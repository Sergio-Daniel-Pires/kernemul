#include "mmu.hpp"
#include "arch.hpp"
#include "../emu.hpp"
#include <stdexcept>
#include <cstring>
#include <cstdlib>
#include <cstdio>

namespace ia32 {
#include <ia32.hpp>
}

namespace x86
{

addr_space& mmu::as_x86(::addr_space& space)
{
	auto* p = dynamic_cast<addr_space*>(&space);
	if (!p) throw std::runtime_error("invalid address space type");
	return *p;
}

const addr_space& mmu::as_x86(const ::addr_space& space)
{
	auto* p = dynamic_cast<const addr_space*>(&space);
	if (!p) throw std::runtime_error("invalid address space type");
	return *p;
}

std::shared_ptr<::addr_space> mmu::create_addr_space()
{
	std::unique_lock lk(mtx_);

	auto space = std::make_shared<addr_space>();
	space->pml4_pa = alloc_phys_locked(page_size(), prot_rw);
	space->mmu_ = this;

	// Spaces after the first alias the kernel half; copying the entries shares the tables below.
	if (!kernel_pml4_pa_)
	{
		kernel_pml4_pa_ = space->pml4_pa;
	}
	else
	{
		for (std::size_t i = 256; i < 512; ++i)
		{
			const auto off = i * sizeof(ia32::pt_entry_64);
			write_phys(space->pml4_pa + off,
				read_phys<ia32::pt_entry_64>(kernel_pml4_pa_ + off));
		}
	}

	spaces_[space->pml4_pa] = space;
	return space;
}

void mmu::destroy_addr_space(std::shared_ptr<::addr_space> space)
{
	std::unique_lock lk(mtx_);

	auto& s = as_x86(*space);
	spaces_.erase(s.pml4_pa);
}

std::shared_ptr<::addr_space> mmu::curr_addr_space(vcpu& cpu)
{
	addr_t cr3 = cpu.reg<addr_t>(x86::cr3);

	std::shared_lock lk(mtx_);
	auto it = spaces_.find(cr3);
	if (it == spaces_.end())
		return nullptr;
	return it->second;
}

void mmu::init_vcpu(vcpu& cpu)
{
	auto cr4_val = cpu.reg<ia32::cr4>(x86::cr4);
	cr4_val.physical_address_extension = 1;
	cr4_val.os_fxsave_fxrstor_support = 1;
	cr4_val.os_xmm_exception_support = 1;
	cr4_val.page_size_extensions = 1;
	// Kernel leaf entries are marked global, which only means anything with this set.
	cr4_val.page_global_enable = 1;
	cr4_val.fsgsbase_enable = 1;
	cr4_val.usermode_instruction_prevention = 1;
	// A kernel that can execute or touch user pages is what an exploited one looks like, and
	// nothing here runs user code in supervisor mode, so both stay on as they do on the real one.
	cr4_val.smep_enable = 1;
	cr4_val.smap_enable = 1;
	cpu.reg(x86::cr4, cr4_val);

	auto efer_val = cpu.reg<ia32::ia32_efer_register>(x86::efer);
	efer_val.syscall_enable = 1;
	efer_val.execute_disable_bit_enable = 1;
	efer_val.ia32e_mode_enable = 1;
	efer_val.ia32e_mode_active = 1;
	cpu.reg(x86::efer, efer_val);

	auto cr0_val = cpu.reg<ia32::cr0>(x86::cr0);
	cr0_val.protection_enable = 1;
	cr0_val.monitor_coprocessor = 1;
	cr0_val.extension_type = 1;
	cr0_val.numeric_error = 1;
	// Without this a supervisor write ignores the read only bit, which is the state a kernel
	// is left in to be patched -- never the state a running one is in.
	cr0_val.write_protect = 1;
	cr0_val.alignment_mask = 1;
	cr0_val.paging_enable = 1;
	cpu.reg(x86::cr0, cr0_val);
}

void mmu::switch_to(vcpu& cpu, std::shared_ptr<::addr_space> space)
{
	auto& s = as_x86(*space);

	std::shared_lock lk(mtx_);

	cpu.reg(x86::cr3, s.pml4_pa);
	cpu.flush_tlb();
}

void mmu::flush_all_tlb()
{
	// uc_ctl_flush_tlb rewrites the tlb, so flushing a running cpu corrupts it under the cpu.
	emu_->run_on_all([&]
	{
		for (auto& cpu : emu_->cpus())
			cpu->flush_tlb();
	});
}

// `supervisor` is the U/S bit, so 1 means user reachable. The cpu takes the strictest bit on
// the walk, which would let every level say user and the leaf decide -- but a guest reading the
// tables sees what each level claims, and on NT a kernel range says supervisor at every level.
addr_t mmu::ensure_table(const addr_t table_pa, const std::size_t index, const bool user)
{
	auto entry = read_phys<ia32::pt_entry_64>(table_pa + index * sizeof(ia32::pt_entry_64));

	if (entry.present)
	{
		// A user page under a table made for a kernel one has to widen it.
		if (user && !entry.supervisor)
		{
			entry.supervisor = 1;
			write_phys(table_pa + index * sizeof(ia32::pt_entry_64), entry);
		}

		return pfn_to_pa(entry.page_frame_number);
	}

	addr_t child = alloc_phys_locked(page_size(), prot_rw);
	entry.flags = 0;
	entry.present = 1;
	entry.write = 1;
	entry.supervisor = user ? 1 : 0;
	// The cpu sets this on every level it walks, so on a running machine an entry that leads
	// anywhere has it. Most of what is mapped here is filled by writing physical memory, which
	// the cpu never walks, and an untouched entry over live memory is not a state hardware
	// leaves behind.
	entry.accessed = 1;
	entry.page_frame_number = child >> page_shift;
	write_phys(table_pa + index * sizeof(ia32::pt_entry_64), entry);

	return child;
}

void mmu::map_page(addr_space& space, addr_t va, addr_t pa, const mem_prot prot)
{
	virt_addr v{ .val = page_align(va) };

	const bool user = !(prot & prot_supervisor);

	addr_t pdpt = ensure_table(space.pml4_pa, v.pml4, user);
	addr_t pd   = ensure_table(pdpt, v.pdpt, user);
	addr_t pt   = ensure_table(pd, v.pd, user);

	ia32::pte_64 entry{};
	entry.present = 1;
	entry.write = (prot & prot_write) ? 1 : 0;
	entry.execute_disable = (prot & prot_exec) ? 0 : 1;
	entry.accessed = 1;
	// A writable page the emulator has already filled has been written, whoever did the
	// writing; hardware would have recorded that here.
	entry.dirty = (prot & prot_write) ? 1 : 0;
	if (user)
		entry.supervisor = 1;
	else
		entry.global = 1;
	entry.page_frame_number = pa >> page_shift;

	write_phys(pt + v.pt * sizeof(ia32::pte_64), entry);

	space.shadow[v.val] = pa;

	// The page, then each table that maps it. A table is a frame like any other and nothing
	// maps it at an ordinary address, so this is the only place it can be described from --
	// and here it costs nothing, because the walk that mapped the page already found them all.
	note_page(pa,   v.val, pt,            0, true);
	note_page(pt,   v.val, pd,            1, true);
	note_page(pd,   v.val, pdpt,          2, true);
	note_page(pdpt, v.val, space.pml4_pa, 3, true);

	// The root is reached through its own self map entry, so it is its own parent.
	note_page(space.pml4_pa, v.val, space.pml4_pa, 4, true);
}

void mmu::unmap_page(addr_space& space, addr_t va)
{
	virt_addr v{ .val = page_align(va) };

	// Before the walk, not after it: the returns below give up on a page whose tables are
	// already gone, and a shadow entry left behind for one is a translation copy_virt still
	// goes through -- a read of memory the guest has given back.
	if (const auto it = space.shadow.find(v.val); it != space.shadow.end())
	{
		const auto pa = it->second;
		space.shadow.erase(it);

		// Only the page. The tables it hung off are still tables, and still map whatever else
		// was in them.
		note_page(pa, v.val, 0, 0, false);
	}

	auto pml4e = read_phys<ia32::pt_entry_64>(space.pml4_pa + v.pml4 * sizeof(ia32::pt_entry_64));
	if (!pml4e.present) return;

	auto pdpte = read_phys<ia32::pt_entry_64>(pfn_to_pa(pml4e.page_frame_number) + v.pdpt * sizeof(ia32::pt_entry_64));
	if (!pdpte.present) return;

	auto pde = read_phys<ia32::pt_entry_64>(pfn_to_pa(pdpte.page_frame_number) + v.pd * sizeof(ia32::pt_entry_64));
	if (!pde.present) return;

	write_phys<std::uint64_t>(pfn_to_pa(pde.page_frame_number) + v.pt * sizeof(ia32::pte_64), 0);
}

void mmu::map_virt(::addr_space& space, addr_t va, std::size_t size, mem_prot prot)
{
	auto& s = as_x86(space);
	std::unique_lock lk(mtx_);

	auto phys_prot = prot & ~prot_supervisor;

	const addr_t start = page_align(va);
	const std::size_t aligned = size_align(size);

	const addr_t pa_block = alloc_phys_locked(aligned, phys_prot);

	for (std::size_t off = 0; off < aligned; off += page_size())
		map_page(s, start + off, pa_block + off, prot);

	flush_all_tlb();
}

void mmu::map_virt_phys(::addr_space& space, addr_t va, addr_t pa, std::size_t size, mem_prot prot)
{
	auto& s = as_x86(space);
	std::unique_lock lk(mtx_);

	const addr_t start = page_align(va);
	const addr_t pa_start = page_align(pa);
	const std::size_t aligned = size_align(size);

	for (std::size_t off = 0; off < aligned; off += page_size())
		map_page(s, start + off, pa_start + off, prot);

	flush_all_tlb();
}

void mmu::unmap_virt(::addr_space& space, addr_t va, std::size_t size)
{
	auto& s = as_x86(space);
	std::unique_lock lk(mtx_);

	const addr_t start = page_align(va);
	const std::size_t aligned = size_align(size);

	for (std::size_t off = 0; off < aligned; off += page_size())
		unmap_page(s, start + off);

	flush_all_tlb();
}

void mmu::copy_virt(const addr_space& space, addr_t va, void* buf, std::size_t size, bool write)
{
	auto* bytes = static_cast<std::uint8_t*>(buf);
	std::size_t done = 0;
	const char* watch_env = write ? std::getenv("KERNEMUL_MEM_WRITE_WATCH") : nullptr;
	const addr_t write_watch = watch_env ? std::strtoull(watch_env, nullptr, 0) : 0;

	while (done < size)
	{
		addr_t cur_va = va + done;
		addr_t page = page_align(cur_va);
		std::size_t page_off = cur_va - page;
		std::size_t chunk = std::min(size - done, page_size() - page_off);

		const auto it = space.shadow.find(page);

		const auto frame = it != space.shadow.end()
			? std::optional<addr_t>(it->second)
			: translate_virt(space, page);

		if (!frame)
			throw std::runtime_error("unmapped virtual address");

		const addr_t pa = *frame + page_off;

		if (write)
		{
			if (write_watch && cur_va <= write_watch + 7 && cur_va + chunk > write_watch)
			{
				const addr_t overlap = std::max(cur_va, write_watch);
				const auto offset = static_cast<std::size_t>(overlap - cur_va);
				const auto n = std::min<std::size_t>(8, chunk - offset);
				std::uint64_t old_value = 0;
				std::uint64_t new_value = 0;
				read_phys(pa + offset, &old_value, n);
				std::memcpy(&new_value, bytes + done + offset, n);
				std::fprintf(stderr,
					"[MMU_WRITE_WATCH] va=0x%llX pa=0x%llX size=%zu old=0x%016llX incoming=0x%016llX\n",
					static_cast<unsigned long long>(overlap),
					static_cast<unsigned long long>(pa + offset), n,
					static_cast<unsigned long long>(old_value),
					static_cast<unsigned long long>(new_value));
			}
			write_phys(pa, bytes + done, chunk);
		}
		else
			read_phys(pa, bytes + done, chunk);

		done += chunk;
	}
}

void mmu::read_virt(const ::addr_space& space, addr_t va, void* buf, std::size_t size)
{
	auto& s = as_x86(space);
	std::shared_lock lk(mtx_);
	copy_virt(s, va, buf, size, false);
}

void mmu::write_virt(const ::addr_space& space, addr_t va, const void* buf, std::size_t size)
{
	auto& s = as_x86(space);
	std::shared_lock lk(mtx_);
	copy_virt(s, va, const_cast<void*>(buf), size, true);
}

void mmu::prot_virt(::addr_space& space, addr_t va, std::size_t size, mem_prot prot)
{
	auto& s = as_x86(space);
	std::unique_lock lk(mtx_);

	const addr_t start = page_align(va);
	const std::size_t aligned = size_align(size);

	for (std::size_t off = 0; off < aligned; off += page_size())
	{
		virt_addr v{ .val = start + off };

		auto pml4e = read_phys<ia32::pt_entry_64>(s.pml4_pa + v.pml4 * sizeof(ia32::pt_entry_64));
		if (!pml4e.present) continue;

		auto pdpte = read_phys<ia32::pt_entry_64>(pfn_to_pa(pml4e.page_frame_number) + v.pdpt * sizeof(ia32::pt_entry_64));
		if (!pdpte.present) continue;

		auto pde = read_phys<ia32::pt_entry_64>(pfn_to_pa(pdpte.page_frame_number) + v.pd * sizeof(ia32::pt_entry_64));
		if (!pde.present) continue;

		addr_t pt_pa = pfn_to_pa(pde.page_frame_number);
		auto pte = read_phys<ia32::pte_64>(pt_pa + v.pt * sizeof(ia32::pte_64));
		if (!pte.present) continue;

		ia32::pte_64 new_pte{};
		new_pte.present = 1;
		new_pte.page_frame_number = pte.page_frame_number;
		// Reprotecting a page does not un-touch it.
		new_pte.accessed = pte.accessed;
		new_pte.dirty = pte.dirty;
		if (prot & prot_write)
			new_pte.write = 1;
		if (!(prot & prot_exec))
			new_pte.execute_disable = 1;
		if (pte.supervisor)
			new_pte.supervisor = 1;
		else
			new_pte.global = 1;

		write_phys(pt_pa + v.pt * sizeof(ia32::pte_64), new_pte);
	}

	flush_all_tlb();
}

// virt_addr ignores bits 63:48 on purpose: MiGetPteAddress masks a virtual address to 48 bits
// before indexing, so NT answers for a non canonical address exactly as it does for its
// canonical twin. The cpu still faults on dereferencing one -- this is the walk, not the load.
std::optional<addr_t> mmu::translate_virt(const addr_space& s, const addr_t page)
{
	const virt_addr v{ .val = page };

	const auto pml4e = read_phys<ia32::pt_entry_64>(
		s.pml4_pa + v.pml4 * sizeof(ia32::pt_entry_64));
	if (!pml4e.present) return std::nullopt;

	const auto pdpte = read_phys<ia32::pt_entry_64>(
		pfn_to_pa(pml4e.page_frame_number) + v.pdpt * sizeof(ia32::pt_entry_64));
	if (!pdpte.present) return std::nullopt;

	const auto pde = read_phys<ia32::pt_entry_64>(
		pfn_to_pa(pdpte.page_frame_number) + v.pd * sizeof(ia32::pt_entry_64));
	if (!pde.present) return std::nullopt;

	const auto pte = read_phys<ia32::pte_64>(
		pfn_to_pa(pde.page_frame_number) + v.pt * sizeof(ia32::pte_64));
	if (!pte.present) return std::nullopt;

	return pfn_to_pa(pte.page_frame_number);
}

std::optional<addr_t> mmu::virt_to_phys(const ::addr_space& space, addr_t va)
{
	auto& s = as_x86(space);
	std::shared_lock lk(mtx_);

	addr_t page = page_align(va);
	auto it = s.shadow.find(page);

	if (it != s.shadow.end())
		return it->second + (va - page);

	if (const auto pa = translate_virt(s, page))
		return *pa + (va - page);

	return std::nullopt;
}

std::optional<addr_t> mmu::phys_to_virt(const ::addr_space& space, addr_t pa)
{
	auto& s = as_x86(space);
	std::shared_lock lk(mtx_);

	addr_t pa_page = pa & ~(page_size() - 1);
	addr_t offset = pa - pa_page;

	for (auto& [va_page, mapped_pa] : s.shadow)
	{
		if (mapped_pa == pa_page)
			return va_page + offset;
	}

	return std::nullopt;
}

addr_t addr_space::alloc(std::size_t size, mem_prot prot)
{
	constexpr std::size_t page = 0x1000;
	auto aligned = (size + page - 1) & ~(page - 1);

	addr_t& cursor = (prot & prot_supervisor) ? kernel_next_ : user_next_;
	addr_t va = cursor;
	cursor += aligned;

	mmu_->map_virt(*this, va, aligned, prot);
	return va;
}

addr_t addr_space::alloc_at(addr_t va, std::size_t size, mem_prot prot)
{
	constexpr std::size_t page = 0x1000;
	const auto aligned = (size + page - 1) & ~(page - 1);

	mmu_->map_virt(*this, va, aligned, prot);
	return va;
}

addr_t addr_space::map_phys(const addr_t pa, const std::size_t size, const mem_prot prot)
{
	constexpr std::size_t page = 0x1000;
	const auto aligned = (size + page - 1) & ~(page - 1);

	addr_t& cursor = (prot & prot_supervisor) ? kernel_next_ : user_next_;
	const addr_t va = cursor;
	cursor += aligned;

	mmu_->map_virt_phys(*this, va, pa, aligned, prot);
	return va;
}

}
