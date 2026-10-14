#include "nt_vm_ops.hpp"
#include "../win_kernel.hpp"
#include "../mdl.hpp"
#include "../object_names.hpp"
#include "../objects.hpp"
#include "../status.hpp"
#include "../string.hpp"
#include "../types.hpp"
#include "../../../util/log.hpp"
#include "../../../util/string.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

namespace
{

enum allocation_type : std::uint32_t
{
	mem_commit  = 0x00001000,
	mem_reserve = 0x00002000,
	mem_release = 0x00008000,
	mem_free    = 0x00010000,
};

enum page_protection : std::uint32_t
{
	page_noaccess          = 0x01,
	page_readonly          = 0x02,
	page_readwrite         = 0x04,
	page_writecopy         = 0x08,
	page_execute           = 0x10,
	page_execute_read      = 0x20,
	page_execute_readwrite = 0x40,
	page_execute_writecopy = 0x80,
};

win_user_mem* user_mem(vcpu& cpu)
{
	const auto t = cpu.thread();
	auto* const proc = t ? dynamic_cast<win_user_proc*>(t->proc().get()) : nullptr;
	return proc ? &proc->mem() : nullptr;
}

mem_prot prot_from_page(const std::uint32_t protect)
{
	switch (protect & 0xFF)
	{
	case page_noaccess:          return static_cast<mem_prot>(0);
	case page_readonly:          return prot_read;
	case page_readwrite:
	case page_writecopy:         return prot_rw;
	case page_execute:           return prot_exec;
	case page_execute_read:      return prot_rx;
	case page_execute_readwrite:
	case page_execute_writecopy: return prot_rwx;
	default:                     return prot_rw;
	}
}

std::uint32_t page_from_prot(const mem_prot prot)
{
	const bool w = (prot & prot_write) != 0;
	const bool x = (prot & prot_exec) != 0;

	if (!(prot & prot_read))
		return page_noaccess;

	if (x)
		return w ? page_execute_readwrite : page_execute_read;

	return w ? page_readwrite : page_readonly;
}

constexpr std::uint32_t memory_basic_information = 0;
constexpr std::uint32_t memory_working_set_ex_information = 4;
constexpr std::uint32_t memory_image_information = 6;

// Nothing here carries an image extension -- the ARM64X and hotpatch metadata.
constexpr std::uint32_t memory_image_extension_information = 14;

constexpr std::uint32_t section_basic_information = 0;

// kernel32 maps this while it is initialising.
constexpr std::string_view shared_section_name = "\\Windows\\SharedSection";
constexpr std::uint64_t shared_section_size = 0x10000;

#pragma pack(push, 8)
struct memory_basic_information_t
{
	addr_t        base_address;
	addr_t        allocation_base;
	std::uint32_t allocation_protect;
	std::uint32_t partition_id;
	std::uint64_t region_size;
	std::uint32_t state;
	std::uint32_t protect;
	std::uint32_t type;
	std::uint32_t padding;
};

struct memory_image_information_t
{
	addr_t        image_base;
	std::uint64_t size_of_image;
	std::uint32_t image_flags;
	std::uint32_t padding;
};

struct memory_working_set_ex_information_t
{
	addr_t        virtual_address;
	std::uint64_t attributes;
};

constexpr std::uint64_t working_set_valid = 1;
constexpr int working_set_protection_shift = 4;

struct section_basic_information_t
{
	addr_t        base_address;
	std::uint32_t allocation_attributes;
	std::uint32_t padding;
	std::int64_t  maximum_size;
};
#pragma pack(pop)

static_assert(sizeof(memory_basic_information_t) == 0x30);
static_assert(sizeof(memory_image_information_t) == 0x18);
static_assert(sizeof(section_basic_information_t) == 0x18);

constexpr std::uint32_t mem_state_commit = 0x1000;
constexpr std::uint32_t mem_state_free = 0x10000;
constexpr std::uint32_t mem_type_private = 0x20000;

// -1 is the pseudo-handle for the caller itself, and only the current process is addressable.
constexpr std::uint64_t current_process_handle = ~std::uint64_t{0};

bool is_current_process(const std::uint64_t handle)
{
	return handle == 0 || handle == current_process_handle;
}

constexpr std::uint32_t protection_modifiers = 0x100 | 0x200 | 0x400;

bool valid_protection(const std::uint32_t protect)
{
	switch (protect & ~protection_modifiers)
	{
	case page_noaccess:
	case page_readonly:
	case page_readwrite:
	case page_writecopy:
	case page_execute:
	case page_execute_read:
	case page_execute_readwrite:
	case page_execute_writecopy:
		return true;

	default:
		return false;
	}
}

struct page_rights
{
	bool read = false;
	bool write = false;
	bool exec = false;
	bool copy_on_write = false;
};

page_rights rights_of(const std::uint32_t protect)
{
	switch (protect & ~protection_modifiers)
	{
	case page_readonly:          return { true, false, false, false };
	case page_readwrite:         return { true, true, false, false };
	case page_writecopy:         return { true, true, false, true };
	case page_execute:           return { false, false, true, false };
	case page_execute_read:      return { true, false, true, false };
	case page_execute_readwrite: return { true, true, true, false };
	case page_execute_writecopy: return { true, true, true, true };
	default:                     return {};
	}
}

// A view may ask for no more than the section was created to allow.
bool view_fits_section(const std::uint32_t view, const std::uint32_t section)
{
	const auto v = rights_of(view);
	const auto s = rights_of(section);

	return (!v.read || s.read || s.exec) && (!v.write || s.write) && (!v.exec || s.exec);
}

constexpr std::uint32_t section_map_write = 0x0002;
constexpr std::uint32_t section_map_read = 0x0004;
constexpr std::uint32_t section_map_execute = 0x0008;

// What the section handle has to grant for a view with this protection.
std::uint32_t section_access_needed(const std::uint32_t view)
{
	const auto v = rights_of(view);
	std::uint32_t needed = 0;

	if (v.write && !v.copy_on_write)
		needed |= section_map_write;

	if (v.exec)
		needed |= section_map_execute;

	if (v.read && (!v.write || v.copy_on_write))
		needed |= section_map_read;

	return needed;
}

constexpr std::uint32_t process_vm_operation = 0x0008;
constexpr std::uint32_t process_vm_read = 0x0010;
constexpr std::uint32_t process_vm_write = 0x0020;
constexpr std::uint32_t process_query_information = 0x0400;
constexpr std::uint32_t process_query_limited_information = 0x1000;

// The address space a process handle names. mem is null for a process that has no user memory
// tracker, which is the kernel's own.
struct vm_target
{
	NTSTATUS status = STATUS_SUCCESS;
	win_user_mem* mem = nullptr;
	std::shared_ptr<addr_space> space;
	std::shared_ptr<process> proc;
	bool remote = false;
};

vm_target resolve_target(win_kernel_state& state, vcpu& cpu, const std::uint64_t handle,
	const std::uint32_t any_of_access)
{
	vm_target target;

	if (is_current_process(handle))
	{
		target.space = cpu.curr_addr_space();
		target.mem = user_mem(cpu);

		if (const auto t = cpu.thread())
			target.proc = t->proc();

		return target;
	}

	const auto entry = state.sys_proc->handle_table().lookup_handle(handle);

	if (!entry)
	{
		target.status = STATUS_INVALID_HANDLE;
		return target;
	}

	const auto object = state.objs.get_object<process_object>(entry->body_addr);

	if (!object)
	{
		target.status = STATUS_OBJECT_TYPE_MISMATCH;
		return target;
	}

	if (!(entry->access & any_of_access))
	{
		target.status = STATUS_ACCESS_DENIED;
		return target;
	}

	target.proc = state.find_process(static_cast<process::id_type>(object->pid));
	target.space = target.proc ? target.proc->addr_space() : nullptr;

	if (!target.space)
	{
		target.status = STATUS_PROCESS_IS_TERMINATING;
		return target;
	}

	if (const auto user = std::dynamic_pointer_cast<win_user_proc>(target.proc))
		target.mem = &user->mem();

	target.remote = true;

	return target;
}

// Data section pages live in guest physical memory, so a view in each process is the same bytes.
addr_t section_backing(vcpu& cpu, section_host& host)
{
	std::scoped_lock lock(host.phys_mtx);

	if (host.phys)
		return host.phys;

	auto& mmu = *cpu.curr_addr_space()->mmu_;

	const auto size = (host.size + mdl_page_size - 1) & ~static_cast<std::uint64_t>(mdl_page_size - 1);
	const auto pa = mmu.alloc_phys(size, prot_rw);

	if (host.file)
	{
		const auto data = host.file->data();
		const auto copied = std::min<std::size_t>(size, data.size());

		if (copied)
			mmu.write_phys(pa, data.data(), copied);
	}

	host.phys = pa;

	return pa;
}

}

// The page tables behind these are the guest's own, so it faults exactly as on real hardware.
void modules::register_ntoskrnl_vm_ops(win_kernel_state& state, proc_module& mod)
{
	auto* st = &state;

	auto allocate = [st](vcpu& cpu, const std::uint64_t process_handle,
		emu_object<addr_t> base_address, emu_object<std::uint64_t> region_size,
		const std::uint32_t allocation_type, const std::uint32_t protect,
		const std::string_view who) -> NTSTATUS
	{
		const auto target = resolve_target(*st, cpu, process_handle, process_vm_operation);

		if (target.status != STATUS_SUCCESS)
			return target.status;

		if (!valid_protection(protect))
			return STATUS_INVALID_PAGE_PROTECTION;

		if (!base_address || !region_size)
			return STATUS_INVALID_PARAMETER;

		const auto requested = region_size.read();

		if (!requested)
			return STATUS_INVALID_PARAMETER;

		const auto wanted = base_address.read();
		const auto size = (requested + mdl_page_size - 1) & ~static_cast<std::uint64_t>(mdl_page_size - 1);

		// Reserve without commit still gets pages: there is no reserved state to hold a range in.
		if (!(allocation_type & (mem_commit | mem_reserve)))
			return STATUS_INVALID_PARAMETER;

		// The address space's own cursor hands out supervisor pages, which ring 3 cannot touch.
		if (auto* const mem = target.mem)
		{
			addr_t base = wanted;
			auto region = static_cast<std::size_t>(size);
			const auto status = mem->allocate(base, region, allocation_type, protect);

			if (status != STATUS_SUCCESS)
			{
				THREAD_LOG_WARN("{}: cannot allocate {} bytes at 0x{:X}: 0x{:X}",
					who, size, wanted, static_cast<std::uint32_t>(status));
				return status;
			}

			base_address.write(base);
			region_size.write(region);

			THREAD_LOG_INFO("{}(size={}, type=0x{:X}, protect=0x{:X}) -> 0x{:X}",
				who, region, allocation_type, protect, base);

			return STATUS_SUCCESS;
		}

		auto& space = *target.space;
		addr_t addr = 0;

		try
		{
			addr = space.alloc(size, prot_from_page(protect) | prot_supervisor);
		}
		catch (const std::exception& e)
		{
			THREAD_LOG_ERR("{}: cannot allocate {} bytes: {}", who, size, e.what());
			return STATUS_NO_MEMORY;
		}

		if (!addr)
			return STATUS_NO_MEMORY;

		if (wanted && wanted != addr)
			THREAD_LOG_WARN("{}: wanted 0x{:X}, allocated 0x{:X} -- the address space picks",
				who, wanted, addr);

		base_address.write(addr);
		region_size.write(size);

		THREAD_LOG_INFO("{}(size={}, type=0x{:X}, protect=0x{:X}) -> 0x{:X}",
			who, size, allocation_type, protect, addr);

		return STATUS_SUCCESS;
	};

	auto allocate_vm = [allocate](vcpu& cpu, const std::uint64_t process_handle,
		emu_object<addr_t> base_address, [[maybe_unused]] const std::uint64_t zero_bits,
		emu_object<std::uint64_t> region_size, const std::uint32_t allocation_type,
		const std::uint32_t protect) -> NTSTATUS
	{
		return allocate(cpu, process_handle, base_address, region_size, allocation_type,
			protect, "NtAllocateVirtualMemory");
	};

	auto allocate_vm_ex = [allocate](vcpu& cpu, const std::uint64_t process_handle,
		emu_object<addr_t> base_address, emu_object<std::uint64_t> region_size,
		const std::uint32_t allocation_type, const std::uint32_t protect,
		const addr_t extended_parameters, const std::uint32_t extended_parameter_count) -> NTSTATUS
	{
		if (extended_parameter_count)
			THREAD_LOG_WARN("NtAllocateVirtualMemoryEx: ignoring {} extended parameter(s) at 0x{:X}",
				extended_parameter_count, extended_parameters);

		return allocate(cpu, process_handle, base_address, region_size, allocation_type,
			protect, "NtAllocateVirtualMemoryEx");
	};

	auto free_vm = [st](vcpu& cpu, const std::uint64_t process_handle,
		emu_object<addr_t> base_address, emu_object<std::uint64_t> region_size,
		const std::uint32_t free_type) -> NTSTATUS
	{
		const auto target = resolve_target(*st, cpu, process_handle, process_vm_operation);

		if (target.status != STATUS_SUCCESS)
			return target.status;

		if (!base_address)
			return STATUS_INVALID_PARAMETER;

		const auto base = mdl_page_base(base_address.read());
		auto size = region_size ? region_size.read() : 0;

		if (auto* const mem = target.mem)
		{
			addr_t start = base;
			auto region = static_cast<std::size_t>(size);
			const auto status = mem->free(start, region, free_type);

			if (status != STATUS_SUCCESS)
				return status;

			base_address.write(start);

			if (region_size)
				region_size.write(region);

			return STATUS_SUCCESS;
		}

		if (free_type & mem_release)
		{
			if (size)
				return STATUS_INVALID_PARAMETER;

			// Nothing records the run length, so a release takes down only the page the base names.
			size = mdl_page_size;
		}

		if (!size)
			return STATUS_INVALID_PARAMETER;

		auto& space = *target.space;

		try
		{
			space.mmu_->unmap_virt(space, base, size);
		}
		catch (const std::exception& e)
		{
			THREAD_LOG_WARN("NtFreeVirtualMemory: 0x{:X} is not mapped: {}", base, e.what());
			return STATUS_MEMORY_NOT_ALLOCATED;
		}

		if (region_size)
			region_size.write(size);

		THREAD_LOG_INFO("NtFreeVirtualMemory(0x{:X}, {} bytes, type=0x{:X})", base, size, free_type);

		return STATUS_SUCCESS;
	};

	auto protect_vm = [st](vcpu& cpu, const std::uint64_t process_handle,
		emu_object<addr_t> base_address, emu_object<std::uint64_t> region_size,
		const std::uint32_t new_protect, emu_object<std::uint32_t> old_protect) -> NTSTATUS
	{
		const auto target = resolve_target(*st, cpu, process_handle, process_vm_operation);

		if (target.status != STATUS_SUCCESS)
			return target.status;

		if (!valid_protection(new_protect))
			return STATUS_INVALID_PAGE_PROTECTION;

		if (!base_address || !region_size)
			return STATUS_INVALID_PARAMETER;

		const auto requested = base_address.read();
		const auto base = mdl_page_base(requested);
		const auto size = (region_size.read() + (requested - base) + mdl_page_size - 1)
			& ~static_cast<std::uint64_t>(mdl_page_size - 1);

		if (auto* const mem = target.mem)
		{
			addr_t start = base;
			auto region = static_cast<std::size_t>(size);
			std::uint32_t previous = page_readwrite;
			const auto status = mem->protect(start, region, new_protect, previous);

			if (status != STATUS_SUCCESS)
				return status;

			if (old_protect)
				old_protect.write(previous);

			base_address.write(start);
			region_size.write(region);

			return STATUS_SUCCESS;
		}

		auto& space = *target.space;

		// No protection can be read back out of the tables, so the caller is told a fresh page's.
		if (old_protect)
			old_protect.write(page_readwrite);

		try
		{
			space.prot_mem(base, size, prot_from_page(new_protect) | prot_supervisor);
		}
		catch (const std::exception& e)
		{
			THREAD_LOG_WARN("NtProtectVirtualMemory: cannot protect 0x{:X}: {}", base, e.what());
			return STATUS_INVALID_ADDRESS;
		}

		base_address.write(base);
		region_size.write(size);

		THREAD_LOG_INFO("NtProtectVirtualMemory(0x{:X}, {} bytes, protect=0x{:X})",
			base, size, new_protect);

		return STATUS_SUCCESS;
	};

	// Reporting the real run would need a VAD tree, so the run here is the one page asked about.
	auto query_vm = [st](vcpu& cpu, const std::uint64_t process_handle, const addr_t base_address,
		const std::uint32_t memory_information_class, const addr_t memory_information,
		const std::uint64_t memory_information_length,
		emu_object<std::uint64_t> return_length) -> NTSTATUS
	{
		const auto target = resolve_target(*st, cpu, process_handle,
			process_query_information | process_query_limited_information);

		if (target.status != STATUS_SUCCESS)
			return target.status;

		if (memory_information_class == memory_working_set_ex_information)
		{
			using entry_t = memory_working_set_ex_information_t;

			const auto count = memory_information_length / sizeof(entry_t);

			if (return_length)
				return_length.write(count * sizeof(entry_t));

			if (!count)
				return STATUS_INFO_LENGTH_MISMATCH;

			auto& space = *cpu.curr_addr_space();
			auto& queried = *target.space;

			for (std::uint64_t i = 0; i < count; ++i)
			{
				emu_object<entry_t> entry(space, memory_information + i * sizeof(entry_t));

				auto info = entry.read();
				const auto page = mdl_page_base(info.virtual_address);
				const auto pa = queried.mmu_->virt_to_phys(queried, page);

				info.attributes = pa
					? working_set_valid
						| (std::uint64_t{page_execute_readwrite} << working_set_protection_shift)
					: 0;

				entry.write(info);
			}

			THREAD_LOG_INFO("NtQueryVirtualMemory(MemoryWorkingSetExInformation): {} page(s)",
				count);

			return STATUS_SUCCESS;
		}

		if (memory_information_class == memory_image_information)
		{
			if (return_length)
				return_length.write(sizeof(memory_image_information_t));

			if (memory_information_length < sizeof(memory_image_information_t))
				return STATUS_INFO_LENGTH_MISMATCH;

			const auto mod = target.proc ? target.proc->find_module_by_addr(base_address) : nullptr;

			if (!mod)
			{
				THREAD_LOG_INFO("NtQueryVirtualMemory(MemoryImageInformation, 0x{:X}): "
					"not in an image", base_address);
				return STATUS_INVALID_ADDRESS;
			}

			memory_image_information_t info{};
			info.image_base = mod->addr;
			info.size_of_image = mod->size;

			emu_object<memory_image_information_t>(*cpu.curr_addr_space(),
				memory_information).write(info);

			THREAD_LOG_INFO("NtQueryVirtualMemory(MemoryImageInformation, 0x{:X}) -> {} at 0x{:X}",
				base_address, mod->name, mod->addr);

			return STATUS_SUCCESS;
		}

		// LdrpProcessMappedModule treats STATUS_NOT_SUPPORTED as "no extension" and carries on;
		// an invalid class lands in the failure path below it and kills the process.
		if (memory_information_class == memory_image_extension_information)
		{
			if (return_length)
				return_length.write(0);

			THREAD_LOG_INFO("NtQueryVirtualMemory(MemoryImageExtensionInformation, 0x{:X}): "
				"no image here has an extension", base_address);

			return STATUS_NOT_SUPPORTED;
		}

		if (memory_information_class != memory_basic_information)
		{
			THREAD_LOG_WARN("NtQueryVirtualMemory: unhandled class {}", memory_information_class);
			return STATUS_INVALID_INFO_CLASS;
		}

		if (return_length)
			return_length.write(sizeof(memory_basic_information_t));

		if (memory_information_length < sizeof(memory_basic_information_t))
			return STATUS_INFO_LENGTH_MISMATCH;

		auto& space = *cpu.curr_addr_space();

		if (auto* const mem = target.mem)
		{
			win::memory_basic_info info{};
			const auto status = mem->query(base_address, info);

			if (status != STATUS_SUCCESS)
				return status;

			emu_object<win::memory_basic_info>(space, memory_information).write(info);

			THREAD_LOG_INFO("NtQueryVirtualMemory(0x{:X}) -> base 0x{:X}, {} bytes, state 0x{:X}",
				base_address, info.base_address, info.region_size, info.state);

			return STATUS_SUCCESS;
		}

		const auto base = mdl_page_base(base_address);
		const bool mapped = target.space->mmu_->virt_to_phys(*target.space, base).has_value();

		memory_basic_information_t info{};
		info.base_address = base;
		info.allocation_base = mapped ? base : 0;
		info.allocation_protect = mapped ? page_readwrite : 0;
		info.region_size = mdl_page_size;
		info.state = mapped ? mem_state_commit : mem_state_free;
		info.protect = mapped ? page_readwrite : 0;
		info.type = mapped ? mem_type_private : 0;

		emu_object<memory_basic_information_t>(space, memory_information).write(info);

		THREAD_LOG_INFO("NtQueryVirtualMemory(0x{:X}) -> {}", base, mapped ? "committed" : "free");

		return STATUS_SUCCESS;
	};

	auto create_section = [st](vcpu& cpu, emu_object<std::uint64_t> section_handle,
		const std::uint32_t desired_access, emu_object<_OBJECT_ATTRIBUTES> object_attributes,
		emu_object<std::int64_t> maximum_size, const std::uint32_t section_page_protection,
		const std::uint32_t allocation_attributes, const std::uint64_t file_handle) -> NTSTATUS
	{
		if (!section_handle)
			return STATUS_INVALID_PARAMETER;

		auto& space = *cpu.curr_addr_space();

		const emu_object<_UNICODE_STRING> name_obj(space, object_attributes
			? guest_va(object_attributes.field(&_OBJECT_ATTRIBUTES::ObjectName).read())
			: 0);

		const auto name = narrow_wstring(win::read_unicode_string(name_obj));
		const auto max_size = maximum_size ? maximum_size.read() : 0;

		const auto resolved = resolve_object_name(*st, cpu, object_attributes);

		if (resolved.status != STATUS_SUCCESS)
			return resolved.status;

		if (resolved.named())
		{
			if (const auto existing = st->objs.lookup_named_object(resolved.key))
			{
				if (!(resolved.attributes & obj_openif))
					return STATUS_OBJECT_NAME_COLLISION;

				if (!st->objs.get_object<section_host>(existing))
					return STATUS_OBJECT_TYPE_MISMATCH;

				section_handle.write(st->sys_proc->handle_table().create_handle(existing,
					desired_access, (resolved.attributes & obj_inherit)
						? win_handle_table::flag_inherit : 0));

				return STATUS_OBJECT_NAME_EXISTS;
			}
		}

		if (!valid_protection(section_page_protection)
			|| (section_page_protection & ~protection_modifiers) == page_noaccess)
			return STATUS_INVALID_PAGE_PROTECTION;

		auto host = std::make_shared<section_host>();
		host->is_image = (allocation_attributes & sec_image) != 0;
		host->protection = section_page_protection & ~protection_modifiers;

		if (file_handle)
		{
			const auto file = st->sys_proc->handle_table().get_object<file_host>(file_handle);

			if (!file)
				return STATUS_INVALID_HANDLE;

			host->file = file->file;
			host->path = file->path;
		}

		const auto file_size = host->file ? static_cast<std::uint64_t>(host->file->size()) : 0;

		if (host->file && file_size == 0 && max_size <= 0)
			return STATUS_MAPPED_FILE_SIZE_ZERO;

		if (!host->file && max_size <= 0)
			return STATUS_INVALID_PARAMETER;

		// A section the file cannot grow into cannot be larger than the file.
		if (host->file && !host->is_image && static_cast<std::uint64_t>(max_size) > file_size
			&& !rights_of(host->protection).write)
			return STATUS_SECTION_TOO_BIG;

		const auto page_mask = static_cast<std::uint64_t>(mdl_page_size - 1);

		host->size = host->file
			? (max_size > 0 && !host->is_image
				? (static_cast<std::uint64_t>(max_size) + page_mask) & ~page_mask
				: file_size)
			: (static_cast<std::uint64_t>(max_size) + page_mask) & ~page_mask;

		const auto size = static_cast<std::int64_t>(host->size);
		const auto path = host->path;

		std::array<std::uint8_t, section_body_size> body{};
		std::memcpy(body.data() + section_body_size_offset, &size, sizeof(size));

		const auto addr = st->objs.create_object(0, body.data(), body.size(),
			std::move(host), prot_rw | prot_supervisor);

		if (!addr)
			return STATUS_INSUFFICIENT_RESOURCES;

		if (resolved.named())
			st->objs.register_named_object(resolved.key, addr);

		const auto handle = st->sys_proc->handle_table().create_handle(addr, desired_access,
			(resolved.attributes & obj_inherit) ? win_handle_table::flag_inherit : 0);
		section_handle.write(handle);

		THREAD_LOG_INFO("NtCreateSection('{}', file=0x{:X}, max_size={}, protection=0x{:X}, "
			"allocation=0x{:X}) -> handle=0x{:X} ({} bytes)",
			name.empty() ? path : name, file_handle, max_size, section_page_protection,
			allocation_attributes, handle, size);

		return STATUS_SUCCESS;
	};

	auto open_section = [st](vcpu& cpu, emu_object<std::uint64_t> section_handle,
		const std::uint32_t desired_access,
		emu_object<_OBJECT_ATTRIBUTES> object_attributes) -> NTSTATUS
	{
		if (section_handle)
			section_handle.write(0);

		auto& space = *cpu.curr_addr_space();

		const emu_object<_UNICODE_STRING> name_obj(space, object_attributes
			? guest_va(object_attributes.field(&_OBJECT_ATTRIBUTES::ObjectName).read())
			: 0);

		const auto name = narrow_wstring(win::read_unicode_string(name_obj));

		const auto resolved = resolve_object_name(*st, cpu, object_attributes);

		if (resolved.status != STATUS_SUCCESS)
			return resolved.status;

		if (resolved.named())
		{
			if (const auto existing = st->objs.lookup_named_object(resolved.key))
			{
				if (!st->objs.get_object<section_host>(existing))
					return STATUS_OBJECT_TYPE_MISMATCH;

				const auto handle = st->sys_proc->handle_table().create_handle(existing,
					desired_access, (resolved.attributes & obj_inherit)
						? win_handle_table::flag_inherit : 0);

				if (section_handle)
					section_handle.write(handle);

				THREAD_LOG_INFO("NtOpenSection('{}') -> handle=0x{:X}", name, handle);

				return STATUS_SUCCESS;
			}
		}

		if (name == shared_section_name)
		{
			auto host = std::make_shared<section_host>();
			host->path = name;
			host->size = shared_section_size;

			const auto size = static_cast<std::int64_t>(host->size);

			std::array<std::uint8_t, section_body_size> body{};
			std::memcpy(body.data() + section_body_size_offset, &size, sizeof(size));

			const auto addr = st->objs.create_object(0, body.data(), body.size(),
				std::move(host), prot_rw | prot_supervisor);

			if (!addr)
				return STATUS_INSUFFICIENT_RESOURCES;

			const auto handle = st->sys_proc->handle_table().create_handle(addr, desired_access);

			if (section_handle)
				section_handle.write(handle);

			THREAD_LOG_INFO("NtOpenSection('{}') -> handle=0x{:X} ({} bytes)",
				name, handle, size);

			return STATUS_SUCCESS;
		}

		THREAD_LOG_WARN("NtOpenSection('{}', access=0x{:X}): nothing here gives a section "
			"that name", name, desired_access);

		return STATUS_OBJECT_NAME_NOT_FOUND;
	};

	auto query_section = [st](vcpu& cpu, const std::uint64_t section_handle,
		const std::uint32_t section_information_class, const addr_t section_information,
		const std::uint64_t section_information_length,
		emu_object<std::uint64_t> return_length) -> NTSTATUS
	{
		const auto host = st->sys_proc->handle_table().get_object<section_host>(section_handle);

		if (!host)
			return STATUS_INVALID_HANDLE;

		if (section_information_class != section_basic_information)
		{
			THREAD_LOG_WARN("NtQuerySection: unhandled class {}", section_information_class);
			return STATUS_INVALID_INFO_CLASS;
		}

		if (return_length)
			return_length.write(sizeof(section_basic_information_t));

		if (section_information_length < sizeof(section_basic_information_t))
			return STATUS_INFO_LENGTH_MISMATCH;

		section_basic_information_t info{};
		info.allocation_attributes = host->is_image ? sec_image : 0;
		info.maximum_size = static_cast<std::int64_t>(host->size);

		emu_object<section_basic_information_t>(*cpu.curr_addr_space(), section_information)
			.write(info);

		THREAD_LOG_INFO("NtQuerySection(0x{:X}, '{}') -> {} bytes",
			section_handle, host->path, host->size);

		return STATUS_SUCCESS;
	};

	// A data view is the section's own pages; an image view is a copy laid out at its addresses.
	state.redirect_ntzw(mod, "MapViewOfSection",
		[st](vcpu& cpu, const std::uint64_t section_handle, const std::uint64_t process_handle,
			emu_object<addr_t> base_address, [[maybe_unused]] const std::uint64_t zero_bits,
			[[maybe_unused]] const std::uint64_t commit_size,
			emu_object<std::int64_t> section_offset, emu_object<std::uint64_t> view_size,
			const std::uint32_t inherit_disposition, const std::uint32_t allocation_type,
			const std::uint32_t win32_protect) -> NTSTATUS
		{
			const auto target = resolve_target(*st, cpu, process_handle, process_vm_operation);

			if (target.status != STATUS_SUCCESS)
				return target.status;

			const auto host = st->sys_proc->handle_table().get_object<section_host>(section_handle);

			if (!host)
				return STATUS_INVALID_HANDLE;

			const bool shared = !host->is_image && target.mem;

			// The offset a view starts at is rounded down to the allocation granularity.
			auto offset = section_offset
				? static_cast<std::uint64_t>(section_offset.read()) : 0;

			if (shared)
				offset &= ~static_cast<std::uint64_t>(win::alloc_granularity - 1);

			if (offset >= host->size)
				return STATUS_INVALID_PARAMETER;

			auto size = view_size ? view_size.read() : 0;

			if (!size)
				size = host->size - offset;

			size = std::min<std::uint64_t>(size, host->size - offset);

			// A data view maps the section's own pages, so the process that made the section
			// and the one it is mapped into read and write the same bytes.
			if (shared)
			{
				if (!valid_protection(win32_protect))
					return STATUS_INVALID_PAGE_PROTECTION;

				if (!view_fits_section(win32_protect, host->protection))
					return STATUS_SECTION_PROTECTION;

				const auto entry = st->sys_proc->handle_table().lookup_handle(section_handle);
				const auto needed = section_access_needed(win32_protect);

				if (!entry || (entry->access & needed) != needed)
					return STATUS_ACCESS_DENIED;

				if (rights_of(win32_protect).copy_on_write)
					THREAD_LOG_WARN("NtMapViewOfSection: copy-on-write is not tracked, so a "
						"write through this view reaches the section");

				const auto pa = section_backing(cpu, *host);

				// A requested base is rounded down to the allocation granularity.
				addr_t at = base_address
					? base_address.read() & ~static_cast<addr_t>(win::alloc_granularity - 1) : 0;
				auto region = static_cast<std::size_t>(size);

				const auto status = target.mem->map_view(at, region, pa + offset, win32_protect);

				if (status != STATUS_SUCCESS)
					return status;

				if (base_address)
					base_address.write(at);

				if (section_offset)
					section_offset.write(static_cast<std::int64_t>(offset));

				if (view_size)
					view_size.write(region);

				THREAD_LOG_INFO("NtMapViewOfSection(0x{:X}, '{}', process=0x{:X}, offset={}, "
					"protect=0x{:X}) -> 0x{:X}, {} bytes shared",
					section_handle, host->path, process_handle, offset, win32_protect, at, region);

				return STATUS_SUCCESS;
			}

			auto& space = *target.space;

			std::vector<std::uint8_t> image;

			if (host->is_image)
			{
				if (!host->file)
					return STATUS_INVALID_FILE_FOR_SECTION;

				image = krnl::pe_virtual_image(host->file->data());

				if (image.empty())
					return STATUS_INVALID_IMAGE_FORMAT;

				if (offset)
					return STATUS_INVALID_PARAMETER;

				size = image.size();
			}

			auto* const mem = target.mem;

			// STATUS_IMAGE_NOT_AT_BASE is not an error: it tells the loader to relocate.
			addr_t preferred = 0;

			if (!image.empty())
				preferred = static_cast<addr_t>(
					reinterpret_cast<const pe::image*>(image.data())
						->nt_hdrs()->optional_hdr.image_base);

			addr_t base = 0;

			if (mem)
			{
				if (preferred)
				{
					addr_t at = preferred;
					auto region = static_cast<std::size_t>(size);

					if (mem->allocate(at, region, mem_commit | mem_reserve,
							page_execute_readwrite) == STATUS_SUCCESS && at == preferred)
						base = at;
				}

				if (!base)
					base = mem->alloc_pages(size, host->is_image
						? page_execute_readwrite : win32_protect);
			}
			else
			{
				base = space.alloc(size, prot_from_page(win32_protect) | prot_supervisor);
			}

			if (!base)
				return STATUS_INSUFFICIENT_RESOURCES;

			if (!image.empty())
			{
				space.write_mem(base, image.data(), image.size());

				if (mem)
					mem->register_image(base, size);
			}
			else if (host->file)
			{
				const auto data = host->file->data();
				const auto available = data.size() > offset ? data.size() - offset : 0;
				const auto copied = std::min<std::size_t>(size, available);

				if (copied)
					space.write_mem(base, data.data() + offset, copied);
			}

			if (base_address)
				base_address.write(base);

			if (view_size)
				view_size.write(size);

			st->views[base] = size;

			const bool rebased = preferred && base != preferred;

			THREAD_LOG_INFO("NtMapViewOfSection(0x{:X}, '{}', offset={}, inherit={}, "
				"allocation=0x{:X}, protect=0x{:X}) -> 0x{:X}, {} bytes{}",
				section_handle, host->path, offset, inherit_disposition, allocation_type,
				win32_protect, base, size, rebased ? " (not at its base)" : "");

			return rebased ? STATUS_IMAGE_NOT_AT_BASE : STATUS_SUCCESS;
		});

	state.redirect_ntzw(mod, "UnmapViewOfSection",
		[st](vcpu& cpu, const std::uint64_t process_handle, const addr_t base_address) -> NTSTATUS
		{
			const auto target = resolve_target(*st, cpu, process_handle, process_vm_operation);

			if (target.status != STATUS_SUCCESS)
				return target.status;

			if (target.mem)
			{
				std::size_t region = 0;

				if (target.mem->unmap_view(base_address, region) == STATUS_SUCCESS)
				{
					THREAD_LOG_INFO("NtUnmapViewOfSection(0x{:X}, process=0x{:X}): {} bytes",
						base_address, process_handle, region);

					return STATUS_SUCCESS;
				}
			}

			const auto it = st->views.find(base_address);

			if (it == st->views.end())
			{
				THREAD_LOG_WARN("NtUnmapViewOfSection: 0x{:X} is not a mapped view", base_address);
				return STATUS_NOT_MAPPED_VIEW;
			}

			auto& space = *target.space;
			space.mmu_->unmap_virt(space, base_address, it->second);

			THREAD_LOG_INFO("NtUnmapViewOfSection(0x{:X}): {} bytes", base_address, it->second);

			st->views.erase(it);

			return STATUS_SUCCESS;
		});

	state.redirect_ntzw(mod, "AllocateVirtualMemory", allocate_vm);
	state.redirect_ntzw(mod, "AllocateVirtualMemoryEx", allocate_vm_ex);
	state.redirect_ntzw(mod, "FreeVirtualMemory", free_vm);
	state.redirect_ntzw(mod, "ProtectVirtualMemory", protect_vm);
	state.redirect_ntzw(mod, "QueryVirtualMemory", query_vm);
	state.redirect_ntzw(mod, "CreateSection", create_section);
	state.redirect_ntzw(mod, "OpenSection", open_section);
	state.redirect_ntzw(mod, "QuerySection", query_section);

	const auto copy_virtual_memory = [st](vcpu& cpu, const std::uint64_t process_handle,
		const addr_t remote_va, const addr_t local_va, const std::uint64_t size,
		emu_object<std::uint64_t> bytes, const bool to_remote) -> NTSTATUS
	{
		if (bytes)
			bytes.write(0);

		auto local = cpu.curr_addr_space();
		auto remote = local;

		if (process_handle && process_handle != current_process_handle)
		{
			const auto entry = st->sys_proc->handle_table().lookup_handle(process_handle);

			if (!entry)
				return STATUS_INVALID_HANDLE;

			const auto process = st->objs.get_object<process_object>(entry->body_addr);

			if (!process)
				return STATUS_OBJECT_TYPE_MISMATCH;

			const auto required = to_remote ? process_vm_write | process_vm_operation : process_vm_read;

			if ((entry->access & required) != required)
				return STATUS_ACCESS_DENIED;

			const auto target = st->find_process(static_cast<process::id_type>(process->pid));
			remote = target ? target->addr_space() : nullptr;

			if (!remote)
				return STATUS_PROCESS_IS_TERMINATING;
		}

		auto& from = to_remote ? *local : *remote;
		auto& to = to_remote ? *remote : *local;
		const auto from_va = to_remote ? local_va : remote_va;
		const auto to_va = to_remote ? remote_va : local_va;

		std::vector<std::uint8_t> buffer;
		std::uint64_t copied = 0;
		NTSTATUS status = STATUS_SUCCESS;

		while (copied < size && status == STATUS_SUCCESS)
		{
			buffer.resize(std::min<std::uint64_t>(0x10000, size - copied));

			try
			{
				from.read_mem(from_va + copied, buffer.data(), buffer.size());
			}
			catch (const std::exception&)
			{
				status = to_remote ? STATUS_ACCESS_VIOLATION : STATUS_PARTIAL_COPY;
				break;
			}

			try
			{
				to.write_mem(to_va + copied, buffer.data(), buffer.size());
			}
			catch (const std::exception&)
			{
				status = to_remote ? STATUS_PARTIAL_COPY : STATUS_ACCESS_VIOLATION;
				break;
			}

			copied += buffer.size();
		}

		if (bytes)
			bytes.write(copied);

		return status;
	};

	state.redirect_ntzw(mod, "ReadVirtualMemory",
		[copy_virtual_memory](vcpu& cpu, const std::uint64_t process_handle,
			const addr_t base_address, const addr_t buffer, const std::uint64_t size,
			emu_object<std::uint64_t> bytes_read) -> NTSTATUS
		{
			return copy_virtual_memory(cpu, process_handle, base_address, buffer, size,
				std::move(bytes_read), false);
		});

	state.redirect_ntzw(mod, "WriteVirtualMemory",
		[copy_virtual_memory](vcpu& cpu, const std::uint64_t process_handle,
			const addr_t base_address, const addr_t buffer, const std::uint64_t size,
			emu_object<std::uint64_t> bytes_written) -> NTSTATUS
		{
			return copy_virtual_memory(cpu, process_handle, base_address, buffer, size,
				std::move(bytes_written), true);
		});

	// Every class is a hint and nothing enforces control flow guard.
	state.redirect_ntzw(mod, "SetInformationVirtualMemory",
		[](vcpu&, const std::uint64_t process_handle,
			const std::uint32_t information_class, const std::uint64_t entry_count,
			const addr_t addresses, const addr_t information,
			const std::uint32_t information_length) -> NTSTATUS
		{
			THREAD_LOG_INFO("NtSetInformationVirtualMemory(handle=0x{:X}, class={}, entries={}, "
				"addresses=0x{:X}, information=0x{:X}/{}): accepted, and nothing here acts on it",
				process_handle, information_class, entry_count, addresses, information,
				information_length);

			return STATUS_SUCCESS;
		});
}
