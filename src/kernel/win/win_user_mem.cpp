#include "win_user_mem.hpp"
#include "../../util/log.hpp"

#include <algorithm>

bool win_user_mem::is_resident(const std::uint32_t prot)
{
	return !(prot & win::page_guard) && (prot & win::page_prot_mask) != win::page_noaccess;
}

mem_prot win_user_mem::to_mem_prot(const std::uint32_t prot)
{
	switch (prot & win::page_prot_mask)
	{
	case win::page_noaccess:               return prot_none;
	case win::page_readonly:               return prot_read;
	case win::page_execute:                return prot_exec;
	case win::page_execute_read:           return prot_rx;
	case win::page_execute_readwrite:
	case win::page_execute_writecopy:      return prot_rwx;
	case win::page_readwrite:
	case win::page_writecopy:
	default:                               return prot_rw;
	}
}

std::uint32_t win_user_mem::to_win_prot(const mem_prot prot)
{
	if (prot & prot_exec)
	{
		if (prot & prot_write) return win::page_execute_readwrite;
		return (prot & prot_read) ? win::page_execute_read : win::page_execute;
	}

	if (prot & prot_write)
		return win::page_readwrite;

	return (prot & prot_read) ? win::page_readonly : win::page_noaccess;
}

win_user_mem::reservation_map::iterator win_user_mem::find_reservation(const addr_t addr)
{
	const auto upper = reservations_.upper_bound(addr);

	if (upper == reservations_.begin())
		return reservations_.end();

	const auto entry = std::prev(upper);

	return addr < entry->first + entry->second.size ? entry : reservations_.end();
}

win_user_mem::reservation_map::const_iterator win_user_mem::find_reservation(const addr_t addr) const
{
	return const_cast<win_user_mem*>(this)->find_reservation(addr);
}

win_user_mem::committed_map::iterator win_user_mem::find_committed(committed_map& committed, const addr_t addr)
{
	const auto upper = committed.upper_bound(addr);

	if (upper == committed.begin())
		return committed.end();

	const auto entry = std::prev(upper);

	return addr < entry->first + entry->second.size ? entry : committed.end();
}

win_user_mem::committed_map::const_iterator win_user_mem::find_committed(const committed_map& committed, const addr_t addr)
{
	const auto upper = committed.upper_bound(addr);

	if (upper == committed.begin())
		return committed.end();

	const auto entry = std::prev(upper);

	return addr < entry->first + entry->second.size ? entry : committed.end();
}

void win_user_mem::split_at(committed_map& committed, const addr_t addr)
{
	const auto it = find_committed(committed, addr);

	if (it == committed.end() || it->first == addr)
		return;

	const auto tail = it->first + it->second.size - addr;
	it->second.size -= tail;

	committed[addr] = committed_region{ .size = tail, .prot = it->second.prot };
}

void win_user_mem::merge_adjacent(committed_map& committed)
{
	for (auto it = committed.begin(); it != committed.end();)
	{
		const auto next = std::next(it);

		if (next != committed.end()
			&& it->first + it->second.size == next->first
			&& it->second.prot == next->second.prot)
		{
			it->second.size += next->second.size;
			committed.erase(next);
			continue;
		}

		++it;
	}
}

void win_user_mem::map_pages(const addr_t base, const std::size_t size, const std::uint32_t prot)
{
	auto& mmu = *space_->mmu_;
	const auto emu_prot = to_mem_prot(prot);

	const auto first = backing_.lower_bound(base);

	if (first == backing_.end() || first->first >= base + size)
	{
		mmu.map_virt(*space_, base, size, emu_prot);
		return;
	}

	for (std::size_t off = 0; off < size; off += page_size())
	{
		const auto page = base + off;
		const auto it = backing_.find(page);

		if (it == backing_.end())
		{
			mmu.map_virt(*space_, page, page_size(), emu_prot);
			continue;
		}

		mmu.map_virt_phys(*space_, page, it->second, page_size(), emu_prot);
		backing_.erase(it);
	}
}

void win_user_mem::evict_pages(const addr_t base, const std::size_t size)
{
	auto& mmu = *space_->mmu_;

	for (std::size_t off = 0; off < size; off += page_size())
	{
		const auto page = base + off;

		if (const auto pa = mmu.virt_to_phys(*space_, page))
			backing_[page] = *pa;
	}

	mmu.unmap_virt(*space_, base, size);
}

void win_user_mem::release_pages(const addr_t base, const std::size_t size)
{
	for (std::size_t off = 0; off < size; off += page_size())
		backing_.erase(base + off);

	space_->mmu_->unmap_virt(*space_, base, size);
}

void win_user_mem::commit_pages(const addr_t base, const std::size_t size, const std::uint32_t prot)
{
	if (is_resident(prot))
		map_pages(base, size, prot);
}

void win_user_mem::apply_prot(const addr_t base, const std::size_t size,
	const std::uint32_t old_prot, const std::uint32_t new_prot)
{
	const bool was_resident = is_resident(old_prot);
	const bool now_resident = is_resident(new_prot);

	if (was_resident && !now_resident)
		evict_pages(base, size);
	else if (!was_resident && now_resident)
		map_pages(base, size, new_prot);
	else if (now_resident)
		space_->mmu_->prot_virt(*space_, base, size, to_mem_prot(new_prot));
}

win_user_mem::reservation_map::iterator win_user_mem::overlapping(const addr_t base, const std::size_t size)
{
	if (const auto it = find_reservation(base); it != reservations_.end())
		return it;

	const auto next = reservations_.lower_bound(base);

	return next != reservations_.end() && next->first < base + size ? next : reservations_.end();
}

addr_t win_user_mem::pick_base(const std::size_t size, const std::size_t alignment)
{
	const auto mask = std::max(alignment, win::alloc_granularity) - 1;

	for (auto candidate = (next_free_ + mask) & ~mask; candidate + size <= win::user_addr_limit;)
	{
		const auto clash = overlapping(candidate, size);

		if (clash == reservations_.end())
		{
			next_free_ = candidate + size;
			return candidate;
		}

		candidate = (clash->first + clash->second.size + mask) & ~mask;
	}

	return 0;
}

NTSTATUS win_user_mem::commit_into(const reservation_map::iterator res, const addr_t base,
	const std::size_t size, const std::uint32_t prot)
{
	const auto end = base + size;

	if (end > res->first + res->second.size)
	{
		LOG_WARN("win_user_mem: commit 0x{:X}+0x{:X} exceeds reservation 0x{:X}+0x{:X}",
			base, size, res->first, res->second.size);
		return STATUS_CONFLICTING_ADDRESSES;
	}

	auto& committed = res->second.committed;

	split_at(committed, base);
	split_at(committed, end);

	auto cursor = base;

	for (auto it = committed.lower_bound(base); it != committed.end() && it->first < end; ++it)
	{
		if (it->first > cursor)
		{
			const auto gap = it->first - cursor;
			commit_pages(cursor, gap, prot);
			committed[cursor] = committed_region{ .size = gap, .prot = prot };
		}

		apply_prot(it->first, it->second.size, it->second.prot, prot);
		it->second.prot = prot;
		cursor = it->first + it->second.size;
	}

	if (cursor < end)
	{
		const auto gap = end - cursor;
		commit_pages(cursor, gap, prot);
		committed[cursor] = committed_region{ .size = gap, .prot = prot };
	}

	merge_adjacent(committed);

	LOG_INFO("win_user_mem: committed 0x{:X}+0x{:X} (prot=0x{:X})", base, size, prot);

	return STATUS_SUCCESS;
}

NTSTATUS win_user_mem::allocate(addr_t& base, std::size_t& size, const std::uint32_t alloc_type,
	const std::uint32_t prot, const std::size_t alignment)
{
	std::scoped_lock lock(mtx_);

	const auto start = page_align(base);
	const auto region = size_align(size + (base - start));

	if (!region)
		return STATUS_INVALID_PARAMETER;

	const bool do_reserve = (alloc_type & win::mem_reserve) != 0;
	const bool do_commit = (alloc_type & win::mem_commit) != 0;

	if (!do_reserve && !do_commit)
		return STATUS_INVALID_PARAMETER;

	if (do_commit && !do_reserve && base)
	{
		if (const auto res = find_reservation(start); res != reservations_.end())
		{
			const auto status = commit_into(res, start, region, prot);

			if (status == STATUS_SUCCESS)
			{
				base = start;
				size = region;
			}

			return status;
		}

		LOG_INFO("win_user_mem: commit at 0x{:X} has no reservation, reserving it", start);
	}

	const auto target = base ? (do_reserve ? base & ~(win::alloc_granularity - 1) : start)
							 : pick_base(region, alignment);

	if (!target || target + region > win::user_addr_limit)
		return STATUS_NO_MEMORY;

	if (const auto conflict = overlapping(target, region); conflict != reservations_.end())
	{
		LOG_WARN("win_user_mem: reserve 0x{:X}+0x{:X} conflicts with 0x{:X}+0x{:X}",
			target, region, conflict->first, conflict->second.size);
		return STATUS_CONFLICTING_ADDRESSES;
	}

	auto& res = reservations_[target];
	res.size = region;
	res.initial_prot = prot;
	res.type = win::mem_private;

	if (do_commit)
	{
		commit_pages(target, region, prot);
		res.committed[target] = committed_region{ .size = region, .prot = prot };
	}

	base = target;
	size = region;

	LOG_INFO("win_user_mem: {} 0x{:X}+0x{:X} (prot=0x{:X})",
		do_commit ? "allocated" : "reserved", target, region, prot);

	return STATUS_SUCCESS;
}

NTSTATUS win_user_mem::free(addr_t& base, std::size_t& size, const std::uint32_t free_type)
{
	std::scoped_lock lock(mtx_);

	const auto start = page_align(base);
	const auto res = find_reservation(start);

	if (free_type & win::mem_release)
	{
		if (size)
			return STATUS_INVALID_PARAMETER;

		if (res == reservations_.end() || res->first != start)
			return STATUS_UNABLE_TO_FREE_VM;

		if (res->second.type != win::mem_private)
			return STATUS_UNABLE_TO_FREE_VM;

		for (const auto& [sub_base, sub] : res->second.committed)
			release_pages(sub_base, sub.size);

		base = res->first;
		size = res->second.size;

		LOG_INFO("win_user_mem: released 0x{:X}+0x{:X}", base, size);

		reservations_.erase(res);

		return STATUS_SUCCESS;
	}

	if (!(free_type & win::mem_decommit))
		return STATUS_INVALID_PARAMETER;

	if (res == reservations_.end())
		return STATUS_MEMORY_NOT_ALLOCATED;

	auto region = size_align(size + (base - start));

	if (!region)
	{
		if (res->first != start)
			return STATUS_UNABLE_TO_FREE_VM;

		region = res->second.size;
	}

	const auto end = start + region;
	auto& committed = res->second.committed;

	split_at(committed, start);
	split_at(committed, end);

	for (auto it = committed.lower_bound(start); it != committed.end() && it->first < end;)
	{
		release_pages(it->first, it->second.size);
		it = committed.erase(it);
	}

	base = start;
	size = region;

	LOG_INFO("win_user_mem: decommitted 0x{:X}+0x{:X}", start, region);

	return STATUS_SUCCESS;
}

NTSTATUS win_user_mem::protect(addr_t& base, std::size_t& size, const std::uint32_t new_prot,
	std::uint32_t& old_prot)
{
	std::scoped_lock lock(mtx_);

	const auto start = page_align(base);
	const auto region = size_align(size + (base - start));
	const auto end = start + region;

	old_prot = win::page_readwrite;

	const auto res = find_reservation(start);

	if (res == reservations_.end())
	{
		space_->mmu_->prot_virt(*space_, start, region, to_mem_prot(new_prot));

		base = start;
		size = region;

		LOG_INFO("win_user_mem: protect 0x{:X}+0x{:X} -> 0x{:X} (untracked)", start, region, new_prot);

		return STATUS_SUCCESS;
	}

	auto& committed = res->second.committed;
	const auto first = find_committed(committed, start);

	if (first == committed.end())
	{
		old_prot = res->second.initial_prot;
		base = start;
		size = region;

		LOG_INFO("win_user_mem: protect 0x{:X}+0x{:X} -> 0x{:X} (reserved)", start, region, new_prot);

		return STATUS_SUCCESS;
	}

	old_prot = first->second.prot;

	split_at(committed, start);
	split_at(committed, end);

	for (auto it = committed.lower_bound(start); it != committed.end() && it->first < end; ++it)
	{
		apply_prot(it->first, it->second.size, it->second.prot, new_prot);
		it->second.prot = new_prot;
	}

	merge_adjacent(committed);

	base = start;
	size = region;

	LOG_INFO("win_user_mem: protect 0x{:X}+0x{:X}: 0x{:X} -> 0x{:X}", start, region, old_prot, new_prot);

	return STATUS_SUCCESS;
}

NTSTATUS win_user_mem::query(const addr_t addr, win::memory_basic_info& info) const
{
	std::scoped_lock lock(mtx_);

	info = {};
	info.base_address = page_align(addr);

	const auto res = find_reservation(addr);

	if (res == reservations_.end())
	{
		const auto next = reservations_.upper_bound(addr);

		info.state = win::mem_free;
		info.protect = win::page_noaccess;
		info.region_size = (next != reservations_.end() ? next->first : win::user_addr_limit + 1)
			- info.base_address;

		return STATUS_SUCCESS;
	}

	info.allocation_base = res->first;
	info.allocation_protect = res->second.initial_prot;
	info.type = res->second.type;

	const auto& committed = res->second.committed;

	if (const auto sub = find_committed(committed, addr); sub != committed.end())
	{
		info.state = win::mem_commit;
		info.protect = sub->second.prot;
		info.region_size = sub->first + sub->second.size - info.base_address;

		return STATUS_SUCCESS;
	}

	const auto next = committed.upper_bound(info.base_address);

	info.state = win::mem_reserve;
	info.protect = win::page_noaccess;
	info.region_size = (next != committed.end() ? next->first : res->first + res->second.size)
		- info.base_address;

	return STATUS_SUCCESS;
}

addr_t win_user_mem::alloc_pages(const std::size_t size, const std::uint32_t prot)
{
	addr_t base = 0;
	auto region = size;

	if (allocate(base, region, win::mem_commit | win::mem_reserve, prot) != STATUS_SUCCESS)
		return 0;

	return base;
}

addr_t win_user_mem::alloc(const std::size_t size, const mem_prot prot)
{
	return alloc_pages(size, to_win_prot(prot));
}

NTSTATUS win_user_mem::map_view(addr_t& base, std::size_t& size, const addr_t pa,
	const std::uint32_t prot)
{
	std::scoped_lock lock(mtx_);

	const auto region = size_align(size);

	if (!region)
		return STATUS_INVALID_PARAMETER;

	const auto target = base ? base : pick_base(region, 0);

	if (!target || target + region > win::user_addr_limit)
		return STATUS_NO_MEMORY;

	if (base && (base & (win::alloc_granularity - 1)))
		return STATUS_MAPPED_ALIGNMENT;

	if (overlapping(target, region) != reservations_.end())
		return STATUS_CONFLICTING_ADDRESSES;

	auto& res = reservations_[target];
	res.size = region;
	res.initial_prot = prot;
	res.type = win::mem_mapped;
	res.committed[target] = committed_region{ .size = region, .prot = prot };

	space_->mmu_->map_virt_phys(*space_, target, pa, region, to_mem_prot(prot));

	base = target;
	size = region;

	LOG_INFO("win_user_mem: mapped view 0x{:X}+0x{:X} of 0x{:X} (prot=0x{:X})",
		target, region, pa, prot);

	return STATUS_SUCCESS;
}

NTSTATUS win_user_mem::unmap_view(const addr_t base, std::size_t& size)
{
	std::scoped_lock lock(mtx_);

	// Any address inside the view names it, and the whole view goes.
	const auto res = find_reservation(base);

	if (res == reservations_.end() || res->second.type != win::mem_mapped)
		return STATUS_NOT_MAPPED_VIEW;

	size = res->second.size;

	for (const auto& [sub_base, sub] : res->second.committed)
		release_pages(sub_base, sub.size);

	reservations_.erase(res);

	return STATUS_SUCCESS;
}

void win_user_mem::register_image(const addr_t base, const std::size_t size)
{
	std::scoped_lock lock(mtx_);

	const auto region = size_align(size);

	auto& res = reservations_[base];
	res.size = region;
	res.initial_prot = win::page_execute_readwrite;
	res.type = win::mem_image;
	res.committed.clear();
	res.committed[base] = committed_region{ .size = region, .prot = win::page_execute_readwrite };

	LOG_INFO("win_user_mem: registered image 0x{:X}+0x{:X}", base, region);
}

void win_user_mem::register_mapped(const addr_t base, const std::size_t size, const std::uint32_t prot)
{
	std::scoped_lock lock(mtx_);

	const auto region = size_align(size);

	auto& res = reservations_[base];
	res.size = region;
	res.initial_prot = prot;
	res.type = win::mem_mapped;
	res.committed.clear();
	res.committed[base] = committed_region{ .size = region, .prot = prot };

	LOG_INFO("win_user_mem: registered mapping 0x{:X}+0x{:X} (prot=0x{:X})", base, region, prot);
}

bool win_user_mem::handle_fault(const addr_t fault_addr)
{
	std::scoped_lock lock(mtx_);

	const auto page = page_align(fault_addr);
	const auto res = find_reservation(page);

	if (res == reservations_.end())
		return false;

	auto& committed = res->second.committed;
	const auto sub = find_committed(committed, page);

	// reserved but never committed: backed on demand, so lazily grown stacks and heaps keep running
	if (sub == committed.end())
	{
		const auto prot = res->second.initial_prot;

		if (!is_resident(prot))
			return false;

		map_pages(page, page_size(), prot);
		committed[page] = committed_region{ .size = page_size(), .prot = prot };
		merge_adjacent(committed);

		LOG_INFO("win_user_mem: demand committed 0x{:X} in 0x{:X}+0x{:X}",
			page, res->first, res->second.size);

		return true;
	}

	// PAGE_NOACCESS, or a genuine violation against a mapped page
	if (!(sub->second.prot & win::page_guard))
		return false;

	split_at(committed, page);
	split_at(committed, page + page_size());

	const auto hit = committed.find(page);
	hit->second.prot &= ~win::page_guard;

	map_pages(page, page_size(), hit->second.prot);
	merge_adjacent(committed);

	LOG_INFO("win_user_mem: guard page hit at 0x{:X}, cleared guard (prot=0x{:X})",
		page, hit->second.prot);

	return true;
}
