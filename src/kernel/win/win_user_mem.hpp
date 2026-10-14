#pragma once
#include "status.hpp"
#include "../../emu/mmu.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>

namespace win
{

constexpr std::uint32_t mem_commit   = 0x00001000;
constexpr std::uint32_t mem_reserve  = 0x00002000;
constexpr std::uint32_t mem_decommit = 0x00004000;
constexpr std::uint32_t mem_release  = 0x00008000;
constexpr std::uint32_t mem_free     = 0x00010000;
constexpr std::uint32_t mem_private  = 0x00020000;
constexpr std::uint32_t mem_mapped   = 0x00040000;
constexpr std::uint32_t mem_image    = 0x01000000;

constexpr std::uint32_t page_noaccess          = 0x001;
constexpr std::uint32_t page_readonly          = 0x002;
constexpr std::uint32_t page_readwrite         = 0x004;
constexpr std::uint32_t page_writecopy         = 0x008;
constexpr std::uint32_t page_execute           = 0x010;
constexpr std::uint32_t page_execute_read      = 0x020;
constexpr std::uint32_t page_execute_readwrite = 0x040;
constexpr std::uint32_t page_execute_writecopy = 0x080;
constexpr std::uint32_t page_guard             = 0x100;

constexpr std::uint32_t page_prot_mask = 0xFF;

constexpr addr_t user_alloc_base  = 0x0000000100000000;
constexpr addr_t user_addr_limit  = 0x00007FFFFFFEFFFF;
constexpr std::size_t alloc_granularity = 0x10000;

struct memory_basic_info
{
	std::uint64_t base_address;
	std::uint64_t allocation_base;
	std::uint32_t allocation_protect;
	std::uint32_t partition_id;
	std::uint64_t region_size;
	std::uint32_t state;
	std::uint32_t protect;
	std::uint32_t type;
	std::uint32_t __unused_alignment;
};

static_assert(sizeof(memory_basic_info) == 0x30);

} // namespace win

// pages that must fault on access (PAGE_GUARD, PAGE_NOACCESS) are left unmapped and resolved lazily
class win_user_mem
{
public:
	explicit win_user_mem(std::shared_ptr<addr_space> space)
		:	space_(std::move(space)) { }

	NTSTATUS allocate(addr_t& base, std::size_t& size, std::uint32_t alloc_type,
		std::uint32_t prot, std::size_t alignment = 0);
	NTSTATUS free(addr_t& base, std::size_t& size, std::uint32_t free_type);
	NTSTATUS protect(addr_t& base, std::size_t& size, std::uint32_t new_prot,
		std::uint32_t& old_prot);
	NTSTATUS query(addr_t addr, win::memory_basic_info& info) const;

	addr_t alloc_pages(std::size_t size, std::uint32_t prot = win::page_readwrite);

	// tracked stand-in for addr_space::alloc, so loader allocations do not race its bump allocator
	addr_t alloc(std::size_t size, mem_prot prot = prot_rw);

	[[nodiscard]] addr_space& space() const { return *space_; }

	void read_mem(addr_t va, void* buf, std::size_t size) const { space_->read_mem(va, buf, size); }
	void write_mem(addr_t va, const void* buf, std::size_t size) { space_->write_mem(va, buf, size); }

	template <typename T>
	[[nodiscard]] T read_mem(addr_t va) const { return space_->read_mem<T>(va); }

	template <typename T>
	void write_mem(addr_t va, const T& val) { space_->write_mem<T>(va, val); }

	// Maps physical pages a section owns, so every process that maps it sees the same bytes.
	NTSTATUS map_view(addr_t& base, std::size_t& size, addr_t pa, std::uint32_t prot);
	NTSTATUS unmap_view(addr_t base, std::size_t& size);

	void register_image(addr_t base, std::size_t size);
	void register_mapped(addr_t base, std::size_t size, std::uint32_t prot);

	bool handle_fault(addr_t fault_addr);

private:
	struct committed_region
	{
		std::size_t size;
		std::uint32_t prot;
	};

	struct reservation
	{
		std::size_t size;
		std::uint32_t initial_prot;
		std::uint32_t type;
		std::map<addr_t, committed_region> committed;
	};

	using committed_map = std::map<addr_t, committed_region>;
	using reservation_map = std::map<addr_t, reservation>;

	[[nodiscard]] std::size_t page_size() const { return space_->mmu_->page_size(); }
	[[nodiscard]] addr_t page_align(addr_t addr) const { return addr & ~(page_size() - 1); }
	[[nodiscard]] std::size_t size_align(std::size_t size) const { return (size + page_size() - 1) & ~(page_size() - 1); }

	[[nodiscard]] reservation_map::iterator find_reservation(addr_t addr);
	[[nodiscard]] reservation_map::const_iterator find_reservation(addr_t addr) const;
	[[nodiscard]] reservation_map::iterator overlapping(addr_t base, std::size_t size);

	NTSTATUS commit_into(reservation_map::iterator res, addr_t base, std::size_t size, std::uint32_t prot);
	addr_t pick_base(std::size_t size, std::size_t alignment);

	void map_pages(addr_t base, std::size_t size, std::uint32_t prot);
	void evict_pages(addr_t base, std::size_t size);
	void release_pages(addr_t base, std::size_t size);
	void commit_pages(addr_t base, std::size_t size, std::uint32_t prot);
	void apply_prot(addr_t base, std::size_t size, std::uint32_t old_prot, std::uint32_t new_prot);

	static committed_map::iterator find_committed(committed_map& committed, addr_t addr);
	static committed_map::const_iterator find_committed(const committed_map& committed, addr_t addr);
	static void split_at(committed_map& committed, addr_t addr);
	static void merge_adjacent(committed_map& committed);

	static bool is_resident(std::uint32_t prot);
	static mem_prot to_mem_prot(std::uint32_t prot);
	static std::uint32_t to_win_prot(mem_prot prot);

	std::shared_ptr<addr_space> space_;
	addr_t next_free_ = win::user_alloc_base;
	reservation_map reservations_;
	// kept alive while their virtual page is unmapped, so arming a guard page does not discard them
	std::map<addr_t, addr_t> backing_;
	mutable std::mutex mtx_;
};
