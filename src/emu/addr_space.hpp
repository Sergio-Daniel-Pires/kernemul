#pragma once
#include "defs.hpp"
#include <cstddef>

class mmu;

struct addr_space
{
	virtual ~addr_space() = default;
	virtual addr_t alloc(std::size_t size, mem_prot prot) = 0;

	// Map at an explicit VA. Used to reproduce a relocated module (an image
	// loaded at its PE ImageBase instead of the kernel-space bump allocator).
	virtual addr_t alloc_at(addr_t va, std::size_t size, mem_prot prot)
	{
		(void)va; (void)size; (void)prot;
		return 0;
	}

	// Both `pa` and `size` are page aligned by the caller.
	virtual addr_t map_phys(addr_t pa, std::size_t size, mem_prot prot) = 0;

	// Backs `pa` and maps it, for an address the machine has rather than one this allocated.
	addr_t alloc_phys(addr_t pa, std::size_t size, mem_prot prot);

	void read_mem(addr_t va, void* buf, std::size_t size);
	void write_mem(addr_t va, const void* buf, std::size_t size);
	void prot_mem(addr_t va, std::size_t size, mem_prot prot);

	template <typename T>
	T read_mem(addr_t va)
	{
		T val{};
		read_mem(va, &val, sizeof(T));
		return val;
	}

	template <typename T>
	void write_mem(addr_t va, const T& val)
	{
		write_mem(va, &val, sizeof(T));
	}

	mmu* mmu_ = nullptr;
};
