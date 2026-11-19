#pragma once
#include "../addr_space.hpp"
#include <unordered_map>

namespace x86
{
	union virt_addr
	{
		std::uint64_t val;
		struct
		{
			std::uint64_t offset : 12;
			std::uint64_t pt     : 9;
			std::uint64_t pd     : 9;
			std::uint64_t pdpt   : 9;
			std::uint64_t pml4   : 9;
			std::uint64_t reserved : 16;
		};
	};

	struct addr_space : ::addr_space
	{
		addr_t alloc(std::size_t size, mem_prot prot) override;
		addr_t alloc_at(addr_t va, std::size_t size, mem_prot prot) override;
		addr_t map_phys(addr_t pa, std::size_t size, mem_prot prot) override;

		addr_t pml4_pa = 0;
		std::unordered_map<addr_t, addr_t> shadow;
		addr_t user_next_ = 0x10000;
		addr_t kernel_next_ = 0xFFFFF80000000000;
	};
}
