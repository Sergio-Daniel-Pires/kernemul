#include "nt_pool_ops.hpp"
#include "../win_kernel.hpp"
#include "../pool.hpp"
#include "../types.hpp"
#include "../../../util/log.hpp"
#include <string_view>

namespace
{

constexpr std::uint64_t pool_flag_use_quota        = 0x001;
constexpr std::uint64_t pool_flag_uninitialized    = 0x002;
constexpr std::uint64_t pool_flag_session          = 0x004;
constexpr std::uint64_t pool_flag_cache_aligned    = 0x008;
constexpr std::uint64_t pool_flag_reserved1        = 0x010;
constexpr std::uint64_t pool_flag_raise_on_failure = 0x020;
constexpr std::uint64_t pool_flag_non_paged        = 0x040;
constexpr std::uint64_t pool_flag_non_paged_exec   = 0x080;
constexpr std::uint64_t pool_flag_paged            = 0x100;
constexpr std::uint64_t pool_flag_special_pool     = 0x200;
constexpr std::uint64_t pool_flag_reserved3        = 0x400;
constexpr std::uint64_t pool_flag_reserved4        = 0x800;

constexpr std::uint64_t pool_flag_type_mask = pool_flag_non_paged
	| pool_flag_non_paged_exec | pool_flag_paged;

// A caller that passed no tag is given 'one0', how an untagged allocation shows in a pool dump.
constexpr std::uint32_t default_pool_tag = 0x30656E6F;

std::uint32_t pool_type_tag(const std::uint32_t tag)
{
	return (tag & 0x7FFFFFFF) ? (tag & 0x7FFFFFFF) : default_pool_tag;
}

// The translation below is the one in the binary.
constexpr std::uint32_t pool_type_paged           = 0x001;
constexpr std::uint32_t pool_type_cache_aligned   = 0x004;
constexpr std::uint32_t pool_type_raise_on_failure = 0x010;
constexpr std::uint32_t pool_type_session         = 0x020;
constexpr std::uint32_t pool_type_reserved3       = 0x040;
constexpr std::uint32_t pool_type_special         = 0x080;
constexpr std::uint32_t pool_type_nx              = 0x200;
constexpr std::uint32_t pool_type_zero            = 0x400;

std::uint64_t pool_flags_from_type(const std::uint32_t pool_type)
{
	auto flags = (pool_type & pool_type_paged) ? pool_flag_paged
		: (pool_type & pool_type_nx) ? pool_flag_non_paged
		: pool_flag_non_paged_exec;

	if (pool_type & pool_type_session)
		flags |= pool_flag_session;

	if (!(pool_type & pool_type_zero))
		flags |= pool_flag_uninitialized;

	if (pool_type & pool_type_cache_aligned)
		flags |= pool_flag_cache_aligned;

	if (pool_type & pool_type_special)
		flags |= pool_flag_special_pool;

	if (pool_type & pool_type_reserved3)
		flags |= pool_flag_reserved3;

	if (pool_type & pool_type_raise_on_failure)
		flags |= pool_flag_raise_on_failure;

	return flags;
}

bool flags_are_valid(const std::uint64_t flags, const std::uint32_t tag)
{
	const auto type = flags & pool_flag_type_mask;

	return type != 0
		&& (type & (type - 1)) == 0
		&& (flags & 0xFFFFF000) == 0
		&& (flags & pool_flag_reserved1) == 0
		&& (flags & pool_flag_reserved4) == 0
		&& tag != 0;
}

addr_t allocate(win_kernel_state& state, const std::string_view who, const std::uint64_t flags,
	const std::uint64_t size, const std::uint32_t tag)
{
	if (!flags_are_valid(flags, tag))
	{
		// Nothing here delivers a guest exception, so a caller asking to be raised at gets null.
		THREAD_LOG_ERR("{}: invalid pool flags 0x{:X} for tag '{}'",
			who, flags, pool_tag_name(tag));

		return 0;
	}

	const auto addr = state.pool.allocate(size, tag,
		(flags & pool_flag_uninitialized) == 0);

	if (!addr)
	{
		THREAD_LOG_ERR("{}: out of pool for {} bytes (tag '{}')",
			who, size, pool_tag_name(tag));

		return 0;
	}

	THREAD_LOG_INFO("{}(flags=0x{:X}, size={}, tag='{}') -> 0x{:X}",
		who, flags, size, pool_tag_name(tag), addr);

	return addr;
}

void release(win_kernel_state& state, const std::string_view who, const addr_t address,
	const std::uint32_t tag)
{
	if (!address)
	{
		THREAD_LOG_ERR("{}: null pointer", who);
		return;
	}

	const auto freed = state.pool.free(address);

	if (!freed)
	{
		// Real NT bugchecks with BAD_POOL_CALLER here; the guest is left standing instead.
		THREAD_LOG_ERR("{}: 0x{:X} is not a live pool allocation", who, address);
		return;
	}

	if (tag && tag != freed->tag)
		THREAD_LOG_ERR("{}: 0x{:X} was allocated with tag '{}', freed with '{}'",
			who, address, pool_tag_name(freed->tag), pool_tag_name(tag));

	THREAD_LOG_INFO("{}(0x{:X}): {} bytes, tag '{}'",
		who, address, freed->size, pool_tag_name(freed->tag));
}

}

// All five allocation entry points are the same call underneath, where the validation lives.
void modules::register_ntoskrnl_pool_ops(win_kernel_state& state, proc_module& mod)
{
	auto* st = &state;

	state.redirect(mod, "ExAllocatePool2",
		[st](vcpu&, const std::uint64_t flags, const std::uint64_t size,
			const std::uint32_t tag) -> addr_t
		{
			return allocate(*st, "ExAllocatePool2", flags, size, tag);
		});

	state.redirect(mod, "ExAllocatePool3",
		[st](vcpu&, const std::uint64_t flags, const std::uint64_t size,
			const std::uint32_t tag, const addr_t extended_parameters,
			const std::uint32_t extended_parameter_count) -> addr_t
		{
			if (extended_parameter_count)
				THREAD_LOG_WARN("ExAllocatePool3: ignoring {} extended parameter(s) at 0x{:X}",
					extended_parameter_count, extended_parameters);

			return allocate(*st, "ExAllocatePool3", flags, size, tag);
		});

	state.redirect(mod, "ExAllocatePoolWithTag",
		[st](vcpu&, const std::uint32_t pool_type, const std::uint64_t size,
			const std::uint32_t tag) -> addr_t
		{
			return allocate(*st, "ExAllocatePoolWithTag",
				pool_flags_from_type(pool_type), size, pool_type_tag(tag));
		});

	state.redirect(mod, "ExAllocatePoolWithQuotaTag",
		[st](vcpu&, const std::uint32_t pool_type, const std::uint64_t size,
			const std::uint32_t tag) -> addr_t
		{
			return allocate(*st, "ExAllocatePoolWithQuotaTag",
				pool_flags_from_type(pool_type) | pool_flag_use_quota, size, pool_type_tag(tag));
		});

	state.redirect(mod, "ExAllocatePool",
		[st](vcpu&, const std::uint32_t pool_type, const std::uint64_t size) -> addr_t
		{
			return allocate(*st, "ExAllocatePool",
				pool_flags_from_type(pool_type), size, default_pool_tag);
		});

	state.redirect(mod, "ExFreePoolWithTag",
		[st](vcpu&, const addr_t address, const std::uint32_t tag)
		{
			release(*st, "ExFreePoolWithTag", address, tag & 0x7FFFFFFF);
		});

	state.redirect(mod, "ExFreePool", [st](vcpu&, const addr_t address)
	{
		release(*st, "ExFreePool", address, 0);
	});

	// GENERAL_LOOKASIDE (x64): Type +0x24, Tag +0x28, Size +0x2C.
	// The packer resolves these dynamically and the driver never calls
	// ExInitializePagedLookasideList, so the fields can still be zero here.
	const auto lookaside_allocate = [st](vcpu& cpu, const std::string_view who,
		const addr_t lookaside) -> addr_t
	{
		std::uint32_t type = 0;
		std::uint32_t tag = 0;
		std::uint32_t size = 0;
		try
		{
			cpu.read_virt_mem(lookaside + 0x24, &type, sizeof(type));
			cpu.read_virt_mem(lookaside + 0x28, &tag, sizeof(tag));
			cpu.read_virt_mem(lookaside + 0x2C, &size, sizeof(size));
		}
		catch (...)
		{
		}

		if (size == 0 || size > 0x100000)
		{
			THREAD_LOG_WARN("{}: lookaside 0x{:X} has Size={} Type=0x{:X} Tag='{}'; using 0x100",
				who, lookaside, size, type, pool_tag_name(tag));
			size = 0x100;
		}

		const auto flags = (type & pool_type_paged)
			? pool_flags_from_type(type)
			: (type ? pool_flags_from_type(type) : pool_flag_paged);

		return allocate(*st, who, flags, size, pool_type_tag(tag));
	};

	state.redirect(mod, "ExAllocateFromPagedLookasideList",
		[lookaside_allocate](vcpu& cpu, const addr_t lookaside) -> addr_t
		{
			return lookaside_allocate(cpu, "ExAllocateFromPagedLookasideList",
				lookaside);
		});

	state.redirect(mod, "ExAllocateFromNPagedLookasideList",
		[lookaside_allocate](vcpu& cpu, const addr_t lookaside) -> addr_t
		{
			return lookaside_allocate(cpu, "ExAllocateFromNPagedLookasideList",
				lookaside);
		});

	const auto lookaside_free = [st](const std::string_view who,
		const addr_t entry)
	{
		release(*st, who, entry, 0);
	};

	state.redirect(mod, "ExFreeToPagedLookasideList",
		[lookaside_free](vcpu&, const addr_t, const addr_t entry)
		{
			lookaside_free("ExFreeToPagedLookasideList", entry);
		});

	state.redirect(mod, "ExFreeToNPagedLookasideList",
		[lookaside_free](vcpu&, const addr_t, const addr_t entry)
		{
			lookaside_free("ExFreeToNPagedLookasideList", entry);
		});
}
