#include "nt_sysinfo_ops.hpp"
#include "../win_kernel.hpp"
#include "../objects.hpp"
#include "../filesystem.hpp"
#include "../process_params.hpp"
#include "../status.hpp"
#include "../types.hpp"
#include "../../../util/log.hpp"
#include <chrono>
#include <string>
#include <string_view>
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

namespace
{

enum system_information_class : std::uint32_t
{
	system_basic_information        = 0,
	system_process_information      = 5,
	system_time_of_day_information  = 3,
	system_numa_processor_map       = 55,
	system_module_information       = 0x0B,
	system_module_information_ex    = 0x4D,
	system_kernel_debugger_information    = 0x23,
	system_kernel_debugger_information_ex = 0x42,
	system_code_integrity_policy_information = 0x5A,
	system_code_integrity_information     = 0x67,

	system_logical_processor_and_group_information = 107,
	system_shadow_stack_information = 183,

	// Nothing is emulated here, so this is the basic information verbatim.
	system_emulation_basic_information = 62,
};

// Neither shape is in ntoskrnl's pdb -- they are what ntdll and drivers agree the query
// returns -- so the sizes are asserted rather than trusted.
#pragma pack(push, 8)
struct rtl_process_module_information_t
{
	std::uint64_t section;
	std::uint64_t mapped_base;
	std::uint64_t image_base;
	std::uint32_t image_size;
	std::uint32_t flags;
	std::uint16_t load_order_index;
	std::uint16_t init_order_index;
	std::uint16_t load_count;
	std::uint16_t offset_to_file_name;
	std::uint8_t  full_path_name[256];
};

struct rtl_process_modules_t
{
	std::uint32_t number_of_modules;
	std::uint32_t padding;
	rtl_process_module_information_t modules[1];
};

struct rtl_process_module_information_ex_t
{
	std::uint16_t next_offset;
	std::uint8_t  padding[6];
	rtl_process_module_information_t base_info;
	std::uint32_t image_checksum;
	std::uint32_t time_date_stamp;
	std::uint64_t default_base;
};

struct system_thread_information_t
{
	std::int64_t  kernel_time;
	std::int64_t  user_time;
	std::int64_t  create_time;
	std::uint32_t wait_time;
	std::uint32_t padding0;
	std::uint64_t start_address;
	std::uint64_t unique_process_id;
	std::uint64_t unique_thread_id;
	std::int32_t  priority;
	std::int32_t  base_priority;
	std::uint32_t context_switches;
	std::uint32_t thread_state;
	std::uint32_t wait_reason;
	std::uint32_t padding1;
};

struct system_process_information_t
{
	std::uint32_t next_entry_offset;
	std::uint32_t number_of_threads;
	std::int64_t  working_set_private_size;
	std::uint32_t hard_fault_count;
	std::uint32_t number_of_threads_high_watermark;
	std::uint64_t cycle_time;
	std::int64_t  create_time;
	std::int64_t  user_time;
	std::int64_t  kernel_time;
	_UNICODE_STRING image_name;
	std::int32_t  base_priority;
	std::uint32_t padding0;
	std::uint64_t unique_process_id;
	std::uint64_t inherited_from_unique_process_id;
	std::uint32_t handle_count;
	std::uint32_t session_id;
	std::uint64_t unique_process_key;
	std::uint64_t peak_virtual_size;
	std::uint64_t virtual_size;
	std::uint32_t page_fault_count;
	std::uint32_t padding1;
	std::uint64_t peak_working_set_size;
	std::uint64_t working_set_size;
	std::uint64_t quota_peak_paged_pool_usage;
	std::uint64_t quota_paged_pool_usage;
	std::uint64_t quota_peak_non_paged_pool_usage;
	std::uint64_t quota_non_paged_pool_usage;
	std::uint64_t pagefile_usage;
	std::uint64_t peak_pagefile_usage;
	std::uint64_t private_page_count;
	std::int64_t  read_operation_count;
	std::int64_t  write_operation_count;
	std::int64_t  other_operation_count;
	std::int64_t  read_transfer_count;
	std::int64_t  write_transfer_count;
	std::int64_t  other_transfer_count;
};
#pragma pack(pop)

static_assert(sizeof(rtl_process_module_information_t) == 0x128);
static_assert(sizeof(rtl_process_module_information_ex_t) == 0x140);
static_assert(offsetof(rtl_process_modules_t, modules) == 8);
static_assert(sizeof(system_thread_information_t) == 0x50);
static_assert(sizeof(system_process_information_t) == 0x100);

constexpr std::uint32_t module_entry_size = sizeof(rtl_process_module_information_t);
constexpr std::uint32_t module_ex_entry_size = sizeof(rtl_process_module_information_ex_t);
constexpr std::uint32_t module_list_header_size =
	static_cast<std::uint32_t>(offsetof(rtl_process_modules_t, modules));

// Read out of PsActiveProcessHead rather than the host side process map, so this query and the
// list a driver walks itself never disagree about what is running.
struct listed_process_t
{
	std::uint64_t pid;
	std::string name;
};

std::vector<listed_process_t> collect_processes(win_kernel_state& state)
{
	std::vector<listed_process_t> out;

	if (!state.active_process_list.address())
		return out;

	state.active_process_list.for_each([&](emu_object<_EPROCESS> entry)
	{
		const auto ep = entry.read();

		// ImageFileName is a fixed 15 byte field that is only null terminated when it fits.
		const auto* const image = reinterpret_cast<const char*>(ep.ImageFileName);
		const auto len = ::strnlen(image, sizeof(ep.ImageFileName));

		out.push_back({ static_cast<std::uint64_t>(
			reinterpret_cast<std::uintptr_t>(ep.UniqueProcessId)), std::string(image, len) });

		return true;
	});

	return out;
}

// What the loaded module list holds, flattened once so both classes describe the same modules
// in the same order as the list the guest can walk itself.
struct loaded_module_t
{
	std::uint64_t base;
	std::uint32_t size;
	std::string path;
};

std::vector<loaded_module_t> collect_modules(win_kernel_state& state)
{
	std::vector<loaded_module_t> out;

	if (!state.loaded_module_list.address())
		return out;

	state.loaded_module_list.for_each([&](emu_object<_KLDR_DATA_TABLE_ENTRY> entry)
	{
		const auto e = entry.read();

		auto path = narrow_wstring(win::read_unicode_string(
			entry.field(&_KLDR_DATA_TABLE_ENTRY::FullDllName)));

		if (path.empty())
			path = narrow_wstring(win::read_unicode_string(
				entry.field(&_KLDR_DATA_TABLE_ENTRY::BaseDllName)));

		out.push_back({ guest_va(e.DllBase), e.SizeOfImage, std::move(path) });

		return true;
	});

	return out;
}

void fill_module_entry(rtl_process_module_information_t& entry, const loaded_module_t& mod,
	const std::uint32_t index)
{
	entry.image_base = mod.base;
	entry.image_size = mod.size;
	entry.load_order_index = static_cast<std::uint16_t>(index);
	entry.load_count = 1;

	const auto kept = std::min(mod.path.size(), sizeof(entry.full_path_name) - 1);
	std::memcpy(entry.full_path_name, mod.path.data(), kept);
	entry.full_path_name[kept] = 0;

	const auto slash = std::string_view(mod.path).substr(0, kept).find_last_of("\\/");

	entry.offset_to_file_name = slash == std::string_view::npos
		? 0
		: static_cast<std::uint16_t>(slash + 1);
}

#pragma pack(push, 4)
struct system_basic_information_t
{
	std::uint32_t reserved;
	std::uint32_t timer_resolution;
	std::uint32_t page_size;
	std::uint32_t number_of_physical_pages;
	std::uint32_t lowest_physical_page_number;
	std::uint32_t highest_physical_page_number;
	std::uint32_t allocation_granularity;
	std::uint64_t minimum_user_mode_address;
	std::uint64_t maximum_user_mode_address;
	std::uint64_t active_processors_affinity_mask;
	std::int8_t   number_of_processors;
};

struct group_affinity_t
{
	std::uint64_t mask;
	std::uint16_t group;
	std::uint16_t reserved[3];
};

enum processor_relationship : std::uint32_t
{
	relation_processor_core    = 0,
	relation_numa_node         = 1,
	relation_cache             = 2,
	relation_processor_package = 3,
	relation_group             = 4,
	relation_processor_die     = 5,
	relation_numa_node_ex      = 6,
	relation_processor_module  = 7,
	relation_all               = 0xFFFF,
};

struct numa_node_relationship_t
{
	std::uint32_t    node_number;
	std::uint8_t     reserved[18];
	std::uint16_t    group_count;
	group_affinity_t group_mask[1];
};

struct processor_relationship_t
{
	std::uint8_t     flags;
	std::uint8_t     efficiency_class;
	std::uint8_t     reserved[20];
	std::uint16_t    group_count;
	group_affinity_t group_mask[1];
};

struct processor_group_info_t
{
	std::uint8_t  maximum_processor_count;
	std::uint8_t  active_processor_count;
	std::uint8_t  reserved[38];
	std::uint64_t active_processor_mask;
};

struct group_relationship_t
{
	std::uint16_t          maximum_group_count;
	std::uint16_t          active_group_count;
	std::uint8_t           reserved[20];
	processor_group_info_t group_info[1];
};

// Size is this entry's own length, what a caller steps by, not the size of the widest member.
struct system_logical_processor_information_ex_t
{
	std::uint32_t relationship;
	std::uint32_t size;
	union
	{
		processor_relationship_t  processor;
		group_relationship_t      group;
		numa_node_relationship_t  numa;
	};
};

constexpr std::size_t logical_processor_header = offsetof(
	system_logical_processor_information_ex_t, processor);

struct system_numa_information_t
{
	std::uint32_t   highest_node_number;
	std::uint32_t   reserved;
	group_affinity_t node[1];
};

struct system_time_of_day_information_t
{
	std::int64_t  boot_time;
	std::int64_t  current_time;
	std::int64_t  time_zone_bias;
	std::uint32_t time_zone_id;
	std::uint32_t reserved;
	std::uint64_t boot_time_bias;
	std::uint64_t sleep_time_bias;
};
#pragma pack(pop)

// The granularity every Windows has had. The clock tick is in defs.hpp, shared with the
// KeTimeIncrement global so the two cannot disagree.
constexpr std::uint32_t allocation_granularity = 0x10000;

constexpr std::uint32_t langid_en_us = 0x0409;

constexpr std::uint64_t current_process_handle = ~std::uint64_t{0};

}

void modules::register_ntoskrnl_sysinfo_ops(win_kernel_state& state, proc_module& mod)
{
	auto* st = &state;

	state.redirect_ntzw(mod, "QuerySystemTime",
		[](vcpu&, emu_object<std::int64_t> system_time) -> NTSTATUS
		{
			if (!system_time)
				return STATUS_INVALID_PARAMETER;

			const auto now = static_cast<std::int64_t>(win_system_time());
			system_time.write(now);
			win::determinism::instance().mock_log("NtQuerySystemTime",
				static_cast<std::uint64_t>(now));

			THREAD_LOG_INFO("NtQuerySystemTime() -> {}", now);

			return STATUS_SUCCESS;
		});

	state.redirect_ntzw(mod, "QueryPerformanceCounter",
		[](vcpu&, emu_object<std::int64_t> performance_counter,
			emu_object<std::int64_t> performance_frequency) -> NTSTATUS
		{
			if (!performance_counter)
				return STATUS_INVALID_PARAMETER;

			auto& det = win::determinism::instance();
			std::int64_t ticks;
			std::int64_t frequency;
			if (det.enabled())
			{
				ticks = det.perf_counter();
				frequency = static_cast<std::int64_t>(det.perf_frequency());
				det.mock_log("NtQueryPerformanceCounter",
					static_cast<std::uint64_t>(ticks));
			}
			else
			{
				const auto now = std::chrono::steady_clock::now().time_since_epoch();
				ticks = std::chrono::duration_cast<win_ticks>(now).count();
				frequency = win_ticks::period::den;
			}

			performance_counter.write(ticks);

			if (performance_frequency)
				performance_frequency.write(frequency);

			THREAD_LOG_INFO("NtQueryPerformanceCounter() -> {}", ticks);

			return STATUS_SUCCESS;
		});

	auto query_system_information = [st](vcpu& cpu, const std::uint32_t system_information_class,
		const addr_t system_information, const std::uint32_t length,
		emu_object<std::uint32_t> return_length) -> NTSTATUS
	{
		auto& space = *cpu.curr_addr_space();

		switch (system_information_class)
		{
		case system_basic_information:
		case system_emulation_basic_information:
		{
			if (return_length)
				return_length.write(sizeof(system_basic_information_t));

			if (length < sizeof(system_basic_information_t))
				return STATUS_INFO_LENGTH_MISMATCH;

			const auto page = static_cast<std::uint32_t>(space.mmu_->page_size());
			const auto cpus = cpu.emu()->cpus().size();

			system_basic_information_t info{};
			info.timer_resolution = clock_increment_100ns;
			info.page_size = page;
			// The declared memory map, not the part of it that happens to be backed. The
			// highest number is inclusive, as NT's is: a driver walks up to and including that
			// frame, and the pfn database is exactly long enough to be indexed by it.
			info.number_of_physical_pages = static_cast<std::uint32_t>(emulated_physical_pages);
			info.lowest_physical_page_number = static_cast<std::uint32_t>(lowest_physical_page);
			info.highest_physical_page_number = static_cast<std::uint32_t>(highest_physical_page);
			info.allocation_granularity = allocation_granularity;
			info.active_processors_affinity_mask = cpus >= 64
				? ~std::uint64_t{0} : (std::uint64_t{1} << cpus) - 1;
			info.number_of_processors = static_cast<std::int8_t>(cpus);

			emu_object<system_basic_information_t>(space, system_information).write(info);

			THREAD_LOG_INFO("NtQuerySystemInformation(SystemBasicInformation): {} cpu(s), "
				"{} physical page(s)", cpus, info.number_of_physical_pages);

			return STATUS_SUCCESS;
		}

		case system_time_of_day_information:
		{
			if (return_length)
				return_length.write(sizeof(system_time_of_day_information_t));

			if (length < sizeof(system_time_of_day_information_t))
				return STATUS_INFO_LENGTH_MISMATCH;

			const auto now = static_cast<std::int64_t>(win_system_time());
			auto& det = win::determinism::instance();
			const auto boot_time = det.enabled() ? det.boot_time_100ns()
				: st->boot_time;

			system_time_of_day_information_t info{};
			info.boot_time = boot_time;
			info.current_time = now;
			info.time_zone_bias = det.enabled() ? det.timezone_bias_100ns() : 0;
			det.mock_log("SystemTimeOfDayInformation",
				static_cast<std::uint64_t>(now));

			emu_object<system_time_of_day_information_t>(space, system_information).write(info);

			THREAD_LOG_INFO("NtQuerySystemInformation(SystemTimeOfDayInformation): up {}s",
				win_ticks(now - boot_time) / std::chrono::seconds(1));

			return STATUS_SUCCESS;
		}

		// ntdll treats a failure here as fatal. One node, every processor in it.
		case system_numa_processor_map:
		{
			system_numa_information_t info{};
			info.highest_node_number = 0;
			const auto cpus = cpu.emu()->cpus().size();
			info.node[0].mask = cpus >= 64 ? ~0ull : (1ull << cpus) - 1;
			info.node[0].group = 0;

			const auto written = std::min<std::size_t>(length, sizeof(info));

			if (return_length)
				return_length.write(static_cast<std::uint32_t>(written));

			if (written < offsetof(system_numa_information_t, node))
				return STATUS_INFO_LENGTH_MISMATCH;

			cpu.curr_addr_space()->write_mem(system_information, &info, written);

			THREAD_LOG_INFO("NtQuerySystemInformation(SystemNumaProcessorMap): "
				"1 node, processor mask 0x{:X}", info.node[0].mask);

			return STATUS_SUCCESS;
		}

		// Each process is followed by its threads and then its name, because the name is pointed
		// at by a UNICODE_STRING whose buffer has to live in the caller's buffer too.
		case system_process_information:
		{
			const auto procs = collect_processes(*st);

			// The name is padded out so the entry after it stays 8 aligned.
			const auto entry_size_for = [](const listed_process_t& proc)
			{
				const auto name_bytes = proc.name.size() * sizeof(char16_t);

				return static_cast<std::uint32_t>(sizeof(system_process_information_t)
					+ sizeof(system_thread_information_t) + ((name_bytes + 7) & ~std::size_t{7}));
			};

			std::uint32_t required = 0;

			for (const auto& proc : procs)
				required += entry_size_for(proc);

			if (return_length)
				return_length.write(required);

			if (length < required)
			{
				THREAD_LOG_INFO("NtQuerySystemInformation(SystemProcessInformation): buffer 0x{:X}"
					" < required 0x{:X} for {} process(es)", length, required, procs.size());

				return STATUS_INFO_LENGTH_MISMATCH;
			}

			std::vector<std::uint8_t> out(required, 0);
			std::uint32_t offset = 0;

			for (std::size_t i = 0; i < procs.size(); ++i)
			{
				const auto& proc = procs[i];
				const auto wide = widen_string(proc.name);
				const auto name_bytes = wide.size() * sizeof(char16_t);
				const auto entry_size = entry_size_for(proc);

				auto* const entry = reinterpret_cast<system_process_information_t*>(
					out.data() + offset);
				auto* const thread = reinterpret_cast<system_thread_information_t*>(
					out.data() + offset + sizeof(system_process_information_t));

				const auto name_at = offset + static_cast<std::uint32_t>(
					sizeof(system_process_information_t) + sizeof(system_thread_information_t));

				std::memcpy(out.data() + name_at, wide.data(), name_bytes);

				entry->next_entry_offset = i + 1 < procs.size() ? entry_size : 0;
				entry->number_of_threads = 1;
				entry->base_priority = 8;
				entry->unique_process_id = proc.pid;
				entry->handle_count = 32;
				entry->image_name.Length = static_cast<unsigned short>(name_bytes);
				entry->image_name.MaximumLength =
					static_cast<unsigned short>(name_bytes + sizeof(char16_t));
				entry->image_name.Buffer = guest_ptr<char16_t>(system_information + name_at);

				// One thread each: nothing here tracks per process threads, and a process with
				// none would read as a process that is already gone.
				thread->unique_process_id = proc.pid;
				thread->unique_thread_id = proc.pid + 4;
				thread->priority = 8;
				thread->base_priority = 8;
				thread->thread_state = 5;  // Waiting
				thread->wait_reason = 13;  // WrQueue

				offset += entry_size;
			}

			space.write_mem(system_information, out.data(), out.size());

			THREAD_LOG_INFO("NtQuerySystemInformation(SystemProcessInformation): {} process(es),"
				" 0x{:X} bytes", procs.size(), required);

			return STATUS_SUCCESS;
		}

		case system_module_information:
		{
			const auto mods = collect_modules(*st);
			const auto count = static_cast<std::uint32_t>(mods.size());
			const auto required = module_list_header_size + count * module_entry_size;

			if (return_length)
				return_length.write(required);

			if (length < module_list_header_size)
				return STATUS_INFO_LENGTH_MISMATCH;

			// As many as fit, and the count says how many that was -- the caller sizes its
			// second call from the required length written above.
			const auto fits = std::min<std::uint32_t>(
				(length - module_list_header_size) / module_entry_size, count);

			std::vector<std::uint8_t> out(module_list_header_size
				+ fits * module_entry_size, 0);

			auto* const header = reinterpret_cast<rtl_process_modules_t*>(out.data());
			header->number_of_modules = fits;

			for (std::uint32_t i = 0; i < fits; ++i)
				fill_module_entry(header->modules[i], mods[i], i);

			space.write_mem(system_information, out.data(), out.size());

			THREAD_LOG_INFO("NtQuerySystemInformation(SystemModuleInformation): {} of {} "
				"module(s), required 0x{:X}, buffer 0x{:X}", fits, count, required, length);

			return fits == count ? STATUS_SUCCESS : STATUS_INFO_LENGTH_MISMATCH;
		}

		case system_module_information_ex:
		{
			const auto mods = collect_modules(*st);
			const auto count = static_cast<std::uint32_t>(mods.size());
			// Two bytes past the last entry, left zero: every entry carries a step, and a walk
			// that follows the last one lands on that zero and stops there.
			const auto required = count * module_ex_entry_size + sizeof(std::uint16_t);

			if (return_length)
				return_length.write(required);

			// A chain rather than a counted array, so a short buffer has no partial form.
			if (length < required)
			{
				THREAD_LOG_INFO("NtQuerySystemInformation(SystemModuleInformationEx): "
					"buffer 0x{:X} < required 0x{:X} for {} module(s)",
					length, required, count);

				return STATUS_INFO_LENGTH_MISMATCH;
			}

			std::vector<std::uint8_t> out(required, 0);

			for (std::uint32_t i = 0; i < count; ++i)
			{
				auto* const entry = reinterpret_cast<rtl_process_module_information_ex_t*>(
					out.data() + i * module_ex_entry_size);

				entry->next_offset = static_cast<std::uint16_t>(module_ex_entry_size);

				fill_module_entry(entry->base_info, mods[i], i);
			}

			if (required)
				space.write_mem(system_information, out.data(), out.size());

			THREAD_LOG_INFO("NtQuerySystemInformation(SystemModuleInformationEx): {} module(s), "
				"0x{:X} bytes", count, required);

			return STATUS_SUCCESS;
		}

		// { BOOLEAN DebuggerEnabled, BOOLEAN DebuggerNotPresent } -- the same answer
		// KdDebuggerEnabled and KdDebuggerNotPresent give, so the two cannot disagree.
		case system_kernel_debugger_information:
		{
			constexpr std::uint8_t info[] = { 0, 1 };

			if (return_length)
				return_length.write(sizeof(info));

			if (length < sizeof(info))
				return STATUS_INFO_LENGTH_MISMATCH;

			space.write_mem(system_information, info, sizeof(info));

			THREAD_LOG_INFO("NtQuerySystemInformation(SystemKernelDebuggerInformation): "
				"no debugger");

			return STATUS_SUCCESS;
		}

		// { BOOLEAN DebuggerAllowed, BOOLEAN DebuggerEnabled, BOOLEAN DebuggerPresent }
		case system_kernel_debugger_information_ex:
		{
			constexpr std::uint8_t info[] = { 1, 0, 0 };

			if (return_length)
				return_length.write(sizeof(info));

			if (length < sizeof(info))
				return STATUS_INFO_LENGTH_MISMATCH;

			space.write_mem(system_information, info, sizeof(info));

			THREAD_LOG_INFO("NtQuerySystemInformation(SystemKernelDebuggerInformationEx): "
				"allowed, not enabled, not present");

			return STATUS_SUCCESS;
		}

		// { ULONG Length, ULONG CodeIntegrityOptions }. Code integrity is on -- a driver that
		// finds it off concludes the machine is already compromised -- but not the hypervisor
		// enforced kind: claiming that promises W^X over kernel memory, and a driver is free to
		// read the page tables and see whether the promise holds. Plenty of real machines run
		// exactly this way, so it is the honest answer rather than a weaker one.
		case system_code_integrity_information:
		{
			constexpr std::uint32_t option_enabled = 0x01;

			struct
			{
				std::uint32_t length;
				std::uint32_t options;
			} info{ sizeof(info), option_enabled };

			if (return_length)
				return_length.write(sizeof(info));

			if (length < sizeof(info))
				return STATUS_INFO_LENGTH_MISMATCH;

			space.write_mem(system_information, info);

			THREAD_LOG_INFO("NtQuerySystemInformation(SystemCodeIntegrityInformation): "
				"enabled, options 0x{:X}", info.options);

			return STATUS_SUCCESS;
		}

		// Whether the machine has hardware enforced stack protection. The cpu this emulator
		// presents does not report CET in its feature leaves, so the answer that agrees with
		// it is that there is none -- a machine, just not one with shadow stacks.
		case system_shadow_stack_information:
		{
			constexpr std::uint32_t no_shadow_stack_support = 0;

			if (return_length)
				return_length.write(sizeof(no_shadow_stack_support));

			if (length < sizeof(no_shadow_stack_support))
				return STATUS_INFO_LENGTH_MISMATCH;

			space.write_mem(system_information, no_shadow_stack_support);

			THREAD_LOG_INFO("NtQuerySystemInformation(SystemShadowStackInformation): the cpu "
				"reports no CET, so nothing here is shadow stack protected");

			return STATUS_SUCCESS;
		}

		// Nothing here holds a policy, and an empty one would read as a policy that allows
		// everything -- so the caller is told there is none rather than shown a permissive one.
		case system_code_integrity_policy_information:
		{
			if (return_length)
				return_length.write(0);

			THREAD_LOG_WARN("NtQuerySystemInformation(SystemCodeIntegrityPolicyInformation): "
				"nothing here holds a code integrity policy");

			return STATUS_NOT_IMPLEMENTED;
		}

		default:
			THREAD_LOG_WARN("NtQuerySystemInformation: unhandled class {}",
				system_information_class);
			return STATUS_INVALID_INFO_CLASS;
		}
	};

	// LdrpInitializeProcess asks on its way up, and a loader that cannot learn the layout stops.
	auto query_processor_topology = [](vcpu& cpu, const addr_t input_buffer,
		const std::uint32_t input_buffer_length, const addr_t system_information,
		const std::uint32_t length, emu_object<std::uint32_t> return_length) -> NTSTATUS
	{
		auto& space = *cpu.curr_addr_space();

		auto wanted = relation_all;

		if (input_buffer && input_buffer_length >= sizeof(std::uint32_t))
			wanted = static_cast<processor_relationship>(space.read_mem<std::uint32_t>(input_buffer));

		const auto cpus = cpu.emu()->cpus().size();
		const std::uint64_t mask = cpus >= 64 ? ~0ull : (1ull << cpus) - 1;

		std::vector<std::uint8_t> answer;

		const auto add = [&answer](const processor_relationship kind, const std::size_t body,
			auto&& fill)
		{
			system_logical_processor_information_ex_t entry{};
			entry.relationship = kind;
			entry.size = static_cast<std::uint32_t>(logical_processor_header + body);

			fill(entry);

			const auto* const bytes = reinterpret_cast<const std::uint8_t*>(&entry);
			answer.insert(answer.end(), bytes, bytes + entry.size);
		};

		if (wanted == relation_processor_core || wanted == relation_all)
		{
			for (std::size_t i = 0; i < cpus; ++i)
			{
				add(relation_processor_core, sizeof(processor_relationship_t),
					[i](auto& entry)
					{
						entry.processor.flags = 0;
						entry.processor.efficiency_class = 0;
						entry.processor.group_count = 1;
						entry.processor.group_mask[0].mask = 1ull << i;
						entry.processor.group_mask[0].group = 0;
					});
			}
		}

		if (wanted == relation_numa_node || wanted == relation_numa_node_ex
			|| wanted == relation_all)
		{
			const auto kind = wanted == relation_numa_node_ex
				? relation_numa_node_ex : relation_numa_node;

			add(kind, sizeof(numa_node_relationship_t),
				[mask](auto& entry)
				{
					entry.numa.node_number = 0;
					entry.numa.group_count = 1;
					entry.numa.group_mask[0].mask = mask;
					entry.numa.group_mask[0].group = 0;
				});
		}

		if (wanted == relation_group || wanted == relation_all)
		{
			add(relation_group, sizeof(group_relationship_t),
				[cpus, mask](auto& entry)
				{
					entry.group.maximum_group_count = 1;
					entry.group.active_group_count = 1;
					entry.group.group_info[0].maximum_processor_count =
						static_cast<std::uint8_t>(cpus);
					entry.group.group_info[0].active_processor_count =
						static_cast<std::uint8_t>(cpus);
					entry.group.group_info[0].active_processor_mask = mask;
				});
		}

		if (answer.empty())
		{
			THREAD_LOG_WARN("NtQuerySystemInformationEx(SystemLogicalProcessorAndGroup"
				"Information): relationship {} is not one this machine describes",
				static_cast<std::uint32_t>(wanted));

			return STATUS_INVALID_PARAMETER;
		}

		if (return_length)
			return_length.write(static_cast<std::uint32_t>(answer.size()));

		if (length < answer.size())
			return STATUS_INFO_LENGTH_MISMATCH;

		space.write_mem(system_information, answer.data(), answer.size());

		THREAD_LOG_INFO("NtQuerySystemInformationEx(SystemLogicalProcessorAndGroupInformation, "
			"relationship={}): {} cpu(s) in 1 group, mask 0x{:X}", static_cast<std::uint32_t>(wanted),
			cpus, mask);

		return STATUS_SUCCESS;
	};

	auto query_system_information_ex = [query_system_information, query_processor_topology](
		vcpu& cpu, const std::uint32_t system_information_class, const addr_t input_buffer,
		const std::uint32_t input_buffer_length, const addr_t system_information,
		const std::uint32_t length, emu_object<std::uint32_t> return_length) -> NTSTATUS
	{
		THREAD_LOG_INFO("NtQuerySystemInformationEx: input=0x{:X}/{}",
			input_buffer, input_buffer_length);

		if (system_information_class == system_logical_processor_and_group_information)
		{
			return query_processor_topology(cpu, input_buffer, input_buffer_length,
				system_information, length, return_length);
		}

		return query_system_information(cpu, system_information_class, system_information,
			length, return_length);
	};

	auto query_langid = [](vcpu&, emu_object<std::uint32_t> language_id,
		const std::string_view who) -> NTSTATUS
	{
		if (!language_id)
			return STATUS_INVALID_PARAMETER;

		language_id.write(langid_en_us);

		THREAD_LOG_INFO("{}() -> 0x{:04X}", who, langid_en_us);

		return STATUS_SUCCESS;
	};

	state.redirect_ntzw(mod, "QueryDefaultLocale",
		[query_langid](vcpu& cpu, const bool user_profile,
			emu_object<std::uint32_t> default_locale_id) -> NTSTATUS
		{
			THREAD_LOG_INFO("NtQueryDefaultLocale: user_profile={}", user_profile);
			return query_langid(cpu, default_locale_id, "NtQueryDefaultLocale");
		});

	state.redirect_ntzw(mod, "QueryDefaultUILanguage",
		[query_langid](vcpu& cpu, emu_object<std::uint32_t> default_ui_language) -> NTSTATUS
		{
			return query_langid(cpu, default_ui_language, "NtQueryDefaultUILanguage");
		});

	state.redirect_ntzw(mod, "QueryInstallUILanguage",
		[query_langid](vcpu& cpu, emu_object<std::uint32_t> install_ui_language) -> NTSTATUS
		{
			return query_langid(cpu, install_ui_language, "NtQueryInstallUILanguage");
		});

	// Nothing here sleeps or dims a display, so the previous state is the one nothing changed.
	state.redirect_ntzw(mod, "SetThreadExecutionState",
		[](vcpu&, const std::uint32_t new_flags,
			emu_object<std::uint32_t> previous_flags) -> NTSTATUS
		{
			constexpr std::uint32_t es_continuous = 0x80000000;

			if (previous_flags)
				previous_flags.write(es_continuous);

			THREAD_LOG_INFO("NtSetThreadExecutionState(0x{:X}): nothing here sleeps", new_flags);

			return STATUS_SUCCESS;
		});

	// The emulator fetches from the same memory the guest writes, so there is nothing to flush.
	state.redirect_ntzw(mod, "FlushInstructionCache",
		[](vcpu&, const std::uint64_t process_handle, const addr_t base_address,
			const std::uint64_t length) -> NTSTATUS
		{
			THREAD_LOG_INFO("NtFlushInstructionCache(0x{:X}, 0x{:X}, {}): the emulator fetches "
				"from guest memory directly", process_handle, base_address, length);

			return STATUS_SUCCESS;
		});

	// Every cpu here shares one host memory model already, so the barrier is implicit.
	state.redirect_ntzw(mod, "FlushProcessWriteBuffers", [](vcpu&)
	{
		THREAD_LOG_INFO("NtFlushProcessWriteBuffers()");
	});

	state.redirect_ntzw(mod, "DuplicateObject",
		[st](vcpu&, const std::uint64_t source_process_handle,
			const std::uint64_t source_handle, const std::uint64_t target_process_handle,
			emu_object<std::uint64_t> target_handle, const std::uint32_t desired_access,
			const std::uint32_t handle_attributes, const std::uint32_t options) -> NTSTATUS
		{
			constexpr std::uint32_t duplicate_close_source = 0x1;
			constexpr std::uint32_t duplicate_same_access = 0x2;

			if (source_process_handle != current_process_handle
				|| target_process_handle != current_process_handle)
			{
				THREAD_LOG_WARN("NtDuplicateObject: there is one process here, so a handle can "
					"only be duplicated within it");
				return STATUS_INVALID_HANDLE;
			}

			auto& handles = st->sys_proc->handle_table();
			const auto entry = handles.lookup_handle(source_handle);

			if (!entry)
				return STATUS_INVALID_HANDLE;

			const auto access = (options & duplicate_same_access) ? entry->access : desired_access;

			st->objs.reference_object(entry->body_addr);
			const auto duplicate = handles.create_handle(entry->body_addr, access);

			if (target_handle)
				target_handle.write(duplicate);

			if (options & duplicate_close_source)
				handles.close_handle(source_handle);

			THREAD_LOG_INFO("NtDuplicateObject(0x{:X}, access=0x{:X}, attributes=0x{:X}, "
				"options=0x{:X}) -> 0x{:X}",
				source_handle, access, handle_attributes, options, duplicate);

			return STATUS_SUCCESS;
		});

	// Nothing here puts an object in the namespace, so every object is already temporary.
	auto make_temporary = [st](vcpu&, const std::uint64_t handle) -> NTSTATUS
	{
		if (!st->sys_proc->handle_table().lookup_handle(handle))
			return STATUS_INVALID_HANDLE;

		THREAD_LOG_INFO("NtMakeTemporaryObject(0x{:X}): nothing here is permanent", handle);

		return STATUS_SUCCESS;
	};

	state.redirect_ntzw(mod, "MakeTemporaryObject", make_temporary);

	state.redirect_ntzw(mod, "QuerySystemInformation", query_system_information);
	state.redirect_ntzw(mod, "QuerySystemInformationEx", query_system_information_ex);

	auto map_nls_file = [st](vcpu& cpu, const std::string& path,
		emu_object<addr_t> base_out, const std::string_view who) -> NTSTATUS
	{
		const auto file = st->fs.open(path);

		if (!file)
		{
			THREAD_LOG_WARN("{}: {} is not in the guest filesystem", who, path);
			return STATUS_OBJECT_NAME_NOT_FOUND;
		}

		auto& space = *cpu.curr_addr_space();
		const auto data = file->data();
		const auto base = space.alloc(data.size(), prot_rw);

		if (!base)
		{
			THREAD_LOG_ERR("{}: no room for {} ({} bytes)", who, path, data.size());
			return STATUS_NO_MEMORY;
		}

		space.write_mem(base, data.data(), data.size());

		if (base_out)
			base_out.write(base);

		THREAD_LOG_INFO("{}: mapped {} ({} bytes) at 0x{:X}", who, path, data.size(), base);

		return STATUS_SUCCESS;
	};

	state.redirect_ntzw(mod, "InitializeNlsFiles",
		[map_nls_file](vcpu& cpu, emu_object<addr_t> base_address,
			emu_object<std::uint32_t> default_locale_id,
			emu_object<std::int64_t> default_casing_table_size) -> NTSTATUS
		{
			// Without this file kernelbase cannot resolve LOCALE_INVARIANT and its DllMain fails.
			const auto status = map_nls_file(cpu, std::string(system32_dir_narrow) + "locale.nls",
				base_address, "NtInitializeNlsFiles");

			if (status != STATUS_SUCCESS)
				return status;

			if (default_locale_id)
				default_locale_id.write(0x0409);

			if (default_casing_table_size)
				default_casing_table_size.write(0);

			return STATUS_SUCCESS;
		});

	state.redirect_ntzw(mod, "GetNlsSectionPtr",
		[st, map_nls_file](vcpu& cpu, const std::uint32_t section_type,
			const std::uint32_t section_data, [[maybe_unused]] const addr_t context_data,
			emu_object<addr_t> section_pointer, emu_object<std::uint32_t> section_size) -> NTSTATUS
		{
			constexpr std::uint32_t nls_section_code_page = 11;

			if (section_type != nls_section_code_page)
			{
				THREAD_LOG_WARN("NtGetNlsSectionPtr: section type {} is not a code page",
					section_type);
				return STATUS_NOT_SUPPORTED;
			}

			const auto path = std::string(system32_dir_narrow) + "c_"
				+ std::to_string(section_data) + ".nls";

			const auto status = map_nls_file(cpu, path, section_pointer, "NtGetNlsSectionPtr");

			if (status != STATUS_SUCCESS)
				return status;

			if (section_size)
			{
				const auto file = st->fs.open(path);
				section_size.write(static_cast<std::uint32_t>(file ? file->size() : 0));
			}

			return STATUS_SUCCESS;
		});
}
