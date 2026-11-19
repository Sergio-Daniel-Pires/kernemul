#pragma once
#include "../../emu/object.hpp"
#include "process_params.hpp"
#include "per_cpu.hpp"
#include "defs.hpp"
#include "../../target.hpp"
#include <algorithm>
#include <cstddef>
#include <string>

namespace win_target
{
#if defined(KERNEMUL_ARCH_ARM64)
	inline constexpr std::uint16_t image_machine          = 0xAA64; // ARM64
	inline constexpr std::uint16_t processor_architecture = 12;     // ARM64

	// ARM64 has no recursive self map to name, so nothing here reports one.
	inline constexpr addr_t pte_base = 0;

	// _CONTEXT calls the program counter something different on each arch, and a crash dump
	// repeats it outside the captured context, where the reader looks for it at a fixed offset.
	inline constexpr std::size_t context_pc_offset = offsetof(_CONTEXT, Pc);
#else
	inline constexpr std::uint16_t image_machine          = 0x8664; // AMD64
	inline constexpr std::uint16_t processor_architecture = 9;      // AMD64

	// MmPteBase: the pml4 slot that points at the pml4, as a virtual address. The slot itself
	// is x86::self_map_pml4_index, which is what the mmu actually writes.
	inline constexpr addr_t pte_base = 0xFFFFF68000000000;

	inline constexpr std::size_t context_pc_offset = offsetof(_CONTEXT, Rip);
#endif
}

// KdDebuggerDataBlock's size as its own header reports it. A triage dump carries a copy of the
// block, so the size is needed away from the code that builds the original.
inline constexpr std::uint32_t kdbg_block_size = 0x3A0;

constexpr std::uint64_t kuser_shared_data_user_va   = 0x7FFE0000;
constexpr std::uint64_t kuser_shared_data_kernel_va  = 0xFFFFF78000000000;

// The memory map the guest is told about, stated once and derived from the one place that
// decides it, because these reach the guest through unrelated calls -- the basic information
// block, MmGetPhysicalMemoryRanges, the length of the pfn database -- and a machine whose
// answers disagree with each other is not one anyone has.
inline constexpr std::uint64_t guest_page_size = 0x1000;

// Frames start high: there is no usable ram under 4gb on a machine with this much of it. See
// mmu::phys_base for why.
inline constexpr std::uint64_t lowest_physical_page = mmu::phys_base / guest_page_size;

// How much ram there is, as a count of frames. 4gb -- the builds this pretends to be do not
// install on less, so a smaller number here is one a real machine never reports.
inline constexpr std::uint64_t emulated_physical_pages = mmu::phys_size / guest_page_size;

// The last frame that exists, inclusive, which is what NT's HighestPhysicalPageNumber means. A
// driver walks up to and including this one.
inline constexpr std::uint64_t highest_physical_page =
	lowest_physical_page + emulated_physical_pages - 1;

// The database is indexed by raw frame number from zero, as NT's is, so it is as long as the
// highest frame rather than as long as the count. The low half describes the hole under 4gb and
// stays zero, which is what a hole's entries read as on a real machine anyway. Every page of it
// is host memory committed up front: at 0x30 bytes an entry this is about 96mb.
inline constexpr std::uint64_t pfn_database_entries = highest_physical_page + 1;

// Only what an image whose NtBuildNumber cannot be read falls back to. The mapped ntoskrnl is
// the authority on its own build; see win_kernel_state::nt_build_number.
inline constexpr std::uint32_t default_build_number = 19045;

// The processor count reaches the guest in several unrelated places, which have to agree.
inline _KUSER_SHARED_DATA make_default_kuser_shared_data(const std::size_t processors)
{
	_KUSER_SHARED_DATA sd{};

	sd.NtMajorVersion = 10;
	sd.NtMinorVersion = 0;
	sd.NtBuildNumber = default_build_number;
	sd.NtProductType = NtProductWinNt;
	sd.ProductTypeIsValid = 1;
	sd.NativeProcessorArchitecture = win_target::processor_architecture;
	sd.ImageNumberLow = win_target::image_machine;
	sd.ImageNumberHigh = win_target::image_machine;
	sd.ActiveProcessorCount = static_cast<std::uint32_t>(processors);
	sd.ActiveGroupCount = 1;
	sd.NumberOfPhysicalPages = static_cast<std::uint32_t>(emulated_physical_pages);
	sd.LargePageMinimum = 0x200000;
	sd.TickCountMultiplier = default_tick_count_multiplier;

	auto& det = win::determinism::instance();
	if (det.enabled())
	{
		sd.NtBuildNumber = det.build_number();
		sd.TickCountMultiplier = det.tick_multiplier();
		sd.ActiveProcessorCount = det.processor_count();
		sd.Cookie = det.security_cookie();
		const auto bias = det.timezone_bias_100ns();
		sd.TimeZoneBias.LowPart = static_cast<std::uint32_t>(bias);
		sd.TimeZoneBias.High1Time = static_cast<std::int32_t>(bias >> 32);
		sd.TimeZoneBias.High2Time = static_cast<std::int32_t>(bias >> 32);
	}

	const auto root_chars = std::min(windows_dir.size(), std::size(sd.NtSystemRoot) - 1);
	std::char_traits<char16_t>::copy(sd.NtSystemRoot, windows_dir.data(), root_chars);
	sd.NtSystemRoot[root_chars] = u'\0';

	return sd;
}

inline _PEB64 make_default_peb(const std::size_t processors,
	const std::uint32_t build = default_build_number)
{
	_PEB64 peb{};

	peb.NumberOfProcessors = static_cast<std::uint32_t>(processors);

	peb.HeapSegmentReserve = 0x100000;
	peb.HeapSegmentCommit = 0x1000;
	peb.HeapDeCommitTotalFreeThreshold = 0x10000;
	peb.HeapDeCommitFreeBlockThreshold = 0x1000;
	peb.MaximumNumberOfHeaps = 0x10;

	peb.OSMajorVersion = 10;
	peb.OSBuildNumber = build;
	peb.OSPlatformId = 2;

	peb.ImageSubsystem = 3;
	peb.ImageSubsystemMajorVersion = 6;

	return peb;
}

inline _TEB64 make_default_teb(addr_t teb_addr, addr_t stack_base,
	std::size_t stack_size, std::uint64_t process_id,
	std::uint32_t thread_id, addr_t peb_address, std::size_t processors)
{
	_TEB64 teb{};

	// RtlGetCurrentProcessorNumber takes a slow path when the bit is clear, so it is set.
	teb.PrimaryGroupAffinity.Mask = affinity_mask(processors);
	teb.PrimaryGroupAffinity.Group = 0;

	teb.NtTib.StackBase = stack_base + stack_size;
	teb.NtTib.StackLimit = stack_base;
	teb.NtTib.Self = teb_addr;
	teb.ClientId.UniqueProcess = process_id;
	teb.ClientId.UniqueThread = thread_id;
	teb.ProcessEnvironmentBlock = peb_address;

	return teb;
}
