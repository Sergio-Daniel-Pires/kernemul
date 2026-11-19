#include "target.hpp"
#include "kernel/map.hpp"
#include "kernel/win/win_kernel.hpp"
#include "emu/calling_conv.hpp"
#include "util/log.hpp"
#include "chunk_emitter.hpp"
#include "util/string.hpp"

#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "emu/guest_emu.hpp"

#if defined(KERNEMUL_ARCH_ARM64)
	#include "kernel/win/arm64_win.hpp"
	#include "emu/arm64/mmu.hpp"
	#include "emu/arm64/calling_conv.hpp"

	namespace guest
	{
		using mmu          = arm64::mmu;
		using calling_conv = arm64_win_conv;
		using win_emulator = arm64_win_emulator;
	}
#else
	#include "kernel/win/x86_win.hpp"
	#include "emu/x86/mmu.hpp"
	#include "emu/x86/calling_conv.hpp"

	namespace guest
	{
		using mmu          = x86::mmu;
		using calling_conv = x86_win_conv;
		using win_emulator = x86_win_emulator;
	}
#endif

// Set by setup_driver, consumed by the module code hook on the first instruction
// of the driver so the watched slot is already mapped when the hook is installed.
// Defined at global scope: kernel.hpp declares it with external linkage.
std::function<void(emu&, vcpu&)> g_arm_pending_watch;

namespace
{

struct guest_image
{
	std::string name;
	bool driver = false;

	std::shared_ptr<thread> entry;
};

// ---- LOCAL MEASUREMENT PATCH (not upstream): guest RIP page tracer ----
static std::vector<std::uint64_t> g_rip_pages;
static std::uint64_t g_rip_total = 0;
static bool g_slot_trace = false;
static bool g_zero_bias_store = false;
static bool g_slot_trace_seen = false;
static std::uint64_t g_slot_trace_last = 0;
static std::uint64_t g_slot_trace_prev_rip = 0;
static addr_t g_rip_min = ~static_cast<addr_t>(0);
static addr_t g_rip_max = 0;
static addr_t g_mod_base = 0;
static std::uint64_t g_mod_size = 0;
// single-step stream: driver RVAs in execution order, capped
static std::vector<std::uint32_t> g_rip_seq;
static std::size_t g_rip_seq_cap = 0;
// first-entry milestones: watched RVAs and the first index inside a range
static std::vector<std::uint64_t> g_watch_rva;
static std::vector<std::uint64_t> g_watch_first;   // index+1, 0 = not yet hit
static std::vector<std::uint64_t> g_watch_last;    // index+1 of the last hit
static std::unordered_map<std::uint64_t, std::size_t> g_watch_index;
static std::uint64_t g_in_lo = 0, g_in_hi = 0;
static std::uint64_t g_in_idx = 0, g_in_rva = 0;
static int g_in_done = 0;
static std::uint64_t g_step_idx = 0;
// record the stream only from the first hit of this RVA (allow-pair analogue)
static std::uint64_t g_seq_from_rva = 0;

namespace
{
struct kernemul_mutation
{
	enum class kind { force_flag, force_reg };
	kind type = kind::force_flag;
	std::uint64_t rva = 0;
	std::uint64_t mask = 0;
	std::uint64_t value = 0;
	int reg = 0;
	std::uint64_t chunk = UINT64_MAX;
	std::uint64_t step = UINT64_MAX;
};

static std::vector<kernemul_mutation> g_kernemul_mutations;
static bool g_kernemul_mutations_ready = false;

static std::uint64_t mutation_u64(const std::string& text)
{
	return std::strtoull(text.c_str(), nullptr, 0);
}

static void parse_kernemul_mutations()
{
	g_kernemul_mutations_ready = true;
	const char* env = std::getenv("KERNEMUL_MUTATIONS");
	if (!env || !*env)
		return;

	std::string list(env);
	std::size_t start = 0;
	while (start <= list.size())
	{
		const auto end = list.find(';', start);
		const std::string spec = list.substr(start,
			end == std::string::npos ? std::string::npos : end - start);
		start = end == std::string::npos ? list.size() + 1 : end + 1;
		if (spec.empty())
			continue;

		std::vector<std::string> fields;
		std::size_t field_start = 0;
		while (field_start <= spec.size())
		{
			const auto field_end = spec.find(':', field_start);
			fields.push_back(spec.substr(field_start,
				field_end == std::string::npos ? std::string::npos
					: field_end - field_start));
			field_start = field_end == std::string::npos
				? spec.size() + 1 : field_end + 1;
		}
		if (fields.size() < 4)
			continue;

		kernemul_mutation m;
		if (fields[0] == "force_flag")
			m.type = kernemul_mutation::kind::force_flag;
		else if (fields[0] == "force_reg")
			m.type = kernemul_mutation::kind::force_reg;
		else
			continue;
		m.rva = mutation_u64(fields[1]);
		if (m.type == kernemul_mutation::kind::force_flag)
		{
			m.mask = mutation_u64(fields[2]);
			m.value = mutation_u64(fields[3]);
		}
		else
		{
			m.reg = static_cast<int>(mutation_u64(fields[2]));
			m.value = mutation_u64(fields[3]);
		}
		for (std::size_t i = 4; i < fields.size(); ++i)
		{
			if (fields[i].rfind("chunk=", 0) == 0)
				m.chunk = mutation_u64(fields[i].substr(6));
			else if (fields[i].rfind("step=", 0) == 0)
				m.step = mutation_u64(fields[i].substr(5));
		}
		g_kernemul_mutations.push_back(m);
	}
}

static void apply_kernemul_mutations(vcpu& cpu, std::uint64_t rva,
	std::uint64_t step)
{
	if (!g_kernemul_mutations_ready)
		parse_kernemul_mutations();
	if (g_kernemul_mutations.empty())
		return;

	for (const auto& m : g_kernemul_mutations)
	{
		if (m.rva != rva)
			continue;
		if (m.chunk != UINT64_MAX &&
			chunk_emitter::instance().chunk_index() != m.chunk)
			continue;
		if (m.step != UINT64_MAX && step != m.step)
			continue;
		if (m.type == kernemul_mutation::kind::force_flag)
		{
			auto flags = cpu.reg<std::uint64_t>(static_cast<reg_t>(17));
			flags = (flags & ~m.mask) | (m.value & m.mask);
			cpu.reg(static_cast<reg_t>(17), flags);
		}
		else
		{
			cpu.reg(static_cast<reg_t>(m.reg), m.value);
		}
	}
}
// Environment-driven flag oracle: force selected flag bits at the n-th
// execution of an RVA, using a table generated from the LWK reference trace.
// This keeps the flow aligned where the two runtimes hash different
// environment addresses into their VM state (see docs/grfn1_f4skip_coverage.md).
struct kernemul_oracle_entry
{
	std::uint64_t mask = 0;
	std::uint64_t value = 0;
};

static std::map<std::pair<std::uint64_t, std::uint64_t>,
	kernemul_oracle_entry> g_flag_oracle;
static std::unordered_set<std::uint64_t> g_flag_oracle_rvas;
static std::unordered_map<std::uint64_t, std::uint64_t> g_flag_oracle_counts;
static bool g_flag_oracle_ready = false;

static void parse_flag_oracle()
{
	g_flag_oracle_ready = true;
	const char* env = std::getenv("KERNEMUL_FLAG_ORACLE");
	if (!env || !*env)
		return;
	std::ifstream stream(env);
	std::string line;
	while (std::getline(stream, line))
	{
		if (line.empty() || line[0] == '#')
			continue;
		std::istringstream fields(line);
		std::uint64_t rva = 0, occurrence = 0, mask = 0, value = 0;
		if (!(fields >> std::hex >> rva >> occurrence >> mask >> value))
			continue;
		g_flag_oracle[{rva, occurrence}] = kernemul_oracle_entry{mask, value};
		g_flag_oracle_rvas.insert(rva);
	}
	LOG_INFO("[FLAG_ORACLE] entries={}", g_flag_oracle.size());
}

static void apply_flag_oracle(vcpu& cpu, std::uint64_t rva)
{
	if (!g_flag_oracle_ready)
		parse_flag_oracle();
	if (g_flag_oracle.empty() || !g_flag_oracle_rvas.count(rva))
		return;
	const std::uint64_t occurrence = ++g_flag_oracle_counts[rva];
	const auto entry = g_flag_oracle.find({rva, occurrence});
	if (entry == g_flag_oracle.end())
		return;
	auto flags = cpu.reg<std::uint64_t>(static_cast<reg_t>(17));
	flags = (flags & ~entry->second.mask) | (entry->second.value & entry->second.mask);
	cpu.reg(static_cast<reg_t>(17), flags);
}
} // namespace

static int g_seq_started = 0;
// one-shot guest register dump at given RVAs (pre-state of that instruction)
static std::uint64_t g_regs_rva[8];
static std::uint64_t g_regs_vals[8][18];
static int g_regs_hit[8];
static int g_regs_count = 0;
// KERNEMUL_LAST_BEFORE=<rva_target>:<rva_gate> -- keep the *last* occurrence of
// rva_target seen before rva_gate is first reached, so a specific call instance
// can be tied to a later consumer instead of the one-shot first hit. Comma-separated
// target:gate pairs are accepted for capturing several points in one run.
struct last_before_watch
{
	std::uint64_t target = 0;
	std::uint64_t gate = 0;
	std::uint64_t regs[18]{};
	long long idx = -1;
	long long gate_idx = -1;
	bool have = false;
	bool snapped = false;
};
static std::vector<last_before_watch> g_lb_watches;
// KERNEMUL_MEM_WRITE_WATCH=<guest-va> -- log guest writes overlapping one qword.
static std::uint64_t g_mem_write_watch = 0;
static std::uint64_t g_mem_write_hits = 0;
static std::uint64_t g_mem_write_max = 128;
static std::uint64_t g_mem_write_addrs = 0;
// KERNEMUL_MEM_WATCH_VALUE=<value> -- only report writes where one of the store
// registers holds this value, so a stack slot reused by many frames yields the
// write that actually stored the value of interest.
static std::uint64_t g_mem_watch_value = 0;
// KERNEMUL_REG_VALUE_WATCH=<value> -- log the first steps whose register file
// holds this value together with the RIP of the previous instruction, i.e. the
// instruction that produced it. Catches stub returns, real-kernel code and VM
// arithmetic alike.
static std::uint64_t g_reg_value_watch = 0;
static int g_reg_value_watch_reg = -1;
static int g_reg_value_hits = 0;
static bool g_mem_last_write_valid = false;
static std::uint64_t g_mem_last_write_idx = 0;
static std::uint64_t g_mem_last_write_rip = 0;
static std::uint64_t g_mem_last_write_rsp = 0;
static std::uint64_t g_mem_last_write_rax = 0;
static std::uint64_t g_mem_last_write_rbx = 0;
static std::uint64_t g_mem_last_write_rcx = 0;
static std::uint64_t g_mem_last_write_rdx = 0;
static std::uint64_t g_mem_last_write_r8 = 0;
static std::uint64_t g_mem_last_write_r9 = 0;
static bool g_mem_watch_load_mapping_logged = false;
static std::uint64_t g_mem_snapshot_rva = 0;
static std::uint64_t g_mem_snapshot_va = 0;
static std::uint64_t g_mem_snapshot_idx = 0;
static bool g_mem_snapshot_idx_done = false;
static std::uint64_t g_mem_snapshot_after_rva = 0;
static std::uint64_t g_mem_snapshot_delta = 0;
static std::vector<std::pair<std::uint64_t, std::uint64_t>> g_mem_snapshot_due;
static std::uint64_t g_mem_snapshot_last_zero_anchor = 0;
static bool g_trace_target_value = false;
static std::uint64_t g_target_value = 0;
static std::uint64_t g_base_value = 0;
static bool g_target_value_track_active = false;
static bool g_target_value_found = false;
static std::uint64_t g_target_value_anchor_idx = 0;
static std::uint64_t g_target_value_first_idx = 0;
static std::uint64_t g_target_value_first_rip = 0;
static std::uint64_t g_target_value_writer_rip = 0;
static std::uint64_t g_target_value_writer_idx = 0;
static int g_target_value_first_reg = -1;
static std::uint64_t g_target_value_prev_rip = 0;
static std::uint64_t g_target_value_prev_idx = 0;
static std::uint64_t g_target_value_prev_rax = 0;
static std::uint64_t g_target_value_prev_rcx = 0;
static bool g_base_value_found = false;
static std::uint64_t g_base_value_first_idx = 0;
static std::uint64_t g_base_value_first_rip = 0;
static std::uint64_t g_base_value_writer_rip = 0;
static std::uint64_t g_base_value_writer_idx = 0;
static int g_base_value_first_reg = -1;
struct load_window_entry
{
	std::uint64_t idx = 0;
	addr_t rip = 0;
	std::size_t size = 0;
};
static std::vector<load_window_entry> g_load_window_ring;
static bool g_trace_load_window = false;
static bool g_mutate_load_lowbyte = false;
static bool g_mutate_load_done = false;

void print_usage()
{
	LOG_INFO("usage: kernemul [image...]");
	LOG_INFO("examples:");
	LOG_INFO("  kernemul test_driver.sys");
	LOG_INFO("  kernemul test_printf.exe test_seh.exe");
	LOG_INFO("  kernemul test_driver.sys test_user.exe");
}

static void install_mem_write_watch(emu& e, vcpu& cpu, const std::uint64_t watch)
{
	// Write-only: the same slot is read by the frames that reuse it, and those
	// reads would otherwise consume the print budget before the writer appears.
	const auto hook = e.hook_mem(watch, watch + 7, prot_write,
		[watch](vcpu& cpu, addr_t addr, std::size_t size, mem_prot access)
		{
			if (addr > watch + 7 || addr + size <= watch)
				return;
			bool value_match = !g_mem_watch_value;
			for (int r = 0; r < 16 && !value_match; ++r)
				value_match = cpu.reg<std::uint64_t>(static_cast<reg_t>(r)) == g_mem_watch_value;
			if (access & prot_write)
			{
				++g_mem_write_addrs;
				if (!value_match)
					return;
				g_mem_last_write_valid = true;
				g_mem_last_write_idx = g_step_idx ? g_step_idx - 1 : 0;
				g_mem_last_write_rip = cpu.pc();
				g_mem_last_write_rsp = cpu.reg<std::uint64_t>(static_cast<reg_t>(4));
				g_mem_last_write_rax = cpu.reg<std::uint64_t>(static_cast<reg_t>(0));
				g_mem_last_write_rbx = cpu.reg<std::uint64_t>(static_cast<reg_t>(3));
				g_mem_last_write_rcx = cpu.reg<std::uint64_t>(static_cast<reg_t>(1));
				g_mem_last_write_rdx = cpu.reg<std::uint64_t>(static_cast<reg_t>(2));
				g_mem_last_write_r8 = cpu.reg<std::uint64_t>(static_cast<reg_t>(8));
				g_mem_last_write_r9 = cpu.reg<std::uint64_t>(static_cast<reg_t>(9));
			}
			if (!value_match)
				return;
			if (g_mem_write_hits++ >= g_mem_write_max)
				return;
			LOG_INFO("[MEM_WATCH_ACCESS] access={} idx={} rip=0x{:X} addr=0x{:X} size={} rax=0x{:X} rcx=0x{:X} rdx=0x{:X} rbx=0x{:X} rsi=0x{:X} rdi=0x{:X} rbp=0x{:X} r8=0x{:X} r9=0x{:X} r10=0x{:X} r11=0x{:X} r12=0x{:X} r13=0x{:X} r14=0x{:X} r15=0x{:X} rsp=0x{:X}",
				static_cast<unsigned>(access), g_step_idx ? g_step_idx - 1 : 0, cpu.pc(), addr, size,
				cpu.reg<std::uint64_t>(static_cast<reg_t>(0)),
				cpu.reg<std::uint64_t>(static_cast<reg_t>(1)),
				cpu.reg<std::uint64_t>(static_cast<reg_t>(2)),
				cpu.reg<std::uint64_t>(static_cast<reg_t>(3)),
				cpu.reg<std::uint64_t>(static_cast<reg_t>(6)),
				cpu.reg<std::uint64_t>(static_cast<reg_t>(7)),
				cpu.reg<std::uint64_t>(static_cast<reg_t>(5)),
				cpu.reg<std::uint64_t>(static_cast<reg_t>(8)),
				cpu.reg<std::uint64_t>(static_cast<reg_t>(9)),
				cpu.reg<std::uint64_t>(static_cast<reg_t>(10)),
				cpu.reg<std::uint64_t>(static_cast<reg_t>(11)),
				cpu.reg<std::uint64_t>(static_cast<reg_t>(12)),
				cpu.reg<std::uint64_t>(static_cast<reg_t>(13)),
				cpu.reg<std::uint64_t>(static_cast<reg_t>(14)),
				cpu.reg<std::uint64_t>(static_cast<reg_t>(15)),
				cpu.reg<std::uint64_t>(static_cast<reg_t>(4)));
		});
	LOG_INFO("[MEM_WATCH_ARM] ea=0x{:X} bytes=8 hook={:p}", watch,
		static_cast<void*>(hook));

	// The VA hook fires for the guest address; the physical alias duplicates every
	// event and shares the same print budget, so it is not armed.
	return;
	// Unicorn's memory callbacks can report the mapped backing address rather
	// than the guest VA. Arm the translated physical alias too, to distinguish
	// a missing writer from an address-domain mismatch in this diagnostic.
	// The physical alias duplicates every event; skip it when a value filter is
	// set so it does not consume the print budget with identical writes.
	if (g_mem_watch_value)
		return;
	if (const auto space = cpu.curr_addr_space())
	{
		if (const auto pa = e.mem()->virt_to_phys(*space, watch); pa && *pa != watch)
		{
			const auto phys_watch = *pa;
			const auto phys_hook = e.hook_mem(phys_watch, phys_watch + 7, prot_rw,
				[phys_watch](vcpu& cpu, addr_t addr, std::size_t size, mem_prot access)
				{
					if (addr > phys_watch + 7 || addr + size <= phys_watch)
						return;
					if (access & prot_write)
					{
						++g_mem_write_addrs;
						g_mem_last_write_valid = true;
						g_mem_last_write_idx = g_step_idx ? g_step_idx - 1 : 0;
						g_mem_last_write_rip = cpu.pc();
						g_mem_last_write_rsp = cpu.reg<std::uint64_t>(static_cast<reg_t>(4));
						g_mem_last_write_rax = cpu.reg<std::uint64_t>(static_cast<reg_t>(0));
						g_mem_last_write_rbx = cpu.reg<std::uint64_t>(static_cast<reg_t>(3));
						g_mem_last_write_rcx = cpu.reg<std::uint64_t>(static_cast<reg_t>(1));
						g_mem_last_write_rdx = cpu.reg<std::uint64_t>(static_cast<reg_t>(2));
						g_mem_last_write_r8 = cpu.reg<std::uint64_t>(static_cast<reg_t>(8));
						g_mem_last_write_r9 = cpu.reg<std::uint64_t>(static_cast<reg_t>(9));
					}
					if (g_mem_write_hits++ >= g_mem_write_max)
						return;
					LOG_INFO("[MEM_WATCH_ACCESS] domain=physical access={} idx={} rip=0x{:X} addr=0x{:X} size={} rax=0x{:X} r8=0x{:X} r9=0x{:X} rdx=0x{:X} rsp=0x{:X}",
						static_cast<unsigned>(access), g_step_idx ? g_step_idx - 1 : 0, cpu.pc(), addr, size,
						cpu.reg<std::uint64_t>(static_cast<reg_t>(0)),
						cpu.reg<std::uint64_t>(static_cast<reg_t>(8)),
						cpu.reg<std::uint64_t>(static_cast<reg_t>(9)),
						cpu.reg<std::uint64_t>(static_cast<reg_t>(2)),
						cpu.reg<std::uint64_t>(static_cast<reg_t>(4)));
				});
			LOG_INFO("[MEM_WATCH_ARM] domain=physical va=0x{:X} pa=0x{:X} hook={:p}",
				watch, phys_watch, static_cast<void*>(phys_hook));
		}
		else
			LOG_INFO("[MEM_WATCH_TRANSLATE] va=0x{:X} physical_alias=unavailable_or_identical", watch);
	}
	else
		LOG_INFO("[MEM_WATCH_TRANSLATE] va=0x{:X} address_space=unavailable", watch);
}

std::shared_ptr<thread> setup_driver(guest::win_emulator& win,
	const std::shared_ptr<guest::calling_conv>& conv, vcpu& cpu, const std::string& name,
	emu& e)
{
	auto& kernel = win.kernel();
	auto& proc = *kernel.sys_proc;

	const auto driver = krnl::map_img(proc, std::string(target::guest_fs_dir) + name, true);

	if (!driver)
	{
		LOG_ERR("failed to map {}", name);
		return {};
	}

	// ---- LOCAL MEASUREMENT PATCH (not upstream) ----
	// Diagnostics that do not need the per-instruction RIP hook.
	if (const char* mw = std::getenv("KERNEMUL_MEM_WRITE_WATCH"))
		g_mem_write_watch = strtoull(mw, nullptr, 0);
	if (const char* mm = std::getenv("KERNEMUL_MEM_WATCH_MAX"))
		g_mem_write_max = strtoull(mm, nullptr, 0);
	if (const char* mv = std::getenv("KERNEMUL_MEM_WATCH_VALUE"))
		g_mem_watch_value = strtoull(mv, nullptr, 0);

	if (const char* t = std::getenv("KERNEMUL_RIP_TRACE"); t && *t == '1')
	{
		g_mod_base = driver->addr;
		g_mod_size = driver->size;
		g_base_value = g_mod_base + 0x6900;
		g_target_value = g_mod_base + 0x6901;
		g_rip_pages.assign((driver->size + 0xFFF) / 0x1000, 0);
		if (const char* n = std::getenv("KERNEMUL_RIP_STEPS"))
			g_rip_seq_cap = static_cast<std::size_t>(strtoull(n, nullptr, 0));
		if (const char* f = std::getenv("KERNEMUL_SEQ_FROM_RVA"))
			g_seq_from_rva = strtoull(f, nullptr, 0);
		if (const char* lb = std::getenv("KERNEMUL_LAST_BEFORE"))
		{
			const std::string spec(lb);
			std::size_t pos = 0;
			while (pos < spec.size())
			{
				const auto comma = spec.find(',', pos);
				const auto end = comma == std::string::npos ? spec.size() : comma;
				const auto colon = spec.find(':', pos);
				if (colon != std::string::npos && colon < end)
				{
					last_before_watch watch;
					watch.target = strtoull(spec.substr(pos, colon - pos).c_str(), nullptr, 0);
					watch.gate = strtoull(spec.substr(colon + 1, end - colon - 1).c_str(), nullptr, 0);
					if (watch.target && watch.gate)
						g_lb_watches.push_back(watch);
				}
				pos = end + 1;
			}
		}
		if (const char* st = std::getenv("KERNEMUL_SLOT_TRACE"))
			g_slot_trace = (*st == '1');
		if (const char* zb = std::getenv("KERNEMUL_ZERO_BIAS_STORE"))
			g_zero_bias_store = (*zb == '1');
		if (const char* sr = std::getenv("KERNEMUL_MEM_SNAPSHOT_RVA"))
			g_mem_snapshot_rva = strtoull(sr, nullptr, 0);
		if (const char* sv = std::getenv("KERNEMUL_MEM_SNAPSHOT_VA"))
			g_mem_snapshot_va = strtoull(sv, nullptr, 0);
		if (const char* si = std::getenv("KERNEMUL_MEM_SNAPSHOT_IDX"))
			g_mem_snapshot_idx = strtoull(si, nullptr, 0);
		if (const char* sr = std::getenv("KERNEMUL_MEM_SNAPSHOT_AFTER_RVA"))
			g_mem_snapshot_after_rva = strtoull(sr, nullptr, 0);
		if (const char* sd = std::getenv("KERNEMUL_MEM_SNAPSHOT_DELTA"))
			g_mem_snapshot_delta = strtoull(sd, nullptr, 0);
		if (const char* tw = std::getenv("KERNEMUL_TRACE_LOAD_WINDOW"))
		{
			g_trace_load_window = (*tw == '1');
			if (g_trace_load_window)
				g_load_window_ring.resize(1024);
		}
		if (const char* tv = std::getenv("KERNEMUL_TRACE_TARGET_VALUE"))
			g_trace_target_value = (*tv == '1');
		if (const char* rv = std::getenv("KERNEMUL_REG_VALUE_WATCH"))
			g_reg_value_watch = strtoull(rv, nullptr, 0);
		if (const char* rr = std::getenv("KERNEMUL_REG_VALUE_WATCH_REG"))
			g_reg_value_watch_reg = static_cast<int>(strtol(rr, nullptr, 0));
		if (const char* ml = std::getenv("KERNEMUL_MUTATE_LOAD_LOWBYTE"))
			g_mutate_load_lowbyte = (*ml == '1');
		if (const char* g = std::getenv("KERNEMUL_REGS_AT"))
		{
			const std::string spec(g);
			std::size_t pos = 0;
			while (pos <= spec.size() && g_regs_count < 8)
			{
				const auto comma = spec.find(',', pos);
				const auto tok = spec.substr(pos, comma == std::string::npos
					? std::string::npos : comma - pos);
				if (!tok.empty())
					g_regs_rva[g_regs_count++] = strtoull(tok.c_str(), nullptr, 0);
				if (comma == std::string::npos) break;
				pos = comma + 1;
			}
		}
		if (g_rip_seq_cap)
			g_rip_seq.reserve(g_rip_seq_cap);
		const addr_t b = driver->addr;
		const addr_t s = driver->size;
		chunk_emitter::instance().set_base(b);
		// RDTSC is the packer's one entropy source.  Pinning it keeps the
		// guest state (and the flow) reproducible across runs.
		if (const char* w = std::getenv("KERNEMUL_WATCH"))
		{
			std::string spec(w);
			std::size_t pos = 0;
			while (pos <= spec.size())
			{
				const auto comma = spec.find(',', pos);
				const auto tok = spec.substr(pos, comma == std::string::npos
					? std::string::npos : comma - pos);
				if (!tok.empty())
				{
					g_watch_rva.push_back(strtoull(tok.c_str(), nullptr, 0));
					g_watch_first.push_back(0);
					g_watch_last.push_back(0);
				}
				if (comma == std::string::npos) break;
				pos = comma + 1;
			}
		}
		if (const char* wf = std::getenv("KERNEMUL_WATCH_FILE"))
		{
			// Same list as KERNEMUL_WATCH, read from a file so thousands of RVAs do
			// not have to fit in an environment variable.
			std::ifstream stream(wf);
			std::string spec((std::istreambuf_iterator<char>(stream)),
				std::istreambuf_iterator<char>());
			std::size_t pos = 0;
			while (pos <= spec.size())
			{
				const auto comma = spec.find(',', pos);
				auto tok = spec.substr(pos, comma == std::string::npos
					? std::string::npos : comma - pos);
				while (!tok.empty() && std::isspace(static_cast<unsigned char>(tok.front())))
					tok.erase(tok.begin());
				while (!tok.empty() && std::isspace(static_cast<unsigned char>(tok.back())))
					tok.pop_back();
				if (!tok.empty())
				{
					g_watch_rva.push_back(strtoull(tok.c_str(), nullptr, 0));
					g_watch_first.push_back(0);
					g_watch_last.push_back(0);
				}
				if (comma == std::string::npos) break;
				pos = comma + 1;
			}
			LOG_INFO("[RIP_TRACE] watch list file {} -> {} sites", wf, g_watch_rva.size());
		}
		if (const char* r = std::getenv("KERNEMUL_FIRST_IN"))
		{
			const std::string spec(r);
			const auto dash = spec.find('-');
			if (dash != std::string::npos)
			{
				g_in_lo = strtoull(spec.substr(0, dash).c_str(), nullptr, 0);
				g_in_hi = strtoull(spec.substr(dash + 1).c_str(), nullptr, 0);
			}
		}
		// A watch list can be long enough (thousands of store sites) that a linear
		// scan per instruction is not affordable; index it once instead.
		for (std::size_t i = 0; i < g_watch_rva.size(); ++i)
			g_watch_index.emplace(g_watch_rva[i], i);

		LOG_INFO("[RIP_TRACE] tracing module {} at 0x{:X} size 0x{:X} steps_cap={} watch={} first_in=0x{:X}-0x{:X}",
			name, b, s, g_rip_seq_cap, g_watch_rva.size(), g_in_lo, g_in_hi);
		e.hook_code(b, b + s - 1, [b, s, &e](vcpu& cpu, addr_t addr, std::size_t insn_size)
		{
			if (addr >= b && addr < b + s)
			{
				const std::uint64_t rva = addr - b;
				const std::uint64_t idx = g_step_idx++;
				apply_kernemul_mutations(cpu, rva, idx);
				apply_flag_oracle(cpu, rva);
				if (chunk_emitter::instance().active())
				{
					// Watch snapshots read guest bytes through the live vcpu.
					// Rebind once: the address space is shared for the run.
					if (!chunk_emitter::instance().has_memory_reader())
						chunk_emitter::instance().set_memory_reader(
							[&cpu](std::uint64_t va, std::uint8_t* dst,
								std::size_t size)
							{
								try
								{
									std::size_t available = size;
									cpu.read_virt_mem(va, dst, available);
									return available == size;
								}
								catch (...)
								{
									return false;
								}
							});
					std::uint64_t gpr_values[16];
					for (int r = 0; r < 16; ++r)
						gpr_values[r] = cpu.reg<std::uint64_t>(static_cast<reg_t>(r));
					const auto flags = cpu.reg<std::uint64_t>(static_cast<reg_t>(17));
					// RKTRACE stores a fixed 15-byte window at RIP, independent
					// of the guest instruction length.  Hash the same window so
					// the flow digests are directly comparable.
					std::uint8_t bytes[15]{};
					std::size_t bytes_read = sizeof(bytes);
					try { cpu.read_virt_mem(addr, bytes, bytes_read); }
					catch (...)
					{
						bytes_read = std::min<std::size_t>(insn_size, sizeof(bytes));
						if (bytes_read)
						{
							try { cpu.read_virt_mem(addr, bytes, bytes_read); }
							catch (...) { bytes_read = 0; }
						}
					}
					chunk_emitter::instance().observe(idx, addr, flags, gpr_values,
						bytes, bytes_read);
				}
				if (g_slot_trace && rva >= 0x23C105C && rva <= 0x23C3156)
				{
					try
					{
						const auto sp = cpu.reg<std::uint64_t>(static_cast<reg_t>(4));
						const auto v = cpu.read_virt_mem<std::uint64_t>(sp + 8);
						if (!g_slot_trace_seen || v != g_slot_trace_last)
						{
							LOG_INFO("[SLOT_TRACE] idx={} rip=0x{:X} rsp=0x{:X} value=0x{:X} prev_rip=0x{:X}",
								idx, addr, sp, v, g_slot_trace_prev_rip);
							g_slot_trace_last = v;
							g_slot_trace_seen = true;
						}
					}
					catch (...) {}
				}
				if (g_reg_value_watch)
				{
					// Rising edge only: a register can hold the value for a long run,
					// and the interesting event is the step that put it there.
					static bool held[16] = {};
					for (int r = 0; r < 16; ++r)
					{
						const bool now =
							cpu.reg<std::uint64_t>(static_cast<reg_t>(r)) == g_reg_value_watch;
						if (now && !held[r] &&
							(g_reg_value_watch_reg < 0 || r == g_reg_value_watch_reg) &&
							++g_reg_value_hits <= 4096)
							LOG_INFO("[REG_VALUE_WATCH] idx={} rip=0x{:X} reg=r{} "
								"prev_rip=0x{:X} prev_rva=0x{:X} rsp=0x{:X}",
								idx, addr, r, g_slot_trace_prev_rip,
								g_slot_trace_prev_rip > b ? g_slot_trace_prev_rip - b : 0,
								cpu.reg<std::uint64_t>(static_cast<reg_t>(4)));
						held[r] = now;
					}
				}
				g_slot_trace_prev_rip = addr;
				if (g_zero_bias_store && rva == 0x17276C)
					cpu.reg(static_cast<reg_t>(10), std::uint64_t{0});
				if (g_trace_target_value && rva == 0x6938)
				{
					g_target_value_track_active = true;
					g_target_value_found = false;
					g_base_value_found = false;
					g_target_value_anchor_idx = idx;
					g_target_value_prev_rip = addr;
					g_target_value_prev_idx = idx;
					g_target_value_prev_rax = cpu.reg<std::uint64_t>(static_cast<reg_t>(0));
					g_target_value_prev_rcx = cpu.reg<std::uint64_t>(static_cast<reg_t>(1));
				}
				if (g_target_value_track_active && (!g_target_value_found || !g_base_value_found))
				{
					for (int r = 0; r < 16; ++r)
					{
						const auto reg_value = cpu.reg<std::uint64_t>(static_cast<reg_t>(r));
						if (!g_base_value_found && reg_value == g_base_value)
						{
							g_base_value_found = true;
							g_base_value_first_idx = idx;
							g_base_value_first_rip = addr;
							g_base_value_writer_idx = g_target_value_prev_idx;
							g_base_value_writer_rip = g_target_value_prev_rip;
							g_base_value_first_reg = r;
						}
						if (!g_target_value_found && reg_value == g_target_value)
						{
							g_target_value_found = true;
							g_target_value_first_idx = idx;
							g_target_value_first_rip = addr;
							g_target_value_writer_idx = g_target_value_prev_idx;
							g_target_value_writer_rip = g_target_value_prev_rip;
							g_target_value_first_reg = r;
							std::uint8_t bytes[15]{};
							const auto distance = addr > g_target_value_writer_rip
								? addr - g_target_value_writer_rip : 0;
							const auto n = static_cast<std::size_t>(std::min<std::uint64_t>(distance, sizeof(bytes)));
							std::string hex;
							if (n)
							{
								try
								{
									cpu.read_virt_mem(g_target_value_writer_rip, bytes, n);
									for (std::size_t j = 0; j < n; ++j)
										hex += fmt::format("{:02X}", static_cast<unsigned>(bytes[j]));
								}
								catch (...)
								{
									hex = "unreadable";
								}
							}
							LOG_INFO("[TARGET_VALUE_ORIGIN] from=0x{:X} idx={} bytes={} pre_rax=0x{:X} pre_rcx=0x{:X} observed_at=0x{:X} post_rax=0x{:X} post_rcx=0x{:X}",
								g_target_value_writer_rip, g_target_value_writer_idx, hex,
								g_target_value_prev_rax, g_target_value_prev_rcx, addr,
								cpu.reg<std::uint64_t>(static_cast<reg_t>(0)),
								cpu.reg<std::uint64_t>(static_cast<reg_t>(1)));
						}
					}
				}
				if (g_trace_target_value && rva == 0x2ADE37B)
				{
					if (g_base_value_found)
						LOG_INFO("[BASE_VALUE_TRACK] value=0x{:X} anchor_idx={} first_seen_idx={} observed_rip=0x{:X} after_rip=0x{:X} after_idx={} reg_index={}",
							g_base_value, g_target_value_anchor_idx, g_base_value_first_idx,
							g_base_value_first_rip, g_base_value_writer_rip,
							g_base_value_writer_idx, g_base_value_first_reg);
					else
						LOG_INFO("[BASE_VALUE_TRACK] value=0x{:X} anchor_idx={} not_seen_in_gprs_before_load",
							g_base_value, g_target_value_anchor_idx);
					if (g_target_value_found)
						LOG_INFO("[TARGET_VALUE_TRACK] value=0x{:X} anchor_idx={} first_seen_idx={} observed_rip=0x{:X} after_rip=0x{:X} after_idx={} reg_index={}",
							g_target_value, g_target_value_anchor_idx, g_target_value_first_idx,
							g_target_value_first_rip, g_target_value_writer_rip,
							g_target_value_writer_idx, g_target_value_first_reg);
					else
						LOG_INFO("[TARGET_VALUE_TRACK] value=0x{:X} anchor_idx={} not_seen_in_gprs_before_load",
							g_target_value, g_target_value_anchor_idx);
					g_target_value_track_active = false;
				}
				if (g_target_value_track_active && (!g_target_value_found || !g_base_value_found))
				{
					g_target_value_prev_rip = addr;
					g_target_value_prev_idx = idx;
					g_target_value_prev_rax = cpu.reg<std::uint64_t>(static_cast<reg_t>(0));
					g_target_value_prev_rcx = cpu.reg<std::uint64_t>(static_cast<reg_t>(1));
				}
				if (g_trace_load_window && !g_load_window_ring.empty())
					g_load_window_ring[idx % g_load_window_ring.size()] = { idx, addr, insn_size };
				if (rva == 0x2ADE37B)
				{
					LOG_INFO("[TARGET_LOAD_IDX] idx={} rip=0x{:X} last_zero_return_idx={} delta={}",
						idx, addr, g_mem_snapshot_last_zero_anchor,
						g_mem_snapshot_last_zero_anchor ? idx - g_mem_snapshot_last_zero_anchor : 0);
					if (g_trace_load_window)
					{
						const auto first = idx > 990 ? idx - 990 : 0;
						for (auto seq = first; seq < idx; ++seq)
						{
							const auto& entry = g_load_window_ring[seq % g_load_window_ring.size()];
							if (entry.idx != seq)
								continue;
							std::uint8_t bytes[15]{};
							const auto n = std::min<std::size_t>(entry.size, sizeof(bytes));
							try
							{
								cpu.read_virt_mem(entry.rip, bytes, n);
								std::string hex;
								for (std::size_t j = 0; j < n; ++j)
									hex += fmt::format("{:02X}", static_cast<unsigned>(bytes[j]));
								LOG_INFO("[LOAD_WINDOW] seq={} idx={} rip=0x{:X} size={} bytes={}",
									seq - first, seq, entry.rip, n, hex);
							}
							catch (const std::exception& ex)
							{
								LOG_INFO("[LOAD_WINDOW] seq={} idx={} rip=0x{:X} unreadable: {}",
									seq - first, seq, entry.rip, ex.what());
							}
						}
					}
				}
				if (g_mem_snapshot_after_rva && g_mem_snapshot_va &&
					rva == g_mem_snapshot_after_rva && g_mem_snapshot_due.size() < 128)
				{
					try
					{
						const auto value = cpu.read_virt_mem<std::uint64_t>(g_mem_snapshot_va);
						if ((value & 0xFF) == 0)
						{
							g_mem_snapshot_last_zero_anchor = idx;
							if (g_mem_snapshot_delta)
							{
								g_mem_snapshot_due.emplace_back(idx + g_mem_snapshot_delta, idx);
								LOG_INFO("[MEM_SNAPSHOT_ANCHOR] idx={} rip=0x{:X} va=0x{:X} lowbyte=0x00 sample_idx={}",
									idx, addr, g_mem_snapshot_va, idx + g_mem_snapshot_delta);
							}
							else
								LOG_INFO("[MEM_SNAPSHOT_ANCHOR] idx={} rip=0x{:X} va=0x{:X} lowbyte=0x00",
									idx, addr, g_mem_snapshot_va);
						}
					}
					catch (const std::exception& ex)
					{
						LOG_INFO("[MEM_SNAPSHOT_ANCHOR] idx={} unreadable: {}", idx, ex.what());
					}
				}
				for (auto it = g_mem_snapshot_due.begin(); it != g_mem_snapshot_due.end();)
				{
					if (idx != it->first)
					{
						++it;
						continue;
					}
					try
					{
						const auto value = cpu.read_virt_mem<std::uint64_t>(g_mem_snapshot_va);
						LOG_INFO("[MEM_SNAPSHOT_REL] anchor_idx={} idx={} rip=0x{:X} va=0x{:X} qword=0x{:016X} lowbyte=0x{:02X}",
							it->second, idx, addr, g_mem_snapshot_va, value, static_cast<unsigned>(value & 0xFF));
					}
					catch (const std::exception& ex)
					{
						LOG_INFO("[MEM_SNAPSHOT_REL] anchor_idx={} idx={} unreadable: {}", it->second, idx, ex.what());
					}
					it = g_mem_snapshot_due.erase(it);
				}
				if (g_mem_snapshot_rva && g_mem_snapshot_va && rva == g_mem_snapshot_rva)
				{
					try
					{
						const auto value = cpu.read_virt_mem<std::uint64_t>(g_mem_snapshot_va);
						LOG_INFO("[MEM_SNAPSHOT] idx={} rip=0x{:X} va=0x{:X} qword=0x{:016X} lowbyte=0x{:02X}",
							idx, addr, g_mem_snapshot_va, value, static_cast<unsigned>(value & 0xFF));
					}
					catch (const std::exception& ex)
					{
						LOG_INFO("[MEM_SNAPSHOT] idx={} rip=0x{:X} va=0x{:X} unreadable: {}",
							idx, addr, g_mem_snapshot_va, ex.what());
					}
				}
				if (g_mem_snapshot_idx && g_mem_snapshot_va && !g_mem_snapshot_idx_done && idx == g_mem_snapshot_idx)
				{
					try
					{
						const auto value = cpu.read_virt_mem<std::uint64_t>(g_mem_snapshot_va);
						LOG_INFO("[MEM_SNAPSHOT_IDX] idx={} rip=0x{:X} va=0x{:X} qword=0x{:016X} lowbyte=0x{:02X}",
							idx, addr, g_mem_snapshot_va, value, static_cast<unsigned>(value & 0xFF));
					}
					catch (const std::exception& ex)
					{
						LOG_INFO("[MEM_SNAPSHOT_IDX] idx={} rip=0x{:X} va=0x{:X} unreadable: {}",
							idx, addr, g_mem_snapshot_va, ex.what());
					}
					g_mem_snapshot_idx_done = true;
				}
				if (g_mem_write_watch && !g_mem_watch_load_mapping_logged && rva == 0x2ADE37B)
				{
					const auto load_ea = cpu.reg<std::uint64_t>(static_cast<reg_t>(0)) +
						cpu.reg<std::uint64_t>(static_cast<reg_t>(2)) * 2 + 1;
					const auto space = cpu.curr_addr_space();
					const auto pa = space ? e.mem()->virt_to_phys(*space, load_ea) : std::nullopt;
					LOG_INFO("[MEM_WATCH_AT_LOAD] idx={} rip=0x{:X} va=0x{:X} cr3=0x{:X} pa=0x{:X} mapped={}",
						idx, addr, load_ea, cpu.reg<std::uint64_t>(static_cast<reg_t>(x86::cr3)),
						pa.value_or(0), pa.has_value());
					if (pa && (g_mem_write_watch == load_ea || g_mem_write_watch == *pa))
					{
						if (g_mem_last_write_valid)
							LOG_INFO("[MEM_WATCH_LAST_WRITER] idx={} rip=0x{:X} rsp=0x{:X} rax=0x{:X} rbx=0x{:X} rcx=0x{:X} rdx=0x{:X} r8=0x{:X} r9=0x{:X}",
								g_mem_last_write_idx, g_mem_last_write_rip, g_mem_last_write_rsp,
								g_mem_last_write_rax, g_mem_last_write_rbx, g_mem_last_write_rcx,
								g_mem_last_write_rdx, g_mem_last_write_r8, g_mem_last_write_r9);
						else
							LOG_INFO("[MEM_WATCH_LAST_WRITER] no guest CPU write observed before load");
					}
					g_mem_watch_load_mapping_logged = true;
				}
				if (g_mutate_load_lowbyte && !g_mutate_load_done && rva == 0x2ADE37B)
				{
					const auto old_rax = cpu.reg<std::uint64_t>(static_cast<reg_t>(0));
					const auto old_rdx = cpu.reg<std::uint64_t>(static_cast<reg_t>(2));
					const auto ea = old_rax + old_rdx * 2 + 1;
					const auto old_value = cpu.read_virt_mem<std::uint64_t>(ea);
					g_mutate_load_done = true;
					if ((old_value & 0xFF) == 1)
					{
						const auto new_value = old_value & ~std::uint64_t{0xFF};
						cpu.write_virt_mem<std::uint64_t>(ea, new_value);
						LOG_INFO("[MUTATE_LOAD_LOWBYTE] idx={} rip=0x{:X} ea=0x{:X} old=0x{:016X} new=0x{:016X}",
							idx, addr, ea, old_value, new_value);
					}
					else
						LOG_INFO("[MUTATE_LOAD_LOWBYTE] SKIP idx={} ea=0x{:X} value=0x{:016X} (low byte != 1)",
							idx, ea, old_value);
				}
				if (g_regs_count)
				{
					for (int t = 0; t < g_regs_count; ++t)
						if (!g_regs_hit[t] && rva == g_regs_rva[t])
						{
							for (int r = 0; r < 18; ++r)
								g_regs_vals[t][r] = cpu.reg<std::uint64_t>(static_cast<reg_t>(r));
							g_regs_hit[t] = 1;
						}
				}
				g_rip_pages[rva >> 12] = 1;
				++g_rip_total;
				for (auto& watch : g_lb_watches)
				{
					if (!watch.snapped && rva == watch.target)
					{
						for (int r = 0; r < 18; ++r)
							watch.regs[r] = cpu.reg<std::uint64_t>(static_cast<reg_t>(r));
						watch.idx = static_cast<long long>(idx);
						watch.have = true;
					}
					if (!watch.snapped && rva == watch.gate)
					{
						watch.snapped = true;
						watch.gate_idx = static_cast<long long>(idx);
					}
				}
				if (addr < g_rip_min) g_rip_min = addr;
				if (addr > g_rip_max) g_rip_max = addr;
				if (g_seq_from_rva && !g_seq_started && rva == g_seq_from_rva)
					g_seq_started = 1;
				if ((!g_seq_from_rva || g_seq_started) && g_rip_seq.size() < g_rip_seq_cap)
					g_rip_seq.push_back(static_cast<std::uint32_t>(rva));				if (!g_in_done && g_in_hi > g_in_lo && rva >= g_in_lo && rva < g_in_hi)
				{
					g_in_idx = idx;
					g_in_rva = rva;
					g_in_done = 1;
				}
				if (!g_watch_index.empty())
				{
					const auto w = g_watch_index.find(rva);
					if (w != g_watch_index.end())
					{
						if (!g_watch_first[w->second])
							g_watch_first[w->second] = idx + 1;
						g_watch_last[w->second] = idx + 1;
					}
				}
			}
		});
	}
	// ---- diagnostics that do not need the per-instruction hook ----
	if (const char* tsc = std::getenv("KERNEMUL_RDTSC_VALUE"))
	{
		const std::uint64_t value = std::strtoull(tsc, nullptr, 0);
		e.hook_insn(0, std::numeric_limits<addr_t>::max(),
			hook_insn_t::rdtsc, [value](vcpu& cpu)
			{
				cpu.reg(static_cast<reg_t>(0),
					value & UINT64_C(0xFFFFFFFF));
				cpu.reg(static_cast<reg_t>(2), value >> 32);
				return true;
			});
		LOG_INFO("[RDTSC] pinned to 0x{:X}", value);
	}


	if (g_mem_write_watch)
		g_arm_pending_watch = [](emu& e, vcpu& cpu) { install_mem_write_watch(e, cpu, g_mem_write_watch); };


	// Up to the first dot, not the last: a variant build is named <service>.<variant>.sys, and
	// stem() would make vgk.calvin.sys a service called "vgk.calvin" that no driver expects.
	const auto file = std::filesystem::path(name).filename().string();
	const auto service = widen_string(file.substr(0, file.find('.')));

	const auto args = kernel.create_driver(*driver, service);

	const auto entry = win.create_kernel_thread(cpu, driver->entry_point);

	if (!entry)
	{
		LOG_ERR("failed to create a thread for {}", name);
		return {};
	}

	conv->set_arg(cpu, *entry, 0, args.driver_object.address());
	conv->set_arg(cpu, *entry, 1, args.registry_path.address());

	// The watched slot can only be hooked once the thread stack exists.
	if (g_arm_pending_watch)
	{
		auto arm = std::move(g_arm_pending_watch);
		g_arm_pending_watch = nullptr;
		arm(e, cpu);
	}

	return entry;
}

bool setup_user_process(guest::win_emulator& win, vcpu& cpu, const std::string& name)
{
	const auto app = win.create_user_process(cpu, name);

	if (!app.thread)
	{
		LOG_ERR("failed to create a process for {}", name);
		return false;
	}

	return true;
}

}

int main(const int argc, const char* const* const argv)
{
	LOG_INFO("kernemul targeting {}", target::name);

	std::vector<std::string> names(argv + 1, argv + argc);

	if (names.empty())
	{
		print_usage();
		return 1;
	}

	std::vector<guest_image> images;
	images.reserve(names.size());

	for (auto& name : names)
	{
		if (!std::filesystem::exists(std::string(target::guest_fs_dir) + name))
		{
			LOG_ERR("{} is not in {}", name, target::guest_fs_dir);
			return 1;
		}

		const bool driver = name.ends_with(".sys");

		images.push_back({ std::move(name), driver, {} });
	}

	auto mem = std::make_shared<guest::mmu>();
	auto conv = std::make_shared<guest::calling_conv>();
	auto e = make_guest_emu(mem, conv);

	guest::win_emulator win(e);

	// ---- LOCAL MEASUREMENT PATCH (not upstream): deterministic trace support ----
	// Multi-vCPU scheduling makes the executed-instruction count vary run to run,
	// which defeats cross-runtime trace alignment. KERNEMUL_VCPUS pins it.
	std::size_t vcpus = vcpu_count;
	if (const char* v = std::getenv("KERNEMUL_VCPUS"))
	{
		const auto n = std::strtoull(v, nullptr, 0);
		if (n >= 1 && n <= 64)
			vcpus = static_cast<std::size_t>(n);
	}
	win.create_vcpus(vcpus);

	const auto cpu = win.cpus().front();

	set_log_cpu(cpu.get());

	for (auto& image : images)
	{
		if (image.driver)
		{
			image.entry = setup_driver(win, conv, *cpu, image.name, *e);

			if (!image.entry)
				return 1;
		}
		else if (!setup_user_process(win, *cpu, image.name))
		{
			return 1;
		}
	}

	win.run_all();

	for (const auto& image : images)
	{
		if (image.entry)
			LOG_INFO("{} returned, status=0x{:X}", image.name, conv->read_ret(*cpu, *image.entry));
		else
			LOG_INFO("{} finished", image.name);
	}

	// ---- LOCAL MEASUREMENT PATCH (not upstream) ----
	if (g_mem_write_watch)
	{
		if (g_mem_last_write_valid)
			LOG_INFO("[MEM_WATCH_SUMMARY] watch=0x{:X} hits={} writes={} last_writer idx={} rip=0x{:X} rsp=0x{:X} rax=0x{:X} rbx=0x{:X} rcx=0x{:X} rdx=0x{:X} r8=0x{:X} r9=0x{:X}",
				g_mem_write_watch, g_mem_write_hits, g_mem_write_addrs,
				g_mem_last_write_idx, g_mem_last_write_rip, g_mem_last_write_rsp,
				g_mem_last_write_rax, g_mem_last_write_rbx, g_mem_last_write_rcx,
				g_mem_last_write_rdx, g_mem_last_write_r8, g_mem_last_write_r9);
		else
			LOG_INFO("[MEM_WATCH_SUMMARY] watch=0x{:X} hits={} no write observed",
				g_mem_write_watch, g_mem_write_hits);
	}

	if (!g_rip_pages.empty())
	{
		std::uint64_t pages = 0;
		for (auto v : g_rip_pages) pages += v ? 1 : 0;
		LOG_INFO("[RIP_TRACE] module base=0x{:X} size=0x{:X} instrs={} pages={}/{} min=0x{:X} max=0x{:X}",
			g_mod_base, g_mod_size, g_rip_total, pages, g_rip_pages.size(),
			g_rip_total ? g_rip_min : 0, g_rip_max);
		if (g_in_hi > g_in_lo)
		{
			if (g_in_done)
				LOG_INFO("[RIP_TRACE] first_in 0x{:X}-0x{:X}: idx={} rva=0x{:X}",
					g_in_lo, g_in_hi, g_in_idx, g_in_rva);
			else
				LOG_INFO("[RIP_TRACE] first_in 0x{:X}-0x{:X}: NOT REACHED", g_in_lo, g_in_hi);
		}
		for (std::size_t i = 0; i < g_watch_rva.size(); ++i)
			LOG_INFO("[RIP_TRACE] watch 0x{:X}: first_idx={} last_idx={}",
				g_watch_rva[i],
				g_watch_first[i] ? static_cast<long long>(g_watch_first[i] - 1) : -1LL,
				g_watch_last[i] ? static_cast<long long>(g_watch_last[i] - 1) : -1LL);
		static const char* rnames[18] = {"rax","rcx","rdx","rbx","rsp","rbp","rsi","rdi",
			"r8","r9","r10","r11","r12","r13","r14","r15","rip","rflags"};
		for (int t = 0; t < g_regs_count; ++t)
		{
			if (!g_regs_hit[t])
			{
				LOG_INFO("[RIP_TRACE] regs_at 0x{:X}: NOT REACHED", g_regs_rva[t]);
				continue;
			}
			LOG_INFO("[RIP_TRACE] regs_at 0x{:X}:", g_regs_rva[t]);
			for (int r = 0; r < 18; ++r)
				LOG_INFO("[RIP_TRACE]   {:>6}={:016X}", rnames[r], g_regs_vals[t][r]);
		}
		for (const auto& watch : g_lb_watches)
		{
			if (!watch.snapped)
				LOG_INFO("[RIP_TRACE] last_before 0x{:X}<-0x{:X}: gate NOT REACHED",
					watch.target, watch.gate);
			else if (!watch.have)
				LOG_INFO("[RIP_TRACE] last_before 0x{:X}<-0x{:X}: target NOT SEEN before gate idx={}",
					watch.target, watch.gate, watch.gate_idx);
			else
			{
				LOG_INFO("[RIP_TRACE] last_before 0x{:X}<-0x{:X}: idx={} (gate idx={})",
					watch.target, watch.gate, watch.idx, watch.gate_idx);
				for (int r = 0; r < 18; ++r)
					LOG_INFO("[RIP_TRACE]   {:>6}={:016X}", rnames[r], watch.regs[r]);
			}
		}
		if (std::FILE* f = std::fopen("/tmp/opencode/kernemul-rip-pages.txt", "w"))
		{
			for (std::size_t i = 0; i < g_rip_pages.size(); ++i)
				if (g_rip_pages[i])
					std::fprintf(f, "0x%llX\n", static_cast<unsigned long long>(g_mod_base + i * 0x1000));
			std::fclose(f);
			LOG_INFO("[RIP_TRACE] wrote /tmp/opencode/kernemul-rip-pages.txt");
		}
		if (!g_rip_seq.empty())
		{
			if (std::FILE* f = std::fopen("/tmp/opencode/kernemul-rip-seq.bin", "wb"))
			{
				std::fwrite("KSEQ1\0", 1, 6, f);
				std::uint32_t prev = 0;
				std::vector<std::uint8_t> buf;
				buf.reserve(1 << 20);
				for (std::uint32_t rva : g_rip_seq)
				{
					const std::int64_t d = static_cast<std::int64_t>(rva)
						- static_cast<std::int64_t>(prev);
					prev = rva;
					std::uint64_t z = (static_cast<std::uint64_t>(d) << 1)
						^ static_cast<std::uint64_t>(d >> 63);
					while (z >= 0x80)
					{
						buf.push_back(static_cast<std::uint8_t>(z | 0x80));
						z >>= 7;
					}
					buf.push_back(static_cast<std::uint8_t>(z));
					if (buf.size() >= (1u << 20))
					{
						std::fwrite(buf.data(), 1, buf.size(), f);
						buf.clear();
					}
				}
				if (!buf.empty())
					std::fwrite(buf.data(), 1, buf.size(), f);
				std::fclose(f);
				LOG_INFO("[RIP_TRACE] wrote /tmp/opencode/kernemul-rip-seq.bin steps={}",
					g_rip_seq.size());
			}
		}
	}

	return 0;
}
