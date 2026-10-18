#include "target.hpp"
#include "kernel/map.hpp"
#include "kernel/win/win_kernel.hpp"
#include "emu/calling_conv.hpp"
#include "util/log.hpp"
#include "util/string.hpp"

#include <algorithm>
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
static std::uint64_t g_in_lo = 0, g_in_hi = 0;
static std::uint64_t g_in_idx = 0, g_in_rva = 0;
static int g_in_done = 0;
static std::uint64_t g_step_idx = 0;
// record the stream only from the first hit of this RVA (allow-pair analogue)
static std::uint64_t g_seq_from_rva = 0;
static int g_seq_started = 0;
// one-shot guest register dump at given RVAs (pre-state of that instruction)
static std::uint64_t g_regs_rva[8];
static std::uint64_t g_regs_vals[8][18];
static int g_regs_hit[8];
static int g_regs_count = 0;

void print_usage()
{
	LOG_INFO("usage: kernemul [image...]");
	LOG_INFO("examples:");
	LOG_INFO("  kernemul test_driver.sys");
	LOG_INFO("  kernemul test_printf.exe test_seh.exe");
	LOG_INFO("  kernemul test_driver.sys test_user.exe");
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
	if (const char* t = std::getenv("KERNEMUL_RIP_TRACE"); t && *t == '1')
	{
		g_mod_base = driver->addr;
		g_mod_size = driver->size;
		g_rip_pages.assign((driver->size + 0xFFF) / 0x1000, 0);
		if (const char* n = std::getenv("KERNEMUL_RIP_STEPS"))
			g_rip_seq_cap = static_cast<std::size_t>(strtoull(n, nullptr, 0));
		if (const char* f = std::getenv("KERNEMUL_SEQ_FROM_RVA"))
			g_seq_from_rva = strtoull(f, nullptr, 0);
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
				}
				if (comma == std::string::npos) break;
				pos = comma + 1;
			}
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
		LOG_INFO("[RIP_TRACE] tracing module {} at 0x{:X} size 0x{:X} steps_cap={} watch={} first_in=0x{:X}-0x{:X}",
			name, b, s, g_rip_seq_cap, g_watch_rva.size(), g_in_lo, g_in_hi);
		e.hook_code(b, b + s - 1, [b, s](vcpu& cpu, addr_t addr, std::size_t)
		{
			if (addr >= b && addr < b + s)
			{
				const std::uint64_t rva = addr - b;
				const std::uint64_t idx = g_step_idx++;
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
				for (std::size_t i = 0; i < g_watch_rva.size(); ++i)
					if (!g_watch_first[i] && rva == g_watch_rva[i])
						g_watch_first[i] = idx + 1;
			}
		});
	}

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
			LOG_INFO("[RIP_TRACE] watch 0x{:X}: first_idx={}",
				g_watch_rva[i], g_watch_first[i] ? static_cast<long long>(g_watch_first[i] - 1) : -1LL);
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
