#pragma once

//
// CV3-lite: emit CV1-compatible chunk checkpoints from Kernemul.
//
// The file is JSON Lines (kchunk-v1); with the per-step CPU digests enabled it
// also carries "cpu":"fnv1a64" in the header and cpu_raw64/cpu_sym64 in every
// record.  Flow is FNV-1a 64 over the same bytes
// the Python side uses, so LWK .chunk files and Kernemul .kchunk files can be
// compared directly.  CPU state is emitted raw at each boundary; the Python
// reader hashes it with the same BLAKE2b-128 rules as the LWK checkpoints.
//
// Enable with KERNEMUL_CHUNK_OUT=/path/file.kchunk.  Optional:
//   KERNEMUL_CHUNK_KIND=insn|block   (default insn)
//   KERNEMUL_CHUNK_EVERY=4096        (instructions per chunk)
//   KERNEMUL_CHUNK_BLOCKS=16         (blocks per chunk for kind=block)
//   KERNEMUL_CHUNK_MODULE=target_driver.sys
//   KERNEMUL_CHUNK_IMAGE_SIZE=0x...
//   KERNEMUL_CHUNK_WATCH=name:address:size,...  (guest bytes at each boundary)
//   KERNEMUL_CHUNK_WRITES=1                     (ordered write digest)
//   KERNEMUL_CHUNK_WRITE_LOG=N                  (first N writes per chunk)
//   KERNEMUL_CHUNK_CTRL=1                      (ordered indirect control-target
//                                               digest)
//   KERNEMUL_CHUNK_CTRL_LOG=N                  (first N indirect transfers per
//                                               chunk, max 64, default 16)
//   KERNEMUL_CHUNK_CTRL_STACK=<base>:<size>    (classify stack targets)
//   KERNEMUL_CHUNK_CTRL_THUNK=<base>:<size>    (classify thunk/other-module)
//   KERNEMUL_CHUNK_CTRL_POOL=<base>:<size>     (classify pool targets)
//   KERNEMUL_CHUNK_STACK=<base>:<size>[,...]   (cpu_sym64 stack classification)
//   KERNEMUL_CHUNK_POOL=<base>:<size>[,...]    (cpu_sym64 pool classification)
//   KERNEMUL_CHUNK_NO_CPU=1                    (debug: disable the per-step
//                                               CPU digests, emit kchunk-v1)
//
// cpu_raw64 / cpu_sym64 are per-chunk FNV-1a 64 digests accumulated over every
// observed step (the same rule as src/trace/lwk_chunk.c):
//   raw - 16 GPRs little-endian, then RFLAGS.
//   sym - for each GPR (class:u8, payload:u64): module -> 1 (payload = RVA),
//         configured stack -> 2 (payload 0), configured pool -> 3 (payload 0),
//         otherwise 0 (payload = value); RFLAGS always class 0 (payload =
//         rflags).  RFLAGS is masked with 0xCD5 (architectural subset).
// Both emitters must produce identical bytes for equivalent state.
//
// KERNEMUL_CHUNK_WRITES mirrors the LWK decoder (src/trace/lwk_chunk.c and
// lwk-tools/chunk_domains.py): mov r/m,r 0x88/0x89, mov r/m,imm 0xC7 /0,
// push r64 0x50-0x57 and call rel32 0xE8, hashed in execution order as
// (effective address RVA, width, value).  Stores whose address cannot be
// decoded are counted in writes_skipped.
//
// KERNEMUL_CHUNK_CTRL mirrors the same decoder for indirect control transfers
// (jmp/call r/m64 0xFF /2 and /4, ret 0xC3, ret imm16 0xC2, ModRM/SIB/disp
// memory operands).  Each transfer is hashed as (ordinal, target_class,
// target), target rebased to the module RVA when it is inside the image.
// Classes: 0 image/RVA, 1 stack, 2 pool/other, 3 thunk/other-module,
// 4 unread (the target lives in memory and the vcpu read failed).
//

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

class chunk_emitter
{
public:
	static chunk_emitter& instance()
	{
		static chunk_emitter value;
		return value;
	}

	bool active() const { return active_; }
	std::uint64_t chunk_index() const { return index_; }

	using memory_reader =
		std::function<bool(std::uint64_t, std::uint8_t*, std::size_t)>;

	void set_memory_reader(memory_reader reader)
	{
		memory_reader_ = std::move(reader);
	}

	bool has_memory_reader() const { return static_cast<bool>(memory_reader_); }

	void set_base(std::uint64_t base)
	{
		if (base_ == 0)
			base_ = base;
	}

	void observe(std::uint64_t seq, std::uint64_t rip, std::uint64_t rflags,
		const std::uint64_t* gprs, const std::uint8_t* opcode,
		std::size_t length)
	{
		if (!active_)
			return;
		if (!pairs_ready_)
			initialize_pairs();
		if (!region_allows(rva(rip)))
			return;
		if (length && opcode_ignored(opcode, length))
			return;
		if (length && gprs && is_rep_string_opcode(opcode, length) &&
			gprs[1] == 0)
			return;
		if (!header_written_)
			write_header();

		if (stop_after_dump_ && dump_steps_ && dumped_ >= dump_steps_)
		{
			// One-shot dump for divergence probes: stop as soon as the window is out.
			finish_stream();
			std::_Exit(0);
		}

		if (started_ && boundary_reached())
		{
			write_record(true, rip, rflags, gprs);
			start_chunk();
		}
		if (!started_)
		{
			first_seq_ = seq;
			first_rva_ = rva(rip);
			flow_ = kFnvOffset;
			started_ = true;
		}

		const std::uint64_t instruction_rva = rva(rip);
		if (dump_rva_ && !dump_armed_ && instruction_rva == dump_rva_)
		{
			dump_armed_ = true;
			dump_from_ = seq;
		}
		if (dump_steps_ && seq >= dump_from_ && dumped_ < dump_steps_)
		{
			if (dump_bin_)
				std::fwrite(&instruction_rva, sizeof(instruction_rva), 1, dump_bin_);
			if (dump_text_)
			{
			std::fprintf(stderr, "[KCHUNK_STEP] seq=%llu rva=0x%llX op=",
				static_cast<unsigned long long>(seq),
				static_cast<unsigned long long>(instruction_rva));
			for (std::size_t dump_index = 0; dump_index < length; ++dump_index)
				std::fprintf(stderr, "%02X", opcode[dump_index]);
			if (dump_regs_ && gprs)
				std::fprintf(stderr, " rax=0x%016llX rbx=0x%016llX rcx=0x%016llX rdx=0x%016llX"
					" rsi=0x%016llX rdi=0x%016llX rbp=0x%016llX rsp=0x%016llX"
					" r8=0x%016llX r9=0x%016llX r10=0x%016llX r11=0x%016llX"
					" r12=0x%016llX r13=0x%016llX r14=0x%016llX r15=0x%016llX"
					" rflags=0x%016llX",
					static_cast<unsigned long long>(gprs[0]),
					static_cast<unsigned long long>(gprs[3]),
					static_cast<unsigned long long>(gprs[1]),
					static_cast<unsigned long long>(gprs[2]),
					static_cast<unsigned long long>(gprs[6]),
					static_cast<unsigned long long>(gprs[7]),
					static_cast<unsigned long long>(gprs[5]),
					static_cast<unsigned long long>(gprs[4]),
					static_cast<unsigned long long>(gprs[8]),
					static_cast<unsigned long long>(gprs[9]),
					static_cast<unsigned long long>(gprs[10]),
					static_cast<unsigned long long>(gprs[11]),
					static_cast<unsigned long long>(gprs[12]),
					static_cast<unsigned long long>(gprs[13]),
					static_cast<unsigned long long>(gprs[14]),
					static_cast<unsigned long long>(gprs[15]),
					static_cast<unsigned long long>(rflags));
			std::fprintf(stderr, "\n");
			}
			++dumped_;
		}
		last_seq_ = seq;
		last_rva_ = instruction_rva;
		if (writes_enabled_ && opcode && gprs)
		{
			store decoded{};
			const int status = decode_store(opcode, length, rip, gprs, decoded);
			if (status > 0)
				record_write(seq, decoded);
			else if (status < 0)
				++writes_skipped_;
			else
			{
				std::uint64_t ea = 0, bytes = 0;
				store other{};
				if (decode_block_copy(opcode, length, gprs, ea, bytes))
					record_block_write(seq, ea, bytes);
				else if (decode_other_store(opcode, length, gprs, rip, other) > 0)
					record_write(seq, other);
			}
		}
		if (ctrl_enabled_ && opcode && gprs)
		{
			ctrl_entry decoded{};
			const int status = decode_ctrl(opcode, length, rip, gprs, decoded);
			if (status > 0)
				record_ctrl(instruction_rva, decoded);
			else if (status < 0)
				++ctrl_skipped_;
		}
		flow_ = fnv_step(flow_, instruction_rva, opcode, length);
		if (cpu_enabled_ && gprs)
		{
			cpu_raw_ = cpu_raw_update(cpu_raw_, rflags, gprs);
			cpu_sym_ = cpu_sym_update(cpu_sym_, rflags, gprs);
		}
		++steps_;
		last_was_boundary_ = is_boundary_instruction(opcode, length);
		if (last_was_boundary_)
			++blocks_;
	}

	void finish()
	{
		if (!active_ || !started_)
		{
			finish_stream();
			return;
		}
		write_record(false, 0, 0, nullptr);
		started_ = false;
		finish_stream();
	}

private:
	static constexpr std::uint64_t kFnvOffset = UINT64_C(0xCBF29CE484222325);
	static constexpr std::uint64_t kFnvPrime = UINT64_C(0x100000001B3);
	static constexpr std::size_t kMaxWriteLog = 256;
	static constexpr std::size_t kMaxCtrlLog = 64;
	static constexpr std::size_t kDefaultCtrlLog = 16;
	static constexpr std::uint8_t kCtrlImage = 0;
	static constexpr std::uint8_t kCtrlStack = 1;
	static constexpr std::uint8_t kCtrlPool = 2;
	static constexpr std::uint8_t kCtrlThunk = 3;
	static constexpr std::uint8_t kCtrlUnread = 4;
	// cpu_sym64 value classes (not the CTRL target classes).
	static constexpr std::uint8_t kCpuScalar = 0;
	static constexpr std::uint8_t kCpuModule = 1;
	static constexpr std::uint8_t kCpuStack = 2;
	static constexpr std::uint8_t kCpuPool = 3;
	static constexpr std::uint64_t kCpuFlagsMask = UINT64_C(0xCD5);
	static constexpr std::size_t kMaxCpuRanges = 4;

	chunk_emitter() { initialize(); }
	~chunk_emitter() { finish(); }
	chunk_emitter(const chunk_emitter&) = delete;
	chunk_emitter& operator=(const chunk_emitter&) = delete;

	static std::uint64_t parse_u64(const char* value, std::uint64_t fallback)
	{
		if (!value || !*value)
			return fallback;
		char* end = nullptr;
		const unsigned long long parsed = std::strtoull(value, &end, 0);
		return end == value ? fallback : static_cast<std::uint64_t>(parsed);
	}

	void initialize()
	{
		const char* path = std::getenv("KERNEMUL_CHUNK_OUT");
		if (!path || !*path)
			return;

		kind_ = "insn";
		if (const char* value = std::getenv("KERNEMUL_CHUNK_KIND"))
			kind_ = value;
		every_ = parse_u64(std::getenv("KERNEMUL_CHUNK_EVERY"), 4096);
		blocks_per_chunk_ = parse_u64(std::getenv("KERNEMUL_CHUNK_BLOCKS"), 16);
		module_ = "target_driver.sys";
		if (const char* value = std::getenv("KERNEMUL_CHUNK_MODULE"))
			module_ = value;
		image_size_ = parse_u64(std::getenv("KERNEMUL_CHUNK_IMAGE_SIZE"), 0);
		dump_steps_ = parse_u64(
			std::getenv("KERNEMUL_CHUNK_DUMP_STEPS"), 0);
		dump_from_ = parse_u64(
			std::getenv("KERNEMUL_CHUNK_DUMP_FROM"), 0);
		dump_regs_ = std::getenv("KERNEMUL_CHUNK_DUMP_REGS") != nullptr;
		dump_rva_ = parse_u64(std::getenv("KERNEMUL_CHUNK_DUMP_RVA"), 0);
		if (dump_rva_)
		{
			if (dump_steps_ == 0)
				dump_steps_ = 1;
			// Arm on the RVA: nothing is dumped until the instruction matches.
			dump_from_ = ~static_cast<std::uint64_t>(0);
		}
		stop_after_dump_ = std::getenv("KERNEMUL_CHUNK_STOP_AFTER_DUMP") != nullptr;
		if (const char* bin = std::getenv("KERNEMUL_CHUNK_DUMP_BIN"))
		{
			dump_bin_ = std::fopen(bin, "wb");
			dump_text_ = false;
		}
		if (std::getenv("KERNEMUL_CHUNK_DUMP_TEXT"))
			dump_text_ = true;
		if (const char* ignore = std::getenv("KERNEMUL_CHUNK_IGNORE_OPCODES"))
		{
			ignore_opcodes_ = ignore;
			std::string current;
			for (const char* cursor = ignore; ; ++cursor)
			{
				if (*cursor == ',' || *cursor == '\0')
				{
					if (!current.empty())
					{
						const auto value = std::strtoul(current.c_str(), nullptr, 16);
						if (value < 256)
							ignore_[value] = true;
					}
					current.clear();
					if (*cursor == '\0')
						break;
				}
				else
				{
					current.push_back(*cursor);
				}
			}
		}
		base_ = parse_u64(std::getenv("KERNEMUL_CHUNK_BASE"), 0);
		writes_enabled_ = parse_u64(std::getenv("KERNEMUL_CHUNK_WRITES"), 0) != 0;
		write_log_cap_ = parse_u64(
			std::getenv("KERNEMUL_CHUNK_WRITE_LOG"), 0);
		if (write_log_cap_ > kMaxWriteLog)
			write_log_cap_ = kMaxWriteLog;
		ctrl_enabled_ = parse_u64(std::getenv("KERNEMUL_CHUNK_CTRL"), 0) != 0;
		ctrl_log_cap_ = parse_u64(
			std::getenv("KERNEMUL_CHUNK_CTRL_LOG"), kDefaultCtrlLog);
		if (ctrl_log_cap_ > kMaxCtrlLog)
			ctrl_log_cap_ = kMaxCtrlLog;
		ctrl_stack_ = parse_range(std::getenv("KERNEMUL_CHUNK_CTRL_STACK"));
		ctrl_thunk_ = parse_range(std::getenv("KERNEMUL_CHUNK_CTRL_THUNK"));
		ctrl_pool_ = parse_range(std::getenv("KERNEMUL_CHUNK_CTRL_POOL"));
		// Per-step CPU digests: on by default.  KERNEMUL_CHUNK_NO_CPU is a
		// debug switch to measure emitter cost, not a comparison mode.
		cpu_enabled_ = parse_u64(std::getenv("KERNEMUL_CHUNK_NO_CPU"), 0) == 0;
		parse_range_list(std::getenv("KERNEMUL_CHUNK_STACK"), cpu_stack_);
		parse_range_list(std::getenv("KERNEMUL_CHUNK_POOL"), cpu_pool_);
		// Ordered classification rules; see load_cpu_rules().
		parse_cpu_rules(std::getenv("KERNEMUL_CHUNK_REGION_RULES"));
		for (const auto& range : cpu_stack_)
			cpu_rule_add(kCpuStack, 1, range.base, range.size);
		for (const auto& range : cpu_pool_)
			cpu_rule_add(kCpuPool, 1, range.base, range.size);
		run_cpu_probe(std::getenv("KERNEMUL_CHUNK_REGION_PROBE"));
		if (const char* watch = std::getenv("KERNEMUL_CHUNK_WATCH"))
			parse_watch(watch);

		if (kind_ != "insn" && kind_ != "block")
			kind_ = "insn";
		if (every_ == 0)
			every_ = 4096;
		if (blocks_per_chunk_ == 0)
			blocks_per_chunk_ = 16;

		stream_.open(path, std::ios::binary | std::ios::trunc);
		active_ = stream_.is_open();
	}

	void finish_stream()
	{
		if (stream_.is_open())
			stream_.flush();
		if (dump_bin_)
		{
			std::fflush(dump_bin_);
			std::fclose(dump_bin_);
			dump_bin_ = nullptr;
		}
		active_ = false;
	}

	// A watch address is accepted as a module RVA, as a guest VA of this
	// image, or as the LWK-form absolute RIP (0x140000000 + RVA, the same
	// convention KERNEMUL_TRACE_IGNORE_PAIRS uses).  The module base is only
	// known when the driver is mapped, so the spec keeps the raw value and the
	// address is resolved at each boundary.
	std::uint64_t watch_address(std::uint64_t value) const
	{
		constexpr std::uint64_t lwk_base = UINT64_C(0x140000000);
		if (value >= lwk_base && value - lwk_base < UINT64_C(0x100000000))
			return base_ + (value - lwk_base);
		if (base_ && value < base_)
			return base_ + value;
		return value;
	}

	void parse_watch(const char* text)
	{
		const std::string list(text);
		std::size_t pos = 0;
		while (pos <= list.size() && watch_.size() < 16)
		{
			const auto comma = list.find(',', pos);
			const std::string item = list.substr(pos,
				comma == std::string::npos ? std::string::npos : comma - pos);
			const auto first = item.find(':');
			const auto second = (first == std::string::npos)
				? std::string::npos : item.find(':', first + 1);
			if (first != std::string::npos && second != std::string::npos)
			{
				watch_region region;
				region.name = item.substr(0, first);
				region.address = std::strtoull(
					item.substr(first + 1, second - first - 1).c_str(),
					nullptr, 0);
				region.size = static_cast<std::uint32_t>(
					std::strtoul(item.substr(second + 1).c_str(), nullptr, 0));
				if (region.size && region.size <= 256)
					watch_.push_back(region);
			}
			if (comma == std::string::npos)
				break;
			pos = comma + 1;
		}
	}

	// -- ordered writes (CV4 H_WRITES) ------------------------------------
	struct store
	{
		std::uint64_t ea = 0;
		std::uint64_t value = 0;
		std::uint8_t width = 0;
	};

	struct write_entry
	{
		std::uint64_t insn_seq = 0;
		std::uint64_t ea = 0;
		std::uint64_t value = 0;
		std::uint8_t width = 0;
	};

	static bool read_signed(const std::uint8_t* data, std::size_t length,
		std::size_t& index, std::size_t size, std::int64_t& out)
	{
		if (index + size > length)
			return false;
		std::uint64_t value = 0;
		for (std::size_t i = 0; i < size; ++i)
			value |= static_cast<std::uint64_t>(data[index + i]) << (8 * i);
		if (size < 8 && (value & (UINT64_C(1) << (size * 8 - 1))))
			value |= ~UINT64_C(0) << (size * 8);
		out = static_cast<std::int64_t>(value);
		index += size;
		return true;
	}

	static int read_modrm_ea(const std::uint8_t* data, std::size_t length,
		std::size_t& index, const std::uint64_t* gprs, int rex_b, int rex_x,
		bool addr32, std::uint64_t rip, std::size_t imm_size,
		std::uint64_t& ea)
	{
		if (index >= length)
			return -1;
		const std::uint8_t modrm = data[index++];
		const unsigned mod = modrm >> 6;
		const unsigned rm = modrm & 7;
		if (mod == 3)
			return 0;
		const std::uint64_t mask = addr32 ? UINT64_C(0xFFFFFFFF) : ~UINT64_C(0);
		std::uint64_t address = 0;
		bool has_base = false, has_index = false, rip_relative = false;
		std::uint64_t base_value = 0, index_value = 0;
		unsigned scale = 1;
		if (rm == 4)
		{
			if (index >= length)
				return -1;
			const std::uint8_t sib = data[index++];
			scale = 1u << (sib >> 6);
			const unsigned index_low = (sib >> 3) & 7;
			if (index_low != 4 || rex_x)
			{
				has_index = true;
				index_value = gprs[(index_low + static_cast<unsigned>(rex_x)) & 15] & mask;
			}
			const unsigned base_low = sib & 7;
			if (mod == 0 && base_low == 5)
			{
				std::int64_t displacement = 0;
				if (!read_signed(data, length, index, 4, displacement))
					return -1;
				address += static_cast<std::uint64_t>(displacement);
			}
			else
			{
				has_base = true;
				base_value = gprs[(base_low + static_cast<unsigned>(rex_b)) & 15] & mask;
			}
		}
		else if (mod == 0 && rm == 5)
		{
			std::int64_t displacement = 0;
			if (!read_signed(data, length, index, 4, displacement))
				return -1;
			if (addr32)
				address += static_cast<std::uint64_t>(displacement) & UINT64_C(0xFFFFFFFF);
			else
			{
				rip_relative = true;
				address += static_cast<std::uint64_t>(displacement);
			}
		}
		else
		{
			has_base = true;
			base_value = gprs[(rm + static_cast<unsigned>(rex_b)) & 15] & mask;
		}
		if (mod == 1)
		{
			std::int64_t displacement = 0;
			if (!read_signed(data, length, index, 1, displacement))
				return -1;
			address += static_cast<std::uint64_t>(displacement);
		}
		else if (mod == 2)
		{
			std::int64_t displacement = 0;
			if (!read_signed(data, length, index, 4, displacement))
				return -1;
			address += static_cast<std::uint64_t>(displacement);
		}
		if (has_base)
			address += base_value;
		if (has_index)
			address += index_value * scale;
		if (rip_relative)
			address += rip + index + imm_size;
		ea = address & mask;
		return 1;
	}

	static int decode_store(const std::uint8_t* opcode, std::size_t length,
		std::uint64_t rip, const std::uint64_t* gprs, store& out)
	{
		if (!opcode || !gprs || !length)
			return 0;
		std::size_t index = 0;
		unsigned rex = 0;
		bool operand16 = false, addr32 = false;
		while (index < length)
		{
			const auto byte = opcode[index];
			if (byte == 0x2E || byte == 0x36 || byte == 0x3E || byte == 0x26 ||
				byte == 0x64 || byte == 0x65 || byte == 0x66 || byte == 0x67 ||
				byte == 0xF0 || byte == 0xF2 || byte == 0xF3)
			{
				if (byte == 0x66)
					operand16 = true;
				if (byte == 0x67)
					addr32 = true;
				++index;
				continue;
			}
			if (byte >= 0x40 && byte <= 0x4F)
			{
				rex = byte;
				++index;
				continue;
			}
			break;
		}
		if (index >= length)
			return 0;
		const std::uint8_t op = opcode[index++];
		const bool rex_w = (rex & 0x08) != 0;
		const int rex_r = (rex & 0x04) ? 8 : 0;
		const int rex_x = (rex & 0x02) ? 8 : 0;
		const int rex_b = (rex & 0x01) ? 8 : 0;
		if (op >= 0x50 && op <= 0x57)
		{
			const unsigned bits = operand16 ? 2 : 8;
			const unsigned reg = (op - 0x50) + static_cast<unsigned>(rex_b);
			out.width = static_cast<std::uint8_t>(bits);
			out.value = gprs[reg & 15] &
				(bits == 2 ? UINT64_C(0xFFFF) : ~UINT64_C(0));
			out.ea = gprs[4] - bits;
			return 1;
		}
		if (op == 0xE8)
		{
			if (operand16)
				return -1;
			out.width = 8;
			out.ea = gprs[4] - 8;
			out.value = rip + index + 4;
			return 1;
		}
		if (op != 0x88 && op != 0x89 && op != 0xC7)
			return 0;
		if (index >= length)
			return -1;
		const std::uint8_t modrm = opcode[index];
		const unsigned group = (modrm >> 3) & 7;
		if (op == 0xC7 && group != 0)
			return 0;
		const unsigned bits = op == 0x88 ? 1u
			: (rex_w ? 8u : (operand16 ? 2u : 4u));
		const std::size_t imm_size = op == 0xC7 ? (operand16 ? 2u : 4u) : 0;
		std::uint64_t ea = 0;
		const int decoded = read_modrm_ea(opcode, length, index, gprs, rex_b,
			rex_x, addr32, rip, imm_size, ea);
		if (decoded <= 0)
			return decoded;
		out.ea = ea;
		out.width = static_cast<std::uint8_t>(bits);
		if (op == 0xC7)
		{
			std::int64_t immediate = 0;
			if (!read_signed(opcode, length, index, imm_size, immediate))
				return -1;
			const std::uint64_t mask = bits == 8 ? ~UINT64_C(0)
				: ((UINT64_C(1) << (bits * 8)) - 1);
			out.value = static_cast<std::uint64_t>(immediate) & mask;
			return 1;
		}
		const unsigned source = (group + static_cast<unsigned>(rex_r)) & 15;
		// bits == 8 must not shift by 64 (undefined behaviour).
		const std::uint64_t mask = bits == 8 ? ~UINT64_C(0)
			: ((UINT64_C(1) << (bits * 8)) - 1);
		out.value = gprs[source] & mask;
		return 1;
	}

	std::uint64_t write_ea_token(std::uint64_t ea) const
	{
		if (base_ && image_size_)
		{
			const std::uint64_t rva = ea - base_;
			if (rva < image_size_)
				return rva;
		}
		return ea;
	}

	// rep movs/stos write memory without a ModRM store, so decode_store cannot
	// see them and the write log would silently miss them.
	static bool decode_block_copy(const std::uint8_t* opcode,
		std::size_t length, const std::uint64_t* gprs, std::uint64_t& ea,
		std::uint64_t& bytes)
	{
		if (!opcode || !gprs || !length)
			return false;
		std::size_t index = 0;
		bool rep = false, rex_w = false;
		while (index < length)
		{
			const auto byte = opcode[index];
			if (byte == 0xF3)
			{
				rep = true;
				++index;
			}
			else if (byte == 0x66 || byte == 0x67 || byte == 0x2E ||
				byte == 0x36 || byte == 0x3E || byte == 0x26 ||
				byte == 0x64 || byte == 0x65)
				++index;
			else if (byte >= 0x40 && byte <= 0x4F)
			{
				if (byte & 8)
					rex_w = true;
				++index;
			}
			else
				break;
		}
		if (!rep || index >= length)
			return false;
		const auto op = opcode[index];
		std::uint64_t size;
		if (op == 0xA4 || op == 0xAA)
			size = 1;
		else if (op == 0xA5 || op == 0xAB)
			size = rex_w ? 8 : 4;
		else
			return false;
		const std::uint64_t count = gprs[1];
		if (!count)
			return false;
		ea = gprs[7];
		bytes = count * size;
		return true;
	}

	// Stores that decode_store does not model: SSE register->memory moves,
	// push [mem] and pop [mem].  The value being written lives in an XMM register
	// or in memory, so it is recorded as opaque (real EA and width, value 0).
	// The LWK emitter does exactly the same, so the digest still compares where,
	// how wide and in what order each unmodelled store went.
	static int decode_other_store(const std::uint8_t* opcode,
		std::size_t length, const std::uint64_t* gprs, std::uint64_t rip,
		store& out)
	{
		if (!opcode || !gprs || !length)
			return 0;
		std::size_t index = 0;
		unsigned rex = 0;
		bool addr32 = false, prefix_f2 = false, prefix_f3 = false;
		while (index < length)
		{
			const auto byte = opcode[index];
			if (byte == 0x2E || byte == 0x36 || byte == 0x3E || byte == 0x26 ||
				byte == 0x64 || byte == 0x65 || byte == 0x66 || byte == 0x67 ||
				byte == 0xF0 || byte == 0xF2 || byte == 0xF3)
			{
				if (byte == 0x67)
					addr32 = true;
				if (byte == 0xF2)
					prefix_f2 = true;
				if (byte == 0xF3)
					prefix_f3 = true;
				++index;
				continue;
			}
			if (byte >= 0x40 && byte <= 0x4F)
			{
				rex = byte;
				++index;
				continue;
			}
			break;
		}
		if (index >= length)
			return 0;
		const auto op = opcode[index++];
		const int rex_b = (rex & 0x01) ? 8 : 0;
		const int rex_x = (rex & 0x02) ? 8 : 0;

		if (op == 0x0F)
		{
			if (index >= length)
				return 0;
			const auto op2 = opcode[index++];
			std::uint8_t width = 0;
			switch (op2)
			{
			case 0x11: width = prefix_f3 ? 4 : (prefix_f2 ? 8 : 16); break;
			case 0x13: case 0x17: case 0xD6: width = 8; break;
			case 0x29: case 0x7F: width = 16; break;
			default: return 0;
			}
			std::uint64_t ea = 0;
			const int decoded = read_modrm_ea(opcode, length, index, gprs,
				rex_b, rex_x, addr32, rip, 0, ea);
			if (decoded <= 0)
				return decoded;
			out.ea = ea;
			out.width = width;
			out.value = 0;
			return 1;
		}

		if (op == 0xFF || op == 0x8F)
		{
			if (index >= length)
				return 0;
			const auto modrm = opcode[index];
			const unsigned group = (modrm >> 3) & 7;
			const unsigned mod = modrm >> 6;
			if (op == 0xFF && group != 6)
				return 0;
			if (op == 0x8F && group != 0)
				return 0;
			if (mod == 3)
			{
				if (op != 0xFF)
					return 0;
				const unsigned reg = (modrm & 7) + static_cast<unsigned>(rex_b);
				out.ea = gprs[4] - 8;
				out.width = 8;
				out.value = gprs[reg & 15];
				return 1;
			}
			std::uint64_t ea = 0;
			const int decoded = read_modrm_ea(opcode, length, index, gprs,
				rex_b, rex_x, addr32, rip, 0, ea);
			if (decoded <= 0)
				return decoded;
			out.ea = (op == 0x8F) ? ea : gprs[4] - 8;
			out.width = 8;
			out.value = 0;
			return 1;
		}
		return 0;
	}

	void record_block_write(std::uint64_t insn_seq, std::uint64_t ea,
		std::uint64_t bytes)
	{
		++writes_block_;
		if (!write_log_cap_)
			return;
		if (write_log_.size() < write_log_cap_)
		{
			write_entry entry;
			entry.insn_seq = insn_seq;
			entry.ea = ea;
			entry.value = bytes;
			entry.width = 0;
			write_log_.push_back(entry);
		}
		else
			write_log_capped_ = true;
	}

	void record_write(std::uint64_t insn_seq, const store& decoded)
	{
		write_hash_ = writes_hash_update(write_hash_,
			write_ea_token(decoded.ea), decoded.value, decoded.width);
		write_sym_ = writes_sym_update(write_sym_, decoded.ea, decoded.value,
			decoded.width);
		++writes_count_;
		if (!write_log_cap_)
			return;
		if (write_log_.size() < write_log_cap_)
		{
			write_entry entry;
			entry.insn_seq = insn_seq;
			entry.ea = decoded.ea;
			entry.value = decoded.value;
			entry.width = decoded.width;
			write_log_.push_back(entry);
		}
		else
			write_log_capped_ = true;
	}

	// Same rule table as cpu_sym64, applied to the store EA and value, so a
	// pointer store is not cross-environment noise.
	std::uint64_t writes_sym_update(std::uint64_t hash, std::uint64_t ea,
		std::uint64_t value, std::uint8_t width) const
	{
		std::uint8_t ea_cls = kCpuScalar, val_cls = kCpuScalar;
		std::uint64_t ea_payload = 0, val_payload = 0;
		cpu_classify(ea, ea_cls, ea_payload);
		cpu_classify(value, val_cls, val_payload);
		hash = fnv_u8(hash, ea_cls);
		hash = fnv_u64(hash, ea_payload);
		hash = fnv_u8(hash, width);
		// Narrow stores carry pointer fragments (the low half of an address
		// looks like a scalar and changes with ASLR), so only full 64-bit
		// values participate.  EA and width still discriminate.
		if (width < 8)
			return hash;
		hash = fnv_u8(hash, val_cls);
		return fnv_u64(hash, val_payload);
	}

	static std::uint64_t writes_hash_update(std::uint64_t hash,
		std::uint64_t ea, std::uint64_t value, std::uint8_t width)
	{
		std::uint8_t bytes[17];
		for (unsigned i = 0; i < 8; ++i)
			bytes[i] = static_cast<std::uint8_t>(ea >> (i * 8));
		bytes[8] = width;
		for (unsigned i = 0; i < 8; ++i)
			bytes[9 + i] = static_cast<std::uint8_t>(value >> (i * 8));
		for (const auto byte : bytes)
		{
			hash ^= byte;
			hash *= kFnvPrime;
		}
		return hash;
	}

	// -- indirect control targets (CTRL_TARGETS) ---------------------------
	// Mirror of the LWK decoder (src/trace/lwk_chunk.c) and of
	// lwk-tools/chunk_domains.py:decode_ctrl_target_status.
	struct ctrl_entry
	{
		std::uint64_t ordinal = 0;
		std::uint64_t rip_rva = 0;
		std::uint64_t target = 0;
		std::uint8_t target_class = kCtrlUnread;
	};

	struct ctrl_range
	{
		std::uint64_t base = 0;
		std::uint64_t size = 0;
	};

	static ctrl_range parse_range(const char* text)
	{
		ctrl_range range;
		if (!text || !*text)
			return range;
		const char* colon = std::strchr(text, ':');
		if (!colon)
			return range;
		range.base = std::strtoull(text, nullptr, 0);
		range.size = std::strtoull(colon + 1, nullptr, 0);
		return range;
	}

	static bool range_contains(const ctrl_range& range, std::uint64_t value)
	{
		return range.size != 0 && value >= range.base &&
			value - range.base < range.size;
	}

	// ``<base>:<size>[,<base>:<size>...]`` (``;`` also accepted), capped.
	static void parse_range_list(const char* text,
		std::vector<ctrl_range>& out)
	{
		if (!text || !*text)
			return;
		const std::string list(text);
		std::size_t pos = 0;
		while (pos <= list.size() && out.size() < kMaxCpuRanges)
		{
			const auto end = list.find_first_of(",;", pos);
			const std::string item = list.substr(pos,
				end == std::string::npos ? std::string::npos : end - pos);
			const ctrl_range range = parse_range(item.c_str());
			if (range.size)
				out.push_back(range);
			if (end == std::string::npos)
				break;
			pos = end + 1;
		}
	}

	static bool range_list_contains(const std::vector<ctrl_range>& ranges,
		std::uint64_t value)
	{
		for (const auto& range : ranges)
			if (range_contains(range, value))
				return true;
		return false;
	}

	std::uint8_t classify_ctrl_target(std::uint64_t target) const
	{
		if (base_ && image_size_ && target - base_ < image_size_)
			return kCtrlImage;
		// Most specific first; an unset range never matches.
		if (range_contains(ctrl_thunk_, target))
			return kCtrlThunk;
		if (range_contains(ctrl_stack_, target))
			return kCtrlStack;
		if (range_contains(ctrl_pool_, target))
			return kCtrlPool;
		return kCtrlPool;
	}

	bool read_guest_qword(std::uint64_t address, std::uint64_t& out) const
	{
		if (!memory_reader_)
			return false;
		std::uint8_t buffer[8] = {};
		if (!memory_reader_(address, buffer, sizeof(buffer)))
			return false;
		std::uint64_t value = 0;
		for (unsigned i = 0; i < 8; ++i)
			value |= static_cast<std::uint64_t>(buffer[i]) << (8 * i);
		out = value;
		return true;
	}

	// Returns 1 when a transfer was decoded (class 4 = the target lives in
	// memory and could not be read), 0 when the instruction is not an indirect
	// transfer, and -1 for an indirect form that is not modelled (retf/iret/
	// far call or jmp), counted in ct_skipped.
	int decode_ctrl(const std::uint8_t* opcode, std::size_t length,
		std::uint64_t rip, const std::uint64_t* gprs, ctrl_entry& out) const
	{
		if (!opcode || !gprs || !length)
			return 0;
		std::size_t index = 0;
		unsigned rex = 0;
		bool addr32 = false;
		while (index < length)
		{
			const auto byte = opcode[index];
			if (byte == 0x2E || byte == 0x36 || byte == 0x3E || byte == 0x26 ||
				byte == 0x64 || byte == 0x65 || byte == 0x66 || byte == 0x67 ||
				byte == 0xF0 || byte == 0xF2 || byte == 0xF3)
			{
				if (byte == 0x67)
					addr32 = true;
				++index;
				continue;
			}
			if (byte >= 0x40 && byte <= 0x4F)
			{
				rex = byte;
				++index;
				continue;
			}
			break;
		}
		if (index >= length)
			return 0;
		const std::uint8_t op = opcode[index++];
		const int rex_b = (rex & 0x01) ? 8 : 0;
		const int rex_x = (rex & 0x02) ? 8 : 0;

		out.target = 0;
		out.target_class = kCtrlUnread;

		auto take_memory = [&](std::uint64_t address) -> int
		{
			if (!read_guest_qword(address, out.target))
				return 1; // unread: recorded, never guessed
			out.target_class = classify_ctrl_target(out.target);
			return 1;
		};

		if (op == 0xC3 || op == 0xC2)
		{
			if (op == 0xC2 && index + 2 > length)
				return -1;
			return take_memory(gprs[4]);
		}
		if (op == 0xCA || op == 0xCB || op == 0xCF)
			return -1; // retf / iret
		if (op != 0xFF)
			return 0;
		if (index >= length)
			return -1;
		const std::uint8_t modrm = opcode[index];
		const unsigned group = (modrm >> 3) & 7;
		if (group == 3 || group == 5)
			return -1; // far call/jmp m16:64
		if (group != 2 && group != 4)
			return 0;
		if ((modrm >> 6) == 3)
		{
			out.target = gprs[((modrm & 7) + static_cast<unsigned>(rex_b)) & 15];
			out.target_class = classify_ctrl_target(out.target);
			return 1;
		}
		std::uint64_t ea = 0;
		if (read_modrm_ea(opcode, length, index, gprs, rex_b, rex_x, addr32,
			rip, 0, ea) <= 0)
			return -1;
		return take_memory(ea);
	}

	static std::uint64_t ctrl_hash_update(std::uint64_t hash,
		std::uint64_t ordinal, std::uint8_t target_class, std::uint64_t token)
	{
		std::uint8_t bytes[17];
		for (unsigned i = 0; i < 8; ++i)
			bytes[i] = static_cast<std::uint8_t>(ordinal >> (i * 8));
		bytes[8] = target_class;
		for (unsigned i = 0; i < 8; ++i)
			bytes[9 + i] = static_cast<std::uint8_t>(token >> (i * 8));
		for (const auto byte : bytes)
		{
			hash ^= byte;
			hash *= kFnvPrime;
		}
		return hash;
	}

	void record_ctrl(std::uint64_t rip_rva, ctrl_entry decoded)
	{
		decoded.ordinal = ctrl_count_;
		decoded.rip_rva = rip_rva;
		std::uint64_t token = decoded.target;
		if (decoded.target_class == kCtrlImage)
			token = decoded.target - base_;
		ctrl_hash_ = ctrl_hash_update(ctrl_hash_, decoded.ordinal,
			decoded.target_class, token);
		++ctrl_count_;
		if (decoded.target_class == kCtrlUnread)
			++ctrl_skipped_;
		if (!ctrl_log_cap_)
			return;
		if (ctrl_log_.size() < ctrl_log_cap_)
			ctrl_log_.push_back(decoded);
		else
			ctrl_log_capped_ = true;
	}

	std::uint64_t rva(std::uint64_t rip) const
	{
		return base_ ? rip - base_ : rip;
	}

	// -- per-step CPU digests (CPU_RAW / CPU_SYMBOLIC) ----------------------
	// Mirror of src/trace/lwk_chunk.c: both emitters must hash the same bytes
	// for equivalent state, so the two sides are directly comparable.
	//
	// One ordered rule per classified region: the first match wins and values
	// that match nothing are SCALAR.  Rules are data (KERNEMUL_CHUNK_REGION_RULES
	// "kind:base:size:mode;..."), so a new edge case is one more rule rather
	// than a change to the classifier.
	struct cpu_rule
	{
		std::uint8_t kind = 0;
		std::uint8_t payload_mode = 1;
		std::uint64_t base = 0;
		std::uint64_t size = 0;
	};

	void cpu_rule_add(std::uint8_t kind, std::uint8_t payload_mode,
		std::uint64_t base, std::uint64_t size)
	{
		if (!size || cpu_rules_.size() >= kMaxCpuRules)
			return;
		cpu_rules_.push_back(cpu_rule{ kind, payload_mode, base, size });
	}

	void parse_cpu_rules(const char* text)
	{
		if (!text || !*text)
			return;
		std::string list(text);
		std::size_t start = 0;
		while (start <= list.size())
		{
			const auto end = list.find(';', start);
			const std::string item = list.substr(start,
				end == std::string::npos ? std::string::npos : end - start);
			start = end == std::string::npos ? list.size() + 1 : end + 1;
			if (item.empty())
				continue;
			unsigned kind = 0, mode = 1;
			unsigned long long base = 0, size = 0;
			if (std::sscanf(item.c_str(), "%u:%llx:%llx:%u", &kind, &base,
					&size, &mode) < 3)
				continue;
			cpu_rule_add(static_cast<std::uint8_t>(kind),
				static_cast<std::uint8_t>(mode ? 1 : 0),
				static_cast<std::uint64_t>(base),
				static_cast<std::uint64_t>(size));
		}
	}

	void run_cpu_probe(const char* text) const
	{
		if (!text || !*text)
			return;
		const std::string list(text);
		std::size_t start = 0;
		while (start <= list.size())
		{
			const auto end = list.find(',', start);
			const std::string item = list.substr(start,
				end == std::string::npos ? std::string::npos : end - start);
			start = end == std::string::npos ? list.size() + 1 : end + 1;
			if (item.empty())
				continue;
			char* stop = nullptr;
			const auto value = std::strtoull(item.c_str(), &stop, 0);
			if (stop == item.c_str())
				continue;
			std::uint8_t cls = kCpuScalar;
			std::uint64_t payload = 0;
			cpu_classify(value, cls, payload);
			std::fprintf(stderr,
				"[CPU_CLASS] value=0x%016llX kind=%u payload=0x%016llX\n",
				static_cast<unsigned long long>(value), static_cast<unsigned>(cls),
				static_cast<unsigned long long>(payload));
		}
	}

	void cpu_classify(std::uint64_t value, std::uint8_t& cls,
		std::uint64_t& payload) const
	{
		if (base_ && image_size_ && value - base_ < image_size_)
		{
			cls = kCpuModule;
			payload = value - base_;
			return;
		}
		for (const auto& rule : cpu_rules_)
		{
			if (value - rule.base < rule.size)
			{
				cls = rule.kind;
				payload = rule.payload_mode ? 0 : value - rule.base;
				return;
			}
		}
		cls = kCpuScalar;
		payload = value;
	}

	static std::uint64_t cpu_raw_update(std::uint64_t hash,
		std::uint64_t rflags, const std::uint64_t* gprs)
	{
		for (unsigned index = 0; index < 16; ++index)
			hash = fnv_u64(hash, gprs[index]);
		return fnv_u64(hash, rflags & kCpuFlagsMask);
	}

	std::uint64_t cpu_sym_update(std::uint64_t hash, std::uint64_t rflags,
		const std::uint64_t* gprs) const
	{
		for (unsigned index = 0; index < 16; ++index)
		{
			std::uint8_t cls = kCpuScalar;
			std::uint64_t payload = 0;
			cpu_classify(gprs[index], cls, payload);
			hash = fnv_u8(hash, cls);
			hash = fnv_u64(hash, payload);
		}
		hash = fnv_u8(hash, kCpuScalar);
		return fnv_u64(hash, rflags & kCpuFlagsMask);
	}

	static std::uint64_t fnv_u64(std::uint64_t hash, std::uint64_t value)
	{
		std::uint8_t bytes[8];
		for (unsigned index = 0; index < 8; ++index)
			bytes[index] = static_cast<std::uint8_t>(value >> (index * 8));
		for (const auto byte : bytes)
		{
			hash ^= byte;
			hash *= kFnvPrime;
		}
		return hash;
	}

	static std::uint64_t fnv_u8(std::uint64_t hash, std::uint8_t value)
	{
		hash ^= value;
		return hash * kFnvPrime;
	}

	// Region pairs mirror the LWK TF controls.  A value is accepted either as a
	// module RVA (0x2D3A8FB) or as an LWK absolute RIP (0x142D3A8FB).
	static std::uint64_t pair_rva(const std::string& text, std::uint64_t base)
	{
		const std::uint64_t value = std::strtoull(text.c_str(), nullptr, 0);
		constexpr std::uint64_t lwk_base = UINT64_C(0x140000000);
		if (value >= lwk_base && value - lwk_base < UINT64_C(0x100000000))
			return value - lwk_base;
		if (base && value >= base)
			return value - base;
		return value;
	}

	static void parse_region_pairs(const char* text, std::uint64_t base,
		std::vector<std::pair<std::uint64_t, std::uint64_t>>& out)
	{
		if (!text || !*text)
			return;
		const std::string list(text);
		std::size_t pos = 0;
		while (pos <= list.size())
		{
			const auto end = list.find_first_of(",;", pos);
			const std::string item = list.substr(pos,
				end == std::string::npos ? std::string::npos : end - pos);
			const auto colon = item.find(':');
			if (colon != std::string::npos)
			{
				const auto start = pair_rva(item.substr(0, colon), base);
				const auto exit_rva = pair_rva(item.substr(colon + 1), base);
				if (start && exit_rva)
					out.emplace_back(start, exit_rva);
			}
			if (end == std::string::npos)
				break;
			pos = end + 1;
		}
	}

	void initialize_pairs()
	{
		pairs_ready_ = true;
		if (const char* value = std::getenv("KERNEMUL_TRACE_IGNORE_PAIRS"))
		{
			ignore_pairs_text_ = value;
			parse_region_pairs(value, base_, ignore_pairs_);
		}
		if (const char* value = std::getenv("KERNEMUL_TRACE_ALLOW_PAIRS"))
		{
			allow_pairs_text_ = value;
			parse_region_pairs(value, base_, allow_pairs_);
		}
		// Unset: bounded default.  Explicit 0: no cap (a long callee between the
		// start and the exit is normal in this packer).
		if (const char* pair_max = std::getenv("KERNEMUL_TRACE_PAIR_MAX"))
			pair_max_ = parse_u64(pair_max, pair_max_);
		if (!ignore_pairs_.empty() || !allow_pairs_.empty())
			std::fprintf(stderr,
				"[KCHUNK_PAIRS] ignore=%zu allow=%zu max=%llu base=0x%llX\n",
				ignore_pairs_.size(), allow_pairs_.size(),
				static_cast<unsigned long long>(pair_max_),
				static_cast<unsigned long long>(base_));
	}

	bool region_allows(std::uint64_t rva)
	{
		if (ignore_pairs_.empty() && allow_pairs_.empty())
			return true;
		// Ignore regions can nest: a frame loop starts inside the F4 callee.
		// A stack keeps the outer exit armed when an inner region pops.
		if (!ignore_stack_.empty())
		{
			const int top = ignore_stack_.back();
			if (rva == ignore_pairs_[top].second)
			{
				ignore_stack_.pop_back();
				pair_steps_ = 0;
				return false;
			}
			// A start inside the active region is ignored: the exit that is
			// already armed is the one that ends it.
			if (pair_max_ && ++pair_steps_ > pair_max_)
			{
				std::fprintf(stderr,
					"[KCHUNK_PAIRS] ignore region 0x%llX:0x%llX did not reach its exit after %llu steps; resuming\n",
					static_cast<unsigned long long>(ignore_pairs_[top].first),
					static_cast<unsigned long long>(ignore_pairs_[top].second),
					static_cast<unsigned long long>(pair_steps_));
				ignore_stack_.pop_back();
				pair_steps_ = 0;
			}
			return false;
		}
		pair_steps_ = 0;
		bool enabled = allow_pairs_.empty();
		if (!allow_pairs_.empty())
		{
			if (allow_active_ >= 0)
			{
				if (rva == allow_pairs_[allow_active_].second)
				{
					allow_active_ = -1;
					return false;
				}
				enabled = true;
			}
			else
			{
				for (std::size_t i = 0; i < allow_pairs_.size(); ++i)
				{
					if (allow_pairs_[i].first != rva)
						continue;
					allow_active_ = static_cast<int>(i);
					enabled = true;
					break;
				}
			}
		}
		for (std::size_t i = 0; i < ignore_pairs_.size(); ++i)
		{
			if (ignore_pairs_[i].first == rva)
			{
				ignore_stack_.push_back(static_cast<int>(i));
				pair_steps_ = 0;
				return false;
			}
		}
		return enabled;
	}

	bool boundary_reached() const
	{
		if (kind_ == "block")
			return blocks_ >= blocks_per_chunk_ && last_was_boundary_;
		return steps_ >= every_;
	}

	void start_chunk()
	{
		started_ = false;
		steps_ = 0;
		blocks_ = 0;
		last_was_boundary_ = false;
		write_hash_ = kFnvOffset;
		write_sym_ = kFnvOffset;
		writes_count_ = 0;
		writes_skipped_ = 0;
		writes_block_ = 0;
		write_log_.clear();
		write_log_capped_ = false;
		ctrl_hash_ = kFnvOffset;
		ctrl_count_ = 0;
		ctrl_skipped_ = 0;
		ctrl_log_.clear();
		ctrl_log_capped_ = false;
		cpu_raw_ = kFnvOffset;
		cpu_sym_ = kFnvOffset;
	}

	static bool is_rep_string_opcode(const std::uint8_t* opcode,
		std::size_t length)
	{
		if (!opcode || !length)
			return false;
		std::size_t index = 0;
		bool rep = false;
		while (index < length)
		{
			const auto byte = opcode[index];
			if (byte == 0xF2 || byte == 0xF3)
			{
				rep = true;
				++index;
			}
			else if (byte == 0x2E || byte == 0x36 || byte == 0x3E ||
				byte == 0x26 || byte == 0x64 || byte == 0x65 ||
				byte == 0x66 || byte == 0x67 ||
				(byte >= 0x40 && byte <= 0x4F))
			{
				++index;
			}
			else
			{
				break;
			}
		}
		if (!rep || index >= length)
			return false;
		const auto first = opcode[index];
		return first == 0x6C || first == 0x6D || first == 0x6E ||
			first == 0x6F || (first >= 0xA4 && first <= 0xA7) ||
			(first >= 0xAA && first <= 0xAF);
	}

	bool opcode_ignored(const std::uint8_t* opcode, std::size_t length) const
	{
		if (!opcode || !length)
			return false;
		std::size_t index = 0;
		while (index < length)
		{
			const auto byte = opcode[index];
			if (byte == 0x2E || byte == 0x36 || byte == 0x3E || byte == 0x26 ||
				byte == 0x64 || byte == 0x65 || byte == 0x66 || byte == 0x67 ||
				byte == 0xF2 || byte == 0xF3 ||
				(byte >= 0x40 && byte <= 0x4F))
				++index;
			else
				break;
		}
		return index < length && ignore_[opcode[index]];
	}

	static bool is_boundary_instruction(const std::uint8_t* opcode,
		std::size_t length)
	{
		if (!opcode || length == 0)
			return false;
		static const std::uint8_t prefixes[] = {
			0x2E, 0x36, 0x3E, 0x26, 0x64, 0x65, 0x66, 0x67, 0xF2, 0xF3
		};
		std::size_t index = 0;
		while (index < length)
		{
			bool is_prefix = false;
			for (const auto prefix : prefixes)
				is_prefix |= opcode[index] == prefix;
			if (!is_prefix)
				break;
			++index;
		}
		if (index >= length)
			return false;
		const std::uint8_t first = opcode[index];
		if (first == 0xE8 || first == 0xE9 || first == 0xEB ||
			first == 0xC2 || first == 0xC3 || first == 0xCA ||
			first == 0xCB || first == 0xCC || first == 0xCD ||
			first == 0xCE || first == 0xCF)
			return true;
		if (first >= 0x70 && first <= 0x7F)
			return true;
		if (first == 0x0F && index + 1 < length)
		{
			const std::uint8_t second = opcode[index + 1];
			if (second >= 0x80 && second <= 0x8F)
				return true;
			if (second == 0x05 || second == 0x07 ||
				second == 0x34 || second == 0x35)
				return true;
		}
		if (first == 0xFF && index + 1 < length)
		{
			const auto group = (opcode[index + 1] >> 3) & 7;
			return group >= 2 && group <= 5;
		}
		if (index > 0)
		{
			// REP-prefixed string operations can retire for a long time.
			if (first == 0x6C || first == 0x6D || first == 0x6E ||
				first == 0x6F || (first >= 0xA4 && first <= 0xA7) ||
				(first >= 0xAA && first <= 0xAF))
				return true;
		}
		return false;
	}

	static std::uint64_t fnv_step(std::uint64_t hash, std::uint64_t rva,
		const std::uint8_t* opcode, std::size_t length)
	{
		for (unsigned index = 0; index < 8; ++index)
		{
			hash ^= (rva >> (index * 8)) & 0xFF;
			hash *= kFnvPrime;
		}
		hash ^= length & 0xFF;
		hash *= kFnvPrime;
		for (std::size_t index = 0; index < length; ++index)
		{
			hash ^= opcode[index];
			hash *= kFnvPrime;
		}
		return hash;
	}

	static void write_hex(std::ofstream& stream, std::uint64_t value)
	{
		char buffer[32];
		std::snprintf(buffer, sizeof(buffer), "0x%016llX",
			static_cast<unsigned long long>(value));
		stream << '"' << buffer << '"';
	}

	void write_header()
	{
		header_written_ = true;
		stream_ << "{\"schema\":\"kchunk-v1\",\"chunk_kind\":\""
			<< kind_ << "\",\"chunk_size\":" << every_
			<< ",\"blocks_per_chunk\":" << blocks_per_chunk_
			<< ",\"base\":";
		write_hex(stream_, base_);
		stream_ << ",\"image_size\":";
		write_hex(stream_, image_size_);
		stream_ << ",\"module\":\"" << module_
			<< "\",\"ignore_opcodes\":\"" << ignore_opcodes_
			<< "\",\"ignore_pairs\":\"" << ignore_pairs_text_
			<< "\",\"allow_pairs\":\"" << allow_pairs_text_
			<< "\",\"flow_hash\":\"fnv1a64\"";
		if (cpu_enabled_)
			stream_ << ",\"cpu\":\"fnv1a64\"";
		stream_ << ",\"watch\":[";
		for (std::size_t i = 0; i < watch_.size(); ++i)
		{
			if (i)
				stream_ << ',';
			stream_ << '"' << watch_[i].name << ':' << watch_[i].size << '"';
		}
		stream_ << ']';
		if (writes_enabled_)
			stream_ << ",\"writes\":\"fnv1a64\",\"write_log\":"
				<< write_log_cap_;
		if (ctrl_enabled_)
			stream_ << ",\"ctrl\":\"fnv1a64\",\"ctrl_log\":" << ctrl_log_cap_;
		stream_ << "}\n";
	}

	void write_record(bool final, std::uint64_t rip, std::uint64_t rflags,
		const std::uint64_t* gprs)
	{
		stream_ << "{\"index\":" << index_++
			<< ",\"first_seq\":" << first_seq_
			<< ",\"last_seq\":" << last_seq_
			<< ",\"first_rva\":";
		write_hex(stream_, first_rva_);
		stream_ << ",\"last_rva\":";
		write_hex(stream_, last_rva_);
		stream_ << ",\"n_steps\":" << steps_
			<< ",\"n_blocks\":" << blocks_
			<< ",\"flow64\":";
		write_hex(stream_, flow_);
		if (cpu_enabled_)
		{
			stream_ << ",\"cpu_raw64\":";
			write_hex(stream_, cpu_raw_);
			stream_ << ",\"cpu_sym64\":";
			write_hex(stream_, cpu_sym_);
		}
		stream_ << ",\"final\":" << (final ? "true" : "false");
		if (final)
		{
			stream_ << ",\"rip\":";
			write_hex(stream_, rva(rip));
			stream_ << ",\"rflags\":" << (rflags & UINT64_C(0xCD5));
			stream_ << ",\"regs\":[";
			for (unsigned index = 0; index < 16; ++index)
			{
				if (index)
					stream_ << ',';
				write_hex(stream_, gprs[index]);
			}
			stream_ << ']';
		}
		if (final && !watch_.empty() && memory_reader_)
		{
			stream_ << ",\"watch\":{";
			bool first = true;
			std::vector<std::uint8_t> buffer;
			for (const auto& region : watch_)
			{
				buffer.assign(region.size, 0);
				if (!memory_reader_(watch_address(region.address),
					buffer.data(), buffer.size()))
					continue;
				if (!first)
					stream_ << ',';
				first = false;
				stream_ << '"' << region.name << "\":\"";
				static const char digits[] = "0123456789ABCDEF";
				for (const auto byte : buffer)
				{
					stream_ << digits[byte >> 4] << digits[byte & 0xF];
				}
				stream_ << '"';
			}
			stream_ << '}';
		}
		if (writes_enabled_)
		{
			stream_ << ",\"writes64\":";
			write_hex(stream_, write_hash_);
			stream_ << ",\"writes_sym64\":";
			write_hex(stream_, write_sym_);
			stream_ << ",\"writes_block\":" << writes_block_
				<< ",\"writes_skipped\":" << writes_skipped_;
			if (write_log_cap_ && !write_log_.empty())
			{
				stream_ << ",\"wlog\":[";
				for (std::size_t i = 0; i < write_log_.size(); ++i)
				{
					char line[96];
					std::snprintf(line, sizeof(line),
						"%llu:0x%016llX:%u:0x%016llX",
						static_cast<unsigned long long>(write_log_[i].insn_seq),
						static_cast<unsigned long long>(write_log_[i].ea),
						static_cast<unsigned>(write_log_[i].width),
						static_cast<unsigned long long>(write_log_[i].value));
					if (i)
						stream_ << ',';
					stream_ << '"' << line << '"';
				}
				stream_ << ']';
				if (write_log_capped_)
					stream_ << ",\"wlog_capped\":true";
			}
		}
		if (ctrl_enabled_)
		{
			stream_ << ",\"ctargets64\":";
			write_hex(stream_, ctrl_hash_);
			stream_ << ",\"ct_skipped\":" << ctrl_skipped_;
			if (ctrl_log_cap_ && !ctrl_log_.empty())
			{
				stream_ << ",\"ctlog\":[";
				for (std::size_t i = 0; i < ctrl_log_.size(); ++i)
				{
					char line[96];
					std::snprintf(line, sizeof(line),
						"%llu:0x%016llX:%u:0x%016llX",
						static_cast<unsigned long long>(ctrl_log_[i].ordinal),
						static_cast<unsigned long long>(ctrl_log_[i].rip_rva),
						static_cast<unsigned>(ctrl_log_[i].target_class),
						static_cast<unsigned long long>(ctrl_log_[i].target));
					if (i)
						stream_ << ',';
					stream_ << '"' << line << '"';
				}
				stream_ << ']';
				if (ctrl_log_capped_)
					stream_ << ",\"ctlog_capped\":true";
			}
		}
		stream_ << "}\n";
	}

	bool active_ = false;
	bool header_written_ = false;
	bool started_ = false;
	bool last_was_boundary_ = false;
	std::string kind_ = "insn";
	std::string module_ = "target_driver.sys";
	std::uint64_t every_ = 4096;
	std::uint64_t blocks_per_chunk_ = 16;
	std::uint64_t image_size_ = 0;
	std::uint64_t base_ = 0;
	std::uint64_t first_seq_ = 0;
	std::uint64_t last_seq_ = 0;
	std::uint64_t first_rva_ = 0;
	std::uint64_t last_rva_ = 0;
	std::uint64_t flow_ = kFnvOffset;
	std::uint64_t steps_ = 0;
	std::uint64_t blocks_ = 0;
	std::uint64_t index_ = 0;
	std::uint64_t dump_steps_ = 0;
	std::uint64_t dump_from_ = 0;
	std::uint64_t dump_rva_ = 0;
	bool dump_armed_ = false;
	bool dump_regs_ = false;
	bool stop_after_dump_ = false;
	// Compact per-step RVA stream (8 bytes each) for whole-run sequence comparisons.
	FILE* dump_bin_ = nullptr;
	bool dump_text_ = true;
	std::uint64_t dumped_ = 0;
	std::string ignore_opcodes_;
	std::string ignore_pairs_text_;
	std::string allow_pairs_text_;
	std::vector<std::pair<std::uint64_t, std::uint64_t>> ignore_pairs_;
	std::vector<std::pair<std::uint64_t, std::uint64_t>> allow_pairs_;
	std::vector<int> ignore_stack_;
	int allow_active_ = -1;
	bool pairs_ready_ = false;
	std::uint64_t pair_steps_ = 0;
	std::uint64_t pair_max_ = UINT64_C(1) << 23;
	bool ignore_[256] = {};
	std::ofstream stream_;

	struct watch_region
	{
		std::string name;
		std::uint64_t address = 0;
		std::uint32_t size = 0;
	};

	std::vector<watch_region> watch_;
	memory_reader memory_reader_;
	bool writes_enabled_ = false;
	std::size_t write_log_cap_ = 0;
	std::uint64_t write_hash_ = kFnvOffset;
	std::uint64_t write_sym_ = kFnvOffset;
	std::uint64_t writes_count_ = 0;
	std::uint64_t writes_skipped_ = 0;
	std::uint64_t writes_block_ = 0;
	std::vector<write_entry> write_log_;
	bool write_log_capped_ = false;
	bool ctrl_enabled_ = false;
	std::size_t ctrl_log_cap_ = 0;
	std::uint64_t ctrl_hash_ = kFnvOffset;
	std::uint64_t ctrl_count_ = 0;
	std::uint64_t ctrl_skipped_ = 0;
	std::vector<ctrl_entry> ctrl_log_;
	bool ctrl_log_capped_ = false;
	ctrl_range ctrl_stack_;
	ctrl_range ctrl_thunk_;
	ctrl_range ctrl_pool_;
	bool cpu_enabled_ = true;
	std::uint64_t cpu_raw_ = kFnvOffset;
	std::uint64_t cpu_sym_ = kFnvOffset;
	std::vector<ctrl_range> cpu_stack_;
	std::vector<ctrl_range> cpu_pool_;
	std::vector<cpu_rule> cpu_rules_;
	static constexpr std::size_t kMaxCpuRules = 16;
};
