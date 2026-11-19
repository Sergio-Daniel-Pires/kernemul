#pragma once

//
// CV0 determinism profile for Kernemul.
//
// The LWK side lives in include/nt/determinism.h and accepts the same flat-key
// JSON profile.  Without KERNEMUL_DETERMINISM_PROFILE (or KERNEMUL_DETERMINISM=1)
// every getter reports "not active" and the original host clock is used.
//

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <string_view>

namespace win
{

class determinism
{
public:
	static determinism& instance()
	{
		static determinism value;
		return value;
	}

	bool enabled() const
	{
		std::lock_guard lock(mutex_);
		return state_.enabled;
	}

	// Advance one clock quantum and return the virtual NT time.
	std::int64_t system_time_100ns()
	{
		std::lock_guard lock(mutex_);
		if (!state_.enabled)
			return 0;

		advance_unlocked();
		if (!logged_)
		{
			logged_ = true;
			std::fprintf(stderr,
				"[DETERMINISM] first system time=0x%016llX increment=%llu\n",
				static_cast<unsigned long long>(state_.last_time_100ns),
				static_cast<unsigned long long>(state_.clock_increment_100ns));
		}
		return state_.last_time_100ns;
	}

	// Same clock, scaled to the profile QPC frequency (100 ns units by default).
	std::int64_t perf_counter()
	{
		std::lock_guard lock(mutex_);
		if (!state_.enabled)
			return 0;

		advance_unlocked();
		const std::uint64_t elapsed = saturated_mul(
			state_.clock_queries, state_.clock_increment_100ns);
		const std::uint64_t scaled = saturated_mul(
			elapsed, state_.perf_frequency) / 10'000'000ULL;
		return static_cast<std::int64_t>(state_.perf_counter_base + scaled);
	}

	std::int64_t boot_time_100ns() const
	{
		std::lock_guard lock(mutex_);
		return state_.enabled ? state_.boot_time_100ns : 0;
	}

	std::int64_t timezone_bias_100ns() const
	{
		std::lock_guard lock(mutex_);
		return state_.enabled ? state_.timezone_bias_100ns : 0;
	}

	std::uint32_t tick_multiplier() const
	{
		std::lock_guard lock(mutex_);
		return state_.enabled ? state_.tick_multiplier : 0;
	}

	std::uint32_t clock_increment_100ns() const
	{
		std::lock_guard lock(mutex_);
		return state_.enabled
			? static_cast<std::uint32_t>(state_.clock_increment_100ns) : 0;
	}

	std::uint64_t perf_frequency() const
	{
		std::lock_guard lock(mutex_);
		return state_.enabled ? state_.perf_frequency : 0;
	}

	std::uint32_t build_number() const
	{
		std::lock_guard lock(mutex_);
		return state_.enabled ? state_.build_number : 0;
	}

	std::uint32_t processor_count() const
	{
		std::lock_guard lock(mutex_);
		return state_.enabled ? state_.processor_count : 0;
	}

	std::uint32_t security_cookie() const
	{
		std::lock_guard lock(mutex_);
		return state_.enabled ? state_.security_cookie : 0;
	}

	std::uint32_t next_random()
	{
		std::lock_guard lock(mutex_);
		if (!state_.enabled)
			return 0;
		std::uint64_t value = state_.rng_seed +
			(++state_.random_calls) * UINT64_C(0x9E3779B97F4A7C15);
		value ^= value >> 30;
		value *= UINT64_C(0xBF58476D1CE4E5B9);
		value ^= value >> 27;
		value *= UINT64_C(0x94D049BB133111EB);
		value ^= value >> 31;
		return static_cast<std::uint32_t>(value >> 32);
	}

	std::uint64_t mock_log(const char* api, std::uint64_t value)
	{
		std::lock_guard lock(mutex_);
		if (!state_.enabled)
			return 0;
		const std::uint64_t sequence = ++state_.mock_calls;
		const char* logging = std::getenv("KERNEMUL_DET_LOG_MOCKS");
		if (logging && *logging && std::strcmp(logging, "0") != 0)
		{
			std::fprintf(stderr, "[DET_MOCK] seq=%llu api=%s value=0x%016llX\n",
				static_cast<unsigned long long>(sequence), api ? api : "?",
				static_cast<unsigned long long>(value));
		}
		return sequence;
	}

private:
	struct state
	{
		bool enabled = false;
		std::int64_t system_time_base_100ns = INT64_C(0x01DC000000000000);
		std::int64_t boot_time_100ns =
			INT64_C(0x01DC000000000000) - INT64_C(36000000000);
		std::uint32_t tick_multiplier = 0x0FA00000u;
		std::uint64_t perf_frequency = 10'000'000;
		std::uint64_t perf_counter_base = 0;
		std::uint64_t clock_increment_100ns = 10'000;
		std::int64_t timezone_bias_100ns = 0;
		std::uint32_t processor_count = 4;
		std::uint32_t build_number = 19045;
		std::uint32_t security_cookie = 0x12345678u;
		std::uint64_t rng_seed = UINT64_C(0x5EEDC0DE);
		std::uint64_t max_clock_queries = 10'000'000;
		std::uint64_t clock_queries = 0;
		std::uint64_t random_calls = 0;
		std::uint64_t mock_calls = 0;
		std::int64_t last_time_100ns = INT64_C(0x01DC000000000000);
	};

	determinism()
	{
		initialize();
	}

	static std::uint64_t saturated_mul(std::uint64_t left, std::uint64_t right)
	{
		if (left != 0 && right > UINT64_MAX / left)
			return UINT64_MAX;
		return left * right;
	}

	void advance_unlocked()
	{
		if (state_.clock_queries >= state_.max_clock_queries)
			return;
		++state_.clock_queries;
		const std::uint64_t delta = saturated_mul(
			state_.clock_queries, state_.clock_increment_100ns);
		const std::uint64_t room = state_.system_time_base_100ns >= 0
			? static_cast<std::uint64_t>(
				INT64_MAX - state_.system_time_base_100ns)
			: static_cast<std::uint64_t>(INT64_MAX);
		state_.last_time_100ns = delta > room
			? (state_.system_time_base_100ns >= 0 ? INT64_MAX : INT64_MIN)
			: state_.system_time_base_100ns +
				static_cast<std::int64_t>(delta);
	}

	void initialize()
	{
		const char* path = std::getenv("KERNEMUL_DETERMINISM_PROFILE");
		if (path && *path)
		{
			if (load_file(path))
			{
				std::fprintf(stderr,
					"[DETERMINISM] profile=%s enabled=1 base=0x%016llX "
					"increment=%llu\n", path,
					static_cast<unsigned long long>(
						state_.system_time_base_100ns),
					static_cast<unsigned long long>(
						state_.clock_increment_100ns));
			}
			else
			{
				state_ = state{};
				state_.enabled = false;
				std::fprintf(stderr,
					"[DETERMINISM] failed to load profile: %s\n", path);
			}
			return;
		}
		const char* enabled = std::getenv("KERNEMUL_DETERMINISM");
		state_.enabled = enabled && *enabled &&
			std::strcmp(enabled, "0") != 0;
	}

	static bool parse_integer(const char* text, std::int64_t& out)
	{
		if (!text || !*text)
			return false;
		bool negative = false;
		if (*text == '-')
		{
			negative = true;
			++text;
		}
		else if (*text == '+')
		{
			++text;
		}
		char* end = nullptr;
		const unsigned long long value = std::strtoull(text, &end, 0);
		if (end == text)
			return false;
		out = negative ? -static_cast<std::int64_t>(value)
			: static_cast<std::int64_t>(value);
		return true;
	}

	static bool json_integer(const std::string& text, std::string_view key,
		std::int64_t& out)
	{
		const std::string pattern = "\"" + std::string(key) + "\"";
		std::size_t position = text.find(pattern);
		if (position == std::string::npos)
			return false;
		position += pattern.size();
		while (position < text.size() &&
			std::isspace(static_cast<unsigned char>(text[position])))
			++position;
		if (position >= text.size() || text[position] != ':')
			return false;
		++position;
		while (position < text.size() &&
			std::isspace(static_cast<unsigned char>(text[position])))
			++position;
		if (position < text.size() && text[position] == '"')
		{
			const std::size_t end = text.find('"', position + 1);
			if (end == std::string::npos)
				return false;
			const std::string value = text.substr(position + 1,
				end - position - 1);
			return parse_integer(value.c_str(), out);
		}
		return parse_integer(text.c_str() + position, out);
	}

	static void apply_i64(const std::string& text, std::string_view key,
		std::int64_t& field)
	{
		std::int64_t value = 0;
		if (json_integer(text, key, value))
			field = value;
	}

	static void apply_u64(const std::string& text, std::string_view key,
		std::uint64_t& field)
	{
		std::int64_t value = 0;
		if (json_integer(text, key, value))
			field = static_cast<std::uint64_t>(value);
	}

	static void apply_u32(const std::string& text, std::string_view key,
		std::uint32_t& field)
	{
		std::int64_t value = 0;
		if (json_integer(text, key, value))
			field = static_cast<std::uint32_t>(value);
	}

	bool load_file(const char* path)
	{
		std::ifstream stream(path, std::ios::binary);
		if (!stream)
			return false;
		std::string text((std::istreambuf_iterator<char>(stream)),
			std::istreambuf_iterator<char>());
		if (text.empty() || text.size() > 65536)
			return false;

		state_ = state{};
		std::int64_t version = 1;
		if (json_integer(text, "version", version) && version != 1)
			return false;

		apply_i64(text, "system_time_base_100ns",
			state_.system_time_base_100ns);
		state_.boot_time_100ns = state_.system_time_base_100ns -
			INT64_C(36000000000);
		apply_i64(text, "boot_time_100ns", state_.boot_time_100ns);
		apply_u32(text, "tick_multiplier", state_.tick_multiplier);
		apply_u64(text, "perf_frequency", state_.perf_frequency);
		apply_u64(text, "perf_counter_base", state_.perf_counter_base);
		apply_u64(text, "clock_increment_100ns",
			state_.clock_increment_100ns);
		apply_i64(text, "timezone_bias_100ns",
			state_.timezone_bias_100ns);
		apply_u32(text, "processor_count", state_.processor_count);
		apply_u32(text, "build_number", state_.build_number);
		apply_u32(text, "security_cookie", state_.security_cookie);
		apply_u64(text, "rng_seed", state_.rng_seed);
		{
			std::int64_t seed = 0;
			if (!json_integer(text, "rng_seed", seed))
				apply_u64(text, "seed", state_.rng_seed);
		}
		apply_u64(text, "max_clock_queries", state_.max_clock_queries);

		if (state_.tick_multiplier == 0)
			state_.tick_multiplier = 0x0FA00000u;
		if (state_.perf_frequency == 0)
			state_.perf_frequency = 10'000'000;
		if (state_.clock_increment_100ns == 0)
			state_.clock_increment_100ns = 10'000;
		if (state_.max_clock_queries == 0)
			state_.max_clock_queries = 10'000'000;

		state_.last_time_100ns = state_.system_time_base_100ns;
		state_.enabled = true;
		return true;
	}

	mutable std::mutex mutex_;
	state state_{};
	bool logged_ = false;
};

} // namespace win
