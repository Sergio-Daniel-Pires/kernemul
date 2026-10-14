#pragma once
#include "../../emu/object.hpp"

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <vector>

namespace win::sec
{

using bytes = std::vector<std::uint8_t>;
using sid_t = bytes;

constexpr std::uint32_t access_read_control    = 0x00020000;
constexpr std::uint32_t access_write_dac       = 0x00040000;
constexpr std::uint32_t access_system_security = 0x01000000;
constexpr std::uint32_t access_maximum_allowed = 0x02000000;
constexpr std::uint32_t access_generic_all     = 0x10000000;
constexpr std::uint32_t access_generic_execute = 0x20000000;
constexpr std::uint32_t access_generic_write   = 0x40000000;
constexpr std::uint32_t access_generic_read    = 0x80000000;
constexpr std::uint32_t access_generic_bits    = 0xF0000000;

constexpr std::uint32_t owner_security_information = 1;
constexpr std::uint32_t group_security_information = 2;
constexpr std::uint32_t dacl_security_information  = 4;

struct generic_mapping
{
	std::uint32_t read;
	std::uint32_t write;
	std::uint32_t execute;
	std::uint32_t all;
};

struct descriptor
{
	std::optional<sid_t> owner;
	std::optional<sid_t> group;
	std::optional<bytes> dacl;
};

[[nodiscard]] std::uint32_t map_generic(std::uint32_t mask, const generic_mapping& mapping);

template <typename T>
void put(bytes& out, const T value)
{
	const auto* p = reinterpret_cast<const std::uint8_t*>(&value);
	out.insert(out.end(), p, p + sizeof(T));
}

[[nodiscard]] sid_t make_sid(std::uint8_t authority, std::initializer_list<std::uint32_t> subs);

[[nodiscard]] bytes make_allow_all_acl(std::span<const sid_t> sids);

[[nodiscard]] std::optional<descriptor> read_descriptor(addr_space& space, addr_t address);

[[nodiscard]] bytes compose_self_relative(const descriptor& d, std::uint32_t information);

}
