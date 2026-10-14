#pragma once
#include <cstdint>

constexpr std::uint64_t ps_attribute_client_id   = 0x10003;
constexpr std::uint64_t ps_attribute_teb_address = 0x10004;
constexpr std::uint64_t ps_attribute_image_name  = 0x20005;

#pragma pack(push, 1)
struct ps_attribute_t
{
	std::uint64_t attribute;
	std::uint64_t size;
	std::uint64_t value;
	std::uint64_t return_length;
};
#pragma pack(pop)

static_assert(sizeof(ps_attribute_t) == 0x20);
