#include "security.hpp"

#include <cstring>
#include <exception>

namespace win::sec
{

namespace
{

constexpr std::size_t relative_header_size = 20;
constexpr std::size_t part_header_size = 8;
constexpr std::size_t max_sub_authorities = 15;

}

std::uint32_t map_generic(std::uint32_t mask, const generic_mapping& mapping)
{
	if (mask & access_generic_read)
		mask |= mapping.read;

	if (mask & access_generic_write)
		mask |= mapping.write;

	if (mask & access_generic_execute)
		mask |= mapping.execute;

	if (mask & access_generic_all)
		mask |= mapping.all;

	return mask & ~access_generic_bits;
}

sid_t make_sid(const std::uint8_t authority, const std::initializer_list<std::uint32_t> subs)
{
	sid_t sid{ 1, static_cast<std::uint8_t>(subs.size()), 0, 0, 0, 0, 0, authority };

	for (const auto sub : subs)
		put(sid, sub);

	return sid;
}

bytes make_allow_all_acl(const std::span<const sid_t> sids)
{
	bytes acl{ 2, 0, 0, 0, 0, 0, 0, 0 };

	for (const auto& sid : sids)
	{
		put<std::uint16_t>(acl, 0);
		put(acl, static_cast<std::uint16_t>(8 + sid.size()));
		put(acl, access_generic_all);
		acl.insert(acl.end(), sid.begin(), sid.end());
	}

	const auto size = static_cast<std::uint16_t>(acl.size());
	const auto count = static_cast<std::uint16_t>(sids.size());

	std::memcpy(acl.data() + 2, &size, sizeof(size));
	std::memcpy(acl.data() + 4, &count, sizeof(count));

	return acl;
}

std::optional<descriptor> read_descriptor(addr_space& space, const addr_t address)
{
	if (!address)
		return std::nullopt;

	try
	{
		std::uint8_t head[relative_header_size]{};
		space.read_mem(address, head, sizeof(head));

		if (head[0] != 1 || !(head[3] & 0x80))
			return std::nullopt;

		const auto read_part = [&](const std::size_t field, const bool acl) -> std::optional<bytes>
		{
			std::uint32_t offset = 0;
			std::memcpy(&offset, head + field, sizeof(offset));

			if (!offset)
				return std::nullopt;

			std::uint8_t part[part_header_size]{};
			space.read_mem(address + offset, part, sizeof(part));

			if (!acl && (part[0] != 1 || part[1] > max_sub_authorities))
				return std::nullopt;

			const std::size_t size = acl ? part[2] | part[3] << 8 : 8 + part[1] * 4;

			if (size < part_header_size)
				return std::nullopt;

			bytes out(size);
			space.read_mem(address + offset, out.data(), out.size());

			return out;
		};

		descriptor d;
		d.owner = read_part(4, false);
		d.group = read_part(8, false);

		if (head[2] & 4)
			d.dacl = read_part(16, true);

		return d;
	}
	catch (const std::exception&)
	{
		return std::nullopt;
	}
}

bytes compose_self_relative(const descriptor& d, const std::uint32_t information)
{
	bytes out(relative_header_size);
	out[0] = 1;
	out[3] = 0x80;

	const auto place = [&](const std::optional<bytes>& part, const std::uint32_t flag,
		const std::size_t field)
	{
		if (!(information & flag) || !part)
			return false;

		const auto at = static_cast<std::uint32_t>(out.size());

		std::memcpy(out.data() + field, &at, sizeof(at));
		out.insert(out.end(), part->begin(), part->end());
		out.resize((out.size() + 3) & ~std::size_t{3});

		return true;
	};

	if (place(d.dacl, dacl_security_information, 16))
		out[2] = 4;

	place(d.owner, owner_security_information, 4);
	place(d.group, group_security_information, 8);

	return out;
}

}
