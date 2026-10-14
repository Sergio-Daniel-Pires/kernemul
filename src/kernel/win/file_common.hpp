#pragma once
#include "dispatcher.hpp"
#include "filesystem.hpp"
#include "objects.hpp"
#include "security.hpp"
#include "status.hpp"
#include "string.hpp"
#include "types.hpp"
#include "win_kernel.hpp"
#include "../../util/string.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

constexpr std::uint32_t file_attribute_normal = 0x80;
constexpr std::uint32_t file_attribute_directory = 0x10;

inline std::string object_path(vcpu& cpu, const emu_object<_OBJECT_ATTRIBUTES>& object_attributes)
{
	const emu_object<_UNICODE_STRING> name(*cpu.curr_addr_space(), object_attributes
		? guest_va(object_attributes.field(&_OBJECT_ATTRIBUTES::ObjectName).read())
		: 0);

	return win_filesystem::normalize(narrow_wstring(win::read_unicode_string(name)));
}

inline std::string object_path(win_kernel_state& state, vcpu& cpu,
	const emu_object<_OBJECT_ATTRIBUTES>& object_attributes)
{
	auto name = object_path(cpu, object_attributes);
	const auto root = object_attributes
		? guest_va(object_attributes.field(&_OBJECT_ATTRIBUTES::RootDirectory).read())
		: 0;

	if (!root)
		return name;

	auto& handles = state.sys_proc->handle_table();
	std::string base;

	if (const auto file = handles.get_object<file_host>(root))
		base = file->path;
	else if (const auto pipe = handles.get_object<pipe_host>(root))
		base = pipe->path;
	else
		return {};

	name.erase(0, name.find_first_not_of('/'));

	return name.empty() ? base : base + "/" + name;
}

inline std::uint64_t file_offset(const emu_object<std::int64_t>& byte_offset,
	const std::uint64_t position)
{
	const auto value = byte_offset ? byte_offset.read() : -2;

	return value < 0 ? position : static_cast<std::uint64_t>(value);
}

struct open_result
{
	NTSTATUS status = STATUS_SUCCESS;
	std::uint64_t information = 0;
};

inline void write_status_block(emu_object<_IO_STATUS_BLOCK> block, const open_result& result)
{
	if (!block)
		return;

	_IO_STATUS_BLOCK status{};
	status.Status = static_cast<std::int32_t>(result.status);
	status.Information = result.information;

	block.write(status);
}

inline std::uint32_t file_handle_access(const std::uint32_t desired)
{
	constexpr win::sec::generic_mapping file_mapping{ 0x120089, 0x120116, 0x1200A0, 0x1F01FF };

	if (desired & win::sec::access_maximum_allowed)
		return file_mapping.all;

	return win::sec::map_generic(desired, file_mapping);
}

template <class host_type>
inline std::uint64_t make_file_handle(win_kernel_state& state, std::shared_ptr<host_type> host,
	const std::uint32_t access)
{
	const std::uint8_t body[sizeof(addr_t)] = {};

	const auto addr = state.objs.create_object(0, body, sizeof(body), std::move(host),
		prot_rw | prot_supervisor);

	return addr ? state.sys_proc->handle_table().create_handle(addr, access) : 0;
}

inline void write_file_bytes(win_file& file, const std::uint64_t offset,
	const std::span<const std::uint8_t> bytes)
{
	std::vector<std::uint8_t> content(file.data().begin(), file.data().end());

	if (content.size() < offset + bytes.size())
		content.resize(offset + bytes.size());

	std::ranges::copy(bytes, content.begin() + static_cast<std::ptrdiff_t>(offset));
	file.write(content);
}

inline void delete_host(win_kernel_state& state, const file_host& host)
{
	if (host.directory)
		state.fs.remove_directory(host.path);
	else if (state.fs.open(host.path) == host.file)
		state.fs.unlink(host.path);
}

inline NTSTATUS finish(emu_object<_IO_STATUS_BLOCK> block, const NTSTATUS status,
	const std::uint64_t information = 0)
{
	write_status_block(block, { status, information });

	return status;
}

inline std::uint64_t write_offset(const file_host& host, const emu_object<std::int64_t>& byte_offset)
{
	return byte_offset && byte_offset.read() == -1 ? host.file->size()
		: file_offset(byte_offset, host.position);
}

inline bool lock_overlaps(const win_file::byte_lock& lock, const std::uint64_t offset,
	const std::uint64_t length)
{
	return length && lock.length && lock.offset < offset + length && offset < lock.offset + lock.length;
}

inline bool lock_blocks(const win_file& file, const std::uint64_t offset, const std::uint64_t length,
	const void* who, const bool writing)
{
	return std::ranges::any_of(file.locks, [&](const win_file::byte_lock& lock)
	{
		const bool mine = lock.owner == who;

		return lock_overlaps(lock, offset, length)
			&& (writing ? !(lock.exclusive && mine) : lock.exclusive && !mine);
	});
}

inline void set_extended_attribute(win_file& file, win_file::extended_attribute attribute)
{
	std::ranges::transform(attribute.name, attribute.name.begin(),
		[](const unsigned char c) { return static_cast<char>(std::toupper(c)); });

	const auto it = std::ranges::find(file.eas, attribute.name, &win_file::extended_attribute::name);

	if (attribute.value.empty())
	{
		if (it != file.eas.end())
			file.eas.erase(it);
	}
	else if (it != file.eas.end())
	{
		*it = std::move(attribute);
	}
	else
	{
		file.eas.push_back(std::move(attribute));
	}
}

struct ea_parse_result
{
	NTSTATUS status = STATUS_SUCCESS;
	std::vector<win_file::extended_attribute> items;
};

inline ea_parse_result parse_ea_chain(const std::span<const std::uint8_t> buffer)
{
	constexpr std::size_t header = 8;
	constexpr std::string_view invalid_name_characters = "\"*+,/:;<=>?\\|[]";

	ea_parse_result result;
	std::size_t offset = 0;

	while (true)
	{
		if (offset + header > buffer.size())
			return { STATUS_EA_LIST_INCONSISTENT, {} };

		const auto* const at = buffer.data() + offset;
		std::uint32_t next;
		std::uint16_t value_length;
		std::memcpy(&next, at, sizeof(next));
		std::memcpy(&value_length, at + 6, sizeof(value_length));

		const std::size_t name_length = at[5];
		const std::size_t end = header + name_length + 1 + value_length;

		if (offset + end > buffer.size() || at[header + name_length] != 0)
			return { STATUS_EA_LIST_INCONSISTENT, {} };

		win_file::extended_attribute item;
		item.name.assign(reinterpret_cast<const char*>(at + header), name_length);
		item.flags = at[4];
		item.value.assign(at + header + name_length + 1, at + end);

		const bool valid = !item.name.empty() && std::ranges::none_of(item.name, [&](const char c)
		{
			return static_cast<unsigned char>(c) < 0x20 || invalid_name_characters.contains(c);
		});

		if (!valid)
			return { STATUS_INVALID_EA_NAME, {} };

		result.items.push_back(std::move(item));

		if (!next)
			return result;

		if (next < end || offset + next >= buffer.size())
			return { STATUS_EA_LIST_INCONSISTENT, {} };

		offset += next;
	}
}

inline void signal_io_event(win_kernel_state& state, vcpu& cpu, const std::uint64_t event_handle)
{
	const auto entry = event_handle ? state.sys_proc->handle_table().lookup_handle(event_handle)
		: std::nullopt;
	const auto host = entry ? state.objs.get_object<dispatcher_host>(entry->body_addr) : nullptr;

	if (!host || (host->type != win::event_notification_object
		&& host->type != win::event_synchronization_object))
	{
		return;
	}

	auto& space = *cpu.curr_addr_space();

	win::set_signal_state(emu_object<_KEVENT>(space, entry->body_addr), 1);
	state.sys_proc->wake_waiters(space, entry->body_addr);
}
