#include "nt_file_ops.hpp"
#include "../win_kernel.hpp"
#include "../file_common.hpp"
#include "../status.hpp"
#include "../string.hpp"
#include "../types.hpp"
#include "../../../util/log.hpp"
#include "../../../util/string.hpp"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace
{
	constexpr std::size_t page_size = 0x1000;

	template <typename T>
	void put(std::vector<std::uint8_t>& out, const std::size_t offset, const T value)
	{
		std::memcpy(out.data() + offset, &value, sizeof(T));
	}

	std::string lower(std::string text)
	{
		std::ranges::transform(text, text.begin(), [](const unsigned char c) { return std::tolower(c); });
		return text;
	}

	bool wildcard_match(const std::string_view pattern, const std::string_view name)
	{
		if (pattern.empty())
			return name.empty();

		if (pattern.front() == '*' || pattern.front() == '<')
		{
			for (std::size_t skip = 0; skip <= name.size(); ++skip)
			{
				if (wildcard_match(pattern.substr(1), name.substr(skip)))
					return true;
			}

			return false;
		}

		const char c = pattern.front() == '>' ? '?' : pattern.front() == '"' ? '.' : pattern.front();

		return !name.empty() && (c == '?' || c == name.front())
			&& wildcard_match(pattern.substr(1), name.substr(1));
	}

	std::size_t entry_name_offset(const std::uint32_t info_class)
	{
		switch (info_class)
		{
		case 1:  return 0x40;
		case 2:  return 0x44;
		case 3:  return 0x5E;
		case 12: return 0x0C;
		case 37: return 0x68;
		case 38: return 0x50;
		default: return 0;
		}
	}

	std::vector<std::uint8_t> build_entry(const std::uint32_t info_class, const win_filesystem::dir_entry& item,
		const std::uint32_t index)
	{
		const auto wide = widen_string(item.name);
		const auto name_bytes = static_cast<std::uint32_t>(wide.size() * sizeof(char16_t));
		const auto name_offset = entry_name_offset(info_class);

		std::vector<std::uint8_t> out(name_offset + name_bytes);

		put<std::uint32_t>(out, 4, index);

		if (info_class == 12)
		{
			put<std::uint32_t>(out, 8, name_bytes);
		}
		else
		{
			put<std::int64_t>(out, 0x28, static_cast<std::int64_t>(item.size));
			put<std::int64_t>(out, 0x30, static_cast<std::int64_t>(item.size));
			put<std::uint32_t>(out, 0x38, item.is_directory ? file_attribute_directory : file_attribute_normal);
			put<std::uint32_t>(out, 0x3C, name_bytes);
		}

		std::memcpy(out.data() + name_offset, wide.data(), name_bytes);

		return out;
	}

	open_result query_directory(win_kernel_state& state, vcpu& cpu, file_host& host,
		const addr_t buffer, const std::uint32_t length, const std::uint32_t info_class,
		const bool single, const emu_object<_UNICODE_STRING>& file_name, const bool restart,
		const bool keep_cursor)
	{
		if (!host.directory)
			return { STATUS_INVALID_PARAMETER, 0 };

		if (!entry_name_offset(info_class))
			return { STATUS_INVALID_INFO_CLASS, 0 };

		if (restart)
		{
			host.position = 0;
			host.dir_started = false;
		}

		const auto first_call = !host.dir_started;

		if (first_call && file_name)
		{
			host.dir_filter = lower(narrow_wstring(win::read_unicode_string(file_name)));

			if (host.dir_filter == "*.*")
				host.dir_filter.clear();
		}

		host.dir_started = true;

		std::vector<win_filesystem::dir_entry> items;

		if (host.path.find('/') != std::string::npos)
			items = { { ".", true, 0 }, { "..", true, 0 } };

		for (auto& entry : state.fs.list_dir(host.path))
			items.push_back(std::move(entry));

		auto& space = *cpu.curr_addr_space();
		std::uint64_t cursor = host.position;
		std::uint64_t written = 0;
		std::uint64_t last_start = 0;
		std::uint64_t last_size = 0;
		bool packed = false;

		for (; cursor < items.size(); ++cursor)
		{
			const auto& item = items[cursor];

			if (!host.dir_filter.empty() && !wildcard_match(host.dir_filter, item.name))
				continue;

			const auto entry = build_entry(info_class, item, static_cast<std::uint32_t>(cursor));

			if (written + entry.size() > length)
			{
				if (packed)
					break;

				host.dir_started = !first_call;

				if (!first_call)
					return { STATUS_SUCCESS, 0 };

				if (length < entry.size() - item.name.size() * sizeof(char16_t))
					return { STATUS_BUFFER_TOO_SMALL, 0 };

				space.write_mem(buffer, entry.data(), length);

				return { STATUS_BUFFER_OVERFLOW, length };
			}

			space.write_mem(buffer + written, entry.data(), entry.size());

			if (packed)
				space.write_mem<std::uint32_t>(buffer + last_start, static_cast<std::uint32_t>(written - last_start));

			last_start = written;
			last_size = entry.size();
			written += (entry.size() + 7) & ~std::size_t{7};
			packed = true;

			if (single)
			{
				++cursor;
				break;
			}
		}

		if (!packed)
		{
			host.position = items.size();
			return { first_call ? STATUS_NO_SUCH_FILE : STATUS_NO_MORE_FILES, 0 };
		}

		if (!keep_cursor)
			host.position = cursor;

		return { STATUS_SUCCESS, last_start + last_size };
	}

	std::shared_ptr<file_host> regular_file(win_kernel_state& state, const std::uint64_t handle,
		NTSTATUS& status)
	{
		auto host = state.sys_proc->handle_table().get_object<file_host>(handle);

		status = !host ? STATUS_INVALID_HANDLE
			: (host->is_device() || host->console || !host->file) ? STATUS_INVALID_DEVICE_REQUEST
			: host->directory ? STATUS_FILE_IS_A_DIRECTORY
			: STATUS_SUCCESS;

		return status == STATUS_SUCCESS ? host : nullptr;
	}

	NTSTATUS page_io(win_kernel_state& state, vcpu& cpu, const bool writing,
		const std::uint64_t file_handle, const std::uint64_t event,
		emu_object<_IO_STATUS_BLOCK> io_status_block, const addr_t segment_array,
		const std::uint32_t length, emu_object<std::int64_t> byte_offset)
	{
		NTSTATUS status;
		const auto host = regular_file(state, file_handle, status);

		if (!host)
			return finish(io_status_block, status);

		if (!length || length % page_size || !segment_array)
			return finish(io_status_block, STATUS_INVALID_PARAMETER);

		auto& space = *cpu.curr_addr_space();
		auto& file = *host->file;

		std::vector<addr_t> pages;

		for (std::size_t i = 0; i < length / page_size; ++i)
			pages.push_back(static_cast<addr_t>(space.read_mem<std::uint64_t>(segment_array + i * 8)));

		const auto offset = writing ? write_offset(*host, byte_offset)
			: file_offset(byte_offset, host->position);

		if (!writing && offset >= file.size())
			return finish(io_status_block, STATUS_END_OF_FILE);

		const auto count = writing ? length : std::min<std::uint64_t>(length, file.size() - offset);

		if (lock_blocks(file, offset, count, host.get(), writing))
			return finish(io_status_block, STATUS_FILE_LOCK_CONFLICT);

		if (writing)
		{
			std::vector<std::uint8_t> bytes(length);

			for (std::size_t i = 0; i < pages.size(); ++i)
				space.read_mem(pages[i], bytes.data() + i * page_size, page_size);

			write_file_bytes(file, offset, bytes);
		}
		else
		{
			std::vector<std::uint8_t> page(page_size);

			for (std::size_t i = 0; i < pages.size(); ++i)
			{
				std::ranges::fill(page, 0);

				if (const auto from = offset + i * page_size; from < file.size())
					std::memcpy(page.data(), file.data().data() + from,
						std::min<std::uint64_t>(page_size, file.size() - from));

				space.write_mem(pages[i], page.data(), page.size());
			}
		}

		host->position = offset + count;
		signal_io_event(state, cpu, event);

		return finish(io_status_block, STATUS_SUCCESS, count);
	}

	std::vector<std::uint8_t> pack_extended_attributes(
		const std::vector<win_file::extended_attribute>& list, const std::size_t length,
		const bool single, std::size_t& count)
	{
		constexpr std::size_t header = 8;

		std::vector<std::uint8_t> out;
		std::size_t previous = 0;
		count = 0;

		for (const auto& ea : list)
		{
			const auto start = (out.size() + 3) & ~std::size_t{3};
			const auto size = header + ea.name.size() + 1 + ea.value.size();

			if (start + size > length)
				break;

			out.resize(start + size);

			if (count)
				put<std::uint32_t>(out, previous, static_cast<std::uint32_t>(start - previous));

			out[start + 4] = ea.flags;
			out[start + 5] = static_cast<std::uint8_t>(ea.name.size());
			put<std::uint16_t>(out, start + 6, static_cast<std::uint16_t>(ea.value.size()));
			std::memcpy(out.data() + start + header, ea.name.data(), ea.name.size());
			std::ranges::copy(ea.value, out.begin() + static_cast<std::ptrdiff_t>(start + header + ea.name.size() + 1));

			previous = start;
			++count;

			if (single)
				break;
		}

		return out;
	}
}

void modules::register_ntoskrnl_file_ops(win_kernel_state& state, proc_module& mod)
{
	auto* st = &state;

	state.redirect_ntzw(mod, "QueryDirectoryFile",
		[st](vcpu& cpu, const std::uint64_t file_handle, const std::uint64_t event,
			[[maybe_unused]] const addr_t apc_routine, [[maybe_unused]] const addr_t apc_context,
			emu_object<_IO_STATUS_BLOCK> io_status_block, const addr_t file_information,
			const std::uint32_t length, const std::uint32_t file_information_class,
			const bool return_single_entry, emu_object<_UNICODE_STRING> file_name,
			const bool restart_scan) -> NTSTATUS
		{
			const auto host = st->sys_proc->handle_table().get_object<file_host>(file_handle);

			if (!host)
				return finish(io_status_block, STATUS_INVALID_HANDLE);

			const auto result = query_directory(*st, cpu, *host, file_information, length,
				file_information_class, return_single_entry, file_name, restart_scan, false);

			signal_io_event(*st, cpu, event);

			return finish(io_status_block, result.status, result.information);
		});

	state.redirect_ntzw(mod, "QueryDirectoryFileEx",
		[st](vcpu& cpu, const std::uint64_t file_handle, const std::uint64_t event,
			[[maybe_unused]] const addr_t apc_routine, [[maybe_unused]] const addr_t apc_context,
			emu_object<_IO_STATUS_BLOCK> io_status_block, const addr_t file_information,
			const std::uint32_t length, const std::uint32_t file_information_class,
			const std::uint32_t query_flags, emu_object<_UNICODE_STRING> file_name) -> NTSTATUS
		{
			constexpr std::uint32_t restart_scan = 0x1;
			constexpr std::uint32_t return_single_entry = 0x2;
			constexpr std::uint32_t index_specified = 0x4;
			constexpr std::uint32_t no_cursor_update = 0x10;

			const auto host = st->sys_proc->handle_table().get_object<file_host>(file_handle);

			if (!host)
				return finish(io_status_block, STATUS_INVALID_HANDLE);

			if (query_flags & index_specified)
				return finish(io_status_block, STATUS_NOT_SUPPORTED);

			const auto result = query_directory(*st, cpu, *host, file_information, length,
				file_information_class, query_flags & return_single_entry, file_name,
				query_flags & restart_scan, query_flags & no_cursor_update);

			signal_io_event(*st, cpu, event);

			return finish(io_status_block, result.status, result.information);
		});

	state.redirect_ntzw(mod, "DeleteFile",
		[st](vcpu& cpu, emu_object<_OBJECT_ATTRIBUTES> object_attributes) -> NTSTATUS
		{
			const auto path = object_path(*st, cpu, object_attributes);

			if (path.empty())
				return STATUS_OBJECT_NAME_INVALID;

			if (st->fs.dir_exists(path))
			{
				return st->fs.remove_directory(path) ? STATUS_SUCCESS : STATUS_DIRECTORY_NOT_EMPTY;
			}

			return st->fs.unlink(path) ? STATUS_SUCCESS : STATUS_OBJECT_NAME_NOT_FOUND;
		});

	state.redirect_ntzw(mod, "LockFile",
		[st](vcpu& cpu, const std::uint64_t file_handle, const std::uint64_t event,
			[[maybe_unused]] const addr_t apc_routine, [[maybe_unused]] const addr_t apc_context,
			emu_object<_IO_STATUS_BLOCK> io_status_block, emu_object<std::int64_t> byte_offset,
			emu_object<std::int64_t> length, [[maybe_unused]] const std::uint32_t key,
			[[maybe_unused]] const bool fail_immediately, const bool exclusive_lock) -> NTSTATUS
		{
			NTSTATUS status;
			const auto host = regular_file(*st, file_handle, status);

			if (!host)
				return finish(io_status_block, status);

			if (!byte_offset || !length || byte_offset.read() < 0 || length.read() < 0)
				return finish(io_status_block, STATUS_INVALID_PARAMETER);

			const auto offset = static_cast<std::uint64_t>(byte_offset.read());
			const auto count = static_cast<std::uint64_t>(length.read());
			auto& locks = host->file->locks;

			if (std::ranges::any_of(locks, [&](const auto& lock)
				{ return lock_overlaps(lock, offset, count) && (lock.exclusive || exclusive_lock); }))
			{
				return finish(io_status_block, STATUS_LOCK_NOT_GRANTED);
			}

			locks.push_back({ offset, count, exclusive_lock, host.get() });
			signal_io_event(*st, cpu, event);

			return finish(io_status_block, STATUS_SUCCESS);
		});

	state.redirect_ntzw(mod, "UnlockFile",
		[st](vcpu&, const std::uint64_t file_handle, emu_object<_IO_STATUS_BLOCK> io_status_block,
			emu_object<std::int64_t> byte_offset, emu_object<std::int64_t> length,
			[[maybe_unused]] const std::uint32_t key) -> NTSTATUS
		{
			NTSTATUS status;
			const auto host = regular_file(*st, file_handle, status);

			if (!host)
				return finish(io_status_block, status);

			if (!byte_offset || !length)
				return finish(io_status_block, STATUS_INVALID_PARAMETER);

			auto& locks = host->file->locks;

			const auto it = std::ranges::find_if(locks, [&](const auto& lock)
			{
				return lock.offset == static_cast<std::uint64_t>(byte_offset.read())
					&& lock.length == static_cast<std::uint64_t>(length.read())
					&& lock.owner == host.get();
			});

			if (it == locks.end())
				return finish(io_status_block, STATUS_RANGE_NOT_LOCKED);

			locks.erase(it);

			return finish(io_status_block, STATUS_SUCCESS);
		});

	state.redirect_ntzw(mod, "CancelIoFile",
		[st](vcpu&, const std::uint64_t file_handle,
			emu_object<_IO_STATUS_BLOCK> io_status_block) -> NTSTATUS
		{
			return finish(io_status_block, st->sys_proc->handle_table().lookup_handle(file_handle)
				? STATUS_SUCCESS : STATUS_INVALID_HANDLE);
		});

	for (const auto* const name : { "CancelIoFileEx", "CancelSynchronousIoFile" })
	{
		state.redirect_ntzw(mod, name,
			[st](vcpu&, const std::uint64_t handle,
				[[maybe_unused]] emu_object<_IO_STATUS_BLOCK> io_request_to_cancel,
				emu_object<_IO_STATUS_BLOCK> io_status_block) -> NTSTATUS
			{
				return finish(io_status_block, st->sys_proc->handle_table().lookup_handle(handle)
					? STATUS_NOT_FOUND : STATUS_INVALID_HANDLE);
			});
	}

	for (const bool writing : { false, true })
	{
		state.redirect_ntzw(mod, writing ? "WriteFileGather" : "ReadFileScatter",
			[st, writing](vcpu& cpu, const std::uint64_t file_handle, const std::uint64_t event,
				[[maybe_unused]] const addr_t apc_routine, [[maybe_unused]] const addr_t apc_context,
				emu_object<_IO_STATUS_BLOCK> io_status_block, const addr_t segment_array,
				const std::uint32_t length, emu_object<std::int64_t> byte_offset,
				[[maybe_unused]] emu_object<std::uint32_t> key) -> NTSTATUS
			{
				return page_io(*st, cpu, writing, file_handle, event, io_status_block, segment_array,
					length, byte_offset);
			});
	}

	state.redirect_ntzw(mod, "SetEaFile",
		[st](vcpu& cpu, const std::uint64_t file_handle,
			emu_object<_IO_STATUS_BLOCK> io_status_block, const addr_t buffer,
			const std::uint32_t length) -> NTSTATUS
		{
			NTSTATUS status;
			const auto host = regular_file(*st, file_handle, status);

			if (!host)
				return finish(io_status_block, status == STATUS_FILE_IS_A_DIRECTORY ? STATUS_NOT_SUPPORTED : status);

			if (!buffer || !length)
				return finish(io_status_block, STATUS_INVALID_PARAMETER);

			std::vector<std::uint8_t> raw(length);
			cpu.curr_addr_space()->read_mem(buffer, raw.data(), raw.size());

			auto parsed = parse_ea_chain(raw);

			for (auto& item : parsed.items)
				set_extended_attribute(*host->file, std::move(item));

			return finish(io_status_block, parsed.status);
		});

	state.redirect_ntzw(mod, "QueryEaFile",
		[st](vcpu& cpu, const std::uint64_t file_handle,
			emu_object<_IO_STATUS_BLOCK> io_status_block, const addr_t buffer,
			const std::uint32_t length, const bool return_single_entry, const addr_t ea_list,
			const std::uint32_t ea_list_length, emu_object<std::uint32_t> ea_index,
			const bool restart_scan) -> NTSTATUS
		{
			NTSTATUS status;
			const auto host = regular_file(*st, file_handle, status);

			if (!host)
				return finish(io_status_block, status == STATUS_FILE_IS_A_DIRECTORY ? STATUS_NO_EAS_ON_FILE : status);

			const auto& all = host->file->eas;

			if (all.empty())
				return finish(io_status_block, STATUS_NO_EAS_ON_FILE);

			std::vector<win_file::extended_attribute> chosen;
			std::size_t start = 0;
			const bool by_list = ea_list && ea_list_length;

			if (by_list)
			{
				std::vector<std::uint8_t> raw(ea_list_length);
				cpu.curr_addr_space()->read_mem(ea_list, raw.data(), raw.size());

				for (std::size_t at = 0;;)
				{
					if (at + 5 > raw.size() || at + 5 + raw[at + 4] > raw.size())
						return finish(io_status_block, STATUS_EA_LIST_INCONSISTENT);

					std::string name(reinterpret_cast<const char*>(raw.data() + at + 5), raw[at + 4]);
					std::ranges::transform(name, name.begin(), [](const unsigned char c) { return std::toupper(c); });

					const auto found = std::ranges::find(all, name, &win_file::extended_attribute::name);
					chosen.push_back(found != all.end() ? *found : win_file::extended_attribute{ name, 0, {} });

					std::uint32_t next;
					std::memcpy(&next, raw.data() + at, sizeof(next));

					if (!next)
						break;

					at += next;
				}
			}
			else
			{
				if (restart_scan)
					host->ea_scan = 0;

				start = ea_index ? ea_index.read() - 1 : host->ea_scan;

				if (ea_index && (ea_index.read() == 0 || ea_index.read() > all.size()))
					return finish(io_status_block, STATUS_NONEXISTENT_EA_ENTRY);

				if (start >= all.size())
					return finish(io_status_block, STATUS_NO_MORE_EAS);

				chosen.assign(all.begin() + static_cast<std::ptrdiff_t>(start), all.end());
			}

			std::size_t count;
			const auto out = pack_extended_attributes(chosen, length, return_single_entry, count);

			if (!count)
				return finish(io_status_block, STATUS_BUFFER_OVERFLOW);

			cpu.curr_addr_space()->write_mem(buffer, out.data(), out.size());

			if (!by_list)
				host->ea_scan = static_cast<std::uint32_t>(start + count);

			return finish(io_status_block, STATUS_SUCCESS, out.size());
		});
}
