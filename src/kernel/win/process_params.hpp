#pragma once
#include "../../emu/object.hpp"
#include "api_set.hpp"
#include "filesystem.hpp"
#include "string.hpp"
#include <pe.hpp>
#include <vector>

constexpr std::string_view  root_dir_narrow     = "C:\\";
constexpr std::string_view  windows_dir_narrow  = "C:\\Windows";
constexpr std::string_view  system32_dir_narrow = "C:\\Windows\\System32\\";

// Where a kernel driver lives on a real machine, which is where one looks for its own image.
constexpr std::string_view  drivers_dir_narrow  = "C:\\Windows\\System32\\drivers\\";

// Where a kernel driver lives on a real machine, which is where one looks for its own image.
constexpr std::u16string_view windows_dir  = u"C:\\Windows";
constexpr std::u16string_view system32_dir = u"C:\\Windows\\System32\\";

namespace win
{

struct environment_block
{
	addr_t address;
	std::uint64_t size;
};

inline environment_block allocate_environment_block(win_user_mem& mem)
{
	std::u16string env;
	env += u"PATH=";
	env += system32_dir;
	env += u'\0';
	env += u"SystemRoot=";
	env += windows_dir;
	env += u'\0';
	env += u"TEMP=C:\\Users\\Default\\AppData\\Local\\Temp";
	env += u'\0';
	env += u'\0';

	const auto size = env.size() * sizeof(char16_t);
	const auto addr = mem.alloc(size, prot_rw);
	mem.write_mem(addr, env.data(), size);
	return { addr, size };
}

inline std::u16string_view dir_from_path(std::u16string_view path)
{
	const auto pos = path.find_last_of(u"\\/");
	return pos != std::u16string_view::npos ? path.substr(0, pos + 1) : path;
}

inline emu_object<_RTL_USER_PROCESS_PARAMETERS> init_process_parameters(
	win_user_mem& mem, std::u16string_view image_path, std::u16string_view cmd = {})
{
	const auto current_dir = dir_from_path(image_path);

	// One block, Length covering both: the loader rebases every Buffer against it on copy.
	std::vector<char16_t> strings;

	struct placement
	{
		std::size_t offset;
		unsigned short length;
	};

	const auto place = [&strings](const std::u16string_view text)
	{
		const placement at{ strings.size() * sizeof(char16_t),
			static_cast<unsigned short>(text.size() * sizeof(char16_t)) };

		// Terminated as well as counted: the loader runs plain string functions over these.
		strings.insert(strings.end(), text.begin(), text.end());
		strings.push_back(u'\0');

		return at;
	};

	const auto dos_path     = place(current_dir);
	const auto dll_path     = place(system32_dir);
	const auto image_name   = place(image_path);
	const auto command_line = place(cmd.empty() ? image_path : cmd);

	constexpr std::size_t strings_at = sizeof(_RTL_USER_PROCESS_PARAMETERS);
	const auto total = strings_at + strings.size() * sizeof(char16_t);
	const auto addr = mem.alloc(total, prot_rw);

	const auto string_at = [addr](const placement& at)
	{
		return _UNICODE_STRING{
			.Length = at.length,
			.MaximumLength = static_cast<unsigned short>(at.length + sizeof(char16_t)),
			.Buffer = guest_ptr<char16_t>(addr + strings_at + at.offset),
		};
	};

	_RTL_USER_PROCESS_PARAMETERS params{};
	params.MaximumLength = static_cast<std::uint32_t>(total);
	params.Length = static_cast<std::uint32_t>(total);
	params.Flags = 0x6001;
	params.ConsoleHandle = guest_ptr(~0ULL);

	params.CurrentDirectory.DosPath = string_at(dos_path);
	params.DllPath = string_at(dll_path);
	params.ImagePathName = string_at(image_name);
	params.CommandLine = string_at(command_line);

	const auto env = allocate_environment_block(mem);
	params.Environment = guest_ptr(env.address);
	params.EnvironmentSize = env.size;

	mem.write_mem(addr, &params, sizeof(params));
	mem.write_mem(addr + strings_at, strings.data(), strings.size() * sizeof(char16_t));

	return emu_object<_RTL_USER_PROCESS_PARAMETERS>(mem.space(), addr,
		"RTL_USER_PROCESS_PARAMETERS", true);
}

struct api_set_result
{
	addr_t address = 0;
	api_set_map map;
};

inline api_set_result load_api_set_from_pe(win_user_mem& mem, std::span<const std::uint8_t> data)
{
	if (data.size() < sizeof(pe::dos_header))
		return {};

	const auto* img = reinterpret_cast<const pe::image*>(data.data());
	if (!img->dos_hdr()->ok())
		return {};

	for (const auto& sec : img->sections())
	{
		if (sec.name() != ".apiset")
			continue;

		if (!sec.pointer_to_raw_data || !sec.size_of_raw_data)
			break;

		const auto offset = static_cast<std::size_t>(sec.pointer_to_raw_data);
		const auto size = static_cast<std::size_t>(sec.size_of_raw_data);
		if (offset + size > data.size())
			break;

		const auto section = data.subspan(offset, size);

		const auto addr = mem.alloc(size, prot_rw);
		mem.write_mem(addr, section.data(), size);

		return { addr, parse_api_set_map(section) };
	}

	return {};
}

inline api_set_result init_api_set_map(win_user_mem& mem, const win_filesystem& fs)
{
	if (const auto file = fs.open(std::string(system32_dir_narrow) + "apisetschema.dll"))
	{
		if (auto result = load_api_set_from_pe(mem, file->data()); result.address)
		{
			LOG_INFO("api set schema: {} names resolve through it", result.map.hosts.size());
			return result;
		}
	}

	const _API_SET_NAMESPACE ns{
		.Version = 6,
		.Size = sizeof(_API_SET_NAMESPACE),
		.Flags = 0,
		.Count = 0,
		.EntryOffset = sizeof(_API_SET_NAMESPACE),
		.HashOffset = sizeof(_API_SET_NAMESPACE),
		.HashFactor = 0,
	};

	const auto addr = mem.alloc(sizeof(ns), prot_rw);
	mem.write_mem(addr, &ns, sizeof(ns));
	return { addr, {} };
}

}
