#pragma once
#include "objects.hpp"
#include "status.hpp"
#include "string.hpp"
#include "types.hpp"
#include "win_kernel.hpp"
#include "../../util/string.hpp"

#include <string>

constexpr std::uint32_t obj_inherit = 0x00000002;
constexpr std::uint32_t obj_openif  = 0x00000080;

struct resolved_object_name
{
	NTSTATUS status = STATUS_SUCCESS;
	std::string name;
	std::string key;
	std::uint32_t attributes = 0;

	[[nodiscard]] bool named() const noexcept { return !key.empty(); }
};

inline resolved_object_name resolve_object_name(win_kernel_state& state, vcpu& cpu,
	const emu_object<_OBJECT_ATTRIBUTES>& object_attributes)
{
	resolved_object_name result;

	if (!object_attributes)
		return result;

	result.attributes = object_attributes.field(&_OBJECT_ATTRIBUTES::Attributes).read();

	const emu_object<_UNICODE_STRING> raw(*cpu.curr_addr_space(),
		guest_va(object_attributes.field(&_OBJECT_ATTRIBUTES::ObjectName).read()));

	auto name = narrow_wstring(win::read_unicode_string(raw));

	if (const auto root = guest_va(object_attributes.field(&_OBJECT_ATTRIBUTES::RootDirectory).read()))
	{
		const auto entry = state.sys_proc->handle_table().lookup_handle(root);

		if (!entry)
		{
			result.status = STATUS_INVALID_HANDLE;
			return result;
		}

		const auto dir = state.objs.get_object<directory_host>(entry->body_addr);

		if (!dir)
		{
			result.status = STATUS_OBJECT_TYPE_MISMATCH;
			return result;
		}

		name.erase(0, name.find_first_not_of('\\'));
		name = name.empty() ? dir->name : dir->name + "\\" + name;
	}

	result.key = name.empty() ? std::string() : object_namespace_key(name);
	result.name = std::move(name);

	return result;
}
