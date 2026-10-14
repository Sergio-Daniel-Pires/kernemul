#include "nt_object_ops.hpp"
#include "../win_kernel.hpp"
#include "../object_names.hpp"
#include "../file_common.hpp"
#include "../objects.hpp"
#include "../pipe.hpp"
#include "../status.hpp"
#include "../process_params.hpp"
#include "../string.hpp"
#include "../types.hpp"
#include "../../../util/string.hpp"
#include "../../../util/log.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace
{

constexpr std::size_t namespace_object_body_size = 0x10;

enum object_information_class : std::uint32_t
{
	object_basic_information       = 0,
	object_name_information        = 1,
	object_type_information        = 2,
	object_types_information       = 3,
	object_handle_flag_information = 4,
};

struct object_kind
{
	std::string type;
	std::string name;
	std::uint32_t valid_access = 0;
};

template <typename T>
void put(std::vector<std::uint8_t>& out, const std::size_t offset, const T value)
{
	std::memcpy(out.data() + offset, &value, sizeof(T));
}

std::optional<object_kind> describe_object(win_kernel_state& state, const addr_t body)
{
	if (const auto file = state.objs.get_object<file_host>(body))
	{
		auto path = file->path;
		std::ranges::replace(path, '/', '\\');

		if (!file->is_device())
		{
			if (path.size() >= 2 && path[1] == ':')
				path.erase(0, 2);

			if (path.empty() || path.front() != '\\')
				path.insert(path.begin(), '\\');

			path = "\\Device\\HarddiskVolume3" + path;
		}

		return object_kind{ "File", file->console ? std::string() : path, 0x1F01FF };
	}

	if (const auto dispatcher = state.objs.get_object<dispatcher_host>(body))
	{
		switch (dispatcher->type)
		{
		case win::event_notification_object:
		case win::event_synchronization_object:
			return object_kind{ "Event", dispatcher->name, 0x1F0003 };

		case win::mutant_object:
			return object_kind{ "Mutant", dispatcher->name, 0x1F0001 };

		case win::semaphore_object:
			return object_kind{ "Semaphore", dispatcher->name, 0x1F0003 };

		case win::timer_notification_object:
		case win::timer_synchronization_object:
			return object_kind{ "Timer", dispatcher->name, 0x1F0003 };

		default:
			return std::nullopt;
		}
	}

	if (const auto directory = state.objs.get_object<directory_host>(body))
		return object_kind{ "Directory", directory->name, 0xF000F };

	if (state.objs.get_object<symbolic_link_host>(body))
		return object_kind{ "SymbolicLink", {}, 0xF0001 };

	if (state.objs.get_object<section_host>(body))
		return object_kind{ "Section", {}, 0xF001F };

	if (state.objs.get_object<thread_object>(body))
		return object_kind{ "Thread", {}, 0x1FFFFF };

	if (state.objs.get_object<process_object>(body))
		return object_kind{ "Process", {}, 0x1FFFFF };

	if (state.objs.get_object<win::sec::token_host>(body))
		return object_kind{ "Token", {}, 0xF01FF };

	return std::nullopt;
}

// WDK types the kernel does not store, so not in the PDB; both are the same on each architecture.
#pragma pack(push, 8)
struct ob_callback_registration_t
{
	std::uint16_t   version;
	std::uint16_t   operation_count;
	std::uint32_t   flags;
	_UNICODE_STRING altitude;
	addr_t          registration_context;
	addr_t          operation_registration;
};

struct ob_operation_registration_t
{
	addr_t        object_type;
	std::uint32_t operations;
	std::uint32_t padding;
	addr_t        pre_operation;
	addr_t        post_operation;
};
#pragma pack(pop)

}

void modules::register_ntoskrnl_object_ops(win_kernel_state& state, proc_module& mod)
{
	auto* sys_proc = state.sys_proc.get();

	state.redirect(mod, "ObReferenceObjectByHandle",
		[sys_proc, &objs = state.objs](vcpu& cpu,
			std::uint64_t handle, [[maybe_unused]] std::uint32_t desired_access,
			[[maybe_unused]] emu_object<void> object_type, [[maybe_unused]] std::uint8_t access_mode,
			emu_object<std::uint64_t> object_out) -> NTSTATUS
		{
			THREAD_LOG_INFO("ObReferenceObjectByHandle(handle=0x{:X}, object_out=0x{:X})",
				handle, object_out.address());

			const auto entry = sys_proc->handle_table().lookup_handle(handle);
			if (!entry)
			{
				THREAD_LOG_WARN("ObReferenceObjectByHandle: invalid handle 0x{:X}", handle);
				return STATUS_INVALID_HANDLE;
			}

			objs.reference_object(entry->body_addr);

			if (object_out)
				object_out.write(entry->body_addr);

			return STATUS_SUCCESS;
		});

	state.redirect(mod, "ObfReferenceObject",
		[&objs = state.objs](vcpu&, addr_t object) -> void
		{
			THREAD_LOG_INFO("ObfReferenceObject(object=0x{:X})", object);
			objs.reference_object(object);
		});

	// Only four object types exist here and nothing ties one to an object, so the type a caller
	// asks for is logged rather than enforced: a mismatch it never made is worse than no check.
	state.redirect(mod, "ObReferenceObjectByPointer",
		[&objs = state.objs](vcpu&, const addr_t object,
			[[maybe_unused]] const std::uint32_t desired_access, const addr_t object_type,
			[[maybe_unused]] const std::uint8_t access_mode) -> NTSTATUS
		{
			if (!object)
				return STATUS_INVALID_PARAMETER;

			if (!objs.has_object(object))
				THREAD_LOG_WARN("ObReferenceObjectByPointer: 0x{:X} is not an object here, so "
					"the reference is not counted", object);
			else
				objs.reference_object(object);

			THREAD_LOG_INFO("ObReferenceObjectByPointer(object=0x{:X}, type=0x{:X})",
				object, object_type);

			return STATUS_SUCCESS;
		});

	auto deref = [&objs = state.objs](vcpu&, addr_t object) -> void
	{
		THREAD_LOG_INFO("ObfDereferenceObject(object=0x{:X})", object);
		objs.dereference_object(object);
	};

	state.redirect(mod, "ObfDereferenceObject", deref);
	state.redirect(mod, "ObfDereferenceObjectWithTag", deref);

	auto close_fn = [st = &state, sys_proc](vcpu& cpu, std::uint64_t handle) -> NTSTATUS
	{
		THREAD_LOG_INFO("NtClose(handle=0x{:X})", handle);

		const auto entry = sys_proc->handle_table().lookup_handle(handle);

		if (entry && (entry->flags & win_handle_table::flag_protect_from_close))
		{
			THREAD_LOG_WARN("NtClose: handle 0x{:X} is protected from close", handle);
			return STATUS_HANDLE_NOT_CLOSABLE;
		}

		const auto counts = entry ? st->objs.counts_of(entry->body_addr) : std::nullopt;
		const bool last_handle = counts && counts->handles == 1;

		// The last handle on a device is what tells its driver the open is over, so the two
		// requests real cleanup sends go out before the handle stops naming anything.
		if (const auto host = sys_proc->handle_table().get_object<file_host>(handle);
			host && host->is_device())
		{
			if (auto* const emulator = st->emulator())
			{
				for (const auto major : { irp_mj_cleanup, irp_mj_close })
				{
					emulator->dispatch_irp(cpu, {
						.major = major,
						.device_object = host->device_object,
						.file_object = host->file_object,
					});
				}
			}

			st->pool.free(host->file_object);
		}

		if (!sys_proc->handle_table().close_handle(handle))
		{
			THREAD_LOG_WARN("NtClose: invalid handle 0x{:X}", handle);
			return STATUS_INVALID_HANDLE;
		}

		if (!last_handle)
			return STATUS_SUCCESS;

		if (const auto pipe = st->objs.get_object<pipe_host>(entry->body_addr))
		{
			st->pipes->close_end(cpu, *pipe);
		}
		else if (const auto host = st->objs.get_object<file_host>(entry->body_addr);
			host && host->file)
		{
			std::erase_if(host->file->locks, [&](const auto& lock) { return lock.owner == host.get(); });

			if (host->delete_on_close)
				delete_host(*st, *host);
		}

		return STATUS_SUCCESS;
	};

	state.redirect_ntzw(mod, "Close", close_fn);

	// Nothing here opens a handle down the path that would notify one, so a filter sees no traffic.
	state.redirect(mod, "ObRegisterCallbacks",
		[&objs = state.objs](vcpu& cpu, emu_object<void> callback_registration,
			emu_object<addr_t> registration_handle) -> NTSTATUS
		{
			if (!callback_registration || !registration_handle)
				return STATUS_INVALID_PARAMETER;

			auto& space = *cpu.curr_addr_space();
			const emu_object<ob_callback_registration_t> registration(space,
				callback_registration.address());

			const auto reg = registration.read();
			const auto altitude = narrow_wstring(win::read_unicode_string(
				emu_object<_UNICODE_STRING>(space, registration.address()
					+ offsetof(ob_callback_registration_t, altitude))));

			THREAD_LOG_INFO("ObRegisterCallbacks(version={}, altitude='{}', {} operation(s))",
				reg.version, altitude, reg.operation_count);

			for (std::uint16_t i = 0; i < reg.operation_count; ++i)
			{
				const emu_object<ob_operation_registration_t> op(space,
					reg.operation_registration + i * sizeof(ob_operation_registration_t));

				const auto entry = op.read();
				const auto type = space.read_mem<addr_t>(entry.object_type);

				THREAD_LOG_INFO("  operation[{}]: type=0x{:X} (at 0x{:X}), operations=0x{:X}, "
					"pre=0x{:X}, post=0x{:X}",
					i, type, entry.object_type, entry.operations,
					entry.pre_operation, entry.post_operation);
			}

			const auto count = reg.operation_count;

			const std::uint8_t body[sizeof(addr_t)] = {};
			const auto handle = objs.create_object(0, body, sizeof(body), {},
				prot_rw | prot_supervisor);

			if (!handle)
				return STATUS_INSUFFICIENT_RESOURCES;

			registration_handle.write(handle);

			THREAD_LOG_WARN("ObRegisterCallbacks(0x{:X}, {} operation(s)) -> 0x{:X}: nothing here "
				"notifies an object callback", callback_registration.address(), count, handle);

			return STATUS_SUCCESS;
		});

	state.redirect(mod, "ObUnRegisterCallbacks", [](vcpu&, const addr_t registration_handle)
	{
		THREAD_LOG_INFO("ObUnRegisterCallbacks(0x{:X})", registration_handle);
	});

	// Nothing here builds a type index table, so the kind comes from the host object instead.
	state.redirect(mod, "ObGetObjectType",
		[st = &state, m = &mod](vcpu& cpu, const addr_t object) -> addr_t
		{
			std::string_view symbol;

			if (st->objs.get_object<file_host>(object))
				symbol = "IoFileObjectType";
			else if (st->objs.get_object<section_host>(object))
				symbol = "MmSectionObjectType";
			else if (st->objs.get_object<thread_object>(object))
				symbol = "PsThreadType";

			if (symbol.empty())
			{
				THREAD_LOG_WARN("ObGetObjectType(0x{:X}): nothing here knows what kind of object "
					"that is", object);
				return 0;
			}

			const auto global = m->find_symbol(symbol);

			if (!global)
			{
				THREAD_LOG_WARN("ObGetObjectType: {} is not in {}", symbol, m->name);
				return 0;
			}

			const auto type = cpu.curr_addr_space()->read_mem<addr_t>(*global);

			THREAD_LOG_INFO("ObGetObjectType(0x{:X}) -> 0x{:X} ({})", object, type, symbol);

			return type;
		});

	state.redirect(mod, "ObOpenObjectByPointer",
		[st = &state](vcpu&, const addr_t object, const std::uint32_t handle_attributes,
			const addr_t passed_access_state, const std::uint32_t desired_access,
			const addr_t object_type, const std::uint8_t access_mode,
			emu_object<std::uint64_t> handle) -> NTSTATUS
		{
			if (!handle)
				return STATUS_INVALID_PARAMETER;

			st->objs.reference_object(object);
			const auto value = st->sys_proc->handle_table().create_handle(object, desired_access);

			handle.write(value);

			THREAD_LOG_INFO("ObOpenObjectByPointer(0x{:X}, attributes=0x{:X}, access_state=0x{:X}, "
				"access=0x{:X}, type=0x{:X}, mode={}) -> handle=0x{:X}",
				object, handle_attributes, passed_access_state, desired_access, object_type,
				access_mode, value);

			return STATUS_SUCCESS;
		});

	// There is one handle table, hanging off the system process, and every process here shares it.
	state.redirect(mod, "ObReferenceProcessHandleTable",
		[](vcpu&, emu_object<_EPROCESS> process) -> addr_t
		{
			const auto table = process
				? guest_va(process.field(&_EPROCESS::ObjectTable).read()) : 0;

			THREAD_LOG_INFO("ObReferenceProcessHandleTable(0x{:X}) -> 0x{:X}",
				process.address(), table);

			return table;
		});

	// Nothing here calls back into the guest, so null says the walk finished without a match.
	state.redirect(mod, "ExEnumHandleTable",
		[](vcpu&, const addr_t handle_table, const addr_t callback, const addr_t context,
			emu_object<std::uint64_t> handle) -> bool
		{
			if (handle)
				handle.write(0);

			THREAD_LOG_WARN("ExEnumHandleTable(table=0x{:X}, callback=0x{:X}, context=0x{:X}): "
				"nothing here calls back into the guest, so no handle is enumerated",
				handle_table, callback, context);

			return false;
		});

	// Nothing builds the directories these names would hang off, so opening one makes it.
	auto attribute_name = [](vcpu& cpu, const emu_object<_OBJECT_ATTRIBUTES>& object_attributes)
	{
		auto& space = *cpu.curr_addr_space();

		const emu_object<_UNICODE_STRING> name(space, object_attributes
			? guest_va(object_attributes.field(&_OBJECT_ATTRIBUTES::ObjectName).read())
			: 0);

		return narrow_wstring(win::read_unicode_string(name));
	};

	auto open_namespace_object = [st = &state](std::shared_ptr<win_object> host,
		const std::string& name, const std::uint32_t desired_access) -> std::uint64_t
	{
		if (const auto existing = st->objs.lookup_named_object(name))
			return st->sys_proc->handle_table().create_handle(existing, desired_access);

		const std::array<std::uint8_t, namespace_object_body_size> body{};
		const auto addr = st->objs.create_object(0, body.data(), body.size(), std::move(host),
			prot_rw | prot_supervisor);

		if (!addr)
			return 0;

		st->objs.register_named_object(name, addr);

		return st->sys_proc->handle_table().create_handle(addr, desired_access);
	};

	state.redirect_ntzw(mod, "OpenDirectoryObject",
		[st = &state, open_namespace_object](vcpu& cpu,
			emu_object<std::uint64_t> directory_handle, const std::uint32_t desired_access,
			emu_object<_OBJECT_ATTRIBUTES> object_attributes) -> NTSTATUS
		{
			if (!directory_handle)
				return STATUS_INVALID_PARAMETER;

			const auto resolved = resolve_object_name(*st, cpu, object_attributes);

			if (resolved.status != STATUS_SUCCESS)
				return resolved.status;

			const auto& name = resolved.name;

			auto host = std::make_shared<directory_host>();
			host->name = name;

			const auto handle = open_namespace_object(std::move(host), name, desired_access);

			if (!handle)
				return STATUS_INSUFFICIENT_RESOURCES;

			directory_handle.write(handle);

			THREAD_LOG_WARN("NtOpenDirectoryObject('{}') -> handle=0x{:X}: the directory is "
				"empty, because nothing here puts anything in one", name, handle);

			return STATUS_SUCCESS;
		});

	// Nothing puts anything in a directory here, so an enumeration of one has nothing to
	// return -- which is a result a caller handles, unlike a failure.
	state.redirect_ntzw(mod, "QueryDirectoryObject",
		[](vcpu&, const std::uint64_t directory_handle, const addr_t buffer,
			const std::uint32_t length, const std::uint8_t return_single_entry,
			const std::uint8_t restart_scan, emu_object<std::uint32_t> context,
			emu_object<std::uint32_t> return_length) -> NTSTATUS
		{
			if (context)
				context.write(0);

			if (return_length)
				return_length.write(0);

			THREAD_LOG_WARN("NtQueryDirectoryObject(handle=0x{:X}, buffer=0x{:X}/{}, single={}, "
				"restart={}): the directory is empty",
				directory_handle, buffer, length, return_single_entry != 0, restart_scan != 0);

			return STATUS_NO_MORE_ENTRIES;
		});

	state.redirect_ntzw(mod, "OpenSymbolicLinkObject",
		[attribute_name, open_namespace_object](vcpu& cpu, emu_object<std::uint64_t> link_handle,
			const std::uint32_t desired_access,
			emu_object<_OBJECT_ATTRIBUTES> object_attributes) -> NTSTATUS
		{
			if (!link_handle)
				return STATUS_INVALID_PARAMETER;

			const auto name = attribute_name(cpu, object_attributes);

			auto host = std::make_shared<symbolic_link_host>();
			host->target = system32_dir_narrow;

			// Windows spells a device path without the trailing separator.
			if (host->target.size() > 1 && host->target.back() == '\\')
				host->target.pop_back();

			const auto handle = open_namespace_object(std::move(host), name, desired_access);

			if (!handle)
				return STATUS_INSUFFICIENT_RESOURCES;

			link_handle.write(handle);

			THREAD_LOG_WARN("NtOpenSymbolicLinkObject('{}') -> handle=0x{:X}: nothing here "
				"creates a link, so it points at System32", name, handle);

			return STATUS_SUCCESS;
		});

	state.redirect_ntzw(mod, "QuerySymbolicLinkObject",
		[st = &state](vcpu&, const std::uint64_t link_handle,
			emu_object<_UNICODE_STRING> link_target,
			emu_object<std::uint32_t> returned_length) -> NTSTATUS
		{
			const auto host =
				st->sys_proc->handle_table().get_object<symbolic_link_host>(link_handle);

			if (!host)
			{
				THREAD_LOG_WARN("NtQuerySymbolicLinkObject: handle 0x{:X} is not a symbolic link",
					link_handle);
				return STATUS_OBJECT_TYPE_MISMATCH;
			}

			const auto target = widen_string(host->target);
			const auto needed = static_cast<std::uint16_t>(target.size() * sizeof(char16_t));

			if (returned_length)
				returned_length.write(needed);

			if (!link_target)
				return STATUS_INVALID_PARAMETER;

			auto value = link_target.read();

			if (value.MaximumLength < needed || !value.Buffer)
				return STATUS_BUFFER_TOO_SMALL;

			link_target.space()->write_mem(guest_va(value.Buffer), target.data(), needed);

			value.Length = needed;
			link_target.write(value);

			THREAD_LOG_INFO("NtQuerySymbolicLinkObject(0x{:X}) -> '{}'", link_handle, host->target);

			return STATUS_SUCCESS;
		});

	state.redirect_ntzw(mod, "QueryObject",
		[st = &state](vcpu& cpu, const std::uint64_t handle, const std::uint32_t information_class,
			const addr_t object_information, const std::uint32_t length,
			emu_object<std::uint32_t> return_length) -> NTSTATUS
		{
			if (information_class == object_types_information)
				return STATUS_NOT_IMPLEMENTED;

			if (information_class > object_handle_flag_information)
				return STATUS_INVALID_INFO_CLASS;

			std::optional<object_kind> kind;
			addr_t body = 0;
			std::uint32_t access = 0x1FFFFF;
			std::uint8_t flags = 0;

			if (handle == ~std::uint64_t{0} || handle == ~std::uint64_t{1})
			{
				kind = object_kind{ handle == ~std::uint64_t{0} ? "Process" : "Thread", {}, 0x1FFFFF };
			}
			else
			{
				const auto entry = st->sys_proc->handle_table().lookup_handle(handle);

				if (!entry)
					return STATUS_INVALID_HANDLE;

				body = entry->body_addr;
				access = entry->access;
				flags = entry->flags;
				kind = describe_object(*st, body);
			}

			const auto name = kind ? kind->name : std::string();
			const auto type = kind ? kind->type : std::string();
			const auto wide = widen_string(information_class == object_type_information ? type : name);

			constexpr std::size_t type_header = 0x68;
			const std::size_t string_bytes = wide.size() * sizeof(char16_t);

			std::size_t needed = 0x38;

			if (information_class == object_name_information)
				needed = 16 + (string_bytes ? string_bytes + sizeof(char16_t) : 0);
			else if (information_class == object_type_information)
				needed = type_header + string_bytes + sizeof(char16_t);
			else if (information_class == object_handle_flag_information)
				needed = 2;

			if (return_length)
				return_length.write(static_cast<std::uint32_t>(needed));

			if (information_class == object_type_information && !kind)
				return STATUS_NOT_IMPLEMENTED;

			if (length < needed)
				return STATUS_INFO_LENGTH_MISMATCH;

			std::vector<std::uint8_t> out(needed);

			const auto put_string = [&](const std::size_t at)
			{
				put<std::uint16_t>(out, 0, static_cast<std::uint16_t>(string_bytes));
				put<std::uint16_t>(out, 2, static_cast<std::uint16_t>(string_bytes + sizeof(char16_t)));
				put<std::uint64_t>(out, 8, object_information + at);
				std::memcpy(out.data() + at, wide.data(), string_bytes);
			};

			if (information_class == object_basic_information)
			{
				const auto counts = st->objs.counts_of(body);

				put<std::uint32_t>(out, 0, (flags & win_handle_table::flag_inherit) ? 0x2 : 0);
				put<std::uint32_t>(out, 4, access);
				put<std::uint32_t>(out, 8, counts ? static_cast<std::uint32_t>(counts->handles) : 1);
				put<std::uint32_t>(out, 12, counts ? static_cast<std::uint32_t>(counts->pointers) : 1);
				put<std::uint32_t>(out, 0x20, static_cast<std::uint32_t>(
					16 + (name.empty() ? 0 : (name.size() + 1) * sizeof(char16_t))));
				put<std::uint32_t>(out, 0x24, static_cast<std::uint32_t>(
					type_header + (type.size() + 1) * sizeof(char16_t)));
			}
			else if (information_class == object_name_information)
			{
				if (string_bytes)
					put_string(16);
			}
			else if (information_class == object_type_information)
			{
				put_string(type_header);
				put<std::uint32_t>(out, 0x54, kind->valid_access);
			}
			else
			{
				out[0] = (flags & win_handle_table::flag_inherit) != 0;
				out[1] = (flags & win_handle_table::flag_protect_from_close) != 0;
			}

			cpu.curr_addr_space()->write_mem(object_information, out.data(), out.size());

			return STATUS_SUCCESS;
		});

	state.redirect_ntzw(mod, "SetInformationObject",
		[st = &state](vcpu& cpu, const std::uint64_t handle, const std::uint32_t information_class,
			const addr_t object_information, const std::uint32_t length) -> NTSTATUS
		{
			if (information_class != object_handle_flag_information)
				return STATUS_INVALID_INFO_CLASS;

			if (length < 2)
				return STATUS_INFO_LENGTH_MISMATCH;

			if (!st->sys_proc->handle_table().lookup_handle(handle))
				return STATUS_INVALID_HANDLE;

			auto& space = *cpu.curr_addr_space();

			const std::uint8_t flags =
				(space.read_mem<std::uint8_t>(object_information) ? win_handle_table::flag_inherit : 0)
				| (space.read_mem<std::uint8_t>(object_information + 1) ? win_handle_table::flag_protect_from_close : 0);

			st->sys_proc->handle_table().set_handle_flags(handle, flags);

			return STATUS_SUCCESS;
		});
}
