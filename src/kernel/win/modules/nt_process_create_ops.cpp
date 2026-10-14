#include "nt_process_create_ops.hpp"
#include "../win_kernel.hpp"
#include "../filesystem.hpp"
#include "../objects.hpp"
#include "../process.hpp"
#include "../ps_attributes.hpp"
#include "../status.hpp"
#include "../thread.hpp"
#include "../types.hpp"
#include "../../../util/log.hpp"
#include "../../../util/string.hpp"
#include <string>
#include <vector>

namespace
{

constexpr std::uint32_t ps_create_success = 6;
constexpr std::uint32_t parameters_normalized = 0x1;
constexpr std::uint32_t thread_create_suspended = 0x1;

std::u16string read_command_line(addr_space& space, const addr_t block)
{
	const auto params = emu_object<_RTL_USER_PROCESS_PARAMETERS>(space, block).read();
	const auto& text = params.CommandLine;

	if (!text.Length || !text.Buffer)
		return {};

	auto at = guest_va(text.Buffer);

	if (!(params.Flags & parameters_normalized))
		at += block;

	std::u16string out(text.Length / sizeof(char16_t), u'\0');
	space.read_mem(at, out.data(), text.Length);

	return out;
}

}

void modules::register_ntoskrnl_process_create_ops(win_kernel_state& state, proc_module& mod)
{
	auto* st = &state;

	state.redirect_ntzw(mod, "CreateUserProcess",
		[st](vcpu& cpu, emu_object<std::uint64_t> process_handle,
			emu_object<std::uint64_t> thread_handle, const std::uint32_t process_access,
			const std::uint32_t thread_access, emu_object<_OBJECT_ATTRIBUTES>,
			emu_object<_OBJECT_ATTRIBUTES>, const std::uint32_t, const std::uint32_t thread_flags,
			const addr_t process_parameters, const addr_t create_info,
			const addr_t attribute_list) -> NTSTATUS
		{
			auto* const emulator = st->emulator();

			if (!process_handle || !thread_handle || !attribute_list || !emulator)
				return STATUS_INVALID_PARAMETER;

			auto& space = *cpu.curr_addr_space();

			const auto total = space.read_mem<std::uint64_t>(attribute_list);
			std::vector<ps_attribute_t> attributes(total > 8 ? (total - 8) / sizeof(ps_attribute_t) : 0);

			if (!attributes.empty())
				space.read_mem(attribute_list + 8, attributes.data(),
					attributes.size() * sizeof(ps_attribute_t));

			std::u16string image;

			for (const auto& attr : attributes)
			{
				if (attr.attribute == ps_attribute_image_name)
				{
					image.resize(attr.size / sizeof(char16_t));
					space.read_mem(attr.value, image.data(), image.size() * sizeof(char16_t));
				}
			}

			auto path = narrow_wstring(image);

			if (path.starts_with("\\??\\"))
				path.erase(0, 4);

			if (path.empty() || !st->fs.exists(path))
				return STATUS_OBJECT_NAME_NOT_FOUND;

			const auto app = emulator->create_user_process(cpu,
				path.substr(path.find_last_of('\\') + 1), true, path);

			if (!app.proc || !app.thread)
				return STATUS_UNSUCCESSFUL;

			if (process_parameters)
				app.proc->set_process_parameters(widen_string(path),
					read_command_line(space, process_parameters));

			auto host = std::make_shared<process_object>();
			host->pid = app.proc->id();

			const std::uint8_t body[sizeof(addr_t)] = {};
			const auto process_body = st->objs.create_object(0, body, sizeof(body),
				std::move(host), prot_rw | prot_supervisor);

			const auto thread = std::static_pointer_cast<win_thread>(app.thread);

			if (!(thread_flags & thread_create_suspended))
				thread->resume();

			auto& handles = st->sys_proc->handle_table();
			const auto process_h = process_body ? handles.create_handle(process_body, process_access) : 0;
			const auto thread_h = handles.create_handle(thread->ethread().address(), thread_access);

			if (!process_h || !thread_h)
				return STATUS_INSUFFICIENT_RESOURCES;

			process_handle.write(process_h);
			thread_handle.write(thread_h);

			for (const auto& attr : attributes)
			{
				if (attr.attribute == ps_attribute_client_id && attr.value
					&& attr.size >= sizeof(_CLIENT_ID))
				{
					space.write_mem(attr.value, thread->client_id().read());

					if (attr.return_length)
						space.write_mem<std::uint64_t>(attr.return_length, sizeof(_CLIENT_ID));
				}
			}

			if (create_info)
			{
				space.write_mem<std::uint32_t>(create_info + 0x08, ps_create_success);
				space.write_mem<std::uint64_t>(create_info + 0x28, app.proc->process_parameters_address());
				space.write_mem<std::uint64_t>(create_info + 0x38, app.proc->peb().address());
			}

			THREAD_LOG_INFO("NtCreateUserProcess('{}') -> pid={}, tid={}", path, app.proc->id(),
				thread->id());

			return STATUS_SUCCESS;
		});
}
