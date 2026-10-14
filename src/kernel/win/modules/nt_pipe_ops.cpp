#include "nt_pipe_ops.hpp"
#include "../win_kernel.hpp"
#include "../file_common.hpp"
#include "../pipe.hpp"
#include "../status.hpp"
#include "../types.hpp"

#include <atomic>
#include <format>

void modules::register_ntoskrnl_pipe_ops(win_kernel_state& state, proc_module& mod)
{
	state.pipes = std::make_shared<pipe_manager>(state);

	auto* st = &state;

	state.redirect_ntzw(mod, "CreateNamedPipeFile",
		[st](vcpu& cpu, emu_object<std::uint64_t> file_handle, const std::uint32_t desired_access,
			emu_object<_OBJECT_ATTRIBUTES> object_attributes,
			emu_object<_IO_STATUS_BLOCK> io_status_block, [[maybe_unused]] const std::uint32_t share_access,
			const std::uint32_t create_disposition, const std::uint32_t create_options,
			const std::uint32_t named_pipe_type, const std::uint32_t read_mode,
			const std::uint32_t completion_mode, const std::uint32_t maximum_instances,
			[[maybe_unused]] const std::uint32_t inbound_quota,
			[[maybe_unused]] const std::uint32_t outbound_quota,
			emu_object<std::int64_t> default_timeout) -> NTSTATUS
		{
			static std::atomic<std::uint64_t> next_unnamed{1};

			if (!file_handle)
				return STATUS_INVALID_PARAMETER;

			file_handle.write(0);

			const auto path = object_path(*st, cpu, object_attributes);
			auto key = pipe_manager::pipe_key(path);

			if (key && key->empty() && object_attributes
				&& guest_va(object_attributes.field(&_OBJECT_ATTRIBUTES::RootDirectory).read()))
			{
				key = std::format("\x01unnamed.{}", next_unnamed.fetch_add(1));
			}

			if (!key || key->empty())
				return finish(io_status_block, STATUS_OBJECT_NAME_INVALID);

			std::shared_ptr<pipe_host> host;
			const auto result = st->pipes->create_server(cpu, {
				.key = *key,
				.disposition = create_disposition,
				.options = create_options,
				.type = named_pipe_type,
				.read_mode = read_mode,
				.completion_mode = completion_mode,
				.max_instances = maximum_instances,
				.default_timeout = default_timeout ? default_timeout.read() : 0,
			}, host);

			if (result.status == STATUS_SUCCESS)
				file_handle.write(make_file_handle(*st, std::move(host), file_handle_access(desired_access)));

			return finish(io_status_block, result.status, result.information);
		});
}
