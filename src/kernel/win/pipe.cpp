#include "pipe.hpp"
#include "../../util/log.hpp"
#include "../../util/string.hpp"

#include <algorithm>

namespace
{
	constexpr std::string_view directory = "device/namedpipe";
	constexpr std::uint32_t file_open = 1;
	constexpr std::uint32_t file_create = 2;
	constexpr std::uint32_t unlimited = 0xFFFFFFFF;
	constexpr std::int64_t default_wait = 500000;
	constexpr std::size_t peek_header = 16;
	constexpr std::size_t wait_name_length = 8;
	constexpr std::size_t wait_timeout_specified = 12;
	constexpr std::size_t wait_name = 14;

	bool stale(const pending_request& request)
	{
		if (!request.synchronous)
			return false;

		const auto t = request.thread.lock();

		return !t || !t->io_pending();
	}

	bool current(const pipe_host& host)
	{
		return host.server || host.generation == host.instance->generation;
	}

	pipe_end& own(pipe_host& host)
	{
		return host.server ? host.instance->server : host.instance->client;
	}

	pipe_end& peer(pipe_host& host)
	{
		return host.server ? host.instance->client : host.instance->server;
	}

	std::vector<std::uint8_t> take(pipe_end& end, const std::uint32_t length, const bool message,
		NTSTATUS& status)
	{
		std::vector<std::uint8_t> out;

		while (out.size() < length && !end.inbox.empty())
		{
			auto& front = end.inbox.front();
			const auto n = std::min<std::size_t>(length - out.size(), front.size());

			out.insert(out.end(), front.begin(), front.begin() + static_cast<std::ptrdiff_t>(n));
			front.erase(front.begin(), front.begin() + static_cast<std::ptrdiff_t>(n));

			if (front.empty())
				end.inbox.pop_front();
			else if (message)
				status = STATUS_BUFFER_OVERFLOW;

			if (message)
				break;
		}

		return out;
	}

	std::shared_ptr<pipe_host> make_host(std::shared_ptr<pipe_instance> instance, const bool server,
		const std::uint32_t read_mode, const bool nonblocking, const std::uint32_t options)
	{
		auto host = std::make_shared<pipe_host>();

		host->path = std::string(directory) + "/" + instance->name->key;
		host->server = server;
		host->generation = instance->generation;
		host->message_read = read_mode == 1;
		host->nonblocking = nonblocking;
		host->synchronous = (options & 0x30) != 0;
		host->instance = std::move(instance);

		return host;
	}
}

std::optional<std::string> pipe_manager::pipe_key(std::string_view path)
{
	if (path.starts_with('/'))
		path.remove_prefix(1);

	if (path == directory || path == "pipe")
		return std::string();

	if (path.starts_with("device/namedpipe/"))
		return std::string(path.substr(directory.size() + 1));

	if (path.starts_with("pipe/"))
		return std::string(path.substr(5));

	return std::nullopt;
}

std::shared_ptr<pipe_host> pipe_manager::open_root(const std::uint32_t options)
{
	auto host = std::make_shared<pipe_host>();

	host->path = std::string(directory);
	host->synchronous = (options & 0x30) != 0;

	return host;
}

void pipe_manager::complete(vcpu& cpu, const pending_request& request, const NTSTATUS status,
	const std::uint64_t information, const std::span<const std::uint8_t> data)
{
	if (stale(request))
		return;

	if (request.space)
	{
		if (!data.empty() && request.buffer)
			request.space->write_mem(request.buffer, data.data(), data.size());

		if (request.io_status)
		{
			_IO_STATUS_BLOCK block{};
			block.Status = static_cast<std::int32_t>(status);
			block.Information = information;

			request.space->write_mem(request.io_status, &block, sizeof(block));
		}
	}

	signal_io_event(state_, cpu, request.event);

	if (const auto t = request.thread.lock())
		t->complete_io(status);
}

void pipe_manager::complete_all(vcpu& cpu, std::vector<pending_request>& queue, const NTSTATUS status)
{
	const auto requests = std::move(queue);
	queue.clear();

	for (const auto& request : requests)
		complete(cpu, request, status, 0);
}

void pipe_manager::service_readers(vcpu& cpu, pipe_end& end)
{
	while (!end.readers.empty() && !end.inbox.empty())
	{
		const auto request = end.readers.front();
		end.readers.erase(end.readers.begin());

		if (stale(request))
			continue;

		NTSTATUS status = STATUS_SUCCESS;
		const auto data = take(end, request.length, request.message_mode, status);

		complete(cpu, request, status, data.size(), data);
	}
}

pipe_io_result pipe_manager::block(vcpu& cpu, const pipe_host& host, pending_request request,
	std::vector<pending_request>& queue, const std::optional<std::int64_t> relative_timeout,
	const NTSTATUS timeout_status)
{
	request.synchronous = host.synchronous;
	request.space = cpu.curr_addr_space();

	std::erase_if(queue, stale);

	if (request.io_status)
	{
		_IO_STATUS_BLOCK block{};
		block.Status = static_cast<std::int32_t>(STATUS_PENDING);

		request.space->write_mem(request.io_status, &block, sizeof(block));
	}

	const auto t = std::dynamic_pointer_cast<win_thread>(cpu.thread());

	if (!request.synchronous || !t)
	{
		request.synchronous = false;
		queue.push_back(std::move(request));

		return { STATUS_PENDING, 0, true };
	}

	win_thread::wait_state wait{};
	wait.pending_io = true;
	wait.timeout_status = timeout_status;

	if (relative_timeout)
	{
		wait.timed = true;
		wait.deadline = static_cast<std::int64_t>(win_system_time()) + *relative_timeout;
	}

	t->begin_wait(std::move(wait));

	request.thread = t;
	queue.push_back(std::move(request));

	thread_scheduler::yield_current(cpu);

	return { STATUS_PENDING, 0, true };
}

pipe_io_result pipe_manager::create_server(vcpu& cpu, const pipe_create_params& params,
	std::shared_ptr<pipe_host>& out)
{
	if (params.key.empty() || params.type > 1 || params.read_mode > 1 || params.completion_mode > 1
		|| (params.read_mode == 1 && params.type == 0) || params.max_instances == 0)
	{
		return { STATUS_INVALID_PARAMETER, 0 };
	}

	std::scoped_lock lock(mtx_);

	auto& name = names_[params.key];
	const bool existed = name != nullptr;

	if (existed)
	{
		if (params.disposition == file_create || name->type != params.type
			|| name->max_instances != params.max_instances)
		{
			return { STATUS_ACCESS_DENIED, 0 };
		}

		if (name->max_instances != unlimited && name->instances.size() >= name->max_instances)
			return { STATUS_INSTANCE_NOT_AVAILABLE, 0 };
	}
	else
	{
		if (params.disposition == file_open)
		{
			names_.erase(params.key);
			return { STATUS_OBJECT_NAME_NOT_FOUND, 0 };
		}

		name = std::make_shared<pipe_name>();
		name->key = params.key;
		name->type = params.type;
		name->max_instances = params.max_instances;
		name->default_timeout = params.default_timeout;
	}

	auto instance = std::make_shared<pipe_instance>();
	instance->name = name;
	instance->server.open = true;
	name->instances.push_back(instance);

	out = make_host(instance, true, params.read_mode, params.completion_mode == 1, params.options);

	complete_all(cpu, name->waiters, STATUS_SUCCESS);

	THREAD_LOG_INFO("pipe '{}': instance {} created, {} pipe", name->key, name->instances.size(),
		name->type ? "message" : "byte");

	return { STATUS_SUCCESS, existed ? 1u : 2u };
}

pipe_io_result pipe_manager::open_client(vcpu& cpu, const std::string& key,
	const std::uint32_t options, std::shared_ptr<pipe_host>& out)
{
	std::scoped_lock lock(mtx_);

	const auto it = names_.find(key);

	if (it == names_.end())
		return { STATUS_OBJECT_NAME_NOT_FOUND, 0 };

	const auto& instances = it->second->instances;

	const auto chosen = std::ranges::find_if(instances,
		[](const auto& i) { return i->state == pipe_listening; });

	if (chosen == instances.end())
	{
		const bool disconnected = std::ranges::any_of(instances,
			[](const auto& i) { return i->state == pipe_disconnected; });

		return { disconnected ? STATUS_PIPE_NOT_AVAILABLE : STATUS_PIPE_BUSY, 0 };
	}

	const auto instance = *chosen;

	instance->state = pipe_connected;
	instance->generation++;
	instance->client.open = true;
	instance->client.inbox.clear();

	out = make_host(instance, false, 0, false, options);

	complete_all(cpu, instance->listeners, STATUS_SUCCESS);

	THREAD_LOG_INFO("pipe '{}': a client connected", key);

	return { STATUS_SUCCESS, 1 };
}

pipe_io_result pipe_manager::read_locked(vcpu& cpu, pipe_host& host, const pipe_request& request)
{
	auto& instance = *host.instance;
	auto& mine = own(host);

	if (!current(host) || (host.server && instance.state == pipe_disconnected))
		return { STATUS_PIPE_DISCONNECTED, 0 };

	if (host.server && instance.state == pipe_listening)
		return { STATUS_PIPE_LISTENING, 0 };

	const bool message = host.message_read && instance.name->type == 1;

	if (!mine.inbox.empty())
	{
		NTSTATUS status = STATUS_SUCCESS;
		const auto data = take(mine, request.length, message, status);

		if (!data.empty())
			cpu.curr_addr_space()->write_mem(request.buffer, data.data(), data.size());

		return { status, data.size() };
	}

	if (!request.length)
		return { STATUS_SUCCESS, 0 };

	if (host.server ? instance.state == pipe_closing : !instance.server.open)
		return { STATUS_PIPE_BROKEN, 0 };

	if (host.nonblocking)
		return { STATUS_PIPE_EMPTY, 0 };

	pending_request wait;
	wait.buffer = request.buffer;
	wait.length = request.length;
	wait.io_status = request.io_status;
	wait.event = request.event;
	wait.message_mode = message;

	return block(cpu, host, std::move(wait), mine.readers);
}

pipe_io_result pipe_manager::read(vcpu& cpu, pipe_host& host, const pipe_request& request)
{
	if (!host.instance)
		return { STATUS_INVALID_DEVICE_REQUEST, 0 };

	std::scoped_lock lock(mtx_);

	return read_locked(cpu, host, request);
}

pipe_io_result pipe_manager::write_locked(vcpu& cpu, pipe_host& host,
	const std::span<const std::uint8_t> data)
{
	auto& instance = *host.instance;

	if (!current(host) || (host.server && instance.state == pipe_disconnected))
		return { STATUS_PIPE_DISCONNECTED, 0 };

	if (host.server && instance.state == pipe_listening)
		return { STATUS_PIPE_LISTENING, 0 };

	if (host.server ? instance.state == pipe_closing : !instance.server.open)
		return { STATUS_PIPE_CLOSING, 0 };

	auto& other = peer(host);

	if (!data.empty() || instance.name->type == 1)
		other.inbox.emplace_back(data.begin(), data.end());

	service_readers(cpu, other);

	return { STATUS_SUCCESS, data.size() };
}

pipe_io_result pipe_manager::write(vcpu& cpu, pipe_host& host,
	const std::span<const std::uint8_t> data)
{
	if (!host.instance)
		return { STATUS_INVALID_DEVICE_REQUEST, 0 };

	std::scoped_lock lock(mtx_);

	return write_locked(cpu, host, data);
}

pipe_io_result pipe_manager::peek(vcpu& cpu, pipe_host& host, const addr_t output_buffer,
	const std::uint32_t output_length)
{
	if (!current(host))
		return { STATUS_PIPE_DISCONNECTED, 0 };

	if (output_length < peek_header)
		return { STATUS_INFO_LENGTH_MISMATCH, 0 };

	auto& inbox = own(host).inbox;
	const bool message = host.instance->name->type == 1;
	const auto room = output_length - peek_header;

	std::uint32_t available = 0;
	std::vector<std::uint8_t> data;

	for (const auto& chunk : inbox)
	{
		available += static_cast<std::uint32_t>(chunk.size());

		if (message && !data.empty())
			continue;

		const auto n = std::min<std::size_t>(room - data.size(), chunk.size());
		data.insert(data.end(), chunk.begin(), chunk.begin() + static_cast<std::ptrdiff_t>(n));
	}

	const auto first = message && !inbox.empty() ? inbox.front().size() : available;

	const std::uint32_t header[4] = {
		host.instance->state,
		available,
		message ? static_cast<std::uint32_t>(inbox.size()) : 0,
		message ? static_cast<std::uint32_t>(first) : 0,
	};

	auto& space = *cpu.curr_addr_space();

	space.write_mem(output_buffer, header, sizeof(header));

	if (!data.empty())
		space.write_mem(output_buffer + peek_header, data.data(), data.size());

	return { data.size() < first ? STATUS_BUFFER_OVERFLOW : STATUS_SUCCESS, peek_header + data.size() };
}

pipe_io_result pipe_manager::wait_for_instance(vcpu& cpu, pipe_host& host, const addr_t input_buffer,
	const std::uint32_t input_length, const pipe_request& request)
{
	if (input_length < wait_name)
		return { STATUS_INVALID_PARAMETER, 0 };

	auto& space = *cpu.curr_addr_space();

	const auto timeout = space.read_mem<std::int64_t>(input_buffer);
	const auto name_length = space.read_mem<std::uint32_t>(input_buffer + wait_name_length);
	const bool specified = space.read_mem<std::uint8_t>(input_buffer + wait_timeout_specified) != 0;

	if (name_length % sizeof(char16_t) || wait_name + name_length > input_length)
		return { STATUS_INVALID_PARAMETER, 0 };

	std::u16string wide(name_length / sizeof(char16_t), u'\0');
	space.read_mem(input_buffer + wait_name, wide.data(), name_length);

	auto key = win_filesystem::normalize(narrow_wstring(wide));
	key.erase(0, key.find_first_not_of('/'));

	const auto it = names_.find(key);

	if (it == names_.end())
		return { STATUS_OBJECT_NAME_NOT_FOUND, 0 };

	const auto& name = it->second;

	if (std::ranges::any_of(name->instances, [](const auto& i) { return i->state == pipe_listening; }))
		return { STATUS_SUCCESS, 0 };

	const auto wanted = specified ? timeout : name->default_timeout;
	const auto ticks = wanted < 0 ? -wanted
		: wanted > 0 ? std::max<std::int64_t>(0, wanted - static_cast<std::int64_t>(win_system_time()))
		: default_wait;

	pending_request wait;
	wait.io_status = request.io_status;
	wait.event = request.event;

	return block(cpu, host, std::move(wait), name->waiters, ticks, STATUS_IO_TIMEOUT);
}

pipe_io_result pipe_manager::control(vcpu& cpu, pipe_host& host, const std::uint32_t code,
	const addr_t input_buffer, const std::uint32_t input_length, const addr_t output_buffer,
	const std::uint32_t output_length, const pipe_request& request)
{
	std::scoped_lock lock(mtx_);

	if (code == fsctl_pipe_wait)
		return wait_for_instance(cpu, host, input_buffer, input_length, request);

	if (!host.instance)
		return { STATUS_NOT_SUPPORTED, 0 };

	auto& instance = *host.instance;

	if (code == fsctl_pipe_peek)
		return peek(cpu, host, output_buffer, output_length);

	if (code == fsctl_pipe_transceive)
	{
		std::vector<std::uint8_t> message(input_length);

		if (input_length)
			cpu.curr_addr_space()->read_mem(input_buffer, message.data(), message.size());

		const auto sent = write_locked(cpu, host, message);

		if (sent.status != STATUS_SUCCESS)
			return sent;

		return read_locked(cpu, host, { output_buffer, output_length, request.io_status, request.event });
	}

	if (code != fsctl_pipe_listen && code != fsctl_pipe_disconnect && code != fsctl_pipe_impersonate)
	{
		THREAD_LOG_WARN("pipe '{}': control code 0x{:X} is not handled", host.path, code);
		return { STATUS_NOT_SUPPORTED, 0 };
	}

	if (!host.server)
		return { STATUS_ILLEGAL_FUNCTION, 0 };

	if (code == fsctl_pipe_impersonate)
		return { instance.state == pipe_connected ? STATUS_SUCCESS : STATUS_PIPE_DISCONNECTED, 0 };

	if (code == fsctl_pipe_disconnect)
	{
		if (instance.state == pipe_disconnected)
			return { STATUS_PIPE_DISCONNECTED, 0 };

		complete_all(cpu, instance.listeners, STATUS_CANCELLED);

		instance.generation++;
		instance.client.open = false;
		instance.client.inbox.clear();
		instance.server.inbox.clear();
		instance.state = pipe_disconnected;

		complete_all(cpu, instance.client.readers, STATUS_PIPE_DISCONNECTED);
		complete_all(cpu, instance.server.readers, STATUS_PIPE_DISCONNECTED);

		return { STATUS_SUCCESS, 0 };
	}

	if (instance.state == pipe_connected)
		return { STATUS_PIPE_CONNECTED, 0 };

	if (instance.state == pipe_closing)
		return { STATUS_PIPE_CLOSING, 0 };

	if (instance.state == pipe_disconnected)
	{
		instance.state = pipe_listening;
		complete_all(cpu, instance.name->waiters, STATUS_SUCCESS);
	}

	if (host.nonblocking)
		return { STATUS_PIPE_LISTENING, 0 };

	pending_request wait;
	wait.io_status = request.io_status;
	wait.event = request.event;

	return block(cpu, host, std::move(wait), instance.listeners);
}

pipe_io_result pipe_manager::query_information(vcpu& cpu, pipe_host& host,
	const std::uint32_t information_class, const addr_t buffer, const std::uint32_t length)
{
	if (!host.instance)
		return { STATUS_INVALID_DEVICE_REQUEST, 0 };

	std::scoped_lock lock(mtx_);

	auto& space = *cpu.curr_addr_space();
	const auto& name = *host.instance->name;

	if (information_class == file_pipe_information_class)
	{
		if (length < 8)
			return { STATUS_INFO_LENGTH_MISMATCH, 0 };

		const std::uint32_t info[2] = { host.message_read, host.nonblocking };
		space.write_mem(buffer, info, sizeof(info));

		return { STATUS_SUCCESS, sizeof(info) };
	}

	if (information_class == file_pipe_local_information_class)
	{
		if (length < 40)
			return { STATUS_INFO_LENGTH_MISMATCH, 0 };

		std::uint32_t waiting = 0;

		for (const auto& chunk : own(host).inbox)
			waiting += static_cast<std::uint32_t>(chunk.size());

		const std::uint32_t info[10] = {
			name.type, 2, name.max_instances, static_cast<std::uint32_t>(name.instances.size()),
			4096, waiting, 4096, 4096, host.instance->state, host.server,
		};
		space.write_mem(buffer, info, sizeof(info));

		return { STATUS_SUCCESS, sizeof(info) };
	}

	return { STATUS_INVALID_INFO_CLASS, 0 };
}

pipe_io_result pipe_manager::set_information(vcpu& cpu, pipe_host& host,
	const std::uint32_t information_class, const addr_t buffer, const std::uint32_t length)
{
	if (!host.instance)
		return { STATUS_INVALID_DEVICE_REQUEST, 0 };

	if (information_class != file_pipe_information_class)
		return { STATUS_INVALID_INFO_CLASS, 0 };

	if (length < 8)
		return { STATUS_INFO_LENGTH_MISMATCH, 0 };

	std::uint32_t info[2]{};
	cpu.curr_addr_space()->read_mem(buffer, info, sizeof(info));

	std::scoped_lock lock(mtx_);

	if (info[0] > 1 || info[1] > 1 || (info[0] == 1 && host.instance->name->type == 0))
		return { STATUS_INVALID_PARAMETER, 0 };

	host.message_read = info[0] == 1;
	host.nonblocking = info[1] == 1;

	return { STATUS_SUCCESS, 0 };
}

void pipe_manager::close_end(vcpu& cpu, pipe_host& host)
{
	if (!host.instance)
		return;

	std::scoped_lock lock(mtx_);

	const auto instance = host.instance;

	if (host.server)
	{
		const auto name = instance->name;

		instance->server.open = false;
		instance->server.readers.clear();
		instance->listeners.clear();

		complete_all(cpu, instance->client.readers, STATUS_PIPE_BROKEN);

		if (instance->state == pipe_connected)
			instance->state = pipe_closing;

		std::erase(name->instances, instance);

		if (name->instances.empty())
		{
			complete_all(cpu, name->waiters, STATUS_SUCCESS);
			names_.erase(name->key);
		}

		return;
	}

	if (host.generation != instance->generation)
		return;

	instance->client.open = false;
	instance->client.readers.clear();

	complete_all(cpu, instance->server.readers, STATUS_PIPE_BROKEN);

	if (instance->state == pipe_connected)
		instance->state = pipe_closing;
}
