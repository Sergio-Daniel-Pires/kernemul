#pragma once
#include "file_common.hpp"
#include "thread.hpp"

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

enum pipe_state : std::uint32_t
{
	pipe_disconnected = 1,
	pipe_listening,
	pipe_connected,
	pipe_closing,
};

constexpr std::uint32_t file_device_named_pipe = 0x11;

constexpr std::uint32_t fsctl_pipe_disconnect = 0x110004;
constexpr std::uint32_t fsctl_pipe_listen = 0x110008;
constexpr std::uint32_t fsctl_pipe_peek = 0x11400C;
constexpr std::uint32_t fsctl_pipe_transceive = 0x11C017;
constexpr std::uint32_t fsctl_pipe_wait = 0x110018;
constexpr std::uint32_t fsctl_pipe_impersonate = 0x11001C;

constexpr std::uint32_t file_pipe_information_class = 23;
constexpr std::uint32_t file_pipe_local_information_class = 24;

struct pending_request
{
	std::weak_ptr<win_thread> thread;
	std::shared_ptr<addr_space> space;
	addr_t buffer = 0;
	std::uint32_t length = 0;
	addr_t io_status = 0;
	std::uint64_t event = 0;
	bool message_mode = false;
	bool synchronous = false;
};

struct pipe_end
{
	std::deque<std::vector<std::uint8_t>> inbox;
	std::vector<pending_request> readers;
	bool open = false;
};

struct pipe_name;

struct pipe_instance
{
	std::shared_ptr<pipe_name> name;
	pipe_state state = pipe_listening;
	std::uint64_t generation = 1;
	pipe_end server;
	pipe_end client;
	std::vector<pending_request> listeners;
};

struct pipe_name
{
	std::string key;
	std::uint32_t type = 0;
	std::uint32_t max_instances = 1;
	std::int64_t default_timeout = 0;
	std::vector<std::shared_ptr<pipe_instance>> instances;
	std::vector<pending_request> waiters;
};

struct pipe_io_result
{
	NTSTATUS status = STATUS_SUCCESS;
	std::uint64_t information = 0;
	bool pending = false;
};

struct pipe_request
{
	addr_t buffer = 0;
	std::uint32_t length = 0;
	addr_t io_status = 0;
	std::uint64_t event = 0;
};

struct pipe_create_params
{
	std::string key;
	std::uint32_t disposition = 0;
	std::uint32_t options = 0;
	std::uint32_t type = 0;
	std::uint32_t read_mode = 0;
	std::uint32_t completion_mode = 0;
	std::uint32_t max_instances = 0;
	std::int64_t default_timeout = 0;
};

class pipe_manager
{
public:
	explicit pipe_manager(win_kernel_state& state) : state_(state) {}

	[[nodiscard]] static std::optional<std::string> pipe_key(std::string_view normalized_path);

	[[nodiscard]] static std::shared_ptr<pipe_host> open_root(std::uint32_t options);

	pipe_io_result create_server(vcpu& cpu, const pipe_create_params& params,
		std::shared_ptr<pipe_host>& out);

	pipe_io_result open_client(vcpu& cpu, const std::string& key, std::uint32_t options,
		std::shared_ptr<pipe_host>& out);

	pipe_io_result read(vcpu& cpu, pipe_host& host, const pipe_request& request);
	pipe_io_result write(vcpu& cpu, pipe_host& host, std::span<const std::uint8_t> data);

	pipe_io_result control(vcpu& cpu, pipe_host& host, std::uint32_t code, addr_t input_buffer,
		std::uint32_t input_length, addr_t output_buffer, std::uint32_t output_length,
		const pipe_request& request);

	pipe_io_result query_information(vcpu& cpu, pipe_host& host, std::uint32_t information_class,
		addr_t buffer, std::uint32_t length);

	pipe_io_result set_information(vcpu& cpu, pipe_host& host, std::uint32_t information_class,
		addr_t buffer, std::uint32_t length);

	void close_end(vcpu& cpu, pipe_host& host);

private:
	void complete(vcpu& cpu, const pending_request& request, NTSTATUS status,
		std::uint64_t information, std::span<const std::uint8_t> data = {});

	void complete_all(vcpu& cpu, std::vector<pending_request>& queue, NTSTATUS status);
	void service_readers(vcpu& cpu, pipe_end& end);

	pipe_io_result block(vcpu& cpu, const pipe_host& host, pending_request request,
		std::vector<pending_request>& queue, std::optional<std::int64_t> relative_timeout = {},
		NTSTATUS timeout_status = STATUS_TIMEOUT);

	pipe_io_result read_locked(vcpu& cpu, pipe_host& host, const pipe_request& request);
	pipe_io_result write_locked(vcpu& cpu, pipe_host& host, std::span<const std::uint8_t> data);
	pipe_io_result peek(vcpu& cpu, pipe_host& host, addr_t output_buffer, std::uint32_t output_length);
	pipe_io_result wait_for_instance(vcpu& cpu, pipe_host& host, addr_t input_buffer,
		std::uint32_t input_length, const pipe_request& request);

	win_kernel_state& state_;
	std::mutex mtx_;
	std::unordered_map<std::string, std::shared_ptr<pipe_name>> names_;
};
