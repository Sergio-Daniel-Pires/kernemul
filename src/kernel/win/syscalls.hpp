#pragma once
#include "../../emu/defs.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>

class vcpu;
struct proc_module;

// Numbers out of the usermode stubs, addresses out of ntoskrnl; win32k's numbers have bit 12 set.
struct win_syscall
{
	virtual ~win_syscall() = default;

	[[nodiscard]] virtual std::uint32_t id(vcpu& cpu) const = 0;

	[[nodiscard]] virtual std::optional<std::uint32_t> decode_id(
		std::span<const std::uint8_t> stub) const = 0;

	// Names `impl` has no symbol for are counted rather than reported.
	void add(std::string_view name, std::span<const std::uint8_t> image,
		const proc_module& impl);

	[[nodiscard]] std::optional<addr_t> find(std::uint32_t id) const;

	[[nodiscard]] std::size_t size() const noexcept { return ids_.size(); }

	[[nodiscard]] std::string_view name(std::uint32_t id) const;

private:
	std::unordered_map<std::uint32_t, addr_t> ids_;
	std::unordered_map<std::uint32_t, std::string> names_;
};

// Null where there is no implementation yet.
std::unique_ptr<win_syscall> make_win_syscall();
