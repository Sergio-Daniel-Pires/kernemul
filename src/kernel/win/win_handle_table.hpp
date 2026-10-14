#pragma once
#include "win_obj_manager.hpp"

#include <cstdint>
#include <shared_mutex>
#include <optional>
#include <unordered_map>

class win_handle_table
{
public:
	using handle_t = std::uint64_t;
	using access_t = std::uint32_t;

	static constexpr std::uint8_t flag_inherit = 0x1;
	static constexpr std::uint8_t flag_protect_from_close = 0x2;

	struct handle_entry
	{
		addr_t body_addr;
		access_t access;
		std::uint8_t flags = 0;
	};

	explicit win_handle_table(win_obj_manager& objs);

	handle_t create_handle(addr_t body_addr, access_t access, std::uint8_t flags = 0);
	bool close_handle(handle_t handle);
	[[nodiscard]] std::optional<handle_entry> lookup_handle(handle_t handle) const;

	bool set_handle_flags(handle_t handle, std::uint8_t flags);
	[[nodiscard]] win_obj_manager& objs() { return objs_; }

	template <typename T>
	[[nodiscard]] std::shared_ptr<T> get_object(handle_t handle) const
	{
		auto entry = lookup_handle(handle);
		if (!entry) return {};
		return objs_.get_object<T>(entry->body_addr);
	}

private:
	handle_t alloc_handle();

	win_obj_manager& objs_;
	// Looked up on every call that takes a handle, written only on open/close.
	mutable std::shared_mutex mtx_;
	std::unordered_map<handle_t, handle_entry> handles_;
	handle_t next_handle_ = 4;
};
