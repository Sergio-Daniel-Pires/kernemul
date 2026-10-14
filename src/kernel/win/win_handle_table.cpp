#include "win_handle_table.hpp"
#include "../../util/log.hpp"

win_handle_table::win_handle_table(win_obj_manager& objs)
	:	objs_(objs) {}

win_handle_table::handle_t win_handle_table::create_handle(addr_t body_addr, access_t access,
	const std::uint8_t flags)
{
	if (!objs_.has_object(body_addr))
	{
		LOG_ERR("win_handle_table: create_handle for unregistered object 0x{:X}", body_addr);
		return 0;
	}

	const auto h = alloc_handle();

	{
		std::unique_lock lock(mtx_);
		handles_[h] = { body_addr, access, flags };
	}

	objs_.increment_handle_count(body_addr);
	return h;
}

bool win_handle_table::close_handle(handle_t handle)
{
	addr_t body_addr;

	{
		std::unique_lock lock(mtx_);
		auto it = handles_.find(handle);
		if (it == handles_.end()) return false;
		body_addr = it->second.body_addr;
		handles_.erase(it);
	}

	objs_.decrement_handle_count(body_addr);
	return true;
}

std::optional<win_handle_table::handle_entry> win_handle_table::lookup_handle(handle_t handle) const
{
	std::shared_lock lock(mtx_);
	auto it = handles_.find(handle);
	if (it == handles_.end()) return std::nullopt;
	return it->second;
}

bool win_handle_table::set_handle_flags(const handle_t handle, const std::uint8_t flags)
{
	std::unique_lock lock(mtx_);
	auto it = handles_.find(handle);
	if (it == handles_.end()) return false;
	it->second.flags = flags;
	return true;
}

win_handle_table::handle_t win_handle_table::alloc_handle()
{
	std::unique_lock lock(mtx_);
	auto h = next_handle_;
	next_handle_ += 4;
	return h;
}
