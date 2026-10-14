#include "syscalls.hpp"
#include "../map.hpp"
#include "../process.hpp"
#include "../../util/log.hpp"

#if defined(KERNEMUL_ARCH_ARM64)
	#include "arm64_win.hpp"
#else
	#include "x86_win.hpp"
#endif

std::unique_ptr<win_syscall> make_win_syscall()
{
#if defined(KERNEMUL_ARCH_ARM64)
	return std::make_unique<arm64_win_syscall>();
#else
	return std::make_unique<x86_win_syscall>();
#endif
}

void win_syscall::add(const std::string_view name, const std::span<const std::uint8_t> image,
	const proc_module& impl)
{
	const auto* const img = reinterpret_cast<const pe::image*>(image.data());

	std::size_t numbered = 0;
	std::size_t unresolved = 0;

	for (const auto exp : img->exports())
	{
		// Nt and Zw are the same stub; a forwarded export has a name where its code would be.
		if (exp.forwarded || !exp.name.starts_with("Nt"))
			continue;

		const auto rva = exp.loc.rva();

		if (rva >= image.size())
			continue;

		const auto number = decode_id(image.subspan(rva));

		if (!number)
			continue;

		names_[*number] = std::string(exp.name);

		const auto handler = impl.find_symbol(exp.name);

		if (!handler)
		{
			++unresolved;
			continue;
		}

		ids_[*number] = *handler;
		++numbered;
	}

	LOG_INFO("syscalls: {} -> {}: {} numbered, {} with no symbol to point at",
		name, impl.name, numbered, unresolved);
}

std::string_view win_syscall::name(const std::uint32_t id) const
{
	const auto it = names_.find(id);

	return it != names_.end() ? std::string_view(it->second) : std::string_view();
}

std::optional<addr_t> win_syscall::find(const std::uint32_t id) const
{
	const auto it = ids_.find(id);

	if (it == ids_.end())
		return std::nullopt;

	return it->second;
}
