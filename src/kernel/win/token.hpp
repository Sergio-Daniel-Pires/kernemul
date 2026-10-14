#pragma once
#include "security.hpp"
#include "status.hpp"
#include "win_obj_manager.hpp"

#include <cstdint>
#include <mutex>
#include <span>
#include <vector>

namespace win::sec
{

enum token_information_class : std::uint32_t
{
	token_user                   = 1,
	token_groups                 = 2,
	token_privileges             = 3,
	token_owner                  = 4,
	token_primary_group          = 5,
	token_default_dacl           = 6,
	token_source                 = 7,
	token_type_class             = 8,
	token_impersonation_level    = 9,
	token_statistics             = 10,
	token_restricted_sids        = 11,
	token_session_id             = 12,
	token_sandbox_inert          = 15,
	token_elevation_type         = 18,
	token_elevation              = 20,
	token_has_restrictions       = 21,
	token_virtualization_allowed = 23,
	token_virtualization_enabled = 24,
	token_integrity_level        = 25,
	token_ui_access              = 26,
	token_mandatory_policy       = 27,
	token_logon_sid              = 28,
	token_is_app_container       = 29,
	token_is_restricted          = 40,
};

constexpr std::uint32_t token_primary       = 1;
constexpr std::uint32_t token_impersonation = 2;

constexpr std::uint32_t security_anonymous      = 0;
constexpr std::uint32_t security_identification = 1;
constexpr std::uint32_t security_impersonation  = 2;

constexpr std::uint32_t token_duplicate         = 0x0002;
constexpr std::uint32_t token_impersonate       = 0x0004;
constexpr std::uint32_t token_query             = 0x0008;
constexpr std::uint32_t token_query_source      = 0x0010;
constexpr std::uint32_t token_adjust_privileges = 0x0020;
constexpr std::uint32_t token_adjust_groups     = 0x0040;
constexpr std::uint32_t token_adjust_default    = 0x0080;
constexpr std::uint32_t token_all_access        = 0x000F01FF;

constexpr std::uint32_t se_privilege_enabled         = 0x00000002;
constexpr std::uint32_t se_privilege_removed         = 0x00000004;
constexpr std::uint32_t se_privilege_used_for_access = 0x80000000;

constexpr std::uint32_t privilege_set_all_necessary = 1;

struct group_entry
{
	sid_t sid;
	std::uint32_t attributes = 0;
};

struct privilege_entry
{
	std::uint32_t luid = 0;
	std::uint32_t attributes = 0;
};

struct token_data
{
	sid_t user;
	std::vector<group_entry> groups;
	std::vector<privilege_entry> privileges;
	sid_t owner;
	sid_t primary_group;
	bytes default_dacl;

	std::uint32_t type = token_primary;
	std::uint32_t impersonation_level = security_anonymous;
	std::uint64_t token_id = 0;
};

struct token_host final : win_object
{
	explicit token_host(token_data d) : data(std::move(d)) {}

	mutable std::mutex mtx;
	token_data data;
};

[[nodiscard]] const generic_mapping& token_generic_mapping();

[[nodiscard]] std::uint32_t map_token_access(std::uint32_t desired);

[[nodiscard]] token_data make_default_token(std::uint64_t token_id);

[[nodiscard]] token_data duplicate_token(const token_data& source, std::uint32_t type,
	std::uint32_t impersonation_level, bool effective_only, std::uint64_t token_id);

[[nodiscard]] NTSTATUS query_token_information(const token_data& token,
	std::uint32_t information_class, addr_t base, bytes& out);

[[nodiscard]] bytes encode_privileges(std::span<const privilege_entry> privileges);

[[nodiscard]] NTSTATUS adjust_privileges(token_data& token, bool disable_all,
	std::span<const privilege_entry> requested, std::vector<privilege_entry>& previous);

[[nodiscard]] NTSTATUS check_access(const descriptor& sd, const token_data& token,
	std::uint32_t desired, const generic_mapping& mapping, std::uint32_t& granted);

}
