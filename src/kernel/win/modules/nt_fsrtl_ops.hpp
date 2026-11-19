#pragma once
#include "../win_kernel.hpp"

namespace modules
{
	void register_ntoskrnl_fsrtl_ops(win_kernel_state& state, proc_module& mod);
}
