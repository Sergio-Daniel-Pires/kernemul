#pragma once

struct win_kernel_state;
struct proc_module;

namespace modules
{
	void register_ntoskrnl_pipe_ops(win_kernel_state& state, proc_module& mod);
}
