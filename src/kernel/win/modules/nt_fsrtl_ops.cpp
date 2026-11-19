#include "nt_fsrtl_ops.hpp"
#include "../win_kernel.hpp"
#include "../../../util/log.hpp"

void modules::register_ntoskrnl_fsrtl_ops(win_kernel_state& state, proc_module& mod)
{
	// Auto-expand push locks keep the lock in guest memory; freeing one releases
	// the backing the allocator handed out.  The teardown path calls this from
	// ntoskrnl itself, so without a redirect the unknown target stops the cpu and
	// the driver's entry thread never returns a status (observed as
	// "unimplemented function ntoskrnl.exe!FsRtlFreeAePushLock").
	state.redirect(mod, "FsRtlFreeAePushLock",
		[](vcpu&, const addr_t lock)
		{
			THREAD_LOG_INFO("FsRtlFreeAePushLock(0x{:X})", lock);
		});
}
