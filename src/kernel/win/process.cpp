#include "process.hpp"
#include "thread.hpp"
#include "win_kernel.hpp"
#include "../thread_scheduler.hpp"
#include "../../util/log.hpp"

win_kernel_proc::win_kernel_proc(id_type id, win_kernel_state& kernel, std::shared_ptr<struct addr_space> space)
	:	windows_process(id, std::move(space), kernel.objs, kernel.fs), kernel_(kernel) {}

addr_t windows_process::find_symbol(const std::string_view mod_name,
	const std::string_view sym) const
{
	if (const auto mod = find_module(mod_name))
	{
		if (const auto addr = mod->find_symbol(sym))
			return *addr;
	}

	LOG_ERR("{}!{} not found", mod_name, sym);
	return 0;
}

std::shared_ptr<thread> windows_process::create_suspended_thread(vcpu& cpu,
	const addr_t start_addr, const std::span<const std::uint64_t> args,
	const std::size_t stack_size)
{
	const auto id = static_cast<thread_id_type>(objs_.allocate_id());
	const auto size = thread_stack_size(stack_size);

	const addr_t stack_base = addr_space_->alloc(size, prot_rw | prot_supervisor);
	auto self = std::static_pointer_cast<windows_process>(shared_from_this());
	auto t = std::make_shared<win_kernel_thread>(
		id, std::move(self), start_addr, stack_base, size, cpu);

	t->set_emulator(emulator_);
	setup_ethread(t, start_addr);

	{
		std::unique_lock lock(thread_mtx_);
		threads_[t->id()] = t;
	}

	// On the queue, but no cpu takes it until it is started.
	scheduler_->enqueue(cpu, t, args);
	return t;
}

std::shared_ptr<thread> win_user_proc::create_suspended_thread(vcpu& cpu,
	const addr_t start_addr, const std::span<const std::uint64_t> args,
	const std::size_t stack_size)
{
	const auto id = static_cast<thread_id_type>(objs_.allocate_id());
	const auto size = thread_stack_size(stack_size);

	const addr_t stack_base = mem_.alloc(size, prot_rw);
	auto self = std::static_pointer_cast<windows_process>(shared_from_this());
	auto t = std::make_shared<win_user_thread>(
		id, std::move(self), mem_, start_addr, stack_base, size, cpu);

	t->set_emulator(emulator_);
	setup_ethread(t, start_addr);

	emulator_->init_thread_teb(*t, cpu, t->teb().address());

	{
		std::unique_lock lock(thread_mtx_);
		threads_[t->id()] = t;
	}

	scheduler_->enqueue(cpu, t, args);
	return t;
}

void windows_process::setup_ethread(const std::shared_ptr<win_thread>& t, const addr_t start_addr)
{
	if (!emulator_)
		return;

	auto& space = *emulator_->emu().default_addr_space();

	ethread_params p{};
	p.eprocess = eprocess_.address();
	p.start_addr = start_addr;
	p.teb = t->teb().address();
	p.stack_limit = t->stack_limit();
	p.stack_base = t->stack_base();
	p.process_id = id_;
	p.thread_id = t->id();
	p.system_thread = t->is_system_thread();

	const auto et = make_default_ethread(p);

	auto obj = emu_object<_ETHREAD>(space,
		objs_.create_object(0, &et, sizeof(et), std::make_shared<thread_object>(t),
			prot_rw | prot_supervisor));

	t->set_ethread(obj);

	if (!eprocess_)
		return;

	auto& espace = *eprocess_.space();

	std::scoped_lock lock(emulator_->kernel().list_mtx_);

	kprocess_thread_list(espace, eprocess_.address()).push_back(obj);
	eprocess_thread_list(espace, eprocess_.address()).push_back(obj);

	auto active = eprocess_.field(&_EPROCESS::ActiveThreads);
	active.write(active.read() + 1);
}

void windows_process::destroy_ethread(const win_thread& t)
{
	const auto& et = t.ethread();

	if (!et || !emulator_)
		return;

	set_thread_state(et, Terminated, false);
	set_thread_exit_time(et, win_system_time());
	set_thread_exit_status(et, t.exit_status());

	win::set_state_at(*et.space(), et.address(), 1);
	wake_waiters(*et.space(), et.address());

	if (!eprocess_)
		return;

	auto& espace = *eprocess_.space();

	std::scoped_lock lock(emulator_->kernel().list_mtx_);

	kprocess_thread_list(espace, eprocess_.address()).remove(et.address());
	eprocess_thread_list(espace, eprocess_.address()).remove(et.address());

	auto active = eprocess_.field(&_EPROCESS::ActiveThreads);

	if (const auto count = active.read(); count > 0)
		active.write(count - 1);
}

void windows_process::terminate_thread(const thread_id_type id)
{
	// Nothing frees the ETHREAD, so it is only unlinked.
	if (const auto t = std::dynamic_pointer_cast<win_thread>(find_thread(id)))
		destroy_ethread(*t);

	process::terminate_thread(id);
}

std::shared_ptr<win_file> windows_process::open_system_image(const std::string_view name) const
{
	if (auto file = fs_.open(std::string(system32_dir_narrow) + std::string(name)))
		return file;

	return fs_.open(std::string(system32_dir_narrow) + "drivers/" + std::string(name));
}

// There is no loader in the guest behind a driver, so the imports are resolved here.
std::shared_ptr<proc_module> win_kernel_proc::load_module(const std::string_view name,
	const bool supervisor)
{
	const auto file = open_system_image(name);

	if (!file)
		return nullptr;

	return krnl::map_img(*this, name, file->data(), supervisor);
}

// A user image resolves its own; doing it here too would map every module a second time.
std::shared_ptr<proc_module> win_user_proc::load_module(const std::string_view name,
	const bool supervisor)
{
	auto file = fs_.open(current_dir_ + std::string(name));

	if (!file)
		file = open_system_image(name);

	// Last, so nothing in the root can stand in front of a real system module. A process is
	// created from a bare name, which leaves it no directory of its own to look in, and the
	// guest filesystem is loaded from a host directory whose root is where an image dropped in
	// to be run actually lands -- so without this the only place a user image can be started
	// from is System32, and a copy left in the root is loaded and then reported missing.
	if (!file)
		file = fs_.open(std::string(root_dir_narrow) + std::string(name));

	if (!file)
		return nullptr;

	return krnl::map_img(*this, name, file->data(), supervisor, true);
}

void win_user_proc::module_add_cb(proc_module& mod)
{
	mem_.register_image(mod.addr, mod.size);

	const bool image = is_process_image(mod);

	ldr_.add_module(mem_, mod.addr, mod.entry_point, mod.size, mod.name,
		!image, image ? ldr_module_list::image_flags : ldr_module_list::dll_flags);
}

void win_kernel_proc::module_add_cb(proc_module& mod)
{
	if (!kernel_.loaded_module_list.address())
		return;

	auto& space = *addr_space_;

	_KLDR_DATA_TABLE_ENTRY entry{};
	entry.DllBase = reinterpret_cast<void*>(mod.addr);
	entry.EntryPoint = reinterpret_cast<void*>(mod.entry_point);
	entry.SizeOfImage = mod.size;

	// A nameless entry is most of the way to useless: code that walks this list is looking for
	// a module by name, so without one it can only ever match on an address it already had.
	const auto wide_name = widen_string(mod.name);

	entry.BaseDllName = win::init_unicode_string(space, wide_name, prot_rw | prot_supervisor);

	// What the kernel reports, which is never a drive letter: a driver goes under drivers\, 
	// everything else sits in system32 itself.
	const std::u16string dir = mod.lookup_name.ends_with(".sys")
		? u"\\SystemRoot\\System32\\drivers\\"
		: u"\\SystemRoot\\System32\\";

	entry.FullDllName = win::init_unicode_string(space, dir + wide_name, prot_rw | prot_supervisor);

	std::scoped_lock lock(kernel_.list_mtx_);
	const auto obj = kernel_.loaded_module_list.push_back(entry);

	kernel_.set_ldr_entry(mod.addr, obj.address());
}

std::shared_ptr<win_thread> windows_process::find_ethread(
	const emu_object<_ETHREAD>& ethread) const
{
	std::shared_lock lock(thread_mtx_);

	for (const auto& [id, t] : threads_)
	{
		const auto win_t = std::dynamic_pointer_cast<win_thread>(t);

		if (win_t && win_t->ethread().address() == ethread.address())
			return win_t;
	}

	return {};
}

void windows_process::for_each_thread(const std::function<void(win_thread&)>& fn) const
{
	std::shared_lock lock(thread_mtx_);

	for (const auto& [id, t] : threads_)
	{
		if (const auto win_t = std::dynamic_pointer_cast<win_thread>(t))
			fn(*win_t);
	}
}

void windows_process::wake_waiters(struct addr_space& space, const addr_t object) const
{
	bool released = false;

	const auto wake = [&](const windows_process& proc)
	{
		proc.for_each_thread([&](win_thread& t)
		{
			if (t.waiting_on(object) && t.try_satisfy(space))
			{
				THREAD_LOG_INFO("wait on 0x{:X} satisfied for tid={}", object, t.id());
				released = true;
			}
		});
	};

	if (emulator_)
	{
		for (const auto& proc : emulator_->kernel().all_processes())
		{
			if (const auto win_proc = std::dynamic_pointer_cast<windows_process>(proc))
				wake(*win_proc);
		}
	}
	else
	{
		wake(*this);
	}

	// Satisfying the wait is only half of releasing the thread: it is runnable again without
	// having been queued, and a cpu parked on an empty answer does not look a second time on
	// its own. An untimed wait leaves nothing for next_wake() to arm either, so that cpu waits
	// on a notify that never comes. start() and resume() are the other two ways a thread turns
	// runnable out of band, and both say so exactly here.
	if (released)
	{
		if (const auto s = scheduler())
			s->wake();
	}
}
