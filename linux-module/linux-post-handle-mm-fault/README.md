in `mm/memory.c` update `handle_mm_fault`

```
pid_t target_pid = -1;
EXPORT_SYMBOL(target_pid);
vm_fault_t handle_mm_fault(struct vm_area_struct *vma, unsigned long address,
                           unsigned int flags, struct pt_regs *regs)
{
//      if (current->pid == target_pid) {
//            printk("PRE handle_mm_fault\n");
//            log_page_fault_details(vma, address, flags);
//        }

        /* If the fault handler drops the mmap_lock, vma may be freed */
        struct mm_struct *mm = vma->vm_mm;
        vm_fault_t ret;

        __set_current_state(TASK_RUNNING);

        ret = sanitize_fault_flags(vma, &flags);
        if (ret)
                goto out;

        if (!arch_vma_access_permitted(vma, flags & FAULT_FLAG_WRITE,
                                            flags & FAULT_FLAG_INSTRUCTION,
                                            flags & FAULT_FLAG_REMOTE)) {
                ret = VM_FAULT_SIGSEGV;
                goto out;
        }

        /*
         * Enable the memcg OOM handling for faults triggered in user
         * space.  Kernel faults are handled more gracefully.
         */
        if (flags & FAULT_FLAG_USER)
                mem_cgroup_enter_user_fault();

        lru_gen_enter_fault(vma);

        if (unlikely(is_vm_hugetlb_page(vma)))
                ret = hugetlb_fault(vma->vm_mm, vma, address, flags);
        else
                ret = __handle_mm_fault(vma, address, flags);

        lru_gen_exit_fault();

        if (flags & FAULT_FLAG_USER) {
                mem_cgroup_exit_user_fault();
                /*
                 * The task may have entered a memcg OOM situation but
                 * if the allocation error was handled gracefully (no
                 * VM_FAULT_OOM), there is no need to kill anything.
                 * Just clean up the OOM state peacefully.
                 */
                if (task_in_memcg_oom(current) && !(ret & VM_FAULT_OOM))
                        mem_cgroup_oom_synchronize(false);
        }
out:
        mm_account_fault(mm, regs, address, flags, ret);
        if (current->pid == target_pid) {
//          printk("POST handle_mm_fault\n");
            log_page_fault_details(vma, address, flags);
            if (redirect_pte) {
                   pr_info("Calling redirect_pte function...\n");
                   redirect_pte(vma, address, flags,regs);
                }
        }

        return ret;
}
EXPORT_SYMBOL_GPL(handle_mm_fault);

```
