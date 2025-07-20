```
#include <linux/sched.h>
#include <linux/kallsyms.h>
#include <linux/uaccess.h>
#include <linux/dcache.h>
#include <linux/path.h>
#include <linux/mount.h>
#include <linux/slab.h>
#include <linux/kdev_t.h>
static void print_vma_info(struct vm_area_struct *vma)
{
    pr_info("========== VMA Info ==========\n");

    /* Address range */
    pr_info("Start: 0x%lx, End: 0x%lx, Size: %lu KB\n",
            vma->vm_start, vma->vm_end,
            (vma->vm_end - vma->vm_start) >> 10);

    /* Permissions and flags */
    pr_info("Permissions: %c%c%c%c%c%c\n",
            (vma->vm_flags & VM_READ)  ? 'r' : '-',
            (vma->vm_flags & VM_WRITE) ? 'w' : '-',
            (vma->vm_flags & VM_EXEC)  ? 'x' : '-',
            (vma->vm_flags & VM_SHARED)? 's' : 'p',
            (vma->vm_flags & VM_GROWSDOWN) ? 'd' : '-',
            (vma->vm_flags & VM_GROWSUP)   ? 'u' : '-');

    pr_info("vm_flags: 0x%lx [", vma->vm_flags);
#define PRINT_FLAG(flag, name) if (vma->vm_flags & flag) pr_cont("%s ", name)
    PRINT_FLAG(VM_IO, "IO");
    PRINT_FLAG(VM_DONTCOPY, "DONTCOPY");
    PRINT_FLAG(VM_DONTEXPAND, "DONTEXPAND");
    PRINT_FLAG(VM_ACCOUNT, "ACCOUNT");
    PRINT_FLAG(VM_HUGETLB, "HUGETLB");
    PRINT_FLAG(VM_LOCKED, "LOCKED");
    PRINT_FLAG(VM_MAYREAD, "MAYREAD");
    PRINT_FLAG(VM_MAYWRITE, "MAYWRITE");
    PRINT_FLAG(VM_MAYEXEC, "MAYEXEC");
    PRINT_FLAG(VM_MAYSHARE, "MAYSHARE");
#undef PRINT_FLAG
    pr_cont("]\n");

    /* File mapping info */
    if (vma->vm_file) {
        char *tmp = (char *)__get_free_page(GFP_KERNEL);
        if (tmp) {
            char *name = d_path(&vma->vm_file->f_path, tmp, PAGE_SIZE);
            pr_info("File: %s\n", IS_ERR(name) ? "(error resolving path)" : name);
            free_page((unsigned long)tmp);
        } else {
            pr_info("File: (path resolution failed)\n");
        }

        pr_info("Inode: %lu, Device: %d:%d\n",
                vma->vm_file->f_inode->i_ino,
                MAJOR(vma->vm_file->f_inode->i_sb->s_dev),
                MINOR(vma->vm_file->f_inode->i_sb->s_dev));
    } else {
        pr_info("File: (none, anonymous mapping)\n");
    }

    pr_info("Page offset (vm_pgoff): 0x%lx\n", vma->vm_pgoff);

    /* Page protection */
    pr_info("Page protection flags: 0x%lx\n", pgprot_val(vma->vm_page_prot));

    /* VM operations */
    if (vma->vm_ops && vma->vm_ops->fault)
        pr_info("vm_ops->fault: %ps\n", vma->vm_ops->fault);
    else if (vma->vm_ops)
        pr_info("vm_ops: %px\n", vma->vm_ops);
    else
        pr_info("vm_ops: (none)\n");

    /* Private data */
    pr_info("vm_private_data: %px\n", vma->vm_private_data);

#if defined(CONFIG_ANON_VMA_NAME)
    if (vma->anon_name)
        pr_info("anon_name: %ps\n", vma->anon_name);
#endif

#ifdef CONFIG_USERFAULTFD
    if (vma->vm_userfaultfd_ctx.ctx)
        pr_info("userfaultfd context present\n");
#endif

#ifdef CONFIG_NUMA
    if (vma->vm_policy)
        pr_info("NUMA policy: %ps\n", vma->vm_policy);
#endif

    pr_info("==============================\n");
}
static void log_page_fault_details(struct vm_area_struct *vma, unsigned long address, unsigned int flags)
{
    //struct mm_struct *mm = current->mm;
    char flags_str[256] = {0};
    char *ptr = flags_str;

    /* Build flags string safely */
#define APPEND_FLAG(flag, str) \
    if (vma->vm_flags & (flag)) { \
        ptr += snprintf(ptr, sizeof(flags_str) - (ptr - flags_str), "%s ", (str)); \
    }

  //  pr_info("=== Page Fault Details ===\n");
  //  pr_info("Process: %s (PID: %d)\n", current->comm, current->pid);
    pr_info("Fault Address: 0x%lx | Fault Flags: 0x%x [%s%s%s]\n",
           address, flags,
           (flags & FAULT_FLAG_WRITE) ? "WRITE " : "",
           (flags & FAULT_FLAG_INSTRUCTION) ? "EXEC " : "",
           (flags & FAULT_FLAG_USER) ? "USER " : "");
/*    
    print_vma_info(vma);   
    
    pr_info("\n--- Page Table Walk ---\n");
    pgd_t *pgd = pgd_offset(mm, address);
    if (pgd_none(*pgd) || pgd_bad(*pgd)) {
        pr_warn("PGD not present or bad\n");
        goto out;
    }
    pr_info("PGD: 0x%lx\n", pgd_val(*pgd));

    p4d_t *p4d = p4d_offset(pgd, address);
    if (p4d_none(*p4d) || p4d_bad(*p4d)) {
        pr_warn("P4D not present or bad\n");
        goto out;
    }
    pr_info("P4D: 0x%lx\n", p4d_val(*p4d));

    pud_t *pud = pud_offset(p4d, address);
    if (pud_none(*pud) || pud_bad(*pud)) {
        pr_warn("PUD not present or bad\n");
        goto out;
    }
    pr_info("PUD: 0x%lx\n", pud_val(*pud));

    pmd_t *pmd = pmd_offset(pud, address);
    if (pmd_none(*pmd) || pmd_bad(*pmd)) {
        pr_warn("PMD not present or bad\n");
        goto out;
    }
    pr_info("PMD: 0x%lx\n", pmd_val(*pmd));

    pte_t *pte = pte_offset_map(pmd, address);
    if (!pte) {
        pr_warn("PTE not mapped\n");
        goto out;
    }
    pr_info("PTE: 0x%llx | Present: %d | Dirty: %d | Accessed: %d\n", 
           (u64)pte_val(*pte), pte_present(*pte), pte_dirty(*pte), pte_young(*pte));
    pte_unmap(pte);
*/
out:
//    pr_info("\n=========================\n");
}
EXPORT_SYMBOL_GPL(__pte_offset_map_lock);
EXPORT_SYMBOL_GPL(__pte_offset_map);

/*
 * By the time we get here, we already hold the mm semaphore
 *
 * The mmap_lock may have been released depending on flags and our
 * return value.  See filemap_fault() and __folio_lock_or_retry().
 */
void (*redirect_pte)(struct vm_area_struct *vma, unsigned long address,unsigned int flags, struct pt_regs *regs) = NULL;

EXPORT_SYMBOL(redirect_pte);
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
```
