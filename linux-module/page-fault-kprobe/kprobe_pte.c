#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/kprobes.h>
#include <linux/mm.h>
#include <linux/sched.h>
#include <linux/version.h>
#include <linux/ptrace.h>
#include <linux/highmem.h>
#include <linux/pgtable.h>

#define MAX_SYMBOL_LEN 64

// Module parameter to specify target PID
static int target_pid = -1;
module_param(target_pid, int, S_IRUGO);
MODULE_PARM_DESC(target_pid, "PID to monitor (default: -1 for all PIDs)");

static unsigned long get_pte_flags(pte_t pte)
{
    return pte_val(pte) & ~(pte_pfn(pte) << PAGE_SHIFT);
}

static unsigned long get_pmd_flags(pmd_t pmd)
{
    return pmd_val(pmd) & ~(pmd_pfn(pmd) << PAGE_SHIFT);
}

static unsigned long get_pud_flags(pud_t pud)
{
    return pud_val(pud) & ~(pud_pfn(pud) << PAGE_SHIFT);
}

static unsigned long get_p4d_flags(p4d_t p4d)
{
    return p4d_val(p4d) & ~(p4d_pfn(p4d) << PAGE_SHIFT);
}

static unsigned long get_pgd_flags(pgd_t pgd)
{
    return pgd_val(pgd) & ~(pgd_pfn(pgd) << PAGE_SHIFT);
}

static void print_page_table_walk(struct vm_area_struct *vma, unsigned long address)
{
    pgd_t *pgd;
    p4d_t *p4d;
    pud_t *pud;
    pmd_t *pmd;
    pte_t *pte;

    struct mm_struct *mm = vma->vm_mm;
    pgd = pgd_offset(mm, address);
    if (pgd_none(*pgd) || pgd_bad(*pgd))
        return;

    pr_info("PID: %d PGD: %lx (flags: %lx)\n",
            current->pid, pgd_val(*pgd), get_pgd_flags(*pgd));

    p4d = p4d_offset(pgd, address);
    if (p4d_none(*p4d) || p4d_bad(*p4d))
        return;

    pr_info("PID: %d P4D: %lx (flags: %lx)\n",
            current->pid, p4d_val(*p4d), get_p4d_flags(*p4d));

    pud = pud_offset(p4d, address);
    if (pud_none(*pud) || pud_bad(*pud))
        return;

    pr_info("PID: %d PUD: %lx (flags: %lx)\n",
            current->pid, pud_val(*pud), get_pud_flags(*pud));

    pmd = pmd_offset(pud, address);
    if (pmd_none(*pmd) || pmd_bad(*pmd))
        return;

    pr_info("PID: %d PMD: %lx (flags: %lx)\n",
            current->pid, pmd_val(*pmd), get_pmd_flags(*pmd));

    pte = pte_offset_map(pmd, address);
    if (!pte)
        return;

    pr_info("PID: %d PTE: %lx (flags: %lx)\n",
            current->pid, pte_val(*pte), get_pte_flags(*pte));
    pte_unmap(pte);
}

static int handler_pre(struct kprobe *p, struct pt_regs *regs)
{
    struct vm_area_struct *vma = (struct vm_area_struct *)regs->di;
    unsigned long address = regs->si;
    unsigned int flags = regs->dx;

    if (!vma || !current->mm)
        return 0;

    // Check if we should log this page fault
    if (target_pid != -1 && current->pid != target_pid)
        return 0;

    pr_info("PID: %d Page fault at: %lx, flags: %x\n", current->pid, address, flags);
    print_page_table_walk(vma, address);

    return 0;
}

static struct kprobe kp = {
    .symbol_name = "handle_mm_fault",
    .pre_handler = handler_pre,
};

static int __init kprobe_init(void)
{
    int ret;

    ret = register_kprobe(&kp);
    if (ret < 0) {
        pr_err("register_kprobe failed, returned %d\n", ret);
        return ret;
    }

    pr_info("Planted kprobe at %p\n", kp.addr);
    pr_info("Monitoring page faults for PID: %d\n", target_pid);
    return 0;
}

static void __exit kprobe_exit(void)
{
    unregister_kprobe(&kp);
    pr_info("kprobe at %p unregistered\n", kp.addr);
}

module_init(kprobe_init);
module_exit(kprobe_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Jatin");
MODULE_DESCRIPTION("Trace page faults for specific PID and print page table entries");
