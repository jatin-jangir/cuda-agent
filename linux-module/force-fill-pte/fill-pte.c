#include <linux/module.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/uaccess.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/slab.h>
#include <linux/device.h>
#include <linux/sysfs.h>
#include <linux/init.h>
#include <asm/pgtable.h>
#include <asm/tlbflush.h>










#define DEVICE_NAME "memredir"
#define BUF_SIZE 256

static int major;
static struct class *cls;

// External symbols defined in kernel
extern int target_pid;
extern void (*redirect_pte)(struct vm_area_struct *vma, unsigned long address, unsigned int flags, struct pt_regs *regs);

static pid_t target_pid_mod = -1;
static unsigned long src_start, src_end;
static unsigned long dst_start, dst_end;





static ssize_t memredir_write(struct file *file, const char __user *buf, size_t count, loff_t *ppos)
{
    char kbuf[BUF_SIZE];
    struct task_struct *task;
    struct mm_struct *mm;
    unsigned long addr;

    if (count >= BUF_SIZE)
        return -EINVAL;

    if (copy_from_user(kbuf, buf, count))
        return -EFAULT;

    kbuf[count] = '\0';

    if (sscanf(kbuf, "%d %lx %lx %lx %lx",
               &target_pid_mod, &src_start, &dst_start, &src_end, &dst_end) != 5) {
        pr_err("memredir: Invalid input format\n");
        return -EINVAL;
    }

    target_pid = target_pid_mod;

    pr_info("memredir: Set PID=%d\n", target_pid);
    pr_info("memredir: src=[%lx - %lx], dst=[%lx - %lx]\n",
            src_start, src_end, dst_start, dst_end);

    task = pid_task(find_vpid(target_pid), PIDTYPE_PID);
    if (!task) {
        pr_err("memredir: Target PID not found\n");
        return -ESRCH;
    }

    mm = task->mm;
    if (!mm) {
        pr_err("memredir: Target process has no mm_struct\n");
        return -EINVAL;
    }

    down_write(&mm->mmap_lock);

    for (addr = src_start; addr < src_end; addr += PAGE_SIZE) {
        unsigned long dst_addr = dst_start + (addr - src_start);

        pr_info("memredir: Processing src 0x%lx -> dst 0x%lx\n", addr, dst_addr);

        // Walk src
        pgd_t *pgd = pgd_offset(mm, addr);
        if (pgd_none(*pgd) || pgd_bad(*pgd)) {
            pr_info("memredir: Bad PGD for src 0x%lx\n", addr);
            continue;
        }

        p4d_t *p4d = p4d_offset(pgd, addr);
        if (p4d_none(*p4d) || p4d_bad(*p4d)) {
            pr_info("memredir: Bad P4D for src 0x%lx\n", addr);
            continue;
        }

        pud_t *pud = pud_offset(p4d, addr);
        if (pud_none(*pud) || pud_bad(*pud)) {
            pr_info("memredir: Bad PUD for src 0x%lx\n", addr);
            continue;
        }

        pmd_t *pmd = pmd_offset(pud, addr);
        if (pmd_none(*pmd) || pmd_bad(*pmd)) {
            pr_info("memredir: Bad PMD for src 0x%lx\n", addr);
            continue;
        }

        pte_t *src_pte = pte_offset_map(pmd, addr);
        if (!src_pte) {
            pr_info("memredir: No PTE for src 0x%lx\n", addr);
            continue;
        }

        if (!pte_present(*src_pte)) {
            pr_info("memredir: PTE not present for src 0x%lx\n", addr);
            pte_unmap(src_pte);
            continue;
        }

        unsigned long pfn = pte_pfn(*src_pte);
        pgprot_t prot = pte_pgprot(*src_pte);

        pr_info("memredir: Got PFN 0x%lx, prot 0x%lx for src 0x%lx\n",
                pfn, pgprot_val(prot), addr);

        pte_unmap(src_pte);

        // Walk dst and allocate tables if missing
        pgd = pgd_offset(mm, dst_addr);
        if (pgd_none(*pgd) || pgd_bad(*pgd)) {
	    pr_err("memredir: Bad or missing PGD for dst 0x%lx\n", dst_addr);
	    continue;
	}

        p4d = p4d_alloc(mm, pgd, dst_addr);
        if (!p4d) {
            pr_err("memredir: Failed p4d_alloc\n");
            continue;
        }

        pud = pud_alloc(mm, p4d, dst_addr);
        if (!pud) {
            pr_err("memredir: Failed pud_alloc\n");
            continue;
        }

        pmd = pmd_alloc(mm, pud, dst_addr);
        if (!pmd) {
            pr_err("memredir: Failed pmd_alloc\n");
            continue;
        }

        pte_t *dst_pte = pte_alloc_map(mm, pmd, dst_addr);
        if (!dst_pte) {
            pr_err("memredir: Failed pte_alloc_map for dst 0x%lx\n", dst_addr);
            continue;
        }

        pte_t new_pte = pfn_pte(pfn, prot);
        set_pte_at(mm, dst_addr, dst_pte, new_pte);

        pr_info("memredir: Force-mapped PFN 0x%lx from src 0x%lx to dst 0x%lx\n",
                pfn, addr, dst_addr);

        pte_unmap(dst_pte);
    }

    up_write(&mm->mmap_lock);

    pr_info("memredir: Done\n");
    target_pid=-1;
    return count;
}




static int memredir_open(struct inode *inode, struct file *file) {
    return 0;
}


static int memredir_release(struct inode *inode, struct file *file) {
    return 0;
}

static struct file_operations memredir_fops = {
    .owner = THIS_MODULE,
    .open = memredir_open,
    .write = memredir_write,
    .release = memredir_release,
};



static int __init memredir_init(void) {
    major = register_chrdev(0, DEVICE_NAME, &memredir_fops);
    if (major < 0)
        return major;

    cls = class_create(DEVICE_NAME);
    if (IS_ERR(cls)) {
        unregister_chrdev(major, DEVICE_NAME);
        return PTR_ERR(cls);
    }

    device_create(cls, NULL, MKDEV(major, 0), NULL, DEVICE_NAME);

//    redirect_pte = redirect_pte_function;
    pr_info("memredir: Module loaded\n");
    return 0;
}

static void __exit memredir_exit(void) {
    redirect_pte = NULL;
    device_destroy(cls, MKDEV(major, 0));
    class_destroy(cls);
    unregister_chrdev(major, DEVICE_NAME);
    pr_info("memredir: Module unloaded\n");
}

module_init(memredir_init);
module_exit(memredir_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Jatin + ChatGPT");
MODULE_DESCRIPTION("Redirect user memory access from dst to src using PTE remap.");
