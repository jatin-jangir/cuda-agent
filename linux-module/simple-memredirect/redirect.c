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

static int memredir_open(struct inode *inode, struct file *file) {
    return 0;
}

static ssize_t memredir_write(struct file *file, const char __user *buf, size_t count, loff_t *ppos) {
    char kbuf[BUF_SIZE];

    if (count >= BUF_SIZE)
        return -EINVAL;

    if (copy_from_user(kbuf, buf, count))
        return -EFAULT;

    kbuf[count] = '\0';

    if (sscanf(kbuf, "%d %lx %lx %lx %lx", &target_pid_mod, &src_start, &dst_start, &src_end, &dst_end) != 5) {
        pr_err("memredir: Invalid input format\n");
        return -EINVAL;
    }

    target_pid = target_pid_mod;

    pr_info("memredir: Set PID=%d\n", target_pid_mod);
    pr_info("memredir: src=[%lx - %lx], dst=[%lx - %lx]\n",
            src_start, src_end, dst_start, dst_end);

    return count;
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

static void redirect_pte_function(struct vm_area_struct *vma, unsigned long addr, unsigned int flags, struct pt_regs *regs) {
    struct mm_struct *mm;
    unsigned long offset, src_addr;
    pte_t *pte_src = NULL, *pte_dst = NULL;
    spinlock_t *src_ptl = NULL, *dst_ptl = NULL;
    pgd_t *pgd;
    p4d_t *p4d;
    pud_t *pud;
    pmd_t *pmd;
    unsigned long src_pfn;
    pgprot_t prot;

    if (!vma || !(mm = vma->vm_mm))
        return;

    if (addr < dst_start || addr >= dst_end)
        return;

    offset = addr - dst_start;
    src_addr = src_start + offset;

    // Source PTE walk
    pgd = pgd_offset(mm, src_addr);
    if (pgd_none(*pgd) || pgd_bad(*pgd)) return;

    p4d = p4d_offset(pgd, src_addr);
    if (p4d_none(*p4d) || p4d_bad(*p4d)) return;

    pud = pud_offset(p4d, src_addr);
    if (pud_none(*pud) || pud_bad(*pud)) return;

    pmd = pmd_offset(pud, src_addr);
    if (pmd_none(*pmd) || pmd_bad(*pmd)) return;

    pte_src = pte_offset_map_lock(mm, pmd, src_addr, &src_ptl);
    if (!pte_src || !pte_present(*pte_src)) {
        if (pte_src)
            pte_unmap_unlock(pte_src, src_ptl);
        return;
    }

    src_pfn = pte_pfn(*pte_src);
    pte_unmap_unlock(pte_src, src_ptl);

    // Destination PTE walk
    pgd = pgd_offset(mm, addr);
    if (pgd_none(*pgd) || pgd_bad(*pgd)) return;

    p4d = p4d_offset(pgd, addr);
    if (p4d_none(*p4d) || p4d_bad(*p4d)) return;

    pud = pud_offset(p4d, addr);
    if (pud_none(*pud) || pud_bad(*pud)) return;

    pmd = pmd_offset(pud, addr);
    if (pmd_none(*pmd) || pmd_bad(*pmd)) return;

    pte_dst = pte_offset_map_lock(mm, pmd, addr, &dst_ptl);
    if (!pte_dst) {
        return;
    }

    prot = vma->vm_page_prot;
    pte_t new_pte = pfn_pte(src_pfn, prot);

    set_pte_at(mm, addr, pte_dst, new_pte);
    pte_unmap_unlock(pte_dst, dst_ptl);

    __flush_tlb_one_user(addr);

    pr_info("memredir: Remapped 0x%lx -> PFN 0x%lx\n", addr, src_pfn);
}

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

    redirect_pte = redirect_pte_function;
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
MODULE_AUTHOR("Jatin");
MODULE_DESCRIPTION("Redirect user memory access from dst to src using PTE remap.");
