#include <linux/init.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/uaccess.h>
#include <linux/mm.h>
#include <linux/sched/mm.h>
#include <linux/pid.h>
#include <linux/slab.h>
#include <linux/cdev.h>

#define DEVICE_NAME "pidaddrinfo"
#define CLASS_NAME  "pidaddr"
#define BUF_SIZE    256

static dev_t dev_number;
static struct class *dev_class;
static struct cdev my_cdev;

static int dev_open(struct inode *inode, struct file *file) {
    return 0;
}

static int dev_release(struct inode *inode, struct file *file) {
    return 0;
}

static void print_page_flags(struct page *page, const char *prefix) {
    pr_info("%s Page flags: %#lx\n", prefix, page->flags);
    if (PageReserved(page)) pr_info("%s  - Reserved\n", prefix);
    if (PageSwapCache(page)) pr_info("%s  - Swap cache\n", prefix);
    if (PageDirty(page)) pr_info("%s  - Dirty\n", prefix);
    if (PageAnon(page)) pr_info("%s  - Anonymous\n", prefix);
    if (PageSlab(page)) pr_info("%s  - Slab\n", prefix);
}

static ssize_t dev_write(struct file *file, const char __user *ubuf, size_t len, loff_t *off) {
    char kbuf[BUF_SIZE];
    pid_t pid;
    char prefix[64];
    unsigned long address;

    if (len >= BUF_SIZE) return -EINVAL;
    if (copy_from_user(kbuf, ubuf, len)) return -EFAULT;
    kbuf[len] = '\0';

    if (sscanf(kbuf, "%d %63s %lx", &pid, prefix, &address) != 3) {
        pr_err("pidaddrinfo: invalid input format. Expected: <pid> <prefix> <address>\n");
        return -EINVAL;
    }

    struct task_struct *task = pid_task(find_get_pid(pid), PIDTYPE_PID);
    if (!task) {
        pr_err("%s Invalid PID: %d\n", prefix, pid);
        return -ESRCH;
    }

    struct mm_struct *mm = get_task_mm(task);
    if (!mm) {
        pr_err("%s Failed to get mm_struct for PID %d\n", prefix, pid);
        return -EINVAL;
    }

    mmap_read_lock(mm);
    struct vm_area_struct *vma = find_vma(mm, address);
    if (!vma || address < vma->vm_start) {
        pr_err("%s Address %#lx is not mapped\n", prefix, address);
        mmap_read_unlock(mm);
        mmput(mm);
        return -EFAULT;
    }

    pr_info("[%s] === VMA Info ===\n", prefix);
    pr_info("[%s] Start: 0x%lx End: 0x%lx\n", prefix, vma->vm_start, vma->vm_end);
    pr_info("[%s] vm_flags: 0x%lx\n", prefix, vma->vm_flags);
    pr_info("[%s] vm_page_prot: 0x%lx\n", prefix, pgprot_val(vma->vm_page_prot));
    pr_info("[%s] vm_ops: %px\n", prefix, vma->vm_ops);
    pr_info("[%s] anon_vma: %px\n", prefix, vma->anon_vma);
    pr_info("[%s] vm_file: %px\n", prefix, vma->vm_file);
    pr_info("[%s] vm_private_data: %px\n", prefix, vma->vm_private_data);

    pr_info("[%s] === Page Table Walk for addr: 0x%lx ===\n", prefix, address);

    pgd_t *pgd = pgd_offset(mm, address);
    if (pgd_none(*pgd) || pgd_bad(*pgd)) goto out;

    p4d_t *p4d = p4d_offset(pgd, address);
    if (p4d_none(*p4d) || p4d_bad(*p4d)) goto out;

    pud_t *pud = pud_offset(p4d, address);
    if (pud_none(*pud) || pud_bad(*pud)) goto out;

    pmd_t *pmd = pmd_offset(pud, address);
    if (pmd_none(*pmd) || pmd_bad(*pmd)) goto out;

    pte_t *pte = pte_offset_map(pmd, address);
    if (!pte || pte_none(*pte)) {
        pr_err("[%s] PTE not present for address %#lx\n", prefix, address);
        goto out;
    }

    unsigned long pteval = pte_val(*pte);
    pr_info("[%s] PTE: 0x%016lx\n", prefix, pteval);

    struct page *page = pte_page(*pte);
    if (page) {
        pr_info("[%s] === Page Struct Info ===\n", prefix);
        pr_info("[%s] Page PFN: 0x%lx\n", prefix, page_to_pfn(page));
        pr_info("[%s] Page struct: %px\n", prefix, page);
        pr_info("[%s] flags: 0x%lx\n", prefix, page->flags);
        pr_info("[%s] refcount: %d\n", prefix, page_ref_count(page));
        pr_info("[%s] mapcount: %d\n", prefix, page_mapcount(page));
        pr_info("[%s] mapping: %px\n", prefix, page->mapping);
        pr_info("[%s] index: 0x%lx\n", prefix, page->index);
    } else {
        pr_err("[%s] Failed to get struct page\n", prefix);
    }

    pr_info("[%s] === Additional Context ===\n", prefix);
    pr_info("[%s] Fault flags: 0x%lx\n", prefix, vma->vm_flags);  // fallback
#ifdef CONFIG_X86_64
    pr_info("[%s] Faulting instruction pointer: 0x%lx\n", prefix, task_pt_regs(task)->ip);
#else
    pr_info("[%s] Faulting instruction pointer: <unsupported arch>\n", prefix);
#endif

    pte_unmap(pte);
    mmap_read_unlock(mm);
    mmput(mm);
    return len;

out:
    pr_err("[%s] Page table walk failed for address %#lx\n", prefix, address);
    mmap_read_unlock(mm);
    mmput(mm);
    return -EFAULT;
}

static struct file_operations fops = {
    .owner = THIS_MODULE,
    .open = dev_open,
    .release = dev_release,
    .write = dev_write,
};

static int __init pidaddrinfo_init(void) {
    alloc_chrdev_region(&dev_number, 0, 1, DEVICE_NAME);
    cdev_init(&my_cdev, &fops);
    cdev_add(&my_cdev, dev_number, 1);

    dev_class = class_create( CLASS_NAME);
    device_create(dev_class, NULL, dev_number, NULL, DEVICE_NAME);

    pr_info("pidaddrinfo loaded. Use /dev/%s\n", DEVICE_NAME);
    return 0;
}

static void __exit pidaddrinfo_exit(void) {
    device_destroy(dev_class, dev_number);
    class_destroy(dev_class);
    cdev_del(&my_cdev);
    unregister_chrdev_region(dev_number, 1);
    pr_info("pidaddrinfo unloaded.\n");
}

module_init(pidaddrinfo_init);
module_exit(pidaddrinfo_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("jatin");
MODULE_DESCRIPTION("Char device: print VMA, PTE, page struct info from <pid> <prefix> <addr>");
