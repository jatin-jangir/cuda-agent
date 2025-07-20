#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/kprobes.h>
#include <linux/mm.h>
#include <linux/sched.h>
#include <linux/ptrace.h>
#include <linux/highmem.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/cdev.h>
#include <linux/device.h>

#define DEVICE_NAME "kprobe_pte"
#define CLASS_NAME "kprobe"

static dev_t dev_num;
static struct cdev kprobe_cdev;
static struct class *kprobe_class;
static int target_pid = -1;

// Data passed between entry and return handler
struct fault_data {
    struct vm_area_struct *vma;
    unsigned long address;
};

static void decode_flags(const char *level, unsigned long val)
{
    pr_info("PID %d %s: 0x%lx\n", current->pid, level, val);
    pr_info("  Flags: 0x%lx\n", val);
    pr_info("    Present:    %d\n", (val & _PAGE_PRESENT) ? 1 : 0);
    pr_info("    Write:      %d\n", (val & _PAGE_RW) ? 1 : 0);
    pr_info("    User:       %d\n", (val & _PAGE_USER) ? 1 : 0);
    pr_info("    Accessed:   %d\n", (val & _PAGE_ACCESSED) ? 1 : 0);
    pr_info("    Dirty:      %d\n", (val & _PAGE_DIRTY) ? 1 : 0);
    pr_info("    NX:         %d\n", (val & _PAGE_NX) ? 1 : 0);
}

static void print_page_table_walk(struct vm_area_struct *vma, unsigned long address)
{
    pgd_t *pgd = pgd_offset(vma->vm_mm, address);
    if (pgd_none(*pgd) || pgd_bad(*pgd)) return;
    decode_flags("PGD", pgd_val(*pgd));

    p4d_t *p4d = p4d_offset(pgd, address);
    if (p4d_none(*p4d) || p4d_bad(*p4d)) return;
    decode_flags("P4D", p4d_val(*p4d));

    pud_t *pud = pud_offset(p4d, address);
    if (pud_none(*pud) || pud_bad(*pud)) return;
    decode_flags("PUD", pud_val(*pud));

    pmd_t *pmd = pmd_offset(pud, address);
    if (pmd_none(*pmd) || pmd_bad(*pmd)) return;
    decode_flags("PMD", pmd_val(*pmd));

    // Only decode kernel-space PTEs
    if (address >= PAGE_OFFSET) {
        pte_t *pte = pte_offset_kernel(pmd, address);
        if (pte) {
            decode_flags("PTE", pte_val(*pte));
            pte_unmap(pte);
        }
    } else {
        pr_info("PID %d: User-space address, skipping PTE decode (restricted)\n", current->pid);
    }
}

// Kretprobe return handler
static int handler_ret(struct kretprobe_instance *ri, struct pt_regs *regs)
{
    struct fault_data *data = (struct fault_data *)ri->data;

    if (!data->vma || !current->mm)
        return 0;

    if (current->pid != target_pid)
        return 0;

    pr_info("AFTER FAULT: PID %d Address: %lx\n", current->pid, data->address);
    print_page_table_walk(data->vma, data->address);
    return 0;
}

// Entry handler saves vma + address for later use
static int handler_entry(struct kretprobe_instance *ri, struct pt_regs *regs)
{
    struct fault_data *data;

    if (!current->mm || current->pid != target_pid)
        return 0;

    data = (struct fault_data *)ri->data;
    data->vma = (struct vm_area_struct *)regs->di;
    data->address = regs->si;

    pr_info("target_pid: %d\n", target_pid);
    pr_info("Page fault at: %lx in PID: %d\n", data->address, current->pid);

    return 0;
}

static struct kretprobe kp_ret = {
    .kp.symbol_name = "handle_mm_fault",
    .entry_handler = handler_entry,
    .handler = handler_ret,
    .data_size = sizeof(struct fault_data),
    .maxactive = 20,
};

// Char device write handler to update PID
static ssize_t kprobe_write(struct file *file, const char __user *buf, size_t len, loff_t *off)
{
    char kbuf[16];
    int new_pid;

    if (len > 15)
        return -EINVAL;

    if (copy_from_user(kbuf, buf, len))
        return -EFAULT;

    kbuf[len] = '\0';

    if (kstrtoint(kbuf, 10, &new_pid) == 0) {
        target_pid = new_pid;
        pr_info("Updated target PID to %d\n", target_pid);
        return len;
    }

    return -EINVAL;
}

static struct file_operations fops = {
    .owner = THIS_MODULE,
    .write = kprobe_write,
};

static int __init kprobe_init(void)
{
    int ret;

    ret = alloc_chrdev_region(&dev_num, 0, 1, DEVICE_NAME);
    if (ret < 0) {
        pr_err("Failed to allocate char device region\n");
        return ret;
    }

    cdev_init(&kprobe_cdev, &fops);
    kprobe_cdev.owner = THIS_MODULE;

    if ((ret = cdev_add(&kprobe_cdev, dev_num, 1)) < 0) {
        pr_err("Failed to add cdev\n");
        goto fail_cdev;
    }

    kprobe_class = class_create(CLASS_NAME);
    if (IS_ERR(kprobe_class)) {
        ret = PTR_ERR(kprobe_class);
        goto fail_class;
    }

    if (IS_ERR(device_create(kprobe_class, NULL, dev_num, NULL, DEVICE_NAME))) {
        ret = PTR_ERR(device_create(kprobe_class, NULL, dev_num, NULL, DEVICE_NAME));
        goto fail_device;
    }

    if ((ret = register_kretprobe(&kp_ret)) < 0) {
        pr_err("register_kretprobe failed, returned %d\n", ret);
        goto fail_kprobe;
    }

    pr_info("Kretprobe planted at %p. Write PID to /dev/%s to update.\n", kp_ret.kp.addr, DEVICE_NAME);
    return 0;

fail_kprobe:
    device_destroy(kprobe_class, dev_num);
fail_device:
    class_destroy(kprobe_class);
fail_class:
    cdev_del(&kprobe_cdev);
fail_cdev:
    unregister_chrdev_region(dev_num, 1);
    return ret;
}

static void __exit kprobe_exit(void)
{
    unregister_kretprobe(&kp_ret);
    device_destroy(kprobe_class, dev_num);
    class_destroy(kprobe_class);
    cdev_del(&kprobe_cdev);
    unregister_chrdev_region(dev_num, 1);
    pr_info("Module exited, kretprobe unregistered\n");
}

module_init(kprobe_init);
module_exit(kprobe_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Jatin");
MODULE_DESCRIPTION("Char dev + kretprobe for PID-based post-page-fault page table tracing")
