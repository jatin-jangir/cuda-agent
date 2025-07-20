#include <linux/module.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/target_pid.h>  // Declares: extern pid_t target_pid;

#define DEVICE_NAME "pte_target"
static int major;
static struct class *cls;

// Extern declaration to use the global target_pid from memory.c
// You can also do this in target_pid.h
// extern pid_t target_pid;

static ssize_t write_pid(struct file *f, const char __user *buf, size_t len, loff_t *off)
{
    char kbuf[16] = {0};

    if (len > 15)
        return -EINVAL;

    if (copy_from_user(kbuf, buf, len))
        return -EFAULT;

    kbuf[len] = '\0';  // Null-terminate the string

    if (kstrtoint(kbuf, 10, &target_pid) == 0) {
        pr_info("Target PID updated to %d\n", target_pid);
        return len;
    }

    return -EINVAL;
}

static struct file_operations fops = {
    .owner = THIS_MODULE,
    .write = write_pid,
};

static int __init pte_target_init(void)
{
    major = register_chrdev(0, DEVICE_NAME, &fops);
    if (major < 0) {
        pr_err("Failed to register character device\n");
        return major;
    }

    cls = class_create("pteclass");
    if (IS_ERR(cls)) {
        unregister_chrdev(major, DEVICE_NAME);
        return PTR_ERR(cls);
    }

    device_create(cls, NULL, MKDEV(major, 0), NULL, DEVICE_NAME);

    pr_info("Char device /dev/%s initialized\n", DEVICE_NAME);
    return 0;
}

static void __exit pte_target_exit(void)
{
    device_destroy(cls, MKDEV(major, 0));
    class_destroy(cls);
    unregister_chrdev(major, DEVICE_NAME);
    pr_info("Char device /dev/%s removed\n", DEVICE_NAME);
}

module_init(pte_target_init);
module_exit(pte_target_exit);
MODULE_LICENSE("GPL");
