#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
int main() {
    int dev = open("/dev/pidaddrinfo", O_WRONLY);
    if (dev < 0) {
        perror("open /dev/pidaddrinfo");
        return 1;
    }

    char buf[256];
    unsigned long address = (unsigned long)&buf; // some valid address
    snprintf(buf, sizeof(buf), "%d %s %lx", getpid(), "myprefix", address);

    write(dev, buf, strlen(buf));
    
    char buf2[256];
    unsigned long address2 = (unsigned long)&buf2; // some valid address
    snprintf(buf2, sizeof(buf2), "%d %s %lx", getpid(), "myprefix2", address2);

    write(dev, buf2, strlen(buf2));

    close(dev);
    return 0;
}
