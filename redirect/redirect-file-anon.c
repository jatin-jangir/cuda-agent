#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>
#include <string.h>

int main() {
    printf("%d\n",getpid());
    int fd = open("test.txt", O_RDONLY);
    struct stat sb;
    fstat(fd, &sb);

    char *file_mapped = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
char *ptr = mmap(NULL, sb.st_size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    printf("File mapped addr: %p\n", file_mapped);
    printf("Malloc'd addr   : %p\n", ptr);

    int dev = open("/dev/memredir", O_WRONLY);
    char buf[256];
    snprintf(buf, sizeof(buf), "%d %lx %lx %lx %lx", getpid(),
             (unsigned long)file_mapped,
             (unsigned long)ptr,
             (unsigned long)(file_mapped + sb.st_size),
             (unsigned long)(ptr + sb.st_size));

    write(dev, buf, strlen(buf));
    printf("file data\n");
    for(int i=0;i<10;i++){
	    printf("%c ",file_mapped[i]);
	}
    printf("\n");
    printf("array data\n");
    for(int i=0;i<10;i++){
            printf("%c ",ptr[i]);
        }
    printf("\n");

    close(dev);
    close(fd);
    munmap(file_mapped, sb.st_size);
    munmap(ptr, sb.st_size);

}
