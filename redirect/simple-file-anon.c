#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

#define SIZE (8 * 1024) // 8KB
#define FILENAME "mmap_test_file"

int main() {
    // Create test file
    int fd = open(FILENAME, O_RDWR | O_CREAT, 0666);
    if (fd == -1) {
        perror("open");
        exit(EXIT_FAILURE);
    }
    ftruncate(fd, SIZE);

    // Create mappings
    void *file_map = mmap(NULL, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    void *anon_map = mmap(NULL, SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    
    if (file_map == MAP_FAILED || anon_map == MAP_FAILED) {
        perror("mmap");
        exit(EXIT_FAILURE);
    }

    printf("Process PID: %d\n", getpid());
    printf("File mapping: %p-%p\n", file_map, (char*)file_map + SIZE);
    printf("Anonymous mapping: %p-%p\n", anon_map, (char*)anon_map + SIZE);

    // Register with kernel module
    FILE *dev = fopen("/dev/pfmonitor", "w");
    if (!dev) {
        perror("fopen /dev/pfmonitor");
        exit(EXIT_FAILURE);
    }
    fprintf(dev, "%d %lx %d %lx %d", getpid(), 
            (unsigned long)file_map, SIZE,
            (unsigned long)anon_map, SIZE);
    fclose(dev);

    // Trigger page faults
    for (int i = 0; i < SIZE; i += 1) {
        ((volatile char*)file_map)[i] = 0xAA;
//        ((volatile char*)anon_map)[i] = 0xBB;
    }



    for (int i = 0; i < SIZE; i += 1) {
//        ((volatile char*)file_map)[i] = 0xAA;
        ((volatile char*)anon_map)[i] = 0xBB;
    }


    printf("Triggered page faults. Check dmesg output.\n");

    // Cleanup
    munmap(file_map, SIZE);
    munmap(anon_map, SIZE);
    close(fd);
    remove(FILENAME);
    return 0;
}
