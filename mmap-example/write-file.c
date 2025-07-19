#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <sys/stat.h>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <filename> <matrix_size>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *filename = argv[1];
    int size = atoi(argv[2]);
    if (size <= 0) {
        fprintf(stderr, "Matrix size must be positive.\n");
        return EXIT_FAILURE;
    }

    // Estimate file size: header + rows * (cols * (number + space) + newline)
    size_t estimated_size = 20 + size * (size * 2 + 1);  // conservative estimate

    // Open file
    int fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd == -1) {
        perror("open");
        return EXIT_FAILURE;
    }

    // Set the file size
    if (ftruncate(fd, estimated_size) == -1) {
        perror("ftruncate");
        close(fd);
        return EXIT_FAILURE;
    }

    // Memory map the file
    char *map = mmap(NULL, estimated_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return EXIT_FAILURE;
    }

    // Write matrix to memory-mapped file
    char *ptr = map;
    int written = sprintf(ptr, "%d %d\n", size, size);
    ptr += written;

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            written = sprintf(ptr, "0 ");
            ptr += written;
        }
        *ptr++ = '\n';
    }

    // Sync changes and cleanup
    if (msync(map, estimated_size, MS_SYNC) == -1) {
        perror("msync");
    }
    if (munmap(map, estimated_size) == -1) {
        perror("munmap");
    }
    close(fd);

    printf("Created %dx%d zero matrix in file: %s\n", size, size, filename);
    return EXIT_SUCCESS;
}
