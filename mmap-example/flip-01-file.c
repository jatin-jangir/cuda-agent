#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>

int main() {
    printf("%d\n",getpid());
    //sleep(30);
    const char *filename = "matrix.txt";
    int fd = open(filename, O_RDWR);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("fstat");
        close(fd);
        return 1;
    }

    size_t filesize = st.st_size;
    char *data = mmap(NULL, filesize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    int rows = 0, cols = 0;
    char *ptr = data;
    printf("%dx%d matrix\n",rows,cols);
    // Parse rows and cols
    sscanf(ptr, "%d %d", &rows, &cols);

    // Move pointer to beginning of matrix data
    while (*ptr != '\n') ptr++;
    ptr++;
    printf("started fliping\n");

    // Traverse and flip bits
    for (int i = 0; i < rows * cols; ) {
        if (*ptr == '1') {
            *ptr = '0';
            i++;
        } else if (*ptr == '0') {
            *ptr = '1';
            i++;
        }
        ptr++;
	//printf("%d \n",i);
//	sleep(2);
    }

    // Optional: flush changes
    msync(data, filesize, MS_SYNC);

    // Clean up
    munmap(data, filesize);
    close(fd);

    return 0;
}
