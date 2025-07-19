#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

int main() {
    const char *filename = "int_array.bin";
    const int num_elements = 1000000;
    int fd;

    // Create file if it doesn't exist
    fd = open(filename, O_RDWR | O_CREAT, 0666);
    if (fd == -1) {
        perror("open");
        exit(EXIT_FAILURE);
    }

    // Ensure the file is large enough
    if (ftruncate(fd, num_elements * sizeof(int)) == -1) {
        perror("ftruncate");
        close(fd);
        exit(EXIT_FAILURE);
    }

    // Memory-map the file
    int *mapped_array = mmap(NULL, num_elements * sizeof(int),
                             PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped_array == MAP_FAILED) {
        perror("mmap");
        close(fd);
        exit(EXIT_FAILURE);
    }

    // Close the file descriptor as it's no longer needed
    close(fd);

    // Modify the memory-mapped array
    for (int i = 0; i < num_elements; i++) {
            //sleep(5);
	    //printf("%d\n",);
	    if ((i + 1) % 50 == 0)
            	mapped_array[i]= 10;
	    else
	    	mapped_array[i] = 100;  // Write values into the array
    }

    printf("Memory-mapped array updated. Check the file for changes.\n");

    // Optional: Sync changes to disk
    if (msync(mapped_array, num_elements * sizeof(int), MS_SYNC) == -1) {
        perror("msync");
    }

    // Unmap the memory
    if (munmap(mapped_array, num_elements * sizeof(int)) == -1) {
        perror("munmap");
        exit(EXIT_FAILURE);
    }

    return 0;
}
