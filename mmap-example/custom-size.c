#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <number_of_elements>\n", argv[0]);
        printf("Example: %s 1000000  (for 1 million integers)\n", argv[0]);
        return 1;
    }
    
    // Parse command line argument for number of elements
    long num_elements = strtol(argv[1], NULL, 10);
    if (num_elements <= 0) {
        printf("Error: Number of elements must be a positive number\n");
        return 1;
    }
    
    // Calculate total size in bytes
    size_t array_size_bytes = num_elements * sizeof(int);
    double array_size_mb = (double)array_size_bytes / (1024.0 * 1024.0);
    
    printf("Allocating array with %ld elements\n", num_elements);
    printf("Total memory: %.2f MB (%zu bytes)\n", array_size_mb, array_size_bytes);
    
    // Create anonymous mmap allocation
    int *large_array = mmap(NULL, array_size_bytes,
                           PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS,
                           -1, 0);
    
    if (large_array == MAP_FAILED) {
        perror("mmap failed");
        return 1;
    }
    
    printf("Successfully mapped memory at address: %p\n", large_array);
    
    // Touch pages to force allocation (write every 1024 elements ~ 4KB page)
   printf("Writing to array to trigger page faults...\n");
//    size_t page_stride = 1024;  // Approximately 4KB per page for integers
    for (size_t i = 0; i < num_elements; i += 1) {
        large_array[i] = i;
    }
    
    // Access pattern that may cause major faults under memory pressure
    printf("Reading from array in multiple passes...\n");
    for (size_t i = 0; i < num_elements; i += 1) {
        int val = large_array[i];
	val++;
    } 
    printf("Cleaning up and unmapping memory...\n");
    munmap(large_array, array_size_bytes);
    
    printf("Program completed successfully\n");
    return 0;
}
