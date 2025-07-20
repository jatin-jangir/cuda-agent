#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <string.h>
#include <errno.h>

#define DATA_SIZE 512  // Size of the UVM memory region
#define DELAY_US 10000  // Delay between accesses in microseconds
#define DEVICE_PATH "/dev/pte_target"

int main() {

    // Print PID
    printf("My PID: %d\n", getpid());

    // Send PID to kernel module
    int dev_fd = open(DEVICE_PATH, O_WRONLY);
    if (dev_fd < 0) {
        perror("Failed to open device");
        return 1;
    }

    char pid_str[16];
    snprintf(pid_str, sizeof(pid_str), "%d", getpid());
    if (write(dev_fd, pid_str, strlen(pid_str)) < 0) {
        perror("Failed to write PID to device");
        close(dev_fd);
        return 1;
    }
    close(dev_fd);

    // Allocate UVM memory
    char *uvm_ptr;
cudaError_t err = cudaMallocManaged((void **)&uvm_ptr, DATA_SIZE, cudaMemAttachGlobal);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate UVM memory: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Mapped address: %p\n", (void *)uvm_ptr);
    // Initialize the UVM memory with some data
    for (int i = 0; i < DATA_SIZE; i++) {
        uvm_ptr[i] = (char)(i % 256);  // Simple pattern
    }

    // Flush writes to make sure they're visible
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to synchronize: %s\n", cudaGetErrorString(err));
        cudaFree(uvm_ptr);
        return 1;
    }

    // Read back the UVM memory from CPU
    printf("UVM memory contents:\n");
    for (int i = 0; i < DATA_SIZE; i++) {
        printf("%02x ", (unsigned char)uvm_ptr[i]);
        if ((i + 1) % 16 == 0) {
            printf("\n");
        }
        fflush(stdout);  // Ensure output is visible immediately
        usleep(DELAY_US); // Small delay to potentially trigger page faults
    }
    printf("\n");

    // Cleanup
    err = cudaFree(uvm_ptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free UVM memory: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
