jatin@test1:~/redirect/test-4$ cat 3.cu 
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <cuda_runtime.h>

#define SIZE 10

__global__ void copy_and_print(char *src, char *dest, size_t count) {
    int idx = threadIdx.x;
    if (idx < count) {
        dest[idx] = src[idx];
        printf("CUDA kernel: dest[%d] = %d\n", idx, dest[idx]);
    }
}

int main() {
    printf("PID: %d\n", getpid());

    // 1. Create anonymous memory using mmap
    char *anon = (char *)mmap(NULL, SIZE, PROT_READ | PROT_WRITE,
                              MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (anon == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    // 2. Fill with alternating 0 and 1
    for (int i = 0; i < SIZE; ++i)
        anon[i] = i % 2;

    // 3. Allocate CUDA UVM buffers
    char *uvm1, *uvm2;
    if (cudaMallocManaged(&uvm1, SIZE) != cudaSuccess ||
        cudaMallocManaged(&uvm2, SIZE) != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed\n");
        return 1;
    }

    // 4. Print addresses
    printf("Anon addr : %p\n", anon);
    printf("UVM1 addr : %p\n", uvm1);
    printf("UVM2 addr : %p\n", uvm2);

    // 5. Pass info to kernel module
    int dev = open("/dev/memredir", O_WRONLY);
    if (dev < 0) {
        perror("open /dev/memredir");
        return 1;
    }

    char buf[256];
    snprintf(buf, sizeof(buf), "%d %lx %lx %lx %lx",
             getpid(),
             (unsigned long)anon,
             (unsigned long)uvm1,
             (unsigned long)(anon + SIZE),
             (unsigned long)(uvm1 + SIZE));
    write(dev, buf, strlen(buf));
    close(dev);

    // 6. Print anon memory
    printf("Anon contents:\n");
    for (int i = 0; i < SIZE; ++i)
        printf("%d ", anon[i]);
    printf("\n");

    // 7. Print UVM1 contents on CPU (after redirect)
    printf("UVM1 contents (CPU):\n");
    for (int i = 0; i < SIZE; ++i)
        printf("%d ", uvm1[i]);
    printf("\n");

    // 8. Launch kernel to copy uvm1 → uvm2 and print
    copy_and_print<<<1, SIZE>>>(uvm1, uvm2, SIZE);
    cudaDeviceSynchronize();

    // 9. Print UVM2 contents on CPU
    printf("UVM2 contents (CPU):\n");
    for (int i = 0; i < SIZE; ++i)
        printf("%d ", uvm2[i]);
    printf("\n");

    // 10. Cleanup
    munmap(anon, SIZE);
    cudaFree(uvm1);
    cudaFree(uvm2);

    return 0;
}
