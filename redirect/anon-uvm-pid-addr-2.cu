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

    // Defensive: Check if pointers are valid
    if (src == NULL || dest == NULL) {
        printf("CUDA kernel ERROR: Null pointer detected! src=%p dest=%p\n", src, dest);
        return;
    }

    // Defensive: Bounds check
    if (idx >= count) {
        printf("CUDA kernel WARNING: Thread index %d out of bounds (count=%zu)\n", idx, count);
        return;
    }

    dest[idx] = src[idx];
    printf("CUDA kernel: dest[%d] = %d\n", idx, dest[idx]);
}

int main() {
    printf("PID: %d\n", getpid());
    int dev = open("/dev/pidaddrinfo", O_WRONLY);
    if (dev < 0) {
        perror("open /dev/pidaddrinfo");
        return 1;
    }


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
    printf("--------------before redirect--------------------\n");
    // 6. Print anon memory
    printf("Anon contents:\n");
    for (int i = 0; i < SIZE; ++i)
        printf("%d ", anon[i]);
    printf("\n");
/*
    // 7. Print UVM1 contents on CPU (after redirect)
    printf("UVM1 contents (CPU):\n");
    for (int i = 0; i < SIZE; ++i)
        printf("%d ", uvm1[i]);
    printf("\n");
*/

    // 4. Print addresses
    printf("Anon addr : %p\n", anon);
    printf("UVM1 addr : %p\n", uvm1);
    printf("UVM2 addr : %p\n", uvm2);
    
    char buf1[256];
    snprintf(buf1, sizeof(buf1), "%d %s %lx", getpid(), "anon-before-memredir", anon);
    write(dev, buf1, strlen(buf1));
    
    char buf2[256];
    snprintf(buf2, sizeof(buf2), "%d %s %lx", getpid(), "uvm1-before-memredir", uvm1);
    write(dev, buf2, strlen(buf2));

    // 5. Pass info to kernel module
    int devm = open("/dev/memredir", O_WRONLY);
    if (devm < 0) {
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
    write(devm, buf, strlen(buf));
    close(devm);
    printf("-----------------after redirect------------------\n");
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


    char buf3[256];
    snprintf(buf3, sizeof(buf3), "%d %s %lx", getpid(), "anon-after-memredir", anon);
    write(dev, buf3, strlen(buf3));

    char buf4[256];
    snprintf(buf4, sizeof(buf4), "%d %s %lx", getpid(), "uvm1-after-memredir", uvm1);
    write(dev, buf4, strlen(buf4));

    // 8. Launch kernel to copy uvm1 → uvm2 and print
    copy_and_print<<<1, SIZE>>>(uvm1, uvm2, SIZE);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA ERROR: %s\n", cudaGetErrorString(err));
    }
    char buf5[256];
    snprintf(buf5, sizeof(buf5), "%d %s %lx", getpid(), "anon-after-cuda-kernel", anon);
    write(dev, buf5, strlen(buf5));

    char buf6[256];
    snprintf(buf6, sizeof(buf6), "%d %s %lx", getpid(), "uvm1-after-cuda-kernel", uvm1);
    write(dev, buf6, strlen(buf6));
    // 7. Print UVM1 contents on CPU (after redirect)
    printf("UVM1 contents (CPU):\n");
    for (int i = 0; i < SIZE; ++i)
        printf("%d ", uvm1[i]);
    printf("\n");
    // 9. Print UVM2 contents on CPU
    printf("UVM2 contents (CPU):\n");
    for (int i = 0; i < SIZE; ++i)
        printf("%d ", uvm2[i]);
    printf("\n");
    // 4. Print addresses
    printf("Anon addr : %p\n", anon);
    printf("UVM1 addr : %p\n", uvm1);
    printf("UVM2 addr : %p\n", uvm2);

    // 10. Cleanup
    munmap(anon, SIZE);
    cudaFree(uvm1);
    cudaFree(uvm2);

    return 0;
}
