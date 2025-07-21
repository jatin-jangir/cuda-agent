#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void copy_to_device_array(char *src, char *dest, size_t count) {
    int idx = threadIdx.x;
    if (idx < count) {
        dest[idx] = src[idx];
        printf("Copied src[%d] = %c to dest[%d]\n", idx, src[idx], idx);
    }
}



int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <file to map>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    printf("PID: %d\n", getpid());

    // Open file and get size
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) {
        perror("open");
        exit(EXIT_FAILURE);
    }

    struct stat sb;
    if (fstat(fd, &sb) < 0) {
        perror("fstat");
        exit(EXIT_FAILURE);
    }

    size_t file_size = sb.st_size;

    // mmap file as read-only
    char *file_mapped = (char *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (file_mapped == MAP_FAILED) {
        perror("mmap file");
        exit(EXIT_FAILURE);
    }

    // Allocate unified memory using cudaMallocManaged
    char *cuda_ptr;
    cudaError_t err = cudaMallocManaged(&cuda_ptr, file_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Print addresses
    printf("File mapped addr: %p\n", file_mapped);
    printf("CUDA managed addr: %p\n", cuda_ptr);

    // Notify kernel module (optional part, unchanged)
    int dev = open("/dev/memredir", O_WRONLY);
    if (dev < 0) {
        perror("open /dev/memredir");
        exit(EXIT_FAILURE);
    }

    char buf[256];
    snprintf(buf, sizeof(buf), "%d %lx %lx %lx %lx",
             getpid(),
             (unsigned long)file_mapped,
             (unsigned long)cuda_ptr,
             (unsigned long)(file_mapped + file_size),
             (unsigned long)(cuda_ptr + file_size));

    write(dev, buf, strlen(buf));
    close(dev);
    close(fd);

    // Show contents of file
    printf("File data:\n");
    for (int i = 0; i < 10 && i < file_size; i++) {
        printf("%c ", file_mapped[i]);
    }
    printf("\n");

    // Access CUDA-managed memory from CPU
    printf("Redirected array data (initially):\n");
    for (int i = 0; i < 10 && i < file_size; i++) {
        printf("%c ", cuda_ptr[i]); // Should show uninitialized or garbage values initially
	//cuda_ptr[i]='1';
    }
    printf("\n");

    const int copy_size = 10;
    char *copy_array;
    cudaMallocManaged(&copy_array, copy_size);

    // Launch kernel to copy data
    copy_to_device_array<<<1, copy_size>>>(cuda_ptr, copy_array, copy_size);

    // Wait for kernel to complete
    cudaDeviceSynchronize();

    // Print copied array
    printf("Copied array from CUDA kernel:\n");
    for (int i = 0; i < copy_size && i < file_size; i++) {
        printf("%c ", copy_array[i]);
    }
    printf("\n");

    // Cleanup
    munmap(file_mapped, file_size);
    cudaFree(cuda_ptr);
    cudaFree(copy_array);

    return 0;
}
