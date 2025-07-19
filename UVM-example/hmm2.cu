#include <stdio.h>
#include <cuda_runtime.h>
// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
// CUDA kernel to copy data from malloc array to cudaMallocManaged array
__global__ void copyKernel(int* src, int* dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {  // Only print once from first thread
        printf("GPU Kernel - Source array address: %p\n", (void*)src);
        printf("GPU Kernel - Destination array address: %p\n", (void*)dst);
        printf("GPU Kernel - First element of src: %d at address: %p\n",
               src[0], (void*)&src[0]);
    }
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

int main() {
    const int size = 10;
    const int bytes = size * sizeof(int);
   
   int access;
cudaDeviceGetAttribute(&access, cudaDevAttrPageableMemoryAccess, 0);
printf("HMM support: %s\n", access ?"Yes":"No");

    // 1. Allocate host memory using malloc
    int* h_array = (int*)malloc(bytes);
    
    // Initialize the malloc array
    for (int i = 0; i < size; i++) {
        h_array[i] = 100;  // Some initial values
    }
    
    // 2. Allocate unified memory using cudaMallocManaged
    int* managed_array;
    cudaMallocManaged(&managed_array, bytes);
    
    // Initialize the managed array
    for (int i = 0; i < size; i++) {
        managed_array[i] = 0;  // Initialize to zeros
    }
     printf("CPU - Malloc array address: %p\n", (void*)h_array);
    printf("CPU - Managed array address: %p\n", (void*)managed_array);
    printf("CPU - First element of malloc array: %d at address: %p\n", 
           h_array[0], (void*)&h_array[0]);
    printf("CPU - First element of managed array: %d at address: %p\n", 
           managed_array[0], (void*)&managed_array[0]);
    // Print initial values
    printf("Initial values:\n");
    printf("Malloc array: ");
    for (int i = 0; i < 20; i++) {
        printf("%d ", h_array[i]);
    }
    printf("\n");
    
    printf("Managed array: ");
    for (int i = 0; i < 20; i++) {
        printf("%d ", managed_array[i]);
    }
    printf("\n\n");

    // Check if system supports direct host memory access
    int pageableMemoryAccess;
    CUDA_CHECK(cudaDeviceGetAttribute(&pageableMemoryAccess, 
                                    cudaDevAttrPageableMemoryAccess, 0));
    printf("System supports pageable memory access: %s\n", 
           pageableMemoryAccess ? "Yes" : "No");
    
    // 5. Launch the kernel to copy from d_array to managed_array
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    copyKernel<<<blocksPerGrid, threadsPerBlock>>>(h_array, managed_array, size);
    
   // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // 6. Synchronize to make sure the kernel is finished
    CUDA_CHECK(cudaDeviceSynchronize()); 
    
    // 7. Print the results
    printf("After kernel execution:\n");
    printf("Malloc array: ");
    for (int i = 0; i < 20; i++) {
        printf("%d ", h_array[i]);
    }
    printf("\n");
    
    printf("Managed array: ");
    for (int i = 0; i < 20; i++) {
        printf("%d ", managed_array[i]);
    }
    printf("\n");
    
    // 8. Free memory
    free(h_array);
//    cudaFree(d_array);
        CUDA_CHECK(cudaFree(managed_array));

    
    return 0;
}
