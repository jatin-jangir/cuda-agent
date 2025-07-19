#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Kernel to set all elements to 1
__global__ void setToOnes(int *array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx>2 && idx < size) {
        array[idx] = 1;
    }
}

int main(int argc, char **argv) {
    int size = 1024000; // Array size
    int *array = NULL;
    
    // 1. Allocate memory using malloc and initialize to 0
cudaMallocManaged(&array,size*sizeof(int))   ;
    if (array == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return 1;
    }
    
    // Initialize to 0
    for (int i = 0; i < size; i++) {
        array[i] = 0;
    }
    
    // Verify initialization
    printf("Before UVM (first 5 elements): ");
    for (int i = 0; i < 5; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
    
    
    // 3. Launch kernel to set all elements to 1
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    setToOnes<<<blocksPerGrid, threadsPerBlock>>>(array, size);
    
    // 4. Verify results
    sleep(0.5);
    printf("After UVM (first 5 elements): ");
    for (int i = 0; i < 5; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaHostUnregister(array);
      //  free(array);
        return 1;
    }
    // Check all elements were set to 1
    bool success = true;
    for (int i = 3; i < size; i++) {
        if (array[i] != 1) {
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("All elements were successfully set to 1(except 1st 3 element)!\n");
    } else {
        printf("Error: Not all elements were set to 1\n");
    }
    
    // Cleanup
    cudaHostUnregister(array);
    //free(array);
    
    return 0;
}
