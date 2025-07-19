#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel to initialize array elements
__global__ void initArray(int *arr, int value, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        arr[idx] = value;
    }
}

int main() {
    const int N = 10;
    int *data;

    // Allocate unified memory accessible by both CPU and GPU
    cudaError_t err = cudaMallocManaged(&data, N * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "cudaMallocManaged failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Launch kernel with 1 block and N threads
    initArray<<<1, N>>>(data, 42, N);

    // Wait for GPU to finish before accessing data on the host
    cudaDeviceSynchronize();

    // Print the array from host
    std::cout << "Array contents:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    // Free unified memory
    cudaFree(data);

    return 0;
}
