#include <iostream>
#include <cuda_runtime.h>

__global__ void init_kernel(float *data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) data[i] = i;
}

int main() {
    const int N = 1<<20;  // 1 million elements
    float *managed_data;
    
    // 1. Unified memory allocation
    cudaMallocManaged(&managed_data, N*sizeof(float));  // [2]

    // 2. First touch on CPU
    for(int i=0; i<N; i++) managed_data[i] = 0.0f;  // Page faults occur here [4]
    std::cout << "CPU initialization complete\n";

    // 3. Kernel launch without prefetch
    init_kernel<<<256,256>>>(managed_data, N);  // GPU page faults occur [5]
    cudaDeviceSynchronize();
    std::cout << "GPU initialization complete\n";

    // 4. Access modified data on CPU
    float sum = 0;
    for(int i=0; i<N; i++) sum += managed_data[i];  // Page faults migrate back [1]
    std::cout << "Data sum: " << sum << "\n";

    cudaFree(managed_data);
    return 0;
}
