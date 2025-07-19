#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void modifyData(int* data, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
	    if ((idx + 1) % 50 == 0)
            	data[idx]= 10;
	    else 
		data[idx] += 1;  // Example modification
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <binary_file>" << std::endl;
        return 1;
    }

    // 1. Open and mmap the file
    int fd = open(argv[1], O_RDWR);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    size_t file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET); // optional
    int* mmapped_data = (int*)mmap(NULL, file_size,
                                   PROT_READ | PROT_WRITE,
                                   MAP_SHARED, fd, 0);
    if (mmapped_data == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    size_t num_ints = file_size / sizeof(int);

    // 2. Register mmap memory with CUDA
    cudaError_t err = cudaHostRegister(mmapped_data, file_size, cudaHostRegisterMapped);
    if (err != cudaSuccess) {
        std::cerr << "cudaHostRegister failed: " << cudaGetErrorString(err) << std::endl;
        munmap(mmapped_data, file_size);
        close(fd);
        return 1;
    }

    // 3. Get device pointer for mmap memory
    int* device_ptr = nullptr;
    cudaHostGetDevicePointer(&device_ptr, mmapped_data, 0);

    // 4. Launch kernel directly on the mapped memory in chunks
    const size_t chunk_size = 1 << 2;  // 1M elements per chunk
    for (size_t offset = 0; offset < num_ints; offset += chunk_size) {
        size_t current_chunk = std::min(chunk_size, num_ints - offset);

        dim3 threads(256);
        dim3 blocks((current_chunk + threads.x - 1) / threads.x);
        modifyData<<<blocks, threads>>>(device_ptr + offset, current_chunk);
        cudaDeviceSynchronize();
	printf("%d\n",offset);
    	sleep(5);
    }

    // 5. Sync changes to disk
    msync(mmapped_data, file_size, MS_SYNC);

    // 6. Cleanup
    cudaHostUnregister(mmapped_data);   // unregister before munmap
    munmap(mmapped_data, file_size);
    close(fd);

    return 0;
}
