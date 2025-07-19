#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <chrono>  // For timing

// CUDA kernel to flip bits (0 ↔ 1)
__global__ void flipBits(int* data, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (data[idx] == 0)
            data[idx] = 1;
        else if (data[idx] == 1)
            data[idx] = 0;
    }
}

int main(int argc, char** argv) {
    auto start_time = std::chrono::high_resolution_clock::now();  // Start timer

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <binary_file>" << std::endl;
        return EXIT_FAILURE;
    }

    const char* filename = argv[1];

    // Open file
    int fd = open(filename, O_RDWR);
    if (fd < 0) {
        perror("open");
        return EXIT_FAILURE;
    }

    // Determine file size
    size_t file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    if (file_size % sizeof(int) != 0) {
        std::cerr << "Error: File size is not a multiple of sizeof(int)" << std::endl;
        close(fd);
        return EXIT_FAILURE;
    }

    // Memory-map the file
    int* mmapped_data = (int*)mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mmapped_data == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return EXIT_FAILURE;
    }

    size_t num_ints = file_size / sizeof(int);
std::cout << "\nFirst 20 integers from mapped pointer before kernel:\n";
    for (int i = 0; i < 20 && i < num_ints; i++) {
        std::cout << mmapped_data[i] << " ";
        if ((i + 1) % 10 == 0) std::cout << "\n";
    }
    std::cout << "\n";

    // Launch kernel in chunks
    const size_t chunk_size = 1 << 10; 
    for (size_t offset = 0; offset < num_ints; offset += chunk_size) {
        size_t current_chunk = std::min(chunk_size, num_ints - offset);

        dim3 threads(256);
        dim3 blocks((current_chunk + threads.x - 1) / threads.x);
        flipBits<<<blocks, threads>>>(mmapped_data + offset, current_chunk);
        cudaDeviceSynchronize();  // Wait for kernel
    }

    std::cout << "\nFirst 20 integers from mapped pointer after kernel:\n";
    for (int i = 0; i < 20 && i < num_ints; i++) {
        std::cout << mmapped_data[i] << " ";
        if ((i + 1) % 10 == 0) std::cout << "\n";
    }
    std::cout << "\n";

    // Flush changes to disk
    msync(mmapped_data, file_size, MS_SYNC);

    // Cleanup
    munmap(mmapped_data, file_size);
    close(fd);

    auto end_time = std::chrono::high_resolution_clock::now();  // End timer
    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << "Bit flipping completed successfully." << std::endl;
    std::cout << "Total execution time: " << duration.count() << " seconds." << std::endl;

    return EXIT_SUCCESS;
}
