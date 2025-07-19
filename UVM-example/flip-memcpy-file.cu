#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>

#define MAX_ROWS 1024
#define MAX_COLS 1024

__global__ void flipBits(char* data, int* row_offsets, int cols, int row) {
    // Calculate position in the file data for this row
    char* row_start = data + row_offsets[row];
    
    // Each thread handles one column
    int col = threadIdx.x;
    if (col >= cols) return;
    
    // Find the start of our number (skip spaces)
    char* num_start = row_start;
    for (int i = 0; i < col; i++) {
        while (*num_start == ' ') num_start++;
        while (*num_start != ' ' && *num_start != '\n') num_start++;
    }
    while (*num_start == ' ') num_start++;
    
    // Flip the bit (0->1 or 1->0)
    if (*num_start == '0') {
        *num_start = '1';
    } else if (*num_start == '1') {
        *num_start = '0';
    }
}

int main() {
    // 1. Open and mmap the file
    int fd = open("matrix.txt", O_RDWR);
    if (fd == -1) {
        perror("Failed to open file");
        return 1;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("Failed to get file size");
        close(fd);
        return 1;
    }

    // Map the file with MAP_SHARED so changes propagate to disk
    char* file_data = (char*)mmap(NULL, sb.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (file_data == MAP_FAILED) {
        perror("Failed to mmap file");
        close(fd);
        return 1;
    }

    // 2. Parse matrix dimensions
    int rows, cols;
    sscanf(file_data, "%d %d", &rows, &cols);
    
    // 3. Create a managed copy of the file data
    char* uvm_data;
    cudaMallocManaged(&uvm_data, sb.st_size);
    memcpy(uvm_data, file_data, sb.st_size);
    
    // 4. Find row offsets (positions where each row starts)
    int* d_row_offsets;
    int row_offsets[MAX_ROWS];
    char* ptr = uvm_data;
    while (*ptr++ != '\n'); // Skip first line
    
    for (int i = 0; i < rows; i++) {
        row_offsets[i] = ptr - uvm_data;
        while (*ptr++ != '\n');
    }
    
    // Allocate managed memory for row offsets
    cudaMallocManaged(&d_row_offsets, rows * sizeof(int));
    memcpy(d_row_offsets, row_offsets, rows * sizeof(int));

    // 5. Flip bits row by row using CUDA
    for (int row = 0; row < rows; row++) {
        flipBits<<<1, cols>>>(uvm_data, d_row_offsets, cols, row);
        cudaDeviceSynchronize();
        
        // Copy the modified data back to the file mapping
        memcpy(file_data, uvm_data, sb.st_size);
        
        // Flush changes to disk
        msync(file_data, sb.st_size, MS_SYNC);
        
        printf("Updated row %d\n", row);
        sleep(20);
    }

    // 6. Cleanup
    munmap(file_data, sb.st_size);
    close(fd);
    cudaFree(d_row_offsets);
    cudaFree(uvm_data);

    printf("Bit flipping completed successfully!\n");
    return 0;
}
