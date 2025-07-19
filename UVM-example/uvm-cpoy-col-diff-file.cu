#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string.h>

#define PORT 10000
#define MAX_MATRIX_SIZE 1024

__global__ void updateColumn(float* matrix, int cols, int col_num, float value) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread updates its own row in the specified column
    matrix[row * cols + col_num] += value;
}

int main(void) {
    cudaFree(0);

    // 1. Read matrix from data.txt using mmap
    int fd = open("data.txt", O_RDWR);
    if (fd == -1) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    struct stat sb;
    if (fstat(fd, &sb)) {
        perror("Error getting file size");
        close(fd);
        exit(EXIT_FAILURE);
    }

    char *file_data = (char *)mmap(NULL, sb.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (file_data == MAP_FAILED) {
        perror("Error mmapping the file");
        close(fd);
        exit(EXIT_FAILURE);
    }

    // Parse matrix dimensions and data
    int rows, cols;
    sscanf(file_data, "%d %d", &rows, &cols);
    
    // 2. Create UVM array for the matrix
    float *uvm_matrix;
    cudaMallocManaged(&uvm_matrix, rows * cols * sizeof(float));
    
    // Skip the first line (dimensions)
    char *data_ptr = file_data;
    while (*data_ptr++ != '\n');
    
    // Fill the UVM matrix
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sscanf(data_ptr, "%f", &uvm_matrix[i * cols + j]);
            while (*data_ptr != ' ' && *data_ptr != '\n' && *data_ptr != '\0') data_ptr++;
            data_ptr++;
        }
    }

    // Set up socket
    int sock = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    bind(sock, (struct sockaddr *)&addr, sizeof(addr));

    while (1) {
        char buffer[16] = {0};
        struct sockaddr_in peer = {0};
        socklen_t inetSize = sizeof(peer);

        // 3. Take col_num input from socket
        recvfrom(sock, buffer, sizeof(buffer), 0, (struct sockaddr *)&peer, &inetSize);
        
        int col_num = atoi(buffer);
        if (col_num < 0 || col_num >= cols) {
            printf("Invalid column number: %d\n", col_num);
            continue;
        }

        // 4. Update each element of that column in UVM memory
        // Each thread will update its own row in the specified column
        float increment_value = 1.0f; // You can change this or receive from socket
	printf("updating array using cuda\n");
        updateColumn<<<(rows + 255)/256, 256>>>(uvm_matrix, cols, col_num, increment_value);
        cudaDeviceSynchronize();
	sleep(5);

        // 5. Update the file
        // First update the in-memory file data
        data_ptr = file_data;
        while (*data_ptr++ != '\n'); // Skip dimensions line
        printf("updating file\n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                char num_str[32];
                int len = sprintf(num_str, "%.2f", uvm_matrix[i * cols + j]);
                memcpy(data_ptr, num_str, len);
                data_ptr += len;
                *data_ptr++ = (j == cols - 1) ? '\n' : ' ';
		sleep(2);
            }
        }
        
        // Ensure changes are written back to file
        msync(file_data, sb.st_size, MS_SYNC);
        
        // Send acknowledgment
        const char *ack = "Column updated successfully";
        sendto(sock, ack, strlen(ack), 0, (struct sockaddr *)&peer, inetSize);
    }

    // Cleanup (unreachable in this example due to infinite loop)
    munmap(file_data, sb.st_size);
    close(fd);
    cudaFree(uvm_matrix);
    return 0;
}
