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
#include <ctype.h>

#define PORT 10000
#define MAX_MATRIX_SIZE 1024

__global__ void updateColumn(float* values, int rows, float increment) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        values[row] += increment;
    }
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

    // Parse matrix dimensions
    int rows, cols;
    sscanf(file_data, "%d %d", &rows, &cols);
    
    // Find all column positions in the file
    float *column_values;
    cudaMallocManaged(&column_values, rows * sizeof(float));
    int *value_positions = (int*)malloc(rows * sizeof(int));
    int *value_lengths = (int*)malloc(rows * sizeof(int));

    // Parse the file to find positions and initial values
    char *ptr = file_data;
    while (*ptr++ != '\n'); // Skip header line
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (j == 0) {
                value_positions[i] = ptr - file_data;
            }
            sscanf(ptr, "%f", &column_values[i]);
            value_lengths[i] = 0;
            while (*ptr != ' ' && *ptr != '\n' && *ptr != '\0') {
                ptr++;
                value_lengths[i]++;
            }
            ptr++; // Skip space or newline
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

        // 2. Take col_num input from socket
        recvfrom(sock, buffer, sizeof(buffer), 0, (struct sockaddr *)&peer, &inetSize);

        int col_num = atoi(buffer);
        if (col_num < 0 || col_num >= cols) {
            printf("Invalid column number: %d\n", col_num);
            continue;
        }

        // 3. Update the specified column using GPU
        printf("Updating column %d using CUDA\n", col_num);
        updateColumn<<<(rows + 255)/256, 256>>>(column_values, rows, 1.0f);
        cudaDeviceSynchronize();

        // 4. Update the file data
        for (int i = 0; i < rows; i++) {
            char num_str[32];
            int len = sprintf(num_str, "%.2f", column_values[i]);
            memcpy(file_data + value_positions[i], num_str, len);
        }

        // Ensure changes are written back to file
        msync(file_data, sb.st_size, MS_SYNC);

        // Send acknowledgment
        const char *ack = "Column updated successfully";
        sendto(sock, ack, strlen(ack), 0, (struct sockaddr *)&peer, inetSize);
    }

    // Cleanup
    munmap(file_data, sb.st_size);
    close(fd);
    cudaFree(column_values);
    free(value_positions);
    free(value_lengths);
    return 0;
}
