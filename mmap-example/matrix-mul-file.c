#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

typedef struct {
    int rows;
    int cols;
    int *data;
} Matrix;

void parse_matrix(char *file_content, Matrix *matrix) {
    char *ptr = file_content;
    
    // Read dimensions
    matrix->rows = atoi(ptr);
    while (*ptr != ' ') ptr++;
    ptr++;
    matrix->cols = atoi(ptr);
    while (*ptr != '\n') ptr++;
    ptr++;
    
    // Allocate memory for matrix data
    matrix->data = malloc(matrix->rows * matrix->cols * sizeof(int));
    
    // Read matrix values
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            while (*ptr == ' ' || *ptr == '\n') ptr++;
            matrix->data[i * matrix->cols + j] = atoi(ptr);
            while (*ptr != ' ' && *ptr != '\n' && *ptr != '\0') ptr++;
        }
    }
}

void matrix_multiply(const Matrix *a, Matrix *result) {
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result->data[i * a->cols + j] = 0;
            for (int k = 0; k < a->cols; k++) {
                result->data[i * a->cols + j] += 
                    a->data[i * a->cols + k] * a->data[k * a->cols + j];
            }
        }
    }
}

void write_result(char *file_content, const Matrix *matrix) {
    char *ptr = file_content;
    
    // Write dimensions
    int written = sprintf(ptr, "%d %d\n", matrix->rows, matrix->cols);
    ptr += written;
    
    // Write matrix data
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            written = sprintf(ptr, "%d ", matrix->data[i * matrix->cols + j]);
            ptr += written;
        }
	//sleep(2);
        *ptr = '\n';
        ptr++;
    }
    *ptr = '\0';
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <matrix_file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *filename = argv[1];
    int fd;
    struct stat sb;
    char *file_content;
    Matrix matrix, result;

    // Open file
    fd = open(filename, O_RDWR);
    if (fd == -1) {
        perror("open");
        return EXIT_FAILURE;
    }

    // Get file size
    if (fstat(fd, &sb) == -1) {
        perror("fstat");
        close(fd);
        return EXIT_FAILURE;
    }

    // Memory map the file
    file_content = mmap(NULL, sb.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (file_content == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return EXIT_FAILURE;
    }

    // Parse matrix from file content
    parse_matrix(file_content, &matrix);

    if (matrix.rows != matrix.cols) {
        fprintf(stderr, "Matrix must be square for A×A multiplication\n");
        munmap(file_content, sb.st_size);
        close(fd);
        free(matrix.data);
        return EXIT_FAILURE;
    }

    // Prepare result matrix
    result.rows = matrix.rows;
    result.cols = matrix.cols;
    result.data = malloc(result.rows * result.cols * sizeof(int));

    // Perform multiplication
    matrix_multiply(&matrix, &result);

    // Resize file if needed (for cases where result might need more space)
    size_t new_size = 0;
    // Calculate required size (simplified estimation)
    new_size = 20 + (matrix.rows * matrix.cols * 10); // 20 for header, 10 per number
    
    if (new_size > sb.st_size) {
        if (ftruncate(fd, new_size) == -1) {
            perror("ftruncate");
            munmap(file_content, sb.st_size);
            close(fd);
            free(matrix.data);
            free(result.data);
            return EXIT_FAILURE;
        }
        
        // Remap if we resized
        munmap(file_content, sb.st_size);
        file_content = mmap(NULL, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (file_content == MAP_FAILED) {
            perror("mmap after resize");
            close(fd);
            free(matrix.data);
            free(result.data);
            return EXIT_FAILURE;
        }
    }

    // Write result back to memory mapped file
    write_result(file_content, &result);

    // Clean up
    free(matrix.data);
    free(result.data);
    if (munmap(file_content, new_size > sb.st_size ? new_size : sb.st_size) == -1) {
        perror("munmap");
    }
    close(fd);

    printf("Matrix multiplication completed successfully. Result written back to file.\n");
    return EXIT_SUCCESS;
}
