#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <string.h>
#include <ctype.h>

int main(int argc, char *argv[]) {
    int fd;
    char *mapped_data;
    struct stat file_info;
    const char *filename = "data.txt";
    
    // Check if filename is provided as argument
    if (argc > 1) {
        filename = argv[1];
    }
    
    // Open the file for reading
    fd = open(filename, O_RDONLY);
    if (fd == -1) {
        perror("Error opening file");
        return EXIT_FAILURE;
    }
    
    // Get file size using fstat
    if (fstat(fd, &file_info) == -1) {
        close(fd);
        perror("Error getting file info");
        return EXIT_FAILURE;
    }
    
    // Check if file is empty
    if (file_info.st_size == 0) {
        printf("File is empty\n");
        close(fd);
        return EXIT_SUCCESS;
    }
    
    // Map the file into memory
    mapped_data = (char*)mmap(NULL, file_info.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped_data == MAP_FAILED) {
        close(fd);
        perror("Error mapping file");
        return EXIT_FAILURE;
    }
    
    // Close file descriptor (not needed after mapping)
    close(fd);
    
    // Parse numbers from the file
    printf("\nParsed numbers:\n");
    char number_buffer[32];
    int buffer_index = 0;
    int number_count = 0;
    
    for (off_t i = 0; i < file_info.st_size; i++) {
        char current_char = mapped_data[i];
        
        // If character is a digit, minus sign, or decimal point
        if (isdigit(current_char) || current_char == '-' || current_char == '.') {
            if (buffer_index < sizeof(number_buffer) - 1) {
                number_buffer[buffer_index++] = current_char;
            }
        }
        // If we hit whitespace or end of file, process the number
        else if (buffer_index > 0) {
            number_buffer[buffer_index] = '\0';
            ++number_count;
            buffer_index = 0;
        }
    }
    
    // Handle last number if file doesn't end with whitespace
    if (buffer_index > 0) {
        number_buffer[buffer_index] = '\0';
        ++number_count;
    }
    
    printf("Total numbers found: %d\n", number_count);
    
    // Unmap the memory
    if (munmap(mapped_data, file_info.st_size) == -1) {
        perror("Error unmapping file");
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
