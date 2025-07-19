#include <sys/mman.h>
#include <stddef.h>
#define ARRAY_SIZE 100  // ~400B for integers

int main() {
    int *arr = mmap(NULL, 
                   sizeof(int) * ARRAY_SIZE,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS,
                   -1, 0);
    
    // Access array elements
    for(int i=0;i<ARRAY_SIZE;i++)arr[i] = 0;  // Triggers page fault (write)
    
    munmap(arr, sizeof(int) * ARRAY_SIZE);
    return 0;
}
