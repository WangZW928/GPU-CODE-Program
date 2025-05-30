#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char** argv){
    int N_elem = 1024;

    dim3 block(256);
    dim3 grid((N_elem + block.x - 1) / block.x); // Calculate grid size for integer division
    printf("Grid size: (%d,%d,%d) -- Block size: (%d,%d,%d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);

    block.x = 512;
    grid.x = (N_elem + block.x - 1) / block.x; // Recalculate grid size after changing block size
    printf("Updated Grid size: (%d,%d,%d) -- Block size: (%d,%d,%d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);

    block.x = 256;
    grid.x = (N_elem + block.x - 1) / block.x; // Recalculate grid size again
    printf("Final Grid size: (%d,%d,%d) -- Block size: (%d,%d,%d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);

    block.x = 128;
    grid.x = (N_elem + block.x - 1) / block.x; // Recalculate grid size again
    printf("Final Grid size: (%d,%d,%d) -- Block size: (%d,%d,%d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
    
    CHECK(cudaDeviceReset());
    return 0;
}