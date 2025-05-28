#include <stdio.h>
#include "../common/common.h"
#include <cuda_runtime.h>

__global__ void checkIndex(void){

    printf("ThreadIdx: (%d,%d,%d) -- BlockIdx: (%d,%d,%d) -- BlockDim: (%d,%d,%d) -- GridDim: (%d,%d,%d)\n",
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z,
           blockDim.x, blockDim.y, blockDim.z,
           gridDim.x, gridDim.y, gridDim.z);
}

int main(int argc,char** argv){
    int N_elem = 6;
    dim3 block(3);
    dim3 grid((N_elem + block.x - 1) / block.x); // this is a tricky way to calculate the grid size for integer division
    printf("Grid size: (%d,%d,%d) -- Block size: (%d,%d,%d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);

    checkIndex<<<grid, block>>>();

    CHECK(cudaDeviceReset());

    return 0;
}