#include "../common/common.h"
#include <stdio.h>

__global__ void HelloWorldGPU(void){

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bdim = blockDim.x;
    printf("Hello World from GPU!, the current block is %d, the current thread is %d, and the block dimension is %d\n", bid, tid, bdim);
}


int main(int argc, char **argv){
    printf("Hello World from CPU!\n");
    HelloWorldGPU<<<3,4>>>();
    CHECK(cudaDeviceReset());
    return 0;
}