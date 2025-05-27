#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common.h"
#include <cstdio>

class GpuTimer {
private:
    cudaEvent_t startEvent, stopEvent;
public:
    GpuTimer() {
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
    }
    ~GpuTimer() {
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }
    void start() {
        cudaEventRecord(startEvent, 0);
    }
    void stop() {
        cudaEventRecord(stopEvent, 0);
    }
    float elapsed() {
        cudaEventSynchronize(stopEvent);
        float ms = 0;
        cudaEventElapsedTime(&ms, startEvent, stopEvent);
        return ms;
    }
};


__global__ void sumArrayGPU(float *a,float *b,float *c){
    int i  = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

void sumArrayCPU(float *a,float *b,float *c,int n){
    int i = 0;
    for(; i < n; i+=4){
        c[i] = a[i] + b[i];
        c[i+1] = a[i+1] + b[i+1];
        c[i+2] = a[i+2] + b[i+2];
        c[i+3] = a[i+3] + b[i+3];
    }
    for(; i < n; i++){
        c[i] = a[i] + b[i];
    }
}

int main(int argc,char** argv){

    int dev = 0;
    cudaSetDevice(dev);
    int N_elements = 1 << 20; // 1 million elements

    printf("Vector size is equal to %d\n", N_elements);

    int nBytes = N_elements * sizeof(float);
    float* a_h = (float*)malloc(nBytes);
    float* b_h = (float*)malloc(nBytes);
    float* c_h = (float*)malloc(nBytes);
    float* c_h_FromGpu = (float*)malloc(nBytes);

    memset(c_h, 0, nBytes);
    memset(c_h_FromGpu, 0, nBytes);

    float* a_d;
    float* b_d; 
    float* c_d;
    CHECK(cudaMalloc((float**)&a_d, nBytes));
    CHECK(cudaMalloc((float**)&b_d, nBytes));
    CHECK(cudaMalloc((float**)&c_d, nBytes));

    /*
    
    [ CPU 内存 ]
+------------------+
| 变量 a_d         |  ---> 指向设备内存（地址写入后）
+------------------+           ↓

                            [ GPU 内存 ]
                            +--------------------+
                            | malloc 的一块内存  | ←—— 分配 nBytes 大小
                            +--------------------+

    
    
    */

    initialData(a_h, N_elements);
    initialData(b_h, N_elements);

    CHECK(cudaMemcpy(a_d,a_h,nBytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_d,b_h,nBytes,cudaMemcpyHostToDevice));

    /*
    [ 主机内存 (CPU) ]         →          [ 设备内存 (GPU) ]
+-------------------+                 +-------------------+
| float a_h[nElem]  | --(memcpy)-->   | float a_d[nElem]  |
| float b_h[nElem]  | --(memcpy)-->   | float b_d[nElem]  |
+-------------------+                 +-------------------+
    */
    
    dim3 block(1024);
    dim3 grid(N_elements / block.x);

    /*
    
    Grid
├── Block 0 (threads 0 ~ 1023)
├── Block 1 (threads 1024 ~ 2047)
├── ...
└── Block 7 (threads 7168 ~ 8191)

    */

    GpuTimer timer;
    timer.start();
    sumArrayGPU<<<grid, block>>>(a_d, b_d, c_d);
    timer.stop();
    float gpu_time_ms = timer.elapsed();
    printf("GPU elapsed time: %f ms\n", gpu_time_ms);

    printf("GPU Configuration: %d blocks, %d threads per block\n", grid.x,block.x);

    CHECK(cudaMemcpy(c_h_FromGpu,c_d,nBytes,cudaMemcpyDeviceToHost));

    double start_cpu = cpuSecond();
    sumArrayCPU(a_h, b_h, c_h, N_elements);
    double end_cpu = cpuSecond();
    printf("CPU time elapsed: %f sec\n", end_cpu - start_cpu);


    checkResult(c_h,c_h_FromGpu,N_elements);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    free(a_h);
    free(b_h);
    free(c_h);
    free(c_h_FromGpu);

    return 0;

}
