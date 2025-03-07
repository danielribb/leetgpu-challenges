#include <cuda_runtime.h>
#include "solve.h"
__global__ void reduceKernel(const float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    float sum = 0.0f;
    if (i < n)
        sum = g_idata[i];
    if (i + blockDim.x < n)
        sum += g_idata[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
         if (tid < s) {
             sdata[tid] += sdata[tid + s];
         }
         __syncthreads();
    }
    
    if (tid == 0) {
         g_odata[blockIdx.x] = sdata[0];
    }
}

void solve(const float* input, float* output, int N) {
    float *d_in, *d_temp;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMemcpy(d_in, input, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_temp, N * sizeof(float));

    int n = N;         
    int blockSize = 256; 

    while (n > 1) {
         int gridSize = (n + blockSize * 2 - 1) / (blockSize * 2);
         size_t sharedMemSize = blockSize * sizeof(float);
         reduceKernel<<<gridSize, blockSize, sharedMemSize>>>(d_in, d_temp, n);
         cudaDeviceSynchronize();

         n = gridSize;
         float* tmp = d_in;
         d_in = d_temp;
         d_temp = tmp;
    }

    cudaMemcpy(output, d_in, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_in);
    cudaFree(d_temp);
}
