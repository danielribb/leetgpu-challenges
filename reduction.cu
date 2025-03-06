#include "solve.h"
#include <cuda_runtime.h>

__global__ void reduction(float *d_input, float *d_output, int N){
   extern __shared__ float sdata[];
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int tid = threadIdx.x;

   float x = (idx < N) ? d_input[idx] : 0.0f;
   sdata[tid] = x;
   __syncthreads();

   for(unsigned int s = blockDim.x/2; s > 0; s>>=1){
        if(tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
   }

   if(tid == 0){
        d_output[blockIdx.x] = sdata[0];
   }
}


void solve(const float* input, float* output, int N) {  
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, input,  N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(256);
    dim3 gridDim((N + blockDim.x - 1)/blockDim.x);
    int shared_size = 256 * sizeof(float);
    reduction<<<gridDim, blockDim, shared_size>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
