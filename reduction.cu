#include "solve.h"
#include <cuda_runtime.h>


__global__ void reduction(const float* input,  float* output, int N){
     extern __shared__ float sdata[];
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int tid = threadIdx.x;

   float x = (idx < N) ? input[idx] : 0.0f;
   sdata[tid] = x;
   __syncthreads();

   for(unsigned int s = blockDim.x/2; s > 0; s>>=1){
        if(tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
   }

   if(tid == 0){
        output[blockIdx.x] = sdata[0];
   }
}


// input, output are device pointers
void solve(const float* input, float* output, int N) {  
    
    int threadsperblock = 256;
    int blockspergrid = (N + threadsperblock - 1)/threadsperblock;
    float shared_mem = threadsperblock;
    reduction<<<blockspergrid, threadsperblock, shared_mem>>>(input, output, N);
    cudaDeviceSynchronize();
}
