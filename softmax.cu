#include "solve.h"
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>

__global__ void max_reduce_kernel(const float* input, float* block_max, int N) {
    extern __shared__ float stdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float x = (idx < N) ? input[tid] : -FLT_MAX;
    stdata[tid] = x;
    __syncthreads();

    for(unsigned  int s = blockDim.x/2; s > 0; s>>=1){
        if(tid < s){
            stdata[tid] = fmax(stdata[tid], stdata[tid + s]);
        }
        __syncthreads();
    }

    if(tid == 0){
        block_max[blockIdx.x] = stdata[0];
    }

}

__global__ void sum_exp_kernel(const float* input, float* block_sum, float global_max, int N) {
    extern __shared__ float stdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockIdx.x + threadIdx.x;

    float x = (idx < N) ? expf(input[idx] - global_max) : 0.0f;
    stdata[tid] = x;
    __syncthreads();

    for(unsigned int s = blockDim.x/2; s > 0; s>>=1){
        if(tid < s){
            stdata[tid] += stdata[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0){
        block_sum[blockIdx.x] = stdata[0];
    }
}

__global__ void softmax_final_kernel(const float* input, float* output, float global_max, float global_sum, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        output[i] = expf(input[i] - global_max)/global_sum;
    }
}

void solve(const float* input, float* output, int N) {
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float *d_block_max, *d_block_sum;
    cudaMalloc(&d_block_max, blocks * sizeof(float));
    cudaMalloc(&d_block_sum, blocks * sizeof(float));
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    max_reduce_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(d_input, d_block_max, N);
    cudaDeviceSynchronize();

    float* h_block_max = (float*)malloc(blocks * sizeof(float));
    cudaMemcpy(h_block_max, d_block_max, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float global_max = -FLT_MAX;
    for (int i = 0; i < blocks; i++) {
        if (h_block_max[i] > global_max) {
            global_max = h_block_max[i];
        }
    }
    free(h_block_max);

    sum_exp_kernel<<<blocks, threadsPerBlock, sharedMemSize>>>(d_input, d_block_sum, global_max, N);
    cudaDeviceSynchronize();

    float* h_block_sum = (float*)malloc(blocks * sizeof(float));
    cudaMemcpy(h_block_sum, d_block_sum, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float global_sum = 0.0f;
    for (int i = 0; i < blocks; i++) {
        global_sum += h_block_sum[i];
    }
    free(h_block_sum);

    softmax_final_kernel<<<blocks, threadsPerBlock>>>(d_input, d_output, global_max, global_sum, N);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_block_max);
    cudaFree(d_block_sum);
}
