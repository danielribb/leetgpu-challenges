#include "solve.h"
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel_reduction(const float *input, const float *kernel, float *output,
                                                  int input_size, int kernel_size) {
    extern __shared__ float s_data[]; 

    int tid = threadIdx.x;
    int out_idx = blockIdx.x;
    int input_start = out_idx; 

    float partial_sum = 0.0f;
    for (int j = tid; j < kernel_size; j += blockDim.x) {
        partial_sum += input[input_start + j] * kernel[j];
    }
    s_data[tid] = partial_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[out_idx] = s_data[0];
    }
}

void solve(const float *input, const float *kernel, float *output, int input_size, int kernel_size) {
    float *d_input, *d_kernel, *d_output;
    int output_size = input_size - kernel_size + 1;

    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    cudaMemcpy(d_input, input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256; 
    int blocksPerGrid = output_size;
    int shared_mem_size = threadsPerBlock * sizeof(float);

    convolution_1d_kernel_reduction<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(d_input, d_kernel, d_output,
                                                                                         input_size, kernel_size);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}
