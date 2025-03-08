#include "solve.h"
#include <cuda_runtime.h>

__global__ void reverse_array(float* input, float *output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N and N - i - 1 >= 0){
        output[i] = input[N - i - 1];
    }
}

// input is device pointer
void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    float* output;
    cudaMalloc(&output, N * sizeof(float));
    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, output,N);
    cudaDeviceSynchronize();
    cudaMemcpy(input, output, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(output);
}
