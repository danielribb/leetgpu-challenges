#include "solve.h"
#include <cuda_runtime.h>

__global__ void matrix_transpose(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < rows and col < cols){
        output[col * rows + row] = input[row * cols + col];
    }
}

void solve(const float* input, float* output, int rows, int cols) {
    float *d_input, *d_output;

    // Allocate device memory
    cudaMalloc(&d_input, rows * cols * sizeof(float));
    cudaMalloc(&d_output, rows * cols * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_input, input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (rows + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // Launch the kernel
    matrix_transpose<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}
