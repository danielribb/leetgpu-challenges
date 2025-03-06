#include "solve.h"
#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i  < width * height  ){
     int pid = i * 4;
     image[pid] = 255 - image[pid];
     image[pid + 1] = 255 - image[pid + 1];
     image[pid + 2] = 255 - image[pid + 2];
    }
}

void solve(unsigned char* image, int width, int height) {
    unsigned char* d_image;
    int image_size = width * height * 4;

    // Allocate device memory
    cudaMalloc(&d_image, image_size * sizeof(unsigned char));

    // Copy input data from host to device
    cudaMemcpy(d_image, image, image_size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, width, height);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(image, d_image, image_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_image);
}
