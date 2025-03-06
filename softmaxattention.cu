#include "solve.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>

__global__ void transpose(const float *kg, float *ht, int N, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < d) {
        ht[col * N + row] = kg[row * d + col];
    }
}

__global__ void matmull(const float *qg, const float *ht, float *matmul, int N, int M, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < d; i++) {
            sum += qg[row * d + i] * ht[i * N + col];
        }
        matmul[row * N + col] = sum;
    }
}

__global__ void scale(float *matmul, int M, int N, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        matmul[row * N + col] /= sqrtf((float)d);
    }
}


__global__ void max_sum_kernel(const float *matmul, float *global_sum, int M, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int row = blockIdx.x;  
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += expf(matmul[row * N + i]);
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        global_sum[row] = sdata[0];
    }
}

__global__ void softmax(float *matmul, int M, int N, const float *global_sum) {
    int row = blockIdx.x;  
    int tid = threadIdx.x;
    for (int col = tid; col < N; col += blockDim.x) {
        float sum_val = global_sum[row];
        matmul[row * N + col] = expf(matmul[row * N + col]) / sum_val;
    }
}

__global__ void matmulfinal(const float *matmul, const float *vg, float *outputg, int N, int M, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    if (row < M && col < d) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += matmul[row * N + i] * vg[i * d + col];
        }
        outputg[row * d + col] = sum;
    }
}

void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    float *qg, *kg, *vg, *matmul, *outputg;
    cudaMalloc(&qg, M * d * sizeof(float));
    cudaMalloc(&kg, N * d * sizeof(float));
    cudaMalloc(&vg, N * d * sizeof(float));
    cudaMalloc(&outputg, M * d * sizeof(float));
    cudaMalloc(&matmul, M * N * sizeof(float));

    cudaMemcpy(qg, Q, M * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kg, K, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vg, V, N * d * sizeof(float), cudaMemcpyHostToDevice);

    float *ht;
    cudaMalloc(&ht, d * N * sizeof(float));
    {
        dim3 blockDimTrans(16, 16);
        dim3 gridDimTrans((d + blockDimTrans.x - 1) / blockDimTrans.x,
                          (N + blockDimTrans.y - 1) / blockDimTrans.y);
        transpose<<<gridDimTrans, blockDimTrans>>>(kg, ht, N, d);
        cudaDeviceSynchronize();
    }

    {
        dim3 blockDimMat(16, 16);
        dim3 gridDimMat((N + blockDimMat.x - 1) / blockDimMat.x,
                        (M + blockDimMat.y - 1) / blockDimMat.y);
        matmull<<<gridDimMat, blockDimMat>>>(qg, ht, matmul, N, M, d);
        cudaDeviceSynchronize();
    }

    {
        dim3 blockDimScale(16, 16);
        dim3 gridDimScale((N + blockDimScale.x - 1) / blockDimScale.x,
                          (M + blockDimScale.y - 1) / blockDimScale.y);
        scale<<<gridDimScale, blockDimScale>>>(matmul, M, N, d);
        cudaDeviceSynchronize();
    }

    float *global_sum;
    cudaMalloc(&global_sum, M * sizeof(float));
    {
        int threadsPerBlockMax = 256;
        dim3 gridDimMax(M);
        size_t shared_size = threadsPerBlockMax * sizeof(float);
        max_sum_kernel<<<gridDimMax, threadsPerBlockMax, shared_size>>>(matmul, global_sum, M, N);
        cudaDeviceSynchronize();
    }

    {
        int threadsPerBlockMax = 256;
        dim3 gridDimSoftmax(M);
        softmax<<<gridDimSoftmax, threadsPerBlockMax>>>(matmul, M, N, global_sum);
        cudaDeviceSynchronize();
    }

    {
        dim3 blockDimFinal(16, 16);
        dim3 gridDimFinal((d + blockDimFinal.x - 1) / blockDimFinal.x,
                          (M + blockDimFinal.y - 1) / blockDimFinal.y);
        matmulfinal<<<gridDimFinal, blockDimFinal>>>(matmul, vg, outputg, N, M, d);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(output, outputg, M * d * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(qg);
    cudaFree(kg);
    cudaFree(vg);
    cudaFree(outputg);
    cudaFree(matmul);
    cudaFree(ht);
    cudaFree(global_sum);
}
