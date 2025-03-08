#include "solve.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>

__global__ void transpose(const float *K, float *ht, int N, int d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < d) {
        ht[col * N + row] = K[row * d + col];
    }
}

__global__ void matmull(const float *Q, const float *kt, float *matmul, int N, int M, int d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < d; i++) {
            sum += Q[row * d + i] * kt[i * N + col];
        }
        matmul[row * N + col] = sum;
    }
}

__global__ void scale(float *matmul, int M, int N, int d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        matmul[row * N + col] /= sqrtf((float)d);
    }
}


__global__ void stable_sum_kernel(const float *matmul, float *global_sum, float *row_max, int M, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int row = blockIdx.y; 

    float max_val = -FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x) {
         float val = matmul[row * N + i];
         if (val > max_val) max_val = val;
    }
    sdata[tid] = max_val;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
         if (tid < s) {
             if (sdata[tid + s] > sdata[tid])
                 sdata[tid] = sdata[tid + s];
         }
         __syncthreads();
    }
    max_val = sdata[0];
    if (tid == 0) {
         row_max[row] = max_val;
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
         local_sum += expf(matmul[row * N + i] - max_val);
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


__global__ void softmax_stable(float *matmul, int M, int N, const float *global_sum, const float *row_max) {
    int row = blockIdx.y;
    int tid = threadIdx.x;
    for (int col = tid; col < N; col += blockDim.x) {
         float max_val = row_max[row];
         float sum_val = global_sum[row];
         matmul[row * N + col] = expf(matmul[row * N + col] - max_val) / sum_val;
    }
}

__global__ void matmulfinal(const float *matmul, const float *V, float *output, int N, int M, int d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < d) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += matmul[row * N + i] * V[i * d + col];
        }
        output[row * d + col] = sum;
    }
}

void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    float *kt;
    cudaMalloc(&kt, d * N * sizeof(float));
    float* matmul;
    cudaMalloc(&matmul, M * N * sizeof(float));

    {
        dim3 blockDimTrans(16, 16);
        dim3 gridDimTrans((d + blockDimTrans.x - 1) / blockDimTrans.x, (N + blockDimTrans.y - 1) / blockDimTrans.y);
        transpose<<<gridDimTrans, blockDimTrans>>>(K, kt, N, d);
        cudaDeviceSynchronize();
    }

    {
        dim3 blockDimMat(16, 16);
        dim3 gridDimMat((N + blockDimMat.x - 1) / blockDimMat.x, (M + blockDimMat.y - 1) / blockDimMat.y);
        matmull<<<gridDimMat, blockDimMat>>>(Q, kt, matmul, N, M, d);
        cudaDeviceSynchronize();
    }

    {
        dim3 blockDimScale(16, 16);
        dim3 gridDimScale((N + blockDimScale.x - 1) / blockDimScale.x, (M + blockDimScale.y - 1) / blockDimScale.y);
        scale<<<gridDimScale, blockDimScale>>>(matmul, M, N, d);
        cudaDeviceSynchronize();
    }

    float *global_sum;
    float *row_max;
    cudaMalloc(&global_sum, M * sizeof(float));
    cudaMalloc(&row_max, M * sizeof(float));
    {
        dim3 blockDimSum(256);
        dim3 gridDimSum(1, M);
        size_t shared_size = 256 * sizeof(float);
        stable_sum_kernel<<<gridDimSum, blockDimSum, shared_size>>>(matmul, global_sum, row_max, M, N);
        cudaDeviceSynchronize();
    }

    {
        dim3 blockDimSoftmax(256);
        dim3 gridDimSoftmax(1, M);
        softmax_stable<<<gridDimSoftmax, blockDimSoftmax>>>(matmul, M, N, global_sum, row_max);
        cudaDeviceSynchronize();
    }

    {
        dim3 blockDimFinal(16, 16);
        dim3 gridDimFinal((d + blockDimFinal.x - 1) / blockDimFinal.x, (M + blockDimFinal.y - 1) / blockDimFinal.y);
        matmulfinal<<<gridDimFinal, blockDimFinal>>>(matmul, V, output, N, M, d);
        cudaDeviceSynchronize();
    }
    
    cudaFree(matmul);
    cudaFree(kt);
    cudaFree(global_sum);
    cudaFree(row_max);
}
