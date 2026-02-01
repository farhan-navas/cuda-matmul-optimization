#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

__global__ void naive_matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < M && c < N) {
        float acc = 0.0;
        for (int k = 0; k < K; ++k) {
            acc += A[r * K + k] * B[k * N + c];
        }
        C[r * N + c] = acc;
    }
}
