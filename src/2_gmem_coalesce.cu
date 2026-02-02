#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

/*
For gmemory coalescing, main thing to note, threadIdx.x is the "fastest" dimension: 
    int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)
So we want to have ascending order threadIdx.x in the same warp!
*/
__global__ void gmem_coalesce_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    const int r = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int c = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (r < M && c < N) {
        float acc = 0.0;
        for (int k = 0; k < K; ++k) {
            acc += A[r * K + k] * B[k * N + c];
        }
        C[r * N + c] = acc;
    }
}