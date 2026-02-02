#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

/*
In naive matmul, each block calculates a certain tile of the output matrix, so in this case our blockDim is (16,16)
this means each thread invoked takes loads one entire row & one entire column from memory,
to calculate ONE output! Naivest implementation with no optimizations.
*/
__global__ void naive_matmul(const float* A, const float* B, float* C, int M, int K, int N) {

    // use threadIdx to "walk" across rows and columns in that block.
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < M && c < N) {
        float acc = 0.0;
        
        // K represents leng(row_A) and len(col_B), 
        // so we use it to load every elem needed for dot product
        for (int k = 0; k < K; ++k) {
            acc += A[r * K + k] * B[k * N + c];
        }
        C[r * N + c] = acc;
    }
}
