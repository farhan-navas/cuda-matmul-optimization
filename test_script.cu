// For a slightly easier implementation, we generate the matrices deterministically with some seed,
// so that all matrices use the same dataset for benchmarking
#include <array>
#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include "src/1_naive.cuh"

// min test for naive kernel
int main() {
    int size = 512;

    int M, K, N;
    M = K = N = size;

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // generate a random 
    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    // generate A and B
    for (auto &x : h_A) x = dist(gen);
    for (auto &x : h_B) x = dist(gen);
    
    // alloc space on device mem
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim(
        (N + blockDim.x - 1) / blockDim.x,
        (M + blockDim.y - 1) / blockDim.y
    );
    
    // run naive kernel, later on manually add the rest
    naive_matmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

    std::cout << "C[0] = " << h_C[0] << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}