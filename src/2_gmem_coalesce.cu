#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

__global__ void gmem_coalesce_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    
}