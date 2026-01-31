#pragma once

__global__ void naive_matmul(const float* A, const float* B, float* C, int M, int K, int N);
