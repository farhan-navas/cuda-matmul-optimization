from memory import UnsafePointer

fn matmul_naive(
    A: UnsafePointer[Float32], 
    B: UnsafePointer[Float32], 
    C: UnsafePointer[mut=True, Float32], # must be passed from a mutable tensor
    M: Int, 
    K: Int, 
    N: Int
):
    for i in range(M):
        for j in range(N):
            var acc: Float32 = 0.0
            for k in range(K):
                acc += A.load(i * K + k) * B.load(k * N + j)
            C.store(i * N + j, acc)

fn main():
    pass
