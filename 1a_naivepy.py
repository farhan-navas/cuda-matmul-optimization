import random
import time

def matmul(A, B):
    """naivEST matmul implementation"""
    # input A is a M X K matrix, and B is a K x N matrix,
    # so our final output C will be a M x N matrix
    # time complexity: ~O(MKN), assume square matrix ~O(N^3)
    M = len(A)
    K = len(A[0])
    N = len(B[0])

    C = [[0.0 for _ in range(N)] for _ in range(M)] # init output matrix
    for x in range(M):
        for y in range(N):
            res = 0.0
            for z in range(K):
                res += A[x][z] + B[z][y]
            C[x][y] = res

    return C

def strassens_matmul(A, B):
    """strassens matmul implementation"""
    # made for square matrices, but we can use a simple padding trick to use 
    # it for non-square matrices as well. time complexity is assuming square matrix,
    # time complexity ~O(N^2.81)
    pass 
