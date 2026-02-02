# CUDA Matmul Kernel Optimization

For this project, I will be trying to optimize a CUDA matrix multiplication kernel to get to the cuBLAS-level performance.

Each iteration will be in separate files and will have:

- **Implementation**: Kernel optimization and algorithmic improvements
- **Performance Profiling**: Detailed timing statistics, memory usage analysis, etc. Done with square matrices for ease of reference
- **Benchmarking**: Comparison against cuBLAS reference implementation

I also want to see if all the parallel computing modules I have taken have actually been useful! And huge thanks and credits to @siboehm for giving me the idea to explore this on my own!

## System Versions

These are the software and compiler versions used for all CUDA experiments on the NUS GPU compute cluster:

| Component           | Version          |
| ------------------- | ---------------- |
| Operating System    | Ubuntu 24.04     |
| CUDA Toolkit        | 12.0 (V12.0.140) |
| NVCC Compiler       | 12.0             |
| GCC (host compiler) | 13.3.0           |
| Clang (optional)    | 18.1.3           |
| CMake               | 3.28.3           |

For all of the experiments we will be using an entire A100 80GB GPU, run using:

```
$ srun -G h100-96 bash -c "hostname; nvidia-smi"

# to build:
mkdir build
cd build
cmake ..
make -j
./sgemm
```

# Iteration 0

For the first iteration, I will just build a baseline naive matrix multiplication in [Python](1a_naivepy.py) and [Mojo](1b_naivemojo.mojo)! I will just measure wall-clock timing and use these numbers to define speedup targets for the subsequent CUDA-focused iterations. Just for fun.

# Iteration 1

First naive CUDA kernel implementation where we just look at correctness, and try our best to make sure everything is running right. We also setup a sanity check benchmark script to check for any errors. For this 1st iteration,
