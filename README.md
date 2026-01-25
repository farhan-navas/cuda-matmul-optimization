# CUDA Matmul Kernel Optimization

For this project, I will be trying to optimize a CUDA matrix multiplication kernel to get to the cuBLAS-level performance.

Each iteration will be in separate files and will have:

- **Implementation**: Kernel optimization and algorithmic improvements
- **Performance Profiling**: Detailed timing statistics, memory usage analysis, etc. Done with square matrices for ease of reference
- **Benchmarking**: Comparison against cuBLAS reference implementation

I also want to see if all the parallel computing modules I have taken have actually been useful! And huge thanks and credits to @ siboehm for giving me the idea to explore this on my own!

# Iteration 1

For the first iteration, I will just build a baseline naive matrix multiplication in Python and Mojo! I will just measure wall-clock timing and use these numbers to define speedup targets for the subsequent CUDA-focused iterations. Just for fun.

# Iteration 2
