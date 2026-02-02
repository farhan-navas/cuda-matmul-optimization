# first time actually really writing cmake
cmake_minimum_required(VERSION 3.19)
set(CMAKE_CUDA_ARCHITECTURES 90)

project(NVIDIA_SGEMM_PRACTICE LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ensure cuda is available
include(CheckLanguage)

set(CMAKE_CXX_STANDARD 20)

# generate executables individually
add_executable(sgemm
    src/1_naive.cu
    src/2_gmem_coalesce.cu
    test_script.cu
)

# in debug mode, add debug symbols to device code
# this disables most optimizations and kills performance
target_compile_options(sgemm PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

# configure header file search paths
target_include_directories(sgemm PRIVATE src)

# for multi-file CUDA kernels
set_target_properties(sgemm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
