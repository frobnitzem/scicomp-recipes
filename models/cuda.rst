CUDA Programming Model
######################

There's a lot of useful discussion of optimizing code
for HPC at
`The CUDA Programming Guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>_`.

TODO
----
* note importance of warp size = 32 as a SIMD operation
* note compiler intrinsics
* note __ballot and __shfl for inter-warp communication
* point out grid-stride for loop pattern example
* CMakeLists.txt discussion
* CUDA with Fortran
