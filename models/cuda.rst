CUDA Programming Model
######################

There's a lot of useful discussion of optimizing code
for HPC at
`The CUDA Programming Guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>`_.

Warp Speed Programming
----------------------

A majority of NVIDIA devices have 32 threads per warp.  This processor
layout is important, since it essentially means you can execute
32 floating point operations simultaneously in SIMD mode.


Device Intrinsics
-----------------

The CUDA programming guide provides several important math intrinsics
that can speed up your CUDA codes.  One of the most important is
``fma(x,y,z)``, which executes ``x*y+z`` with the same speed
as a single multiplication.

Other intrinsics are listed `here <https://docs.nvidia.com/cuda/archive/11.0/cuda-c-programming-guide/index.html#standard-functions>`_.


Inter-Warp Communication
------------------------

Threads within a warp can operate in a tightly coupled way
by exchanging data using
`warp-level primitives <https://developer.nvidia.com/blog/using-cuda-warp-level-primitives>`_.

This is not just useful for loops, but can be used
to manually implement reduction or even to coordinate
the warp to act as a group.


For Loop Example
----------------

Here's a minimal example to run ``Y += A*X`` using a CUDA kernel.
It shows the `grid-stride-for-loop pattern <https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops>`_::

    __global__ void saxpy(size_t n, float a, float *x, float *y) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < n;
             i += blockDim.x * gridDim.x) {
              y[i] = a * x[i] + y[i];
          }
    }

    int main() {
        int numSMs; // the V100 has 160 SMs, each with 32 "CUDA cores"
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
        const size_t N = (1 << 20) + 3; // 1M elements plus 3
        float *x, *y;
        cudaMalloc(&x, 2*N*sizeof(float));
        y = x + N; // Note: this is inefficient, since y is 4,
                   // but not 8-byte aligned (y & 0x0F == 3*4);
        saxpy<<<32*numSMs, 32>>>(N, 2.0, x, y);
        return 0;
    }

TODO
----
* CMakeLists.txt discussion
* CUDA with Fortran
