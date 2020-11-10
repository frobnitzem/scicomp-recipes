NCCL Example
############

Here's an example created based on the `NVIDIA Docs <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html#example-2-one-device-per-process-or-thread>`_.

.. code-block:: C++

    // helper.hh
    #include <iostream>
    #include <mpi.h>
    #include <assert.h>
    #include <memory>

    #include "cuda_runtime.h"
    #include "nccl.h"
    #include <unistd.h>
    #include <stdint.h>

    #define MPICHECK(cmd) do {                          \
      int e = cmd;                                      \
      if( e != MPI_SUCCESS ) {                          \
	printf("Failed: MPI error %s:%d '%d'\n",        \
	    __FILE__,__LINE__, e);   \
	exit(EXIT_FAILURE);                             \
      }                                                 \
    } while(0)

    #define CUDACHECK(cmd) do {                         \
      cudaError_t e = cmd;                              \
      if( e != cudaSuccess ) {                          \
	printf("Failed: Cuda error %s:%d '%s'\n",       \
	    __FILE__,__LINE__,cudaGetErrorString(e));   \
	exit(EXIT_FAILURE);                             \
      }                                                 \
    } while(0)

    #define NCCLCHECK(cmd) do {                         \
      ncclResult_t r = cmd;                             \
      if (r!= ncclSuccess) {                            \
	printf("Failed, NCCL error %s:%d '%s'\n",       \
	    __FILE__,__LINE__,ncclGetErrorString(r));   \
	exit(EXIT_FAILURE);                             \
      }                                                 \
    } while(0)

    struct MPIH {
	int ranks, rank;
	MPI_Comm comm;

	MPIH(int *argc, char **argv[]) : comm(MPI_COMM_WORLD) {
	    int provided;
	    MPICHECK( MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided) );
	    assert(provided >= MPI_THREAD_FUNNELED);
	    MPICHECK( MPI_Comm_size( comm, &ranks) );
	    MPICHECK( MPI_Comm_rank( comm, &rank ) );
	}
	~MPIH() {
	    MPI_Finalize();
	}
    };
    using MPIp = std::shared_ptr<MPIH>;

    struct NCCLH {
	MPIp mpi;
	ncclUniqueId id;
	ncclComm_t comm;
	cudaStream_t stream;

	NCCLH(MPIp _mpi) : mpi(_mpi) {
            int numGPUs;
	    if (mpi->rank == 0) ncclGetUniqueId(&id);
	    MPICHECK( MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, mpi->comm) );
            CUDACHECK( cudaGetDeviceCount(&numGPUs) );
	    CUDACHECK( cudaSetDevice(mpi->rank % numGPUs) );
            CUDACHECK( cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) );
	    NCCLCHECK( ncclCommInitRank(&comm, mpi->ranks, id, mpi->rank) );
	}
	~NCCLH() {
	    CUDACHECK(cudaStreamDestroy(stream));
	    ncclCommDestroy(comm);
	}
    };
    using NCCLp = std::shared_ptr<NCCLH>;

It's always a good idea to wrap up initialization and finalization
code inside classes to manage them.

Note that we're using shared-pointers here, since lots of different
parts of the code might end up storing the NCCL-helper struct above.
Shared pointers act just like pointers (de-reference with ``*`` or ``->``).
However, they track reference-counts and destroy the object when it hits zero.

With this out of the way, the main code is simple::

    // allreduce.cu
    #include "helper.hh"

    int size = 8*1024*1024; // 64 MB
    int run(NCCLp nccl, const float *send, float *recv) {
        NCCLCHECK(ncclAllReduce((const void*)send, (void*)recv, size, ncclFloat, ncclSum,
                                nccl->comm, nccl->stream));
        return 0;
    }

    int main(int argc, char *argv[]) {
        auto mpi = std::make_shared<MPIH>(&argc, &argv);
        auto nccl = std::make_shared<NCCLH>(mpi);

        float *sendbuff, *recvbuff;
        CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
        CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));

        std::cout << "Hello" << std::endl;
        run(nccl, sendbuff, recvbuff);
        CUDACHECK(cudaStreamSynchronize(nccl->stream));

        return 0;
    }

Here's the cmake magic needed to compile it,

.. code-block:: cmake

    # CMakeLists.txt
    CMAKE_MINIMUM_REQUIRED(VERSION 3.17)

    PROJECT(use_nccl CXX CUDA)

    # Dependency Packages
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
    find_package(MPI REQUIRED)
    find_package(NCCL REQUIRED)

    # Global project properties
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED True)

    add_executable(allreduce allreduce.cu)
    target_link_libraries(allreduce PUBLIC NCCL MPI::MPI_CXX)

    set_property(TARGET allreduce PROPERTY CUDA_ARCHITECTURES 70)
    install(TARGETS allreduce DESTINATION bin)

and the biggest file in the distribution,

.. code-block:: cmake

    # Find the nccl libraries
    # from https://github.com/xuhdev/pytorch/blob/a3b4accf014e18bf84f58d3018854435cbc3d55b/cmake/Modules/FindNCCL.cmake
    #
    # The following variables are optionally searched for defaults
    #  NCCL_ROOT: Base directory where all NCCL components are found
    #  NCCL_INCLUDE_DIR: Directory where NCCL header is found
    #  NCCL_LIB_DIR: Directory where NCCL library is found
    #
    # The following are set after configuration is done:
    #  NCCL_FOUND
    #  NCCL_INCLUDE_DIRS
    #  NCCL_LIBRARIES
    #
    # The path hints include CUDA_TOOLKIT_ROOT_DIR seeing as some folks
    # install NCCL in the same location as the CUDA toolkit.
    # See https://github.com/caffe2/caffe2/issues/1601

    set(NCCL_INCLUDE_DIR $ENV{NCCL_INCLUDE_DIR} CACHE PATH "Folder contains NVIDIA NCCL headers")
    set(NCCL_LIB_DIR $ENV{NCCL_LIB_DIR} CACHE PATH "Folder contains NVIDIA NCCL libraries")
    set(NCCL_VERSION $ENV{NCCL_VERSION} CACHE STRING "Version of NCCL to build with")

    list(APPEND NCCL_ROOT ${NCCL_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR})
    # Compatible layer for CMake <3.12. NCCL_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
    list(APPEND CMAKE_PREFIX_PATH ${NCCL_ROOT})

    find_path(NCCL_INCLUDE_DIRS
      NAMES nccl.h
      HINTS ${NCCL_INCLUDE_DIR})

    if (USE_STATIC_NCCL)
      MESSAGE(STATUS "USE_STATIC_NCCL is set. Linking with static NCCL library.")
      SET(NCCL_LIBNAME "nccl_static")
      if (NCCL_VERSION)  # Prefer the versioned library if a specific NCCL version is specified
        set(CMAKE_FIND_LIBRARY_SUFFIXES ".a.${NCCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
      endif()
    else()
      SET(NCCL_LIBNAME "nccl")
      if (NCCL_VERSION)  # Prefer the versioned library if a specific NCCL version is specified
        set(CMAKE_FIND_LIBRARY_SUFFIXES ".so.${NCCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
      endif()
    endif()

    find_library(NCCL_LIBRARIES
      NAMES ${NCCL_LIBNAME}
      HINTS ${NCCL_LIB_DIR})

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIRS NCCL_LIBRARIES)

    if(NCCL_FOUND)
      set (NCCL_HEADER_FILE "${NCCL_INCLUDE_DIRS}/nccl.h")
      message (STATUS "Determining NCCL version from the header file: ${NCCL_HEADER_FILE}")
      file (STRINGS ${NCCL_HEADER_FILE} NCCL_MAJOR_VERSION_DEFINED
            REGEX "^[ \t]*#define[ \t]+NCCL_MAJOR[ \t]+[0-9]+.*$" LIMIT_COUNT 1)
      if (NCCL_MAJOR_VERSION_DEFINED)
        string (REGEX REPLACE "^[ \t]*#define[ \t]+NCCL_MAJOR[ \t]+" ""
                NCCL_MAJOR_VERSION ${NCCL_MAJOR_VERSION_DEFINED})
        message (STATUS "NCCL_MAJOR_VERSION: ${NCCL_MAJOR_VERSION}")
      endif ()
      message(STATUS "Found NCCL (include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES})")
      # Create a new-style imported target (NCCL)
      if (USE_STATIC_NCCL)
          add_library(NCCL STATIC IMPORTED)
      else()
          add_library(NCCL SHARED IMPORTED)
      endif ()
      set_property(TARGET NCCL PROPERTY
                   IMPORTED_LOCATION ${NCCL_LIBRARIES})
      set_property(TARGET NCCL PROPERTY
                   LANGUAGE CUDA)
      target_include_directories(NCCL INTERFACE ${NCCL_INCLUDE_DIRS})

      mark_as_advanced(NCCL_ROOT_DIR NCCL_INCLUDE_DIRS NCCL_LIBRARIES)
    endif()

I built NCCL from their `github source <https://github.com/NVIDIA/nccl>`_
(using ``make src.build CUDA_HOME=$CUDA_DIR NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"``)
and left it in its build directory (nccl/build).  Then ran
``cmake -DCMAKE_PREFIX_PATH=/path/to/nccl/build ..``.  You'll need
cuda and MPI modules loaded, and MPI build flags enabled.

You can run some quick tests on this using interactive mode,

.. code-block:: bash

    bsub -nnodes 1 -W 30 -P CHM101 -Is $SHELL
    # run 6 ranks per node
    jsrun --smpiargs=-gpu -r 6 -g 1 -c 7 -b packed:7 -EOMP_NUM_THREADS=7 ./allreduce

The performance plots below were gathered following the recipe
for :doc:`/perf/scaling`.

.. image:: nccl.svg
   :alt: NCCL Scaling Plot
   :align: center

The "bus" bandwidths reported are computed using the simple
formula 2x(bytes per rank in allreduce) / time.  
This makes them smaller than the true interconnect bandwidth.
Times collected were averages over 10 allreduces with no blocking
in-between.  The first 10 are marked as "cold" and the second
10 are marked as "warm".  NCCL's first run has a noticeable, but
worthwhile initialization cost.

Also, the performance was insensitive to the layout of resource sets:

.. code-block:: bash

  jsrun --smpiargs=-gpu -n $((6*nodes)) -g1 -c7 -b packed:7
  jsrun --smpiargs=-gpu -n $((2*nodes)) -g3 -c21 -a3 -b packed:7
  jsrun --smpiargs=-gpu -n $nodes       -g6 -c42 -a6 -b packed:7


.. admonition:: Contributed by

   David M. Rogers

