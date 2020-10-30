SLATE Example
#############

Check for a module -- it might be there already.
If not, installing with spack is now straightforward.
See the :doc:`/dev/spack` recipe for more details.

Following the `slate tutorial <https://bitbucket.org/icl/slate-tutorial/src/master/>`_, it's possible to
extract this gem::

    #include <iostream>
    #include <slate/slate.hh>
    #include <blas.hh>

    template <typename scalar_type>
    scalar_type make( blas::real_type<scalar_type> re,
                      blas::real_type<scalar_type> im ) {
        return re;
    }
    template <typename T>
    std::complex<T> make( T re, T im ) {
            return std::complex<T>( re, im );
    }
    // generate random matrix A
    template <typename scalar_type>
    void fill_tile( slate::Tile<scalar_type> T ) {
        scalar_type* A = T.data();
        int64_t lda = T.stride();
        for (int64_t j = 0; j < T.nb(); ++j) {
            for (int64_t i = 0; i < T.mb(); ++i) {
                A[ i + j*lda ] = make<scalar_type>( rand() / double(RAND_MAX),
                                                    rand() / double(RAND_MAX) );
            }
        }
    }
    template <typename matrix_type>
    void random_matrix( matrix_type& A ) {
        for (int64_t j = 0; j < A.nt(); ++j) {
            for (int64_t i = 0; i < A.mt(); ++i) {
                if (A.tileIsLocal( i, j )) {
                    try {
                        fill_tile( A(i,j) );
                    } catch (...) { // ignore missing tiles
                    }
                }
            }
        }
    }

    template <typename scalar_type>
    void test_gemm(int p, int q) {
        double alpha = 2.0, beta = 1.0;
        int64_t m=10000, n=8000, k=5000, nb=256;
        slate::Matrix<double> A( m, k, nb, p, q, MPI_COMM_WORLD );
        slate::Matrix<double> B( k, n, nb, p, q, MPI_COMM_WORLD );
        slate::Matrix<double> C( m, n, nb, p, q, MPI_COMM_WORLD );
        A.insertLocalTiles(slate::Target::Devices);
        B.insertLocalTiles(slate::Target::Devices);
        C.insertLocalTiles(slate::Target::Devices);
        random_matrix( A );
        random_matrix( B );
        random_matrix( C );

        // C = alpha A B + beta C
        slate::gemm( alpha, A, B, beta, C, {
            { slate::Option::Lookahead, 1 },
            { slate::Option::Target, slate::Target::Devices },  // on GPU devices
        } );
    }

    int main( int argc, char** argv ) {
        int mpi_size, mpi_rank, provided = 0;
        assert( ! MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided ) );
        assert( provided == MPI_THREAD_MULTIPLE );
        assert( ! MPI_Comm_size( MPI_COMM_WORLD, &mpi_size ) );
        assert( ! MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank ) );

        if( argc != 3 ) {
        err:
            std::cout << "Usage: mpirun -np <p>*<q> "
                      << argv[0] << " <p> <q>" << std::endl;
            return 1;
        }
        int p = atoi(argv[1]);
        int q = atoi(argv[2]);
        if (p < 0 || mpi_size != p*q) goto err;

        // so random_matrix is different on different ranks.
        srand( 100 * mpi_rank );

        test_gemm< float >(p, q);
        test_gemm< double >(p, q);
        test_gemm< std::complex<float> >(p, q);
        test_gemm< std::complex<double> >(p, q);

        return MPI_Finalize();
    }

I get the following timings using gcc 6.4.0 and CUDA 10.1.243 on 1 Summit node.

.. code-block:: bash

    $ time jsrun --smpiargs=-gpu -r 6 -g 1 -c 7 -b packed:7 -EOMP_NUM_THREADS=7 ./gemm 2 3

    # CPU timing
    real	0m25.461s
    user	0m0.142s
    sys	0m0.015s

    # GPU timing
    real	0m10.973s
    user	0m0.105s
    sys	0m0.052s


Note that this includes generating 8 random matrices and doing 4 matrix-multiplies.

The following 2 cmake files make compilation easy.

.. code-block:: cmake

    # CMakeLists.txt
    SET(TARGET "gemm")

    SET(TARGET_SRC
	gemm.cc
       )

    CMAKE_MINIMUM_REQUIRED(VERSION 3.17)

    PROJECT(${TARGET} CXX CUDA)

    # Dependency Packages
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
    find_package(MPI REQUIRED)
    find_package(SLATE REQUIRED)
    find_package(CUDAToolkit REQUIRED) # TODO: create SLATE::CUDA
    find_package(OpenMP REQUIRED) # TODO: add to SLATE

    add_executable(${TARGET} ${TARGET_SRC})
    set_target_properties(
	${TARGET} PROPERTIES
	CXX_STANDARD 17
	CUDA_STANDARD 11
	CXX_STANDARD_REQUIRED true
	CXX_EXTENSIONS false
    )

    # TODO: ideally a target like SLATE::CUDA would add the cublas dep.
    #target_link_libraries(${TARGET} PRIVATE SLATE::CUDA MPI::MPI_CXX)
    target_link_libraries(${TARGET} PRIVATE SLATE CUDA::cublas CUDA::cudart OpenMP::OpenMP_CXX MPI::MPI_CXX)
    set_property(TARGET ${TARGET} PROPERTY CUDA_ARCHITECTURES 70)

    install (TARGETS ${TARGET} DESTINATION bin)

.. code-block:: cmake

    # cmake/FindSLATE.cmake
    # Find the slate library
    # THIS IS A WORK IN PROGRESS - SINCE IT DOESN't SET CUDA/OpenMP DEPENDENCIES CORRECTLY
    #
    # The following variables are optionally searched for defaults
    #  SLATE_ROOT: Base directory where all SLATE components are found
    #  SLATE_INCLUDE_DIR: Directory where SLATE header is found
    #  SLATE_LIB_DIR: Directory where SLATE library is found
    #
    # The following are set after configuration is done:
    #  SLATE_FOUND
    #  SLATE_INCLUDE_DIRS
    #  SLATE_LIBRARIES

    set(SLATE_INCLUDE_DIR $ENV{SLATE_INCLUDE_DIR} CACHE PATH "Folder contains SLATE headers")
    set(SLATE_LIB_DIR $ENV{SLATE_LIB_DIR} CACHE PATH "Folder contains SLATE libraries")
    set(SLATE_VERSION $ENV{SLATE_VERSION} CACHE STRING "Version of SLATE to build with")

    # Compatible layer for CMake <3.12. SLATE_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
    list(APPEND CMAKE_PREFIX_PATH ${SLATE_ROOT})

    find_path(SLATE_INCLUDE_DIRS
      NAMES slate/slate.hh
      HINTS ${SLATE_INCLUDE_DIR})

    if (USE_STATIC_SLATE)
      MESSAGE(STATUS "USE_STATIC_SLATE is set. Linking with static SLATE library.")
      if (SLATE_VERSION)  # Prefer the versioned library if a specific SLATE version is specified
	set(CMAKE_FIND_LIBRARY_SUFFIXES ".a.${SLATE_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
      endif()
    else()
      if (SLATE_VERSION)  # Prefer the versioned library if a specific SLATE version is specified
	set(CMAKE_FIND_LIBRARY_SUFFIXES ".so.${SLATE_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
      endif()
    endif()

    find_library(SLATE_LIBRARIES
      NAMES "slate"
      HINTS ${SLATE_LIB_DIR})

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(SLATE DEFAULT_MSG SLATE_INCLUDE_DIRS SLATE_LIBRARIES)

    if(SLATE_FOUND)
      set (SLATE_HEADER_FILE "${SLATE_INCLUDE_DIRS}/slate/slate.hh")
      message (STATUS "Determining SLATE version from the header file: ${SLATE_HEADER_FILE}")
      # e.g. #define SLATE_VERSION 20201000
      file (STRINGS ${SLATE_HEADER_FILE} SLATE_VERSION_DEFINED
	  REGEX "^[ \t]*#define[ \t]+SLATE_VERSION[ \t]+[0-9]+.*$" LIMIT_COUNT 1)
      if (SLATE_VERSION_DEFINED)
	string (REGEX REPLACE "^[ \t]*#define[ \t]+SLATE_VERSION[ \t]+" ""
		SLATE_VERSION ${SLATE_VERSION_DEFINED})
	message (STATUS "SLATE_VERSION: ${SLATE_VERSION}")
      endif ()
      message(STATUS "Found SLATE (include: ${SLATE_INCLUDE_DIRS}, library: ${SLATE_LIBRARIES})")
      # Create a new-style imported target (SLATE)
      if (USE_STATIC_SLATE)
	  add_library(SLATE STATIC IMPORTED)
      else()
	  add_library(SLATE SHARED IMPORTED)
      endif ()
      target_include_directories(SLATE INTERFACE ${SLATE_INCLUDE_DIRS})
      set_target_properties(
	  SLATE PROPERTIES
	  IMPORTED_LOCATION ${SLATE_LIBRARIES}
	  CXX_STANDARD 17
	  CUDA_STANDARD 11
	  CXX_STANDARD_REQUIRED true
	  CXX_EXTENSIONS false
      )
      #set_property(TARGET SLATE PROPERTY
      #             LANGUAGE CUDA)

      mark_as_advanced(SLATE_ROOT_DIR SLATE_INCLUDE_DIRS SLATE_LIBRARIES)
    endif()

.. admonition:: Contributed by

   David M. Rogers

