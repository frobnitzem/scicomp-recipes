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

.. admonition:: Contributed by

   David M. Rogers

