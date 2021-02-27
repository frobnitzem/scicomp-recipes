Simple Kokkos Example
#####################

The `Kokkos C++ library <https://kokkos.org/>`_
provides data layout and location
information to the compiler by enriching the type system.

For example an **N**x**3** array on the GPU could be declared using::

    #include <iostream>
    #include <mpi.h>
    #include <Kokkos_Core.hpp>
        
    using namespace Kokkos;
        
    typedef View<double *[3],LayoutLeft,CudaSpace> t_crd;
        
    class RunCross {
        public:
            RunCross(const t_crd &a_,
                     const t_crd &b_,
                     const t_crd &c_) : a(a_), b(b_), c(c_) {};
            typedef int size_type;
            KOKKOS_INLINE_FUNCTION void operator()(const size_type i) const {
                c(i,0) = a(i,1)*b(i,2)-a(i,2)*b(i,1);
                c(i,1) = a(i,2)*b(i,0)-a(i,0)*b(i,2);
                c(i,2) = a(i,0)*b(i,1)-a(i,1)*b(i,0);
            }
        
        private:
            t_crd a, b, c;
    };
        
    int run(int N) {
        t_crd a("A", N);
        t_crd b("B", N);
        t_crd c("C", N);
        
        parallel_for(N, RunCross(a, b, c));
        
        return 0;
    }
        
    int main(int argc, char *argv[]) {
        int n = -1;
        
        MPI_Init(&argc, &argv);
        Kokkos::initialize(argc, argv);
        n = run(10000);
        Kokkos::finalize();
        MPI_Finalize();
        return n;
    }

TODO: 

  * Compiling instructions

