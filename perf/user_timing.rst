User-Level Timers
#################

Yes, there are code instrumentation strategies that give
much more detail.  This recipe shows how to squeeze as much
performance data from the user-level as possible -- in the
traditional *debug-by-printf* style.

The inspiration for this recipe comes from the
cool per-thread trace plots in
`SLATE <https://bitbucket.org/icl/slate/src/master/include/slate/internal/Trace.hh>`_.

.. code-block:: c++

    // perf.hh
    #ifndef PERF_HH
    #define PERF_HH
    #include <stdint.h>
    #include <iostream>

    #ifdef _OPENMP
    #  include <omp.h>
    #else
    // mini-openmp compatibility layer
    #include <mpi.h>
    inline double omp_get_wtime() {
	return MPI_Wtime();
    }
    inline int omp_get_max_threads() {
	return 1;
    }
    inline int omp_get_num_threads(void) {
	return 1;
    }
    inline int omp_get_thread_num(void) {
	return 0;
    }
    #endif

    namespace Performance {
    struct PerfData {
	double t0, t1;
	double bytes, flops;
	PerfData(double _t0, uint64_t _bytes, uint64_t _flops)
	    : t0(_t0), bytes(_bytes), flops(_flops) {}
    };

    struct PerfAvg {
	/** Accumulates:
	 *
	 *   min = min_i x_i
	 *   max = max_i x_i
	 *   s = \sum_i x_i
	 *   v = \sum_i (x_i - s/n)^2
	 *   n = \sum_i 1
	 */
	PerfAvg(double x) : min(x), max(x), s(x), v(0.0), n(1) { }
	void add(const double x) {
	    // Old Trick From: https://manual.gromacs.org/2020/reference-manual/averages.html
	    if(x < min) min = x;
	    if(x > max) max = x;
	    v += (s - n*x)*(s - n*x) / (n*(n+1.0));
	    s += x;
	    n += 1;
	}
	double min, max, s, v;
	int n;
    };

    struct Timers {
        /**
         *  This class is declared (global) inside the perf.cc and must exist only there.
         */
	static void append(const std::string &label, PerfData &&datum);
	static void show(std::ostream &os);
	static void on();
	static void off();
    };

    class Timed {
	public:
	    Timed(const std::string &_label, uint64_t bytes, uint64_t flops)
		: label(_label), datum(omp_get_wtime(),
					bytes, flops) {}
	    ~Timed() {
		datum.t1 = omp_get_wtime();
		Timers::append(label, std::move(datum));
	    }
	private:
	    const std::string label;
	    PerfData datum;
    };
    }
    #endif

Declaring a ``Performance::Timed x()`` variable inside a block of code
has the effect of calling ``omp_get_wtime()`` when the object is created
and also calling its destructor when it's done.  The destructor
appends the ``PerfData`` object to a global variable stored
statically in ``perf.cc``.

Be careful to name your ``Timed`` variables, otherwise they'll
get immediately destroyed, and all the timer values
will be on the order of microseconds.

.. code-block:: c++

    // perf.cc
    #include "perf.hh"
    #include <vector>
    #include <tuple>
    #include <map>
    #include <cmath>

    namespace Performance {
        using d2 = std::tuple<double,double>;

        static bool tracing = false;
        static int num_threads = omp_get_max_threads();
        static std::vector<std::map<std::string,std::map<d2,PerfAvg>>> events(num_threads);

        void Timers::append(const std::string &label, PerfData &&datum) {
            if(!tracing) return;
            auto &self = events[omp_get_thread_num()];
            auto it = self.find(label);
            d2 key(datum.bytes, datum.flops);
            if (it == self.end()) {
                auto v = std::map<d2,PerfAvg>();
                v.emplace(key, PerfAvg(datum.t1-datum.t0));
                self[label] = v;
            } else {
                auto &et = it->second;
                auto ev = et.find(key);
                if (ev == et.end()) {
                    et.emplace(key, PerfAvg(datum.t1-datum.t0));
                } else {
                    ev->second.add(datum.t1-datum.t0);
                }
            }
        }
        void Timers::on() { tracing = true; }
        void Timers::off() { tracing = false; }

        void Timers::show(std::ostream &os) {
            const char hdr1[] = "[ ";
            const char hdr2[] = ", ";
            const char hdr3[] = "{ ";
            const char hdr4[] = "  , ";
            const char *ahdr = hdr1;

            for(auto self : events) {
                os << ahdr;
                ahdr = hdr2;

                const char *bhdr = hdr3;
                for(auto et : self) { // all events for thread
                    os << bhdr << "\"" << et.first << "\" : [";
                    bhdr = hdr4;
                    int i = 0;
                    for(auto ev : et.second) { // all map values
                       auto x = ev.second;
                       if(i != 0)
                           os << ",";
                       i = 1;
                       os << "\n      { \"Bytes\": " << std::get<0>(ev.first) << std::endl
                          << "      , \"Flops\": " << std::get<1>(ev.first) << std::endl
                          << "      , \"Calls\": " << x.n << std::endl
                          << "      , \"Min\": "   << x.min << std::endl
                          << "      , \"Max\": "   << x.max << std::endl
                          << "      , \"Avg\": "   << x.s/x.n << std::endl
                          << "      , \"Stddev\": " << std::sqrt(x.v/x.n) << " }";
                    }
                    os << "\n      ]\n";
                }
                os << "  }\n";
            }
            os << "]\n";
        }
    }

Most of the code here is the pretty-printer, which
outputs the event log in json-style.  The outer list
has one element per OMP thread.

The inner dictionary runs over labels, and there's a list
of all the individual data points for each label.

By way of example, here's an OpenMP code that
does some time-integration to observe the butterfly effect.

.. code-block:: c++

    // main.cc

    #include <iostream>
    #include <assert.h>
    #include "perf.hh"

    double run(int N) { // Lorenz attractor
        Performance::Timed timer("integrate", 0, 14*N);
        double x=1.0, y=0.0, z=0.0;
        const double dt = 1e-4;
        const double beta = 8.0/3.0;

        y += omp_get_thread_num()*1e-14;
        for(int i=0; i<N; i++) {
              double dx = 10.0*(y - x),
                     dy = x * (28.0 - z) - y,
                     dz = x * y - beta*z;
              x += dx*dt;
              y += dy*dt;
              z += dz*dt;
        }
        return x;
    }

    int main(int argc, char *argv[]) {
        int N = 1000000;
        double results[128];
        double sm, lg;
        assert(omp_get_max_threads() <= 128);
        std::cout << "max_threads = " << omp_get_max_threads() << std::endl;

        Performance::Timers::on();
        #pragma omp parallel
        { results[omp_get_thread_num()] = run(N);
        }
        sm = lg = results[0];
        #pragma omp parallel reduction(min:sm) reduction(max:lg)
        { Performance::Timed timer("Reduce", 1, 1);
          sm = results[omp_get_thread_num()];
          lg = sm;
        }

        std::cout << "Ending x = " << std::endl;
        std::cout << 0.5*(sm+lg) << " +/- " << lg-sm << std::endl;

        Performance::Timers::show(std::cout);
    }


These kinds of performance timers are useful because they incur minimal
overhead.  Unless you are timing something that gets called
thousands of times, you can mostly leave them in your code and forget about them.

.. admonition:: Contributed by

   David M. Rogers

