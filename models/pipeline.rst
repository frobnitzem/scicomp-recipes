Assembly Line
#############

The practice of forming an assembly line dates back at least to the
ancient Romans.  In computational work, you sometimes
want to get each CPU and GPU to perform the same task on
different work items -- handing them off to the next worker
after each is done.

CUDA accomplishes this with its ``streams`` API,
while a fully-CPU implementation would use locks
to wait on ``ready`` events.

Both methods are combined in the ``Event`` class below::

    #include <unistd.h>
    #include <omp.h>
    #include <stdio.h>
    #include <mutex>

    //#define ENABLE_CUDA

    #ifdef ENABLE_CUDA
    #include <cuda_runtime.h>
    #define CHECKCUDA(cmd) do {                         \
      cudaError_t e = cmd;                              \
      if( e != cudaSuccess ) {                          \
        printf("Failed: Cuda error %s:%d '%s'\n",       \
            __FILE__,__LINE__,cudaGetErrorString(e));   \
        exit(EXIT_FAILURE);                             \
      }                                                 \
    } while(0)
    #else
    // skip CUDA commands in non-CUDA mode
    #define CHECKCUDA(cmd)
    typedef void* cudaEvent_t;
    typedef void* cudaStream_t;
    #endif

    struct Event {
        Event() {
            CHECKCUDA( cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) );
            sync.lock();
        }
        Event(const Event &) = delete;
        Event &operator=(Event &) = delete;

        ~Event() {
            CHECKCUDA( cudaEventDestroy(ev) );
        }
        void record(cudaStream_t stream) {
            CHECKCUDA( cudaEventRecord(ev, stream) );
            sync.unlock();
        }
        void wait(cudaStream_t stream) {
            sync.lock(); // will not succeed until record call has completed
            CHECKCUDA( cudaEventSynchronize(ev) );
            //CHECKCUDA( cudaStreamWaitEvent(stream, ev, 0) );
            // leave in the locked state so that another record() is needed.
        }

        std::mutex sync;
        cudaEvent_t ev;
    };

    // do work
    void work(cudaStream_t stream, const char *name, int i, int time) {
        //CHECKCUDA( cudaStreamSynchronize(stream) );
        usleep(time*1000); //  do some work on stream A
        printf("%s item %d\n", name, i);
    }

    int main(int argc, char *argv[]) {
        cudaStream_t sA, sB;
        CHECKCUDA( cudaStreamCreateWithFlags(&sA, cudaStreamNonBlocking) );
        CHECKCUDA( cudaStreamCreateWithFlags(&sB, cudaStreamNonBlocking) );

        Event a[2], b[2]; // Events, named after recording process.

        #pragma omp parallel num_threads(2)
        {
            int thread  = omp_get_thread_num();
            int threads = omp_get_num_threads();
            printf("thread %d/%d\n", thread, threads);
            for(int i=0; i<10; i++) {
                if(threads != 2 || thread == 0) { // A
                    if(i > 1) b[i%2].wait(sA); // wait for resources returned from B2
                    work(sA, "A", i, 100);
                    a[i%2].record(sA);
                }

                if(threads != 2 || thread == 1) { // B
                    a[i%2].wait(sB);
                    work(sB, "B", i, 300);
                    b[i%2].record(sB);
                }
            }
        }

        CHECKCUDA( cudaStreamDestroy(sB) );
        CHECKCUDA( cudaStreamDestroy(sA) );
        return 0;
    }

Worker ``A`` works on item ``A1``, then signals (records)
``Event A1`` and moves on to ``A2``.
Worker ``B`` waits for ``Event A1``, processes it into item ``B1``,
then signales (records) ``Event B1``.
Eventually worker ``A`` completes ``A2``, and moves onto ``A3``.
Here, however, it can just re-use resources from ``A1``,
as long as it first waits for ``Event B2``.

In actual application code, worker ``A`` allocates two memory
resources, ``A1`` and ``A2``, and calls ``std::swap(A1,A2)`` every time it
completes a step.  This way ``A`` always writes to ``A1``
and ``B`` always reads from ``A2``.
Worker ``B`` can do something similar.
This strategy is known as double-buffering.

There's a quirk with CUDA events, where they don't register
until recorded.  We get around this by requiring a wait on
a CPU-lock so that ``cudaEventSynchronize`` can't get called
until after the ``cudaEventRecord`` has been called.
This CPU-lock has the side-effect of creating all the necessary
synchronization for a CPU-only implementation.

Unfortunately, ``cudaStreamWaitEvent`` can't be called when
waiting for the event, since it doesn't block the CPU.
This would make both GPU-only and mixed CPU/GPU implementations
incorrect.  For GPU-only implementations, subsequent events
would get recorded too early.
The mixed CPU/GPU implementation would get into trouble when
waiting starting a CPU operation that required prior GPU
events to have completed.

The trade-off is a few milliseconds of kernel launch latency
as well as synchronization latency from cudaStreamWaitEvent itself.
It's possible to eek out more performance by using
a vector of events (getting around the "too early recording"
problem) and uncommenting the ``cudaStreamSynchronize``
call for every CPU function.

