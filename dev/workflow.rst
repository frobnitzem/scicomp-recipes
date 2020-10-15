Setting Up Your Project Environment
###################################

Let's say you're just getting started on project id CHM101.
A standard workflow on Summit is to build and install project
sources to ``/ccs/proj/CHM101``, stage input/output files to
``/gpfs/alpine/proj-shared/CHM101``,
and use ``/gpfs/alpine/scratch/$USER/CHM101``
during a run.

You should also strongly consider
having each MPI rank do all file-per-process I/O
to ``/mnt/bb/$USER``, as described in
`Using the NVME <https://docs.olcf.ornl.gov/systems/summit_user_guide.html#current-nvme-usage>_`.

Installing Software
===================

Here's the process of querying system information and installing
cp2k - a quantum chemistry package with lots of dependencies.

.. code-block:: shell

    cd /ccs/proj/CHM101
    git clone --depth 1 --recurse-submodules https://github.com/cp2k/cp2k.git cp2k

Going through the dependency list, `make` is there, but
`python`-3 comes from a module.

.. code-block:: shell

    module avail python

It's good practice to collect these into a standard environment.
We'll end up with the following,

.. code-block:: shell

    #!/bin/bash
    # /ccs/proj/CHM101/cp2k-env.sh
    # to use this, run "source /ccs/proj/CHM101/cp2k-env.sh"

    module load cmake
    module load gcc
    module load python/3
    module load spectrum-mpi
    module load cuda
    module load openblas
    module load netlib-scalapack # includes blas, CPU only
    # non-threaded BLAS is preferred (TODO)
    module load fftw/3
    module load gsl

CP2K provides a toolchain to compiler its other dependencies,

.. code-block:: shell

    cd cp2k/tools/toolchain
    ./install_cp2k_toolchain.sh --help

    -j <n>                    Number of processors to use

                 On Summit, you can compile on the head node,
                 but stick to less than 4 or 8 processors
                 to be kind to other users.

    --mpi-mode=openmpi

                 Summit is an IBM system, and uses spectrum-mpi
                 as provided by a module.  It is based on openmpi.
                 Spectrum updates are deployed every 6 months or so.
                 Upgrades and downtimes are announced weeks in advanced,
                 and put on the facility calendar.
                 Old modules become deprecated (but still available).
                 Keep track of your build process, because you will probably 
                 want to re-build on these updates.

    --math-mode=openblas
                 
                 IBM has ESSL as a BLAS library, but it doesn't include
                 all the lapack functions.  We also provide a netlib-scalapack
                 library that includes lapack/blas, as well as a separate openblas
                 library.

    --enable-cuda=yes         You should definitely use GPUs when on Summit.
    --gpu-ver=V100

                 Summit's hardware is documented at
                 https://docs.olcf.ornl.gov/systems/summit_user_guide.html#system-overview
                 The V100 GPUs have compute capability 70, so the usual flag
                 passed to nvcc is --gpu_arch=sm_70.

    --enable-cray=no          Summit's vendor is IBM, not Cray.

                 Package choices, below, are mostly informed by available
                 modules and/or the difficulty of building those libraries
                 manually.  Work incrementally if possible.

                 Usually, you can get important public, core libraries turned
                 into modules by emailing the help desk at help@olcf.ornl.gov.
                 But be sure you have tried them first and it's what you really want
                 (so you have a complete request to email).

    --with-gcc=system         Provided by gcc module

    --with-cmake=system       Provided by cmake module

    --with-openmpi=system     Provided by spectrum-mpi module

    --with-fftw=system        Provided by the fftw/3 module

    --with-reflapack=no
    --with-acml=no
    --with-mkl=no
    --with-cosma=no           Replaces scalapack, we'll try keeping scalapack first.

    --with-openblas=system    Provided by the openblas module (CPU only).
    --with-scalapack=system   Provided by the netlib-scalapack module (CPU only).

    --with-elpa=no            ELPA works using GPU on Summit, but this
                              automated build isn't working. [I tried]

    --with-ptscotch=no        No module is available, can revisit if PEXSI is needed.
    --with-superlu=no         not using PEXSI right away.
    --with-pexsi=no

    --with-gsl=system         provided by the gsl module
    --with-hdf5=system        provided by hdf5 module

                 Ask the tool to install all of the following chemistry-specific
                 libraries locally:

    --with-libxc=install      The tool will install.
    --with-libint=install
    --with-spglib=install
    --with-sirius=install
    --with-spfft=install
    --with-libvdwxc=install
    --with-libsmm=install     We'll see how this works with CUDA.
    --with-libxsmm=no         x86_x64 is different than IBM's PPC (ppc64le)

After running `install_cp2k_toolchain.sh` with these options,

    ./install_cp2k_toolchain.sh --with-libxsmm=no --with-openblas=system \
         --with-mkl=no --with-scalapack=system --with-acml=no \
         --with-fftw=system --with-reflapack=no  --enable-cuda --enable-omp \
         --gpu-ver=V100 --dry-run
         --math-mode=

TODO
====
   
Complete this example with:

 * an lsf script, explain what the launch node is

 * I/O paths (write to /gpfs), expected IO throughput
   ~ 10 Mb/s in file-per-process mode, expect latency

 * saving software version and parameters in output

 * collecting profiling / timing data


