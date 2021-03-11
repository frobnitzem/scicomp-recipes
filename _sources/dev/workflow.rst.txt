Setting Up Your Project Environment
###################################

Let's say you're just getting started on project id CHM101.
A standard workflow on Summit is to build and install project
sources to ``/ccs/proj/CHM101``, stage input/output files to
``/gpfs/alpine/proj-shared/CHM101``,
and use ``/gpfs/alpine/scratch/$USER/CHM101``
during a run.

A good project layout will usually look like:

.. code-block:: shell

    /ccs/proj/CHM101
        /spack         -- local spack installation for dependencies
        /cp2k-June2021 -- project's main application code
        /user1
           /python     -- local python environment
           /analysis1  -- user1's scripts for analysis
        README.md      -- explanation of your project layout, file locations and storage/transfer policy

    /gpfs/alpine/proj-shared/CHM101
        /user1
            /experiment1 -- config files, lsf (batch script), and key outputs
        /user2

    /gpfs/alpine/world-shared/CHM101
        /user1
            /shared-dataset

    /gpfs/alpine/scratch/user1/CHM101
        /experiment1 -- runtime scratch space for experiment1

Installing Software
===================

The first step is, of course, to download and install your application
codes and libraries.  For optimal performance, you'll want to do these
steps yourself or have a way to check that they are performing correctly.

For example, the :doc:`apps/CP2K` program depends largely
on GPU-enabled libraries for using the GPUs.

It's important to check the available modules for available system-specific,
optimized dependencies.

.. code-block:: shell

    module spider mpi

It's good practice to document your module environment
by creating a setup script.  Here's an example
from cp2k.

.. code-block:: shell

    #!/bin/bash
    # /ccs/proj/CHM101/cp2k-June2021/env.sh
    # to use this, run "source /ccs/proj/CHM101/cp2k-June2021/env.sh"

    module load cmake
    module load gcc
    module load python/3
    module load spectrum-mpi
    module load cuda
    module load hdf5
    module load openblas
    module load netlib-scalapack # includes blas, CPU only
    # non-threaded BLAS is preferred (TODO)
    module load fftw/3
    module load gsl

Useful Info
-----------

On Summit, you should compile on the login nodes,
but stick to less than 4 or 8 processors
to be kind to other users.

Summit is an IBM system, and uses spectrum-mpi
as provided by a module.  It is based on openmpi 4.0.1,
and supports the MPI 3.2 standard.
(https://www.ibm.com/support/knowledgecenter/SSZTET_10.3/smpi_overview.html).
Spectrum updates are deployed every 6 months or so.
Upgrades and downtimes are announced weeks in advanced,
and put on the facility calendar.
Old modules become deprecated (but still available).
Keep track of your build process, because you will probably 
want to re-build on these updates.

IBM has ESSL as a BLAS library, but it doesn't include
all the lapack functions.  We also provide a netlib-scalapack
library that includes lapack/blas, as well as a separate openblas
library.

Summit's hardware is documented at
https://docs.olcf.ornl.gov/systems/summit_user_guide.html#system-overview
The V100 GPUs have compute capability 70, so the usual flag
passed to nvcc is --gpu_arch=sm_70.

Compilation and package choices should be informed by available
modules and/or the difficulty of building those libraries
manually.  Work incrementally if possible.

Usually, you can get important public, core libraries turned
into modules by emailing the help desk at help@olcf.ornl.gov.
But be sure you have tried them first and it's what you really want
(so you have a complete request to email).

To test out your environment and develop LSF batch scripts
for running jobs, you can run interactively.

.. code-block:: shell

    bsub -P CHM101 -nnodes 1 -q debug -W 30 -Is $SHELL

Note that your interactive jobs are running on a launch node,
which is shared with other users.  You should limit yourself
to one processor on the launch nodes, and use jsrun to

Don't compile on the launch nodes.  If you must, use jsrun
(equivalent of mpirun) to run make on a summit node instead.

When creating job scripts, it's a good idea to test your
launch configuration using https://jobstepviewer.olcf.ornl.gov/.

