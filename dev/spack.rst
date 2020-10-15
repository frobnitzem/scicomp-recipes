Compiling Using Spack
#####################

.. epigraph::

  Give me six hours to chop down a tree and I will spend
  the first four sharpening the axe.

  -- Abraham Lincoln

High-performance computing systems are large and complex,
with many moving parts.  Being a productive developer
requires having a stable, working set of development tools,
libraries, and systems at ready.

`Spack <https://spack.readthedocs.io/en/latest/index.html>`_
is a package manager for scientific software
stacks that allows multiple versions and variations of
each software to be installed simultaneously.
For example, gromacs has both MPI and non-MPI,
GPU and non-GPU, as well as single and double precision variants.
Package managers for most Linux distributions use global
installation locations and a single, standardized set
of options.  Ubuntu linux defaults to installing
the single-precision, non-MPI, non-GPU gromacs.
In contrast, spack creates parallel
installs so that all :math:`2^3` variants of gromacs
can be installed without conflicts.

This solution is much nicer than naming the libraries
and header files of every variant differently.
Instead, each variant is installed into a directory like:

.. code-block:: shell

  base/archname/compilername/pkgname-ver-hash

Installing Spack
----------------

Every computing will have their own specific guidelines.
Most already use some form of spack to build their
software.  For example, see

* `NERSC Spack <https://github.com/NERSC/spack>`_


.. note::
  Ideally, one spack program would exist on
  each computing system.  Configuration files would exist
  at the system-level, project-level, and user-level.

  Each level would install packages into its respective
  visibility level (system, project, or user).
  
  In such a setup, ``config.yaml`` files at each level
  would set their own values for ``install_tree``,
  ``build_stage``, and ``source_cache``, while allowing
  searching higher levels using ``upstreams.yaml``
  as explained in
  `spack chaining <https://spack.readthedocs.io/en/latest/chain.html>`_.

At present, most systems expect you to install your own copy of spack,
and then configure ``compilers.yaml`` and
``packages.yaml`` to point to key system-installed infrastructure
like vendor-specific compilers, MPI, Blas/Lapack/MKL, and
common/complicated libraries like FFTW, HDF5/netcdf, PETSC, papi,
trilinos, etc.

.. code-block:: bash

  $ git clone https://github.com/spack/spack.git
  $ source spack/share/spack/setup-env.sh 
  $ spack install kokkos+aggressive_vectorization+cuda+enable_lambda+openmp~serial cxxstd=c++11 host_arch=HSW gpu_arch=Volta70

Quick Reference
===============

After getting spack installed and running, use the following steps
to setup the libaries and programming language tools mentioned
in this book.

.. code-block:: bash

  $ git clone https://bitbucket.org/icldistcomp/parsec.git
  $ cd parsec
  $ spack repo add $PWD/contrib/spack
  $ spack install parsec@devel
  $ spack load parsec


TODO
----

Here are some steps that would help future users:

#. build a ready-made docker image
#. document some strategies for directory tree layout

