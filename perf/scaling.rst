Making a Scaling Plot
#####################

.. warning::

    This cut and paste code makes it simpler,
    but always exercise caution when ramping up
    to the whole machine.
    Especially watch out for unintended outputs like
    *log files* that can explode when node counts go up.

.. code-block:: bash

    # run.lsf
    #!/bin/bash
    #BSUB -P chm101
    #BSUB -W 0:10
    #BSUB -alloc_flags NVME
    #BSUB -nnodes 16
    #BSUB -J scaling
    #BSUB -o scaling.%J

    . /ccs/proj/chm101/setup-env.sh

    EXE=$PWD/the_code
    COPY=$PWD/the_copy_script

    echo "Starting scaling calculation of $EXE"
    ver=`git rev-parse HEAD`
    if [ $? -eq 0 ]; then
      echo "Git commit hash = $ver"
      echo "Plus Diffs"
      echo "--------------------------------------------------------"
      git diff $ver | cat
      echo "--------------------------------------------------------"
      echo
    fi

    let nodes=(LSB_MAX_NUM_PROCESSORS-1)/42
    let min_nodes=nodes/16

    mkdir -p $PROJWORK/chm101/scaling
    cd $PROJWORK/chm101/scaling
    export OMP_NUM_THREADS=7

    while [ $nodes -ge $min_nodes ]; do
       echo "Starting copy on $nodes nodes at `date`"
       jsrun -n $nodes -c7 -b packed:7 $COPY
       echo "Starting run on $nodes nodes at `date`"
       jsrun -n $((6*$nodes)) -g1 -c7 -b packed:7 $EXE
       let nodes=nodes/2
    done

There are some really useful coventions demonstrated here:

 * In the bsub flags, we request the node-local buffer/disk drives (NVME).

 * The job output file is named something useful. (-o)

 * There is a setup-env.sh script that loads all the needed modules.

   This prevents us from having to load modules on login (which pollutes
   the module space) and from forgetting to load modules and having
   the job crash.

 * We can calculate number of nodes from the number of processors
   (Summit has 42 cores per node)

 * One GPU per MPI rank and 7 CPUs is a good starting point
   for optimization.  Always use ``-b`` to bind ranks
   to cores.

 * Printing out the name and full version of the source used is
   really helpful for doing retrospectives later.

   This code assumes you're using git.  Hopefully that's the
   case.  If your job-scripts live in a different area, you
   could version them too.  None of this is a substitute
   for your executable printing its own version and timing info.

 * Printing as much timing output as possible allows you more
   chances to recover in case the run fails unexpectedly.

 * Start scaling runs with small nnodes and check sizes, etc.
   Within a run, you should start at the highest node count
   since it should complete faster.

  * Staging input files to the NVME is good practice.
    
    For details,
    see the `user guide <https://docs.olcf.ornl.gov/systems/summit_user_guide.html?highlight=smpi%20args#current-nvme-usage>`_.

The copy script mentioned would do well to use
the MPI rank, user-name, and jobid variables
to copy outputs to /mnt/bb like so:

.. code-block:: bash

    #!/bin/bash
    # the_copy_script

    set -e
    dir=/mnt/bb/$USER/$OMPI_COMM_WORLD_RANK
    [ -d $dir ] && rm -fr $dir
    mkdir -p $dir
    echo "creating output dir $dir for job $LSB_JOBID"
    cp /ccs/proj/chm101/inp.$OMPI_COMM_WORLD_RANK.dat $dir/

This allows each rank to access a collision-free
space from the others.

.. admonition:: Contributed by

   David M. Rogers

