Making a Strong-Scaling Plot
############################

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
    #BSUB -nnodes 10
    #BSUB -J scaling
    #BSUB -o scaling.%J

    . /ccs/proj/chm101/setup-env.sh

    EXE=$PWD/the_code

    echo "Starting strong scaling calculation of $EXE v. 0.2.0"
    let nodes=(LSB_MAX_NUM_PROCESSORS-1)/42
    let min_nodes=nodes/16

    mkdir -p $PROJWORK/chm101/scaling
    cd $PROJWORK/chm101/scaling
    export OMP_NUM_THREADS=7

    while [ $nodes -ge $min_nodes ]; do
       echo "Starting copy on $nodes nodes at `date`"
       jsrun -n $(())
       echo "Starting run on $nodes nodes at `date`"
       jsrun -n $((6*$nodes)) -g 1 -c 7 -b packed:7 $EXE
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
   (Summit has 42 per node)

 * One GPU per MPI rank and 7 CPUs is a good starting point
   for optimization.

 * Printing as much timing output as possible allows you more
   chances to recover in case the run fails unexpectedly.

 * Printing out the name and version of the executable is
   really helpful for doing retrospectives later.

 * Start scaling runs with small nnodes and check sizes, etc.
   Within a run, you should start at the highest node count
   since it should complete faster.

Staging input files to the NVME is good practice.  For details,
see the `user guide <https://docs.olcf.ornl.gov/systems/summit_user_guide.html?highlight=smpi%20args#current-nvme-usage>`_.


