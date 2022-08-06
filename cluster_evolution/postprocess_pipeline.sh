#!/bin/bash
#SBATCH -J fof_halo_run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH -t 24:00:00



# needed for bash in deepthought2 (cf. https://www.glue.umd.edu/hpcc/help/jobs.html#basic)
. ~/.profile

# designed to be submitted/ ran in the script directory
export LANG=en_US
module load gcc
module load openmpi
source ~/py-virtual-envs/master/bin/activate


SCRIPT="/lustre/fgarcia4/glob_clust_evo/halo_data/fs070_postprocess_pipeline.py"

# source /homes/fgarcia4/py-virtual-envs/clean-install/bin/activate.csh
# which python
# for rockstar --mca btl ^openib

# number of cores specified above
mpirun python3 $SCRIPT  2>&1 

