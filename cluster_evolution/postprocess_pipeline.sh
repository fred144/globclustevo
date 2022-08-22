#!/bin/bash
#SBATCH -J postprocess_pipeline
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 72:00:00



# needed for bash in deepthought2 (cf. https://www.glue.umd.edu/hpcc/help/jobs.html#basic)
. ~/.profile

# designed to be submitted/ ran in the script directory
export LANG=en_US
module load gcc
module load openmpi
source ~/py-virtual-envs/master/bin/activate


SCRIPT1="/lustre/fgarcia4/glob_clust_evo/cluster_evolution/fs070_postprocess_pipeline.py"
SCRIPT2="/lustre/fgarcia4/glob_clust_evo/cluster_evolution/fs035_postprocess_pipeline.py"

# source /homes/fgarcia4/py-virtual-envs/clean-install/bin/activate.csh
# which python
# for rockstar --mca btl ^openib

# number of cores specified above
mpirun python3 $SCRIPT1  2>&1 
mpirun python3 $SCRIPT2  2>&1 
