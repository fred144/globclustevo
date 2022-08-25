#!/bin/bash
#SBATCH -J renderer
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=16
#SBATCH -t 36:00:00

# needed for bash in deepthought2 (cf. https://www.glue.umd.edu/hpcc/help/jobs.html#basic)
. ~/.profile

# designed to be submitted/ ran in the script directory
export LANG=en_US
module load gcc
module load openmpi
source ~/py-virtual-envs/master/bin/activate

SCRIPT="/lustre/fgarcia4/glob_clust_evo/visuals/movie_gas_projection.py"

# number of cores specified above
mpirun python3 $SCRIPT  2>&1 
