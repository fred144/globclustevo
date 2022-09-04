#!/bin/bash
#SBATCH -J dm_finder
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH -t 48:00:00



# needed for bash in deepthought2 (cf. https://www.glue.umd.edu/hpcc/help/jobs.html#basic)
. ~/.profile

# designed to be submitted/ ran in the script directory
export LANG=en_US
module load gcc
module load openmpi
source ~/py-virtual-envs/master/bin/activate


SCRIPT1="/lustre/fgarcia4/glob_clust_evo/dm/fs070_dm_halo_finder.py"
SCRIPT2="/lustre/fgarcia4/glob_clust_evo/dm/fs035_dm_halo_finder.py"

# source /homes/fgarcia4/py-virtual-envs/clean-install/bin/activate.csh
# which python
# for rockstar --mca btl ^openib

# number of cores specified above
mpirun python3 $SCRIPT1  2>&1 
mpirun python3 $SCRIPT2  2>&1 
