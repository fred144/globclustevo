#!/bin/bash
#SBATCH -J script_submit
#SBATCH -n 8
#SBATCH --mem-per-cpu=16024  
#SBATCH --exclusive
#SBATCH -t 12:00:00



#needed for bash in deepthought2 (cf. https://www.glue.umd.edu/hpcc/help/jobs.html#basic)
. ~/.profile

set -x

NPE=8;                            # process number
export LANG=en_US


module load gcc
module load openmpi

STDOUT="sh_log"
SCRIPT="/lustre/fgarcia4/glob_clust_evo/cluster_evolution/main_movie.py"
#SCRIPT="profiler.py"

# cd ~
# source /homes/fgarcia4/py-virtual-envs/clean-install/bin/activate.csh
# which python
# cd /lustre/fgarcia4/glob-clust-evo/luminosity_mapping
# cd sbatch_logs


mpirun -np $NPE python3 $SCRIPT  2>&1 | tee -a $STDOUT

