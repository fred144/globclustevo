#!/bin/bash
#SBATCH -J script_submit
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6000  
#SBATCH --exclusive
#SBATCH -t 12:00:00



#needed for bash in deepthought2 (cf. https://www.glue.umd.edu/hpcc/help/jobs.html#basic)
. ~/.profile


export LANG=en_US


module load gcc
module load openmpi


SCRIPT="/lustre/fgarcia4/glob_clust_evo/cluster_evolution/main_movie.py"
#SCRIPT="profiler.py"

# cd ~
# source /homes/fgarcia4/py-virtual-envs/clean-install/bin/activate.csh
# which python
# cd /lustre/fgarcia4/glob-clust-evo/luminosity_mapping
# cd sbatch_logs
# for rockstar --mca btl ^openib
# number of cores specified above
mpirun python3 $SCRIPT  2>&1 

