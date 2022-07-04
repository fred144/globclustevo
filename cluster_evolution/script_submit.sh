#!/bin/bash
#SBATCH -J script_submit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=8000  
#SBATCH --exclusive
#SBATCH -t 24:00:00



#needed for bash in deepthought2 (cf. https://www.glue.umd.edu/hpcc/help/jobs.html#basic)
. ~/.profile


export LANG=en_US


module load gcc
module load openmpi


SCRIPT="/lustre/fgarcia4/glob_clust_evo/cluster_evolution/fs035_gas_proj_movie.py"
#SCRIPT="profiler.py"
source ~/py-virtual-envs/master/bin/activate
# cd ~
# source /homes/fgarcia4/py-virtual-envs/clean-install/bin/activate.csh
# which python
# cd /lustre/fgarcia4/glob-clust-evo/luminosity_mapping
# cd sbatch_logs
# for rockstar --mca btl ^openib
# number of cores specified above
mpirun python3 $SCRIPT  2>&1 

