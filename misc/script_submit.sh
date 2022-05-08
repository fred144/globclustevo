#!/bin/bash
#SBATCH -J script_submit
#SBATCH -n 1
#SBATCH --mem-per-cpu=32024  
#SBATCH -t 12:00:00

#SBATCH --mail-user=fgarcia4@umd.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

#needed for bash in deepthought2 (cf. https://www.glue.umd.edu/hpcc/help/jobs.html#basic)
. ~/.profile

set -x

NPE=1;                            # process number
export LANG=en_US


module load gcc
module load openmpi

STDOUT="sh_log"
SCRIPT="/lustre/fgarcia4/glob-clust-evo/halo_data/halo_finder.py"
#SCRIPT="profiler.py"

cd ~
source /homes/fgarcia4/py-virtual-envs/clean-install/bin/activate.csh
which python
#cd /lustre/fgarcia4/glob-clust-evo/luminosity_mapping
cd sbatch_logs


mpirun -np $NPE python3 $SCRIPT  2>&1 | tee -a $STDOUT
deactivate
