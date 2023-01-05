#!/bin/bash
#SBATCH -J gas_renderer
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=5990
#SBATCH -t 24:00:00

# needed for bash in deepthought2 (cf. https://www.glue.umd.edu/hpcc/help/jobs.html#basic)
#. ~/.bashrc
. ~/.profile

# designed to be submitted/ ran in the script directory
export LANG=en_US
#module unload intel
#module unload intel-mpi
#module load gcc
#module load openmpi
module load python
module load gcc
module load openmpi
source ~/scratch/master/bin/activate


#SCRIPT="/lustre/fgarcia4/glob_clust_evo/visuals/fs070_gas_panel_projection.py"
SCRIPT= "/scratch/zt1/project/ricotti-prj/user/fgarcia4/globclustevo/visuals/low-sfe.py"
#cd /scratch/zt1/project/ricotti-prj/user/fgarcia4/globclustevo/visuals/

# number of cores specified above
#mpirun python3 $SCRIPT  2>&1 
mpirun $SCRIPT  2>&1 
