#!/bin/bash
# The line above this is the "shebang" line.  It must be first line in script
#-----------------------------------------------------
#	Open OnDemand Job Template
#	For a basic python job using MPI
#-----------------------------------------------------
#
# Slurm sbatch parameters section:
#	Request 60 tasks with one CPU core each
#SBATCH -n  60
#SBATCH -c 1
#	Request 5 minutes of walltime
#SBATCH -t 5
#	Request 1 GB of memory per CPU core
#SBATCH --mem-per-cpu=1024
#	Do not allow other jobs to run on same node
#SBATCH --exclusive
#	Run on debug partition for rapid turnaround.  You will need
#	to change this (remove the line) if walltime > 15 minutes
#SBATCH --partition=debug
#       Do not inherit the environment of the process running the
#       sbatch command.  This requires you to explicitly set up the
#       environment for the job in this script, improving reproducibility
#SBATCH --export=NONE
#

# This job will run the code in hello_mpi.py script from the submission dir
# We create a working directory on parallel file system and run the job
# from there, and then make a symlink to the working dir in the submission dir

# Section to ensure we have the "module" command defined
unalias tap >& /dev/null
if [ -f ~/.bash_profile ]; then
	source ~/.bash_profile
elif [ -f ~/.profile ]; then
	source ~/.profile
fi

# Set SLURM_EXPORT_ENV to ALL.  This prevents the --export=NONE flag
# from being passed to mpirun/srun/etc, which can cause issues.
# We want the environment of the job script to be passed to all 
# tasks/processes of the job
export SLURM_EXPORT_ENV=ALL

# Module load section
# First clear our module list 
module purge
# and reload the standard modules
module load hpcc/deepthought2
# Load the desired compiler, MPI, and python modules
# NOTE: You need to use the same compiler and MPI module used
# when compiling the python (and its mpi4py package).  The values
# listed below are correct, you may need to change them if you change
# the python version.
module load gcc/8.4.0
module load openmpi/3.1.5
module load python/3.7.7

# Section to make a scratch directory for this job
# Because different MPI tasks, which might be on different nodes, need
# access to it, we put it in lustre.  We include the SLURM jobid in the
# directory name to avoid interference if multiple jobs running at same time.
TMPWORKDIR="/lustre/$USER/ood-mpi4py.${SLURM_JOBID}"
mkdir $TMPWORKDIR
cd $TMPWORKDIR

# Section to output information identifying the job, etc.
echo "Slurm job ${SLURM_JOBID} running on"
hostname
echo "To run on ${SLURM_NTASKS} CPU cores across ${SLURM_JOB_NUM_NODES} nodes"
echo "All nodes: ${SLURM_JOB_NODELIST}"
date
pwd
echo "Loaded modules are:"
module list


# Setting this variable will suppress the warnings
# about lack of CUDA support on non-GPU enabled nodes.  We
# are not using CUDA, so warning is harmless.
export OMPI_MCA_mpi_cuda_support=0

# Get the full path to our python executable.  It is best
# to provide full path of our executable to , etc.
MYPYTHON=`which python`
echo "Using python $MYPYTHON"

# Run our script using mpirun
# We do not specify the number of tasks here, and instead rely on
# it defaulting to the number of tasks requested of Slurm
mpirun  ${MYPYTHON} ${SLURM_SUBMIT_DIR}/hello_mpi.py > hello.out 2>&1
# Save the exit code from the previous command
ECODE=$?

# Symlink our working dir back into submit dir
ln -s ${TMPWORKDIR} ${SLURM_SUBMIT_DIR}/work-dir

echo "Job finished with exit code $ECODE.  Working dir is $TMPWORKDIR"
date

# Exit with the cached exit code
exit $ECODE

