#!/bin/bash
#SBATCH -J Tree_jobs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=25
#SBATCH --mem-per-cpu=4G
#SBATCH --time=11:59:00
#SBATCH --array=[150-190]               # job array with index values 0, 1, 2, ..., 298, 299
#SBATCH --export=ALL
#SBATCH --mail-type=NONE
#SBATCH --mail-user=jiaxuanl@princeton.edu
#SBATCH -o ./logs/tree/dwarfTree.%a.o
#SBATCH --account=jennyg

now=$(date +"%T")
echo "start time ... $now"
echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)


# module purge
# . /home/jiaxuanl/Research/setup_env.sh
# SBATCH --mem=64G
module load anaconda3/2024.2
# module load lsst

cd /home/jiaxuanl/Research/SatGen/

# python -u TreeGen_Sub.py 100

python -u TreeGen_Sub.py 100

# try:
#     python TreeGen.py 19
# except:
#     python TreeGen.py 19


now=$(date +"%T")
echo "end time ... $now"