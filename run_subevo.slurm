#!/bin/bash
#SBATCH -J SubEvo_jobs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=50
#SBATCH --mem-per-cpu=4G
#SBATCH --time=72:00:00
#SBATCH --array=[150-200]
#SBATCH --export=ALL
#SBATCH --mail-type=NONE
#SBATCH --mail-user=jiaxuanl@princeton.edu
#SBATCH -o ./logs/evo/subEvo.%a.o
#SBATCH --account=jennyg

now=$(date +"%T")
echo "start time ... $now"
echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

module load anaconda3/2024.2

cd /home/jiaxuanl/Research/SatGen/

# Each array task selects one host-mass bin from the available OUTPUT_TREE files.
# Update the array range above if the number of available mass bins changes.
python -u SubEvo.py ${SLURM_CPUS_PER_TASK}

now=$(date +"%T")
echo "end time ... $now"
