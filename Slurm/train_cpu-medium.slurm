#!/bin/bash
#SBATCH --job-name=train_cpu-medium
#SBATCH --output=Data/Outputs/%x_%j.out
#SBATCH --error=Data/Outputs/%x_%j.err
#SBATCH --mem-per-cpu=1G
#SBATCH --time=24:00:00
#SBATCH --partition=cpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20


echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"


echo "[$SHELL] ## Run script"
python3 run_train.py
echo "[$SHELL] ## Script finished"