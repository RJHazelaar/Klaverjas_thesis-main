#!/bin/bash
#SBATCH --job-name=test_cpu-short
#SBATCH --output=Data/Outputs/%x_%j.out
#SBATCH --error=Data/Outputs/%x_%j.err
#SBATCH --mem-per-cpu=500M
#SBATCH --time=04:00:00
#SBATCH --partition=cpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20


echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"


echo "[$SHELL] ## Run script"
python3 run_test.py
echo "[$SHELL] ## Script finished"