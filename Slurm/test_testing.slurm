#!/bin/bash
#SBATCH --job-name=test_testing
#SBATCH --output=Data/Outputs/%x_%j.out
#SBATCH --error=Data/Outputs/%x_%j.err
#SBATCH --mem-per-cpu=500M
#SBATCH --time=01:00:00
#SBATCH --partition=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15


echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"


echo "[$SHELL] ## Run script"
python3 run_test.py
echo "[$SHELL] ## Script finished"