#!/bin/bash
#SBATCH --job-name=test_cpu-medium
#SBATCH --output=Data/Outputs/%x_%j.out
#SBATCH --error=Data/Outputs/%x_%j.err
#SBATCH --mem-per-cpu=500M
#SBATCH --time=20:00:00
#SBATCH --partition=cpu-medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20


echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
python3 test_agent2.py
echo "[$SHELL] ## Script finished"

echo "[$SHELL] #### Finished Python test. Have a nice day"