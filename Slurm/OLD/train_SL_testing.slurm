#!/bin/bash
#SBATCH --job-name=train_nn_SL_testing
#SBATCH --output=Data/Outputs/%x_%j.out
#SBATCH --error=Data/Outputs/%x_%j.err
#SBATCH --mem-per-cpu=30G
#SBATCH --time=01:00:00
#SBATCH --partition=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1


echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
python3 train_nn_SL.py
echo "[$SHELL] ## Script finished"

echo "[$SHELL] #### Finished Python test. Have a nice day"