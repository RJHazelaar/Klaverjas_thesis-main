#!/bin/bash
#SBATCH --job-name=process_data
#SBATCH --output=Data/Outputs/%x_%j.out
#SBATCH --error=Data/Outputs/%x_%j.err
#SBATCH --mem-per-cpu=5G
#SBATCH --time=00:30:00 
#SBATCH --partition=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15


echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
python3 Data/SL_Data/process_data.py
echo "[$SHELL] ## Script finished"

echo "[$SHELL] #### Finished Python test. Have a nice day"