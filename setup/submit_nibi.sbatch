#!/bin/bash
#SBATCH --account=def-craffel
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=/home/bogi/scratch/results/prisk-pressure/slurm/%j.out
#SBATCH --error=/home/bogi/scratch/results/prisk-pressure/slurm/%j.err


# Show node name
hostname

# Set offline mode
export OFFLINE_MODE=0

# Source start env
source setup/start_env.sh

# Execute the command passed as argument
eval "$@"
