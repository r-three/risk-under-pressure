#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --account=aip-craffel

# Show node name
hostname

# Set offline mode
export OFFLINE_MODE=0

# Source start env
source setup/start_env.sh

# Execute the command passed as argument
eval "$@"
