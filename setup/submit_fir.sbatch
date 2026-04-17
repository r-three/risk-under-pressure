#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --partition=gpubase_bygpu_b3
#SBATCH --account=def-craffel_gpu

hostname

export OFFLINE_MODE=0

source setup/start_env.sh

eval "$@"
