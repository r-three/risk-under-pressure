#!/bin/bash
# Run Phase 2 evaluation for all experiments (metrics + metrics by category).
# Usage: bash run_evaluations.sh
# Requires: must be run from the project root.
# Note: runs directly on the login node (no GPU needed — pure post-processing).

set -e

source setup/start_env.sh

python run_evaluation.py \
    --results-dir /home/ehghaghi/scratch/ehghaghi/outputs/pressure_sensitivity \
    --format csv \
    --output /home/ehghaghi/scratch/ehghaghi/outputs/pressure_sensitivity/metrics.csv \
    --print-table \
    | tee /home/ehghaghi/scratch/ehghaghi/outputs/pressure_sensitivity/summary.txt

# python run_evaluation.py \
#     --results-dir /home/ehghaghi/scratch/ehghaghi/outputs/attack_sensitivity \
#     --format csv \
#     --output /home/ehghaghi/scratch/ehghaghi/outputs/attack_sensitivity/metrics.csv \
#     --print-table \
#     | tee /home/ehghaghi/scratch/ehghaghi/outputs/attack_sensitivity/summary.txt

# python run_evaluation.py \
#     --results-dir /home/ehghaghi/scratch/ehghaghi/outputs/training_stage \
#     --format csv \
#     --output /home/ehghaghi/scratch/ehghaghi/outputs/training_stage/metrics.csv \
#     --print-table \
#     | tee /home/ehghaghi/scratch/ehghaghi/outputs/training_stage/summary.txt

# python run_evaluation.py \
#     --results-dir /home/ehghaghi/scratch/ehghaghi/outputs/open_vs_closed \
#     --format csv \
#     --output /home/ehghaghi/scratch/ehghaghi/outputs/open_vs_closed/metrics.csv \
#     --print-table \
#     | tee /home/ehghaghi/scratch/ehghaghi/outputs/open_vs_closed/summary.txt

# python run_evaluation.py \
#     --results-dir /home/ehghaghi/scratch/ehghaghi/outputs/model_family \
#     --format csv \
#     --output /home/ehghaghi/scratch/ehghaghi/outputs/model_family/metrics.csv \
#     --print-table \
#     | tee /home/ehghaghi/scratch/ehghaghi/outputs/model_family/summary.txt