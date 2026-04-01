#!/bin/bash
# Run Phase 3 plotting for all experiments (risk curves + category plots).
# Usage: bash run_plots.sh
# Requires: must be run from the project root.
# Note: runs directly on the login node (no GPU needed — pure post-processing).

set -e

source setup/start_env.sh

# python plot_results.py \
#     --metrics-csv /home/ehghaghi/scratch/ehghaghi/outputs/pressure_sensitivity/metrics.csv \
#     --category-metrics-csv /home/ehghaghi/scratch/ehghaghi/outputs/pressure_sensitivity/metrics_by_category.csv \
#     --output-dir /home/ehghaghi/scratch/ehghaghi/outputs/pressure_sensitivity/plots

# python plot_results.py \
#     --metrics-csv /home/ehghaghi/scratch/ehghaghi/outputs/attack_sensitivity/metrics.csv \
#     --category-metrics-csv /home/ehghaghi/scratch/ehghaghi/outputs/attack_sensitivity/metrics_by_category.csv \
#     --output-dir /home/ehghaghi/scratch/ehghaghi/outputs/attack_sensitivity/plots

# python plot_results.py \
#     --metrics-csv /home/ehghaghi/scratch/ehghaghi/outputs/training_stage/metrics.csv \
#     --category-metrics-csv /home/ehghaghi/scratch/ehghaghi/outputs/training_stage/metrics_by_category.csv \
#     --output-dir /home/ehghaghi/scratch/ehghaghi/outputs/training_stage/plots

# python plot_results.py \
#     --metrics-csv /home/ehghaghi/scratch/ehghaghi/outputs/open_vs_closed/metrics.csv \
#     --category-metrics-csv /home/ehghaghi/scratch/ehghaghi/outputs/open_vs_closed/metrics_by_category.csv \
#     --output-dir /home/ehghaghi/scratch/ehghaghi/outputs/open_vs_closed/plots


python plot_results.py \
    --metrics-csv /home/ehghaghi/scratch/ehghaghi/outputs/model_family/metrics.csv \
    --category-metrics-csv /home/ehghaghi/scratch/ehghaghi/outputs/model_family/metrics_by_category.csv \
    --output-dir /home/ehghaghi/scratch/ehghaghi/outputs/model_family/plots
