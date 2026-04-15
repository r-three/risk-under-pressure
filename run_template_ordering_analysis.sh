#!/bin/bash
# Analyze template ordering experiment results: compute mean/std ASR per template
# and generate bar chart comparisons across models.
# Usage: bash run_template_ordering_analysis.sh
# Requires: must be run from the project root.

set -e

source setup/start_env.sh

python scripts/analyze_template_ordering.py \
    --results-dir /home/ehghaghi/scratch/ehghaghi/outputs/template_ordering \
    --plots-dir /home/ehghaghi/scratch/ehghaghi/outputs/results/plots \
    --output /home/ehghaghi/scratch/ehghaghi/outputs/results/template_stats.csv
