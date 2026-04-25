#!/bin/bash
# Plot the token count or FLOPS count vs risk plots as the last step

set -e

source setup/start_env.sh

BASE=/home/ehghaghi/projects/aip-craffel/ehghaghi/prisk-pressure/results/multitrial_experiments/pressure_sensitivity

# Token cost axis — total tokens consumed by the attack up to each λ
uv run python plot_results.py \
    --metrics-csv $BASE/harmbench/qwen2.5-0.5b-instruct/cost_metrics.csv \
    --output-dir  $BASE/harmbench/qwen2.5-0.5b-instruct/plots_tokens \
    --x-axis tokens

# FLOP cost axis — total TFLOPs consumed by the attack up to each λ
uv run python plot_results.py \
    --metrics-csv $BASE/harmbench/qwen2.5-0.5b-instruct/cost_metrics.csv \
    --output-dir  $BASE/harmbench/qwen2.5-0.5b-instruct/plots_flops \
    --x-axis flops
