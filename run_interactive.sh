#!/bin/bash
# Launch an interactive L40S session and run a single experiment.
# Usage: bash run_interactive.sh <experiment_yaml> [extra args...]
# Example: bash run_interactive.sh configs/experiments/pressure_sensitivity.yaml --n-prompts 5

if [ -z "$1" ]; then
    echo "Usage: bash run_interactive.sh <experiment_yaml> [extra args...]"
    exit 1
fi

EXPERIMENT="$1"
shift
EXTRA_ARGS="$*"

#srun --gres=gpu:l40s:1 --cpus-per-task=8 --mem=128GB \
#     --partition=gpubase_l40s_b3 --account=aip-craffel \
#     --time=02:00:00 --pty bash -c \
#     "source setup/start_env.sh && python scripts/run_inference.py --experiment $EXPERIMENT --resume $EXTRA_ARGS"

srun --account=def-craffel --nodes=1 --gpus-per-node=h100:1 \
     --cpus-per-task=32 --mem=480G --time=08:00:00 --pty bash -c \
     "source setup/start_env.sh && python scripts/run_inference.py --experiment $EXPERIMENT --resume $EXTRA_ARGS"