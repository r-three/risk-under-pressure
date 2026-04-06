#!/bin/bash
# Submit all pRisk-Pressure experiments to the Killarney cluster.
# Usage: bash run_experiments.sh
# Requires: must be run from the project root on a klogin* node.

set -e

source setup/start_env.sh

submit "prisk_pressure_sensitivity" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity.yaml --resume"

#submit "prisk_attack_sensitivity" \
#    "python scripts/run_inference.py --experiment configs/experiments/attack_sensitivity.yaml --resume"

#submit "prisk_training_stage" \
#    "python scripts/run_inference.py --experiment configs/experiments/training_stage.yaml --resume"

#submit "prisk_open_vs_closed" \
#    "python scripts/run_inference.py --experiment configs/experiments/open_vs_closed.yaml --resume"
