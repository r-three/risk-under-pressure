#!/bin/bash
# Submit all pRisk-Pressure experiments to the Killarney cluster.
# Usage: bash run_experiments.sh
# Requires: must be run from the project root on a klogin* node.

set -e

source setup/start_env.sh

# submit "rup_pressure_sensitivity" \
#     "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity.yaml --seeds 1997 2 100 --resume"

# submit "rup_attack_sensitivity" \
#     "python scripts/run_inference.py --experiment configs/experiments/attack_sensitivity.yaml --seeds 42 123 456 --resume"

# submit "rup_open_vs_closed" \
#     "python scripts/run_inference.py --experiment configs/experiments/open_vs_closed.yaml --seeds 42 123 456 --resume"

# submit "rup_attacker_comparison" \
#     "python scripts/run_inference.py --experiment configs/experiments/attacker_comparison.yaml --seeds 42 123 456 --resume"

submit "rup_template_ordering" \
    "python scripts/compute_template_ordering.py --experiment configs/experiments/template_ordering.yaml --seeds 42 --resume"
