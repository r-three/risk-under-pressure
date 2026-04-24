#!/bin/bash
# Run Phase 2.5 cost evaluation for all experiments.
# Augments each metrics.csv with token and FLOP cost columns → cost_metrics.csv.
# No re-running of experiments needed — costs are derived post-hoc from JSONL records.
#
# Usage: bash run_cost_evaluations.sh
# Requires: must be run from the project root.
# Note: runs directly on the login node (no GPU needed — pure post-processing).

set -e

source setup/start_env.sh

BASE=/home/ehghaghi/projects/aip-craffel/ehghaghi/prisk-pressure/results/multitrial_experiments/pressure_sensitivity

# --------------------------------------------------------------------------- #
# HarmBench
# --------------------------------------------------------------------------- #

python scripts/compute_attack_costs.py \
    --results-dir $BASE/harmbench/qwen2.5-0.5b-instruct \
    --metrics-csv  $BASE/harmbench/qwen2.5-0.5b-instruct/metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/harmbench/qwen2.5-3b-instruct \
#     --metrics-csv  $BASE/harmbench/qwen2.5-3b-instruct/metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/harmbench/qwen2.5-7b-instruct \
#     --metrics-csv  $BASE/harmbench/qwen2.5-7b-instruct/metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/harmbench/qwen3-4b-saferl \
#     --metrics-csv  $BASE/harmbench/qwen3-4b-saferl/metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/harmbench/tulu2-7b-base \
#     --metrics-csv  $BASE/harmbench/tulu2-7b-base/metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/harmbench/tulu2-7b-sft \
#     --metrics-csv  $BASE/harmbench/tulu2-7b-sft/metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/harmbench/tulu2-7b-dpo \
#     --metrics-csv  $BASE/harmbench/tulu2-7b-dpo/metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/harmbench/tulu3-8b-base \
#     --metrics-csv  $BASE/harmbench/tulu3-8b-base/metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/harmbench/tulu3-8b-sft \
#     --metrics-csv  $BASE/harmbench/tulu3-8b-sft/metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/harmbench/tulu3-8b-dpo \
#     --metrics-csv  $BASE/harmbench/tulu3-8b-dpo/metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/harmbench/tulu3-8b-rlvr \
#     --metrics-csv  $BASE/harmbench/tulu3-8b-rlvr/metrics.csv

# --------------------------------------------------------------------------- #
# JailbreakBench
# --------------------------------------------------------------------------- #

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/jailbreakbench/qwen2.5-0.5b-instruct \
#     --metrics-csv  $BASE/jailbreakbench/qwen2.5-0.5b-instruct/metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/jailbreakbench/qwen2.5-3b-instruct \
#     --metrics-csv  $BASE/jailbreakbench/qwen2.5-3b-instruct/metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/jailbreakbench/qwen2.5-7b-instruct \
#     --metrics-csv  $BASE/jailbreakbench/qwen2.5-7b-instruct/metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/jailbreakbench/qwen3-4b-saferl \
#     --metrics-csv  $BASE/jailbreakbench/qwen3-4b-saferl/metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/jailbreakbench/tulu2-7b-base \
#     --metrics-csv  $BASE/jailbreakbench/tulu2-7b-base/metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/jailbreakbench/tulu2-7b-sft \
#     --metrics-csv  $BASE/jailbreakbench/tulu2-7b-sft/metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/jailbreakbench/tulu2-7b-dpo \
#     --metrics-csv  $BASE/jailbreakbench/tulu2-7b-dpo/metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/jailbreakbench/tulu3-8b-base \
#     --metrics-csv  $BASE/jailbreakbench/tulu3-8b-base/metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/jailbreakbench/tulu3-8b-sft \
#     --metrics-csv  $BASE/jailbreakbench/tulu3-8b-sft/metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/jailbreakbench/tulu3-8b-dpo \
#     --metrics-csv  $BASE/jailbreakbench/tulu3-8b-dpo/metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/jailbreakbench/tulu3-8b-rlvr \
#     --metrics-csv  $BASE/jailbreakbench/tulu3-8b-rlvr/metrics.csv
