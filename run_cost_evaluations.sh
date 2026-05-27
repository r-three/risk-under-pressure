#!/bin/bash
# Run Phase 2.5 cost evaluation for all experiments.
# Augments each metrics.csv with exact token and FLOP cost columns → cost_metrics.csv.
# No re-running of experiments needed — costs are derived post-hoc from JSONL records.
# Token counts use each model's exact HuggingFace tokenizer (CPU-only, no GPU needed).
# All three token-consuming components are counted: target model, judge, and PAIR attacker.
#
# Usage: bash run_cost_evaluations.sh
# Requires: must be run from the project root.
# Note: runs directly on the login node (no GPU needed — pure post-processing).

set -e

source setup/start_env.sh

BASE=/home/ehghaghi/projects/aip-craffel/ehghaghi/prisk-pressure/results/multitrial_experiments/pressure_sensitivity
PLOTS_BASE=/home/ehghaghi/projects/aip-craffel/ehghaghi/prisk-pressure/results/plots/multitrial_experiments/pressure_sensitivity

# --------------------------------------------------------------------------- #
# HarmBench
# --------------------------------------------------------------------------- #

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/harmbench/qwen2.5-0.5b-instruct \
#     --metrics-csv  $PLOTS_BASE/harmbench/qwen2.5-0.5b-instruct/metrics.csv \
#     --output       $PLOTS_BASE/harmbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/harmbench/qwen2.5-3b-instruct \
#     --metrics-csv  $PLOTS_BASE/harmbench/qwen2.5-3b-instruct/metrics.csv \
#     --output       $PLOTS_BASE/harmbench/qwen2.5-3b-instruct/cost/cost_metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/harmbench/qwen2.5-7b-instruct \
#     --metrics-csv  $PLOTS_BASE/harmbench/qwen2.5-7b-instruct/metrics.csv \
#     --output       $PLOTS_BASE/harmbench/qwen2.5-7b-instruct/cost/cost_metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/harmbench/qwen3-4b-saferl \
#     --metrics-csv  $PLOTS_BASE/harmbench/qwen3-4b-saferl/metrics.csv \
#     --output       $PLOTS_BASE/harmbench/qwen3-4b-saferl/cost/cost_metrics.csv

python scripts/compute_attack_costs.py \
    --results-dir $BASE/harmbench/qwen3-8b \
    --metrics-csv  $PLOTS_BASE/harmbench/qwen3-8b/metrics.csv \
    --output       $PLOTS_BASE/harmbench/qwen3-8b/cost/cost_metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/harmbench/tulu2-7b-base \
#     --metrics-csv  $PLOTS_BASE/harmbench/tulu2-7b-base/metrics.csv \
#     --output       $PLOTS_BASE/harmbench/tulu2-7b-base/cost/cost_metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/harmbench/tulu2-7b-sft \
#     --metrics-csv  $PLOTS_BASE/harmbench/tulu2-7b-sft/metrics.csv \
#     --output       $PLOTS_BASE/harmbench/tulu2-7b-sft/cost/cost_metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/harmbench/tulu2-7b-dpo \
#     --metrics-csv  $PLOTS_BASE/harmbench/tulu2-7b-dpo/metrics.csv \
#     --output       $PLOTS_BASE/harmbench/tulu2-7b-dpo/cost/cost_metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/harmbench/tulu3-8b-base \
#     --metrics-csv  $PLOTS_BASE/harmbench/tulu3-8b-base/metrics.csv \
#     --output       $PLOTS_BASE/harmbench/tulu3-8b-base/cost/cost_metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/harmbench/tulu3-8b-sft \
#     --metrics-csv  $PLOTS_BASE/harmbench/tulu3-8b-sft/metrics.csv \
#     --output       $PLOTS_BASE/harmbench/tulu3-8b-sft/cost/cost_metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/harmbench/tulu3-8b-dpo \
#     --metrics-csv  $PLOTS_BASE/harmbench/tulu3-8b-dpo/metrics.csv \
#     --output       $PLOTS_BASE/harmbench/tulu3-8b-dpo/cost/cost_metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/harmbench/tulu3-8b-rlvr \
#     --metrics-csv  $PLOTS_BASE/harmbench/tulu3-8b-rlvr/metrics.csv \
#     --output       $PLOTS_BASE/harmbench/tulu3-8b-rlvr/cost/cost_metrics.csv

# --------------------------------------------------------------------------- #
# JailbreakBench
# --------------------------------------------------------------------------- #

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/jailbreakbench/qwen2.5-0.5b-instruct \
#     --metrics-csv  $PLOTS_BASE/jailbreakbench/qwen2.5-0.5b-instruct/metrics.csv \
#     --output       $PLOTS_BASE/jailbreakbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/jailbreakbench/qwen2.5-3b-instruct \
#     --metrics-csv  $PLOTS_BASE/jailbreakbench/qwen2.5-3b-instruct/metrics.csv \
#     --output       $PLOTS_BASE/jailbreakbench/qwen2.5-3b-instruct/cost/cost_metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/jailbreakbench/qwen2.5-7b-instruct \
#     --metrics-csv  $PLOTS_BASE/jailbreakbench/qwen2.5-7b-instruct/metrics.csv \
#     --output       $PLOTS_BASE/jailbreakbench/qwen2.5-7b-instruct/cost/cost_metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/jailbreakbench/qwen3-4b-saferl \
#     --metrics-csv  $PLOTS_BASE/jailbreakbench/qwen3-4b-saferl/metrics.csv \
#     --output       $PLOTS_BASE/jailbreakbench/qwen3-4b-saferl/cost/cost_metrics.csv

python scripts/compute_attack_costs.py \
    --results-dir $BASE/jailbreakbench/qwen3-8b \
    --metrics-csv  $PLOTS_BASE/jailbreakbench/qwen3-8b/metrics.csv \
    --output       $PLOTS_BASE/jailbreakbench/qwen3-8b/cost/cost_metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/jailbreakbench/tulu2-7b-base \
#     --metrics-csv  $PLOTS_BASE/jailbreakbench/tulu2-7b-base/metrics.csv \
#     --output       $PLOTS_BASE/jailbreakbench/tulu2-7b-base/cost/cost_metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/jailbreakbench/tulu2-7b-sft \
#     --metrics-csv  $PLOTS_BASE/jailbreakbench/tulu2-7b-sft/metrics.csv \
#     --output       $PLOTS_BASE/jailbreakbench/tulu2-7b-sft/cost/cost_metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/jailbreakbench/tulu2-7b-dpo \
#     --metrics-csv  $PLOTS_BASE/jailbreakbench/tulu2-7b-dpo/metrics.csv \
#     --output       $PLOTS_BASE/jailbreakbench/tulu2-7b-dpo/cost/cost_metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/jailbreakbench/tulu3-8b-base \
#     --metrics-csv  $PLOTS_BASE/jailbreakbench/tulu3-8b-base/metrics.csv \
#     --output       $PLOTS_BASE/jailbreakbench/tulu3-8b-base/cost/cost_metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/jailbreakbench/tulu3-8b-sft \
#     --metrics-csv  $PLOTS_BASE/jailbreakbench/tulu3-8b-sft/metrics.csv \
#     --output       $PLOTS_BASE/jailbreakbench/tulu3-8b-sft/cost/cost_metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/jailbreakbench/tulu3-8b-dpo \
#     --metrics-csv  $PLOTS_BASE/jailbreakbench/tulu3-8b-dpo/metrics.csv \
#     --output       $PLOTS_BASE/jailbreakbench/tulu3-8b-dpo/cost/cost_metrics.csv

# python scripts/compute_attack_costs.py \
#     --results-dir $BASE/jailbreakbench/tulu3-8b-rlvr \
#     --metrics-csv  $PLOTS_BASE/jailbreakbench/tulu3-8b-rlvr/metrics.csv \
#     --output       $PLOTS_BASE/jailbreakbench/tulu3-8b-rlvr/cost/cost_metrics.csv
