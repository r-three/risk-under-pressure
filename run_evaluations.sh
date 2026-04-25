#!/bin/bash
# Run Phase 2 evaluation for all experiments (metrics + metrics by category).
# Produces metrics.csv and metrics_by_category.csv for every model directory.
# Bootstrap CIs (1000 resamples) are computed here and stored in risk_lower/risk_upper.
# Seed-based CIs are computed at plot time — no extra evaluation step needed.
#
# Usage: bash run_evaluations.sh
# Requires: must be run from the project root.
# Note: runs directly on the login node (no GPU needed — pure post-processing).

set -e

source setup/start_env.sh

BASE=/home/ehghaghi/projects/aip-craffel/ehghaghi/prisk-pressure/results/multitrial_experiments/pressure_sensitivity
OUTPUT=/home/ehghaghi/scratch/aip-craffel/ehghaghi/prisk-pressure/results/multitrial_experiments/pressure_sensitivity

# --------------------------------------------------------------------------- #
# HarmBench
# --------------------------------------------------------------------------- #

python run_evaluation.py \
    --experiment configs/experiments/HB_qwen2.5_0.5b.yaml \
    --results-dir $BASE/harmbench/qwen2.5-0.5b-instruct \
    --format csv \
    --output $OUTPUT/harmbench/qwen2.5-0.5b-instruct/metrics.csv \
    --print-table \
    | tee $BASE/harmbench/qwen2.5-0.5b-instruct/summary.txt

python run_evaluation.py \
    --experiment configs/experiments/HB_qwen2.5_3b.yaml \
    --results-dir $BASE/harmbench/qwen2.5-3b-instruct \
    --format csv \
    --output $OUTPUT/harmbench/qwen2.5-3b-instruct/metrics.csv \
    --print-table \
    | tee $BASE/harmbench/qwen2.5-3b-instruct/summary.txt

python run_evaluation.py \
    --experiment configs/experiments/HB_qwen2.5_7b.yaml \
    --results-dir $BASE/harmbench/qwen2.5-7b-instruct \
    --format csv \
    --output $OUTPUT/harmbench/qwen2.5-7b-instruct/metrics.csv \
    --print-table \
    | tee $BASE/harmbench/qwen2.5-7b-instruct/summary.txt

python run_evaluation.py \
    --experiment configs/experiments/HB_qwen3_4b_saferl.yaml \
    --results-dir $BASE/harmbench/qwen3-4b-saferl \
    --format csv \
    --output $OUTPUT/harmbench/qwen3-4b-saferl/metrics.csv \
    --print-table \
    | tee $BASE/harmbench/qwen3-4b-saferl/summary.txt

python run_evaluation.py \
    --experiment configs/experiments/HB_tulu2_7b_base.yaml \
    --results-dir $BASE/harmbench/tulu2-7b-base \
    --format csv \
    --output $OUTPUT/harmbench/tulu2-7b-base/metrics.csv \
    --print-table \
    | tee $BASE/harmbench/tulu2-7b-base/summary.txt

python run_evaluation.py \
    --experiment configs/experiments/HB_tulu2_7b_sft.yaml \
    --results-dir $BASE/harmbench/tulu2-7b-sft \
    --format csv \
    --output $OUTPUT/harmbench/tulu2-7b-sft/metrics.csv \
    --print-table \
    | tee $BASE/harmbench/tulu2-7b-sft/summary.txt

python run_evaluation.py \
    --experiment configs/experiments/HB_tulu2_7b_dpo.yaml \
    --results-dir $BASE/harmbench/tulu2-7b-dpo \
    --format csv \
    --output $OUTPUT/harmbench/tulu2-7b-dpo/metrics.csv \
    --print-table \
    | tee $BASE/harmbench/tulu2-7b-dpo/summary.txt

python run_evaluation.py \
    --experiment configs/experiments/HB_tulu3_8b_base.yaml \
    --results-dir $BASE/harmbench/tulu3-8b-base \
    --format csv \
    --output $OUTPUT/harmbench/tulu3-8b-base/metrics.csv \
    --print-table \
    | tee $BASE/harmbench/tulu3-8b-base/summary.txt

python run_evaluation.py \
    --experiment configs/experiments/HB_tulu3_8b_sft.yaml \
    --results-dir $BASE/harmbench/tulu3-8b-sft \
    --format csv \
    --output $OUTPUT/harmbench/tulu3-8b-sft/metrics.csv \
    --print-table \
    | tee $BASE/harmbench/tulu3-8b-sft/summary.txt

python run_evaluation.py \
    --experiment configs/experiments/HB_tulu3_8b_dpo.yaml \
    --results-dir $BASE/harmbench/tulu3-8b-dpo \
    --format csv \
    --output $OUTPUT/harmbench/tulu3-8b-dpo/metrics.csv \
    --print-table \
    | tee $BASE/harmbench/tulu3-8b-dpo/summary.txt

python run_evaluation.py \
    --experiment configs/experiments/HB_tulu3_8b_rlvr.yaml \
    --results-dir $BASE/harmbench/tulu3-8b-rlvr \
    --format csv \
    --output $OUTPUT/harmbench/tulu3-8b-rlvr/metrics.csv \
    --print-table \
    | tee $BASE/harmbench/tulu3-8b-rlvr/summary.txt

# --------------------------------------------------------------------------- #
# JailbreakBench
# --------------------------------------------------------------------------- #

python run_evaluation.py \
    --experiment configs/experiments/JB_qwen2.5_0.5b.yaml \
    --results-dir $BASE/jailbreakbench/qwen2.5-0.5b-instruct \
    --format csv \
    --output $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/metrics.csv \
    --print-table \
    | tee $BASE/jailbreakbench/qwen2.5-0.5b-instruct/summary.txt

python run_evaluation.py \
    --experiment configs/experiments/JB_qwen2.5_3b.yaml \
    --results-dir $BASE/jailbreakbench/qwen2.5-3b-instruct \
    --format csv \
    --output $OUTPUT/jailbreakbench/qwen2.5-3b-instruct/metrics.csv \
    --print-table \
    | tee $BASE/jailbreakbench/qwen2.5-3b-instruct/summary.txt

python run_evaluation.py \
    --experiment configs/experiments/JB_qwen2.5_7b.yaml \
    --results-dir $BASE/jailbreakbench/qwen2.5-7b-instruct \
    --format csv \
    --output $OUTPUT/jailbreakbench/qwen2.5-7b-instruct/metrics.csv \
    --print-table \
    | tee $BASE/jailbreakbench/qwen2.5-7b-instruct/summary.txt

python run_evaluation.py \
    --experiment configs/experiments/JB_qwen3_4b_saferl.yaml \
    --results-dir $BASE/jailbreakbench/qwen3-4b-saferl \
    --format csv \
    --output $OUTPUT/jailbreakbench/qwen3-4b-saferl/metrics.csv \
    --print-table \
    | tee $BASE/jailbreakbench/qwen3-4b-saferl/summary.txt

python run_evaluation.py \
    --experiment configs/experiments/JB_tulu2_7b_base.yaml \
    --results-dir $BASE/jailbreakbench/tulu2-7b-base \
    --format csv \
    --output $OUTPUT/jailbreakbench/tulu2-7b-base/metrics.csv \
    --print-table \
    | tee $BASE/jailbreakbench/tulu2-7b-base/summary.txt

python run_evaluation.py \
    --experiment configs/experiments/JB_tulu2_7b_sft.yaml \
    --results-dir $BASE/jailbreakbench/tulu2-7b-sft \
    --format csv \
    --output $OUTPUT/jailbreakbench/tulu2-7b-sft/metrics.csv \
    --print-table \
    | tee $BASE/jailbreakbench/tulu2-7b-sft/summary.txt

python run_evaluation.py \
    --experiment configs/experiments/JB_tulu2_7b_dpo.yaml \
    --results-dir $BASE/jailbreakbench/tulu2-7b-dpo \
    --format csv \
    --output $OUTPUT/jailbreakbench/tulu2-7b-dpo/metrics.csv \
    --print-table \
    | tee $BASE/jailbreakbench/tulu2-7b-dpo/summary.txt

python run_evaluation.py \
    --experiment configs/experiments/JB_tulu3_8b_base.yaml \
    --results-dir $BASE/jailbreakbench/tulu3-8b-base \
    --format csv \
    --output $OUTPUT/jailbreakbench/tulu3-8b-base/metrics.csv \
    --print-table \
    | tee $BASE/jailbreakbench/tulu3-8b-base/summary.txt

python run_evaluation.py \
    --experiment configs/experiments/JB_tulu3_8b_sft.yaml \
    --results-dir $BASE/jailbreakbench/tulu3-8b-sft \
    --format csv \
    --output $OUTPUT/jailbreakbench/tulu3-8b-sft/metrics.csv \
    --print-table \
    | tee $BASE/jailbreakbench/tulu3-8b-sft/summary.txt

python run_evaluation.py \
    --experiment configs/experiments/JB_tulu3_8b_dpo.yaml \
    --results-dir $BASE/jailbreakbench/tulu3-8b-dpo \
    --format csv \
    --output $OUTPUT/jailbreakbench/tulu3-8b-dpo/metrics.csv \
    --print-table \
    | tee $BASE/jailbreakbench/tulu3-8b-dpo/summary.txt

python run_evaluation.py \
    --experiment configs/experiments/JB_tulu3_8b_rlvr.yaml \
    --results-dir $BASE/jailbreakbench/tulu3-8b-rlvr \
    --format csv \
    --output $OUTPUT/jailbreakbench/tulu3-8b-rlvr/metrics.csv \
    --print-table \
    | tee $BASE/jailbreakbench/tulu3-8b-rlvr/summary.txt
