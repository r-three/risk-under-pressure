#!/bin/bash
# Run Phase 2 evaluation for all experiments (metrics + metrics by category).
# Reads from $SCRATCH/rup, which is where run_HB_experiments.sh / run_JB_experiments.sh
# write their inference output. Produces metrics.csv and summary.txt per model directory.
#
# Usage: bash run_evaluations.sh
# Requires: must be run from the project root.
# Note: runs directly on the login node (no GPU needed — pure post-processing).

set -e

source setup/start_env.sh

BASE=$SCRATCH/rup
OUTPUT=$SCRATCH/rup/plots

EVAL="python run_evaluation.py --experiment configs/experiments/base.yaml --format csv --print-table"

# =============================================================================
# HarmBench
# =============================================================================

# --- MODEL SIZE STUDY — Qwen2.5-Instruct: 0.5B, 3B, 7B ---
# Paper: Figure 1 right

# $EVAL \
#     --results-dir $BASE/harmbench/qwen2.5-0.5b-instruct \
#     --output $OUTPUT/harmbench/qwen2.5-0.5b-instruct/metrics.csv \
#     | tee $BASE/harmbench/qwen2.5-0.5b-instruct/summary.txt

# $EVAL \
#     --results-dir $BASE/harmbench/qwen2.5-3b-instruct \
#     --output $OUTPUT/harmbench/qwen2.5-3b-instruct/metrics.csv \
#     | tee $BASE/harmbench/qwen2.5-3b-instruct/summary.txt

# $EVAL \
#     --results-dir $BASE/harmbench/qwen2.5-7b-instruct \
#     --output $OUTPUT/harmbench/qwen2.5-7b-instruct/metrics.csv \
#     | tee $BASE/harmbench/qwen2.5-7b-instruct/summary.txt

# --- TRAINING STAGE STUDY — Tulu3 8B: Base → SFT → DPO → RLVR ---
# Paper: Table 1, Figure 1 left

# $EVAL \
#     --results-dir $BASE/harmbench/tulu3-8b-base \
#     --output $OUTPUT/harmbench/tulu3-8b-base/metrics.csv \
#     | tee $BASE/harmbench/tulu3-8b-base/summary.txt

# $EVAL \
#     --results-dir $BASE/harmbench/tulu3-8b-sft \
#     --output $OUTPUT/harmbench/tulu3-8b-sft/metrics.csv \
#     | tee $BASE/harmbench/tulu3-8b-sft/summary.txt

# $EVAL \
#     --results-dir $BASE/harmbench/tulu3-8b-dpo \
#     --output $OUTPUT/harmbench/tulu3-8b-dpo/metrics.csv \
#     | tee $BASE/harmbench/tulu3-8b-dpo/summary.txt

# $EVAL \
#     --results-dir $BASE/harmbench/tulu3-8b-rlvr \
#     --output $OUTPUT/harmbench/tulu3-8b-rlvr/metrics.csv \
#     | tee $BASE/harmbench/tulu3-8b-rlvr/summary.txt

# --- SAFETY ALIGNMENT STUDY — Qwen3-4B base vs Qwen3-4B-SafeRL ---
# Paper: Table 1 (Qwen3 rows)

$EVAL \
    --results-dir $BASE/harmbench/qwen3-4b \
    --output $OUTPUT/harmbench/qwen3-4b/metrics.csv \
    | tee $BASE/harmbench/qwen3-4b/summary.txt

# $EVAL \
#     --results-dir $BASE/harmbench/qwen3-4b-saferl \
#     --output $OUTPUT/harmbench/qwen3-4b-saferl/metrics.csv \
#     | tee $BASE/harmbench/qwen3-4b-saferl/summary.txt

# =============================================================================
# JailbreakBench
# =============================================================================

# --- MODEL SIZE STUDY — Qwen2.5-Instruct: 0.5B, 3B, 7B ---
# Paper: Figure 1 right

# $EVAL \
#     --results-dir $BASE/jailbreakbench/qwen2.5-0.5b-instruct \
#     --output $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/metrics.csv \
#     | tee $BASE/jailbreakbench/qwen2.5-0.5b-instruct/summary.txt

# $EVAL \
#     --results-dir $BASE/jailbreakbench/qwen2.5-3b-instruct \
#     --output $OUTPUT/jailbreakbench/qwen2.5-3b-instruct/metrics.csv \
#     | tee $BASE/jailbreakbench/qwen2.5-3b-instruct/summary.txt

# $EVAL \
#     --results-dir $BASE/jailbreakbench/qwen2.5-7b-instruct \
#     --output $OUTPUT/jailbreakbench/qwen2.5-7b-instruct/metrics.csv \
#     | tee $BASE/jailbreakbench/qwen2.5-7b-instruct/summary.txt

# --- TRAINING STAGE STUDY — Tulu3 8B: Base → SFT → DPO → RLVR ---
# Paper: Table 1, Figure 1 left

# $EVAL \
#     --results-dir $BASE/jailbreakbench/tulu3-8b-base \
#     --output $OUTPUT/jailbreakbench/tulu3-8b-base/metrics.csv \
#     | tee $BASE/jailbreakbench/tulu3-8b-base/summary.txt

# $EVAL \
#     --results-dir $BASE/jailbreakbench/tulu3-8b-sft \
#     --output $OUTPUT/jailbreakbench/tulu3-8b-sft/metrics.csv \
#     | tee $BASE/jailbreakbench/tulu3-8b-sft/summary.txt

# $EVAL \
#     --results-dir $BASE/jailbreakbench/tulu3-8b-dpo \
#     --output $OUTPUT/jailbreakbench/tulu3-8b-dpo/metrics.csv \
#     | tee $BASE/jailbreakbench/tulu3-8b-dpo/summary.txt

# $EVAL \
#     --results-dir $BASE/jailbreakbench/tulu3-8b-rlvr \
#     --output $OUTPUT/jailbreakbench/tulu3-8b-rlvr/metrics.csv \
#     | tee $BASE/jailbreakbench/tulu3-8b-rlvr/summary.txt

# --- SAFETY ALIGNMENT STUDY — Qwen3-4B base vs Qwen3-4B-SafeRL ---

$EVAL \
    --results-dir $BASE/jailbreakbench/qwen3-4b \
    --output $OUTPUT/jailbreakbench/qwen3-4b/metrics.csv \
    | tee $BASE/jailbreakbench/qwen3-4b/summary.txt

# $EVAL \
#     --results-dir $BASE/jailbreakbench/qwen3-4b-saferl \
#     --output $OUTPUT/jailbreakbench/qwen3-4b-saferl/metrics.csv \
#     | tee $BASE/jailbreakbench/qwen3-4b-saferl/summary.txt
