#!/bin/bash
# Run Phase 2 evaluation for all experiments (metrics + metrics by category).
# Reads from $SCRATCH/rup, which is where run_HB_experiments.sh / run_JB_experiments.sh
# write their inference output. Produces metrics.csv and summary.txt per model directory.
#
# Usage: bash run_evaluations.sh
# Requires: must be run from the project root.
# Note: runs directly on the login node (no GPU needed — pure post-processing).
#
# JUDGE selects the safety judge (a config name under configs/models/, without .yaml).
# It defaults to llama3.1_8b_instruct_judge, which keeps every path and job name
# exactly as it was before judges became selectable. Any other judge writes to its
# own tree under $SCRATCH/rup/judges/<judge_model_id>/ so runs never overwrite:
#   JUDGE=olmo3_7b_instruct_judge     bash run_evaluations.sh
#   JUDGE=gemma3_4b_it_judge          bash run_evaluations.sh
# See run_judge_ablation.sh to sweep all judges in one go.

set -e

source setup/start_env.sh
source setup/judge_env.sh

BASE=$RUN_ROOT
OUTPUT=$PLOT_ROOT

EVAL="python scripts/run_evaluation.py --experiment configs/experiments/base.yaml --format csv --print-table"

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

# --- MODEL SIZE STUDY (2nd family) — Gemma 3 Instruction-Tuned: 270M, 1B, 4B ---
# Companion to the Qwen2.5 ladder above (configs/experiments/paper/model_size_gemma3.yaml)

$EVAL \
    --results-dir $BASE/harmbench/gemma3-270m-it \
    --output $OUTPUT/harmbench/gemma3-270m-it/metrics.csv \
    | tee $BASE/harmbench/gemma3-270m-it/summary.txt

$EVAL \
    --results-dir $BASE/harmbench/gemma3-1b-it \
    --output $OUTPUT/harmbench/gemma3-1b-it/metrics.csv \
    | tee $BASE/harmbench/gemma3-1b-it/summary.txt

$EVAL \
    --results-dir $BASE/harmbench/gemma3-4b-it \
    --output $OUTPUT/harmbench/gemma3-4b-it/metrics.csv \
    | tee $BASE/harmbench/gemma3-4b-it/summary.txt

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

# --- TRAINING STAGE STUDY (2nd ladder) — OLMo 2 1B: Base → SFT → DPO → RLVR1 → Instruct ---
# Companion to the Tulu3 8B ladder above; `-Instruct` is a second RLVR round on `-RLVR1`

# $EVAL \
#     --results-dir $BASE/harmbench/olmo2-1b-base \
#     --output $OUTPUT/harmbench/olmo2-1b-base/metrics.csv \
#     | tee $BASE/harmbench/olmo2-1b-base/summary.txt

# $EVAL \
#     --results-dir $BASE/harmbench/olmo2-1b-sft \
#     --output $OUTPUT/harmbench/olmo2-1b-sft/metrics.csv \
#     | tee $BASE/harmbench/olmo2-1b-sft/summary.txt

# $EVAL \
#     --results-dir $BASE/harmbench/olmo2-1b-dpo \
#     --output $OUTPUT/harmbench/olmo2-1b-dpo/metrics.csv \
#     | tee $BASE/harmbench/olmo2-1b-dpo/summary.txt

# $EVAL \
#     --results-dir $BASE/harmbench/olmo2-1b-rlvr1 \
#     --output $OUTPUT/harmbench/olmo2-1b-rlvr1/metrics.csv \
#     | tee $BASE/harmbench/olmo2-1b-rlvr1/summary.txt

# $EVAL \
#     --results-dir $BASE/harmbench/olmo2-1b-instruct \
#     --output $OUTPUT/harmbench/olmo2-1b-instruct/metrics.csv \
#     | tee $BASE/harmbench/olmo2-1b-instruct/summary.txt

# --- SAFETY ALIGNMENT STUDY — Qwen3-4B base vs Qwen3-4B-SafeRL ---
# Paper: Table 1 (Qwen3 rows)

# $EVAL \
#     --results-dir $BASE/harmbench/qwen3-4b \
#     --output $OUTPUT/harmbench/qwen3-4b/metrics.csv \
#     | tee $BASE/harmbench/qwen3-4b/summary.txt

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

# --- MODEL SIZE STUDY (2nd family) — Gemma 3 Instruction-Tuned: 270M, 1B, 4B ---
# Companion to the Qwen2.5 ladder above (configs/experiments/paper/model_size_gemma3.yaml)

$EVAL \
    --results-dir $BASE/jailbreakbench/gemma3-270m-it \
    --output $OUTPUT/jailbreakbench/gemma3-270m-it/metrics.csv \
    | tee $BASE/jailbreakbench/gemma3-270m-it/summary.txt

$EVAL \
    --results-dir $BASE/jailbreakbench/gemma3-1b-it \
    --output $OUTPUT/jailbreakbench/gemma3-1b-it/metrics.csv \
    | tee $BASE/jailbreakbench/gemma3-1b-it/summary.txt

$EVAL \
    --results-dir $BASE/jailbreakbench/gemma3-4b-it \
    --output $OUTPUT/jailbreakbench/gemma3-4b-it/metrics.csv \
    | tee $BASE/jailbreakbench/gemma3-4b-it/summary.txt

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

# --- TRAINING STAGE STUDY (2nd ladder) — OLMo 2 1B: Base → SFT → DPO → RLVR1 → Instruct ---
# Companion to the Tulu3 8B ladder above; `-Instruct` is a second RLVR round on `-RLVR1`

# $EVAL \
#     --results-dir $BASE/jailbreakbench/olmo2-1b-base \
#     --output $OUTPUT/jailbreakbench/olmo2-1b-base/metrics.csv \
#     | tee $BASE/jailbreakbench/olmo2-1b-base/summary.txt

# $EVAL \
#     --results-dir $BASE/jailbreakbench/olmo2-1b-sft \
#     --output $OUTPUT/jailbreakbench/olmo2-1b-sft/metrics.csv \
#     | tee $BASE/jailbreakbench/olmo2-1b-sft/summary.txt

# $EVAL \
#     --results-dir $BASE/jailbreakbench/olmo2-1b-dpo \
#     --output $OUTPUT/jailbreakbench/olmo2-1b-dpo/metrics.csv \
#     | tee $BASE/jailbreakbench/olmo2-1b-dpo/summary.txt

# $EVAL \
#     --results-dir $BASE/jailbreakbench/olmo2-1b-rlvr1 \
#     --output $OUTPUT/jailbreakbench/olmo2-1b-rlvr1/metrics.csv \
#     | tee $BASE/jailbreakbench/olmo2-1b-rlvr1/summary.txt

# $EVAL \
#     --results-dir $BASE/jailbreakbench/olmo2-1b-instruct \
#     --output $OUTPUT/jailbreakbench/olmo2-1b-instruct/metrics.csv \
#     | tee $BASE/jailbreakbench/olmo2-1b-instruct/summary.txt

# --- SAFETY ALIGNMENT STUDY — Qwen3-4B base vs Qwen3-4B-SafeRL ---

# $EVAL \
#     --results-dir $BASE/jailbreakbench/qwen3-4b \
#     --output $OUTPUT/jailbreakbench/qwen3-4b/metrics.csv \
#     | tee $BASE/jailbreakbench/qwen3-4b/summary.txt

# $EVAL \
#     --results-dir $BASE/jailbreakbench/qwen3-4b-saferl \
#     --output $OUTPUT/jailbreakbench/qwen3-4b-saferl/metrics.csv \
#     | tee $BASE/jailbreakbench/qwen3-4b-saferl/summary.txt
