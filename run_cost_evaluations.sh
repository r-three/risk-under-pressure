#!/bin/bash
# Run Phase 2.5 cost evaluation for all experiments.
# Augments each metrics.csv with exact token and FLOP cost columns → cost_metrics.csv.
# No re-running of experiments needed — costs are derived post-hoc from JSONL records.
# Token counts use each model's exact HuggingFace tokenizer (CPU-only, no GPU needed).
# All three token-consuming components are counted: target model, judge, and PAIR attacker.
#
# Reads inference results from $SCRATCH/rup (written by run_HB/JB_experiments.sh).
# Reads metrics.csv from $SCRATCH/rup/plots (written by run_evaluations.sh).
#
# Usage: bash run_cost_evaluations.sh
# Requires: must be run from the project root.
# Note: runs directly on the login node (no GPU needed — pure post-processing).
#
# JUDGE selects the safety judge (a config name under configs/models/, without .yaml).
# It defaults to llama3.1_8b_instruct_judge, which keeps every path and job name
# exactly as it was before judges became selectable. Any other judge writes to its
# own tree under $SCRATCH/rup/judges/<judge_model_id>/ so runs never overwrite:
#   JUDGE=olmo3_7b_instruct_judge     bash run_cost_evaluations.sh
#   JUDGE=gemma3_4b_it_judge          bash run_cost_evaluations.sh
# See run_judge_ablation.sh to sweep all judges in one go.

set -e

source setup/start_env.sh
source setup/judge_env.sh

BASE=$RUN_ROOT
OUTPUT=$PLOT_ROOT

# --pricing-config populates the dollar-cost columns (mean_{target,judge,attacker,total}_dollars).
# Without it those columns stay NaN and `--x-axis dollars` plots nothing.
COST="python scripts/compute_attack_costs.py --pricing-config configs/pricing.yaml --judge-model $JUDGE_ID"

# =============================================================================
# HarmBench
# =============================================================================

# --- MODEL SIZE STUDY — Qwen2.5-Instruct: 0.5B, 3B, 7B ---
# Paper: Figure 1 right

# $COST \
#     --results-dir $BASE/harmbench/qwen2.5-0.5b-instruct \
#     --metrics-csv $OUTPUT/harmbench/qwen2.5-0.5b-instruct/metrics.csv \
#     --output      $OUTPUT/harmbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv

# $COST \
#     --results-dir $BASE/harmbench/qwen2.5-3b-instruct \
#     --metrics-csv $OUTPUT/harmbench/qwen2.5-3b-instruct/metrics.csv \
#     --output      $OUTPUT/harmbench/qwen2.5-3b-instruct/cost/cost_metrics.csv

# $COST \
#     --results-dir $BASE/harmbench/qwen2.5-7b-instruct \
#     --metrics-csv $OUTPUT/harmbench/qwen2.5-7b-instruct/metrics.csv \
#     --output      $OUTPUT/harmbench/qwen2.5-7b-instruct/cost/cost_metrics.csv

# --- MODEL SIZE STUDY (2nd family) — Gemma 3 Instruction-Tuned: 270M, 1B, 4B ---
# Companion to the Qwen2.5 ladder above (configs/experiments/paper/model_size_gemma3.yaml)

$COST \
    --results-dir $BASE/harmbench/gemma3-270m-it \
    --metrics-csv $OUTPUT/harmbench/gemma3-270m-it/metrics.csv \
    --output      $OUTPUT/harmbench/gemma3-270m-it/cost/cost_metrics.csv

$COST \
    --results-dir $BASE/harmbench/gemma3-1b-it \
    --metrics-csv $OUTPUT/harmbench/gemma3-1b-it/metrics.csv \
    --output      $OUTPUT/harmbench/gemma3-1b-it/cost/cost_metrics.csv

$COST \
    --results-dir $BASE/harmbench/gemma3-4b-it \
    --metrics-csv $OUTPUT/harmbench/gemma3-4b-it/metrics.csv \
    --output      $OUTPUT/harmbench/gemma3-4b-it/cost/cost_metrics.csv

# --- TRAINING STAGE STUDY — Tulu3 8B: Base → SFT → DPO → RLVR ---
# Paper: Table 1, Figure 1 left

# $COST \
#     --results-dir $BASE/harmbench/tulu3-8b-base \
#     --metrics-csv $OUTPUT/harmbench/tulu3-8b-base/metrics.csv \
#     --output      $OUTPUT/harmbench/tulu3-8b-base/cost/cost_metrics.csv

# $COST \
#     --results-dir $BASE/harmbench/tulu3-8b-sft \
#     --metrics-csv $OUTPUT/harmbench/tulu3-8b-sft/metrics.csv \
#     --output      $OUTPUT/harmbench/tulu3-8b-sft/cost/cost_metrics.csv

# $COST \
#     --results-dir $BASE/harmbench/tulu3-8b-dpo \
#     --metrics-csv $OUTPUT/harmbench/tulu3-8b-dpo/metrics.csv \
#     --output      $OUTPUT/harmbench/tulu3-8b-dpo/cost/cost_metrics.csv

# $COST \
#     --results-dir $BASE/harmbench/tulu3-8b-rlvr \
#     --metrics-csv $OUTPUT/harmbench/tulu3-8b-rlvr/metrics.csv \
#     --output      $OUTPUT/harmbench/tulu3-8b-rlvr/cost/cost_metrics.csv

# --- TRAINING STAGE STUDY (2nd ladder) — OLMo 2 1B: Base → SFT → DPO → RLVR1 → Instruct ---
# Companion to the Tulu3 8B ladder above; `-Instruct` is a second RLVR round on `-RLVR1`

# $COST \
#     --results-dir $BASE/harmbench/olmo2-1b-base \
#     --metrics-csv $OUTPUT/harmbench/olmo2-1b-base/metrics.csv \
#     --output      $OUTPUT/harmbench/olmo2-1b-base/cost/cost_metrics.csv

# $COST \
#     --results-dir $BASE/harmbench/olmo2-1b-sft \
#     --metrics-csv $OUTPUT/harmbench/olmo2-1b-sft/metrics.csv \
#     --output      $OUTPUT/harmbench/olmo2-1b-sft/cost/cost_metrics.csv

# $COST \
#     --results-dir $BASE/harmbench/olmo2-1b-dpo \
#     --metrics-csv $OUTPUT/harmbench/olmo2-1b-dpo/metrics.csv \
#     --output      $OUTPUT/harmbench/olmo2-1b-dpo/cost/cost_metrics.csv

# $COST \
#     --results-dir $BASE/harmbench/olmo2-1b-rlvr1 \
#     --metrics-csv $OUTPUT/harmbench/olmo2-1b-rlvr1/metrics.csv \
#     --output      $OUTPUT/harmbench/olmo2-1b-rlvr1/cost/cost_metrics.csv

# $COST \
#     --results-dir $BASE/harmbench/olmo2-1b-instruct \
#     --metrics-csv $OUTPUT/harmbench/olmo2-1b-instruct/metrics.csv \
#     --output      $OUTPUT/harmbench/olmo2-1b-instruct/cost/cost_metrics.csv

# --- SAFETY ALIGNMENT STUDY — Qwen3-4B base vs Qwen3-4B-SafeRL ---
# Paper: Table 1 (Qwen3 rows)

# $COST \
#     --results-dir $BASE/harmbench/qwen3-4b \
#     --metrics-csv $OUTPUT/harmbench/qwen3-4b/metrics.csv \
#     --output      $OUTPUT/harmbench/qwen3-4b/cost/cost_metrics.csv

# $COST \
#     --results-dir $BASE/harmbench/qwen3-4b-saferl \
#     --metrics-csv $OUTPUT/harmbench/qwen3-4b-saferl/metrics.csv \
#     --output      $OUTPUT/harmbench/qwen3-4b-saferl/cost/cost_metrics.csv

# =============================================================================
# JailbreakBench
# =============================================================================

# --- MODEL SIZE STUDY — Qwen2.5-Instruct: 0.5B, 3B, 7B ---
# Paper: Figure 1 right

# $COST \
#     --results-dir $BASE/jailbreakbench/qwen2.5-0.5b-instruct \
#     --metrics-csv $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/metrics.csv \
#     --output      $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv

# $COST \
#     --results-dir $BASE/jailbreakbench/qwen2.5-3b-instruct \
#     --metrics-csv $OUTPUT/jailbreakbench/qwen2.5-3b-instruct/metrics.csv \
#     --output      $OUTPUT/jailbreakbench/qwen2.5-3b-instruct/cost/cost_metrics.csv

# $COST \
#     --results-dir $BASE/jailbreakbench/qwen2.5-7b-instruct \
#     --metrics-csv $OUTPUT/jailbreakbench/qwen2.5-7b-instruct/metrics.csv \
#     --output      $OUTPUT/jailbreakbench/qwen2.5-7b-instruct/cost/cost_metrics.csv

# --- MODEL SIZE STUDY (2nd family) — Gemma 3 Instruction-Tuned: 270M, 1B, 4B ---
# Companion to the Qwen2.5 ladder above (configs/experiments/paper/model_size_gemma3.yaml)

$COST \
    --results-dir $BASE/jailbreakbench/gemma3-270m-it \
    --metrics-csv $OUTPUT/jailbreakbench/gemma3-270m-it/metrics.csv \
    --output      $OUTPUT/jailbreakbench/gemma3-270m-it/cost/cost_metrics.csv

$COST \
    --results-dir $BASE/jailbreakbench/gemma3-1b-it \
    --metrics-csv $OUTPUT/jailbreakbench/gemma3-1b-it/metrics.csv \
    --output      $OUTPUT/jailbreakbench/gemma3-1b-it/cost/cost_metrics.csv

$COST \
    --results-dir $BASE/jailbreakbench/gemma3-4b-it \
    --metrics-csv $OUTPUT/jailbreakbench/gemma3-4b-it/metrics.csv \
    --output      $OUTPUT/jailbreakbench/gemma3-4b-it/cost/cost_metrics.csv

# --- TRAINING STAGE STUDY — Tulu3 8B: Base → SFT → DPO → RLVR ---
# Paper: Table 1, Figure 1 left

# $COST \
#     --results-dir $BASE/jailbreakbench/tulu3-8b-base \
#     --metrics-csv $OUTPUT/jailbreakbench/tulu3-8b-base/metrics.csv \
#     --output      $OUTPUT/jailbreakbench/tulu3-8b-base/cost/cost_metrics.csv

# $COST \
#     --results-dir $BASE/jailbreakbench/tulu3-8b-sft \
#     --metrics-csv $OUTPUT/jailbreakbench/tulu3-8b-sft/metrics.csv \
#     --output      $OUTPUT/jailbreakbench/tulu3-8b-sft/cost/cost_metrics.csv

# $COST \
#     --results-dir $BASE/jailbreakbench/tulu3-8b-dpo \
#     --metrics-csv $OUTPUT/jailbreakbench/tulu3-8b-dpo/metrics.csv \
#     --output      $OUTPUT/jailbreakbench/tulu3-8b-dpo/cost/cost_metrics.csv

# $COST \
#     --results-dir $BASE/jailbreakbench/tulu3-8b-rlvr \
#     --metrics-csv $OUTPUT/jailbreakbench/tulu3-8b-rlvr/metrics.csv \
#     --output      $OUTPUT/jailbreakbench/tulu3-8b-rlvr/cost/cost_metrics.csv

# --- TRAINING STAGE STUDY (2nd ladder) — OLMo 2 1B: Base → SFT → DPO → RLVR1 → Instruct ---
# Companion to the Tulu3 8B ladder above; `-Instruct` is a second RLVR round on `-RLVR1`

# $COST \
#     --results-dir $BASE/jailbreakbench/olmo2-1b-base \
#     --metrics-csv $OUTPUT/jailbreakbench/olmo2-1b-base/metrics.csv \
#     --output      $OUTPUT/jailbreakbench/olmo2-1b-base/cost/cost_metrics.csv

# $COST \
#     --results-dir $BASE/jailbreakbench/olmo2-1b-sft \
#     --metrics-csv $OUTPUT/jailbreakbench/olmo2-1b-sft/metrics.csv \
#     --output      $OUTPUT/jailbreakbench/olmo2-1b-sft/cost/cost_metrics.csv

# $COST \
#     --results-dir $BASE/jailbreakbench/olmo2-1b-dpo \
#     --metrics-csv $OUTPUT/jailbreakbench/olmo2-1b-dpo/metrics.csv \
#     --output      $OUTPUT/jailbreakbench/olmo2-1b-dpo/cost/cost_metrics.csv

# $COST \
#     --results-dir $BASE/jailbreakbench/olmo2-1b-rlvr1 \
#     --metrics-csv $OUTPUT/jailbreakbench/olmo2-1b-rlvr1/metrics.csv \
#     --output      $OUTPUT/jailbreakbench/olmo2-1b-rlvr1/cost/cost_metrics.csv

# $COST \
#     --results-dir $BASE/jailbreakbench/olmo2-1b-instruct \
#     --metrics-csv $OUTPUT/jailbreakbench/olmo2-1b-instruct/metrics.csv \
#     --output      $OUTPUT/jailbreakbench/olmo2-1b-instruct/cost/cost_metrics.csv

# --- SAFETY ALIGNMENT STUDY — Qwen3-4B base vs Qwen3-4B-SafeRL ---

# $COST \
#     --results-dir $BASE/jailbreakbench/qwen3-4b \
#     --metrics-csv $OUTPUT/jailbreakbench/qwen3-4b/metrics.csv \
#     --output      $OUTPUT/jailbreakbench/qwen3-4b/cost/cost_metrics.csv

# $COST \
#     --results-dir $BASE/jailbreakbench/qwen3-4b-saferl \
#     --metrics-csv $OUTPUT/jailbreakbench/qwen3-4b-saferl/metrics.csv \
#     --output      $OUTPUT/jailbreakbench/qwen3-4b-saferl/cost/cost_metrics.csv
