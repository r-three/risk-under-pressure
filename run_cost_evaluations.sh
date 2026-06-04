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

set -e

source setup/start_env.sh

BASE=$SCRATCH/rup
OUTPUT=$SCRATCH/rup/plots

COST="python scripts/compute_attack_costs.py"

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

# --- SAFETY ALIGNMENT STUDY — Qwen3-4B base vs Qwen3-4B-SafeRL ---
# Paper: Table 1 (Qwen3 rows)

$COST \
    --results-dir $BASE/harmbench/qwen3-4b \
    --metrics-csv $OUTPUT/harmbench/qwen3-4b/metrics.csv \
    --output      $OUTPUT/harmbench/qwen3-4b/cost/cost_metrics.csv

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

# --- SAFETY ALIGNMENT STUDY — Qwen3-4B base vs Qwen3-4B-SafeRL ---

$COST \
    --results-dir $BASE/jailbreakbench/qwen3-4b \
    --metrics-csv $OUTPUT/jailbreakbench/qwen3-4b/metrics.csv \
    --output      $OUTPUT/jailbreakbench/qwen3-4b/cost/cost_metrics.csv

# $COST \
#     --results-dir $BASE/jailbreakbench/qwen3-4b-saferl \
#     --metrics-csv $OUTPUT/jailbreakbench/qwen3-4b-saferl/metrics.csv \
#     --output      $OUTPUT/jailbreakbench/qwen3-4b-saferl/cost/cost_metrics.csv
