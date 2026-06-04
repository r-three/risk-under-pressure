#!/bin/bash
# run_cost_plots.sh — Cost-axis (tokens / TFLOPs) risk curves for all experiments.
# Reads cost_metrics.csv from $SCRATCH/rup/plots (written by run_cost_evaluations.sh).
#
# Follows the same structure as run_plots.sh:
#   1. Per-model plots   — seed-aggregated mean/CI per model (tokens + flops)
#   2. Comparison plots  — ablation sets: Qwen2.5 size, Tulu3 training phases,
#                          safety alignment, best-per-family, attack transfer (GCG)
#
# All output goes to:
#   $OUTPUT/{harmbench,jailbreakbench}/$model/{tokens,flops}/  (per-model)
#   $OUTPUT/{harmbench,jailbreakbench}/ablations/<name>/{tokens,flops}/
#
# Usage: bash run_cost_plots.sh
# Requires: must be run from the project root.

set -e

source setup/start_env.sh

OUTPUT=$SCRATCH/rup/plots

COST_PLOT="python plot_cost_curves.py"

# =============================================================================
# Per-model plots (seed-aggregated CI)
# =============================================================================

# --------------------------------------------------------------------------- #
# HarmBench
# --------------------------------------------------------------------------- #

# --- MODEL SIZE STUDY — Qwen2.5-Instruct: 0.5B, 3B, 7B ---
# Paper: Figure 1 right

# for axis in tokens flops; do
#     $COST_PLOT \
#         --cost-csv  $OUTPUT/harmbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv \
#         --cost-category-csv $OUTPUT/harmbench/qwen2.5-0.5b-instruct/cost/cost_metrics_by_category.csv \
#         --output-dir $OUTPUT/harmbench/qwen2.5-0.5b-instruct/$axis \
#         --x-axis $axis --skip-missing
# done

# for axis in tokens flops; do
#     $COST_PLOT \
#         --cost-csv  $OUTPUT/harmbench/qwen2.5-3b-instruct/cost/cost_metrics.csv \
#         --cost-category-csv $OUTPUT/harmbench/qwen2.5-3b-instruct/cost/cost_metrics_by_category.csv \
#         --output-dir $OUTPUT/harmbench/qwen2.5-3b-instruct/$axis \
#         --x-axis $axis --skip-missing
# done

# for axis in tokens flops; do
#     $COST_PLOT \
#         --cost-csv  $OUTPUT/harmbench/qwen2.5-7b-instruct/cost/cost_metrics.csv \
#         --cost-category-csv $OUTPUT/harmbench/qwen2.5-7b-instruct/cost/cost_metrics_by_category.csv \
#         --output-dir $OUTPUT/harmbench/qwen2.5-7b-instruct/$axis \
#         --x-axis $axis --skip-missing
# done

# --- TRAINING STAGE STUDY — Tulu3 8B: Base → SFT → DPO → RLVR ---
# Paper: Table 1, Figure 1 left

# for axis in tokens flops; do
#     $COST_PLOT \
#         --cost-csv  $OUTPUT/harmbench/tulu3-8b-base/cost/cost_metrics.csv \
#         --cost-category-csv $OUTPUT/harmbench/tulu3-8b-base/cost/cost_metrics_by_category.csv \
#         --output-dir $OUTPUT/harmbench/tulu3-8b-base/$axis \
#         --x-axis $axis --skip-missing
# done

# for axis in tokens flops; do
#     $COST_PLOT \
#         --cost-csv  $OUTPUT/harmbench/tulu3-8b-sft/cost/cost_metrics.csv \
#         --cost-category-csv $OUTPUT/harmbench/tulu3-8b-sft/cost/cost_metrics_by_category.csv \
#         --output-dir $OUTPUT/harmbench/tulu3-8b-sft/$axis \
#         --x-axis $axis --skip-missing
# done

# for axis in tokens flops; do
#     $COST_PLOT \
#         --cost-csv  $OUTPUT/harmbench/tulu3-8b-dpo/cost/cost_metrics.csv \
#         --cost-category-csv $OUTPUT/harmbench/tulu3-8b-dpo/cost/cost_metrics_by_category.csv \
#         --output-dir $OUTPUT/harmbench/tulu3-8b-dpo/$axis \
#         --x-axis $axis --skip-missing
# done

# for axis in tokens flops; do
#     $COST_PLOT \
#         --cost-csv  $OUTPUT/harmbench/tulu3-8b-rlvr/cost/cost_metrics.csv \
#         --cost-category-csv $OUTPUT/harmbench/tulu3-8b-rlvr/cost/cost_metrics_by_category.csv \
#         --output-dir $OUTPUT/harmbench/tulu3-8b-rlvr/$axis \
#         --x-axis $axis --skip-missing
# done

# --- SAFETY ALIGNMENT STUDY — Qwen3-4B base vs Qwen3-4B-SafeRL ---
# Paper: Table 1 (Qwen3 rows)

for axis in tokens flops; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/harmbench/qwen3-4b/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/harmbench/qwen3-4b/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/qwen3-4b/$axis \
        --x-axis $axis --skip-missing
done

# for axis in tokens flops; do
#     $COST_PLOT \
#         --cost-csv  $OUTPUT/harmbench/qwen3-4b-saferl/cost/cost_metrics.csv \
#         --cost-category-csv $OUTPUT/harmbench/qwen3-4b-saferl/cost/cost_metrics_by_category.csv \
#         --output-dir $OUTPUT/harmbench/qwen3-4b-saferl/$axis \
#         --x-axis $axis --skip-missing
# done

# --------------------------------------------------------------------------- #
# JailbreakBench
# --------------------------------------------------------------------------- #

# --- MODEL SIZE STUDY — Qwen2.5-Instruct: 0.5B, 3B, 7B ---
# Paper: Figure 1 right

# for axis in tokens flops; do
#     $COST_PLOT \
#         --cost-csv  $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv \
#         --cost-category-csv $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/cost/cost_metrics_by_category.csv \
#         --output-dir $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/$axis \
#         --x-axis $axis --skip-missing
# done

# for axis in tokens flops; do
#     $COST_PLOT \
#         --cost-csv  $OUTPUT/jailbreakbench/qwen2.5-3b-instruct/cost/cost_metrics.csv \
#         --cost-category-csv $OUTPUT/jailbreakbench/qwen2.5-3b-instruct/cost/cost_metrics_by_category.csv \
#         --output-dir $OUTPUT/jailbreakbench/qwen2.5-3b-instruct/$axis \
#         --x-axis $axis --skip-missing
# done

# for axis in tokens flops; do
#     $COST_PLOT \
#         --cost-csv  $OUTPUT/jailbreakbench/qwen2.5-7b-instruct/cost/cost_metrics.csv \
#         --cost-category-csv $OUTPUT/jailbreakbench/qwen2.5-7b-instruct/cost/cost_metrics_by_category.csv \
#         --output-dir $OUTPUT/jailbreakbench/qwen2.5-7b-instruct/$axis \
#         --x-axis $axis --skip-missing
# done

# --- TRAINING STAGE STUDY — Tulu3 8B: Base → SFT → DPO → RLVR ---
# Paper: Table 1, Figure 1 left

# for axis in tokens flops; do
#     $COST_PLOT \
#         --cost-csv  $OUTPUT/jailbreakbench/tulu3-8b-base/cost/cost_metrics.csv \
#         --cost-category-csv $OUTPUT/jailbreakbench/tulu3-8b-base/cost/cost_metrics_by_category.csv \
#         --output-dir $OUTPUT/jailbreakbench/tulu3-8b-base/$axis \
#         --x-axis $axis --skip-missing
# done

# for axis in tokens flops; do
#     $COST_PLOT \
#         --cost-csv  $OUTPUT/jailbreakbench/tulu3-8b-sft/cost/cost_metrics.csv \
#         --cost-category-csv $OUTPUT/jailbreakbench/tulu3-8b-sft/cost/cost_metrics_by_category.csv \
#         --output-dir $OUTPUT/jailbreakbench/tulu3-8b-sft/$axis \
#         --x-axis $axis --skip-missing
# done

# for axis in tokens flops; do
#     $COST_PLOT \
#         --cost-csv  $OUTPUT/jailbreakbench/tulu3-8b-dpo/cost/cost_metrics.csv \
#         --cost-category-csv $OUTPUT/jailbreakbench/tulu3-8b-dpo/cost/cost_metrics_by_category.csv \
#         --output-dir $OUTPUT/jailbreakbench/tulu3-8b-dpo/$axis \
#         --x-axis $axis --skip-missing
# done

# for axis in tokens flops; do
#     $COST_PLOT \
#         --cost-csv  $OUTPUT/jailbreakbench/tulu3-8b-rlvr/cost/cost_metrics.csv \
#         --cost-category-csv $OUTPUT/jailbreakbench/tulu3-8b-rlvr/cost/cost_metrics_by_category.csv \
#         --output-dir $OUTPUT/jailbreakbench/tulu3-8b-rlvr/$axis \
#         --x-axis $axis --skip-missing
# done

# --- SAFETY ALIGNMENT STUDY — Qwen3-4B base vs Qwen3-4B-SafeRL ---

for axis in tokens flops; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/jailbreakbench/qwen3-4b/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/jailbreakbench/qwen3-4b/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/jailbreakbench/qwen3-4b/$axis \
        --x-axis $axis --skip-missing
done

# for axis in tokens flops; do
#     $COST_PLOT \
#         --cost-csv  $OUTPUT/jailbreakbench/qwen3-4b-saferl/cost/cost_metrics.csv \
#         --cost-category-csv $OUTPUT/jailbreakbench/qwen3-4b-saferl/cost/cost_metrics_by_category.csv \
#         --output-dir $OUTPUT/jailbreakbench/qwen3-4b-saferl/$axis \
#         --x-axis $axis --skip-missing
# done

# =============================================================================
# Cross-model comparison plots
# =============================================================================

# echo "=== Comparison plots: HarmBench (tokens) ==="
# $COST_PLOT \
#     --cost-csv \
#         $OUTPUT/harmbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv \
#         $OUTPUT/harmbench/qwen2.5-3b-instruct/cost/cost_metrics.csv \
#         $OUTPUT/harmbench/qwen2.5-7b-instruct/cost/cost_metrics.csv \
#         $OUTPUT/harmbench/tulu3-8b-base/cost/cost_metrics.csv \
#         $OUTPUT/harmbench/tulu3-8b-sft/cost/cost_metrics.csv \
#         $OUTPUT/harmbench/tulu3-8b-dpo/cost/cost_metrics.csv \
#         $OUTPUT/harmbench/tulu3-8b-rlvr/cost/cost_metrics.csv \
#         $OUTPUT/harmbench/qwen3-4b/cost/cost_metrics.csv \
#         $OUTPUT/harmbench/qwen3-4b-saferl/cost/cost_metrics.csv \
#     --output-dir $OUTPUT/harmbench/comparison_plots/tokens \
#     --x-axis tokens --title "HarmBench — All Models" \
#     --mode comparison --skip-missing

# echo "=== Comparison plots: HarmBench (flops) ==="
# $COST_PLOT \
#     --cost-csv \
#         $OUTPUT/harmbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv \
#         $OUTPUT/harmbench/qwen2.5-3b-instruct/cost/cost_metrics.csv \
#         $OUTPUT/harmbench/qwen2.5-7b-instruct/cost/cost_metrics.csv \
#         $OUTPUT/harmbench/tulu3-8b-base/cost/cost_metrics.csv \
#         $OUTPUT/harmbench/tulu3-8b-sft/cost/cost_metrics.csv \
#         $OUTPUT/harmbench/tulu3-8b-dpo/cost/cost_metrics.csv \
#         $OUTPUT/harmbench/tulu3-8b-rlvr/cost/cost_metrics.csv \
#         $OUTPUT/harmbench/qwen3-4b/cost/cost_metrics.csv \
#         $OUTPUT/harmbench/qwen3-4b-saferl/cost/cost_metrics.csv \
#     --output-dir $OUTPUT/harmbench/comparison_plots/flops \
#     --x-axis flops --title "HarmBench — All Models" \
#     --mode comparison --skip-missing

# echo "=== Comparison plots: JailbreakBench (tokens) ==="
# $COST_PLOT \
#     --cost-csv \
#         $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv \
#         $OUTPUT/jailbreakbench/qwen2.5-3b-instruct/cost/cost_metrics.csv \
#         $OUTPUT/jailbreakbench/qwen2.5-7b-instruct/cost/cost_metrics.csv \
#         $OUTPUT/jailbreakbench/tulu3-8b-base/cost/cost_metrics.csv \
#         $OUTPUT/jailbreakbench/tulu3-8b-sft/cost/cost_metrics.csv \
#         $OUTPUT/jailbreakbench/tulu3-8b-dpo/cost/cost_metrics.csv \
#         $OUTPUT/jailbreakbench/tulu3-8b-rlvr/cost/cost_metrics.csv \
#         $OUTPUT/jailbreakbench/qwen3-4b/cost/cost_metrics.csv \
#         $OUTPUT/jailbreakbench/qwen3-4b-saferl/cost/cost_metrics.csv \
#     --output-dir $OUTPUT/jailbreakbench/comparison_plots/tokens \
#     --x-axis tokens --title "JailbreakBench — All Models" \
#     --mode comparison --skip-missing

# echo "=== Comparison plots: JailbreakBench (flops) ==="
# $COST_PLOT \
#     --cost-csv \
#         $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv \
#         $OUTPUT/jailbreakbench/qwen2.5-3b-instruct/cost/cost_metrics.csv \
#         $OUTPUT/jailbreakbench/qwen2.5-7b-instruct/cost/cost_metrics.csv \
#         $OUTPUT/jailbreakbench/tulu3-8b-base/cost/cost_metrics.csv \
#         $OUTPUT/jailbreakbench/tulu3-8b-sft/cost/cost_metrics.csv \
#         $OUTPUT/jailbreakbench/tulu3-8b-dpo/cost/cost_metrics.csv \
#         $OUTPUT/jailbreakbench/tulu3-8b-rlvr/cost/cost_metrics.csv \
#         $OUTPUT/jailbreakbench/qwen3-4b/cost/cost_metrics.csv \
#         $OUTPUT/jailbreakbench/qwen3-4b-saferl/cost/cost_metrics.csv \
#     --output-dir $OUTPUT/jailbreakbench/comparison_plots/flops \
#     --x-axis flops --title "JailbreakBench — All Models" \
#     --mode comparison --skip-missing

# =============================================================================
# Ablation plots
# =============================================================================

# --- Ablation 1: Qwen2.5 model size ---
# Paper: Figure 1 right

# for axis in tokens flops; do
#     $COST_PLOT \
#         --cost-csv \
#             $OUTPUT/harmbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv \
#             $OUTPUT/harmbench/qwen2.5-3b-instruct/cost/cost_metrics.csv \
#             $OUTPUT/harmbench/qwen2.5-7b-instruct/cost/cost_metrics.csv \
#         --cost-category-csv \
#             $OUTPUT/harmbench/qwen2.5-0.5b-instruct/cost/cost_metrics_by_category.csv \
#             $OUTPUT/harmbench/qwen2.5-3b-instruct/cost/cost_metrics_by_category.csv \
#             $OUTPUT/harmbench/qwen2.5-7b-instruct/cost/cost_metrics_by_category.csv \
#         --output-dir $OUTPUT/harmbench/ablations/qwen_size/$axis \
#         --x-axis $axis --title "HarmBench — Qwen2.5 Model Size" \
#         --mode comparison --skip-missing
# done

# for axis in tokens flops; do
#     $COST_PLOT \
#         --cost-csv \
#             $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv \
#             $OUTPUT/jailbreakbench/qwen2.5-3b-instruct/cost/cost_metrics.csv \
#             $OUTPUT/jailbreakbench/qwen2.5-7b-instruct/cost/cost_metrics.csv \
#         --cost-category-csv \
#             $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/cost/cost_metrics_by_category.csv \
#             $OUTPUT/jailbreakbench/qwen2.5-3b-instruct/cost/cost_metrics_by_category.csv \
#             $OUTPUT/jailbreakbench/qwen2.5-7b-instruct/cost/cost_metrics_by_category.csv \
#         --output-dir $OUTPUT/jailbreakbench/ablations/qwen_size/$axis \
#         --x-axis $axis --title "JailbreakBench — Qwen2.5 Model Size" \
#         --mode comparison --skip-missing
# done

# --- Ablation 2: Tulu3 training stages ---
# Paper: Table 1, Figure 1 left

# for axis in tokens flops; do
#     $COST_PLOT \
#         --cost-csv \
#             $OUTPUT/harmbench/tulu3-8b-base/cost/cost_metrics.csv \
#             $OUTPUT/harmbench/tulu3-8b-sft/cost/cost_metrics.csv \
#             $OUTPUT/harmbench/tulu3-8b-dpo/cost/cost_metrics.csv \
#             $OUTPUT/harmbench/tulu3-8b-rlvr/cost/cost_metrics.csv \
#         --cost-category-csv \
#             $OUTPUT/harmbench/tulu3-8b-base/cost/cost_metrics_by_category.csv \
#             $OUTPUT/harmbench/tulu3-8b-sft/cost/cost_metrics_by_category.csv \
#             $OUTPUT/harmbench/tulu3-8b-dpo/cost/cost_metrics_by_category.csv \
#             $OUTPUT/harmbench/tulu3-8b-rlvr/cost/cost_metrics_by_category.csv \
#         --output-dir $OUTPUT/harmbench/ablations/tulu3_training/$axis \
#         --x-axis $axis --title "HarmBench — Tulu3 Training Stages" \
#         --mode comparison --skip-missing
# done

# for axis in tokens flops; do
#     $COST_PLOT \
#         --cost-csv \
#             $OUTPUT/jailbreakbench/tulu3-8b-base/cost/cost_metrics.csv \
#             $OUTPUT/jailbreakbench/tulu3-8b-sft/cost/cost_metrics.csv \
#             $OUTPUT/jailbreakbench/tulu3-8b-dpo/cost/cost_metrics.csv \
#             $OUTPUT/jailbreakbench/tulu3-8b-rlvr/cost/cost_metrics.csv \
#         --cost-category-csv \
#             $OUTPUT/jailbreakbench/tulu3-8b-base/cost/cost_metrics_by_category.csv \
#             $OUTPUT/jailbreakbench/tulu3-8b-sft/cost/cost_metrics_by_category.csv \
#             $OUTPUT/jailbreakbench/tulu3-8b-dpo/cost/cost_metrics_by_category.csv \
#             $OUTPUT/jailbreakbench/tulu3-8b-rlvr/cost/cost_metrics_by_category.csv \
#         --output-dir $OUTPUT/jailbreakbench/ablations/tulu3_training/$axis \
#         --x-axis $axis --title "JailbreakBench — Tulu3 Training Stages" \
#         --mode comparison --skip-missing
# done

# --- Ablation 3: Safety alignment — Qwen3-4B base vs Qwen3-4B-SafeRL ---
# Paper: Table 1 (Qwen3 rows)

# for axis in tokens flops; do
#     $COST_PLOT \
#         --cost-csv \
#             $OUTPUT/harmbench/qwen3-4b/cost/cost_metrics.csv \
#             $OUTPUT/harmbench/qwen3-4b-saferl/cost/cost_metrics.csv \
#         --cost-category-csv \
#             $OUTPUT/harmbench/qwen3-4b/cost/cost_metrics_by_category.csv \
#             $OUTPUT/harmbench/qwen3-4b-saferl/cost/cost_metrics_by_category.csv \
#         --output-dir $OUTPUT/harmbench/ablations/safety_alignment/$axis \
#         --x-axis $axis --title "HarmBench — Safety Alignment (Qwen3-4B)" \
#         --mode comparison --skip-missing
# done

# for axis in tokens flops; do
#     $COST_PLOT \
#         --cost-csv \
#             $OUTPUT/jailbreakbench/qwen3-4b/cost/cost_metrics.csv \
#             $OUTPUT/jailbreakbench/qwen3-4b-saferl/cost/cost_metrics.csv \
#         --cost-category-csv \
#             $OUTPUT/jailbreakbench/qwen3-4b/cost/cost_metrics_by_category.csv \
#             $OUTPUT/jailbreakbench/qwen3-4b-saferl/cost/cost_metrics_by_category.csv \
#         --output-dir $OUTPUT/jailbreakbench/ablations/safety_alignment/$axis \
#         --x-axis $axis --title "JailbreakBench — Safety Alignment (Qwen3-4B)" \
#         --mode comparison --skip-missing
# done

# --- Ablation 4: Best per family ---

# BEST_HB_QWEN25=$(python scripts/select_best_model.py --metrics-dir $OUTPUT/harmbench \
#     --models qwen2.5-0.5b-instruct qwen2.5-3b-instruct qwen2.5-7b-instruct)
# BEST_HB_TULU3=$(python scripts/select_best_model.py --metrics-dir $OUTPUT/harmbench \
#     --models tulu3-8b-base tulu3-8b-sft tulu3-8b-dpo tulu3-8b-rlvr)
# BEST_JB_QWEN25=$(python scripts/select_best_model.py --metrics-dir $OUTPUT/jailbreakbench \
#     --models qwen2.5-0.5b-instruct qwen2.5-3b-instruct qwen2.5-7b-instruct)
# BEST_JB_TULU3=$(python scripts/select_best_model.py --metrics-dir $OUTPUT/jailbreakbench \
#     --models tulu3-8b-base tulu3-8b-sft tulu3-8b-dpo tulu3-8b-rlvr)

# for axis in tokens flops; do
#     $COST_PLOT \
#         --cost-csv \
#             $OUTPUT/harmbench/$BEST_HB_QWEN25/cost/cost_metrics.csv \
#             $OUTPUT/harmbench/$BEST_HB_TULU3/cost/cost_metrics.csv \
#             $OUTPUT/harmbench/qwen3-4b/cost/cost_metrics.csv \
#             $OUTPUT/harmbench/qwen3-4b-saferl/cost/cost_metrics.csv \
#         --cost-category-csv \
#             $OUTPUT/harmbench/$BEST_HB_QWEN25/cost/cost_metrics_by_category.csv \
#             $OUTPUT/harmbench/$BEST_HB_TULU3/cost/cost_metrics_by_category.csv \
#             $OUTPUT/harmbench/qwen3-4b/cost/cost_metrics_by_category.csv \
#             $OUTPUT/harmbench/qwen3-4b-saferl/cost/cost_metrics_by_category.csv \
#         --output-dir $OUTPUT/harmbench/ablations/best_per_family/$axis \
#         --x-axis $axis --title "HarmBench — Best per Family" \
#         --mode comparison --skip-missing
# done

# for axis in tokens flops; do
#     $COST_PLOT \
#         --cost-csv \
#             $OUTPUT/jailbreakbench/$BEST_JB_QWEN25/cost/cost_metrics.csv \
#             $OUTPUT/jailbreakbench/$BEST_JB_TULU3/cost/cost_metrics.csv \
#             $OUTPUT/jailbreakbench/qwen3-4b/cost/cost_metrics.csv \
#             $OUTPUT/jailbreakbench/qwen3-4b-saferl/cost/cost_metrics.csv \
#         --cost-category-csv \
#             $OUTPUT/jailbreakbench/$BEST_JB_QWEN25/cost/cost_metrics_by_category.csv \
#             $OUTPUT/jailbreakbench/$BEST_JB_TULU3/cost/cost_metrics_by_category.csv \
#             $OUTPUT/jailbreakbench/qwen3-4b/cost/cost_metrics_by_category.csv \
#             $OUTPUT/jailbreakbench/qwen3-4b-saferl/cost/cost_metrics_by_category.csv \
#         --output-dir $OUTPUT/jailbreakbench/ablations/best_per_family/$axis \
#         --x-axis $axis --title "JailbreakBench — Best per Family" \
#         --mode comparison --skip-missing
# done

# --- Ablation 5: Attack transfer (GCG surrogate: qwen2.5-0.5b → qwen3-4b) ---

# _normalize_attack() {
#     local src="$1" dst="$2"
#     python3 -c "
# import pandas as pd
# df = pd.read_csv('$src')
# df['attack_id'] = df['attack_id'].str.replace(r'transfer_gcg_from_.*', 'gcg', regex=True)
# df.to_csv('$dst', index=False)
# " 2>/dev/null || cp "$src" "$dst"
# }
#
# NORM_HB_Q3=/tmp/norm_hb_qwen3_cost.csv
# NORM_JB_Q3=/tmp/norm_jb_qwen3_cost.csv
# NORM_HB_Q3_CAT=/tmp/norm_hb_qwen3_cost_cat.csv
# NORM_JB_Q3_CAT=/tmp/norm_jb_qwen3_cost_cat.csv
#
# _normalize_attack $OUTPUT/harmbench/qwen3-4b/cost/cost_metrics.csv             $NORM_HB_Q3
# _normalize_attack $OUTPUT/jailbreakbench/qwen3-4b/cost/cost_metrics.csv         $NORM_JB_Q3
# _normalize_attack $OUTPUT/harmbench/qwen3-4b/cost/cost_metrics_by_category.csv  $NORM_HB_Q3_CAT
# _normalize_attack $OUTPUT/jailbreakbench/qwen3-4b/cost/cost_metrics_by_category.csv $NORM_JB_Q3_CAT
#
# for axis in tokens flops; do
#     $COST_PLOT \
#         --cost-csv \
#             $OUTPUT/harmbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv \
#             $NORM_HB_Q3 \
#         --cost-category-csv \
#             $OUTPUT/harmbench/qwen2.5-0.5b-instruct/cost/cost_metrics_by_category.csv \
#             $NORM_HB_Q3_CAT \
#         --output-dir $OUTPUT/harmbench/ablations/attack_transfer_gcg/$axis \
#         --x-axis $axis --attacks gcg \
#         --title "HarmBench — GCG Transfer: Qwen2.5-0.5B → Qwen3-4B" \
#         --mode comparison --skip-missing
# done
#
# for axis in tokens flops; do
#     $COST_PLOT \
#         --cost-csv \
#             $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv \
#             $NORM_JB_Q3 \
#         --cost-category-csv \
#             $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/cost/cost_metrics_by_category.csv \
#             $NORM_JB_Q3_CAT \
#         --output-dir $OUTPUT/jailbreakbench/ablations/attack_transfer_gcg/$axis \
#         --x-axis $axis --attacks gcg \
#         --title "JailbreakBench — GCG Transfer: Qwen2.5-0.5B → Qwen3-4B" \
#         --mode comparison --skip-missing
# done

echo "Done."
