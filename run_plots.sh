#!/bin/bash
# Run Phase 3 plotting for all experiments.
# Reads from $SCRATCH/rup/plots (written by run_evaluations.sh).
#
# Per-model plots: one figure per attack + combined, written to each model's plots/ dir.
# Comparison plots: all models overlaid per attack, written to comparison_plots/.
#
# Usage: bash run_plots.sh
# Requires: must be run from the project root.
#
# JUDGE selects which judge's results to plot (a config name under configs/models/,
# without .yaml); defaults to llama3.1_8b_instruct_judge and reads $SCRATCH/rup/plots
# as before. Other judges read $SCRATCH/rup/judges/<judge_model_id>/plots.
#   JUDGE=gemma3_4b_it_judge bash run_plots.sh

set -e

source setup/start_env.sh
source setup/judge_env.sh

OUTPUT=$PLOT_ROOT

# =============================================================================
# Per-model plots (seed-aggregated CI)
# =============================================================================

# --------------------------------------------------------------------------- #
# HarmBench
# --------------------------------------------------------------------------- #

# --- MODEL SIZE STUDY — Qwen2.5-Instruct: 0.5B, 3B, 7B ---
# Paper: Figure 1 right

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/harmbench/qwen2.5-0.5b-instruct/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/harmbench/qwen2.5-0.5b-instruct/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/qwen2.5-0.5b-instruct/plots/seeds \
    --ci-method seeds

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/harmbench/qwen2.5-3b-instruct/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/harmbench/qwen2.5-3b-instruct/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/qwen2.5-3b-instruct/plots/seeds \
    --ci-method seeds

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/harmbench/qwen2.5-7b-instruct/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/harmbench/qwen2.5-7b-instruct/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/qwen2.5-7b-instruct/plots/seeds \
    --ci-method seeds

# --- MODEL SIZE STUDY (2nd family) — Gemma 3 Instruction-Tuned: 270M, 1B, 4B ---
# Companion to the Qwen2.5 ladder above

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/harmbench/gemma3-270m-it/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/harmbench/gemma3-270m-it/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/gemma3-270m-it/plots/seeds \
    --ci-method seeds

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/harmbench/gemma3-1b-it/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/harmbench/gemma3-1b-it/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/gemma3-1b-it/plots/seeds \
    --ci-method seeds

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/harmbench/gemma3-4b-it/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/harmbench/gemma3-4b-it/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/gemma3-4b-it/plots/seeds \
    --ci-method seeds

# --- TRAINING STAGE STUDY — Tulu3 8B: Base → SFT → DPO → RLVR ---
# Paper: Table 1, Figure 1 left

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/harmbench/tulu3-8b-base/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/harmbench/tulu3-8b-base/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/tulu3-8b-base/plots/seeds \
    --ci-method seeds

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/harmbench/tulu3-8b-sft/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/harmbench/tulu3-8b-sft/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/tulu3-8b-sft/plots/seeds \
    --ci-method seeds

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/harmbench/tulu3-8b-dpo/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/harmbench/tulu3-8b-dpo/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/tulu3-8b-dpo/plots/seeds \
    --ci-method seeds

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/harmbench/tulu3-8b-rlvr/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/harmbench/tulu3-8b-rlvr/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/tulu3-8b-rlvr/plots/seeds \
    --ci-method seeds

# --- TRAINING STAGE STUDY (2nd ladder) — OLMo 2 1B: Base → SFT → DPO → RLVR1 → Instruct ---
# Companion to the Tulu3 8B ladder above

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/harmbench/olmo2-1b-base/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/harmbench/olmo2-1b-base/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/olmo2-1b-base/plots/seeds \
    --ci-method seeds

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/harmbench/olmo2-1b-sft/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/harmbench/olmo2-1b-sft/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/olmo2-1b-sft/plots/seeds \
    --ci-method seeds

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/harmbench/olmo2-1b-dpo/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/harmbench/olmo2-1b-dpo/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/olmo2-1b-dpo/plots/seeds \
    --ci-method seeds

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/harmbench/olmo2-1b-rlvr1/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/harmbench/olmo2-1b-rlvr1/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/olmo2-1b-rlvr1/plots/seeds \
    --ci-method seeds

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/harmbench/olmo2-1b-instruct/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/harmbench/olmo2-1b-instruct/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/olmo2-1b-instruct/plots/seeds \
    --ci-method seeds

# --- SAFETY ALIGNMENT STUDY — Qwen3-4B base vs Qwen3-4B-SafeRL ---
# Paper: Table 1 (Qwen3 rows)

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/harmbench/qwen3-4b/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/harmbench/qwen3-4b/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/qwen3-4b/plots/seeds \
    --ci-method seeds

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/harmbench/qwen3-4b-saferl/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/harmbench/qwen3-4b-saferl/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/qwen3-4b-saferl/plots/seeds \
    --ci-method seeds

# --------------------------------------------------------------------------- #
# JailbreakBench
# --------------------------------------------------------------------------- #

# --- MODEL SIZE STUDY — Qwen2.5-Instruct: 0.5B, 3B, 7B ---
# Paper: Figure 1 right

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/plots/seeds \
    --ci-method seeds

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/jailbreakbench/qwen2.5-3b-instruct/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/jailbreakbench/qwen2.5-3b-instruct/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/qwen2.5-3b-instruct/plots/seeds \
    --ci-method seeds

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/jailbreakbench/qwen2.5-7b-instruct/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/jailbreakbench/qwen2.5-7b-instruct/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/qwen2.5-7b-instruct/plots/seeds \
    --ci-method seeds

# --- MODEL SIZE STUDY (2nd family) — Gemma 3 Instruction-Tuned: 270M, 1B, 4B ---
# Companion to the Qwen2.5 ladder above

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/jailbreakbench/gemma3-270m-it/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/jailbreakbench/gemma3-270m-it/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/gemma3-270m-it/plots/seeds \
    --ci-method seeds

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/jailbreakbench/gemma3-1b-it/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/jailbreakbench/gemma3-1b-it/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/gemma3-1b-it/plots/seeds \
    --ci-method seeds

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/jailbreakbench/gemma3-4b-it/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/jailbreakbench/gemma3-4b-it/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/gemma3-4b-it/plots/seeds \
    --ci-method seeds

# --- TRAINING STAGE STUDY — Tulu3 8B: Base → SFT → DPO → RLVR ---
# Paper: Table 1, Figure 1 left

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/jailbreakbench/tulu3-8b-base/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/jailbreakbench/tulu3-8b-base/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/tulu3-8b-base/plots/seeds \
    --ci-method seeds

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/jailbreakbench/tulu3-8b-sft/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/jailbreakbench/tulu3-8b-sft/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/tulu3-8b-sft/plots/seeds \
    --ci-method seeds

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/jailbreakbench/tulu3-8b-dpo/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/jailbreakbench/tulu3-8b-dpo/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/tulu3-8b-dpo/plots/seeds \
    --ci-method seeds

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/jailbreakbench/tulu3-8b-rlvr/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/jailbreakbench/tulu3-8b-rlvr/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/tulu3-8b-rlvr/plots/seeds \
    --ci-method seeds

# --- TRAINING STAGE STUDY (2nd ladder) — OLMo 2 1B: Base → SFT → DPO → RLVR1 → Instruct ---
# Companion to the Tulu3 8B ladder above

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/jailbreakbench/olmo2-1b-base/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/jailbreakbench/olmo2-1b-base/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/olmo2-1b-base/plots/seeds \
    --ci-method seeds

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/jailbreakbench/olmo2-1b-sft/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/jailbreakbench/olmo2-1b-sft/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/olmo2-1b-sft/plots/seeds \
    --ci-method seeds

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/jailbreakbench/olmo2-1b-dpo/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/jailbreakbench/olmo2-1b-dpo/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/olmo2-1b-dpo/plots/seeds \
    --ci-method seeds

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/jailbreakbench/olmo2-1b-rlvr1/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/jailbreakbench/olmo2-1b-rlvr1/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/olmo2-1b-rlvr1/plots/seeds \
    --ci-method seeds

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/jailbreakbench/olmo2-1b-instruct/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/jailbreakbench/olmo2-1b-instruct/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/olmo2-1b-instruct/plots/seeds \
    --ci-method seeds

# --- SAFETY ALIGNMENT STUDY — Qwen3-4B base vs Qwen3-4B-SafeRL ---

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/jailbreakbench/qwen3-4b/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/jailbreakbench/qwen3-4b/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/qwen3-4b/plots/seeds \
    --ci-method seeds

python scripts/plot_results.py \
    --metrics-csv $OUTPUT/jailbreakbench/qwen3-4b-saferl/metrics_summary.csv \
    --category-metrics-csv $OUTPUT/jailbreakbench/qwen3-4b-saferl/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/qwen3-4b-saferl/plots/seeds \
    --ci-method seeds

# =============================================================================
# Cross-model comparison plots (seed-aggregated summary CSVs)
# =============================================================================

echo "=== Comparison plots: HarmBench ==="
python scripts/plot_results.py \
    --metrics-csv \
        $OUTPUT/harmbench/qwen2.5-0.5b-instruct/metrics_summary.csv \
        $OUTPUT/harmbench/qwen2.5-3b-instruct/metrics_summary.csv \
        $OUTPUT/harmbench/qwen2.5-7b-instruct/metrics_summary.csv \
        $OUTPUT/harmbench/tulu3-8b-base/metrics_summary.csv \
        $OUTPUT/harmbench/tulu3-8b-sft/metrics_summary.csv \
        $OUTPUT/harmbench/tulu3-8b-dpo/metrics_summary.csv \
        $OUTPUT/harmbench/tulu3-8b-rlvr/metrics_summary.csv \
        $OUTPUT/harmbench/qwen3-4b/metrics_summary.csv \
        $OUTPUT/harmbench/qwen3-4b-saferl/metrics_summary.csv \
    --category-metrics-csv \
        $OUTPUT/harmbench/qwen2.5-0.5b-instruct/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/qwen2.5-3b-instruct/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/qwen2.5-7b-instruct/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/tulu3-8b-base/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/tulu3-8b-sft/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/tulu3-8b-dpo/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/tulu3-8b-rlvr/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/qwen3-4b/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/qwen3-4b-saferl/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/comparison_plots \
    --title "HarmBench — All Models"

echo "=== Comparison plots: JailbreakBench ==="
python scripts/plot_results.py \
    --metrics-csv \
        $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/metrics_summary.csv \
        $OUTPUT/jailbreakbench/qwen2.5-3b-instruct/metrics_summary.csv \
        $OUTPUT/jailbreakbench/qwen2.5-7b-instruct/metrics_summary.csv \
        $OUTPUT/jailbreakbench/tulu3-8b-base/metrics_summary.csv \
        $OUTPUT/jailbreakbench/tulu3-8b-sft/metrics_summary.csv \
        $OUTPUT/jailbreakbench/tulu3-8b-dpo/metrics_summary.csv \
        $OUTPUT/jailbreakbench/tulu3-8b-rlvr/metrics_summary.csv \
        $OUTPUT/jailbreakbench/qwen3-4b/metrics_summary.csv \
        $OUTPUT/jailbreakbench/qwen3-4b-saferl/metrics_summary.csv \
    --category-metrics-csv \
        $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/qwen2.5-3b-instruct/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/qwen2.5-7b-instruct/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/tulu3-8b-base/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/tulu3-8b-sft/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/tulu3-8b-dpo/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/tulu3-8b-rlvr/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/qwen3-4b/metrics_summary.csv \
        $OUTPUT/jailbreakbench/qwen3-4b-saferl/metrics_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/comparison_plots \
    --title "JailbreakBench — All Models"

# =============================================================================
# Ablation plots
# =============================================================================

# --- Ablation 1: Qwen2.5 model size ---
# Paper: Figure 1 right

python scripts/plot_results.py \
    --metrics-csv \
        $OUTPUT/harmbench/qwen2.5-0.5b-instruct/metrics_summary.csv \
        $OUTPUT/harmbench/qwen2.5-3b-instruct/metrics_summary.csv \
        $OUTPUT/harmbench/qwen2.5-7b-instruct/metrics_summary.csv \
    --category-metrics-csv \
        $OUTPUT/harmbench/qwen2.5-0.5b-instruct/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/qwen2.5-3b-instruct/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/qwen2.5-7b-instruct/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/ablations/qwen_size \
    --title "HarmBench — Qwen2.5 Model Size Ablation"

python scripts/plot_results.py \
    --metrics-csv \
        $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/metrics_summary.csv \
        $OUTPUT/jailbreakbench/qwen2.5-3b-instruct/metrics_summary.csv \
        $OUTPUT/jailbreakbench/qwen2.5-7b-instruct/metrics_summary.csv \
    --category-metrics-csv \
        $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/qwen2.5-3b-instruct/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/qwen2.5-7b-instruct/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/ablations/qwen_size \
    --title "JailbreakBench — Qwen2.5 Model Size Ablation"

# --- Ablation 1b: Gemma 3 model size (2nd family) ---

python scripts/plot_results.py \
    --metrics-csv \
        $OUTPUT/harmbench/gemma3-270m-it/metrics_summary.csv \
        $OUTPUT/harmbench/gemma3-1b-it/metrics_summary.csv \
        $OUTPUT/harmbench/gemma3-4b-it/metrics_summary.csv \
    --category-metrics-csv \
        $OUTPUT/harmbench/gemma3-270m-it/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/gemma3-1b-it/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/gemma3-4b-it/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/ablations/gemma_size \
    --title "HarmBench — Gemma 3 Model Size Ablation"

python scripts/plot_results.py \
    --metrics-csv \
        $OUTPUT/jailbreakbench/gemma3-270m-it/metrics_summary.csv \
        $OUTPUT/jailbreakbench/gemma3-1b-it/metrics_summary.csv \
        $OUTPUT/jailbreakbench/gemma3-4b-it/metrics_summary.csv \
    --category-metrics-csv \
        $OUTPUT/jailbreakbench/gemma3-270m-it/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/gemma3-1b-it/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/gemma3-4b-it/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/ablations/gemma_size \
    --title "JailbreakBench — Gemma 3 Model Size Ablation"

# --- Ablation 2: Tulu3 training stages ---
# Paper: Table 1, Figure 1 left

python scripts/plot_results.py \
    --metrics-csv \
        $OUTPUT/harmbench/tulu3-8b-base/metrics_summary.csv \
        $OUTPUT/harmbench/tulu3-8b-sft/metrics_summary.csv \
        $OUTPUT/harmbench/tulu3-8b-dpo/metrics_summary.csv \
        $OUTPUT/harmbench/tulu3-8b-rlvr/metrics_summary.csv \
    --category-metrics-csv \
        $OUTPUT/harmbench/tulu3-8b-base/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/tulu3-8b-sft/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/tulu3-8b-dpo/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/tulu3-8b-rlvr/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/ablations/tulu3_training \
    --title "HarmBench — Tulu3 Training Phase Ablation"

python scripts/plot_results.py \
    --metrics-csv \
        $OUTPUT/jailbreakbench/tulu3-8b-base/metrics_summary.csv \
        $OUTPUT/jailbreakbench/tulu3-8b-sft/metrics_summary.csv \
        $OUTPUT/jailbreakbench/tulu3-8b-dpo/metrics_summary.csv \
        $OUTPUT/jailbreakbench/tulu3-8b-rlvr/metrics_summary.csv \
    --category-metrics-csv \
        $OUTPUT/jailbreakbench/tulu3-8b-base/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/tulu3-8b-sft/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/tulu3-8b-dpo/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/tulu3-8b-rlvr/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/ablations/tulu3_training \
    --title "JailbreakBench — Tulu3 Training Phase Ablation"

# --- Ablation 2b: OLMo 2 1B training stages (2nd ladder, 5 rungs) ---

python scripts/plot_results.py \
    --metrics-csv \
        $OUTPUT/harmbench/olmo2-1b-base/metrics_summary.csv \
        $OUTPUT/harmbench/olmo2-1b-sft/metrics_summary.csv \
        $OUTPUT/harmbench/olmo2-1b-dpo/metrics_summary.csv \
        $OUTPUT/harmbench/olmo2-1b-rlvr1/metrics_summary.csv \
        $OUTPUT/harmbench/olmo2-1b-instruct/metrics_summary.csv \
    --category-metrics-csv \
        $OUTPUT/harmbench/olmo2-1b-base/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/olmo2-1b-sft/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/olmo2-1b-dpo/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/olmo2-1b-rlvr1/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/olmo2-1b-instruct/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/ablations/olmo2_training \
    --title "HarmBench — OLMo 2 1B Training Stage Ablation"

python scripts/plot_results.py \
    --metrics-csv \
        $OUTPUT/jailbreakbench/olmo2-1b-base/metrics_summary.csv \
        $OUTPUT/jailbreakbench/olmo2-1b-sft/metrics_summary.csv \
        $OUTPUT/jailbreakbench/olmo2-1b-dpo/metrics_summary.csv \
        $OUTPUT/jailbreakbench/olmo2-1b-rlvr1/metrics_summary.csv \
        $OUTPUT/jailbreakbench/olmo2-1b-instruct/metrics_summary.csv \
    --category-metrics-csv \
        $OUTPUT/jailbreakbench/olmo2-1b-base/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/olmo2-1b-sft/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/olmo2-1b-dpo/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/olmo2-1b-rlvr1/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/olmo2-1b-instruct/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/ablations/olmo2_training \
    --title "JailbreakBench — OLMo 2 1B Training Stage Ablation"

# --- Ablation 3: Safety alignment — Qwen3-4B base vs Qwen3-4B-SafeRL ---
# Paper: Table 1 (Qwen3 rows)

python scripts/plot_results.py \
    --metrics-csv \
        $OUTPUT/harmbench/qwen3-4b/metrics_summary.csv \
        $OUTPUT/harmbench/qwen3-4b-saferl/metrics_summary.csv \
    --category-metrics-csv \
        $OUTPUT/harmbench/qwen3-4b/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/qwen3-4b-saferl/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/ablations/safety_alignment \
    --title "HarmBench — Safety Alignment (Qwen3-4B)"

python scripts/plot_results.py \
    --metrics-csv \
        $OUTPUT/jailbreakbench/qwen3-4b/metrics_summary.csv \
        $OUTPUT/jailbreakbench/qwen3-4b-saferl/metrics_summary.csv \
    --category-metrics-csv \
        $OUTPUT/jailbreakbench/qwen3-4b/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/qwen3-4b-saferl/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/ablations/safety_alignment \
    --title "JailbreakBench — Safety Alignment (Qwen3-4B)"

# --- Ablation 4: Best per family ---

BEST_HB_QWEN25=$(python scripts/select_best_model.py --metrics-dir $OUTPUT/harmbench \
    --models qwen2.5-0.5b-instruct qwen2.5-3b-instruct qwen2.5-7b-instruct)
BEST_HB_TULU3=$(python scripts/select_best_model.py --metrics-dir $OUTPUT/harmbench \
    --models tulu3-8b-base tulu3-8b-sft tulu3-8b-dpo tulu3-8b-rlvr)
BEST_JB_QWEN25=$(python scripts/select_best_model.py --metrics-dir $OUTPUT/jailbreakbench \
    --models qwen2.5-0.5b-instruct qwen2.5-3b-instruct qwen2.5-7b-instruct)
BEST_JB_TULU3=$(python scripts/select_best_model.py --metrics-dir $OUTPUT/jailbreakbench \
    --models tulu3-8b-base tulu3-8b-sft tulu3-8b-dpo tulu3-8b-rlvr)

python scripts/plot_results.py \
    --metrics-csv \
        $OUTPUT/harmbench/$BEST_HB_QWEN25/metrics_summary.csv \
        $OUTPUT/harmbench/$BEST_HB_TULU3/metrics_summary.csv \
        $OUTPUT/harmbench/qwen3-4b/metrics_summary.csv \
        $OUTPUT/harmbench/qwen3-4b-saferl/metrics_summary.csv \
    --category-metrics-csv \
        $OUTPUT/harmbench/$BEST_HB_QWEN25/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/$BEST_HB_TULU3/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/qwen3-4b/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/qwen3-4b-saferl/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/ablations/best_per_family \
    --title "HarmBench — Best per Family"

python scripts/plot_results.py \
    --metrics-csv \
        $OUTPUT/jailbreakbench/$BEST_JB_QWEN25/metrics_summary.csv \
        $OUTPUT/jailbreakbench/$BEST_JB_TULU3/metrics_summary.csv \
        $OUTPUT/jailbreakbench/qwen3-4b/metrics_summary.csv \
        $OUTPUT/jailbreakbench/qwen3-4b-saferl/metrics_summary.csv \
    --category-metrics-csv \
        $OUTPUT/jailbreakbench/$BEST_JB_QWEN25/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/$BEST_JB_TULU3/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/qwen3-4b/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/qwen3-4b-saferl/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/ablations/best_per_family \
    --title "JailbreakBench — Best per Family"

# --- Ablation 5: Attack transfer (GCG surrogate: qwen2.5-0.5b) ---

python scripts/plot_results.py \
    --metrics-csv \
        $OUTPUT/harmbench/qwen2.5-0.5b-instruct/metrics_summary.csv \
        $OUTPUT/harmbench/qwen3-4b/metrics_summary.csv \
    --category-metrics-csv \
        $OUTPUT/harmbench/qwen2.5-0.5b-instruct/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/qwen3-4b/metrics_by_category_summary.csv \
    --attacks gcg \
    --output-dir $OUTPUT/harmbench/ablations/attack_transfer_gcg \
    --title "HarmBench — Attack Transfer (GCG)"

python scripts/plot_results.py \
    --metrics-csv \
        $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/metrics_summary.csv \
        $OUTPUT/jailbreakbench/qwen3-4b/metrics_summary.csv \
    --category-metrics-csv \
        $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/qwen3-4b/metrics_by_category_summary.csv \
    --attacks gcg \
    --output-dir $OUTPUT/jailbreakbench/ablations/attack_transfer_gcg \
    --title "JailbreakBench — Attack Transfer (GCG)"

echo "Done."
