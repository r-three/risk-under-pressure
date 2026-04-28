#!/bin/bash
# Run Phase 3 plotting for all experiments.
#
# Per-model plots: one figure per attack + combined, written to each model's plots/ dir.
# Comparison plots: all models overlaid per attack, written to comparison_plots/.
#
# Usage: bash run_plots.sh
# Requires: must be run from the project root.

set -e

source setup/start_env.sh

BASE=/home/ehghaghi/projects/aip-craffel/ehghaghi/prisk-pressure/results/multitrial_experiments/pressure_sensitivity
OUTPUT=/home/ehghaghi/projects/aip-craffel/ehghaghi/prisk-pressure/results/plots/multitrial_experiments/pressure_sensitivity

HB_MODELS=(
    qwen2.5-0.5b-instruct
    qwen2.5-3b-instruct
    qwen2.5-7b-instruct
    qwen3-4b-saferl
    tulu2-7b-base
    tulu2-7b-sft
    tulu2-7b-dpo
    tulu3-8b-base
    tulu3-8b-sft
    tulu3-8b-dpo
    tulu3-8b-rlvr
)

JB_MODELS=(
    qwen2.5-0.5b-instruct
    qwen2.5-3b-instruct
    qwen2.5-7b-instruct
    qwen3-4b-saferl
    tulu2-7b-base
    tulu2-7b-sft
    tulu2-7b-dpo
    tulu3-8b-base
    tulu3-8b-sft
    tulu3-8b-dpo
    tulu3-8b-rlvr
)

# --------------------------------------------------------------------------- #
# Per-model plots (seed-aggregated CI)
# --------------------------------------------------------------------------- #

echo "=== Per-model plots: HarmBench ==="
for model in "${HB_MODELS[@]}"; do
    echo "  $model"
    python plot_results.py \
        --metrics-csv $OUTPUT/harmbench/$model/metrics.csv \
        --category-metrics-csv $OUTPUT/harmbench/$model/metrics_by_category_summary.csv \
        --output-dir $OUTPUT/harmbench/$model/plots/seeds \
        --ci-method seeds
done

echo "=== Per-model plots: JailbreakBench ==="
for model in "${JB_MODELS[@]}"; do
    echo "  $model"
    python plot_results.py \
        --metrics-csv $OUTPUT/jailbreakbench/$model/metrics.csv \
        --category-metrics-csv $OUTPUT/jailbreakbench/$model/metrics_by_category_summary.csv \
        --output-dir $OUTPUT/jailbreakbench/$model/plots/seeds \
        --ci-method seeds
done

# --------------------------------------------------------------------------- #
# Cross-model comparison plots (seed-aggregated summary CSVs)
# --------------------------------------------------------------------------- #

echo "=== Comparison plots: HarmBench ==="
python plot_results.py \
    --metrics-csv \
        $OUTPUT/harmbench/qwen2.5-0.5b-instruct/metrics_summary.csv \
        $OUTPUT/harmbench/qwen2.5-3b-instruct/metrics_summary.csv \
        $OUTPUT/harmbench/qwen2.5-7b-instruct/metrics_summary.csv \
        $OUTPUT/harmbench/qwen3-4b-saferl/metrics_summary.csv \
        $OUTPUT/harmbench/tulu2-7b-base/metrics_summary.csv \
        $OUTPUT/harmbench/tulu2-7b-sft/metrics_summary.csv \
        $OUTPUT/harmbench/tulu2-7b-dpo/metrics_summary.csv \
        $OUTPUT/harmbench/tulu3-8b-base/metrics_summary.csv \
        $OUTPUT/harmbench/tulu3-8b-sft/metrics_summary.csv \
        $OUTPUT/harmbench/tulu3-8b-dpo/metrics_summary.csv \
        $OUTPUT/harmbench/tulu3-8b-rlvr/metrics_summary.csv \
    --category-metrics-csv \
        $OUTPUT/harmbench/qwen2.5-0.5b-instruct/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/qwen2.5-3b-instruct/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/qwen2.5-7b-instruct/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/qwen3-4b-saferl/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/tulu2-7b-base/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/tulu2-7b-sft/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/tulu2-7b-dpo/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/tulu3-8b-base/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/tulu3-8b-sft/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/tulu3-8b-dpo/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/tulu3-8b-rlvr/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/comparison_plots \
    --title "HarmBench — All Models"

echo "=== Comparison plots: JailbreakBench ==="
python plot_results.py \
    --metrics-csv \
        $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/metrics_summary.csv \
        $OUTPUT/jailbreakbench/qwen2.5-3b-instruct/metrics_summary.csv \
        $OUTPUT/jailbreakbench/qwen2.5-7b-instruct/metrics_summary.csv \
        $OUTPUT/jailbreakbench/qwen3-4b-saferl/metrics_summary.csv \
        $OUTPUT/jailbreakbench/tulu2-7b-base/metrics_summary.csv \
        $OUTPUT/jailbreakbench/tulu2-7b-sft/metrics_summary.csv \
        $OUTPUT/jailbreakbench/tulu2-7b-dpo/metrics_summary.csv \
        $OUTPUT/jailbreakbench/tulu3-8b-base/metrics_summary.csv \
        $OUTPUT/jailbreakbench/tulu3-8b-sft/metrics_summary.csv \
        $OUTPUT/jailbreakbench/tulu3-8b-dpo/metrics_summary.csv \
        $OUTPUT/jailbreakbench/tulu3-8b-rlvr/metrics_summary.csv \
    --category-metrics-csv \
        $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/qwen2.5-3b-instruct/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/qwen2.5-7b-instruct/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/qwen3-4b-saferl/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/tulu2-7b-base/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/tulu2-7b-sft/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/tulu2-7b-dpo/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/tulu3-8b-base/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/tulu3-8b-sft/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/tulu3-8b-dpo/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/tulu3-8b-rlvr/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/comparison_plots \
    --title "JailbreakBench — All Models"

# --------------------------------------------------------------------------- #
# Ablation plots
# --------------------------------------------------------------------------- #

# Best-per-family selection (lowest mean AURC)
echo "=== Selecting best models per family ==="
BEST_HB_QWEN25=$(python scripts/select_best_model.py --metrics-dir $OUTPUT/harmbench \
    --models qwen2.5-0.5b-instruct qwen2.5-3b-instruct qwen2.5-7b-instruct)
BEST_HB_TULU3=$(python scripts/select_best_model.py --metrics-dir $OUTPUT/harmbench \
    --models tulu3-8b-base tulu3-8b-sft tulu3-8b-dpo tulu3-8b-rlvr)
BEST_HB_TULU2=$(python scripts/select_best_model.py --metrics-dir $OUTPUT/harmbench \
    --models tulu2-7b-base tulu2-7b-sft tulu2-7b-dpo)

BEST_JB_QWEN25=$(python scripts/select_best_model.py --metrics-dir $OUTPUT/jailbreakbench \
    --models qwen2.5-0.5b-instruct qwen2.5-3b-instruct qwen2.5-7b-instruct)
BEST_JB_TULU3=$(python scripts/select_best_model.py --metrics-dir $OUTPUT/jailbreakbench \
    --models tulu3-8b-base tulu3-8b-sft tulu3-8b-dpo tulu3-8b-rlvr)
BEST_JB_TULU2=$(python scripts/select_best_model.py --metrics-dir $OUTPUT/jailbreakbench \
    --models tulu2-7b-base tulu2-7b-sft tulu2-7b-dpo)

echo "  HB best Qwen2.5: $BEST_HB_QWEN25"
echo "  HB best Tulu3:   $BEST_HB_TULU3"
echo "  HB best Tulu2:   $BEST_HB_TULU2"
echo "  JB best Qwen2.5: $BEST_JB_QWEN25"
echo "  JB best Tulu3:   $BEST_JB_TULU3"
echo "  JB best Tulu2:   $BEST_JB_TULU2"

# --- Ablation 1: Qwen2.5 model size ---

echo "=== Ablation: Qwen2.5 size — HarmBench ==="
python plot_results.py \
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

echo "=== Ablation: Qwen2.5 size — JailbreakBench ==="
python plot_results.py \
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

# --- Ablation 2: Tulu3 training phases ---

echo "=== Ablation: Tulu3 training — HarmBench ==="
python plot_results.py \
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

echo "=== Ablation: Tulu3 training — JailbreakBench ==="
python plot_results.py \
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

# --- Ablation 3: Tulu2 training phases ---

echo "=== Ablation: Tulu2 training — HarmBench ==="
python plot_results.py \
    --metrics-csv \
        $OUTPUT/harmbench/tulu2-7b-base/metrics_summary.csv \
        $OUTPUT/harmbench/tulu2-7b-sft/metrics_summary.csv \
        $OUTPUT/harmbench/tulu2-7b-dpo/metrics_summary.csv \
    --category-metrics-csv \
        $OUTPUT/harmbench/tulu2-7b-base/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/tulu2-7b-sft/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/tulu2-7b-dpo/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/ablations/tulu2_training \
    --title "HarmBench — Tulu2 Training Phase Ablation"

echo "=== Ablation: Tulu2 training — JailbreakBench ==="
python plot_results.py \
    --metrics-csv \
        $OUTPUT/jailbreakbench/tulu2-7b-base/metrics_summary.csv \
        $OUTPUT/jailbreakbench/tulu2-7b-sft/metrics_summary.csv \
        $OUTPUT/jailbreakbench/tulu2-7b-dpo/metrics_summary.csv \
    --category-metrics-csv \
        $OUTPUT/jailbreakbench/tulu2-7b-base/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/tulu2-7b-sft/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/tulu2-7b-dpo/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/ablations/tulu2_training \
    --title "JailbreakBench — Tulu2 Training Phase Ablation"

# --- Ablation 4: Best per family ---

echo "=== Ablation: Best per family — HarmBench ==="
python plot_results.py \
    --metrics-csv \
        $OUTPUT/harmbench/$BEST_HB_QWEN25/metrics_summary.csv \
        $OUTPUT/harmbench/$BEST_HB_TULU3/metrics_summary.csv \
        $OUTPUT/harmbench/$BEST_HB_TULU2/metrics_summary.csv \
        $OUTPUT/harmbench/qwen3-4b-saferl/metrics_summary.csv \
    --category-metrics-csv \
        $OUTPUT/harmbench/$BEST_HB_QWEN25/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/$BEST_HB_TULU3/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/$BEST_HB_TULU2/metrics_by_category_summary.csv \
        $OUTPUT/harmbench/qwen3-4b-saferl/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/harmbench/ablations/best_per_family \
    --title "HarmBench — Best per Family"

echo "=== Ablation: Best per family — JailbreakBench ==="
python plot_results.py \
    --metrics-csv \
        $OUTPUT/jailbreakbench/$BEST_JB_QWEN25/metrics_summary.csv \
        $OUTPUT/jailbreakbench/$BEST_JB_TULU3/metrics_summary.csv \
        $OUTPUT/jailbreakbench/$BEST_JB_TULU2/metrics_summary.csv \
        $OUTPUT/jailbreakbench/qwen3-4b-saferl/metrics_summary.csv \
    --category-metrics-csv \
        $OUTPUT/jailbreakbench/$BEST_JB_QWEN25/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/$BEST_JB_TULU3/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/$BEST_JB_TULU2/metrics_by_category_summary.csv \
        $OUTPUT/jailbreakbench/qwen3-4b-saferl/metrics_by_category_summary.csv \
    --output-dir $OUTPUT/jailbreakbench/ablations/best_per_family \
    --title "JailbreakBench — Best per Family"

# --- Per-model plots: qwen3-8b ---

echo "=== Per-model plots: qwen3-8b — HarmBench ==="
python plot_results.py \
    --metrics-csv $OUTPUT/harmbench/qwen3-8b/metrics.csv \
    --category-metrics-csv $OUTPUT/harmbench/qwen3-8b/metrics_by_category.csv \
    --output-dir $OUTPUT/harmbench/qwen3-8b/plots

echo "=== Per-model plots: qwen3-8b — JailbreakBench ==="
python plot_results.py \
    --metrics-csv $OUTPUT/jailbreakbench/qwen3-8b/metrics.csv \
    --category-metrics-csv $OUTPUT/jailbreakbench/qwen3-8b/metrics_by_category.csv \
    --output-dir $OUTPUT/jailbreakbench/qwen3-8b/plots

# --- Ablation 5: Attack transfer (GCG only) — qwen3-8b vs tulu3-8b-dpo ---

echo "=== Ablation: Attack transfer GCG — HarmBench ==="
python plot_results.py \
    --metrics-csv \
        $OUTPUT/harmbench/qwen3-8b/metrics.csv \
        $OUTPUT/harmbench/tulu3-8b-dpo/metrics_summary.csv \
    --category-metrics-csv \
        $OUTPUT/harmbench/qwen3-8b/metrics_by_category.csv \
        $OUTPUT/harmbench/tulu3-8b-dpo/metrics_by_category_summary.csv \
    --attacks gcg \
    --output-dir $OUTPUT/harmbench/ablations/attack_transfer_gcg \
    --title "HarmBench — Attack Transfer (GCG)"

echo "=== Ablation: Attack transfer GCG — JailbreakBench ==="
python plot_results.py \
    --metrics-csv \
        $OUTPUT/jailbreakbench/qwen3-8b/metrics.csv \
        $OUTPUT/jailbreakbench/tulu3-8b-dpo/metrics_summary.csv \
    --category-metrics-csv \
        $OUTPUT/jailbreakbench/qwen3-8b/metrics_by_category.csv \
        $OUTPUT/jailbreakbench/tulu3-8b-dpo/metrics_by_category_summary.csv \
    --attacks gcg \
    --output-dir $OUTPUT/jailbreakbench/ablations/attack_transfer_gcg \
    --title "JailbreakBench — Attack Transfer (GCG)"

echo "Done."
