#!/bin/bash
# run_cost_plots.sh — Cost-axis (tokens / TFLOPs) risk curves for all experiments.
#
# Follows the same structure as run_plots.sh:
#   1. Per-model plots   — seed traces + seed-aggregated mean/CI (per model)
#   2. Comparison plots  — ablation sets: Qwen2.5 size, Tulu2/3 training phases,
#                          best-per-family, attack transfer (GCG only)
#
# All output goes to:
#   $OUTPUT/{harmbench,jailbreakbench}/$model/{tokens,flops}/  (per-model)
#   $OUTPUT/{harmbench,jailbreakbench}/ablations/<name>/{tokens,flops}/
#
# Requires: must be run from the project root.
# Models without cost_metrics.csv are silently skipped (--skip-missing).
#
# Usage:
#   bash run_cost_plots.sh

set -e

source setup/start_env.sh

BASE=/home/ehghaghi/projects/aip-craffel/ehghaghi/prisk-pressure/results/multitrial_experiments/pressure_sensitivity
PLOTS_BASE=/home/ehghaghi/projects/aip-craffel/ehghaghi/prisk-pressure/results/plots/multitrial_experiments/pressure_sensitivity
OUTPUT=/home/ehghaghi/projects/aip-craffel/ehghaghi/prisk-pressure/results/plots/cost

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
# Per-model plots (seed traces + seed-aggregated CI)
# --------------------------------------------------------------------------- #

echo "=== Per-model cost plots: HarmBench ==="
for model in "${HB_MODELS[@]}"; do
    COST_CSV=$BASE/harmbench/$model/cost/cost_metrics.csv
    if [ ! -f "$COST_CSV" ]; then
        echo "  Skipping $model (no cost data)"
        continue
    fi
    echo "  $model — tokens"
    python plot_cost_curves.py \
        --cost-csv  "$COST_CSV" \
        --output-dir "$OUTPUT/harmbench/$model/tokens" \
        --x-axis tokens

    echo "  $model — flops"
    python plot_cost_curves.py \
        --cost-csv  "$COST_CSV" \
        --output-dir "$OUTPUT/harmbench/$model/flops" \
        --x-axis flops
done

echo "=== Per-model cost plots: JailbreakBench ==="
for model in "${JB_MODELS[@]}"; do
    COST_CSV=$BASE/jailbreakbench/$model/cost/cost_metrics.csv
    if [ ! -f "$COST_CSV" ]; then
        echo "  Skipping $model (no cost data)"
        continue
    fi
    echo "  $model — tokens"
    python plot_cost_curves.py \
        --cost-csv  "$COST_CSV" \
        --output-dir "$OUTPUT/jailbreakbench/$model/tokens" \
        --x-axis tokens

    echo "  $model — flops"
    python plot_cost_curves.py \
        --cost-csv  "$COST_CSV" \
        --output-dir "$OUTPUT/jailbreakbench/$model/flops" \
        --x-axis flops
done

# --------------------------------------------------------------------------- #
# Best-per-family selection (lowest mean AURC, same criterion as run_plots.sh)
# --------------------------------------------------------------------------- #

echo "=== Selecting best models per family ==="

BEST_HB_QWEN25=$(python scripts/select_best_model.py \
    --metrics-dir "$PLOTS_BASE/harmbench" \
    --models qwen2.5-0.5b-instruct qwen2.5-3b-instruct qwen2.5-7b-instruct)
BEST_HB_TULU3=$(python scripts/select_best_model.py \
    --metrics-dir "$PLOTS_BASE/harmbench" \
    --models tulu3-8b-base tulu3-8b-sft tulu3-8b-dpo tulu3-8b-rlvr)
BEST_HB_TULU2=$(python scripts/select_best_model.py \
    --metrics-dir "$PLOTS_BASE/harmbench" \
    --models tulu2-7b-base tulu2-7b-sft tulu2-7b-dpo)

BEST_JB_QWEN25=$(python scripts/select_best_model.py \
    --metrics-dir "$PLOTS_BASE/jailbreakbench" \
    --models qwen2.5-0.5b-instruct qwen2.5-3b-instruct qwen2.5-7b-instruct)
BEST_JB_TULU3=$(python scripts/select_best_model.py \
    --metrics-dir "$PLOTS_BASE/jailbreakbench" \
    --models tulu3-8b-base tulu3-8b-sft tulu3-8b-dpo tulu3-8b-rlvr)
BEST_JB_TULU2=$(python scripts/select_best_model.py \
    --metrics-dir "$PLOTS_BASE/jailbreakbench" \
    --models tulu2-7b-base tulu2-7b-sft tulu2-7b-dpo)

echo "  HB best Qwen2.5: $BEST_HB_QWEN25"
echo "  HB best Tulu3:   $BEST_HB_TULU3"
echo "  HB best Tulu2:   $BEST_HB_TULU2"
echo "  JB best Qwen2.5: $BEST_JB_QWEN25"
echo "  JB best Tulu3:   $BEST_JB_TULU3"
echo "  JB best Tulu2:   $BEST_JB_TULU2"

# --------------------------------------------------------------------------- #
# Ablation 1: Qwen2.5 model size
# --------------------------------------------------------------------------- #

echo "=== Ablation: Qwen2.5 size — HarmBench (tokens) ==="
python plot_cost_curves.py \
    --cost-csv \
        "$BASE/harmbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv" \
        "$BASE/harmbench/qwen2.5-3b-instruct/cost/cost_metrics.csv" \
        "$BASE/harmbench/qwen2.5-7b-instruct/cost/cost_metrics.csv" \
    --output-dir "$OUTPUT/harmbench/ablations/qwen_size/tokens" \
    --x-axis tokens \
    --title "HarmBench — Qwen2.5 Size Ablation" \
    --mode comparison \
    --skip-missing

echo "=== Ablation: Qwen2.5 size — HarmBench (flops) ==="
python plot_cost_curves.py \
    --cost-csv \
        "$BASE/harmbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv" \
        "$BASE/harmbench/qwen2.5-3b-instruct/cost/cost_metrics.csv" \
        "$BASE/harmbench/qwen2.5-7b-instruct/cost/cost_metrics.csv" \
    --output-dir "$OUTPUT/harmbench/ablations/qwen_size/flops" \
    --x-axis flops \
    --title "HarmBench — Qwen2.5 Size Ablation" \
    --mode comparison \
    --skip-missing

echo "=== Ablation: Qwen2.5 size — JailbreakBench (tokens) ==="
python plot_cost_curves.py \
    --cost-csv \
        "$BASE/jailbreakbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv" \
        "$BASE/jailbreakbench/qwen2.5-3b-instruct/cost/cost_metrics.csv" \
        "$BASE/jailbreakbench/qwen2.5-7b-instruct/cost/cost_metrics.csv" \
    --output-dir "$OUTPUT/jailbreakbench/ablations/qwen_size/tokens" \
    --x-axis tokens \
    --title "JailbreakBench — Qwen2.5 Size Ablation" \
    --mode comparison \
    --skip-missing

echo "=== Ablation: Qwen2.5 size — JailbreakBench (flops) ==="
python plot_cost_curves.py \
    --cost-csv \
        "$BASE/jailbreakbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv" \
        "$BASE/jailbreakbench/qwen2.5-3b-instruct/cost/cost_metrics.csv" \
        "$BASE/jailbreakbench/qwen2.5-7b-instruct/cost/cost_metrics.csv" \
    --output-dir "$OUTPUT/jailbreakbench/ablations/qwen_size/flops" \
    --x-axis flops \
    --title "JailbreakBench — Qwen2.5 Size Ablation" \
    --mode comparison \
    --skip-missing

# --------------------------------------------------------------------------- #
# Ablation 2: Tulu3 training phases (base → sft → dpo → rlvr)
# --------------------------------------------------------------------------- #

echo "=== Ablation: Tulu3 training — HarmBench (tokens) ==="
python plot_cost_curves.py \
    --cost-csv \
        "$BASE/harmbench/tulu3-8b-base/cost/cost_metrics.csv" \
        "$BASE/harmbench/tulu3-8b-sft/cost/cost_metrics.csv" \
        "$BASE/harmbench/tulu3-8b-dpo/cost/cost_metrics.csv" \
        "$BASE/harmbench/tulu3-8b-rlvr/cost/cost_metrics.csv" \
    --output-dir "$OUTPUT/harmbench/ablations/tulu3_training/tokens" \
    --x-axis tokens \
    --title "HarmBench — Tulu3 Training Phase Ablation" \
    --mode comparison \
    --skip-missing

echo "=== Ablation: Tulu3 training — HarmBench (flops) ==="
python plot_cost_curves.py \
    --cost-csv \
        "$BASE/harmbench/tulu3-8b-base/cost/cost_metrics.csv" \
        "$BASE/harmbench/tulu3-8b-sft/cost/cost_metrics.csv" \
        "$BASE/harmbench/tulu3-8b-dpo/cost/cost_metrics.csv" \
        "$BASE/harmbench/tulu3-8b-rlvr/cost/cost_metrics.csv" \
    --output-dir "$OUTPUT/harmbench/ablations/tulu3_training/flops" \
    --x-axis flops \
    --title "HarmBench — Tulu3 Training Phase Ablation" \
    --mode comparison \
    --skip-missing

echo "=== Ablation: Tulu3 training — JailbreakBench (tokens) ==="
python plot_cost_curves.py \
    --cost-csv \
        "$BASE/jailbreakbench/tulu3-8b-base/cost/cost_metrics.csv" \
        "$BASE/jailbreakbench/tulu3-8b-sft/cost/cost_metrics.csv" \
        "$BASE/jailbreakbench/tulu3-8b-dpo/cost/cost_metrics.csv" \
        "$BASE/jailbreakbench/tulu3-8b-rlvr/cost/cost_metrics.csv" \
    --output-dir "$OUTPUT/jailbreakbench/ablations/tulu3_training/tokens" \
    --x-axis tokens \
    --title "JailbreakBench — Tulu3 Training Phase Ablation" \
    --mode comparison \
    --skip-missing

echo "=== Ablation: Tulu3 training — JailbreakBench (flops) ==="
python plot_cost_curves.py \
    --cost-csv \
        "$BASE/jailbreakbench/tulu3-8b-base/cost/cost_metrics.csv" \
        "$BASE/jailbreakbench/tulu3-8b-sft/cost/cost_metrics.csv" \
        "$BASE/jailbreakbench/tulu3-8b-dpo/cost/cost_metrics.csv" \
        "$BASE/jailbreakbench/tulu3-8b-rlvr/cost/cost_metrics.csv" \
    --output-dir "$OUTPUT/jailbreakbench/ablations/tulu3_training/flops" \
    --x-axis flops \
    --title "JailbreakBench — Tulu3 Training Phase Ablation" \
    --mode comparison \
    --skip-missing

# --------------------------------------------------------------------------- #
# Ablation 3: Tulu2 training phases (base → sft → dpo)
# --------------------------------------------------------------------------- #

echo "=== Ablation: Tulu2 training — HarmBench (tokens) ==="
python plot_cost_curves.py \
    --cost-csv \
        "$BASE/harmbench/tulu2-7b-base/cost/cost_metrics.csv" \
        "$BASE/harmbench/tulu2-7b-sft/cost/cost_metrics.csv" \
        "$BASE/harmbench/tulu2-7b-dpo/cost/cost_metrics.csv" \
    --output-dir "$OUTPUT/harmbench/ablations/tulu2_training/tokens" \
    --x-axis tokens \
    --title "HarmBench — Tulu2 Training Phase Ablation" \
    --mode comparison \
    --skip-missing

echo "=== Ablation: Tulu2 training — HarmBench (flops) ==="
python plot_cost_curves.py \
    --cost-csv \
        "$BASE/harmbench/tulu2-7b-base/cost/cost_metrics.csv" \
        "$BASE/harmbench/tulu2-7b-sft/cost/cost_metrics.csv" \
        "$BASE/harmbench/tulu2-7b-dpo/cost/cost_metrics.csv" \
    --output-dir "$OUTPUT/harmbench/ablations/tulu2_training/flops" \
    --x-axis flops \
    --title "HarmBench — Tulu2 Training Phase Ablation" \
    --mode comparison \
    --skip-missing

echo "=== Ablation: Tulu2 training — JailbreakBench (tokens) ==="
python plot_cost_curves.py \
    --cost-csv \
        "$BASE/jailbreakbench/tulu2-7b-base/cost/cost_metrics.csv" \
        "$BASE/jailbreakbench/tulu2-7b-sft/cost/cost_metrics.csv" \
        "$BASE/jailbreakbench/tulu2-7b-dpo/cost/cost_metrics.csv" \
    --output-dir "$OUTPUT/jailbreakbench/ablations/tulu2_training/tokens" \
    --x-axis tokens \
    --title "JailbreakBench — Tulu2 Training Phase Ablation" \
    --mode comparison \
    --skip-missing

echo "=== Ablation: Tulu2 training — JailbreakBench (flops) ==="
python plot_cost_curves.py \
    --cost-csv \
        "$BASE/jailbreakbench/tulu2-7b-base/cost/cost_metrics.csv" \
        "$BASE/jailbreakbench/tulu2-7b-sft/cost/cost_metrics.csv" \
        "$BASE/jailbreakbench/tulu2-7b-dpo/cost/cost_metrics.csv" \
    --output-dir "$OUTPUT/jailbreakbench/ablations/tulu2_training/flops" \
    --x-axis flops \
    --title "JailbreakBench — Tulu2 Training Phase Ablation" \
    --mode comparison \
    --skip-missing

# --------------------------------------------------------------------------- #
# Ablation 4: Best per family
# --------------------------------------------------------------------------- #

echo "=== Ablation: Best per family — HarmBench (tokens) ==="
python plot_cost_curves.py \
    --cost-csv \
        "$BASE/harmbench/$BEST_HB_QWEN25/cost/cost_metrics.csv" \
        "$BASE/harmbench/$BEST_HB_TULU3/cost/cost_metrics.csv" \
        "$BASE/harmbench/$BEST_HB_TULU2/cost/cost_metrics.csv" \
        "$BASE/harmbench/qwen3-4b-saferl/cost/cost_metrics.csv" \
    --output-dir "$OUTPUT/harmbench/ablations/best_per_family/tokens" \
    --x-axis tokens \
    --title "HarmBench — Best per Family" \
    --mode comparison \
    --skip-missing

echo "=== Ablation: Best per family — HarmBench (flops) ==="
python plot_cost_curves.py \
    --cost-csv \
        "$BASE/harmbench/$BEST_HB_QWEN25/cost/cost_metrics.csv" \
        "$BASE/harmbench/$BEST_HB_TULU3/cost/cost_metrics.csv" \
        "$BASE/harmbench/$BEST_HB_TULU2/cost/cost_metrics.csv" \
        "$BASE/harmbench/qwen3-4b-saferl/cost/cost_metrics.csv" \
    --output-dir "$OUTPUT/harmbench/ablations/best_per_family/flops" \
    --x-axis flops \
    --title "HarmBench — Best per Family" \
    --mode comparison \
    --skip-missing

echo "=== Ablation: Best per family — JailbreakBench (tokens) ==="
python plot_cost_curves.py \
    --cost-csv \
        "$BASE/jailbreakbench/$BEST_JB_QWEN25/cost/cost_metrics.csv" \
        "$BASE/jailbreakbench/$BEST_JB_TULU3/cost/cost_metrics.csv" \
        "$BASE/jailbreakbench/$BEST_JB_TULU2/cost/cost_metrics.csv" \
        "$BASE/jailbreakbench/qwen3-4b-saferl/cost/cost_metrics.csv" \
    --output-dir "$OUTPUT/jailbreakbench/ablations/best_per_family/tokens" \
    --x-axis tokens \
    --title "JailbreakBench — Best per Family" \
    --mode comparison \
    --skip-missing

echo "=== Ablation: Best per family — JailbreakBench (flops) ==="
python plot_cost_curves.py \
    --cost-csv \
        "$BASE/jailbreakbench/$BEST_JB_QWEN25/cost/cost_metrics.csv" \
        "$BASE/jailbreakbench/$BEST_JB_TULU3/cost/cost_metrics.csv" \
        "$BASE/jailbreakbench/$BEST_JB_TULU2/cost/cost_metrics.csv" \
        "$BASE/jailbreakbench/qwen3-4b-saferl/cost/cost_metrics.csv" \
    --output-dir "$OUTPUT/jailbreakbench/ablations/best_per_family/flops" \
    --x-axis flops \
    --title "JailbreakBench — Best per Family" \
    --mode comparison \
    --skip-missing

# --------------------------------------------------------------------------- #
# Ablation 5: Attack transfer (GCG only) — qwen3-8b vs tulu3-8b-dpo
# --------------------------------------------------------------------------- #

echo "=== Ablation: Attack transfer GCG — HarmBench (tokens) ==="
python plot_cost_curves.py \
    --cost-csv \
        "$BASE/harmbench/qwen3-8b/cost/cost_metrics.csv" \
        "$BASE/harmbench/tulu3-8b-dpo/cost/cost_metrics.csv" \
    --output-dir "$OUTPUT/harmbench/ablations/attack_transfer_gcg/tokens" \
    --x-axis tokens \
    --attacks gcg \
    --title "HarmBench — Attack Transfer (GCG)" \
    --mode comparison \
    --skip-missing

echo "=== Ablation: Attack transfer GCG — HarmBench (flops) ==="
python plot_cost_curves.py \
    --cost-csv \
        "$BASE/harmbench/qwen3-8b/cost/cost_metrics.csv" \
        "$BASE/harmbench/tulu3-8b-dpo/cost/cost_metrics.csv" \
    --output-dir "$OUTPUT/harmbench/ablations/attack_transfer_gcg/flops" \
    --x-axis flops \
    --attacks gcg \
    --title "HarmBench — Attack Transfer (GCG)" \
    --mode comparison \
    --skip-missing

echo "=== Ablation: Attack transfer GCG — JailbreakBench (tokens) ==="
python plot_cost_curves.py \
    --cost-csv \
        "$BASE/jailbreakbench/qwen3-8b/cost/cost_metrics.csv" \
        "$BASE/jailbreakbench/tulu3-8b-dpo/cost/cost_metrics.csv" \
    --output-dir "$OUTPUT/jailbreakbench/ablations/attack_transfer_gcg/tokens" \
    --x-axis tokens \
    --attacks gcg \
    --title "JailbreakBench — Attack Transfer (GCG)" \
    --mode comparison \
    --skip-missing

echo "=== Ablation: Attack transfer GCG — JailbreakBench (flops) ==="
python plot_cost_curves.py \
    --cost-csv \
        "$BASE/jailbreakbench/qwen3-8b/cost/cost_metrics.csv" \
        "$BASE/jailbreakbench/tulu3-8b-dpo/cost/cost_metrics.csv" \
    --output-dir "$OUTPUT/jailbreakbench/ablations/attack_transfer_gcg/flops" \
    --x-axis flops \
    --attacks gcg \
    --title "JailbreakBench — Attack Transfer (GCG)" \
    --mode comparison \
    --skip-missing

echo ""
echo "Done. Cost plots written to: $OUTPUT"
