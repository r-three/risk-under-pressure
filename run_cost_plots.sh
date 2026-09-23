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
#
# JUDGE selects the safety judge (a config name under configs/models/, without .yaml).
# It defaults to llama3.1_8b_instruct_judge, which keeps every path and job name
# exactly as it was before judges became selectable. Any other judge writes to its
# own tree under $SCRATCH/rup/judges/<judge_model_id>/ so runs never overwrite:
#   JUDGE=olmo3_7b_instruct_judge     bash run_cost_plots.sh
#   JUDGE=gemma3_4b_it_judge          bash run_cost_plots.sh
# See run_judge_ablation.sh to sweep all judges in one go.

set -e

source setup/start_env.sh
source setup/judge_env.sh

OUTPUT=$PLOT_ROOT

COST_PLOT="python scripts/plot_cost_curves.py"

# Cost axes to plot. seconds = measured L40S attack wall-clock (only seeds whose
# results.jsonl carry per-step timing contribute; untimed seeds drop out as NaN).
# dollars = hosted per-token cost, requires run_cost_evaluations.sh to have been
# run with --pricing-config. Override to a subset, e.g. AXES="dollars" bash run_cost_plots.sh
#
# TWO COST FRAMINGS. tokens/flops/dollars charge target + judge + attacker: the cost of
# REPRODUCING this measurement. Their *_nojudge counterparts charge target + attacker only:
# the cost of mounting the ATTACK, since a real adversary reads the response themselves and
# never pays to run an LLM judge over every reply. The judge fires once per step on every
# attack, so on the cheap template attacks it can dominate the total.
#
# Use the *_nojudge axes whenever the comparison spans different judges (run_judge_ablation.sh):
# they are invariant to the judge, whereas part of any gap on the judge-inclusive axes is just
# the judge's own size and rate moving. Both are plotted by default so either can be quoted.
#
# There is no seconds_nojudge — that axis is measured wall-clock with the judge inside the
# timed region, so it cannot be split after the fact.
#
#   AXES="flops flops_nojudge"  bash run_cost_plots.sh   # just the FLOP pair
#   AXES="dollars_nojudge"      bash run_cost_plots.sh   # attacker's bill only
# AXES="${AXES:-tokens flops seconds dollars tokens_nojudge flops_nojudge dollars_nojudge}"
AXES="${AXES:-flops}"
# =============================================================================
# Per-model plots (seed-aggregated CI)
# =============================================================================

# --------------------------------------------------------------------------- #
# HarmBench
# --------------------------------------------------------------------------- #

# --- MODEL SIZE STUDY — Qwen2.5-Instruct: 0.5B, 3B, 7B ---
# Paper: Figure 1 right

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/harmbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/harmbench/qwen2.5-0.5b-instruct/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/qwen2.5-0.5b-instruct/$axis \
        --x-axis $axis --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/harmbench/qwen2.5-3b-instruct/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/harmbench/qwen2.5-3b-instruct/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/qwen2.5-3b-instruct/$axis \
        --x-axis $axis --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/harmbench/qwen2.5-7b-instruct/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/harmbench/qwen2.5-7b-instruct/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/qwen2.5-7b-instruct/$axis \
        --x-axis $axis --skip-missing
done

# --- MODEL SIZE STUDY (2nd family) — Gemma 3 Instruction-Tuned: 270M, 1B, 4B ---
# Companion to the Qwen2.5 ladder above

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/harmbench/gemma3-270m-it/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/harmbench/gemma3-270m-it/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/gemma3-270m-it/$axis \
        --x-axis $axis --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/harmbench/gemma3-1b-it/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/harmbench/gemma3-1b-it/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/gemma3-1b-it/$axis \
        --x-axis $axis --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/harmbench/gemma3-4b-it/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/harmbench/gemma3-4b-it/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/gemma3-4b-it/$axis \
        --x-axis $axis --skip-missing
done

# --- TRAINING STAGE STUDY — Tulu3 8B: Base → SFT → DPO → RLVR ---
# Paper: Table 1, Figure 1 left

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/harmbench/tulu3-8b-base/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/harmbench/tulu3-8b-base/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/tulu3-8b-base/$axis \
        --x-axis $axis --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/harmbench/tulu3-8b-sft/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/harmbench/tulu3-8b-sft/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/tulu3-8b-sft/$axis \
        --x-axis $axis --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/harmbench/tulu3-8b-dpo/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/harmbench/tulu3-8b-dpo/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/tulu3-8b-dpo/$axis \
        --x-axis $axis --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/harmbench/tulu3-8b-rlvr/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/harmbench/tulu3-8b-rlvr/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/tulu3-8b-rlvr/$axis \
        --x-axis $axis --skip-missing
done

# --- TRAINING STAGE STUDY (2nd ladder) — OLMo 2 1B: Base → SFT → DPO → RLVR1 → Instruct ---
# Companion to the Tulu3 8B ladder above

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/harmbench/olmo2-1b-base/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/harmbench/olmo2-1b-base/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/olmo2-1b-base/$axis \
        --x-axis $axis --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/harmbench/olmo2-1b-sft/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/harmbench/olmo2-1b-sft/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/olmo2-1b-sft/$axis \
        --x-axis $axis --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/harmbench/olmo2-1b-dpo/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/harmbench/olmo2-1b-dpo/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/olmo2-1b-dpo/$axis \
        --x-axis $axis --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/harmbench/olmo2-1b-rlvr1/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/harmbench/olmo2-1b-rlvr1/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/olmo2-1b-rlvr1/$axis \
        --x-axis $axis --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/harmbench/olmo2-1b-instruct/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/harmbench/olmo2-1b-instruct/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/olmo2-1b-instruct/$axis \
        --x-axis $axis --skip-missing
done

# --- SAFETY ALIGNMENT STUDY — Qwen3-4B base vs Qwen3-4B-SafeRL ---
# Paper: Table 1 (Qwen3 rows)

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/harmbench/qwen3-4b/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/harmbench/qwen3-4b/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/qwen3-4b/$axis \
        --x-axis $axis --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/harmbench/qwen3-4b-saferl/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/harmbench/qwen3-4b-saferl/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/qwen3-4b-saferl/$axis \
        --x-axis $axis --skip-missing
done

# --------------------------------------------------------------------------- #
# JailbreakBench
# --------------------------------------------------------------------------- #

# --- MODEL SIZE STUDY — Qwen2.5-Instruct: 0.5B, 3B, 7B ---
# Paper: Figure 1 right

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/$axis \
        --x-axis $axis --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/jailbreakbench/qwen2.5-3b-instruct/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/jailbreakbench/qwen2.5-3b-instruct/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/jailbreakbench/qwen2.5-3b-instruct/$axis \
        --x-axis $axis --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/jailbreakbench/qwen2.5-7b-instruct/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/jailbreakbench/qwen2.5-7b-instruct/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/jailbreakbench/qwen2.5-7b-instruct/$axis \
        --x-axis $axis --skip-missing
done

# --- MODEL SIZE STUDY (2nd family) — Gemma 3 Instruction-Tuned: 270M, 1B, 4B ---
# Companion to the Qwen2.5 ladder above

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/jailbreakbench/gemma3-270m-it/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/jailbreakbench/gemma3-270m-it/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/jailbreakbench/gemma3-270m-it/$axis \
        --x-axis $axis --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/jailbreakbench/gemma3-1b-it/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/jailbreakbench/gemma3-1b-it/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/jailbreakbench/gemma3-1b-it/$axis \
        --x-axis $axis --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/jailbreakbench/gemma3-4b-it/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/jailbreakbench/gemma3-4b-it/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/jailbreakbench/gemma3-4b-it/$axis \
        --x-axis $axis --skip-missing
done

# --- TRAINING STAGE STUDY — Tulu3 8B: Base → SFT → DPO → RLVR ---
# Paper: Table 1, Figure 1 left

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/jailbreakbench/tulu3-8b-base/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/jailbreakbench/tulu3-8b-base/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/jailbreakbench/tulu3-8b-base/$axis \
        --x-axis $axis --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/jailbreakbench/tulu3-8b-sft/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/jailbreakbench/tulu3-8b-sft/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/jailbreakbench/tulu3-8b-sft/$axis \
        --x-axis $axis --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/jailbreakbench/tulu3-8b-dpo/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/jailbreakbench/tulu3-8b-dpo/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/jailbreakbench/tulu3-8b-dpo/$axis \
        --x-axis $axis --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/jailbreakbench/tulu3-8b-rlvr/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/jailbreakbench/tulu3-8b-rlvr/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/jailbreakbench/tulu3-8b-rlvr/$axis \
        --x-axis $axis --skip-missing
done

# --- TRAINING STAGE STUDY (2nd ladder) — OLMo 2 1B: Base → SFT → DPO → RLVR1 → Instruct ---
# Companion to the Tulu3 8B ladder above

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/jailbreakbench/olmo2-1b-base/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/jailbreakbench/olmo2-1b-base/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/jailbreakbench/olmo2-1b-base/$axis \
        --x-axis $axis --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/jailbreakbench/olmo2-1b-sft/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/jailbreakbench/olmo2-1b-sft/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/jailbreakbench/olmo2-1b-sft/$axis \
        --x-axis $axis --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/jailbreakbench/olmo2-1b-dpo/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/jailbreakbench/olmo2-1b-dpo/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/jailbreakbench/olmo2-1b-dpo/$axis \
        --x-axis $axis --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/jailbreakbench/olmo2-1b-rlvr1/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/jailbreakbench/olmo2-1b-rlvr1/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/jailbreakbench/olmo2-1b-rlvr1/$axis \
        --x-axis $axis --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/jailbreakbench/olmo2-1b-instruct/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/jailbreakbench/olmo2-1b-instruct/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/jailbreakbench/olmo2-1b-instruct/$axis \
        --x-axis $axis --skip-missing
done

# --- SAFETY ALIGNMENT STUDY — Qwen3-4B base vs Qwen3-4B-SafeRL ---

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/jailbreakbench/qwen3-4b/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/jailbreakbench/qwen3-4b/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/jailbreakbench/qwen3-4b/$axis \
        --x-axis $axis --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv  $OUTPUT/jailbreakbench/qwen3-4b-saferl/cost/cost_metrics.csv \
        --cost-category-csv $OUTPUT/jailbreakbench/qwen3-4b-saferl/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/jailbreakbench/qwen3-4b-saferl/$axis \
        --x-axis $axis --skip-missing
done

# =============================================================================
# Cross-model comparison plots
# =============================================================================

for bench in harmbench jailbreakbench; do
    case $bench in
        harmbench)      bench_title="HarmBench" ;;
        jailbreakbench) bench_title="JailbreakBench" ;;
    esac

    for axis in $AXES; do
        echo "=== Comparison plots: $bench_title ($axis) ==="
        $COST_PLOT \
            --cost-csv \
                $OUTPUT/$bench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv \
                $OUTPUT/$bench/qwen2.5-3b-instruct/cost/cost_metrics.csv \
                $OUTPUT/$bench/qwen2.5-7b-instruct/cost/cost_metrics.csv \
                $OUTPUT/$bench/tulu3-8b-base/cost/cost_metrics.csv \
                $OUTPUT/$bench/tulu3-8b-sft/cost/cost_metrics.csv \
                $OUTPUT/$bench/tulu3-8b-dpo/cost/cost_metrics.csv \
                $OUTPUT/$bench/tulu3-8b-rlvr/cost/cost_metrics.csv \
                $OUTPUT/$bench/qwen3-4b/cost/cost_metrics.csv \
                $OUTPUT/$bench/qwen3-4b-saferl/cost/cost_metrics.csv \
            --output-dir $OUTPUT/$bench/comparison_plots/$axis \
            --x-axis $axis --title "$bench_title — All Models" \
            --mode comparison --skip-missing
    done
done

# =============================================================================
# Ablation plots
# =============================================================================

# --- Ablation 1: Qwen2.5 model size ---
# Paper: Figure 1 right

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv \
            $OUTPUT/harmbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv \
            $OUTPUT/harmbench/qwen2.5-3b-instruct/cost/cost_metrics.csv \
            $OUTPUT/harmbench/qwen2.5-7b-instruct/cost/cost_metrics.csv \
        --cost-category-csv \
            $OUTPUT/harmbench/qwen2.5-0.5b-instruct/cost/cost_metrics_by_category.csv \
            $OUTPUT/harmbench/qwen2.5-3b-instruct/cost/cost_metrics_by_category.csv \
            $OUTPUT/harmbench/qwen2.5-7b-instruct/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/ablations/qwen_size/$axis \
        --x-axis $axis --title "HarmBench — Qwen2.5 Model Size" \
        --mode comparison --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv \
            $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv \
            $OUTPUT/jailbreakbench/qwen2.5-3b-instruct/cost/cost_metrics.csv \
            $OUTPUT/jailbreakbench/qwen2.5-7b-instruct/cost/cost_metrics.csv \
        --cost-category-csv \
            $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/cost/cost_metrics_by_category.csv \
            $OUTPUT/jailbreakbench/qwen2.5-3b-instruct/cost/cost_metrics_by_category.csv \
            $OUTPUT/jailbreakbench/qwen2.5-7b-instruct/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/jailbreakbench/ablations/qwen_size/$axis \
        --x-axis $axis --title "JailbreakBench — Qwen2.5 Model Size" \
        --mode comparison --skip-missing
done

# --- Ablation 1b: Gemma 3 model size (2nd family) ---

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv \
            $OUTPUT/harmbench/gemma3-270m-it/cost/cost_metrics.csv \
            $OUTPUT/harmbench/gemma3-1b-it/cost/cost_metrics.csv \
            $OUTPUT/harmbench/gemma3-4b-it/cost/cost_metrics.csv \
        --cost-category-csv \
            $OUTPUT/harmbench/gemma3-270m-it/cost/cost_metrics_by_category.csv \
            $OUTPUT/harmbench/gemma3-1b-it/cost/cost_metrics_by_category.csv \
            $OUTPUT/harmbench/gemma3-4b-it/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/ablations/gemma_size/$axis \
        --x-axis $axis --title "HarmBench — Gemma 3 Model Size" \
        --mode comparison --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv \
            $OUTPUT/jailbreakbench/gemma3-270m-it/cost/cost_metrics.csv \
            $OUTPUT/jailbreakbench/gemma3-1b-it/cost/cost_metrics.csv \
            $OUTPUT/jailbreakbench/gemma3-4b-it/cost/cost_metrics.csv \
        --cost-category-csv \
            $OUTPUT/jailbreakbench/gemma3-270m-it/cost/cost_metrics_by_category.csv \
            $OUTPUT/jailbreakbench/gemma3-1b-it/cost/cost_metrics_by_category.csv \
            $OUTPUT/jailbreakbench/gemma3-4b-it/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/jailbreakbench/ablations/gemma_size/$axis \
        --x-axis $axis --title "JailbreakBench — Gemma 3 Model Size" \
        --mode comparison --skip-missing
done

# --- Ablation 2: Tulu3 training stages ---
# Paper: Table 1, Figure 1 left

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv \
            $OUTPUT/harmbench/tulu3-8b-base/cost/cost_metrics.csv \
            $OUTPUT/harmbench/tulu3-8b-sft/cost/cost_metrics.csv \
            $OUTPUT/harmbench/tulu3-8b-dpo/cost/cost_metrics.csv \
            $OUTPUT/harmbench/tulu3-8b-rlvr/cost/cost_metrics.csv \
        --cost-category-csv \
            $OUTPUT/harmbench/tulu3-8b-base/cost/cost_metrics_by_category.csv \
            $OUTPUT/harmbench/tulu3-8b-sft/cost/cost_metrics_by_category.csv \
            $OUTPUT/harmbench/tulu3-8b-dpo/cost/cost_metrics_by_category.csv \
            $OUTPUT/harmbench/tulu3-8b-rlvr/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/ablations/tulu3_training/$axis \
        --x-axis $axis --title "HarmBench — Tulu3 Training Stages" \
        --mode comparison --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv \
            $OUTPUT/jailbreakbench/tulu3-8b-base/cost/cost_metrics.csv \
            $OUTPUT/jailbreakbench/tulu3-8b-sft/cost/cost_metrics.csv \
            $OUTPUT/jailbreakbench/tulu3-8b-dpo/cost/cost_metrics.csv \
            $OUTPUT/jailbreakbench/tulu3-8b-rlvr/cost/cost_metrics.csv \
        --cost-category-csv \
            $OUTPUT/jailbreakbench/tulu3-8b-base/cost/cost_metrics_by_category.csv \
            $OUTPUT/jailbreakbench/tulu3-8b-sft/cost/cost_metrics_by_category.csv \
            $OUTPUT/jailbreakbench/tulu3-8b-dpo/cost/cost_metrics_by_category.csv \
            $OUTPUT/jailbreakbench/tulu3-8b-rlvr/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/jailbreakbench/ablations/tulu3_training/$axis \
        --x-axis $axis --title "JailbreakBench — Tulu3 Training Stages" \
        --mode comparison --skip-missing
done

# --- Ablation 2b: OLMo 2 1B training stages (2nd ladder, 5 rungs) ---

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv \
            $OUTPUT/harmbench/olmo2-1b-base/cost/cost_metrics.csv \
            $OUTPUT/harmbench/olmo2-1b-sft/cost/cost_metrics.csv \
            $OUTPUT/harmbench/olmo2-1b-dpo/cost/cost_metrics.csv \
            $OUTPUT/harmbench/olmo2-1b-rlvr1/cost/cost_metrics.csv \
            $OUTPUT/harmbench/olmo2-1b-instruct/cost/cost_metrics.csv \
        --cost-category-csv \
            $OUTPUT/harmbench/olmo2-1b-base/cost/cost_metrics_by_category.csv \
            $OUTPUT/harmbench/olmo2-1b-sft/cost/cost_metrics_by_category.csv \
            $OUTPUT/harmbench/olmo2-1b-dpo/cost/cost_metrics_by_category.csv \
            $OUTPUT/harmbench/olmo2-1b-rlvr1/cost/cost_metrics_by_category.csv \
            $OUTPUT/harmbench/olmo2-1b-instruct/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/ablations/olmo2_training/$axis \
        --x-axis $axis --title "HarmBench — OLMo 2 1B Training Stages" \
        --mode comparison --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv \
            $OUTPUT/jailbreakbench/olmo2-1b-base/cost/cost_metrics.csv \
            $OUTPUT/jailbreakbench/olmo2-1b-sft/cost/cost_metrics.csv \
            $OUTPUT/jailbreakbench/olmo2-1b-dpo/cost/cost_metrics.csv \
            $OUTPUT/jailbreakbench/olmo2-1b-rlvr1/cost/cost_metrics.csv \
            $OUTPUT/jailbreakbench/olmo2-1b-instruct/cost/cost_metrics.csv \
        --cost-category-csv \
            $OUTPUT/jailbreakbench/olmo2-1b-base/cost/cost_metrics_by_category.csv \
            $OUTPUT/jailbreakbench/olmo2-1b-sft/cost/cost_metrics_by_category.csv \
            $OUTPUT/jailbreakbench/olmo2-1b-dpo/cost/cost_metrics_by_category.csv \
            $OUTPUT/jailbreakbench/olmo2-1b-rlvr1/cost/cost_metrics_by_category.csv \
            $OUTPUT/jailbreakbench/olmo2-1b-instruct/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/jailbreakbench/ablations/olmo2_training/$axis \
        --x-axis $axis --title "JailbreakBench — OLMo 2 1B Training Stages" \
        --mode comparison --skip-missing
done

# --- Ablation 3: Safety alignment — Qwen3-4B base vs Qwen3-4B-SafeRL ---
# Paper: Table 1 (Qwen3 rows)

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv \
            $OUTPUT/harmbench/qwen3-4b/cost/cost_metrics.csv \
            $OUTPUT/harmbench/qwen3-4b-saferl/cost/cost_metrics.csv \
        --cost-category-csv \
            $OUTPUT/harmbench/qwen3-4b/cost/cost_metrics_by_category.csv \
            $OUTPUT/harmbench/qwen3-4b-saferl/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/ablations/safety_alignment/$axis \
        --x-axis $axis --title "HarmBench — Safety Alignment (Qwen3-4B)" \
        --mode comparison --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv \
            $OUTPUT/jailbreakbench/qwen3-4b/cost/cost_metrics.csv \
            $OUTPUT/jailbreakbench/qwen3-4b-saferl/cost/cost_metrics.csv \
        --cost-category-csv \
            $OUTPUT/jailbreakbench/qwen3-4b/cost/cost_metrics_by_category.csv \
            $OUTPUT/jailbreakbench/qwen3-4b-saferl/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/jailbreakbench/ablations/safety_alignment/$axis \
        --x-axis $axis --title "JailbreakBench — Safety Alignment (Qwen3-4B)" \
        --mode comparison --skip-missing
done

# --- Ablation 4: Best per family ---

BEST_HB_QWEN25=$(python scripts/select_best_model.py --metrics-dir $OUTPUT/harmbench \
    --models qwen2.5-0.5b-instruct qwen2.5-3b-instruct qwen2.5-7b-instruct)
BEST_HB_TULU3=$(python scripts/select_best_model.py --metrics-dir $OUTPUT/harmbench \
    --models tulu3-8b-base tulu3-8b-sft tulu3-8b-dpo tulu3-8b-rlvr)
BEST_JB_QWEN25=$(python scripts/select_best_model.py --metrics-dir $OUTPUT/jailbreakbench \
    --models qwen2.5-0.5b-instruct qwen2.5-3b-instruct qwen2.5-7b-instruct)
BEST_JB_TULU3=$(python scripts/select_best_model.py --metrics-dir $OUTPUT/jailbreakbench \
    --models tulu3-8b-base tulu3-8b-sft tulu3-8b-dpo tulu3-8b-rlvr)

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv \
            $OUTPUT/harmbench/$BEST_HB_QWEN25/cost/cost_metrics.csv \
            $OUTPUT/harmbench/$BEST_HB_TULU3/cost/cost_metrics.csv \
            $OUTPUT/harmbench/qwen3-4b/cost/cost_metrics.csv \
            $OUTPUT/harmbench/qwen3-4b-saferl/cost/cost_metrics.csv \
        --cost-category-csv \
            $OUTPUT/harmbench/$BEST_HB_QWEN25/cost/cost_metrics_by_category.csv \
            $OUTPUT/harmbench/$BEST_HB_TULU3/cost/cost_metrics_by_category.csv \
            $OUTPUT/harmbench/qwen3-4b/cost/cost_metrics_by_category.csv \
            $OUTPUT/harmbench/qwen3-4b-saferl/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/ablations/best_per_family/$axis \
        --x-axis $axis --title "HarmBench — Best per Family" \
        --mode comparison --skip-missing
done

for axis in $AXES; do
    $COST_PLOT \
        --cost-csv \
            $OUTPUT/jailbreakbench/$BEST_JB_QWEN25/cost/cost_metrics.csv \
            $OUTPUT/jailbreakbench/$BEST_JB_TULU3/cost/cost_metrics.csv \
            $OUTPUT/jailbreakbench/qwen3-4b/cost/cost_metrics.csv \
            $OUTPUT/jailbreakbench/qwen3-4b-saferl/cost/cost_metrics.csv \
        --cost-category-csv \
            $OUTPUT/jailbreakbench/$BEST_JB_QWEN25/cost/cost_metrics_by_category.csv \
            $OUTPUT/jailbreakbench/$BEST_JB_TULU3/cost/cost_metrics_by_category.csv \
            $OUTPUT/jailbreakbench/qwen3-4b/cost/cost_metrics_by_category.csv \
            $OUTPUT/jailbreakbench/qwen3-4b-saferl/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/jailbreakbench/ablations/best_per_family/$axis \
        --x-axis $axis --title "JailbreakBench — Best per Family" \
        --mode comparison --skip-missing
done

# --- Ablation 5: Attack transfer (GCG surrogate: qwen2.5-0.5b → qwen3-4b) ---

_normalize_attack() {
    local src="$1" dst="$2"
    python3 -c "
import pandas as pd
df = pd.read_csv('$src')
df['attack_id'] = df['attack_id'].str.replace(r'transfer_gcg_from_.*', 'gcg', regex=True)
df.to_csv('$dst', index=False)
" 2>/dev/null || cp "$src" "$dst"
}
#
NORM_HB_Q3=/tmp/norm_hb_qwen3_cost.csv
NORM_JB_Q3=/tmp/norm_jb_qwen3_cost.csv
NORM_HB_Q3_CAT=/tmp/norm_hb_qwen3_cost_cat.csv
NORM_JB_Q3_CAT=/tmp/norm_jb_qwen3_cost_cat.csv
#
_normalize_attack $OUTPUT/harmbench/qwen3-4b/cost/cost_metrics.csv             $NORM_HB_Q3
_normalize_attack $OUTPUT/jailbreakbench/qwen3-4b/cost/cost_metrics.csv         $NORM_JB_Q3
_normalize_attack $OUTPUT/harmbench/qwen3-4b/cost/cost_metrics_by_category.csv  $NORM_HB_Q3_CAT
_normalize_attack $OUTPUT/jailbreakbench/qwen3-4b/cost/cost_metrics_by_category.csv $NORM_JB_Q3_CAT
#
for axis in tokens flops; do
    $COST_PLOT \
        --cost-csv \
            $OUTPUT/harmbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv \
            $NORM_HB_Q3 \
        --cost-category-csv \
            $OUTPUT/harmbench/qwen2.5-0.5b-instruct/cost/cost_metrics_by_category.csv \
            $NORM_HB_Q3_CAT \
        --output-dir $OUTPUT/harmbench/ablations/attack_transfer_gcg/$axis \
        --x-axis $axis --attacks gcg \
        --title "HarmBench — GCG Transfer: Qwen2.5-0.5B → Qwen3-4B" \
        --mode comparison --skip-missing
done
#
for axis in tokens flops; do
    $COST_PLOT \
        --cost-csv \
            $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv \
            $NORM_JB_Q3 \
        --cost-category-csv \
            $OUTPUT/jailbreakbench/qwen2.5-0.5b-instruct/cost/cost_metrics_by_category.csv \
            $NORM_JB_Q3_CAT \
        --output-dir $OUTPUT/jailbreakbench/ablations/attack_transfer_gcg/$axis \
        --x-axis $axis --attacks gcg \
        --title "JailbreakBench — GCG Transfer: Qwen2.5-0.5B → Qwen3-4B" \
        --mode comparison --skip-missing
done

# =============================================================================
# RL model comparison — how different targets fare under the RL (GRPO) attack.
# --attacks rl restricts each comparison to the RL curve, one line per model
# (→ cost_comparison_rl.{png} per output dir). RL only ran on HarmBench for these
# targets (JB has a single RL model, so no JB comparison is meaningful).
# =============================================================================

# --- RL across Qwen2.5 model sizes (0.5B / 3B / 7B) ---
for axis in $AXES; do
    $COST_PLOT \
        --cost-csv \
            $OUTPUT/harmbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv \
            $OUTPUT/harmbench/qwen2.5-3b-instruct/cost/cost_metrics.csv \
            $OUTPUT/harmbench/qwen2.5-7b-instruct/cost/cost_metrics.csv \
        --cost-category-csv \
            $OUTPUT/harmbench/qwen2.5-0.5b-instruct/cost/cost_metrics_by_category.csv \
            $OUTPUT/harmbench/qwen2.5-3b-instruct/cost/cost_metrics_by_category.csv \
            $OUTPUT/harmbench/qwen2.5-7b-instruct/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/ablations/rl_qwen_size/$axis \
        --x-axis $axis --attacks rl \
        --title "HarmBench — RL (GRPO): Qwen2.5 Model Size" \
        --mode comparison --skip-missing
done

# --- RL across Gemma 3 model sizes (270M / 1B / 4B) ---
# The cross-family read: does the adaptive attacker's advantage track size the same
# way outside Qwen? Compare against ablations/rl_qwen_size on the same axis.
for axis in $AXES; do
    $COST_PLOT \
        --cost-csv \
            $OUTPUT/harmbench/gemma3-270m-it/cost/cost_metrics.csv \
            $OUTPUT/harmbench/gemma3-1b-it/cost/cost_metrics.csv \
            $OUTPUT/harmbench/gemma3-4b-it/cost/cost_metrics.csv \
        --cost-category-csv \
            $OUTPUT/harmbench/gemma3-270m-it/cost/cost_metrics_by_category.csv \
            $OUTPUT/harmbench/gemma3-1b-it/cost/cost_metrics_by_category.csv \
            $OUTPUT/harmbench/gemma3-4b-it/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/ablations/rl_gemma_size/$axis \
        --x-axis $axis --attacks rl \
        --title "HarmBench — RL (GRPO): Gemma 3 Model Size" \
        --mode comparison --skip-missing
done

# --- RL across Tulu3 training stages (Base / SFT / DPO / RLVR) ---
for axis in $AXES; do
    $COST_PLOT \
        --cost-csv \
            $OUTPUT/harmbench/tulu3-8b-base/cost/cost_metrics.csv \
            $OUTPUT/harmbench/tulu3-8b-sft/cost/cost_metrics.csv \
            $OUTPUT/harmbench/tulu3-8b-dpo/cost/cost_metrics.csv \
            $OUTPUT/harmbench/tulu3-8b-rlvr/cost/cost_metrics.csv \
        --cost-category-csv \
            $OUTPUT/harmbench/tulu3-8b-base/cost/cost_metrics_by_category.csv \
            $OUTPUT/harmbench/tulu3-8b-sft/cost/cost_metrics_by_category.csv \
            $OUTPUT/harmbench/tulu3-8b-dpo/cost/cost_metrics_by_category.csv \
            $OUTPUT/harmbench/tulu3-8b-rlvr/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/ablations/rl_tulu3_training/$axis \
        --x-axis $axis --attacks rl \
        --title "HarmBench — RL (GRPO): Tulu3 Training Stages" \
        --mode comparison --skip-missing
done

# --- RL across OLMo 2 1B training stages (Base / SFT / DPO / RLVR1 / Instruct) ---
# The five-rung read: compare against ablations/rl_tulu3_training on the same axis.
# The RLVR1 -> Instruct segment is the marginal effect of a second RLVR round, which the
# four-rung Tulu3 ladder cannot show.
for axis in $AXES; do
    $COST_PLOT \
        --cost-csv \
            $OUTPUT/harmbench/olmo2-1b-base/cost/cost_metrics.csv \
            $OUTPUT/harmbench/olmo2-1b-sft/cost/cost_metrics.csv \
            $OUTPUT/harmbench/olmo2-1b-dpo/cost/cost_metrics.csv \
            $OUTPUT/harmbench/olmo2-1b-rlvr1/cost/cost_metrics.csv \
            $OUTPUT/harmbench/olmo2-1b-instruct/cost/cost_metrics.csv \
        --cost-category-csv \
            $OUTPUT/harmbench/olmo2-1b-base/cost/cost_metrics_by_category.csv \
            $OUTPUT/harmbench/olmo2-1b-sft/cost/cost_metrics_by_category.csv \
            $OUTPUT/harmbench/olmo2-1b-dpo/cost/cost_metrics_by_category.csv \
            $OUTPUT/harmbench/olmo2-1b-rlvr1/cost/cost_metrics_by_category.csv \
            $OUTPUT/harmbench/olmo2-1b-instruct/cost/cost_metrics_by_category.csv \
        --output-dir $OUTPUT/harmbench/ablations/rl_olmo2_training/$axis \
        --x-axis $axis --attacks rl \
        --title "HarmBench — RL (GRPO): OLMo 2 1B Training Stages" \
        --mode comparison --skip-missing
done

# --- RL across ALL HarmBench targets (Qwen2.5 sizes + Tulu3 stages) on one canvas ---
for axis in $AXES; do
    $COST_PLOT \
        --cost-csv \
            $OUTPUT/harmbench/qwen2.5-0.5b-instruct/cost/cost_metrics.csv \
            $OUTPUT/harmbench/qwen2.5-3b-instruct/cost/cost_metrics.csv \
            $OUTPUT/harmbench/qwen2.5-7b-instruct/cost/cost_metrics.csv \
            $OUTPUT/harmbench/tulu3-8b-base/cost/cost_metrics.csv \
            $OUTPUT/harmbench/tulu3-8b-sft/cost/cost_metrics.csv \
            $OUTPUT/harmbench/tulu3-8b-dpo/cost/cost_metrics.csv \
            $OUTPUT/harmbench/tulu3-8b-rlvr/cost/cost_metrics.csv \
            $OUTPUT/harmbench/gemma3-270m-it/cost/cost_metrics.csv \
            $OUTPUT/harmbench/gemma3-1b-it/cost/cost_metrics.csv \
            $OUTPUT/harmbench/gemma3-4b-it/cost/cost_metrics.csv \
            $OUTPUT/harmbench/olmo2-1b-base/cost/cost_metrics.csv \
            $OUTPUT/harmbench/olmo2-1b-sft/cost/cost_metrics.csv \
            $OUTPUT/harmbench/olmo2-1b-dpo/cost/cost_metrics.csv \
            $OUTPUT/harmbench/olmo2-1b-rlvr1/cost/cost_metrics.csv \
            $OUTPUT/harmbench/olmo2-1b-instruct/cost/cost_metrics.csv \
        --output-dir $OUTPUT/harmbench/ablations/rl_all_models/$axis \
        --x-axis $axis --attacks rl \
        --title "HarmBench — RL (GRPO): All Models" \
        --mode comparison --skip-missing
done

echo "Done."
