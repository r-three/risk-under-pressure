#!/bin/bash
# run_judge_cost_main.sh — metrics + cost (token/FLOP/$) for the MAIN-PAPER targets only,
# under one or more alternative judges.
#
# Why this exists rather than `bash run_judge_ablation.sh eval`:
# run_evaluations.sh / run_cost_evaluations.sh walk EVERY target in the project, including
# the gemma3-* and olmo2-* ladders added after the judge sweep was launched. Those targets
# have no results under $SCRATCH/rup/judges/<judge>/, so with `set -e` the sweep aborts on
# the first missing one. This script restricts the walk to the nine models in Table 1 and
# skips (with a warning) anything that has not been run, instead of dying.
#
# Usage:
#   bash run_judge_cost_main.sh                       # both new judges, both benchmarks
#   JUDGES="olmo3_7b_instruct_judge" bash run_judge_cost_main.sh
#   BENCHES="harmbench" bash run_judge_cost_main.sh
#   STAGE=cost bash run_judge_cost_main.sh            # skip metrics.csv, redo costs only
#
# Then build the cross-judge comparison:
#   python scripts/compare_judges.py --benchmark harmbench
#
# NOTE ON WHICH AXIS TO COMPARE. The judges differ in size (llama3.1-8b 8.03B,
# olmo3-7b 7.30B, gemma3-4b 3.88B), so on the `flops` (total) axis part of any cross-judge
# difference is just the judge's own forward pass getting cheaper. Compare conclusions on
# `flops_nojudge` (target + attacker), which is invariant to the judge; use the `flops`
# axis only to quantify how much of the accounted compute the judge contributes.

# Deliberately NOT -e: a missing model must warn, not abort. Also not -u — setup/start_env.sh
# reads $OFFLINE_MODE with no default, which is fine unquoted but trips `set -u`.
set -o pipefail

: "${OFFLINE_MODE:=0}"
export OFFLINE_MODE

source setup/start_env.sh

JUDGES="${JUDGES:-olmo3_7b_instruct_judge gemma3_4b_it_judge}"
BENCHES="${BENCHES:-harmbench jailbreakbench}"
STAGE="${STAGE:-all}"        # all | metrics | cost

# The nine targets of Table 1. Deliberately excludes the gemma3-* / olmo2-* ladders.
MAIN_MODELS="${MAIN_MODELS:-tulu3-8b-base tulu3-8b-sft tulu3-8b-dpo tulu3-8b-rlvr \
qwen2.5-0.5b-instruct qwen2.5-3b-instruct qwen2.5-7b-instruct qwen3-4b qwen3-4b-saferl}"

EVAL="python scripts/run_evaluation.py --experiment configs/experiments/base.yaml --format csv --print-table"

skipped=0
done_n=0

for judge in $JUDGES; do
    # Sets JUDGE_ID / RUN_ROOT / PLOT_ROOT for this judge.
    JUDGE="$judge" source setup/judge_env.sh || { echo "skip judge $judge"; continue; }

    COST="python scripts/compute_attack_costs.py \
        --pricing-config configs/pricing.yaml \
        --judge-model $JUDGE_ID"

    for bench in $BENCHES; do
        for model in $MAIN_MODELS; do
            rdir="$RUN_ROOT/$bench/$model"
            odir="$PLOT_ROOT/$bench/$model"

            if [ ! -d "$rdir" ]; then
                echo "SKIP  $judge / $bench / $model  (no results dir)"
                skipped=$((skipped + 1))
                continue
            fi

            mkdir -p "$odir/cost"

            if [ "$STAGE" = "all" ] || [ "$STAGE" = "metrics" ]; then
                echo "EVAL  $judge / $bench / $model"
                $EVAL --results-dir "$rdir" --output "$odir/metrics.csv" \
                    | tee "$rdir/summary.txt" || {
                        echo "WARN  eval failed: $judge / $bench / $model" >&2
                        skipped=$((skipped + 1)); continue; }
            fi

            if [ "$STAGE" = "all" ] || [ "$STAGE" = "cost" ]; then
                if [ ! -f "$odir/metrics.csv" ]; then
                    echo "SKIP  $judge / $bench / $model  (no metrics.csv — run STAGE=metrics first)"
                    skipped=$((skipped + 1)); continue
                fi
                echo "COST  $judge / $bench / $model"
                $COST --results-dir "$rdir" \
                      --metrics-csv "$odir/metrics.csv" \
                      --output      "$odir/cost/cost_metrics.csv" || {
                        echo "WARN  cost failed: $judge / $bench / $model" >&2
                        skipped=$((skipped + 1)); continue; }
            fi

            done_n=$((done_n + 1))
        done
    done
done

echo
echo "Done: $done_n model/benchmark/judge combinations processed, $skipped skipped."
echo "Compare with: python scripts/compare_judges.py --benchmark harmbench"
