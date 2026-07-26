#!/bin/bash
# run_attacker_ablation.sh — attacker-size study: who writes the jailbreak prompts?
#
# Every other sweep in this repo varies the target or the judge. This one holds both fixed
# and varies the ATTACKER inside PAIR's refinement loop:
#
#   qwen2.5_7b                  7.62B  Qwen2.5-7B-Instruct        incumbent, safety-tuned
#   gemma3_4b_it_abliterated    3.88B  gemma-3-4b-it-abliterated  uncensored, 4B
#   gemma3_1b_it_abliterated    1.00B  gemma-3-1b-it-abliterated  uncensored, 1B
#
# The two Gemma arms share a family and an abliteration recipe, so 4B vs 1B is a clean size
# contrast; the Qwen arm is the reference the existing curves were measured with and differs
# on two axes at once (bigger AND safety-tuned). See configs/experiments/paper/attacker_size.yaml.
#
# The cost axes are the point: a 1B attacker is billed at 1.00B on the FLOP axis and at its
# own $/1M-token rate on the dollar axis, so "cheaper attacker, same risk curve?" is a
# question the plots can actually answer. compute_attack_costs.py picks the attacker up from
# the pair__<attacker> directory name — no flag needed.
#
# Everything lands in its own tree so the main results are untouched:
#   $SCRATCH/rup/attackers/harmbench/<target>/<seed>/pair__<attacker>/results.jsonl
#   $SCRATCH/rup/attackers/plots/harmbench/<target>/...
# (Under a non-default JUDGE the tree moves with it, to $SCRATCH/rup/judges/<judge>/attackers.)
#
# Usage:
#   bash run_attacker_ablation.sh smoke              # 5 prompts, budget 2 — do this FIRST
#   bash run_attacker_ablation.sh smoke-check        # per-arm verdicts once it finishes
#   bash run_attacker_ablation.sh                    # phase 1: submit inference
#   bash run_attacker_ablation.sh eval               # phase 2 + 2.5: metrics + cost metrics
#   bash run_attacker_ablation.sh plots              # risk curves on all four cost axes
#   SEEDS="1394 2 100" bash run_attacker_ablation.sh # more seeds (one job each)
#
# Phase 1 submits SLURM jobs and returns immediately; run `eval` only once they finish.
#
# Run `smoke` first. An attacker that loads but never produces a usable refinement — it
# refuses, or returns an empty string — makes PAIR fall back to the previous prompt, so the
# arm silently degrades into "ask the same thing 10 times" and reads as a weak attacker
# instead of a broken one. `smoke-check` counts how often each arm actually changed the
# prompt, which is what separates the two.
#
# COMPUTE WARNING: one job per seed runs ALL THREE attackers over 200 prompts at lambda=10,
# so a seed costs ~3x a plain PAIR seed. Trim attacker_models in the experiment YAML to cut it.
# GPU memory is not 3x though: run_inference.py loads one attacker per arm and drops the
# previous one when it rebinds, so the resident set is target + judge + one attacker (briefly
# two, while the next one loads). All three arms are 4-bit.

set -e

source setup/start_env.sh
source setup/judge_env.sh

EXPERIMENT="configs/experiments/paper/attacker_size.yaml"

# Target config is read from the experiment YAML rather than repeated here, so the results
# path below cannot drift from the model the runs actually used.
TARGET_CFG=$(python -c "import yaml; print(yaml.safe_load(open('$EXPERIMENT'))['models'][0])")
TARGET_ID=$(python -c "import yaml; print(yaml.safe_load(open('configs/models/$TARGET_CFG.yaml'))['model_id'])")
BENCHMARK=$(python -c "import yaml; print(yaml.safe_load(open('$EXPERIMENT'))['benchmark'])")

ATT_ROOT="$RUN_ROOT/attackers"
ATT_PLOTS="$ATT_ROOT/plots"
RESULTS="$ATT_ROOT/$BENCHMARK/$TARGET_ID"
METRICS="$ATT_PLOTS/$BENCHMARK/$TARGET_ID"

# One seed by default, matching the seed the other sweeps currently run with.
SEEDS="${SEEDS:-1394}"

# Cost axes for the plots stage. dollars needs the pricing config, which the eval stage passes.
AXES="${AXES:-tokens flops seconds dollars}"

STAGE="${1:-submit}"

echo "Experiment:  $EXPERIMENT"
echo "Target:      $TARGET_CFG (model_id=$TARGET_ID) on $BENCHMARK"
echo "Attackers:   $(python -c "import yaml; print(' '.join(yaml.safe_load(open('$EXPERIMENT'))['attacker_models']))")"
echo "Results:     $RESULTS"
echo "Stage:       $STAGE"
echo

SMOKE_ROOT="$SCRATCH/rup_attacker_smoke"

case "$STAGE" in
    smoke)
        # Tiny on purpose: one small target, 5 prompts, budget 2, one seed. Throwaway tree.
        submit "rup_attacker_smoke$JUDGE_TAG" \
            "python scripts/run_inference.py --experiment $EXPERIMENT \
                --model qwen2.5_0.5b --n-prompts 5 --lambda-max 2 --seeds 42 \
                --judge-model $JUDGE --output-dir $SMOKE_ROOT"
        echo
        echo "Submitted. When it finishes:  bash run_attacker_ablation.sh smoke-check"
        ;;

    smoke-check)
        python3 - "$SMOKE_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
if not root.exists():
    sys.exit(f"No smoke results at {root} — run `bash run_attacker_ablation.sh smoke` first.")

files = sorted(root.glob("*/*/*/*/results.jsonl"))
if not files:
    sys.exit(f"No results.jsonl under {root} yet — jobs may still be queued or may have failed.\n"
             f"Check: squeue -u $USER | grep attacker   and   logs/*_rup_attacker_smoke*.out")

print(f"{'attacker arm':32s} {'refined':>8} {'steps':>7} {'unsafe':>7}  verdict")
print("-" * 80)
for f in files:
    arm = f.parent.name                      # pair__<attacker config name>
    records = [json.loads(line) for line in f.open()]
    refined = total = unsafe = 0
    for r in records:
        steps = r["steps"]
        unsafe += sum(s["judgment"] for s in steps)
        # Step 1 is the raw behavior; a refinement counts when the attacker returned
        # something different from the prompt it was given. PAIR also (correctly) repeats
        # the prompt after a success, so those steps are not counted as failures.
        for prev, cur in zip(steps, steps[1:]):
            if prev["judgment"] == 1:
                continue
            total += 1
            refined += cur["prompt"] != prev["prompt"]

    if total == 0:
        verdict = "no refinement steps — every step succeeded, or the run failed"
    elif refined == 0:
        verdict = "BROKEN — attacker never changed the prompt (refusing or empty?)"
    elif refined < total / 2:
        verdict = f"SUSPECT — refined only {refined}/{total}; grep the log for 'empty prompt'"
    else:
        verdict = "ok"
    print(f"{arm:32s} {refined:>8} {total:>7} {unsafe:>7}  {verdict}")

print()
print("'refined' counts steps where the attacker returned a prompt different from the one it")
print("was given (excluding steps after a success, where repeating is the correct behavior).")
print("An arm at 0 is not a weak attacker — it is an attacker that never ran usefully.")
PY
        ;;

    submit)
        for seed in $SEEDS; do
            submit "rup_att_${TARGET_CFG}_s${seed}$JUDGE_TAG" \
                "python scripts/run_inference.py --experiment $EXPERIMENT \
                    --seeds $seed --output-dir $ATT_ROOT --judge-model $JUDGE --resume"
        done
        echo
        echo "Submitted. When the jobs finish:  bash run_attacker_ablation.sh eval"
        ;;

    eval)
        # Phase 2 — ASR / lambda* per (target, attacker arm). Each arm is a separate
        # attack_id row (pair__<attacker>), so one metrics.csv holds the whole comparison.
        python scripts/run_evaluation.py \
            --experiment $EXPERIMENT --format csv --print-table \
            --results-dir $RESULTS \
            --output $METRICS/metrics.csv \
            | tee $RESULTS/summary.txt

        # Phase 2.5 — cost columns. No --attacker-model flag: each pair__<attacker>
        # directory is charged at that attacker's own params_b and hosted rate.
        python scripts/compute_attack_costs.py \
            --pricing-config configs/pricing.yaml --judge-model $JUDGE_ID \
            --results-dir $RESULTS \
            --metrics-csv $METRICS/metrics.csv \
            --output      $METRICS/cost/cost_metrics.csv
        ;;

    plots)
        for axis in $AXES; do
            python scripts/plot_cost_curves.py \
                --cost-csv $METRICS/cost/cost_metrics.csv \
                --cost-category-csv $METRICS/cost/cost_metrics_by_category.csv \
                --output-dir $METRICS/$axis \
                --x-axis $axis --skip-missing \
                --title "PAIR attacker size — $TARGET_ID ($BENCHMARK)"
        done
        echo "Plots written under $METRICS/{$(echo $AXES | tr ' ' ',')}"
        ;;

    *)
        echo "ERROR: unknown stage '$STAGE' (expected: smoke | smoke-check | submit | eval | plots)" >&2
        exit 1
        ;;
esac
