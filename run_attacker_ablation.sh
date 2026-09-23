#!/bin/bash
# run_attacker_ablation.sh — attacker-size study: who writes the jailbreak prompts?
#
# Every other sweep in this repo varies the target or the judge with the attacker fixed at
# Qwen2.5-7B. This one crosses two abliterated Gemma attackers with the full target grid:
#
#   attackers: gemma3_4b_it_abliterated  3.88B      gemma3_1b_it_abliterated  1.00B
#   targets:   qwen2.5 0.5B / 3B / 7B    +   tulu3-8b base / sft / dpo / rlvr
#   attacks:   pair (prompts the attacker)  +  rl (GRPO trains it)
#   judge:     llama3.1-8b throughout
#
# Both attackers share a family and an abliteration recipe, so 4B vs 1B is a clean size
# contrast. Per target it asks "does attacker size matter?"; across the target grid it asks
# the sharper question — whether a small attacker only keeps up against weak targets and
# falls off as the target hardens. See configs/experiments/paper/attacker_size.yaml.
#
# The cost axes are the point: a 1B attacker is billed at 1.00B on the FLOP axis and at its
# own $/1M-token rate on the dollar axis, so "cheaper attacker, same risk curve?" is a
# question the plots can actually answer. compute_attack_costs.py picks the attacker up from
# the pair__<attacker> directory name — no flag needed.
#
# Everything lands in its own tree so the main results are untouched:
#   $SCRATCH/rup/attackers/harmbench/<target>/<seed>/pair__<attacker>/results.jsonl
#   $SCRATCH/rup/attackers/harmbench/<target>/<seed>/rl__<attacker>/results.jsonl
#   $SCRATCH/rup/attackers/plots/harmbench/<target>/...
#   $SCRATCH/rup/attackers/plots/harmbench/ablations/{qwen_size,tulu3_training}/...
# (Under a non-default JUDGE the tree moves with it, to $SCRATCH/rup/judges/<judge>/attackers.)
#
# Usage:
#   bash run_attacker_ablation.sh smoke              # 5 prompts, budget 2 — do this FIRST
#   bash run_attacker_ablation.sh smoke-check        # per-arm verdicts once it finishes
#   bash run_attacker_ablation.sh                    # phase 1: submit inference
#   bash run_attacker_ablation.sh eval               # phase 2 + 2.5: metrics + cost metrics
#   bash run_attacker_ablation.sh plots              # per-target + cross-target cost curves
#   ATTACKS=pair bash run_attacker_ablation.sh       # PAIR arms only (RL is much pricier)
#   TARGETS="qwen2.5_0.5b" bash run_attacker_ablation.sh    # one target
#   SEEDS="1394 2 100" bash run_attacker_ablation.sh        # more seeds
#   BENCHMARK=jailbreakbench bash run_attacker_ablation.sh  # the other benchmark
#
# Phase 1 submits SLURM jobs and returns immediately; run `eval` only once they finish.
#
# Run `smoke` first. An attacker that loads but never produces a usable refinement — it
# refuses, or returns an empty string — makes PAIR fall back to the previous prompt, so the
# arm silently degrades into "ask the same thing 10 times" and reads as a weak attacker
# instead of a broken one. `smoke-check` counts how often each arm actually changed the
# prompt, which is what separates the two.
#
# COMPUTE WARNING: this is the largest sweep in the repo. Per seed it is 7 targets x 2
# attackers x 2 attacks over 200 prompts at lambda=10 — 28 attack runs, submitted as 7 jobs
# (one per target, each looping its own arms). RL is per-prompt GRPO and dominates the bill:
# run ATTACKS=pair across the grid first, then add RL. The seed list multiplies all of it.
#
# GPU memory does not scale with the arm count: PAIR loads one 4-bit attacker at a time, and
# the RL path holds exactly one trainable attacker (bf16 + LoRA, ~8 GB at 4B), freeing it
# before building the next arm's.

set -e

source setup/start_env.sh
source setup/judge_env.sh

EXPERIMENT="configs/experiments/paper/attacker_size.yaml"

# Targets and benchmark come from the experiment YAML rather than being repeated here, so the
# result paths below cannot drift from what the runs actually used. Override TARGETS to run a
# subset (config names, space separated).
TARGETS="${TARGETS:-$(python -c "import yaml; print(' '.join(yaml.safe_load(open('$EXPERIMENT'))['models']))")}"
BENCHMARK="${BENCHMARK:-$(python -c "import yaml; print(yaml.safe_load(open('$EXPERIMENT'))['benchmark'])")}"

# config name -> model_id, which is what run_inference.py names the result directory after.
target_id() { python -c "import yaml; print(yaml.safe_load(open('configs/models/$1.yaml'))['model_id'])"; }

# Benchmark goes in the job name as well as the path: HB and JB runs of the same target must
# not look like the same job to should_skip_job.
case "$BENCHMARK" in
    harmbench)      BM_TAG="HB" ;;
    jailbreakbench) BM_TAG="JB" ;;
    *)              echo "ERROR: unknown benchmark '$BENCHMARK'" >&2; exit 1 ;;
esac

ATT_ROOT="$RUN_ROOT/attackers"
ATT_PLOTS="$ATT_ROOT/plots"
RESULTS="$ATT_ROOT/$BENCHMARK"       # per target: $RESULTS/<model_id>
METRICS="$ATT_PLOTS/$BENCHMARK"      # per target: $METRICS/<model_id>

# One seed by default, matching the seed the other sweeps currently run with.
SEEDS="${SEEDS:-1394}"

# Attacks to run; both use an attacker LLM and both honour attacker_models. Restrict with
# ATTACKS=pair (cheap, do this first) or ATTACKS=rl. The list goes into the job name so a
# later ATTACKS=rl run is not skipped as "already completed" by should_skip_job.
ATTACKS="${ATTACKS:-$(python -c "import yaml; print(' '.join(yaml.safe_load(open('$EXPERIMENT'))['attacks']))")}"
ATTACK_TAG="_$(echo $ATTACKS | tr ' ' '-')"

# GRPO group size, read from the attack config so the LoRA-aware attacker FLOP reconstruction
# in compute_attack_costs.py cannot drift from the value the runs actually used.
RL_GENS=$(python -c "import yaml; print(yaml.safe_load(open('configs/attacks/rl.yaml'))['extra']['num_generations'])")

# Cost axes for the plots stage. dollars needs the pricing config, which the eval stage passes.
AXES="${AXES:-tokens flops seconds dollars}"

STAGE="${1:-submit}"

echo "Experiment:  $EXPERIMENT"
echo "Benchmark:   $BENCHMARK"
echo "Targets:     $TARGETS"
echo "Attackers:   $(python -c "import yaml; print(' '.join(yaml.safe_load(open('$EXPERIMENT'))['attacker_models']))")"
echo "Attacks:     $ATTACKS"
echo "Results:     $RESULTS/<target>"
echo "Stage:       $STAGE"
echo

SMOKE_ROOT="$SCRATCH/rup_attacker_smoke"

case "$STAGE" in
    smoke)
        # Tiny on purpose: one small target, 5 prompts, budget 2, one seed. Throwaway tree.
        # Worth running for RL specifically: the GRPO path applies LoRA to a fixed list of
        # projection names, and this is the first non-Qwen attacker to go through it.
        submit "rup_attacker_smoke_${BM_TAG}$ATTACK_TAG$JUDGE_TAG" \
            "python scripts/run_inference.py --experiment $EXPERIMENT \
                --model qwen2.5_0.5b --attacks $ATTACKS --n-prompts 5 --lambda-max 2 --seeds 42 \
                --benchmark $BENCHMARK \
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
        # One job per (target, seed): each loops its own attacker arms internally, so the
        # attacker models are loaded once per job rather than once per target.
        for target in $TARGETS; do
            for seed in $SEEDS; do
                submit "rup_att_${BM_TAG}_${target}${ATTACK_TAG}_s${seed}$JUDGE_TAG" \
                    "python scripts/run_inference.py --experiment $EXPERIMENT \
                        --model $target --attacks $ATTACKS --seeds $seed \
                        --benchmark $BENCHMARK \
                        --output-dir $ATT_ROOT --judge-model $JUDGE --resume"
            done
        done
        echo
        echo "Submitted. When the jobs finish:  bash run_attacker_ablation.sh eval"
        ;;

    eval)
        for target in $TARGETS; do
            tid=$(target_id $target)
            echo "=== $tid ==="
            # Phase 2 — ASR / lambda* per (attack, attacker arm). Every arm is its own
            # attack_id row (pair__<attacker>, rl__<attacker>), so one metrics.csv per target
            # holds that target's whole comparison, PAIR and RL side by side.
            python scripts/run_evaluation.py \
                --experiment $EXPERIMENT --format csv --print-table \
                --results-dir $RESULTS/$tid \
                --output $METRICS/$tid/metrics.csv \
                | tee $RESULTS/$tid/summary.txt

            # Phase 2.5 — cost columns. No --attacker-model flag: every pair__/rl__ directory
            # is charged at that attacker's own params_b and hosted rate. --rl-num-generations
            # reconstructs which RL candidates took a LoRA update (8x vs 2x attacker FLOPs).
            python scripts/compute_attack_costs.py \
                --pricing-config configs/pricing.yaml --judge-model $JUDGE_ID \
                --rl-num-generations $RL_GENS \
                --results-dir $RESULTS/$tid \
                --metrics-csv $METRICS/$tid/metrics.csv \
                --output      $METRICS/$tid/cost/cost_metrics.csv
        done
        ;;

    plots)
        # Per-target: every arm as a series, so attacker size is read off one figure.
        for target in $TARGETS; do
            tid=$(target_id $target)
            for axis in $AXES; do
                python scripts/plot_cost_curves.py \
                    --cost-csv $METRICS/$tid/cost/cost_metrics.csv \
                    --cost-category-csv $METRICS/$tid/cost/cost_metrics_by_category.csv \
                    --output-dir $METRICS/$tid/$axis \
                    --x-axis $axis --skip-missing \
                    --title "Attacker size — $tid ($BENCHMARK)"
            done
        done

        # Cross-target: --mode comparison overlays the targets per arm, which is the view that
        # shows whether the small attacker's penalty grows as the target hardens. Mirrors the
        # ablation sets in run_cost_plots.sh. --skip-missing covers a trimmed TARGETS list.
        _comparison() {
            local name="$1" title="$2"; shift 2
            local csvs=() cats=()
            for t in "$@"; do
                csvs+=("$METRICS/$(target_id $t)/cost/cost_metrics.csv")
                cats+=("$METRICS/$(target_id $t)/cost/cost_metrics_by_category.csv")
            done
            for axis in $AXES; do
                python scripts/plot_cost_curves.py \
                    --cost-csv "${csvs[@]}" \
                    --cost-category-csv "${cats[@]}" \
                    --output-dir $ATT_PLOTS/$BENCHMARK/ablations/$name/$axis \
                    --x-axis $axis --mode comparison --skip-missing \
                    --title "$title"
            done
        }
        _comparison qwen_size "$BENCHMARK — Qwen2.5 size x attacker size" \
            qwen2.5_0.5b qwen2.5_3b qwen2.5_7b
        _comparison tulu3_training "$BENCHMARK — Tulu3 training stage x attacker size" \
            tulu3_8b_base tulu3_8b_sft tulu3_8b_dpo tulu3_8b_rlvr

        echo "Per-target plots:  $METRICS/<target>/{$(echo $AXES | tr ' ' ',')}"
        echo "Comparison plots:  $ATT_PLOTS/$BENCHMARK/ablations/{qwen_size,tulu3_training}/"
        ;;

    *)
        echo "ERROR: unknown stage '$STAGE' (expected: smoke | smoke-check | submit | eval | plots)" >&2
        exit 1
        ;;
esac
