#!/bin/bash
# run_judge_smoke.sh — validate a safety judge before spending real compute on it.
#
# Loads each judge and runs a tiny attack (1 target, 5 prompts, budget 2, jailbroken only)
# so you find out in ~10 minutes, not 23 hours, whether the judge loads at all and whether
# it produces sane labels.
#
# WHY THIS EXISTS: a judge can load cleanly and still be broken. If its rubric never reaches
# the model — wrong chat template, dropped system prompt — it returns SAFE for everything,
# which is indistinguishable from a well-aligned target in every downstream risk curve. That
# failure is silent and it poisons the whole sweep, so check it up front.
#
# Everything lands in a throwaway tree ($SCRATCH/rup_judge_smoke) and never touches the real
# results under $SCRATCH/rup.
#
# Usage:
#   bash run_judge_smoke.sh              # submit the smoke jobs (returns immediately)
#   bash run_judge_smoke.sh check        # after they finish: print label counts
#   JUDGES="gemma3_4b_it_judge" bash run_judge_smoke.sh
#
# Requires: must be run from the project root on a klogin*/Alliance login node.

set -e

source setup/start_env.sh

# The incumbent judge is included by default as a REFERENCE, not because it needs testing:
# "3 unsafe / 10 steps" only means something next to what the known-good judge scores on the
# exact same prompts. Drop it from JUDGES once you have that baseline.
JUDGES="${JUDGES:-llama3.1_8b_instruct_judge olmo3_7b_instruct_judge gemma3_4b_it_judge}"

SMOKE_ROOT="$SCRATCH/rup_judge_smoke"

# Tiny on purpose: one small target, one cheap attack, 5 prompts, budget 2.
SMOKE_ARGS="--experiment configs/experiments/base.yaml \
--model qwen2.5_0.5b --attacks jailbroken --n-prompts 5 --lambda-max 2 --seeds 42"

STAGE="${1:-submit}"

case "$STAGE" in
    submit)
        for judge in $JUDGES; do
            # Reuse judge_env for name validation and the config -> model_id lookup, then
            # override the output root: smoke results must not land in the real tree.
            # judge_env `return`s non-zero on an unknown name rather than exiting (it is
            # meant to be sourced), so check the status explicitly instead of trusting set -e.
            export JUDGE="$judge"
            if ! source setup/judge_env.sh >/dev/null; then
                echo "ERROR: could not resolve judge '$judge' — aborting." >&2
                exit 1
            fi

            echo "Submitting smoke test: $judge (model_id=$JUDGE_ID)"
            submit "rup_judge_smoke_${JUDGE_ID//[.-]/_}" \
                "python scripts/run_inference.py $SMOKE_ARGS \
                    --judge-model $judge --output-dir $SMOKE_ROOT/$JUDGE_ID"
        done
        echo
        echo "Submitted. When the jobs finish:  bash run_judge_smoke.sh check"
        echo "Watch progress:                   squeue -u \$USER --noheader -o '%j %t' | grep smoke"
        ;;

    check)
        python3 - "$SMOKE_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
if not root.exists():
    sys.exit(f"No smoke results at {root} — run `bash run_judge_smoke.sh` first.")

files = sorted(root.glob("*/*/*/*/*/results.jsonl"))
if not files:
    sys.exit(f"No results.jsonl under {root} yet — jobs may still be queued or may have failed.\n"
             f"Check: squeue -u $USER | grep smoke   and   logs/*_rup_judge_smoke_*.out")

print(f"{'judge':26s} {'unsafe':>7} {'steps':>7} {'rate':>7}  verdict")
print("-" * 68)
for f in files:
    judge = f.relative_to(root).parts[0]
    records = [json.loads(line) for line in f.open()]
    steps = [s for r in records for s in r["steps"]]
    unsafe = sum(s["judgment"] for s in steps)
    total = len(steps)
    rate = unsafe / total if total else 0.0

    if total == 0:
        verdict = "NO STEPS — run failed"
    elif unsafe == 0:
        verdict = "SUSPECT — all SAFE, check the rubric reached the judge"
    elif unsafe == total:
        verdict = "SUSPECT — all UNSAFE, check output parsing"
    else:
        verdict = "ok"
    print(f"{judge:26s} {unsafe:>7} {total:>7} {rate:>6.0%}  {verdict}")

print()
print("Compare each judge against llama3.1-8b-instruct on the same 5 prompts. An all-SAFE")
print("judge is the dangerous case: it looks exactly like a perfectly aligned target.")
print("Also grep the job logs for 'Chat template rejected a system role' — benign (the rubric")
print("is folded into the user turn instead), but it tells you which template took that path.")
PY
        ;;

    *)
        echo "ERROR: unknown stage '$STAGE' (expected: submit | check)" >&2
        exit 1
        ;;
esac
