#!/bin/bash
# run_judge_ablation.sh — replicate the full experiment sweep under each safety judge.
#
# The judge is the measurement instrument for every risk curve in this project: it decides
# what counts as a successful jailbreak, so ASR, lambda*, and every cost axis are all
# conditioned on it. This script re-runs the whole sweep under alternative judges so that
# conditioning can be reported rather than assumed.
#
# Judges (see configs/models/*_judge.yaml):
#   llama3.1_8b_instruct_judge      8.03B  Meta       2024-07   (default / incumbent)
#   olmo3_7b_instruct_judge         7.30B  AI2        2025-11   fully open, reproducible
#   gemma3_4b_it_judge              3.88B  Google     2025-03   smallest — capacity probe
# (Sizes are the text-only LM; Gemma 3 is a multimodal checkpoint whose vision tower
#  never runs on a text-only judging call. See the config for the split.)
#
# Each judge writes to its own tree, so nothing overwrites anything:
#   default judge  -> $SCRATCH/rup/{harmbench,jailbreakbench}/...
#                     $SCRATCH/rup/plots/...
#   other judges   -> $SCRATCH/rup/judges/<judge_model_id>/{harmbench,jailbreakbench}/...
#                     $SCRATCH/rup/judges/<judge_model_id>/plots/...
#
# Usage:
#   bash run_judge_ablation.sh                  # phase 1: submit inference for the new judges
#   bash run_judge_ablation.sh eval             # phase 2+2.5: metrics + cost metrics
#   bash run_judge_ablation.sh plots            # cost-axis plots
#   JUDGES="gemma3_4b_it_judge" bash run_judge_ablation.sh   # restrict to one judge
#
# Phase 1 submits SLURM jobs and returns immediately. Run `eval` only after those jobs
# finish — it reads the results.jsonl trees they produce.
#
# Run `bash run_judge_smoke.sh` FIRST. It validates each judge in ~10 minutes; a judge that
# loads but never receives its rubric labels everything SAFE, and you would not notice until
# the curves came out wrong 3x the sweep later.
#
# COMPUTE WARNING: this multiplies the sweep by the number of judges. With the default
# JUDGES list that is 2x the HB + JB + RL-HB + RL-JB cost. Trim JUDGES, or comment out
# stages below, before launching.

set -e

# Judges to sweep. The incumbent llama3.1_8b_instruct_judge is deliberately NOT here —
# those runs already exist under $SCRATCH/rup. Add it back to regenerate them.
JUDGES="${JUDGES:- flow_judge_v01}"

STAGE="${1:-submit}"

for judge in $JUDGES; do
    echo
    echo "============================================================================"
    echo "JUDGE: $judge   (stage: $STAGE)"
    echo "============================================================================"

    case "$STAGE" in
        submit)
            # Phase 1 — inference. Static attacks (pair/jailbroken/gcg) on both benchmarks,
            # then the per-prompt GRPO adaptive attack on both benchmarks.
            JUDGE="$judge" bash run_HB_experiments.sh
            JUDGE="$judge" bash run_JB_experiments.sh
            # JUDGE="$judge" bash run_rl_HB_experiments.sh
            # JUDGE="$judge" bash run_rl_JB_experiments.sh
            ;;
        eval)
            # Phase 2 — ASR / lambda* metrics, then phase 2.5 — token, FLOP, second and
            # dollar cost columns. run_cost_evaluations.sh passes --judge-model so the
            # judge's own params_b and $/1M-token rate are the ones charged.
            JUDGE="$judge" bash run_evaluations.sh
            JUDGE="$judge" bash run_cost_evaluations.sh
            ;;
        plots)
            JUDGE="$judge" bash run_cost_plots.sh
            ;;
        *)
            echo "ERROR: unknown stage '$STAGE' (expected: submit | eval | plots)" >&2
            exit 1
            ;;
    esac
done

echo
echo "Done: stage '$STAGE' for judges: $JUDGES"
