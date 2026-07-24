#!/bin/bash
# Submit the per-prompt RL/GRPO adaptive-attack SMOKE TEST as a single GPU job.
# Tiny scale (2 prompts, per-prompt query budget 8, group 4) to confirm the RL path works
# (attack -> evaluate -> cost) before launching the real experiments. No separate training
# phase — per-prompt GRPO runs inside run_inference.py --attack rl.
#
# Usage: bash run_rl_smoke.sh
# Requires: must be run from the project root on a klogin* (or Alliance) login node.
#
# The pipeline runs on a GPU node (see scripts/rl_smoke_job.sh) and writes everything under
# $SCRATCH/rl_smoke; model + dataset caches go to $SCRATCH/huggingface (set in start_env.sh).
# Watch progress:  tail -f logs/<jobid>_rup_rl_smoke.out   — success prints "SMOKE TEST PASSED".

set -e

source setup/start_env.sh

submit "rup_rl_smoke" "bash scripts/rl_smoke_job.sh"
