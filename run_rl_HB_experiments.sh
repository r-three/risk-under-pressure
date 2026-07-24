#!/bin/bash
# Submit the per-prompt RL/GRPO adaptive-attack experiments on HarmBench.
# Adaptive attacker following "The Attacker Moves Second" (Nasr et al., 2025, arXiv:2510.09023):
# for EACH behavior, Qwen2.5-7B-Instruct runs a short GRPO optimization against the target
# (group rollouts + weight updates) up to a per-prompt query budget (= lambda-max).
#
# There is NO separate training phase — per-prompt GRPO happens inside run_inference.py, so RL is
# submitted just like any other attack (e.g. gcg), one job per (model, seed).
#
# Usage: bash run_rl_HB_experiments.sh
# Requires: must be run from the project root on a klogin* node.
#
# COMPUTE WARNING: per-prompt GRPO is the most expensive attack (a mini optimization, with
# backward passes, per behavior). n-prompts=200 × 9 targets is a large sweep — comment out
# targets you don't need. Watch 48 GB VRAM (7B bf16 attacker + LoRA optimizer + 4-bit target
# + 4-bit judge + KV caches).

set -e

source setup/start_env.sh

# Subset + budget knobs (tune for compute). base.yaml default lambda_max=10.
RL_HB="python scripts/run_inference.py --experiment configs/experiments/base.yaml --attack rl --n-prompts 200 --lambda-max 10 --output-dir $SCRATCH/rup --resume"

# =============================================================================
# MODEL SIZE STUDY — Qwen2.5-Instruct: 0.5B, 3B, 7B
# =============================================================================
submit "rup_HB_rl_qwen2.5_0.5b_s1395" "$RL_HB --model qwen2.5_0.5b --seeds 1395"
submit "rup_HB_rl_qwen2.5_3b_s1395"   "$RL_HB --model qwen2.5_3b --seeds 1395"
submit "rup_HB_rl_qwen2.5_7b_s1395"   "$RL_HB --model qwen2.5_7b --seeds 1395"

# =============================================================================
# TRAINING STAGE STUDY — Tulu3 8B: Base -> SFT -> DPO -> RLVR
# =============================================================================
submit "rup_HB_rl_tulu3_8b_base_s1395" "$RL_HB --model tulu3_8b_base --seeds 1395"
submit "rup_HB_rl_tulu3_8b_sft_s1395"  "$RL_HB --model tulu3_8b_sft --seeds 1395"
submit "rup_HB_rl_tulu3_8b_dpo_s1395"  "$RL_HB --model tulu3_8b_dpo --seeds 1395"
submit "rup_HB_rl_tulu3_8b_rlvr_s1395" "$RL_HB --model tulu3_8b_rlvr --seeds 1395"

# =============================================================================
# SAFETY ALIGNMENT STUDY — Qwen3-4B base vs Qwen3-4B-SafeRL
# =============================================================================
submit "rup_HB_rl_qwen3_4b_s1395"        "$RL_HB --model qwen3_4b --seeds 1395"
submit "rup_HB_rl_qwen3_4b_saferl_s1395" "$RL_HB --model qwen3_4b_saferl --seeds 1395"
