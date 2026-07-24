#!/bin/bash
# Submit the per-prompt RL/GRPO adaptive-attack experiments.
# Adaptive attacker following "The Attacker Moves Second" (Nasr et al., 2025, arXiv:2510.09023):
# for EACH behavior, Qwen2.5-7B-Instruct runs a short GRPO optimization against the target
# (group rollouts + weight updates) up to a per-prompt query budget (= lambda_max).
#
# There is NO separate training phase — per-prompt GRPO happens inside run_inference.py, so RL is
# submitted just like any other attack (e.g. gcg), one job per (model, seed).
#
# Usage: bash run_rl_experiments.sh
# Requires: must be run from the project root on a klogin* node.
#
# COMPUTE WARNING: per-prompt GRPO is the most expensive attack (a mini optimization, with
# backward passes, per behavior). Start on a SUBSET of behaviors (--n-prompts ~50) and a modest
# per-prompt budget (--lambda-max, and num_generations in configs/attacks/rl.yaml). Watch 48 GB
# VRAM (7B bf16 attacker + LoRA optimizer + 4-bit target + 4-bit judge + KV caches).

set -e

source setup/start_env.sh

# Subset + budget knobs (tune for compute). base.yaml default lambda_max=10; RL likes a bit more.
RL_HB="python scripts/run_inference.py --experiment configs/experiments/base.yaml --attack rl --n-prompts 50 --lambda-max 16 --output-dir $SCRATCH/rup --resume"
RL_JB="python scripts/run_inference.py --experiment configs/experiments/base.yaml --benchmark jailbreakbench --attack rl --n-prompts 50 --lambda-max 16 --output-dir $SCRATCH/rup --resume"

# =============================================================================
# MODEL SIZE STUDY — Qwen2.5-Instruct: 0.5B, 3B, 7B (HarmBench)
# =============================================================================
submit "rup_HB_rl_qwen2.5_0.5b_s1997" "$RL_HB --model qwen2.5_0.5b --seeds 1997"
submit "rup_HB_rl_qwen2.5_3b_s1997"   "$RL_HB --model qwen2.5_3b --seeds 1997"
submit "rup_HB_rl_qwen2.5_7b_s1997"   "$RL_HB --model qwen2.5_7b --seeds 1997"

# =============================================================================
# TRAINING STAGE STUDY — Tulu3 8B: Base -> SFT -> DPO -> RLVR (HarmBench)
# =============================================================================
submit "rup_HB_rl_tulu3_8b_base_s1997" "$RL_HB --model tulu3_8b_base --seeds 1997"
submit "rup_HB_rl_tulu3_8b_sft_s1997"  "$RL_HB --model tulu3_8b_sft --seeds 1997"
submit "rup_HB_rl_tulu3_8b_dpo_s1997"  "$RL_HB --model tulu3_8b_dpo --seeds 1997"
submit "rup_HB_rl_tulu3_8b_rlvr_s1997" "$RL_HB --model tulu3_8b_rlvr --seeds 1997"

# =============================================================================
# JailbreakBench (same targets)
# =============================================================================
submit "rup_JB_rl_qwen2.5_0.5b_s1997" "$RL_JB --model qwen2.5_0.5b --seeds 1997"
