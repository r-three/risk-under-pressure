#!/bin/bash
# Submit the per-prompt RL/GRPO adaptive-attack experiments on HarmBench.
# Adaptive attacker following "The Attacker Moves Second" (Nasr et al., 2025, arXiv:2510.09023,
# App. A.2): for EACH behavior, Qwen2.5-7B-Instruct runs a short GRPO optimization against the
# target. A SESSION is 5 sequential in-context refinements scored best-of-5; a GROUP is 8
# independent sessions, over whose returns the group-relative policy gradient is taken.
#
# BUDGET is set in GROUPS (--rl-groups). One group = 8 sessions x 5 rounds = 40 rollouts, and
# closing a group is what triggers a weight update, so N groups gives N-1 updates per behavior
# (the last group's update is skipped — the adapter resets before anything could sample from it).
# --rl-groups 2 => lambda_max 81 (1 raw probe + 80 rollouts) and 1 GRPO update. That is the
# smallest budget at which an update can influence a scored query; at 1 group the attack is
# best-of-N sampling from an untrained attacker and should not be reported as RL.
#
# Reporting a SMALLER lambda needs no separate run: risk and cost both truncate to
# steps[0:lambda], so this run yields the whole curve for every lambda <= 81. Scale up with
# --rl-groups 5 / 10 (lambda_max 201 / 401) once the 2-group arm shows the attacker moving.
#
# EARLY STOP: a behavior ends at the first GROUP BOUNDARY where a confirmed jailbreak (binary
# label) also clears stop_score_threshold on the continuous scorer (configs/attacks/rl.yaml).
# Group boundaries only, so no session loses its best-of-N and no GRPO group is left partial.
# Requiring the binary label means a recorded success always exists when we stop, so risk at
# every lambda is untouched and only post-success queries are saved — the same semantics as
# PAIR's and GCG's early stop, which keeps RL's cost axis comparable to theirs.
#
# There is NO separate training phase — per-prompt GRPO happens inside run_inference.py, so RL is
# submitted just like any other attack (e.g. gcg), one job per (model, seed).
#
# Usage: bash run_rl_HB_experiments.sh
# Requires: must be run from the project root on a klogin* node.
#
# COMPUTE WARNING: this submits 170 JOBS (17 models x 10 seeds), and per-prompt GRPO is by far
# the most expensive attack here. Per behavior: up to 81 target queries, 3 judge passes per query
# (one verdict generation + two prefill passes for the continuous P(UNSAFE) the reward needs), and
# 40 attacker backward passes for group 1's GRPO update. Across the sweep that is roughly
# 2.8M target generations, 8.3M judge passes and 1.4M 7B backward passes — before the
# group-boundary early stop trims it, which it will for every behavior that jailbreaks.
#
# n-prompts=200 is the full HarmBench set; 17 targets x 10 seeds is the full grid. Trim with
# SEEDS="1394" (back to the single-seed sweep) or by commenting out targets you don't need.
# Watch 48 GB VRAM (7B bf16 attacker + LoRA optimizer + 4-bit target + 4-bit judge + KV caches).
#
# JUDGE selects the safety judge (a config name under configs/models/, without .yaml).
# It defaults to llama3.1_8b_instruct_judge, which keeps every path and job name
# exactly as it was before judges became selectable. Any other judge writes to its
# own tree under $SCRATCH/rup/judges/<judge_model_id>/ so runs never overwrite:
#   JUDGE=olmo3_7b_instruct_judge     bash run_rl_HB_experiments.sh
#   JUDGE=gemma3_4b_it_judge          bash run_rl_HB_experiments.sh
# See run_judge_ablation.sh to sweep all judges in one go.

set -e

source setup/start_env.sh
source setup/judge_env.sh

# Subset + budget knobs. --rl-groups 2 = 1 raw probe + 2 groups of 8 sessions x 5
# rounds = lambda_max 81; see configs/attacks/rl.yaml. One group means no GRPO at all.
RL_HB="python scripts/run_inference.py --experiment configs/experiments/base.yaml --attack rl --n-prompts 200 --rl-groups 2 --output-dir $RUN_ROOT --judge-model $JUDGE --resume"

# Seeds. Each (model, seed) is submitted as its OWN job so they queue and run independently —
# 10 seeds means 10x the jobs, not one 10x-longer job. 1394 is first and unchanged, so results
# already on disk for it resume rather than re-run (every command below keeps --resume).
# Override to run a subset:  SEEDS="1394 42" bash run_rl_HB_experiments.sh
SEEDS="${SEEDS:-1394 42 123 256 512 1024 1997 2002 5431 7919}"

# submit_rl <job-short-name> <model-config>
# Fans one model out into one job per seed.
submit_rl() {
    local short="$1" cfg="$2" seed
    for seed in $SEEDS; do
        submit "rup_HB_rl_${short}_s${seed}$JUDGE_TAG" \
            "$RL_HB --model $cfg --seeds $seed"
    done
}

# =============================================================================
# MODEL SIZE STUDY — Qwen2.5-Instruct: 0.5B, 3B, 7B
# =============================================================================
submit_rl "qwen2.5_0.5b" "qwen2.5_0.5b"
submit_rl "qwen2.5_3b" "qwen2.5_3b"
submit_rl "qwen2.5_7b" "qwen2.5_7b"

# =============================================================================
# MODEL SIZE STUDY (2nd family) — Gemma 3 Instruction-Tuned: 270M, 1B, 4B
# Companion to the Qwen2.5 ladder above. Does the adaptive attacker's advantage over the
# static attacks hold across a second family, or is it a Qwen artifact?
#
# The RL attacker (Qwen2.5-7B bf16 + LoRA) is unchanged; only the target family differs.
# Two things to watch on the first run:
#   - 270M is far smaller than any target this attack has faced. A tiny target is easy to
#     break but also easy to make incoherent; check the responses, not just the ASR.
#   - the shaping reward calls target.sequence_nll, which needs a causal_lm. All three
#     Gemma configs take that path (including 4B, which is multimodal on the hub) — see
#     configs/models/gemma3_4b_it.yaml.
# =============================================================================
submit_rl "gemma3_270m" "gemma3_270m_it"
submit_rl "gemma3_1b" "gemma3_1b_it"
submit_rl "gemma3_4b" "gemma3_4b_it"

# =============================================================================
# TRAINING STAGE STUDY — Tulu3 8B: Base -> SFT -> DPO -> RLVR
# =============================================================================
submit_rl "tulu3_8b_base" "tulu3_8b_base"
submit_rl "tulu3_8b_sft" "tulu3_8b_sft"
submit_rl "tulu3_8b_dpo" "tulu3_8b_dpo"
submit_rl "tulu3_8b_rlvr" "tulu3_8b_rlvr"

# =============================================================================
# TRAINING STAGE STUDY (2nd ladder) — OLMo 2 1B: Base -> SFT -> DPO -> RLVR1 -> Instruct
# Companion to the Tulu3 8B ladder above. Five rungs: `-Instruct` is a SECOND RLVR round
# stacked on `-RLVR1`, so this is the only ladder here where the marginal effect of one
# further RLVR round on adaptive-attack resistance is visible.
#
# The base rung has no chat template and no safety training at all; expect it to fall fast
# and to produce rambling continuations rather than refusals. That is the control, not a bug.
# =============================================================================
submit_rl "olmo2_1b_base" "olmo2_1b_base"
submit_rl "olmo2_1b_sft" "olmo2_1b_sft"
submit_rl "olmo2_1b_dpo" "olmo2_1b_dpo"
submit_rl "olmo2_1b_rlvr1" "olmo2_1b_rlvr1"
submit_rl "olmo2_1b_instruct" "olmo2_1b_instruct"

# =============================================================================
# SAFETY ALIGNMENT STUDY — Qwen3-4B base vs Qwen3-4B-SafeRL
# =============================================================================
submit_rl "qwen3_4b" "qwen3_4b"
submit_rl "qwen3_4b_saferl" "qwen3_4b_saferl"
