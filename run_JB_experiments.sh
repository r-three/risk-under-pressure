#!/bin/bash
# Submit JailbreakBench per-model pressure sensitivity experiments.
# Usage: bash run_JB_experiments.sh
# Requires: must be run from the project root on a klogin* node.
#
# All runs use configs/experiments/base.yaml; model, attack and seeds are CLI overrides.
#
# ONE JOB PER (MODEL, ATTACK, SEED). Each attack is submitted separately rather than letting
# one job walk the whole attack list, so SLURM can run pair / jailbroken / gcg concurrently
# for the same target instead of serialising them behind each other. GCG is far and away the
# slowest of the three (128 candidate forward passes per step), so under the old one-job-per-
# model layout the cheap attacks finished in minutes and then sat waiting on it inside the
# same 23-hour allocation.
#
# This is safe to parallelise: run_inference.py writes to
#   $RUN_ROOT/<benchmark>/<model_id>/<seed>/<attack>/results.jsonl
# so two attacks on the same target never touch the same file.
#
# Set ATTACKS to run a subset:
#   ATTACKS="gcg"            bash run_JB_experiments.sh   # just the slow one
#   ATTACKS="pair jailbroken" bash run_JB_experiments.sh   # just the fast ones
#
# NOTE ON JOB NAMES: these now carry the attack (rup_JB_<model>_<attack>_s<seed>). Runs
# submitted under the older, attack-less names will not be recognised by should_skip_job, so a
# previously-completed model may be resubmitted. That is cheap, not destructive — every job
# passes --resume, so run_inference.py skips prompts already present in results.jsonl and
# exits almost immediately if there is nothing left to do.
#
# WHAT IS CURRENTLY ENABLED: everything — all 17 targets (Qwen2.5, Gemma 3, Tulu3,
# OLMo 2, Qwen3) x 3 attacks x 10 seeds = 510 jobs. Comment out a submit_model line
# to drop a target, or narrow the sweep with ATTACKS=... (see above).
#
# JUDGE selects the safety judge (a config name under configs/models/, without .yaml).
# It defaults to llama3.1_8b_instruct_judge, which keeps every path exactly as it was before
# judges became selectable. Any other judge writes to its own tree under
# $SCRATCH/rup/judges/<judge_model_id>/ so runs never overwrite:
#   JUDGE=olmo3_7b_instruct_judge     bash run_JB_experiments.sh
#   JUDGE=gemma3_4b_it_judge          bash run_JB_experiments.sh
# See run_judge_ablation.sh to sweep all judges in one go.

set -e

source setup/start_env.sh
source setup/judge_env.sh

# --n-prompts 100 is the whole of JailbreakBench (it has 100 behaviors); base.yaml's 200 is
# a HarmBench number. Do not drop this flag.
BASE="python scripts/run_inference.py --experiment configs/experiments/base.yaml --benchmark jailbreakbench --n-prompts 100 --output-dir $RUN_ROOT --judge-model $JUDGE --resume"

# Static attacks only. The RL/GRPO adaptive attack lives in run_rl_JB_experiments.sh
# because of its per-prompt training phase.
ATTACKS="${ATTACKS:-pair jailbroken gcg}"

# submit_model <job-short-name> <model-config> <seed>
# Fans the model out into one job per attack, so they queue and run independently.
submit_model() {
    local short="$1" cfg="$2" seed="$3"
    local atk
    for atk in $ATTACKS; do
        submit "rup_JB_${short}_${atk}_s${seed}$JUDGE_TAG" \
            "$BASE --model $cfg --attack $atk --seeds $seed"
    done
}


# =============================================================================
# MODEL SIZE STUDY — Qwen2.5-Instruct: 0.5B, 3B, 7B
# Paper: Figure 1 right
# =============================================================================

# --- Qwen2.5-0.5B (also gcg surrogate for attack transfer) ---
submit_model qwen2.5_0.5b qwen2.5_0.5b 1394
submit_model qwen2.5_0.5b qwen2.5_0.5b 2
submit_model qwen2.5_0.5b qwen2.5_0.5b 100
submit_model qwen2.5_0.5b qwen2.5_0.5b 42
submit_model qwen2.5_0.5b qwen2.5_0.5b 5431
submit_model qwen2.5_0.5b qwen2.5_0.5b 2002
submit_model qwen2.5_0.5b qwen2.5_0.5b 256
submit_model qwen2.5_0.5b qwen2.5_0.5b 512
submit_model qwen2.5_0.5b qwen2.5_0.5b 123
submit_model qwen2.5_0.5b qwen2.5_0.5b 5

# --- Qwen2.5-3B ---
submit_model qwen2.5_3b qwen2.5_3b 1394
submit_model qwen2.5_3b qwen2.5_3b 2
submit_model qwen2.5_3b qwen2.5_3b 100
submit_model qwen2.5_3b qwen2.5_3b 42
submit_model qwen2.5_3b qwen2.5_3b 5431
submit_model qwen2.5_3b qwen2.5_3b 2002
submit_model qwen2.5_3b qwen2.5_3b 256
submit_model qwen2.5_3b qwen2.5_3b 512
submit_model qwen2.5_3b qwen2.5_3b 123
submit_model qwen2.5_3b qwen2.5_3b 5

# --- Qwen2.5-7B ---
submit_model qwen2.5_7b qwen2.5_7b 1394
submit_model qwen2.5_7b qwen2.5_7b 2
submit_model qwen2.5_7b qwen2.5_7b 100
submit_model qwen2.5_7b qwen2.5_7b 42
submit_model qwen2.5_7b qwen2.5_7b 5431
submit_model qwen2.5_7b qwen2.5_7b 2002
submit_model qwen2.5_7b qwen2.5_7b 256
submit_model qwen2.5_7b qwen2.5_7b 512
submit_model qwen2.5_7b qwen2.5_7b 123
submit_model qwen2.5_7b qwen2.5_7b 5


# =============================================================================
# MODEL SIZE STUDY (2nd family) — Gemma 3 Instruction-Tuned: 270M, 1B, 4B
# Companion to the Qwen2.5 ladder above (configs/experiments/paper/model_size_gemma3.yaml).
# A second family is what separates "risk falls with size" from "risk falls with size
# in Qwen": different vendor, tokenizer and safety recipe, and it reaches an order of
# magnitude smaller at the bottom rung.
#
# Gemma 3 is the first non-Qwen/non-Llama target to go through the GCG path, and the 4B
# checkpoint is multimodal (loaded text-only via causal_lm — see configs/models/
# gemma3_4b_it.yaml). Watch the gcg job for the 4B arm before uncommenting more seeds.
# =============================================================================

# --- Gemma3-270M ---
submit_model gemma3_270m gemma3_270m_it 1394
submit_model gemma3_270m gemma3_270m_it 2
submit_model gemma3_270m gemma3_270m_it 100
submit_model gemma3_270m gemma3_270m_it 42
submit_model gemma3_270m gemma3_270m_it 5431
submit_model gemma3_270m gemma3_270m_it 2002
submit_model gemma3_270m gemma3_270m_it 256
submit_model gemma3_270m gemma3_270m_it 512
submit_model gemma3_270m gemma3_270m_it 123
submit_model gemma3_270m gemma3_270m_it 5

# --- Gemma3-1B ---
submit_model gemma3_1b gemma3_1b_it 1394
submit_model gemma3_1b gemma3_1b_it 2
submit_model gemma3_1b gemma3_1b_it 100
submit_model gemma3_1b gemma3_1b_it 42
submit_model gemma3_1b gemma3_1b_it 5431
submit_model gemma3_1b gemma3_1b_it 2002
submit_model gemma3_1b gemma3_1b_it 256
submit_model gemma3_1b gemma3_1b_it 512
submit_model gemma3_1b gemma3_1b_it 123
submit_model gemma3_1b gemma3_1b_it 5

# --- Gemma3-4B ---
submit_model gemma3_4b gemma3_4b_it 1394
submit_model gemma3_4b gemma3_4b_it 2
submit_model gemma3_4b gemma3_4b_it 100
submit_model gemma3_4b gemma3_4b_it 42
submit_model gemma3_4b gemma3_4b_it 5431
submit_model gemma3_4b gemma3_4b_it 2002
submit_model gemma3_4b gemma3_4b_it 256
submit_model gemma3_4b gemma3_4b_it 512
submit_model gemma3_4b gemma3_4b_it 123
submit_model gemma3_4b gemma3_4b_it 5


# =============================================================================
# TRAINING STAGE STUDY — Tulu3 8B: Base -> SFT -> DPO -> RLVR
# Paper: Table 1, Figure 1 left
# =============================================================================

# --- Tulu3-8B Base ---
submit_model tulu3_8b_base tulu3_8b_base 1394
submit_model tulu3_8b_base tulu3_8b_base 2
submit_model tulu3_8b_base tulu3_8b_base 100
submit_model tulu3_8b_base tulu3_8b_base 42
submit_model tulu3_8b_base tulu3_8b_base 5431
submit_model tulu3_8b_base tulu3_8b_base 2002
submit_model tulu3_8b_base tulu3_8b_base 256
submit_model tulu3_8b_base tulu3_8b_base 512
submit_model tulu3_8b_base tulu3_8b_base 123
submit_model tulu3_8b_base tulu3_8b_base 5

# --- Tulu3-8B SFT ---
submit_model tulu3_8b_sft tulu3_8b_sft 1394
submit_model tulu3_8b_sft tulu3_8b_sft 2
submit_model tulu3_8b_sft tulu3_8b_sft 100
submit_model tulu3_8b_sft tulu3_8b_sft 42
submit_model tulu3_8b_sft tulu3_8b_sft 5431
submit_model tulu3_8b_sft tulu3_8b_sft 2002
submit_model tulu3_8b_sft tulu3_8b_sft 256
submit_model tulu3_8b_sft tulu3_8b_sft 512
submit_model tulu3_8b_sft tulu3_8b_sft 123
submit_model tulu3_8b_sft tulu3_8b_sft 5

# --- Tulu3-8B DPO ---
submit_model tulu3_8b_dpo tulu3_8b_dpo 1394
submit_model tulu3_8b_dpo tulu3_8b_dpo 2
submit_model tulu3_8b_dpo tulu3_8b_dpo 100
submit_model tulu3_8b_dpo tulu3_8b_dpo 42
submit_model tulu3_8b_dpo tulu3_8b_dpo 5431
submit_model tulu3_8b_dpo tulu3_8b_dpo 2002
submit_model tulu3_8b_dpo tulu3_8b_dpo 256
submit_model tulu3_8b_dpo tulu3_8b_dpo 512
submit_model tulu3_8b_dpo tulu3_8b_dpo 123
submit_model tulu3_8b_dpo tulu3_8b_dpo 5

# --- Tulu3-8B RLVR ---
submit_model tulu3_8b_rlvr tulu3_8b_rlvr 1394
submit_model tulu3_8b_rlvr tulu3_8b_rlvr 2
submit_model tulu3_8b_rlvr tulu3_8b_rlvr 100
submit_model tulu3_8b_rlvr tulu3_8b_rlvr 42
submit_model tulu3_8b_rlvr tulu3_8b_rlvr 5431
submit_model tulu3_8b_rlvr tulu3_8b_rlvr 2002
submit_model tulu3_8b_rlvr tulu3_8b_rlvr 256
submit_model tulu3_8b_rlvr tulu3_8b_rlvr 512
submit_model tulu3_8b_rlvr tulu3_8b_rlvr 123
submit_model tulu3_8b_rlvr tulu3_8b_rlvr 5


# =============================================================================
# TRAINING STAGE STUDY (2nd ladder) — OLMo 2 1B: Base -> SFT -> DPO -> RLVR1 -> Instruct
# Companion to the Tulu3 8B ladder above (configs/experiments/paper/training_stage_olmo2.yaml).
#
# FIVE rungs, not four. allenai's `-Instruct` is a SECOND RLVR round (on RLVR-MATH) stacked
# on `-RLVR1` (on RLVR-GSM-MATH-IF-Mixed-Constraints), not a rename of it — so this is the
# only ladder here that shows whether a further round of RLVR keeps moving the risk curve
# or whether the effect saturates after the first.
#
# Absolute risk is NOT comparable to the 8B Tulu3 ladder (1.48B vs 8B). What compares is the
# shape of each progression relative to its own base rung. All five load unquantized, so no
# rung differs from another in precision.
# =============================================================================

# --- OLMo2-1B Base (pre-trained only) ---
submit_model olmo2_1b_base olmo2_1b_base 1394
submit_model olmo2_1b_base olmo2_1b_base 2
submit_model olmo2_1b_base olmo2_1b_base 100
submit_model olmo2_1b_base olmo2_1b_base 42
submit_model olmo2_1b_base olmo2_1b_base 5431
submit_model olmo2_1b_base olmo2_1b_base 2002
submit_model olmo2_1b_base olmo2_1b_base 256
submit_model olmo2_1b_base olmo2_1b_base 512
submit_model olmo2_1b_base olmo2_1b_base 123
submit_model olmo2_1b_base olmo2_1b_base 5

# --- OLMo2-1B SFT ---
submit_model olmo2_1b_sft olmo2_1b_sft 1394
submit_model olmo2_1b_sft olmo2_1b_sft 2
submit_model olmo2_1b_sft olmo2_1b_sft 100
submit_model olmo2_1b_sft olmo2_1b_sft 42
submit_model olmo2_1b_sft olmo2_1b_sft 5431
submit_model olmo2_1b_sft olmo2_1b_sft 2002
submit_model olmo2_1b_sft olmo2_1b_sft 256
submit_model olmo2_1b_sft olmo2_1b_sft 512
submit_model olmo2_1b_sft olmo2_1b_sft 123
submit_model olmo2_1b_sft olmo2_1b_sft 5

# --- OLMo2-1B DPO ---
submit_model olmo2_1b_dpo olmo2_1b_dpo 1394
submit_model olmo2_1b_dpo olmo2_1b_dpo 2
submit_model olmo2_1b_dpo olmo2_1b_dpo 100
submit_model olmo2_1b_dpo olmo2_1b_dpo 42
submit_model olmo2_1b_dpo olmo2_1b_dpo 5431
submit_model olmo2_1b_dpo olmo2_1b_dpo 2002
submit_model olmo2_1b_dpo olmo2_1b_dpo 256
submit_model olmo2_1b_dpo olmo2_1b_dpo 512
submit_model olmo2_1b_dpo olmo2_1b_dpo 123
submit_model olmo2_1b_dpo olmo2_1b_dpo 5

# --- OLMo2-1B RLVR round 1 ---
submit_model olmo2_1b_rlvr1 olmo2_1b_rlvr1 1394
submit_model olmo2_1b_rlvr1 olmo2_1b_rlvr1 2
submit_model olmo2_1b_rlvr1 olmo2_1b_rlvr1 100
submit_model olmo2_1b_rlvr1 olmo2_1b_rlvr1 42
submit_model olmo2_1b_rlvr1 olmo2_1b_rlvr1 5431
submit_model olmo2_1b_rlvr1 olmo2_1b_rlvr1 2002
submit_model olmo2_1b_rlvr1 olmo2_1b_rlvr1 256
submit_model olmo2_1b_rlvr1 olmo2_1b_rlvr1 512
submit_model olmo2_1b_rlvr1 olmo2_1b_rlvr1 123
submit_model olmo2_1b_rlvr1 olmo2_1b_rlvr1 5

# --- OLMo2-1B Instruct = RLVR round 2 (final release) ---
submit_model olmo2_1b_instruct olmo2_1b_instruct 1394
submit_model olmo2_1b_instruct olmo2_1b_instruct 2
submit_model olmo2_1b_instruct olmo2_1b_instruct 100
submit_model olmo2_1b_instruct olmo2_1b_instruct 42
submit_model olmo2_1b_instruct olmo2_1b_instruct 5431
submit_model olmo2_1b_instruct olmo2_1b_instruct 2002
submit_model olmo2_1b_instruct olmo2_1b_instruct 256
submit_model olmo2_1b_instruct olmo2_1b_instruct 512
submit_model olmo2_1b_instruct olmo2_1b_instruct 123
submit_model olmo2_1b_instruct olmo2_1b_instruct 5


# =============================================================================
# SAFETY ALIGNMENT STUDY — Qwen3-4B base vs Qwen3-4B-SafeRL
# Paper: Table 1 (Qwen3 rows)
# =============================================================================

# --- Qwen3-4B ---
submit_model qwen3_4b qwen3_4b 1394
submit_model qwen3_4b qwen3_4b 2
submit_model qwen3_4b qwen3_4b 100
submit_model qwen3_4b qwen3_4b 42
submit_model qwen3_4b qwen3_4b 5431
submit_model qwen3_4b qwen3_4b 2002
submit_model qwen3_4b qwen3_4b 256
submit_model qwen3_4b qwen3_4b 512
submit_model qwen3_4b qwen3_4b 123
submit_model qwen3_4b qwen3_4b 5

# --- Qwen3-4B-SafeRL ---
submit_model qwen3_4b_saferl qwen3_4b_saferl 1394
submit_model qwen3_4b_saferl qwen3_4b_saferl 2
submit_model qwen3_4b_saferl qwen3_4b_saferl 100
submit_model qwen3_4b_saferl qwen3_4b_saferl 42
submit_model qwen3_4b_saferl qwen3_4b_saferl 5431
submit_model qwen3_4b_saferl qwen3_4b_saferl 2002
submit_model qwen3_4b_saferl qwen3_4b_saferl 256
submit_model qwen3_4b_saferl qwen3_4b_saferl 512
submit_model qwen3_4b_saferl qwen3_4b_saferl 123
submit_model qwen3_4b_saferl qwen3_4b_saferl 5


echo
echo "Submitted: attacks [$ATTACKS] x the enabled models above."
echo "Watch:  squeue -u $USER --noheader -o '%j %t' | grep rup_JB"
