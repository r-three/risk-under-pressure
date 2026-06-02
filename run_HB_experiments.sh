#!/bin/bash
# Submit HarmBench per-model pressure sensitivity experiments.
# Usage: bash run_HB_experiments.sh
# Requires: must be run from the project root on a klogin* node.
#
# All runs use configs/experiments/base.yaml; model and seeds are CLI overrides.
# Each seed is submitted as a separate job for fine-grained control.
# To add a new model, add a 10-line block following the same pattern.

set -e

source setup/start_env.sh

BASE="python scripts/run_inference.py --experiment configs/experiments/base.yaml --output-dir $SCRATCH/rup --resume"

# =============================================================================
# MODEL SIZE STUDY — Qwen2.5-Instruct: 0.5B, 3B, 7B
# Paper: Figure 1 right
# =============================================================================

# --- Qwen2.5-0.5B (also gcg surrogate for attack transfer) ---
# submit "rup_HB_qwen2.5_0.5b_s1997" "$BASE --model qwen2.5_0.5b --seeds 1997"
# submit "rup_HB_qwen2.5_0.5b_s2"    "$BASE --model qwen2.5_0.5b --seeds 2"
# submit "rup_HB_qwen2.5_0.5b_s100"  "$BASE --model qwen2.5_0.5b --seeds 100"
# submit "rup_HB_qwen2.5_0.5b_s42"   "$BASE --model qwen2.5_0.5b --seeds 42"
# submit "rup_HB_qwen2.5_0.5b_s5431" "$BASE --model qwen2.5_0.5b --seeds 5431"
# submit "rup_HB_qwen2.5_0.5b_s2002" "$BASE --model qwen2.5_0.5b --seeds 2002"
# submit "rup_HB_qwen2.5_0.5b_s256"  "$BASE --model qwen2.5_0.5b --seeds 256"
# submit "rup_HB_qwen2.5_0.5b_s512"  "$BASE --model qwen2.5_0.5b --seeds 512"
# submit "rup_HB_qwen2.5_0.5b_s123"  "$BASE --model qwen2.5_0.5b --seeds 123"
# submit "rup_HB_qwen2.5_0.5b_s5"    "$BASE --model qwen2.5_0.5b --seeds 5"

# --- Qwen2.5-3B ---
# submit "rup_HB_qwen2.5_3b_s1997" "$BASE --model qwen2.5_3b --seeds 1997"
# submit "rup_HB_qwen2.5_3b_s2"    "$BASE --model qwen2.5_3b --seeds 2"
# submit "rup_HB_qwen2.5_3b_s100"  "$BASE --model qwen2.5_3b --seeds 100"
# submit "rup_HB_qwen2.5_3b_s42"   "$BASE --model qwen2.5_3b --seeds 42"
# submit "rup_HB_qwen2.5_3b_s5431" "$BASE --model qwen2.5_3b --seeds 5431"
# submit "rup_HB_qwen2.5_3b_s2002" "$BASE --model qwen2.5_3b --seeds 2002"
# submit "rup_HB_qwen2.5_3b_s256"  "$BASE --model qwen2.5_3b --seeds 256"
# submit "rup_HB_qwen2.5_3b_s512"  "$BASE --model qwen2.5_3b --seeds 512"
# submit "rup_HB_qwen2.5_3b_s123"  "$BASE --model qwen2.5_3b --seeds 123"
# submit "rup_HB_qwen2.5_3b_s5"    "$BASE --model qwen2.5_3b --seeds 5"

# --- Qwen2.5-7B ---
# submit "rup_HB_qwen2.5_7b_s1997" "$BASE --model qwen2.5_7b --seeds 1997"
# submit "rup_HB_qwen2.5_7b_s2"    "$BASE --model qwen2.5_7b --seeds 2"
# submit "rup_HB_qwen2.5_7b_s100"  "$BASE --model qwen2.5_7b --seeds 100"
# submit "rup_HB_qwen2.5_7b_s42"   "$BASE --model qwen2.5_7b --seeds 42"
# submit "rup_HB_qwen2.5_7b_s5431" "$BASE --model qwen2.5_7b --seeds 5431"
# submit "rup_HB_qwen2.5_7b_s2002" "$BASE --model qwen2.5_7b --seeds 2002"
# submit "rup_HB_qwen2.5_7b_s256"  "$BASE --model qwen2.5_7b --seeds 256"
# submit "rup_HB_qwen2.5_7b_s512"  "$BASE --model qwen2.5_7b --seeds 512"
# submit "rup_HB_qwen2.5_7b_s123"  "$BASE --model qwen2.5_7b --seeds 123"
# submit "rup_HB_qwen2.5_7b_s5"    "$BASE --model qwen2.5_7b --seeds 5"

# =============================================================================
# TRAINING STAGE STUDY — Tulu3 8B: Base → SFT → DPO → RLVR
# Paper: Table 1, Figure 1 left
# =============================================================================

# --- Tulu3-8B Base ---
# submit "rup_HB_tulu3_8b_base_s1997" "$BASE --model tulu3_8b_base --seeds 1997"
# submit "rup_HB_tulu3_8b_base_s2"    "$BASE --model tulu3_8b_base --seeds 2   "
# submit "rup_HB_tulu3_8b_base_s100"  "$BASE --model tulu3_8b_base --seeds 100 "
# submit "rup_HB_tulu3_8b_base_s42"   "$BASE --model tulu3_8b_base --seeds 42  "
# submit "rup_HB_tulu3_8b_base_s5431" "$BASE --model tulu3_8b_base --seeds 5431"
# submit "rup_HB_tulu3_8b_base_s2002" "$BASE --model tulu3_8b_base --seeds 2002"
# submit "rup_HB_tulu3_8b_base_s256"  "$BASE --model tulu3_8b_base --seeds 256 "
# submit "rup_HB_tulu3_8b_base_s512"  "$BASE --model tulu3_8b_base --seeds 512 "
# submit "rup_HB_tulu3_8b_base_s123"  "$BASE --model tulu3_8b_base --seeds 123 "
# submit "rup_HB_tulu3_8b_base_s5"    "$BASE --model tulu3_8b_base --seeds 5   "

# --- Tulu3-8B SFT ---
# submit "rup_HB_tulu3_8b_sft_s1997" "$BASE --model tulu3_8b_sft --seeds 1997"
# submit "rup_HB_tulu3_8b_sft_s2"    "$BASE --model tulu3_8b_sft --seeds 2   "
# submit "rup_HB_tulu3_8b_sft_s100"  "$BASE --model tulu3_8b_sft --seeds 100 "
# submit "rup_HB_tulu3_8b_sft_s42"   "$BASE --model tulu3_8b_sft --seeds 42  "
# submit "rup_HB_tulu3_8b_sft_s5431" "$BASE --model tulu3_8b_sft --seeds 5431"
# submit "rup_HB_tulu3_8b_sft_s2002" "$BASE --model tulu3_8b_sft --seeds 2002"
# submit "rup_HB_tulu3_8b_sft_s256"  "$BASE --model tulu3_8b_sft --seeds 256 "
# submit "rup_HB_tulu3_8b_sft_s512"  "$BASE --model tulu3_8b_sft --seeds 512 "
# submit "rup_HB_tulu3_8b_sft_s123"  "$BASE --model tulu3_8b_sft --seeds 123 "
# submit "rup_HB_tulu3_8b_sft_s5"    "$BASE --model tulu3_8b_sft --seeds 5   "

# --- Tulu3-8B DPO ---
# submit "rup_HB_tulu3_8b_dpo_s1997" "$BASE --model tulu3_8b_dpo --seeds 1997"
# submit "rup_HB_tulu3_8b_dpo_s2"    "$BASE --model tulu3_8b_dpo --seeds 2   "
# submit "rup_HB_tulu3_8b_dpo_s100"  "$BASE --model tulu3_8b_dpo --seeds 100 "
# submit "rup_HB_tulu3_8b_dpo_s42"   "$BASE --model tulu3_8b_dpo --seeds 42  "
# submit "rup_HB_tulu3_8b_dpo_s5431" "$BASE --model tulu3_8b_dpo --seeds 5431"
# submit "rup_HB_tulu3_8b_dpo_s2002" "$BASE --model tulu3_8b_dpo --seeds 2002"
# submit "rup_HB_tulu3_8b_dpo_s256"  "$BASE --model tulu3_8b_dpo --seeds 256 "
# submit "rup_HB_tulu3_8b_dpo_s512"  "$BASE --model tulu3_8b_dpo --seeds 512 "
# submit "rup_HB_tulu3_8b_dpo_s123"  "$BASE --model tulu3_8b_dpo --seeds 123 "
# submit "rup_HB_tulu3_8b_dpo_s5"    "$BASE --model tulu3_8b_dpo --seeds 5   "

# --- Tulu3-8B RLVR ---
# submit "rup_HB_tulu3_8b_rlvr_s1997" "$BASE --model tulu3_8b_rlvr --seeds 1997"
# submit "rup_HB_tulu3_8b_rlvr_s2"    "$BASE --model tulu3_8b_rlvr --seeds 2   "
# submit "rup_HB_tulu3_8b_rlvr_s100"  "$BASE --model tulu3_8b_rlvr --seeds 100 "
# submit "rup_HB_tulu3_8b_rlvr_s42"   "$BASE --model tulu3_8b_rlvr --seeds 42  "
# submit "rup_HB_tulu3_8b_rlvr_s5431" "$BASE --model tulu3_8b_rlvr --seeds 5431"
# submit "rup_HB_tulu3_8b_rlvr_s2002" "$BASE --model tulu3_8b_rlvr --seeds 2002"
# submit "rup_HB_tulu3_8b_rlvr_s256"  "$BASE --model tulu3_8b_rlvr --seeds 256 "
# submit "rup_HB_tulu3_8b_rlvr_s512"  "$BASE --model tulu3_8b_rlvr --seeds 512 "
# submit "rup_HB_tulu3_8b_rlvr_s123"  "$BASE --model tulu3_8b_rlvr --seeds 123 "
# submit "rup_HB_tulu3_8b_rlvr_s5"    "$BASE --model tulu3_8b_rlvr --seeds 5   "


# =============================================================================
# SAFETY ALIGNMENT STUDY — Qwen3-4B base vs Qwen3-4B-SafeRL
# Paper: Table 1 (Qwen3 rows)
# =============================================================================

# --- Qwen3-4B (base) ---
submit "rup_HB_qwen3_4b_s1997" "$BASE --model qwen3_4b --seeds 1997"
submit "rup_HB_qwen3_4b_s2"    "$BASE --model qwen3_4b --seeds 2"
submit "rup_HB_qwen3_4b_s100"  "$BASE --model qwen3_4b --seeds 100"
submit "rup_HB_qwen3_4b_s42"   "$BASE --model qwen3_4b --seeds 42"
submit "rup_HB_qwen3_4b_s5431" "$BASE --model qwen3_4b --seeds 5431"
submit "rup_HB_qwen3_4b_s2002" "$BASE --model qwen3_4b --seeds 2002"
submit "rup_HB_qwen3_4b_s256"  "$BASE --model qwen3_4b --seeds 256"
submit "rup_HB_qwen3_4b_s512"  "$BASE --model qwen3_4b --seeds 512"
submit "rup_HB_qwen3_4b_s123"  "$BASE --model qwen3_4b --seeds 123"
# submit "rup_HB_qwen3_4b_s5"    "$BASE --model qwen3_4b --seeds 5"

# --- Qwen3-4B-SafeRL ---
# submit "rup_HB_qwen3_4b_saferl_s1997" "$BASE --model qwen3_4b_saferl --seeds 1997"
# submit "rup_HB_qwen3_4b_saferl_s2"    "$BASE --model qwen3_4b_saferl --seeds 2"
# submit "rup_HB_qwen3_4b_saferl_s100"  "$BASE --model qwen3_4b_saferl --seeds 100"
# submit "rup_HB_qwen3_4b_saferl_s42"   "$BASE --model qwen3_4b_saferl --seeds 42"
# submit "rup_HB_qwen3_4b_saferl_s5431" "$BASE --model qwen3_4b_saferl --seeds 5431"
# submit "rup_HB_qwen3_4b_saferl_s2002" "$BASE --model qwen3_4b_saferl --seeds 2002"
# submit "rup_HB_qwen3_4b_saferl_s256"  "$BASE --model qwen3_4b_saferl --seeds 256"
# submit "rup_HB_qwen3_4b_saferl_s512"  "$BASE --model qwen3_4b_saferl --seeds 512"
# submit "rup_HB_qwen3_4b_saferl_s123"  "$BASE --model qwen3_4b_saferl --seeds 123"
# submit "rup_HB_qwen3_4b_saferl_s5"    "$BASE --model qwen3_4b_saferl --seeds 5"
