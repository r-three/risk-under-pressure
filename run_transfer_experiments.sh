#!/bin/bash
# Submit attack transfer experiments to Killarney L40S GPU.
#
# Replays pre-computed GCG trajectories from a source model against all target
# models. One job per (source, target, benchmark, seed-batch). Results land in:
#   $OUTPUT/{benchmark}/{target_model}_seed{seed}/transfer_gcg_from_{source_model}/
#
# Usage: bash run_transfer_experiments.sh
# Requires: must be run from the project root on a klogin* node.

set -e

source setup/start_env.sh

# Source results live in the project BASE (one model-level dir per model).
# Transfer outputs go to the scratch OUTPUT tree.
BASE=/home/ehghaghi/projects/aip-craffel/ehghaghi/prisk-pressure/results/multitrial_experiments/pressure_sensitivity
OUTPUT=/home/ehghaghi/scratch/aip-craffel/ehghaghi/prisk-pressure/results/multitrial_experiments/pressure_sensitivity

# Seed batches — same grouping as run_HB/JB_experiments.sh
# b1: 1997 2 100 | b2: 42 5431 2002 | b3: 256 512 123 | b4: 5

# ============================================================
# Source: tulu3-8b-sft  (GCG → all other models)
# ============================================================

SRC_HB_T3SFT="--experiment configs/experiments/HB_tulu3_8b_sft.yaml --source-results-dir $BASE/harmbench/tulu3-8b-sft --source-model tulu3-8b-sft --output-dir $OUTPUT --resume"
SRC_JB_T3SFT="--experiment configs/experiments/JB_tulu3_8b_sft.yaml --source-results-dir $BASE/jailbreakbench/tulu3-8b-sft --source-model tulu3-8b-sft --output-dir $OUTPUT --resume"

# ============================================================
# Source: tulu3-8b-dpo → Qwen3-8B
# (transfer to models without weight access; text-only mode)
# ============================================================

SRC_HB_T3DPO="--experiment configs/experiments/HB_tulu3_8b_dpo.yaml --source-results-dir $BASE/harmbench/tulu3-8b-dpo --source-model tulu3-8b-dpo --output-dir $OUTPUT --resume"
SRC_JB_T3DPO="--experiment configs/experiments/JB_tulu3_8b_dpo.yaml --source-results-dir $BASE/jailbreakbench/tulu3-8b-dpo --source-model tulu3-8b-dpo --output-dir $OUTPUT --resume"

# --- tulu3-8b-dpo → Qwen3-8B (text-only: enable_thinking=false in config) ---
# submit "rup_tr_gcg_t3dpo_q3_8b_HB_b1" "python scripts/run_transfer_inference.py $SRC_HB_T3DPO --target-models qwen3_8b --seeds 1997 2 100"
# submit "rup_tr_gcg_t3dpo_q3_8b_HB_b2" "python scripts/run_transfer_inference.py $SRC_HB_T3DPO --target-models qwen3_8b --seeds 42 5431 2002"
# submit "rup_tr_gcg_t3dpo_q3_8b_HB_b3" "python scripts/run_transfer_inference.py $SRC_HB_T3DPO --target-models qwen3_8b --seeds 256 512 123"
# submit "rup_tr_gcg_t3dpo_q3_8b_HB_b4" "python scripts/run_transfer_inference.py $SRC_HB_T3DPO --target-models qwen3_8b --seeds 5"
# submit "rup_tr_gcg_t3dpo_q3_8b_JB_b1" "python scripts/run_transfer_inference.py $SRC_JB_T3DPO --target-models qwen3_8b --seeds 1997 2 100"
# submit "rup_tr_gcg_t3dpo_q3_8b_JB_b2" "python scripts/run_transfer_inference.py $SRC_JB_T3DPO --target-models qwen3_8b --seeds 42 5431 2002"
# submit "rup_tr_gcg_t3dpo_q3_8b_JB_b3" "python scripts/run_transfer_inference.py $SRC_JB_T3DPO --target-models qwen3_8b --seeds 256 512 123"
# submit "rup_tr_gcg_t3dpo_q3_8b_JB_b4" "python scripts/run_transfer_inference.py $SRC_JB_T3DPO --target-models qwen3_8b --seeds 5"

# ============================================================
# Source: qwen2.5-0.5b-instruct → Qwen3-8B
# ============================================================

SRC_HB_Q25_05B="--experiment configs/experiments/HB_qwen2.5_0.5b.yaml --source-results-dir $BASE/harmbench/qwen2.5-0.5b-instruct --source-model qwen2.5-0.5b-instruct --output-dir $OUTPUT --resume"
SRC_JB_Q25_05B="--experiment configs/experiments/JB_qwen2.5_0.5b.yaml --source-results-dir $BASE/jailbreakbench/qwen2.5-0.5b-instruct --source-model qwen2.5-0.5b-instruct --output-dir $OUTPUT --resume"

# --- qwen2.5-0.5b-instruct → qwen3-8b ---
submit "rup_tr_gcg_q25_05b_q3_8b_HB_b1" "python scripts/run_transfer_inference.py $SRC_HB_Q25_05B --target-models qwen3_8b --seeds 1997 2 100"
submit "rup_tr_gcg_q25_05b_q3_8b_HB_b2" "python scripts/run_transfer_inference.py $SRC_HB_Q25_05B --target-models qwen3_8b --seeds 42 5431 2002"
submit "rup_tr_gcg_q25_05b_q3_8b_HB_b3" "python scripts/run_transfer_inference.py $SRC_HB_Q25_05B --target-models qwen3_8b --seeds 256 512 123"
submit "rup_tr_gcg_q25_05b_q3_8b_HB_b4" "python scripts/run_transfer_inference.py $SRC_HB_Q25_05B --target-models qwen3_8b --seeds 5"
submit "rup_tr_gcg_q25_05b_q3_8b_JB_b1" "python scripts/run_transfer_inference.py $SRC_JB_Q25_05B --target-models qwen3_8b --seeds 1997 2 100"
submit "rup_tr_gcg_q25_05b_q3_8b_JB_b2" "python scripts/run_transfer_inference.py $SRC_JB_Q25_05B --target-models qwen3_8b --seeds 42 5431 2002"
submit "rup_tr_gcg_q25_05b_q3_8b_JB_b3" "python scripts/run_transfer_inference.py $SRC_JB_Q25_05B --target-models qwen3_8b --seeds 256 512 123"
submit "rup_tr_gcg_q25_05b_q3_8b_JB_b4" "python scripts/run_transfer_inference.py $SRC_JB_Q25_05B --target-models qwen3_8b --seeds 5"

