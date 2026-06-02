#!/bin/bash
# Submit attack transfer experiments to Killarney L40S GPU.
#
# Replays pre-computed GCG trajectories from a source model against target models.
# Source traces are read from $SCRATCH/rup (output of run_HB_experiments.sh / run_JB_experiments.sh).
# Transfer outputs go to the same $SCRATCH/rup tree under:
#   $SCRATCH/rup/{benchmark}/{target_model_id}/{seed}/transfer_gcg_from_{source_model_id}/results.jsonl
#
# Each seed is submitted as a separate job for fine-grained control.
# Usage: bash run_transfer_experiments.sh
# Requires: must be run from the project root on a klogin* node.

set -e

source setup/start_env.sh

EXP="configs/experiments/base.yaml"
CMD="python scripts/run_transfer_inference.py --experiment $EXP --output-dir $SCRATCH/rup --resume"

# =============================================================================
# ATTACK TRANSFER STUDY — qwen2.5-0.5b-instruct (GCG surrogate) → Qwen3-8B
# Paper: Figure 2 left
# Source traces: $SCRATCH/rup/harmbench(jailbreakbench)/qwen2.5-0.5b-instruct/{seed}/gcg/
# =============================================================================

SRC_HB="--benchmark harmbench   --source-results-dir $SCRATCH/rup/harmbench   --source-model qwen2.5-0.5b-instruct --target-models qwen3_8b"
SRC_JB="--benchmark jailbreakbench --n-prompts 100 --source-results-dir $SCRATCH/rup/jailbreakbench --source-model qwen2.5-0.5b-instruct --target-models qwen3_8b"

# --- HarmBench ---
# submit "rup_tr_HB_q25_05b_q3_8b_s1997" "$CMD $SRC_HB --seeds 1997"
# submit "rup_tr_HB_q25_05b_q3_8b_s2"    "$CMD $SRC_HB --seeds 2"
# submit "rup_tr_HB_q25_05b_q3_8b_s100"  "$CMD $SRC_HB --seeds 100"
# submit "rup_tr_HB_q25_05b_q3_8b_s42"   "$CMD $SRC_HB --seeds 42"
# submit "rup_tr_HB_q25_05b_q3_8b_s5431" "$CMD $SRC_HB --seeds 5431"
# submit "rup_tr_HB_q25_05b_q3_8b_s2002" "$CMD $SRC_HB --seeds 2002"
# submit "rup_tr_HB_q25_05b_q3_8b_s256"  "$CMD $SRC_HB --seeds 256"
# submit "rup_tr_HB_q25_05b_q3_8b_s512"  "$CMD $SRC_HB --seeds 512"
# submit "rup_tr_HB_q25_05b_q3_8b_s123"  "$CMD $SRC_HB --seeds 123"
# submit "rup_tr_HB_q25_05b_q3_8b_s5"    "$CMD $SRC_HB --seeds 5"

# --- JailbreakBench ---
# submit "rup_tr_JB_q25_05b_q3_8b_s1997" "$CMD $SRC_JB --seeds 1997"
# submit "rup_tr_JB_q25_05b_q3_8b_s2"    "$CMD $SRC_JB --seeds 2"
# submit "rup_tr_JB_q25_05b_q3_8b_s100"  "$CMD $SRC_JB --seeds 100"
# submit "rup_tr_JB_q25_05b_q3_8b_s42"   "$CMD $SRC_JB --seeds 42"
# submit "rup_tr_JB_q25_05b_q3_8b_s5431" "$CMD $SRC_JB --seeds 5431"
# submit "rup_tr_JB_q25_05b_q3_8b_s2002" "$CMD $SRC_JB --seeds 2002"
# submit "rup_tr_JB_q25_05b_q3_8b_s256"  "$CMD $SRC_JB --seeds 256"
# submit "rup_tr_JB_q25_05b_q3_8b_s512"  "$CMD $SRC_JB --seeds 512"
# submit "rup_tr_JB_q25_05b_q3_8b_s123"  "$CMD $SRC_JB --seeds 123"
# submit "rup_tr_JB_q25_05b_q3_8b_s5"    "$CMD $SRC_JB --seeds 5"
