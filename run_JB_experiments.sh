#!/bin/bash
# Submit all pRisk-Pressure experiments to the Killarney cluster.
# Usage: bash run_experiments.sh
# Requires: must be run from the project root on a klogin* node.

set -e

source setup/start_env.sh

# --- JailBreakBench pressure sensitivity (10 seeds) ---

# Qwen models
submit "rup_JB_qwen2.5_0.5b_batch1" \
    "python scripts/run_inference.py --experiment configs/experiments/JB_qwen2.5_0.5b.yaml --seeds 1997 2 100 --resume"
submit "rup_JB_qwen2.5_0.5b_batch2" \
    "python scripts/run_inference.py --experiment configs/experiments/JB_qwen2.5_0.5b.yaml --seeds 42 5431 2002 --resume"
submit "rup_JB_qwen2.5_0.5b_batch3" \
    "python scripts/run_inference.py --experiment configs/experiments/JB_qwen2.5_0.5b.yaml --seeds 256 512 123 --resume"
submit "rup_JB_qwen2.5_0.5b_batch4" \
    "python scripts/run_inference.py --experiment configs/experiments/JB_qwen2.5_0.5b.yaml --seeds 5 --resume"

# submit "rup_JB_qwen2.5_3b_batch1" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_qwen2.5_3b.yaml --seeds 1997 2 100 --resume"
# submit "rup_JB_qwen2.5_3b_batch2" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_qwen2.5_3b.yaml --seeds 42 5431 2002 --resume"
# submit "rup_JB_qwen2.5_3b_batch3" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_qwen2.5_3b.yaml --seeds 256 512 123 --resume"
# submit "rup_JB_qwen2.5_3b_batch4" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_qwen2.5_3b.yaml --seeds 5 --resume"

# submit "rup_JB_qwen2.5_7b_batch1" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_qwen2.5_7b.yaml --seeds 1997 2 100 --resume"
# submit "rup_JB_qwen2.5_7b_batch2" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_qwen2.5_7b.yaml --seeds 42 5431 2002 --resume"
# submit "rup_JB_qwen2.5_7b_batch3" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_qwen2.5_7b.yaml --seeds 256 512 123 --resume"
# submit "rup_JB_qwen2.5_7b_batch4" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_qwen2.5_7b.yaml --seeds 5 --resume"

# Qwen safety model
# submit "rup_JB_qwen3_4b_saferl_s1997" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_qwen3_4b_saferl.yaml --seeds 1997 --resume"
# submit "rup_JB_qwen3_4b_saferl_s2" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_qwen3_4b_saferl.yaml --seeds 2 --resume"
# submit "rup_JB_qwen3_4b_saferl_s100" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_qwen3_4b_saferl.yaml --seeds 100 --resume"
# submit "rup_JB_qwen3_4b_saferl_s42" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_qwen3_4b_saferl.yaml --seeds 42 --resume"
# submit "rup_JB_qwen3_4b_saferl_s5431" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_qwen3_4b_saferl.yaml --seeds 5431 --resume"
# submit "rup_JB_qwen3_4b_saferl_s2002" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_qwen3_4b_saferl.yaml --seeds 2002 --resume"
# submit "rup_JB_qwen3_4b_saferl_s256" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_qwen3_4b_saferl.yaml --seeds 256 --resume"
# submit "rup_JB_qwen3_4b_saferl_s512" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_qwen3_4b_saferl.yaml --seeds 512 --resume"
# submit "rup_JB_qwen3_4b_saferl_s123" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_qwen3_4b_saferl.yaml --seeds 123 --resume"
# submit "rup_JB_qwen3_4b_saferl_s5" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_qwen3_4b_saferl.yaml --seeds 5 --resume"

# Tulu2 models
# submit "rup_JB_tulu2_7b_base_batch1" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu2_7b_base.yaml --seeds 1997 2 100 --resume"
# submit "rup_JB_tulu2_7b_base_batch2" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu2_7b_base.yaml --seeds 42 5431 2002 --resume"
# submit "rup_JB_tulu2_7b_base_batch3" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu2_7b_base.yaml --seeds 256 512 123 --resume"
# submit "rup_JB_tulu2_7b_base_batch4" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu2_7b_base.yaml --seeds 5 --resume"

# submit "rup_JB_tulu2_7b_sft_batch1" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu2_7b_sft.yaml --seeds 1997 2 100 --resume"
# submit "rup_JB_tulu2_7b_sft_batch2" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu2_7b_sft.yaml --seeds 42 5431 2002 --resume"
# submit "rup_JB_tulu2_7b_sft_batch3" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu2_7b_sft.yaml --seeds 256 512 123 --resume"
# submit "rup_JB_tulu2_7b_sft_batch4" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu2_7b_sft.yaml --seeds 5 --resume"

# submit "rup_JB_tulu2_7b_dpo_batch1" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu2_7b_dpo.yaml --seeds 1997 2 100 --resume"
# submit "rup_JB_tulu2_7b_dpo_batch2" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu2_7b_dpo.yaml --seeds 42 5431 2002 --resume"
# submit "rup_JB_tulu2_7b_dpo_batch3" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu2_7b_dpo.yaml --seeds 256 512 123 --resume"
# submit "rup_JB_tulu2_7b_dpo_batch4" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu2_7b_dpo.yaml --seeds 5 --resume"

# Tulu3 models
# submit "rup_JB_tulu3_8b_base_batch1" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu3_8b_base.yaml --seeds 1997 2 100 --resume"
# submit "rup_JB_tulu3_8b_base_batch2" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu3_8b_base.yaml --seeds 42 5431 2002 --resume"
# submit "rup_JB_tulu3_8b_base_batch3" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu3_8b_base.yaml --seeds 256 512 123 --resume"
# submit "rup_JB_tulu3_8b_base_batch4" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu3_8b_base.yaml --seeds 5 --resume"

# submit "rup_JB_tulu3_8b_sft_batch1" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu3_8b_sft.yaml --seeds 1997 2 100 --resume"
# submit "rup_JB_tulu3_8b_sft_batch2" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu3_8b_sft.yaml --seeds 42 5431 2002 --resume"
# submit "rup_JB_tulu3_8b_sft_batch3" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu3_8b_sft.yaml --seeds 256 512 123 --resume"
# submit "rup_JB_tulu3_8b_sft_batch4" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu3_8b_sft.yaml --seeds 5 --resume"

# submit "rup_JB_tulu3_8b_dpo_batch1" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu3_8b_dpo.yaml --seeds 1997 2 100 --resume"
# submit "rup_JB_tulu3_8b_dpo_batch2" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu3_8b_dpo.yaml --seeds 42 5431 2002 --resume"
# submit "rup_JB_tulu3_8b_dpo_batch3" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu3_8b_dpo.yaml --seeds 256 512 123 --resume"
# submit "rup_JB_tulu3_8b_dpo_batch4" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu3_8b_dpo.yaml --seeds 5 --resume"
submit "rup_JB_tulu3_8b_dpo_batch1" \
    "python scripts/run_inference.py --experiment configs/experiments/JB_tulu3_8b_dpo.yaml --seeds 100 --resume"
submit "rup_JB_tulu3_8b_dpo_batch2" \
    "python scripts/run_inference.py --experiment configs/experiments/JB_tulu3_8b_dpo.yaml --seeds 2002 --resume"
submit "rup_JB_tulu3_8b_dpo_batch3" \
    "python scripts/run_inference.py --experiment configs/experiments/JB_tulu3_8b_dpo.yaml --seeds 123 --resume"


# submit "rup_JB_tulu3_8b_rlvr_batch1" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu3_8b_rlvr.yaml --seeds 1997 2 100 --resume"
# submit "rup_JB_tulu3_8b_rlvr_batch2" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu3_8b_rlvr.yaml --seeds 42 5431 2002 --resume"
# submit "rup_JB_tulu3_8b_rlvr_batch3" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu3_8b_rlvr.yaml --seeds 256 512 123 --resume"
# submit "rup_JB_tulu3_8b_rlvr_batch4" \
#     "python scripts/run_inference.py --experiment configs/experiments/JB_tulu3_8b_rlvr.yaml --seeds 5 --resume"

submit "rup_JB_tulu3_8b_rlvr_batch1" \
    "python scripts/run_inference.py --experiment configs/experiments/JB_tulu3_8b_rlvr.yaml --seeds 100 --resume"
submit "rup_JB_tulu3_8b_rlvr_batch2" \
    "python scripts/run_inference.py --experiment configs/experiments/JB_tulu3_8b_rlvr.yaml --seeds 2002 --resume"
submit "rup_JB_tulu3_8b_rlvr_batch3" \
    "python scripts/run_inference.py --experiment configs/experiments/JB_tulu3_8b_rlvr.yaml --seeds 123 --resume"