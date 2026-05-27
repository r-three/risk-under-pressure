#!/bin/bash
# Submit HarmBench per-model pressure sensitivity experiments.
# Usage: bash run_HB_experiments.sh
# Requires: must be run from the project root on a klogin* node.

set -e

source setup/start_env.sh

# --- Per-model pressure sensitivity ---

# Qwen models
submit "rup_pressure_sensitivity_qwen2.5_0.5b_batch1" \
    "python scripts/run_inference.py --experiment configs/experiments/HB_qwen2.5_0.5b.yaml --seeds 100 --attacks gcg --resume"
# submit "rup_pressure_sensitivity_qwen2.5_0.5b_batch2" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_qwen2.5_0.5b.yaml --seeds 42 5431 2002 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_qwen2.5_0.5b_batch3" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_qwen2.5_0.5b.yaml --seeds 256 512 123 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_qwen2.5_0.5b_batch4" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_qwen2.5_0.5b.yaml --seeds 5 --attacks jailbroken --resume"

# submit "rup_pressure_sensitivity_qwen2.5_3b_batch1" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_qwen2.5_3b.yaml --seeds 1997 2 100 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_qwen2.5_3b_batch2" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_qwen2.5_3b.yaml --seeds 42 5431 2002 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_qwen2.5_3b_batch3" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_qwen2.5_3b.yaml --seeds 256 512 123 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_qwen2.5_3b_batch4" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_qwen2.5_3b.yaml --seeds 5 --attacks jailbroken --resume"

# submit "rup_pressure_sensitivity_qwen2.5_7b_batch1" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_qwen2.5_7b.yaml --seeds 1997 2 100 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_qwen2.5_7b_batch2" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_qwen2.5_7b.yaml --seeds 42 5431 2002 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_qwen2.5_7b_batch3" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_qwen2.5_7b.yaml --seeds 256 512 123 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_qwen2.5_7b_batch4" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_qwen2.5_7b.yaml --seeds 5 --attacks jailbroken --resume"

# # Qwen safety model
# submit "rup_pressure_sensitivity_qwen3_4b_saferl_batch1" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_qwen3_4b_saferl.yaml --seeds 1997 2 100 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_qwen3_4b_saferl_batch2" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_qwen3_4b_saferl.yaml --seeds 42 5431 2002 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_qwen3_4b_saferl_batch3" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_qwen3_4b_saferl.yaml --seeds 256 512 123 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_qwen3_4b_saferl_batch4" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_qwen3_4b_saferl.yaml --seeds 5 --attacks jailbroken --resume"

# # Tulu2 models
# submit "rup_pressure_sensitivity_tulu2_7b_base_batch1" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu2_7b_base.yaml --seeds 1997 2 100 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_tulu2_7b_base_batch2" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu2_7b_base.yaml --seeds 42 5431 2002 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_tulu2_7b_base_batch3" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu2_7b_base.yaml --seeds 256 512 123 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_tulu2_7b_base_batch4" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu2_7b_base.yaml --seeds 5 --attacks jailbroken --resume"

# submit "rup_pressure_sensitivity_tulu2_7b_sft_batch1" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu2_7b_sft.yaml --seeds 1997 2 100 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_tulu2_7b_sft_batch2" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu2_7b_sft.yaml --seeds 42 5431 2002 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_tulu2_7b_sft_batch3" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu2_7b_sft.yaml --seeds 256 512 123 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_tulu2_7b_sft_batch4" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu2_7b_sft.yaml --seeds 5 --attacks jailbroken --resume"

# submit "rup_pressure_sensitivity_tulu2_7b_dpo_batch1" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu2_7b_dpo.yaml --seeds 1997 2 100 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_tulu2_7b_dpo_batch2" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu2_7b_dpo.yaml --seeds 42 5431 2002 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_tulu2_7b_dpo_batch3" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu2_7b_dpo.yaml --seeds 256 512 123 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_tulu2_7b_dpo_batch4" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu2_7b_dpo.yaml --seeds 5 --attacks jailbroken --resume"

# # Tulu3 models
# submit "rup_pressure_sensitivity_tulu3_8b_base_batch1" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu3_8b_base.yaml --seeds 1997 2 100 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_tulu3_8b_base_batch2" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu3_8b_base.yaml --seeds 42 5431 2002 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_tulu3_8b_base_batch3" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu3_8b_base.yaml --seeds 256 512 123 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_tulu3_8b_base_batch4" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu3_8b_base.yaml --seeds 5 --attacks jailbroken --resume"

# submit "rup_pressure_sensitivity_tulu3_8b_sft_batch1" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu3_8b_sft.yaml --seeds 1997 2 100 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_tulu3_8b_sft_batch2" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu3_8b_sft.yaml --seeds 42 5431 2002 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_tulu3_8b_sft_batch3" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu3_8b_sft.yaml --seeds 256 512 123 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_tulu3_8b_sft_batch4" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu3_8b_sft.yaml --seeds 5 --attacks jailbroken --resume"

# submit "rup_pressure_sensitivity_tulu3_8b_dpo_batch1" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu3_8b_dpo.yaml --seeds 1997 2 100 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_tulu3_8b_dpo_batch2" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu3_8b_dpo.yaml --seeds 42 5431 2002 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_tulu3_8b_dpo_batch3" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu3_8b_dpo.yaml --seeds 256 512 123 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_tulu3_8b_dpo_batch4" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu3_8b_dpo.yaml --seeds 5 --attacks jailbroken --resume"

# submit "rup_pressure_sensitivity_tulu3_8b_rlvr_batch1" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu3_8b_rlvr.yaml --seeds 1997 2 100 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_tulu3_8b_rlvr_batch2" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu3_8b_rlvr.yaml --seeds 42 5431 2002 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_tulu3_8b_rlvr_batch3" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu3_8b_rlvr.yaml --seeds 256 512 123 --attacks jailbroken --resume"
# submit "rup_pressure_sensitivity_tulu3_8b_rlvr_batch4" \
#     "python scripts/run_inference.py --experiment configs/experiments/HB_tulu3_8b_rlvr.yaml --seeds 5 --attacks jailbroken --resume"
