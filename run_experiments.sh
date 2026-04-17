#!/bin/bash
# Submit all pRisk-Pressure experiments to the Killarney cluster.
# Usage: bash run_experiments.sh
# Requires: must be run from the project root on a klogin* node.

set -e

source setup/start_env.sh

# submit "prisk_pressure_sensitivity_batch1" \
#     "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity.yaml --seeds 1997 2 100 --resume"

# submit "prisk_pressure_sensitivity_batch2" \
#     "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity.yaml --seeds 42 5431 2002 --resume"

# submit "prisk_pressure_sensitivity_batch3" \
#     "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity.yaml --seeds 256 512 123 5--resume"

# submit "prisk_pressure_sensitivity_JB" \
#     "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_JB.yaml --seeds 1997 2 100 42 5431 --resume"

# submit "prisk_attack_sensitivity" \
#     "python scripts/run_inference.py --experiment configs/experiments/attack_sensitivity.yaml --seeds 42 123 456 --resume"

# submit "prisk_open_vs_closed" \
#     "python scripts/run_inference.py --experiment configs/experiments/open_vs_closed.yaml --seeds 42 123 456 --resume"

# submit "prisk_attacker_comparison" \
#     "python scripts/run_inference.py --experiment configs/experiments/attacker_comparison.yaml --seeds 42 123 456 --resume"

# submit "prisk_template_ordering" \
#     "python scripts/compute_template_ordering.py --experiment configs/experiments/template_ordering.yaml --seeds 42 123 456 --resume"

# --- Per-model pressure sensitivity ---

# Qwen models
submit "prisk_pressure_sensitivity_qwen2.5_0.5b_batch1" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_qwen2.5_0.5b.yaml --seeds 1997 2 100 --resume"
submit "prisk_pressure_sensitivity_qwen2.5_0.5b_batch2" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_qwen2.5_0.5b.yaml --seeds 42 5431 2002 --resume"
# submit "prisk_pressure_sensitivity_qwen2.5_0.5b_batch3" \
#     "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_qwen2.5_0.5b.yaml --seeds 256 512 123 5 --resume"

submit "prisk_pressure_sensitivity_qwen2.5_3b_batch1" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_qwen2.5_3b.yaml --seeds 1997 2 100 --resume"
submit "prisk_pressure_sensitivity_qwen2.5_3b_batch2" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_qwen2.5_3b.yaml --seeds 42 5431 2002 --resume"
submit "prisk_pressure_sensitivity_qwen2.5_3b_batch3" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_qwen2.5_3b.yaml --seeds 256 512 123 5 --resume"

submit "prisk_pressure_sensitivity_qwen2.5_7b_batch1" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_qwen2.5_7b.yaml --seeds 1997 2 100 --resume"
submit "prisk_pressure_sensitivity_qwen2.5_7b_batch2" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_qwen2.5_7b.yaml --seeds 42 5431 2002 --resume"
submit "prisk_pressure_sensitivity_qwen2.5_7b_batch3" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_qwen2.5_7b.yaml --seeds 256 512 123 5 --resume"

# Qwen safety model
submit "prisk_pressure_sensitivity_qwen3_4b_saferl_batch1" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_qwen3_4b_saferl.yaml --seeds 1997 2 100 --resume"
submit "prisk_pressure_sensitivity_qwen3_4b_saferl_batch2" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_qwen3_4b_saferl.yaml --seeds 42 5431 2002 --resume"
submit "prisk_pressure_sensitivity_qwen3_4b_saferl_batch3" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_qwen3_4b_saferl.yaml --seeds 256 512 123 5 --resume"

# Tulu2 models
submit "prisk_pressure_sensitivity_tulu2_7b_base_batch1" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_tulu2_7b_base.yaml --seeds 1997 2 100 --resume"
submit "prisk_pressure_sensitivity_tulu2_7b_base_batch2" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_tulu2_7b_base.yaml --seeds 42 5431 2002 --resume"
submit "prisk_pressure_sensitivity_tulu2_7b_base_batch3" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_tulu2_7b_base.yaml --seeds 256 512 123 5 --resume"

submit "prisk_pressure_sensitivity_tulu2_7b_sft_batch1" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_tulu2_7b_sft.yaml --seeds 1997 2 100 --resume"
submit "prisk_pressure_sensitivity_tulu2_7b_sft_batch2" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_tulu2_7b_sft.yaml --seeds 42 5431 2002 --resume"
submit "prisk_pressure_sensitivity_tulu2_7b_sft_batch3" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_tulu2_7b_sft.yaml --seeds 256 512 123 5 --resume"

submit "prisk_pressure_sensitivity_tulu2_7b_dpo_batch1" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_tulu2_7b_dpo.yaml --seeds 1997 2 100 --resume"
submit "prisk_pressure_sensitivity_tulu2_7b_dpo_batch2" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_tulu2_7b_dpo.yaml --seeds 42 5431 2002 --resume"
submit "prisk_pressure_sensitivity_tulu2_7b_dpo_batch3" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_tulu2_7b_dpo.yaml --seeds 256 512 123 5 --resume"

# Tulu3 models
submit "prisk_pressure_sensitivity_tulu3_8b_base_batch1" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_tulu3_8b_base.yaml --seeds 1997 2 100 --resume"
submit "prisk_pressure_sensitivity_tulu3_8b_base_batch2" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_tulu3_8b_base.yaml --seeds 42 5431 2002 --resume"
submit "prisk_pressure_sensitivity_tulu3_8b_base_batch3" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_tulu3_8b_base.yaml --seeds 256 512 123 5 --resume"

submit "prisk_pressure_sensitivity_tulu3_8b_sft_batch1" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_tulu3_8b_sft.yaml --seeds 1997 2 100 --resume"
submit "prisk_pressure_sensitivity_tulu3_8b_sft_batch2" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_tulu3_8b_sft.yaml --seeds 42 5431 2002 --resume"
submit "prisk_pressure_sensitivity_tulu3_8b_sft_batch3" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_tulu3_8b_sft.yaml --seeds 256 512 123 5 --resume"

submit "prisk_pressure_sensitivity_tulu3_8b_dpo_batch1" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_tulu3_8b_dpo.yaml --seeds 1997 2 100 --resume"
submit "prisk_pressure_sensitivity_tulu3_8b_dpo_batch2" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_tulu3_8b_dpo.yaml --seeds 42 5431 2002 --resume"
submit "prisk_pressure_sensitivity_tulu3_8b_dpo_batch3" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_tulu3_8b_dpo.yaml --seeds 256 512 123 5 --resume"

submit "prisk_pressure_sensitivity_tulu3_8b_rlvr_batch1" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_tulu3_8b_rlvr.yaml --seeds 1997 2 100 --resume"
submit "prisk_pressure_sensitivity_tulu3_8b_rlvr_batch2" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_tulu3_8b_rlvr.yaml --seeds 42 5431 2002 --resume"
submit "prisk_pressure_sensitivity_tulu3_8b_rlvr_batch3" \
    "python scripts/run_inference.py --experiment configs/experiments/pressure_sensitivity_tulu3_8b_rlvr.yaml --seeds 256 512 123 5 --resume"