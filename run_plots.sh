#!/bin/bash
# Run Phase 3 plotting for all experiments (risk curves + category plots).
# Each model is plotted twice:
#   plots/bootstrap/ — per-seed bootstrap CIs (1000 resamples, stored in metrics.csv)
#   plots/seeds/     — cross-seed t-distribution CIs (empirical variance across 10 seeds)
#
# Usage: bash run_plots.sh
# Requires: must be run from the project root.
# Note: runs directly on the login node (no GPU needed — pure post-processing).

set -e

source setup/start_env.sh

BASE=/home/ehghaghi/projects/aip-craffel/ehghaghi/prisk-pressure/results/multitrial_experiments/pressure_sensitivity

# --------------------------------------------------------------------------- #
# HarmBench
# --------------------------------------------------------------------------- #

python plot_results.py \
    --metrics-csv $BASE/harmbench/qwen2.5-0.5b-instruct/metrics.csv \
    --category-metrics-csv $BASE/harmbench/qwen2.5-0.5b-instruct/metrics_by_category.csv \
    --output-dir $BASE/harmbench/qwen2.5-0.5b-instruct/plots/bootstrap \
    --ci-method bootstrap

python plot_results.py \
    --metrics-csv $BASE/harmbench/qwen2.5-0.5b-instruct/metrics.csv \
    --category-metrics-csv $BASE/harmbench/qwen2.5-0.5b-instruct/metrics_by_category.csv \
    --output-dir $BASE/harmbench/qwen2.5-0.5b-instruct/plots/seeds \
    --ci-method seeds

# python plot_results.py \
#     --metrics-csv $BASE/harmbench/qwen2.5-3b-instruct/metrics.csv \
#     --category-metrics-csv $BASE/harmbench/qwen2.5-3b-instruct/metrics_by_category.csv \
#     --output-dir $BASE/harmbench/qwen2.5-3b-instruct/plots/bootstrap \
#     --ci-method bootstrap
# python plot_results.py \
#     --metrics-csv $BASE/harmbench/qwen2.5-3b-instruct/metrics.csv \
#     --category-metrics-csv $BASE/harmbench/qwen2.5-3b-instruct/metrics_by_category.csv \
#     --output-dir $BASE/harmbench/qwen2.5-3b-instruct/plots/seeds \
#     --ci-method seeds

# python plot_results.py \
#     --metrics-csv $BASE/harmbench/qwen2.5-7b-instruct/metrics.csv \
#     --category-metrics-csv $BASE/harmbench/qwen2.5-7b-instruct/metrics_by_category.csv \
#     --output-dir $BASE/harmbench/qwen2.5-7b-instruct/plots/bootstrap \
#     --ci-method bootstrap
# python plot_results.py \
#     --metrics-csv $BASE/harmbench/qwen2.5-7b-instruct/metrics.csv \
#     --category-metrics-csv $BASE/harmbench/qwen2.5-7b-instruct/metrics_by_category.csv \
#     --output-dir $BASE/harmbench/qwen2.5-7b-instruct/plots/seeds \
#     --ci-method seeds

# python plot_results.py \
#     --metrics-csv $BASE/harmbench/qwen3-4b-saferl/metrics.csv \
#     --category-metrics-csv $BASE/harmbench/qwen3-4b-saferl/metrics_by_category.csv \
#     --output-dir $BASE/harmbench/qwen3-4b-saferl/plots/bootstrap \
#     --ci-method bootstrap
# python plot_results.py \
#     --metrics-csv $BASE/harmbench/qwen3-4b-saferl/metrics.csv \
#     --category-metrics-csv $BASE/harmbench/qwen3-4b-saferl/metrics_by_category.csv \
#     --output-dir $BASE/harmbench/qwen3-4b-saferl/plots/seeds \
#     --ci-method seeds

# python plot_results.py \
#     --metrics-csv $BASE/harmbench/tulu2-7b-base/metrics.csv \
#     --category-metrics-csv $BASE/harmbench/tulu2-7b-base/metrics_by_category.csv \
#     --output-dir $BASE/harmbench/tulu2-7b-base/plots/bootstrap \
#     --ci-method bootstrap
# python plot_results.py \
#     --metrics-csv $BASE/harmbench/tulu2-7b-base/metrics.csv \
#     --category-metrics-csv $BASE/harmbench/tulu2-7b-base/metrics_by_category.csv \
#     --output-dir $BASE/harmbench/tulu2-7b-base/plots/seeds \
#     --ci-method seeds

# python plot_results.py \
#     --metrics-csv $BASE/harmbench/tulu2-7b-sft/metrics.csv \
#     --category-metrics-csv $BASE/harmbench/tulu2-7b-sft/metrics_by_category.csv \
#     --output-dir $BASE/harmbench/tulu2-7b-sft/plots/bootstrap \
#     --ci-method bootstrap
# python plot_results.py \
#     --metrics-csv $BASE/harmbench/tulu2-7b-sft/metrics.csv \
#     --category-metrics-csv $BASE/harmbench/tulu2-7b-sft/metrics_by_category.csv \
#     --output-dir $BASE/harmbench/tulu2-7b-sft/plots/seeds \
#     --ci-method seeds

# python plot_results.py \
#     --metrics-csv $BASE/harmbench/tulu2-7b-dpo/metrics.csv \
#     --category-metrics-csv $BASE/harmbench/tulu2-7b-dpo/metrics_by_category.csv \
#     --output-dir $BASE/harmbench/tulu2-7b-dpo/plots/bootstrap \
#     --ci-method bootstrap
# python plot_results.py \
#     --metrics-csv $BASE/harmbench/tulu2-7b-dpo/metrics.csv \
#     --category-metrics-csv $BASE/harmbench/tulu2-7b-dpo/metrics_by_category.csv \
#     --output-dir $BASE/harmbench/tulu2-7b-dpo/plots/seeds \
#     --ci-method seeds

# python plot_results.py \
#     --metrics-csv $BASE/harmbench/tulu3-8b-base/metrics.csv \
#     --category-metrics-csv $BASE/harmbench/tulu3-8b-base/metrics_by_category.csv \
#     --output-dir $BASE/harmbench/tulu3-8b-base/plots/bootstrap \
#     --ci-method bootstrap
# python plot_results.py \
#     --metrics-csv $BASE/harmbench/tulu3-8b-base/metrics.csv \
#     --category-metrics-csv $BASE/harmbench/tulu3-8b-base/metrics_by_category.csv \
#     --output-dir $BASE/harmbench/tulu3-8b-base/plots/seeds \
#     --ci-method seeds

# python plot_results.py \
#     --metrics-csv $BASE/harmbench/tulu3-8b-sft/metrics.csv \
#     --category-metrics-csv $BASE/harmbench/tulu3-8b-sft/metrics_by_category.csv \
#     --output-dir $BASE/harmbench/tulu3-8b-sft/plots/bootstrap \
#     --ci-method bootstrap
# python plot_results.py \
#     --metrics-csv $BASE/harmbench/tulu3-8b-sft/metrics.csv \
#     --category-metrics-csv $BASE/harmbench/tulu3-8b-sft/metrics_by_category.csv \
#     --output-dir $BASE/harmbench/tulu3-8b-sft/plots/seeds \
#     --ci-method seeds

# python plot_results.py \
#     --metrics-csv $BASE/harmbench/tulu3-8b-dpo/metrics.csv \
#     --category-metrics-csv $BASE/harmbench/tulu3-8b-dpo/metrics_by_category.csv \
#     --output-dir $BASE/harmbench/tulu3-8b-dpo/plots/bootstrap \
#     --ci-method bootstrap
# python plot_results.py \
#     --metrics-csv $BASE/harmbench/tulu3-8b-dpo/metrics.csv \
#     --category-metrics-csv $BASE/harmbench/tulu3-8b-dpo/metrics_by_category.csv \
#     --output-dir $BASE/harmbench/tulu3-8b-dpo/plots/seeds \
#     --ci-method seeds

# python plot_results.py \
#     --metrics-csv $BASE/harmbench/tulu3-8b-rlvr/metrics.csv \
#     --category-metrics-csv $BASE/harmbench/tulu3-8b-rlvr/metrics_by_category.csv \
#     --output-dir $BASE/harmbench/tulu3-8b-rlvr/plots/bootstrap \
#     --ci-method bootstrap
# python plot_results.py \
#     --metrics-csv $BASE/harmbench/tulu3-8b-rlvr/metrics.csv \
#     --category-metrics-csv $BASE/harmbench/tulu3-8b-rlvr/metrics_by_category.csv \
#     --output-dir $BASE/harmbench/tulu3-8b-rlvr/plots/seeds \
#     --ci-method seeds

# --------------------------------------------------------------------------- #
# JailbreakBench
# --------------------------------------------------------------------------- #

python plot_results.py \
    --metrics-csv $BASE/jailbreakbench/qwen2.5-0.5b-instruct/metrics.csv \
    --category-metrics-csv $BASE/jailbreakbench/qwen2.5-0.5b-instruct/metrics_by_category.csv \
    --output-dir $BASE/jailbreakbench/qwen2.5-0.5b-instruct/plots/bootstrap \
    --ci-method bootstrap

python plot_results.py \
    --metrics-csv $BASE/jailbreakbench/qwen2.5-0.5b-instruct/metrics.csv \
    --category-metrics-csv $BASE/jailbreakbench/qwen2.5-0.5b-instruct/metrics_by_category.csv \
    --output-dir $BASE/jailbreakbench/qwen2.5-0.5b-instruct/plots/seeds \
    --ci-method seeds

# python plot_results.py \
#     --metrics-csv $BASE/jailbreakbench/qwen2.5-3b-instruct/metrics.csv \
#     --category-metrics-csv $BASE/jailbreakbench/qwen2.5-3b-instruct/metrics_by_category.csv \
#     --output-dir $BASE/jailbreakbench/qwen2.5-3b-instruct/plots/bootstrap \
#     --ci-method bootstrap
# python plot_results.py \
#     --metrics-csv $BASE/jailbreakbench/qwen2.5-3b-instruct/metrics.csv \
#     --category-metrics-csv $BASE/jailbreakbench/qwen2.5-3b-instruct/metrics_by_category.csv \
#     --output-dir $BASE/jailbreakbench/qwen2.5-3b-instruct/plots/seeds \
#     --ci-method seeds

# python plot_results.py \
#     --metrics-csv $BASE/jailbreakbench/qwen2.5-7b-instruct/metrics.csv \
#     --category-metrics-csv $BASE/jailbreakbench/qwen2.5-7b-instruct/metrics_by_category.csv \
#     --output-dir $BASE/jailbreakbench/qwen2.5-7b-instruct/plots/bootstrap \
#     --ci-method bootstrap
# python plot_results.py \
#     --metrics-csv $BASE/jailbreakbench/qwen2.5-7b-instruct/metrics.csv \
#     --category-metrics-csv $BASE/jailbreakbench/qwen2.5-7b-instruct/metrics_by_category.csv \
#     --output-dir $BASE/jailbreakbench/qwen2.5-7b-instruct/plots/seeds \
#     --ci-method seeds

# python plot_results.py \
#     --metrics-csv $BASE/jailbreakbench/qwen3-4b-saferl/metrics.csv \
#     --category-metrics-csv $BASE/jailbreakbench/qwen3-4b-saferl/metrics_by_category.csv \
#     --output-dir $BASE/jailbreakbench/qwen3-4b-saferl/plots/bootstrap \
#     --ci-method bootstrap
# python plot_results.py \
#     --metrics-csv $BASE/jailbreakbench/qwen3-4b-saferl/metrics.csv \
#     --category-metrics-csv $BASE/jailbreakbench/qwen3-4b-saferl/metrics_by_category.csv \
#     --output-dir $BASE/jailbreakbench/qwen3-4b-saferl/plots/seeds \
#     --ci-method seeds

# python plot_results.py \
#     --metrics-csv $BASE/jailbreakbench/tulu2-7b-base/metrics.csv \
#     --category-metrics-csv $BASE/jailbreakbench/tulu2-7b-base/metrics_by_category.csv \
#     --output-dir $BASE/jailbreakbench/tulu2-7b-base/plots/bootstrap \
#     --ci-method bootstrap
# python plot_results.py \
#     --metrics-csv $BASE/jailbreakbench/tulu2-7b-base/metrics.csv \
#     --category-metrics-csv $BASE/jailbreakbench/tulu2-7b-base/metrics_by_category.csv \
#     --output-dir $BASE/jailbreakbench/tulu2-7b-base/plots/seeds \
#     --ci-method seeds

# python plot_results.py \
#     --metrics-csv $BASE/jailbreakbench/tulu2-7b-sft/metrics.csv \
#     --category-metrics-csv $BASE/jailbreakbench/tulu2-7b-sft/metrics_by_category.csv \
#     --output-dir $BASE/jailbreakbench/tulu2-7b-sft/plots/bootstrap \
#     --ci-method bootstrap
# python plot_results.py \
#     --metrics-csv $BASE/jailbreakbench/tulu2-7b-sft/metrics.csv \
#     --category-metrics-csv $BASE/jailbreakbench/tulu2-7b-sft/metrics_by_category.csv \
#     --output-dir $BASE/jailbreakbench/tulu2-7b-sft/plots/seeds \
#     --ci-method seeds

# python plot_results.py \
#     --metrics-csv $BASE/jailbreakbench/tulu2-7b-dpo/metrics.csv \
#     --category-metrics-csv $BASE/jailbreakbench/tulu2-7b-dpo/metrics_by_category.csv \
#     --output-dir $BASE/jailbreakbench/tulu2-7b-dpo/plots/bootstrap \
#     --ci-method bootstrap
# python plot_results.py \
#     --metrics-csv $BASE/jailbreakbench/tulu2-7b-dpo/metrics.csv \
#     --category-metrics-csv $BASE/jailbreakbench/tulu2-7b-dpo/metrics_by_category.csv \
#     --output-dir $BASE/jailbreakbench/tulu2-7b-dpo/plots/seeds \
#     --ci-method seeds

# python plot_results.py \
#     --metrics-csv $BASE/jailbreakbench/tulu3-8b-base/metrics.csv \
#     --category-metrics-csv $BASE/jailbreakbench/tulu3-8b-base/metrics_by_category.csv \
#     --output-dir $BASE/jailbreakbench/tulu3-8b-base/plots/bootstrap \
#     --ci-method bootstrap
# python plot_results.py \
#     --metrics-csv $BASE/jailbreakbench/tulu3-8b-base/metrics.csv \
#     --category-metrics-csv $BASE/jailbreakbench/tulu3-8b-base/metrics_by_category.csv \
#     --output-dir $BASE/jailbreakbench/tulu3-8b-base/plots/seeds \
#     --ci-method seeds

# python plot_results.py \
#     --metrics-csv $BASE/jailbreakbench/tulu3-8b-sft/metrics.csv \
#     --category-metrics-csv $BASE/jailbreakbench/tulu3-8b-sft/metrics_by_category.csv \
#     --output-dir $BASE/jailbreakbench/tulu3-8b-sft/plots/bootstrap \
#     --ci-method bootstrap
# python plot_results.py \
#     --metrics-csv $BASE/jailbreakbench/tulu3-8b-sft/metrics.csv \
#     --category-metrics-csv $BASE/jailbreakbench/tulu3-8b-sft/metrics_by_category.csv \
#     --output-dir $BASE/jailbreakbench/tulu3-8b-sft/plots/seeds \
#     --ci-method seeds

# python plot_results.py \
#     --metrics-csv $BASE/jailbreakbench/tulu3-8b-dpo/metrics.csv \
#     --category-metrics-csv $BASE/jailbreakbench/tulu3-8b-dpo/metrics_by_category.csv \
#     --output-dir $BASE/jailbreakbench/tulu3-8b-dpo/plots/bootstrap \
#     --ci-method bootstrap
# python plot_results.py \
#     --metrics-csv $BASE/jailbreakbench/tulu3-8b-dpo/metrics.csv \
#     --category-metrics-csv $BASE/jailbreakbench/tulu3-8b-dpo/metrics_by_category.csv \
#     --output-dir $BASE/jailbreakbench/tulu3-8b-dpo/plots/seeds \
#     --ci-method seeds

# python plot_results.py \
#     --metrics-csv $BASE/jailbreakbench/tulu3-8b-rlvr/metrics.csv \
#     --category-metrics-csv $BASE/jailbreakbench/tulu3-8b-rlvr/metrics_by_category.csv \
#     --output-dir $BASE/jailbreakbench/tulu3-8b-rlvr/plots/bootstrap \
#     --ci-method bootstrap
# python plot_results.py \
#     --metrics-csv $BASE/jailbreakbench/tulu3-8b-rlvr/metrics.csv \
#     --category-metrics-csv $BASE/jailbreakbench/tulu3-8b-rlvr/metrics_by_category.csv \
#     --output-dir $BASE/jailbreakbench/tulu3-8b-rlvr/plots/seeds \
#     --ci-method seeds
