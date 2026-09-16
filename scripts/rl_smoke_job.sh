#!/bin/bash
# GPU job body for the per-prompt RL/GRPO adaptive-attack smoke test.
# Runs the pipeline (attack -> evaluate -> cost -> verify) on ONE GPU node, tiny scale, so you
# can confirm the RL path works before launching real experiments. There is NO separate training
# phase — per-prompt GRPO happens inside run_inference.py --attack rl.
#
# Launched by run_rl_smoke.sh via the SLURM `submit` helper (GRPO needs a visible GPU).
# All artifacts and model caches live under $SCRATCH (set by setup/start_env.sh).

set -e

[ -z "$SCRATCH" ] && source setup/start_env.sh

OUT="$SCRATCH/rl_smoke"
TARGET_CFG="qwen2.5_0.5b"
MODEL_ID="qwen2.5-0.5b-instruct"      # model_id from configs/models/qwen2.5_0.5b.yaml

echo "SCRATCH=$SCRATCH | HF_HOME=$HF_HOME | OUT=$OUT"

echo "=============================================================="
echo "[1/3] Run the per-prompt RL attack (2 prompts, budget 9, 4 sessions x 2 rounds)"
echo "=============================================================="
python scripts/run_inference.py \
    --experiment configs/experiments/base.yaml \
    --model "$TARGET_CFG" --attack rl_smoke \
    --n-prompts 2 --seeds 1997 --lambda-max 9 \
    --output-dir "$OUT"
test -f "$OUT/harmbench/$MODEL_ID/1997/rl/results.jsonl" \
    && echo "OK: results.jsonl written" \
    || { echo "FAIL: results.jsonl missing"; exit 1; }

echo "=============================================================="
echo "[2/3] Compute risk metrics + FLOP costs"
echo "=============================================================="
python scripts/run_evaluation.py \
    --experiment configs/experiments/base.yaml --format csv \
    --results-dir "$OUT/harmbench/$MODEL_ID" \
    --output "$OUT/plots/harmbench/$MODEL_ID/metrics.csv"
python scripts/compute_attack_costs.py \
    --results-dir "$OUT/harmbench/$MODEL_ID" \
    --metrics-csv "$OUT/plots/harmbench/$MODEL_ID/metrics.csv" \
    --output      "$OUT/plots/harmbench/$MODEL_ID/cost/cost_metrics.csv"

echo "=============================================================="
echo "[3/3] Verify the RL rows/costs are present"
echo "=============================================================="
python - "$OUT/plots/harmbench/$MODEL_ID/cost/cost_metrics.csv" <<'PY'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
rl = df[df["attack_id"] == "rl"]
assert not rl.empty, "no rl rows in cost_metrics.csv"
top = rl.sort_values("lambda").iloc[-1]
print(f"rl @ lambda={int(top['lambda'])}: risk={top.get('risk', float('nan'))} "
      f"total_tflops={top['mean_total_tflops']:.4f}")
assert top["mean_total_tflops"] > 0, "RL cost not computed"
print("SMOKE TEST PASSED")
PY
