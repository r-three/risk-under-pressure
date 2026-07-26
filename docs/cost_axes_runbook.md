# Cost-Axes Runbook — wall-clock (seconds) & dollars

How to populate the two new cost axes for every experiment/attack, and what (if anything)
must be re-run. Companion to the "Cost axes" section in the main README.

For *how the numbers are computed* — measurement boundaries, the pricing formula, NaN semantics —
see [`cost_axes_computation.md`](cost_axes_computation.md).

## TL;DR — the rerun matrix

| Axis | Re-run experiments (GPU inference)? | What to run |
|---|---|---|
| **dollars** | ❌ **No** — computed post-hoc from token counts already in `results.jsonl` | re-run cost mapping with `--pricing-config`, then re-plot `--x-axis dollars` |
| **seconds** | ✅ **Yes**, on a single **L40S** — wall-clock is *measured* during the run, and old results have no per-step timing (they show `NaN`) | re-run the attacks on L40S, then the usual eval → cost → plot |

`metrics.csv` (risk) never needs re-running for either axis.

---

## 1. Dollars — no rerun, works on all existing experiments now

Per model (CPU, login node):

```bash
python scripts/compute_attack_costs.py \
    --results-dir  $SCRATCH/rup/<bench>/<model> \
    --metrics-csv  $SCRATCH/rup/plots/<bench>/<model>/metrics.csv \
    --output       $SCRATCH/rup/plots/<bench>/<model>/cost/cost_metrics.csv \
    --pricing-config configs/pricing.yaml         # <-- the only new flag

python scripts/plot_cost_curves.py \
    --cost-csv $SCRATCH/rup/plots/<bench>/<model>/cost/cost_metrics.csv \
    --output-dir $SCRATCH/rup/plots/<bench>/<model>/dollars \
    --x-axis dollars
```

Output columns added to `cost_metrics.csv`:
`mean_target_dollars`, `mean_judge_dollars`, `mean_attacker_dollars`, `mean_total_dollars`
(each component priced by **its own model's** rate; total = their sum), plus
`mean_attacker_tokens`. Without `--pricing-config` these are `NaN`.

Rates live in `configs/pricing.yaml` (placeholder Together AI reference rates — **verify before
publishing**).

---

## 2. Seconds — measured attack wall-clock, requires L40S re-inference

Existing `results.jsonl` predate the timing instrumentation → `mean_total_seconds = NaN`. To
populate it, re-run the attacks on **one L40S** (fixed hardware, for comparability), then:

```bash
python scripts/run_inference.py --attack <gcg|pair|jailbroken|rl> ... --output-dir $SCRATCH/rup
bash run_evaluations.sh
bash run_cost_evaluations.sh
python scripts/plot_cost_curves.py --cost-csv <...> --output-dir <dir> --x-axis seconds
```

Each `StepResult` now carries a measured `seconds`, and each `TrialRecord.metadata.gpu` records
the GPU so you can assert/filter to L40S. The first step per process carries CUDA warm-up.

### Which experiments to re-run for the seconds axis
Re-run the standard experiment scripts on a **fresh seed** (e.g. `1394`) on **L40S** — timing is
captured automatically (see section 4). No `--attack`/flag changes needed.
- **RL** — `run_rl_HB_experiments.sh` / `run_rl_JB_experiments.sh` (already instrumented).
- **Baselines (PAIR / JailBroken / GCG, and GCG-transfer if reported)** —
  `run_HB_experiments.sh` / `run_JB_experiments.sh`.
- Targets covered by those scripts: `qwen2.5_0.5b/3b/7b`, `tulu3_8b_{base,sft,dpo,rlvr}`,
  `qwen3_4b`, `qwen3_4b_saferl` (HarmBench primary; JB if you report JB time).

The scripts run the full `n_prompts` (200 HB / 100 JB). If you only need a representative time
axis and want to save compute, a fixed ~50-prompt subset per (attack, model) is enough — but the
full run also refreshes risk numbers on the new seed, so it's the simpler path.

Dollars needs **none** of these reruns.

---

## 3. Bash-script status (batch runs)

**Wired.** The batch flow now produces all four axes for every model:

```bash
bash run_cost_evaluations.sh    # $COST carries --pricing-config configs/pricing.yaml
bash run_cost_plots.sh          # AXES="tokens flops seconds dollars"
```

- `run_cost_evaluations.sh` passes `--pricing-config configs/pricing.yaml` via the `$COST` variable.
- `run_cost_plots.sh` drives every loop from `$AXES`, overridable per invocation:
  `AXES="dollars" bash run_cost_plots.sh` re-plots just one axis.

### Known gap: the summary table is TFLOPs-only

`<axis>/cost_summary_table.csv` is **not** axis-aware. `save_summary_table()` in
`plot_cost_curves.py` copies the precomputed `cost_summary_metrics.csv` — which
`compute_attack_costs.py` builds with `compute_col="mean_total_tflops"` — into every axis
directory unchanged. So `seconds/cost_summary_table.csv` and `dollars/cost_summary_table.csv`
are byte-identical to the TFLOPs table, and their `C@τ` / `CAURC` / `R@c` values are in TFLOPs,
not seconds or dollars.

The **curves** (`cost_aggregated_*.png`, `cost_comparison_*.png`) are correct per axis — only the
scalar tables are affected. For numeric cross-model/ablation comparison on seconds or dollars,
read the plotted curves, or plumb `compute_col` (and axis-appropriate `c_targets`) through
`compute_cost_summary_metrics` first.

---

## 4. Capturing wall-clock when you submit runs (operational notes)

**Timing is automatic — no flags.** The instrumentation lives in the run path (`run_trial` for
baselines, `run_prompt_rl` for RL), so **every** `run_inference.py` execution writes a measured
`seconds` into each `StepResult` and a `metadata.gpu` into each `TrialRecord`, straight into
`results.jsonl`. What's recorded is the **attack-compute interval per step** (CUDA-synced
`generate + judge + refine`; for RL, per-candidate `env.score` + the round's amortized
group-generation/GRPO-update) — **not** start-to-end wall time (model load / queue excluded).

So the existing experiment scripts capture wall-clock as-is:
- `run_HB_experiments.sh` / `run_JB_experiments.sh` → baselines (pair / jailbroken / gcg).
- `run_rl_HB_experiments.sh` / `run_rl_JB_experiments.sh` → RL. One `submit "<job>" "$BASE --model
  <m> --seeds <s>"` per (model, seed) is all that's needed.

**Two conditions that gate a usable seconds axis:**

1. **Run on one fixed GPU — L40S.** The `submit` function picks the GPU by login host, so submit
   from the Killarney login node:

   | Login host | GPU |
   |---|---|
   | `klogin*` → `submit_killarney.sbatch` | **L40S** ✅ |
   | `*.fir.alliancecan.ca` | H100 ❌ |
   | `trig-login01` (Trillium) | A100 ❌ |

   All attacks (baselines **and** RL) must land on the same GPU or their times aren't comparable.

2. **`--resume` only times prompts that actually execute.** A fresh seed (nothing in
   `results.jsonl` yet) runs — and times — every prompt. Re-running a seed that already has
   results **skips** those prompts, leaving them `NaN` for seconds. Keep baselines and RL on the
   **same, fresh seed** (e.g. all on `1394`) so every attack is timed and aggregates cleanly.

Dollars is the only axis *not* produced by the run — add it post-hoc at the cost step
(`--pricing-config`, section 1).

---

## Verification
- Dollars end-to-end (no rerun): run section 1 on one model; confirm the four dollar columns
  appear and `mean_total_dollars` == target + judge + attacker.
- Seconds: after an L40S rerun, confirm `metadata.gpu` is the L40S, `mean_total_seconds` is
  finite and monotonic in λ, and summed per-step `seconds` ≈ the per-trial `elapsed_seconds`.
- Unit tests: `uv run pytest tests/test_cost_axes.py`.
