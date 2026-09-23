# How the wall-clock and dollar cost axes are computed

Reference for the two non-token cost axes: what `mean_total_seconds` and `mean_total_dollars`
actually measure, where the numbers come from, and how to read them. Companion to
[`cost_axes_runbook.md`](cost_axes_runbook.md), which covers *how to produce* them.

The short version: **seconds is measured at run time; dollars is derived post-hoc from token
counts.** That asymmetry drives everything else on this page — dollars can be recomputed for any
experiment ever run, seconds only exists for runs made after the timing instrumentation landed.

---

## 1. Wall-clock — `mean_total_seconds`

Measured during the run and written into each `StepResult.seconds` in `results.jsonl`. Nothing is
modeled or extrapolated.

### Baselines (PAIR / JailBroken / GCG)

One timed interval per step, in `src/rup/pipeline/budgeted_refinement.py:86-110`. Both ends are
CUDA-synced so the `perf_counter` delta reflects real device time rather than async kernel-launch
time:

```python
_cuda_sync(); t_step = time.perf_counter()

y = model.generate(p_used)      # target
z = judge.judge(p_used, y)      # judge
p = attack.refine(p_used, y, z, t)   # attacker — skipped when z == 1 (early stop)

_cuda_sync(); sec = time.perf_counter() - t_step
```

So one step = `generate + judge + refine`, with all three components fused into a single scalar.

> **Seconds is not decomposable.** Unlike tokens and FLOPs, there are no per-component
> `*_seconds` columns — only the fused total. If you need to attribute time to target vs. judge
> vs. attacker, the instrumentation would have to be split first.

On the success step the `refine` call is skipped (the loop breaks), so that step's time covers
only `generate + judge`.

### RL (GRPO)

RL needs different handling because GRPO does per-*round* work that belongs to no single query.
In `src/rup/pipeline/rl_refinement.py`:

1. **The raw-behavior probe** (`p^0`, mirroring PAIR) is timed on its own — line 256-258.
2. **Each candidate** gets its own `env.score()` interval — line 295-299.
3. **The round's shared attacker work** — group generation (`gen_time`) plus the GRPO update
   (`update_time`) — is amortized evenly across the candidates actually scored, line 349-352:

```python
overhead = (gen_time + update_time) / scored
for sr in round_steps:
    sr.seconds = (sr.seconds or 0.0) + overhead
```

The GRPO update is skipped when the round already succeeded (`t_star is None` guard), so a
successful round's steps carry only `gen_time / scored` as overhead.

### What the interval excludes

Model loading, tokenizer init, SLURM queue time, and checkpoint I/O are all outside the timed
region. The axis sums the per-step intervals, so it measures **attack compute**, not
"time to jailbreak from a cold start."

`TrialRecord.metadata.elapsed_seconds` *does* record start-to-end per trial (from `t0`), but the
cost axis does not use it. A useful sanity check is that summed per-step `seconds` ≲
`elapsed_seconds`.

The first step in each process still absorbs CUDA warm-up, which inflates it relative to steady
state.

### Hardware comparability

`TrialRecord.metadata.gpu` records the device (via `_gpu_name()`), because times are only
comparable within one GPU model. All current timed results are on `NVIDIA L40S`. The axis label is
hardcoded to match — `"Cumulative attack seconds (L40S)"` in
`scripts/plot_cost_curves.py:137` — so if you ever time on other hardware, that label and any
cross-GPU aggregation both need revisiting.

---

## 2. Dollars — `mean_total_dollars`

Computed entirely post-hoc from token counts already present in `results.jsonl`. This is why the
dollar axis needs **no experiment reruns** — it applies to every experiment already on disk.

### The formula

Each component is priced at **its own model's** rate
(`_component_dollars`, `src/rup/metrics/cost_mapper.py:192`):

```python
component_$ = (in_tok * usd_per_1m_input + out_tok * usd_per_1m_output) / 1_000_000.0
```

and the three components are summed:

```
mean_total_dollars = mean_target_dollars + mean_judge_dollars + mean_attacker_dollars
```

The input/output split comes from the `StepCost` dataclass (`cost_mapper.py:52-82`):
`{target,judge,att}_in` are prompt tokens *read*, `_out` are tokens *generated*. Token counts use
each model's exact HuggingFace tokenizer, so they are not estimates.

### Which model prices which component

| Component | Model | Source |
|---|---|---|
| target | the model under attack | `record.model_id` |
| judge | `llama3.1-8b-instruct` | `--judge-model` (default) |
| attacker | `qwen2.5-7b-instruct` | `_ATTACKER_MODEL_ID`, `cost_mapper.py:142` (hardcoded) |

Rates are read from `configs/pricing.yaml` by `load_pricing()`. The dollar columns are `NaN`
unless `--pricing-config` is passed — `_step_dollars` returns `(nan, nan, nan)` when no pricing
config was loaded (`cost_mapper.py:198-209`).

> **Unpriced models silently cost $0.** `_price()` (`cost_mapper.py:181`) falls back to `0.0`
> with a *one-time* `warnings.warn`. If you add a target and forget to price it, its dollar curve
> is a flat zero rather than an error. `pricing.yaml` must cover every target, the judge, and the
> attacker.

### Two interpretation caveats

**Dollars is currently a rescaled token axis.** `configs/pricing.yaml` sets
`usd_per_1m_input == usd_per_1m_output` for every model, so the in/out split has no effect today.
The only thing separating dollars from tokens is the size tier — `$0.10` (≤4B), `$0.20` (4.1–8B),
`$0.30` (8.1–21B). The two axes therefore diverge *only* across models in different tiers, or if
you introduce asymmetric input/output rates. Rates are flagged in the config as placeholder
Together AI reference values — **verify before publishing.**

**GCG's search cost has no dollar equivalent.** GCG's 128-candidate gradient search is counted on
the FLOP axis (`_GCG_NUM_CANDIDATES`, `gcg_backward_mult=3.0`) but gradient-based search is not a
hosted-API operation, so no per-token price applies. This is why GCG reads as cheap in dollars yet
expensive in seconds and TFLOPs — measured on `qwen2.5-3b-instruct`:

| Attack | seconds | dollars | tokens | TFLOPs |
|---|---|---|---|---|
| gcg | 56.5 | $0.0020 | 18,843 | 248.2 |
| pair | 60.6 | $0.0005 | 2,950 | 59.5 |
| jailbroken | 28.7 | $0.0002 | 1,404 | 25.3 |
| rl | 27.5 | $0.0003 | 1,573 | 31.2 |

The dollar axis prices the **queries**, not the **search**. Read it as "cost to rent this attack
from a hosted API," which is the wrong model for GCG and the right one for PAIR/RL.

---

## 3. Shared accumulation over the pressure budget

Both axes accumulate identically, in `aggregate_costs` (`cost_mapper.py:565-580`). Per-step costs
are cumulative-summed once per trial, then read off at the budget:

```python
k = min(lam, n_steps)   # steps actually consumed — early-stop aware
c = cumul[k]
```

so an early-stopped trial contributes its true shorter cost, not a budget-length extrapolation.
The result is averaged over **all** trials in the seed — successes and failures alike:

```python
"mean_total_seconds": s[6] / n,
"mean_total_dollars": (s[7] + s[8] + s[9]) / n,
```

### NaN propagation (seconds only)

Untimed steps become `math.nan` (`cost_mapper.py:552`), and because the accumulator is a plain
sum, **one untimed step makes that entire (seed, λ) cell NaN.** Consequences:

- Pre-instrumentation runs have `mean_total_seconds = NaN` for all λ ≥ 1.
- `λ = 0` is a genuine `0.0`, not NaN — zero steps consumed, so there is nothing to sum.
- At seed aggregation, `plot_cost_curves.py:330` uses pandas `.mean()`, which **skips NaN**. Seeds
  without timing therefore drop out silently rather than poisoning the curve.

That last point is convenient but has a real consequence: for an attack where only some seeds are
timed, the x-value (seconds) comes from a subset of the seeds that produced the y-value (risk).
Currently seconds rests on seed 1394 alone, so the seconds axis carries **no x-axis confidence
interval**, while dollars aggregates over every seed normally.

---

## 4. Summary

| | wall-clock | dollars |
|---|---|---|
| Origin | measured at run time | derived post-hoc from tokens |
| Needs reruns? | yes — only instrumented runs have it | no — works on all existing results |
| Per-component breakdown | ✗ fused into one number | ✓ target / judge / attacker |
| Hardware-dependent | yes (L40S) | no |
| Missing data | `NaN`, drops out at aggregation | `NaN` without `--pricing-config` |
| Multi-seed CI | no (single timed seed) | yes |
| Blind spot | cold-start / load time excluded | GCG search not priced |

**Known gap:** `<axis>/cost_summary_table.csv` is not axis-aware — it is always the TFLOPs table.
See the corresponding section in [`cost_axes_runbook.md`](cost_axes_runbook.md#known-gap-the-summary-table-is-tflops-only).
For numeric comparison on seconds or dollars, read the curves.
