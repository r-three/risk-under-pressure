<table border="0" cellpadding="0" cellspacing="0" style="border: none; border-collapse: collapse; background: transparent;">
  <tr style="border: none; background: transparent;">
    <td style="border: none; padding: 0;"><img src="figures/logo.png" height="60" alt="Risk Under Pressure logo"/></td>
    <td valign="middle" style="border: none; padding-left: 12px;"><h1 style="margin: 0;">Risk Under Pressure</h1></td>
  </tr>
</table>

**Compute-Aware Evaluation of Adversarial Robustness in Language Models**

[![Paper](https://img.shields.io/badge/paper-preprint-blue)](https://arxiv.org/pdf/2606.11409)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Most jailbreak benchmarks report attack success rate (ASR) at a fixed query budget — which
implicitly treats a cheap template jailbreak and an expensive gradient-based GCG attack as
equivalent. They're not: compute costs across attack strategies vary by orders of magnitude, so a
high ASR can mean "trivially broken" or "extremely expensive to break," and you can't tell which
from ASR alone.

**Risk Under Pressure** replaces the query-count axis with cumulative FLOPs — a hardware-agnostic
measure of actual attacker effort. Instead of "did the attack succeed within N queries?", you get
*risk-compute curves* showing how jailbreak success scales with compute budget, summarized by
two metrics: compute to reach a target risk level (`C@τ`) and risk gained per FLOP (`AE`).

![Risk Under Pressure Framework](figures/rup_framework.png)

---

## Contents

| Where | What |
|---|---|
| [Install](#install) · [Quickstart](#quickstart) | Get one attack running in ~10 minutes |
| [**Reproducing the paper**](#reproducing-the-paper) | Every table and figure → the exact command |
| [Where results land](#where-results-land) | Directory layout of `$RUN_ROOT` |
| [Reference](#reference) | Models, attacks, benchmarks, judges |
| [Extending](#extending-the-framework) | Add a model / attack / benchmark |
| [Known gotchas](#known-gotchas) | Read before launching a sweep |
| [`docs/design_notes.md`](docs/design_notes.md) | *Why* these models, judges, attackers and prices |
| [`docs/ablations.md`](docs/ablations.md) | Judge, attacker and severity ablations in depth |
| [`docs/rl_attack.md`](docs/rl_attack.md) | The GRPO adaptive attacker: budget, reward, LoRA cost |
| [`docs/cost_axes_computation.md`](docs/cost_axes_computation.md) | How seconds and dollars are computed |
| [`docs/cost_axes_runbook.md`](docs/cost_axes_runbook.md) | How to populate those axes |
| [`docs/python_api.md`](docs/python_api.md) | Using the components directly from Python |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Code conventions, cost-model requirements |

---

## Install

```bash
git clone https://github.com/Malikeh97/risk-under-pressure && cd risk-under-pressure
uv venv && source .venv/bin/activate
uv pip install -e .

cp .env.example .env     # then set HF_TOKEN (gated models: Llama, Tulu3)
```

Python 3.11, CUDA 12.6, one GPU. `HF_TOKEN` is the only required key; `~/hf_token.txt` works as a
fallback. Everything caches to `$SCRATCH/huggingface`.

**On a SLURM cluster**, build the environment as a job instead, then let the run scripts activate
it for you:

```bash
mkdir -p logs && sbatch setup/create_env_killarney_uv.sh   # or _fir_ / _trillium_
```

`setup/start_env.sh` loads the modules, activates `.venv`, sets `SCRATCH`, and defines the
`submit` helper that every `run_*.sh` uses (it skips jobs already running or finished in the last
2 days). `setup/judge_env.sh` then derives `RUN_ROOT` / `PLOT_ROOT` from the selected `JUDGE`.
Cluster profiles live in `setup/submit_{killarney,fir,trillium}.sbatch` — one GPU, 8 CPUs,
128 GB, 23 h. **Edit the `--account` line to your own allocation.**

---

## Quickstart

One model, one attack, 5 prompts, no cluster — this is the whole pipeline in miniature:

```bash
python scripts/run_inference.py \
    --experiment configs/experiments/base.yaml \
    --model qwen2.5_0.5b --attack jailbroken \
    --n-prompts 5 --lambda-max 4 --seeds 42 \
    --output-dir outputs/demo

python scripts/run_evaluation.py \
    --results-dir outputs/demo/harmbench/qwen2.5-0.5b-instruct \
    --pressure-levels 0 1 2 4 \
    --format csv --output outputs/demo/metrics.csv --print-table

python scripts/compute_attack_costs.py \
    --results-dir outputs/demo/harmbench/qwen2.5-0.5b-instruct \
    --metrics-csv outputs/demo/metrics.csv \
    --pricing-config configs/pricing.yaml

python scripts/plot_cost_curves.py \
    --cost-csv outputs/demo/cost_metrics.csv \
    --output-dir outputs/demo/plots --x-axis flops
```

---

## Reproducing the paper

### The four phases

Every result in the paper is produced by the same pipeline. Only Phase 1 needs a GPU.

| Phase | Script | Driver | GPU |
|---|---|---|---|
| **1 — Run attacks** | `scripts/run_inference.py` | `run_{HB,JB}_experiments.sh`, `run_rl_*_experiments.sh` | ✅ |
| **2 — Risk metrics** | `scripts/run_evaluation.py` | `run_evaluations.sh` | ❌ |
| **2.5 — Cost metrics** | `scripts/compute_attack_costs.py` | `run_cost_evaluations.sh` | ❌ |
| **2.6 — Severity** (optional) | `scripts/score_severity.py` | `run_severity_scoring.sh` | ✅ |
| **3 — Plots** | `scripts/plot_results.py`, `scripts/plot_cost_curves.py` | `run_plots.sh`, `run_cost_plots.sh` | ❌ |

Phases 2 and 2.5 also emit per-harm-category variants of every file automatically.

### Paper artifact → command

The `run_*.sh` drivers are the **canonical path** — they are what produced the numbers in the
paper. Each is a list of `submit …` lines grouped by study, with most lines **commented out** so
you only pay for what you need. Reproducing an artifact means: uncomment its block, run the
driver, wait, then run Phases 2/2.5/3.

| Paper artifact | Phase 1 | Then |
|---|---|---|
| **Table 1** — HarmBench, 9 models × 3 attacks | `run_HB_experiments.sh` → *MODEL SIZE* + *TRAINING STAGE* + *SAFETY ALIGNMENT* blocks | eval → cost; read `cost_summary_metrics.csv` |
| **Fig. training stage** (HB) | `run_HB_experiments.sh` → *TRAINING STAGE STUDY* (Tulu3 ×4) | `run_cost_plots.sh` → `ablations/tulu3_training` |
| **Fig. model size** (HB) | `run_HB_experiments.sh` → *MODEL SIZE STUDY* (Qwen2.5 ×3) | `run_cost_plots.sh` → `ablations/qwen_size` |
| **Fig. safety alignment** (HB) | `run_HB_experiments.sh` → *SAFETY ALIGNMENT* (Qwen3-4B ±SafeRL) | `run_cost_plots.sh` → `ablations/safety_alignment` |
| **Fig. attack transfer** | Qwen2.5-0.5B GCG first, then `run_transfer_experiments.sh` | eval → cost → plots |
| **Fig. per-category** | same runs as Table 1 | `--category-metrics-csv` is passed automatically by `run_cost_plots.sh` |
| **App. JailbreakBench** (all of the above) | `run_JB_experiments.sh` (same blocks; `--n-prompts 100`) | identical Phase 2/3 |
| **App. RL adaptive attack** | `run_rl_HB_experiments.sh`, `run_rl_JB_experiments.sh` | eval → cost; RL columns appear automatically |
| **App. Gemma 3** (model size, 2nd family) | `run_{HB,JB}_experiments.sh` → *MODEL SIZE (2nd family)* block | `ablations/gemma_size` |
| **App. OLMo 2** (training stage, 2nd family) | `run_{HB,JB}_experiments.sh` → *TRAINING STAGE (2nd family)* block | `ablations/olmo2_training` |
| **App. judge robustness** (Flow-Judge) | `bash run_rejudge.sh` — re-scores recorded responses, no re-run | `scripts/judge_agreement.py` (κ), `scripts/compare_judges.py` |
| **App. judge-excluded cost** | *no new runs* | `AXES=flops_nojudge bash run_cost_plots.sh`; `cost_summary_metrics_nojudge.csv` |
| **App. wall-clock / dollar axes** | *no new runs for dollars*; seconds needs a fresh L40S run | `AXES="seconds dollars" bash run_cost_plots.sh` — see [runbook](docs/cost_axes_runbook.md) |
| **App. severity (0–5)** | `bash run_severity_scoring.sh` — re-reads existing `results.jsonl` | `bash run_severity_scoring.sh report` |
| **App. cross-benchmark consistency** | *no new runs* | Spearman ρ over the two benchmarks' `cost_summary_metrics.csv` (computed ad hoc — no dedicated script) |
| **App. attacker ablation** | `bash run_attacker_ablation.sh` | `… eval`, `… plots` |

### Seeds

The paper reports **10 seeds**. In `run_{HB,JB}_experiments.sh` each model block lists all ten
(`1394 2 100 42 5431 2002 256 512 123 5`) with **only `1394` uncommented** — uncomment the rest to
reproduce the published confidence intervals. The RL drivers instead take a `SEEDS` variable
(default: all ten of `1394 42 123 256 512 1024 1997 2002 5431 7919`); narrow it with
`SEEDS="1394 42" bash run_rl_HB_experiments.sh`.

Each (model, attack, seed) is a separate job, so ten seeds means ten jobs, not one ten-times-longer
job. Every command passes `--resume`, so a job that hits the 23 h limit is simply resubmitted.

### Step by step on SLURM

```bash
# Phase 1 — edit the driver to uncomment your study's block, then:
bash run_HB_experiments.sh                 # HarmBench, static attacks
bash run_JB_experiments.sh                 # JailbreakBench
bash run_rl_HB_experiments.sh              # RL/GRPO adaptive attack (most expensive)

ATTACKS="gcg" bash run_HB_experiments.sh   # subset: GCG is ~5× the others

# Phase 2 + 2.5 — login node, no GPU (uncomment the matching model blocks)
bash run_evaluations.sh
bash run_cost_evaluations.sh

# Phase 3
bash run_plots.sh                          # risk vs λ
bash run_cost_plots.sh                     # risk vs compute
AXES="flops tokens dollars" bash run_cost_plots.sh
```

Select a different safety judge anywhere with `JUDGE=<config-name>` (see
[Reference](#reference)); each judge writes to its own tree, so nothing is overwritten.

### Without SLURM

The drivers are convenience wrappers — every one of them ultimately runs the command shown in
[Quickstart](#quickstart). To reproduce one cell of Table 1 directly:

```bash
python scripts/run_inference.py \
    --experiment configs/experiments/base.yaml \
    --benchmark harmbench --model tulu3_8b_sft --attack gcg \
    --seeds 1394 --output-dir $RUN_ROOT --resume
```

`configs/experiments/paper/*.yaml` bundle the model lists per study
(`model_size.yaml`, `training_stage.yaml`, `safety_alignment.yaml`, `attack_transfer.yaml`,
`attacker_size.yaml`, plus the `_gemma3` / `_olmo2` second families) if you prefer one command per
study over one per cell. Note their `seeds:` lists do **not** include 1394 — pass `--seeds`
explicitly to match the published runs.

### Before you burn GPU-hours

```bash
pytest tests/                              # 163 CPU-only unit tests, no downloads, ~seconds
bash run_judge_smoke.sh                    # validate each judge against a 22-case control set
bash run_judge_smoke.sh check              # → hard pass/fail gate
bash run_rl_smoke.sh                       # full RL path end-to-end; prints SMOKE TEST PASSED
bash run_severity_scoring.sh dry-run       # exact judge-call count, no GPU
bash run_attacker_ablation.sh smoke        # verifies each attacker actually rewrites prompts
```

All smoke runs write to throwaway trees (`$SCRATCH/rl_smoke`, `$SCRATCH/rup_judge_smoke`, …) and
never touch `$SCRATCH/rup`. The judge gate matters most: a judge that loads but never receives its
rubric returns SAFE for everything, which is indistinguishable from a perfectly aligned target in
every downstream curve.

---

## Where results land

`RUN_ROOT` is `$SCRATCH/rup` for the default judge, `$SCRATCH/rup/judges/<judge_id>` otherwise;
`PLOT_ROOT` is `$RUN_ROOT/plots`.

```
$RUN_ROOT/
├── <benchmark>/                      # harmbench | jailbreakbench
│   └── <model_id>/<seed>/<attack>/   # attack: pair | jailbroken | gcg | rl
│       ├── results.jsonl             # one TrialRecord per prompt (each step: response,
│       │                             #   judgment, token counts, seconds, metadata.gpu)
│       ├── training_trace.jsonl      # RL only: per-GRPO-round rollouts, rewards, advantages
│       ├── severity_scores.jsonl     # Phase 2.6, per executed step
│       └── rejudge__<judge_id>.jsonl # offline re-scoring sidecar
└── plots/  (= $PLOT_ROOT)
    └── <benchmark>/<model_id>/
        ├── metrics.csv  metrics_summary.csv  metrics_by_category.csv
        ├── severity_metrics.csv  severity_summary.csv
        ├── cost/cost_metrics.csv
        │   cost_summary_metrics.csv          # ← C@τ, AE, CAURC: the paper's tables
        │   cost_summary_metrics_nojudge.csv  # ← same, judge FLOPs excluded
        ├── plots/seeds/*.png                 # risk vs λ
        └── <axis>/*.png                      # risk vs cost; axis ∈ tokens|flops|seconds|dollars
```

Variant attack directories: `pair__<attacker>` / `rl__<attacker>` (attacker ablation),
`transfer_gcg_from_<source_model_id>` (transfer).

---

## Reference

### Models

| Family | Configs | Role in the paper |
|---|---|---|
| **Qwen2.5 Instruct** | `qwen2.5_{0.5b,3b,7b}` | model-size study; 7B is also the default attacker |
| **Gemma 3 Instruct** | `gemma3_{270m,1b,4b}_it` | model-size study, 2nd family |
| **Tulu3 8B** | `tulu3_8b_{base,sft,dpo,rlvr}` | training-stage study |
| **OLMo 2 1B** | `olmo2_1b_{base,sft,dpo,rlvr1,instruct}` | training-stage study, 2nd family (5 rungs) |
| **Qwen3** | `qwen3_4b`, `qwen3_4b_saferl`, `qwen3_8b` | safety alignment; 8B is the transfer target |

Adding a model is a YAML file — see [Extending](#extending-the-framework). Quantization policy,
the OLMo 2 fifth rung, and the Gemma 3 dual loader path are explained in
[`docs/design_notes.md`](docs/design_notes.md).

### Attacks

| Attack | Type | Per-step compute |
|---|---|---|
| **JailBroken** | template | `2N·L_gen + 2N_J·L_J` |
| **PAIR** | black-box, attacker LLM | `+ 2N_A·L_att` |
| **GCG** | white-box, gradient | `(128 + β_bwd)·2N·L_opt + 2N·L_gen + 2N_J·L_J` |
| **RL (GRPO)** | adaptive, trains the attacker | `{0, 2, 8}·N_A·L_att + 2N·L_gen + 2N_J·L_J` per query |
| **Transfer** | replay | same as JailBroken |

N = target params, N_A = attacker, N_J = judge, L = tokens. RL's attacker term is LoRA-aware and
billed per candidate: `0` for the raw probe, `8N` on a GRPO-updated round, `2N` on the winning
round — see [`docs/rl_attack.md`](docs/rl_attack.md) for the budget arithmetic, reward and early
stopping. Formulas live in `src/rup/metrics/cost_mapper.py`.

### Benchmarks and judges

HarmBench (200 behaviors, 6 categories) and JailbreakBench (100, 10). The judge defines what counts
as a jailbreak, so every number is conditioned on it — which is why three are wired in:

| `JUDGE=` | Model | `params_b` |
|---|---|---|
| `llama3.1_8b_instruct_judge` | Llama-3.1-8B-Instruct | 8.03 (default) |
| `olmo3_7b_instruct_judge` | Olmo-3-7B-Instruct | 7.30 |
| `gemma3_4b_it_judge` | Gemma-3-4B-IT | 3.88 |
| `flow_judge_v01` | Flow-Judge-v0.1 (Phi-3.5 lineage) | 3.82 |

### Cost axes

`--x-axis {flops,tokens,seconds,dollars}` plus a `_nojudge` variant of each except `seconds`:

| Axis | Charges | Answers |
|---|---|---|
| `flops` / `tokens` / `dollars` | target + judge + attacker | cost to **reproduce this measurement** |
| `*_nojudge` | target + attacker | what the **attack costs an adversary** |

The split is not cosmetic: the judge is 10% of GCG's FLOPs but ~59% of JailBroken's, so it changes
which attack is cheapest. **Use `_nojudge` for anything comparing across judges.** Details:
[`docs/cost_axes_computation.md`](docs/cost_axes_computation.md).

### Severity (0–5)

`run_severity_scoring.sh` re-reads responses already in `results.jsonl` with a graded rubric — no
experiment is re-run, and only executed steps are scored. Two rubrics ship: `detail` (default;
how much harmful detail is on the page) and `uplift` (the original; real-world efficacy). The
paper uses `detail`, because `uplift` put 89% of steps at 0 and scored 53% of *successful*
jailbreaks as 0. Full metric definitions: [`docs/ablations.md`](docs/ablations.md).

---

## Extending the Framework

**A new model** is a YAML file — no Python:

```yaml
# configs/models/my_llama_3b.yaml
model_id: "llama-3.2-3b-instruct"
backend: "huggingface"
hf_name: "meta-llama/Llama-3.2-3B-Instruct"
params_b: 3.21          # required — drives the FLOP axis
model_type: "instruct"
quantization: "4bit"
device: "cuda"
generation: {max_new_tokens: 512, temperature: 0.7, do_sample: true, top_p: 0.9}
```

**A new attack**: add `configs/attacks/<id>.yaml`, implement `AttackPolicy.{initialize,refine}` in
`src/rup/attacks/`, register it in `factory.py`, **and add its per-step FLOP formula to
`step_cost()` in `src/rup/metrics/cost_mapper.py`** — the cost metrics are only as honest as that
formula.

**A new benchmark**: implement `Benchmark` in `src/rup/benchmarks/` and register it. See
[CONTRIBUTING.md](CONTRIBUTING.md).

---

## Known gotchas

- **The run scripts ship mostly commented out.** Uncommenting is the interface. `run_HB/JB_experiments.sh`
  currently enable Qwen2.5 / Tulu3 / Qwen3 at seed 1394 only — and their header comments claim the
  opposite (that Gemma 3 and OLMo 2 are the enabled ladders). Trust the code, not the header.
- **`run_transfer_experiments.sh` is entirely inert** — every `submit` line is commented, so running
  it today is a no-op. Uncomment the seed lines you want.
- **Seed 1394 is in no experiment YAML** but is what every enabled static-attack line passes.
- **`run_cost_plots.sh` pins `AXES=flops`** (line 55); the multi-axis default is commented out just
  above. Override with `AXES="..."`.
- **`run_rejudge.sh` and `run_judge_cost_main.sh` must run on a login node** — on compute nodes
  `$SCRATCH` resolves elsewhere and the `rup` tree isn't there. Both deliberately skip-and-warn
  instead of `set -e` aborting on a missing model.
- **`scripts/bootstrap_rl_cost_ci.py` hardcodes an absolute repo root and output path** and is not
  portable as written.
- **`--rl-num-generations` must match `num_generations`** in `configs/attacks/rl.yaml` (default 8),
  or the RL cost reconstruction is wrong.
- **`mean_total_seconds` is NaN for older runs** — wall-clock is measured, not modeled, so it only
  exists for runs made after the timing instrumentation, on one GPU.

---

## Citation

```bibtex
@article{ehghaghi2026riskpressure,
  title         = {Risk Under Pressure: Compute-Aware Evaluation of Adversarial Robustness in Language Models},
  author        = {Ehghaghi, Malikeh and Ecsedi, Boglarka and Chechik, Marsha and Raffel, Colin},
  journal       = {arXiv preprint arXiv:2606.11409},
  year          = {2026},
  eprint        = {2606.11409},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2606.11409},
}
```

## Contributing

New models, attacks and benchmarks are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
Released under the [MIT License](LICENSE).
