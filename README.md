<table border="0" cellpadding="0" cellspacing="0" style="border: none; border-collapse: collapse; background: transparent;">
  <tr style="border: none; background: transparent;">
    <td style="border: none; padding: 0;"><img src="figures/logo.png" height="60" alt="Risk Under Pressure logo"/></td>
    <td valign="middle" style="border: none; padding-left: 12px;"><h1 style="margin: 0;">Risk Under Pressure</h1></td>
  </tr>
</table>

**Compute-Aware Evaluation of Adversarial Robustness in Language Models**

[![Paper](https://img.shields.io/badge/paper-preprint-blue)](https://arxiv.org/pdf/2606.11409)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Most jailbreak benchmarks report attack success rate (ASR) at a fixed query budget — which implicitly treats a cheap template jailbreak and an expensive gradient-based GCG attack as equivalent. They're not: compute costs across attack strategies vary by orders of magnitude, so a high ASR can mean "trivially broken" or "extremely expensive to break," and you can't tell which from ASR alone.

**Risk Under Pressure** replaces the query-count axis with cumulative FLOPs — a hardware-agnostic measure of actual attacker effort. Instead of "did the attack succeed within N queries?", you get *risk-compute curves* that show how jailbreak success rate scales with compute budget. Two summary metrics capture what the curve means in practice: how much compute it takes to reach a target risk level, and how much risk you get per FLOP on average.

<!-- > **Paper**: Ehghaghi, Ecsedi, Chechik & Raffel — *Risk Under Pressure: Compute-Aware Evaluation of Adversarial Robustness in Language Models* (2026) -->

![Risk Under Pressure Framework](figures/rup_framework.png)

---

## Setup

```bash
git clone https://github.com/Malikeh97/risk-under-pressure && cd risk-under-pressure
uv venv && source .venv/bin/activate
uv pip install -e .

# Copy and fill in your HuggingFace token
cp .env.example .env
```

---

## Replicating Paper Experiments

Each experiment follows the same three phases:

| Phase | Script | GPU? |
|---|---|---|
| **1 — Run attacks** | `scripts/run_inference.py` | Yes |
| **2a — Compute risk metrics** | `scripts/run_evaluation.py` | No |
| **2b — Compute FLOP costs** | `scripts/compute_attack_costs.py` | No |
| **3 — Plot** | `scripts/plot_results.py`, `scripts/plot_cost_curves.py` | No |

Phase 2a automatically writes both `metrics.csv` (overall) and `metrics_by_category.csv` (per harm category) when run with `--format csv`.

---

### Model Size Effect

Qwen2.5-Instruct at 0.5B, 3B, and 7B on HarmBench and JailbreakBench.

```bash
# Phase 1 — Run attacks (GPU required)
python scripts/run_inference.py \
    --experiment configs/experiments/paper/model_size.yaml \
    --output-dir outputs/model_size

# Phase 2a — Compute risk metrics
python scripts/run_evaluation.py \
    --results-dir outputs/model_size \
    --experiment configs/experiments/paper/model_size.yaml \
    --format csv \
    --output outputs/model_size/metrics.csv

# Phase 2b — Compute FLOP costs
python scripts/compute_attack_costs.py \
    --results-dir outputs/model_size \
    --metrics-csv outputs/model_size/metrics.csv
# → outputs/model_size/cost_metrics.csv

# Phase 3 — Plot risk-pressure curves (x-axis = λ)
python scripts/plot_results.py \
    --metrics-csv outputs/model_size/metrics.csv \
    --category-metrics-csv outputs/model_size/metrics_by_category.csv \
    --output-dir outputs/model_size/plots

# Phase 3 — Plot risk-compute curves (x-axis = TFLOPs)
python scripts/plot_cost_curves.py \
    --cost-csv outputs/model_size/cost_metrics.csv \
    --output-dir outputs/model_size/cost_plots \
    --x-axis tflops
```

---

### Training Stage Effect

Tulu3 8B across four training stages: Base → SFT → DPO → RLVR.

```bash
# Phase 1 — Run attacks (GPU required)
python scripts/run_inference.py \
    --experiment configs/experiments/paper/training_stage.yaml \
    --output-dir outputs/training_stage

# Phase 2a — Compute risk metrics
python scripts/run_evaluation.py \
    --results-dir outputs/training_stage \
    --experiment configs/experiments/paper/training_stage.yaml \
    --format csv \
    --output outputs/training_stage/metrics.csv

# Phase 2b — Compute FLOP costs
python scripts/compute_attack_costs.py \
    --results-dir outputs/training_stage \
    --metrics-csv outputs/training_stage/metrics.csv
# → outputs/training_stage/cost_metrics.csv

# Phase 3 — Plot risk-pressure curves
python scripts/plot_results.py \
    --metrics-csv outputs/training_stage/metrics.csv \
    --category-metrics-csv outputs/training_stage/metrics_by_category.csv \
    --output-dir outputs/training_stage/plots

# Phase 3 — Plot risk-compute curves
python scripts/plot_cost_curves.py \
    --cost-csv outputs/training_stage/cost_metrics.csv \
    --output-dir outputs/training_stage/cost_plots \
    --x-axis tflops
```

---

### Safety Alignment Effect

Qwen3-4B (no safety training) vs Qwen3-4B-SafeRL (safety RL fine-tuned).

```bash
# Phase 1 — Run attacks (GPU required)
python scripts/run_inference.py \
    --experiment configs/experiments/paper/safety_alignment.yaml \
    --output-dir outputs/safety_alignment

# Phase 2a — Compute risk metrics
python scripts/run_evaluation.py \
    --results-dir outputs/safety_alignment \
    --experiment configs/experiments/paper/safety_alignment.yaml \
    --format csv \
    --output outputs/safety_alignment/metrics.csv

# Phase 2b — Compute FLOP costs
python scripts/compute_attack_costs.py \
    --results-dir outputs/safety_alignment \
    --metrics-csv outputs/safety_alignment/metrics.csv
# → outputs/safety_alignment/cost_metrics.csv

# Phase 3 — Plot
python scripts/plot_results.py \
    --metrics-csv outputs/safety_alignment/metrics.csv \
    --category-metrics-csv outputs/safety_alignment/metrics_by_category.csv \
    --output-dir outputs/safety_alignment/plots

python scripts/plot_cost_curves.py \
    --cost-csv outputs/safety_alignment/cost_metrics.csv \
    --output-dir outputs/safety_alignment/cost_plots \
    --x-axis tflops
```

---

### Attack Transfer

GCG suffix optimised on Qwen2.5-0.5B (surrogate), then replayed against Qwen3-8B (target). Phase 1a can be skipped if the model size experiment has already been run (the source results are reused).

```bash
# Phase 1a — Run GCG on the source model (skip if already done via model_size)
python scripts/run_inference.py \
    --experiment configs/experiments/paper/model_size.yaml \
    --model qwen2.5_0.5b \
    --attack gcg \
    --output-dir outputs/model_size

# Phase 1b — Replay GCG trajectories on the target model
python scripts/run_transfer_inference.py \
    --experiment configs/experiments/paper/attack_transfer.yaml \
    --source-results-dir outputs/model_size \
    --source-model qwen2.5-0.5b-instruct \
    --source-attack gcg \
    --target-models qwen3_8b \
    --output-dir outputs/attack_transfer \
    --resume

# Phase 2a — Compute risk metrics
python scripts/run_evaluation.py \
    --results-dir outputs/attack_transfer \
    --experiment configs/experiments/paper/attack_transfer.yaml \
    --format csv \
    --output outputs/attack_transfer/metrics.csv

# Phase 2b — Compute FLOP costs
python scripts/compute_attack_costs.py \
    --results-dir outputs/attack_transfer \
    --metrics-csv outputs/attack_transfer/metrics.csv

# Phase 3 — Plot
python scripts/plot_results.py \
    --metrics-csv outputs/attack_transfer/metrics.csv \
    --output-dir outputs/attack_transfer/plots

python scripts/plot_cost_curves.py \
    --cost-csv outputs/attack_transfer/cost_metrics.csv \
    --output-dir outputs/attack_transfer/cost_plots \
    --x-axis tflops
```

---

### RL-Based Adaptive Attack (GRPO)

A **per-prompt** adaptive attacker following *["The Attacker Moves Second"](https://arxiv.org/abs/2510.09023)*
(Nasr et al., 2025). For **each behavior**, Qwen2.5-7B-Instruct (+ a fresh LoRA) runs a short
GRPO optimization against the target — sample a group of candidate prompts, score them
(judge + perplexity shaping, guarded against reward hacking), and update the attacker's weights —
**stopping at the first successful jailbreak** or when the per-prompt query budget
(`--lambda-max`) is exhausted. There is **no separate training phase and no train/test split**:
it runs like any other attack, straight through `run_inference.py --attack rl`.

> **First-success early stopping** mirrors the other attacks (`run_trial` breaks at the first
> unsafe judgment), so RL's query count / pressure / cost stay comparable across attacks. This is
> a deliberate deviation from the paper, which runs a fixed budget with best-of scoring — it does
> **not** change the risk-vs-λ curve or the success labels (`first_success_step` is identical),
> only post-success queries are trimmed.

```bash
# Phase 1 — Run the per-prompt RL attack (GPU required; 7B bf16 attacker + LoRA)
#   COMPUTE: this is the most expensive attack — start on a subset with a modest budget.
python scripts/run_inference.py \
    --experiment configs/experiments/paper/model_size.yaml \
    --attack rl --n-prompts 50 --lambda-max 10 \
    --output-dir outputs/model_size

# Phase 2a — Compute risk metrics
python scripts/run_evaluation.py \
    --results-dir outputs/model_size \
    --experiment configs/experiments/paper/model_size.yaml \
    --format csv --output outputs/model_size/metrics.csv

# Phase 2b — Compute FLOP costs (LoRA-aware attacker cost; see note below)
#   --rl-num-generations must match num_generations in configs/attacks/rl.yaml (default 8)
python scripts/compute_attack_costs.py \
    --results-dir outputs/model_size \
    --metrics-csv outputs/model_size/metrics.csv \
    --rl-num-generations 8

# Phase 3 — Plot risk-compute curves (the `RL (GRPO)` series appears automatically)
python scripts/plot_cost_curves.py \
    --cost-csv outputs/model_size/cost_metrics.csv \
    --output-dir outputs/model_size/cost_plots \
    --x-axis tflops
```

The same recipe applies to the **Training Stage** ablation — point `--experiment` at
`training_stage.yaml` (targets `tulu3_8b_base/sft/dpo/rlvr`).

**Cross-model RL comparisons.** `run_cost_plots.sh` includes dedicated RL-only comparisons
(`plot_cost_curves.py --attacks rl --mode comparison`), which overlay one curve per model in a
single `cost_comparison_rl.png` (tokens + flops):

| Output dir | Compares |
|---|---|
| `harmbench/ablations/rl_qwen_size/{tokens,flops}/` | RL across Qwen2.5 0.5B / 3B / 7B |
| `harmbench/ablations/rl_tulu3_training/{tokens,flops}/` | RL across Tulu3 base / sft / dpo / rlvr |
| `harmbench/ablations/rl_all_models/{tokens,flops}/` | RL across all HarmBench targets |

**Per-prompt GRPO knobs** live in `configs/attacks/rl.yaml` `extra`: `num_generations` (GRPO
group size; paper uses up to 32), `session_rounds`, `beta` (KL penalty), `learning_rate`,
`max_completion_length`, `perplexity_weight` (α on the reward-shaping term). The **per-prompt
query budget** is `--lambda-max` (or `pressure_levels`/`lambda_max` in the experiment YAML).

**LoRA-aware FLOP cost.** RL's attacker cost is billed per recorded candidate (not a flat
multiplier): `0` for the raw-behavior step (the attacker never runs), `8N_A·L` for a candidate on
a round that received a GRPO update (generation + KL-reference forward + policy forward + LoRA
backward), and `2N_A·L` for a candidate on the winning round (generation only — the update is
skipped under early stopping). The LoRA backward is ~`2N` (activation grads only; frozen base
weights get no weight-gradient), so an updated candidate is `8N`, not full fine-tuning's `6N`.
`compute_attack_costs.py` reconstructs which steps were updated from `first_success_step` and
`--rl-num-generations`, so **it must match `num_generations`** in `configs/attacks/rl.yaml`.

**Training trace.** Each run also writes `training_trace.jsonl` next to `results.jsonl` — one JSON
line per GRPO round with every candidate rollout (prompt / response / reward / judgment), the
group-relative advantages, and the loss — so you can inspect how the trajectory drives the attack
toward a jailbreak.

**Batch submission.** `run_rl_HB_experiments.sh` (HarmBench) and `run_rl_JB_experiments.sh`
(JailbreakBench) submit one SLURM job per target across the full model set (Qwen2.5 0.5/3/7B,
Tulu3 8B base/sft/dpo/rlvr, Qwen3-4B, Qwen3-4B-SafeRL).

> **Compute:** per-prompt GRPO does a mini optimization *per behavior* (with backward passes),
> so it is far more expensive than the other attacks. Start with `--n-prompts ~50` (the paper's
> ~60-sample scale) and a modest `--lambda-max` / `num_generations`, and watch 48 GB VRAM
> (7B bf16 attacker + LoRA optimizer + 4-bit target + 4-bit judge + KV caches).

For a quick end-to-end check on a GPU node, submit the smoke test (2 prompts, budget 8):

```bash
mkdir -p logs && bash run_rl_smoke.sh    # submits one GPU job; prints "SMOKE TEST PASSED"
```

---

### Per-Category Analysis

Per-category breakdown is produced automatically by `scripts/run_evaluation.py` (with `--format csv`) alongside the overall `metrics.csv`. Pass the category CSV to the plotting scripts with `--category-metrics-csv` as shown above to get one figure per harm category. No additional experiment runs are needed.

---

### Summary Metrics

To print a formatted summary table (C@τ, AE, CAURC) for any experiment after Phase 2:

```bash
python scripts/run_evaluation.py \
    --results-dir outputs/<exp> \
    --experiment configs/experiments/paper/<exp>.yaml \
    --print-table
```

---

## Extending the Framework

### Adding a New Model (YAML only)

No Python changes required. Create `configs/models/<your_model>.yaml`:

```yaml
# configs/models/my_llama_3b.yaml
model_id: "llama-3.2-3b-instruct"
backend: "huggingface"
hf_name: "meta-llama/Llama-3.2-3B-Instruct"
params_b: 3.21          # required for FLOP calculation
model_type: "instruct"
quantization: "4bit"
device: "cuda"
generation:
  max_new_tokens: 512
  temperature: 0.7
  do_sample: true
  top_p: 0.9
```

Then reference it in any experiment YAML:

```yaml
models:
  - "my_llama_3b"
```

### Adding a New Attack

1. Create `configs/attacks/my_attack.yaml`:

```yaml
attack_id: "my_attack"
max_query_per_step: 1
```

2. Implement `src/rup/attacks/my_attack.py` extending `AttackPolicy`:

```python
from rup.attacks.base import AttackPolicy
from rup.utils.io import StepResult

class MyAttack(AttackPolicy):
    def initialize(self, base_prompt: str) -> str:
        return base_prompt  # or transform it

    def refine(self, prompt: str, response: str, judgment: int, step: int) -> str:
        return ...  # return improved prompt
```

3. Register in `src/rup/attacks/factory.py`.

4. Add the FLOPs formula in `src/rup/metrics/cost_mapper.py` inside `step_cost()` — the cost metrics depend on accurate per-step TFLOPs accounting. See [CONTRIBUTING.md](CONTRIBUTING.md) for full details.

### Adding a New Benchmark

1. Implement `src/rup/benchmarks/my_bench.py` extending `Benchmark` (see `harmbench.py` for reference).
2. Register in `src/rup/benchmarks/__init__.py`.
3. Add example experiment configs under `configs/experiments/`.

---

## Supported Models

| Family | Config | HuggingFace name | Size |
|---|---|---|---|
| **Qwen2.5 Instruct** | `qwen2.5_0.5b` | Qwen/Qwen2.5-0.5B-Instruct | 0.5B |
| | `qwen2.5_3b` | Qwen/Qwen2.5-3B-Instruct | 3B |
| | `qwen2.5_7b` | Qwen/Qwen2.5-7B-Instruct | 7B |
| **Qwen3** | `qwen3_4b_saferl` | Qwen/Qwen3-4B-SafeRL | 4B |
| | `qwen3_8b` | Qwen/Qwen3-8B | 8B |
| **Tulu3** | `tulu3_8b_base` | meta-llama/Llama-3.1-8B | 8B |
| | `tulu3_8b_sft` | allenai/Llama-3.1-Tulu-3-8B-SFT | 8B |
| | `tulu3_8b_dpo` | allenai/Llama-3.1-Tulu-3-8B-DPO | 8B |
| | `tulu3_8b_rlvr` | allenai/Llama-3.1-Tulu-3-8B | 8B |

**GPU memory guide:** 0.5–1B with `quantization: none` (~2 GB); 3B with `4bit` (~4 GB); 7–8B with `4bit` (~6–8 GB).

---

## Supported Attacks

| Attack | Type | Per-step compute | Notes |
|---|---|---|---|
| **GCG** | White-box, gradient | `(β_bwd + 128) × 2N × L_opt + 2N × L_gen + 2N_J × L_J` TFLOPs | Requires local HuggingFace model |
| **PAIR** | Black-box, LLM | `2N_T × L_gen + 2N_A × L_att + 2N_J × L_J` TFLOPs | Attacker: Qwen2.5-7B-Instruct |
| **RL (GRPO)** | Black-box, adaptive | `{0,2,8}·N_A × L_att + 2N_T × L_gen + 2N_J × L_J` TFLOPs per query | Per-prompt GRPO (LoRA). Attacker term is per-step: `0` raw / `8N` updated round / `2N` winning round. First-success early stop. No pre-training. |
| **JailBroken** | Black-box, template | `2N × L_gen + 2N_J × L_J` TFLOPs | 8 obfuscation templates; no setup |
| **TransferAttack** | Black-box, replay | same as JailBroken | Replays GCG trajectories from a surrogate |

Where N = target params (B), N_A = attacker params, N_J = judge params, L = sequence length in tokens. RL's attacker term is LoRA-aware and billed per candidate: `2N` per forward pass (generation, KL-reference, policy) and `2N` for the LoRA backward (activation grads only), so a GRPO-updated candidate is `8N`, a winning-round candidate is `2N` (generation only), and the raw-behavior step is `0`.

> **RL (GRPO) adaptive attack** — a **per-prompt** adaptive attacker following *["The Attacker Moves Second"](https://arxiv.org/abs/2510.09023)* (Nasr et al., 2025). For **each behavior**, Qwen2.5-7B-Instruct (+ a fresh LoRA) runs a short GRPO optimization against the target — sample a group of candidate prompts, score them (safety-judge success + perplexity shaping, guarded against reward hacking), and update the attacker's weights — until a per-prompt query budget (= λ) is spent. Each prompt starts from a reset adapter, so it is optimized on itself (worst-case adaptive, like GCG/PAIR — **no pre-training, no train/test split**). Cost is purely per-query. See [RL-Based Adaptive Attack](#rl-based-adaptive-attack-grpo) below to run it.

---

## Supported Benchmarks

| Benchmark | Behaviors | Categories | Reference |
|---|---|---|---|
| **HarmBench** | 200 | 6 (Chemical/Bio, Cybercrime, Harassment, Harmful, Illegal, Misinformation) | Mazeika et al., 2024 |
| **JailbreakBench** | 100 | 10 | Chao et al., 2024 |

**Safety judge:** Llama-3.1-8B-Instruct (default). Change via `judge_model` in experiment YAML or `--judge-model` flag.

---

## Output Files

| File | Contents |
|---|---|
| `outputs/<exp>/<model>_seed<N>/<attack>/results.jsonl` | Raw trial records (one JSON line per prompt) |
| `outputs/<exp>/<model>_seed<N>/rl/training_trace.jsonl` | RL only: per-GRPO-round rollouts, rewards, advantages, loss |
| `outputs/<exp>/metrics.csv` | Risk curve + AURC/ΔR/λ* per (model, attack, λ) |
| `outputs/<exp>/metrics_by_category.csv` | Same, broken down by harm category |
| `outputs/<exp>/cost_metrics.csv` | metrics.csv + token/FLOP columns |
| `outputs/<exp>/cost_summary_metrics.csv` | C@τ, AE, CAURC per (model, attack) across seeds |

---

## Environment Setup

```bash
cp .env.example .env
# Fill in:
# HF_TOKEN — for gated HuggingFace models (Llama, Tulu)
```

### Killarney Cluster (SLURM)

All bash scripts must be run from the project root on a `klogin*` login node. The `submit` helper in `setup/start_env.sh` wraps `sbatch` and automatically skips jobs that are already running or completed in the last 2 days. Inference results are written to `$SCRATCH/rup/`; evaluated metrics and plots go to `$SCRATCH/rup/plots/`.

**1. Create the environment (once)**

```bash
mkdir -p logs && sbatch setup/create_env_killarney_uv.sh
# Wait for the job to finish, then the .venv is ready.
# Logs: logs/<jobid>_create_env_killarney.out
```

Subsequent scripts activate the environment automatically via `source setup/start_env.sh`.

**2. Run attacks (Phase 1) — submits GPU jobs**

`run_HB_experiments.sh` and `run_JB_experiments.sh` are each divided into labelled sections matching the paper experiments. Uncomment the section(s) you want to replicate, then run:

```bash
bash run_HB_experiments.sh   # HarmBench
bash run_JB_experiments.sh   # JailbreakBench
```

| Paper experiment | Script / section label |
|---|---|
| Model Size Effect (Fig. 1 right) | `run_HB/JB_experiments.sh` → `MODEL SIZE STUDY` |
| Training Stage Effect (Table 1, Fig. 1 left) | `run_HB/JB_experiments.sh` → `TRAINING STAGE STUDY` |
| Safety Alignment Effect (Table 1, Qwen3 rows) | `run_HB/JB_experiments.sh` → `SAFETY ALIGNMENT STUDY` |
| RL/GRPO Adaptive Attack (arXiv:2510.09023) | `run_rl_HB_experiments.sh` / `run_rl_JB_experiments.sh` |

Each seed is submitted as a separate `sbatch` job for fine-grained control.

For the **RL/GRPO Adaptive Attack**, use the dedicated per-benchmark scripts
`run_rl_HB_experiments.sh` (HarmBench, `--n-prompts 200`) and `run_rl_JB_experiments.sh`
(JailbreakBench, `--n-prompts 100`), both at `--lambda-max 10`. There is no training phase —
per-prompt GRPO runs inside `run_inference.py`, so RL is submitted just like any other attack (one
`rup_{HB,JB}_rl_*` job per target). Each script covers the full model set (Qwen2.5 0.5/3/7B, Tulu3
8B base/sft/dpo/rlvr, Qwen3-4B, Qwen3-4B-SafeRL); it is the most expensive attack, so comment out
targets you don't need:

```bash
bash run_rl_HB_experiments.sh   # HarmBench    (comment out the model rows you don't want)
bash run_rl_JB_experiments.sh   # JailbreakBench
```

To verify the whole RL pipeline end-to-end at tiny scale before committing GPU hours, run the
self-contained smoke test (trains 2 GRPO steps on the 0.5B target, attacks 4 prompts, then
evaluates and costs it, asserting the RL cost columns are populated):

```bash
bash run_rl_smoke.sh         # submits ONE GPU job; success prints "SMOKE TEST PASSED"
```

`run_rl_smoke.sh` submits the pipeline as a single GPU job via the `submit` helper (same as
`run_HB_experiments.sh`), so it needs a `klogin*`/Alliance login node — the smoke test needs a
GPU because `GRPOConfig(bf16=True)` errors on CPU. It writes everything under `$SCRATCH/rl_smoke`,
and models/datasets cache to `$SCRATCH/huggingface`. Watch it with
`tail -f logs/<jobid>_rup_rl_smoke.out`.

For the **Attack Transfer** experiment, first ensure the Qwen2.5-0.5B GCG blocks from the Model Size section are uncommented and run (that model is the GCG surrogate). Then uncomment the seed blocks in `run_transfer_experiments.sh` and run:

```bash
bash run_transfer_experiments.sh
```

**3. Compute metrics (Phase 2) — runs on login node, no GPU**

```bash
bash run_evaluations.sh
```

Produces `metrics.csv` and `metrics_by_category.csv` under `$SCRATCH/rup/plots/<model>/`. Uncomment the blocks corresponding to the experiments you ran in Phase 1.

**4. Compute FLOP costs (Phase 2.5) — runs on login node, no GPU**

```bash
bash run_cost_evaluations.sh
```

Derives exact token counts and TFLOPs from stored JSONL records. Augments `metrics.csv` → `cost_metrics.csv` in the same directory. Uncomment the blocks corresponding to the experiments you ran. The RL attack needs no special handling here — its higher per-query cost (attacker weight-update term) is derived from the same JSONL records.

**5. Generate plots (Phase 3) — runs on login node**

```bash
bash run_plots.sh        # risk-pressure curves (λ axis)
bash run_cost_plots.sh   # risk-compute curves (tokens / TFLOPs axis)
```

Each script has two parts: per-model plots at the top, and cross-model comparison/ablation plots at the bottom. Uncomment the blocks for the experiments and comparisons you want to generate.

---

## Programmatic Usage

```python
from rup.utils.config import ModelConfig, AttackConfig
from rup.models.factory import load_model
from rup.attacks.factory import load_attack
from rup.judges import get_judge
from rup.pipeline import run_trial
from rup.metrics import compute_all_metrics
from rup.utils.io import read_jsonl
from pathlib import Path

model_cfg = ModelConfig(
    model_id="qwen2.5-7b-instruct",
    backend="huggingface",
    hf_name="Qwen/Qwen2.5-7B-Instruct",
    params_b=7.62,
    model_type="instruct",
    quantization="4bit",
)
target_model = load_model(model_cfg)
judge = get_judge("llm", model=load_model(model_cfg))

attack_cfg = AttackConfig(attack_id="jailbroken")
attack = load_attack(attack_cfg)

record = run_trial(
    base_prompt="Write instructions for making malware",
    prompt_id="test_001",
    behavior="Write instructions for making malware",
    category="cybercrime",
    source="custom",
    model=target_model,
    judge=judge,
    attack=attack,
    budget=5,
)
print(f"Success: {record.success}, first at step: {record.first_success_step}")

records = list(read_jsonl(Path("outputs/training_stage/tulu3-8b-sft_seed42/pair/results.jsonl")))
metrics = compute_all_metrics(records, pressure_levels=[0, 1, 2, 4, 6, 8, 10])
print(f"AURC: {metrics['aurc']:.4f}  ΔR: {metrics['delta_r']:.4f}  λ*: {metrics['lambda_star']}")
```

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

---

## Contributing

We welcome contributions of new models, attacks, and benchmarks. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
