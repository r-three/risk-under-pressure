# Contributing to Risk Under Pressure

We welcome contributions of new models, attacks, and benchmarks. The framework is designed so that the most common contributions require only YAML files and minimal Python.

## Adding a New Model

**YAML-only (most common case):** Create `configs/models/<model_id>.yaml`:

```yaml
model_id: "my-model-7b-instruct"
backend: "huggingface"            # huggingface | openai | anthropic | google
hf_name: "org/My-Model-7B-Instruct"
params_b: 7.0                     # billions of parameters (required for FLOP calculation)
model_type: "instruct"            # instruct | base
quantization: "4bit"              # none | 4bit | 8bit
device: "cuda"
generation:
  max_new_tokens: 512
  temperature: 0.7
  do_sample: true
  top_p: 0.9
```

Then reference it by filename (without `.yaml`) in any experiment config:

```yaml
models:
  - "my-model-7b-instruct"
```

**Custom backend:** If the model uses a backend not yet supported, implement `src/rup/models/my_backend.py` extending `BaseModel` (see `hf_model.py` for reference) and register it in `src/rup/models/factory.py`.

Open a PR with the YAML file and any backend code.

---

## Adding a New Attack

1. **Create the attack config** `configs/attacks/my_attack.yaml`:

```yaml
attack_id: "my_attack"
max_query_per_step: 1
# optional: attacker_model, templates, extra fields
```

2. **Implement the attack** in `src/rup/attacks/my_attack.py`:

```python
from rup.attacks.base import AttackPolicy

class MyAttack(AttackPolicy):
    def initialize(self, base_prompt: str) -> str:
        """Return the first prompt to try (step 0)."""
        return base_prompt

    def refine(self, prompt: str, response: str, judgment: int, step: int) -> str:
        """Return an improved prompt given the previous prompt, response, and judgment.
        judgment=1 means unsafe (attack succeeded); judgment=0 means safe (refine further).
        """
        return improved_prompt
```

3. **Register** in `src/rup/attacks/factory.py`:

```python
from .my_attack import MyAttack

def load_attack(cfg: AttackConfig) -> AttackPolicy:
    ...
    elif cfg.attack_id == "my_attack":
        return MyAttack(cfg)
```

4. **Add the FLOPs formula** in `src/rup/metrics/cost_mapper.py`, inside `step_cost()`. The cost metrics (C@τ, AE, CAURC) depend on accurate per-step TFLOPs accounting. Add a branch for your attack alongside the existing `gcg` and `pair` cases. The base formula for a single forward pass is `2 × N_B × L / 1000` TFLOPs, where `N_B` is model size in billions and `L` is sequence length in tokens. Judge cost (`judge_tfl`) is already computed for all attacks — only add extra terms your attack introduces beyond target + judge. If your attack reuses a helper model (e.g. an attacker LLM like PAIR's), add its forward pass cost analogously to the `pair` branch. Also update the docstring table in `README.md` under **Supported Attacks**.

---

## Adding a New Benchmark

1. **Implement** `src/rup/benchmarks/my_bench.py` extending `Benchmark`:

```python
from rup.benchmarks.base import Benchmark, BenchmarkPrompt

class MyBench(Benchmark):
    def load(self, n=None, seed=42, categories=None) -> list[BenchmarkPrompt]:
        ...  # return list of BenchmarkPrompt(id, text, behavior, category, source)
```

2. **Register** in `src/rup/benchmarks/__init__.py`.

3. **Add example experiment configs** under `configs/experiments/` with `benchmark: "my_bench"`.

---

## Code Style

- Default to writing no comments unless the *why* is non-obvious.
- Tests go in `tests/` and run with `pytest`.
- Keep imports relative within `src/rup/`.
- Format with `ruff format` before opening a PR.

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

## Opening a PR

1. Fork the repository and create a feature branch.
2. Add or modify the YAML config(s).
3. If you added Python code, include a test in `tests/`.
4. Open a pull request with a short description of what model/attack/benchmark you're adding and why it's useful.
