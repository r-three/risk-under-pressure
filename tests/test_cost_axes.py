"""Unit tests for the seconds (measured wall-clock) and dollars (per-token) cost axes.

No models/tokenizers are loaded: _count_tokens and the model registry are stubbed, so these
run on CPU without network access.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rup.metrics import cost_mapper as cm
from rup.utils.io import StepResult, TrialRecord


def _record(attack_id, steps, success, t_star, model_id="qwen2.5-0.5b-instruct"):
    return TrialRecord(
        prompt_id="p", base_prompt="b", behavior="beh", category="c", source="s",
        model_id=model_id, attack_id=attack_id, budget=len(steps), steps=steps,
        success=success, first_success_step=t_star, final_prompt="f",
    )


@pytest.fixture(autouse=True)
def _stub(monkeypatch):
    # 1 token per whitespace word (deterministic; no tokenizer download)
    monkeypatch.setattr(cm, "_count_tokens", lambda text, hf_id: max(1, len((text or "").split())))
    monkeypatch.setattr(cm, "_hf_id", lambda mid: mid)
    monkeypatch.setattr(cm, "_params_b", lambda mid: 1.0)
    monkeypatch.setattr(cm, "_judge_params_b", lambda mid: 1.0)
    monkeypatch.setattr(cm, "load_model_registry", lambda *a, **k: None)
    # reset pricing globals between tests
    cm.MODEL_PRICE_IN.clear()
    cm.MODEL_PRICE_OUT.clear()
    cm._PRICING_LOADED = False
    cm._PRICING_WARNED.clear()
    yield


def test_mean_total_seconds_is_cumulative():
    steps = [
        StepResult(step=1, prompt="a a", response="r r", judgment=0, seconds=2.0),
        StepResult(step=2, prompt="b b", response="r r", judgment=1, seconds=3.0),
    ]
    out = cm.aggregate_costs([_record("pair", steps, True, 2)], [1, 2], judge_model_id="judge")
    assert out[1]["mean_total_seconds"] == pytest.approx(2.0)   # through step 1
    assert out[2]["mean_total_seconds"] == pytest.approx(5.0)   # + step 2


def test_seconds_nan_when_a_consumed_step_is_untimed():
    steps = [
        StepResult(step=1, prompt="a", response="r", judgment=0, seconds=1.0),
        StepResult(step=2, prompt="b", response="r", judgment=1, seconds=None),
    ]
    out = cm.aggregate_costs([_record("pair", steps, True, 2)], [1, 2], judge_model_id="judge")
    assert out[1]["mean_total_seconds"] == pytest.approx(1.0)   # step 1 timed
    assert math.isnan(out[2]["mean_total_seconds"])            # step 2 untimed → NaN


def test_mean_total_dollars_from_pricing(tmp_path):
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text(
        "models:\n"
        "  qwen2.5-0.5b-instruct: {usd_per_1m_input: 1.0, usd_per_1m_output: 2.0}\n"
        "  judge: {usd_per_1m_input: 3.0, usd_per_1m_output: 3.0}\n"
        "  qwen2.5-7b-instruct: {usd_per_1m_input: 0.0, usd_per_1m_output: 0.0}\n"
    )
    # jailbroken: no attacker; target_in = 3 words, target_out = 2 words
    steps = [StepResult(step=1, prompt="w w w", response="x x", judgment=1, seconds=1.0)]
    out = cm.aggregate_costs(
        [_record("jailbroken", steps, True, 1)], [1],
        judge_model_id="judge", pricing_path=str(pricing),
    )[1]
    # each component priced by its OWN model's rate
    assert out["mean_target_dollars"] == pytest.approx((3 * 1.0 + 2 * 2.0) / 1e6)
    assert out["mean_attacker_dollars"] == pytest.approx(0.0)          # no attacker in jailbroken
    assert out["mean_judge_dollars"] > 0.0                             # judge input priced at $3/1M
    # total == sum of the three components
    assert out["mean_total_dollars"] == pytest.approx(
        out["mean_target_dollars"] + out["mean_judge_dollars"] + out["mean_attacker_dollars"]
    )


def test_dollars_nan_without_pricing_config():
    steps = [StepResult(step=1, prompt="w", response="x", judgment=1, seconds=1.0)]
    out = cm.aggregate_costs([_record("jailbroken", steps, True, 1)], [1], judge_model_id="judge")
    assert math.isnan(out[1]["mean_total_dollars"])


# --------------------------------------------------------------------------- #
# Judge ablation — each judge must be sized and priced by its OWN config, not by
# the llama3.1-8b default that judge_model_id falls back to.
# --------------------------------------------------------------------------- #

# (judge model_id, params_b from configs/models/*_judge.yaml)
# Gemma 3 is a multimodal checkpoint; params_b is the text-only LM, because a
# text-only judging call never runs the vision tower.
_JUDGES = [
    ("llama3.1-8b-instruct", 8.03),
    ("olmo3-7b-instruct", 7.30),
    ("gemma3-4b-it", 3.88),
]


@pytest.mark.parametrize("judge_id,expected_params_b", _JUDGES)
def test_judge_registry_has_size_and_price(judge_id, expected_params_b, monkeypatch):
    """Every ablation judge resolves to its real size, tokenizer and hosted rate."""
    monkeypatch.undo()  # drop the _stub fixture's fakes; we want the real registry here
    cm.MODEL_PRICE_IN.clear()
    cm.MODEL_PRICE_OUT.clear()
    cm._PRICING_LOADED = False

    cm.load_model_registry("configs")
    cm.load_pricing("configs/pricing.yaml")

    assert cm._judge_params_b(judge_id) == pytest.approx(expected_params_b)
    assert cm._hf_id(judge_id)                      # tokenizer mapping exists
    assert judge_id in cm.MODEL_PRICE_IN, f"{judge_id} missing from configs/pricing.yaml"
    assert judge_id in cm.MODEL_PRICE_OUT


def test_judge_choice_changes_judge_flops(monkeypatch):
    """Judge FLOPs scale with the judge's params_b — a 3.88B judge must not be
    billed as if it were the 8.03B default."""
    monkeypatch.undo()
    monkeypatch.setattr(cm, "_count_tokens", lambda text, hf_id: max(1, len((text or "").split())))
    cm.load_model_registry("configs")

    steps = [StepResult(step=1, prompt="a a a", response="b b", judgment=1)]
    rec = _record("jailbroken", steps, True, 1, model_id="qwen2.5-7b-instruct")

    flops = {
        j: cm.aggregate_costs([rec], [1], judge_model_id=j, configs_dir="configs")[1]["mean_judge_tflops"]
        for j, _ in _JUDGES
    }
    # FLOPs = 2 x params_b x tokens, so the ratio between two judges is their size ratio.
    assert flops["gemma3-4b-it"] < flops["olmo3-7b-instruct"] < flops["llama3.1-8b-instruct"]
    assert flops["gemma3-4b-it"] / flops["llama3.1-8b-instruct"] == pytest.approx(3.88 / 8.03, rel=1e-6)


# --------------------------------------------------------------------------- #
# Attacker ablation — PAIR's attacker is swappable per run
# (configs/experiments/paper/attacker_size.yaml), so it must be sized and priced
# by its OWN config too, not by the qwen2.5-7b default it falls back to.
# --------------------------------------------------------------------------- #

# (attacker model_id, params_b from configs/models/*.yaml)
_ATTACKERS = [
    ("qwen2.5-7b-instruct", 7.62),        # incumbent
    ("gemma3-4b-it-abliterated", 3.88),
    ("gemma3-1b-it-abliterated", 1.00),
]


@pytest.mark.parametrize("attacker_id,expected_params_b", _ATTACKERS)
def test_attacker_registry_has_size_and_price(attacker_id, expected_params_b, monkeypatch):
    """Every ablation attacker resolves to its real size, tokenizer and hosted rate."""
    monkeypatch.undo()
    cm.MODEL_PRICE_IN.clear()
    cm.MODEL_PRICE_OUT.clear()
    cm._PRICING_LOADED = False

    cm.load_model_registry("configs")
    cm.load_pricing("configs/pricing.yaml")

    assert cm._attacker_params_b(attacker_id) == pytest.approx(expected_params_b)
    assert cm._attacker_hf_id(attacker_id)
    assert attacker_id in cm.MODEL_PRICE_IN, f"{attacker_id} missing from configs/pricing.yaml"
    assert attacker_id in cm.MODEL_PRICE_OUT


def test_attacker_choice_changes_attacker_flops(monkeypatch):
    """Attacker FLOPs scale with the attacker's params_b — the whole point of the
    ablation is that a 1B attacker is not billed as if it were the 7.62B default."""
    monkeypatch.undo()
    monkeypatch.setattr(cm, "_count_tokens", lambda text, hf_id: max(1, len((text or "").split())))
    cm.load_model_registry("configs")

    # PAIR, one failed step: the attacker runs and is charged.
    steps = [StepResult(step=1, prompt="a a a", response="b b", judgment=0)]
    rec = _record("pair", steps, False, None, model_id="qwen2.5-7b-instruct")

    def att_tflops(attacker_id):
        costs = cm.aggregate_costs([rec], [1], configs_dir="configs",
                                   attacker_model_id=attacker_id)[1]
        # attacker TFLOPs are not a column of their own; back them out of the total
        return costs["mean_total_tflops"] - costs["mean_target_tflops"] - costs["mean_judge_tflops"]

    flops = {a: att_tflops(a) for a, _ in _ATTACKERS}
    assert flops["gemma3-1b-it-abliterated"] < flops["gemma3-4b-it-abliterated"] \
        < flops["qwen2.5-7b-instruct"]
    assert flops["gemma3-1b-it-abliterated"] / flops["gemma3-4b-it-abliterated"] \
        == pytest.approx(1.00 / 3.88, rel=1e-6)


def test_attacker_resolved_from_results_dir_name(monkeypatch):
    """`pair__<config>` result dirs name their attacker; plain dirs fall back."""
    monkeypatch.undo()
    cm.load_model_registry("configs")

    # Written by run_inference.py when the experiment sets attacker_models: the folder
    # carries the CONFIG name, and this maps it back to the model_id costs are keyed by.
    assert cm.attacker_from_attack_id("pair__gemma3_1b_it_abliterated") == "gemma3-1b-it-abliterated"
    assert cm.attacker_from_attack_id("pair__gemma3_4b_it_abliterated") == "gemma3-4b-it-abliterated"
    # Single-attacker runs (and rl) keep the default.
    assert cm.attacker_from_attack_id("pair") == "qwen2.5-7b-instruct"
    assert cm.attacker_from_attack_id("rl") == "qwen2.5-7b-instruct"
    assert cm.attacker_from_attack_id("pair", default="other") == "other"
    # A config that no longer exists warns rather than crashing an old cost recompute.
    with pytest.warns(UserWarning):
        assert cm.attacker_from_attack_id("pair__deleted_model") == "qwen2.5-7b-instruct"


def test_unknown_attacker_falls_back_without_crashing(monkeypatch):
    """Costing an old results tree must not die on an attacker the registry lost."""
    monkeypatch.undo()
    monkeypatch.setattr(cm, "_count_tokens", lambda text, hf_id: max(1, len((text or "").split())))
    cm.load_model_registry("configs")

    steps = [StepResult(step=1, prompt="a a", response="b", judgment=0)]
    rec = _record("pair", steps, False, None, model_id="qwen2.5-7b-instruct")
    with pytest.warns(UserWarning):
        out = cm.aggregate_costs([rec], [1], configs_dir="configs",
                                 attacker_model_id="model-that-vanished")[1]
    assert out["mean_total_tflops"] > 0.0
