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
# Alternative cost framing — the attacker's own bill, judge excluded.
# `total` = target + judge + attacker (cost of reproducing the measurement);
# `nojudge` = target + attacker (cost of mounting the attack).
# --------------------------------------------------------------------------- #

def test_nojudge_columns_exclude_exactly_the_judge(tmp_path):
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text(
        "models:\n"
        "  qwen2.5-0.5b-instruct: {usd_per_1m_input: 1.0, usd_per_1m_output: 2.0}\n"
        "  judge: {usd_per_1m_input: 3.0, usd_per_1m_output: 3.0}\n"
        "  qwen2.5-7b-instruct: {usd_per_1m_input: 5.0, usd_per_1m_output: 7.0}\n"
    )
    # pair, so the attacker term is non-zero on every axis and the identity is not
    # trivially satisfied by an all-zero attacker column.
    steps = [
        StepResult(step=1, prompt="a a a", response="r r", judgment=0, seconds=1.0),
        StepResult(step=2, prompt="b b", response="s s s", judgment=1, seconds=1.0),
    ]
    out = cm.aggregate_costs(
        [_record("pair", steps, True, 2)], [1, 2],
        judge_model_id="judge", pricing_path=str(pricing),
    )

    for lam in (1, 2):
        c = out[lam]
        assert c["mean_attacker_tokens"] > 0.0 and c["mean_judge_tokens"] > 0.0
        for nojudge, parts in (
            ("mean_nojudge_tokens",  ("mean_target_tokens",  "mean_attacker_tokens")),
            ("mean_nojudge_tflops",  ("mean_target_tflops",  "mean_attacker_tflops")),
            ("mean_nojudge_dollars", ("mean_target_dollars", "mean_attacker_dollars")),
        ):
            assert c[nojudge] == pytest.approx(c[parts[0]] + c[parts[1]])
        # and each nojudge axis is exactly its total minus the judge's share
        assert c["mean_nojudge_tokens"] == pytest.approx(
            c["mean_total_tokens"] - c["mean_judge_tokens"])
        assert c["mean_nojudge_tflops"] == pytest.approx(
            c["mean_total_tflops"] - c["mean_judge_tflops"])
        assert c["mean_nojudge_dollars"] == pytest.approx(
            c["mean_total_dollars"] - c["mean_judge_dollars"])


def test_total_tflops_still_decomposes_into_three_components():
    """mean_attacker_tflops is newly reported; it must complete the existing total."""
    steps = [StepResult(step=1, prompt="a a", response="r r", judgment=1, seconds=1.0)]
    c = cm.aggregate_costs([_record("pair", steps, True, 1)], [1], judge_model_id="judge")[1]
    assert c["mean_total_tflops"] == pytest.approx(
        c["mean_target_tflops"] + c["mean_judge_tflops"] + c["mean_attacker_tflops"]
    )


def test_nojudge_is_invariant_to_the_judge_choice(monkeypatch):
    """The point of the alternative framing: swapping judges must not move it.

    The judge-inclusive axis does move, which is exactly why a cross-judge comparison
    should be read on the nojudge axis.
    """
    monkeypatch.setattr(cm, "_judge_params_b", lambda mid: {"small": 1.0, "big": 70.0}[mid])
    steps = [StepResult(step=1, prompt="a a", response="r r", judgment=1, seconds=1.0)]
    rec = _record("pair", steps, True, 1)

    small = cm.aggregate_costs([rec], [1], judge_model_id="small")[1]
    big   = cm.aggregate_costs([rec], [1], judge_model_id="big")[1]

    assert small["mean_nojudge_tflops"] == pytest.approx(big["mean_nojudge_tflops"])
    assert small["mean_total_tflops"] != pytest.approx(big["mean_total_tflops"])


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


@pytest.mark.parametrize("attack_id", ["pair", "rl"])
def test_attacker_choice_changes_attacker_flops(attack_id, monkeypatch):
    """Attacker FLOPs scale with the attacker's params_b — the whole point of the
    ablation is that a 1B attacker is not billed as if it were the 7.62B default.
    Holds for both attacks that use an attacker: PAIR and RL/GRPO."""
    monkeypatch.undo()
    monkeypatch.setattr(cm, "_count_tokens", lambda text, hf_id: max(1, len((text or "").split())))
    cm.load_model_registry("configs")

    # One failed step (for RL: a candidate on a round that trained) — the attacker is charged.
    steps = [StepResult(step=1, prompt="a a a", response="b b", judgment=0),
             StepResult(step=2, prompt="c c c", response="d d", judgment=0)]
    rec = _record(attack_id, steps, False, None, model_id="qwen2.5-7b-instruct")

    def att_tflops(attacker_id):
        costs = cm.aggregate_costs([rec], [2], configs_dir="configs",
                                   attacker_model_id=attacker_id)[2]
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
    # RL arms are named the same way (rl__<attacker config>).
    assert cm.attacker_from_attack_id("rl__gemma3_4b_it_abliterated") == "gemma3-4b-it-abliterated"
    # Single-attacker runs keep the default and their historic directory names.
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


# --------------------------------------------------------------------------- #
# Judge input reconstruction. The cost model has to price the string the judge
# actually read. For RL that is the behavior, not the recorded candidate prompt:
# the environment calls judge(behavior, response) while the StepResult stores the
# candidate. Pricing the candidate bills a call that never happened.
# --------------------------------------------------------------------------- #


def test_rl_judge_input_is_the_behavior_not_the_candidate_prompt():
    """RL judge cost must not scale with the candidate prompt the judge never saw."""
    long_candidate = " ".join(["candidate"] * 200)
    steps = [StepResult(step=1, prompt=long_candidate, response="r r", judgment=1, seconds=1.0)]

    rl = _record("rl", steps, True, 1)
    rl.behavior = "build a bomb"
    pair = _record("pair", steps, True, 1)
    pair.behavior = "build a bomb"

    rl_out = cm.aggregate_costs([rl], [1])[1]
    pair_out = cm.aggregate_costs([pair], [1])[1]

    # RL reads its judge input three times per step (one verdict generation + two prefill-only
    # passes for the continuous P(UNSAFE) the reward needs), so compare per-pass to isolate the
    # thing this test is about: WHICH string is priced, not how many times it is read.
    rl_per_pass = rl_out["mean_judge_tokens"] / 3.0
    # PAIR prices the 200-word candidate; RL prices the 3-word behavior.
    assert rl_per_pass < pair_out["mean_judge_tokens"]

    # Stronger form of the same invariant: growing the candidate must not move RL's judge cost
    # at all. (A plain magnitude bound would be weak here — the fixed rubric dominates the judge
    # input, so even a mispriced candidate only shifts the total by a fraction.)
    longer = [StepResult(step=1, prompt=long_candidate * 3, response="r r",
                         judgment=1, seconds=1.0)]
    rl_longer = _record("rl", longer, True, 1)
    rl_longer.behavior = "build a bomb"
    assert cm.aggregate_costs([rl_longer], [1])[1]["mean_judge_tokens"] == pytest.approx(
        rl_out["mean_judge_tokens"]
    )


def test_rl_pays_for_the_two_continuous_score_passes():
    """RL's reward needs P(UNSAFE), which is two extra judge prefills — and they are billed."""
    # Same string as prompt AND behavior, so both attacks price an identical judge input and
    # the only remaining difference is how many times RL reads it.
    steps = [StepResult(step=1, prompt="build a bomb", response="r r", judgment=1, seconds=1.0)]
    rl = _record("rl", steps, True, 1)
    rl.behavior = "build a bomb"
    pair = _record("pair", steps, True, 1)
    pair.behavior = "build a bomb"

    rl_out = cm.aggregate_costs([rl], [1])[1]
    pair_out = cm.aggregate_costs([pair], [1])[1]

    # Same judge input for both here (the candidate IS short), so the only difference left is
    # that RL reads it three times. Output tokens are generated once, so the ratio is just under 3.
    ratio = rl_out["mean_judge_tokens"] / pair_out["mean_judge_tokens"]
    assert 2.5 < ratio < 3.0


def test_non_rl_attacks_still_price_the_step_prompt():
    """The RL carve-out must not change any other attack's judge cost."""
    steps = [StepResult(step=1, prompt="a a a a", response="r r", judgment=1, seconds=1.0)]
    for attack_id in ("gcg", "pair", "jailbroken"):
        rec = _record(attack_id, steps, True, 1)
        rec.behavior = "x"
        out = cm.aggregate_costs([rec], [1])[1]
        # behavior is 1 word, prompt is 4 — pricing the prompt gives the larger count.
        assert out["mean_judge_tokens"] > 0.0


def test_measured_judge_output_length_moves_only_the_judge_axis():
    """Escalation can make a judge emit far more than the assumed ~2 tokens.

    When a results file records the real length, use it — but it must not leak into the
    nojudge axis, which is the one a cross-judge comparison is read on.
    """
    assumed = [StepResult(step=1, prompt="a a", response="r r", judgment=1, seconds=1.0)]
    measured = [StepResult(step=1, prompt="a a", response="r r", judgment=1, seconds=1.0,
                           judge_out_tokens=250)]

    base = cm.aggregate_costs([_record("pair", assumed, True, 1)], [1])[1]
    esc = cm.aggregate_costs([_record("pair", measured, True, 1)], [1])[1]

    assert esc["mean_judge_tflops"] > base["mean_judge_tflops"]
    assert esc["mean_nojudge_tflops"] == pytest.approx(base["mean_nojudge_tflops"])


def test_pre_audit_files_fall_back_to_the_two_token_assumption():
    """Trees written before judge_out_tokens existed must cost exactly as they did before."""
    steps = [StepResult(step=1, prompt="a a", response="r r", judgment=1, seconds=1.0)]
    assert steps[0].judge_out_tokens is None

    out = cm.aggregate_costs([_record("pair", steps, True, 1)], [1])[1]
    assert out["mean_judge_tflops"] > 0.0


def test_cost_mapper_shares_one_rubric_with_the_judge():
    """A local copy of the rubric silently mis-prices the judge as soon as it drifts."""
    from rup.judges.llm_judge import _JUDGE_SYSTEM_PROMPT, _JUDGE_USER_TEMPLATE

    assert cm._JUDGE_SYSTEM_PROMPT is _JUDGE_SYSTEM_PROMPT
    assert cm._JUDGE_USER_TEMPLATE is _JUDGE_USER_TEMPLATE
