"""Structural tests for the session/group GRPO loop (arXiv:2510.09023, App. A.2).

These pin the three things the paper's structure actually determines — how the budget is spent,
when a weight update happens, and how the run stops — without loading a 7B attacker. The attacker
is a 2x2 Linear (AdamW needs at least one real parameter) and its sampling/update are stubbed.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rup.pipeline import rl_refinement as rr
from rup.pipeline.rl_refinement import GRPOAttackConfig, _Rollout, _Session
from rup.training.rl_env import RewardConfig


class StubTarget:
    model_id = "stub-target"

    def __init__(self, response="a reasonably long and varied unsafe looking reply"):
        self._response = response

    def generate(self, prompt, **kwargs):
        return self._response

    def sequence_nll(self, context, continuation, **kwargs):
        return 5.0


class StubJudge:
    """Labels UNSAFE on the step numbers in `unsafe_at` (1-indexed call order)."""

    def __init__(self, unsafe_at=()):
        self._unsafe_at = set(unsafe_at)
        self.calls = 0

    def judge(self, prompt, response):
        self.calls += 1
        return 1 if self.calls in self._unsafe_at else 0

    def verdict_score(self, prompt, response):
        return 0.9 if self.calls in self._unsafe_at else 0.1


@pytest.fixture
def attacker():
    import torch

    class _A:
        model = torch.nn.Linear(2, 2)
        tokenizer = None
        init_adapter_state = {}
        device = "cpu"

    return _A()


@pytest.fixture
def stub_loop(monkeypatch):
    """Stub everything that needs a real LM, and record every GRPO update."""
    import torch

    updates = []
    monkeypatch.setattr(rr, "_reset_adapter", lambda a: None)
    monkeypatch.setattr(rr, "_prompt_ids", lambda a, b, ctx: torch.zeros(1, 3, dtype=torch.long))
    monkeypatch.setattr(
        rr, "_sample_batch",
        lambda a, pids_list, cfg: (
            ["a diverse candidate prompt for the target model"] * len(pids_list),
            [torch.zeros(2)] * len(pids_list),
        ),
    )

    def _fake_update(a, opt, group, cfg):
        updates.append([s.ret for s in group])
        return 0.0, [0.0] * len(group)

    monkeypatch.setattr(rr, "_grpo_update", _fake_update)
    return updates


def _run(attacker, budget, judge, g=2, r=2):
    cfg = GRPOAttackConfig(num_generations=g, session_rounds=r)
    return rr.run_prompt_rl(
        base_prompt="do the harmful thing", prompt_id="p1", behavior="do the harmful thing",
        category="c", source="s", target=StubTarget(), judge=judge, attacker=attacker,
        budget=budget, cfg=cfg, reward_config=RewardConfig(),
    )


# --------------------------------------------------------------------------- #
# Budget arithmetic: one round = one target query, and the budget is the ONLY
# thing that stops the loop.
# --------------------------------------------------------------------------- #

def test_budget_is_spent_exactly_when_nothing_is_jailbroken(attacker, stub_loop):
    rec = _run(attacker, budget=9, judge=StubJudge())
    assert len(rec.steps) == 9
    assert [s.step for s in rec.steps] == list(range(1, 10))
    assert rec.metadata["rl_sessions"] == 4       # 1 raw probe + 4 sessions x 2 rounds = 9


def test_a_jailbreak_stops_the_run_only_at_the_group_boundary(attacker, stub_loop):
    """Best-of-N rules out a mid-group exit; the stop waits for the group to close."""
    rec = _run(attacker, budget=9, judge=StubJudge(unsafe_at=[2]), g=2, r=2)
    # The jailbreak lands on step 2, the first rollout of group 1 (steps 2..5). The run neither
    # stops there (that would cut a session short of its best-of-N) nor spends the full budget.
    assert len(rec.steps) == 5
    assert rec.success is True
    assert rec.first_success_step == 2, "t* is read back off the steps, not used to stop"


def test_first_success_step_matches_the_risk_curve_definition(attacker, stub_loop):
    """risk_curve calls a record successful at lambda if any judgment in steps[0:lambda] == 1."""
    rec = _run(attacker, budget=9, judge=StubJudge(unsafe_at=[4, 7]))
    assert rec.first_success_step == 4
    for lam in range(1, 10):
        expected = any(s.judgment == 1 for s in rec.steps[:lam])
        assert (rec.first_success_step <= lam) is expected


def test_no_success_is_recorded_when_nothing_is_jailbroken(attacker, stub_loop):
    rec = _run(attacker, budget=9, judge=StubJudge())
    assert rec.success is False and rec.first_success_step is None


# --------------------------------------------------------------------------- #
# The update rule: a group trains only if it is COMPLETE and budget remains, so
# a final group's update — which nothing could be sampled from — is not paid for.
# --------------------------------------------------------------------------- #

def test_only_the_non_final_complete_group_is_trained_on(attacker, stub_loop):
    _run(attacker, budget=9, judge=StubJudge())
    # 4 sessions -> 2 groups of 2. Group 1 closes at q=5 (< 9) and trains; group 2 closes at
    # q=9, with no budget left for anything to be sampled from it, so it is skipped.
    assert len(stub_loop) == 1
    assert len(stub_loop[0]) == 2  # the group's two session returns


def test_a_budget_too_small_to_close_a_group_never_updates(attacker, stub_loop):
    """Below one full group the attack is best-of-N sampling, not RL — and costs no backward."""
    _run(attacker, budget=3, judge=StubJudge())   # 1 raw + 1 session only
    assert stub_loop == []


def test_two_full_groups_train_once(attacker, stub_loop):
    _run(attacker, budget=13, judge=StubJudge())  # 1 raw + 6 sessions = 3 groups
    assert len(stub_loop) == 2                    # groups 1 and 2 train; group 3 is final


# --------------------------------------------------------------------------- #
# Session return = best score across its rounds (App. A.2).
# --------------------------------------------------------------------------- #

def test_session_return_is_best_of_n():
    sess = _Session([_Rollout(None, None, 0.2), _Rollout(None, None, 0.9),
                     _Rollout(None, None, 0.4)])
    assert sess.ret == pytest.approx(0.9)


def test_empty_session_return_is_zero():
    assert _Session([]).ret == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# The cost model must reconstruct the same structure from the recorded steps
# alone — it has no access to the runner's control flow.
# --------------------------------------------------------------------------- #

def test_cost_model_agrees_with_the_runner_on_which_steps_were_trained():
    from rup.metrics.cost_mapper import _rl_attacker_mult

    n_steps, g, r = 9, 2, 2
    mults = [_rl_attacker_mult(i, n_steps, g, r) for i in range(n_steps)]
    assert mults[0] == 0.0                     # raw behavior: the attacker never runs
    assert mults[1:5] == [8.0] * 4             # group 1 (sessions 1-2) trained
    assert mults[5:9] == [2.0] * 4             # group 2 (sessions 3-4) was the final group


def test_cost_model_charges_no_backward_when_no_group_closes():
    from rup.metrics.cost_mapper import _rl_attacker_mult

    mults = [_rl_attacker_mult(i, 3, 2, 2) for i in range(3)]
    assert mults == [0.0, 2.0, 2.0]            # generation only — matches "never updates" above


def test_update_wall_clock_is_billed_to_the_group_that_paid_for_it(attacker, monkeypatch):
    """The seconds axis must include the backward passes, not just the target queries."""
    import time as _time
    import torch

    monkeypatch.setattr(rr, "_reset_adapter", lambda a: None)
    monkeypatch.setattr(rr, "_prompt_ids", lambda a, b, ctx: torch.zeros(1, 3, dtype=torch.long))
    monkeypatch.setattr(rr, "_sample_batch",
                        lambda a, pids_list, cfg: (
                            ["a diverse candidate prompt here"] * len(pids_list),
                            [torch.zeros(2)] * len(pids_list),
                        ))

    def _slow_update(a, opt, group, cfg):
        _time.sleep(0.05)
        return 0.0, [0.0] * len(group)

    monkeypatch.setattr(rr, "_grpo_update", _slow_update)
    rec = _run(attacker, budget=9, judge=StubJudge())

    trained = [s.seconds for s in rec.steps[1:5]]   # group 1 — the one that updated
    untrained = [s.seconds for s in rec.steps[5:9]]  # group 2 — final, never updated
    assert min(trained) > max(untrained), "the update's wall-clock never reached the steps"
    assert sum(trained) - sum(untrained) == pytest.approx(0.05, abs=0.04)


def test_session_count_is_not_overcounted(attacker, stub_loop):
    rec = _run(attacker, budget=9, judge=StubJudge())
    assert rec.metadata["rl_sessions"] == 4
    assert rec.metadata["rl_session_rounds"] == 2
    assert rec.metadata["rl_num_generations"] == 2


# --------------------------------------------------------------------------- #
# Lockstep: the group's sessions advance one round at a time so each round can
# be generated as a single batch. This makes low-lambda behaviour breadth-first
# (8 diverse opening attempts) rather than depth-first (one session explored to
# depth 5 before the second is even started).
# --------------------------------------------------------------------------- #

def test_group_rounds_are_interleaved_across_sessions(attacker, stub_loop, monkeypatch):
    """Round r of every session must be spent before round r+1 of any session."""
    import torch

    seen = []

    def _batch(a, pids_list, cfg):
        seen.append(len(pids_list))
        n = len(pids_list)
        return [f"candidate number {len(seen)} for a target model"] * n, [torch.zeros(2)] * n

    monkeypatch.setattr(rr, "_sample_batch", _batch)
    rec = _run(attacker, budget=9, judge=StubJudge(), g=2, r=2)

    # 2 groups x 2 rounds = 4 batched attacker calls, each covering both sessions at once —
    # not 4 sessions x 2 rounds = 8 sequential ones.
    assert seen == [2, 2, 2, 2]
    # Steps 2,3 are round 1 of sessions 1,2; steps 4,5 are round 2 of the same two sessions.
    prompts = [s.prompt for s in rec.steps[1:5]]
    assert prompts[0] == prompts[1], "round 1 of both sessions is one batch"
    assert prompts[2] == prompts[3], "round 2 of both sessions is one batch"
    assert prompts[0] != prompts[2], "the two rounds are distinct generations"


def test_sample_batch_left_pads_and_strips_padding():
    """Generation continues from the right edge, so prompts must be LEFT-padded."""
    import torch

    PAD = 99

    class _Tok:
        pad_token_id = PAD

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(str(int(i)) for i in ids)

    class _Model:
        def __init__(self):
            self.seen = None

        def generate(self, input_ids=None, attention_mask=None, **kw):
            self.seen = (input_ids.clone(), attention_mask.clone())
            # Two completions of different length; the shorter is pad-filled by generate.
            comps = torch.tensor([[7, 8, 9], [7, PAD, PAD]])
            return torch.cat([input_ids, comps], dim=1)

    class _A:
        tokenizer = _Tok()
        model = _Model()
        device = "cpu"

    a = _A()
    short, long = torch.tensor([[1, 2]]), torch.tensor([[3, 4, 5, 6]])
    texts, comps = rr._sample_batch(a, [short, long], GRPOAttackConfig())

    ids, attn = a.model.seen
    # The 2-token prompt sits at the RIGHT of the 4-wide batch, padded on the left.
    assert ids[0].tolist() == [PAD, PAD, 1, 2]
    assert attn[0].tolist() == [0, 0, 1, 1]
    assert ids[1].tolist() == [3, 4, 5, 6] and attn[1].tolist() == [1, 1, 1, 1]

    # Batch padding is stripped, so the log-prob pass never sees a pad token.
    assert comps[0].tolist() == [7, 8, 9]
    assert comps[1].tolist() == [7], "trailing pad must not survive into the rollout"
    assert texts == ["7 8 9", "7"]


# --------------------------------------------------------------------------- #
# Group-boundary early stop: a CONFIRMED jailbreak (binary label) that also
# clears the scorer threshold ends the behavior — but only at a group boundary,
# so no session is cut short of its best-of-N and no GRPO group is left partial.
# --------------------------------------------------------------------------- #

class ThresholdJudge:
    """Returns (label, score) from a fixed schedule keyed on judge-call order."""

    def __init__(self, schedule):
        self._schedule = dict(schedule)   # call index -> (label, score)
        self.calls = 0

    def judge(self, prompt, response):
        self.calls += 1
        return self._schedule.get(self.calls, (0, 0.1))[0]

    def verdict_score(self, prompt, response):
        return self._schedule.get(self.calls, (0, 0.1))[1]


def test_stops_at_the_group_boundary_after_a_confirmed_jailbreak(attacker, stub_loop):
    # Call 3 is round 1 of session 2 in group 1 (call 1 is the raw probe).
    judge = ThresholdJudge({3: (1, 0.95)})
    rec = _run(attacker, budget=13, judge=judge, g=2, r=2)
    # Group 1 is steps 2..5; the run stops at its boundary rather than continuing to 13.
    assert len(rec.steps) == 5
    assert rec.metadata["rl_stopped_early"] is True
    assert rec.first_success_step == 3


def test_the_group_containing_the_jailbreak_still_completes(attacker, stub_loop):
    """Stopping mid-group would leave a session short of its best-of-N. It must not happen."""
    judge = ThresholdJudge({2: (1, 0.95)})   # the very first rollout of group 1
    rec = _run(attacker, budget=13, judge=judge, g=2, r=2)
    assert len(rec.steps) == 5, "the group must run to its boundary, not stop on the rollout"


def test_no_update_is_paid_for_on_the_group_we_stop_after(attacker, stub_loop):
    judge = ThresholdJudge({3: (1, 0.95)})
    _run(attacker, budget=13, judge=judge, g=2, r=2)
    assert stub_loop == [], "training a group we are about to discard is wasted compute"


def test_cost_model_infers_the_skipped_update_from_the_truncated_steps():
    """cost_mapper sees only the steps; it must not bill a backward pass that never ran."""
    from rup.metrics.cost_mapper import _rl_attacker_mult

    mults = [_rl_attacker_mult(i, 5, 2, 2) for i in range(5)]
    assert mults == [0.0, 2.0, 2.0, 2.0, 2.0]


def test_a_low_confidence_jailbreak_does_not_stop_the_run(attacker, stub_loop):
    """Below the threshold the attack keeps going, hunting for a stronger jailbreak."""
    judge = ThresholdJudge({3: (1, 0.20)})   # labelled unsafe, but scores under 0.5
    rec = _run(attacker, budget=13, judge=judge, g=2, r=2)
    assert len(rec.steps) == 13
    assert rec.metadata["rl_stopped_early"] is False
    # The success is still recorded — the threshold gates STOPPING, never labelling.
    assert rec.success is True and rec.first_success_step == 3


def test_a_high_score_without_the_binary_label_does_not_stop_the_run(attacker, stub_loop):
    """Stopping on the scorer alone would truncate a trial with no recorded success."""
    judge = ThresholdJudge({3: (0, 0.99)})
    rec = _run(attacker, budget=13, judge=judge, g=2, r=2)
    assert len(rec.steps) == 13
    assert rec.success is False


def test_raw_behavior_jailbreak_stops_immediately(attacker, stub_loop):
    """A behavior the target answers outright must not cost a whole group."""
    rec = _run(attacker, budget=13, judge=ThresholdJudge({1: (1, 0.99)}), g=2, r=2)
    assert len(rec.steps) == 1
    assert rec.first_success_step == 1 and rec.metadata["rl_stopped_early"] is True


def test_early_stop_preserves_the_risk_curve_definition(attacker, stub_loop):
    """Truncating must not change risk at any lambda: success at t* implies success beyond it."""
    rec = _run(attacker, budget=13, judge=ThresholdJudge({3: (1, 0.95)}), g=2, r=2)
    t = rec.first_success_step
    for lam in range(1, 14):
        risk_curve_says = any(s.judgment == 1 for s in rec.steps[:lam])
        assert risk_curve_says == (lam >= t), f"risk changed at lambda={lam}"


# --------------------------------------------------------------------------- #
# The KL leash. The naive `log pi - log pi_ref` has the right expectation but the
# WRONG gradient (autograd sees beta*grad(lp), a unconditional push-down), so a
# policy that drifted below the reference gets pushed further away. This matters
# most when a group has no successes: the advantages are small and the KL term is
# effectively the whole update.
# --------------------------------------------------------------------------- #

def _kl_grad(theta_val):
    """d/dtheta of the k3 penalty for a 2-token policy whose reference sits at logit 0."""
    import torch

    th = torch.tensor(float(theta_val), requires_grad=True)
    lp = torch.log_softmax(torch.stack([th, torch.zeros(())]), 0)[:1].unsqueeze(0)
    ref = torch.log_softmax(torch.zeros(2), 0)[:1].unsqueeze(0).detach()
    rr._kl_penalty(lp, ref).mean().backward()
    return th.grad.item()


def test_kl_penalty_pulls_toward_the_reference_from_both_sides():
    # Gradient DESCENT moves theta by -grad, and the reference is at theta = 0.
    assert -_kl_grad(+1.5) < 0, "a policy above the reference must be pulled down toward it"
    assert -_kl_grad(-1.5) > 0, "a policy below the reference must be pulled UP toward it"


def test_kl_penalty_is_zero_only_at_the_reference():
    import torch

    at_ref = rr._kl_penalty(torch.tensor([[-0.7]]), torch.tensor([[-0.7]])).item()
    assert at_ref == pytest.approx(0.0, abs=1e-9)
    for lp in (-0.2, -1.5, -5.0):
        assert rr._kl_penalty(torch.tensor([[lp]]), torch.tensor([[-0.7]])).item() > 0.0


def test_kl_penalty_survives_a_large_divergence():
    """A summed-logprob difference over 256 tokens would overflow exp(); clamping must hold."""
    import torch

    v = rr._kl_penalty(torch.tensor([[-500.0]]), torch.tensor([[0.0]]))
    assert torch.isfinite(v).all()


def test_a_group_with_no_successes_still_updates(attacker, stub_loop):
    """All 8 sessions refused is NOT a reason to skip: the dense reward still ranks them."""
    rec = _run(attacker, budget=9, judge=StubJudge(), g=2, r=2)   # never jailbroken
    assert rec.success is False
    assert len(stub_loop) == 1, "the non-final group must still train"


def test_a_flat_group_skips_the_update(attacker, monkeypatch):
    """Identical returns give identically-zero advantages — 40 backward passes for nothing."""
    import torch

    monkeypatch.setattr(rr, "_reset_adapter", lambda a: None)
    monkeypatch.setattr(rr, "_prompt_ids", lambda a, b, ctx: torch.zeros(1, 3, dtype=torch.long))
    monkeypatch.setattr(rr, "_sample_batch", lambda a, pl, cfg: (
        ["a diverse candidate prompt here"] * len(pl), [torch.zeros(2)] * len(pl)))

    calls = []
    real = rr._grpo_update
    monkeypatch.setattr(rr, "_grpo_update",
                        lambda a, o, g, c: (calls.append(1), real(a, o, g, c))[1])

    class _Flat:
        """Every rollout scores exactly the same, so the group carries no signal."""
        def judge(self, p, r): return 0
        def verdict_score(self, p, r): return 0.3

    rec = _run(attacker, budget=9, judge=_Flat(), g=2, r=2)
    assert calls, "the update must still be attempted"
    # ...and returns (None, None) rather than running the backward passes for a zero gradient.
    assert real(attacker, torch.optim.AdamW([attacker.model.weight], lr=1e-5),
                [_Session([_Rollout(None, None, 0.3)]),
                 _Session([_Rollout(None, None, 0.3)])],
                GRPOAttackConfig()) == (None, None)
    assert len(rec.steps) == 9
