"""Unit tests for the offline re-judge materializer.

Pure logic — no GPU, no judge, no model. The materializer takes a stored trajectory plus a new
set of labels and rebuilds the TrialRecord a live run under the new judge would have produced.

Re-labelling a fixed trajectory is exact in two of three cases and lossy in the third:

  A. the new judge succeeds at some stored step t*' <= K       exact
  B. no new success and K == budget                            exact
  C. no new success and K < budget                             right-censored at K

Case C is the only lossy one and it only arises when the new judge is stricter than the one that
produced the tree. These tests pin all three, plus the truncation invariant that keeps the cost
axis honest (cost_mapper sums steps[:lambda], so a trailing step is a real overcharge).
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rup.utils.io import StepResult, TrialRecord


def _load_module():
    path = Path(__file__).parent.parent / "scripts" / "rejudge_offline.py"
    spec = importlib.util.spec_from_file_location("_rejudge_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rj = _load_module()


def _record(judgments, budget=10, attack_id="pair", behavior="do the harmful thing"):
    """A trial that stopped at the first 1, exactly as the live pipeline records it."""
    steps = []
    for i, j in enumerate(judgments, start=1):
        steps.append(StepResult(step=i, prompt=f"p{i}", response=f"r{i}", judgment=j, seconds=1.0))
    t_star = next((i for i, j in enumerate(judgments, start=1) if j == 1), None)
    return TrialRecord(
        prompt_id="jbb_0000", base_prompt="bp", behavior=behavior, category="c",
        source="jailbreakbench", model_id="m", attack_id=attack_id, budget=budget,
        steps=steps, success=t_star is not None, first_success_step=t_star,
        final_prompt="fp",
    )


def _sidecar_row(rec, new_labels, parsed=None, branch="first_line"):
    """Sidecar row pairing each stored step with its new label."""
    parsed = [True] * len(new_labels) if parsed is None else parsed
    steps = [
        {
            "step": st.step, "judgment": lab, "orig_judgment": st.judgment,
            "parsed": p, "branch": branch if p else "unparsed",
            "escalated": False, "out_tokens": 3, "raw": "SAFE" if lab == 0 else "UNSAFE",
        }
        for st, lab, p in zip(rec.steps, new_labels, parsed)
    ]
    n_flips = sum(1 for s in steps if s["judgment"] is not None
                  and s["judgment"] != s["orig_judgment"])
    return {
        "prompt_id": rec.prompt_id, "judge": "olmo3-7b-instruct",
        "rubric_id": "default", "rubric_sha1": "deadbeef",
        "n_unparsed": sum(1 for p in parsed if not p), "n_flips": n_flips,
        "steps": steps,
    }


# ─────────────────────────────────────────────────── apply_policy


@pytest.mark.parametrize("policy,expected", [("safe", 0), ("orig", 1), ("drop", None)])
def test_apply_policy_resolves_unparsed(policy, expected):
    assert rj.apply_policy(None, orig_label=1, policy=policy) == expected


def test_apply_policy_passes_a_real_label_through_unchanged():
    for policy in ("safe", "orig", "drop"):
        assert rj.apply_policy(1, orig_label=0, policy=policy) == 1
        assert rj.apply_policy(0, orig_label=1, policy=policy) == 0


def test_apply_policy_rejects_an_unknown_policy():
    with pytest.raises(ValueError, match="unknown unparsed policy"):
        rj.apply_policy(None, 0, "guess")


# ─────────────────────────────────────────────────── Case A: exact, either direction


def test_more_permissive_judge_shortens_and_truncates():
    """Success moves earlier. Trailing steps must be dropped, or the cost axis overcharges."""
    rec = _record([0, 0, 0, 1])                     # original stopped at 4
    row = _sidecar_row(rec, [0, 1, 1, 1])           # new judge succeeds at 2

    out, prov = rj.materialize_record(rec, row, "safe")

    assert out.first_success_step == 2
    assert out.success is True
    assert len(out.steps) == 2                      # THE truncation invariant
    assert [s.step for s in out.steps] == [1, 2]
    assert prov["censored"] is False
    assert prov["orig_first_success_step"] == 4


def test_stricter_judge_that_still_succeeds_later_is_exact():
    """t*' later than the original is still exact, as long as it lands inside the stored steps."""
    rec = _record([0, 0, 1])
    row = _sidecar_row(rec, [0, 0, 1])
    out, prov = rj.materialize_record(rec, row, "safe")
    assert out.first_success_step == 3
    assert len(out.steps) == 3
    assert prov["censored"] is False


def test_audit_fields_are_carried_onto_the_materialized_steps():
    rec = _record([0, 1])
    row = _sidecar_row(rec, [0, 1])
    out, _ = rj.materialize_record(rec, row, "safe")

    assert out.steps[1].judge_branch == "first_line"
    assert out.steps[1].judge_parsed is True
    assert out.steps[1].judge_out_tokens == 3
    assert out.steps[1].judge_raw == "UNSAFE"
    # The response and prompt must be the originals, untouched.
    assert out.steps[1].response == "r2"


# ─────────────────────────────────────────────────── Case B: exact


def test_budget_exhausted_with_no_new_success_is_not_censored():
    rec = _record([0] * 10, budget=10)              # ran the full budget, never succeeded
    row = _sidecar_row(rec, [0] * 10)

    out, prov = rj.materialize_record(rec, row, "safe")

    assert out.success is False
    assert out.first_success_step is None
    assert len(out.steps) == 10
    assert prov["censored"] is False                # no data is missing
    assert prov["censored_at"] is None


# ─────────────────────────────────────────────────── Case C: censored


def test_unflagged_success_is_marked_censored():
    """The lossy case: the old judge stopped the run, the new one disagrees, no data past there."""
    rec = _record([0, 0, 1], budget=10)             # stopped at 3 of a possible 10
    row = _sidecar_row(rec, [0, 0, 0])              # new judge un-flags it

    out, prov = rj.materialize_record(rec, row, "safe")

    assert out.success is False
    assert out.first_success_step is None
    assert len(out.steps) == 3                      # all stored steps kept
    assert prov["censored"] is True
    assert prov["censored_at"] == 3


def test_censoring_requires_a_stricter_judge():
    """A judge that only ever agrees-or-flags-more cannot produce censoring."""
    rec = _record([0, 0, 1], budget=10)
    row = _sidecar_row(rec, [0, 1, 1])              # more permissive
    _, prov = rj.materialize_record(rec, row, "safe")
    assert prov["censored"] is False


# ─────────────────────────────────────────────────── unparsed handling


def test_unparsed_step_under_safe_policy_does_not_create_success():
    rec = _record([0, 1], budget=10)
    row = _sidecar_row(rec, [0, None], parsed=[True, False])

    out, prov = rj.materialize_record(rec, row, "safe")

    assert out.success is False
    assert prov["censored"] is True                 # the old success was not reproduced
    assert out.steps[1].judge_parsed is False
    assert out.steps[1].judge_branch == "unparsed"


def test_unparsed_step_under_orig_policy_reproduces_the_old_label():
    rec = _record([0, 1], budget=10)
    row = _sidecar_row(rec, [0, None], parsed=[True, False])

    out, prov = rj.materialize_record(rec, row, "orig")

    assert out.success is True                      # fell back to the stored 1
    assert out.first_success_step == 2
    assert prov["censored"] is False


def test_unparsed_judgment_never_lands_in_the_record_as_none():
    """StepResult.judgment is typed int and read by success_at — it must never be None."""
    rec = _record([0, 1], budget=10)
    row = _sidecar_row(rec, [None, None], parsed=[False, False])
    out, _ = rj.materialize_record(rec, row, "drop")
    assert all(isinstance(s.judgment, int) for s in out.steps)


# ─────────────────────────────────────────────────── identity / no-op proof


def test_identity_relabelling_is_a_no_op():
    """The only way to trust the materializer when labels change is to prove it when they don't.

    Re-judging with the same labels must reproduce the original record exactly, modulo the
    metadata provenance and the audit fields.
    """
    rec = _record([0, 0, 1], budget=10)
    row = _sidecar_row(rec, [0, 0, 1])

    out, _ = rj.materialize_record(rec, row, "safe")

    assert out.success == rec.success
    assert out.first_success_step == rec.first_success_step
    assert len(out.steps) == len(rec.steps)
    for a, b in zip(out.steps, rec.steps):
        assert (a.step, a.prompt, a.response, a.judgment, a.seconds) == \
               (b.step, b.prompt, b.response, b.judgment, b.seconds)
    for lam in range(0, rec.budget + 1):
        assert out.success_at(lam) == rec.success_at(lam)


def test_success_at_matches_the_new_labels():
    rec = _record([0, 0, 0, 1], budget=10)
    row = _sidecar_row(rec, [0, 1, 1, 1])
    out, _ = rj.materialize_record(rec, row, "safe")

    assert out.success_at(1) is False
    assert out.success_at(2) is True
    assert out.success_at(10) is True


def test_partial_depth_sidecar_truncates_rather_than_guessing():
    """--max-steps produces a shallower sidecar; unjudged steps must not be invented."""
    rec = _record([0, 0, 0, 1], budget=10)
    row = _sidecar_row(rec, [0, 0, 0, 1])
    row["steps"] = row["steps"][:2]                 # only 2 of 4 judged

    out, prov = rj.materialize_record(rec, row, "safe")

    assert len(out.steps) == 2
    assert out.success is False
    assert prov["censored"] is True                 # only 2 of a 10 budget -> right-censored


# ─────────────────────────────────────────────────── round trip through disk


def test_materialize_writes_a_loadable_tree(tmp_path):
    src_dir = tmp_path / "jailbreakbench" / "m" / "1394" / "pair"
    src_dir.mkdir(parents=True)
    src = src_dir / "results.jsonl"

    rec = _record([0, 0, 1], budget=10)
    src.write_text(json.dumps(rec.to_dict()) + "\n")

    sidecar_name = rj.sidecar_filename("olmo3-7b-instruct")
    (src_dir / sidecar_name).write_text(json.dumps(_sidecar_row(rec, [0, 1, 1])) + "\n")

    out_root = tmp_path / "out"
    manifest = rj.materialize(
        results_dir=tmp_path / "jailbreakbench", out_root=out_root,
        judge_id="olmo3-7b-instruct", sidecar_name=sidecar_name, policy="safe",
    )

    written = out_root / "m" / "1394" / "pair" / "results.jsonl"
    assert written.exists()

    from rup.utils.io import read_jsonl
    (loaded,) = list(read_jsonl(written))
    assert loaded.first_success_step == 2
    assert len(loaded.steps) == 2
    assert loaded.metadata["rejudge"]["judge"] == "olmo3-7b-instruct"

    assert manifest["totals"]["trials"] == 1
    assert manifest["totals"]["censored"] == 0


def test_sidecar_filename_is_always_judge_slugged():
    assert rj.sidecar_filename("olmo3-7b-instruct") == "rejudge__olmo3_7b_instruct.jsonl"
    assert rj.sidecar_filename("gemma3-4b-it") == "rejudge__gemma3_4b_it.jsonl"
    # Re-judging with the incumbent is a real validation run, so it gets a file too.
    assert rj.sidecar_filename("llama3.1-8b-instruct") == "rejudge__llama3_1_8b_instruct.jsonl"
    assert rj.sidecar_filename("x", override="custom.jsonl") == "custom.jsonl"


def test_discover_results_matches_the_evaluation_layout(tmp_path):
    root = tmp_path / "jailbreakbench"
    for attack in ("pair", "gcg"):
        d = root / "tulu3-8b-sft" / "1394" / attack
        d.mkdir(parents=True)
        (d / "results.jsonl").write_text("")

    found = rj.discover_results(root)
    assert set(found) == {("tulu3-8b-sft", "pair"), ("tulu3-8b-sft", "gcg")}
