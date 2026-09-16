"""Unit tests for inter-judge agreement statistics."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rup.metrics.agreement import (
    agreement_table,
    cohens_kappa,
    confusion,
    error_rates,
    kappa_ci,
)


def test_confusion_counts():
    c = confusion([0, 0, 1, 1], [0, 1, 0, 1])
    assert c == {"n00": 1, "n01": 1, "n10": 1, "n11": 1}


def test_confusion_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        confusion([0, 1], [0])


def test_kappa_perfect_and_inverted():
    assert cohens_kappa([0, 1, 0, 1], [0, 1, 0, 1]) == 1.0
    assert cohens_kappa([0, 1, 0, 1], [1, 0, 1, 0]) == -1.0


def test_kappa_at_chance_is_zero():
    assert cohens_kappa([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.0)


def test_kappa_is_undefined_against_a_saturated_judge():
    """The headline case: a judge that says UNSAFE to everything cannot be correlated.

    Arithmetically kappa comes out 0.0 here, which would be reported as "no agreement beyond
    chance" — indistinguishable from a judge that genuinely disagrees. It has to be nan, or a
    degenerate instrument reads as merely uncorrelated.
    """
    assert math.isnan(cohens_kappa([0, 1, 0, 1], [1, 1, 1, 1]))
    assert math.isnan(cohens_kappa([0, 1, 0, 1], [0, 0, 0, 0]))
    assert math.isnan(cohens_kappa([1, 1, 1, 1], [1, 1, 1, 1]))


def test_kappa_empty_is_nan():
    assert math.isnan(cohens_kappa([], []))


def test_kappa_ci_brackets_the_point_estimate():
    a = [0, 1] * 50
    b = [0, 1] * 45 + [1, 0] * 5
    k, lo, hi = kappa_ci(a, b, n_bootstrap=200, seed=1)
    assert lo <= k <= hi
    assert 0.5 < k < 1.0


def test_kappa_ci_is_wider_when_clustered_by_prompt():
    """Steps within a trial are correlated; resampling them independently fakes precision."""
    # 20 prompts x 5 steps, with the disagreement concentrated in whole prompts.
    a, b, groups = [], [], []
    for p in range(20):
        for s in range(5):
            a.append(s % 2)
            b.append(s % 2 if p >= 4 else 1 - s % 2)
            groups.append(f"prompt{p}")

    _, lo_flat, hi_flat = kappa_ci(a, b, n_bootstrap=300, seed=7)
    _, lo_clu, hi_clu = kappa_ci(a, b, groups=groups, n_bootstrap=300, seed=7)

    assert (hi_clu - lo_clu) > (hi_flat - lo_flat)


def test_kappa_ci_of_a_saturated_pair_is_nan():
    k, lo, hi = kappa_ci([0, 1] * 10, [1] * 20, n_bootstrap=50)
    assert math.isnan(k) and math.isnan(lo) and math.isnan(hi)


def test_agreement_table_covers_every_pair_and_reports_flip_direction():
    labels = {
        "llama": [0, 0, 1, 1],
        "gemma": [0, 1, 1, 1],   # flags one extra
        "olmo": [1, 1, 1, 1],    # saturated
    }
    rows = agreement_table(labels, n_bootstrap=50)

    assert len(rows) == 3  # 3 choose 2
    pairs = {(r["judge_a"], r["judge_b"]) for r in rows}
    assert pairs == {("gemma", "llama"), ("gemma", "olmo"), ("llama", "olmo")}

    lg = next(r for r in rows if {r["judge_a"], r["judge_b"]} == {"llama", "gemma"})
    # gemma is judge_a alphabetically; it flags item 1 that llama calls safe.
    assert lg["n10"] + lg["n01"] == 1
    assert lg["raw_agreement"] == pytest.approx(0.75)

    olmo_rows = [r for r in rows if "olmo" in (r["judge_a"], r["judge_b"])]
    assert all(math.isnan(r["kappa"]) for r in olmo_rows)


def test_error_rates_on_a_refusal_population():
    """FPR on known refusals — the number the judge ablation turns on."""
    truth = [0] * 10                      # all refusals
    pred = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # judge flags 2 of them
    r = error_rates(pred, truth)

    assert r["fpr"] == pytest.approx(0.2)
    assert r["fp"] == 2
    assert math.isnan(r["fnr"])           # no positives in the population


def test_error_rates_catches_the_all_safe_judge_only_with_positives():
    """A refusal-only control set gives a degenerate all-SAFE judge a perfect score."""
    refusals_only = error_rates(pred=[0] * 10, truth=[0] * 10)
    assert refusals_only["fpr"] == 0.0    # looks perfect...

    with_positives = error_rates(pred=[0] * 10, truth=[0] * 5 + [1] * 5)
    assert with_positives["fpr"] == 0.0
    assert with_positives["fnr"] == 1.0   # ...and here the failure is visible


def test_error_rates_full_confusion():
    r = error_rates(pred=[1, 0, 1, 0], truth=[1, 1, 0, 0])
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (1, 1, 1, 1)
    assert r["precision"] == pytest.approx(0.5)
    assert r["recall"] == pytest.approx(0.5)
    assert r["accuracy"] == pytest.approx(0.5)
