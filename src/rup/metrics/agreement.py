"""Inter-judge agreement statistics for the judge-identity ablation.

Cohen's kappa is the statistic a reviewer asks for when they doubt a judge, and until the
offline re-judge existed it was not computable here: the judge sits inside the attack loop as
the early-stopping criterion, so two judges produced different trajectories and there were no
matched (prompt, response) pairs to compare. Re-judging stored responses makes the labels
paired, which is what these functions consume.

Two details that matter for honesty:

  * Bootstrap over PROMPTS, not steps. Steps within a trial share a prompt and an attack
    trajectory, so resampling steps treats correlated observations as independent and returns a
    confidence interval several times too narrow.
  * kappa is undefined, not zero, when a judge's labels are constant. A judge that says UNSAFE
    to everything has no variance to correlate; reporting 0.0 would read as "measured no
    agreement" when the truth is "this instrument cannot be compared". Return nan.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "confusion",
    "cohens_kappa",
    "kappa_ci",
    "agreement_table",
    "error_rates",
]


def confusion(a: Sequence[int], b: Sequence[int]) -> Dict[str, int]:
    """2x2 counts for two binary labellings of the same items."""
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    out = {"n00": 0, "n01": 0, "n10": 0, "n11": 0}
    for x, y in zip(a, b):
        out[f"n{int(x)}{int(y)}"] += 1
    return out


def cohens_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Cohen's kappa for two binary labellings.

    Returns nan when either labelling is constant (kappa is undefined there, and a saturated
    judge is exactly the case this ablation has to be able to report).
    """
    n = len(a)
    if n == 0:
        return math.nan
    c = confusion(a, b)
    po = (c["n00"] + c["n11"]) / n
    pa1 = (c["n10"] + c["n11"]) / n     # rater a says 1
    pb1 = (c["n01"] + c["n11"]) / n     # rater b says 1

    # A constant rater makes kappa arithmetically 0 rather than a division by zero, which is
    # worse than an error: 0.0 reads as "measured no agreement beyond chance" when the truth is
    # "this rater has no variance, so agreement is not defined". The broken Olmo judge labelled
    # every step UNSAFE, and reporting rho/kappa = 0 against it is what made a degenerate
    # instrument look merely uncorrelated. Say undefined.
    if pa1 in (0.0, 1.0) or pb1 in (0.0, 1.0):
        return math.nan

    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if math.isclose(pe, 1.0):
        return math.nan
    return (po - pe) / (1 - pe)


def kappa_ci(
    a: Sequence[int],
    b: Sequence[int],
    groups: Optional[Sequence] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """(kappa, lo, hi) with a percentile bootstrap CI.

    Args:
        groups: cluster label per item — pass the prompt_id so resampling draws whole prompts.
                Omitting it resamples items independently, which understates the interval
                whenever several steps come from the same trial.
    """
    import random

    point = cohens_kappa(a, b)
    n = len(a)
    if n == 0 or math.isnan(point):
        return point, math.nan, math.nan

    rng = random.Random(seed)

    if groups is None:
        clusters = [[i] for i in range(n)]
    else:
        by_group: Dict[object, List[int]] = {}
        for i, g in enumerate(groups):
            by_group.setdefault(g, []).append(i)
        clusters = list(by_group.values())

    draws = []
    for _ in range(n_bootstrap):
        idx: List[int] = []
        for _ in range(len(clusters)):
            idx.extend(clusters[rng.randrange(len(clusters))])
        k = cohens_kappa([a[i] for i in idx], [b[i] for i in idx])
        if not math.isnan(k):
            draws.append(k)

    if not draws:
        return point, math.nan, math.nan
    draws.sort()
    lo = draws[max(0, int((alpha / 2) * len(draws)) - 1)]
    hi = draws[min(len(draws) - 1, int((1 - alpha / 2) * len(draws)))]
    return point, lo, hi


def agreement_table(
    labels: Dict[str, Sequence[int]],
    groups: Optional[Sequence] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> List[dict]:
    """Pairwise agreement for every judge pair in `labels`.

    Returns one row per pair with kappa + CI, raw agreement, and the off-diagonal flip counts
    (n01/n10), which are what show WHICH direction a judge disagrees in.
    """
    names = sorted(labels)
    rows = []
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            a, b = labels[x], labels[y]
            k, lo, hi = kappa_ci(a, b, groups=groups, n_bootstrap=n_bootstrap, seed=seed)
            c = confusion(a, b)
            n = len(a) or 1
            rows.append({
                "judge_a": x, "judge_b": y, "n": len(a),
                "kappa": k, "kappa_lo": lo, "kappa_hi": hi,
                "raw_agreement": (c["n00"] + c["n11"]) / n,
                **c,
                "rate_a": (c["n10"] + c["n11"]) / n,
                "rate_b": (c["n01"] + c["n11"]) / n,
            })
    return rows


def error_rates(pred: Sequence[int], truth: Sequence[int]) -> Dict[str, float]:
    """FPR/FNR/precision/recall of `pred` against reference labels `truth`.

    FPR here is the number the judge ablation turns on: on a population of known refusals it is
    the fraction of refusals a judge calls jailbreaks. A refusal-only control set is not enough
    on its own, though — an all-SAFE judge scores a perfect 0% FPR on it — so always report FNR
    on a compliance population alongside.
    """
    c = confusion(truth, pred)          # truth first: n<truth><pred>
    tn, fp, fn, tp = c["n00"], c["n01"], c["n10"], c["n11"]
    return {
        "n": len(pred),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "fpr": fp / (fp + tn) if (fp + tn) else math.nan,
        "fnr": fn / (fn + tp) if (fn + tp) else math.nan,
        "precision": tp / (tp + fp) if (tp + fp) else math.nan,
        "recall": tp / (tp + fn) if (tp + fn) else math.nan,
        "accuracy": (tp + tn) / len(pred) if len(pred) else math.nan,
    }
