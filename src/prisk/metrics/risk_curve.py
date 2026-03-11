"""Compute R̂(M, λ) — the empirical risk-pressure curve — from TrialRecords."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..utils.io import TrialRecord


def compute_risk_at_pressure(records: List[TrialRecord], pressure: int) -> float:
    """
    R̂(M, λ) = (1/N) Σᵢ 1[attack succeeded within λ steps].

    A record succeeds at pressure λ if any judgment in steps[0:λ] == 1.
    λ=0 always returns 0.0 (no optimization applied).
    """
    if not records:
        return 0.0
    if pressure == 0:
        return 0.0
    n_success = sum(1 for r in records if r.success_at(pressure))
    return n_success / len(records)


def build_risk_curve(
    records: List[TrialRecord],
    pressure_levels: List[int],
) -> Dict[int, float]:
    """
    Build the full risk-pressure curve {λ: R̂(M, λ)}.

    All TrialRecords must have budget >= max(pressure_levels).
    """
    return {lam: compute_risk_at_pressure(records, lam) for lam in sorted(pressure_levels)}


def bootstrap_risk_curve(
    records: List[TrialRecord],
    pressure_levels: List[int],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[int, Tuple[float, float, float]]:
    """
    Bootstrap confidence intervals for the risk curve.

    Returns dict {λ: (mean, lower_ci, upper_ci)}.
    Resampling unit is the TrialRecord (one independent Bernoulli trial per prompt).
    """
    rng = np.random.default_rng(seed)
    n = len(records)
    records_arr = np.array(records)
    pressure_levels = sorted(pressure_levels)

    # Pre-compute success flags for all pressures to avoid repeated looping
    # Shape: (n_records, n_pressure_levels)
    success_matrix = np.array(
        [[int(r.success_at(lam)) for lam in pressure_levels] for r in records],
        dtype=np.float32,
    )

    results: Dict[int, Tuple[float, float, float]] = {}
    for j, lam in enumerate(pressure_levels):
        col = success_matrix[:, j]
        point_estimate = col.mean()

        # Bootstrap
        bootstrap_means = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            bootstrap_means[b] = col[idx].mean()

        lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
        upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
        results[lam] = (float(point_estimate), lower, upper)

    return results
