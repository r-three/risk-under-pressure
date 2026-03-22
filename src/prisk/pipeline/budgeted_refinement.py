"""
Algorithm 1: Budgeted Iterative Refinement for Prompt Optimization.

Input:  base prompt p, target model M, unsafe evaluator E, refinement policy π, budget λ
Output: TrialRecord with final prompt p^(λ), success flag s, first-success step t*
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ..utils.io import StepResult, TrialRecord
from ..utils.logging import get_logger

if TYPE_CHECKING:
    from ..attacks.base import AttackPolicy
    from ..judges.base import SafetyJudge
    from ..models.base import BaseModel

logger = get_logger(__name__)


def run_trial(
    base_prompt: str,
    prompt_id: str,
    behavior: str,
    category: str,
    source: str,
    model: "BaseModel",
    judge: "SafetyJudge",
    attack: "AttackPolicy",
    budget: int,
) -> TrialRecord:
    """
    Execute Algorithm 1 for a single (prompt, model, attack) triple.

    Args:
        base_prompt:  original harmful behavior text (p from benchmark)
        prompt_id:    benchmark-assigned identifier
        behavior:     human-readable behavior description
        category:     harm category
        source:       benchmark name
        model:        target model M
        judge:        safety evaluator E(·)
        attack:       refinement policy π
        budget:       maximum refinement iterations λ

    Returns:
        TrialRecord containing all steps, success flag, and first-success step.
    """
    t0 = time.time()

    # Initialize: p^(0) ← p  (attack may apply initial transformation)
    p = attack.initialize(base_prompt)
    s = 0          # success flag
    t_star = None  # first-success step

    steps: list[StepResult] = []

    for t in range(1, budget + 1):
        # y^(t-1) ← M(p^(t-1))
        y = model.generate(p)

        # z^(t-1) ← E(y^(t-1))
        z = judge.judge(p, y)

        steps.append(StepResult(step=t, prompt=p, response=y, judgment=z))

        # Record first success and stop — no further refinement needed
        if z == 1:
            s = 1
            t_star = t
            logger.debug(f"[{prompt_id}] First success at step {t}")
            break

        # p^(t) ← π(p^(t-1), y^(t-1), z^(t-1), t)
        p = attack.refine(p, y, z, t)

    elapsed = time.time() - t0
    logger.info(
        f"[{model.model_id}/{attack.attack_id}] {prompt_id}: "
        f"success={bool(s)} t*={t_star} budget={budget} ({elapsed:.1f}s)"
    )

    return TrialRecord(
        prompt_id=prompt_id,
        base_prompt=base_prompt,
        behavior=behavior,
        category=category,
        source=source,
        model_id=model.model_id,
        attack_id=attack.attack_id,
        budget=budget,
        steps=steps,
        success=bool(s),
        first_success_step=t_star,
        final_prompt=p,
        metadata={"elapsed_seconds": elapsed},
    )
