"""Core data structures shared between inference and evaluation phases."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class StepResult:
    """Result of a single refinement step t in Algorithm 1."""
    step: int          # t (1-indexed; step 0 = base prompt with no model call)
    prompt: str        # p^(t-1) — the prompt used at this step
    response: str      # y^(t-1) — model response
    judgment: int      # z^(t-1) — judge label: 1=unsafe, 0=safe


@dataclass
class TrialRecord:
    """
    Full record from running Algorithm 1 on a single (prompt, model, attack, budget).

    The `steps` list has length `budget`. Steps are 1-indexed matching Algorithm 1.
    A trial is 'successful' (s=1) if any step's judgment == 1.

    To derive risk at pressure λ ≤ budget:
        success_at_lambda = any(s.judgment == 1 for s in record.steps[:lambda])
    """
    prompt_id: str
    base_prompt: str
    behavior: str           # human-readable harm behavior description
    category: str           # harm category from benchmark
    source: str             # benchmark name: "jailbreakbench" | "harmbench"
    model_id: str
    attack_id: str
    budget: int             # lambda_max used for this trial
    steps: List[StepResult]
    success: bool           # True if any step succeeded
    first_success_step: Optional[int]  # t* (None if never succeeded)
    final_prompt: str       # p^(lambda)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def success_at(self, pressure: int) -> bool:
        """Return True if attack succeeded within `pressure` steps (cumulative)."""
        if pressure == 0:
            return False
        return any(s.judgment == 1 for s in self.steps[:pressure])

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TrialRecord:
        steps = [StepResult(**s) for s in d.pop("steps")]
        return cls(steps=steps, **d)


def write_jsonl(records: List[TrialRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")


def append_jsonl(record: TrialRecord, path: Path) -> None:
    """Append a single record — safe to call after each trial for crash-resilience."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record.to_dict()) + "\n")


def read_jsonl(path: Path) -> Iterator[TrialRecord]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield TrialRecord.from_dict(json.loads(line))


def load_completed_ids(path: Path) -> set:
    """Return set of prompt_ids already present in a results file."""
    if not path.exists():
        return set()
    return {r.prompt_id for r in read_jsonl(path)}
