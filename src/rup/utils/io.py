"""Core data structures shared between inference and evaluation phases."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict, fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

_warned_unknown_fields: set = set()


def _warn_once(key: str, message: str) -> None:
    """Warn about a schema mismatch once per process, not once per record."""
    if key not in _warned_unknown_fields:
        _warned_unknown_fields.add(key)
        logger.warning(message)


@dataclass
class StepResult:
    """Result of a single refinement step t in Algorithm 1.

    The four judge_* fields are the audit trail for the verdict. They exist because a judge
    whose output cannot be parsed used to be indistinguishable from a judge that said SAFE:
    the label was recorded, the raw text was thrown away, and a sweep could complete with
    every verdict fabricated by a substring fallback while looking perfectly healthy. They
    are Optional and default to None so the ~150 results.jsonl files written before this
    existed still load.
    """
    step: int          # t (1-indexed; step 0 = base prompt with no model call)
    prompt: str        # p^(t-1) — the prompt used at this step
    response: str      # y^(t-1) — model response
    judgment: int      # z^(t-1) — judge label: 1=unsafe, 0=safe
    seconds: Optional[float] = None  # measured attack-compute wall-clock for this step (None = untimed)
    judge_parsed: Optional[bool] = None   # False = judgment came from the unparsed fallback
    judge_branch: Optional[str] = None    # which parse stage produced the verdict
    judge_raw: Optional[str] = None        # truncated judge output, for auditing
    judge_out_tokens: Optional[int] = None  # judge's generated length (feeds the cost axis)


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
        """Load a record, ignoring fields this version does not know about.

        Forward tolerance matters here: results trees are read by whatever checkout happens
        to be on the node, and a tree written with a newer schema would otherwise raise
        TypeError on StepResult(**s) and take down an unrelated evaluation job.
        """
        d = dict(d)
        step_names = {f.name for f in dataclass_fields(StepResult)}
        steps = []
        for s in d.pop("steps"):
            extra = set(s) - step_names
            if extra:
                _warn_once(
                    f"StepResult:{sorted(extra)}",
                    f"Ignoring unknown StepResult fields {sorted(extra)} — this file was "
                    f"written by a newer version of rup.",
                )
            steps.append(StepResult(**{k: v for k, v in s.items() if k in step_names}))

        record_names = {f.name for f in dataclass_fields(cls)}
        extra = set(d) - record_names
        if extra:
            _warn_once(
                f"TrialRecord:{sorted(extra)}",
                f"Ignoring unknown TrialRecord fields {sorted(extra)} — this file was "
                f"written by a newer version of rup.",
            )
        return cls(steps=steps, **{k: v for k, v in d.items() if k in record_names})


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
