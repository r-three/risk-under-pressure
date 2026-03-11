"""Abstract base class for safety judges."""

from __future__ import annotations

from abc import ABC, abstractmethod


class SafetyJudge(ABC):
    """
    Binary safety classifier: 1 = unsafe (jailbreak succeeded), 0 = safe.

    The judge is the evaluator E(·) in Algorithm 1.
    """

    @abstractmethod
    def judge(self, prompt: str, response: str) -> int:
        """
        Classify whether response is unsafe given the prompt context.

        Returns:
            1 if the response is unsafe / harmful
            0 if the response is safe / refused
        """
        ...

    def judge_batch(self, pairs: list[tuple[str, str]]) -> list[int]:
        """Judge multiple (prompt, response) pairs. Override for batched efficiency."""
        return [self.judge(p, r) for p, r in pairs]
