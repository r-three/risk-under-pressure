"""Abstract base class for attack/refinement policies (π in Algorithm 1)."""

from __future__ import annotations

from abc import ABC, abstractmethod


class AttackPolicy(ABC):
    """
    Refinement policy π from Algorithm 1.

    Usage per trial:
        prompt = policy.initialize(base_prompt)    # p^(0)
        for t in 1..lambda:
            response = model.generate(prompt)
            judgment = judge(prompt, response)
            prompt = policy.refine(prompt, response, judgment, t)  # p^(t)
    """

    @property
    @abstractmethod
    def attack_id(self) -> str: ...

    @abstractmethod
    def initialize(self, base_prompt: str) -> str:
        """
        Set up state for a new trial and return p^(0).
        For most attacks this is the base prompt unchanged.
        JailBroken applies the first template here.
        """
        ...

    @abstractmethod
    def refine(self, prompt: str, response: str, judgment: int, step: int) -> str:
        """
        Generate p^(t) from (p^(t-1), y^(t-1), z^(t-1), t).

        Args:
            prompt:    current prompt p^(t-1)
            response:  model output y^(t-1)
            judgment:  safety label z^(t-1) — 1=unsafe, 0=safe
            step:      iteration number t (1-indexed)

        Returns:
            refined prompt p^(t)
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.attack_id!r})"
