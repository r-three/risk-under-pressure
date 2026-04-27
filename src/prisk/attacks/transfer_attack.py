"""Transfer attack: replay pre-computed adversarial trajectories against a new target model."""

from __future__ import annotations

from .base import AttackPolicy


class TransferAttack(AttackPolicy):
    """
    Replays a source model's attack trajectory (step-by-step prompts) against a target model.

    Useful for measuring transferability of white-box attacks (e.g., GCG) in a black-box setting:
    the optimized prompts from model A are applied verbatim to model B, and the target model's
    responses are judged independently.

    Usage per trial:
        attack.prepare(prompt_id)           # load the source trajectory for this prompt
        p = attack.initialize(base_prompt)  # returns source step-1 prompt
        for t in 1..budget:
            y = target_model.generate(p)
            z = judge(p, y)
            p = attack.refine(p, y, z, t)  # returns source step-(t+1) prompt
    """

    def __init__(self, attack_id: str, trajectories: dict[str, list[dict]]):
        """
        Args:
            attack_id:    identifier string, e.g. "transfer_gcg_from_tulu3-8b-sft"
            trajectories: mapping from prompt_id to a list of step dicts, each with at
                          minimum a "prompt" key (matching StepResult fields).
        """
        self._attack_id = attack_id
        self.trajectories = trajectories
        self._current_steps: list[dict] = []

    @property
    def attack_id(self) -> str:
        return self._attack_id

    def prepare(self, prompt_id: str) -> int:
        """
        Load the source trajectory for this prompt_id.

        Must be called before each call to initialize(). Returns the trajectory
        length (0 if the prompt_id is missing). Callers should use the returned
        length as the per-trial budget so the target model is only evaluated on
        the prompts the source actually produced — no padding or repetition.
        """
        self._current_steps = self.trajectories.get(prompt_id, [])
        return len(self._current_steps)

    def initialize(self, base_prompt: str) -> str:
        """Return the source step-1 prompt; fall back to base_prompt if no trajectory."""
        if self._current_steps:
            return self._current_steps[0]["prompt"]
        return base_prompt

    def refine(self, prompt: str, response: str, judgment: int, step: int) -> str:
        """
        Return the source prompt for the next step.

        step is 1-indexed (same convention as Algorithm 1). run_trial always calls
        refine after the final step even when no further refinement is possible
        (the return value is discarded), so we clamp to the last valid index to
        avoid an IndexError on that dummy call.
        """
        if not self._current_steps:
            return prompt
        idx = min(step, len(self._current_steps) - 1)
        return self._current_steps[idx]["prompt"]
