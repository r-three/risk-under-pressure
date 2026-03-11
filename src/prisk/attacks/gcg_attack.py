"""GCG (Greedy Coordinate Gradient) attack policy.

GCG is a white-box gradient-based attack that optimizes an adversarial suffix.
It requires access to model logits and therefore only works with local HuggingFace
models — not API models.
"""

from __future__ import annotations

from typing import Optional

from ..utils.config import AttackConfig
from ..utils.logging import get_logger
from .base import AttackPolicy

logger = get_logger(__name__)

_GCG_SUFFIX_INIT = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"


class GCGAttack(AttackPolicy):
    """
    GCG attack: appends an adversarial suffix that is iteratively optimized
    via greedy coordinate gradient descent.

    Wraps EasyJailbreak's GCG implementation when available. The EasyJailbreak
    attacker is configured with n_steps=1 per call so Algorithm 1 controls
    the budget.

    NOTE: Requires a HuggingFace (local) model. Raises NotImplementedError
    for API-based models.
    """

    def __init__(self, config: AttackConfig, target_model=None):
        self._config = config
        self._target_model = target_model
        self._current_suffix: str = _GCG_SUFFIX_INIT
        self._base_prompt: Optional[str] = None
        self._gcg_attacker = None

    @property
    def attack_id(self) -> str:
        return "gcg"

    def _check_model_compatible(self) -> None:
        if self._target_model is None:
            return
        # Check that this is a HuggingFace model
        from ..models.hf_model import HFModel
        if not isinstance(self._target_model, HFModel):
            raise NotImplementedError(
                "GCG is a white-box attack and requires a local HuggingFace model. "
                f"Got: {type(self._target_model).__name__}"
            )

    def _try_init_easyjailbreak(self) -> bool:
        try:
            # EasyJailbreak GCG integration placeholder
            # Actual integration depends on EasyJailbreak API version
            import easyjailbreak  # type: ignore  # noqa: F401
            return True
        except ImportError:
            return False

    def initialize(self, base_prompt: str) -> str:
        self._check_model_compatible()
        self._base_prompt = base_prompt
        self._current_suffix = _GCG_SUFFIX_INIT
        return f"{base_prompt} {self._current_suffix}"

    def refine(self, prompt: str, response: str, judgment: int, step: int) -> str:
        if judgment == 1:
            return prompt  # already succeeded

        # Update suffix via one gradient step
        self._current_suffix = self._gradient_step(step)
        return f"{self._base_prompt} {self._current_suffix}"

    def _gradient_step(self, step: int) -> str:
        """
        Perform one GCG gradient step.

        When EasyJailbreak is installed and a compatible model is provided,
        delegates to its GCG implementation. Otherwise returns the current suffix
        (no-op fallback for testing without GPU).
        """
        if self._target_model is None:
            return self._current_suffix

        try:
            from easyjailbreak.attacker import GCG  # type: ignore

            # Initialize EasyJailbreak GCG on first step
            if self._gcg_attacker is None:
                self._gcg_attacker = GCG(
                    attack_model=self._target_model._model,
                    tokenizer=self._target_model._tokenizer,
                    num_steps=1,
                )
            updated = self._gcg_attacker.step(self._base_prompt, self._current_suffix)
            return updated
        except (ImportError, AttributeError):
            logger.warning(
                "EasyJailbreak GCG not available or incompatible API. "
                "Using static suffix (no gradient updates)."
            )
            return self._current_suffix
