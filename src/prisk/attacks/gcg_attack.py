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


class _EJModelAdapter:
    """Wraps HFModel as an EasyJailbreak WhiteBoxModelBase."""

    def __init__(self, hf_model):
        from easyjailbreak.models import WhiteBoxModelBase  # type: ignore
        super().__init__(hf_model._model, hf_model._tokenizer)
        self.model_name = "adapter"

        tokenizer = hf_model._tokenizer
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            msgs = [{"role": "user", "content": "{prompt}"}]
            prefix = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            self.format_str = prefix + "{response}"
        else:
            self.format_str = "### Instruction:\n{prompt}\n\n### Response:\n{response}"

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def embed_layer(self):
        return self.model.get_input_embeddings()

    @property
    def vocab_size(self):
        return self.model.config.vocab_size

    @property
    def bos_token_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def batch_encode(self, texts, **kwargs):
        kwargs.setdefault("return_tensors", "pt")
        kwargs.setdefault("padding", True)
        return self.tokenizer(texts, **kwargs)

    def batch_decode(self, token_ids, **kwargs):
        return self.tokenizer.batch_decode(token_ids, **kwargs)

    def tokenize(self, text, **kwargs):
        kwargs.setdefault("return_tensors", "pt")
        return self.tokenizer(text, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


# Patch _EJModelAdapter to inherit from WhiteBoxModelBase at import time
def _make_adapter_class():
    try:
        from easyjailbreak.models import WhiteBoxModelBase  # type: ignore

        class _Adapter(WhiteBoxModelBase):
            def __init__(self, hf_model):
                super().__init__(hf_model._model, hf_model._tokenizer)
                self.model_name = "adapter"

                tokenizer = hf_model._tokenizer
                if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
                    msgs = [{"role": "user", "content": "{prompt}"}]
                    prefix = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True
                    )
                    self.format_str = prefix + "{response}"
                else:
                    self.format_str = (
                        "### Instruction:\n{prompt}\n\n### Response:\n{response}"
                    )

            @property
            def device(self):
                return next(self.model.parameters()).device

            @property
            def embed_layer(self):
                return self.model.get_input_embeddings()

            @property
            def vocab_size(self):
                return self.model.config.vocab_size

            @property
            def bos_token_id(self):
                return self.tokenizer.bos_token_id

            @property
            def eos_token_id(self):
                return self.tokenizer.eos_token_id

            @property
            def pad_token_id(self):
                return self.tokenizer.pad_token_id

            @property
            def dtype(self):
                return next(self.model.parameters()).dtype

            def batch_encode(self, texts, **kwargs):
                kwargs.setdefault("return_tensors", "pt")
                kwargs.setdefault("padding", True)
                return self.tokenizer(texts, **kwargs)

            def batch_decode(self, token_ids, **kwargs):
                return self.tokenizer.batch_decode(token_ids, **kwargs)

            def tokenize(self, text, **kwargs):
                kwargs.setdefault("return_tensors", "pt")
                return self.tokenizer(text, **kwargs)

            def generate(self, *args, **kwargs):
                return self.model.generate(*args, **kwargs)

            def __call__(self, *args, **kwargs):
                return self.model(*args, **kwargs)

        return _Adapter
    except ImportError:
        return None


_EJModelAdapter = _make_adapter_class()


class GCGAttack(AttackPolicy):
    """
    GCG attack: appends an adversarial suffix that is iteratively optimized
    via greedy coordinate gradient descent.

    Uses EasyJailbreak's MutationTokenGradient and ReferenceLossSelector
    directly (one step per call) so Algorithm 1 controls the budget.

    NOTE: Requires a HuggingFace (local) model. Raises NotImplementedError
    for API-based models.
    """

    def __init__(self, config: AttackConfig, target_model=None):
        self._config = config
        self._target_model = target_model
        self._current_suffix: str = _GCG_SUFFIX_INIT
        self._base_prompt: Optional[str] = None
        self._ej_model = None   # _EJModelAdapter, initialized lazily
        self._mutator = None    # MutationTokenGradient
        self._selector = None   # ReferenceLossSelector

    @property
    def attack_id(self) -> str:
        return "gcg"

    def _check_model_compatible(self) -> None:
        if self._target_model is None:
            return
        from ..models.hf_model import HFModel
        if not isinstance(self._target_model, HFModel):
            raise NotImplementedError(
                "GCG is a white-box attack and requires a local HuggingFace model. "
                f"Got: {type(self._target_model).__name__}"
            )

    def initialize(self, base_prompt: str) -> str:
        self._check_model_compatible()
        self._base_prompt = base_prompt
        self._current_suffix = _GCG_SUFFIX_INIT
        return f"{base_prompt} {self._current_suffix}"

    def refine(self, prompt: str, response: str, judgment: int, step: int) -> str:
        if judgment == 1:
            return prompt  # already succeeded

        self._current_suffix = self._gradient_step(step)
        return f"{self._base_prompt} {self._current_suffix}"

    def _gradient_step(self, step: int) -> str:
        """
        Perform one GCG gradient step using EasyJailbreak's lower-level components:
        MutationTokenGradient generates candidates, ReferenceLossSelector picks the best.
        Falls back to the current suffix (no-op) on any error.
        """
        if self._target_model is None:
            return self._current_suffix

        try:
            from easyjailbreak.mutation.gradient.token_gradient import (  # type: ignore
                MutationTokenGradient,
            )
            from easyjailbreak.selector import ReferenceLossSelector  # type: ignore
            from easyjailbreak.datasets import JailbreakDataset, Instance  # type: ignore

            # Lazy init of adapter and EJ components
            if self._ej_model is None:
                if _EJModelAdapter is None:
                    raise ImportError("EasyJailbreak WhiteBoxModelBase not available")
                self._ej_model = _EJModelAdapter(self._target_model)
                self._mutator = MutationTokenGradient(
                    self._ej_model, num_turb_sample=128, top_k=256
                )
                self._selector = ReferenceLossSelector(self._ej_model)

            # Build EJ dataset; jailbreak_prompt uses {query} placeholder
            jailbreak_prompt = "{query} " + self._current_suffix
            instance = Instance(
                query=self._base_prompt,
                jailbreak_prompt=jailbreak_prompt,
                reference_responses=["Sure, here is"],
            )
            dataset = JailbreakDataset([instance])

            # One GCG step: mutate → select best candidate
            mutated = self._mutator(dataset)
            best = self._selector.select(mutated)

            # Extract updated suffix from jailbreak_prompt ("{query} <suffix>")
            new_jbp = best[0].jailbreak_prompt
            prefix = "{query} "
            if new_jbp.startswith(prefix):
                return new_jbp[len(prefix):]
            return new_jbp

        except Exception as e:
            logger.error(
                f"GCG gradient step failed ({type(e).__name__}): {e}. "
                "Using static suffix (no gradient updates). "
                "Common cause: model loaded with quantization (4bit/8bit) which disables "
                "gradient computation — load the model with quantization=none for GCG."
            )
            return self._current_suffix
