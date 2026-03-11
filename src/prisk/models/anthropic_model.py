"""Anthropic API model wrapper."""

from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from ..utils.config import ModelConfig
from ..utils.logging import get_logger
from .base import BaseModel

logger = get_logger(__name__)


class AnthropicModel(BaseModel):
    """Wraps the Anthropic messages API."""

    def __init__(self, config: ModelConfig):
        self._config = config
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise ImportError("Install 'anthropic': pip install anthropic>=0.28.0") from e
        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    @property
    def model_id(self) -> str:
        return self._config.model_id

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
    def generate(self, prompt: str, **kwargs) -> str:
        gen = self._config.generation
        response = self._client.messages.create(
            model=self._config.hf_name or self._config.model_id,
            max_tokens=kwargs.get("max_new_tokens", gen.max_new_tokens),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
