"""OpenAI API model wrapper."""

from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from ..utils.config import ModelConfig
from ..utils.logging import get_logger
from .base import BaseModel

logger = get_logger(__name__)


class OpenAIModel(BaseModel):
    """Wraps the OpenAI chat completions API."""

    def __init__(self, config: ModelConfig):
        self._config = config
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError("Install 'openai': pip install openai>=1.30.0") from e
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    @property
    def model_id(self) -> str:
        return self._config.model_id

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
    def generate(self, prompt: str, **kwargs) -> str:
        gen = self._config.generation
        system_prompt = kwargs.get("system_prompt", None)
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(
            model=self._config.hf_name or self._config.model_id,
            messages=messages,
            max_tokens=kwargs.get("max_new_tokens", gen.max_new_tokens),
            temperature=kwargs.get("temperature", gen.temperature),
        )
        return response.choices[0].message.content.strip()
