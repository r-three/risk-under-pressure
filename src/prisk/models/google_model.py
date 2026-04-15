"""Google Generative AI model wrapper (google-genai SDK)."""

from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from ..utils.config import ModelConfig
from ..utils.logging import get_logger
from .base import BaseModel

logger = get_logger(__name__)


class GoogleModel(BaseModel):
    """Wraps the Google Generative AI (Gemini) API via the google-genai SDK."""

    def __init__(self, config: ModelConfig):
        self._config = config
        try:
            from google import genai  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Install 'google-genai': pip install google-genai"
            ) from e
        self._client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        self._model_name = self._config.hf_name or self._config.model_id

    @property
    def model_id(self) -> str:
        return self._config.model_id

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
    def generate(self, prompt: str, **kwargs) -> str:
        from google.genai import types  # type: ignore

        gen = self._config.generation
        system_prompt = kwargs.get("system_prompt", None)
        config_kwargs = dict(
            max_output_tokens=kwargs.get("max_new_tokens", gen.max_new_tokens),
            temperature=kwargs.get("temperature", gen.temperature),
        )
        if system_prompt is not None:
            config_kwargs["system_instruction"] = system_prompt
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return response.text.strip()
