"""Google Generative AI model wrapper."""

from __future__ import annotations

import os

from tenacity import retry, stop_after_attempt, wait_exponential

from ..utils.config import ModelConfig
from ..utils.logging import get_logger
from .base import BaseModel

logger = get_logger(__name__)


class GoogleModel(BaseModel):
    """Wraps the Google Generative AI (Gemini) API."""

    def __init__(self, config: ModelConfig):
        self._config = config
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Install 'google-generativeai': pip install google-generativeai>=0.5.0"
            ) from e
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self._client = genai.GenerativeModel(self._config.hf_name or self._config.model_id)

    @property
    def model_id(self) -> str:
        return self._config.model_id

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
    def generate(self, prompt: str, **kwargs) -> str:
        import google.generativeai as genai  # type: ignore

        gen = self._config.generation
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=kwargs.get("max_new_tokens", gen.max_new_tokens),
            temperature=kwargs.get("temperature", gen.temperature),
        )
        response = self._client.generate_content(prompt, generation_config=generation_config)
        return response.text.strip()
