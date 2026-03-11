"""Model factory with singleton caching."""

from __future__ import annotations

from typing import Dict

from ..utils.config import ModelConfig
from .base import BaseModel

_CACHE: Dict[str, BaseModel] = {}


def load_model(config: ModelConfig) -> BaseModel:
    """Load a model from config, returning a cached instance if already loaded."""
    if config.model_id in _CACHE:
        return _CACHE[config.model_id]

    model: BaseModel
    if config.backend == "huggingface":
        from .hf_model import HFModel
        model = HFModel(config)
    elif config.backend == "openai":
        from .openai_model import OpenAIModel
        model = OpenAIModel(config)
    elif config.backend == "anthropic":
        from .anthropic_model import AnthropicModel
        model = AnthropicModel(config)
    elif config.backend == "google":
        from .google_model import GoogleModel
        model = GoogleModel(config)
    else:
        raise ValueError(f"Unknown backend: {config.backend!r}")

    _CACHE[config.model_id] = model
    return model


def clear_cache() -> None:
    """Release all cached models (useful for testing or memory management)."""
    _CACHE.clear()
