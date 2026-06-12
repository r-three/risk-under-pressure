"""Model factory with singleton caching."""

from __future__ import annotations

from typing import Dict

from ..utils.config import ModelConfig
from .base import BaseModel
from .hf_model import HFModel

_CACHE: Dict[str, BaseModel] = {}


def load_model(config: ModelConfig) -> BaseModel:
    """Load a model from config, returning a cached instance if already loaded."""
    if config.model_id in _CACHE:
        return _CACHE[config.model_id]

    model: BaseModel = HFModel(config)
    _CACHE[config.model_id] = model
    return model


def clear_cache() -> None:
    """Release all cached models (useful for testing or memory management)."""
    _CACHE.clear()
