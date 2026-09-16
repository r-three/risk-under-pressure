"""Model factory with singleton caching."""

from __future__ import annotations

from typing import Dict

from ..utils.config import ModelConfig
from .base import BaseModel
from .hf_model import HFModel

_CACHE: Dict[str, BaseModel] = {}


def _cache_key(config: ModelConfig) -> str:
    """Cache identity is the whole config, not just model_id.

    Two config files can legitimately name the same checkpoint and still need separate
    instances: gemma3_4b_it.yaml (target) and gemma3_4b_it_judge.yaml (judge) both have
    model_id `gemma3-4b-it` but load it through different loaders with different generation
    caps. Keying on model_id alone would hand the target whichever one loaded first — a
    64-token multimodal judge standing in for the model under test, with no error raised
    and a plausible-looking risk curve at the end of it.

    Identical configs still serialise identically and share one instance, so the common
    case (qwen2.5_7b as both a target and the PAIR attacker) is unaffected.
    """
    return config.model_dump_json()


def load_model(config: ModelConfig) -> BaseModel:
    """Load a model from config, returning a cached instance if already loaded."""
    key = _cache_key(config)
    if key in _CACHE:
        return _CACHE[key]

    model: BaseModel = HFModel(config)
    _CACHE[key] = model
    return model


def clear_cache() -> None:
    """Release all cached models (useful for testing or memory management)."""
    _CACHE.clear()
