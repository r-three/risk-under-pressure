"""HarmBench benchmark loader."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import List

from .base import Benchmark, BenchmarkPrompt

_HF_DATASET = "walledai/HarmBench"
# Deterministic UUID namespace so the same behavior always gets the same ID
_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


class HarmBench(Benchmark):
    """Loads behaviors from the HarmBench standard subset (train split)."""

    def __init__(self, cache_dir: str | Path = "outputs/cache/harmbench"):
        self._cache_dir = Path(cache_dir)

    @property
    def name(self) -> str:
        return "harmbench"

    def _load_all(self) -> List[BenchmarkPrompt]:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:
            raise ImportError("Install 'datasets' to use HarmBench: pip install datasets") from e

        ds = load_dataset(_HF_DATASET, "standard", split="train", cache_dir=str(self._cache_dir))
        prompts = []
        for row in ds:
            text = row.get("Behavior") or ""
            category = row.get("SemanticCategory") or "unknown"
            prompt_id = str(uuid.uuid5(_NAMESPACE, text))
            prompts.append(BenchmarkPrompt(
                prompt_id=prompt_id,
                text=text,
                category=category,
                source="harmbench",
            ))
        return prompts
