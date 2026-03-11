"""HarmBench benchmark loader."""

from __future__ import annotations

from pathlib import Path
from typing import List

from .base import Benchmark, BenchmarkPrompt

_HF_DATASET = "walledai/HarmBench"


class HarmBench(Benchmark):
    """Loads behaviors from HarmBench via HuggingFace datasets."""

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

        ds = load_dataset(_HF_DATASET, split="train", cache_dir=str(self._cache_dir))
        prompts = []
        for i, row in enumerate(ds):
            category = row.get("category") or row.get("FunctionalCategory") or "unknown"
            text = row.get("behavior") or row.get("Behavior") or row.get("prompt") or ""
            prompts.append(BenchmarkPrompt(
                prompt_id=f"hb_{i:04d}",
                text=text,
                category=category,
                source="harmbench",
            ))
        return prompts
