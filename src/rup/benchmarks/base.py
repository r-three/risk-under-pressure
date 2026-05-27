"""Abstract base class for safety benchmarks."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BenchmarkPrompt:
    prompt_id: str
    text: str           # the harmful behavior / base prompt
    category: str       # harm category
    source: str         # benchmark name


class Benchmark(ABC):
    """Abstract benchmark that yields a list of harmful prompts."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def _load_all(self) -> List[BenchmarkPrompt]: ...

    def load(
        self,
        n: Optional[int] = None,
        seed: int = 42,
        categories: Optional[List[str]] = None,
    ) -> List[BenchmarkPrompt]:
        """Load up to n prompts, optionally filtered by category, reproducibly sampled."""
        prompts = self._load_all()
        if categories:
            prompts = [p for p in prompts if p.category in categories]
        if n is not None and n < len(prompts):
            rng = random.Random(seed)
            prompts = rng.sample(prompts, n)
        return prompts
