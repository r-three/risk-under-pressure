"""JailbreakBench benchmark loader."""

from __future__ import annotations

import csv
import urllib.request
from pathlib import Path
from typing import List

from .base import Benchmark, BenchmarkPrompt

_BEHAVIORS_URL = (
    "https://raw.githubusercontent.com/JailbreakBench/jailbreakbench/v1.0.0/"
    "src/jailbreakbench/data/behaviors.csv"
)


class JailbreakBench(Benchmark):
    """Loads adversarial behaviors from JailbreakBench."""

    def __init__(self, cache_dir: str | Path = "outputs/cache/jailbreakbench"):
        self._cache_dir = Path(cache_dir)

    @property
    def name(self) -> str:
        return "jailbreakbench"

    def _load_all(self) -> List[BenchmarkPrompt]:
        try:
            return self._load_via_package()
        except ImportError:
            return self._load_via_csv()

    def _load_via_package(self) -> List[BenchmarkPrompt]:
        import importlib.util

        spec = importlib.util.find_spec("jailbreakbench")
        if spec is None or spec.origin is None:
            raise ImportError("jailbreakbench not installed")

        # Load directly (not via the package wrapper) to retain the Index column,
        # which the wrapper drops before exposing the dataset.
        from datasets import load_dataset
        dataset = load_dataset("dedeswim/JBB-Behaviors", "behaviors", split="harmful")

        prompts = []
        for row in dataset:
            behavior_id = row["Behavior"].lower().replace(" ", "_")[:50]
            prompts.append(BenchmarkPrompt(
                prompt_id=f"jbb_{row['Index']:04d}_{behavior_id}",
                text=row["Goal"],
                category=row["Category"],
                source="jailbreakbench",
            ))
        return prompts

    def _load_via_csv(self) -> List[BenchmarkPrompt]:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self._cache_dir / "behaviors.csv"
        if not cache_file.exists():
            urllib.request.urlretrieve(_BEHAVIORS_URL, cache_file)

        prompts = []
        with open(cache_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompts.append(BenchmarkPrompt(
                    prompt_id=f"jbb_{len(prompts):04d}_{row.get('BehaviorID', len(prompts))}",
                    text=row.get("Behavior", ""),
                    category=row.get("SemanticCategory", "unknown"),
                    source="jailbreakbench",
                ))
        return prompts
