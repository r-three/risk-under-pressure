"""JailbreakBench benchmark loader."""

from __future__ import annotations

import csv
import urllib.request
from pathlib import Path
from typing import List

from .base import Benchmark, BenchmarkPrompt

_BEHAVIORS_URL = (
    "https://raw.githubusercontent.com/JailbreakBench/jailbreakbench/main/"
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
        import jailbreakbench as jbb  # type: ignore

        dataset = jbb.read_dataset()
        prompts = []
        for item in dataset.behaviors:
            prompts.append(BenchmarkPrompt(
                prompt_id=f"jbb_{item.BehaviorID}",
                text=item.Behavior,
                category=getattr(item, "SemanticCategory", "unknown"),
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
                    prompt_id=f"jbb_{row.get('BehaviorID', len(prompts))}",
                    text=row.get("Behavior", ""),
                    category=row.get("SemanticCategory", "unknown"),
                    source="jailbreakbench",
                ))
        return prompts
