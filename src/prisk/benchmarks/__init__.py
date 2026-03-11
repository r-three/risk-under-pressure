from .base import Benchmark, BenchmarkPrompt
from .jailbreakbench import JailbreakBench
from .harmbench import HarmBench

_REGISTRY = {
    "jailbreakbench": JailbreakBench,
    "harmbench": HarmBench,
}


def get_benchmark(name: str, **kwargs) -> Benchmark:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown benchmark '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


__all__ = ["Benchmark", "BenchmarkPrompt", "JailbreakBench", "HarmBench", "get_benchmark"]
