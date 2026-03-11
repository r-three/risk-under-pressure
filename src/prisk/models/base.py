"""Abstract base class for target models."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Minimal interface that all target models must implement."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Canonical identifier for this model (e.g. 'qwen2.5-7b-instruct')."""
        ...

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Single-turn generation. Returns response text only.
        The model has no memory of previous calls.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.model_id!r})"
