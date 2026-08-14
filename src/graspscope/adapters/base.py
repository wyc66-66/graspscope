from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from graspscope.schema import Prediction, Sample

_REGISTRY: dict[str, type[Adapter]] = {}


class Adapter(ABC):
    name: str = "base"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def predict(self, samples: list[Sample], vocab: list[str]) -> list[Prediction]:
        raise NotImplementedError


def register_adapter(cls: type[Adapter]) -> type[Adapter]:
    _REGISTRY[cls.name] = cls
    return cls


def get_adapter(name: str, config: dict[str, Any] | None = None) -> Adapter:
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown adapter '{name}'. Known: {known}")
    return _REGISTRY[name](config)


def list_adapters() -> list[str]:
    return sorted(_REGISTRY)
