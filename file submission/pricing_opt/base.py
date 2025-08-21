"""
Shared abstract base‑class *and* the plugin registry.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict
from .data_types import OptimiserReturn


class BasePricingOptimiser(ABC):
    name: str = "base"
    _REGISTRY: Dict[str, type["BasePricingOptimiser"]] = {}

    # ------------------------------------------------------------------  
    # factory helpers
    # ------------------------------------------------------------------
    @classmethod
    def register(cls, subclass: type["BasePricingOptimiser"], *aliases: str) -> None:
        keys = aliases or (subclass.name,)
        for k in keys:
            cls._REGISTRY[k.lower()] = subclass

    @classmethod
    def from_mode(cls, mode: str, *args, **kwargs) -> "BasePricingOptimiser":
        try:
            sub = cls._REGISTRY[mode.lower()]
        except KeyError as e:
            raise KeyError(
                f"Unknown optimiser '{mode}'. "
                f"Available: {sorted(cls._REGISTRY)}"
            ) from e
        return sub(*args, **kwargs)

    # ------------------------------------------------------------------
    # interface every subclass must implement
    # ------------------------------------------------------------------
    @abstractmethod
    def solve(self, **kwargs) -> OptimiserReturn: ...
