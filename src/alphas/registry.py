from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class AlphaSpec:
    name: str
    import_path: str  # e.g. "src.alphas.trend_momentum:trend_momentum"
    version: str
    default_params: Dict[str, Any]


class AlphaRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, AlphaSpec] = {}

    def register(self, spec: AlphaSpec) -> None:
        self._registry[spec.name] = spec

    def unregister(self, name: str) -> None:
        self._registry.pop(name, None)

    def resolve(self, name: str) -> Callable[..., Any]:
        spec = self._registry.get(name)
        if not spec:
            raise KeyError(f"Alpha '{name}' not found")
        module_name, func_name = spec.import_path.split(":", 1)
        module = importlib.import_module(module_name)
        fn = getattr(module, func_name)
        return fn

    def list(self) -> Dict[str, AlphaSpec]:
        return dict(self._registry)


