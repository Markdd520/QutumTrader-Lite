# qtlite/core/registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, TypeVar, Any

T = TypeVar("T")


@dataclass
class RegisteredItem:
    """Metadata for registered objects."""
    name: str
    cls_or_func: Callable[..., Any]
    category: str | None = None
    desc: str | None = None


class Registry:
    """
    Simple name → object registry with decorator support.

    用来注册：
    - 因子
    - 合成器
    - 策略
    - 风控模块
    """

    def __init__(self, name: str):
        self.name = name
        self._items: Dict[str, RegisteredItem] = {}

    def register(self, name: str | None = None,
                 category: str | None = None,
                 desc: str | None = None) -> Callable[[T], T]:
        """
        Decorator to register a class or function.

        Example
        -------
        @factor_registry.register(category="volatility", desc="Upward volatility")
        class UpVol(Factor):
            ...
        """

        def _decorator(obj: T) -> T:
            key = name or obj.__name__
            if key in self._items:
                raise ValueError(f"{self.name}: duplicate key '{key}'")
            self._items[key] = RegisteredItem(
                name=key,
                cls_or_func=obj,  # type: ignore[arg-type]
                category=category,
                desc=desc,
            )
            return obj

        return _decorator

    def get(self, name: str) -> Callable[..., Any]:
        return self._items[name].cls_or_func

    def items(self) -> Dict[str, RegisteredItem]:
        return dict(self._items)


# global registries (can be imported elsewhere)
factor_registry = Registry("factor_registry")
combiner_registry = Registry("combiner_registry")
strategy_registry = Registry("strategy_registry")
risk_registry = Registry("risk_registry")
