"""Rejestr strategii wykorzystywany przez marketplace oraz silnik autotradingu."""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, Iterator, MutableMapping, Type

from .engine import BaseStrategy, StrategyMetadata

__all__ = ["StrategyRegistry", "registry", "strategy"]


class StrategyRegistry:
    """Łatwy w testach rejestr strategii."""

    def __init__(self) -> None:
        self._strategies: MutableMapping[str, Type[BaseStrategy]] = OrderedDict()

    def register(self, strategy_cls: Type[BaseStrategy]) -> Type[BaseStrategy]:
        key = strategy_cls.__name__.lower()
        self._strategies[key] = strategy_cls
        return strategy_cls

    def get(self, name: str) -> Type[BaseStrategy]:
        key = (name or "").strip().lower()
        try:
            return self._strategies[key]
        except KeyError as exc:  # pragma: no cover - defensywne logowanie
            raise KeyError(f"Strategia '{name}' nie jest zarejestrowana") from exc

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return item.strip().lower() in self._strategies
        if isinstance(item, type) and issubclass(item, BaseStrategy):
            return item.__name__.lower() in self._strategies
        return False

    def __iter__(self) -> Iterator[Type[BaseStrategy]]:
        return iter(self._strategies.values())

    def items(self) -> Iterable[tuple[str, Type[BaseStrategy]]]:
        return self._strategies.items()

    def metadata(self) -> Dict[str, StrategyMetadata]:
        return {
            name: getattr(cls, "metadata", StrategyMetadata(name=cls.__name__, description=""))
            for name, cls in self._strategies.items()
        }


def strategy(strategy_cls: Type[BaseStrategy]) -> Type[BaseStrategy]:
    """Dekorator ułatwiający rejestrację."""

    return registry.register(strategy_cls)


registry = StrategyRegistry()
