"""Interfejsy strategii oraz kontrakty walk-forward."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence


@dataclass(slots=True)
class MarketSnapshot:
    """Minimalny zestaw danych przekazywany do strategii.

    Zawiera standardowe pola OHLCV wraz z opcjonalnymi wskaźnikami
    wyliczonymi wcześniej w łańcuchu przetwarzania. Wszystkie wartości
    powinny być w strefie czasu UTC, a ceny wyrażone w walucie kwotowanej
    instrumentu.
    """

    symbol: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    indicators: Mapping[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class StrategySignal:
    """Rezultat działania strategii (np. sygnał wejścia/wyjścia)."""

    symbol: str
    side: str
    confidence: float
    metadata: Mapping[str, float]


class StrategyEngine(abc.ABC):
    """Bazowy interfejs silników strategii."""

    @abc.abstractmethod
    def on_data(self, snapshot: MarketSnapshot) -> Sequence[StrategySignal]:
        """Przetwarza nowy pakiet danych i zwraca sygnały."""

    @abc.abstractmethod
    def warm_up(self, history: Sequence[MarketSnapshot]) -> None:
        """Pozwala przygotować wskaźniki przed startem live/paper."""


class WalkForwardOptimizer(Protocol):
    """Kontrakt dla optymalizatorów walk-forward."""

    def split(
        self, data: Sequence[MarketSnapshot]
    ) -> Sequence[tuple[Sequence[MarketSnapshot], Sequence[MarketSnapshot]]]:
        ...

    def select_parameters(self, in_sample: Sequence[MarketSnapshot]) -> Mapping[str, float]:
        ...


__all__ = [
    "MarketSnapshot",
    "StrategySignal",
    "StrategyEngine",
    "WalkForwardOptimizer",
]
