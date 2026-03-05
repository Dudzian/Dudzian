"""Interfejsy strategii oraz kontrakty walk-forward."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence


@dataclass(slots=True)
class MarketSnapshot:
    """Minimalny zestaw danych przekazywany do strategii.

    Zawiera standardowe pola OHLCV wraz z opcjonalnymi wskaźnikami
    wyliczonymi wcześniej w łańcuchu przetwarzania. Wszystkie wartości
    powinny być w strefie czasu UTC, a ceny wyrażone w walucie kwotowanej
    instrumentu.
    """

    symbol: str
    timestamp: int  # unix epoch ms/s (ujednolicone po stronie loadera)
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    indicators: Mapping[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class SignalLeg:
    """Pojedyncza noga sygnału wielonogowego."""

    symbol: str
    side: str
    quantity: float | None = None
    exchange: str | None = None
    confidence: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class StrategySignal:
    """Rezultat działania strategii (np. sygnał wejścia/wyjścia)."""

    symbol: str
    side: str  # "BUY" / "SELL" / "FLAT" itp.
    confidence: float  # 0.0–1.0
    quantity: float | None = None
    intent: str = "single"
    legs: Sequence[SignalLeg] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.legs is None:
            object.__setattr__(self, "legs", tuple())
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        normalized_intent = (self.intent or "single").strip() or "single"
        object.__setattr__(self, "intent", normalized_intent)


class StrategyEngine(abc.ABC):
    """Bazowy interfejs silników strategii z ujednoliconym lifecycle."""

    def prepare(self, *, context: Any | None = None) -> None:
        """Przygotowuje strategię do pracy (np. wstrzyknięcie zależności)."""

    def warmup(self, history: Sequence[MarketSnapshot]) -> None:
        """Pozwala przygotować wskaźniki przed startem live/paper."""

        legacy = self.__class__.__dict__.get("warm_up")
        if legacy is not None and legacy is not StrategyEngine.warm_up:
            legacy(self, history)

    def decide(self, snapshot: MarketSnapshot) -> Sequence[StrategySignal]:
        """Przetwarza nowy pakiet danych i zwraca sygnały."""

        legacy = self.__class__.__dict__.get("on_data")
        if legacy is not None and legacy is not StrategyEngine.on_data:
            return legacy(self, snapshot)
        raise NotImplementedError("decide() must be implemented by strategy engine")

    def teardown(self) -> None:
        """Sprząta zasoby po zakończeniu pracy strategii."""

    # ------------------------------------------------------------------
    # Backward compatibility shims (on_data / warm_up)
    def on_data(self, snapshot: MarketSnapshot) -> Sequence[StrategySignal]:
        return self.decide(snapshot)

    def warm_up(self, history: Sequence[MarketSnapshot]) -> None:
        return self.warmup(history)


class BaseStrategy(StrategyEngine):
    """Wspólna baza dla silników strategii.

    Zapewnia standardowe hooki na integrację z risk/portfolio, przechowuje
    zależność na datafeed/runtime oraz metadane strategii (w tym wymagane
    dane rynkowe). Konkretne strategie implementują ujednolicony lifecycle
    ``prepare -> warmup -> decide -> teardown``.
    """

    def __init__(
        self,
        *,
        required_data: Sequence[str] | None = None,
        risk_hooks: Sequence[str] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self._datafeed: Any | None = None
        self.required_data = tuple(required_data or ())
        self.risk_hooks = tuple(risk_hooks or ())
        base_metadata: dict[str, object] = {
            "required_data": self.required_data,
            "risk_hooks": self.risk_hooks,
        }
        base_metadata.update(metadata or {})
        self.metadata: Mapping[str, object] = MappingProxyType(base_metadata)

    def bind_datafeed(self, datafeed: Any) -> None:
        """Podpina datafeed/runtime dostarczający snapshoty i wskaźniki."""

        self._datafeed = datafeed

    @property
    def datafeed(self) -> Any | None:  # pragma: no cover - getter
        return self._datafeed

    def before_risk_checks(self, signals: Sequence[StrategySignal]) -> Sequence[StrategySignal]:
        """Hook pozwalający na pre-processing sygnałów przed risk/portfolio."""

        return signals

    def after_risk_checks(self, signals: Sequence[StrategySignal]) -> Sequence[StrategySignal]:
        """Hook uruchamiany po risk/portfolio (np. do logowania/telemetrii)."""

        return signals


def ensure_positive_int(value: int, *, field: str, min_value: int = 1) -> int:
    value = int(value)
    if value < min_value:
        raise ValueError(f"{field} must be at least {min_value}")
    return value


def ensure_positive_float(value: float, *, field: str, allow_zero: bool = False) -> float:
    value = float(value)
    if allow_zero:
        if value < 0:
            raise ValueError(f"{field} must be non-negative")
    elif value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def ensure_ratio(
    value: float,
    *,
    field: str,
    lower: float = 0.0,
    upper: float = 1.0,
    inclusive_lower: bool = False,
    inclusive_upper: bool = False,
) -> float:
    value = float(value)
    lower_ok = value >= lower if inclusive_lower else value > lower
    upper_ok = value <= upper if inclusive_upper else value < upper
    if not (lower_ok and upper_ok):
        bound_repr = (
            f"[{lower:g}, {upper:g}]"
            if inclusive_lower or inclusive_upper
            else f"({lower:g}, {upper:g})"
        )
        raise ValueError(f"{field} must be in the range {bound_repr}")
    return value


def clamp_range(value: float, *, field: str, lower: float, upper: float) -> float:
    value = float(value)
    if not lower <= value <= upper:
        raise ValueError(f"{field} must be between {lower} and {upper}")
    return value


class WalkForwardOptimizer(Protocol):
    """Kontrakt dla optymalizatorów walk-forward."""

    def split(
        self, data: Sequence[MarketSnapshot]
    ) -> Sequence[tuple[Sequence[MarketSnapshot], Sequence[MarketSnapshot]]]: ...

    def select_parameters(self, in_sample: Sequence[MarketSnapshot]) -> Mapping[str, float]: ...


__all__ = [
    "MarketSnapshot",
    "SignalLeg",
    "StrategySignal",
    "BaseStrategy",
    "StrategyEngine",
    "WalkForwardOptimizer",
    "ensure_positive_int",
    "ensure_positive_float",
    "ensure_ratio",
    "clamp_range",
]
