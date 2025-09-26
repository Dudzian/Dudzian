"""Fundamenty silnika strategii w architekturze modułowej.

Moduł dostarcza bazowe klasy danych oraz abstrakcyjny interfejs strategii,
który będzie współdzielony przez marketplace presetów, moduł backtestingu oraz
silnik autotradingu. Zależymy jedynie od ``logging_utils`` (centralny logger)
i standardowych bibliotek, aby zapewnić możliwość wykorzystania w lekkich
testach jednostkowych.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Mapping, MutableMapping, Optional, Protocol

from KryptoLowca.logging_utils import get_logger

__all__ = [
    "StrategyError",
    "StrategyMetadata",
    "StrategyContext",
    "StrategySignal",
    "DataProvider",
    "BaseStrategy",
]


logger = get_logger(__name__)


class StrategyError(RuntimeError):
    """Wspólny typ wyjątków rzucanych przez strategie."""


@dataclass(slots=True)
class StrategyMetadata:
    """Opis strategii wyświetlany w marketplace i GUI."""

    name: str
    description: str
    author: str = "internal"
    version: str = "0.1.0"
    tags: tuple[str, ...] = ()
    exchanges: tuple[str, ...] = ("binance",)
    timeframes: tuple[str, ...] = ("1h",)
    risk_level: str = "balanced"


@dataclass(slots=True)
class StrategyContext:
    """Kontekst wywołania strategii przekazywany na każdym kroku."""

    symbol: str
    timeframe: str
    portfolio_value: float
    position: float
    timestamp: datetime
    metadata: StrategyMetadata
    extra: MutableMapping[str, Any] = field(default_factory=dict)

    def require_demo_mode(self) -> None:
        """Prosta walidacja – blokada live tradingu przed audytem."""

        if self.extra.get("mode", "demo") != "demo":
            raise StrategyError(
                "Strategia nie została jeszcze certyfikowana do trybu LIVE. "
                "Uruchom ją w środowisku paper trading przed przełączeniem."
            )


@dataclass(slots=True)
class StrategySignal:
    """Struktura sygnału zwracanego przez strategie."""

    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float = 0.0
    size: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def ensure_bounds(self) -> "StrategySignal":
        if not 0.0 <= self.confidence <= 1.0:
            raise StrategyError("confidence musi być w zakresie 0-1")
        if self.size is not None and self.size < 0:
            raise StrategyError("size nie może być ujemne")
        return self


class DataProvider(Protocol):
    """Minimalny interfejs wymagany do pobierania danych przez strategie."""

    async def get_ohlcv(self, symbol: str, timeframe: str, *, limit: int = 500) -> Mapping[str, Any]:
        ...

    async def get_ticker(self, symbol: str) -> Mapping[str, Any]:  # pragma: no cover - interfejs
        ...


class BaseStrategy(ABC):
    """Abstrakcyjna strategia – punkty rozszerzeń dla konkretnych presetów."""

    metadata: StrategyMetadata

    def __init__(self, metadata: StrategyMetadata | None = None) -> None:
        self.metadata = metadata or StrategyMetadata(
            name=self.__class__.__name__,
            description="Brak opisu",
        )
        self._logger = get_logger(f"{__name__}.{self.metadata.name}")
        self._prepared = False

    async def prepare(self, context: StrategyContext, data_provider: DataProvider) -> None:
        """Hook inicjalizacyjny uruchamiany przez silnik przed tradingiem."""

        if self._prepared:
            return
        context.require_demo_mode()
        await self._warmup(context, data_provider)
        self._prepared = True
        self._logger.info(
            "Strategia %s przygotowana (symbol=%s, timeframe=%s)",
            self.metadata.name,
            context.symbol,
            context.timeframe,
        )

    async def _warmup(self, context: StrategyContext, data_provider: DataProvider) -> None:
        """Domyślna implementacja pobiera historię do cache'u."""

        try:
            await data_provider.get_ohlcv(context.symbol, context.timeframe, limit=200)
        except Exception as exc:  # pragma: no cover - zależne od providerów
            self._logger.warning("Błąd pobierania danych podczas warm-up", exc_info=exc)
            raise StrategyError("Nie udało się pobrać danych historycznych") from exc

    async def handle_market_data(
        self,
        context: StrategyContext,
        market_payload: Mapping[str, Any],
    ) -> StrategySignal:
        """Główny punkt wejścia – deleguje generowanie sygnału."""

        if not self._prepared:
            raise StrategyError("Strategia nie została przygotowana – wywołaj prepare()")
        try:
            signal = await self.generate_signal(context, market_payload)
            return signal.ensure_bounds()
        except StrategyError:
            raise
        except Exception as exc:  # pragma: no cover - defensywna warstwa
            self._logger.exception("Nieoczekiwany błąd strategii")
            raise StrategyError("Strategia zgłosiła nieobsługiwany wyjątek") from exc

    @abstractmethod
    async def generate_signal(
        self,
        context: StrategyContext,
        market_payload: Mapping[str, Any],
    ) -> StrategySignal:
        """Metoda implementowana przez konkretne strategie."""

    async def notify_fill(self, context: StrategyContext, fill_data: Mapping[str, Any]) -> None:
        """Hook pozwalający zareagować na wykonane zlecenie."""

        self._logger.debug("Fill data: %s", dict(fill_data))

    async def shutdown(self) -> None:
        """Umożliwia zwolnienie zasobów (np. sesji HTTP)."""

        self._prepared = False
        self._logger.info("Strategia %s zamknięta", self.metadata.name)
