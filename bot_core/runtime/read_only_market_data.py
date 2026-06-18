"""Read-only market data contract for functional preview.

This module defines a local/static preview contract only. It never performs
network I/O, account/balance/order/credential access, file loading/export,
cloud telemetry, or runtime loop startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

from bot_core.runtime.preview_modes import (
    PreviewMode,
    PreviewModePolicy,
    RuntimeCapability,
    build_preview_mode_policy,
)


class ReadOnlyMarketDataError(ValueError):
    """Raised when read-only market data contract validation fails closed."""


@dataclass(frozen=True, slots=True)
class MarketQuote:
    """Immutable market quote with injected timestamp only."""

    symbol: str
    bid: float
    ask: float
    last: float
    timestamp: str


@dataclass(frozen=True, slots=True)
class MarketCandle:
    """Immutable OHLCV candle with injected timestamp only."""

    symbol: str
    timeframe: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class ReadOnlyMarketDataSnapshot:
    """Immutable deterministic snapshot of quotes keyed by sorted symbols."""

    symbols: tuple[str, ...]
    quotes: tuple[MarketQuote, ...]


@dataclass(frozen=True, slots=True)
class ReadOnlyMarketDataPolicy:
    """Preview gate proving that only read-only market fetch is enabled."""

    preview_policy: PreviewModePolicy

    def __post_init__(self) -> None:
        """Fail closed for manually constructed, untrusted preview policies."""

        self._validate()

    def _validate(self) -> None:
        """Re-validate policy contents when a policy object crosses a boundary."""

        if self.preview_policy.mode is not PreviewMode.READ_ONLY_MARKET:
            raise ReadOnlyMarketDataError(
                "read-only market data requires PreviewMode.READ_ONLY_MARKET"
            )
        if self.preview_policy.capabilities != (RuntimeCapability.READ_ONLY_MARKET_FETCH,):
            raise ReadOnlyMarketDataError(
                "read-only market data requires exactly READ_ONLY_MARKET_FETCH"
            )

    @classmethod
    def for_preview_mode(
        cls, mode: str | PreviewMode = PreviewMode.READ_ONLY_MARKET
    ) -> "ReadOnlyMarketDataPolicy":
        return cls(
            preview_policy=build_preview_mode_policy(
                mode,
                [RuntimeCapability.READ_ONLY_MARKET_FETCH],
            )
        )


@runtime_checkable
class ReadOnlyMarketDataProvider(Protocol):
    """Provider surface restricted to market-data reads only.

    Implementations must not expose account, balance, position, order, fill,
    credential, export, file I/O, cloud sink, network I/O, or runtime-loop methods.
    """

    def get_quote(self, symbol: str) -> MarketQuote: ...

    def get_candles(self, symbol: str, timeframe: str, limit: int) -> tuple[MarketCandle, ...]: ...

    def snapshot(self, symbols: tuple[str, ...]) -> ReadOnlyMarketDataSnapshot: ...


class InMemoryReadOnlyMarketDataProvider:
    """Deterministic local-only provider for unit contract proof.

    Data is supplied by the caller as in-memory immutable value objects. The provider
    does not read files/env/secrets, does not write files, does not use network or
    sockets, and does not start timers, workers, schedulers, or loops.
    """

    def __init__(
        self,
        *,
        quotes: Mapping[str, MarketQuote],
        candles: Mapping[tuple[str, str], tuple[MarketCandle, ...]],
        policy: ReadOnlyMarketDataPolicy | None = None,
    ) -> None:
        self.policy = policy or ReadOnlyMarketDataPolicy.for_preview_mode()
        self.policy._validate()
        self._quotes = {self._normalize_symbol(symbol): quote for symbol, quote in quotes.items()}
        self._candles = {
            (self._normalize_symbol(symbol), self._validate_timeframe(timeframe)): tuple(items)
            for (symbol, timeframe), items in candles.items()
        }
        for symbol, quote in self._quotes.items():
            if quote.symbol != symbol:
                raise ReadOnlyMarketDataError("quote symbol must match mapping key")
        for (symbol, timeframe), items in self._candles.items():
            for candle in items:
                if candle.symbol != symbol or candle.timeframe != timeframe:
                    raise ReadOnlyMarketDataError("candle symbol/timeframe must match mapping key")

    def get_quote(self, symbol: str) -> MarketQuote:
        normalized = self._normalize_symbol(symbol)
        try:
            return self._quotes[normalized]
        except KeyError as exc:
            raise ReadOnlyMarketDataError(f"unknown market data symbol: {normalized}") from exc

    def get_candles(self, symbol: str, timeframe: str, limit: int) -> tuple[MarketCandle, ...]:
        normalized = self._normalize_symbol(symbol)
        normalized_timeframe = self._validate_timeframe(timeframe)
        if limit <= 0:
            raise ReadOnlyMarketDataError("candle limit must be positive")
        try:
            values = self._candles[(normalized, normalized_timeframe)]
        except KeyError as exc:
            raise ReadOnlyMarketDataError(
                f"missing candles for {normalized} {normalized_timeframe}"
            ) from exc
        return values[:limit]

    def snapshot(self, symbols: tuple[str, ...]) -> ReadOnlyMarketDataSnapshot:
        normalized_symbols = tuple(sorted(self._normalize_symbol(symbol) for symbol in symbols))
        return ReadOnlyMarketDataSnapshot(
            symbols=normalized_symbols,
            quotes=tuple(self.get_quote(symbol) for symbol in normalized_symbols),
        )

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        normalized = symbol.strip().upper()
        if not normalized:
            raise ReadOnlyMarketDataError("symbol must be non-empty")
        return normalized

    @staticmethod
    def _validate_timeframe(timeframe: str) -> str:
        normalized = timeframe.strip()
        if not normalized:
            raise ReadOnlyMarketDataError("timeframe must be non-empty")
        return normalized


__all__ = [
    "InMemoryReadOnlyMarketDataProvider",
    "MarketCandle",
    "MarketQuote",
    "ReadOnlyMarketDataError",
    "ReadOnlyMarketDataPolicy",
    "ReadOnlyMarketDataProvider",
    "ReadOnlyMarketDataSnapshot",
]
