"""Warstwa zgodności dla historycznego silnika backtestowego."""

from __future__ import annotations

from typing import Sequence

from KryptoLowca.exchange_manager import ExchangeManager
from bot_core.backtest.trend_following import (
    DEFAULT_FEE,
    MIN_SL_PCT,
    BacktestConfig,
    EntryParams,
    ExchangeLike,
    ExitParams,
    StrategyParams,
    TradeFill,
    TradeRecord,
    TrendBacktestEngine,
)

__all__ = [
    "DEFAULT_FEE",
    "MIN_SL_PCT",
    "BacktestConfig",
    "BacktestEngine",
    "EntryParams",
    "ExitParams",
    "StrategyParams",
    "TradeFill",
    "TradeRecord",
]


class _ExchangeManagerAdapter(ExchangeLike):
    """Adapter udostępniający API ExchangeManagera dla natywnego silnika."""

    def __init__(self, manager: ExchangeManager) -> None:
        self._manager = manager

    def load_markets(self) -> None:  # pragma: no cover - delegacja bez logiki
        self._manager.load_markets()

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, *, limit: int = 3000
    ) -> Sequence[Sequence[float]]:
        return self._manager.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit) or []

    def quantize_amount(self, symbol: str, amount: float) -> float:
        return float(self._manager.quantize_amount(symbol, amount))

    def min_notional(self, symbol: str) -> float | None:
        raw = self._manager.min_notional(symbol)
        return float(raw) if raw is not None else None


class BacktestEngine(TrendBacktestEngine):
    """Zachowuje API legacy, delegując do natywnego silnika trend-following."""

    def __init__(self, ex: ExchangeManager | None = None) -> None:
        manager = ex or ExchangeManager(exchange_id="binance")
        self.ex = manager
        super().__init__(_ExchangeManagerAdapter(manager))

