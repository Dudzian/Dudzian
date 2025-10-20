"""Implementacje dostawców danych zgodne z protokołem ``DataProvider``."""
from __future__ import annotations

import inspect
from typing import Any, Mapping

from KryptoLowca.logging_utils import get_logger
from KryptoLowca.strategies.base import DataProvider

logger = get_logger(__name__)


class ExchangeDataProvider(DataProvider):
    """Adapter wykorzystujący ``ExchangeManager`` do pobierania danych rynkowych."""

    def __init__(self, exchange_manager: Any, *, default_limit: int = 500) -> None:
        self._exchange_manager = exchange_manager
        self._default_limit = int(max(1, default_limit))

    async def get_ohlcv(self, symbol: str, timeframe: str, *, limit: int = 500) -> Mapping[str, Any]:
        candles_limit = int(limit or self._default_limit)
        fetch = getattr(self._exchange_manager, "fetch_ohlcv", None)
        if not callable(fetch):
            raise AttributeError("exchange_manager nie udostępnia metody fetch_ohlcv")

        try:
            candles = fetch(symbol, timeframe, limit=candles_limit)
            if inspect.isawaitable(candles):
                candles = await candles
        except Exception as exc:  # pragma: no cover - log defensywny
            logger.exception("ExchangeDataProvider.get_ohlcv failed for %s %s", symbol, timeframe)
            raise
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "limit": candles_limit,
            "candles": list(candles) if candles is not None else [],
        }

    async def get_ticker(self, symbol: str) -> Mapping[str, Any]:
        ticker: Any = None
        # najpierw próbujemy natywnej metody menedżera
        fetch_ticker = getattr(self._exchange_manager, "fetch_ticker", None)
        if callable(fetch_ticker):
            try:
                ticker = fetch_ticker(symbol)
                if inspect.isawaitable(ticker):
                    ticker = await ticker
            except Exception as exc:  # pragma: no cover - log defensywny
                logger.exception("ExchangeDataProvider.get_ticker failed for %s", symbol)
                raise

        if ticker is None:
            exchange = getattr(self._exchange_manager, "exchange", None)
            if exchange is None:
                return {}
            fetch_ticker = getattr(exchange, "fetch_ticker", None)
            if not callable(fetch_ticker):
                return {}
        try:
            ticker = fetch_ticker(symbol)
            if inspect.isawaitable(ticker):
                ticker = await ticker
        except Exception as exc:  # pragma: no cover - log defensywny
            logger.exception("ExchangeDataProvider.get_ticker failed for %s", symbol)
            raise

        if ticker is None:
            return {}
        if isinstance(ticker, Mapping):
            return ticker
        if isinstance(ticker, dict):  # pragma: no cover - zachowanie defensywne
            return ticker
        try:
            return dict(ticker)
        except Exception:  # pragma: no cover - fallback
            return {"symbol": symbol, "last": float(ticker) if isinstance(ticker, (int, float)) else None}


__all__ = ["ExchangeDataProvider"]

