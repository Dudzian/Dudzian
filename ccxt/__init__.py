"""Minimalny stub biblioteki ccxt na potrzeby testów jednostkowych."""
from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace
from typing import Any, Dict, Optional


class AuthenticationError(Exception):
    """Zastępczy wyjątek ccxt AuthenticationError."""


base = SimpleNamespace(errors=SimpleNamespace(AuthenticationError=AuthenticationError))


class _BaseExchange:
    """Minimalny kontener udostępniający parse_ticker dla testów kontraktowych."""

    def parse_ticker(self, data: Dict[str, Any], symbol: Optional[str] = None) -> Dict[str, Any]:
        bid = self._extract_bid(data)
        ask = self._extract_ask(data)
        last = self._extract_last(data)
        return {
            "symbol": symbol or data.get("symbol"),
            "bid": bid,
            "ask": ask,
            "last": last,
            "info": data,
        }

    def _extract_bid(self, data: Dict[str, Any]) -> float:
        for key in ("bid", "bidPrice", "bidPx", "bid1Price"):
            value = data.get(key)
            if value is not None:
                return float(value)
        return 0.0

    def _extract_ask(self, data: Dict[str, Any]) -> float:
        for key in ("ask", "askPrice", "askPx", "ask1Price"):
            value = data.get(key)
            if value is not None:
                return float(value)
        return 0.0

    def _extract_last(self, data: Dict[str, Any]) -> float:
        for key in ("last", "lastPrice", "close", "c"):
            value = data.get(key)
            if isinstance(value, (list, tuple)):
                value = value[0] if value else None
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return 0.0


class binance(_BaseExchange):
    pass


class bitstamp(_BaseExchange):
    pass


class bybit(_BaseExchange):
    pass


class okx(_BaseExchange):
    pass


class kraken(_BaseExchange):
    pass


class zonda(_BaseExchange):
    pass

__all__ = ["AuthenticationError", "base"]

# Public API dla prostych testów kontraktowych
__all__ += ["binance", "bitstamp", "bybit", "okx", "kraken", "zonda"]

# Ułatwienie dla testów odwołujących się do globalnego `ccxt` bez importu.
builtins.ccxt = sys.modules[__name__]
