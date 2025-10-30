"""Wspólne testy zdrowia dla adapterów giełdowych."""
from __future__ import annotations

import time
from typing import Mapping, Sequence

from bot_core.exchanges.base import ExchangeAdapter
from bot_core.exchanges.health import HealthCheck


def build_standard_health_checks(
    adapter: ExchangeAdapter,
    *,
    sample_symbol: str | None = None,
    ohlcv_interval: str = "1m",
    clock: callable = time.time,
    max_clock_drift: float = 300.0,
) -> Sequence[HealthCheck]:
    """Buduje zestaw testów zdrowia dla adaptera."""

    def _connectivity() -> Mapping[str, object]:
        symbols = tuple(adapter.fetch_symbols())
        if not symbols:
            raise RuntimeError("Giełda nie zwróciła żadnych symboli")
        return {"symbols": len(symbols)}

    def _authorization() -> Mapping[str, object]:
        snapshot = adapter.fetch_account_snapshot()
        if snapshot.total_equity < 0:
            raise RuntimeError("Saldo konta jest ujemne – podejrzenie błędu")
        return {
            "total_equity": snapshot.total_equity,
            "available_margin": snapshot.available_margin,
        }

    def _clock_sync() -> Mapping[str, object]:
        symbols = tuple(adapter.fetch_symbols())
        if not symbols:
            raise RuntimeError("Brak symbolu referencyjnego do health-checku")
        symbol = sample_symbol or symbols[0]
        candles = adapter.fetch_ohlcv(symbol, ohlcv_interval, limit=1)
        if not candles:
            raise RuntimeError("Brak świec OHLCV w health-checku")
        candle = candles[-1]
        timestamp = float(candle[0])
        if timestamp > 1e12:
            timestamp /= 1000.0
        drift = abs(clock() - timestamp)
        if drift > max_clock_drift:
            raise RuntimeError(f"Dryf zegara przekracza dopuszczalne {max_clock_drift}s")
        return {"clock_drift": drift, "symbol": symbol}

    return (
        HealthCheck(name="connectivity", check=_connectivity, critical=True),
        HealthCheck(name="authorization", check=_authorization, critical=True),
        HealthCheck(name="clock_sync", check=_clock_sync, critical=False),
    )


__all__ = ["build_standard_health_checks"]
