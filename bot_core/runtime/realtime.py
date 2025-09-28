"""Pętla czasu rzeczywistego dla strategii dziennych."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Sequence

from bot_core.exchanges.base import OrderResult
from bot_core.runtime.controller import ControllerSignal, DailyTrendController, TradingController


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


_INTERVAL_SECONDS: dict[str, float] = {
    "1m": 60.0,
    "3m": 180.0,
    "5m": 300.0,
    "15m": 900.0,
    "30m": 1_800.0,
    "1h": 3_600.0,
    "2h": 7_200.0,
    "4h": 14_400.0,
    "6h": 21_600.0,
    "8h": 28_800.0,
    "12h": 43_200.0,
    "1d": 86_400.0,
    "3d": 259_200.0,
    "1w": 604_800.0,
    "1M": 2_592_000.0,
}


def _interval_seconds(interval: str, fallback: float) -> float:
    seconds = _INTERVAL_SECONDS.get(interval)
    if seconds is None:
        seconds = _INTERVAL_SECONDS.get(interval.lower())
    if seconds is None:
        seconds = fallback
    return max(fallback, seconds)


@dataclass(slots=True)
class DailyTrendRealtimeRunner:
    """Uruchamia cykl strategii dziennej i deleguje egzekucję do TradingController."""

    controller: DailyTrendController
    trading_controller: TradingController
    history_bars: int = 120
    clock: Callable[[], datetime] = _utc_now

    def run_once(self) -> list[OrderResult]:
        """Wykonuje pojedynczą iterację strategii w oparciu o aktualny czas."""

        now = self.clock()
        end_ms = int(now.timestamp() * 1000)
        interval_seconds = _interval_seconds(self.controller.interval, max(1.0, self.controller.tick_seconds))
        lookback_ms = int(interval_seconds * max(1, self.history_bars) * 1000)
        start_ms = max(0, end_ms - lookback_ms)

        collected: Sequence[ControllerSignal] = self.controller.collect_signals(start=start_ms, end=end_ms)
        signals = [item.signal for item in collected]

        if not signals:
            self.trading_controller.maybe_report_health()
            return []

        return self.trading_controller.process_signals(signals)


__all__ = ["DailyTrendRealtimeRunner"]
