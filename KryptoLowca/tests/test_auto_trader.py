"""Tests for safety guards in AutoTrader."""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Tuple

import pytest

from KryptoLowca.auto_trader import AutoTrader


class DummyEmitter:
    def __init__(self) -> None:
        self.logs: List[Tuple[str, str, str]] = []

    def on(self, *_, **__) -> None:  # pragma: no cover - interface placeholder
        return None

    def off(self, *_, **__) -> None:  # pragma: no cover - interface placeholder
        return None

    def emit(self, event: str, **payload: Any) -> None:
        self.logs.append(("event", event, str(payload)))

    def log(self, message: str, level: str = "INFO", component: str | None = None) -> None:
        self.logs.append(("log", level, message))


class DummyVar:
    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        return self._value


class NoSignalAI:
    ai_threshold_bps = 5.0

    def predict_series(self, *_, **__) -> None:
        return None


class DummyGUI:
    def __init__(self, demo: bool, allow_live: bool) -> None:
        self.timeframe_var = DummyVar("1m")
        self.ai_mgr = NoSignalAI()
        self.ex_mgr = self
        self._demo = demo
        self._allow_live = allow_live
        self.executed: List[Tuple[str, str, float]] = []
        self.db = DummyDB()

    def is_demo_mode_active(self) -> bool:
        return self._demo

    def is_live_trading_allowed(self) -> bool:
        return self._allow_live

    def _bridge_execute_trade(self, symbol: str, side: str, price: float) -> None:
        self.executed.append((symbol, side, price))

    # Exchange-like API -------------------------------------------------
    def fetch_ticker(self, symbol: str) -> Dict[str, float]:
        return {"last": 100.0}


class DummyDB:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        self.sync = self

    def log_performance_metric(self, payload: Dict[str, Any]) -> int:
        self.records.append(payload)
        return len(self.records)


@pytest.fixture()
def demo_autotrader() -> Callable[[DummyGUI], AutoTrader]:
    def factory(gui: DummyGUI) -> AutoTrader:
        emitter = DummyEmitter()
        trader = AutoTrader(emitter, gui, lambda: "BTC/USDT", auto_trade_interval_s=0.01)
        trader.enable_auto_trade = True
        return trader

    return factory


def _run_loop(trader: AutoTrader, duration: float = 0.1) -> None:
    worker = threading.Thread(target=trader._auto_trade_loop, daemon=True)
    worker.start()
    time.sleep(duration)
    trader._stop.set()
    worker.join(timeout=1.0)
    trader.stop()


def test_no_fallback_trade_without_signal(demo_autotrader: Callable[[DummyGUI], AutoTrader]) -> None:
    gui = DummyGUI(demo=True, allow_live=True)
    trader = demo_autotrader(gui)
    _run_loop(trader)
    assert gui.executed == []
    assert any("no valid model signal" in msg for kind, _, msg in trader.emitter.logs if kind == "log")


def test_live_trading_blocked_without_confirmation(demo_autotrader: Callable[[DummyGUI], AutoTrader]) -> None:
    gui = DummyGUI(demo=False, allow_live=False)
    trader = demo_autotrader(gui)
    _run_loop(trader)
    assert gui.executed == []
    assert any("live trading requires explicit confirmation" in msg for kind, _, msg in trader.emitter.logs if kind == "log")


def test_metrics_persisted_on_trade_close(demo_autotrader: Callable[[DummyGUI], AutoTrader]) -> None:
    gui = DummyGUI(demo=True, allow_live=True)
    trader = demo_autotrader(gui)
    trader._on_trade_closed("BTC/USDT", "BUY", 100.0, 110.0, 5.0, time.time())
    metrics = gui.db.records
    assert metrics, "Brak zapisanych metryk po zamknięciu transakcji"
    names = {entry["metric"] for entry in metrics}
    assert "auto_trader_expectancy" in names
    assert any(entry["symbol"] == "BTC/USDT" for entry in metrics)
