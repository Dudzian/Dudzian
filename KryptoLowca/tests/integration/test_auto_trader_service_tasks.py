import threading
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

import pytest

from bot_core.auto_trader import AutoTrader


class StubEmitter:
    def __init__(self) -> None:
        self.logs: List[Tuple[str, Optional[str], str]] = []

    def on(self, *_: Any, **__: Any) -> None:  # pragma: no cover - interface only
        return None

    def off(self, *_: Any, **__: Any) -> None:  # pragma: no cover - interface only
        return None

    def emit(self, *_: Any, **__: Any) -> None:  # pragma: no cover - interface only
        return None

    def log(self, message: str, level: str = "INFO", component: Optional[str] = None) -> None:
        self.logs.append((level, component, message))


class StubGUI:
    def __init__(self, paper_balance: float = 10_000.0) -> None:
        self.paper_balance = paper_balance
        self.timeframe_var = SimpleNamespace(get=lambda: "1m")
        self.network_var = SimpleNamespace(get=lambda: "demo")

    def get_portfolio_snapshot(self, symbol: str) -> dict[str, Any]:  # pragma: no cover - helper only
        return {
            "portfolio_value": self.paper_balance,
            "position": 0.0,
            "symbol": symbol,
        }


def _make_trader(auto_trade_interval_s: float = 0.01) -> AutoTrader:
    return AutoTrader(
        StubEmitter(),
        StubGUI(),
        lambda: "BTC/USDT",
        walkforward_interval_s=None,
        auto_trade_interval_s=auto_trade_interval_s,
    )


def test_confirm_auto_trade_starts_background_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    trader = _make_trader()
    loop_triggered = threading.Event()

    def fake_loop(self: AutoTrader) -> None:
        loop_triggered.set()
        self._auto_trade_stop.set()

    monkeypatch.setattr(AutoTrader, "_auto_trade_loop", fake_loop)

    trader.start()
    try:
        assert trader.is_running()
        assert not loop_triggered.is_set()

        trader.confirm_auto_trade(True)
        assert loop_triggered.wait(1.0), "Auto-trade loop should run after confirmation"
    finally:
        trader.stop()

    assert not trader.is_running()


def test_disabling_auto_trade_resets_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    trader = _make_trader()
    loop_triggered = threading.Event()
    loop_runs: List[int] = []

    def fake_loop(self: AutoTrader) -> None:
        loop_runs.append(1)
        loop_triggered.set()
        self._auto_trade_stop.set()

    monkeypatch.setattr(AutoTrader, "_auto_trade_loop", fake_loop)

    trader.start()
    try:
        trader.confirm_auto_trade(True)
        assert loop_triggered.wait(1.0)
        assert loop_runs, "Background loop should execute"

        trader.set_enable_auto_trade(False)
        trader.set_enable_auto_trade(True)

        loop_triggered.clear()
        # Without reconfirmation the loop should remain idle
        assert not loop_triggered.wait(0.05)

        trader.confirm_auto_trade(True)
        assert loop_triggered.wait(1.0)
        assert len(loop_runs) == 2
    finally:
        trader.stop()
