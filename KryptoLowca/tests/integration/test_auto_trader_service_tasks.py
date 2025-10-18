import asyncio
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

import pytest

from bot_core.alerts import AlertSeverity
from KryptoLowca.auto_trader import AutoTrader


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


@pytest.mark.asyncio
async def test_service_task_failure_triggers_alert_and_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    emitter = StubEmitter()
    gui = StubGUI()
    trader = AutoTrader(
        emitter,
        gui,
        lambda: "BTC/USDT",
        walkforward_interval_s=None,
        auto_trade_interval_s=0.01,
    )

    alerts: List[Tuple[str, AlertSeverity, str, dict[str, Any], Optional[BaseException]]] = []

    def fake_alert(
        message: str,
        *,
        severity: AlertSeverity,
        source: str,
        context: Optional[dict[str, Any]] = None,
        exception: Optional[BaseException] = None,
    ) -> None:
        alerts.append((message, severity, source, context or {}, exception))

    monkeypatch.setattr("KryptoLowca.auto_trader.emit_alert", fake_alert)

    async def failing_service_loop(self: AutoTrader, symbol: str, timeframe: str) -> None:
        await asyncio.sleep(0)
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(AutoTrader, "_symbol_service_loop", failing_service_loop)

    await trader._ensure_service_schedule([("BTC/USDT", "1m")])
    for _ in range(10):
        if ("BTC/USDT", "1m") not in trader._service_tasks:
            break
        await asyncio.sleep(0.01)

    assert ("BTC/USDT", "1m") not in trader._service_tasks
    assert trader._is_symbol_on_cooldown("BTC/USDT")

    error_logs = [log for log in emitter.logs if log[0] == "ERROR" and log[1] == "AutoTrader"]
    assert any("Service task for BTC/USDT@1m crashed" in log[2] for log in error_logs)

    assert alerts, "Alert should be emitted for crashed service task"
    message, severity, source, context, exception = alerts[-1]
    assert severity is AlertSeverity.ERROR
    assert source == "autotrader"
    assert context.get("symbol") == "BTC/USDT"
    assert context.get("timeframe") == "1m"
    assert isinstance(exception, RuntimeError)

