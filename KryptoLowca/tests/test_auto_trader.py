"""Tests for safety guards in AutoTrader."""
from __future__ import annotations

import sys
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd
import pytest

# Ensure repository root is available on sys.path when running the test as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from KryptoLowca.alerts import get_alert_dispatcher
from KryptoLowca.auto_trader import AutoTrader
from KryptoLowca.config_manager import StrategyConfig
from KryptoLowca.risk_management import RiskManagement


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


class DummyDB:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        self.risk_audits: List[Dict[str, Any]] = []
        self.sync = self

    def log_performance_metric(self, payload: Dict[str, Any]) -> int:
        self.records.append(payload)
        return len(self.records)

    def log_risk_audit(self, payload: Dict[str, Any]) -> int:
        self.risk_audits.append(payload)
        return len(self.risk_audits)


class DummyRiskManager:
    def __init__(self, fraction: float = 0.05) -> None:
        self.fraction = fraction
        self.calls = 0

    def calculate_position_size(
        self,
        symbol: str,
        signal: Any,
        market_data: Any,
        portfolio: Dict[str, Any],
        return_details: bool = False,
    ) -> Any:
        self.calls += 1
        if return_details:
            return self.fraction, {"recommended_size": self.fraction, "reasoning": "ok"}
        return self.fraction


class DummyCfg:
    def __init__(self, strategy: StrategyConfig | Dict[str, Any] | None = None) -> None:
        self._strategy = strategy

    def load_strategy_config(self) -> StrategyConfig | Dict[str, Any]:
        if self._strategy is None:
            return StrategyConfig()
        return self._strategy


class DummyGUI:
    def __init__(
        self,
        demo: bool,
        allow_live: bool,
        *,
        ai: Any | None = None,
        risk_mgr: DummyRiskManager | None = None,
        strategy: StrategyConfig | Dict[str, Any] | None = None,
        paper_balance: float = 10_000.0,
    ) -> None:
        self.timeframe_var = DummyVar("1m")
        self.network_var = DummyVar("Testnet" if demo else "Live")
        self.ai_mgr = ai or NoSignalAI()
        self.ex_mgr = self
        self._demo = demo
        self._allow_live = allow_live
        self.executed: List[Tuple[str, str, float]] = []
        self.db = DummyDB()
        self.paper_balance = paper_balance
        self._open_positions: Dict[str, Dict[str, Any]] = {}
        self.risk_mgr = risk_mgr or DummyRiskManager()
        self.cfg = DummyCfg(strategy)

    def is_demo_mode_active(self) -> bool:
        return self._demo

    def is_live_trading_allowed(self) -> bool:
        return self._allow_live

    def _bridge_execute_trade(self, symbol: str, side: str, price: float) -> None:
        self.executed.append((symbol, side, price))

    # Exchange-like API -------------------------------------------------
    def fetch_ticker(self, symbol: str) -> Dict[str, float]:
        return {"last": 100.0}

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", limit: int = 20) -> List[List[float]]:
        base_ts = int(time.time() * 1000) - limit * 60_000
        data: List[List[float]] = []
        price = 100.0
        for i in range(limit):
            ts = base_ts + i * 60_000
            open_ = price
            high = open_ * 1.001
            low = open_ * 0.999
            close = open_ * 1.0005
            volume = 10.0 + i
            data.append([ts, open_, high, low, close, volume])
            price = close
        return data


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
    assert any(
        "live trading requires explicit confirmation" in msg
        for kind, _, msg in trader.emitter.logs
        if kind == "log"
    )


def test_metrics_persisted_on_trade_close(demo_autotrader: Callable[[DummyGUI], AutoTrader]) -> None:
    gui = DummyGUI(demo=True, allow_live=True)
    trader = demo_autotrader(gui)
    trader._on_trade_closed("BTC/USDT", "BUY", 100.0, 110.0, 5.0, time.time())
    metrics = gui.db.records
    assert metrics, "Brak zapisanych metryk po zamknięciu transakcji"
    names = {entry["metric"] for entry in metrics}
    assert "auto_trader_expectancy" in names
    assert any(entry["symbol"] == "BTC/USDT" for entry in metrics)


class SignalAI:
    def __init__(self, value: float = 0.02) -> None:
        self.ai_threshold_bps = 5.0
        self._value = value

    def predict_series(self, df: pd.DataFrame | None = None, **_: Any) -> pd.Series:
        if df is None or df.empty:
            return pd.Series([self._value])
        return pd.Series([self._value] * len(df), index=df.index)


def test_risk_manager_blocks_trade_on_zero_fraction(
    demo_autotrader: Callable[[DummyGUI], AutoTrader]
) -> None:
    risk_mgr = DummyRiskManager(fraction=0.0)
    gui = DummyGUI(demo=True, allow_live=True, ai=SignalAI(0.02), risk_mgr=risk_mgr)
    trader = demo_autotrader(gui)
    _run_loop(trader, duration=0.2)

    assert gui.executed == []
    assert risk_mgr.calls > 0
    risk_events = [payload for kind, event, payload in trader.emitter.logs if kind == "event" and event == "risk_guard_event"]
    assert any("risk_fraction_zero" in payload for payload in risk_events)
    assert gui.db.risk_audits, "Powinien zostać zapisany audyt ryzyka"
    assert any(entry["reason"] == "risk_fraction_zero" for entry in gui.db.risk_audits)


def test_risk_audit_contains_limit_events(demo_autotrader: Callable[[DummyGUI], AutoTrader]) -> None:
    risk_mgr = DummyRiskManager(fraction=0.5)
    gui = DummyGUI(demo=True, allow_live=True, ai=SignalAI(0.02), risk_mgr=risk_mgr)
    trader = demo_autotrader(gui)
    _run_loop(trader, duration=0.2)

    assert gui.db.risk_audits, "Audyt ryzyka powinien być zarejestrowany"
    limit_event_found = False
    engine_keys_present = False
    for audit in gui.db.risk_audits:
        details = audit.get("details") or {}
        limit_events = details.get("limit_events") or []
        if any(evt.get("type") == "trade_risk_pct" for evt in limit_events):
            limit_event_found = True
            engine = details.get("risk_engine") or {}
            engine_keys_present = "recommended_size" in engine
            break
    assert limit_event_found, "Powinien istnieć wpis z limit_events"
    assert engine_keys_present, "Detal risk_engine powinien zawierać recommended_size"


def test_risk_audit_emits_alert_and_cooldown(
    demo_autotrader: Callable[[DummyGUI], AutoTrader]
) -> None:
    dispatcher = get_alert_dispatcher()
    dispatcher.clear()
    events: List[Any] = []

    def _listener(event):
        events.append(event)

    token = dispatcher.register(_listener, name="risk-alert-test")
    try:
        risk_mgr = DummyRiskManager(fraction=0.5)
        gui = DummyGUI(demo=True, allow_live=True, ai=SignalAI(0.02), risk_mgr=risk_mgr)
        trader = demo_autotrader(gui)
        _run_loop(trader, duration=0.2)
    finally:
        dispatcher.unregister(token)

    assert gui.db.risk_audits, "Powinien powstać wpis audytu"
    assert any(event.source == "risk_guard" for event in events), "Powinien zostać wysłany alert risk_guard"
    cooldown_values = [event.context.get("cooldown_until") for event in events if event.source == "risk_guard"]
    assert any(value for value in cooldown_values), "Alert powinien zawierać informację o cooldown"
    cooldown_details = [
        (audit.get("details") or {})
        for audit in gui.db.risk_audits
        if (audit.get("details") or {}).get("cooldown_until")
    ]
    assert cooldown_details, "Audyt powinien zawierać cooldown"


def test_compliance_flags_block_live_mode(demo_autotrader: Callable[[DummyGUI], AutoTrader]) -> None:
    gui = DummyGUI(demo=False, allow_live=True)
    trader = demo_autotrader(gui)
    custom_cfg = StrategyConfig(
        preset="CUSTOM",
        mode="live",
        max_leverage=1.0,
        max_position_notional_pct=0.02,
        trade_risk_pct=0.01,
        default_sl=0.02,
        default_tp=0.04,
        violation_cooldown_s=120,
        reduce_only_after_violation=True,
        compliance_confirmed=False,
        api_keys_configured=False,
        acknowledged_risk_disclaimer=False,
    )
    trader._strategy_config = custom_cfg
    trader._strategy_override = True

    decision = trader._evaluate_risk(
        "BTC/USDT",
        "BUY",
        100.0,
        {"strength": 0.6, "confidence": 0.7},
        None,
    )

    assert decision.should_trade is False
    assert decision.reason == "compliance_requirements_not_met"
    assert "missing" in decision.details


def test_position_sizing_with_real_risk_management(
    demo_autotrader: Callable[[DummyGUI], AutoTrader]
) -> None:
    gui = DummyGUI(demo=True, allow_live=True, ai=SignalAI(0.05))
    gui.risk_mgr = RiskManagement({"max_risk_per_trade": 0.03, "max_portfolio_risk": 0.2})
    trader = demo_autotrader(gui)

    base = 100.0
    closes = [base + i * 0.5 for i in range(200)]
    df = pd.DataFrame(
        {
            "open": [c * 0.999 for c in closes],
            "high": [c * 1.001 for c in closes],
            "low": [c * 0.998 for c in closes],
            "close": closes,
            "volume": [100 + i for i in range(200)],
        }
    )

    decision = trader._evaluate_risk(
        "BTC/USDT",
        "BUY",
        float(df["close"].iloc[-1]),
        {"strength": 0.7, "confidence": 0.6},
        df,
    )

    assert decision.should_trade is True
    assert "risk_engine" in decision.details
    engine = decision.details["risk_engine"]
    assert engine["recommended_size"] >= 0.0
    assert decision.fraction <= decision.details["strategy_trade_risk_pct"]
