"""Tests for safety guards in AutoTrader."""
from __future__ import annotations

import threading
import time
from dataclasses import asdict
from types import MethodType
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd
import pytest

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
        self.sync = self

    def log_performance_metric(self, payload: Dict[str, Any]) -> int:
        self.records.append(payload)
        return len(self.records)


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


def _convert_portfolio(portfolio_ctx: Dict[str, Any], fallback_price: float) -> Dict[str, Dict[str, float]]:
    converted: Dict[str, Dict[str, float]] = {}
    if not isinstance(portfolio_ctx, dict):
        return converted
    raw_positions = portfolio_ctx.get("positions", {})
    if not isinstance(raw_positions, dict):
        return converted
    for sym, pos in raw_positions.items():
        if not isinstance(pos, dict):
            continue
        try:
            qty = abs(float(pos.get("qty", 0.0) or 0.0))
        except Exception:
            qty = 0.0
        try:
            entry = float(pos.get("entry") or pos.get("price") or fallback_price)
        except Exception:
            entry = fallback_price
        notional = abs(float(pos.get("notional", qty * entry))) if entry or qty else 0.0
        size = qty if entry <= 0 else (notional / entry if entry else qty)
        converted[sym] = {
            "size": size,
            "volatility": float(pos.get("volatility", 0.2)),
            "entry_price": entry,
        }
    return converted


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


def test_real_risk_management_integration_handles_reduce_only(
    demo_autotrader: Callable[[DummyGUI], AutoTrader]
) -> None:
    risk_engine = RiskManagement(
        {
            "max_risk_per_trade": 0.015,
            "max_portfolio_risk": 0.2,
            "max_positions": 5,
        }
    )
    original_calc = RiskManagement.calculate_position_size

    def adapted_calculate_position_size(
        self: RiskManagement,
        *,
        symbol: str,
        signal: Dict[str, Any],
        market_data: Any,
        portfolio: Dict[str, Any],
        return_details: bool = False,
    ) -> Any:
        market_df = market_data if isinstance(market_data, pd.DataFrame) else pd.DataFrame(market_data)
        fallback_price = float(market_df["close"].iloc[-1]) if "close" in market_df and not market_df.empty else 0.0
        converted_portfolio = _convert_portfolio(portfolio, fallback_price)
        sizing = original_calc(self, symbol, signal, market_df, converted_portfolio)
        detail_payload = asdict(sizing)
        detail_payload["engine"] = "RiskManagement"
        if return_details:
            return sizing.recommended_size, detail_payload
        return sizing.recommended_size

    risk_engine.calculate_position_size = MethodType(adapted_calculate_position_size, risk_engine)

    gui = DummyGUI(demo=True, allow_live=True, ai=SignalAI(0.03), risk_mgr=risk_engine)
    trader = demo_autotrader(gui)

    periods = 120
    index = pd.date_range("2024-01-01", periods=periods, freq="min")
    steps = pd.Series(range(periods), dtype=float, index=index)
    close = 100.0 + 0.05 * steps
    market_df = pd.DataFrame(
        {
            "open": close - 0.02,
            "high": close + 0.04,
            "low": close - 0.06,
            "close": close,
            "volume": 100 + steps * 0.5,
        },
        index=index,
    )

    symbol = "BTC/USDT"
    risk_engine.historical_returns[symbol] = market_df["close"].pct_change().dropna()
    price = float(market_df["close"].iloc[-1])
    signal_payload = AutoTrader._build_signal_payload(symbol, "BUY", 0.03)

    portfolio_ctx = trader._build_portfolio_context(symbol, price)
    expected_sizing = original_calc(
        risk_engine,
        symbol,
        signal_payload,
        market_df,
        _convert_portfolio(portfolio_ctx, price),
    )

    decision = trader._evaluate_risk(symbol, "BUY", price, signal_payload, market_df)

    assert decision.should_trade is True
    assert decision.fraction == pytest.approx(expected_sizing.recommended_size, rel=1e-6)
    assert decision.details["engine"] == "RiskManagement"
    assert decision.details["risk_adjusted_size"] == pytest.approx(expected_sizing.risk_adjusted_size, rel=1e-6)
    assert decision.mode == "demo"

    tight_cfg = StrategyConfig(
        preset="SAFE",
        mode="demo",
        max_leverage=0.05,
        max_position_notional_pct=0.02,
        trade_risk_pct=0.01,
        default_sl=0.02,
        default_tp=0.04,
        violation_cooldown_s=60,
        reduce_only_after_violation=True,
    ).validate()
    trader._update_strategy_config(tight_cfg)

    gui.paper_balance = 100.0
    gui._open_positions[symbol] = {"qty": 5.0, "entry": price, "side": "LONG"}

    violation_decision = trader._evaluate_risk(symbol, "BUY", price, signal_payload, market_df)

    assert violation_decision.should_trade is False
    assert violation_decision.reason == "max_leverage_exceeded"
    assert any(
        event.get("type") == "max_leverage"
        for event in violation_decision.details.get("limit_events", [])
    )
    ro_until = trader._reduce_only_until.get(symbol, 0.0)
    assert ro_until > time.time()

    reduce_only_decision = trader._evaluate_risk(symbol, "BUY", price, signal_payload, market_df)

    assert reduce_only_decision.should_trade is False
    assert reduce_only_decision.reason == "reduce_only_active"
