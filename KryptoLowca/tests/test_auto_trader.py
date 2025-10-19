"""Integration tests for the service-based AutoTrader loop."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import pytest
import pandas as pd
from bot_core.runtime.metadata import RiskManagerSettings

from KryptoLowca.auto_trader import AutoTrader
from KryptoLowca.backtest.simulation import BacktestFill
from KryptoLowca.config_manager import StrategyConfig
from KryptoLowca.core.services import ExecutionService, PaperTradingAdapter, RiskAssessment, SignalService
from KryptoLowca.core.services.data_provider import ExchangeDataProvider  # noqa: F401 - ensures module importable
from KryptoLowca.core.services.risk_service import RiskService
from KryptoLowca.strategies.base import BaseStrategy, StrategyContext, StrategyMetadata, StrategySignal
from KryptoLowca.strategies.base.registry import StrategyRegistry

# Opcjonalny import okna ważności backtestu — gdy brak, użyjemy sensownego domyślnego
try:  # pragma: no cover - tylko dla kompatybilności
    from KryptoLowca.config_manager import BACKTEST_VALIDITY_WINDOW_S  # type: ignore
except Exception:  # pragma: no cover
    BACKTEST_VALIDITY_WINDOW_S = 24 * 3600  # 24h fallback


class DummyEmitter:
    def __init__(self) -> None:
        self.logs: List[Tuple[str, str, str]] = []
        self.events: List[Tuple[str, Mapping[str, Any]]] = []

    def on(self, *_, **__) -> None:  # pragma: no cover - interface only
        return None

    def off(self, *_, **__) -> None:  # pragma: no cover - interface only
        return None

    def emit(self, event: str, **payload: Any) -> None:
        self.events.append((event, dict(payload)))

    def log(self, message: str, level: str = "INFO", component: Optional[str] = None) -> None:
        self.logs.append((level, component or "", message))


class DummyVar:
    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        return self._value


class RecordingExecutionAdapter:
    def __init__(self) -> None:
        self.orders: List[Mapping[str, Any]] = []

    async def submit_order(self, *, symbol: str, side: str, size: float, **kwargs: Any) -> Mapping[str, Any]:
        payload = {"symbol": symbol, "side": side, "size": size, **kwargs}
        self.orders.append(payload)
        return {"status": "ok", **payload}


class StubDataProvider:
    def __init__(self, *, price: float = 100.0) -> None:
        self.price = float(price)
        self.ohlcv_calls: Dict[str, int] = {}

    async def get_ohlcv(self, symbol: str, timeframe: str, *, limit: int = 500) -> Mapping[str, Any]:
        self.ohlcv_calls[symbol] = self.ohlcv_calls.get(symbol, 0) + 1
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "limit": limit,
            "close": self.price,
            "candles": [[0, self.price, self.price, self.price, self.price, 1.0]],
        }

    async def get_ticker(self, symbol: str) -> Mapping[str, Any]:
        return {"symbol": symbol, "last": self.price}


class AcceptAllRiskService(RiskService):
    def __init__(self, *, size: float = 50.0) -> None:
        super().__init__()
        self.size = float(size)
        self.calls = 0

    def assess(self, signal: StrategySignal, context: StrategyContext, market_state: Mapping[str, float]) -> RiskAssessment:
        self.calls += 1
        return RiskAssessment(
            allow=True,
            reason="ok",
            size=self.size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )


class RejectingRiskService(RiskService):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def assess(self, signal: StrategySignal, context: StrategyContext, market_state: Mapping[str, float]) -> RiskAssessment:
        self.calls += 1
        return RiskAssessment(allow=False, reason="blocked", size=0.0)


class DummyGUI:
    def __init__(self, *, paper_balance: float = 10_000.0) -> None:
        self.timeframe_var = DummyVar("1m")
        self.paper_balance = paper_balance
        self._open_positions: Dict[str, Dict[str, Any]] = {}
        self.network_var = DummyVar("demo")

    def get_portfolio_snapshot(self, symbol: str) -> Mapping[str, Any]:
        return {
            "portfolio_value": self.paper_balance,
            "position": 0.0,
            "symbol": symbol,
        }


@dataclass
class StrategyHarness:
    registry: StrategyRegistry
    signal_service: SignalService


@pytest.fixture
def strategy_harness() -> StrategyHarness:
    registry = StrategyRegistry()

    @registry.register
    class DummyStrategy(BaseStrategy):
        metadata = StrategyMetadata(
            name="DummyStrategy",
            description="Test strategy",
            timeframes=("1m", "5m"),
        )

        async def generate_signal(
            self,
            context: StrategyContext,
            market_payload: Mapping[str, Any],
        ) -> StrategySignal:
            return StrategySignal(
                symbol=context.symbol,
                action="BUY",
                confidence=0.9,
                size=50.0,
            )

    return StrategyHarness(registry=registry, signal_service=SignalService(strategy_registry=registry))


def test_autotrader_applies_runtime_risk_profile() -> None:
    emitter = DummyEmitter()
    gui = DummyGUI()
    adapter = RecordingExecutionAdapter()
    trader = AutoTrader(
        emitter,
        gui,
        lambda: "BTC/USDT",
        auto_trade_interval_s=0.5,
        walkforward_interval_s=None,
        signal_service=SignalService(),
        risk_service=RiskService(),
        execution_service=ExecutionService(adapter),
        data_provider=StubDataProvider(),
    )

    cfg = trader._get_strategy_config()
    assert cfg.max_position_notional_pct == pytest.approx(0.05)
    assert cfg.trade_risk_pct == pytest.approx(0.015)
    assert cfg.max_leverage == pytest.approx(3.0)
    assert trader._risk_manager_settings.max_risk_per_trade == pytest.approx(0.05)
    assert trader._risk_manager_settings.max_daily_loss_pct == pytest.approx(0.015)
    assert trader._risk_service.max_position_notional_pct == pytest.approx(0.05)
    assert trader._risk_service.max_daily_loss_pct == pytest.approx(0.015)
    assert trader._risk_service.max_portfolio_risk_pct == pytest.approx(0.10)
    assert trader._risk_service.max_positions == 5
    assert trader._risk_service.emergency_stop_drawdown_pct == pytest.approx(0.10)


def test_autotrader_update_risk_manager_settings_applies_changes() -> None:
    emitter = DummyEmitter()
    gui = DummyGUI()
    adapter = RecordingExecutionAdapter()
    trader = AutoTrader(
        emitter,
        gui,
        lambda: "BTC/USDT",
        auto_trade_interval_s=0.5,
        walkforward_interval_s=None,
        signal_service=SignalService(),
        risk_service=RiskService(),
        execution_service=ExecutionService(adapter),
        data_provider=StubDataProvider(),
    )

    new_settings = RiskManagerSettings(
        max_risk_per_trade=0.08,
        max_daily_loss_pct=0.18,
        max_portfolio_risk=0.3,
        max_positions=9,
        emergency_stop_drawdown=0.28,
    )
    profile_payload = {
        "max_position_pct": 0.08,
        "max_daily_loss_pct": 0.18,
        "trade_risk_pct": 0.03,
        "max_open_positions": 9,
        "hard_drawdown_pct": 0.28,
        "max_leverage": 2.5,
    }

    trader.update_risk_manager_settings(
        new_settings,
        profile_name="growth",
        profile_config=profile_payload,
    )

    assert trader._risk_manager_settings is new_settings
    assert trader._risk_profile_name == "growth"
    assert trader._risk_service.max_portfolio_risk_pct == pytest.approx(0.3)
    assert trader._risk_service.max_positions == 9
    cfg = trader._get_strategy_config()
    assert cfg.max_position_notional_pct == pytest.approx(0.08)
    assert cfg.trade_risk_pct == pytest.approx(0.08)


def _configured_trader(
    *,
    symbol_source: Callable[[], Iterable[Tuple[str, str]] | Iterable[str] | str],
    data_provider: StubDataProvider,
    signal_service: SignalService,
    risk_service: RiskService,
    execution_adapter: RecordingExecutionAdapter,
    strategy_mode: str = "demo",
    exchange_overrides: Optional[Mapping[str, Any]] = None,
) -> Tuple[AutoTrader, DummyEmitter, RecordingExecutionAdapter]:
    emitter = DummyEmitter()
    execution_service = ExecutionService(execution_adapter)
    trader = AutoTrader(
        emitter,
        DummyGUI(),
        symbol_source,
        auto_trade_interval_s=0.05,
        walkforward_interval_s=None,
        signal_service=signal_service,
        risk_service=risk_service,
        execution_service=execution_service,
        data_provider=data_provider,
    )
    trader.enable_auto_trade = True
    trader.confirm_auto_trade(True)
    strategy_kwargs: Dict[str, Any] = {
        "preset": "DummyStrategy",
        "mode": strategy_mode,
        "max_leverage": 1.0,
        "max_position_notional_pct": 0.5,
        "trade_risk_pct": 0.1,
        "default_sl": 0.01,
        "default_tp": 0.02,
        "violation_cooldown_s": 1,
        "reduce_only_after_violation": False,
    }
    if strategy_mode == "live":
        now_ts = time.time()
        strategy_kwargs.update(
            compliance_confirmed=True,
            api_keys_configured=True,
            acknowledged_risk_disclaimer=True,
            backtest_passed_at=now_ts,
        )
    exchange_cfg: Dict[str, Any] = {"testnet": strategy_mode != "live"}
    if exchange_overrides:
        exchange_cfg.update(exchange_overrides)
    exchange_cfg.setdefault("adapter", execution_adapter)
    trader.configure(
        strategy=StrategyConfig(**strategy_kwargs),
        exchange=exchange_cfg,
    )
    trader.confirm_auto_trade(True)
    return trader, emitter, execution_adapter


def _run_for(trader: AutoTrader, duration: float) -> None:
    trader.confirm_auto_trade(True)
    trader.start()
    time.sleep(duration)
    trader.stop()


def _wait_until(predicate: Callable[[], bool], timeout: float = 1.5) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def test_live_mode_requires_backtest_confirmation(strategy_harness: StrategyHarness) -> None:
    provider = StubDataProvider(price=100.0)
    adapter = RecordingExecutionAdapter()
    trader, emitter, _ = _configured_trader(
        symbol_source=lambda: [("BTC/USDT", "1m")],
        data_provider=provider,
        signal_service=strategy_harness.signal_service,
        risk_service=AcceptAllRiskService(size=50.0),
        execution_adapter=adapter,
    )

    assert trader._strategy_config.mode == "demo"

    live_cfg = StrategyConfig(
        preset="DummyStrategy",
        mode="live",
        max_leverage=1.0,
        max_position_notional_pct=0.5,
        trade_risk_pct=0.1,
        default_sl=0.01,
        default_tp=0.02,
        violation_cooldown_s=1,
        reduce_only_after_violation=False,
        compliance_confirmed=True,
        api_keys_configured=True,
        acknowledged_risk_disclaimer=True,
    )

    trader.configure(strategy=live_cfg)

    assert trader._strategy_config.mode == "demo"
    assert any(
        level == "WARNING" and "backtest" in message.lower()
        for level, _, message in emitter.logs
    )


def test_live_mode_rejects_stale_backtest(strategy_harness: StrategyHarness) -> None:
    provider = StubDataProvider(price=102.0)
    adapter = RecordingExecutionAdapter()
    trader, emitter, _ = _configured_trader(
        symbol_source=lambda: [("BTC/USDT", "1m")],
        data_provider=provider,
        signal_service=strategy_harness.signal_service,
        risk_service=AcceptAllRiskService(size=50.0),
        execution_adapter=adapter,
    )

    assert trader._strategy_config.mode == "demo"

    stale_ts = time.time() - (BACKTEST_VALIDITY_WINDOW_S + 10)
    live_cfg = StrategyConfig(
        preset="DummyStrategy",
        mode="live",
        max_leverage=1.0,
        max_position_notional_pct=0.5,
        trade_risk_pct=0.1,
        default_sl=0.01,
        default_tp=0.02,
        violation_cooldown_s=1,
        reduce_only_after_violation=False,
        compliance_confirmed=True,
        api_keys_configured=True,
        acknowledged_risk_disclaimer=True,
        backtest_passed_at=stale_ts,
    )

    trader.configure(strategy=live_cfg)

    assert trader._strategy_config.mode == "demo"
    assert any(
        level == "WARNING" and "przeterminowany" in message.lower()
        for level, _, message in emitter.logs
    )


def test_live_mode_requires_compliance_flag(strategy_harness: StrategyHarness) -> None:
    provider = StubDataProvider(price=103.0)
    adapter = RecordingExecutionAdapter()
    trader, _, adapter = _configured_trader(
        symbol_source=lambda: [("BTC/USDT", "1m")],
        data_provider=provider,
        signal_service=strategy_harness.signal_service,
        risk_service=AcceptAllRiskService(size=50.0),
        execution_adapter=adapter,
    )

    live_cfg = StrategyConfig(
        preset="DummyStrategy",
        mode="live",
        max_leverage=1.0,
        max_position_notional_pct=0.5,
        trade_risk_pct=0.1,
        default_sl=0.01,
        default_tp=0.02,
        violation_cooldown_s=1,
        reduce_only_after_violation=False,
        compliance_confirmed=True,
        api_keys_configured=True,
        acknowledged_risk_disclaimer=True,
        backtest_passed_at=time.time(),
    )

    trader._compliance_live_allowed = False
    trader.configure(strategy=live_cfg)

    assert trader._strategy_config.mode == "live"

    adapter.orders.clear()
    _run_for(trader, 0.4)
    assert adapter.orders == []

    trader._compliance_live_allowed = True
    adapter.orders.clear()
    _run_for(trader, 0.4)
    assert _wait_until(lambda: len(adapter.orders) >= 1)


def test_services_execute_order(strategy_harness: StrategyHarness) -> None:
    provider = StubDataProvider(price=101.0)
    adapter = RecordingExecutionAdapter()
    trader, emitter, adapter = _configured_trader(
        symbol_source=lambda: [("BTC/USDT", "1m")],
        data_provider=provider,
        signal_service=strategy_harness.signal_service,
        risk_service=AcceptAllRiskService(size=75.0),
        execution_adapter=adapter,
    )

    _run_for(trader, 0.4)

    assert any(event[0] == "auto_trade_tick" for event in emitter.events)
    paper_adapter = trader._paper_adapter
    assert paper_adapter is not None
    assert _wait_until(lambda: "BTC/USDT" in getattr(paper_adapter, "_portfolios", {}))


def test_paper_trading_mode_switch(strategy_harness: StrategyHarness) -> None:
    provider = StubDataProvider(price=101.0)
    adapter = RecordingExecutionAdapter()
    trader, _, adapter = _configured_trader(
        symbol_source=lambda: [("BTC/USDT", "1m")],
        data_provider=provider,
        signal_service=strategy_harness.signal_service,
        risk_service=AcceptAllRiskService(size=25.0),
        execution_adapter=adapter,
    )
    assert trader._paper_enabled
    assert isinstance(getattr(trader._execution_service, "_adapter", None), PaperTradingAdapter)

    live_cfg = StrategyConfig(
        preset="DummyStrategy",
        mode="live",
        max_leverage=1.0,
        max_position_notional_pct=0.5,
        trade_risk_pct=0.1,
        default_sl=0.01,
        default_tp=0.02,
        violation_cooldown_s=1,
        reduce_only_after_violation=False,
        compliance_confirmed=True,
        api_keys_configured=True,
        acknowledged_risk_disclaimer=True,
        backtest_passed_at=time.time(),
    )

    trader.configure(strategy=live_cfg, exchange={"testnet": False, "adapter": adapter})
    assert not trader._paper_enabled
    assert getattr(trader._execution_service, "_adapter", None) is adapter

    trader.configure(
        strategy=StrategyConfig(
            preset="DummyStrategy",
            mode="demo",
            max_leverage=1.0,
            max_position_notional_pct=0.5,
            trade_risk_pct=0.1,
            default_sl=0.01,
            default_tp=0.02,
            violation_cooldown_s=1,
            reduce_only_after_violation=False,
        ),
        exchange={"testnet": True},
    )
    assert trader._paper_enabled
    assert isinstance(getattr(trader._execution_service, "_adapter", None), PaperTradingAdapter)


def test_autotrader_defaults_to_paper_without_exchange_config(
    strategy_harness: StrategyHarness,
) -> None:
    provider = StubDataProvider(price=100.0)
    emitter = DummyEmitter()
    gui = DummyGUI()
    execution_adapter = RecordingExecutionAdapter()
    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "BTC/USDT",
        auto_trade_interval_s=0.05,
        walkforward_interval_s=None,
        signal_service=strategy_harness.signal_service,
        risk_service=AcceptAllRiskService(size=25.0),
        execution_service=ExecutionService(execution_adapter),
        data_provider=provider,
    )
    trader.enable_auto_trade = True
    trader.confirm_auto_trade(True)
    trader.configure(
        strategy=StrategyConfig(
            preset="DummyStrategy",
            mode="demo",
            max_leverage=1.0,
            max_position_notional_pct=0.5,
            trade_risk_pct=0.1,
            default_sl=0.01,
            default_tp=0.02,
            violation_cooldown_s=1,
            reduce_only_after_violation=False,
        )
    )

    assert trader._paper_enabled
    paper_adapter = getattr(trader._execution_service, "_adapter", None)
    assert isinstance(paper_adapter, PaperTradingAdapter)
    result = paper_adapter.submit_order(symbol="BTC/USDT", side="buy", size=0.5)
    assert result.get("status") != "skipped"


def test_paper_trading_adapter_apply_fill_charges_fees() -> None:
    adapter = PaperTradingAdapter(initial_balance=1_000.0)
    state = adapter._ensure_state("BTC/USDT")
    timestamp = datetime.now(timezone.utc)

    buy_fill = BacktestFill(
        order_id=1,
        side="buy",
        size=0.5,
        price=100.0,
        fee=0.25,
        slippage=0.0,
        timestamp=timestamp,
        partial=False,
    )
    adapter._apply_fill(state, buy_fill)

    expected_cash = adapter._initial_balance - (buy_fill.price * buy_fill.size) - buy_fill.fee
    assert state.cash == pytest.approx(expected_cash)
    assert state.position == pytest.approx(0.5)

    sell_fill = BacktestFill(
        order_id=2,
        side="sell",
        size=0.5,
        price=110.0,
        fee=0.3,
        slippage=0.0,
        timestamp=timestamp,
        partial=False,
    )
    adapter._apply_fill(state, sell_fill)

    expected_cash += (sell_fill.price * sell_fill.size) - sell_fill.fee
    assert state.cash == pytest.approx(expected_cash)
    assert state.position == pytest.approx(0.0, abs=1e-9)
    assert state.avg_price == pytest.approx(0.0)

    total_fees = sum(fill.fee for fill in state.fills)
    assert total_fees == pytest.approx(buy_fill.fee + sell_fill.fee)


def test_risk_rejection_applies_cooldown(strategy_harness: StrategyHarness) -> None:
    provider = StubDataProvider(price=99.0)
    adapter = RecordingExecutionAdapter()
    risk_service = RejectingRiskService()
    trader, emitter, _ = _configured_trader(
        symbol_source=lambda: [("ETH/USDT", "5m")],
        data_provider=provider,
        signal_service=strategy_harness.signal_service,
        risk_service=risk_service,
        execution_adapter=adapter,
    )

    _run_for(trader, 0.3)

    assert risk_service.calls == 1
    assert not adapter.orders
    assert any("Cooldown applied" in log[2] for log in emitter.logs)


def test_scheduler_handles_multiple_symbols(strategy_harness: StrategyHarness) -> None:
    provider = StubDataProvider(price=105.0)
    adapter = RecordingExecutionAdapter()
    trader, _, adapter = _configured_trader(
        symbol_source=lambda: [("BTC/USDT", "1m"), ("ETH/USDT", "5m")],
        data_provider=provider,
        signal_service=strategy_harness.signal_service,
        risk_service=AcceptAllRiskService(size=25.0),
        execution_adapter=adapter,
    )

    _run_for(trader, 0.6)

    assert provider.ohlcv_calls.get("BTC/USDT", 0) > 0
    assert provider.ohlcv_calls.get("ETH/USDT", 0) > 0
    paper_adapter = trader._paper_adapter
    assert paper_adapter is not None
    assert _wait_until(
        lambda: {"BTC/USDT", "ETH/USDT"}.issubset(
            set(getattr(paper_adapter, "_portfolios", {}).keys())
        ),
        timeout=2.0,
    )


def test_obtain_prediction_executes_async_coroutine(strategy_harness: StrategyHarness) -> None:
    class AsyncPredictAI:
        ai_threshold_bps = 5.0

        def __init__(self) -> None:
            self.series = pd.Series([0.0, 0.0125])
            self.calls = 0

        async def predict_series(self, *, symbol: str, timeframe: str, bars: int = 256) -> pd.Series:
            self.calls += 1
            return self.series

    class PriceProvider:
        def __init__(self, price: float) -> None:
            self.price = float(price)
            self.symbols: List[str] = []

        def get_latest_price(self, symbol: str) -> float:
            self.symbols.append(symbol)
            return self.price

    class AsyncDummyGUI(DummyGUI):
        def __init__(self, ai: AsyncPredictAI) -> None:
            super().__init__()
            self.ai_mgr = ai
            self.executed: List[Tuple[str, str, float]] = []

        def _bridge_execute_trade(self, symbol: str, side: str, price: float) -> None:
            self.executed.append((symbol, side, price))

    ai = AsyncPredictAI()
    price_provider = PriceProvider(101.25)
    emitter = DummyEmitter()
    gui = AsyncDummyGUI(ai)
    execution_adapter = RecordingExecutionAdapter()

    trader = AutoTrader(
        emitter,
        gui,
        symbol_getter=lambda: "BTC/USDT",
        auto_trade_interval_s=0.05,
        walkforward_interval_s=None,
        signal_service=strategy_harness.signal_service,
        risk_service=AcceptAllRiskService(size=50.0),
        execution_service=ExecutionService(execution_adapter),
        data_provider=None,
        market_data_provider=price_provider,
    )
    trader.enable_auto_trade = True
    trader.confirm_auto_trade(True)
    trader.configure(
        strategy=StrategyConfig(
            preset="DummyStrategy",
            mode="demo",
            max_leverage=1.0,
            max_position_notional_pct=0.5,
            trade_risk_pct=0.1,
            default_sl=0.01,
            default_tp=0.02,
            violation_cooldown_s=1,
            reduce_only_after_violation=False,
        )
    )

    last_pred, df, last_price = trader._obtain_prediction(ai, "BTC/USDT", "1m", None)
    assert df is None
    assert last_pred is not None
    assert last_pred == pytest.approx(ai.series.iloc[-1])
    assert last_price == pytest.approx(price_provider.price)

    _run_for(trader, 0.3)

    assert gui.executed
    assert any("Auto-trade executed" in message for _, _, message in emitter.logs)


@pytest.mark.asyncio
async def test_resolve_prediction_result_handles_running_event_loop(
    strategy_harness: StrategyHarness,
) -> None:
    provider = StubDataProvider(price=100.0)
    adapter = RecordingExecutionAdapter()
    trader, _, _ = _configured_trader(
        symbol_source=lambda: [("BTC/USDT", "1m")],
        data_provider=provider,
        signal_service=strategy_harness.signal_service,
        risk_service=AcceptAllRiskService(size=25.0),
        execution_adapter=adapter,
    )

    async def async_series() -> pd.Series:
        return pd.Series([0.5, 1.25])

    result = trader._resolve_prediction_result(
        async_series(), context="test-running-loop"
    )

    assert isinstance(result, pd.Series)
    assert result.iloc[-1] == pytest.approx(1.25)
