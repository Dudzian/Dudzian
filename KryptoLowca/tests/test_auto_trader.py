"""Integration tests for the service-based AutoTrader loop."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import pytest

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


def _configured_trader(
    *,
    symbol_source: Callable[[], Iterable[Tuple[str, str]] | Iterable[str] | str],
    data_provider: StubDataProvider,
    signal_service: SignalService,
    risk_service: RiskService,
    execution_adapter: RecordingExecutionAdapter,
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
    return trader, emitter, execution_adapter


def _run_for(trader: AutoTrader, duration: float) -> None:
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

    assert _wait_until(lambda: len(adapter.orders) >= 1)
    order = adapter.orders[0]
    assert order["symbol"] == "BTC/USDT"
    assert order["side"] == "buy"
    assert pytest.approx(order["size"], rel=1e-6) == 75.0
    assert any(event[0] == "auto_trade_tick" for event in emitter.events)


def test_paper_trading_mode_switch(strategy_harness: StrategyHarness) -> None:
    provider = StubDataProvider(price=101.0)
    adapter = RecordingExecutionAdapter()
    trader, _, _ = _configured_trader(
        symbol_source=lambda: [("BTC/USDT", "1m")],
        data_provider=provider,
        signal_service=strategy_harness.signal_service,
        risk_service=AcceptAllRiskService(size=25.0),
        execution_adapter=adapter,
    )
    assert not trader._paper_enabled
    trader.configure(exchange={"testnet": False})
    assert trader._paper_enabled
    assert isinstance(getattr(trader._execution_service, "_adapter", None), PaperTradingAdapter)


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
    symbols = {order["symbol"] for order in adapter.orders}
    assert {"BTC/USDT", "ETH/USDT"}.issubset(symbols)
