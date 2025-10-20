"""Integration tests for the service-based AutoTrader loop."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import pytest
import pandas as pd
from types import SimpleNamespace

from bot_core.config.models import RiskProfileConfig
from bot_core.decision.models import DecisionCandidate
from bot_core.exchanges.base import AccountSnapshot, OrderRequest
from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.risk.base import RiskCheckResult
from bot_core.runtime.metadata import RiskManagerSettings

from KryptoLowca.auto_trader import AutoTrader
from KryptoLowca.backtest.simulation import BacktestFill
from KryptoLowca.config_manager import StrategyConfig
from KryptoLowca.core.services import ExecutionService, PaperTradingAdapter, RiskAssessment, SignalService
from KryptoLowca.core.services.data_provider import ExchangeDataProvider  # noqa: F401 - ensures module importable
from KryptoLowca.core.services.risk_service import RiskService
from KryptoLowca.managers.risk_manager_adapter import RiskManager
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

    def _bridge_execute_trade(self, symbol: str, side: str, price: float) -> None:
        side_norm = (side or "").lower()
        symbol_str = "" if symbol is None else str(symbol)
        symbol_key = symbol_str.upper() or symbol_str
        if side_norm == "buy":
            self._open_positions[symbol_key] = {
                "side": "buy",
                "qty": 1.0,
                "entry": float(price),
            }
        elif side_norm == "sell":
            self._open_positions.pop(symbol_key, None)


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
    assert trader._risk_manager_adapter is not None
    assert isinstance(trader._risk_manager_adapter, RiskManager)


class DenyingRiskManager:
    def __init__(self) -> None:
        self.calls = 0

    def calculate_position_size(self, *args: Any, **kwargs: Any) -> Tuple[float, Mapping[str, Any]]:
        self.calls += 1
        return (
            0.25,
            {
                "allowed": False,
                "reason": "Limit pozycji przekroczony",
                "recommended_size": 0.0,
            },
        )

    def latest_guard_state(self) -> Mapping[str, Any]:
        return {"gross_notional": 12_345.0, "active_positions": 1}


def test_risk_denial_from_adapter_blocks_trade() -> None:
    emitter = DummyEmitter()
    gui = DummyGUI()
    gui.risk_mgr = DenyingRiskManager()
    adapter = RecordingExecutionAdapter()
    trader = AutoTrader(
        emitter,
        gui,
        lambda: "BTC/USDT",
        auto_trade_interval_s=0.5,
        walkforward_interval_s=None,
        signal_service=SignalService(),
        execution_service=ExecutionService(adapter),
        data_provider=StubDataProvider(),
    )

    decision = trader._evaluate_risk(
        "BTC/USDT",
        "BUY",
        100.0,
        trader._build_signal_payload("BTC/USDT", "BUY", 1.0),
        None,
    )

    assert not decision.should_trade
    assert decision.state == "lock"
    assert decision.reason == "risk_engine_denied"
    assert decision.details["risk_engine_allowed"] is False
    assert decision.details["risk_engine"]["allowed"] is False
    assert decision.details["risk_engine"]["reason"] == "Limit pozycji przekroczony"
    assert decision.details["risk_engine"]["state"]["gross_notional"] == pytest.approx(12_345.0)
    assert any(event.get("type") == "risk_engine_denied" for event in decision.details["limit_events"])


class AdjustingRiskManager:
    def __init__(self, *, recommended: float = 0.004, max_qty: float = 12.5) -> None:
        self.recommended = float(recommended)
        self.max_qty = float(max_qty)
        self.calls = 0

    def calculate_position_size(self, *args: Any, **kwargs: Any) -> Tuple[float, Mapping[str, Any]]:
        self.calls += 1
        return (
            self.recommended,
            {
                "allowed": True,
                "reason": "clamped",
                "recommended_size": self.recommended,
                "adjustments": {"max_quantity": self.max_qty},
            },
        )

    def latest_guard_state(self) -> Mapping[str, Any]:
        return {"gross_notional": 2_222.0}


def test_risk_adjustment_from_adapter_sets_warning() -> None:
    emitter = DummyEmitter()
    gui = DummyGUI()
    gui.risk_mgr = AdjustingRiskManager()
    adapter = RecordingExecutionAdapter()
    trader = AutoTrader(
        emitter,
        gui,
        lambda: "BTC/USDT",
        auto_trade_interval_s=0.5,
        walkforward_interval_s=None,
        signal_service=SignalService(),
        execution_service=ExecutionService(adapter),
        data_provider=StubDataProvider(),
    )

    decision = trader._evaluate_risk(
        "BTC/USDT",
        "BUY",
        100.0,
        trader._build_signal_payload("BTC/USDT", "BUY", 1.0),
        None,
    )

    assert decision.should_trade
    assert decision.state == "warn"
    assert decision.reason == "risk_clamped"
    assert decision.fraction == pytest.approx(0.004)
    assert decision.details["risk_engine_allowed"] is True
    risk_info = decision.details["risk_engine"]
    assert risk_info["allowed"] is True
    assert risk_info["adjustments"]["max_quantity"] == pytest.approx(12.5)
    assert any(event.get("type") == "risk_engine_adjustment" for event in decision.details["limit_events"])
    assert any(event.get("type") == "risk_engine_clamp" for event in decision.details["limit_events"])


def test_autotrader_appends_decision_to_log() -> None:
    emitter = DummyEmitter()
    gui = DummyGUI()
    trader = AutoTrader(
        emitter,
        gui,
        lambda: "BTC/USDT",
        auto_trade_interval_s=0.5,
        walkforward_interval_s=None,
        signal_service=SignalService(),
        execution_service=ExecutionService(RecordingExecutionAdapter()),
        data_provider=StubDataProvider(),
    )

    gui.risk_mgr = RiskManager(
        config={
            "max_risk_per_trade": 0.05,
            "max_daily_loss_pct": 0.2,
            "max_positions": 5,
        },
        mode="paper",
        decision_log=trader._risk_decision_log,
    )

    decision = trader._evaluate_risk(
        "BTC/USDT",
        "BUY",
        100.0,
        trader._build_signal_payload("BTC/USDT", "BUY", 1.0),
        None,
    )

    tail = trader._risk_decision_log.tail(limit=5)
    sources = {entry.get("metadata", {}).get("source") for entry in tail}
    assert "risk_manager_adapter" in sources
    assert "auto_trader" in sources

    auto_entry = next(
        entry for entry in reversed(tail) if entry.get("metadata", {}).get("source") == "auto_trader"
    )
    assert auto_entry["symbol"] == "BTCUSDT"
    expected_allowed = bool(decision.should_trade and decision.state != "lock")
    assert auto_entry["allowed"] is expected_allowed
    assert auto_entry["metadata"]["mode"] == decision.mode


def test_core_auto_trade_denial_records_decision(monkeypatch: pytest.MonkeyPatch) -> None:
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
        execution_service=ExecutionService(adapter),
        data_provider=StubDataProvider(),
    )

    class FakeRiskEngine:
        def __init__(self, result: RiskCheckResult) -> None:
            self._result = result
            self.snapshot_calls: list[str] = []
            self.fill_calls: list[Dict[str, object]] = []

        def apply_pre_trade_checks(
            self,
            request: OrderRequest,
            *,
            account: AccountSnapshot,
            profile_name: str,
        ) -> RiskCheckResult:
            self.snapshot_calls.append(profile_name)
            self.last_account = account
            return self._result

        def snapshot_state(self, profile_name: str) -> Mapping[str, object]:
            return {"gross_notional": 123.0, "profile": profile_name}

        def on_fill(
            self,
            *,
            profile_name: str,
            symbol: str,
            side: str,
            position_value: float,
            pnl: float,
            timestamp: datetime | None = None,
        ) -> None:
            self.fill_calls.append(
                {
                    "profile_name": profile_name,
                    "symbol": symbol,
                    "side": side,
                    "position_value": position_value,
                    "pnl": pnl,
                    "timestamp": timestamp,
                }
            )

    class FakeExecutionService:
        def __init__(self) -> None:
            self.calls: list[tuple[OrderRequest, Mapping[str, object]]] = []

        def execute(self, request: OrderRequest, context: Mapping[str, object]) -> Any:
            self.calls.append((request, context))
            return SimpleNamespace(
                filled_quantity=request.quantity,
                avg_price=request.price,
                realized_pnl=10.0,
            )

    class FakeConnector:
        def __init__(self, default_notional: float) -> None:
            self.default_notional = default_notional
            self.risk_profile = "balanced"
            self.ai_manager = object()
            self.calls: list[tuple[str, float]] = []

        def candidate_from_signal(
            self,
            *,
            symbol: str,
            signal: float,
            timestamp: object | None,
            notional: float,
            metadata: Mapping[str, object] | None = None,
        ) -> DecisionCandidate:
            self.calls.append((symbol, signal))
            return DecisionCandidate(
                strategy="test",
                action="enter",
                risk_profile=self.risk_profile,
                symbol=symbol,
                notional=notional,
                expected_return_bps=signal * 10_000.0,
                expected_probability=0.6,
                metadata={"source": "test"},
            )

    risk_result = RiskCheckResult(
        allowed=False,
        reason="Limit pozycji przekroczony",
        adjustments={"max_quantity": 0.5},
        metadata={"engine": "threshold"},
    )
    fake_engine = FakeRiskEngine(risk_result)
    fake_execution = FakeExecutionService()
    fake_connector = FakeConnector(default_notional=1_000.0)

    trader._core_risk_engine = fake_engine
    trader._core_execution_service = fake_execution
    trader._core_ai_connector = fake_connector
    trader._core_ai_default_notional = 1_000.0
    trader._core_account_equity = 10_000.0
    trader._core_risk_profile = "balanced"
    trader._risk_profile_name = "balanced"

    df = pd.DataFrame(
        {"close": [100.0]}, index=pd.to_datetime(["2024-01-01T00:00:00Z"])
    )
    monkeypatch.setattr(
        trader,
        "_obtain_prediction",
        lambda *_, **__: (0.02, df, 100.0),
    )

    audited: list[RiskDecision] = []

    def capture_audit(symbol: str, side: str, decision: RiskDecision, price: float) -> None:
        audited.append(decision)

    monkeypatch.setattr(trader, "_emit_risk_audit", capture_audit)

    handled = trader._handle_core_auto_trade("BTC/USDT", "1m")
    assert handled is True
    assert fake_execution.calls == []
    assert audited, "Powinien zostać zarejestrowany wpis audytu ryzyka"
    decision = audited[-1]
    assert not decision.should_trade
    assert decision.reason == "risk_engine_denied"
    assert decision.details["risk_engine_allowed"] is False
    assert decision.details["risk_engine"]["state"]["gross_notional"] == pytest.approx(123.0)
    limit_events = decision.details.get("limit_events", [])
    assert any(evt.get("type") == "risk_engine_denied" for evt in limit_events)
    assert any(evt.get("type") == "risk_engine_adjustment" for evt in limit_events)
    assert decision.details["recommended_size"] == pytest.approx(0.005)

    tail = trader._risk_decision_log.tail(limit=1)
    assert tail, "RiskDecisionLog powinien otrzymać wpis"
    entry = tail[-1]
    assert entry["symbol"] == "BTCUSDT"
    assert entry["allowed"] is False
    assert entry["reason"] == "risk_engine_denied"
    assert entry["metadata"]["source"] == "auto_trader"


def test_core_auto_trade_allowed_records_decision(monkeypatch: pytest.MonkeyPatch) -> None:
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
        execution_service=ExecutionService(adapter),
        data_provider=StubDataProvider(),
    )

    class FakeRiskEngine:
        def __init__(self, result: RiskCheckResult) -> None:
            self._result = result
            self.fill_calls: list[Dict[str, object]] = []

        def apply_pre_trade_checks(
            self,
            request: OrderRequest,
            *,
            account: AccountSnapshot,
            profile_name: str,
        ) -> RiskCheckResult:
            self.last_profile = profile_name
            self.last_account = account
            return self._result

        def snapshot_state(self, profile_name: str) -> Mapping[str, object]:
            return {"gross_notional": 123.0, "profile": profile_name}

        def on_fill(
            self,
            *,
            profile_name: str,
            symbol: str,
            side: str,
            position_value: float,
            pnl: float,
            timestamp: datetime | None = None,
        ) -> None:
            self.fill_calls.append(
                {
                    "profile_name": profile_name,
                    "symbol": symbol,
                    "side": side,
                    "position_value": position_value,
                    "pnl": pnl,
                    "timestamp": timestamp,
                }
            )

    class FakeExecutionService:
        def __init__(self) -> None:
            self.calls: list[tuple[OrderRequest, Mapping[str, object]]] = []

        def execute(self, request: OrderRequest, context: Mapping[str, object]) -> Any:
            self.calls.append((request, context))
            return SimpleNamespace(
                filled_quantity=request.quantity,
                avg_price=request.price,
                realized_pnl=25.0,
                timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            )

    class FakeConnector:
        def __init__(self, default_notional: float) -> None:
            self.default_notional = default_notional
            self.risk_profile = "balanced"
            self.ai_manager = object()

        def candidate_from_signal(
            self,
            *,
            symbol: str,
            signal: float,
            timestamp: object | None,
            notional: float,
            metadata: Mapping[str, object] | None = None,
        ) -> DecisionCandidate:
            return DecisionCandidate(
                strategy="test",
                action="enter",
                risk_profile=self.risk_profile,
                symbol=symbol,
                notional=notional,
                expected_return_bps=signal * 10_000.0,
                expected_probability=0.6,
                metadata={"source": "test"},
            )

    risk_result = RiskCheckResult(
        allowed=True,
        reason="ok",
        adjustments=None,
        metadata={"engine": "threshold"},
    )
    fake_engine = FakeRiskEngine(risk_result)
    fake_execution = FakeExecutionService()
    fake_connector = FakeConnector(default_notional=1_000.0)

    trader._core_risk_engine = fake_engine
    trader._core_execution_service = fake_execution
    trader._core_ai_connector = fake_connector
    trader._core_ai_default_notional = 1_000.0
    trader._core_account_equity = 10_000.0
    trader._core_risk_profile = "balanced"
    trader._risk_profile_name = "balanced"

    df = pd.DataFrame(
        {"close": [100.0]}, index=pd.to_datetime(["2024-01-01T00:00:00Z"])
    )
    monkeypatch.setattr(
        trader,
        "_obtain_prediction",
        lambda *_, **__: (0.03, df, 100.0),
    )

    audited: list[RiskDecision] = []

    def capture_audit(symbol: str, side: str, decision: RiskDecision, price: float) -> None:
        audited.append(decision)

    monkeypatch.setattr(trader, "_emit_risk_audit", capture_audit)

    handled = trader._handle_core_auto_trade("BTC/USDT", "1m")
    assert handled is True
    assert fake_execution.calls, "Powinno zostać złożone zlecenie core"
    execution_request, _ = fake_execution.calls[-1]
    assert execution_request.quantity == pytest.approx(10.0)
    assert fake_engine.fill_calls, "Powinien zostać wywołany on_fill"
    fill_entry = fake_engine.fill_calls[-1]
    assert fill_entry["symbol"] == "BTCUSDT"
    assert fill_entry["side"] == "buy"
    assert fill_entry["position_value"] == pytest.approx(1000.0)
    assert fill_entry["pnl"] == pytest.approx(25.0)
    assert fill_entry["timestamp"] == datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)

    assert audited, "Powinien zostać zarejestrowany audyt"
    decision = audited[-1]
    assert decision.should_trade
    assert decision.state == "ok"
    assert decision.fraction == pytest.approx(0.1)
    assert decision.details["risk_engine_allowed"] is True
    assert decision.details["risk_engine"]["state"]["gross_notional"] == pytest.approx(123.0)
    assert "limit_events" not in decision.details or not decision.details["limit_events"]

    tail = trader._risk_decision_log.tail(limit=1)
    assert tail and tail[-1]["allowed"] is True
    assert tail[-1]["metadata"]["source"] == "auto_trader"


def test_core_auto_trade_short_updates_position(monkeypatch: pytest.MonkeyPatch) -> None:
    emitter = DummyEmitter()
    gui = DummyGUI()
    adapter = RecordingExecutionAdapter()
    trader = AutoTrader(
        emitter,
        gui,
        lambda: "ETH/USDT",
        auto_trade_interval_s=0.5,
        walkforward_interval_s=None,
        signal_service=SignalService(),
        execution_service=ExecutionService(adapter),
        data_provider=StubDataProvider(),
    )

    class FakeRiskEngine:
        def __init__(self, result: RiskCheckResult) -> None:
            self._result = result
            self.fill_calls: list[Dict[str, object]] = []

        def apply_pre_trade_checks(
            self,
            request: OrderRequest,
            *,
            account: AccountSnapshot,
            profile_name: str,
        ) -> RiskCheckResult:
            return self._result

        def snapshot_state(self, profile_name: str) -> Mapping[str, object]:
            return {"gross_notional": 42.0, "profile": profile_name}

        def on_fill(
            self,
            *,
            profile_name: str,
            symbol: str,
            side: str,
            position_value: float,
            pnl: float,
            timestamp: datetime | None = None,
        ) -> None:
            self.fill_calls.append(
                {
                    "profile_name": profile_name,
                    "symbol": symbol,
                    "side": side,
                    "position_value": position_value,
                    "pnl": pnl,
                    "timestamp": timestamp,
                }
            )

    class FakeExecutionService:
        def __init__(self) -> None:
            self.calls: list[tuple[OrderRequest, Mapping[str, object]]] = []

        def execute(self, request: OrderRequest, context: Mapping[str, object]) -> Any:
            self.calls.append((request, context))
            return SimpleNamespace(
                filled_quantity=request.quantity,
                avg_price=request.price,
                realized_pnl=-12.5,
            )

    class FakeConnector:
        def __init__(self) -> None:
            self.default_notional = 2_000.0
            self.risk_profile = "balanced"
            self.ai_manager = object()

        def candidate_from_signal(
            self,
            *,
            symbol: str,
            signal: float,
            timestamp: object | None,
            notional: float,
            metadata: Mapping[str, object] | None = None,
        ) -> DecisionCandidate:
            return DecisionCandidate(
                strategy="test",
                action="enter",
                risk_profile=self.risk_profile,
                symbol=symbol,
                notional=notional,
                expected_return_bps=signal * 10_000.0,
                expected_probability=0.55,
                metadata={},
            )

    risk_result = RiskCheckResult(
        allowed=True,
        reason="ok",
        adjustments=None,
        metadata={},
    )

    fake_engine = FakeRiskEngine(risk_result)
    fake_execution = FakeExecutionService()
    fake_connector = FakeConnector()

    trader._core_risk_engine = fake_engine
    trader._core_execution_service = fake_execution
    trader._core_ai_connector = fake_connector
    trader._core_ai_default_notional = fake_connector.default_notional
    trader._core_account_equity = 20_000.0
    trader._core_risk_profile = "balanced"
    trader._risk_profile_name = "balanced"

    df = pd.DataFrame(
        {"close": [200.0]}, index=pd.to_datetime(["2024-01-01T00:00:00Z"])
    )
    monkeypatch.setattr(
        trader,
        "_obtain_prediction",
        lambda *_, **__: (-0.05, df, 200.0),
    )

    handled = trader._handle_core_auto_trade("ETH/USDT", "1m")
    assert handled is True
    assert fake_execution.calls, "Powinno zostać złożone zlecenie core"
    fill_entry = fake_engine.fill_calls[-1]
    assert fill_entry["symbol"] == "ETHUSDT"
    assert fill_entry["side"] == "sell"
    assert fill_entry["position_value"] == pytest.approx(2000.0)
    assert fill_entry["pnl"] == pytest.approx(-12.5)


def test_paper_trading_adapter_uses_gui_balance() -> None:
    emitter = DummyEmitter()
    gui = DummyGUI(paper_balance=2_500.0)
    adapter = RecordingExecutionAdapter()
    trader = AutoTrader(
        emitter,
        gui,
        lambda: "BTC/USDT",
        auto_trade_interval_s=0.5,
        walkforward_interval_s=None,
        signal_service=SignalService(),
        execution_service=ExecutionService(adapter),
        data_provider=StubDataProvider(),
    )

    trader._enable_paper_trading()
    assert trader._paper_adapter is not None
    snapshot = trader._paper_adapter.portfolio_snapshot("BTC/USDT")
    assert snapshot["value"] == pytest.approx(2_500.0)


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
        execution_service=ExecutionService(adapter),
        data_provider=StubDataProvider(),
    )

    trader._core_risk_profile = "baseline"

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
    assert trader._core_risk_profile == "growth"
    assert trader._risk_service.max_portfolio_risk_pct == pytest.approx(0.3)
    assert trader._risk_service.max_positions == 9
    cfg = trader._get_strategy_config()
    assert cfg.max_position_notional_pct == pytest.approx(0.08)
    assert cfg.trade_risk_pct == pytest.approx(0.08)
    assert trader._risk_manager_adapter is not None


def test_autotrader_registers_core_risk_profile_with_bootstrap() -> None:
    engine = ThresholdRiskEngine()
    bootstrap = SimpleNamespace(
        risk_engine=engine,
        environment=SimpleNamespace(name="paper", environment=SimpleNamespace(value="paper"), ai=None),
        risk_profile_name="paper",
        execution_service=None,
        ai_manager=None,
        ai_model_bindings=(),
        ai_threshold_bps=None,
    )

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
        execution_service=ExecutionService(adapter),
        data_provider=StubDataProvider(),
        bootstrap_context=bootstrap,
        core_risk_engine=engine,
    )

    profile_name = trader._core_risk_profile or trader._risk_profile_name or "paper"
    profiles = getattr(engine, "_profiles", {})
    assert profile_name in profiles

    snapshot = trader._core_account_snapshot()
    request = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=1.0,
        order_type="market",
        price=100.0,
    )
    result = engine.apply_pre_trade_checks(request, account=snapshot, profile_name=profile_name)
    assert result.reason is not None
    assert "ATR" in result.reason


def test_autotrader_reloads_core_risk_profile_on_update() -> None:
    engine = ThresholdRiskEngine()
    bootstrap = SimpleNamespace(
        risk_engine=engine,
        environment=SimpleNamespace(name="paper", environment=SimpleNamespace(value="paper"), ai=None),
        risk_profile_name="paper",
        execution_service=None,
        ai_manager=None,
        ai_model_bindings=(),
        ai_threshold_bps=None,
    )

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
        execution_service=ExecutionService(adapter),
        data_provider=StubDataProvider(),
        bootstrap_context=bootstrap,
        core_risk_engine=engine,
    )

    new_settings = RiskManagerSettings(
        max_risk_per_trade=0.04,
        max_daily_loss_pct=0.12,
        max_portfolio_risk=0.2,
        max_positions=6,
        emergency_stop_drawdown=0.18,
    )
    profile_cfg = RiskProfileConfig(
        name="growth",
        max_daily_loss_pct=0.12,
        max_position_pct=0.04,
        target_volatility=0.18,
        max_leverage=2.5,
        stop_loss_atr_multiple=1.4,
        max_open_positions=6,
        hard_drawdown_pct=0.18,
    )

    trader.update_risk_manager_settings(
        new_settings,
        profile_name="growth",
        profile_config=profile_cfg,
    )

    profiles = getattr(engine, "_profiles", {})
    assert "growth" in profiles

    request = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=1.0,
        order_type="market",
        price=100.0,
    )
    snapshot = trader._core_account_snapshot()
    result = engine.apply_pre_trade_checks(request, account=snapshot, profile_name="growth")
    assert result.reason is not None
    assert "ATR" in result.reason


def test_autotrader_update_account_equity_adjusts_snapshot() -> None:
    emitter = DummyEmitter()
    gui = DummyGUI(paper_balance=5_000.0)
    adapter = RecordingExecutionAdapter()
    trader = AutoTrader(
        emitter,
        gui,
        lambda: "BTC/USDT",
        auto_trade_interval_s=0.5,
        walkforward_interval_s=None,
        signal_service=SignalService(),
        execution_service=ExecutionService(adapter),
        data_provider=StubDataProvider(),
    )

    trader.update_account_equity(4_321.5)
    snapshot = trader._core_account_snapshot()
    assert getattr(trader, "_core_account_equity", 0.0) == pytest.approx(4_321.5)
    assert snapshot.total_equity == pytest.approx(4_321.5)
    assert snapshot.available_margin == pytest.approx(4_321.5)

    trader.update_account_equity(-50.0)
    snapshot = trader._core_account_snapshot()
    assert getattr(trader, "_core_account_equity", 0.0) == pytest.approx(0.0)
    assert snapshot.total_equity == pytest.approx(0.0)


def _configured_trader(
    *,
    symbol_source: Callable[[], Iterable[Tuple[str, str]] | Iterable[str] | str],
    data_provider: StubDataProvider,
    signal_service: SignalService,
    risk_service: RiskService | None,
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


@pytest.mark.asyncio
async def test_trade_once_updates_dummy_gui_positions(strategy_harness: StrategyHarness) -> None:
    provider = StubDataProvider(price=102.5)
    execution_adapter = RecordingExecutionAdapter()
    gui = DummyGUI()
    trader = AutoTrader(
        DummyEmitter(),
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
        ),
    )

    assert trader._paper_enabled
    assert gui._open_positions == {}

    await trader._trade_once("BTC/USDT", "1m")

    assert "BTC/USDT" in gui._open_positions
    position = gui._open_positions["BTC/USDT"]
    assert position["side"] == "buy"
    assert position["qty"] > 0.0


@pytest.mark.asyncio
async def test_trade_once_sell_clears_dummy_gui_positions() -> None:
    provider = StubDataProvider(price=103.7)
    execution_adapter = RecordingExecutionAdapter()
    gui = DummyGUI()
    gui._open_positions["BTC/USDT"] = {"side": "buy", "qty": 1.0, "entry": 100.0}

    registry = StrategyRegistry()

    @registry.register
    class DummySellStrategy(BaseStrategy):
        metadata = StrategyMetadata(
            name="DummySellStrategy",
            description="Test strategy sell",
            timeframes=("1m",),
        )

        async def generate_signal(
            self,
            context: StrategyContext,
            market_payload: Mapping[str, Any],
        ) -> StrategySignal:
            return StrategySignal(
                symbol=context.symbol,
                action="SELL",
                confidence=0.9,
                size=25.0,
            )

    trader = AutoTrader(
        DummyEmitter(),
        gui,
        symbol_getter=lambda: "BTC/USDT",
        auto_trade_interval_s=0.05,
        walkforward_interval_s=None,
        signal_service=SignalService(strategy_registry=registry),
        risk_service=AcceptAllRiskService(size=25.0),
        execution_service=ExecutionService(execution_adapter),
        data_provider=provider,
    )

    trader.enable_auto_trade = True
    trader.confirm_auto_trade(True)
    trader.configure(
        strategy=StrategyConfig(
            preset="DummySellStrategy",
            mode="demo",
            max_leverage=1.0,
            max_position_notional_pct=0.5,
            trade_risk_pct=0.1,
            default_sl=0.01,
            default_tp=0.02,
            violation_cooldown_s=1,
            reduce_only_after_violation=False,
        ),
    )

    assert trader._paper_enabled

    await trader._trade_once("BTC/USDT", "1m")

    assert gui._open_positions == {}


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
