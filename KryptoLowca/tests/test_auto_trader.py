"""Integration tests for the service-based AutoTrader loop."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple
from types import SimpleNamespace

import pytest
import pandas as pd
from bot_core.runtime.metadata import RiskManagerSettings
from bot_core.risk.events import RiskDecisionLog

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


class RecordingRiskEngine:
    def __init__(self, *, decision_log: RiskDecisionLog | None = None) -> None:
        self._decision_log = decision_log
        self.fills: List[Mapping[str, Any]] = []

    def on_fill(self, **payload: Any) -> None:
        self.fills.append(dict(payload))


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
        risk_service=RiskService(),
        execution_service=ExecutionService(adapter),
        data_provider=StubDataProvider(),
    )

    trader._enable_paper_trading()
    assert trader._paper_adapter is not None
    snapshot = trader._paper_adapter.portfolio_snapshot("BTC/USDT")
    assert snapshot["value"] == pytest.approx(2_500.0)



def test_post_core_fill_merges_metadata_from_result() -> None:
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

    decision_log = RiskDecisionLog(clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc))
    risk_engine = RecordingRiskEngine(decision_log=decision_log)
    trader._core_risk_engine = risk_engine
    trader._core_risk_profile = "core-profile"

    order_request = SimpleNamespace(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.25,
        price=24_950.0,
        metadata={
            "decision_candidate": {
                "risk_profile": "core-profile",
                "metadata": {"decision_time": "2024-04-01T12:30:00Z"},
            },
            "note": "from-strategy",
        },
    )
    fill_timestamp = datetime.fromtimestamp(1_700_000_000_000 / 1000.0, tz=timezone.utc)
    order_result = SimpleNamespace(
        avg_price=24_930.0,
        filled_quantity=0.25,
        raw_response={
            "timestamp": 1_700_000_000_000,
            "pnl": "12.5",
            "fill_id": "abc123",
        },
    )

    trader._post_core_fill("BTCUSDT", "BUY", order_request, order_result)

    assert len(risk_engine.fills) == 1
    fill_payload = risk_engine.fills[0]
    assert fill_payload["profile_name"] == "core-profile"
    assert fill_payload["symbol"] == "BTCUSDT"
    assert fill_payload["side"] == "buy"
    assert fill_payload["pnl"] == pytest.approx(12.5)
    assert fill_payload["position_value"] == pytest.approx(24_930.0 * 0.25)
    assert fill_payload["timestamp"] == fill_timestamp

    entries = decision_log.tail(limit=1)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["profile"] == "core-profile"
    assert entry["symbol"] == "BTCUSDT"
    assert entry["side"] == "buy"
    assert entry["quantity"] == pytest.approx(0.25)
    assert entry["price"] == pytest.approx(24_930.0)
    metadata = entry["metadata"]
    assert metadata["source"] == "auto_trader_core_fill"
    expected_notional = 24_930.0 * 0.25
    assert metadata["metrics"]["pnl"] == pytest.approx(12.5)
    assert metadata["metrics"]["notional"] == pytest.approx(expected_notional)
    assert metadata["metrics"]["position_value"] == pytest.approx(expected_notional)
    assert metadata["metrics"]["position_delta"] == pytest.approx(expected_notional)
    assert metadata["request"]["note"] == "from-strategy"
    assert metadata["fill"]["fill_id"] == "abc123"
    assert metadata["fill_timestamp"] == fill_timestamp.isoformat()


def test_post_core_fill_prefers_request_metadata_when_missing() -> None:
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
        risk_service=RiskService(),
        execution_service=ExecutionService(adapter),
        data_provider=StubDataProvider(),
    )

    decision_log = RiskDecisionLog(clock=lambda: datetime(2024, 1, 2, tzinfo=timezone.utc))
    risk_engine = RecordingRiskEngine(decision_log=decision_log)
    trader._core_risk_engine = risk_engine
    trader._core_risk_profile = None

    class _Connector:
        risk_profile = "connector-default"

    trader._core_ai_connector = _Connector()

    request_timestamp = datetime(2024, 4, 2, 15, 30, 45, tzinfo=timezone.utc)
    order_request = SimpleNamespace(
        symbol="ETHUSDT",
        side="sell",
        quantity=0.4,
        price=1_850.0,
        metadata={
            "pnl": -7.5,
            "decision_candidate": {
                "risk_profile": "risk-beta",
                "metadata": {"timestamp": request_timestamp.isoformat()},
            },
        },
    )
    order_result = SimpleNamespace(
        avg_price=None,
        filled_quantity=0.4,
        raw_response={"fill_id": "xyz"},
    )

    trader._post_core_fill("ETHUSDT", "SELL", order_request, order_result)

    assert len(risk_engine.fills) == 1
    fill_payload = risk_engine.fills[0]
    assert fill_payload["profile_name"] == "risk-beta"
    assert fill_payload["side"] == "sell"
    assert fill_payload["symbol"] == "ETHUSDT"
    assert fill_payload["pnl"] == pytest.approx(-7.5)
    assert fill_payload["position_value"] == pytest.approx(0.0)
    assert fill_payload["timestamp"] == request_timestamp

    entry = decision_log.tail(limit=1)[0]
    metadata = entry["metadata"]
    assert entry["profile"] == "risk-beta"
    assert entry["side"] == "sell"
    expected_notional = 1_850.0 * 0.4
    assert metadata["metrics"]["pnl"] == pytest.approx(-7.5)
    assert metadata["metrics"]["notional"] == pytest.approx(expected_notional)
    assert metadata["request"]["decision_candidate"]["risk_profile"] == "risk-beta"
    assert metadata["fill"]["fill_id"] == "xyz"
    assert metadata["fill_timestamp"] == request_timestamp.isoformat()
    assert metadata["metrics"]["position_value"] == pytest.approx(0.0)
    assert metadata["metrics"]["position_delta"] == pytest.approx(-expected_notional)


def test_post_core_fill_reads_nested_sequence_sources() -> None:
    emitter = DummyEmitter()
    gui = DummyGUI()
    adapter = RecordingExecutionAdapter()
    trader = AutoTrader(
        emitter,
        gui,
        lambda: "SOL/USDT",
        auto_trade_interval_s=0.5,
        walkforward_interval_s=None,
        signal_service=SignalService(),
        risk_service=RiskService(),
        execution_service=ExecutionService(adapter),
        data_provider=StubDataProvider(),
    )

    log_clock = lambda: datetime(2024, 1, 3, tzinfo=timezone.utc)
    decision_log = RiskDecisionLog(clock=log_clock)
    risk_engine = RecordingRiskEngine(decision_log=decision_log)
    trader._core_risk_engine = risk_engine
    trader._core_risk_profile = "seq-profile"

    request_metadata = {
        "decision_candidate": {
            "risk_profile": "seq-profile",
            "metadata": {"note": "nested"},
        }
    }
    order_request = SimpleNamespace(
        symbol="SOLUSDT",
        side="buy",
        quantity=3.0,
        price=None,
        metadata=request_metadata,
    )

    nested_timestamp_ns = 1_700_200_300_400_000_000
    nested_raw = {
        "fills": [
            {
                "info": {
                    "realizedPnl": "3.25",
                    "executedAt": nested_timestamp_ns,
                }
            }
        ],
        "extra": [{"events": [{"time": nested_timestamp_ns // 1_000_000}]}],
    }
    order_result = SimpleNamespace(
        avg_price=24.5,
        filled_quantity=3.0,
        raw_response=nested_raw,
    )

    trader._post_core_fill("SOLUSDT", "BUY", order_request, order_result)

    assert len(risk_engine.fills) == 1
    payload = risk_engine.fills[0]
    assert payload["profile_name"] == "seq-profile"
    assert payload["pnl"] == pytest.approx(3.25)
    expected_ts = datetime.fromtimestamp(nested_timestamp_ns / 1_000_000_000, tz=timezone.utc)
    assert payload["timestamp"] == expected_ts

    entry = decision_log.tail(limit=1)[0]
    metadata = entry["metadata"]
    assert metadata["fill_timestamp"] == expected_ts.isoformat()
    assert metadata["fill"]["fills"][0]["info"]["realizedPnl"] == "3.25"
    assert metadata["metrics"]["pnl"] == pytest.approx(3.25)
    assert metadata["metrics"]["notional"] == pytest.approx(24.5 * 3.0)


def test_post_core_fill_handles_namespace_and_pair_sequences() -> None:
    emitter = DummyEmitter()
    gui = DummyGUI()
    adapter = RecordingExecutionAdapter()
    trader = AutoTrader(
        emitter,
        gui,
        lambda: "ADA/USDT",
        auto_trade_interval_s=0.5,
        walkforward_interval_s=None,
        signal_service=SignalService(),
        risk_service=RiskService(),
        execution_service=ExecutionService(adapter),
        data_provider=StubDataProvider(),
    )

    decision_log = RiskDecisionLog(clock=lambda: datetime(2024, 1, 4, tzinfo=timezone.utc))
    risk_engine = RecordingRiskEngine(decision_log=decision_log)
    trader._core_risk_engine = risk_engine
    trader._core_risk_profile = None

    raw_timestamp_ms = 1_701_111_222_333
    raw_response = SimpleNamespace(
        values=[("profit_loss", "8.75"), ("ignored", None)],
        details=SimpleNamespace(events=[SimpleNamespace(ts=raw_timestamp_ms)]),
    )
    request_timestamp_ms = raw_timestamp_ms + 5_000
    request_metadata = SimpleNamespace(
        decision_candidate=SimpleNamespace(risk_profile="ns-profile"),
        tags=("alpha", "beta"),
        extra=[["timestamp", request_timestamp_ms]],
    )
    order_request = SimpleNamespace(
        symbol="ADAUSDT",
        side="sell",
        quantity=1.5,
        price=1.25,
        metadata=request_metadata,
    )
    order_result = SimpleNamespace(
        avg_price=1.3,
        filled_quantity=None,
        raw_response=raw_response,
    )

    trader._post_core_fill("ADAUSDT", "SELL", order_request, order_result)

    assert len(risk_engine.fills) == 1
    payload = risk_engine.fills[0]
    assert payload["profile_name"] == "ns-profile"
    assert payload["symbol"] == "ADAUSDT"
    assert payload["side"] == "sell"
    assert payload["pnl"] == pytest.approx(8.75)
    assert payload["position_value"] == pytest.approx(0.0)
    expected_timestamp = datetime.fromtimestamp(raw_timestamp_ms / 1000.0, tz=timezone.utc)
    assert payload["timestamp"] == expected_timestamp

    entry = decision_log.tail(limit=1)[0]
    metadata = entry["metadata"]
    assert metadata["metrics"]["pnl"] == pytest.approx(8.75)
    assert metadata["metrics"]["notional"] == pytest.approx(1.3 * 1.5)
    assert metadata["fill_timestamp"] == expected_timestamp.isoformat()
    assert metadata["fill"]["values"][0] == ["profit_loss", "8.75"]
    assert metadata["request"]["tags"] == ["alpha", "beta"]
    assert metadata["request"]["extra"][0] == ["timestamp", request_timestamp_ms]


def test_post_core_fill_sanitizes_invalid_numeric_sources() -> None:
    from decimal import Decimal

    emitter = DummyEmitter()
    gui = DummyGUI()
    adapter = RecordingExecutionAdapter()
    trader = AutoTrader(
        emitter,
        gui,
        lambda: "XRP/USDT",
        auto_trade_interval_s=0.5,
        walkforward_interval_s=None,
        signal_service=SignalService(),
        risk_service=RiskService(),
        execution_service=ExecutionService(adapter),
        data_provider=StubDataProvider(),
    )

    decision_log = RiskDecisionLog(clock=lambda: datetime(2024, 1, 5, tzinfo=timezone.utc))
    risk_engine = RecordingRiskEngine(decision_log=decision_log)
    trader._core_risk_engine = risk_engine
    trader._core_risk_profile = "nan-profile"

    order_request = SimpleNamespace(
        symbol="XRPUSDT",
        side="buy",
        quantity="0.8",
        price=Decimal("0.52"),
        metadata={"decision_candidate": {"risk_profile": "nan-profile"}},
    )
    raw_timestamp = "2024-04-03T12:30:00Z"
    order_result = SimpleNamespace(
        avg_price=Decimal("NaN"),
        filled_quantity=float("nan"),
        pnl=float("nan"),
        timestamp=pd.NaT,
        raw_response={
            "fills": [
                {"profit": "15.75", "time": 1_700_000_999_000},
                {"details": {"ts": raw_timestamp}},
            ]
        },
    )

    trader._post_core_fill("XRPUSDT", "BUY", order_request, order_result)

    assert len(risk_engine.fills) == 1
    fill_payload = risk_engine.fills[0]
    assert fill_payload["profile_name"] == "nan-profile"
    assert fill_payload["symbol"] == "XRPUSDT"
    assert fill_payload["side"] == "buy"
    assert fill_payload["pnl"] == pytest.approx(15.75)
    assert fill_payload["timestamp"] == datetime(2024, 4, 3, 12, 30, tzinfo=timezone.utc)
    assert fill_payload["position_value"] == pytest.approx(0.52 * 0.8)

    entries = decision_log.tail(limit=1)
    assert len(entries) == 1
    entry = entries[0]
    metrics = entry["metadata"]["metrics"]
    assert metrics["avg_price"] == pytest.approx(0.52)
    assert metrics["filled_quantity"] == pytest.approx(0.8)
    assert metrics["notional"] == pytest.approx(0.52 * 0.8)
    assert metrics["pnl"] == pytest.approx(15.75)
    assert metrics["position_value"] == pytest.approx(0.52 * 0.8)
    assert metrics["position_delta"] == pytest.approx(0.52 * 0.8)
    assert entry["metadata"]["fill_timestamp"] == datetime(
        2024, 4, 3, 12, 30, tzinfo=timezone.utc
    ).isoformat()


def test_post_core_fill_merges_notional_from_nested_sources() -> None:
    emitter = DummyEmitter()
    gui = DummyGUI()
    adapter = RecordingExecutionAdapter()
    trader = AutoTrader(
        emitter,
        gui,
        lambda: "SOL/USDT",
        auto_trade_interval_s=0.5,
        walkforward_interval_s=None,
        signal_service=SignalService(),
        risk_service=RiskService(),
        execution_service=ExecutionService(adapter),
        data_provider=StubDataProvider(),
    )

    decision_log = RiskDecisionLog(clock=lambda: datetime(2024, 1, 6, tzinfo=timezone.utc))
    risk_engine = RecordingRiskEngine(decision_log=decision_log)
    trader._core_risk_engine = risk_engine
    trader._core_risk_profile = "nested"

    nested_raw = {
        "order": {
            "fills": [
                {
                    "price": "25123.45",
                    "executedQty": "0.32",
                    "quoteQty": "8039.504",
                }
            ],
            "cost": "8039.504",
            "transactTime": 1_700_000_000_500,
        }
    }
    request_metadata = {
        "decision_candidate": {"risk_profile": "nested"},
        "details": [
            {"averagePrice": "25123.45"},
            {"baseQty": "0.32"},
            {"quoteAmount": "8039.504"},
        ],
    }
    order_request = SimpleNamespace(
        symbol="SOLUSDT",
        side="buy",
        quantity=None,
        price=None,
        metadata=request_metadata,
    )
    order_result = SimpleNamespace(
        avg_price=None,
        filled_quantity=None,
        pnl="18.0",
        raw_response=nested_raw,
    )

    trader._post_core_fill("SOLUSDT", "BUY", order_request, order_result)

    assert len(risk_engine.fills) == 1
    fill_payload = risk_engine.fills[0]
    assert fill_payload["profile_name"] == "nested"
    assert fill_payload["side"] == "buy"
    assert fill_payload["symbol"] == "SOLUSDT"
    assert fill_payload["pnl"] == pytest.approx(18.0)
    assert fill_payload["timestamp"] == datetime(
        2023, 11, 14, 22, 13, 20, 500000, tzinfo=timezone.utc
    )
    assert fill_payload["position_value"] == pytest.approx(8039.504)

    entry = decision_log.tail(limit=1)[0]
    metrics = entry["metadata"]["metrics"]
    assert metrics["avg_price"] == pytest.approx(25123.45)
    assert metrics["filled_quantity"] == pytest.approx(0.32)
    assert metrics["notional"] == pytest.approx(8039.504)
    assert metrics["pnl"] == pytest.approx(18.0)
    assert metrics["position_value"] == pytest.approx(8039.504)
    assert metrics["position_delta"] == pytest.approx(8039.504)
    assert entry["metadata"]["fill_timestamp"] == datetime(
        2023, 11, 14, 22, 13, 20, 500000, tzinfo=timezone.utc
    ).isoformat()


def test_post_core_fill_records_fee_metrics_from_nested_sources() -> None:
    emitter = DummyEmitter()
    gui = DummyGUI()
    adapter = RecordingExecutionAdapter()
    trader = AutoTrader(
        emitter,
        gui,
        lambda: "AVAX/USDT",
        auto_trade_interval_s=0.5,
        walkforward_interval_s=None,
        signal_service=SignalService(),
        risk_service=RiskService(),
        execution_service=ExecutionService(adapter),
        data_provider=StubDataProvider(),
    )

    decision_log = RiskDecisionLog(clock=lambda: datetime(2024, 1, 8, tzinfo=timezone.utc))
    risk_engine = RecordingRiskEngine(decision_log=decision_log)
    trader._core_risk_engine = risk_engine
    trader._core_risk_profile = "fees"

    raw_response = {
        "fills": [
            {
                "feeAmount": "1.25",
                "feeCurrency": "USDT",
            }
        ]
    }
    metadata = {
        "decision_candidate": {"risk_profile": "fees"},
        "settlement": {"commissionAsset": "USDT"},
    }
    order_request = SimpleNamespace(
        symbol="AVAXUSDT",
        side="buy",
        quantity=0.75,
        price=125.5,
        metadata=metadata,
    )
    order_result = SimpleNamespace(
        avg_price=125.5,
        filled_quantity=0.75,
        raw_response=raw_response,
    )

    trader._post_core_fill("AVAXUSDT", "BUY", order_request, order_result)

    entry = decision_log.tail(limit=1)[0]
    metrics = entry["metadata"]["metrics"]
    assert metrics["avg_price"] == pytest.approx(125.5)
    assert metrics["filled_quantity"] == pytest.approx(0.75)
    assert metrics["notional"] == pytest.approx(125.5 * 0.75)
    assert metrics["position_value"] == pytest.approx(125.5 * 0.75)
    assert metrics["fee"] == pytest.approx(1.25)
    assert metrics["fee_currency"] == "USDT"
    assert metrics["position_delta"] == pytest.approx(125.5 * 0.75)


def test_post_core_fill_handles_nested_fee_objects() -> None:
    emitter = DummyEmitter()
    gui = DummyGUI()
    adapter = RecordingExecutionAdapter()
    trader = AutoTrader(
        emitter,
        gui,
        lambda: "XRP/USDT",
        auto_trade_interval_s=0.5,
        walkforward_interval_s=None,
        signal_service=SignalService(),
        risk_service=RiskService(),
        execution_service=ExecutionService(adapter),
        data_provider=StubDataProvider(),
    )

    decision_log = RiskDecisionLog(clock=lambda: datetime(2024, 3, 12, tzinfo=timezone.utc))
    risk_engine = RecordingRiskEngine(decision_log=decision_log)
    trader._core_risk_engine = risk_engine
    trader._core_risk_profile = "fees-nested"

    raw_response = {
        "fills": [
            {
                "fee": {"cost": "0.75", "currency": "USDT"},
                "commission": {"amount": "0.75", "asset": "USDT"},
            }
        ],
        "settlement": {"fees": [{"value": "0.75", "currency": "USDT"}]},
    }
    metadata = {
        "decision_candidate": {"risk_profile": "fees-nested"},
        "audit": {"fees": [{"amount": "0.75", "currency": "USDT"}]},
    }
    order_request = SimpleNamespace(
        symbol="XRPUSDT",
        side="buy",
        quantity=10.0,
        price=0.5,
        metadata=metadata,
    )
    order_result = SimpleNamespace(
        avg_price=0.5,
        filled_quantity=10.0,
        raw_response=raw_response,
    )

    trader._post_core_fill("XRPUSDT", "BUY", order_request, order_result)

    entry = decision_log.tail(limit=1)[0]
    metrics = entry["metadata"]["metrics"]
    assert metrics["fee"] == pytest.approx(0.75)
    assert metrics["fee_currency"] == "USDT"
    assert metrics["notional"] == pytest.approx(5.0)
    assert metrics["position_delta"] == pytest.approx(5.0)


def test_post_core_fill_logs_fee_rate_and_liquidity_role() -> None:
    emitter = DummyEmitter()
    gui = DummyGUI()
    adapter = RecordingExecutionAdapter()
    trader = AutoTrader(
        emitter,
        gui,
        lambda: "LTC/USDT",
        auto_trade_interval_s=0.5,
        walkforward_interval_s=None,
        signal_service=SignalService(),
        risk_service=RiskService(),
        execution_service=ExecutionService(adapter),
        data_provider=StubDataProvider(),
    )

    decision_log = RiskDecisionLog(clock=lambda: datetime(2024, 4, 9, tzinfo=timezone.utc))
    risk_engine = RecordingRiskEngine(decision_log=decision_log)
    trader._core_risk_engine = risk_engine
    trader._core_risk_profile = "fee-role"

    raw_response = {
        "fills": [
            {"isMaker": True, "feeRate": "0.0004"},
            {"details": {"makerFlag": "false"}},
        ],
        "execution": {"liquidityType": "maker"},
    }
    metadata = {"decision_candidate": {"risk_profile": "fee-role"}}

    order_request = SimpleNamespace(
        symbol="LTCUSDT",
        side="buy",
        quantity=1.5,
        price=90.0,
        metadata=metadata,
    )
    order_result = SimpleNamespace(
        avg_price=90.0,
        filled_quantity=1.5,
        raw_response=raw_response,
    )

    trader._post_core_fill("LTCUSDT", "BUY", order_request, order_result)

    entry = decision_log.tail(limit=1)[0]
    metadata_payload = entry["metadata"]
    metrics = metadata_payload["metrics"]
    assert metrics["fee_rate"] == pytest.approx(0.0004)
    assert metadata_payload["liquidity"] == "maker"


def test_post_core_fill_collects_identifier_fields() -> None:
    emitter = DummyEmitter()
    gui = DummyGUI()
    adapter = RecordingExecutionAdapter()
    trader = AutoTrader(
        emitter,
        gui,
        lambda: "BNB/USDT",
        auto_trade_interval_s=0.5,
        walkforward_interval_s=None,
        signal_service=SignalService(),
        risk_service=RiskService(),
        execution_service=ExecutionService(adapter),
        data_provider=StubDataProvider(),
    )

    decision_log = RiskDecisionLog(clock=lambda: datetime(2024, 2, 11, tzinfo=timezone.utc))
    risk_engine = RecordingRiskEngine(decision_log=decision_log)
    trader._core_risk_engine = risk_engine
    trader._core_risk_profile = "identifiers"

    raw_response = {
        "order": {
            "orderId": "ord-1001",
            "fills": [
                {
                    "tradeId": 874563,
                    "execId": "fill-009",
                }
            ],
        }
    }
    metadata = {
        "client": {
            "clientOrderId": "cli-7788",
        }
    }

    order_request = SimpleNamespace(
        symbol="BNBUSDT",
        side="buy",
        quantity=2.0,
        price=315.25,
        metadata=metadata,
    )
    order_result = SimpleNamespace(
        avg_price=315.25,
        filled_quantity=2.0,
        raw_response=raw_response,
        order_id="ord-1001",
    )

    trader._post_core_fill("BNBUSDT", "BUY", order_request, order_result)

    entry = decision_log.tail(limit=1)[0]
    identifiers = entry["metadata"].get("identifiers")
    assert identifiers is not None
    assert identifiers["order_id"] == "ord-1001"
    assert identifiers["client_order_id"] == "cli-7788"
    assert identifiers["trade_id"] == "874563"
    assert identifiers["fill_id"] == "fill-009"


def test_post_core_fill_prefers_explicit_position_metrics() -> None:
    emitter = DummyEmitter()
    gui = DummyGUI()
    adapter = RecordingExecutionAdapter()
    trader = AutoTrader(
        emitter,
        gui,
        lambda: "DOGE/USDT",
        auto_trade_interval_s=0.5,
        walkforward_interval_s=None,
        signal_service=SignalService(),
        risk_service=RiskService(),
        execution_service=ExecutionService(adapter),
        data_provider=StubDataProvider(),
    )

    decision_log = RiskDecisionLog(clock=lambda: datetime(2024, 3, 3, tzinfo=timezone.utc))
    risk_engine = RecordingRiskEngine(decision_log=decision_log)
    trader._core_risk_engine = risk_engine
    trader._core_risk_profile = "position-explicit"

    raw_response = {
        "details": {
            "metrics": {
                "positionValue": "9876.5",
            }
        },
        "changes": [
            {"positionChange": "-9876.5"},
            {"other": None},
        ],
    }
    metadata = {
        "decision_candidate": {"risk_profile": "position-explicit"},
        "position": {"position_notional": "4321.0"},
    }

    order_request = SimpleNamespace(
        symbol="DOGEUSDT",
        side="sell",
        quantity=2_000.0,
        price=0.15,
        metadata=metadata,
    )
    order_result = SimpleNamespace(
        avg_price=0.15,
        filled_quantity=2_000.0,
        raw_response=raw_response,
    )

    trader._post_core_fill("DOGEUSDT", "SELL", order_request, order_result)

    assert len(risk_engine.fills) == 1
    payload = risk_engine.fills[0]
    assert payload["profile_name"] == "position-explicit"
    assert payload["symbol"] == "DOGEUSDT"
    assert payload["side"] == "sell"
    assert payload["position_value"] == pytest.approx(9876.5)

    entry = decision_log.tail(limit=1)[0]
    metrics = entry["metadata"]["metrics"]
    assert metrics["position_value"] == pytest.approx(9876.5)
    assert metrics["position_delta"] == pytest.approx(-9876.5)
    # Notional calculation should remain unaffected by explicit overrides.
    assert metrics["notional"] == pytest.approx(0.15 * 2_000.0)


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
