from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Mapping

import pandas as pd

from bot_core.auto_trader import (
    AutoTrader,
    AutoTraderDecisionScheduler,
    AutoTraderLifecycleManager,
    DecisionAuditLog,
    GuardrailTrigger,
)
from bot_core.ai.regime import MarketRegime, MarketRegimeAssessment
from bot_core.config.models import DecisionEngineConfig, DecisionOrchestratorThresholds
from bot_core.decision.orchestrator import DecisionOrchestrator
from bot_core.execution.bridge import ExchangeAdapterExecutionService
from bot_core.exchanges.base import ExchangeAdapter, ExchangeCredentials, OrderRequest, OrderResult
from bot_core.runtime.journal import InMemoryTradingDecisionJournal, aggregate_decision_statistics


class _Emitter:
    def __init__(self) -> None:
        self.events: list[tuple[str, Mapping[str, Any]]] = []

    def log(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def emit(self, event: str, **payload: Any) -> None:
        self.events.append((event, payload))


class _Var:
    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        return self._value


class _GUI:
    def __init__(self) -> None:
        self.timeframe_var = _Var("1h")
        self._demo = True
        self.ai_mgr = None

    def is_demo_mode_active(self) -> bool:
        return self._demo


class _MarketDataProvider:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def get_historical(self, symbol: str, timeframe: str, limit: int = 256) -> pd.DataFrame:
        del symbol, timeframe, limit
        return self.frame


class _AIManagerStub:
    ai_threshold_bps = 5.0
    is_degraded = False

    def assess_market_regime(self, symbol: str, market_data: pd.DataFrame) -> MarketRegimeAssessment:
        del symbol, market_data
        return MarketRegimeAssessment(
            regime=MarketRegime.TREND,
            confidence=0.9,
            risk_score=0.2,
            metrics={"atr": 1.0},
        )

    def get_regime_summary(self, symbol: str) -> None:
        del symbol
        return None

    def predict_series(self, symbol: str, df: pd.DataFrame) -> pd.Series:
        del symbol
        index = df.index[-1:]
        return pd.Series([0.02], index=index)

    def build_decision_engine_payload(
        self,
        *,
        strategy: str,
        action: str,
        risk_profile: str,
        symbol: str,
        notional: float,
        features: Mapping[str, float],
    ) -> Mapping[str, object]:
        return {
            "candidate": {
                "strategy": strategy,
                "action": action,
                "risk_profile": risk_profile,
                "symbol": symbol,
                "notional": notional,
                "features": dict(features),
            },
            "ai": {
                "direction": "buy",
                "prediction_bps": 25.0,
                "threshold_bps": self.ai_threshold_bps,
            },
        }


class _RiskServiceStub:
    def __init__(self) -> None:
        self.decisions: list[Any] = []

    def evaluate_decision(self, decision: Any) -> Any:
        self.decisions.append(decision)
        return SimpleNamespace(approved=True)

    def attach_decision_orchestrator(self, orchestrator: DecisionOrchestrator) -> None:
        self._orchestrator = orchestrator

    def snapshot_state(self, profile: str) -> Mapping[str, Any]:
        return {
            "profile": profile,
            "positions": {},
            "start_of_day_equity": 1_000_000.0,
            "last_equity": 1_000_000.0,
            "peak_equity": 1_000_000.0,
            "daily_realized_pnl": 0.0,
        }


@dataclass
class _RecordedOrder:
    request: OrderRequest
    timestamp: datetime


class _StubExchangeAdapter(ExchangeAdapter):
    def __init__(self) -> None:
        super().__init__(ExchangeCredentials(key_id="paper"))
        self.orders: list[_RecordedOrder] = []

    def configure_network(self, *, ip_allowlist=None) -> None:
        return None

    def fetch_account_snapshot(self):  # pragma: no cover - not required for the test
        raise NotImplementedError

    def fetch_symbols(self):  # pragma: no cover - not used
        return []

    def fetch_ohlcv(self, symbol, interval, start=None, end=None, limit=None):  # pragma: no cover - not used
        return []

    def place_order(self, request: OrderRequest) -> OrderResult:
        self.orders.append(_RecordedOrder(request=request, timestamp=datetime.now(timezone.utc)))
        price = request.price if request.price is not None else 100.0
        return OrderResult(
            order_id=f"paper-{len(self.orders)}",
            status="filled",
            filled_quantity=request.quantity,
            avg_price=price,
            raw_response={"fee": 0.0, "fee_asset": "USDT"},
        )

    def cancel_order(self, *_args: Any, **_kwargs: Any) -> bool:  # pragma: no cover - not used in test
        return True

    def stream_public_data(self, *_args: Any, **_kwargs: Any):  # pragma: no cover - not used
        if False:
            yield None

    def stream_private_data(self, *_args: Any, **_kwargs: Any):  # pragma: no cover - not used
        if False:
            yield None


class _AlertRouterStub:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    def dispatch(self, message: Any) -> None:
        self.messages.append(message)

    def cancel_order(self, order_id: str, *, symbol=None) -> None:  # pragma: no cover - not used
        return None

    def stream_public_data(self, *, channels):  # pragma: no cover - not used
        raise NotImplementedError

    def stream_private_data(self, *, channels):  # pragma: no cover - not used
        raise NotImplementedError


def _decision_engine_config() -> DecisionEngineConfig:
    thresholds = DecisionOrchestratorThresholds(
        max_cost_bps=50.0,
        min_net_edge_bps=-10.0,
        max_daily_loss_pct=0.5,
        max_drawdown_pct=0.6,
        max_position_ratio=100.0,
        max_open_positions=100,
        max_latency_ms=500.0,
    )
    return DecisionEngineConfig(
        orchestrator=thresholds,
        profile_overrides={},
        stress_tests=None,
        min_probability=0.0,
        require_cost_data=False,
        penalty_cost_bps=0.0,
    )


def _market_frame() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=120, freq="min")
    base = pd.Series(range(len(idx)), index=idx, dtype="float64")
    return pd.DataFrame(
        {
            "open": base + 100.0,
            "high": base + 101.0,
            "low": base + 99.0,
            "close": base + 100.5,
            "volume": 1_000.0,
        }
    )


def test_auto_trader_scheduler_with_exchange_bridge() -> None:
    emitter = _Emitter()
    gui = _GUI()
    ai_manager = _AIManagerStub()
    gui.ai_mgr = ai_manager
    market_provider = _MarketDataProvider(_market_frame())
    orchestrator = DecisionOrchestrator(_decision_engine_config())
    risk_service = _RiskServiceStub()
    adapter = _StubExchangeAdapter()
    journal = InMemoryTradingDecisionJournal()
    execution_service = ExchangeAdapterExecutionService(
        adapter=lambda: adapter,
        journal=journal,
        backoff_base=0.0,
    )

    bootstrap = SimpleNamespace(
        decision_orchestrator=orchestrator,
        risk_engine=risk_service,
        decision_engine_config=_decision_engine_config(),
        risk_profile_name="paper",
        portfolio_id="demo",
    )

    trader = AutoTrader(
        emitter,
        gui,
        lambda: "BTCUSDT",
        market_data_provider=market_provider,
        risk_service=risk_service,
        execution_service=execution_service,
        bootstrap_context=bootstrap,
        decision_journal=journal,
    )

    scheduler = AutoTraderDecisionScheduler(trader, interval_s=0.01)
    scheduler.start_in_background()
    time.sleep(0.05)
    scheduler.stop_background()

    assert adapter.orders, "scheduler should submit at least one order"
    summary = aggregate_decision_statistics(journal)
    assert summary["by_status"].get("filled", 0) >= 1
    assert summary["by_status"].get("approved", 0) >= 1
    assert summary["by_status"].get("trade", 0) >= 1
    assert "BTCUSDT" in summary["by_symbol"]
    assert risk_service.decisions, "risk service should have evaluated decisions"
    events = list(journal.export())
    assert any(entry.get("event") == "decision_composed" for entry in events)
    assert any(entry.get("event") == "risk_evaluated" for entry in events)


def test_autonomous_lifecycle_runs_247_flow() -> None:
    emitter = _Emitter()
    gui = _GUI()
    ai_manager = _AIManagerStub()
    gui.ai_mgr = ai_manager
    market_provider = _MarketDataProvider(_market_frame())
    orchestrator = DecisionOrchestrator(_decision_engine_config())
    risk_service = _RiskServiceStub()
    adapter = _StubExchangeAdapter()
    journal = InMemoryTradingDecisionJournal()
    audit_log = DecisionAuditLog()
    alert_router = _AlertRouterStub()
    execution_service = ExchangeAdapterExecutionService(
        adapter=lambda: adapter,
        journal=journal,
        backoff_base=0.0,
    )

    bootstrap = SimpleNamespace(
        decision_orchestrator=orchestrator,
        risk_engine=risk_service,
        decision_engine_config=_decision_engine_config(),
        risk_profile_name="paper",
        portfolio_id="demo",
        alert_router=alert_router,
    )

    trader = AutoTrader(
        emitter,
        gui,
        lambda: "BTCUSDT",
        market_data_provider=market_provider,
        risk_service=risk_service,
        execution_service=execution_service,
        bootstrap_context=bootstrap,
        decision_journal=journal,
        decision_audit_log=audit_log,
        auto_trade_interval_s=0.0,
    )
    trader.ai_manager = ai_manager
    trader.alert_router = alert_router

    guardrail_counter = {"count": 0}

    def _inject_guardrail(self: AutoTrader, signal: str, *_args: object, **_kwargs: object) -> str:
        if signal in {"buy", "sell"} and guardrail_counter["count"] == 0:
            guardrail_counter["count"] += 1
            self._last_guardrail_reasons = ["exchange degradation"]  # type: ignore[attr-defined]
            self._last_guardrail_triggers = [  # type: ignore[attr-defined]
                GuardrailTrigger(
                    name="exchange_degradation",
                    label="Exchange degradation",
                    comparator=">=",
                    threshold=0.5,
                    unit="score",
                    value=0.9,
                )
            ]
            return "hold"
        return signal

    trader._apply_signal_guardrails = _inject_guardrail.__get__(trader, AutoTrader)  # type: ignore[assignment]

    scheduler = AutoTraderDecisionScheduler(trader, interval_s=0.01)
    lifecycle = AutoTraderLifecycleManager(trader, scheduler=scheduler)
    lifecycle.start()
    time.sleep(0.15)
    lifecycle.stop()

    events = list(journal.export())
    assert any(entry.get("event") == "scheduler_bootstrap" for entry in events)
    assert any(entry.get("event") == "decision_guardrail" for entry in events)
    assert any(entry.get("event") == "order_filled" for entry in events)
    assert adapter.orders, "expected trade execution after guardrail release"

    audit_entries = audit_log.query_dicts(stage="lifecycle_bootstrap")
    assert audit_entries, "expected lifecycle bootstrap audit entry"

    assert any(message.category == "auto_trader.guardrail" for message in alert_router.messages)
    assert getattr(trader, "_auto_trade_user_confirmed", False)
