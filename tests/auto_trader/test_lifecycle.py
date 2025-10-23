from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from bot_core.ai.regime import MarketRegime, MarketRegimeAssessment, RiskLevel
from bot_core.config.models import DecisionEngineConfig, DecisionOrchestratorThresholds
from bot_core.auto_trader.app import AutoTrader, RiskDecision
from bot_core.observability import metrics as metrics_module


class DummyEmitter:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def emit(self, event: str, **payload: object) -> None:
        self.events.append((event, dict(payload)))

    def log(self, *_args: object, **_kwargs: object) -> None:  # pragma: no cover - not used in tests
        return None


class SequenceMarketDataProvider:
    def __init__(self, frames: list[pd.DataFrame]) -> None:
        self._frames = deque(frames)

    def get_historical(self, symbol: str, timeframe: str, limit: int = 256) -> pd.DataFrame:
        frame = self._frames[0]
        self._frames.rotate(-1)
        return frame


class SequencedAIManager:
    def __init__(self, scenarios: list[dict[str, object]]) -> None:
        self._scenarios = deque(scenarios)
        self.ai_threshold_bps = 5.0
        self.is_degraded = False

    def assess_market_regime(self, symbol: str, market_data: pd.DataFrame) -> MarketRegimeAssessment:
        return self._scenarios[0]["assessment"]  # type: ignore[return-value]

    def get_regime_summary(self, symbol: str):
        return self._scenarios[0].get("summary")

    def predict_series(self, symbol: str, market_data: pd.DataFrame, feature_cols: list[str] | None = None) -> pd.Series:
        scenario = self._scenarios[0]
        index = market_data.index[-1:]
        series = pd.Series([scenario.get("prediction", 0.0)], index=index)
        self._scenarios.rotate(-1)
        return series


class RecordingExecutionService:
    def __init__(self) -> None:
        self.decisions: list[RiskDecision] = []

    def execute_decision(self, decision: RiskDecision) -> None:
        self.decisions.append(decision)


class StubAlertRouter:
    def __init__(self) -> None:
        self.messages: list = []

    def dispatch(self, message) -> None:  # pragma: no cover - simple collector
        self.messages.append(message)


@dataclass
class SimpleSchedule:
    strategy: str
    interval: object
    next_run: datetime


class SequencedOrchestrator:
    def __init__(self, strategies: list[str]) -> None:
        self._strategies = deque(strategies)
        self._schedules: list[SimpleSchedule] = []

    def select_strategy(self, regime: MarketRegime) -> str | None:
        if not self._strategies:
            return None
        return self._strategies.popleft()

    def schedule_strategy_recalibration(self, strategy: str, interval, *, first_run: datetime | None = None) -> SimpleSchedule:
        schedule = SimpleSchedule(strategy=strategy, interval=interval, next_run=first_run or datetime.now(timezone.utc))
        self._schedules.append(schedule)
        return schedule

    def due_recalibrations(self, now: datetime | None = None) -> tuple[SimpleSchedule, ...]:
        return tuple(self._schedules)

    def mark_recalibrated(self, strategy: str) -> None:
        self._schedules = [item for item in self._schedules if item.strategy != strategy]


class SummaryStub:
    def __init__(self, **values: object) -> None:
        self.__dict__.update(values)

    def __getattr__(self, item: str) -> object:  # pragma: no cover - fallback for optional metrics
        return self.__dict__.get(item, 0.0)

    def to_dict(self) -> dict[str, object]:  # pragma: no cover - helper for logging paths
        return dict(self.__dict__)


def _build_assessment(regime: MarketRegime, *, risk: float) -> MarketRegimeAssessment:
    return MarketRegimeAssessment(
        regime=regime,
        confidence=0.75,
        risk_score=risk,
        metrics={"volatility": 0.1},
    )


def test_auto_trader_lifecycle_with_guardrails_and_recalibrations() -> None:
    metrics_module._GLOBAL_REGISTRY = metrics_module.MetricsRegistry()

    frames = [
        pd.DataFrame({"close": [100, 101, 102, 103, 104]}, index=pd.date_range("2024-01-01", periods=5, freq="h")),
        pd.DataFrame({"close": [105, 104, 103, 102, 101]}, index=pd.date_range("2024-01-02", periods=5, freq="h")),
        pd.DataFrame({"close": [100, 101, 100, 99, 98]}, index=pd.date_range("2024-01-03", periods=5, freq="h")),
    ]

    scenarios = [
        {
            "assessment": _build_assessment(MarketRegime.TREND, risk=0.2),
            "summary": None,
            "prediction": 0.003,
        },
        {
            "assessment": _build_assessment(MarketRegime.MEAN_REVERSION, risk=0.35),
            "summary": None,
            "prediction": 0.004,
        },
        {
            "assessment": _build_assessment(MarketRegime.TREND, risk=0.62),
            "summary": None,
            "prediction": -0.005,
        },
    ]

    base_thresholds = DecisionOrchestratorThresholds(
        max_cost_bps=250.0,
        min_net_edge_bps=-50.0,
        max_daily_loss_pct=1.5,
        max_drawdown_pct=2.0,
        max_position_ratio=12.0,
        max_open_positions=20,
        max_latency_ms=1500.0,
    )
    paper_thresholds = DecisionOrchestratorThresholds(
        max_cost_bps=200.0,
        min_net_edge_bps=10.0,
        max_daily_loss_pct=1.0,
        max_drawdown_pct=1.5,
        max_position_ratio=8.0,
        max_open_positions=10,
        max_latency_ms=1200.0,
        max_trade_notional=50_000.0,
    )
    orchestrator = SequencedOrchestrator([
        "trend_following",
        "mean_reversion",
        "capital_preservation",
    ])

    provider = SequenceMarketDataProvider(frames)
    ai_manager = SequencedAIManager(scenarios)
    emitter = DummyEmitter()
    alert_router = StubAlertRouter()
    execution_service = RecordingExecutionService()

    decision_config = DecisionEngineConfig(
        orchestrator=base_thresholds,
        profile_overrides={"paper": paper_thresholds},
        min_probability=0.35,
        require_cost_data=True,
        penalty_cost_bps=12.5,
        evaluation_history_limit=128,
    )

    bootstrap_context = SimpleNamespace(
        risk_profile_name="paper",
        portfolio_id="autotrader",
        environment="paper",
        decision_engine_config=decision_config,
        execution_service=execution_service,
        alert_router=alert_router,
    )

    trader = AutoTrader(
        emitter=emitter,
        gui=SimpleNamespace(is_demo_mode_active=lambda: False),
        symbol_getter=lambda: "BTCUSDT",
        market_data_provider=provider,
        enable_auto_trade=True,
        auto_trade_interval_s=0.0,
        bootstrap_context=bootstrap_context,
    )
    trader.ai_manager = ai_manager
    trader.decision_orchestrator = orchestrator
    trader.alert_router = alert_router
    trader.execution_service = execution_service
    trader.core_execution_service = execution_service
    trader.core_risk_engine = None
    trader.risk_service = None
    trader._adjust_strategy_parameters = lambda *args, **kwargs: None  # type: ignore[assignment]
    trader._evaluate_decision_candidate = lambda **kwargs: SimpleNamespace(  # type: ignore[assignment]
        accepted=True,
        reasons=(),
        thresholds_snapshot={},
        model_name="stub",
    )

    trader._thresholds["auto_trader"]["signal_guardrails"] = {"effective_risk_cap": 0.5}

    trader.schedule_strategy_recalibration("mean_reversion", interval_s=0.0)

    for _ in range(3):
        trader._auto_trade_loop()

    assert len(execution_service.decisions) == 2
    assert trader.current_strategy == "capital_preservation"

    registry = metrics_module.get_global_metrics_registry()
    guardrail_metric = registry.get("auto_trader_guardrail_blocks_total")
    guardrail_names = {trigger.name for trigger in trader._last_guardrail_triggers}
    assert trader._last_guardrail_reasons, f"guardrail reasons missing (names={guardrail_names})"
    assert guardrail_metric.value(
        labels={
            "environment": "paper",
            "portfolio": "autotrader",
            "risk_profile": "paper",
            "guardrail": "effective_risk",
        }
    ) == 1.0

    recalibration_metric = registry.get("auto_trader_recalibrations_triggered_total")
    assert recalibration_metric.value(
        labels={
            "environment": "paper",
            "portfolio": "autotrader",
            "risk_profile": "paper",
            "strategy": "mean_reversion",
        }
    ) == 1.0

    categories = {message.category for message in alert_router.messages}
    assert "auto_trader.guardrail" in categories
    assert "auto_trader.recalibration" in categories

    assert orchestrator.due_recalibrations() == ()

    snapshot = trader.build_lifecycle_snapshot(bucket_s=3600.0, tz=timezone.utc)
    assert snapshot["symbol"] == "BTCUSDT"
    assert snapshot["strategy"]["current"] == "capital_preservation"
    assert snapshot["guardrails"]["summary"]["total"] >= 1
    assert snapshot["guardrails"]["last_reasons"]
    assert snapshot["risk_decisions"]["history_size"] == len(trader._risk_evaluations)
    thresholds_snapshot = snapshot["risk_decisions"]["thresholds"]
    assert thresholds_snapshot["risk_profile"] == "paper"
    assert thresholds_snapshot["source"] == "profile_override"
    assert thresholds_snapshot["max_trade_notional"] == pytest.approx(50_000.0)
    assert thresholds_snapshot["min_probability"] == pytest.approx(0.35)
    assert thresholds_snapshot["require_cost_data"] is True
    assert thresholds_snapshot["evaluation_history_limit"] == 128
    assert snapshot["metrics"]["cycles_total"] == pytest.approx(3.0)
    assert snapshot["metrics"]["guardrail_blocks_total"] >= 1.0
    assert not snapshot["cooldown"]["active"]


def test_auto_trader_guardrail_timeline_for_24_7_regime_rotation() -> None:
    metrics_module._GLOBAL_REGISTRY = metrics_module.MetricsRegistry()

    total_iterations = 24
    guardrail_start = 8
    base_timestamp = datetime(2024, 2, 1, tzinfo=timezone.utc)

    frames = [
        pd.DataFrame(
            {"close": [100 + idx, 100 + idx + 1, 100 + idx + 2, 100 + idx + 3, 100 + idx + 4]},
            index=pd.date_range(base_timestamp + timedelta(hours=idx), periods=5, freq="h"),
        )
        for idx in range(total_iterations)
    ]

    regimes = [
        MarketRegime.TREND,
        MarketRegime.MEAN_REVERSION,
        MarketRegime.DAILY,
    ]
    scenarios: list[dict[str, object]] = []
    for idx in range(total_iterations):
        risk = 0.35 if idx < guardrail_start else 0.62
        prediction = 0.005 if idx % 2 == 0 else -0.004
        scenarios.append(
            {
                "assessment": _build_assessment(regimes[idx % len(regimes)], risk=risk),
                "summary": None,
                "prediction": prediction,
            }
        )

    base_thresholds = DecisionOrchestratorThresholds(
        max_cost_bps=150.0,
        min_net_edge_bps=-25.0,
        max_daily_loss_pct=2.5,
        max_drawdown_pct=4.0,
        max_position_ratio=15.0,
        max_open_positions=30,
        max_latency_ms=1000.0,
    )
    active_thresholds = DecisionOrchestratorThresholds(
        max_cost_bps=120.0,
        min_net_edge_bps=5.0,
        max_daily_loss_pct=1.2,
        max_drawdown_pct=2.2,
        max_position_ratio=6.0,
        max_open_positions=8,
        max_latency_ms=800.0,
    )

    orchestrator = SequencedOrchestrator(
        [
            "trend_following",
            "mean_reversion",
            "volatility_breakout",
            "intraday_breakout",
        ]
        * (guardrail_start // 4)
    )

    provider = SequenceMarketDataProvider(frames)
    ai_manager = SequencedAIManager(scenarios)
    emitter = DummyEmitter()
    alert_router = StubAlertRouter()
    execution_service = RecordingExecutionService()

    decision_config = DecisionEngineConfig(
        orchestrator=base_thresholds,
        profile_overrides={"active": active_thresholds},
        min_probability=0.42,
        require_cost_data=False,
        penalty_cost_bps=8.0,
        evaluation_history_limit=256,
    )

    bootstrap_context = SimpleNamespace(
        execution_service=execution_service,
        alert_router=alert_router,
        environment="live",
        portfolio_id="autotrader_live",
        risk_profile_name="active",
        decision_engine_config=decision_config,
    )

    trader = AutoTrader(
        emitter=emitter,
        gui=SimpleNamespace(is_demo_mode_active=lambda: False),
        symbol_getter=lambda: "BTCUSDT",
        market_data_provider=provider,
        enable_auto_trade=True,
        auto_trade_interval_s=0.0,
        bootstrap_context=bootstrap_context,
    )
    trader.ai_manager = ai_manager
    trader.decision_orchestrator = orchestrator
    trader._attach_decision_orchestrator()
    trader.alert_router = alert_router
    trader._adjust_strategy_parameters = lambda *args, **kwargs: None  # type: ignore[assignment]
    trader._evaluate_decision_candidate = lambda **kwargs: SimpleNamespace(  # type: ignore[assignment]
        accepted=True,
        reasons=(),
        thresholds_snapshot={},
        model_name="stub",
    )
    trader.risk_service = lambda decision: {"approved": decision.should_trade}
    trader._thresholds["auto_trader"]["signal_guardrails"] = {"effective_risk_cap": 0.5}

    orchestrator.schedule_strategy_recalibration(
        "mean_reversion",
        interval=timedelta(hours=6),
        first_run=base_timestamp + timedelta(hours=guardrail_start // 2),
    )

    for _ in range(total_iterations):
        trader._auto_trade_loop()

    guardrail_iterations = total_iterations - guardrail_start

    trade_evaluations = sum(
        1
        for entry in trader._risk_evaluations
        if bool(entry.get("decision", {}).get("should_trade"))
    )
    assert len(execution_service.decisions) == trade_evaluations
    assert 0 < trade_evaluations < total_iterations

    registry = metrics_module.get_global_metrics_registry()
    guardrail_metric = registry.get("auto_trader_guardrail_blocks_total")
    assert guardrail_metric.value(
        labels={
            "environment": "live",
            "portfolio": "autotrader_live",
            "risk_profile": "active",
            "guardrail": "effective_risk",
        }
    ) == float(guardrail_iterations)

    recalibration_metric = registry.get("auto_trader_recalibrations_triggered_total")
    assert recalibration_metric.value(
        labels={
            "environment": "live",
            "portfolio": "autotrader_live",
            "risk_profile": "active",
            "strategy": "mean_reversion",
        }
    ) == 1.0

    with trader._lock:
        for index, entry in enumerate(trader._risk_evaluations):
            entry["timestamp"] = (base_timestamp + timedelta(hours=index)).timestamp()

    evaluations = list(trader._risk_evaluations)
    assert len(evaluations) == total_iterations
    guardrail_evaluations = sum(
        1
        for entry in evaluations
        if entry.get("decision", {})
        .get("details", {})
        .get("guardrail_reasons")
    )
    assert guardrail_evaluations == guardrail_iterations
    guardrail_rate = guardrail_evaluations / len(evaluations)
    assert guardrail_rate == pytest.approx(
        guardrail_iterations / len(evaluations), rel=1e-6
    )

    first_ts = evaluations[0]["timestamp"]
    last_ts = evaluations[-1]["timestamp"]
    expected_span = max(0, len(evaluations) - 1) * 3600.0
    assert last_ts - first_ts >= expected_span - 1e-6

    guardrail_messages = [
        message for message in alert_router.messages if message.category == "auto_trader.guardrail"
    ]
    assert len(guardrail_messages) == guardrail_iterations
    assert any(message.category == "auto_trader.recalibration" for message in alert_router.messages)

    assert trader.current_strategy == "capital_preservation"

    snapshot = trader.build_lifecycle_snapshot(bucket_s=3600.0, tz=timezone.utc)
    assert snapshot["environment"] == "live"
    assert snapshot["guardrails"]["summary"]["total"] == guardrail_iterations
    assert snapshot["metrics"]["guardrail_blocks_total"] == pytest.approx(
        float(guardrail_iterations)
    )
    assert snapshot["risk_decisions"]["history_size"] == len(trader._risk_evaluations)
    thresholds_snapshot = snapshot["risk_decisions"]["thresholds"]
    assert thresholds_snapshot["risk_profile"] == "active"
    assert thresholds_snapshot["source"] == "profile_override"
    assert thresholds_snapshot["max_cost_bps"] == pytest.approx(120.0)
    assert thresholds_snapshot["penalty_cost_bps"] == pytest.approx(8.0)
    assert thresholds_snapshot["require_cost_data"] is False
    assert snapshot["metrics"]["cycles_total"] == pytest.approx(float(total_iterations))
    assert snapshot["controller"]["auto_trade"]["active"] is False
