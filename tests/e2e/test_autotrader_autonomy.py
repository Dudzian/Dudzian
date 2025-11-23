from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import threading
from pathlib import Path

import pandas as pd
import pytest

from bot_core.ai.regime import MarketRegime, MarketRegimeAssessment
from bot_core.auto_trader import AutoTrader, AutoTraderAIGovernorRunner, ScheduleState
from bot_core.auto_trader.audit import DecisionAuditLog
from bot_core.execution import ExecutionContext, ExecutionService
from bot_core.runtime.journal import InMemoryTradingDecisionJournal
from bot_core.config.models import DecisionEngineConfig, DecisionOrchestratorThresholds
from bot_core.decision.orchestrator import DecisionOrchestrator

from tests.e2e.fixtures import FakeExecutionService


class _Emitter:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def emit(self, name: str, **payload: object) -> None:
        self.events.append((name, dict(payload)))


@dataclass(slots=True)
class _StaticAIManager:
    assessment: MarketRegimeAssessment
    prediction: float
    probability: float
    model_name: str = "static_model"

    def assess_market_regime(self, symbol: str, market_data: pd.DataFrame) -> MarketRegimeAssessment:
        return self.assessment

    def get_regime_summary(self, symbol: str) -> None:
        return None

    def predict_series(self, symbol: str, market_data: pd.DataFrame) -> pd.Series:
        return pd.Series([self.prediction] * len(market_data), index=market_data.index)

    def prediction_probability(self, symbol: str, market_data: pd.DataFrame) -> float:
        return self.probability

    def get_active_model(self, symbol: str) -> str:
        return self.model_name


def _market_data(rows: int = 120) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="h")
    prices = pd.Series(100.0 + pd.Series(range(rows)).rolling(5, min_periods=1).mean(), index=index)
    return pd.DataFrame({
        "open": prices,
        "high": prices + 2.0,
        "low": prices - 2.0,
        "close": prices + 1.0,
        "volume": 1000 + pd.Series(range(rows), index=index),
    })


def _build_runner() -> AutoTraderAIGovernorRunner:
    thresholds = DecisionOrchestratorThresholds(
        max_cost_bps=15.0,
        min_net_edge_bps=4.0,
        max_daily_loss_pct=0.02,
        max_drawdown_pct=0.08,
        max_position_ratio=0.35,
        max_open_positions=5,
        max_latency_ms=250.0,
    )
    config = DecisionEngineConfig(
        orchestrator=thresholds,
        profile_overrides={},
        stress_tests=None,
        min_probability=0.55,
        require_cost_data=False,
        penalty_cost_bps=0.0,
    )
    orchestrator = DecisionOrchestrator(config)
    orchestrator.record_strategy_performance(
        "scalping_alpha",
        MarketRegime.TREND,
        hit_rate=0.78,
        pnl=15.0,
        sharpe=1.2,
    )
    orchestrator.record_strategy_performance(
        "hedge_guardian",
        MarketRegime.MEAN_REVERSION,
        hit_rate=0.6,
        pnl=6.0,
        sharpe=0.55,
    )
    orchestrator.record_strategy_performance(
        "grid_balanced",
        MarketRegime.DAILY,
        hit_rate=0.64,
        pnl=8.5,
        sharpe=0.7,
    )
    return AutoTraderAIGovernorRunner(orchestrator)


def _build_trader(
    ai_manager: _StaticAIManager,
    *,
    execution_service: ExecutionService | None = None,
    environment: str = "paper",
    symbol: str = "BTCUSDT",
) -> tuple[
    AutoTrader,
    InMemoryTradingDecisionJournal,
    _Emitter,
    DecisionAuditLog,
    ExecutionContext,
]:
    emitter = _Emitter()
    journal = InMemoryTradingDecisionJournal()
    audit_log = DecisionAuditLog()
    gui = type(
        "GuiStub",
        (),
        {"ai_mgr": ai_manager, "portfolio_manager": None, "decision_journal": journal},
    )()
    trader = AutoTrader(
        emitter=emitter,
        gui=gui,
        symbol_getter=lambda: symbol,
        market_data_provider=lambda *_, **__: _market_data(),
        enable_auto_trade=True,
        trusted_auto_confirm=True,
        decision_audit_log=audit_log,
        decision_journal=journal,
        execution_service=execution_service,
    )
    trader.risk_service = None
    trader.core_risk_engine = None
    trader._environment_name = environment
    trader._base_metric_labels = {
        **trader._base_metric_labels,
        "environment": environment,
    }
    trader._apply_active_mode_overrides = lambda: None  # type: ignore[assignment]
    context = ExecutionContext(
        portfolio_id="autotrader",
        risk_profile="paper" if environment == "paper" else "live",
        environment=environment,
        metadata={},
    )
    return trader, journal, emitter, audit_log, context


def _open_schedule_state(mode: str) -> ScheduleState:
    return ScheduleState(
        mode=mode,
        is_open=True,
        window=None,
        next_transition=None,
        reference_time=datetime.now(timezone.utc),
    )


def test_autotrader_paper_switches_to_growth_profile() -> None:
    assessment = MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=0.86,
        risk_score=0.32,
        metrics={"trend_strength": 0.8},
        symbol="BTCUSDT",
    )
    ai_manager = _StaticAIManager(assessment=assessment, prediction=0.015, probability=0.71)
    trader, journal, emitter, _, context = _build_trader(ai_manager)

    report = trader.run_single_cycle(
        execution_context=context,
        schedule_state=_open_schedule_state(trader.schedule_mode),
    )

    assert trader._risk_profile_name == "aggressive"
    assert trader.current_strategy == "trend_following"
    assert report.metadata.get("decision_state") == "trade"
    assert journal.export(), "journal should capture decision events"
    assert any(event[0] == "auto_trader.decision_audit" for event in emitter.events)
    assert report.metrics["cycles_total"] >= 1
    assert trader._execution_context is None


def test_autotrader_live_enforces_conservative_profile_on_high_risk() -> None:
    assessment = MarketRegimeAssessment(
        regime=MarketRegime.DAILY,
        confidence=0.74,
        risk_score=0.88,
        metrics={"volatility": 0.12},
        symbol="ETHUSDT",
    )
    ai_manager = _StaticAIManager(assessment=assessment, prediction=0.0, probability=0.4)
    trader, journal, emitter, _, context = _build_trader(ai_manager)

    report = trader.run_single_cycle(
        execution_context=context,
        schedule_state=_open_schedule_state(trader.schedule_mode),
    )

    assert trader._risk_profile_name == "conservative"
    assert trader.current_strategy == "capital_preservation"
    assert report.metadata.get("decision_state") == "hold"
    decision_events = [event for event in journal.export() if event.get("event") == "decision_composed"]
    assert decision_events and decision_events[0].get("risk_profile") == "conservative"
    assert any(
        event[0] == "auto_trader.decision_audit" and event[1].get("stage") == "risk_profile_transition"
        for event in emitter.events
    )
    assert report.metrics["cycles_total"] >= 1


def test_autotrader_paper_executes_order_and_records_audit() -> None:
    assessment = MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=0.91,
        risk_score=0.28,
        metrics={"trend_strength": 0.9},
        symbol="BTCUSDT",
    )
    ai_manager = _StaticAIManager(assessment=assessment, prediction=0.022, probability=0.83)
    service = FakeExecutionService()
    trader, _, _, audit_log, context = _build_trader(ai_manager, execution_service=service)

    report = trader.run_single_cycle(
        execution_context=context,
        schedule_state=_open_schedule_state(trader.schedule_mode),
    )

    assert service.executed, "usługa egzekucji powinna zostać wywołana"
    recorded = service.executed[0]
    assert recorded.request.symbol == "BTCUSDT"
    assert recorded.request.quantity > 0
    assert recorded.context.environment == "paper"
    assert recorded.request.metadata is not None
    assert recorded.request.metadata.get("mode") == trader.schedule_mode
    assert report.decision is not None and report.decision.should_trade
    assert report.metrics["cycles_total"] >= 1

    stages = audit_log.to_dicts(limit=10)
    assert any(entry["stage"] == "execution_submitted" for entry in stages)
    assert any(
        entry["stage"] == "execution_submitted"
        and entry["payload"].get("order", {}).get("symbol") == "BTCUSDT"
        for entry in stages
    )


def test_autotrader_live_execution_failure_records_audit() -> None:
    assessment = MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=0.88,
        risk_score=0.35,
        metrics={"trend_strength": 0.85},
        symbol="ETHUSDT",
    )
    ai_manager = _StaticAIManager(assessment=assessment, prediction=0.018, probability=0.78)
    service = FakeExecutionService(should_fail=True, failure=RuntimeError("boom"))
    trader, _, _, audit_log, context = _build_trader(
        ai_manager,
        execution_service=service,
        environment="live",
        symbol="ETHUSDT",
    )

    report = trader.run_single_cycle(
        execution_context=context,
        schedule_state=_open_schedule_state(trader.schedule_mode),
    )

    assert service.executed, "nawet w przypadku błędu powinien wystąpić jeden attempt"
    recorded = service.executed[0]
    assert recorded.request.symbol == "ETHUSDT"
    assert recorded.context.environment == "live"
    assert report.decision is not None
    assert report.metrics["cycles_total"] >= 1
    assert trader._execution_context is None

    stages = audit_log.to_dicts(limit=10)
    assert any(entry["stage"] == "execution_failed" for entry in stages)
    assert not any(
        entry["stage"] == "execution_submitted"
        and entry["payload"].get("order", {}).get("symbol") == "ETHUSDT"
        for entry in stages
    )


def test_autotrader_ai_governor_snapshot_reports_mode() -> None:
    assessment = MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=0.9,
        risk_score=0.4,
        metrics={"trend_strength": 0.95},
        symbol="BTCUSDT",
    )
    ai_manager = _StaticAIManager(assessment=assessment, prediction=0.02, probability=0.82)
    trader, _, _, _, context = _build_trader(ai_manager)

    trader.run_single_cycle(
        execution_context=context,
        schedule_state=_open_schedule_state(trader.schedule_mode),
    )
    snapshot = trader.build_auto_mode_snapshot(include_history=False)
    governor = snapshot.get("ai_governor", {})

    assert governor.get("last_decision", {}).get("mode") in {"scalping", "grid", "hedge"}
    telemetry = governor.get("telemetry", {})
    cycle_metrics = telemetry.get("cycleMetrics", {})
    assert cycle_metrics, "powinny istnieć metryki cyklu"
    assert "strategy_switch_total" in cycle_metrics


def test_ai_governor_runner_handles_all_modes() -> None:
    runner = _build_runner()

    scalping_history = runner.run_until(mode="scalping")
    assert scalping_history[-1].mode == "scalping"

    hedge_history = runner.run_until(mode="hedge")
    assert hedge_history[-1].mode == "hedge"

    grid_history = runner.run_until(mode="grid")
    assert grid_history[-1].mode == "grid"

    snapshot = runner.snapshot()
    telemetry = snapshot.get("telemetry", {})
    assert telemetry.get("cycleMetrics", {}).get("cycles_total", 0.0) >= 3
    assert telemetry.get("riskMetrics", {}).get("risk_score") is not None


def test_autotrader_cycle_report_without_new_decision() -> None:
    assessment = MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=0.92,
        risk_score=0.22,
        metrics={"trend_strength": 0.88},
        symbol="BTCUSDT",
    )
    ai_manager = _StaticAIManager(assessment=assessment, prediction=0.02, probability=0.8)
    trader, _, _, _, context = _build_trader(ai_manager)

    trader._enforce_work_schedule = lambda *_args, **_kwargs: False  # type: ignore[assignment]
    before_cycles = trader._metric_cycle_total.value(labels=trader._base_metric_labels)

    report = trader.run_single_cycle(
        execution_context=context,
        schedule_state=_open_schedule_state(trader.schedule_mode),
    )

    assert report.decision is None
    assert report.metadata == {}
    assert report.metrics["cycles_total"] == before_cycles


def test_autotrader_cycle_report_exposes_telemetry() -> None:
    assessment = MarketRegimeAssessment(
        regime=MarketRegime.MEAN_REVERSION,
        confidence=0.81,
        risk_score=0.35,
        metrics={"volatility": 0.08},
        symbol="ADAUSDT",
    )
    ai_manager = _StaticAIManager(assessment=assessment, prediction=0.01, probability=0.76)
    trader, _, _, _, context = _build_trader(ai_manager)

    report = trader.run_single_cycle(
        execution_context=context,
        schedule_state=_open_schedule_state(trader.schedule_mode),
    )

    telemetry = report.telemetry or {}
    latency = telemetry.get("cycleLatency", {})
    assert latency.get("lastMs", 0.0) >= 0.0
    transitions = telemetry.get("modeTransitions", [])
    assert transitions and transitions[0].get("mode")
    guardrails = telemetry.get("guardrails", {})
    assert "active" in guardrails


def test_autotrader_run_forever_respects_limit() -> None:
    assessment = MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=0.85,
        risk_score=0.25,
        metrics={"trend_strength": 0.9},
        symbol="BTCUSDT",
    )
    ai_manager = _StaticAIManager(assessment=assessment, prediction=0.02, probability=0.82)
    trader, _, _, _, _ = _build_trader(ai_manager)

    reports = list(trader.run_forever(limit=2))

    assert len(reports) == 2
    assert all(report.metadata for report in reports)


def test_e2e_suite_rejects_private_autotrader_fields() -> None:
    root = Path(__file__).resolve().parents[2] / "tests"
    tokens = ["._schedule_mode", "._execution_context"]
    offenders: list[tuple[str, str]] = []
    this_file = Path(__file__).resolve()
    for file_path in root.rglob("*.py"):
        if file_path.resolve() == this_file:
            continue
        text = file_path.read_text(encoding="utf-8")
        for token in tokens:
            if token in text:
                offenders.append((str(file_path.relative_to(root)), token))
    if offenders:
        formatted = ", ".join(f"{path}:{token}" for path, token in offenders)
        pytest.fail(f"Private AutoTrader fields referenced in tests: {formatted}")

