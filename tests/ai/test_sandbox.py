from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import json
import pytest

from bot_core.ai.inference import DecisionModelInference
from bot_core.ai.manager import AIManager
from bot_core.ai.sandbox import (
    SandboxAlertConfig,
    SandboxBudgetConfig,
    SandboxBudgetExceeded,
    SandboxCostGuard,
    SandboxResourceSample,
    TradingStubStreamIngestor,
    RiskLimitSummary,
    default_feature_builder,
    load_sandbox_config,
)
from bot_core.ai.models import ModelScore
from bot_core.observability.metrics import MetricsRegistry
from bot_core.runtime.journal import InMemoryTradingDecisionJournal


def test_stream_ingest_orders_events() -> None:
    ingestor = TradingStubStreamIngestor("multi_asset_performance")
    events = list(
        ingestor.iter_events(
            instruments=["BTC/USDT"],
            event_types=["snapshot", "increment"],
        )
    )
    assert len(events) == 3
    assert [event.event_type for event in events] == ["snapshot", "increment", "increment"]
    assert events[0].payload["close"] == 43090.0
    assert events[-1].payload["close"] == 43150.0
    summary = ingestor.summary()
    assert summary["events"] == len(list(ingestor.iter_events()))


def test_stream_ingest_filters_and_sequence() -> None:
    ingestor = TradingStubStreamIngestor("multi_asset_performance")
    increments = list(ingestor.iter_events(event_types=["increment"]))
    assert increments, "expected increment events"
    assert all(event.event_type == "increment" for event in increments)
    assert [event.sequence for event in increments] == list(range(len(increments)))
    timestamps = [event.timestamp for event in increments]
    assert timestamps == sorted(timestamps)


def test_stream_ingest_risk_state_events() -> None:
    ingestor = TradingStubStreamIngestor("multi_asset_performance")
    risk_events = list(ingestor.iter_events(event_types=["risk_state"]))
    assert len(risk_events) == 2
    assert {event.instrument.symbol for event in risk_events} == {"BTC/USDT", "GLOBAL"}
    btc_risk = [event for event in risk_events if event.instrument.symbol == "BTC/USDT"]
    assert btc_risk, "expected BTC risk state event"
    features = default_feature_builder(btc_risk[0])
    assert features["portfolio_value"] == pytest.approx(1_200_000.0)
    assert features["limit_btc_notional_current_value"] == pytest.approx(210_000.0)
    assert "limit_btc_notional_utilization" in features


def test_cost_guard_exceeds_budget() -> None:
    base_config = load_sandbox_config()
    metrics_registry = MetricsRegistry()
    guard = SandboxCostGuard(
        budgets=SandboxBudgetConfig(wall_time_seconds=0.05, cpu_utilization_percent=5.0, gpu_utilization_percent=None),
        metrics=base_config.metrics,
        alerts=SandboxAlertConfig(enabled=False, source=base_config.alerts.source, severity=base_config.alerts.severity),
        metrics_registry=metrics_registry,
        sampler=lambda: SandboxResourceSample(cpu_percent=12.0, elapsed_seconds=0.1),
        metric_labels={"scenario": "test"},
    )
    guard.start()
    with pytest.raises(SandboxBudgetExceeded):
        guard.update()
    counter = metrics_registry.counter(base_config.metrics.cpu_counter[0], base_config.metrics.cpu_counter[1])
    assert counter.value(labels={"scenario": "test"}) >= 12.0


def test_manager_run_sandbox_scenario_writes_dashboard(tmp_path: Path) -> None:
    manager = AIManager()
    inference = Mock(spec=DecisionModelInference)
    inference.score.side_effect = lambda features, context=None: ModelScore(
        expected_return_bps=1.5, success_probability=0.75
    )
    manager._decision_inferences["default"] = inference  # type: ignore[attr-defined]
    manager._decision_default_name = "default"  # type: ignore[attr-defined]
    journal = InMemoryTradingDecisionJournal()
    dashboard_path = tmp_path / "sandbox_annotations.json"
    metrics_registry = MetricsRegistry()
    result = manager.run_sandbox_scenario(
        "load-test",
        dataset="multi_asset_performance",
        dashboard_output=dashboard_path,
        metrics_registry=metrics_registry,
        decision_journal=journal,
        event_types=["increment"],
    )
    assert result.scenario == "load-test"
    assert result.processed_events == result.event_type_counts.get("increment")
    assert inference.score.call_count == len(result.decisions)
    exported = list(journal.export())
    assert exported, "journal should capture sandbox decisions"
    assert exported[0]["event"] == "sandbox_decision"
    payload = json.loads(dashboard_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "stage6.ai.sandbox"
    assert payload["scenario"] == "load-test"
    assert payload["processed_events"] == result.processed_events
    assert payload["event_type_counts"]["increment"] == result.processed_events
    assert payload["risk_limit_summary"] == {}
    assert result.risk_limit_summary == {}
    config = load_sandbox_config()
    base_labels = dict(config.metric_labels)
    base_labels.setdefault("dataset", result.dataset.stem)
    base_labels["scenario"] = "load-test"
    base_labels["event_type"] = "increment"
    events_counter = config.metrics.events_counter
    assert events_counter is not None
    events_metric = metrics_registry.counter(events_counter[0], events_counter[1])
    assert events_metric.value(labels=base_labels) == result.processed_events
    decisions_counter = config.metrics.decisions_counter
    assert decisions_counter is not None
    decisions_metric = metrics_registry.counter(decisions_counter[0], decisions_counter[1])
    assert decisions_metric.value(labels=base_labels) == len(result.decisions)


def test_manager_run_sandbox_with_instrument_filter() -> None:
    manager = AIManager()
    inference = Mock(spec=DecisionModelInference)
    inference.score.side_effect = lambda features, context=None: ModelScore(
        expected_return_bps=2.0, success_probability=0.5
    )
    manager._decision_inferences["default"] = inference  # type: ignore[attr-defined]
    manager._decision_default_name = "default"  # type: ignore[attr-defined]
    metrics_registry = MetricsRegistry()
    result = manager.run_sandbox_scenario(
        "filter-test",
        dataset="multi_asset_performance",
        metrics_registry=metrics_registry,
        instruments=["BTC/USDT"],
    )
    assert result.event_type_counts
    assert set(result.event_type_counts) == {"snapshot", "increment", "risk_state"}
    assert result.processed_events == sum(result.event_type_counts.values())
    assert inference.score.call_count == len(result.decisions)
    assert "BTC/USDT" in result.risk_limit_summary
    btc_limits = result.risk_limit_summary["BTC/USDT"]
    assert btc_limits
    assert all(isinstance(summary, RiskLimitSummary) for summary in btc_limits)
    notional_summary = next(summary for summary in btc_limits if summary.code == "BTC_NOTIONAL")
    assert notional_summary.observations == 1
    assert notional_summary.max_utilization == pytest.approx(210000.0 / 400000.0)
    assert notional_summary.max_threshold_utilization == pytest.approx(210000.0 / 300000.0)
    assert notional_summary.hard_limit_breaches == 0
    assert notional_summary.threshold_breaches == 0
    config = load_sandbox_config()
    base_labels = dict(config.metric_labels)
    base_labels.setdefault("dataset", result.dataset.stem)
    base_labels["scenario"] = "filter-test"
    base_labels["instrument"] = "BTC/USDT"
    base_labels["limit_code"] = "BTC_NOTIONAL"
    gauge_config = config.metrics.risk_limit_utilization_gauge
    assert gauge_config is not None
    utilization_gauge = metrics_registry.gauge(gauge_config[0], gauge_config[1])
    assert utilization_gauge.value(labels={**base_labels, "dimension": "current_value"}) == pytest.approx(210000.0)
    assert utilization_gauge.value(labels={**base_labels, "dimension": "max_value"}) == pytest.approx(400000.0)
    assert utilization_gauge.value(labels={**base_labels, "dimension": "threshold_value"}) == pytest.approx(300000.0)
    assert utilization_gauge.value(labels={**base_labels, "dimension": "hard_utilization"}) == pytest.approx(210000.0 / 400000.0)
    assert utilization_gauge.value(labels={**base_labels, "dimension": "threshold_utilization"}) == pytest.approx(210000.0 / 300000.0)
    breach_config = config.metrics.risk_limit_breach_counter
    assert breach_config is not None
    breach_counter = metrics_registry.counter(breach_config[0], breach_config[1])
    assert breach_counter.value(labels={**base_labels, "breach_type": "hard"}) == 0.0
    assert breach_counter.value(labels={**base_labels, "breach_type": "threshold"}) == 0.0
