from datetime import datetime, timedelta, timezone

import json

from bot_core.runtime.journal import (
    InMemoryTradingDecisionJournal,
    aggregate_decision_statistics,
    log_decision_event,
    log_model_change_event,
)


def test_log_and_aggregate_decisions() -> None:
    journal = InMemoryTradingDecisionJournal()
    base_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)

    log_decision_event(
        journal,
        event="order_submitted",
        environment="paper",
        portfolio="demo",
        risk_profile="default",
        timestamp=base_time,
        symbol="BTCUSDT",
        side="buy",
        quantity=1.0,
        price=20000.0,
        status="submitted",
        latency_ms=5.0,
        confidence=0.95,
    )

    log_decision_event(
        journal,
        event="order_filled",
        environment="paper",
        portfolio="demo",
        risk_profile="default",
        timestamp=base_time + timedelta(seconds=5),
        symbol="ETHUSDT",
        side="sell",
        quantity=0.5,
        price=1500.0,
        status="filled",
        latency_ms=12.0,
        confidence=0.8,
    )

    summary = aggregate_decision_statistics(journal)
    assert summary["total"] == 2
    assert summary["by_status"]["submitted"] == 1
    assert summary["by_status"]["filled"] == 1
    assert summary["by_symbol"]["BTCUSDT"] == 1
    assert summary["by_symbol"]["ETHUSDT"] == 1
    assert summary["avg_latency_ms"] > 0
    assert summary["p95_latency_ms"] >= summary["avg_latency_ms"]

    window_summary = aggregate_decision_statistics(
        journal,
        start=base_time + timedelta(seconds=1),
        end=base_time + timedelta(seconds=10),
    )
    assert window_summary["total"] == 1
    assert window_summary["by_symbol"] == {"ETHUSDT": 1}


def test_log_model_change_event_records_metrics() -> None:
    journal = InMemoryTradingDecisionJournal()
    timestamp = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)

    log_model_change_event(
        journal,
        environment="paper",
        portfolio="demo",
        risk_profile="default",
        model_name="decision_engine",
        new_version="v2",
        previous_version="v1",
        source="champion",
        fallback=False,
        decided_at=timestamp,
        retraining_id="decision_engine:20240501T120000Z",
        metrics_current={"mae": 0.5, "directional_accuracy": 0.7},
        metrics_previous={"mae": 0.8},
        metrics_delta={"mae": -0.3},
        metadata={"promotion_reason": "auto"},
    )

    exported = list(journal.export())
    assert len(exported) == 1
    entry = exported[0]
    assert entry["event"] == "model_change"
    assert entry["strategy"] == "decision_engine"
    assert entry["model_version"] == "v2"
    assert entry["model_previous_version"] == "v1"
    assert entry["retraining_id"] == "decision_engine:20240501T120000Z"
    assert entry["metric_mae_current"] == "0.5"
    assert entry["metric_mae_previous"] == "0.8"
    assert entry["metric_mae_delta"] == "-0.3"
    assert entry["promotion_reason"] == "auto"
    assert json.loads(entry["metrics_current_json"]) == {"mae": 0.5, "directional_accuracy": 0.7}
    assert json.loads(entry["metrics_previous_json"]) == {"mae": 0.8}
    assert json.loads(entry["metrics_delta_json"]) == {"mae": -0.3}
