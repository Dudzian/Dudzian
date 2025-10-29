from datetime import datetime, timedelta, timezone

from bot_core.runtime.journal import (
    InMemoryTradingDecisionJournal,
    aggregate_decision_statistics,
    log_decision_event,
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
