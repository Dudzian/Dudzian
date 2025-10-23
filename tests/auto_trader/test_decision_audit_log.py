from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from bot_core.auto_trader.audit import DecisionAuditLog


def test_decision_audit_log_summary_export_and_trace() -> None:
    log = DecisionAuditLog(max_entries=10)
    notifications: list[str] = []

    def _listener(record) -> None:
        notifications.append(record.stage)

    log.add_listener(_listener)

    base = datetime(2024, 3, 1, tzinfo=timezone.utc)
    log.record(
        "strategy_selected",
        "BTCUSDT",
        mode="live",
        payload={"decision_id": "DEC-1", "strategy": "trend"},
        timestamp=base,
    )
    log.record(
        "order_submitted",
        "BTCUSDT",
        mode="live",
        payload={"decision_id": "DEC-1", "order_id": "ORD-1"},
        timestamp=base + timedelta(seconds=5),
    )
    log.record(
        "order_filled",
        "BTCUSDT",
        mode="live",
        payload={"decision_id": "DEC-2"},
        risk_snapshot={"equity": 101_000.0},
        timestamp=base + timedelta(seconds=10),
    )

    assert notifications == ["strategy_selected", "order_submitted", "order_filled"]

    summary = log.summarize()
    assert summary["count"] == 3
    assert summary["decision_ids"]["DEC-1"] == 2
    assert summary["with_risk_snapshot"] == 1

    df = log.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert list(df["stage"]) == ["strategy_selected", "order_submitted", "order_filled"]

    grouped = log.group_by_decision()
    assert set(grouped.keys()) == {"DEC-1", "DEC-2"}
    assert len(grouped["DEC-1"]) == 2

    trace = log.trace_decision("DEC-1")
    assert [entry["stage"] for entry in trace] == ["strategy_selected", "order_submitted"]

    exported = log.export()
    clone = DecisionAuditLog()
    loaded = clone.load(exported)
    assert loaded == 3
    assert clone.summarize()["count"] == 3

    removed = log.trim(before=base + timedelta(seconds=7))
    assert removed == 2
    assert log.summarize()["count"] == 1

    log.remove_listener(_listener)
