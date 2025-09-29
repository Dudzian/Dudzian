from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bot_core.data.ohlcv import GapAuditRecord

import scripts.gap_audit_report as report


def _record(**kwargs) -> GapAuditRecord:
    defaults = {
        "timestamp": datetime.now(timezone.utc),
        "environment": "paper",
        "exchange": "binance_spot",
        "symbol": "BTCUSDT",
        "interval": "1h",
        "status": "ok",
        "gap_minutes": 0.0,
        "row_count": 100,
        "last_timestamp": "1700000000000",
        "warnings_in_window": 0,
        "incident_minutes": None,
    }
    defaults.update(kwargs)
    return GapAuditRecord(**defaults)


def test_gap_audit_record_from_dict_roundtrip() -> None:
    original = _record(
        timestamp=datetime(2024, 5, 1, 12, 30, tzinfo=timezone.utc),
        gap_minutes=12.3456,
        row_count=123,
        warnings_in_window=2,
        incident_minutes=45.6,
    )
    payload = original.to_dict()
    parsed = GapAuditRecord.from_dict(payload)
    assert parsed.timestamp == original.timestamp
    assert parsed.gap_minutes == round(original.gap_minutes, 3)
    assert parsed.row_count == original.row_count
    assert parsed.warnings_in_window == original.warnings_in_window
    assert parsed.incident_minutes == round(original.incident_minutes, 3)


def test_load_records_filters_environment_and_time(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    records = [
        _record(timestamp=now - timedelta(hours=1), environment="paper", status="warning"),
        _record(timestamp=now - timedelta(hours=3), environment="paper", status="incident"),
        _record(timestamp=now - timedelta(hours=1), environment="prod", status="ok"),
    ]

    audit_path = tmp_path / "audit.jsonl"
    with audit_path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item.to_dict()) + "\n")

    loaded = report.load_records(audit_path, environment="paper", since_hours=2)
    assert len(loaded) == 1
    assert loaded[0].status == "warning"


def test_summarize_records_counts_events_within_window() -> None:
    now = datetime.now(timezone.utc)
    records = [
        _record(status="warning", timestamp=now - timedelta(hours=1)),
        _record(status="warning", timestamp=now - timedelta(hours=5)),
        _record(status="incident", timestamp=now - timedelta(hours=2)),
        _record(status="sms_escalated", timestamp=now - timedelta(hours=3)),
    ]

    summaries = report.summarize_records(records, window_hours=4)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.warning_count == 1  # tylko ostrzeżenie z ostatnich 4h
    assert summary.incident_count == 1
    assert summary.sms_count == 1
    assert summary.last_record.status == "warning"


def test_format_summary_table_renders_rows() -> None:
    record = _record(
        status="incident",
        gap_minutes=60.0,
        incident_minutes=30.0,
        row_count=500,
        last_timestamp="1700012345000",
        timestamp=datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc),
    )
    summary = report.GapSummary(
        exchange=record.exchange,
        symbol=record.symbol,
        interval=record.interval,
        last_record=record,
        warning_count=2,
        incident_count=1,
        sms_count=0,
    )

    table = report.format_summary_table([summary])
    assert "incident" in table
    assert "60.00" in table
    assert "1700012345000" in table
    assert "2" in table
    assert "1" in table

