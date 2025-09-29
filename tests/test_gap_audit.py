from __future__ import annotations

import json
from datetime import datetime, timezone

from bot_core.data.ohlcv.audit import GapAuditRecord, JSONLGapAuditLogger


def test_jsonl_gap_audit_logger_appends(tmp_path) -> None:
    path = tmp_path / "audit.jsonl"
    logger = JSONLGapAuditLogger(path)
    record = GapAuditRecord(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        environment="demo",
        exchange="binance_spot",
        symbol="BTCUSDT",
        interval="1h",
        status="warning",
        gap_minutes=30.0,
        row_count=1200,
        last_timestamp="2024-01-01T00:00:00+00:00",
        warnings_in_window=1,
        incident_minutes=None,
    )

    logger.log(record)

    content = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 1
    payload = json.loads(content[0])
    assert payload["status"] == "warning"
    assert payload["symbol"] == "BTCUSDT"
