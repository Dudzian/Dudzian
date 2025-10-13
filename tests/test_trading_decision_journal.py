from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bot_core.runtime.journal import JsonlTradingDecisionJournal, TradingDecisionEvent


def _event(timestamp: datetime) -> TradingDecisionEvent:
    return TradingDecisionEvent(
        event_type="order_executed",
        timestamp=timestamp,
        environment="paper",
        portfolio="paper-1",
        risk_profile="balanced",
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.1,
        price=30_000.0,
        status="filled",
        metadata={"order_id": "abc-1"},
    )


def test_jsonl_journal_writes_events(tmp_path: Path) -> None:
    journal = JsonlTradingDecisionJournal(directory=tmp_path, retention_days=7)
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    journal.record(_event(timestamp))

    file_path = tmp_path / "decisions-20240101.jsonl"
    assert file_path.exists()
    payload = json.loads(file_path.read_text(encoding="utf-8").strip())
    assert payload["event"] == "order_executed"
    assert payload["portfolio"] == "paper-1"
    assert payload["order_id"] == "abc-1"


def test_jsonl_journal_purges_old_files(tmp_path: Path) -> None:
    journal = JsonlTradingDecisionJournal(directory=tmp_path, retention_days=2)
    older = datetime(2024, 1, 1, tzinfo=timezone.utc)
    newer = older + timedelta(days=2)

    journal.record(_event(older))
    old_file = tmp_path / "decisions-20240101.jsonl"
    assert old_file.exists()

    journal.record(_event(newer))
    assert not old_file.exists()
    assert (tmp_path / "decisions-20240103.jsonl").exists()


def test_trading_decision_event_new_fields() -> None:
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    event = TradingDecisionEvent(
        event_type="strategy_signal",
        timestamp=timestamp,
        environment="demo",
        portfolio="paper",
        risk_profile="balanced",
        schedule="mean_reversion_intraday",
        strategy="mean_reversion",
        confidence=0.75,
        latency_ms=180.0,
        telemetry_namespace="demo.multi_strategy.mean_reversion_intraday",
    )
    payload = event.as_dict()
    assert payload["schedule"] == "mean_reversion_intraday"
    assert payload["strategy"] == "mean_reversion"
    assert payload["confidence"] == "0.75"
    assert payload["latency_ms"] == "180"
    assert payload["telemetry_namespace"].startswith("demo.multi_strategy")
