from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

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
        schedule_run_id="mean_reversion_intraday:2024-01-01T00:00:00+00:00",
        strategy_instance_id="mean_reversion_v1",
        signal_id="signal-123",
        primary_exchange="binance",
        secondary_exchange="kraken",
        base_asset="BTC",
        quote_asset="USDT",
        instrument_type="spot",
        data_feed="normalized_backtest",
        risk_budget_bucket="balanced",
        confidence=0.75,
        latency_ms=180.0,
        telemetry_namespace="demo.multi_strategy.mean_reversion_intraday",
    )
    payload = event.as_dict()
    assert payload["schedule"] == "mean_reversion_intraday"
    assert payload["strategy"] == "mean_reversion"
    assert payload["schedule_run_id"].startswith("mean_reversion_intraday")
    assert payload["strategy_instance_id"] == "mean_reversion_v1"
    assert payload["signal_id"] == "signal-123"
    assert payload["primary_exchange"] == "binance"
    assert payload["secondary_exchange"] == "kraken"
    assert payload["base_asset"] == "BTC"
    assert payload["quote_asset"] == "USDT"
    assert payload["instrument_type"] == "spot"
    assert payload["data_feed"] == "normalized_backtest"
    assert payload["risk_budget_bucket"] == "balanced"
    assert payload["confidence"] == "0.75"
    assert payload["latency_ms"] == "180"
    assert payload["telemetry_namespace"].startswith("demo.multi_strategy")


def test_trading_decision_event_metadata_does_not_override_system_fields() -> None:
    event = TradingDecisionEvent(
        event_type="decision_evaluation",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        environment="paper",
        portfolio="paper-01",
        risk_profile="balanced",
        metadata={
            "event": "tampered",
            "timestamp": "2020-01-01T00:00:00+00:00",
            "environment": "prod",
            "portfolio": "other",
            "risk_profile": "aggressive",
            "custom": "ok",
        },
    )

    payload = event.as_dict()

    assert payload["event"] == "decision_evaluation"
    assert payload["timestamp"] == "2024-01-01T00:00:00+00:00"
    assert payload["environment"] == "paper"
    assert payload["portfolio"] == "paper-01"
    assert payload["risk_profile"] == "balanced"

    assert payload["meta_event"] == "tampered"
    assert payload["meta_timestamp"] == "2020-01-01T00:00:00+00:00"
    assert payload["meta_environment"] == "prod"
    assert payload["meta_portfolio"] == "other"
    assert payload["meta_risk_profile"] == "aggressive"
    assert payload["custom"] == "ok"


def test_trading_decision_event_metadata_serializes_complex_values_as_json() -> None:
    event = TradingDecisionEvent(
        event_type="decision_evaluation",
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        environment="paper",
        portfolio="paper-01",
        risk_profile="balanced",
        metadata={
            "simple": "value",
            "number": 12,
            "flag": True,
            "missing": None,
            "payload": {"z": 2, "a": 1},
            "items": ["x", {"k": 1}],
        },
    )

    payload = event.as_dict()

    assert payload["simple"] == "value"
    assert payload["number"] == "12"
    assert payload["flag"] == "true"
    assert payload["missing"] == "null"
    assert payload["payload"] == '{"a":1,"z":2}'
    assert payload["items"] == '["x",{"k":1}]'


def test_jsonl_export_is_lenient_but_logs_decode_failures(tmp_path: Path, caplog) -> None:
    journal = JsonlTradingDecisionJournal(directory=tmp_path, retention_days=7)
    file_path = tmp_path / "decisions-20240101.jsonl"
    file_path.write_text(
        "\n".join(
            [
                '{"event":"ok_1","timestamp":"2024-01-01T00:00:00+00:00"}',
                "{bad json",
                '{"event":"ok_2","timestamp":"2024-01-01T00:01:00+00:00"}',
                "",
            ]
        ),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="bot_core.runtime.journal"):
        exported = journal.export()

    assert [item["event"] for item in exported] == ["ok_1", "ok_2"]
    assert "skipped malformed decision journal lines" in caplog.text
    assert "decisions-20240101.jsonl" in caplog.text


def test_jsonl_export_logs_file_read_errors(tmp_path: Path, monkeypatch, caplog) -> None:
    journal = JsonlTradingDecisionJournal(directory=tmp_path, retention_days=7)
    file_path = tmp_path / "decisions-20240101.jsonl"
    file_path.write_text(
        '{"event":"ok","timestamp":"2024-01-01T00:00:00+00:00"}\n', encoding="utf-8"
    )

    original_open = Path.open

    def _raising_open(self: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self == file_path:
            raise OSError("disk read error")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _raising_open)

    with caplog.at_level(logging.WARNING, logger="bot_core.runtime.journal"):
        exported = journal.export()

    assert exported == ()
    assert "failed to read decision journal file" in caplog.text
    assert "decisions-20240101.jsonl" in caplog.text


def test_jsonl_journal_rejects_filename_pattern_with_directory_separator(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="filename_pattern"):
        JsonlTradingDecisionJournal(directory=tmp_path, filename_pattern="%Y/%m/decisions-%d.jsonl")
