from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from bot_core.runtime.journal import JsonlTradingDecisionJournal, TradingDecisionEvent

from scripts.export_decision_engine_summary import main as export_summary


def _record_event(
    journal: JsonlTradingDecisionJournal,
    *,
    timestamp: datetime,
    status: str,
    symbol: str,
    strategy: str,
    schedule: str,
    metadata: dict[str, str],
) -> None:
    latency_raw = metadata.pop("latency_ms", None)
    if isinstance(latency_raw, str):
        latency_value = float(latency_raw)
    else:
        latency_value = latency_raw

    event = TradingDecisionEvent(
        event_type="decision_evaluation",
        timestamp=timestamp,
        environment="paper",
        portfolio="paper-01",
        risk_profile="balanced",
        symbol=symbol,
        side="BUY",
        strategy=strategy,
        schedule=schedule,
        status=status,
        latency_ms=latency_value,
        metadata=metadata,
    )
    journal.record(event)


def test_export_decision_engine_summary(tmp_path: Path) -> None:
    journal = JsonlTradingDecisionJournal(directory=tmp_path)
    base_time = datetime(2024, 5, 1, tzinfo=timezone.utc)

    _record_event(
        journal,
        timestamp=base_time,
        status="accepted",
        symbol="BTC/USDT",
        strategy="daily",
        schedule="d1",
        metadata={
            "expected_probability": "0.68",
            "expected_return_bps": "12.0",
            "notional": "1000",
            "cost_bps": "1.2",
            "net_edge_bps": "6.5",
            "model_success_probability": "0.72",
            "model_expected_return_bps": "8.5",
            "model_name": "gbm_v5",
            "decision_thresholds": json.dumps({"min_probability": 0.6, "max_cost_bps": 18.0}),
            "generated_at": "2024-05-01T00:00:00Z",
            "latency_ms": "42.0",
        },
    )

    _record_event(
        journal,
        timestamp=base_time.replace(day=2),
        status="rejected",
        symbol="ETH/USDT",
        strategy="daily",
        schedule="d1",
        metadata={
            "expected_probability": "0.4",
            "expected_return_bps": "3.0",
            "notional": "500",
            "cost_bps": "2.5",
            "net_edge_bps": "1.0",
            "model_success_probability": "0.41",
            "model_expected_return_bps": "2.5",
            "model_name": "gbm_v6",
            "decision_thresholds": json.dumps({"min_probability": 0.65, "max_cost_bps": 12.0}),
            "decision_reasons": "too_costly",
            "latency_ms": "55.0",
        },
    )

    output_path = tmp_path / "summary.json"
    exit_code = export_summary(
        [
            "--ledger",
            str(tmp_path),
            "--output",
            str(output_path),
            "--environment",
            "paper",
            "--portfolio",
            "paper-01",
            "--history-limit",
            "10",
            "--include-history",
            "--history-size",
            "2",
            "--pretty",
        ]
    )
    assert exit_code == 0

    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["type"] == "decision_engine_summary"
    assert summary["total"] == 2
    assert summary["accepted"] == 1
    assert summary["rejected"] == 1
    assert summary["filters"]["portfolio"] == "paper-01"
    assert summary["rejection_reasons"]["too_costly"] == 1
    assert summary["latest_model"] == "gbm_v6"
    assert summary["history_limit"] == 10
    assert len(summary["history"]) == 2
    assert summary["history"][0]["candidate"]["symbol"] == "BTC/USDT"


def test_export_decision_engine_summary_requires_data(tmp_path: Path) -> None:
    output_path = tmp_path / "summary.json"
    exit_code = export_summary(
        [
            "--ledger",
            str(tmp_path),
            "--output",
            str(output_path),
            "--require-evaluations",
        ]
    )
    assert exit_code == 2
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["total"] == 0
    assert summary["accepted"] == 0
