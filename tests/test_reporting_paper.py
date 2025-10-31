"""Testy raportowania symulatora paper tradingu."""
from __future__ import annotations

import csv
import json
from datetime import date, datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import pytest


from bot_core.execution.base import ExecutionContext
from bot_core.execution.paper import MarketMetadata, PaperTradingExecutionService
from bot_core.exchanges.base import OrderRequest
from bot_core.reporting import generate_daily_paper_report
from bot_core.runtime.journal import InMemoryTradingDecisionJournal, TradingDecisionEvent


@pytest.fixture
def paper_service() -> PaperTradingExecutionService:
    timestamps = [
        datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc).timestamp(),
        datetime(2024, 1, 2, 10, 0, 1, tzinfo=timezone.utc).timestamp(),
        datetime(2024, 1, 2, 10, 0, 2, tzinfo=timezone.utc).timestamp(),
        datetime(2024, 1, 2, 12, 15, tzinfo=timezone.utc).timestamp(),
        datetime(2024, 1, 2, 12, 15, 1, tzinfo=timezone.utc).timestamp(),
        datetime(2024, 1, 2, 12, 15, 2, tzinfo=timezone.utc).timestamp(),
        datetime(2024, 1, 2, 15, 45, tzinfo=timezone.utc).timestamp(),
        datetime(2024, 1, 2, 15, 45, 1, tzinfo=timezone.utc).timestamp(),
        datetime(2024, 1, 2, 15, 45, 2, tzinfo=timezone.utc).timestamp(),
    ]
    iterator = iter(timestamps)

    def _time_source() -> float:
        return next(iterator)

    markets = {
        "BTCUSDT": MarketMetadata(
            base_asset="BTC",
            quote_asset="USDT",
            min_quantity=0.0001,
            min_notional=10.0,
        )
    }

    service = PaperTradingExecutionService(
        markets,
        initial_balances={"USDT": 100_000.0},
        maker_fee=0.0002,
        taker_fee=0.0004,
        slippage_bps=0.0,
        time_source=_time_source,
    )
    return service


@pytest.fixture
def execution_context() -> ExecutionContext:
    return ExecutionContext(
        portfolio_id="paper-demo",
        risk_profile="balanced",
        environment="paper",
        metadata={"leverage": "2"},
    )


def test_generate_daily_report(tmp_path: Path, paper_service: PaperTradingExecutionService, execution_context: ExecutionContext) -> None:
    order_buy = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.01,
        order_type="market",
        price=30_000.0,
    )
    order_sell = OrderRequest(
        symbol="BTCUSDT",
        side="sell",
        quantity=0.005,
        order_type="market",
        price=31_000.0,
    )

    paper_service.execute(order_buy, execution_context)
    paper_service.execute(order_sell, execution_context)

    journal = InMemoryTradingDecisionJournal()
    journal.record(
        TradingDecisionEvent(
            event_type="signal_received",
            timestamp=datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc),
            environment="paper",
            portfolio="paper-demo",
            risk_profile="balanced",
            symbol="BTCUSDT",
            side="buy",
            quantity=0.01,
            metadata={"reason": "trend"},
        )
    )
    journal.record(
        TradingDecisionEvent(
            event_type="signal_received",
            timestamp=datetime(2024, 1, 1, 20, 0, tzinfo=timezone.utc),
            environment="paper",
            portfolio="paper-demo",
            risk_profile="balanced",
        )
    )

    artifacts = generate_daily_paper_report(
        execution_service=paper_service,
        output_dir=tmp_path,
        decision_journal=journal,
        report_date=date(2024, 1, 2),
        tz=timezone.utc,
    )

    assert artifacts.ledger_rows == 2
    assert artifacts.decision_events == 1
    assert artifacts.archive_path.exists()

    with ZipFile(artifacts.archive_path) as archive:
        ledger_data = archive.read("ledger.csv").decode("utf-8").splitlines()
        reader = csv.DictReader(ledger_data)
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["symbol"] == "BTCUSDT"
        assert rows[0]["side"] == "buy"
        assert rows[1]["side"] == "sell"

        decisions = archive.read("decisions.jsonl").decode("utf-8").strip().splitlines()
        assert len(decisions) == 1
        decision_record = json.loads(decisions[0])
        assert decision_record["event"] == "signal_received"
        assert decision_record["symbol"] == "BTCUSDT"

        summary = json.loads(archive.read("summary.json"))
        assert summary["ledger_rows"] == 2
        assert summary["decision_events"] == 1
        assert summary["report_date"] == "2024-01-02"
        assert summary["fees_paid"] > 0.0


def test_generate_empty_report(tmp_path: Path, paper_service: PaperTradingExecutionService) -> None:
    artifacts = generate_daily_paper_report(
        execution_service=paper_service,
        output_dir=tmp_path,
        report_date=date(2024, 1, 5),
        tz=timezone.utc,
    )

    assert artifacts.ledger_rows == 0
    assert artifacts.decision_events == 0
    with ZipFile(artifacts.archive_path) as archive:
        ledger_data = archive.read("ledger.csv").decode("utf-8")
        assert "timestamp_utc" in ledger_data
        assert archive.read("summary.json")
        with pytest.raises(KeyError):
            archive.read("decisions.jsonl")
