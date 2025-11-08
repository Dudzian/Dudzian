from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bot_core.cli import main as cli_main
from bot_core.compliance.tax import StaticFXRateProvider, TaxReportGenerator
from bot_core.database.manager import DatabaseManager
from bot_core.execution import LedgerEntry
from bot_core.reporting.tax import TaxReportExporter

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "compliance" / "tax_methods.yaml"
SCHEMA_PATH = REPO_ROOT / "docs" / "schemas" / "tax_report.json"

PL_TAX_RATE = 0.19
USA_SHORT_TAX_RATE = 0.37
USA_LONG_TAX_RATE = 0.20
UK_SHORT_TAX_RATE = 0.20
UK_LONG_TAX_RATE = 0.10


class _StubOrderStore:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def fetch_trades(self, limit: int = 1000, symbol: str | None = None, since=None, **_: object):
        rows = self._rows
        if symbol:
            rows = [row for row in rows if str(row.get("symbol")) == symbol]
        if since:
            filtered: list[dict[str, object]] = []
            for row in rows:
                ts_raw = row.get("ts")
                if isinstance(ts_raw, str):
                    ts = datetime.fromisoformat(ts_raw)
                elif isinstance(ts_raw, datetime):
                    ts = ts_raw
                elif isinstance(ts_raw, (int, float)):
                    ts = datetime.fromtimestamp(float(ts_raw), timezone.utc)
                else:
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= since:
                    filtered.append(row)
            rows = filtered
        return rows[:limit]


@pytest.fixture()
def sample_trades() -> list[LedgerEntry]:
    return [
        LedgerEntry(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp(),
            order_id="buy-ledger-1",
            symbol="BINANCE:BTC/USDT",
            side="buy",
            quantity=0.5,
            price=20000.0,
            fee=5.0,
            fee_asset="USDT",
            status="filled",
            leverage=1.0,
            position_value=0.0,
        ),
        LedgerEntry(
            timestamp=datetime(2024, 1, 5, tzinfo=timezone.utc).timestamp(),
            order_id="buy-ledger-2",
            symbol="COINBASE:BTC-USD",
            side="buy",
            quantity=0.5,
            price=22000.0,
            fee=5.0,
            fee_asset="USD",
            status="filled",
            leverage=1.0,
            position_value=0.0,
        ),
    ]


def _build_generator(
    sample_trades: list[LedgerEntry], **generator_kwargs: object
) -> TaxReportGenerator:
    generator = TaxReportGenerator(config_path=CONFIG_PATH, **generator_kwargs)
    generator.ingest_ledger_entries(sample_trades)
    sale_time = datetime(2024, 1, 10, tzinfo=timezone.utc)
    store = _StubOrderStore(
        [
            {
                "symbol": "KRAKEN:BTCUSD",
                "side": "SELL",
                "quantity": -0.75,
                "price": 24000.0,
                "fee": 18.0,
                "ts": sale_time.timestamp(),
                "order_id": "sell-local-1",
                "mode": "kraken",
            }
        ]
    )
    generator.ingest_local_order_store(store)
    return generator


def test_cost_basis_methods(sample_trades: list[LedgerEntry]) -> None:
    report_end = datetime(2024, 1, 15, tzinfo=timezone.utc)
    fifo_report = _build_generator(sample_trades).generate_report("pl", end=report_end)
    lifo_report = _build_generator(sample_trades).generate_report("usa", end=report_end)
    average_report = _build_generator(sample_trades).generate_report("uk", end=report_end)

    assert len(fifo_report.events) == 1
    fifo_event = fifo_report.events[0]
    assert pytest.approx(fifo_event.cost_basis, rel=1e-6) == pytest.approx(15507.5)
    assert pytest.approx(fifo_event.realized_gain, rel=1e-6) == pytest.approx(2474.5)
    assert fifo_event.long_term_gain == pytest.approx(0.0)
    assert pytest.approx(fifo_event.short_term_gain, rel=1e-6) == pytest.approx(
        fifo_event.realized_gain
    )
    assert pytest.approx(fifo_event.short_term_quantity, rel=1e-6) == pytest.approx(
        fifo_event.quantity
    )
    assert fifo_event.long_term_quantity == pytest.approx(0.0)
    fifo_expected_avg = sum(
        matched.holding_period_days * matched.quantity for matched in fifo_event.matched_lots
    ) / fifo_event.quantity
    assert pytest.approx(
        fifo_event.average_holding_period_days, rel=1e-6
    ) == pytest.approx(fifo_expected_avg)
    assert pytest.approx(fifo_event.short_term_tax, rel=1e-6) == pytest.approx(
        fifo_event.short_term_gain * PL_TAX_RATE
    )
    assert fifo_event.long_term_tax == pytest.approx(0.0)
    assert pytest.approx(
        fifo_event.total_tax_liability, rel=1e-6
    ) == pytest.approx(fifo_event.short_term_tax)
    assert pytest.approx(fifo_report.open_lots[0].quantity, rel=1e-6) == 0.25
    fifo_open_expected = (report_end - datetime(2024, 1, 5, tzinfo=timezone.utc)).total_seconds() / 86400.0
    assert pytest.approx(
        fifo_report.open_lots[0].holding_period_days, rel=1e-6
    ) == pytest.approx(fifo_open_expected)
    fifo_breakdown = {entry.asset: entry for entry in fifo_report.asset_breakdown}
    assert fifo_breakdown["BTC"].disposed_quantity == pytest.approx(0.75)
    assert fifo_breakdown["BTC"].open_quantity == pytest.approx(0.25)
    assert fifo_breakdown["BTC"].realized_gain == pytest.approx(fifo_event.realized_gain)
    assert fifo_breakdown["BTC"].short_term_gain == pytest.approx(
        fifo_event.short_term_gain
    )
    assert fifo_breakdown["BTC"].short_term_quantity == pytest.approx(
        fifo_event.short_term_quantity
    )
    assert fifo_breakdown["BTC"].long_term_quantity == pytest.approx(0.0)
    assert fifo_breakdown["BTC"].short_term_tax == pytest.approx(fifo_event.short_term_tax)
    assert fifo_breakdown["BTC"].long_term_tax == pytest.approx(0.0)
    assert fifo_breakdown["BTC"].total_tax_liability == pytest.approx(
        fifo_event.total_tax_liability
    )
    assert pytest.approx(
        fifo_breakdown["BTC"].average_holding_period_days,
        rel=1e-6,
    ) == pytest.approx(fifo_event.average_holding_period_days)
    assert pytest.approx(
        fifo_breakdown["BTC"].open_short_term_quantity, rel=1e-6
    ) == pytest.approx(fifo_report.open_lots[0].quantity)
    assert fifo_breakdown["BTC"].open_long_term_quantity == pytest.approx(0.0)
    assert pytest.approx(
        fifo_breakdown["BTC"].open_short_term_cost_basis,
        rel=1e-6,
    ) == pytest.approx(fifo_report.open_lots[0].cost_basis + fifo_report.open_lots[0].fee)
    assert fifo_breakdown["BTC"].open_long_term_cost_basis == pytest.approx(0.0)
    assert pytest.approx(
        fifo_breakdown["BTC"].open_average_holding_period_days,
        rel=1e-6,
    ) == pytest.approx(fifo_report.open_lots[0].holding_period_days)
    fifo_venue = {entry.venue: entry for entry in fifo_report.venue_breakdown}
    assert fifo_venue["KRAKEN"].disposed_quantity == pytest.approx(fifo_event.quantity)
    assert fifo_venue["KRAKEN"].realized_gain == pytest.approx(fifo_event.realized_gain)
    assert pytest.approx(
        fifo_venue["KRAKEN"].short_term_quantity, rel=1e-6
    ) == pytest.approx(fifo_event.short_term_quantity)
    assert fifo_venue["KRAKEN"].total_tax_liability == pytest.approx(
        fifo_event.total_tax_liability
    )
    assert "COINBASE" in fifo_venue
    assert fifo_venue["COINBASE"].open_quantity == pytest.approx(
        fifo_report.open_lots[0].quantity
    )
    assert fifo_venue["COINBASE"].disposed_quantity == pytest.approx(0.0)
    assert pytest.approx(
        fifo_report.totals.short_term_quantity, rel=1e-6
    ) == pytest.approx(fifo_event.short_term_quantity)
    assert fifo_report.totals.long_term_quantity == pytest.approx(0.0)
    assert pytest.approx(
        fifo_report.totals.average_holding_period_days,
        rel=1e-6,
    ) == pytest.approx(fifo_event.average_holding_period_days)
    assert pytest.approx(
        fifo_report.totals.unrealized_short_term_quantity, rel=1e-6
    ) == pytest.approx(fifo_report.totals.unrealized_quantity)
    assert fifo_report.totals.unrealized_long_term_quantity == pytest.approx(0.0)
    assert pytest.approx(
        fifo_report.totals.unrealized_short_term_cost_basis, rel=1e-6
    ) == pytest.approx(fifo_report.totals.unrealized_cost_basis)
    assert fifo_report.totals.unrealized_long_term_cost_basis == pytest.approx(0.0)
    assert pytest.approx(
        fifo_report.totals.average_open_holding_period_days, rel=1e-6
    ) == pytest.approx(fifo_report.open_lots[0].holding_period_days)
    assert pytest.approx(fifo_report.totals.short_term_tax, rel=1e-6) == pytest.approx(
        fifo_event.short_term_tax
    )
    assert fifo_report.totals.long_term_tax == pytest.approx(0.0)
    assert pytest.approx(
        fifo_report.totals.total_tax_liability, rel=1e-6
    ) == pytest.approx(fifo_event.total_tax_liability)
    assert len(fifo_report.period_breakdown) == 1
    fifo_period = fifo_report.period_breakdown[0]
    assert fifo_period.period == "2024-01"
    assert fifo_period.proceeds == pytest.approx(fifo_event.proceeds)
    assert fifo_period.cost_basis == pytest.approx(fifo_event.cost_basis)
    assert fifo_period.short_term_gain == pytest.approx(fifo_event.short_term_gain)
    assert fifo_period.long_term_gain == pytest.approx(0.0)
    assert fifo_period.short_term_quantity == pytest.approx(fifo_event.short_term_quantity)
    assert fifo_period.long_term_quantity == pytest.approx(0.0)
    assert pytest.approx(
        fifo_period.average_holding_period_days, rel=1e-6
    ) == pytest.approx(fifo_event.average_holding_period_days)
    assert pytest.approx(fifo_period.short_term_tax, rel=1e-6) == pytest.approx(
        fifo_event.short_term_tax
    )
    assert fifo_period.long_term_tax == pytest.approx(0.0)
    assert fifo_period.total_tax_liability == pytest.approx(fifo_event.total_tax_liability)

    lifo_event = lifo_report.events[0]
    assert pytest.approx(lifo_event.cost_basis, rel=1e-6) == pytest.approx(16007.5)
    assert pytest.approx(lifo_event.realized_gain, rel=1e-6) == pytest.approx(1974.5)
    assert lifo_event.long_term_gain == pytest.approx(0.0)
    assert pytest.approx(lifo_event.short_term_gain, rel=1e-6) == pytest.approx(
        lifo_event.realized_gain
    )
    assert pytest.approx(lifo_event.short_term_quantity, rel=1e-6) == pytest.approx(
        lifo_event.quantity
    )
    assert lifo_event.long_term_quantity == pytest.approx(0.0)
    assert pytest.approx(
        lifo_event.short_term_tax, rel=1e-6
    ) == pytest.approx(max(lifo_event.short_term_gain, 0.0) * USA_SHORT_TAX_RATE)
    assert pytest.approx(
        lifo_event.long_term_tax, rel=1e-6
    ) == pytest.approx(max(lifo_event.long_term_gain, 0.0) * USA_LONG_TAX_RATE)
    assert pytest.approx(
        lifo_event.total_tax_liability, rel=1e-6
    ) == pytest.approx(lifo_event.short_term_tax + lifo_event.long_term_tax)
    lifo_expected_avg = sum(
        matched.holding_period_days * matched.quantity for matched in lifo_event.matched_lots
    ) / lifo_event.quantity
    assert pytest.approx(
        lifo_event.average_holding_period_days, rel=1e-6
    ) == pytest.approx(lifo_expected_avg)
    lifo_breakdown = {entry.asset: entry for entry in lifo_report.asset_breakdown}
    assert lifo_breakdown["BTC"].cost_basis == pytest.approx(lifo_event.cost_basis)
    assert lifo_breakdown["BTC"].short_term_quantity == pytest.approx(
        lifo_event.short_term_quantity
    )
    assert lifo_breakdown["BTC"].average_holding_period_days == pytest.approx(
        lifo_event.average_holding_period_days
    )
    assert pytest.approx(
        lifo_breakdown["BTC"].open_short_term_quantity, rel=1e-6
    ) == pytest.approx(lifo_report.open_lots[0].quantity)
    assert lifo_breakdown["BTC"].open_long_term_quantity == pytest.approx(0.0)
    assert pytest.approx(
        lifo_breakdown["BTC"].open_short_term_cost_basis,
        rel=1e-6,
    ) == pytest.approx(lifo_report.open_lots[0].cost_basis + lifo_report.open_lots[0].fee)
    assert lifo_breakdown["BTC"].open_long_term_cost_basis == pytest.approx(0.0)
    assert pytest.approx(
        lifo_breakdown["BTC"].open_average_holding_period_days,
        rel=1e-6,
    ) == pytest.approx(lifo_report.open_lots[0].holding_period_days)
    assert pytest.approx(
        lifo_breakdown["BTC"].short_term_tax, rel=1e-6
    ) == pytest.approx(lifo_event.short_term_tax)
    assert pytest.approx(
        lifo_breakdown["BTC"].total_tax_liability, rel=1e-6
    ) == pytest.approx(lifo_event.total_tax_liability)
    lifo_venue = {entry.venue: entry for entry in lifo_report.venue_breakdown}
    assert lifo_venue["KRAKEN"].disposed_quantity == pytest.approx(lifo_event.quantity)
    assert pytest.approx(lifo_venue["KRAKEN"].realized_gain, rel=1e-6) == pytest.approx(
        lifo_event.realized_gain
    )
    assert pytest.approx(
        lifo_venue["KRAKEN"].short_term_quantity, rel=1e-6
    ) == pytest.approx(lifo_event.short_term_quantity)
    assert pytest.approx(
        lifo_venue["KRAKEN"].total_tax_liability, rel=1e-6
    ) == pytest.approx(lifo_event.total_tax_liability)
    assert lifo_venue["BINANCE"].open_quantity == pytest.approx(
        lifo_report.open_lots[0].quantity
    )
    lifo_open_expected = (report_end - datetime(2024, 1, 1, tzinfo=timezone.utc)).total_seconds() / 86400.0
    assert pytest.approx(
        lifo_report.open_lots[0].holding_period_days, rel=1e-6
    ) == pytest.approx(lifo_open_expected)
    assert pytest.approx(
        lifo_report.totals.unrealized_short_term_quantity, rel=1e-6
    ) == pytest.approx(lifo_report.totals.unrealized_quantity)
    assert lifo_report.totals.unrealized_long_term_quantity == pytest.approx(0.0)
    assert pytest.approx(
        lifo_report.totals.unrealized_short_term_cost_basis, rel=1e-6
    ) == pytest.approx(lifo_report.totals.unrealized_cost_basis)
    assert lifo_report.totals.unrealized_long_term_cost_basis == pytest.approx(0.0)
    assert pytest.approx(
        lifo_report.totals.average_open_holding_period_days, rel=1e-6
    ) == pytest.approx(lifo_report.open_lots[0].holding_period_days)
    assert pytest.approx(lifo_report.totals.short_term_tax, rel=1e-6) == pytest.approx(
        lifo_event.short_term_tax
    )
    assert pytest.approx(lifo_report.totals.long_term_tax, rel=1e-6) == pytest.approx(
        lifo_event.long_term_tax
    )
    assert pytest.approx(
        lifo_report.totals.total_tax_liability, rel=1e-6
    ) == pytest.approx(lifo_event.total_tax_liability)
    assert len(lifo_report.period_breakdown) == 1
    lifo_period = lifo_report.period_breakdown[0]
    assert lifo_period.period == "2024-01"
    assert lifo_period.proceeds == pytest.approx(lifo_event.proceeds)
    assert lifo_period.cost_basis == pytest.approx(lifo_event.cost_basis)
    assert lifo_period.short_term_gain == pytest.approx(lifo_event.short_term_gain)
    assert lifo_period.short_term_quantity == pytest.approx(lifo_event.short_term_quantity)
    assert lifo_period.long_term_quantity == pytest.approx(0.0)
    assert pytest.approx(
        lifo_period.average_holding_period_days, rel=1e-6
    ) == pytest.approx(lifo_event.average_holding_period_days)
    assert pytest.approx(lifo_period.short_term_tax, rel=1e-6) == pytest.approx(
        lifo_event.short_term_tax
    )
    assert lifo_period.long_term_tax == pytest.approx(lifo_event.long_term_tax)
    assert pytest.approx(lifo_period.total_tax_liability, rel=1e-6) == pytest.approx(
        lifo_event.total_tax_liability
    )

    average_event = average_report.events[0]
    assert pytest.approx(average_event.cost_basis, rel=1e-6) == pytest.approx(15757.5)
    assert pytest.approx(average_event.realized_gain, rel=1e-6) == pytest.approx(2224.5)
    assert average_event.long_term_gain == pytest.approx(0.0)
    assert pytest.approx(
        average_event.short_term_tax, rel=1e-6
    ) == pytest.approx(max(average_event.short_term_gain, 0.0) * UK_SHORT_TAX_RATE)
    assert pytest.approx(
        average_event.long_term_tax, rel=1e-6
    ) == pytest.approx(max(average_event.long_term_gain, 0.0) * UK_LONG_TAX_RATE)
    assert pytest.approx(
        average_event.total_tax_liability, rel=1e-6
    ) == pytest.approx(average_event.short_term_tax + average_event.long_term_tax)
    assert average_report.open_lots[0].quantity == pytest.approx(0.25)
    average_open_expected = (
        report_end - datetime(2024, 1, 1, tzinfo=timezone.utc)
    ).total_seconds() / 86400.0
    assert pytest.approx(
        average_report.open_lots[0].holding_period_days, rel=1e-6
    ) == pytest.approx(average_open_expected)
    average_breakdown = {entry.asset: entry for entry in average_report.asset_breakdown}
    assert average_breakdown["BTC"].open_cost_basis == pytest.approx(
        average_report.open_lots[0].cost_basis + average_report.open_lots[0].fee
    )
    assert average_breakdown["BTC"].short_term_gain == pytest.approx(
        average_event.short_term_gain
    )
    assert pytest.approx(
        average_breakdown["BTC"].short_term_quantity,
        rel=1e-6,
    ) == pytest.approx(average_event.short_term_quantity)
    assert average_breakdown["BTC"].average_holding_period_days == pytest.approx(
        average_event.average_holding_period_days
    )
    assert pytest.approx(
        average_breakdown["BTC"].open_short_term_quantity, rel=1e-6
    ) == pytest.approx(average_report.open_lots[0].quantity)
    assert average_breakdown["BTC"].open_long_term_quantity == pytest.approx(0.0)
    assert pytest.approx(
        average_breakdown["BTC"].open_short_term_cost_basis,
        rel=1e-6,
    ) == pytest.approx(
        average_report.open_lots[0].cost_basis + average_report.open_lots[0].fee
    )
    assert average_breakdown["BTC"].open_long_term_cost_basis == pytest.approx(0.0)
    assert pytest.approx(
        average_breakdown["BTC"].open_average_holding_period_days, rel=1e-6
    ) == pytest.approx(average_report.open_lots[0].holding_period_days)
    assert pytest.approx(
        average_breakdown["BTC"].short_term_tax, rel=1e-6
    ) == pytest.approx(average_event.short_term_tax)
    assert pytest.approx(
        average_breakdown["BTC"].total_tax_liability, rel=1e-6
    ) == pytest.approx(average_event.total_tax_liability)
    average_venue = {entry.venue: entry for entry in average_report.venue_breakdown}
    assert average_venue["KRAKEN"].disposed_quantity == pytest.approx(
        average_event.quantity
    )
    assert pytest.approx(
        average_venue["KRAKEN"].short_term_quantity, rel=1e-6
    ) == pytest.approx(average_event.short_term_quantity)
    assert pytest.approx(
        average_venue["KRAKEN"].total_tax_liability, rel=1e-6
    ) == pytest.approx(average_event.total_tax_liability)
    assert average_venue[None].open_quantity == pytest.approx(
        average_report.open_lots[0].quantity
    )
    assert pytest.approx(
        average_report.totals.unrealized_short_term_cost_basis, rel=1e-6
    ) == pytest.approx(average_report.totals.unrealized_cost_basis)
    assert average_report.totals.unrealized_long_term_cost_basis == pytest.approx(0.0)
    assert pytest.approx(
        average_report.totals.short_term_tax, rel=1e-6
    ) == pytest.approx(average_event.short_term_tax)
    assert pytest.approx(
        average_report.totals.long_term_tax, rel=1e-6
    ) == pytest.approx(average_event.long_term_tax)
    assert pytest.approx(
        average_report.totals.total_tax_liability, rel=1e-6
    ) == pytest.approx(average_event.total_tax_liability)
    assert len(average_report.period_breakdown) == 1
    average_period = average_report.period_breakdown[0]
    assert average_period.period == "2024-01"
    assert average_period.proceeds == pytest.approx(average_event.proceeds)
    assert average_period.cost_basis == pytest.approx(average_event.cost_basis)
    assert average_period.short_term_gain == pytest.approx(average_event.short_term_gain)
    assert average_period.short_term_quantity == pytest.approx(average_event.short_term_quantity)
    assert average_period.long_term_quantity == pytest.approx(0.0)
    assert pytest.approx(
        average_period.average_holding_period_days, rel=1e-6
    ) == pytest.approx(average_event.average_holding_period_days)
    assert pytest.approx(average_period.short_term_tax, rel=1e-6) == pytest.approx(
        average_event.short_term_tax
    )
    assert average_period.long_term_tax == pytest.approx(average_event.long_term_tax)
    assert pytest.approx(average_period.total_tax_liability, rel=1e-6) == pytest.approx(
        average_event.total_tax_liability
    )


def test_generator_applies_fx_conversion(sample_trades: list[LedgerEntry]) -> None:
    provider = StaticFXRateProvider({"USDT": 4.0, "USD": 4.0}, base_currency="PLN")
    generator = _build_generator(
        sample_trades,
        base_currency="PLN",
        fx_rate_provider=provider,
    )
    report_end = datetime(2024, 1, 15, tzinfo=timezone.utc)
    report = generator.generate_report("pl", end=report_end)
    assert report.base_currency == "PLN"
    assert len(report.events) == 1
    event = report.events[0]
    assert event.proceeds == pytest.approx(72000.0)
    assert event.cost_basis == pytest.approx(62030.0)
    assert event.realized_gain == pytest.approx(9898.0)
    assert event.short_term_gain == pytest.approx(event.realized_gain)
    assert pytest.approx(event.short_term_tax, rel=1e-6) == pytest.approx(
        event.short_term_gain * PL_TAX_RATE
    )
    assert event.long_term_tax == pytest.approx(0.0)
    assert report.totals.realized_gain == pytest.approx(event.realized_gain)
    assert pytest.approx(report.totals.total_tax_liability, rel=1e-6) == pytest.approx(
        event.total_tax_liability
    )
    assert report.totals.unrealized_cost_basis == pytest.approx(
        report.open_lots[0].cost_basis + report.open_lots[0].fee
    )
    breakdown = {entry.asset: entry for entry in report.asset_breakdown}
    assert breakdown["BTC"].proceeds == pytest.approx(event.proceeds)
    assert breakdown["BTC"].cost_basis == pytest.approx(event.cost_basis)
    assert breakdown["BTC"].realized_gain == pytest.approx(event.realized_gain)
    venue_breakdown = {entry.venue: entry for entry in report.venue_breakdown}
    assert venue_breakdown["KRAKEN"].proceeds == pytest.approx(event.proceeds)
    assert venue_breakdown["KRAKEN"].cost_basis == pytest.approx(event.cost_basis)
    assert len(report.period_breakdown) == 1
    period = report.period_breakdown[0]
    assert period.period == "2024-01"
    assert period.proceeds == pytest.approx(event.proceeds)
    assert period.cost_basis == pytest.approx(event.cost_basis)


def test_exporter_outputs(tmp_path, sample_trades: list[LedgerEntry]) -> None:
    report_end = datetime(2024, 1, 15, tzinfo=timezone.utc)
    report = _build_generator(sample_trades).generate_report("pl", end=report_end)
    exporter = TaxReportExporter(default_schema=SCHEMA_PATH)
    json_path = tmp_path / "tax_report.json"
    output, signature = exporter.export(report, path=json_path, fmt="json", hmac_key="secret")
    assert output.exists()
    assert signature.exists()
    expected_hmac = hmac.new(b"secret", output.read_bytes(), hashlib.sha256).hexdigest()
    assert signature.read_text(encoding="utf-8").strip() == expected_hmac

    csv_path = tmp_path / "tax_report.csv"
    csv_output, csv_signature = exporter.export(report, path=csv_path, fmt="csv", hmac_key="secret")
    assert csv_output.exists() and csv_signature.exists()
    pdf_path = tmp_path / "tax_report.pdf"
    pdf_output, pdf_signature = exporter.export(report, path=pdf_path, fmt="pdf", hmac_key="secret")
    assert pdf_output.read_bytes().startswith(b"%PDF")
    assert pdf_signature.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["asset_breakdown"], "JSON powinien zawierać sekcję asset_breakdown"
    assert payload["venue_breakdown"], "JSON powinien zawierać sekcję venue_breakdown"
    assert payload["period_breakdown"], "JSON powinien zawierać sekcję period_breakdown"
    assert "short_term_gain" in payload["totals"]
    assert "short_term_quantity" in payload["totals"]
    assert "average_holding_period_days" in payload["totals"]
    assert "unrealized_short_term_quantity" in payload["totals"]
    assert "total_tax_liability" in payload["totals"]
    assert payload["events"][0]["short_term_tax"] >= 0
    assert payload["open_lots"][0]["holding_period_days"] >= 0
    assert payload.get("base_currency") == report.base_currency
    csv_contents = csv_output.read_text(encoding="utf-8")
    assert "asset_breakdown" in csv_contents
    assert "venue_breakdown" in csv_contents
    assert "period_breakdown" in csv_contents
    assert "short_term_gain" in csv_contents
    assert "average_holding_period_days" in csv_contents
    assert "open_short_term_quantity" in csv_contents
    assert "total_tax_liability" in csv_contents


def test_generator_deduplicates_overlapping_sources(sample_trades: list[LedgerEntry]) -> None:
    generator = TaxReportGenerator(config_path=CONFIG_PATH)
    generator.ingest_ledger_entries(sample_trades)
    sale_time = datetime(2024, 1, 12, tzinfo=timezone.utc)
    local_store = _StubOrderStore(
        [
            {
                "symbol": "BINANCE:BTC/USDT",
                "side": "SELL",
                "quantity": 0.5,
                "price": 26000.0,
                "fee": 0.0,
                "ts": sale_time.timestamp(),
                "order_id": "dup-sale-1",
            }
        ]
    )
    generator.ingest_local_order_store(local_store)
    ledger_sale = LedgerEntry(
        timestamp=sale_time.timestamp(),
        order_id="dup-sale-1",
        symbol="BINANCE:BTC/USDT",
        side="sell",
        quantity=0.5,
        price=26000.0,
        fee=7.5,
        fee_asset="USDT",
        status="filled",
        leverage=1.0,
        position_value=0.0,
    )
    generator.ingest_ledger_entries([ledger_sale])
    report = generator.generate_report("pl")
    matching = [event for event in report.events if event.event_id == "dup-sale-1"]
    assert len(matching) == 1
    event = matching[0]
    assert pytest.approx(event.quantity, rel=1e-6) == 0.5
    assert pytest.approx(event.proceeds, rel=1e-6) == pytest.approx(13000.0)
    assert pytest.approx(event.fee, rel=1e-6) == pytest.approx(7.5)
    assert event.source == "ledger"


def test_tax_liability_ignores_losses() -> None:
    generator = TaxReportGenerator(config_path=CONFIG_PATH)
    generator.ingest_ledger_entries(
        [
            LedgerEntry(
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp(),
                order_id="loss-buy",
                symbol="BINANCE:ETH/USDT",
                side="buy",
                quantity=1.0,
                price=2000.0,
                fee=10.0,
                fee_asset="USDT",
                status="filled",
                leverage=1.0,
                position_value=0.0,
            ),
            LedgerEntry(
                timestamp=datetime(2024, 2, 1, tzinfo=timezone.utc).timestamp(),
                order_id="loss-sell",
                symbol="BINANCE:ETH/USDT",
                side="sell",
                quantity=1.0,
                price=1500.0,
                fee=5.0,
                fee_asset="USDT",
                status="filled",
                leverage=1.0,
                position_value=0.0,
            ),
        ]
    )
    report = generator.generate_report("pl")
    assert len(report.events) == 1
    event = report.events[0]
    assert event.realized_gain < 0
    assert event.short_term_tax == pytest.approx(0.0)
    assert event.long_term_tax == pytest.approx(0.0)
    assert report.totals.total_tax_liability == pytest.approx(0.0)
    assert report.totals.short_term_tax == pytest.approx(0.0)
    assert report.totals.long_term_tax == pytest.approx(0.0)
    breakdown = {entry.asset: entry for entry in report.asset_breakdown}
    assert breakdown["ETH"].total_tax_liability == pytest.approx(0.0)
    period = report.period_breakdown[0]
    assert period.total_tax_liability == pytest.approx(0.0)


def test_time_range_includes_prior_buys_and_excludes_pre_period_disposals() -> None:
    generator = TaxReportGenerator(config_path=CONFIG_PATH)
    entries = [
        LedgerEntry(
            timestamp=datetime(2023, 12, 1, tzinfo=timezone.utc).timestamp(),
            order_id="buy-2023",
            symbol="BINANCE:BTC/USDT",
            side="buy",
            quantity=1.0,
            price=20000.0,
            fee=10.0,
            fee_asset="USDT",
            status="filled",
            leverage=1.0,
            position_value=0.0,
        ),
        LedgerEntry(
            timestamp=datetime(2023, 12, 20, tzinfo=timezone.utc).timestamp(),
            order_id="sell-2023",
            symbol="BINANCE:BTC/USDT",
            side="sell",
            quantity=0.4,
            price=21000.0,
            fee=2.0,
            fee_asset="USDT",
            status="filled",
            leverage=1.0,
            position_value=0.0,
        ),
        LedgerEntry(
            timestamp=datetime(2024, 1, 10, tzinfo=timezone.utc).timestamp(),
            order_id="sell-2024",
            symbol="BINANCE:BTC/USDT",
            side="sell",
            quantity=0.6,
            price=23000.0,
            fee=3.0,
            fee_asset="USDT",
            status="filled",
            leverage=1.0,
            position_value=0.0,
        ),
        LedgerEntry(
            timestamp=datetime(2024, 2, 1, tzinfo=timezone.utc).timestamp(),
            order_id="buy-2024",
            symbol="BINANCE:BTC/USDT",
            side="buy",
            quantity=0.5,
            price=25000.0,
            fee=5.0,
            fee_asset="USDT",
            status="filled",
            leverage=1.0,
            position_value=0.0,
        ),
    ]
    generator.ingest_ledger_entries(entries)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 31, tzinfo=timezone.utc)
    report = generator.generate_report("pl", start=start, end=end)
    assert len(report.events) == 1
    event = report.events[0]
    assert event.event_id == "sell-2024"
    assert event.disposal_time == datetime(2024, 1, 10, tzinfo=timezone.utc)
    assert event.cost_basis == pytest.approx(12006.0)
    assert event.realized_gain == pytest.approx(13800.0 - 3.0 - 12006.0)
    assert pytest.approx(event.short_term_tax, rel=1e-6) == pytest.approx(
        max(event.short_term_gain, 0.0) * PL_TAX_RATE
    )
    assert event.long_term_tax == pytest.approx(0.0)
    assert report.totals.realized_gain == pytest.approx(event.realized_gain)
    assert pytest.approx(report.totals.total_tax_liability, rel=1e-6) == pytest.approx(
        event.total_tax_liability
    )
    assert all(evt.disposal_time >= start for evt in report.events)
    assert all(evt.disposal_time <= end for evt in report.events)
    assert not report.open_lots
    breakdown = {entry.asset: entry for entry in report.asset_breakdown}
    assert breakdown["BTC"].disposed_quantity == pytest.approx(0.6)
    assert breakdown["BTC"].realized_gain == pytest.approx(event.realized_gain)
    assert breakdown["BTC"].total_tax_liability == pytest.approx(event.total_tax_liability)


def test_cli_tax_report(tmp_path, sample_trades: list[LedgerEntry]) -> None:
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    ledger_file = ledger_dir / "ledger-20240101.jsonl"
    sale_time = datetime(2024, 1, 10, tzinfo=timezone.utc).timestamp()
    entries = sample_trades + [
        LedgerEntry(
            timestamp=sale_time,
            order_id="sell-ledger-1",
            symbol="BINANCE:BTC/USDT",
            side="sell",
            quantity=0.25,
            price=25000.0,
            fee=10.0,
            fee_asset="USDT",
            status="filled",
            leverage=1.0,
            position_value=0.0,
        ),
    ]
    with ledger_file.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.to_mapping(), ensure_ascii=False))
            handle.write("\n")

    output_path = tmp_path / "cli_report.json"
    argv = [
        "compliance",
        "tax-report",
        "--jurisdiction",
        "pl",
        "--output",
        str(output_path),
        "--format",
        "json",
        "--ledger-dir",
        str(ledger_dir),
        "--hmac-key",
        "cli-secret",
        "--config",
        str(CONFIG_PATH),
        "--schema",
        str(SCHEMA_PATH),
    ]
    result = cli_main(argv)
    assert result == 0
    assert output_path.exists()
    assert output_path.with_suffix(".json.sig").exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["jurisdiction"] == "pl"
    assert payload["events"], "Raport powinien zawierać zdarzenia zbycia"
    assert payload["asset_breakdown"]
    assert payload["venue_breakdown"]
    assert payload["period_breakdown"]
    assert "short_term_gain" in payload["totals"]
    assert "unrealized_short_term_quantity" in payload["totals"]
    assert "total_tax_liability" in payload["totals"]
    assert "short_term_tax" in payload["events"][0]
    assert "holding_period_days" in payload["open_lots"][0]
    assert payload.get("base_currency") is None


def test_cli_tax_report_with_orders_db(tmp_path, sample_trades: list[LedgerEntry]) -> None:
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    ledger_file = ledger_dir / "ledger-20240101.jsonl"
    with ledger_file.open("w", encoding="utf-8") as handle:
        for entry in sample_trades:
            handle.write(json.dumps(entry.to_mapping(), ensure_ascii=False))
            handle.write("\n")

    db_path = tmp_path / "orders.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"
    manager = DatabaseManager(db_url)
    manager.sync.init_db()
    manager.sync.record_trade(
        {
            "symbol": "BINANCE:BTC/USDT",
            "side": "SELL",
            "quantity": 0.75,
            "price": 24000.0,
            "fee": 15.0,
            "mode": "paper",
        }
    )

    output_path = tmp_path / "cli_report_orders.json"
    argv = [
        "compliance",
        "tax-report",
        "--jurisdiction",
        "pl",
        "--output",
        str(output_path),
        "--format",
        "json",
        "--ledger-dir",
        str(ledger_dir),
        "--orders-db",
        db_url,
        "--hmac-key",
        "cli-secret",
        "--config",
        str(CONFIG_PATH),
        "--schema",
        str(SCHEMA_PATH),
    ]

    result = cli_main(argv)
    assert result == 0
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["events"], "Sprzedaże z bazy zleceń powinny zostać uwzględnione"
    assert payload["asset_breakdown"], "Sekcja asset_breakdown powinna być obecna"
    assert payload["venue_breakdown"], "Sekcja venue_breakdown powinna być obecna"
    assert payload["period_breakdown"], "Sekcja period_breakdown powinna być obecna"
    assert "long_term_gain" in payload["totals"]
    assert (
        "unrealized_short_term_quantity" in payload["totals"]
        or "unrealized_long_term_quantity" in payload["totals"]
    )
    assert "holding_period_days" in payload["open_lots"][0]
    assert "total_tax_liability" in payload["totals"]


def test_cli_tax_report_with_fx_rates(tmp_path, sample_trades: list[LedgerEntry]) -> None:
    ledger_dir = tmp_path / "ledger"
    ledger_dir.mkdir()
    ledger_file = ledger_dir / "ledger-20240101.jsonl"
    sale_time = datetime(2024, 1, 10, tzinfo=timezone.utc).timestamp()
    entries = sample_trades + [
        LedgerEntry(
            timestamp=sale_time,
            order_id="sell-ledger-1",
            symbol="BINANCE:BTC/USDT",
            side="sell",
            quantity=0.25,
            price=25000.0,
            fee=10.0,
            fee_asset="USDT",
            status="filled",
            leverage=1.0,
            position_value=0.0,
        ),
    ]
    with ledger_file.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.to_mapping(), ensure_ascii=False))
            handle.write("\n")

    fx_file = tmp_path / "fx_rates.json"
    fx_file.write_text(json.dumps({"USDT": 4.0, "USD": 4.0}), encoding="utf-8")

    output_path = tmp_path / "cli_fx_report.json"
    argv = [
        "compliance",
        "tax-report",
        "--jurisdiction",
        "pl",
        "--output",
        str(output_path),
        "--format",
        "json",
        "--ledger-dir",
        str(ledger_dir),
        "--hmac-key",
        "cli-secret",
        "--config",
        str(CONFIG_PATH),
        "--schema",
        str(SCHEMA_PATH),
        "--base-currency",
        "PLN",
        "--fx-rates-file",
        str(fx_file),
    ]
    result = cli_main(argv)
    assert result == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["base_currency"] == "PLN"
    assert payload["events"], "Raport powinien zawierać zdarzenia"
    assert payload["period_breakdown"], "Sekcja period_breakdown powinna być obecna"
    assert "total_tax_liability" in payload["totals"]
    proceeds = payload["events"][0]["proceeds"]
    assert proceeds == pytest.approx(0.25 * 25000.0 * 4.0)


def test_holding_period_classification() -> None:
    generator = TaxReportGenerator(config_path=CONFIG_PATH)
    generator.ingest_ledger_entries(
        [
            LedgerEntry(
                timestamp=datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp(),
                order_id="long-term-buy",
                symbol="BINANCE:BTC/USDT",
                side="buy",
                quantity=0.3,
                price=10000.0,
                fee=0.0,
                fee_asset="USDT",
                status="filled",
                leverage=1.0,
                position_value=0.0,
            ),
            LedgerEntry(
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp(),
                order_id="short-term-buy",
                symbol="BINANCE:BTC/USDT",
                side="buy",
                quantity=0.4,
                price=20000.0,
                fee=0.0,
                fee_asset="USDT",
                status="filled",
                leverage=1.0,
                position_value=0.0,
            ),
            LedgerEntry(
                timestamp=datetime(2024, 2, 1, tzinfo=timezone.utc).timestamp(),
                order_id="mixed-sale",
                symbol="BINANCE:BTC/USDT",
                side="sell",
                quantity=0.5,
                price=25000.0,
                fee=50.0,
                fee_asset="USDT",
                status="filled",
                leverage=1.0,
                position_value=0.0,
            ),
        ]
    )
    report = generator.generate_report("pl")
    assert len(report.events) == 1
    event = report.events[0]
    assert pytest.approx(event.realized_gain, rel=1e-6) == pytest.approx(5450.0)
    assert pytest.approx(event.short_term_gain, rel=1e-6) == pytest.approx(980.0)
    assert pytest.approx(event.long_term_gain, rel=1e-6) == pytest.approx(4470.0)
    assert pytest.approx(event.short_term_tax, rel=1e-6) == pytest.approx(
        event.short_term_gain * PL_TAX_RATE
    )
    assert pytest.approx(event.long_term_tax, rel=1e-6) == pytest.approx(
        event.long_term_gain * PL_TAX_RATE
    )
    totals = report.totals
    assert pytest.approx(totals.short_term_gain, rel=1e-6) == pytest.approx(980.0)
    assert pytest.approx(totals.long_term_gain, rel=1e-6) == pytest.approx(4470.0)
    assert pytest.approx(totals.short_term_tax, rel=1e-6) == pytest.approx(
        event.short_term_tax
    )
    assert pytest.approx(totals.long_term_tax, rel=1e-6) == pytest.approx(
        event.long_term_tax
    )
    assert pytest.approx(totals.total_tax_liability, rel=1e-6) == pytest.approx(
        event.total_tax_liability
    )
    breakdown = {entry.asset: entry for entry in report.asset_breakdown}
    assert pytest.approx(breakdown["BTC"].short_term_gain, rel=1e-6) == pytest.approx(980.0)
    assert pytest.approx(breakdown["BTC"].long_term_gain, rel=1e-6) == pytest.approx(4470.0)
    assert pytest.approx(breakdown["BTC"].short_term_tax, rel=1e-6) == pytest.approx(
        event.short_term_tax
    )
    assert pytest.approx(breakdown["BTC"].long_term_tax, rel=1e-6) == pytest.approx(
        event.long_term_tax
    )
    assert pytest.approx(breakdown["BTC"].total_tax_liability, rel=1e-6) == pytest.approx(
        event.total_tax_liability
    )
    period = report.period_breakdown[0]
    assert pytest.approx(period.total_tax_liability, rel=1e-6) == pytest.approx(
        event.total_tax_liability
    )
