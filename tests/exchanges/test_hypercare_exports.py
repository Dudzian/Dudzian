import json
from datetime import datetime, timezone
from pathlib import Path

from bot_core.exchanges.bitmex.futures import BitmexFuturesAdapter
from bot_core.exchanges.deribit.futures import DeribitFuturesAdapter
from bot_core.exchanges.signal_quality import SignalQualityReporter


def test_deribit_hypercare_export_creates_snapshot(tmp_path: Path) -> None:
    signal_dir = tmp_path / "signal_quality"
    checklist_dir = tmp_path / "hypercare"
    today_suffix = datetime.now(timezone.utc).date().isoformat()

    reporter = SignalQualityReporter(
        exchange_id=DeribitFuturesAdapter.name,
        report_dir=signal_dir,
        enable_csv_export=True,
        csv_dir=signal_dir,
    )
    reporter.record_success(
        backend="rest",
        symbol="BTC/USDT",
        side="buy",
        order_type="limit",
        requested_quantity=2,
        requested_price=50_000,
        filled_quantity=2,
        executed_price=50_001,
        latency=0.42,
    )

    json_path, csv_path = DeribitFuturesAdapter.export_hypercare_assets(
        signal_quality_dir=signal_dir,
        report_dir=checklist_dir,
        daily_csv_dir=tmp_path,
        reporter=reporter,
    )

    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    assert payload["exchange"] == DeribitFuturesAdapter.name
    assert payload["checklist_id"] == DeribitFuturesAdapter.hypercare_checklist_id
    assert payload["signed"] is True
    assert payload.get("signal_quality_snapshot")

    snapshot = json.loads(Path(payload["signal_quality_snapshot"]).read_text(encoding="utf-8"))
    assert snapshot.get("total", 0) > 0

    assert (signal_dir / f"{DeribitFuturesAdapter.name}.json").exists()
    assert (signal_dir / f"{DeribitFuturesAdapter.name}-{today_suffix}.csv").exists()
    assert csv_path is not None and Path(csv_path).exists()


def test_bitmex_hypercare_export_creates_snapshot(tmp_path: Path) -> None:
    signal_dir = tmp_path / "signal_quality"
    checklist_dir = tmp_path / "hypercare"
    today_suffix = datetime.now(timezone.utc).date().isoformat()

    reporter = SignalQualityReporter(
        exchange_id=BitmexFuturesAdapter.name,
        report_dir=signal_dir,
        enable_csv_export=True,
        csv_dir=signal_dir,
    )
    reporter.record_success(
        backend="rest",
        symbol="ETH/USDT",
        side="sell",
        order_type="market",
        requested_quantity=1,
        requested_price=None,
        filled_quantity=1,
        executed_price=1900.5,
        latency=0.21,
    )

    json_path, csv_path = BitmexFuturesAdapter.export_hypercare_assets(
        signal_quality_dir=signal_dir,
        report_dir=checklist_dir,
        daily_csv_dir=tmp_path,
        reporter=reporter,
    )

    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    assert payload["exchange"] == BitmexFuturesAdapter.name
    assert payload["checklist_id"] == BitmexFuturesAdapter.hypercare_checklist_id
    assert payload["signed_by"] == "exchange_ops"
    assert payload.get("signal_quality_snapshot")

    snapshot = json.loads(Path(payload["signal_quality_snapshot"]).read_text(encoding="utf-8"))
    assert snapshot.get("total", 0) > 0

    assert (signal_dir / f"{BitmexFuturesAdapter.name}.json").exists()
    assert (signal_dir / f"{BitmexFuturesAdapter.name}-{today_suffix}.csv").exists()
    assert csv_path is not None and Path(csv_path).exists()


def test_signal_quality_csv_appends_with_header_once(tmp_path: Path) -> None:
    signal_dir = tmp_path / "signal_quality"

    reporter = SignalQualityReporter(
        exchange_id="custom_exchange",
        report_dir=signal_dir,
        enable_csv_export=True,
        csv_dir=signal_dir,
    )

    reporter.record_success(
        backend="rest",
        symbol="BTC/USDT",
        side="buy",
        order_type="limit",
        requested_quantity=1,
        requested_price=30_000,
        filled_quantity=1,
        executed_price=30_001,
        latency=0.1,
    )
    reporter.record_failure(
        backend="rest",
        symbol="ETH/USDT",
        side="sell",
        order_type="market",
        requested_quantity=1,
        requested_price=None,
        error=RuntimeError("failed"),
    )

    today_suffix = datetime.now(timezone.utc).date().isoformat()
    csv_path = signal_dir / f"custom_exchange-{today_suffix}.csv"
    assert csv_path.exists()

    rows = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0] == "timestamp,exchange,total,failures,fill_ratio,slippage_bps,watchdog_alerts"
    assert len(rows) == 3
