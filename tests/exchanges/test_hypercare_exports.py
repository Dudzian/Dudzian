import json
from pathlib import Path

from bot_core.exchanges.bitmex.futures import BitmexFuturesAdapter
from bot_core.exchanges.deribit.futures import DeribitFuturesAdapter


def test_deribit_hypercare_export_creates_snapshot(tmp_path: Path) -> None:
    signal_dir = tmp_path / "signal_quality"
    checklist_dir = tmp_path / "hypercare"

    json_path, csv_path = DeribitFuturesAdapter.export_hypercare_assets(
        signal_quality_dir=signal_dir,
        report_dir=checklist_dir,
        daily_csv_dir=tmp_path,
    )

    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    assert payload["exchange"] == DeribitFuturesAdapter.name
    assert payload["checklist_id"] == DeribitFuturesAdapter.hypercare_checklist_id
    assert payload["signed"] is True
    assert payload.get("signal_quality_snapshot")

    assert (signal_dir / f"{DeribitFuturesAdapter.name}.json").exists()
    assert (signal_dir / f"{DeribitFuturesAdapter.name}.csv").exists()
    assert csv_path is not None and Path(csv_path).exists()


def test_bitmex_hypercare_export_creates_snapshot(tmp_path: Path) -> None:
    signal_dir = tmp_path / "signal_quality"
    checklist_dir = tmp_path / "hypercare"

    json_path, csv_path = BitmexFuturesAdapter.export_hypercare_assets(
        signal_quality_dir=signal_dir,
        report_dir=checklist_dir,
        daily_csv_dir=tmp_path,
    )

    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    assert payload["exchange"] == BitmexFuturesAdapter.name
    assert payload["checklist_id"] == BitmexFuturesAdapter.hypercare_checklist_id
    assert payload["signed_by"] == "exchange_ops"
    assert payload.get("signal_quality_snapshot")

    assert (signal_dir / f"{BitmexFuturesAdapter.name}.json").exists()
    assert (signal_dir / f"{BitmexFuturesAdapter.name}.csv").exists()
    assert csv_path is not None and Path(csv_path).exists()
