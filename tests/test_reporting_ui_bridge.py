from __future__ import annotations

"""Testy warstwy mostkującej raporty do UI."""

import json
import os
import tarfile
import zipfile
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from pathlib import Path

import pytest

from bot_core.ai.validation import ModelQualityReport, record_model_quality_report
from bot_core.reporting import ui_bridge
from bot_core.reporting.model_quality import load_champion_overview


def write_fixture_text(path: str | os.PathLike[str], content: str) -> None:
    """Zapisuje deterministyczne treści fixture z wymuszonymi końcami linii LF."""

    Path(path).write_text(content, encoding="utf-8", newline="\n")


def test_package_exports_ui_bridge():
    import bot_core.reporting as reporting

    assert reporting.ui_bridge is ui_bridge


def _quality_report(
    version: str, directional: float, mae: float, status: str = "improved"
) -> ModelQualityReport:
    metrics = {"summary": {"directional_accuracy": directional, "mae": mae}}
    return ModelQualityReport(
        model_name="demo",
        version=version,
        evaluated_at=datetime.now(timezone.utc),
        metrics=metrics,
        status=status,
    )


def test_cmd_overview_empty_directory(tmp_path, capsys):
    args = SimpleNamespace(base_dir=str(tmp_path))

    return_code = ui_bridge.cmd_overview(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert payload["reports"] == []
    assert payload["categories"] == []
    assert payload["base_directory"] == str(tmp_path)
    assert payload["filters"] == {
        "since": None,
        "until": None,
        "categories": [],
        "summary_status": "any",
        "limit": None,
        "offset": None,
        "sort_key": "updated_at",
        "sort_direction": "desc",
        "query": None,
        "has_exports": "any",
    }
    assert payload["pagination"] == {
        "total_count": 0,
        "returned_count": 0,
        "limit": None,
        "offset": None,
        "has_more": False,
        "has_previous": False,
    }
    assert payload["summary"] == {
        "report_count": 0,
        "category_count": 0,
        "total_size": 0,
        "export_count": 0,
        "has_summary": False,
        "has_exports": False,
        "invalid_summary_count": 0,
        "missing_summary_count": 0,
        "latest_updated_at": None,
        "earliest_updated_at": None,
    }


def test_cmd_overview_lists_reports_with_exports(tmp_path, capsys):
    base_dir = tmp_path / "reports"
    report_dir = base_dir / "audit" / "2024-01-01"
    report_dir.mkdir(parents=True)

    summary_path = report_dir / "summary.json"
    summary_payload = {"report_date": "2024-01-01", "trades": 15}
    write_fixture_text(summary_path, json.dumps(summary_payload))

    export_path = report_dir / "result.csv"
    export_content = "timestamp,profit\n2024-01-01T00:00:00Z,42.5\n"
    write_fixture_text(export_path, export_content)

    # deterministyczne czasy modyfikacji
    fixed_timestamp = 1_700_000_000
    os.utime(summary_path, (fixed_timestamp, fixed_timestamp))
    os.utime(export_path, (fixed_timestamp, fixed_timestamp))

    args = SimpleNamespace(base_dir=str(base_dir))

    return_code = ui_bridge.cmd_overview(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert payload["base_directory"] == str(base_dir)
    assert len(payload["reports"]) == 1
    assert len(payload["categories"]) == 1
    assert payload["filters"] == {
        "since": None,
        "until": None,
        "categories": [],
        "summary_status": "any",
        "limit": None,
        "offset": None,
        "sort_key": "updated_at",
        "sort_direction": "desc",
        "query": None,
        "has_exports": "any",
    }
    category_entry = payload["categories"][0]
    assert category_entry["id"] == "audit"
    assert category_entry["label"] == "audit"
    assert category_entry["count"] == 1
    assert category_entry["has_summary"] is True
    assert category_entry["has_exports"] is True
    assert category_entry["total_size"] == len(export_content.encode("utf-8"))
    assert category_entry["invalid_summary_count"] == 0
    assert category_entry["export_count"] == 1
    assert category_entry["missing_summary_count"] == 0
    expected_timestamp = datetime.fromtimestamp(fixed_timestamp, tz=timezone.utc).isoformat()
    assert category_entry["earliest_updated_at"] == expected_timestamp

    report_entry = payload["reports"][0]
    assert report_entry["relative_path"] == "audit/2024-01-01"
    assert report_entry["display_name"] == "2024-01-01"
    assert Path(report_entry["summary_path"]).resolve() == summary_path.resolve()
    assert category_entry["latest_updated_at"] == report_entry["updated_at"]
    assert report_entry["total_size"] == len(export_content.encode("utf-8"))
    assert report_entry["export_count"] == 1
    assert report_entry["created_at"] == expected_timestamp
    assert report_entry["has_summary"] is True

    exports = report_entry["exports"]
    assert len(exports) == 1
    export_entry = exports[0]
    assert export_entry["relative_path"] == "audit/2024-01-01/result.csv"
    assert Path(export_entry["absolute_path"]).resolve() == export_path.resolve()
    assert export_entry["size"] == len(export_content.encode("utf-8"))

    summary = payload["summary"]
    assert summary["report_count"] == 1
    assert summary["category_count"] == 1
    assert summary["total_size"] == len(export_content.encode("utf-8"))
    assert summary["export_count"] == 1
    assert summary["has_summary"] is True
    assert summary["has_exports"] is True
    assert summary["invalid_summary_count"] == 0
    assert summary["missing_summary_count"] == 0
    assert summary["latest_updated_at"] == expected_timestamp
    assert summary["earliest_updated_at"] == expected_timestamp


def test_cmd_overview_detects_directories_without_summary(tmp_path, capsys):
    base_dir = tmp_path / "reports"
    orphan_dir = base_dir / "diagnostics" / "2024-03-10"
    orphan_dir.mkdir(parents=True)

    metrics_path = orphan_dir / "metrics.jsonl"
    metrics_content = "{}\n"
    write_fixture_text(metrics_path, metrics_content)
    fixed_timestamp = 1_700_100_000
    os.utime(metrics_path, (fixed_timestamp, fixed_timestamp))

    args = SimpleNamespace(base_dir=str(base_dir))

    return_code = ui_bridge.cmd_overview(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert payload["base_directory"] == str(base_dir)
    assert len(payload["categories"]) == 1
    assert payload["filters"] == {
        "since": None,
        "until": None,
        "categories": [],
        "summary_status": "any",
        "limit": None,
        "offset": None,
        "sort_key": "updated_at",
        "sort_direction": "desc",
        "query": None,
        "has_exports": "any",
    }
    category_entry = payload["categories"][0]
    assert category_entry["id"] == "diagnostics"
    assert category_entry["label"] == "diagnostics"
    assert category_entry["count"] == 1
    assert category_entry["has_summary"] is False
    assert category_entry["has_exports"] is True
    assert category_entry["total_size"] == len(metrics_content.encode("utf-8"))
    assert category_entry["invalid_summary_count"] == 0
    assert category_entry["export_count"] == 1
    assert category_entry["missing_summary_count"] == 1
    expected_timestamp = datetime.fromtimestamp(fixed_timestamp, tz=timezone.utc).isoformat()
    assert category_entry["earliest_updated_at"] == expected_timestamp
    assert len(payload["reports"]) == 1

    report_entry = payload["reports"][0]
    assert report_entry["relative_path"] == "diagnostics/2024-03-10"
    assert report_entry["summary_path"] is None
    assert report_entry["display_name"] == "diagnostics/2024-03-10"
    assert (
        Path(report_entry["absolute_path"])
        .as_posix()
        .endswith("diagnostics/2024-03-10/metrics.jsonl")
    )
    assert category_entry["latest_updated_at"] == report_entry["updated_at"]
    assert report_entry["total_size"] == len(metrics_content.encode("utf-8"))
    assert report_entry["export_count"] == 1
    assert report_entry["created_at"] == expected_timestamp
    assert report_entry["has_summary"] is False

    exports = report_entry["exports"]
    assert len(exports) == 1
    export_entry = exports[0]
    assert export_entry["relative_path"] == "diagnostics/2024-03-10/metrics.jsonl"
    assert Path(export_entry["absolute_path"]).resolve() == metrics_path.resolve()

    summary = payload["summary"]
    assert summary["report_count"] == 1
    assert summary["category_count"] == 1
    assert summary["has_summary"] is False
    assert summary["has_exports"] is True
    assert summary["total_size"] == len(metrics_content.encode("utf-8"))
    assert summary["export_count"] == 1
    assert summary["invalid_summary_count"] == 0
    assert summary["latest_updated_at"] == expected_timestamp
    assert summary["earliest_updated_at"] == expected_timestamp
    assert summary["missing_summary_count"] == 1


def test_cmd_delete_removes_report_directory(tmp_path, capsys):
    base_dir = tmp_path / "reports"
    report_dir = base_dir / "audit" / "2024-01-01"
    export_dir = report_dir / "exports"
    export_dir.mkdir(parents=True)

    summary_path = report_dir / "summary.json"
    summary_content = "{}"
    write_fixture_text(summary_path, summary_content)

    export_path = export_dir / "result.csv"
    export_content = "id,value\n1,2\n"
    write_fixture_text(export_path, export_content)

    expected_size = len(summary_content.encode("utf-8")) + len(export_content.encode("utf-8"))

    args = SimpleNamespace(base_dir=str(base_dir), path="audit/2024-01-01")

    return_code = ui_bridge.cmd_delete(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert payload["status"] == "deleted"
    assert payload["relative_path"] == "audit/2024-01-01"
    assert payload["removed_files"] == 2
    assert payload["removed_directories"] == 2
    assert payload["removed_size"] == expected_size
    assert not report_dir.exists()
    assert not export_dir.exists()
    assert not summary_path.exists()


def test_cmd_delete_dry_run_returns_preview(tmp_path, capsys):
    base_dir = tmp_path / "reports"
    report_dir = base_dir / "audit" / "2024-01-01"
    export_dir = report_dir / "exports"
    export_dir.mkdir(parents=True)

    summary_path = report_dir / "summary.json"
    summary_content = "{}"
    write_fixture_text(summary_path, summary_content)

    export_path = export_dir / "result.csv"
    export_content = "id,value\n1,2\n"
    write_fixture_text(export_path, export_content)

    expected_size = len(summary_content.encode("utf-8")) + len(export_content.encode("utf-8"))

    args = SimpleNamespace(base_dir=str(base_dir), path="audit/2024-01-01", dry_run=True)

    return_code = ui_bridge.cmd_delete(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert payload["status"] == "preview"
    assert payload["dry_run"] is True
    assert payload["removed_files"] == 2
    assert payload["removed_directories"] == 2
    assert payload["removed_size"] == expected_size
    # Ścieżki nadal istnieją, bo operacja była tylko symulacją.
    assert report_dir.exists()
    assert export_dir.exists()
    assert summary_path.exists()


def test_cmd_delete_handles_missing_report(tmp_path, capsys):
    base_dir = tmp_path / "reports"
    base_dir.mkdir(parents=True)

    args = SimpleNamespace(base_dir=str(base_dir), path="missing/report")

    return_code = ui_bridge.cmd_delete(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert payload["status"] == "not_found"
    assert payload["relative_path"] == "missing/report"
    assert payload["removed_files"] == 0
    assert payload["removed_directories"] == 0
    assert payload["removed_size"] == 0


def test_cmd_delete_rejects_paths_outside_base(tmp_path, capsys):
    base_dir = tmp_path / "reports"
    base_dir.mkdir()

    args = SimpleNamespace(base_dir=str(base_dir), path="../outside")

    return_code = ui_bridge.cmd_delete(args)

    captured = capsys.readouterr()

    assert return_code == 2
    assert captured.out == ""
    assert "musi znajdować się" in captured.err


def test_cmd_delete_supports_absolute_paths(tmp_path, capsys):
    base_dir = tmp_path / "reports"
    report_file = base_dir / "standalone.json"
    report_file.parent.mkdir(parents=True)
    report_content = "{}"
    write_fixture_text(report_file, report_content)

    expected_size = len(report_content.encode("utf-8"))

    args = SimpleNamespace(base_dir=str(base_dir), path=str(report_file))

    return_code = ui_bridge.cmd_delete(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert payload["status"] == "deleted"
    assert payload["relative_path"] == "standalone.json"
    assert payload["removed_files"] == 1
    assert payload["removed_directories"] == 0
    assert payload["removed_size"] == expected_size
    assert not report_file.exists()


def _build_purge_args(
    base_dir: Path,
    *,
    dry_run: bool = False,
    summary_status: str = "any",
    limit: int | None = None,
    offset: int | None = None,
    categories: list[str] | None = None,
    signal_quality_dir: Path | None = None,
    retention_days: int = 30,
) -> SimpleNamespace:
    return SimpleNamespace(
        base_dir=str(base_dir),
        since=None,
        until=None,
        categories=categories,
        summary_status=summary_status,
        limit=limit,
        offset=offset,
        sort_key="updated_at",
        sort_direction="desc",
        query=None,
        has_exports="any",
        dry_run=dry_run,
        signal_quality_dir=str(signal_quality_dir) if signal_quality_dir is not None else None,
        signal_quality_retention_days=retention_days,
    )


def _build_archive_args(
    base_dir: Path,
    *,
    destination: Path | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    archive_format: str = "directory",
) -> SimpleNamespace:
    return SimpleNamespace(
        base_dir=str(base_dir),
        destination=str(destination) if destination is not None else None,
        since=None,
        until=None,
        categories=None,
        summary_status="any",
        limit=None,
        offset=None,
        sort_key="updated_at",
        sort_direction="desc",
        query=None,
        has_exports="any",
        dry_run=dry_run,
        overwrite=overwrite,
        format=archive_format,
    )


def test_cmd_purge_dry_run_collects_metrics(tmp_path, capsys):
    base_dir = tmp_path / "reports"

    first_report = base_dir / "audit" / "2024-01-01"
    first_exports = first_report / "exports"
    first_exports.mkdir(parents=True)
    first_summary = first_report / "summary.json"
    first_summary_content = "{}"
    write_fixture_text(first_summary, first_summary_content)
    first_export = first_exports / "result.csv"
    first_export_content = "id,value\n1,2\n"
    write_fixture_text(first_export, first_export_content)

    second_report = base_dir / "diagnostics" / "2024-01-02"
    second_exports = second_report / "exports"
    second_exports.mkdir(parents=True)
    second_summary = second_report / "summary.json"
    second_summary_content = "{}"
    write_fixture_text(second_summary, second_summary_content)
    second_export = second_exports / "data.json"
    second_export_content = '{"ok": true}'
    write_fixture_text(second_export, second_export_content)

    # removed_size dotyczy wyłącznie celów raportowych; signal_quality_cleanup
    # raportuje swój rozmiar osobno
    expected_size = sum(
        len(content.encode("utf-8"))
        for content in (
            first_summary_content,
            first_export_content,
            second_summary_content,
            second_export_content,
        )
    )

    signal_quality_dir = tmp_path / "signal_quality"
    signal_quality_dir.mkdir(parents=True)
    stale_quality = signal_quality_dir / "binance.json"
    write_fixture_text(stale_quality, "{}")
    old_ts = (datetime.now(timezone.utc) - timedelta(days=60)).timestamp()
    os.utime(stale_quality, (old_ts, old_ts))

    args = _build_purge_args(base_dir, dry_run=True, signal_quality_dir=signal_quality_dir)

    return_code = ui_bridge.cmd_purge(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert payload["status"] == "preview"
    assert payload["planned_count"] == 2
    assert payload["matched_count"] == 2
    assert payload["removed_files"] == 4
    assert payload["removed_directories"] == 4
    assert payload["removed_size"] == expected_size
    assert payload["dry_run"] is True
    assert len(payload["targets"]) == 2
    assert first_report.exists()
    assert second_report.exists()

    cleanup = payload.get("signal_quality_cleanup")
    assert cleanup["status"] == "preview"
    assert cleanup["removed"] == 1
    assert cleanup["targets"]
    assert cleanup["targets"][0]["status"] == "preview"


def test_cmd_purge_deletes_reports(tmp_path, capsys):
    base_dir = tmp_path / "reports"

    report_a = base_dir / "audit" / "2024-01-01"
    report_a_exports = report_a / "exports"
    report_a_exports.mkdir(parents=True)
    write_fixture_text((report_a / "summary.json"), "{}")
    write_fixture_text((report_a_exports / "data.csv"), "id,value\n1,2\n")

    report_b = base_dir / "audit" / "2024-01-02"
    report_b_exports = report_b / "exports"
    report_b_exports.mkdir(parents=True)
    write_fixture_text((report_b / "summary.json"), "{}")
    write_fixture_text((report_b_exports / "data.csv"), "id,value\n1,3\n")

    signal_quality_dir = tmp_path / "signal_quality"
    signal_quality_dir.mkdir(parents=True)
    stale_quality = signal_quality_dir / "binance.json"
    write_fixture_text(stale_quality, "{}")
    old_ts = (datetime.now(timezone.utc) - timedelta(days=90)).timestamp()
    os.utime(stale_quality, (old_ts, old_ts))

    args = _build_purge_args(base_dir, signal_quality_dir=signal_quality_dir)

    return_code = ui_bridge.cmd_purge(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert payload["status"] == "completed"
    assert payload["deleted_count"] == 2
    assert payload["planned_count"] == 2
    assert payload["removed_files"] == 4
    assert payload["removed_directories"] == 4
    assert not report_a.exists()
    assert not report_b.exists()

    cleanup = payload.get("signal_quality_cleanup")
    assert cleanup["status"] == "completed"
    assert cleanup["removed"] == 1
    assert not stale_quality.exists()


def test_cmd_purge_honors_limit_and_filters(tmp_path, capsys):
    base_dir = tmp_path / "reports"

    report_a = base_dir / "audit" / "2024-01-01"
    report_a_exports = report_a / "exports"
    report_a_exports.mkdir(parents=True)
    write_fixture_text((report_a / "summary.json"), "{}")
    write_fixture_text((report_a_exports / "data.csv"), "1")

    report_b = base_dir / "audit" / "2024-01-02"
    report_b_exports = report_b / "exports"
    report_b_exports.mkdir(parents=True)
    write_fixture_text((report_b / "summary.json"), "{}")
    write_fixture_text((report_b_exports / "data.csv"), "1")

    report_c = base_dir / "diagnostics" / "2024-01-03"
    report_c_exports = report_c / "exports"
    report_c_exports.mkdir(parents=True)
    write_fixture_text((report_c_exports / "data.csv"), "1")

    signal_quality_dir = tmp_path / "signal_quality"
    signal_quality_dir.mkdir(parents=True)

    args = _build_purge_args(
        base_dir,
        limit=1,
        categories=["diagnostics"],
        summary_status="missing",
        signal_quality_dir=signal_quality_dir,
    )

    return_code = ui_bridge.cmd_purge(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert payload["status"] == "completed"
    assert payload["planned_count"] == 1
    assert payload["matched_count"] == 1
    assert payload["deleted_count"] == 1
    assert not report_c.exists()
    assert report_a.exists()
    assert report_b.exists()
    cleanup = payload.get("signal_quality_cleanup")
    assert cleanup["status"] in {"empty", "missing"}


def test_cmd_purge_empty_returns_status(tmp_path, capsys):
    base_dir = tmp_path / "reports"
    base_dir.mkdir(parents=True)

    signal_quality_dir = tmp_path / "signal_quality"
    signal_quality_dir.mkdir(parents=True)

    args = _build_purge_args(base_dir, signal_quality_dir=signal_quality_dir)

    return_code = ui_bridge.cmd_purge(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert payload["status"] == "empty"
    assert payload["planned_count"] == 0
    assert payload["matched_count"] == 0
    assert payload["removed_files"] == 0
    assert payload["removed_directories"] == 0
    assert payload["removed_size"] == 0
    cleanup = payload.get("signal_quality_cleanup")
    assert cleanup["status"] in {"empty", "missing"}


def test_cmd_archive_dry_run_collects_metrics(tmp_path, capsys):
    base_dir = tmp_path / "reports"

    report_dir = base_dir / "audit" / "2024-01-01"
    report_dir.mkdir(parents=True)
    summary_path = report_dir / "summary.json"
    summary_content = "{}"
    write_fixture_text(summary_path, summary_content)
    export_path = report_dir / "exports" / "data.csv"
    export_path.parent.mkdir(parents=True)
    export_content = "timestamp,value\n2024-01-01T00:00:00Z,1\n"
    write_fixture_text(export_path, export_content)

    expected_files = 2
    expected_directories = 2
    expected_size = len(summary_content.encode("utf-8")) + len(export_content.encode("utf-8"))

    destination = tmp_path / "archives"
    args = _build_archive_args(base_dir, destination=destination, dry_run=True)

    return_code = ui_bridge.cmd_archive(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert payload["status"] == "preview"
    assert payload["destination_directory"] == str(destination.resolve(strict=False))
    assert payload["format"] == "directory"
    assert payload["copied_files"] == expected_files
    assert payload["copied_directories"] == expected_directories
    assert payload["copied_size"] == expected_size
    assert payload["copied_count"] == 1
    assert destination.exists() is False


def test_cmd_archive_copies_reports(tmp_path, capsys):
    base_dir = tmp_path / "reports"

    report_dir = base_dir / "daily" / "2024-02-01"
    report_dir.mkdir(parents=True)
    summary_path = report_dir / "summary.json"
    write_fixture_text(summary_path, "{}")
    export_path = report_dir / "exports" / "result.json"
    export_path.parent.mkdir(parents=True)
    export_payload = {"profit": 42}
    write_fixture_text(export_path, json.dumps(export_payload))

    destination = tmp_path / "archives"
    args = _build_archive_args(base_dir, destination=destination)

    return_code = ui_bridge.cmd_archive(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert payload["status"] == "completed"
    assert payload["copied_count"] == 1
    archived_summary = destination / "daily" / "2024-02-01" / "summary.json"
    archived_export = destination / "daily" / "2024-02-01" / "exports" / "result.json"
    assert archived_summary.exists()
    assert archived_export.exists()
    assert archived_export.read_text(encoding="utf-8") == json.dumps(export_payload)
    # źródłowy raport pozostaje nienaruszony
    assert summary_path.exists()
    assert export_path.exists()


def test_cmd_archive_creates_zip_package(tmp_path, capsys):
    base_dir = tmp_path / "reports"

    report_dir = base_dir / "daily" / "2024-02-02"
    report_dir.mkdir(parents=True)
    summary_path = report_dir / "summary.json"
    write_fixture_text(summary_path, "{}")
    export_dir = report_dir / "exports"
    export_dir.mkdir()
    export_path = export_dir / "result.json"
    export_payload = {"value": 123}
    write_fixture_text(export_path, json.dumps(export_payload))

    destination = tmp_path / "archives"
    args = _build_archive_args(base_dir, destination=destination, archive_format="zip")

    return_code = ui_bridge.cmd_archive(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert payload["status"] == "completed"
    assert payload["format"] == "zip"

    archive_file = destination / "daily" / "2024-02-02.zip"
    assert archive_file.exists()

    with zipfile.ZipFile(archive_file, "r") as archive:
        names = archive.namelist()
        assert "2024-02-02/summary.json" in names
        assert "2024-02-02/exports/result.json" in names
        content = archive.read("2024-02-02/exports/result.json").decode("utf-8")
        assert json.loads(content) == export_payload

    # źródłowe pliki pozostają na miejscu
    assert summary_path.exists()
    assert export_path.exists()


def test_cmd_archive_creates_tar_package(tmp_path, capsys):
    base_dir = tmp_path / "reports"

    report_dir = base_dir / "daily" / "2024-02-03"
    report_dir.mkdir(parents=True)
    summary_path = report_dir / "summary.json"
    write_fixture_text(summary_path, "{}")
    export_path = report_dir / "exports" / "data.csv"
    export_path.parent.mkdir(parents=True)
    write_fixture_text(export_path, "id,value\n1,2\n")

    destination = tmp_path / "archives"
    args = _build_archive_args(base_dir, destination=destination, archive_format="tar")

    return_code = ui_bridge.cmd_archive(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert payload["status"] == "completed"
    assert payload["format"] == "tar"

    archive_file = destination / "daily" / "2024-02-03.tar.gz"
    assert archive_file.exists()

    with tarfile.open(archive_file, "r:gz") as archive:
        names = archive.getnames()
        assert "2024-02-03/summary.json" in names
        assert "2024-02-03/exports/data.csv" in names
        member = archive.extractfile("2024-02-03/exports/data.csv")
        assert member is not None
        content = member.read().decode("utf-8")
        assert content == "id,value\n1,2\n"

    assert summary_path.exists()
    assert export_path.exists()


def test_cmd_archive_requires_overwrite_for_existing_targets(tmp_path, capsys):
    base_dir = tmp_path / "reports"

    report_dir = base_dir / "audit" / "2024-03-01"
    report_dir.mkdir(parents=True)
    write_fixture_text((report_dir / "summary.json"), "{}")

    destination = tmp_path / "archives"
    existing_copy = destination / "audit" / "2024-03-01"
    existing_copy.mkdir(parents=True)
    write_fixture_text((existing_copy / "summary.json"), '{"existing": true}')

    args = _build_archive_args(base_dir, destination=destination)

    return_code = ui_bridge.cmd_archive(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert payload["status"] == "error"
    assert payload["copied_count"] == 0
    assert payload["errors"]

    # ponowne wywołanie z nadpisaniem powinno się powieść
    overwrite_args = _build_archive_args(base_dir, destination=destination, overwrite=True)
    overwrite_return_code = ui_bridge.cmd_archive(overwrite_args)
    overwrite_payload = json.loads(capsys.readouterr().out)

    assert overwrite_return_code == 0
    assert overwrite_payload["status"] == "completed"
    assert (destination / "audit" / "2024-03-01" / "summary.json").exists()


def test_cmd_archive_rejects_destination_inside_base(tmp_path, capsys):
    base_dir = tmp_path / "reports"
    base_dir.mkdir(parents=True)

    args = _build_archive_args(base_dir, destination=Path("."))

    return_code = ui_bridge.cmd_archive(args)

    captured = capsys.readouterr()

    assert return_code == 2
    assert "katalogu raportów" in captured.err.lower()


def test_cmd_overview_marks_invalid_summary(tmp_path, capsys):
    base_dir = tmp_path / "reports"
    report_dir = base_dir / "daily" / "2024-05-07"
    report_dir.mkdir(parents=True)

    summary_path = report_dir / "summary.json"
    write_fixture_text(summary_path, '{"report_date": }')

    export_path = report_dir / "snapshot.csv"
    export_content = "timestamp,value\n2024-05-07T00:00:00Z,1\n"
    write_fixture_text(export_path, export_content)

    fixed_timestamp = 1_700_200_000
    os.utime(summary_path, (fixed_timestamp, fixed_timestamp))
    os.utime(export_path, (fixed_timestamp, fixed_timestamp))

    args = SimpleNamespace(base_dir=str(base_dir))

    return_code = ui_bridge.cmd_overview(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert len(payload["reports"]) == 1
    report_entry = payload["reports"][0]
    category_entry = payload["categories"][0]
    assert report_entry["summary"] is None
    assert isinstance(report_entry["summary_error"], str)
    assert "Invalid JSON" in report_entry["summary_error"]
    assert report_entry["has_summary"] is False
    expected_timestamp = datetime.fromtimestamp(fixed_timestamp, tz=timezone.utc).isoformat()
    assert report_entry["created_at"] == expected_timestamp
    assert category_entry["earliest_updated_at"] == expected_timestamp

    categories = payload["categories"]
    assert len(categories) == 1
    category = categories[0]
    assert category["has_summary"] is False
    assert category["invalid_summary_count"] == 1
    assert category["export_count"] == 1
    assert category["missing_summary_count"] == 0

    summary = payload["summary"]
    assert summary["report_count"] == 1
    assert summary["category_count"] == 1
    assert summary["has_summary"] is False
    assert summary["has_exports"] is True
    assert summary["total_size"] == len(export_content.encode("utf-8"))
    assert summary["export_count"] == 1
    assert summary["invalid_summary_count"] == 1
    assert summary["latest_updated_at"] == expected_timestamp
    assert summary["earliest_updated_at"] == expected_timestamp
    assert summary["missing_summary_count"] == 0


def test_cmd_overview_filters_by_since(tmp_path, capsys):
    base_dir = tmp_path / "reports"

    old_report_dir = base_dir / "daily" / "2024-01-01"
    old_report_dir.mkdir(parents=True)
    old_summary = old_report_dir / "summary.json"
    write_fixture_text(old_summary, json.dumps({"report_date": "2024-01-01"}))
    old_export = old_report_dir / "data.csv"
    write_fixture_text(old_export, "a,b\n1,2\n")

    recent_report_dir = base_dir / "daily" / "2024-01-20"
    recent_report_dir.mkdir(parents=True)
    recent_summary = recent_report_dir / "summary.json"
    write_fixture_text(recent_summary, json.dumps({"report_date": "2024-01-20"}))
    recent_export = recent_report_dir / "data.csv"
    write_fixture_text(recent_export, "a,b\n3,4\n")

    old_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    recent_dt = datetime(2024, 1, 20, tzinfo=timezone.utc)
    old_timestamp = int(old_dt.timestamp())
    recent_timestamp = int(recent_dt.timestamp())
    os.utime(old_summary, (old_timestamp, old_timestamp))
    os.utime(old_export, (old_timestamp, old_timestamp))
    os.utime(recent_summary, (recent_timestamp, recent_timestamp))
    os.utime(recent_export, (recent_timestamp, recent_timestamp))

    since = datetime(2024, 1, 10, tzinfo=timezone.utc)
    args = SimpleNamespace(base_dir=str(base_dir), since=since, until=None)

    return_code = ui_bridge.cmd_overview(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert len(payload["reports"]) == 1
    assert payload["reports"][0]["relative_path"] == "daily/2024-01-20"
    assert payload["filters"] == {
        "since": since.isoformat(),
        "until": None,
        "categories": [],
        "summary_status": "any",
        "limit": None,
        "offset": None,
        "sort_key": "updated_at",
        "sort_direction": "desc",
        "query": None,
        "has_exports": "any",
    }
    assert payload["summary"]["report_count"] == 1


def test_cmd_overview_filters_by_until(tmp_path, capsys):
    base_dir = tmp_path / "reports"

    early_dir = base_dir / "daily" / "2024-01-01"
    early_dir.mkdir(parents=True)
    write_fixture_text((early_dir / "summary.json"), json.dumps({"report_date": "2024-01-01"}))
    write_fixture_text((early_dir / "data.csv"), "a,b\n1,2\n")

    late_dir = base_dir / "daily" / "2024-01-20"
    late_dir.mkdir(parents=True)
    write_fixture_text((late_dir / "summary.json"), json.dumps({"report_date": "2024-01-20"}))
    write_fixture_text((late_dir / "data.csv"), "a,b\n3,4\n")

    early_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    late_dt = datetime(2024, 1, 20, tzinfo=timezone.utc)
    os.utime(early_dir / "summary.json", (early_dt.timestamp(), early_dt.timestamp()))
    os.utime(early_dir / "data.csv", (early_dt.timestamp(), early_dt.timestamp()))
    os.utime(late_dir / "summary.json", (late_dt.timestamp(), late_dt.timestamp()))
    os.utime(late_dir / "data.csv", (late_dt.timestamp(), late_dt.timestamp()))

    until = datetime(2024, 1, 10, tzinfo=timezone.utc)
    args = SimpleNamespace(base_dir=str(base_dir), since=None, until=until)

    return_code = ui_bridge.cmd_overview(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert len(payload["reports"]) == 1
    assert payload["reports"][0]["relative_path"] == "daily/2024-01-01"
    assert payload["filters"] == {
        "since": None,
        "until": until.isoformat(),
        "categories": [],
        "summary_status": "any",
        "limit": None,
        "offset": None,
        "sort_key": "updated_at",
        "sort_direction": "desc",
        "query": None,
        "has_exports": "any",
    }


def test_cmd_overview_rejects_invalid_range(tmp_path, capsys):
    since = datetime(2024, 2, 1, tzinfo=timezone.utc)
    until = datetime(2024, 1, 1, tzinfo=timezone.utc)
    args = SimpleNamespace(base_dir=str(tmp_path), since=since, until=until)

    return_code = ui_bridge.cmd_overview(args)

    captured = capsys.readouterr()

    assert return_code == 2
    assert captured.out == ""
    assert "--since must be earlier" in captured.err


def test_cmd_overview_category_filter(tmp_path, capsys):
    base_dir = tmp_path / "reports"

    audit_dir = base_dir / "audit" / "2024-02-01"
    audit_dir.mkdir(parents=True)
    write_fixture_text((audit_dir / "summary.json"), json.dumps({"report_date": "2024-02-01"}))
    write_fixture_text((audit_dir / "export.csv"), "row\n")

    diagnostics_dir = base_dir / "diagnostics" / "2024-02-02"
    diagnostics_dir.mkdir(parents=True)
    write_fixture_text(
        (diagnostics_dir / "summary.json"), json.dumps({"report_date": "2024-02-02"})
    )
    write_fixture_text((diagnostics_dir / "export.csv"), "row\n")

    args = SimpleNamespace(base_dir=str(base_dir), categories=["audit"])

    return_code = ui_bridge.cmd_overview(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert payload["filters"] == {
        "since": None,
        "until": None,
        "categories": ["audit"],
        "summary_status": "any",
        "limit": None,
        "offset": None,
        "sort_key": "updated_at",
        "sort_direction": "desc",
        "query": None,
        "has_exports": "any",
    }
    assert len(payload["reports"]) == 1
    assert payload["reports"][0]["category"] == "audit"
    assert all(report["category"] == "audit" for report in payload["reports"])
    assert len(payload["categories"]) == 1
    assert payload["categories"][0]["id"] == "audit"


def test_cmd_overview_query_filter(tmp_path, capsys):
    base_dir = tmp_path / "reports"

    audit_dir = base_dir / "audit" / "2024-05-01"
    audit_dir.mkdir(parents=True)
    write_fixture_text((audit_dir / "summary.json"), json.dumps({"report_date": "2024-05-01"}))

    diagnostics_dir = base_dir / "diagnostics" / "2024-05-02"
    diagnostics_dir.mkdir(parents=True)
    write_fixture_text(
        (diagnostics_dir / "summary.json"), json.dumps({"report_date": "2024-05-02"})
    )

    args = SimpleNamespace(base_dir=str(base_dir), query="diag")

    return_code = ui_bridge.cmd_overview(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert len(payload["reports"]) == 1
    assert payload["reports"][0]["category"] == "diagnostics"
    assert payload["filters"]["query"] == "diag"


def test_cmd_overview_query_filter_casefold(tmp_path, capsys):
    base_dir = tmp_path / "reports"

    report_dir = base_dir / "audit" / "2024-05-03"
    report_dir.mkdir(parents=True)
    write_fixture_text((report_dir / "summary.json"), json.dumps({"report_date": "2024-05-03"}))

    args = SimpleNamespace(base_dir=str(base_dir), query="AUDIT")

    return_code = ui_bridge.cmd_overview(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert len(payload["reports"]) == 1
    assert payload["reports"][0]["category"] == "audit"
    assert payload["filters"]["query"] == "AUDIT"


def test_cmd_overview_query_filter_blank_ignored(tmp_path, capsys):
    base_dir = tmp_path / "reports"
    base_dir.mkdir()

    args = SimpleNamespace(base_dir=str(base_dir), query="   ")

    return_code = ui_bridge.cmd_overview(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert payload["filters"]["query"] is None


def test_cmd_overview_category_filter_no_matches(tmp_path, capsys):
    base_dir = tmp_path / "reports"
    base_dir.mkdir()

    args = SimpleNamespace(base_dir=str(base_dir), categories=["nonexistent"])

    return_code = ui_bridge.cmd_overview(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert payload["reports"] == []
    assert payload["categories"] == []
    assert payload["filters"] == {
        "since": None,
        "until": None,
        "categories": ["nonexistent"],
        "summary_status": "any",
        "limit": None,
        "offset": None,
        "sort_key": "updated_at",
        "sort_direction": "desc",
        "query": None,
        "has_exports": "any",
    }


def test_cmd_overview_summary_status_filters(tmp_path, capsys):
    base_dir = tmp_path / "reports"

    valid_dir = base_dir / "audit" / "2024-04-01"
    valid_dir.mkdir(parents=True)
    write_fixture_text(
        (valid_dir / "summary.json"),
        json.dumps({"report_date": "2024-04-01", "status": "ok"}),
    )
    write_fixture_text((valid_dir / "export.csv"), "row\n")

    missing_dir = base_dir / "diagnostics" / "2024-04-02"
    missing_dir.mkdir(parents=True)
    write_fixture_text((missing_dir / "metrics.json"), "{}")

    invalid_dir = base_dir / "audit" / "2024-04-03"
    invalid_dir.mkdir(parents=True)
    write_fixture_text((invalid_dir / "summary.json"), "{\n")
    write_fixture_text((invalid_dir / "export.csv"), "row\n")

    args = SimpleNamespace(base_dir=str(base_dir), summary_status="valid")

    return_code = ui_bridge.cmd_overview(args)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert [report["relative_path"] for report in payload["reports"]] == ["audit/2024-04-01"]
    assert payload["filters"]["summary_status"] == "valid"
    assert payload["summary"]["missing_summary_count"] == 0
    assert payload["summary"]["invalid_summary_count"] == 0

    args.summary_status = "missing"
    return_code = ui_bridge.cmd_overview(args)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert [report["relative_path"] for report in payload["reports"]] == ["diagnostics/2024-04-02"]
    assert payload["filters"]["summary_status"] == "missing"
    assert payload["reports"][0]["summary_path"] is None
    assert payload["summary"]["missing_summary_count"] == 1

    args.summary_status = "invalid"
    return_code = ui_bridge.cmd_overview(args)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert [report["relative_path"] for report in payload["reports"]] == ["audit/2024-04-03"]
    assert payload["filters"]["summary_status"] == "invalid"
    assert payload["reports"][0]["summary_error"].startswith("Invalid JSON")
    assert payload["summary"]["invalid_summary_count"] == 1


def test_cmd_overview_summary_status_invalid_value(tmp_path, capsys):
    args = SimpleNamespace(base_dir=str(tmp_path), summary_status="unsupported")

    return_code = ui_bridge.cmd_overview(args)

    captured = capsys.readouterr()

    assert return_code == 2
    assert captured.out == ""
    assert "Unsupported summary status filter" in captured.err


def test_cmd_overview_has_exports_filter(tmp_path, capsys):
    base_dir = tmp_path / "reports"

    with_exports = base_dir / "audit" / "2024-06-01"
    with_exports.mkdir(parents=True)
    write_fixture_text((with_exports / "summary.json"), json.dumps({"report_date": "2024-06-01"}))
    write_fixture_text((with_exports / "export.csv"), "row\n")

    without_exports = base_dir / "diagnostics" / "2024-06-02"
    without_exports.mkdir(parents=True)
    write_fixture_text(
        (without_exports / "summary.json"), json.dumps({"report_date": "2024-06-02"})
    )

    args = SimpleNamespace(base_dir=str(base_dir), has_exports="yes")

    return_code = ui_bridge.cmd_overview(args)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert [report["relative_path"] for report in payload["reports"]] == ["audit/2024-06-01"]
    assert payload["filters"]["has_exports"] == "yes"
    assert payload["reports"][0]["has_exports"] is True

    args.has_exports = "no"
    return_code = ui_bridge.cmd_overview(args)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert [report["relative_path"] for report in payload["reports"]] == ["diagnostics/2024-06-02"]
    assert payload["filters"]["has_exports"] == "no"
    assert payload["reports"][0]["has_exports"] is False


def test_cmd_overview_has_exports_invalid_value(tmp_path, capsys):
    args = SimpleNamespace(base_dir=str(tmp_path), has_exports="maybe")

    return_code = ui_bridge.cmd_overview(args)

    captured = capsys.readouterr()

    assert return_code == 2
    assert captured.out == ""
    assert "Unsupported exports filter value" in captured.err


def test_cmd_overview_limits_number_of_reports(tmp_path, capsys):
    base_dir = tmp_path / "reports"

    first_dir = base_dir / "daily" / "2024-05-01"
    first_dir.mkdir(parents=True)
    write_fixture_text((first_dir / "summary.json"), json.dumps({"report_date": "2024-05-01"}))
    write_fixture_text((first_dir / "export.csv"), "row\n")

    second_dir = base_dir / "daily" / "2024-05-02"
    second_dir.mkdir(parents=True)
    write_fixture_text((second_dir / "summary.json"), json.dumps({"report_date": "2024-05-02"}))
    write_fixture_text((second_dir / "export.csv"), "row\n")

    third_dir = base_dir / "daily" / "2024-05-03"
    third_dir.mkdir(parents=True)
    write_fixture_text((third_dir / "summary.json"), json.dumps({"report_date": "2024-05-03"}))
    write_fixture_text((third_dir / "export.csv"), "row\n")

    timestamps = [
        datetime(2024, 5, 1, 12, tzinfo=timezone.utc),
        datetime(2024, 5, 2, 12, tzinfo=timezone.utc),
        datetime(2024, 5, 3, 12, tzinfo=timezone.utc),
    ]
    for directory, dt in zip([first_dir, second_dir, third_dir], timestamps, strict=True):
        summary = directory / "summary.json"
        export = directory / "export.csv"
        ts = int(dt.timestamp())
        os.utime(summary, (ts, ts))
        os.utime(export, (ts, ts))

    args = SimpleNamespace(base_dir=str(base_dir), limit=2)

    return_code = ui_bridge.cmd_overview(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert [report["relative_path"] for report in payload["reports"]] == [
        "daily/2024-05-03",
        "daily/2024-05-02",
    ]
    assert payload["filters"]["limit"] == 2
    assert payload["filters"]["offset"] is None
    assert payload["summary"]["report_count"] == 3
    assert payload["pagination"] == {
        "total_count": 3,
        "returned_count": 2,
        "limit": 2,
        "offset": None,
        "has_more": True,
        "has_previous": False,
    }


def test_cmd_overview_rejects_nonpositive_limit(tmp_path, capsys):
    args = SimpleNamespace(base_dir=str(tmp_path), limit=0)

    return_code = ui_bridge.cmd_overview(args)

    captured = capsys.readouterr()

    assert return_code == 2
    assert captured.out == ""
    assert "--limit must be greater than zero" in captured.err


def test_cmd_overview_skips_reports_with_offset(tmp_path, capsys):
    base_dir = tmp_path / "reports"

    for index, day in enumerate(["2024-06-01", "2024-06-02", "2024-06-03"], start=1):
        report_dir = base_dir / "daily" / day
        report_dir.mkdir(parents=True)
        write_fixture_text((report_dir / "summary.json"), json.dumps({"report_date": day}))
        write_fixture_text((report_dir / "export.csv"), "row\n")
        timestamp = datetime(2024, 6, index, 12, tzinfo=timezone.utc)
        ts = int(timestamp.timestamp())
        os.utime(report_dir / "summary.json", (ts, ts))
        os.utime(report_dir / "export.csv", (ts, ts))

    args = SimpleNamespace(base_dir=str(base_dir), offset=1)

    return_code = ui_bridge.cmd_overview(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert [report["relative_path"] for report in payload["reports"]] == [
        "daily/2024-06-02",
        "daily/2024-06-01",
    ]
    assert payload["filters"]["offset"] == 1
    assert payload["filters"]["limit"] is None
    assert payload["summary"]["report_count"] == 3
    assert payload["pagination"] == {
        "total_count": 3,
        "returned_count": 2,
        "limit": None,
        "offset": 1,
        "has_more": False,
        "has_previous": True,
    }

    args.limit = 1
    return_code = ui_bridge.cmd_overview(args)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert [report["relative_path"] for report in payload["reports"]] == [
        "daily/2024-06-02",
    ]
    assert payload["filters"]["offset"] == 1
    assert payload["filters"]["limit"] == 1
    assert payload["summary"]["report_count"] == 3
    assert payload["pagination"] == {
        "total_count": 3,
        "returned_count": 1,
        "limit": 1,
        "offset": 1,
        "has_more": True,
        "has_previous": True,
    }


def test_cmd_overview_rejects_negative_offset(tmp_path, capsys):
    args = SimpleNamespace(base_dir=str(tmp_path), offset=-1)

    return_code = ui_bridge.cmd_overview(args)

    captured = capsys.readouterr()

    assert return_code == 2
    assert captured.out == ""
    assert "--offset must be zero or positive" in captured.err


def test_cmd_overview_sorts_by_name(tmp_path, capsys):
    base_dir = tmp_path / "reports"

    first_dir = base_dir / "daily" / "2024-02-01"
    second_dir = base_dir / "daily" / "2024-02-02"
    first_dir.mkdir(parents=True)
    second_dir.mkdir(parents=True)

    write_fixture_text((first_dir / "summary.json"), json.dumps({"report_date": "2024-02-01"}))
    write_fixture_text((second_dir / "summary.json"), json.dumps({"report_date": "2024-02-02"}))

    write_fixture_text((first_dir / "export.csv"), "row\n")
    write_fixture_text((second_dir / "export.csv"), "row\n")

    # ustawiamy identyczne znaczniki czasu, aby kolejność zależała tylko od sortowania
    fixed_timestamp = 1_700_500_000
    for path in [
        first_dir / "summary.json",
        first_dir / "export.csv",
        second_dir / "summary.json",
        second_dir / "export.csv",
    ]:
        os.utime(path, (fixed_timestamp, fixed_timestamp))

    args = SimpleNamespace(base_dir=str(base_dir), sort_key="name", sort_direction="asc")

    return_code = ui_bridge.cmd_overview(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert [report["relative_path"] for report in payload["reports"]] == [
        "daily/2024-02-01",
        "daily/2024-02-02",
    ]
    assert payload["filters"]["sort_key"] == "name"
    assert payload["filters"]["sort_direction"] == "asc"


def test_cmd_overview_sorts_by_size_descending(tmp_path, capsys):
    base_dir = tmp_path / "reports"

    small_dir = base_dir / "audit" / "small"
    large_dir = base_dir / "audit" / "large"
    small_dir.mkdir(parents=True)
    large_dir.mkdir(parents=True)

    write_fixture_text((small_dir / "summary.json"), json.dumps({"report_date": "2024-03-01"}))
    write_fixture_text((large_dir / "summary.json"), json.dumps({"report_date": "2024-03-02"}))

    write_fixture_text((small_dir / "export.csv"), "a\n")
    write_fixture_text((large_dir / "export.csv"), "a\n" * 100)

    ts_small = 1_700_600_000
    ts_large = 1_700_700_000
    os.utime(small_dir / "summary.json", (ts_small, ts_small))
    os.utime(small_dir / "export.csv", (ts_small, ts_small))
    os.utime(large_dir / "summary.json", (ts_large, ts_large))
    os.utime(large_dir / "export.csv", (ts_large, ts_large))

    args = SimpleNamespace(base_dir=str(base_dir), sort_key="size", sort_direction="desc")

    return_code = ui_bridge.cmd_overview(args)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert return_code == 0
    assert [report["relative_path"] for report in payload["reports"]] == [
        "audit/large",
        "audit/small",
    ]
    assert payload["filters"]["sort_key"] == "size"
    assert payload["filters"]["sort_direction"] == "desc"


def test_cmd_overview_rejects_invalid_sort_options(tmp_path, capsys):
    args = SimpleNamespace(base_dir=str(tmp_path), sort_key="unknown")

    return_code = ui_bridge.cmd_overview(args)

    captured = capsys.readouterr()

    assert return_code == 2
    assert captured.out == ""
    assert "Unsupported sort key" in captured.err

    args = SimpleNamespace(base_dir=str(tmp_path), sort_key="name", sort_direction="sideways")

    return_code = ui_bridge.cmd_overview(args)

    captured = capsys.readouterr()

    assert return_code == 2
    assert captured.out == ""
    assert "Unsupported sort direction" in captured.err


def _write_report(tmp_path: Path, name: str) -> Path:
    report_dir = tmp_path / name
    report_dir.mkdir()
    write_fixture_text((report_dir / "summary.json"), "{}")
    write_fixture_text((report_dir / "details.txt"), "details")
    return report_dir


def test_delete_report_directory(tmp_path: Path) -> None:
    report_dir = _write_report(tmp_path, "report-123")

    result = ui_bridge.delete_report(str(report_dir), root=str(tmp_path))

    assert result["status"] == "ok"
    assert not report_dir.exists()
    assert result["removed_entries"] >= 2  # pliki + katalog
    assert result["size_bytes"] > 0


def test_delete_report_outside_root_guarded(tmp_path: Path) -> None:
    report_dir = _write_report(tmp_path, "report-456")
    outside = tmp_path.parent / "report-outside"
    outside.mkdir()

    result = ui_bridge.delete_report(str(outside), root=str(tmp_path))

    assert result["status"] == "forbidden"
    assert outside.exists()


def test_delete_report_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    result = ui_bridge.delete_report(str(missing), root=str(tmp_path))

    assert result["status"] == "not_found"


def test_cli_delete_returns_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report_dir = _write_report(tmp_path, "report-cli")

    exit_code = ui_bridge.main(
        [
            "delete",
            str(report_dir),
            "--root",
            str(tmp_path),
        ]
    )
    # main zwraca kod wyjścia, wynik wypisywany jest na stdout, co testujemy poniżej
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "deleted"

    # katalog powinien zostać usunięty
    assert not report_dir.exists()


def test_list_reports_includes_relative_paths(tmp_path: Path) -> None:
    report_dir = _write_report(tmp_path, "report-list")

    payload = ui_bridge.list_reports(root=str(tmp_path))

    assert payload["status"] == "ok"
    reports = {entry["relative_path"]: entry for entry in payload["reports"]}
    assert "report-list" in reports
    entry = reports["report-list"]
    assert entry["type"] == "directory"
    assert entry["size_bytes"] > 0
    assert entry["path"].endswith("report-list")


def test_cmd_promote_creates_audit_entry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    quality_dir = tmp_path / "quality"
    audit_dir = tmp_path / "audit"

    record_model_quality_report(
        _quality_report("v1", directional=0.61, mae=15.0), history_root=quality_dir
    )
    record_model_quality_report(
        _quality_report("v2", directional=0.6, mae=14.5, status="ok"), history_root=quality_dir
    )

    args = SimpleNamespace(
        model="demo",
        version="v2",
        quality_dir=str(quality_dir),
        audit_dir=str(audit_dir),
        reason="Manual override",
    )

    return_code = ui_bridge.cmd_promote(args)
    captured = capsys.readouterr()

    assert return_code == 0
    payload = json.loads(captured.out)
    assert payload["status"] == "ok"
    assert payload["model"] == "demo"
    assert payload["version"] == "v2"

    summary_path = Path(payload["audit_entry"]["summary_path"])
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "promoted"
    assert summary["candidate"]["version"] == "v2"
    assert summary["reason"] == "Manual override"

    overview = load_champion_overview("demo", base_dir=quality_dir)
    assert overview is not None
    assert overview["champion"]["version"] == "v2"
    assert overview["challengers"], "Challengers list should contain previous champion"
    assert overview["challengers"][0]["report"]["version"] == "v1"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
