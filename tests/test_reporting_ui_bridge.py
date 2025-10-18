from __future__ import annotations

import json
from pathlib import Path

import pytest

from bot_core.reporting import ui_bridge


def _write_report(tmp_path: Path, name: str) -> Path:
    report_dir = tmp_path / name
    report_dir.mkdir()
    (report_dir / "summary.json").write_text("{}", encoding="utf-8")
    (report_dir / "details.txt").write_text("details", encoding="utf-8")
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
    assert payload["status"] == "ok"

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


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
