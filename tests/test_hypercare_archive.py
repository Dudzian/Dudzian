from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import json

from bot_core.runtime.hypercare_archive import archive_hypercare_reports


def _create_report(path: Path, payload: dict[str, object] | None = None) -> None:
    path.write_text(json.dumps(payload or {"type": "demo"}), encoding="utf-8")


def test_archive_hypercare_reports_copies_all_files(tmp_path: Path) -> None:
    stage5 = tmp_path / "stage5.json"
    stage6 = tmp_path / "stage6.json"
    full = tmp_path / "full.json"
    extra = tmp_path / "notes.txt"
    stage5_sig = stage5.with_suffix(".sig")
    stage6_sig = stage6.with_suffix(".sig")

    for path in (stage5, stage6, full, extra, stage5_sig, stage6_sig):
        _create_report(path, {"source": path.name})

    archive_dir = tmp_path / "archive"
    timestamp = datetime(2024, 5, 20, 12, 30, tzinfo=timezone.utc)

    target_dir = archive_hypercare_reports(
        archive_dir=archive_dir,
        stage5_summary=stage5,
        stage6_summary=stage6,
        stage5_signature=stage5_sig,
        stage6_signature=stage6_sig,
        full_summary=full,
        extra_files=[extra],
        timestamp=timestamp,
    )

    assert target_dir.exists()
    expected_files = {
        "stage5.json",
        "stage6.json",
        "stage5.sig",
        "stage6.sig",
        "full.json",
        "notes.txt",
    }
    archived = {path.name for path in target_dir.iterdir()}
    assert expected_files.issubset(archived)


def test_archive_hypercare_reports_raises_for_missing_required(tmp_path: Path) -> None:
    stage5 = tmp_path / "stage5.json"
    stage6 = tmp_path / "stage6.json"
    _create_report(stage6)

    archive_dir = tmp_path / "archive"

    try:
        archive_hypercare_reports(
            archive_dir=archive_dir,
            stage5_summary=stage5,
            stage6_summary=stage6,
        )
    except FileNotFoundError as exc:
        assert "stage5.json" in str(exc)
    else:  # pragma: no cover - sanity guard
        assert False, "expected FileNotFoundError"


def test_archive_hypercare_reports_ignores_missing_optionals(tmp_path: Path) -> None:
    stage5 = tmp_path / "stage5.json"
    stage6 = tmp_path / "stage6.json"
    _create_report(stage5)
    _create_report(stage6)

    archive_dir = tmp_path / "archive"

    target_dir = archive_hypercare_reports(
        archive_dir=archive_dir,
        stage5_summary=stage5,
        stage6_summary=stage6,
        stage5_signature=stage5.with_suffix(".sig"),  # does not exist
        stage6_signature=stage6.with_suffix(".sig"),  # does not exist
        full_summary=tmp_path / "full.json",  # does not exist
        extra_files=[tmp_path / "missing.json"],
    )

    assert target_dir.exists()
    assert (target_dir / "stage5.json").exists()
    assert (target_dir / "stage6.json").exists()
    assert not (target_dir / "full.json").exists()
