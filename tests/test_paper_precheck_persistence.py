"""Testy zapisu raportów paper_precheck do katalogu audytu."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from scripts import paper_precheck


def test_persist_precheck_report_creates_unique_file(tmp_path: Path) -> None:
    payload = {"status": "ok", "foo": "bar"}
    created_at = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    metadata_first = paper_precheck.persist_precheck_report(
        payload,
        environment_name="binance paper",
        base_dir=tmp_path,
        created_at=created_at,
    )

    metadata_second = paper_precheck.persist_precheck_report(
        payload,
        environment_name="binance paper",
        base_dir=tmp_path,
        created_at=created_at,
    )

    first_path = Path(metadata_first["path"])
    second_path = Path(metadata_second["path"])

    assert first_path.exists()
    assert second_path.exists()
    assert first_path != second_path
    assert metadata_first["sha256"] == metadata_second["sha256"]
    assert metadata_first["created_at"] == created_at.replace(microsecond=0).isoformat()
    assert metadata_first["environment"] == "binance paper"
    assert metadata_first["size_bytes"] > 0
    assert first_path.read_text(encoding="utf-8").strip().startswith("{")
