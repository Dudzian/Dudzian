from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

from bot_core.security.rotation import RotationRegistry
from bot_core.security.rotation_report import (
    RotationRecord,
    RotationSummary,
    build_rotation_summary_entry,
    write_rotation_summary,
)


def test_build_rotation_summary_entry_includes_signature(tmp_path: Path) -> None:
    registry_path = tmp_path / "rotation.json"
    registry = RotationRegistry(registry_path)

    executed_at = datetime(2024, 5, 1, 10, 0, tzinfo=timezone.utc)
    status = registry.status("binance_paper", "trading", interval_days=30.0, now=executed_at)

    record = RotationRecord(
        environment="paper",
        key="binance_paper",
        purpose="trading",
        registry_path=registry_path,
        status_before=status,
        rotated_at=executed_at,
        interval_days=30.0,
        metadata={"exchange": "binance"},
    )

    summary = RotationSummary(
        operator="SecOps",
        executed_at=executed_at,
        records=[record],
        notes="Rotacja planowa",
    )

    entry = build_rotation_summary_entry(summary, signing_key=b"demo_key", signing_key_id="stage5")

    assert entry["type"] == "stage5_key_rotation"
    assert entry["stats"]["total"] == 1
    assert entry["records"][0]["environment"] == "paper"
    assert entry["records"][0]["next_due_at"].startswith("2024-05-31")
    assert entry["signature"]["key_id"] == "stage5"


def test_write_rotation_summary_persists_json(tmp_path: Path) -> None:
    registry_path = tmp_path / "rotation.json"
    registry = RotationRegistry(registry_path)

    executed_at = datetime(2024, 6, 1, 12, 30, tzinfo=timezone.utc)
    status = registry.status("kraken_live", "trading", interval_days=45.0, now=executed_at)

    record = RotationRecord(
        environment="live",
        key="kraken_live",
        purpose="trading",
        registry_path=registry_path,
        status_before=status,
        rotated_at=executed_at,
        interval_days=45.0,
    )

    summary = RotationSummary(operator="Ops", executed_at=executed_at, records=[record])

    output_path = tmp_path / "report.json"
    result_path = write_rotation_summary(summary, output=output_path)

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["records"][0]["registry_path"].endswith("rotation.json")
    assert payload["records"][0]["was_due"] is True
    assert payload["stats"]["total"] == 1
