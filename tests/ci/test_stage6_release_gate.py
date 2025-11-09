"""Testy smoke dla bramki Stage6 w pipeline wydawniczym."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import verify_stage6_hypercare_summary as stage6_cli


def _minimal_stage6_payload() -> dict[str, object]:
    return {
        "type": "stage6_hypercare_summary",
        "overall_status": "ok",
        "issues": [],
        "warnings": [],
        "components": {
            "observability": {"status": "ok"},
            "resilience": {"status": "ok"},
            "portfolio": {"status": "ok"},
        },
    }


def test_stage6_gate_fails_without_summary(tmp_path: Path) -> None:
    missing_path = tmp_path / "stage6_hypercare_summary.json"
    exit_code = stage6_cli.run([missing_path.as_posix(), "--require-signature"])
    assert exit_code == 1


def test_stage6_gate_requires_signature(tmp_path: Path) -> None:
    summary_path = tmp_path / "stage6_hypercare_summary.json"
    summary_path.write_text(json.dumps(_minimal_stage6_payload()), encoding="utf-8")

    exit_code = stage6_cli.run([summary_path.as_posix(), "--require-signature"])
    assert exit_code == 2
