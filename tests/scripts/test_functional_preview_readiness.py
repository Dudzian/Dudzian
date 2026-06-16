from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/ci/functional_preview_readiness.py"
REPORT = REPO_ROOT / "reports/ci/functional_preview_readiness.json"
REQUIRED_SECTIONS = {
    "data_source_market_feed",
    "scanner_opportunity_pipeline",
    "ai_decision_governor",
    "paper_terminal_order_lifecycle",
    "portfolio_positions_trades",
    "alerts_telemetry_audit",
    "settings_config_api_keys",
    "runtime_session_control_plane",
    "live_safety_hard_gate",
}
VALID_STATUSES = {"functional", "partial", "static_mock_only", "missing", "unknown"}


def _load_report() -> dict[str, object]:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_script_generates_deterministic_report(tmp_path: Path) -> None:
    output = tmp_path / "readiness.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    generated = json.loads(output.read_text(encoding="utf-8"))
    tracked = _load_report()
    assert generated == tracked
    assert generated["evaluated_at"] == "2026-06-16T00:00:00Z"


@pytest.mark.parametrize("section_name", sorted(REQUIRED_SECTIONS))
def test_report_section_schema(section_name: str) -> None:
    payload = _load_report()
    sections = payload["sections"]
    assert isinstance(sections, dict)
    assert section_name in sections
    section = sections[section_name]
    assert isinstance(section, dict)
    assert section["status"] in VALID_STATUSES
    assert isinstance(section["evidence_files"], list)
    assert section["evidence_files"], f"{section_name} needs evidence files"
    assert all(isinstance(item, str) for item in section["evidence_files"])
    assert all((REPO_ROOT / item).exists() for item in section["evidence_files"])
    assert isinstance(section["runtime_backed"], bool)
    assert isinstance(section["static_qml_only"], bool)
    assert isinstance(section["supports_test_server"], bool)
    assert isinstance(section["supports_read_only_real_data"], bool)
    assert isinstance(section["paper_only_execution_safe"], bool)
    assert isinstance(section["gaps"], list)
    assert all(isinstance(item, str) and item for item in section["gaps"])
    assert isinstance(section["recommended_next_step"], str)
    assert section["recommended_next_step"].strip()


def test_report_does_not_overstate_functional_readiness() -> None:
    payload = _load_report()
    sections = payload["sections"]
    assert isinstance(sections, dict)
    assert all(section["status"] != "functional" for section in sections.values())
    assert all(not section["supports_test_server"] for section in sections.values())
    assert all(not section["supports_read_only_real_data"] for section in sections.values())
    for name, section in sections.items():
        if section["status"] == "static_mock_only":
            assert section["static_qml_only"] is True
            assert section["runtime_backed"] is False
        if section["runtime_backed"] is False:
            assert section["status"] in {"partial", "static_mock_only", "missing", "unknown"}, name


def test_live_safety_is_not_marked_as_complete() -> None:
    payload = _load_report()
    sections = payload["sections"]
    assert isinstance(sections, dict)
    safety = sections["live_safety_hard_gate"]
    assert safety["paper_only_execution_safe"] is True
    assert safety["status"] == "partial"
    assert any("not for every backend/runtime/live adapter" in gap for gap in safety["gaps"])
