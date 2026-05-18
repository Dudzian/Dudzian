from __future__ import annotations

import json
from pathlib import Path

from scripts.audit.marketing_parity_skip_report import main, write_skip_reports


def test_write_skip_reports_writes_infra_skip_contract(tmp_path: Path) -> None:
    md_path = tmp_path / "docs" / "audit" / "marketing_parity_report.md"
    json_path = tmp_path / "docs" / "audit" / "marketing_parity_report.json"
    reason = "missing_stress_lab_report_skipped_due_to_missing_upstream_report"

    write_skip_reports(md_path, json_path, reason=reason)

    assert md_path.parent.exists()
    assert json_path.parent.exists()
    markdown = md_path.read_text(encoding="utf-8")
    assert "infrastructure precondition" in markdown
    assert "not a parity pass" in markdown
    assert "status=skipped" in markdown
    assert f"Reason: `{reason}`." in markdown

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["status"] == "skipped"
    assert payload["reason"] == reason
    assert payload["classification"] == "infra_skip"
    assert payload["parity_validated"] is False
    assert payload["result"] == "non_passing"
    assert payload["status"] != "passed"
    assert payload["required"] == [
        "MARKETING_PARITY_MIRROR_S3",
        "MARKETING_PARITY_MIRROR_GIT",
    ]


def test_main_supports_custom_reason(tmp_path: Path) -> None:
    md_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"

    exit_code = main(
        [
            "--audit-output",
            str(md_path),
            "--json-output",
            str(json_path),
            "--reason",
            "mirror_temporarily_unavailable",
        ]
    )

    assert exit_code == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["reason"] == "mirror_temporarily_unavailable"

    markdown = md_path.read_text(encoding="utf-8")
    assert "Reason: `mirror_temporarily_unavailable`." in markdown
    assert "infrastructure precondition failure" in markdown
