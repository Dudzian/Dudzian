from __future__ import annotations

import json
from pathlib import Path

import pytest

from deployment.platform_readiness import EvidenceValidationError, _validated_results, load_contract
from deployment.windows_stage10_preflight import main, prepare_report


def test_missing_real_host_and_artifact_never_pass() -> None:
    report = prepare_report()
    assert report["overall_status"] == "BLOCKED"
    assert [c["status"] for c in report["checks"][:2]] == ["UNVERIFIED", "UNVERIFIED"]
    assert report["stage10_execution_authorized"] is False
    assert report["stage10_qualification"] == "NOT_RUN"
    assert report["next_task"] == "S9-S10-PRE-01B"


def test_even_all_pass_components_cannot_be_stage10_evidence(tmp_path: Path) -> None:
    files = []
    for kind in ("runner", "artifact"):
        path = tmp_path / f"{kind}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "report_type": f"windows-stage10-read-only-{kind}-preflight",
                    "overall_status": "PASS",
                    "stage10_qualification": "NOT_RUN",
                    "checks": [{"name": "test", "status": "PASS", "details": "fixture check"}],
                }
            )
        )
        files.append(path)
    report = prepare_report(runner_report=files[0], artifact_report=files[1])
    assert [c["status"] for c in report["checks"][:2]] == ["PASS", "PASS"]
    assert report["overall_status"] == "BLOCKED"
    with pytest.raises(EvidenceValidationError):
        _validated_results(load_contract(), [report], "revision", "https://github.com", "42")


@pytest.mark.parametrize("content", ["{", "[]", "{}", '{"overall_status": []}'])
def test_malformed_imported_report_fails_closed(tmp_path: Path, content: str) -> None:
    path = tmp_path / "report.json"
    path.write_text(content)
    assert prepare_report(runner_report=path)["overall_status"] == "FAIL"


def test_cli_writes_blocked_current_report_and_nonzero_exit(tmp_path: Path) -> None:
    path = tmp_path / "preflight.json"
    assert main(["--output", str(path)]) == 1
    assert json.loads(path.read_text())["overall_status"] == "BLOCKED"


def test_missing_status_source_fails_closed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("deployment.windows_stage10_preflight.CURRENT_STATUS", tmp_path / "absent")
    assert prepare_report()["overall_status"] == "FAIL"


def test_cli_cannot_overwrite_inputs_or_follow_output_links(tmp_path: Path) -> None:
    existing = tmp_path / "input.json"
    existing.write_text("preserve qualified input")
    assert main(["--output", str(existing)]) == 1
    assert existing.read_text() == "preserve qualified input"
    linked = tmp_path / "link"
    linked.symlink_to(tmp_path, target_is_directory=True)
    assert main(["--output", str(linked / "new.json")]) == 1
    assert not (tmp_path / "new.json").exists()
