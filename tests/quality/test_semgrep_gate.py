from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.quality import semgrep_gate


ROOT = Path(__file__).resolve().parents[2]


def test_ci_does_not_mask_community_scan_errors() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/quality-security.yml").read_text())
    steps = workflow["jobs"]["security-static"]["steps"]
    community = next(step for step in steps if "community baseline" in step.get("name", ""))

    assert community.get("continue-on-error", False) is False
    assert "scripts/quality/semgrep_gate.py" in community["run"]


def test_ci_checks_both_scan_reports_and_preserves_failure_artifacts() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/quality-security.yml").read_text())
    steps = workflow["jobs"]["security-static"]["steps"]
    blocking = next(step for step in steps if "CryptoHunter blocking" in step.get("name", ""))
    upload = next(step for step in steps if step.get("name") == "Upload security reports")

    assert "scripts/quality/semgrep_gate.py" in blocking["run"]
    assert "--blocking" in blocking["run"]
    assert upload["if"] == "always()"
    assert "semgrep-community.summary.json" in upload["with"]["path"]
    assert "semgrep-blocking.summary.json" in upload["with"]["path"]
    assert "semgrep-community.json" not in upload["with"]["path"]
    assert "semgrep-blocking.json" not in upload["with"]["path"]


def test_independent_scanners_still_run_after_an_incomplete_community_scan() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/quality-security.yml").read_text())
    job = workflow["jobs"]["security-static"]
    assert job["timeout-minutes"] == 30
    expected = {
        "Semgrep CryptoHunter blocking rules",
        "Semgrep CryptoHunter rule fixtures",
        "Install verified Betterleaks stable",
        "Betterleaks working tree (offline, blocking)",
    }
    independent = {step["name"]: step for step in job["steps"] if step.get("name") in expected}
    assert set(independent) == expected
    for step in independent.values():
        assert step["if"] == "${{ !cancelled() }}"
        assert not step.get("continue-on-error", False)


def _complete_report() -> dict[str, object]:
    return {
        "version": "1.179.0",
        "errors": [],
        "results": [],
        "skipped_rules": [],
        "paths": {"scanned": ["bot_core/example.py"], "skipped": []},
    }


def _write_report(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_empty_findings_are_accepted_only_with_complete_report(tmp_path: Path) -> None:
    assert (
        semgrep_gate.validate_report(_write_report(tmp_path, _complete_report()), blocking=False)
        == []
    )


@pytest.mark.parametrize("error_type", ["ParseError", "PartialParsing", "Timeout", "OutOfMemory"])
def test_all_reported_errors_refuse_zero_finding_report(tmp_path: Path, error_type: str) -> None:
    data = _complete_report()
    data["errors"] = [{"type": error_type, "level": "warn", "path": "bot_core/example.py"}]

    assert semgrep_gate.validate_report(_write_report(tmp_path, data), blocking=False)


@pytest.mark.parametrize(
    "reason",
    [
        "exceeded_size_limit",
        "analysis_failed_parser_or_internal_error",
        "too_big",
        "minified",
        "binary",
        "too_many_matches",
        "Nonexistent_file",
        "insufficient_permissions",
        "unknown_new_reason",
    ],
)
def test_unanalysed_targets_are_not_counted_as_complete(tmp_path: Path, reason: str) -> None:
    data = _complete_report()
    data["paths"] = {
        "scanned": ["bot_core/example.py"],
        "skipped": [{"path": "bot_core/other.py", "reason": reason}],
    }

    assert semgrep_gate.validate_report(_write_report(tmp_path, data), blocking=False)


def test_existing_explicit_scope_filters_are_visible_and_retained(tmp_path: Path) -> None:
    data = _complete_report()
    data["paths"] = {
        "scanned": ["bot_core/example.py"],
        "skipped": [{"path": "tests/example.py", "reason": "cli_exclude_flags_match"}],
    }

    assert semgrep_gate.validate_report(_write_report(tmp_path, data), blocking=False) == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", "unknown"),
        ("errors", None),
        ("errors", [{}]),
        ("results", None),
        ("skipped_rules", [{}]),
        ("paths", {}),
        ("paths", {"scanned": [], "skipped": []}),
        ("paths", {"scanned": [None], "skipped": []}),
        ("paths", {"scanned": ["example.py"]}),
        ("paths", {"scanned": ["example.py"], "skipped": None}),
        ("paths", {"scanned": ["example.py"], "skipped": [{}]}),
    ],
)
def test_incomplete_or_malformed_report_refuses(tmp_path: Path, field: str, value: object) -> None:
    data = _complete_report()
    data[field] = value

    assert semgrep_gate.validate_report(_write_report(tmp_path, data), blocking=False)


def test_missing_or_invalid_json_refuses(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert semgrep_gate.validate_report(missing, blocking=False)
    missing.write_text("{", encoding="utf-8")
    assert semgrep_gate.validate_report(missing, blocking=False)


def test_duplicate_json_keys_cannot_hide_scan_errors(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    raw = json.dumps(_complete_report())
    path.write_text(raw.replace('"errors": []', '"errors": [{"type":"Timeout"}], "errors": []'))

    assert semgrep_gate.validate_report(path, blocking=False)


def test_oversize_report_refuses(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(semgrep_gate, "MAX_REPORT_BYTES", 8)

    assert semgrep_gate.validate_report(_write_report(tmp_path, _complete_report()), blocking=False)


def test_findings_policy_is_independent_of_scan_completeness(tmp_path: Path) -> None:
    data = _complete_report()
    data["results"] = [{"check_id": "synthetic-advisory"}]
    path = _write_report(tmp_path, data)

    assert semgrep_gate.validate_report(path, blocking=False) == []
    assert semgrep_gate.validate_report(path, blocking=True)
    data["errors"] = [{"type": "Timeout", "path": "other.py"}]
    assert semgrep_gate.validate_report(_write_report(tmp_path, data), blocking=False)


def test_public_summary_does_not_publish_finding_details(tmp_path: Path) -> None:
    data = _complete_report()
    data["results"] = [{"check_id": "synthetic", "extra": {"lines": "synthetic-private-detail"}}]
    data["errors"] = [{"type": "SyntheticPrivateType", "path": "synthetic-private-path"}]
    data["paths"] = {
        "scanned": ["synthetic-private-scanned"],
        "skipped": [{"path": "synthetic-private-skipped", "reason": "SyntheticPrivateReason"}],
    }
    path = _write_report(tmp_path, data)

    issues = semgrep_gate.validate_report(path, blocking=False)
    semgrep_gate.write_summary(path, issues, 0)

    raw = path.with_suffix(".summary.json").read_text()
    assert "private" not in raw.lower()
    summary = json.loads(raw)
    assert summary["results_count"] == 1
    assert summary["issue_counts"] == {"scanner_error": 1, "incomplete_target": 1}
    assert summary["target_selection_counts"] == {"incomplete_or_unknown": 1}


def test_invalid_report_and_unknown_issue_text_are_not_published(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_bytes(b'"\xffsynthetic-private-detail"')

    issues = semgrep_gate.validate_report(path, blocking=False)
    assert issues == ["report_invalid"]
    semgrep_gate.write_summary(path, [*issues, "synthetic-private-issue"], None)

    raw = path.with_suffix(".summary.json").read_text()
    assert "private" not in raw
    assert json.loads(raw)["issue_counts"] == {"report_invalid": 2}


@pytest.mark.parametrize("returncode", [1, 2, 7, -9])
def test_scanner_exit_cannot_be_replaced_by_clean_report(
    monkeypatch, tmp_path: Path, returncode: int
) -> None:
    path = tmp_path / "report.json"

    def fake_run(command, **kwargs):
        assert "--timeout=60" in command
        assert "--timeout-threshold=0" in command
        assert "--jobs=2" in command
        _write_report(tmp_path, _complete_report())
        return subprocess.CompletedProcess(command, returncode)

    monkeypatch.setattr(semgrep_gate.subprocess, "run", fake_run)
    assert semgrep_gate.main(["--report", str(path), "--config", "rules.yml", "bot_core"]) == 2


def test_stale_report_is_removed_before_scanner_start(monkeypatch, tmp_path: Path) -> None:
    path = _write_report(tmp_path, _complete_report())
    semgrep_gate.write_summary(path, [], 0)

    def fake_run(command, **kwargs):
        assert not path.exists()
        assert not path.with_suffix(".summary.json").exists()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(semgrep_gate.subprocess, "run", fake_run)
    assert semgrep_gate.main(["--report", str(path), "--config", "rules.yml", "bot_core"]) == 2


def test_missing_scanner_refuses_without_publishing_exception(monkeypatch, tmp_path: Path, capsys):
    def fake_run(command, **kwargs):
        raise FileNotFoundError("synthetic-private-exception")

    monkeypatch.setattr(semgrep_gate.subprocess, "run", fake_run)
    assert (
        semgrep_gate.main(
            ["--report", str(tmp_path / "report.json"), "--config", "rules.yml", "bot_core"]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "private" not in captured.out + captured.err
    summary = json.loads((tmp_path / "report.summary.json").read_text())
    assert summary["issue_counts"] == {"scanner_unavailable": 1}


def test_summary_write_failure_refuses_without_publishing_exception(monkeypatch, tmp_path, capsys):
    def fake_run(command, **kwargs):
        _write_report(tmp_path, _complete_report())
        return subprocess.CompletedProcess(command, 0)

    def fail_summary(*args):
        raise OSError("synthetic-private-exception")

    monkeypatch.setattr(semgrep_gate.subprocess, "run", fake_run)
    monkeypatch.setattr(semgrep_gate, "write_summary", fail_summary)

    assert (
        semgrep_gate.main(
            ["--report", str(tmp_path / "report.json"), "--config", "rules.yml", "bot_core"]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "nie zapisano podsumowania" in captured.err
    assert "private" not in captured.out + captured.err


@pytest.mark.parametrize("errors,returncode", [([], 0), ([{"type": "Timeout"}], 2)])
def test_cli_invokes_scanner_and_checks_actual_output(
    monkeypatch, tmp_path: Path, capsys, errors: list, returncode: int
) -> None:
    scanner = tmp_path / "semgrep.py"
    data = _complete_report()
    data["errors"] = errors
    data["paths"] = {
        "scanned": ["synthetic-private-scanned"],
        "skipped": [],
    }
    scanner.write_text(
        "import pathlib, sys\n"
        "print('synthetic-private-stdout')\n"
        "print('synthetic-private-stderr', file=sys.stderr)\n"
        "assert '--strict' in sys.argv and '--verbose' in sys.argv\n"
        "assert '--max-target-bytes=0' in sys.argv\n"
        f"pathlib.Path(sys.argv[sys.argv.index('--output')+1]).write_text({json.dumps(data)!r})\n",
        encoding="utf-8",
    )
    report = tmp_path / "actual.json"
    real_run = subprocess.run

    def run_synthetic_scanner(command, **kwargs):
        assert command[0] == "semgrep"
        return real_run([sys.executable, str(scanner), *command[1:]], **kwargs)

    monkeypatch.setattr(semgrep_gate.subprocess, "run", run_synthetic_scanner)

    assert (
        semgrep_gate.main(["--report", str(report), "--config", "rules.yml", "bot_core"])
        == returncode
    )
    assert json.loads(report.read_text())["errors"] == errors
    captured = capsys.readouterr()
    assert "synthetic-private" not in captured.out + captured.err
    assert "synthetic-private-stdout" in report.with_suffix(".stdout.log").read_text()
    assert "synthetic-private-stderr" in report.with_suffix(".stderr.log").read_text()
