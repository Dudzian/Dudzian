from pathlib import Path

import pytest
import json

from scripts.audit.marketing_parity_check import ParityError, main, run_parity_check


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_run_parity_check_succeeds(tmp_path: Path) -> None:
    stress_dir = tmp_path / "stress_lab"
    local_stress = stress_dir / "ci_report.json"
    local_index = tmp_path / "signal_quality" / "index.csv"
    mirror_root = tmp_path / "mirror"

    write_file(local_stress, "{}\n")
    write_file(local_index, "exchange,score\nbinance,1\n")

    # Mirror has identical copies
    write_file(mirror_root / "stress_lab" / "ci_report.json", local_stress.read_text())
    write_file(mirror_root / "signal_quality" / "index.csv", local_index.read_text())

    run_parity_check(
        stress_lab_dir=stress_dir,
        signal_quality_index=local_index,
        mirror_dir=mirror_root,
        report_output=tmp_path / "report.md",
    )


def test_run_parity_check_writes_json(tmp_path: Path) -> None:
    stress_dir = tmp_path / "stress_lab"
    local_stress = stress_dir / "ci_report.json"
    local_index = tmp_path / "signal_quality" / "index.csv"
    mirror_root = tmp_path / "mirror"
    json_path = tmp_path / "report.json"

    write_file(local_stress, "{}\n")
    write_file(local_index, "exchange,score\nbinance,1\n")

    # Mirror has identical copies
    write_file(mirror_root / "stress_lab" / "ci_report.json", local_stress.read_text())
    write_file(mirror_root / "signal_quality" / "index.csv", local_index.read_text())

    run_parity_check(
        stress_lab_dir=stress_dir,
        signal_quality_index=local_index,
        mirror_dir=mirror_root,
        report_output=tmp_path / "report.md",
        json_output=json_path,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["missing_in_mirror"] == []
    assert payload["mismatched"] == []
    assert any(entry["file"] == "signal_quality/index.csv" for entry in payload["local_hashes"])


def test_run_parity_check_detects_mismatch(tmp_path: Path) -> None:
    stress_dir = tmp_path / "stress_lab"
    local_stress = stress_dir / "ci_report.json"
    local_index = tmp_path / "signal_quality" / "index.csv"
    mirror_root = tmp_path / "mirror"

    write_file(local_stress, "{}\n")
    write_file(local_index, "exchange,score\nbinance,1\n")

    # Mirror has different content
    write_file(mirror_root / "stress_lab" / "ci_report.json", "different")
    write_file(mirror_root / "signal_quality" / "index.csv", local_index.read_text())

    with pytest.raises(ParityError):
        run_parity_check(
            stress_lab_dir=stress_dir,
            signal_quality_index=local_index,
            mirror_dir=mirror_root,
            report_output=tmp_path / "report.md",
        )


def test_run_parity_check_requires_mirror_dir(tmp_path: Path) -> None:
    stress_dir = tmp_path / "stress_lab"
    local_index = tmp_path / "signal_quality" / "index.csv"

    write_file(stress_dir / "ci_report.json", "{}\n")
    write_file(local_index, "exchange,score\nbinance,1\n")

    with pytest.raises(ParityError, match="Mirror directory not found"):
        run_parity_check(
            stress_lab_dir=stress_dir,
            signal_quality_index=local_index,
            mirror_dir=tmp_path / "missing_mirror",
            report_output=tmp_path / "report.md",
        )


def test_run_parity_check_requires_files(tmp_path: Path) -> None:
    mirror_root = tmp_path / "mirror"
    mirror_root.mkdir()

    with pytest.raises(ParityError, match="Signal quality index missing"):
        run_parity_check(
            stress_lab_dir=None,
            signal_quality_index=tmp_path / "signal_quality" / "index.csv",
            mirror_dir=mirror_root,
            report_output=tmp_path / "report.md",
        )


def test_main_exit_code(tmp_path: Path) -> None:
    stress_dir = tmp_path / "stress_lab"
    local_stress = stress_dir / "ci_report.json"
    local_index = tmp_path / "signal_quality" / "index.csv"
    mirror_root = tmp_path / "mirror"

    write_file(local_stress, "{}\n")
    write_file(local_index, "exchange,score\nbinance,1\n")
    write_file(mirror_root / "stress_lab" / "ci_report.json", local_stress.read_text())
    write_file(mirror_root / "signal_quality" / "index.csv", local_index.read_text())

    exit_code = main(
        [
            "--stress-lab-dir",
            str(stress_dir),
            "--signal-quality-index",
            str(local_index),
            "--mirror-dir",
            str(mirror_root),
            "--audit-output",
            str(tmp_path / "report.md"),
        ]
    )

    assert exit_code == 0


def test_main_reports_failure_and_exit_code(tmp_path: Path) -> None:
    stress_dir = tmp_path / "stress_lab"
    local_stress = stress_dir / "ci_report.json"
    local_index = tmp_path / "signal_quality" / "index.csv"
    mirror_root = tmp_path / "mirror"

    write_file(local_stress, "{}\n")
    write_file(local_index, "exchange,score\nbinance,1\n")
    write_file(mirror_root / "stress_lab" / "ci_report.json", "different")
    write_file(mirror_root / "signal_quality" / "index.csv", local_index.read_text())

    exit_code = main(
        [
            "--stress-lab-dir",
            str(stress_dir),
            "--signal-quality-index",
            str(local_index),
            "--mirror-dir",
            str(mirror_root),
            "--audit-output",
            str(tmp_path / "report.md"),
        ]
    )

    assert exit_code == 1
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Rozbieżności hashy" in report


def test_main_missing_mirror_creates_stub_report(tmp_path: Path) -> None:
    local_index = tmp_path / "signal_quality" / "index.csv"

    write_file(local_index, "exchange,score\nbinance,1\n")

    exit_code = main(
        [
            "--signal-quality-index",
            str(local_index),
            "--mirror-dir",
            str(tmp_path / "missing_mirror"),
            "--audit-output",
            str(tmp_path / "report.md"),
            "--json-output",
            str(tmp_path / "report.json"),
        ]
    )

    assert exit_code == 1
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Błąd walidacji" in report
    json_payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert json_payload["status"] == "failed"
    assert "Mirror directory not found" in json_payload["error"]


def test_main_failure_overwrites_existing_report(tmp_path: Path) -> None:
    stress_dir = tmp_path / "stress_lab"
    local_stress = stress_dir / "ci_report.json"
    local_index = tmp_path / "signal_quality" / "index.csv"
    mirror_root = tmp_path / "mirror"
    audit_path = tmp_path / "report.md"

    write_file(local_stress, "{}\n")
    write_file(local_index, "exchange,score\nbinance,1\n")
    write_file(mirror_root / "stress_lab" / "ci_report.json", "different")
    write_file(mirror_root / "signal_quality" / "index.csv", local_index.read_text())

    # Pre-existing success report that must be replaced on failure
    audit_path.write_text("stale-success-report", encoding="utf-8")

    exit_code = main(
        [
            "--stress-lab-dir",
            str(stress_dir),
            "--signal-quality-index",
            str(local_index),
            "--mirror-dir",
            str(mirror_root),
            "--audit-output",
            str(audit_path),
        ]
    )

    assert exit_code == 1
    report = audit_path.read_text(encoding="utf-8")
    assert "Błąd walidacji" in report
    assert "stale-success-report" not in report
