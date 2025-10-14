from __future__ import annotations

from pathlib import Path

import pytest

from scripts import enforce_coverage


@pytest.fixture()
def coverage_file(tmp_path: Path) -> Path:
    xml = tmp_path / "coverage.xml"
    xml.write_text(
        """
        <coverage line-rate="0.92">
          <packages>
            <package name="bot_core.strategies" line-rate="0.95" />
            <package name="bot_core.runtime" line-rate="0.89" />
          </packages>
        </coverage>
        """.strip(),
        encoding="utf-8",
    )
    return xml


def test_script_passes_with_default_threshold(coverage_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = enforce_coverage.main(
        ["--coverage-file", str(coverage_file), "--minimum", "80", "--package", "bot_core.strategies=85"]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Pokrycie całkowite" in out
    assert "Pakiet bot_core.strategies" in out


def test_script_fails_when_threshold_not_met(coverage_file: Path) -> None:
    exit_code = enforce_coverage.main(
        ["--coverage-file", str(coverage_file), "--minimum", "95"]
    )
    assert exit_code == 1


def test_script_errors_on_missing_package(coverage_file: Path) -> None:
    exit_code = enforce_coverage.main(
        ["--coverage-file", str(coverage_file), "--package", "missing.module"]
    )
    assert exit_code == 1


def test_script_errors_when_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "absent.xml"
    exit_code = enforce_coverage.main(["--coverage-file", str(missing)])
    assert exit_code == 2
