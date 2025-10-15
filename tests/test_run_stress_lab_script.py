from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stress_lab import main as run_stress_lab


@pytest.mark.parametrize("fail_on_breach", [False, True])
def test_run_stress_lab_cli(tmp_path: Path, fail_on_breach: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / ("report_fail.json" if fail_on_breach else "report.json")
    monkeypatch.setenv("STRESS_LAB_SIGNING_KEY", "unit-test-secret")

    argv = [
        "--config",
        "config/core.yaml",
        "--output",
        str(output_path),
    ]
    if fail_on_breach:
        argv.append("--fail-on-breach")

    exit_code = run_stress_lab(argv)
    assert exit_code == 0
    assert output_path.exists()
