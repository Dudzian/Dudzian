from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.failover_drill import main as run_failover_drill


@pytest.mark.parametrize("fail_on_breach", [False, True])
def test_failover_drill_cli(tmp_path: Path, fail_on_breach: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / ("resilience_fail.json" if fail_on_breach else "resilience_ok.json")
    monkeypatch.setenv("STAGE6_RESILIENCE_SIGNING_KEY", "unit-test-secret")

    argv = [
        "--config",
        "config/core.yaml",
        "--output",
        str(output_path),
    ]
    if fail_on_breach:
        argv.append("--fail-on-breach")

    exit_code = run_failover_drill(argv)
    assert exit_code == 0
    assert output_path.exists()
