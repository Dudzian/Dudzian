from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from bot_core.risk.simulation import ProfileSimulationResult, RiskSimulationReport, StressTestResult


def _write_sample_report(path: Path) -> None:
    report = RiskSimulationReport(
        generated_at="2024-06-01T00:00:00Z",
        base_equity=100_000.0,
        profiles=(
            ProfileSimulationResult(
                profile="balanced",
                base_equity=100_000.0,
                final_equity=90_000.0,
                total_return_pct=-0.1,
                max_drawdown_pct=0.15,
                worst_daily_loss_pct=0.07,
                realized_volatility=0.2,
                breaches=(),
                stress_tests=(
                    StressTestResult(
                        name="flash_crash",
                        status="failed",
                        metrics={"assets": ["BTCUSDT"], "severity": "critical"},
                        notes="Głębokie naruszenie limitów",
                    ),
                ),
                sample_size=120,
            ),
        ),
    )
    report.write_json(path)


def test_run_stress_lab_cli(tmp_path: Path) -> None:
    risk_report = tmp_path / "risk_simulation_report.json"
    _write_sample_report(risk_report)

    json_path = tmp_path / "stress_lab.json"
    csv_path = tmp_path / "stress_lab.csv"
    overrides_path = tmp_path / "stress_lab_overrides.csv"
    signature_path = tmp_path / "stress_lab.sig"
    key_path = tmp_path / "hmac.key"
    key_path.write_bytes(b"top-secret")

    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_stress_lab.py",
            "--risk-report",
            str(risk_report),
            "--output-json",
            str(json_path),
            "--output-csv",
            str(csv_path),
            "--overrides-csv",
            str(overrides_path),
            "--signing-key",
            str(key_path),
            "--signature-path",
            str(signature_path),
        ],
        check=True,
        env=env,
    )
    assert result.returncode == 0

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "stage6.risk.stress_lab.report"
    assert payload["overrides"]

    signature = json.loads(signature_path.read_text(encoding="utf-8"))
    assert signature["schema"] == "stage6.risk.stress_lab.report.signature"
    assert signature["signature"]["algorithm"] == "HMAC-SHA256"

    csv_lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert csv_lines[0].startswith("profile,scenario,severity")
    overrides_lines = overrides_path.read_text(encoding="utf-8").strip().splitlines()
    assert overrides_lines[0].startswith("symbol,risk_budget")
