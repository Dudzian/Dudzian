from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

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


def _cleanup(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _inject_packaging_stub(env: dict[str, str], workspace: Path) -> Path:
    stub_root = (workspace / "stubs").resolve()
    package_dir = stub_root / "packaging"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text(
        "from .version import Version, InvalidVersion\n",
        encoding="utf-8",
    )
    (package_dir / "version.py").write_text(
        "class InvalidVersion(Exception):\n    ...\n\n"
        "class Version:\n    def __init__(self, version: str) -> None:\n        self._value = version\n\n"
        "    def __str__(self) -> str:\n        return self._value\n",
        encoding="utf-8",
    )
    pythonpath = env.get("PYTHONPATH") or ""
    env["PYTHONPATH"] = (
        f"{stub_root}{os.pathsep}{pythonpath}" if pythonpath else str(stub_root)
    )
    return stub_root


def test_runbook_evaluate_command() -> None:
    risk_report = Path("var/audit/stage6/risk_simulation_report.json")
    report_dir = risk_report.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_sample_report(risk_report)

    json_path = Path("var/audit/stage6/stress_lab_report.json")
    csv_path = Path("var/audit/stage6/stress_lab_insights.csv")
    overrides_path = Path("var/audit/stage6/stress_lab_overrides.csv")
    signature_path = Path("var/audit/stage6/stress_lab_report.json.sig")
    key_path = Path("secrets/hypercare/stage6_hmac.key")
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(b"top-secret")

    _cleanup([json_path, csv_path, overrides_path, signature_path])

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [".", env.get("PYTHONPATH")]))
    stub_root = _inject_packaging_stub(env, Path("var/audit/stage6"))

    command = [sys.executable, "scripts/run_stress_lab.py", "evaluate"]
    command.extend(
        [
            "--risk-report",
            str(risk_report),
            "--config",
            "config/core.yaml",
            "--governor",
            "stage6_core",
            "--output-json",
            str(json_path),
            "--output-csv",
            str(csv_path),
            "--overrides-csv",
            str(overrides_path),
            "--signing-key",
            str(key_path),
            "--signing-key-id",
            "stage6",
            "--signature-path",
            str(signature_path),
        ]
    )

    try:
        subprocess.run(command, check=True, env=env)
    finally:
        key_path.unlink(missing_ok=True)  # ensure key not re-used between tests
        try:
            key_path.parent.rmdir()
        except OSError:
            pass
        _cleanup([stub_root])

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

    _cleanup([risk_report, json_path, csv_path, overrides_path, signature_path])


def test_runbook_run_command() -> None:
    candidate_outputs = [
        Path("var/audit/stage6/stress_lab/stress_lab_report.json"),
        Path("config/var/audit/stage6/stress_lab/stress_lab_report.json"),
    ]
    for path in candidate_outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    candidate_signatures = [path.with_suffix(path.suffix + ".sig") for path in candidate_outputs]
    _cleanup([*candidate_outputs, *candidate_signatures])

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [".", env.get("PYTHONPATH")]))
    stub_root = _inject_packaging_stub(env, Path("var/audit/stage6"))
    env["STRESS_LAB_SIGNING_KEY"] = "unit-test-secret"

    market_intel_root = Path("config/var/audit/stage6/market_intel")
    market_intel_root.mkdir(parents=True, exist_ok=True)
    for symbol in ("btcusdt", "ethusdt"):
        market_file = market_intel_root / f"{symbol}.json"
        if not market_file.exists():
            market_file.write_text(json.dumps({"metrics": []}), encoding="utf-8")

    command = [sys.executable, "scripts/run_stress_lab.py", "run"]
    command.extend(["--config", "config/core.yaml"])

    try:
        subprocess.run(command, check=True, env=env)

        output_path = next((path for path in candidate_outputs if path.exists()), None)
        assert output_path is not None
        signature_path = output_path.with_suffix(output_path.suffix + ".sig")
        assert signature_path.exists()

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert {"generated_at", "scenarios", "thresholds"}.issubset(payload)

        signature = json.loads(signature_path.read_text(encoding="utf-8"))
        assert signature.get("algorithm") == "HMAC-SHA256"
    finally:
        _cleanup([*candidate_outputs, *candidate_signatures, stub_root])
        for symbol in ("btcusdt", "ethusdt"):
            try:
                (market_intel_root / f"{symbol}.json").unlink()
            except FileNotFoundError:
                pass
