from __future__ import annotations

import json
from pathlib import Path

import pytest


from scripts.run_stress_lab import main as run_stress_lab, _prepare_argv


ROOT = Path(__file__).resolve().parents[1]


def _patch_stress_lab_runner(
    monkeypatch: pytest.MonkeyPatch, *, has_failures: bool, run_called: list[bool] | None = None
):
    from scripts import run_stress_lab as module

    class _StubReport:
        def __init__(self) -> None:
            self.insights: list[str] = []
            self.overrides: list[str] = []

        def write_json(self, path: Path) -> None:
            path.write_text(
                json.dumps({"schema": "stage6.risk.stress_lab.report"}),
                encoding="utf-8",
            )

        def write_signature(self, path: Path, *, key: bytes, key_id: str | None) -> None:
            path.write_text(json.dumps({"signature": "ok"}), encoding="utf-8")

        def has_failures(self) -> bool:
            return has_failures

    class _StubLab:
        def __init__(self, config: object) -> None:
            self._config = config

        def run(self) -> _StubReport:
            if run_called is not None:
                run_called.append(True)
            return _StubReport()

    monkeypatch.setattr(module, "StressLab", _StubLab)
    return module


@pytest.mark.parametrize("fail_on_breach", [False, True])
def test_run_stress_lab_cli(tmp_path: Path, fail_on_breach: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / ("report_fail.json" if fail_on_breach else "report.json")
    monkeypatch.setenv("STRESS_LAB_SIGNING_KEY", "unit-test-secret")
    _patch_stress_lab_runner(monkeypatch, has_failures=False)

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
    assert output_path.with_suffix(output_path.suffix + ".sig").exists()


def test_run_stress_lab_cli_fails_on_breach(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "stress_lab_report.json"
    monkeypatch.setenv("STRESS_LAB_SIGNING_KEY", "unit-test-secret")
    module = _patch_stress_lab_runner(monkeypatch, has_failures=True)

    exit_code = module.main([
        "--config",
        "config/core.yaml",
        "--output",
        str(output_path),
        "--fail-on-breach",
    ])

    assert exit_code == 3
    assert output_path.exists()


def test_run_stress_lab_cli_missing_signing_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module_calls: list[bool] = []
    module = _patch_stress_lab_runner(monkeypatch, has_failures=False, run_called=module_calls)
    monkeypatch.delenv("STRESS_LAB_SIGNING_KEY", raising=False)
    monkeypatch.delenv("MISSING_KEY", raising=False)

    exit_code = module.main([
        "--config",
        "config/core.yaml",
        "--output",
        str(tmp_path / "report.json"),
        "--signing-key-env",
        "MISSING_KEY",
    ])

    assert exit_code == 2
    assert not module_calls
    assert not (tmp_path / "report.json").exists()


def test_run_stress_lab_cli_missing_signing_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module_calls: list[bool] = []
    module = _patch_stress_lab_runner(monkeypatch, has_failures=False, run_called=module_calls)
    monkeypatch.delenv("STRESS_LAB_SIGNING_KEY", raising=False)

    exit_code = module.main([
        "--config",
        "config/core.yaml",
        "--output",
        str(tmp_path / "report.json"),
        "--signing-key-path",
        str(tmp_path / "missing.key"),
    ])

    assert exit_code == 2
    assert not module_calls
    assert not (tmp_path / "report.json").exists()


def test_prepare_argv_injects_subcommand() -> None:
    assert _prepare_argv(["--risk-report", "report.json"]) == [
        "evaluate",
        "--risk-report",
        "report.json",
    ]
    assert _prepare_argv(["--output-json", "out.json", "--risk-report=report.json"]) == [
        "evaluate",
        "--output-json",
        "out.json",
        "--risk-report=report.json",
    ]
    assert _prepare_argv(["--config", "config/core.yaml"]) == [
        "run",
        "--config",
        "config/core.yaml",
    ]
