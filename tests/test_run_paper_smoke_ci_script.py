"""Testy dla skryptu run_paper_smoke_ci.py."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import run_paper_smoke_ci


def _build_completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["python"], returncode=returncode, stdout=stdout, stderr=stderr)


def test_build_command_creates_directories(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text("core: {}", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    run_daily_trend = tmp_path / "scripts" / "run_daily_trend.py"
    run_daily_trend.write_text("print('stub')", encoding="utf-8")

    command, paths = run_paper_smoke_ci._build_command(
        config_path=config_path,
        environment="binance_paper",
        output_dir=tmp_path / "output",
        operator="Tester",
        auto_publish_required=True,
        extra_run_daily_trend_args=[],
    )

    assert "--paper-smoke-auto-publish-required" in command
    assert paths["summary"].parent.exists()
    assert paths["json_log"].parent.exists()
    assert paths["audit_log"].parent.exists()


def test_build_command_accepts_extra_args(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text("core: {}", encoding="utf-8")
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "run_daily_trend.py").write_text("print('stub')", encoding="utf-8")

    command, _ = run_paper_smoke_ci._build_command(
        config_path=config_path,
        environment="binance_paper",
        output_dir=tmp_path / "output",
        operator="Tester",
        auto_publish_required=False,
        extra_run_daily_trend_args=["--date-window 2024-01-01:2024-02-01", "--run-once"],
    )

    assert command.count("--date-window") == 1
    assert "2024-01-01:2024-02-01" in command
    assert "--run-once" in command


def test_main_runs_smoke_and_prints_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    run_daily_trend = scripts_dir / "run_daily_trend.py"
    run_daily_trend.write_text("print('stub run')", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text("core: {}", encoding="utf-8")

    summary_payload = {"status": "ok", "publish": {"status": "ok"}}

    def fake_run(cmd, text, check):  # noqa: ANN001
        summary_arg = cmd.index("--paper-smoke-summary-json")
        summary_path = Path(cmd[summary_arg + 1])
        summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")
        return _build_completed(returncode=0)

    monkeypatch.setattr(run_paper_smoke_ci.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    result = run_paper_smoke_ci.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_paper",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert result == 0
    summary_file = tmp_path / "output" / "paper_smoke_summary.json"
    assert summary_file.exists()


def test_main_propagates_non_zero_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    run_daily_trend = scripts_dir / "run_daily_trend.py"
    run_daily_trend.write_text("print('stub run')", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("core: {}", encoding="utf-8")

    def fake_run(cmd, text, check):  # noqa: ANN001
        return _build_completed(returncode=6)

    monkeypatch.setattr(run_paper_smoke_ci.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    exit_code = run_paper_smoke_ci.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_paper",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert exit_code == 6


def test_main_writes_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    run_daily_trend = scripts_dir / "run_daily_trend.py"
    run_daily_trend.write_text("print('stub run')", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("core: {}", encoding="utf-8")

    summary_payload = {
        "status": "ok",
        "publish": {"status": "ok"},
    }

    def fake_run(cmd, text, check):  # noqa: ANN001
        summary_arg = cmd.index("--paper-smoke-summary-json")
        summary_path = Path(cmd[summary_arg + 1])
        summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")
        return _build_completed(returncode=0)

    monkeypatch.setattr(run_paper_smoke_ci.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    env_file = tmp_path / "env" / "paper_smoke.env"
    exit_code = run_paper_smoke_ci.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_paper",
            "--output-dir",
            str(tmp_path / "output"),
            "--env-file",
            str(env_file),
            "--operator",
            "CI Operator",
        ]
    )

    assert exit_code == 0
    content = env_file.read_text(encoding="utf-8")
    lines = dict(line.split("=", 1) for line in content.strip().splitlines())
    assert lines["PAPER_SMOKE_OPERATOR"] == "CI Operator"
    assert lines["PAPER_SMOKE_STATUS"] == "ok"
    assert lines["PAPER_SMOKE_PUBLISH_STATUS"] == "ok"


def test_main_dry_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    run_daily_trend = scripts_dir / "run_daily_trend.py"
    run_daily_trend.write_text("print('stub run')", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("core: {}", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    exit_code = run_paper_smoke_ci.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_paper",
            "--output-dir",
            str(tmp_path / "output"),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "dry_run"
    assert exit_code == 0


def test_main_allows_optional_publish(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    run_daily_trend = scripts_dir / "run_daily_trend.py"
    run_daily_trend.write_text("print('stub run')", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("core: {}", encoding="utf-8")

    summary_payload = {"status": "ok"}

    def fake_run(cmd, text, check):  # noqa: ANN001
        assert "--paper-smoke-auto-publish" in cmd and "--paper-smoke-auto-publish-required" not in cmd
        summary_arg = cmd.index("--paper-smoke-summary-json")
        Path(cmd[summary_arg + 1]).write_text(json.dumps(summary_payload), encoding="utf-8")
        return _build_completed(returncode=0)

    monkeypatch.setattr(run_paper_smoke_ci.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    exit_code = run_paper_smoke_ci.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_paper",
            "--output-dir",
            str(tmp_path / "output"),
            "--allow-auto-publish-failure",
        ]
    )

    assert exit_code == 0
