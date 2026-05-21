from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_local_bot.py"
SAFE_CONFIG = REPO_ROOT / "config" / "e2e" / "demo_paper.yml"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _mutated_config(tmp_path: Path, dotted_path: str, value: object) -> Path:
    payload = yaml.safe_load(SAFE_CONFIG.read_text(encoding="utf-8"))
    current = payload
    segments = dotted_path.split(".")
    for segment in segments[:-1]:
        current = current[segment]
    current[segments[-1]] = value
    path = tmp_path / f"mutated_{dotted_path.replace('.', '_')}.yml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_preview_plan_demo_is_safe_and_no_runtime() -> None:
    result = _run("--preview-plan", "--mode", "demo", "--config", str(SAFE_CONFIG))
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["runtime_started"] is False
    assert payload["exchange_io"] == "disabled"
    assert payload["order_execution"] == "disabled"
    assert payload["api_keys_required"] is False


def test_preview_plan_rejects_live_mode() -> None:
    result = _run("--preview-plan", "--mode", "live", "--config", str(SAFE_CONFIG))
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked"
    assert payload["reason"] == "preview_plan_forbids_live_mode"


def test_preview_plan_rejects_execution_live_enabled(tmp_path: Path) -> None:
    cfg = _mutated_config(tmp_path, "execution.live.enabled", True)
    result = _run("--preview-plan", "--mode", "demo", "--config", str(cfg))
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert "unsafe_config:execution.live.enabled" in payload["issues"]


def test_preview_plan_rejects_trading_enable_live_mode(tmp_path: Path) -> None:
    cfg = _mutated_config(tmp_path, "trading.enable_live_mode", True)
    result = _run("--preview-plan", "--mode", "demo", "--config", str(cfg))
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert "unsafe_config:trading.enable_live_mode" in payload["issues"]


def test_preview_plan_rejects_trading_enable_paper_mode(tmp_path: Path) -> None:
    cfg = _mutated_config(tmp_path, "trading.enable_paper_mode", False)
    result = _run("--preview-plan", "--mode", "demo", "--config", str(cfg))
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert "unsafe_config:trading.enable_paper_mode" in payload["issues"]


def test_preview_plan_rejects_execution_default_mode_live(tmp_path: Path) -> None:
    cfg = _mutated_config(tmp_path, "execution.default_mode", "live")
    result = _run("--preview-plan", "--mode", "demo", "--config", str(cfg))
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert "unsafe_config:execution.default_mode" in payload["issues"]


def test_preview_plan_rejects_force_paper_when_offline_disabled(tmp_path: Path) -> None:
    cfg = _mutated_config(tmp_path, "execution.force_paper_when_offline", False)
    result = _run("--preview-plan", "--mode", "demo", "--config", str(cfg))
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert "unsafe_config:execution.force_paper_when_offline" in payload["issues"]
