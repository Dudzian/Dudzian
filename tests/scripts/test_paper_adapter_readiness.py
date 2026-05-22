from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/paper_adapter_readiness.py"
SAFE_CONFIG = REPO_ROOT / "config/e2e/demo_paper.yml"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_paper_adapter_readiness_safe_demo_json() -> None:
    result = _run("--config", str(SAFE_CONFIG), "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["exchange_io"]["enabled"] is False
    assert payload["order_submission"]["enabled"] is False
    assert payload["api_keys_required"] is False


def test_paper_adapter_readiness_fails_on_execution_live_enabled(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe.yml"
    data = yaml.safe_load(SAFE_CONFIG.read_text(encoding="utf-8"))
    data.setdefault("execution", {}).setdefault("live", {})["enabled"] = True
    unsafe.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    result = _run("--config", str(unsafe), "--json")
    assert result.returncode != 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert any("execution.live.enabled" in issue for issue in payload["issues"])


def test_paper_adapter_readiness_fails_on_live_defaults(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe_live.yml"
    data = yaml.safe_load(SAFE_CONFIG.read_text(encoding="utf-8"))
    data.setdefault("execution", {})["default_mode"] = "live"
    data.setdefault("trading", {})["enable_live_mode"] = True
    unsafe.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    result = _run("--config", str(unsafe), "--json")
    assert result.returncode != 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"


def test_paper_adapter_readiness_json_contract_contains_required_fields() -> None:
    result = _run("--config", str(SAFE_CONFIG), "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    for key in ("adapter_readiness", "exchange_io", "order_submission", "issues"):
        assert key in payload
