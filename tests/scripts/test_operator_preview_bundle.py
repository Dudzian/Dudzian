from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/operator_preview_bundle.py"
SAFE_CONFIG = REPO_ROOT / "config/e2e/demo_paper.yml"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_operator_preview_bundle_safe_demo_json() -> None:
    result = _run(
        "--mode",
        "demo",
        "--config",
        str(SAFE_CONFIG),
        "--duration-seconds",
        "5",
        "--max-signals",
        "1",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["mode"] == "demo"
    assert len(payload["steps"]) == 4
    assert payload["summary"]["steps_total"] == 4
    assert payload["summary"]["steps_passed"] == 4
    assert payload["summary"]["real_orders_submitted"] is False
    assert payload["summary"]["exchange_io"] == "disabled"
    assert payload["summary"]["api_keys_required"] is False
    assert payload["summary"]["runtime_loop_started"] is False
    assert payload["summary"]["controller_backed_preview"] is True
    assert payload["issues"] == []
    assert payload["safety_contract_version"] == "operator_preview_bundle.v1"


def test_operator_preview_bundle_live_mode_blocked() -> None:
    result = _run("--mode", "live", "--config", str(SAFE_CONFIG), "--json")
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked"
    assert payload["reason"] == "operator_preview_bundle_forbids_live_mode"
    assert payload["steps"] == []


def test_operator_preview_bundle_child_payload_sanity() -> None:
    result = _run("--mode", "demo", "--config", str(SAFE_CONFIG), "--max-signals", "1", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    names = [step["name"] for step in payload["steps"]]
    assert names == [
        "demo_paper_precheck",
        "preview_plan",
        "mock_runtime_preview",
        "controller_mock_preview",
    ]
    controller_payload = payload["steps"][-1]["payload"]
    assert controller_payload["real_orders_submitted"] is False
    assert controller_payload["runtime_loop_started"] is False
    assert controller_payload["api_keys_required"] is False


def test_operator_preview_bundle_no_api_keys_required(monkeypatch) -> None:
    for key in ("API_KEY", "API_SECRET", "BINANCE_API_KEY", "BINANCE_API_SECRET"):
        monkeypatch.delenv(key, raising=False)
    result = _run("--mode", "demo", "--config", str(SAFE_CONFIG), "--json")
    assert result.returncode == 0, result.stderr


def test_operator_preview_bundle_output_cp1252_safe() -> None:
    result = _run("--mode", "demo", "--config", str(SAFE_CONFIG), "--json")
    assert result.returncode == 0, result.stderr
    result.stdout.encode("cp1252")
