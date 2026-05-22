from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/paper_runtime_dry_run.py"
SAFE_CONFIG = REPO_ROOT / "config/e2e/demo_paper.yml"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_paper_runtime_dry_run_happy_path_json() -> None:
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
    assert len(payload["steps"]) == 3
    assert [step["name"] for step in payload["steps"]] == [
        "preview_plan",
        "mock_runtime_preview",
        "controller_mock_preview",
    ]
    assert all(step["exit_code"] == 0 for step in payload["steps"])
    summary = payload["summary"]
    assert summary["steps_total"] == 3
    assert summary["steps_passed"] == 3
    assert summary["bounded_preview_loop"] is True
    assert summary["production_runtime_loop_started"] is False
    assert summary["runtime_loop_started"] is False
    assert summary["exchange_io"] == "disabled"
    assert summary["api_keys_required"] is False
    assert summary["secrets_read"] is False
    assert summary["keychain_read"] is False
    assert summary["env_values_read"] is False
    assert summary["real_orders_submitted"] is False
    assert summary["order_execution"] == "mocked_or_disabled"
    assert summary["controller_backed_preview"] is True
    assert payload["issues"] == []
    assert payload["safety_contract_version"] == "paper_runtime_dry_run.v1"


def test_paper_runtime_dry_run_live_blocked() -> None:
    result = _run("--mode", "live", "--config", str(SAFE_CONFIG), "--json")
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked"
    assert payload["reason"] == "paper_runtime_dry_run_forbids_live_mode"
    assert payload["steps"] == []
    assert "live_mode_not_allowed" in payload["issues"]


def test_paper_runtime_dry_run_invalid_duration_low_blocked() -> None:
    result = _run(
        "--mode", "demo", "--config", str(SAFE_CONFIG), "--duration-seconds", "0", "--json"
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["reason"] == "paper_runtime_dry_run_invalid_duration"
    assert payload["steps"] == []


def test_paper_runtime_dry_run_invalid_duration_high_blocked() -> None:
    result = _run(
        "--mode",
        "demo",
        "--config",
        str(SAFE_CONFIG),
        "--duration-seconds",
        "999",
        "--json",
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["reason"] == "paper_runtime_dry_run_invalid_duration"
    assert payload["steps"] == []


def test_paper_runtime_dry_run_invalid_max_signals_low_blocked() -> None:
    result = _run("--mode", "demo", "--config", str(SAFE_CONFIG), "--max-signals", "0", "--json")
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["reason"] == "paper_runtime_dry_run_invalid_max_signals"
    assert payload["steps"] == []


def test_paper_runtime_dry_run_invalid_max_signals_high_blocked() -> None:
    result = _run(
        "--mode",
        "demo",
        "--config",
        str(SAFE_CONFIG),
        "--max-signals",
        "999",
        "--json",
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["reason"] == "paper_runtime_dry_run_invalid_max_signals"
    assert payload["steps"] == []


def test_paper_runtime_dry_run_no_api_keys_required(monkeypatch) -> None:
    for key in ("API_KEY", "API_SECRET", "BINANCE_API_KEY", "BINANCE_API_SECRET"):
        monkeypatch.delenv(key, raising=False)
    result = _run("--mode", "demo", "--config", str(SAFE_CONFIG), "--json")
    assert result.returncode == 0, result.stderr


def test_paper_runtime_dry_run_output_cp1252_safe() -> None:
    result = _run("--mode", "demo", "--config", str(SAFE_CONFIG), "--json")
    assert result.returncode == 0, result.stderr
    result.stdout.encode("cp1252")


def test_paper_runtime_dry_run_source_safety_contract() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden = (
        "ccxt",
        "create_order",
        "fetch_balance",
        "fetch_ticker",
        "load_markets",
        "get_secret",
        "keychain",
        "os.environ",
        "getenv",
        "shell=True",
        "TradingController(",
    )
    for token in forbidden:
        assert token not in source
