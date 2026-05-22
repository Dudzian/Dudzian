from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/sandbox_testnet_readiness.py"
SAFE_CONFIG = REPO_ROOT / "config/e2e/demo_paper.yml"


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
    parts = dotted_path.split(".")
    for segment in parts[:-1]:
        current = current.setdefault(segment, {})
    current[parts[-1]] = value
    mutated = tmp_path / f"mutated_{parts[-1]}.yml"
    mutated.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return mutated


def test_sandbox_testnet_readiness_safe_repo_config_json() -> None:
    result = _run("--config", str(SAFE_CONFIG), "--environment", "binance_paper", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["environment"] == "binance_paper"
    assert payload["safety_contract_version"] == "sandbox_testnet_readiness.v1"
    readiness = payload["sandbox_testnet_readiness"]
    assert readiness["static_only"] is True
    assert readiness["exchange_io"] == "disabled"
    assert readiness["order_submission"] == "disabled"
    assert readiness["secrets_read"] is False
    assert readiness["api_keys_required"] is False
    assert readiness["runtime_loop_started"] is False
    assert readiness["live_mode_allowed"] is False
    assert payload["issues"] == []


def test_sandbox_testnet_readiness_blocks_execution_live_enabled(tmp_path: Path) -> None:
    cfg = _mutated_config(tmp_path, "execution.live.enabled", True)
    result = _run("--config", str(cfg), "--environment", "binance_paper", "--json")
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked"
    assert "unsafe_config:execution.live.enabled" in payload["issues"]


def test_sandbox_testnet_readiness_blocks_trading_enable_live_mode(tmp_path: Path) -> None:
    cfg = _mutated_config(tmp_path, "trading.enable_live_mode", True)
    result = _run("--config", str(cfg), "--environment", "binance_paper", "--json")
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert "unsafe_config:trading.enable_live_mode" in payload["issues"]


def test_sandbox_testnet_readiness_blocks_trading_disable_paper_mode(tmp_path: Path) -> None:
    cfg = _mutated_config(tmp_path, "trading.enable_paper_mode", False)
    result = _run("--config", str(cfg), "--environment", "binance_paper", "--json")
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert "unsafe_config:trading.enable_paper_mode" in payload["issues"]


def test_sandbox_testnet_readiness_blocks_force_paper_when_offline_disabled(tmp_path: Path) -> None:
    cfg = _mutated_config(tmp_path, "execution.force_paper_when_offline", False)
    result = _run("--config", str(cfg), "--environment", "binance_paper", "--json")
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert "unsafe_config:execution.force_paper_when_offline" in payload["issues"]


def test_sandbox_testnet_readiness_blocks_execution_default_mode_live(tmp_path: Path) -> None:
    cfg = _mutated_config(tmp_path, "execution.default_mode", "live")
    result = _run("--config", str(cfg), "--environment", "binance_paper", "--json")
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert "unsafe_config:execution.default_mode" in payload["issues"]


def test_sandbox_testnet_readiness_missing_config() -> None:
    result = _run(
        "--config", "config/e2e/missing_sandbox.yml", "--environment", "binance_paper", "--json"
    )
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["reason"] == "config_not_found"


def test_sandbox_testnet_readiness_no_api_keys_required(monkeypatch) -> None:
    for key in ("API_KEY", "API_SECRET", "BINANCE_API_KEY", "BINANCE_API_SECRET"):
        monkeypatch.delenv(key, raising=False)
    result = _run("--config", str(SAFE_CONFIG), "--environment", "binance_paper", "--json")
    assert result.returncode == 0, result.stderr


def test_sandbox_testnet_readiness_output_cp1252_safe() -> None:
    result = _run("--config", str(SAFE_CONFIG), "--environment", "binance_paper", "--json")
    assert result.returncode == 0, result.stderr
    result.stdout.encode("cp1252")


def test_sandbox_testnet_readiness_source_has_no_ccxt_or_exchange_calls() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "ccxt" not in source
    assert "create_order" not in source
    assert "fetch_balance" not in source
    assert "load_markets" not in source
    assert "keychain" not in source
