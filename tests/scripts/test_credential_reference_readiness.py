from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/credential_reference_readiness.py"
SAFE_CONFIG = REPO_ROOT / "config/e2e/demo_paper.yml"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_credential_reference_readiness_safe_repo_config_json() -> None:
    result = _run("--config", str(SAFE_CONFIG), "--environment", "binance_paper", "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)

    assert payload["status"] == "ok"
    assert payload["environment"] == "binance_paper"
    assert payload["config_shape"] == "e2e_overlay"
    assert payload["safety_contract_version"] == "credential_reference_readiness.v1"
    readiness = payload["credential_reference_readiness"]
    assert readiness["static_only"] is True
    assert readiness["exchange_io"] == "disabled"
    assert readiness["order_submission"] == "disabled"
    assert readiness["secrets_read"] is False
    assert readiness["keychain_read"] is False
    assert readiness["env_values_read"] is False
    assert readiness["credential_values_read"] is False
    assert readiness["credential_values_present"] is False
    assert readiness["api_keys_required"] is False
    assert readiness["runtime_loop_started"] is False
    assert readiness["live_mode_allowed"] is False
    assert payload["issues"] == []


def _unsafe_config(tmp_path: Path, path: str, value: object) -> Path:
    config = yaml.safe_load(SAFE_CONFIG.read_text(encoding="utf-8"))
    parts = path.split(".")
    current = config
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value
    target = tmp_path / "unsafe.yml"
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target


def test_unsafe_execution_live_enabled_blocked(tmp_path: Path) -> None:
    unsafe = _unsafe_config(tmp_path, "execution.live.enabled", True)
    result = _run("--config", str(unsafe), "--json")
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked"
    assert any("unsafe_config:execution.live.enabled" in i for i in payload["issues"])


def test_unsafe_trading_enable_live_mode_blocked(tmp_path: Path) -> None:
    unsafe = _unsafe_config(tmp_path, "trading.enable_live_mode", True)
    result = _run("--config", str(unsafe), "--json")
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert any("unsafe_config:trading.enable_live_mode" in i for i in payload["issues"])


def test_unsafe_trading_enable_paper_mode_blocked(tmp_path: Path) -> None:
    unsafe = _unsafe_config(tmp_path, "trading.enable_paper_mode", False)
    result = _run("--config", str(unsafe), "--json")
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert any("unsafe_config:trading.enable_paper_mode" in i for i in payload["issues"])


def test_unsafe_force_paper_when_offline_blocked(tmp_path: Path) -> None:
    unsafe = _unsafe_config(tmp_path, "execution.force_paper_when_offline", False)
    result = _run("--config", str(unsafe), "--json")
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert any("unsafe_config:execution.force_paper_when_offline" in i for i in payload["issues"])


def test_inline_secret_value_blocked(tmp_path: Path) -> None:
    config = yaml.safe_load(SAFE_CONFIG.read_text(encoding="utf-8"))
    config.setdefault("credentials", {})["api_key"] = "REAL_VALUE_SHOULD_NOT_BE_INLINE"
    target = tmp_path / "inline_secret.yml"
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run("--config", str(target), "--json")
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["reason"] == "credential_reference_readiness_inline_secret_value"
    assert any("inline_secret_value:credentials.api_key" in i for i in payload["issues"])


def test_inline_nested_secret_value_blocked(tmp_path: Path) -> None:
    config = yaml.safe_load(SAFE_CONFIG.read_text(encoding="utf-8"))
    config.setdefault("exchanges", {}).setdefault("binance", {})["apiSecret"] = (
        "REAL_VALUE_SHOULD_NOT_BE_INLINE"
    )
    target = tmp_path / "nested_secret.yml"
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run("--config", str(target), "--json")
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert any("inline_secret_value:exchanges.binance.apiSecret" in i for i in payload["issues"])


def test_missing_config() -> None:
    result = _run("--config", "config/e2e/does_not_exist.yml", "--json")
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["reason"] == "config_not_found"


def test_no_api_keys_env_needed(monkeypatch) -> None:
    for key in ("API_KEY", "API_SECRET", "BINANCE_API_KEY", "BINANCE_API_SECRET"):
        monkeypatch.delenv(key, raising=False)
    result = _run("--config", str(SAFE_CONFIG), "--json")
    assert result.returncode == 0, result.stderr


def test_output_cp1252_safe() -> None:
    result = _run("--config", str(SAFE_CONFIG), "--json")
    assert result.returncode == 0, result.stderr
    result.stdout.encode("cp1252")


def test_source_safety_no_forbidden_paths() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden = [
        "ccxt",
        "create_order",
        "fetch_balance",
        "fetch_ticker",
        "load_markets",
        "keychain",
        "get_secret",
        "os.environ",
        "getenv",
        "apiKey",
        "apiSecret",
    ]
    for marker in forbidden:
        assert marker not in source
