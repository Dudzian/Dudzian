from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/security_packaging_readiness.py")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--json"], capture_output=True, text=True, check=False
    )


def test_happy_path_demo_paper() -> None:
    result = _run("--config", "config/e2e/demo_paper.yml")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    readiness = payload["security_packaging_readiness"]

    assert payload["safety_contract_version"] == "security_packaging_readiness.v1"
    assert readiness["installer_fingerprint_contract_checked"] is True
    assert readiness["packaged_config_contract_checked"] is True
    assert readiness["installer_safe"] is True
    assert readiness["first_run_safe"] is True
    assert readiness["live_mode_enabled"] is False
    assert readiness["paper_mode_enabled"] is True
    assert readiness["credentials_onboarding_separate_from_install"] is True
    assert readiness["api_keys_bundled"] is False
    assert readiness["env_file_bundled"] is False
    assert readiness["local_db_bundled"] is False
    assert readiness["logs_bundled"] is False
    assert readiness["reports_bundled"] is False
    assert readiness["tmp_artifacts_bundled"] is False
    assert readiness["test_secrets_bundled"] is False
    assert readiness["secrets_read"] is False
    assert readiness["keychain_read"] is False
    assert readiness["env_values_read"] is False
    assert readiness["exchange_io"] == "disabled"
    assert readiness["order_submission"] == "disabled"
    assert readiness["runtime_loop_started"] is False
    assert readiness["production_runtime_loop_started"] is False


def test_aggregates_contract_versions() -> None:
    result = _run("--config", "config/e2e/demo_paper.yml")
    payload = json.loads(result.stdout)
    contracts = payload["contracts"]
    assert (
        contracts["installer_fingerprint_readiness"]["safety_contract_version"]
        == "installer_fingerprint_readiness.v1"
    )
    assert (
        contracts["packaged_config_readiness"]["safety_contract_version"]
        == "packaged_config_readiness.v1"
    )
    assert payload["status"] in {"ok", "warning", "blocked"}


def test_unsafe_config_propagation(tmp_path: Path) -> None:
    cfg = tmp_path / "unsafe.yml"
    cfg.write_text(
        "trading:\n  enable_live_mode: true\n  enable_paper_mode: true\nexecution:\n  default_mode: paper\n  force_paper_when_offline: true\n  live:\n    enabled: false\n",
        encoding="utf-8",
    )
    result = _run("--config", str(cfg))
    payload = json.loads(result.stdout)
    assert payload["status"] in {"blocked", "warning"}
    assert (
        "unsafe_config:trading.enable_live_mode" in payload["issues"]
        or "child_contract_failed" in payload["issues"]
    )


def test_source_safety() -> None:
    source = SCRIPT.read_text(encoding="utf-8").lower()
    forbidden = [
        "ccxt",
        "create_order",
        "fetch_balance",
        "fetch_ticker",
        "load_markets",
        "get_secret",
        "set_secret",
        "keyring.get_password",
        "os.environ",
        "getenv(",
        "dotenv",
        "requests.",
        "httpx.",
        "urllib.",
        "shell=true",
        "write_text",
        "write_bytes",
        "build_installer",
        "pyinstaller",
    ]
    for token in forbidden:
        assert token not in source


def test_cp1252_safe_output() -> None:
    result = _run("--config", "config/e2e/demo_paper.yml")
    assert result.returncode == 0
    result.stdout.encode("cp1252")


def test_modes_and_invalid_mode() -> None:
    for mode in ("install", "first-run"):
        result = _run("--mode", mode, "--config", "config/e2e/demo_paper.yml")
        assert result.returncode == 0
        assert json.loads(result.stdout)["mode"] == mode

    invalid = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mode",
            "invalid",
            "--config",
            "config/e2e/demo_paper.yml",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid.returncode != 0
