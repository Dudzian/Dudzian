from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/packaged_config_readiness.py")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--json"], capture_output=True, text=True, check=False
    )


def test_happy_path_demo_paper() -> None:
    result = _run("--config", "config/e2e/demo_paper.yml")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    readiness = payload["packaged_config_readiness"]
    assert payload["status"] == "ok"
    assert payload["safety_contract_version"] == "packaged_config_readiness.v1"
    assert readiness["live_mode_enabled"] is False
    assert readiness["paper_mode_enabled"] is True
    assert readiness["force_paper_when_offline"] is True
    assert readiness["api_keys_required_for_install"] is False
    assert readiness["api_keys_bundled"] is False
    assert readiness["env_file_bundled"] is False
    assert readiness["local_db_bundled"] is False
    assert readiness["logs_bundled"] is False
    assert readiness["reports_bundled"] is False
    assert readiness["secrets_read"] is False
    assert readiness["keychain_read"] is False
    assert readiness["env_values_read"] is False
    assert readiness["exchange_io"] == "disabled"
    assert readiness["order_submission"] == "disabled"
    assert readiness["runtime_loop_started"] is False
    assert readiness["production_runtime_loop_started"] is False


def test_unsafe_live_config_blocked(tmp_path: Path) -> None:
    cfg = tmp_path / "unsafe.yml"
    cfg.write_text(
        "trading:\n  enable_live_mode: true\n  enable_paper_mode: true\nexecution:\n  default_mode: paper\n  force_paper_when_offline: true\n  live:\n    enabled: false\n",
        encoding="utf-8",
    )
    result = _run("--config", str(cfg))
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "blocked"
    assert any(issue.startswith("unsafe_config:") for issue in payload["issues"])


def test_missing_config_handled() -> None:
    result = _run("--config", "config/e2e/missing.yml")
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["reason"] == "config_not_found"


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
        "keyring",
        "os.environ",
        "getenv(",
        "dotenv",
        "requests.",
        "httpx.",
        "urllib.",
        "write_text",
        "write_bytes",
        "home()",
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
