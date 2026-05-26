from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/safe_exe_preview_readiness.py")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--json"],
        capture_output=True,
        text=True,
        check=False,
    )


def test_happy_path_safe_output() -> None:
    result = _run()
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    readiness = payload["safe_exe_preview_readiness"]

    assert payload["safety_contract_version"] == "safe_exe_preview_readiness.v1"
    assert readiness["allowed_entrypoint"] == "scripts/run_local_bot.py"
    assert readiness["allowed_default_args"] == ["--mode", "demo", "--preview-plan"]
    assert readiness["live_mode_allowed"] is False
    assert readiness["build_performed"] is False
    assert readiness["exe_build_performed"] is False
    assert readiness["installer_build_performed"] is False
    assert readiness["pyinstaller_build_performed"] is False
    assert readiness["briefcase_build_performed"] is False
    assert readiness["secrets_read"] is False
    assert readiness["keychain_read"] is False
    assert readiness["env_values_read"] is False
    assert readiness["dot_env_read"] is False
    assert readiness["exchange_io"] == "disabled"
    assert readiness["order_submission"] == "disabled"
    assert readiness["runtime_loop_started"] is False
    assert readiness["production_runtime_loop_started"] is False


def test_artifact_policy() -> None:
    payload = json.loads(_run().stdout)
    readiness = payload["safe_exe_preview_readiness"]

    expected = {
        ".env",
        "trading.db",
        "logs",
        "reports",
        "var/security",
        "*secret*",
        "*token*",
        "*keychain*",
    }
    assert expected.issubset(set(readiness["denied_artifact_patterns"]))
    assert readiness["env_file_bundled"] is False
    assert readiness["local_db_bundled"] is False
    assert readiness["logs_bundled"] is False
    assert readiness["reports_bundled"] is False
    assert readiness["tmp_artifacts_bundled"] is False
    assert readiness["test_secrets_bundled"] is False
    assert readiness["cache_artifacts_bundled"] is False
    assert readiness["local_user_data_bundled"] is False
    assert readiness["keychain_artifacts_bundled"] is False


def test_source_safety() -> None:
    source = SCRIPT.read_text(encoding="utf-8").lower()
    forbidden = [
        "ccxt",
        "create_order",
        "fetch_balance",
        "load_markets",
        "get_secret",
        "set_secret",
        "os.environ",
        "getenv(",
        "dotenv",
        'read_text(".env")',
        "path.home",
        "shell=true",
        "subprocess.run",
        "briefcase build",
    ]
    for token in forbidden:
        assert token not in source


def test_cp1252_safe_output() -> None:
    result = _run()
    assert result.returncode == 0
    result.stdout.encode("cp1252")


def test_invalid_mode_rejected() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--mode", "live", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0


def test_optional_extended_entrypoint_and_live_blocked() -> None:
    payload = json.loads(_run().stdout)
    readiness = payload["safe_exe_preview_readiness"]

    assert readiness["optional_extended_entrypoint"] == "scripts/operator_preview_bundle.py"
    assert readiness["optional_extended_args"] == ["--mode", "demo", "--json"]
    assert readiness["live_entrypoint_allowed"] is False
    assert "live" in readiness["forbidden_modes"]
