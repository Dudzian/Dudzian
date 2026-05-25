from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/release_integrity_readiness.py")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--json"], capture_output=True, text=True, check=False
    )


def test_happy_warning_path() -> None:
    result = _run()
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    readiness = payload["release_integrity_readiness"]
    assert payload["safety_contract_version"] == "release_integrity_readiness.v1"
    assert payload["status"] == "warning"
    assert readiness["release_integrity_contract_present"] is True
    assert readiness["release_integrity_contract_version"] == "release_integrity_readiness.v1"
    assert readiness["static_only"] is True
    assert readiness["local_only"] is True
    assert readiness["release_signing_ready"] is False
    assert readiness["certificate_material_read"] is False
    assert readiness["hash_manifest_generation_performed"] is False
    assert readiness["artifact_scan_performed"] is False
    assert readiness["artifact_built"] is False
    assert readiness["secrets_read"] is False
    assert readiness["keychain_read"] is False
    assert readiness["env_values_read"] is False
    assert readiness["exchange_io"] == "disabled"
    assert readiness["order_submission"] == "disabled"
    assert readiness["runtime_loop_started"] is False
    assert readiness["production_runtime_loop_started"] is False


def test_docs_policy_detection() -> None:
    payload = json.loads(_run().stdout)
    readiness = payload["release_integrity_readiness"]
    assert isinstance(readiness["release_process_docs_present"], bool)
    assert isinstance(readiness["release_channel_policy_present"], bool)
    assert isinstance(readiness["hash_manifest_policy_present"], bool)
    assert isinstance(readiness["signing_policy_present"], bool)
    if Path("docs/deploy/release_process.md").exists():
        assert readiness["release_process_docs_path"] == "docs/deploy/release_process.md"


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
        "subprocess",
        "subprocess.run",
        "os.system",
        "pyinstaller",
        "write_text",
        "write_bytes",
    ]
    for token in forbidden:
        assert token not in source


def test_cp1252_safe_output() -> None:
    result = _run()
    assert result.returncode == 0
    result.stdout.encode("cp1252")


def test_mode_contract() -> None:
    for mode in ("prebuild", "release"):
        result = _run("--mode", mode)
        assert result.returncode == 0
        assert json.loads(result.stdout)["mode"] == mode

    invalid = subprocess.run(
        [sys.executable, str(SCRIPT), "--mode", "invalid", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid.returncode != 0
