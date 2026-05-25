from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/installer_fingerprint_readiness.py")


def _run(*args: str) -> tuple[int, dict[str, object], str]:
    cmd = [sys.executable, str(SCRIPT), *args, "--json"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    payload = json.loads(result.stdout)
    return result.returncode, payload, result.stdout


def test_installer_fingerprint_readiness_safe_output() -> None:
    code, payload, stdout = _run()
    assert code == 0
    readiness = payload["installer_fingerprint_readiness"]
    assert payload["safety_contract_version"] == "installer_fingerprint_readiness.v1"
    assert readiness["license_activation_performed"] is False
    assert readiness["secrets_read"] is False
    assert readiness["keychain_read"] is False
    assert readiness["env_values_read"] is False
    assert readiness["api_keys_required"] is False
    assert readiness["exchange_io"] == "disabled"
    assert readiness["order_submission"] == "disabled"
    assert readiness["runtime_loop_started"] is False
    assert readiness["production_runtime_loop_started"] is False
    stdout.encode("cp1252")


def test_installer_fingerprint_readiness_does_not_expose_raw_identifiers() -> None:
    _, payload, _ = _run()
    readiness = payload["installer_fingerprint_readiness"]
    assert readiness["fingerprint_value_exposed"] is False
    assert readiness["raw_machine_identifiers_exposed"] is False
    preview = readiness["fingerprint_preview"]
    if isinstance(preview, str):
        assert len(preview) <= 24
        assert "..." in preview or "masked" in preview

    lowered = json.dumps(payload).lower()
    forbidden = [
        '"raw_hwid"',
        '"raw_machine_id"',
        '"mac_address"',
        '"serial_number"',
        '"motherboard_serial"',
    ]
    for token in forbidden:
        assert token not in lowered


def test_installer_fingerprint_readiness_source_safety() -> None:
    source = SCRIPT.read_text(encoding="utf-8").lower()
    forbidden = [
        "ccxt",
        "create_order",
        "fetch_balance",
        "fetch_ticker",
        "load_markets",
        "get_secret",
        "set_secret",
        "os.environ",
        "getenv(",
        "requests.",
        "httpx.",
        "urllib.",
        "open(",
        "write_text",
        "write_bytes",
    ]
    for token in forbidden:
        assert token not in source


def test_installer_fingerprint_readiness_modes_are_safe() -> None:
    for mode in ("install", "first-run"):
        code, payload, _ = _run("--mode", mode)
        assert code == 0
        assert payload["mode"] == mode
        readiness = payload["installer_fingerprint_readiness"]
        assert readiness["installer_safe"] is True
        assert readiness["first_run_safe"] is True


def test_installer_fingerprint_readiness_invalid_mode_rejected() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--mode", "invalid", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
