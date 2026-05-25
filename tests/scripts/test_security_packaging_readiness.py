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
    assert readiness["release_integrity_contract_checked"] is True
    assert readiness["release_integrity_readiness_present"] is True
    assert readiness["release_integrity_contract_version"] == "release_integrity_readiness.v1"
    assert readiness["release_integrity_readiness_status"] in {"warning", "blocked"}
    assert readiness["packaged_config_contract_checked"] is True
    assert readiness["installer_safe"] is True
    assert readiness["first_run_safe"] is True
    assert readiness["live_mode_enabled"] is False
    assert readiness["paper_mode_enabled"] is True
    assert readiness["credentials_onboarding_separate_from_install"] is True
    assert readiness["artifact_hygiene_checked"] is True
    assert readiness["artifact_exclude_policy_present"] is True
    assert readiness["artifact_exclude_policy_version"] == "security_packaging_artifact_policy.v1"
    assert set(
        (
            ".env",
            "trading.db",
            "bot_core/logs",
            "logs",
            "reports",
            "test-results",
            "var/security",
            "*api_key*",
            "*api_secret*",
            "*secret*",
            "*token*",
            "*keychain*",
        )
    ).issubset(set(readiness["denied_artifact_patterns"]))
    assert readiness["api_keys_bundled"] is False
    assert readiness["env_file_bundled"] is False
    assert readiness["local_db_bundled"] is False
    assert readiness["logs_bundled"] is False
    assert readiness["reports_bundled"] is False
    assert readiness["tmp_artifacts_bundled"] is False
    assert readiness["test_secrets_bundled"] is False
    assert readiness["cache_artifacts_bundled"] is False
    assert readiness["local_user_data_bundled"] is False
    assert readiness["keychain_artifacts_bundled"] is False
    assert readiness["secrets_read"] is False
    assert readiness["keychain_read"] is False
    assert readiness["env_values_read"] is False
    assert readiness["exchange_io"] == "disabled"
    assert readiness["order_submission"] == "disabled"
    assert readiness["runtime_loop_started"] is False
    assert readiness["production_runtime_loop_started"] is False
    assert readiness["safe_default_launch_checked"] is True
    assert readiness["safe_default_launch_policy_present"] is True
    assert (
        readiness["safe_default_launch_policy_version"]
        == "security_packaging_safe_launch_policy.v1"
    )
    assert readiness["preview_or_demo_default"] is True
    assert readiness["api_keys_required"] is False
    assert readiness["api_keys_required_for_launch"] is False
    assert readiness["real_orders_submitted"] is False
    assert readiness["live_launch_requires_explicit_reconfiguration"] is True
    assert readiness["live_launch_blocked_by_default"] is True
    assert readiness["packaged_shortcut_live_target_allowed"] is False
    assert readiness["packaged_shortcut_preview_target_allowed"] is True
    assert readiness["packaged_shortcut_demo_target_allowed"] is True
    assert "live" in readiness["unsafe_launch_modes_blocked"]


def test_default_args_safety() -> None:
    result = _run("--config", "config/e2e/demo_paper.yml")
    readiness = json.loads(result.stdout)["security_packaging_readiness"]
    default_args = readiness["packaged_shortcut_default_args"]
    assert isinstance(default_args, list)
    lowered = [str(item).lower() for item in default_args]
    joined = " ".join(lowered)
    assert "live" not in lowered
    assert any(token in joined for token in ("demo", "preview", "paper"))
    for forbidden in ("api_key", "api_secret", "secret", ".env", "binance", "bybit", "okx"):
        assert forbidden not in joined


def test_existing_live_blocked_sanity() -> None:
    run_local = subprocess.run(
        [
            sys.executable,
            "scripts/run_local_bot.py",
            "--mode",
            "live",
            "--config",
            "config/e2e/demo_paper.yml",
            "--preview-plan",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_local.returncode != 0

    operator_bundle = subprocess.run(
        [
            sys.executable,
            "scripts/operator_preview_bundle.py",
            "--mode",
            "live",
            "--config",
            "config/e2e/demo_paper.yml",
            "--duration-seconds",
            "5",
            "--max-signals",
            "1",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert operator_bundle.returncode != 0


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
        "open(",
        "path.home()",
    ]
    for token in forbidden:
        assert token not in source


def test_cp1252_safe_output() -> None:
    result = _run("--config", "config/e2e/demo_paper.yml")
    assert result.returncode == 0
    result.stdout.encode("cp1252")


def test_release_partial_is_preserved() -> None:
    result = _run("--config", "config/e2e/demo_paper.yml")
    payload = json.loads(result.stdout)
    readiness = payload["security_packaging_readiness"]
    assert readiness["release_signing_ready"] is False
    assert readiness["release_hash_manifest_ready"] in {True, False}
    assert readiness["release_integrity_status"] in {"warning", "partial", "blocked"}
    assert readiness["release_signing_ready"] is False
    assert payload["status"] in {"warning", "blocked"}
    assert "release_integrity_partial" in payload["issues"]
    assert "release_signing_not_ready" in payload["issues"]
    assert "artifact_scan_not_performed" in payload["issues"]
    assert (
        payload["contracts"]["release_integrity_readiness"]["safety_contract_version"]
        == "release_integrity_readiness.v1"
    )


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


def test_release_hash_manifest_fields_propagated() -> None:
    result = _run("--config", "config/e2e/demo_paper.yml")
    payload = json.loads(result.stdout)
    readiness = payload["security_packaging_readiness"]
    contract = payload["contracts"]["release_integrity_readiness"]["release_integrity_readiness"]

    assert readiness["release_hash_manifest_ready"] == contract["hash_manifest_ready"]
    assert readiness["release_hash_manifest_algorithm"] in {"sha256", "sha384", "sha512", None}
    assert isinstance(readiness["release_hash_manifest_policy_present"], bool)
    assert readiness["release_hash_manifest_generation_performed"] is False
    assert "hash_manifest_algorithm" in contract
    assert "release_signing_not_ready" in payload["issues"]
    assert "artifact_scan_not_performed" in payload["issues"]
