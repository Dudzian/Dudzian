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


def test_hash_manifest_readiness_fields() -> None:
    payload = json.loads(_run().stdout)
    readiness = payload["release_integrity_readiness"]
    assert isinstance(readiness["hash_manifest_policy_present"], bool)
    assert readiness["hash_manifest_algorithm"] in {"sha256", "sha384", "sha512", None}
    assert isinstance(readiness["hash_manifest_ready"], bool)
    assert readiness["hash_manifest_required_for_release"] is True
    assert readiness["hash_manifest_generation_performed"] is False
    assert readiness["hash_manifest_artifact_path"] is None
    assert readiness["hash_manifest_artifact_exists"] is False
    assert readiness["hash_manifest_includes_source_commit"] is True
    assert readiness["hash_manifest_includes_build_id"] is True
    assert readiness["hash_manifest_includes_artifact_size"] is True
    assert readiness["hash_manifest_includes_artifact_sha"] is True
    assert readiness["hash_manifest_is_prebuild_policy_only"] is True
    assert readiness["hash_manifest_final_artifact_scan_required"] is True
    assert readiness["hash_manifest_final_artifact_scan_performed"] is False


def test_release_issues_visible_and_warning_status() -> None:
    payload = json.loads(_run().stdout)
    assert payload["status"] == "warning"
    assert "release_signing_not_ready" in payload["issues"]
    assert "artifact_scan_not_performed" in payload["issues"]
    assert any(
        issue in payload["issues"]
        for issue in ("hash_manifest_not_generated", "final_artifact_scan_not_performed")
    )


def test_release_channel_policy_fields_and_issues() -> None:
    payload = json.loads(_run().stdout)
    readiness = payload["release_integrity_readiness"]
    assert isinstance(readiness["release_channel_policy_present"], bool)
    assert readiness["release_channel_policy_version"] == "release_channel_policy.v1"
    channels = readiness["supported_release_channels"]
    assert set(("dev", "test", "rc", "ga")).issubset(set(channels))
    assert readiness["default_release_channel"] in channels
    assert (
        readiness["current_release_channel"] is None
        or readiness["current_release_channel"] in channels
    )
    assert readiness["dev_channel_allowed_without_signing"] is True
    assert readiness["test_channel_allowed_without_signing"] is True
    assert readiness["rc_channel_requires_hash_manifest"] is True
    assert readiness["rc_channel_requires_artifact_scan"] is True
    assert readiness["rc_channel_requires_signing_decision"] is True
    assert readiness["ga_channel_requires_hash_manifest"] is True
    assert readiness["ga_channel_requires_artifact_scan"] is True
    assert readiness["ga_channel_requires_signing"] is True
    assert readiness["ga_channel_requires_release_notes"] is True
    assert readiness["ga_channel_requires_source_commit"] is True
    assert readiness["ga_channel_requires_build_id"] is True
    assert isinstance(readiness["promotion_policy_present"], bool)
    assert readiness["rc_to_ga_promotion_requires_clean_security_manifest"] is True
    assert readiness["rc_to_ga_promotion_requires_no_known_blockers"] is True
    assert readiness["release_channel_gate_performed"] is False
    assert readiness["release_channel_gate_result"] == "not_performed"
    assert readiness["release_channel_is_prebuild_policy_only"] is True
    assert "release_channel_gate_not_performed" in payload["issues"]
    assert "ga_release_not_ready" in payload["issues"]
    assert "release_signing_not_ready" in payload["issues"]
    assert "artifact_scan_not_performed" in payload["issues"]
    assert payload["status"] == "warning"


def test_release_promotion_gate_policy_fields_and_issues() -> None:
    payload = json.loads(_run().stdout)
    readiness = payload["release_integrity_readiness"]
    assert readiness["promotion_gate_policy_present"] is True
    assert readiness["promotion_gate_policy_version"] == "release_promotion_gate_policy.v1"
    assert readiness["promotion_gate_performed"] is False
    assert readiness["promotion_gate_result"] == "not_performed"
    assert readiness["rc_to_ga_promotion_performed"] is False
    assert readiness["rc_to_ga_promotion_ready"] is False
    assert isinstance(readiness["rc_to_ga_blockers"], list)
    assert readiness["rc_to_ga_blockers"]
    assert readiness["rc_to_ga_requires_clean_security_manifest"] is True
    assert readiness["rc_to_ga_requires_no_known_blockers"] is True
    assert readiness["rc_to_ga_requires_hash_manifest"] is True
    assert readiness["rc_to_ga_requires_final_artifact_scan"] is True
    assert readiness["rc_to_ga_requires_signing"] is True
    assert readiness["rc_to_ga_requires_release_notes"] is True
    assert readiness["rc_to_ga_requires_source_commit"] is True
    assert readiness["rc_to_ga_requires_build_id"] is True
    assert readiness["rc_to_ga_requires_reproducible_build_record"] is True
    assert readiness["rc_to_ga_is_prebuild_policy_only"] is True
    assert "promotion_gate_not_performed" in payload["issues"]
    assert "rc_to_ga_promotion_not_ready" in payload["issues"]
    assert "release_signing_not_ready" in payload["issues"]
    assert "artifact_scan_not_performed" in payload["issues"]
    assert payload["status"] == "warning"
