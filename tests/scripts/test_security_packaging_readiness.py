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
        "pyinstaller.__main__",
        "briefcase build",
        "write_text",
        "write_bytes",
        "build_installer",
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


def test_release_channel_policy_fields_propagated() -> None:
    payload = json.loads(_run("--config", "config/e2e/demo_paper.yml").stdout)
    readiness = payload["security_packaging_readiness"]
    contract = payload["contracts"]["release_integrity_readiness"]["release_integrity_readiness"]
    assert readiness["release_channel_policy_present"] == contract["release_channel_policy_present"]
    assert readiness["release_channel_policy_version"] == contract["release_channel_policy_version"]
    assert readiness["supported_release_channels"] == contract["supported_release_channels"]
    assert readiness["default_release_channel"] == contract["default_release_channel"]
    assert readiness["release_channel_gate_performed"] is False
    assert readiness["release_channel_gate_result"] == "not_performed"
    assert "release_channel_gate_not_performed" in payload["issues"]
    assert "ga_release_not_ready" in payload["issues"]
    assert "release_signing_not_ready" in payload["issues"]
    assert "artifact_scan_not_performed" in payload["issues"]


def test_release_promotion_gate_fields_propagated() -> None:
    payload = json.loads(_run("--config", "config/e2e/demo_paper.yml").stdout)
    readiness = payload["security_packaging_readiness"]
    contract = payload["contracts"]["release_integrity_readiness"]["release_integrity_readiness"]
    assert readiness["promotion_gate_policy_present"] == contract["promotion_gate_policy_present"]
    assert readiness["promotion_gate_policy_version"] == contract["promotion_gate_policy_version"]
    assert readiness["promotion_gate_performed"] is False
    assert readiness["promotion_gate_result"] == "not_performed"
    assert readiness["rc_to_ga_promotion_ready"] is False
    assert readiness["rc_to_ga_promotion_performed"] is False
    assert readiness["rc_to_ga_blockers"] == contract["rc_to_ga_blockers"]
    assert "promotion_gate_not_performed" in payload["issues"]
    assert "rc_to_ga_promotion_not_ready" in payload["issues"]


def test_safe_exe_preview_fields_propagated() -> None:
    payload = json.loads(_run("--config", "config/e2e/demo_paper.yml").stdout)
    readiness = payload["security_packaging_readiness"]
    checks = payload["checks"]
    contract = payload["contracts"]["safe_exe_preview_readiness"]

    assert contract["safety_contract_version"] == "safe_exe_preview_readiness.v1"
    assert readiness["safe_exe_preview_contract_checked"] is True
    assert readiness["safe_exe_preview_contract_version"] == "safe_exe_preview_readiness.v1"
    assert readiness["safe_exe_preview_ready"] is True
    assert readiness["safe_exe_preview_status"] == "ok"
    assert readiness["safe_exe_preview_allowed_entrypoint"] == "scripts/run_local_bot.py"
    assert readiness["safe_exe_preview_allowed_default_args"] == [
        "--mode",
        "demo",
        "--preview-plan",
    ]
    assert readiness["safe_exe_preview_live_mode_allowed"] is False
    assert readiness["safe_exe_preview_build_performed"] is False
    assert readiness["safe_exe_preview_exe_build_performed"] is False
    assert readiness["safe_exe_preview_installer_build_performed"] is False
    assert readiness["safe_exe_preview_pyinstaller_build_performed"] is False
    assert readiness["safe_exe_preview_briefcase_build_performed"] is False
    assert readiness["safe_exe_preview_release_upload_performed"] is False
    assert readiness["safe_exe_preview_promotion_performed"] is False
    assert readiness["safe_exe_preview_final_artifact_scan_performed"] is False
    assert readiness["safe_exe_preview_final_hash_manifest_generated"] is False
    assert readiness["safe_exe_preview_exchange_io"] == "disabled"
    assert readiness["safe_exe_preview_order_submission"] == "disabled"
    assert readiness["safe_exe_preview_runtime_loop_started"] is False
    assert readiness["safe_exe_preview_production_runtime_loop_started"] is False

    assert checks["safe_exe_preview_contract_referenced"] is True
    assert checks["safe_exe_preview_entrypoint_allowlisted"] is True
    assert checks["safe_exe_preview_default_args_safe"] is True
    assert checks["safe_exe_preview_live_blocked_by_policy"] is True
    assert checks["safe_exe_preview_build_not_performed"] is True
    assert checks["safe_exe_preview_contract_checked"] is True
    assert checks["safe_exe_preview_contract_version_ok"] is True
    assert checks["safe_exe_preview_release_boundary_not_performed"] is True
    assert checks["safe_exe_preview_runtime_boundary_not_started"] is True


def test_safe_exe_preview_warning_propagation(monkeypatch) -> None:
    import scripts.security_packaging_readiness as spr

    real_run_child = spr._run_child
    payload_ok = json.loads(_run("--config", "config/e2e/demo_paper.yml").stdout)

    def fake_run_child(command: list[str]):
        if command[1].endswith("safe_exe_preview_readiness.py"):
            warning_payload = {
                "status": "warning",
                "safety_contract_version": "safe_exe_preview_readiness.v1",
                "safe_exe_preview_readiness": payload_ok["contracts"]["safe_exe_preview_readiness"][
                    "safe_exe_preview_readiness"
                ],
                "issues": ["simulated_warning"],
            }
            return warning_payload, None
        return real_run_child(command)

    monkeypatch.setattr(spr, "_run_child", fake_run_child)
    payload, _ = spr.build_payload("first-run", Path("config/e2e/demo_paper.yml"))
    assert payload["security_packaging_readiness"]["safe_exe_preview_ready"] is False
    assert "safe_exe_preview_readiness_not_ok" in payload["issues"]


def test_safe_exe_preview_blocked_child_blocks_aggregator(monkeypatch) -> None:
    import scripts.security_packaging_readiness as spr

    real_run_child = spr._run_child
    payload_ok = json.loads(_run("--config", "config/e2e/demo_paper.yml").stdout)

    def fake_run_child(command: list[str]):
        if command[1].endswith("safe_exe_preview_readiness.py"):
            blocked_payload = {
                "status": "blocked",
                "safety_contract_version": "safe_exe_preview_readiness.v1",
                "safe_exe_preview_readiness": payload_ok["contracts"]["safe_exe_preview_readiness"][
                    "safe_exe_preview_readiness"
                ],
                "issues": ["simulated_blocked"],
            }
            return blocked_payload, None
        return real_run_child(command)

    monkeypatch.setattr(spr, "_run_child", fake_run_child)
    payload, _ = spr.build_payload("first-run", Path("config/e2e/demo_paper.yml"))
    assert payload["status"] == "blocked"
    assert "child_contract_failed" in payload["issues"]
    assert "safe_exe_preview_readiness_not_ok" in payload["issues"]


def test_safe_exe_preview_invalid_json_error_propagation(monkeypatch) -> None:
    import scripts.security_packaging_readiness as spr

    real_run_child = spr._run_child

    def fake_run_child(command: list[str]):
        if command[1].endswith("safe_exe_preview_readiness.py"):
            return None, "child_payload_invalid_json"
        return real_run_child(command)

    monkeypatch.setattr(spr, "_run_child", fake_run_child)
    payload, _ = spr.build_payload("first-run", Path("config/e2e/demo_paper.yml"))
    readiness = payload["security_packaging_readiness"]
    assert readiness["safe_exe_preview_contract_checked"] is False
    assert readiness["safe_exe_preview_ready"] is False
    assert readiness["safe_exe_preview_contract_version"] is None
    assert "safe_exe_preview_contract_error:child_payload_invalid_json" in payload["issues"]
    assert "safe_exe_preview_contract_missing" in payload["issues"]


def test_safe_exe_preview_version_and_entrypoint_mismatch(monkeypatch) -> None:
    import scripts.security_packaging_readiness as spr

    real_run_child = spr._run_child
    payload_ok = json.loads(_run("--config", "config/e2e/demo_paper.yml").stdout)
    bad = payload_ok["contracts"]["safe_exe_preview_readiness"]["safe_exe_preview_readiness"].copy()
    bad["allowed_entrypoint"] = "scripts/live_runner.py"
    bad["allowed_default_args"] = ["--mode", "live"]

    def fake_run_child(command: list[str]):
        if command[1].endswith("safe_exe_preview_readiness.py"):
            return {
                "status": "ok",
                "safety_contract_version": "safe_exe_preview_readiness.v0",
                "safe_exe_preview_readiness": bad,
                "issues": [],
            }, None
        return real_run_child(command)

    monkeypatch.setattr(spr, "_run_child", fake_run_child)
    payload, _ = spr.build_payload("first-run", Path("config/e2e/demo_paper.yml"))
    checks = payload["checks"]
    assert payload["security_packaging_readiness"]["safe_exe_preview_ready"] is False
    assert checks["safe_exe_preview_contract_version_ok"] is False
    assert checks["safe_exe_preview_entrypoint_allowlisted"] is False
    assert checks["safe_exe_preview_default_args_safe"] is False
    assert "safe_exe_preview_contract_version_mismatch" in payload["issues"]
    assert "safe_exe_preview_entrypoint_not_allowlisted" in payload["issues"]
    assert "safe_exe_preview_default_args_unsafe" in payload["issues"]


def test_safe_exe_preview_build_runtime_exchange_mismatch(monkeypatch) -> None:
    import scripts.security_packaging_readiness as spr

    real_run_child = spr._run_child
    payload_ok = json.loads(_run("--config", "config/e2e/demo_paper.yml").stdout)
    bad = payload_ok["contracts"]["safe_exe_preview_readiness"]["safe_exe_preview_readiness"].copy()
    bad["build_performed"] = True
    bad["pyinstaller_build_performed"] = True
    bad["release_upload_performed"] = True
    bad["runtime_loop_started"] = True
    bad["exchange_io"] = "enabled"
    bad["order_submission"] = "enabled"

    def fake_run_child(command: list[str]):
        if command[1].endswith("safe_exe_preview_readiness.py"):
            return {
                "status": "ok",
                "safety_contract_version": "safe_exe_preview_readiness.v1",
                "safe_exe_preview_readiness": bad,
                "issues": [],
            }, None
        return real_run_child(command)

    monkeypatch.setattr(spr, "_run_child", fake_run_child)
    payload, _ = spr.build_payload("first-run", Path("config/e2e/demo_paper.yml"))
    checks = payload["checks"]
    assert payload["status"] == "blocked"
    assert checks["safe_exe_preview_build_not_performed"] is False
    assert checks["safe_exe_preview_release_boundary_not_performed"] is False
    assert checks["safe_exe_preview_runtime_boundary_not_started"] is False
    assert "safe_exe_preview_build_performed" in payload["issues"]
    assert "safe_exe_preview_release_boundary_performed" in payload["issues"]
    assert "safe_exe_preview_runtime_boundary_started" in payload["issues"]
    assert "safe_exe_preview_exchange_or_order_enabled" in payload["issues"]


def test_safe_exe_preview_build_plan_fields_propagated() -> None:
    payload = json.loads(_run("--config", "config/e2e/demo_paper.yml").stdout)
    readiness = payload["security_packaging_readiness"]
    checks = payload["checks"]
    contract = payload["contracts"]["safe_exe_preview_build_plan"]
    assert contract["safety_contract_version"] == "safe_exe_preview_build_plan.v1"
    assert readiness["safe_exe_preview_build_plan_ready"] is True
    assert readiness["safe_exe_preview_build_plan_status"] == "ok"
    assert readiness["safe_exe_preview_build_plan_build_plan_only"] is True
    assert checks["safe_exe_preview_build_plan_contract_version_ok"] is True


def test_safe_exe_preview_build_plan_warning_propagation(monkeypatch) -> None:
    import scripts.security_packaging_readiness as spr

    real_run_child = spr._run_child

    def fake_run_child(command: list[str]):
        if command[1].endswith("safe_exe_preview_build_plan.py"):
            return {
                "status": "warning",
                "safety_contract_version": "safe_exe_preview_build_plan.v1",
                "safe_exe_preview_build_plan": {"preview_only": True, "build_plan_only": True},
                "issues": [],
            }, None
        return real_run_child(command)

    monkeypatch.setattr(spr, "_run_child", fake_run_child)
    payload, _ = spr.build_payload("first-run", Path("config/e2e/demo_paper.yml"))
    assert payload["status"] == "blocked"
    assert "safe_exe_preview_build_plan_readiness_not_ok" in payload["issues"]


def test_safe_exe_profile_validator_fields_propagated() -> None:
    payload = json.loads(_run("--config", "config/e2e/demo_paper.yml").stdout)
    readiness = payload["security_packaging_readiness"]
    checks = payload["checks"]
    contracts = payload["contracts"]
    assert "safe_exe_preview_profile_validator" in contracts
    assert readiness["safe_exe_preview_profile_validator_contract_checked"] is True
    assert (
        readiness["safe_exe_preview_profile_validator_contract_version"]
        == "safe_exe_preview_profile_validator.v1"
    )
    assert checks["safe_exe_preview_profile_validator_contract_version_ok"] is True
    assert checks["safe_exe_preview_profiles_valid"] is True
    assert checks["safe_exe_preview_profiles_complete"] is True
    assert checks["safe_exe_preview_profile_validator_ready"] is True
    assert checks["contracts_checked"] is True


def test_safe_exe_profile_validator_failure_propagation(monkeypatch) -> None:
    import scripts.security_packaging_readiness as spr

    real = spr._run_child

    def fake(command: list[str]):
        if command[1].endswith("safe_exe_preview_profile_validator.py"):
            return {
                "status": "blocked",
                "safety_contract_version": "safe_exe_preview_profile_validator.v0",
                "safe_exe_preview_profile_validator": {
                    "profile_count": 3,
                    "forbidden_tokens_present": True,
                    "all_profiles_exist": False,
                    "all_platforms_match": True,
                    "all_entrypoints_allowlisted": True,
                    "all_output_paths_preview_scoped": True,
                    "all_work_paths_preview_scoped": True,
                },
                "issues": ["x"],
            }, None
        return real(command)

    monkeypatch.setattr(spr, "_run_child", fake)
    payload, _ = spr.build_payload("first-run", Path("config/e2e/demo_paper.yml"))
    assert payload["status"] == "blocked"
    assert "safe_exe_preview_profile_validator_contract_version_mismatch" in payload["issues"]
    assert "safe_exe_preview_profile_validator_not_ok" in payload["issues"]
    assert "safe_exe_preview_profiles_invalid" in payload["issues"]
    assert "safe_exe_preview_profiles_incomplete" in payload["issues"]
    assert "safe_exe_preview_profile_validator_child_issues_present" in payload["issues"]


def test_safe_exe_profile_validator_ok_with_child_issues_blocks(monkeypatch) -> None:
    import scripts.security_packaging_readiness as spr

    real = spr._run_child
    payload_ok = json.loads(_run("--config", "config/e2e/demo_paper.yml").stdout)
    validator_ok = payload_ok["contracts"]["safe_exe_preview_profile_validator"][
        "safe_exe_preview_profile_validator"
    ]

    def fake(command: list[str]):
        if command[1].endswith("safe_exe_preview_profile_validator.py"):
            return {
                "status": "ok",
                "safety_contract_version": "safe_exe_preview_profile_validator.v1",
                "safe_exe_preview_profile_validator": validator_ok,
                "issues": ["simulated_profile_issue"],
            }, None
        return real(command)

    monkeypatch.setattr(spr, "_run_child", fake)
    payload, _ = spr.build_payload("first-run", Path("config/e2e/demo_paper.yml"))
    readiness = payload["security_packaging_readiness"]
    assert payload["status"] == "blocked"
    assert readiness["safe_exe_preview_profile_validator_ready"] is False
    assert "safe_exe_preview_profile_validator_child_issues_present" in payload["issues"]


def test_safe_exe_command_renderer_fields_and_contract_propagated() -> None:
    payload = json.loads(_run("--config", "config/e2e/demo_paper.yml").stdout)
    readiness = payload["security_packaging_readiness"]
    checks = payload["checks"]
    contracts = payload["contracts"]
    assert "safe_exe_preview_command_renderer" in contracts
    assert readiness["safe_exe_preview_command_renderer_contract_checked"] is True
    assert (
        readiness["safe_exe_preview_command_renderer_contract_version"]
        == "safe_exe_preview_command_renderer.v1"
    )
    assert readiness["safe_exe_preview_command_renderer_status"] == "ok"
    assert checks["safe_exe_preview_command_renderer_contract_checked"] is True
    assert checks["safe_exe_preview_command_renderer_contract_version_ok"] is True
    assert checks["safe_exe_preview_command_renderer_ready"] is True
    assert checks["safe_exe_preview_commands_render_only"] is True
    assert checks["safe_exe_preview_command_execution_blocked"] is True
    assert checks["safe_exe_preview_no_subprocess_or_shell"] is True
    assert checks["safe_exe_preview_all_commands_rendered"] is True
    assert checks["safe_exe_preview_all_entrypoints_allowlisted"] is True
    assert checks["safe_exe_preview_all_paths_preview_scoped"] is True
    assert checks["safe_exe_preview_no_forbidden_command_tokens"] is True
    assert checks["contracts_checked"] is True


def test_safe_exe_command_renderer_version_mismatch_blocks(monkeypatch) -> None:
    import scripts.security_packaging_readiness as spr

    real = spr._run_child

    def fake(command: list[str]):
        if command[1].endswith("safe_exe_preview_command_renderer.py"):
            real_payload, _ = real(command)
            return {
                "status": "ok",
                "safety_contract_version": "safe_exe_preview_command_renderer.v0",
                "safe_exe_preview_command_renderer": real_payload[
                    "safe_exe_preview_command_renderer"
                ],  # type: ignore[index]
                "issues": [],
            }, None
        return real(command)

    monkeypatch.setattr(spr, "_run_child", fake)
    payload, _ = spr.build_payload("first-run", Path("config/e2e/demo_paper.yml"))
    assert payload["status"] == "blocked"
    assert "safe_exe_preview_command_renderer_contract_version_mismatch" in payload["issues"]


def test_safe_exe_command_renderer_child_error_blocks(monkeypatch) -> None:
    import scripts.security_packaging_readiness as spr

    real = spr._run_child

    def fake(command: list[str]):
        if command[1].endswith("safe_exe_preview_command_renderer.py"):
            return None, "child_payload_invalid_json"
        return real(command)

    monkeypatch.setattr(spr, "_run_child", fake)
    payload, _ = spr.build_payload("first-run", Path("config/e2e/demo_paper.yml"))
    assert payload["status"] == "blocked"
    assert "child_contract_failed" in payload["issues"]
    assert (
        "safe_exe_preview_command_renderer_contract_error:child_payload_invalid_json"
        in payload["issues"]
    )
    assert "safe_exe_preview_command_renderer_contract_missing" in payload["issues"]


def test_safe_exe_command_renderer_issues_present_blocks(monkeypatch) -> None:
    import scripts.security_packaging_readiness as spr

    real = spr._run_child
    base = json.loads(_run("--config", "config/e2e/demo_paper.yml").stdout)["contracts"][
        "safe_exe_preview_command_renderer"
    ]["safe_exe_preview_command_renderer"]

    def fake(command: list[str]):
        if command[1].endswith("safe_exe_preview_command_renderer.py"):
            return {
                "status": "ok",
                "safety_contract_version": "safe_exe_preview_command_renderer.v1",
                "safe_exe_preview_command_renderer": base,
                "issues": ["simulated"],
            }, None
        return real(command)

    monkeypatch.setattr(spr, "_run_child", fake)
    payload, _ = spr.build_payload("first-run", Path("config/e2e/demo_paper.yml"))
    assert payload["status"] == "blocked"
    assert "safe_exe_preview_command_renderer_child_issues_present" in payload["issues"]
