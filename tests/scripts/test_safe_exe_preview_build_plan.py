from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

SCRIPT = Path("scripts/safe_exe_preview_build_plan.py")


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
    readiness = payload["safe_exe_preview_build_plan"]

    assert payload["safety_contract_version"] == "safe_exe_preview_build_plan.v1"
    assert payload["status"] == "ok"
    assert readiness["preview_only"] is True
    assert readiness["build_plan_only"] is True
    assert readiness["build_performed"] is False
    assert readiness["exe_build_performed"] is False
    assert readiness["installer_build_performed"] is False
    assert readiness["pyinstaller_build_performed"] is False
    assert readiness["briefcase_build_performed"] is False
    assert readiness["allowed_entrypoint"] == "scripts/run_local_bot.py"
    assert readiness["allowed_default_args"] == ["--mode", "demo", "--preview-plan"]
    assert readiness["build_command_execution_allowed"] is False
    assert readiness["build_command_executed"] is False


def test_artifact_policy_and_bundling_flags() -> None:
    payload = json.loads(_run().stdout)
    readiness = payload["safe_exe_preview_build_plan"]
    assert set(
        [
            ".env",
            "*.env",
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
        ]
    ).issubset(set(readiness["denied_artifact_patterns"]))
    assert readiness["env_file_bundled"] is False
    assert readiness["local_db_bundled"] is False
    assert readiness["logs_bundled"] is False
    assert readiness["reports_bundled"] is False
    assert readiness["tmp_artifacts_bundled"] is False
    assert readiness["test_secrets_bundled"] is False
    assert readiness["cache_artifacts_bundled"] is False
    assert readiness["local_user_data_bundled"] is False
    assert readiness["keychain_artifacts_bundled"] is False


def test_security_boundaries_and_preview_command_data() -> None:
    payload = json.loads(_run().stdout)
    readiness = payload["safe_exe_preview_build_plan"]
    checks = payload["checks"]

    assert isinstance(readiness["build_command_preview"], list)
    assert readiness["build_command_execution_allowed"] is False
    assert readiness["build_command_executed"] is False
    assert readiness["live_mode_allowed"] is False
    assert readiness["live_entrypoint_allowed"] is False
    assert readiness["api_keys_required"] is False
    assert readiness["api_keys_required_for_build"] is False
    assert readiness["api_keys_required_for_launch"] is False
    assert readiness["secrets_read"] is False
    assert readiness["keychain_read"] is False
    assert readiness["env_values_read"] is False
    assert readiness["dot_env_read"] is False
    assert readiness["home_directory_scanned"] is False
    assert readiness["exchange_io"] == "disabled"
    assert readiness["order_submission"] == "disabled"
    assert readiness["runtime_loop_started"] is False
    assert readiness["production_runtime_loop_started"] is False
    assert checks["entrypoint_allowlisted"] is True
    assert checks["default_args_safe"] is True
    assert checks["live_blocked_by_policy"] is True
    assert checks["artifact_policy_present"] is True
    assert checks["build_not_performed"] is True
    assert checks["release_boundary_not_performed"] is True
    assert checks["runtime_boundary_not_started"] is True
    assert checks["exchange_or_order_disabled"] is True
    assert checks["preview_build_profiles_present"] is True
    assert readiness["preview_build_profiles_exist"]["windows"] is True
    assert readiness["preview_build_profiles_exist"]["linux"] is True
    assert readiness["preview_build_profiles_exist"]["macos"] is True


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
        "getenv",
        "dotenv",
        "path.home",
        "shell=true",
        "subprocess.run",
        "pyinstaller.__main__",
        "briefcase build",
        "write_text",
        "write_bytes",
        "open(",
    ]
    for token in forbidden:
        assert token not in source


def test_preview_profiles_path_contract_and_safety() -> None:
    profiles = {
        "linux": Path("deploy/packaging/profiles/preview/linux.toml"),
        "macos": Path("deploy/packaging/profiles/preview/macos.toml"),
        "windows": Path("deploy/packaging/profiles/preview/windows.toml"),
    }
    repo_root = Path.cwd()
    sensitive_tokens = (".env", "api_key", "api_secret", "secret", "token", "keychain")
    home_tokens = ("/home/", "\\Users\\", "~/", "%USERPROFILE%")

    for platform, profile_path in profiles.items():
        profile = tomllib.loads(profile_path.read_text(encoding="utf-8"))
        assert profile["platform"] == platform

        pyinstaller = profile["pyinstaller"]
        briefcase = profile["briefcase"]

        entry_raw = pyinstaller["entrypoint"]
        entry_resolved = (profile_path.parent / entry_raw.replace("\\", "/")).resolve()
        assert entry_resolved.exists()
        assert entry_resolved.relative_to(repo_root).as_posix() == "scripts/run_local_bot.py"
        assert pyinstaller["runtime_name"] == "dudzian-bot-preview"

        dist_resolved = (profile_path.parent / pyinstaller["dist_dir"].replace("\\", "/")).resolve()
        work_resolved = (profile_path.parent / pyinstaller["work_dir"].replace("\\", "/")).resolve()
        briefcase_project_resolved = (
            profile_path.parent / briefcase["project"].replace("\\", "/")
        ).resolve()
        briefcase_out_resolved = (
            profile_path.parent / briefcase["output_dir"].replace("\\", "/")
        ).resolve()

        assert str(dist_resolved.relative_to(repo_root)).startswith("dist/preview/")
        assert str(work_resolved.relative_to(repo_root)).startswith("var/build/preview/")
        assert briefcase_project_resolved.relative_to(repo_root).as_posix() == "ui/briefcase"
        assert str(briefcase_out_resolved.relative_to(repo_root)).startswith("dist/preview/")

        raw_blob = json.dumps(profile).lower()
        assert "live" not in raw_blob
        for token in sensitive_tokens:
            assert token not in raw_blob
        for token in home_tokens:
            assert token.lower() not in raw_blob
        assert "pyinstaller --" not in raw_blob
        assert "briefcase build" not in raw_blob
