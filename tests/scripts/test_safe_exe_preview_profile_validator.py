from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/safe_exe_preview_profile_validator.py")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--json"],
        check=False,
        capture_output=True,
        text=True,
    )


def test_happy_path() -> None:
    result = _run()
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["safety_contract_version"] == "safe_exe_preview_profile_validator.v1"
    validator = payload["safe_exe_preview_profile_validator"]
    assert validator["profiles_checked"] is True
    assert validator["profile_count"] == 3
    assert validator["all_profiles_exist"] is True
    assert validator["all_entrypoints_allowlisted"] is True
    assert validator["all_output_paths_preview_scoped"] is True
    assert validator["all_work_paths_preview_scoped"] is True
    assert validator["forbidden_tokens_present"] is False
    assert validator["all_toml_valid"] is True
    assert validator["all_runtime_names_ok"] is True
    assert validator["all_hidden_imports_ok"] is True
    assert validator["no_hidden_import_forbidden"] is True
    assert validator["all_briefcase_projects_ok"] is True
    assert validator["all_briefcase_apps_ok"] is True
    assert validator["all_briefcase_outputs_preview_scoped"] is True
    assert validator["no_profile_forbidden_tokens"] is True
    assert validator["build_not_performed"] is True
    assert validator["pyinstaller_build_performed"] is False
    assert validator["briefcase_build_performed"] is False
    assert validator["installer_build_performed"] is False
    assert validator["secrets_read"] is False
    assert validator["keychain_read"] is False
    assert validator["env_values_read"] is False
    assert validator["dot_env_read"] is False
    assert validator["home_directory_scanned"] is False
    assert validator["exchange_io"] == "disabled"
    assert validator["order_submission"] == "disabled"
    assert validator["runtime_loop_started"] is False
    assert validator["production_runtime_loop_started"] is False


def test_artifact_policy_and_bundling_flags() -> None:
    payload = json.loads(_run().stdout)
    validator = payload["safe_exe_preview_profile_validator"]
    assert validator["artifact_policy_checked"] is True
    assert validator["artifact_exclude_policy_version"] == "security_packaging_artifact_policy.v1"
    assert validator["denied_artifact_patterns"] == [
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
    assert validator["env_file_bundled"] is False
    assert validator["local_db_bundled"] is False
    assert validator["logs_bundled"] is False
    assert validator["reports_bundled"] is False
    assert validator["tmp_artifacts_bundled"] is False
    assert validator["test_secrets_bundled"] is False
    assert validator["cache_artifacts_bundled"] is False
    assert validator["local_user_data_bundled"] is False
    assert validator["keychain_artifacts_bundled"] is False


def test_per_platform_contract() -> None:
    payload = json.loads(_run().stdout)
    profiles = payload["safe_exe_preview_profile_validator"]["profiles"]
    for platform in ("linux", "macos", "windows"):
        prof = profiles[platform]
        assert prof["platform"] == platform
        assert prof["toml_valid"] is True
        assert prof["entrypoint_resolved"] == "scripts/run_local_bot.py"
        assert prof["dist_dir_resolved"].startswith(f"dist/preview/{platform}")
        assert prof["work_dir_resolved"].startswith(f"var/build/preview/pyinstaller/{platform}")
        assert prof["briefcase_project_resolved"] == "ui/briefcase"
        assert prof["briefcase_output_dir_resolved"].startswith(
            f"dist/preview/briefcase/{platform}"
        )


def test_invalid_mode_rejected() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--mode", "live", "--json"],
        check=False,
        capture_output=True,
        text=True,
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
    ]
    for token in forbidden:
        assert token not in source


def test_cp1252_safe_output() -> None:
    result = _run()
    assert result.returncode == 0
    result.stdout.encode("cp1252")


def test_blocked_missing_profile(monkeypatch) -> None:
    import scripts.safe_exe_preview_profile_validator as validator

    monkeypatch.setitem(
        validator.PROFILE_PATHS, "linux", "deploy/packaging/profiles/preview/missing.toml"
    )
    payload = validator._build_payload("preview")
    assert payload["status"] == "blocked"
    assert "preview_profile_missing:linux" in payload["issues"]


def test_blocked_invalid_toml(monkeypatch, tmp_path: Path) -> None:
    import scripts.safe_exe_preview_profile_validator as validator

    bad = tmp_path / "bad.toml"
    bad.write_text("platform='linux'\n[pyinstaller\nentrypoint='x'", encoding="utf-8")
    monkeypatch.setitem(validator.PROFILE_PATHS, "linux", str(bad))
    payload = validator._build_payload("preview")
    assert payload["status"] == "blocked"
    assert "preview_profile_invalid_toml:linux" in payload["issues"]
    profile = payload["safe_exe_preview_profile_validator"]["profiles"]["linux"]
    assert profile["file_exists"] is True
    assert profile["toml_valid"] is False


def test_blocked_path_escape_and_hidden_import_forbidden(monkeypatch) -> None:
    import scripts.safe_exe_preview_profile_validator as validator

    original = validator._validate_profile

    def fake_profile(platform: str, path: Path, root: Path):
        summary, issues = original(platform, path, root)
        if platform == "linux":
            summary["entrypoint_allowlisted"] = False
            summary["dist_dir_preview_scoped"] = False
            summary["work_dir_preview_scoped"] = False
            summary["briefcase_project_ok"] = False
            summary["briefcase_output_dir_preview_scoped"] = False
            summary["hidden_import_forbidden"] = True
            issues.extend(
                [
                    "preview_profile_entrypoint_invalid:linux",
                    "preview_profile_dist_dir_out_of_scope:linux",
                    "preview_profile_work_dir_out_of_scope:linux",
                    "preview_profile_briefcase_project_invalid:linux",
                    "preview_profile_briefcase_output_out_of_scope:linux",
                    "preview_profile_hidden_import_forbidden:linux",
                ]
            )
        return summary, issues

    monkeypatch.setattr(validator, "_validate_profile", fake_profile)
    payload = validator._build_payload("preview")
    assert payload["status"] == "blocked"
    assert "preview_profile_entrypoint_invalid:linux" in payload["issues"]
    assert "preview_profile_dist_dir_out_of_scope:linux" in payload["issues"]
    assert "preview_profile_work_dir_out_of_scope:linux" in payload["issues"]
    assert "preview_profile_briefcase_project_invalid:linux" in payload["issues"]
    assert "preview_profile_briefcase_output_out_of_scope:linux" in payload["issues"]
    assert "preview_profile_hidden_import_forbidden:linux" in payload["issues"]


def test_forbidden_token_issue_semantics(monkeypatch, tmp_path: Path) -> None:
    import scripts.safe_exe_preview_profile_validator as validator

    profile = tmp_path / "linux.toml"
    profile.write_text(
        """
platform = "linux"
[pyinstaller]
entrypoint = "../../../etc/passwd"
runtime_name = "dudzian-bot-preview"
dist_dir = "../../../tmp"
work_dir = "../../../tmp"
hidden_imports = ["bot_core.runtime.bootstrap", "bot_core.runtime.pipeline", "bot_core.runtime.config", "live.exchange"]
[briefcase]
project = "ui/briefcase"
app = "BotTradingShell"
output_dir = "dist/preview/briefcase/linux"
note = "api_key"
""",
        encoding="utf-8",
    )
    monkeypatch.setitem(validator.PROFILE_PATHS, "linux", str(profile))
    payload = validator._build_payload("preview")
    assert "preview_profile_forbidden_token_present:linux" in payload["issues"]
    assert "preview_profile_hidden_import_forbidden:linux" in payload["issues"]
