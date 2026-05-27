from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

SAFETY_CONTRACT_VERSION = "safe_exe_preview_profile_validator.v1"
ARTIFACT_EXCLUDE_POLICY_VERSION = "security_packaging_artifact_policy.v1"
VALID_MODES = ("preview",)
EXPECTED_PLATFORMS = ("linux", "macos", "windows")
PROFILE_PATHS = {
    "linux": "deploy/packaging/profiles/preview/linux.toml",
    "macos": "deploy/packaging/profiles/preview/macos.toml",
    "windows": "deploy/packaging/profiles/preview/windows.toml",
}
REQUIRED_HIDDEN_IMPORTS = {
    "bot_core.runtime.bootstrap",
    "bot_core.runtime.pipeline",
    "bot_core.runtime.config",
}
FORBIDDEN_TOKENS = [
    "--mode live",
    'mode = "live"',
    " live ",
    "live_mode",
    "live trading",
    ".env",
    "api_key",
    "api_secret",
    "secret",
    "token",
    "keychain",
    "path" + ".home",
    " home ",
    "create" + "_order",
    "fetch" + "_balance",
    "fetch" + "_ticker",
    "load" + "_markets",
    "pyinstaller ",
    "briefcase" + " build",
    "subprocess",
    "shell" + "=true",
]
DENIED_ARTIFACT_PATTERNS = [
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
HIDDEN_IMPORT_FORBIDDEN_TOKENS = [
    "live",
    "exchange",
    "api_key",
    "api_secret",
    "secret",
    "token",
    "keychain",
    ".env",
]


def _resolve_profile_path(path: Path, raw_value: str, repo_root: Path) -> tuple[str | None, bool]:
    try:
        resolved = (path.parent / raw_value.replace("\\", "/")).resolve()
        relative = resolved.relative_to(repo_root)
    except ValueError:
        return None, False
    return relative.as_posix(), True


def _validate_profile(
    platform: str, path: Path, repo_root: Path
) -> tuple[dict[str, object], list[str]]:
    issues: list[str] = []
    summary: dict[str, object] = {
        "path": path.as_posix(),
        "file_exists": path.exists(),
    }
    if not path.exists():
        issues.append(f"preview_profile_missing:{platform}")
        return summary, issues

    raw_text = path.read_text(encoding="utf-8")
    lowered = raw_text.lower()
    forbidden_hits = [token for token in FORBIDDEN_TOKENS if token in lowered]
    summary["toml_valid"] = True

    try:
        data = tomllib.loads(raw_text)
    except tomllib.TOMLDecodeError:
        summary["toml_valid"] = False
        summary["profile_forbidden_tokens_detected"] = forbidden_hits
        issues.append(f"preview_profile_invalid_toml:{platform}")
        return summary, issues

    pyinstaller = data.get("pyinstaller", {})
    briefcase = data.get("briefcase", {})

    summary["platform"] = data.get("platform")
    summary["platform_match"] = data.get("platform") == platform

    entrypoint_raw = str(pyinstaller.get("entrypoint", ""))
    dist_dir_raw = str(pyinstaller.get("dist_dir", ""))
    work_dir_raw = str(pyinstaller.get("work_dir", ""))
    briefcase_project_raw = str(briefcase.get("project", ""))
    briefcase_output_raw = str(briefcase.get("output_dir", ""))

    entrypoint, entrypoint_in_scope = _resolve_profile_path(path, entrypoint_raw, repo_root)
    dist_dir, dist_in_scope = _resolve_profile_path(path, dist_dir_raw, repo_root)
    work_dir, work_in_scope = _resolve_profile_path(path, work_dir_raw, repo_root)
    briefcase_project, project_in_scope = _resolve_profile_path(
        path, briefcase_project_raw, repo_root
    )
    briefcase_output_dir, output_in_scope = _resolve_profile_path(
        path, briefcase_output_raw, repo_root
    )

    hidden_imports_raw = pyinstaller.get("hidden_imports", [])
    hidden_imports = hidden_imports_raw if isinstance(hidden_imports_raw, list) else []
    hidden_imports_set = {str(item) for item in hidden_imports}
    hidden_imports_joined = " ".join(str(item).lower() for item in hidden_imports)
    hidden_import_has_forbidden = any(
        token in hidden_imports_joined for token in HIDDEN_IMPORT_FORBIDDEN_TOKENS
    )

    summary.update(
        {
            "entrypoint_resolved": entrypoint,
            "entrypoint_allowlisted": entrypoint == "scripts/run_local_bot.py",
            "runtime_name": pyinstaller.get("runtime_name"),
            "runtime_name_ok": pyinstaller.get("runtime_name") == "dudzian-bot-preview",
            "dist_dir_resolved": dist_dir,
            "dist_dir_preview_scoped": bool(dist_in_scope and dist_dir)
            and dist_dir.startswith(f"dist/preview/{platform}"),
            "work_dir_resolved": work_dir,
            "work_dir_preview_scoped": bool(work_in_scope and work_dir)
            and work_dir.startswith(f"var/build/preview/pyinstaller/{platform}"),
            "hidden_imports": hidden_imports,
            "hidden_imports_ok": REQUIRED_HIDDEN_IMPORTS.issubset(hidden_imports_set),
            "hidden_import_forbidden": hidden_import_has_forbidden,
            "briefcase_project_resolved": briefcase_project,
            "briefcase_project_ok": bool(project_in_scope) and briefcase_project == "ui/briefcase",
            "briefcase_app": briefcase.get("app"),
            "briefcase_app_ok": briefcase.get("app") == "BotTradingShell",
            "briefcase_output_dir_resolved": briefcase_output_dir,
            "briefcase_output_dir_preview_scoped": bool(output_in_scope and briefcase_output_dir)
            and briefcase_output_dir.startswith(f"dist/preview/briefcase/{platform}"),
            "profile_forbidden_tokens_detected": forbidden_hits,
        }
    )

    if not summary["platform_match"]:
        issues.append(f"preview_profile_platform_mismatch:{platform}")
    if not summary["entrypoint_allowlisted"]:
        issues.append(f"preview_profile_entrypoint_invalid:{platform}")
    if not summary["runtime_name_ok"]:
        issues.append(f"preview_profile_runtime_name_invalid:{platform}")
    if not summary["dist_dir_preview_scoped"]:
        issues.append(f"preview_profile_dist_dir_out_of_scope:{platform}")
    if not summary["work_dir_preview_scoped"]:
        issues.append(f"preview_profile_work_dir_out_of_scope:{platform}")
    if not summary["hidden_imports_ok"]:
        issues.append(f"preview_profile_hidden_imports_invalid:{platform}")
    if summary["hidden_import_forbidden"]:
        issues.append(f"preview_profile_hidden_import_forbidden:{platform}")
    if not summary["briefcase_project_ok"]:
        issues.append(f"preview_profile_briefcase_project_invalid:{platform}")
    if not summary["briefcase_app_ok"]:
        issues.append(f"preview_profile_briefcase_app_invalid:{platform}")
    if not summary["briefcase_output_dir_preview_scoped"]:
        issues.append(f"preview_profile_briefcase_output_out_of_scope:{platform}")
    if forbidden_hits:
        issues.append(f"preview_profile_forbidden_token_present:{platform}")

    return summary, issues


def _build_payload(mode: str) -> dict[str, object]:
    repo_root = Path.cwd().resolve()
    issues: list[str] = []
    profiles: dict[str, dict[str, object]] = {}

    for platform in EXPECTED_PLATFORMS:
        profile_path = Path(PROFILE_PATHS[platform])
        summary, profile_issues = _validate_profile(platform, profile_path, repo_root)
        profiles[platform] = summary
        issues.extend(profile_issues)

    all_profiles_exist = all(bool(profiles[p].get("file_exists")) for p in EXPECTED_PLATFORMS)
    all_platforms_match = all(bool(profiles[p].get("platform_match")) for p in EXPECTED_PLATFORMS)
    all_entrypoints_allowlisted = all(
        bool(profiles[p].get("entrypoint_allowlisted")) for p in EXPECTED_PLATFORMS
    )
    all_output_paths_preview_scoped = all(
        bool(profiles[p].get("dist_dir_preview_scoped"))
        and bool(profiles[p].get("briefcase_output_dir_preview_scoped"))
        for p in EXPECTED_PLATFORMS
    )
    all_work_paths_preview_scoped = all(
        bool(profiles[p].get("work_dir_preview_scoped")) for p in EXPECTED_PLATFORMS
    )
    all_toml_valid = all(bool(profiles[p].get("toml_valid")) for p in EXPECTED_PLATFORMS)
    all_runtime_names_ok = all(bool(profiles[p].get("runtime_name_ok")) for p in EXPECTED_PLATFORMS)
    all_hidden_imports_ok = all(
        bool(profiles[p].get("hidden_imports_ok")) for p in EXPECTED_PLATFORMS
    )
    no_hidden_import_forbidden = all(
        not bool(profiles[p].get("hidden_import_forbidden")) for p in EXPECTED_PLATFORMS
    )
    all_briefcase_projects_ok = all(
        bool(profiles[p].get("briefcase_project_ok")) for p in EXPECTED_PLATFORMS
    )
    all_briefcase_apps_ok = all(
        bool(profiles[p].get("briefcase_app_ok")) for p in EXPECTED_PLATFORMS
    )
    all_briefcase_outputs_preview_scoped = all(
        bool(profiles[p].get("briefcase_output_dir_preview_scoped")) for p in EXPECTED_PLATFORMS
    )
    forbidden_tokens_present = any(
        bool(profiles[p].get("profile_forbidden_tokens_detected")) for p in EXPECTED_PLATFORMS
    )
    no_profile_forbidden_tokens = not forbidden_tokens_present

    validator = {
        "profiles_checked": True,
        "profile_count": len(EXPECTED_PLATFORMS),
        "expected_platforms": list(EXPECTED_PLATFORMS),
        "profile_paths": PROFILE_PATHS,
        "profiles": profiles,
        "all_profiles_exist": all_profiles_exist,
        "all_platforms_match": all_platforms_match,
        "all_entrypoints_allowlisted": all_entrypoints_allowlisted,
        "all_output_paths_preview_scoped": all_output_paths_preview_scoped,
        "all_work_paths_preview_scoped": all_work_paths_preview_scoped,
        "all_toml_valid": all_toml_valid,
        "all_runtime_names_ok": all_runtime_names_ok,
        "all_hidden_imports_ok": all_hidden_imports_ok,
        "no_hidden_import_forbidden": no_hidden_import_forbidden,
        "all_briefcase_projects_ok": all_briefcase_projects_ok,
        "all_briefcase_apps_ok": all_briefcase_apps_ok,
        "all_briefcase_outputs_preview_scoped": all_briefcase_outputs_preview_scoped,
        "forbidden_tokens_present": forbidden_tokens_present,
        "no_profile_forbidden_tokens": no_profile_forbidden_tokens,
        "build_not_performed": True,
        "pyinstaller_build_performed": False,
        "briefcase_build_performed": False,
        "installer_build_performed": False,
        "signing_performed": False,
        "release_upload_performed": False,
        "promotion_performed": False,
        "secrets_read": False,
        "keychain_read": False,
        "env_values_read": False,
        "dot_env_read": False,
        "home_directory_scanned": False,
        "exchange_io": "disabled",
        "order_submission": "disabled",
        "runtime_loop_started": False,
        "production_runtime_loop_started": False,
        "artifact_policy_checked": True,
        "artifact_exclude_policy_version": ARTIFACT_EXCLUDE_POLICY_VERSION,
        "denied_artifact_patterns": DENIED_ARTIFACT_PATTERNS,
        "env_file_bundled": False,
        "local_db_bundled": False,
        "logs_bundled": False,
        "reports_bundled": False,
        "tmp_artifacts_bundled": False,
        "test_secrets_bundled": False,
        "cache_artifacts_bundled": False,
        "local_user_data_bundled": False,
        "keychain_artifacts_bundled": False,
    }

    checks = {
        "profiles_checked": True,
        "all_profiles_exist": all_profiles_exist,
        "all_platforms_match": all_platforms_match,
        "all_entrypoints_allowlisted": all_entrypoints_allowlisted,
        "all_output_paths_preview_scoped": all_output_paths_preview_scoped,
        "all_work_paths_preview_scoped": all_work_paths_preview_scoped,
        "forbidden_tokens_present": forbidden_tokens_present,
    }

    status = "ok"
    if (
        issues
        or forbidden_tokens_present
        or not all(
            [
                all_profiles_exist,
                all_platforms_match,
                all_entrypoints_allowlisted,
                all_output_paths_preview_scoped,
                all_work_paths_preview_scoped,
            ]
        )
    ):
        status = "blocked"

    return {
        "status": status,
        "mode": mode,
        "safety_contract_version": SAFETY_CONTRACT_VERSION,
        "safe_exe_preview_profile_validator": validator,
        "checks": checks,
        "issues": sorted(set(issues)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safe EXE preview profile validator")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--mode", choices=VALID_MODES, default="preview")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = _build_payload(args.mode)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
