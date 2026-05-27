from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

SAFETY_CONTRACT_VERSION = "safe_exe_preview_command_renderer.v1"
ARTIFACT_EXCLUDE_POLICY_VERSION = "security_packaging_artifact_policy.v1"
VALID_MODES = ("preview",)
EXPECTED_PLATFORMS = ("linux", "macos", "windows")
PROFILE_PATHS = {
    "linux": "deploy/packaging/profiles/preview/linux.toml",
    "macos": "deploy/packaging/profiles/preview/macos.toml",
    "windows": "deploy/packaging/profiles/preview/windows.toml",
}
FORBIDDEN_TOKENS = (
    "live",
    "api_key",
    "api_secret",
    "secret",
    "token",
    "keychain",
    ".env",
    "trading.db",
    "/home/",
    "~",
    "&&",
    ";",
    "|",
    "`",
    "$(",
)
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


def _resolve_profile_path(path: Path, raw_value: str, repo_root: Path) -> tuple[str | None, bool]:
    try:
        resolved = (path.parent / raw_value.replace("\\", "/")).resolve()
        relative = resolved.relative_to(repo_root)
    except ValueError:
        return None, False
    return relative.as_posix(), True


def _forbidden_hit(command: list[str]) -> bool:
    lowered = " ".join(command).lower()
    return any(token in lowered for token in FORBIDDEN_TOKENS)


def _validate_command_shape(command: list[str]) -> bool:
    return isinstance(command, list) and all(isinstance(item, str) for item in command)


def _render_for_platform(platform: str, repo_root: Path) -> tuple[dict[str, object], list[str]]:
    issues: list[str] = []
    profile_path = Path(PROFILE_PATHS[platform])
    if not profile_path.is_absolute():
        profile_path = repo_root / profile_path
    summary: dict[str, object] = {
        "path": profile_path.as_posix(),
        "file_exists": profile_path.exists(),
    }
    if not profile_path.exists():
        issues.append(f"preview_command_profile_missing:{platform}")
        return summary, issues

    try:
        data = tomllib.loads(profile_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        issues.append(f"preview_command_profile_invalid_toml:{platform}")
        summary["toml_valid"] = False
        return summary, issues

    summary["toml_valid"] = True
    pyinstaller = data.get("pyinstaller", {})
    briefcase = data.get("briefcase", {})

    entrypoint_raw = str(pyinstaller.get("entrypoint", ""))
    dist_dir_raw = str(pyinstaller.get("dist_dir", ""))
    work_dir_raw = str(pyinstaller.get("work_dir", ""))
    briefcase_project_raw = str(briefcase.get("project", ""))
    briefcase_output_raw = str(briefcase.get("output_dir", ""))

    entrypoint, _ = _resolve_profile_path(profile_path, entrypoint_raw, repo_root)
    dist_dir, dist_ok = _resolve_profile_path(profile_path, dist_dir_raw, repo_root)
    work_dir, work_ok = _resolve_profile_path(profile_path, work_dir_raw, repo_root)
    briefcase_project, _ = _resolve_profile_path(profile_path, briefcase_project_raw, repo_root)
    briefcase_output, _ = _resolve_profile_path(profile_path, briefcase_output_raw, repo_root)

    pyinstaller_entrypoint = (
        "scripts/run_local_bot.py" if entrypoint == "scripts/run_local_bot.py" else ""
    )
    pyinstaller_command = (
        [
            "pyinstaller",
            "--noconfirm",
            "--name",
            str(pyinstaller.get("runtime_name", "dudzian-bot-preview")),
            "--distpath",
            dist_dir or "",
            "--workpath",
            work_dir or "",
            pyinstaller_entrypoint,
        ]
        if pyinstaller_entrypoint
        else []
    )
    briefcase_command = [
        "briefcase",
        "build",
        "--project",
        briefcase_project or "",
        "--app",
        str(briefcase.get("app", "BotTradingShell")),
        "--output",
        briefcase_output or "",
    ]

    entrypoint_ok = entrypoint == "scripts/run_local_bot.py"
    dist_scoped = bool(dist_ok and dist_dir) and dist_dir.startswith(f"dist/preview/{platform}")
    work_scoped = bool(work_ok and work_dir) and work_dir.startswith(
        f"var/build/preview/pyinstaller/{platform}"
    )

    if not entrypoint_ok:
        issues.append(f"preview_command_entrypoint_invalid:{platform}")
    if not dist_scoped:
        issues.append(f"preview_command_output_out_of_scope:{platform}")
    if not work_scoped:
        issues.append(f"preview_command_work_dir_out_of_scope:{platform}")

    if not _validate_command_shape(pyinstaller_command) or not _validate_command_shape(
        briefcase_command
    ):
        issues.append(f"preview_command_not_renderable:{platform}")

    no_forbidden_tokens = not (
        _forbidden_hit(pyinstaller_command) or _forbidden_hit(briefcase_command)
    )
    if not no_forbidden_tokens:
        issues.append(f"preview_command_forbidden_token_present:{platform}")

    pyinstaller_shape_ok = _validate_command_shape(pyinstaller_command)
    briefcase_shape_ok = _validate_command_shape(briefcase_command)
    command_renderable = all(
        (
            bool(summary.get("toml_valid")),
            entrypoint_ok,
            dist_scoped,
            work_scoped,
            pyinstaller_shape_ok,
            briefcase_shape_ok,
            no_forbidden_tokens,
        )
    )

    summary.update(
        {
            "entrypoint": entrypoint,
            "entrypoint_ok": entrypoint_ok,
            "dist_scoped": dist_scoped,
            "work_scoped": work_scoped,
            "allowed_default_args": ["--mode", "demo", "--preview-plan"],
            "pyinstaller_command_preview": pyinstaller_command,
            "briefcase_command_preview": briefcase_command,
            "pyinstaller_command_shape_ok": pyinstaller_shape_ok,
            "briefcase_command_shape_ok": briefcase_shape_ok,
            "no_forbidden_tokens": no_forbidden_tokens,
            "dist_dir": dist_dir,
            "work_dir": work_dir,
            "briefcase_output_dir": briefcase_output,
            "command_renderable": command_renderable,
        }
    )
    return summary, issues


def _build_payload(mode: str) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    per_platform: dict[str, dict[str, object]] = {}
    issues: list[str] = []

    for platform in EXPECTED_PLATFORMS:
        rendered, platform_issues = _render_for_platform(platform, repo_root)
        per_platform[platform] = rendered
        issues.extend(platform_issues)

    checks = {
        "all_profiles_present": all(per_platform[p].get("file_exists") for p in EXPECTED_PLATFORMS),
        "all_commands_rendered": all(
            isinstance(per_platform[p].get("pyinstaller_command_preview"), list)
            and isinstance(per_platform[p].get("briefcase_command_preview"), list)
            for p in EXPECTED_PLATFORMS
        ),
        "all_entrypoints_allowlisted": all(
            per_platform[p].get("entrypoint") == "scripts/run_local_bot.py"
            for p in EXPECTED_PLATFORMS
        ),
        "all_output_paths_preview_scoped": all(
            str(per_platform[p].get("dist_dir", "")).startswith(f"dist/preview/{p}")
            for p in EXPECTED_PLATFORMS
        ),
        "all_work_paths_preview_scoped": all(
            str(per_platform[p].get("work_dir", "")).startswith(
                f"var/build/preview/pyinstaller/{p}"
            )
            for p in EXPECTED_PLATFORMS
        ),
        "no_forbidden_tokens": not any(
            "preview_command_forbidden_token_present" in issue for issue in issues
        ),
    }

    status = "ok" if not issues else "blocked"
    renderer = {
        "platforms": per_platform,
        "command_render_only": True,
        "command_execution_allowed": False,
        "command_executed": False,
        "subprocess_invoked": False,
        "shell_used": False,
        "exe_build_performed": False,
        "installer_build_performed": False,
        "pyinstaller_build_performed": False,
        "briefcase_build_performed": False,
        "signing_performed": False,
        "codesign_performed": False,
        "notarization_performed": False,
        "release_upload_performed": False,
        "promotion_performed": False,
        "final_artifact_scan_performed": False,
        "final_hash_manifest_generated": False,
        "runtime_loop_started": False,
        "production_runtime_loop_started": False,
        "exchange_io": "disabled",
        "order_submission": "disabled",
        "real_orders_submitted": False,
        "api_keys_required": False,
        "secrets_read": False,
        "keychain_read": False,
        "env_values_read": False,
        "dot_env_read": False,
        "home_directory_scanned": False,
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

    return {
        "status": status,
        "mode": mode,
        "safe_exe_preview_command_renderer": renderer,
        "checks": checks,
        "issues": sorted(set(issues)),
        "safety_contract_version": SAFETY_CONTRACT_VERSION,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safe EXE preview command renderer")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--mode", choices=VALID_MODES, default="preview")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = _build_payload(args.mode)
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
