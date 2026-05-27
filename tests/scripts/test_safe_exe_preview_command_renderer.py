from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/safe_exe_preview_command_renderer.py")
FORBIDDEN_TOKENS = [
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
]


def _run(*extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--json", *extra],
        check=False,
        capture_output=True,
        text=True,
    )


def _payload() -> dict[str, object]:
    result = _run()
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_happy_path_flags_and_platforms() -> None:
    payload = _payload()
    renderer = payload["safe_exe_preview_command_renderer"]

    assert payload["status"] == "ok"
    assert payload["safety_contract_version"] == "safe_exe_preview_command_renderer.v1"
    assert payload["mode"] == "preview"
    assert renderer["command_render_only"] is True
    assert renderer["command_execution_allowed"] is False
    assert renderer["command_executed"] is False
    assert renderer["subprocess_invoked"] is False
    assert renderer["shell_used"] is False
    assert renderer["exe_build_performed"] is False
    assert renderer["installer_build_performed"] is False
    assert renderer["pyinstaller_build_performed"] is False
    assert renderer["briefcase_build_performed"] is False
    assert renderer["runtime_loop_started"] is False
    assert renderer["production_runtime_loop_started"] is False
    assert renderer["exchange_io"] == "disabled"
    assert renderer["order_submission"] == "disabled"
    assert set(renderer["platforms"].keys()) == {"linux", "macos", "windows"}
    for rendered in renderer["platforms"].values():
        assert rendered["command_renderable"] is True


def test_command_shape_and_scope() -> None:
    payload = _payload()
    platforms = payload["safe_exe_preview_command_renderer"]["platforms"]

    for platform, rendered in platforms.items():
        for key in ("pyinstaller_command_preview", "briefcase_command_preview"):
            command = rendered[key]
            assert isinstance(command, list)
            assert all(isinstance(item, str) for item in command)
            lowered = " ".join(command).lower()
            for token in FORBIDDEN_TOKENS:
                assert token not in lowered

        pyinstaller_cmd = rendered["pyinstaller_command_preview"]
        assert pyinstaller_cmd[-1] == "scripts/run_local_bot.py"
        assert rendered["entrypoint"] == "scripts/run_local_bot.py"
        assert rendered["allowed_default_args"] == ["--mode", "demo", "--preview-plan"]
        assert str(rendered["dist_dir"]).startswith(f"dist/preview/{platform}")
        assert str(rendered["work_dir"]).startswith(f"var/build/preview/pyinstaller/{platform}")


def test_artifact_policy_present() -> None:
    payload = _payload()
    renderer = payload["safe_exe_preview_command_renderer"]

    assert renderer["artifact_policy_checked"] is True
    assert renderer["artifact_exclude_policy_version"] == "security_packaging_artifact_policy.v1"
    deny = renderer["denied_artifact_patterns"]
    assert ".env" in deny
    assert "trading.db" in deny
    assert renderer["env_file_bundled"] is False
    assert renderer["local_db_bundled"] is False
    assert renderer["logs_bundled"] is False
    assert renderer["reports_bundled"] is False
    assert renderer["tmp_artifacts_bundled"] is False
    assert renderer["test_secrets_bundled"] is False
    assert renderer["cache_artifacts_bundled"] is False
    assert renderer["local_user_data_bundled"] is False
    assert renderer["keychain_artifacts_bundled"] is False


def test_invalid_mode_rejected() -> None:
    result = _run("--mode", "live")
    assert result.returncode != 0


def test_source_safety_no_risky_tokens() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
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
        "shell=True",
        "subprocess.run",
        "pyinstaller.__main__",
        "write_text",
        "write_bytes",
        "open(",
    ]
    for token in forbidden:
        assert token not in source


def test_profile_failure_cases(monkeypatch, tmp_path) -> None:
    import scripts.safe_exe_preview_command_renderer as renderer

    base = tmp_path / "deploy/packaging/profiles/preview"
    base.mkdir(parents=True)
    good = Path("deploy/packaging/profiles/preview/linux.toml").read_text(encoding="utf-8")
    (base / "linux.toml").write_text(good, encoding="utf-8")
    (base / "windows.toml").write_text("platform = 'windows'\n[pyinstaller", encoding="utf-8")

    monkeypatch.setattr(
        renderer,
        "PROFILE_PATHS",
        {
            "linux": str(base / "linux.toml"),
            "macos": str(base / "macos.toml"),
            "windows": str(base / "windows.toml"),
        },
    )

    payload = renderer._build_payload("preview")
    assert payload["status"] == "blocked"
    issues = payload["issues"]
    assert "preview_command_profile_missing:macos" in issues
    assert "preview_command_profile_invalid_toml:windows" in issues

    bad = """
platform = "linux"
[pyinstaller]
entrypoint = "../../../../scripts/not_allowed.py"
runtime_name = "secret-demo"
dist_dir = "../../../../dist/preview/linux"
work_dir = "../../../../var/build/preview/pyinstaller/linux"
[briefcase]
project = "../../../../ui/briefcase"
app = "BotTradingShell"
output_dir = "../../../../dist/preview/briefcase/linux"
"""
    (base / "linux.toml").write_text(
        bad.replace("dist/preview/linux", "dist/live/linux"), encoding="utf-8"
    )
    monkeypatch.setattr(
        renderer,
        "PROFILE_PATHS",
        {
            "linux": str(base / "linux.toml"),
            "macos": str(base / "linux.toml"),
            "windows": str(base / "linux.toml"),
        },
    )
    payload2 = renderer._build_payload("preview")
    assert "preview_command_output_out_of_scope:linux" in payload2["issues"]
    assert "preview_command_entrypoint_invalid:linux" in payload2["issues"]
    platform_linux = payload2["safe_exe_preview_command_renderer"]["platforms"]["linux"]
    assert platform_linux["command_renderable"] is False
    assert platform_linux["pyinstaller_command_preview"] == []


def test_repo_root_resolution_not_from_cwd(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT.resolve()), "--json"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    for rendered in payload["safe_exe_preview_command_renderer"]["platforms"].values():
        assert rendered["command_renderable"] is True
