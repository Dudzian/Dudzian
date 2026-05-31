from __future__ import annotations

import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path("scripts/preview_artifact_launch_plan.py")
CACHE_SCRIPT = Path("scripts/preview_artifact_cache.py")
REQUIRED_EVIDENCE = (
    "preview_artifact_seal.json",
    "preview_artifact_hashes.sha256",
    "main_executable.sha256",
    "leak_triage_summary.json",
    "leak_triage_summary.tsv",
)


def _repo_tmp_root(tmp_path: Path) -> Path:
    root = Path("var/tmp") / f"preview_launch_plan_{tmp_path.name}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    return root


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_cache_as_importable(monkeypatch: pytest.MonkeyPatch):
    cache_module = _load_module(CACHE_SCRIPT, "preview_artifact_cache")
    monkeypatch.setitem(sys.modules, "preview_artifact_cache", cache_module)
    return cache_module


def _run(*args: str) -> tuple[int, dict[str, object], str]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--json"],
        capture_output=True,
        text=True,
        encoding="cp1252",
        errors="replace",
        check=False,
    )
    return proc.returncode, json.loads(proc.stdout.strip()), proc.stdout


def _host_main_binary_name() -> str:
    return "dudzian-bot-preview.exe" if os.name == "nt" else "dudzian-bot-preview"


def _make_artifact(
    root: Path,
    binary_name: str | None = None,
    executable: bool = True,
    evidence: bool = True,
) -> Path:
    artifact_root = root / "run_ok"
    binary_dir = artifact_root / "dist/preview/linux/dudzian-bot-preview"
    binary_dir.mkdir(parents=True)
    binary = binary_dir / (binary_name or _host_main_binary_name())
    binary.write_text("x", encoding="utf-8")
    if executable:
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    else:
        binary.chmod(binary.stat().st_mode & ~0o111)
    if evidence:
        _make_evidence(artifact_root)
    return binary


def _make_evidence(artifact_root: Path) -> None:
    evidence_dir = artifact_root / "evidence"
    evidence_dir.mkdir(parents=True)
    for name in REQUIRED_EVIDENCE:
        (evidence_dir / name).write_text("x", encoding="utf-8")


def test_happy_path_with_evidence_builds_demo_launch_plan(tmp_path: Path) -> None:
    root = _repo_tmp_root(tmp_path)
    try:
        binary = _make_artifact(root, evidence=True)

        code, payload, _stdout = _run("--root", str(root))

        assert code == 0
        assert payload["status"] == "ok"
        assert payload["safety_contract_version"] == "preview_artifact_launch_plan.v1"
        assert payload["artifact_found"] is True
        assert payload["executable_valid"] is True
        assert payload["evidence_required"] is True
        assert payload["evidence_present"] is True
        assert payload["seal_evidence_present"] is True
        assert payload["hash_evidence_present"] is True
        assert payload["leak_triage_evidence_present"] is True
        assert payload["artifact_verified"] is True
        assert payload["launch_plan_ready"] is True
        assert payload["selected_executable"] == str(binary.resolve(strict=False))
        assert Path(str(payload["selected_executable"])).is_file()
        assert payload["launch_command_preview"] == [
            str(binary.resolve(strict=False)),
            "--mode",
            "demo",
            "--preview-plan",
        ]
        assert all(isinstance(part, str) for part in payload["launch_command_preview"])
        assert payload["command_execution_allowed"] is False
        assert payload["command_executed"] is False
        assert payload["live_mode_allowed"] is False
        assert payload["subprocess_invoked"] is False
        assert payload["shell_used"] is False
        assert payload["exchange_io"] == "disabled"
        assert payload["order_submission"] == "disabled"
        assert payload["runtime_loop_started"] is False
        assert payload["production_runtime_loop_started"] is False
        assert payload["secrets_read"] is False
        assert payload["keychain_read"] is False
        assert payload["env_values_read"] is False
        assert payload["dot_env_read"] is False
        assert payload["home_directory_scanned"] is False
        assert payload["artifact_policy_checked"] is True
        assert payload["artifact_exclude_policy_version"] == (
            "security_packaging_artifact_policy.v1"
        )
        for pattern in (
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
        ):
            assert pattern in payload["denied_artifact_patterns"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_executable_without_evidence_blocks_launch_plan(tmp_path: Path) -> None:
    root = _repo_tmp_root(tmp_path)
    try:
        _make_artifact(root, evidence=False)

        code, payload, _stdout = _run("--root", str(root))

        assert code == 2
        assert payload["status"] == "blocked"
        assert payload["issues"] == ["preview_artifact_evidence_missing"]
        assert payload["artifact_found"] is True
        assert payload["executable_valid"] is True
        assert payload["evidence_required"] is True
        assert payload["evidence_present"] is False
        assert payload["artifact_verified"] is False
        assert payload["launch_plan_ready"] is False
        assert payload["launch_command_preview"] == []
        assert payload["command_execution_allowed"] is False
        assert payload["command_executed"] is False
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_root_out_of_scope_blocks_without_scanning(tmp_path: Path) -> None:
    _make_artifact(tmp_path, evidence=True)

    code, payload, _stdout = _run("--root", str(tmp_path))

    assert code == 2
    assert payload["status"] == "blocked"
    assert payload["issues"] == ["preview_artifact_root_out_of_scope"]
    assert payload["artifact_found"] is False
    assert payload["executable_valid"] is False
    assert payload["artifact_verified"] is False
    assert payload["launch_plan_ready"] is False
    assert payload["launch_command_preview"] == []
    assert payload["home_directory_scanned"] is False
    assert payload["command_execution_allowed"] is False
    assert payload["command_executed"] is False


@pytest.mark.parametrize("make_dist", (False, True))
def test_missing_artifact_blocks_without_launch_command(tmp_path: Path, make_dist: bool) -> None:
    root = _repo_tmp_root(tmp_path)
    try:
        if make_dist:
            (root / "dist/preview/linux/dudzian-bot-preview").mkdir(parents=True)

        code, payload, _stdout = _run("--root", str(root))

        assert code == 2
        assert payload["status"] == "blocked"
        assert payload["issues"] in (
            ["preview_artifact_missing"],
            ["preview_artifact_executable_invalid"],
        )
        assert payload["artifact_found"] is False
        assert payload["executable_valid"] is False
        assert payload["artifact_verified"] is False
        assert payload["launch_plan_ready"] is False
        assert payload["launch_command_preview"] == []
        assert payload["command_execution_allowed"] is False
        assert payload["command_executed"] is False
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize("suffix", (".exe", ".cmd", ".bat", ".ps1"))
def test_windows_candidate_suffixes_are_ready_only_with_complete_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    preview_cache = _load_cache_as_importable(monkeypatch)
    monkeypatch.setattr(preview_cache, "_is_windows_platform", lambda: True)
    launch_plan = _load_module(SCRIPT, f"preview_artifact_launch_plan_windows_accept_{suffix[1:]}")
    root = _repo_tmp_root(tmp_path)
    try:
        binary = _make_artifact(
            root,
            binary_name=f"dudzian-bot-preview{suffix}",
            executable=False,
            evidence=True,
        )

        payload = launch_plan._build_payload(str(root))

        assert payload["status"] == "ok"
        assert payload["selected_executable"] == str(binary.resolve(strict=False))
        assert payload["artifact_found"] is True
        assert payload["executable_valid"] is True
        assert payload["evidence_present"] is True
        assert payload["artifact_verified"] is True
        assert payload["launch_plan_ready"] is True
        assert payload["launch_command_preview"] == [
            str(binary.resolve(strict=False)),
            "--mode",
            "demo",
            "--preview-plan",
        ]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_windows_suffixless_candidate_is_blocked_in_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview_cache = _load_cache_as_importable(monkeypatch)
    monkeypatch.setattr(preview_cache, "_is_windows_platform", lambda: True)
    launch_plan = _load_module(SCRIPT, "preview_artifact_launch_plan_windows_block")
    root = _repo_tmp_root(tmp_path)
    try:
        _make_artifact(
            root,
            binary_name="dudzian-bot-preview",
            executable=True,
            evidence=True,
        )

        payload = launch_plan._build_payload(str(root))

        assert payload["status"] == "blocked"
        assert payload["issues"] == ["preview_artifact_executable_invalid"]
        assert payload["artifact_found"] is False
        assert payload["executable_valid"] is False
        assert payload["artifact_verified"] is False
        assert payload["launch_plan_ready"] is False
        assert payload["launch_command_preview"] == []
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_source_safety_contract_terms_are_absent() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for forbidden in (
        "shell=True",
        "subprocess.run",
        "os.environ",
        "getenv",
        "dotenv",
        "keyring",
        "ccxt",
        "create_order",
        "fetch_balance",
        "load_markets",
        "pyinstaller.__main__",
        "briefcase build",
        "open(",
    ):
        assert forbidden not in source


def test_stdout_is_cp1252_encodable(tmp_path: Path) -> None:
    root = _repo_tmp_root(tmp_path)
    try:
        _make_artifact(root, evidence=True)

        code, _payload, stdout = _run("--root", str(root))

        assert code == 0
        stdout.encode("cp1252")
    finally:
        shutil.rmtree(root, ignore_errors=True)
