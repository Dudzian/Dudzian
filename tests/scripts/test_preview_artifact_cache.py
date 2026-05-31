from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path("scripts/preview_artifact_cache.py")


def _load_preview_artifact_cache_module():
    spec = importlib.util.spec_from_file_location("preview_artifact_cache", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*args: str) -> tuple[int, dict[str, object]]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--json"],
        capture_output=True,
        text=True,
        encoding="cp1252",
        errors="replace",
        check=False,
    )
    return proc.returncode, json.loads(proc.stdout.strip())


def _host_main_binary_name() -> str:
    return "dudzian-bot-preview.exe" if os.name == "nt" else "dudzian-bot-preview"


def _make_source(
    root: Path,
    executable: bool = True,
    binary_name: str | None = None,
) -> Path:
    src = root / "src"
    src.mkdir()
    exe = src / (binary_name or _host_main_binary_name())
    exe.write_text("x", encoding="utf-8")
    if executable:
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    else:
        exe.chmod(exe.stat().st_mode & ~0o111)
    return src


def _make_evidence(root: Path) -> Path:
    ev = root / "ev"
    ev.mkdir()
    for f in [
        "preview_artifact_seal.json",
        "preview_artifact_hashes.sha256",
        "main_executable.sha256",
        "leak_triage_summary.json",
        "leak_triage_summary.tsv",
    ]:
        (ev / f).write_text("x", encoding="utf-8")
    return ev


def _mk_complete(
    root: Path,
    name: str,
    executable: bool = True,
    binary_name: str | None = None,
) -> Path:
    run = root / name
    exe = run / "dist/preview/linux/dudzian-bot-preview"
    exe.mkdir(parents=True)
    main = exe / (binary_name or _host_main_binary_name())
    main.write_text("x", encoding="utf-8")
    if executable:
        main.chmod(main.stat().st_mode | stat.S_IXUSR)
    else:
        main.chmod(main.stat().st_mode & ~0o111)
    ev = run / "evidence"
    ev.mkdir(parents=True)
    for f in [
        "preview_artifact_seal.json",
        "preview_artifact_hashes.sha256",
        "main_executable.sha256",
        "leak_triage_summary.json",
        "leak_triage_summary.tsv",
    ]:
        (ev / f).write_text("x", encoding="utf-8")
    return run


def _module_args(**overrides: object) -> SimpleNamespace:
    values = {
        "root": "",
        "ttl_hours": 24,
        "stage": "",
        "source": "",
        "evidence_dir": "",
        "store": False,
        "locate_latest": False,
        "dry_run": False,
        "json": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _run_loaded_store(
    preview_cache,
    root: Path,
    src: Path,
    evidence: Path,
    stage: str = "EXE-PREVIEW-14_12",
) -> tuple[int, dict[str, object]]:
    args = _module_args(
        root=str(root),
        stage=stage,
        source=str(src),
        evidence_dir=str(evidence),
        store=True,
    )
    payload = preview_cache._base_payload(args)
    return preview_cache._do_store(args, payload), payload


def _run_loaded_locate(
    preview_cache, root: Path, ttl_hours: float = 24
) -> tuple[int, dict[str, object]]:
    args = _module_args(root=str(root), ttl_hours=ttl_hours, locate_latest=True)
    payload = preview_cache._base_payload(args)
    return preview_cache._do_locate(args, payload), payload


def test_source_env_blocks_store(tmp_path: Path) -> None:
    src = _make_source(tmp_path)
    (src / ".env").write_text("x", encoding="utf-8")
    ev = _make_evidence(tmp_path)
    code, payload = _run(
        "--root",
        str(tmp_path / "cache"),
        "--stage",
        "EXE-PREVIEW-14_12",
        "--source",
        str(src),
        "--evidence-dir",
        str(ev),
        "--store",
    )
    assert code == 2 and payload["status"] == "blocked"
    assert payload["forbidden_artifacts"]


def test_source_token_secret_keychain_block_store(tmp_path: Path) -> None:
    src = _make_source(tmp_path)
    (src / "my_token_file.txt").write_text("x", encoding="utf-8")
    ev = _make_evidence(tmp_path)
    code, payload = _run(
        "--root",
        str(tmp_path / "cache"),
        "--stage",
        "EXE-PREVIEW-14_12",
        "--source",
        str(src),
        "--evidence-dir",
        str(ev),
        "--store",
    )
    assert code == 2 and payload["status"] == "blocked"


def test_source_trading_db_blocks_store(tmp_path: Path) -> None:
    src = _make_source(tmp_path)
    (src / "trading.db").write_text("x", encoding="utf-8")
    ev = _make_evidence(tmp_path)
    code, payload = _run(
        "--root",
        str(tmp_path / "cache"),
        "--stage",
        "EXE-PREVIEW-14_12",
        "--source",
        str(src),
        "--evidence-dir",
        str(ev),
        "--store",
    )
    assert code == 2 and payload["status"] == "blocked"


def test_non_executable_main_binary_blocks_store(tmp_path: Path) -> None:
    src = _make_source(tmp_path, executable=False, binary_name="dudzian-bot-preview")
    ev = _make_evidence(tmp_path)
    code, payload = _run(
        "--root",
        str(tmp_path / "cache"),
        "--stage",
        "EXE-PREVIEW-14_12",
        "--source",
        str(src),
        "--evidence-dir",
        str(ev),
        "--store",
    )
    assert code == 2 and payload["status"] == "blocked"
    assert "executable_not_executable" in payload["missing_files"]


def test_locate_non_executable_cache_blocked(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    _mk_complete(root, "run_bad", executable=False, binary_name="dudzian-bot-preview")
    code, payload = _run("--root", str(root), "--locate-latest", "--ttl-hours", "24")
    assert code == 2 and payload["cache_hit"] is False
    assert "executable_not_executable" in payload["missing_files"]


def test_windows_executable_check_rejects_suffixless_preview_with_posix_bit(
    tmp_path: Path, monkeypatch
) -> None:
    preview_cache = _load_preview_artifact_cache_module()
    exe = tmp_path / "dudzian-bot-preview"
    exe.write_text("x", encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)

    monkeypatch.setattr(preview_cache, "_is_windows_platform", lambda: True)

    assert preview_cache._is_executable_file(exe) is False


def test_windows_executable_check_rejects_preview_without_posix_bit(
    tmp_path: Path, monkeypatch
) -> None:
    preview_cache = _load_preview_artifact_cache_module()
    exe = tmp_path / "dudzian-bot-preview"
    exe.write_text("x", encoding="utf-8")
    exe.chmod(exe.stat().st_mode & ~0o111)

    monkeypatch.setattr(preview_cache, "_is_windows_platform", lambda: True)

    assert preview_cache._is_executable_file(exe) is False


def test_windows_executable_check_accepts_windows_suffixes_without_posix_bit(
    tmp_path: Path, monkeypatch
) -> None:
    preview_cache = _load_preview_artifact_cache_module()
    monkeypatch.setattr(preview_cache, "_is_windows_platform", lambda: True)

    for suffix in (".exe", ".bat", ".cmd", ".ps1", ".EXE"):
        exe = tmp_path / f"dudzian-bot-preview{suffix}"
        exe.write_text("x", encoding="utf-8")
        exe.chmod(exe.stat().st_mode & ~0o111)

        assert preview_cache._is_executable_file(exe) is True


def test_posix_executable_check_requires_execute_bit(tmp_path: Path, monkeypatch) -> None:
    preview_cache = _load_preview_artifact_cache_module()
    exe = tmp_path / "dudzian-bot-preview"
    exe.write_text("x", encoding="utf-8")
    exe.chmod(exe.stat().st_mode & ~0o111)

    monkeypatch.setattr(preview_cache, "_is_windows_platform", lambda: False)

    assert preview_cache._is_executable_file(exe) is False


def test_invalid_stage_rejected(tmp_path: Path) -> None:
    src = _make_source(tmp_path)
    ev = _make_evidence(tmp_path)
    invalid_stages = (
        "",
        "../evil",
        "foo/bar",
        r"foo\bar",
        "/tmp/cache",
        "C:/tmp/cache",
        "evil.stage",
    )
    for bad in invalid_stages:
        code, payload = _run(
            "--root",
            str(tmp_path / "cache"),
            "--stage",
            bad,
            "--source",
            str(src),
            "--evidence-dir",
            str(ev),
            "--store",
        )
        assert code == 2 and payload["status"] == "error"
        assert payload["errors"] == ["invalid_stage"]


def test_valid_stage_accepted(tmp_path: Path) -> None:
    src = _make_source(tmp_path)
    ev = _make_evidence(tmp_path)
    code, payload = _run(
        "--root",
        str(tmp_path / "cache"),
        "--stage",
        "EXE-PREVIEW-14_12",
        "--source",
        str(src),
        "--evidence-dir",
        str(ev),
        "--store",
    )
    assert code == 0 and payload["cache_complete"] is True


def test_make_source_default_uses_host_compatible_store_binary(tmp_path: Path) -> None:
    src = _make_source(tmp_path)
    expected = src / _host_main_binary_name()

    assert expected.is_file()
    assert (expected.name.endswith(".exe")) is (os.name == "nt")

    ev = _make_evidence(tmp_path)
    code, payload = _run(
        "--root",
        str(tmp_path / "cache"),
        "--stage",
        "EXE-PREVIEW-14_12",
        "--source",
        str(src),
        "--evidence-dir",
        str(ev),
        "--store",
    )
    assert code == 0 and payload["cache_complete"] is True


def test_valid_stage_with_hyphen_accepted(tmp_path: Path) -> None:
    src = _make_source(tmp_path)
    ev = _make_evidence(tmp_path)
    code, payload = _run(
        "--root",
        str(tmp_path / "cache"),
        "--stage",
        "EXE-PREVIEW-14-12",
        "--source",
        str(src),
        "--evidence-dir",
        str(ev),
        "--store",
    )
    assert code == 0 and payload["cache_complete"] is True


@pytest.mark.parametrize("suffix", (".exe", ".cmd", ".bat", ".ps1"))
def test_windows_store_accepts_executable_suffix_candidates_without_posix_bit(
    tmp_path: Path, monkeypatch, suffix: str
) -> None:
    preview_cache = _load_preview_artifact_cache_module()
    monkeypatch.setattr(preview_cache, "_is_windows_platform", lambda: True)
    src = _make_source(tmp_path, executable=False, binary_name=f"dudzian-bot-preview{suffix}")
    ev = _make_evidence(tmp_path)

    code, payload = _run_loaded_store(preview_cache, tmp_path / "cache", src, ev)

    assert code == 0 and payload["cache_complete"] is True


def test_windows_store_prefers_exe_candidate_over_suffixless_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    preview_cache = _load_preview_artifact_cache_module()
    monkeypatch.setattr(preview_cache, "_is_windows_platform", lambda: True)
    src = _make_source(tmp_path, executable=False, binary_name="dudzian-bot-preview.exe")
    suffixless = src / "dudzian-bot-preview"
    suffixless.write_text("x", encoding="utf-8")
    suffixless.chmod(suffixless.stat().st_mode | stat.S_IXUSR)
    ev = _make_evidence(tmp_path)

    code, payload = _run_loaded_store(preview_cache, tmp_path / "cache", src, ev)

    assert code == 0 and payload["cache_complete"] is True


def test_windows_store_blocks_suffixless_candidate_with_posix_bit(
    tmp_path: Path, monkeypatch
) -> None:
    preview_cache = _load_preview_artifact_cache_module()
    monkeypatch.setattr(preview_cache, "_is_windows_platform", lambda: True)
    src = _make_source(tmp_path, executable=True, binary_name="dudzian-bot-preview")
    ev = _make_evidence(tmp_path)

    code, payload = _run_loaded_store(preview_cache, tmp_path / "cache", src, ev)

    assert code == 2 and payload["status"] == "blocked"
    assert "executable_not_executable" in payload["missing_files"]


def test_posix_store_accepts_suffixless_candidate_with_execute_bit(
    tmp_path: Path, monkeypatch
) -> None:
    preview_cache = _load_preview_artifact_cache_module()
    monkeypatch.setattr(preview_cache, "_is_windows_platform", lambda: False)
    src = _make_source(tmp_path, executable=True, binary_name="dudzian-bot-preview")
    ev = _make_evidence(tmp_path)

    code, payload = _run_loaded_store(preview_cache, tmp_path / "cache", src, ev)

    assert code == 0 and payload["cache_complete"] is True


def test_posix_store_blocks_suffixless_candidate_without_execute_bit(
    tmp_path: Path, monkeypatch
) -> None:
    preview_cache = _load_preview_artifact_cache_module()
    monkeypatch.setattr(preview_cache, "_is_windows_platform", lambda: False)
    src = _make_source(tmp_path, executable=False, binary_name="dudzian-bot-preview")
    ev = _make_evidence(tmp_path)

    code, payload = _run_loaded_store(preview_cache, tmp_path / "cache", src, ev)

    assert code == 2 and payload["status"] == "blocked"
    assert "executable_not_executable" in payload["missing_files"]


@pytest.mark.parametrize("suffix", (".exe", ".cmd", ".bat", ".ps1"))
def test_windows_locate_latest_accepts_executable_suffix_candidates_without_posix_bit(
    tmp_path: Path, monkeypatch, suffix: str
) -> None:
    preview_cache = _load_preview_artifact_cache_module()
    monkeypatch.setattr(preview_cache, "_is_windows_platform", lambda: True)
    root = tmp_path / "cache"
    run = _mk_complete(
        root, "run_good", executable=False, binary_name=f"dudzian-bot-preview{suffix}"
    )

    code, payload = _run_loaded_locate(preview_cache, root)

    assert code == 0 and payload["cache_hit"] is True
    assert payload["selected_cache_dir"] == str(run)


def test_windows_locate_latest_prefers_exe_candidate_over_suffixless_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    preview_cache = _load_preview_artifact_cache_module()
    monkeypatch.setattr(preview_cache, "_is_windows_platform", lambda: True)
    root = tmp_path / "cache"
    run = _mk_complete(root, "run_good", executable=False, binary_name="dudzian-bot-preview.exe")
    suffixless = run / "dist/preview/linux/dudzian-bot-preview/dudzian-bot-preview"
    suffixless.write_text("x", encoding="utf-8")
    suffixless.chmod(suffixless.stat().st_mode | stat.S_IXUSR)

    code, payload = _run_loaded_locate(preview_cache, root)

    assert code == 0 and payload["cache_hit"] is True
    assert payload["selected_cache_dir"] == str(run)


def test_windows_locate_latest_blocks_suffixless_candidate_with_posix_bit(
    tmp_path: Path, monkeypatch
) -> None:
    preview_cache = _load_preview_artifact_cache_module()
    monkeypatch.setattr(preview_cache, "_is_windows_platform", lambda: True)
    root = tmp_path / "cache"
    _mk_complete(root, "run_bad", executable=True, binary_name="dudzian-bot-preview")

    code, payload = _run_loaded_locate(preview_cache, root)

    assert code == 2 and payload["cache_hit"] is False
    assert "executable_not_executable" in payload["missing_files"]


def test_dry_run_forbidden_still_blocked(tmp_path: Path) -> None:
    src = _make_source(tmp_path)
    (src / ".env").write_text("x", encoding="utf-8")
    ev = _make_evidence(tmp_path)
    root = tmp_path / "cache"
    code, payload = _run(
        "--root",
        str(root),
        "--stage",
        "EXE-PREVIEW-14_12",
        "--source",
        str(src),
        "--evidence-dir",
        str(ev),
        "--store",
        "--dry-run",
    )
    assert code == 2 and payload["status"] == "blocked"
    assert not root.exists()


def test_stale_candidate_diagnostics(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    old = _mk_complete(root, "run_old")
    ts = time.time() - 48 * 3600
    os.utime(old, (ts, ts))
    code, payload = _run("--root", str(root), "--locate-latest", "--ttl-hours", "24")
    assert code == 2
    assert payload["stale_candidates"]


def test_evidence_secret_name_blocks_store(tmp_path: Path) -> None:
    src = _make_source(tmp_path)
    ev = tmp_path / "secret_evidence"
    ev.mkdir()
    for f in [
        "preview_artifact_seal.json",
        "preview_artifact_hashes.sha256",
        "main_executable.sha256",
        "leak_triage_summary.json",
        "leak_triage_summary.tsv",
    ]:
        (ev / f).write_text("x", encoding="utf-8")
    code, payload = _run(
        "--root",
        str(tmp_path / "cache"),
        "--stage",
        "EXE-PREVIEW-14_12",
        "--source",
        str(src),
        "--evidence-dir",
        str(ev),
        "--store",
    )
    assert code == 2 and payload["status"] == "blocked"
    assert "evidence_path_forbidden" in payload["errors"]


def test_evidence_token_name_blocks_store(tmp_path: Path) -> None:
    src = _make_source(tmp_path)
    ev = tmp_path / "token_cache"
    ev.mkdir()
    for f in [
        "preview_artifact_seal.json",
        "preview_artifact_hashes.sha256",
        "main_executable.sha256",
        "leak_triage_summary.json",
        "leak_triage_summary.tsv",
    ]:
        (ev / f).write_text("x", encoding="utf-8")
    code, payload = _run(
        "--root",
        str(tmp_path / "cache"),
        "--stage",
        "EXE-PREVIEW-14_12",
        "--source",
        str(src),
        "--evidence-dir",
        str(ev),
        "--store",
    )
    assert code == 2 and payload["status"] == "blocked"


def test_required_evidence_copied_selectively(tmp_path: Path) -> None:
    src = _make_source(tmp_path)
    ev = _make_evidence(tmp_path)
    (ev / "extra.txt").write_text("extra", encoding="utf-8")
    code, payload = _run(
        "--root",
        str(tmp_path / "cache"),
        "--stage",
        "EXE-PREVIEW-14_12",
        "--source",
        str(src),
        "--evidence-dir",
        str(ev),
        "--store",
    )
    assert code == 0
    selected = Path(payload["selected_cache_dir"])
    assert (selected / "evidence" / "preview_artifact_seal.json").is_file()
    assert not (selected / "evidence" / "extra.txt").exists()


def test_dry_run_forbidden_evidence_blocked_and_no_copy(tmp_path: Path) -> None:
    src = _make_source(tmp_path)
    ev = tmp_path / "secret_evidence"
    ev.mkdir()
    for f in [
        "preview_artifact_seal.json",
        "preview_artifact_hashes.sha256",
        "main_executable.sha256",
        "leak_triage_summary.json",
        "leak_triage_summary.tsv",
    ]:
        (ev / f).write_text("x", encoding="utf-8")
    root = tmp_path / "cache"
    code, payload = _run(
        "--root",
        str(root),
        "--stage",
        "EXE-PREVIEW-14_12",
        "--source",
        str(src),
        "--evidence-dir",
        str(ev),
        "--store",
        "--dry-run",
    )
    assert code == 2 and payload["status"] == "blocked"
    assert not root.exists()


def test_cp1252_safe_output(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path / "cache"), "--json"],
        capture_output=True,
        check=False,
    )
    proc.stdout.decode("cp1252")


def test_source_safety_scan() -> None:
    src = SCRIPT.read_text(encoding="utf-8").lower()
    for token in [
        "os.environ",
        "getenv",
        "dotenv",
        "requests",
        "httpx",
        "urllib",
        "import subprocess",
        "path.home",
        "open(",
        "ccxt",
        "create_order",
        "fetch_balance",
        "load_markets",
    ]:
        assert token not in src
