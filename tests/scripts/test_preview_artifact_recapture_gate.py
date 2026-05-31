from __future__ import annotations

import argparse
import json
import os
import pytest
import stat
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path("scripts/preview_artifact_recapture_gate.py")
REQUIRED_EVIDENCE = [
    "preview_artifact_seal.json",
    "preview_artifact_hashes.sha256",
    "main_executable.sha256",
    "leak_triage_summary.json",
    "leak_triage_summary.tsv",
]


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


def _run_loaded_gate(root: Path, ttl_hours: float = 24) -> tuple[int, dict[str, object]]:
    from scripts import preview_artifact_recapture_gate as gate

    args = argparse.Namespace(
        root=str(root),
        ttl_hours=ttl_hours,
        json=True,
        dry_run_cleanup=False,
    )
    payload = gate._base_payload(args)
    locate = gate._locate_latest(root, str(root), ttl_hours)
    for key in (
        "candidates",
        "stale_candidates",
        "incomplete_candidates",
        "selected_cache_dir",
        "selected_executable",
        "selected_evidence_dir",
        "missing_files",
        "cache_hit",
        "cache_complete",
        "cache_fresh",
    ):
        payload[key] = locate[key]
    payload["errors"] = locate["errors"]
    gate._finalize(payload)
    return (0 if payload["status"] == "ok" else 2), payload


def _host_main_binary_name() -> str:
    return "dudzian-bot-preview.exe" if os.name == "nt" else "dudzian-bot-preview"


def _mk_complete(
    root: Path,
    name: str,
    executable: bool = True,
    binary_name: str | None = None,
) -> Path:
    run = root / name
    dist = run / "dist/preview/linux/dudzian-bot-preview"
    dist.mkdir(parents=True)
    exe = dist / (binary_name or _host_main_binary_name())
    exe.write_text("exe", encoding="utf-8")
    if executable:
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    evidence = run / "evidence"
    evidence.mkdir(parents=True)
    for item in REQUIRED_EVIDENCE:
        (evidence / item).write_text("evidence", encoding="utf-8")
    return run


def _age(path: Path, hours: float) -> None:
    ts = time.time() - hours * 3600
    for item in sorted(path.rglob("*"), reverse=True):
        os.utime(item, (ts, ts), follow_symlinks=False)
    os.utime(path, (ts, ts), follow_symlinks=False)


def test_root_missing_controlled_rebuild_without_crash(tmp_path: Path) -> None:
    code, payload = _run("--root", str(tmp_path / "missing"), "--ttl-hours", "24")
    assert code == 2
    assert payload["status"] == "blocked"
    assert payload["decision"] == "CONTROLLED_REBUILD_REQUIRED"
    assert payload["cache_hit"] is False
    assert payload["errors"] == ["root_missing"]


def test_fresh_complete_cache_uses_cached_artifact(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    run = _mk_complete(root, "run_fresh")
    code, payload = _run("--root", str(root), "--ttl-hours", "24")
    assert code == 0
    assert payload["decision"] == "USE_CACHED_ARTIFACT"
    assert payload["cache_hit"] is True
    assert payload["cache_complete"] is True
    assert payload["cache_fresh"] is True
    assert payload["selected_cache_dir"] == str(run)
    assert payload["selected_executable"] == str(
        run / "dist/preview/linux/dudzian-bot-preview" / _host_main_binary_name()
    )
    assert payload["selected_evidence_dir"] == str(run / "evidence")
    assert payload["missing_files"] == []
    assert payload["required_files"][1] == (
        "dist/preview/linux/dudzian-bot-preview/<main_executable_candidate>"
    )
    assert str(run / "dist/preview/linux/dudzian-bot-preview" / _host_main_binary_name()).endswith(
        Path(str(payload["selected_executable"])).name
    )


def test_mk_complete_default_uses_host_compatible_main_binary(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    run = _mk_complete(root, "run_fresh")
    expected = run / "dist/preview/linux/dudzian-bot-preview" / _host_main_binary_name()

    assert expected.is_file()
    assert (expected.name.endswith(".exe")) is (os.name == "nt")

    code, payload = _run("--root", str(root), "--ttl-hours", "24")

    assert code == 0
    assert payload["decision"] == "USE_CACHED_ARTIFACT"
    assert payload["cache_hit"] is True
    assert payload["selected_executable"] == str(expected)


def test_stale_complete_cache_requires_rebuild_with_diagnostic(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    old = _mk_complete(root, "run_old")
    _age(old, 48)
    code, payload = _run("--root", str(root), "--ttl-hours", "24")
    assert code == 2
    assert payload["decision"] == "CONTROLLED_REBUILD_REQUIRED"
    assert payload["cache_hit"] is False
    assert payload["stale_candidates"]


def test_incomplete_cache_requires_rebuild_with_diagnostic(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    run = _mk_complete(root, "run_incomplete")
    (run / "evidence" / REQUIRED_EVIDENCE[0]).unlink()
    code, payload = _run("--root", str(root), "--ttl-hours", "24")
    assert code == 2
    assert payload["decision"] == "CONTROLLED_REBUILD_REQUIRED"
    assert payload["cache_hit"] is False
    assert payload["incomplete_candidates"]
    assert f"evidence/{REQUIRED_EVIDENCE[0]}" in payload["missing_files"]


def test_non_executable_executable_file_is_incomplete_candidate(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    _mk_complete(root, "run_bad_mode", executable=False, binary_name="dudzian-bot-preview")
    code, payload = _run("--root", str(root), "--ttl-hours", "24")
    assert code == 2
    assert payload["cache_hit"] is False
    assert payload["incomplete_candidates"]
    assert "executable_not_executable" in payload["missing_files"]


@pytest.mark.parametrize("suffix", (".exe", ".cmd", ".bat", ".ps1"))
def test_windows_recapture_gate_accepts_executable_suffix_without_posix_bit(
    tmp_path: Path, monkeypatch, suffix: str
) -> None:
    from scripts import preview_artifact_cache as preview_cache

    monkeypatch.setattr(preview_cache, "_is_windows_platform", lambda: True)
    root = tmp_path / "cache"
    run = _mk_complete(
        root, "run_fresh", executable=False, binary_name=f"dudzian-bot-preview{suffix}"
    )

    code, payload = _run_loaded_gate(root)

    assert code == 0
    assert payload["cache_hit"] is True
    assert payload["decision"] == "USE_CACHED_ARTIFACT"
    assert payload["selected_executable"] == str(
        run / "dist/preview/linux/dudzian-bot-preview" / f"dudzian-bot-preview{suffix}"
    )
    assert str(payload["selected_executable"]).endswith(f"dudzian-bot-preview{suffix}")
    assert (
        f"dist/preview/linux/dudzian-bot-preview/dudzian-bot-preview{suffix}"
        in payload["required_main_executable_candidates"]
    )


def test_windows_suffixless_only_main_binary_is_blocked(tmp_path: Path, monkeypatch) -> None:
    from scripts import preview_artifact_cache as preview_cache

    monkeypatch.setattr(preview_cache, "_is_windows_platform", lambda: True)
    root = tmp_path / "cache"
    _mk_complete(root, "run_suffixless", binary_name="dudzian-bot-preview")

    code, payload = _run_loaded_gate(root)

    assert code == 2
    assert payload["cache_hit"] is False
    assert payload["decision"] == "CONTROLLED_REBUILD_REQUIRED"
    assert "executable_not_executable" in payload["missing_files"]


def test_missing_main_executable_reports_candidate_lookup_diagnostic(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts import preview_artifact_cache as preview_cache

    monkeypatch.setattr(preview_cache, "_is_windows_platform", lambda: True)
    root = tmp_path / "cache"
    run = _mk_complete(root, "run_missing", binary_name="dudzian-bot-preview.exe")
    (run / "dist/preview/linux/dudzian-bot-preview/dudzian-bot-preview.exe").unlink()

    code, payload = _run_loaded_gate(root)

    assert code == 2
    assert payload["cache_hit"] is False
    assert payload["decision"] == "CONTROLLED_REBUILD_REQUIRED"
    assert payload["missing_files"] == ["main_executable_candidate_missing"]
    assert payload["required_files"][1] == (
        "dist/preview/linux/dudzian-bot-preview/<main_executable_candidate>"
    )
    assert (
        "dist/preview/linux/dudzian-bot-preview/dudzian-bot-preview.exe"
        in payload["required_main_executable_candidates"]
    )


def test_legacy_manifest_suffixless_main_requirement_is_logical_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts import preview_artifact_cache as preview_cache

    monkeypatch.setattr(preview_cache, "_is_windows_platform", lambda: True)
    root = tmp_path / "cache"
    run = _mk_complete(
        root, "run_manifest", executable=False, binary_name="dudzian-bot-preview.exe"
    )
    manifest = {
        "required_files": [
            "dist/preview/linux/dudzian-bot-preview",
            "dist/preview/linux/dudzian-bot-preview/dudzian-bot-preview",
            *[f"evidence/{name}" for name in REQUIRED_EVIDENCE],
        ]
    }
    (run / "cache_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    code, payload = _run_loaded_gate(root)

    assert code == 0
    assert payload["cache_hit"] is True
    assert payload["missing_files"] == []
    assert payload["selected_executable"].endswith("dudzian-bot-preview.exe")


def test_multiple_caches_selects_newest_fresh_complete(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    older = _mk_complete(root, "run_older")
    newer = _mk_complete(root, "run_newer")
    _age(older, 2)
    now = time.time()
    os.utime(newer, (now, now))
    code, payload = _run("--root", str(root), "--ttl-hours", "24")
    assert code == 0
    assert payload["decision"] == "USE_CACHED_ARTIFACT"
    assert payload["selected_cache_dir"] == str(newer)


def test_cleanup_removes_stale_cache_before_locate(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    stale = _mk_complete(root, "run_stale")
    fresh = _mk_complete(root, "run_fresh")
    _age(stale, 48)
    code, payload = _run("--root", str(root), "--ttl-hours", "24")
    assert code == 0
    assert payload["decision"] == "USE_CACHED_ARTIFACT"
    assert payload["selected_cache_dir"] == str(fresh)
    assert not stale.exists()
    assert payload["cleanup_summary"]["removed_count"] == 1


def test_dry_run_cleanup_reports_but_does_not_remove_stale_cache(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    stale = _mk_complete(root, "run_stale")
    _age(stale, 48)
    code, payload = _run("--root", str(root), "--ttl-hours", "24", "--dry-run-cleanup")
    assert code == 2
    assert stale.exists()
    assert payload["cleanup_dry_run"] is True
    assert payload["cleanup_summary"]["candidates"]
    assert payload["cleanup_summary"]["removed_count"] == 0
    assert payload["stale_candidates"]


def test_forbidden_artifact_inside_cache_blocks_cache_hit(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    run = _mk_complete(root, "run_forbidden")
    (run / "dist/preview/linux/dudzian-bot-preview" / ".env").write_text(
        "forbidden", encoding="utf-8"
    )
    code, payload = _run("--root", str(root), "--ttl-hours", "24")
    assert code == 2
    assert payload["decision"] == "BLOCKED_CACHE_ERROR"
    assert payload["cache_hit"] is False
    assert any("forbidden_artifact" in item for item in payload["missing_files"])


def test_symlink_out_of_root_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    run = _mk_complete(root, "run_symlink")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (run / "evidence" / "outside_link").symlink_to(outside)
    code, payload = _run("--root", str(root), "--ttl-hours", "24")
    assert code == 2
    assert payload["decision"] == "BLOCKED_CACHE_ERROR"
    assert payload["cache_hit"] is False
    assert any("symlink_outside_root" in item for item in payload["missing_files"])


def test_cp1252_safe_output(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    _mk_complete(root, "run_cp1252")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), "--json"],
        capture_output=True,
        check=False,
    )
    proc.stdout.decode("cp1252")


def test_source_safety_scan() -> None:
    src = SCRIPT.read_text(encoding="utf-8").lower()
    forbidden = [
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
    ]
    for token in forbidden:
        assert token not in src
