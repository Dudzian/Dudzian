from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path("scripts/preview_artifact_cache.py")


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


def _make_source(root: Path, executable: bool = True) -> Path:
    src = root / "src"
    src.mkdir()
    exe = src / "dudzian-bot-preview"
    exe.write_text("x", encoding="utf-8")
    if executable:
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
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


def _mk_complete(root: Path, name: str, executable: bool = True) -> Path:
    run = root / name
    exe = run / "dist/preview/linux/dudzian-bot-preview"
    exe.mkdir(parents=True)
    main = exe / "dudzian-bot-preview"
    main.write_text("x", encoding="utf-8")
    if executable:
        main.chmod(main.stat().st_mode | stat.S_IXUSR)
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
    src = _make_source(tmp_path, executable=False)
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
    _mk_complete(root, "run_bad", executable=False)
    code, payload = _run("--root", str(root), "--locate-latest", "--ttl-hours", "24")
    assert code == 2 and payload["cache_hit"] is False
    assert "executable_not_executable" in payload["missing_files"]


def test_invalid_stage_rejected(tmp_path: Path) -> None:
    src = _make_source(tmp_path)
    ev = _make_evidence(tmp_path)
    for bad in ("../evil", "foo/bar"):
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
