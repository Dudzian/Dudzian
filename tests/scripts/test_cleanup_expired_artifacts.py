from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path("scripts/cleanup_expired_artifacts.py")


def _run(*args: str) -> tuple[int, dict[str, object]]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--json"],
        capture_output=True,
        text=True,
        encoding="cp1252",
        errors="replace",
        check=False,
    )
    payload = json.loads(proc.stdout.strip())
    return proc.returncode, payload


def test_root_missing_ok(tmp_path: Path) -> None:
    code, payload = _run("--root", str(tmp_path / "missing"))
    assert code == 0
    assert payload["status"] == "ok"
    assert payload["removed_count"] == 0


def test_dry_run_does_not_delete_old_dir(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    run_dir = root / "old_run"
    run_dir.mkdir(parents=True)
    old = time.time() - (48 * 3600)
    os.utime(run_dir, (old, old))
    code, payload = _run("--root", str(root), "--ttl-hours", "24", "--dry-run")
    assert code == 0
    assert run_dir.exists()
    assert payload["candidates"]
    assert payload["removed_count"] == 0


def test_non_dry_run_deletes_old_dir(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    run_dir = root / "old_run"
    run_dir.mkdir(parents=True)
    old = time.time() - (48 * 3600)
    os.utime(run_dir, (old, old))
    code, payload = _run("--root", str(root), "--ttl-hours", "24")
    assert code == 0
    assert not run_dir.exists()
    assert payload["removed_count"] == 1


def test_fresh_dir_remains(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    run_dir = root / "fresh_run"
    run_dir.mkdir(parents=True)
    code, payload = _run("--root", str(root), "--ttl-hours", "24")
    assert code == 0
    assert run_dir.exists()
    assert payload["removed_count"] == 0


def test_outside_root_not_deleted(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    outside = tmp_path / "outside_old"
    outside.mkdir(parents=True)
    old = time.time() - (48 * 3600)
    os.utime(outside, (old, old))
    root.mkdir(parents=True)
    code, payload = _run("--root", str(root), "--ttl-hours", "24")
    assert code == 0
    assert outside.exists()
    assert payload["removed_count"] == 0


def test_invalid_ttl_rejected(tmp_path: Path) -> None:
    code, payload = _run("--root", str(tmp_path / "cache"), "--ttl-hours", "0")
    assert code == 2
    assert payload["status"] == "error"


def test_cp1252_safe_output(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir(parents=True)
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
        "keyring",
        "requests",
        "httpx",
        "urllib",
        "import subprocess",
        "path.home",
        "open(",
        "shell=true",
        "ccxt",
        "create_order",
        "fetch_balance",
        "load_markets",
    ]
    for token in forbidden:
        assert token not in src
