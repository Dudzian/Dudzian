"""Live Windows Stage-6 process-lock and SQLite crash probe.

This module deliberately uses subprocesses and the production implementations.  Its
worker entry points are private implementation details; only :func:`run_probe` may
produce acceptance results.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
from typing import Any

WINDOWS_STAGE6_ITEMS = ("WINDOWS_FILE_LOCKING", "WINDOWS_SQLITE_CRASH_INTEGRITY")
WORKER_TIMEOUT_SECONDS = 20.0
PROCESS_EXIT_TIMEOUT_SECONDS = 10.0

ACCOUNT_ID = "acct_01890f4c-7b9a-7cc1-8a2b-123456789abc"
DEVICE_ID = "dev_01890f4c-7b9a-7cc1-8a2b-123456789abc"
STAGE6_STATE_STORE_IDENTITY_FINGERPRINT_SHA256 = (
    "7d3346f76f18052e314232add85c79db1b397cf0ba514cc681ea2697d6c8322f"
)


class Stage6ProbeError(RuntimeError):
    """A labelled, fail-closed Stage-6 failure."""

    def __init__(self, label: str, item: str, detail: str) -> None:
        self.label = label
        self.item = item
        self.detail = detail
        super().__init__(f"[{label}] {detail}")


def _scope(path: Path) -> Any:
    from bot_core.runtime.core_host import CoreHostScope

    return CoreHostScope(ACCOUNT_ID, DEVICE_ID, path)


def _candidate_metadata(_path: Path, generation: int) -> Any:
    from bot_core.persistence.state_store import StateStoreMetadata

    return StateStoreMetadata(
        account_id=ACCOUNT_ID,
        device_installation_id=DEVICE_ID,
        state_store_schema_version=1,
        state_store_identity_fingerprint_sha256=STAGE6_STATE_STORE_IDENTITY_FINGERPRINT_SHA256,
        environment="PAPER",
        protected_freshness_generation=generation,
        # These caller placeholders are replaced by derive_prepared_metadata.
        state_fingerprint_sha256="0" * 64,
        transaction_fingerprint_sha256="0" * 64,
        history_tail_fingerprint_sha256="0" * 64,
    )


def _commit_generation(store: Any, path: Path, generation: int) -> None:
    expected = None if generation == 1 else generation - 1
    prepared = store.derive_prepared_metadata(
        _candidate_metadata(path, generation), expected_current_generation=expected
    )
    store.commit_prepared_metadata(prepared, expected_current_generation=expected)


def _worker_lock_holder(path: Path) -> int:
    from bot_core.runtime.core_host import CoreHostProcessLock

    lock = CoreHostProcessLock(_scope(path))
    lock.acquire()
    print("LOCK_HELD", flush=True)
    threading.Event().wait()
    return 70  # pragma: no cover - the acceptance parent forcibly terminates this worker


def _worker_lock_contender(path: Path) -> int:
    from bot_core.runtime.core_host import CoreHostAlreadyRunningError, CoreHostProcessLock

    lock = CoreHostProcessLock(_scope(path))
    try:
        lock.acquire()
    except CoreHostAlreadyRunningError as exc:
        print(f"LOCK_CONFLICT\t{type(exc).__name__}\t{exc}", flush=True)
        return 0
    print("PROTECTED_SECTION_ENTERED", flush=True)
    lock.release()
    return 71


def _worker_lock_reacquire(path: Path) -> int:
    from bot_core.runtime.core_host import CoreHostProcessLock

    lock = CoreHostProcessLock(_scope(path))
    lock.acquire()
    print("LOCK_REACQUIRED", flush=True)
    lock.release()
    return 0


def _worker_sqlite_uncommitted(path: Path) -> int:
    connection = sqlite3.connect(path, isolation_level=None)
    mode = connection.execute("PRAGMA journal_mode").fetchone()
    if mode is None or str(mode[0]).lower() != "wal":
        raise RuntimeError("existing StateStore is not in WAL mode")
    connection.execute("BEGIN IMMEDIATE")
    cursor = connection.execute(
        "UPDATE state_store_metadata SET environment='TESTNET' WHERE singleton_key=1"
    )
    if cursor.rowcount != 1:
        raise RuntimeError("uncommitted deterministic write did not update metadata")
    print("UNCOMMITTED_WRITE_HELD", flush=True)
    threading.Event().wait()
    return 72  # pragma: no cover


def _worker_sqlite_committed(path: Path) -> int:
    from bot_core.persistence.state_store import SQLiteStateStore

    store = SQLiteStateStore(path)
    _commit_generation(store, path, 2)
    snapshot = store.read_verified_snapshot()
    if snapshot is None or snapshot.metadata.protected_freshness_generation != 2:
        raise RuntimeError("generation 2 was not verified after public commit")
    print(
        "COMMITTED_STATE_VERIFIED\t"
        + json.dumps(snapshot.metadata.to_mapping(), sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    threading.Event().wait()
    return 73  # pragma: no cover


def _command(worker: str, path: Path) -> list[str]:
    return [sys.executable, "-u", "-m", __name__, "--worker", worker, "--path", str(path)]


def _start(worker: str, path: Path) -> subprocess.Popen[str]:
    return subprocess.Popen(
        _command(worker, path),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _read_marker(process: subprocess.Popen[str], expected: str) -> str:
    if process.stdout is None:
        raise RuntimeError("worker stdout pipe is unavailable")
    lines: queue.Queue[str] = queue.Queue(maxsize=1)

    def read() -> None:
        lines.put(process.stdout.readline())

    threading.Thread(target=read, daemon=True).start()
    try:
        line = lines.get(timeout=WORKER_TIMEOUT_SECONDS).rstrip("\r\n")
    except queue.Empty as exc:
        raise TimeoutError(f"worker did not emit {expected} before deadline") from exc
    if not line.startswith(expected):
        stderr = process.stderr.read() if process.poll() is not None and process.stderr else ""
        raise RuntimeError(f"expected {expected!r}, received {line!r}; stderr={stderr!r}")
    return line


def _kill_and_wait(process: subprocess.Popen[str]) -> None:
    process.kill()  # Windows Popen.kill() calls TerminateProcess; no graceful cleanup runs.
    process.wait(timeout=PROCESS_EXIT_TIMEOUT_SECONDS)


def _finish(process: subprocess.Popen[str], expected: str) -> str:
    stdout, stderr = process.communicate(timeout=WORKER_TIMEOUT_SECONDS)
    line = stdout.strip()
    if process.returncode != 0 or not line.startswith(expected):
        raise RuntimeError(
            f"worker returned {process.returncode}, stdout={line!r}, stderr={stderr.strip()!r}"
        )
    return line


def _integrity_check(path: Path) -> str:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute("PRAGMA integrity_check").fetchall()
    finally:
        connection.close()
    if rows != [("ok",)]:
        raise RuntimeError(f"PRAGMA integrity_check returned {rows!r}")
    return "ok"


def _verify_snapshot(path: Path, expected: Any) -> Any:
    from bot_core.persistence.state_store import SQLiteStateStore

    # This production reopen intentionally happens before native integrity verification.
    with SQLiteStateStore(path) as store:
        actual = store.read_verified_snapshot()
    if actual is None or actual != expected:
        raise RuntimeError("read_verified_snapshot did not exactly match expected state")
    if _integrity_check(path) != "ok":  # defensive: _integrity_check already fails closed
        raise RuntimeError("SQLite integrity verification did not pass")
    return actual


def _file_lock_scenarios(path: Path, processes: list[subprocess.Popen[str]]) -> None:
    holder = _start("lock-holder", path)
    processes.append(holder)
    try:
        _read_marker(holder, "LOCK_HELD")
        contender = _start("lock-contender", path)
        processes.append(contender)
        _finish(contender, "LOCK_CONFLICT\tCoreHostAlreadyRunningError\t")
        processes.remove(contender)
    except BaseException as exc:
        raise Stage6ProbeError(
            "FILE_LOCK_CONFLICT", "WINDOWS_FILE_LOCKING", f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        _kill_and_wait(holder)
        processes.remove(holder)
        if not _scope(path).lock_path.is_file():
            raise RuntimeError("lock locator disappeared after forced holder death")
        reacquirer = _start("lock-reacquire", path)
        processes.append(reacquirer)
        _finish(reacquirer, "LOCK_REACQUIRED")
        processes.remove(reacquirer)
    except BaseException as exc:
        raise Stage6ProbeError(
            "FILE_LOCK_CRASH_RELEASE",
            "WINDOWS_FILE_LOCKING",
            f"{type(exc).__name__}: {exc}",
        ) from exc


def _sqlite_scenarios(path: Path, processes: list[subprocess.Popen[str]]) -> None:
    from bot_core.persistence.state_store import SQLiteStateStore

    try:
        with SQLiteStateStore(path) as store:
            _commit_generation(store, path, 1)
            baseline = store.read_verified_snapshot()
        if baseline is None or baseline.metadata.protected_freshness_generation != 1:
            raise RuntimeError("canonical generation-1 baseline verification failed")
    except BaseException as exc:
        raise Stage6ProbeError(
            "SQLITE_BASELINE", "WINDOWS_SQLITE_CRASH_INTEGRITY", f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        uncommitted = _start("sqlite-uncommitted", path)
        processes.append(uncommitted)
        _read_marker(uncommitted, "UNCOMMITTED_WRITE_HELD")
        _kill_and_wait(uncommitted)
        processes.remove(uncommitted)
    except BaseException as exc:
        raise Stage6ProbeError(
            "SQLITE_UNCOMMITTED_CRASH",
            "WINDOWS_SQLITE_CRASH_INTEGRITY",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    try:
        _verify_snapshot(path, baseline)
    except BaseException as exc:
        raise Stage6ProbeError(
            "SQLITE_POST_CRASH_VERIFY",
            "WINDOWS_SQLITE_CRASH_INTEGRITY",
            f"{type(exc).__name__}: {exc}",
        ) from exc

    try:
        committed = _start("sqlite-committed", path)
        processes.append(committed)
        line = _read_marker(committed, "COMMITTED_STATE_VERIFIED\t")
        expected_metadata = json.loads(line.split("\t", 1)[1])
        _kill_and_wait(committed)
        processes.remove(committed)
    except BaseException as exc:
        raise Stage6ProbeError(
            "SQLITE_COMMITTED_CRASH",
            "WINDOWS_SQLITE_CRASH_INTEGRITY",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    try:
        with SQLiteStateStore(path) as store:
            recovered = store.read_verified_snapshot()
        if recovered is None or recovered.metadata.to_mapping() != expected_metadata:
            raise RuntimeError(
                "recovered generation-2 metadata/fingerprints differ from committed state"
            )
        if recovered.metadata.protected_freshness_generation != 2:
            raise RuntimeError("recovered committed state is not exactly generation 2")
        if _integrity_check(path) != "ok":
            raise RuntimeError("committed-state SQLite integrity verification did not pass")
    except BaseException as exc:
        raise Stage6ProbeError(
            "SQLITE_POST_CRASH_VERIFY",
            "WINDOWS_SQLITE_CRASH_INTEGRITY",
            f"{type(exc).__name__}: {exc}",
        ) from exc


def run_probe(scratch_parent: Path) -> dict[str, str]:
    """Execute the Windows-only live probe in an owned scratch directory."""

    if os.name != "nt":
        raise Stage6ProbeError(
            "FILE_LOCK_CONFLICT", "WINDOWS_FILE_LOCKING", "real Windows host is required"
        )
    root = Path(tempfile.mkdtemp(prefix="cryptohunter-stage6-", dir=scratch_parent))
    processes: list[subprocess.Popen[str]] = []
    primary: BaseException | None = None
    try:
        _file_lock_scenarios(root / "lock-state.sqlite3", processes)
        _sqlite_scenarios(root / "state.sqlite3", processes)
        return {item: "PASS" for item in WINDOWS_STAGE6_ITEMS}
    except BaseException as exc:
        primary = exc
        raise
    finally:
        cleanup_errors: list[str] = []
        for process in processes:
            try:
                if process.poll() is None:
                    _kill_and_wait(process)
            except BaseException as exc:
                cleanup_errors.append(f"worker {process.pid}: {type(exc).__name__}: {exc}")
        try:
            shutil.rmtree(root)
        except OSError as exc:
            cleanup_errors.append(f"scratch {root}: {type(exc).__name__}: {exc}")
        if cleanup_errors:
            detail = "; ".join(cleanup_errors)
            if primary is None:
                raise Stage6ProbeError("SQLITE_POST_CRASH_VERIFY", WINDOWS_STAGE6_ITEMS[-1], f"cleanup failed: {detail}")
            print(f"[CLEANUP] secondary failure: {detail}", file=sys.stderr)


def _worker_main(name: str, path: Path) -> int:
    workers = {
        "lock-holder": _worker_lock_holder,
        "lock-contender": _worker_lock_contender,
        "lock-reacquire": _worker_lock_reacquire,
        "sqlite-uncommitted": _worker_sqlite_uncommitted,
        "sqlite-committed": _worker_sqlite_committed,
    }
    return workers[name](path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=(
        "lock-holder", "lock-contender", "lock-reacquire",
        "sqlite-uncommitted", "sqlite-committed",
    ))
    parser.add_argument("--path", type=Path)
    args = parser.parse_args(argv)
    if args.worker is not None:
        if args.path is None:
            parser.error("--path is required for a worker")
        return _worker_main(args.worker, args.path)
    parser.error("workers are internal; invoke run_probe through deployment.windows_acceptance")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
