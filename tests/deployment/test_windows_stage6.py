from __future__ import annotations

import ast
from pathlib import Path
import re
import sys
from types import ModuleType

import pytest

import deployment.windows_stage6_probe as probe


SOURCE_PATH = Path(probe.__file__)
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")


def _function_source(name: str) -> str:
    tree = ast.parse(SOURCE)
    node = next(
        item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return ast.get_source_segment(SOURCE, node) or ""


def test_live_entrypoint_is_windows_only_and_never_static_pass() -> None:
    assert 'if os.name != "nt"' in _function_source("run_probe")
    if sys.platform != "win32":
        with pytest.raises(probe.Stage6ProbeError, match="real Windows host is required"):
            probe.run_probe(Path.cwd())


def test_state_store_identity_is_stable_and_never_path_derived() -> None:
    first = probe._candidate_metadata(Path("first-scratch/state.sqlite3"), 1)
    second = probe._candidate_metadata(Path("other-scratch/different-name.sqlite3"), 1)
    generation_two = probe._candidate_metadata(Path("third-scratch/state.sqlite3"), 2)
    fingerprints = {
        first.state_store_identity_fingerprint_sha256,
        second.state_store_identity_fingerprint_sha256,
        generation_two.state_store_identity_fingerprint_sha256,
    }
    assert fingerprints == {probe.STAGE6_STATE_STORE_IDENTITY_FINGERPRINT_SHA256}
    assert re.fullmatch(r"[0-9a-f]{64}", fingerprints.pop())
    candidate = _function_source("_candidate_metadata")
    assert "resolve" not in candidate
    assert "hashlib" not in candidate


def test_file_lock_workers_use_production_contract_and_one_exact_scope() -> None:
    holder = _function_source("_worker_lock_holder")
    contender = _function_source("_worker_lock_contender")
    reacquire = _function_source("_worker_lock_reacquire")
    assert "CoreHostProcessLock(_scope(path))" in holder
    assert "CoreHostProcessLock(_scope(path))" in contender
    assert "CoreHostProcessLock(_scope(path))" in reacquire
    assert "CoreHostAlreadyRunningError" in contender
    assert "PROTECTED_SECTION_ENTERED" in contender


def test_lock_conflict_precedes_forced_death_and_reacquire_uses_stale_locator() -> None:
    scenario = _function_source("_file_lock_scenarios")
    conflict = scenario.index('"LOCK_CONFLICT\\tCoreHostAlreadyRunningError\\t"')
    kill = scenario.index("_kill_and_wait(holder)")
    stale = scenario.index("lock_path.is_file()")
    reacquire = scenario.index('"lock-reacquire"')
    assert conflict < kill < stale < reacquire
    assert ".unlink(" not in scenario
    assert ".replace(" not in scenario


def test_all_parent_waits_are_bounded_and_marker_driven() -> None:
    assert "lines.get(timeout=WORKER_TIMEOUT_SECONDS)" in _function_source("_read_marker")
    assert "communicate(timeout=WORKER_TIMEOUT_SECONDS)" in _function_source("_finish")
    assert "wait(timeout=PROCESS_EXIT_TIMEOUT_SECONDS)" in _function_source("_kill_and_wait")
    assert "time.sleep" not in SOURCE


def test_uncommitted_marker_is_after_begin_and_write_without_commit_or_rollback() -> None:
    worker = _function_source("_worker_sqlite_uncommitted")
    begin = worker.index('"BEGIN IMMEDIATE"')
    write = worker.index("UPDATE state_store_metadata")
    marker = worker.index('"UNCOMMITTED_WRITE_HELD"')
    assert begin < write < marker
    assert 'execute("COMMIT")' not in worker
    assert 'execute("ROLLBACK")' not in worker


def test_baseline_and_committed_generation_use_public_state_store_api() -> None:
    commit = _function_source("_commit_generation")
    committed = _function_source("_worker_sqlite_committed")
    scenario = _function_source("_sqlite_scenarios")
    assert "derive_prepared_metadata" in commit
    assert "commit_prepared_metadata" in commit
    assert committed.index("_commit_generation") < committed.index("read_verified_snapshot")
    assert committed.index("read_verified_snapshot") < committed.index("COMMITTED_STATE_VERIFIED")
    assert scenario.index("UNCOMMITTED_WRITE_HELD") < scenario.index("_kill_and_wait(uncommitted)")
    assert scenario.index("COMMITTED_STATE_VERIFIED") < scenario.index("_kill_and_wait(committed)")
    verification = _function_source("_verify_snapshot")
    assert verification.index("read_verified_snapshot") < verification.index("_integrity_check")
    assert "unlink" not in scenario + verification
    assert "-wal" not in scenario + verification
    assert "-shm" not in scenario + verification


def test_post_crash_pass_requires_both_exact_semantics_and_integrity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = object()

    class Store:
        snapshot = expected

        def __init__(self, _path: Path) -> None:
            pass

        def __enter__(self) -> Store:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def read_verified_snapshot(self) -> object:
            return self.snapshot

    fake = ModuleType("bot_core.persistence.state_store")
    fake.SQLiteStateStore = Store  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bot_core.persistence.state_store", fake)
    monkeypatch.setattr(probe, "_integrity_check", lambda _path: "ok")
    assert probe._verify_snapshot(Path("state.sqlite3"), expected) is expected

    Store.snapshot = object()
    with pytest.raises(RuntimeError, match="exactly match"):
        probe._verify_snapshot(Path("state.sqlite3"), expected)

    Store.snapshot = expected
    monkeypatch.setattr(probe, "_integrity_check", lambda _path: "not ok")
    with pytest.raises(RuntimeError, match="integrity"):
        probe._verify_snapshot(Path("state.sqlite3"), expected)


def test_forced_termination_is_not_graceful(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    class Process:
        def kill(self) -> None:
            calls.append("kill")

        def wait(self, *, timeout: float) -> int:
            calls.append(timeout)
            return 1

    probe._kill_and_wait(Process())  # type: ignore[arg-type]
    assert calls == ["kill", probe.PROCESS_EXIT_TIMEOUT_SECONDS]
    assert "terminate" not in _function_source("_kill_and_wait")
