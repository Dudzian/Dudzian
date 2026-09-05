from __future__ import annotations

import multiprocessing
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from bot_core.runtime.core_host import (
    CoreHost,
    CoreHostAlreadyRunningError,
    CoreHostProcessLock,
    CoreHostScope,
)


class Resource:
    def __init__(self, events: list[str], name: str) -> None:
        events.append(f"create:{name}")
        self._events = events
        self._name = name

    def close(self) -> None:
        self._events.append(f"close:{self._name}")


def scope(tmp_path: Path, name: str = "state.sqlite") -> CoreHostScope:
    return CoreHostScope("account-1", "device-1", tmp_path / name)


def lock_worker(
    lock_scope: CoreHostScope,
    acquired: Any,
    release: Any,
    result: Any,
) -> None:
    lock = CoreHostProcessLock(lock_scope)
    try:
        lock.acquire()
    except CoreHostAlreadyRunningError:
        result.put("busy")
        return
    result.put("held")
    acquired.set()
    release.wait()
    lock.release()


def test_lock_acquire_release_and_reacquire(tmp_path: Path) -> None:
    lock = CoreHostProcessLock(scope(tmp_path))
    lock.acquire()
    assert lock.held
    lock.release()
    lock.release()
    assert not lock.held
    with CoreHostProcessLock(scope(tmp_path)) as replacement:
        assert replacement.held


def test_stale_locator_file_is_not_authority(tmp_path: Path) -> None:
    lock_scope = scope(tmp_path)
    lock_scope.lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_scope.lock_path.write_text("stale diagnostic content", encoding="utf-8")
    with CoreHostProcessLock(lock_scope) as lock:
        assert lock.held


def test_different_state_store_scopes_do_not_contend(tmp_path: Path) -> None:
    with CoreHostProcessLock(scope(tmp_path, "one.sqlite")):
        with CoreHostProcessLock(scope(tmp_path, "two.sqlite")) as second:
            assert second.held


def test_real_cross_process_exclusion_and_clean_release(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    acquired, release, result = context.Event(), context.Event(), context.Queue()
    owner = context.Process(target=lock_worker, args=(scope(tmp_path), acquired, release, result))
    owner.start()
    assert acquired.wait(timeout=10)
    assert result.get(timeout=10) == "held"
    contender_acquired, contender_release = context.Event(), context.Event()
    contender = context.Process(
        target=lock_worker,
        args=(scope(tmp_path), contender_acquired, contender_release, result),
    )
    contender.start()
    contender.join(timeout=10)
    assert contender.exitcode == 0
    assert result.get(timeout=10) == "busy"
    release.set()
    owner.join(timeout=10)
    assert owner.exitcode == 0
    with CoreHostProcessLock(scope(tmp_path)) as replacement:
        assert replacement.held


def test_abnormal_process_death_releases_os_lock(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    acquired, release, result = context.Event(), context.Event(), context.Queue()
    owner = context.Process(target=lock_worker, args=(scope(tmp_path), acquired, release, result))
    owner.start()
    assert acquired.wait(timeout=10)
    assert result.get(timeout=10) == "held"
    owner.terminate()
    owner.join(timeout=10)
    assert owner.exitcode is not None
    with CoreHostProcessLock(scope(tmp_path)) as replacement:
        assert replacement.held


def factories(events: list[str]) -> tuple[Callable[[], Resource], Callable[[], Resource]]:
    return (
        lambda: Resource(events, "session"),
        lambda: Resource(events, "store"),
    )


def test_core_host_follows_canonical_order_and_closes_in_reverse(tmp_path: Path) -> None:
    events: list[str] = []
    session_factory, store_factory = factories(events)
    host = CoreHost(
        scope(tmp_path),
        runtime_session_factory=session_factory,
        state_store_factory=store_factory,
    )
    host.start()
    assert host.owns_process_lock
    assert events == ["create:session", "create:store"]
    host.close()
    host.close()
    assert events == ["create:session", "create:store", "close:store", "close:session"]
    assert not host.owns_process_lock


def test_second_core_has_no_authority_pipeline_side_effects(tmp_path: Path) -> None:
    first_events: list[str] = []
    session_factory, store_factory = factories(first_events)
    first = CoreHost(
        scope(tmp_path),
        runtime_session_factory=session_factory,
        state_store_factory=store_factory,
    )
    first.start()
    counts = {"session": 0, "store": 0, "recovery": 0, "readiness": 0}

    def counted(name: str) -> Resource:
        counts[name] += 1
        return Resource([], name)

    second = CoreHost(
        scope(tmp_path),
        runtime_session_factory=lambda: counted("session"),
        state_store_factory=lambda: counted("store"),
    )
    with pytest.raises(CoreHostAlreadyRunningError):
        second.start()
    assert counts == {"session": 0, "store": 0, "recovery": 0, "readiness": 0}
    first.close()


def test_runtime_session_failure_releases_lock_without_opening_store(tmp_path: Path) -> None:
    store_calls = 0

    def fail_session() -> Resource:
        raise LookupError("session failed")

    def open_store() -> Resource:
        nonlocal store_calls
        store_calls += 1
        return Resource([], "store")

    host = CoreHost(
        scope(tmp_path), runtime_session_factory=fail_session, state_store_factory=open_store
    )
    with pytest.raises(LookupError, match="session failed"):
        host.start()
    assert store_calls == 0
    assert not host.owns_process_lock
    with CoreHostProcessLock(scope(tmp_path)):
        pass


def test_state_store_failure_closes_session_and_releases_lock(tmp_path: Path) -> None:
    events: list[str] = []

    def fail_store() -> Resource:
        raise OSError("store failed")

    host = CoreHost(
        scope(tmp_path),
        runtime_session_factory=lambda: Resource(events, "session"),
        state_store_factory=fail_store,
    )
    with pytest.raises(OSError, match="store failed"):
        host.start()
    assert events == ["create:session", "close:session"]
    assert not host.owns_process_lock
    with CoreHostProcessLock(scope(tmp_path)):
        pass


def test_lock_does_not_create_state_store(tmp_path: Path) -> None:
    lock_scope = scope(tmp_path)
    with CoreHostProcessLock(lock_scope):
        pass
    assert not lock_scope.state_store_path.exists()
    assert list(tmp_path.iterdir()) == [lock_scope.lock_path]
