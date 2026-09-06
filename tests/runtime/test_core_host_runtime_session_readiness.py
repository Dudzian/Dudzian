from __future__ import annotations

import re
import inspect
from collections.abc import Callable
from pathlib import Path

import pytest

from bot_core.persistence.local_durable_evidence import LocalDurableEvidenceRegistry
from bot_core.persistence.state_store import SQLiteStateStore
from bot_core.runtime.core_host import CoreHost, CoreHostProcessLock, CoreHostScope
from bot_core.runtime.core_host_recovery_types import (
    CoreHostRecoveryClassification,
    CoreHostStartupDisposition,
    StartupSubsystemRecoveryClassification,
    StartupSubsystemRecoveryResult,
)
from bot_core.runtime.runtime_session import RuntimeSession, create_runtime_session
from tests.persistence.test_protected_freshness_handoff import Boundary, record
from tests.persistence.test_state_store_records import _commit, _metadata
from tests.runtime.test_core_host_startup_recovery import (
    CountingSQLiteStateStore,
    Session,
    _coordinator,
    _scope,
)

ACCOUNT = "acct_01890f47-8f2a-7abc-8def-1234567890ab"
DEVICE = "dev_01890f4c-7b9a-7abc-8def-1234567890ab"
ID_RE = re.compile(r"^run_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


class Resource:
    def __init__(self, events: list[str], name: str) -> None:
        self.events, self.name, self.closed = events, name, False
        events.append(f"create:{name}")

    def close(self) -> None:
        self.closed = True
        self.events.append(f"close:{self.name}")


class Recovery:
    def __init__(self, classification, events):  # type: ignore[no-untyped-def]
        self.classification, self.events = classification, events

    def recover(self) -> StartupSubsystemRecoveryResult:
        self.events.append("recover")
        return StartupSubsystemRecoveryResult(self.classification)


def scope(path: Path) -> CoreHostScope:
    return CoreHostScope(ACCOUNT, DEVICE, path)


def _initialize(path: Path) -> tuple[Boundary, int]:
    with SQLiteStateStore(path) as store:
        metadata = _commit(store, _metadata())
    boundary = Boundary(
        record(
            "COMMITTED",
            committed_generation=metadata.protected_freshness_generation,
            committed_state_fingerprint_sha256=metadata.state_fingerprint_sha256,
        )
    )
    return boundary, metadata.protected_freshness_generation


def _real_host(
    path: Path,
    boundary: Boundary,
    *,
    runtime_session_factory: Callable[[], RuntimeSession] | None = None,
    hook: Callable[[str, RuntimeSession], None] | None = None,
    store_factory: Callable[[], SQLiteStateStore] | None = None,
) -> tuple[CoreHost, list[SQLiteStateStore], list[LocalDurableEvidenceRegistry]]:
    stores: list[SQLiteStateStore] = []
    registries: list[LocalDurableEvidenceRegistry] = []

    def open_store() -> SQLiteStateStore:
        store = store_factory() if store_factory is not None else SQLiteStateStore(path)
        stores.append(store)
        return store

    def recovery_factory(scope, store):  # type: ignore[no-untyped-def]
        evidence = LocalDurableEvidenceRegistry()
        registries.append(evidence)
        return _coordinator(path, store, boundary, scope=scope, evidence=evidence)

    return (
        CoreHost(
            _scope(path),
            runtime_session_factory=runtime_session_factory,
            state_store_factory=open_store,
            startup_recovery_factory=recovery_factory,
            runtime_session_publication_hook=hook,
        ),
        stores,
        registries,
    )


def _session_ids(path: Path) -> list[str]:
    with SQLiteStateStore(path) as store:
        snapshot = store.read_verified_snapshot()
        assert snapshot is not None
        return [
            item.record_key
            for item in snapshot.immutable_history
            if item.representation_name == "RuntimeSession canonical identity/history"
        ]


def test_real_core_owned_session_is_created_after_lock_before_store(tmp_path: Path) -> None:
    events: list[str] = []
    path = tmp_path / "ordered.db"
    boundary, _ = _initialize(path)

    class Lock(CoreHostProcessLock):
        def acquire(self) -> None:
            super().acquire()
            events.append("lock_acquired")

    sessions: list[RuntimeSession] = []

    def session_factory() -> RuntimeSession:
        events.append("runtime_session_created")
        session = create_runtime_session(_metadata().device_installation_id)
        sessions.append(session)
        return session

    def store_factory() -> SQLiteStateStore:
        events.append("store_opened")
        return SQLiteStateStore(path)

    def recovery_factory(scope, store):  # type: ignore[no-untyped-def]
        recovery = _coordinator(path, store, boundary, scope=scope)

        class RecordingRecovery:
            def recover(self):  # type: ignore[no-untyped-def]
                result = recovery.recover()
                events.append("p1b_recovery")
                return result

            def runtime_session_history_publisher(self):  # type: ignore[no-untyped-def]
                return recovery.runtime_session_history_publisher()

        return RecordingRecovery()

    host = CoreHost(
        _scope(path),
        runtime_session_factory=session_factory,
        state_store_factory=store_factory,
        startup_recovery_factory=recovery_factory,
        runtime_session_publication_hook=lambda stage, session: events.append(
            f"publication_{stage}"
        ),
        lock_factory=Lock,
    )
    host.start()
    events.append("proceed_disposition")
    assert ID_RE.fullmatch(sessions[0].runtime_session_id)
    assert events == [
        "lock_acquired",
        "runtime_session_created",
        "store_opened",
        "p1b_recovery",
        "publication_before",
        "publication_after",
        "proceed_disposition",
    ]
    assert host.startup_disposition is CoreHostStartupDisposition.PROCEED_TO_LATER_STARTUP_GATES
    host.close()


def test_production_ids_are_fresh_canonical_uuid7() -> None:
    first, second = create_runtime_session(DEVICE), create_runtime_session(DEVICE)
    assert first.runtime_session_id != second.runtime_session_id
    assert ID_RE.fullmatch(first.runtime_session_id)
    assert ID_RE.fullmatch(second.runtime_session_id)


def test_corehost_has_no_authority_replacing_publisher_factory() -> None:
    assert "runtime_session_publisher_factory" not in inspect.signature(CoreHost).parameters


@pytest.mark.parametrize("_iteration", range(10))
def test_empty_is_setup_required_without_publication(tmp_path: Path, _iteration: int) -> None:
    events: list[str] = []
    host = CoreHost(
        scope(tmp_path / f"empty-{_iteration}.db"),
        state_store_factory=lambda: Resource(events, "store"),
        startup_recovery_factory=lambda scope, store: Recovery(
            StartupSubsystemRecoveryClassification.EMPTY_UNINITIALIZED, events
        ),
        runtime_session_publication_hook=lambda stage, session: events.append("publication"),
    )
    host.start()
    host.start()
    assert host.startup_disposition is CoreHostStartupDisposition.SETUP_REQUIRED
    assert "publication" not in events and events.count("recover") == 1
    host.close()


@pytest.mark.parametrize("_iteration", range(10))
def test_initialized_real_publication(tmp_path: Path, _iteration: int) -> None:
    path = tmp_path / f"initialized-{_iteration}.db"
    boundary, generation = _initialize(path)
    events: list[str] = []
    host, _, _ = _real_host(path, boundary, hook=lambda stage, session: events.append(stage))
    host.start()
    assert events == ["before", "after"]
    assert host.startup_recovery_result is not None
    assert (
        host.startup_recovery_result.classification
        is CoreHostRecoveryClassification.INITIALIZED_RECOVERY_COMPLETE
    )
    assert host.startup_disposition is CoreHostStartupDisposition.PROCEED_TO_LATER_STARTUP_GATES
    with SQLiteStateStore(path) as observer:
        snapshot = observer.read_verified_snapshot()
        assert snapshot is not None
        assert snapshot.metadata.protected_freshness_generation == generation + 1
    assert not hasattr(CoreHostStartupDisposition, "READY")
    host.close()


@pytest.mark.parametrize("_iteration", range(10))
def test_repeated_same_process_start_publishes_once(tmp_path: Path, _iteration: int) -> None:
    path = tmp_path / f"repeated-{_iteration}.db"
    boundary, generation = _initialize(path)
    events: list[str] = []
    host, _, _ = _real_host(path, boundary, hook=lambda stage, session: events.append(stage))
    host.start()
    host.start()
    assert events == ["before", "after"]
    assert len(_session_ids(path)) == 1
    with SQLiteStateStore(path) as observer:
        snapshot = observer.read_verified_snapshot()
        assert snapshot is not None
        assert snapshot.metadata.protected_freshness_generation == generation + 1
    host.close()


def test_non_runtime_session_initialized_fails_closed(tmp_path: Path) -> None:
    events: list[str] = []
    store = Resource(events, "store")
    session = Session(events)
    host = CoreHost(
        scope(tmp_path / "invalid-session.db"),
        runtime_session_factory=lambda: session,
        state_store_factory=lambda: store,
        startup_recovery_factory=lambda scope, store: Recovery(
            StartupSubsystemRecoveryClassification.INITIALIZED_DURABLE_RECOVERY_RESOLVED,
            events,
        ),
    )
    with pytest.raises(RuntimeError, match="requires a RuntimeSession"):
        host.start()
    assert store.closed and session.close_count == 1 and not host.owns_process_lock
    assert host.startup_recovery_result is None and host.startup_disposition is None


@pytest.mark.parametrize("attribute", ["_lock", "_runtime_session", "_state_store"])
def test_publication_hook_cannot_replace_real_owner_and_tampering_fails_closed(
    tmp_path: Path, attribute: str
) -> None:
    path = tmp_path / f"tamper-{attribute}.db"
    boundary, generation = _initialize(path)
    host: CoreHost

    def tamper(stage: str, session: RuntimeSession) -> None:
        if stage == "after":
            setattr(host, attribute, object())

    host, _, _ = _real_host(path, boundary, hook=tamper)
    with pytest.raises(RuntimeError, match="topology changed"):
        host.start()
    assert not host.owns_process_lock and host.startup_disposition is None
    with SQLiteStateStore(path) as observer:
        snapshot = observer.read_verified_snapshot()
        assert snapshot is not None
        assert snapshot.metadata.protected_freshness_generation == generation + 1
        assert len(_session_ids(path)) == 1


@pytest.mark.parametrize(
    "tamper_kind",
    ["release_lock", "replace_session", "replace_store", "clear_recovery_result"],
)
def test_before_publication_topology_tamper_is_fenced_before_durable_mutation(
    tmp_path: Path, tamper_kind: str
) -> None:
    class CountingRuntimeSession(RuntimeSession):
        def __init__(self) -> None:
            generated = create_runtime_session(_metadata().device_installation_id)
            super().__init__(generated.runtime_session_id, generated.device_installation_id)
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1
            super().close()

    path = tmp_path / f"before-tamper-{tamper_kind}.db"
    boundary, generation = _initialize(path)
    sessions: list[CountingRuntimeSession] = []
    foreign_session = CountingRuntimeSession()
    foreign_store = Resource([], "foreign-store")
    host: CoreHost

    def session_factory() -> RuntimeSession:
        session = CountingRuntimeSession()
        sessions.append(session)
        return session

    def tamper(stage: str, session: RuntimeSession) -> None:
        if stage != "before":
            return
        if tamper_kind == "release_lock":
            assert host._lock is not None and host._lock.held
            host._lock.release()
        elif tamper_kind == "replace_session":
            setattr(host, "_runtime_session", foreign_session)
        elif tamper_kind == "replace_store":
            setattr(host, "_state_store", foreign_store)
        else:
            setattr(host, "_startup_recovery_result", None)

    host, stores, _ = _real_host(
        path,
        boundary,
        runtime_session_factory=session_factory,
        hook=tamper,
        store_factory=lambda: CountingSQLiteStateStore(path),
    )
    with pytest.raises(RuntimeError, match="before RuntimeSession publication"):
        host.start()

    assert boundary.calls.count("prepare") == 0
    assert sessions[0].closed and sessions[0].close_count == 1
    assert isinstance(stores[0], CountingSQLiteStateStore)
    assert stores[0]._closed and stores[0].close_count == 1
    assert host.startup_recovery_result is None
    assert host.startup_disposition is None
    assert not host.owns_process_lock
    assert foreign_session.close_count == 0 and not foreign_session.closed
    assert not foreign_store.closed
    with SQLiteStateStore(path) as observer:
        snapshot = observer.read_verified_snapshot()
        assert snapshot is not None
        assert snapshot.metadata.protected_freshness_generation == generation
        assert _session_ids(path) == []
    with CoreHostProcessLock(_scope(path)) as replacement:
        assert replacement.held


@pytest.mark.parametrize("_iteration", range(10))
def test_sequential_real_corehost_restart_uses_fresh_session(
    tmp_path: Path, _iteration: int
) -> None:
    path = tmp_path / f"restart-{_iteration}.db"
    boundary, generation = _initialize(path)
    host_a, _, _ = _real_host(path, boundary)
    host_a.start()
    session_a = host_a._runtime_session
    assert isinstance(session_a, RuntimeSession)
    id_a = session_a.runtime_session_id
    host_a.close()
    host_b, _, _ = _real_host(path, boundary)
    host_b.start()
    session_b = host_b._runtime_session
    assert isinstance(session_b, RuntimeSession)
    id_b = session_b.runtime_session_id
    assert id_a != id_b and _session_ids(path).count(id_a) == _session_ids(path).count(id_b) == 1
    with SQLiteStateStore(path) as observer:
        snapshot = observer.read_verified_snapshot()
        assert snapshot is not None
        assert snapshot.metadata.protected_freshness_generation == generation + 2
    host_b.close()


@pytest.mark.parametrize("_iteration", range(10))
def test_same_id_collision_through_corehost_fails_before_prepare(
    tmp_path: Path, _iteration: int
) -> None:
    path = tmp_path / f"collision-{_iteration}.db"
    boundary, _ = _initialize(path)
    host_a, _, _ = _real_host(path, boundary)
    host_a.start()
    session_a = host_a._runtime_session
    assert isinstance(session_a, RuntimeSession)
    old_id = session_a.runtime_session_id
    host_a.close()
    before_prepares = boundary.calls.count("prepare")
    with SQLiteStateStore(path) as observer:
        before = observer.read_verified_snapshot()
    replay = RuntimeSession(old_id, _metadata().device_installation_id)
    host_b, stores, _ = _real_host(path, boundary, runtime_session_factory=lambda: replay)
    with pytest.raises(RuntimeError, match="collision"):
        host_b.start()
    with SQLiteStateStore(path) as observer:
        assert observer.read_verified_snapshot() == before
    assert boundary.calls.count("prepare") == before_prepares
    assert replay.closed and stores[0]._closed and not host_b.owns_process_lock
    assert host_b.startup_recovery_result is None and host_b.startup_disposition is None


@pytest.mark.parametrize(
    "window",
    [
        "before_prepare",
        "prepared_before_local",
        "local_before_finalize",
        "ack_loss",
        "after_finalize",
    ],
)
@pytest.mark.parametrize("_iteration", range(10))
def test_real_crash_restart_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, window: str, _iteration: int
) -> None:
    path = tmp_path / f"{window}-{_iteration}.db"
    boundary, generation = _initialize(path)
    sessions: list[RuntimeSession] = []

    def session_factory() -> RuntimeSession:
        session = create_runtime_session(_metadata().device_installation_id)
        sessions.append(session)
        return session

    hook = None
    if window == "before_prepare":
        boundary.before_prepare = lambda: (_ for _ in ()).throw(RuntimeError("before prepare"))
    elif window == "local_before_finalize":
        boundary.fail_finalize = True
    elif window == "ack_loss":
        boundary.ack_lost = True
    elif window == "after_finalize":
        hook = lambda stage, session: (
            (_ for _ in ()).throw(RuntimeError("after finalize")) if stage == "after" else None
        )

    stores: list[SQLiteStateStore] = []

    def open_a() -> SQLiteStateStore:
        store = SQLiteStateStore(path)
        stores.append(store)
        if window == "prepared_before_local":
            monkeypatch.setattr(
                store,
                "commit_prepared_state",
                lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("before local")),
            )
        return store

    host_a, _, _ = _real_host(
        path,
        boundary,
        runtime_session_factory=session_factory,
        hook=hook,
        store_factory=open_a,
    )
    with pytest.raises(RuntimeError):
        host_a.start()
    id_a = sessions[0].runtime_session_id
    after_a = _session_ids(path)
    if window in {"before_prepare", "prepared_before_local"}:
        assert id_a not in after_a
        assert len(after_a) == 0
    else:
        assert after_a.count(id_a) == 1

    boundary.before_prepare = None
    boundary.fail_finalize = False
    boundary.ack_lost = False
    host_b, _, _ = _real_host(path, boundary, runtime_session_factory=session_factory)
    host_b.start()
    id_b = sessions[1].runtime_session_id
    final_ids = _session_ids(path)
    assert id_b != id_a and final_ids.count(id_b) == 1
    assert final_ids.count(id_a) == (
        0 if window in {"before_prepare", "prepared_before_local"} else 1
    )
    assert host_b.startup_disposition is CoreHostStartupDisposition.PROCEED_TO_LATER_STARTUP_GATES
    with SQLiteStateStore(path) as observer:
        final = observer.read_verified_snapshot()
        assert final is not None
        expected = generation + (1 if window in {"before_prepare", "prepared_before_local"} else 2)
        assert final.metadata.protected_freshness_generation == expected
        assert boundary.value["lifecycle"] == "COMMITTED"
        assert boundary.value["committed_generation"] == expected
    host_b.close()


@pytest.mark.parametrize("_iteration", range(10))
def test_failed_publication_exact_cleanup(tmp_path: Path, _iteration: int) -> None:
    path = tmp_path / f"cleanup-{_iteration}.db"
    boundary, _ = _initialize(path)
    boundary.before_prepare = lambda: (_ for _ in ()).throw(RuntimeError("publication failed"))
    sessions: list[RuntimeSession] = []

    def session_factory() -> RuntimeSession:
        session = create_runtime_session(_metadata().device_installation_id)
        sessions.append(session)
        return session

    host, stores, _ = _real_host(path, boundary, runtime_session_factory=session_factory)
    with pytest.raises(RuntimeError, match="publication failed"):
        host.start()
    assert stores[0]._closed and sessions[0].closed and not host.owns_process_lock
    assert host.startup_recovery_result is None and host.startup_disposition is None
    with CoreHostProcessLock(_scope(path)) as replacement:
        assert replacement.held
