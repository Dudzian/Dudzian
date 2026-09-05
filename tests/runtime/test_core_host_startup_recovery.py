from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any, cast

import pytest

from bot_core.persistence.local_durable_evidence import LocalDurableEvidenceRegistry
from bot_core.persistence.migration_completion import DurableMigrationCompletionCoordinator
from bot_core.persistence.migration_execution_engine import MigrationExecutionCoordinator
from bot_core.persistence.migration_execution import (
    MigrationExecutionAuthority,
    MigrationExecutionPlan,
    MigrationSqlOperation,
)
from bot_core.persistence.migration_protocol import (
    DurableMigrationLifecycleCoordinator,
    MigrationDefinition,
    MigrationRegistry,
)
from bot_core.persistence.physical_schema_registry import (
    StateStorePhysicalSchemaError,
    StateStorePhysicalSchemaRegistry,
)
from bot_core.persistence.protected_freshness_handoff import (
    ProtectedFreshnessHandoffCoordinator,
)
from bot_core.persistence.secret_handoff import (
    DurableSecretHandoffExecutionCoordinator,
    DurableSecretHandoffLifecycleCoordinator,
    ExternalOutcome,
)
from bot_core.persistence.state_store import SQLiteStateStore
from bot_core.runtime.core_host import CoreHost, CoreHostProcessLock, CoreHostScope
from bot_core.runtime.core_host_startup_recovery import (
    CoreHostStartupRecoveryCoordinator,
    CoreHostStartupRecoveryError,
)
from bot_core.runtime.core_host_recovery_types import (
    CoreHostRecoveryClassification,
    StartupSubsystemRecoveryClassification,
    StartupSubsystemRecoveryResult,
)
from tests.persistence.test_protected_freshness_handoff import Boundary, record
from tests.persistence.test_protected_freshness_recovery import prepared
from tests.persistence.test_durable_lifecycle_cas import _initialize_with_migration
from tests.persistence.test_migration_completion_orchestrator import _operation
from tests.persistence.test_migration_execution_engine import _registry, _target_fingerprint
from tests.persistence.test_state_store_records import _commit, _metadata
from tests.persistence.test_secret_handoff import descriptor
from tests.persistence.test_secret_handoff_execution_orchestrator import Port


class Session:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1
        if self.events is not None:
            self.events.append("session:close")


class SecretPort:
    def begin(self, descriptor: Any) -> Any:
        raise AssertionError("empty handoff set must not begin external mutation")

    def reconcile(self, descriptor: Any) -> Any:
        raise AssertionError("empty handoff set must not reconcile external mutation")

    def cleanup(self, descriptor: Any) -> None:
        raise AssertionError("empty handoff set must not clean up external mutation")


class CountingSQLiteStateStore(SQLiteStateStore):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.close_count = 0

    def close(self) -> None:
        if not self._closed:
            self.close_count += 1
        super().close()


def _scope(path: Path, *, account: str | None = None, device: str | None = None) -> CoreHostScope:
    metadata = _metadata()
    return CoreHostScope(
        account or metadata.account_id,
        device or metadata.device_installation_id,
        path,
    )


def _coordinator(
    path: Path,
    store: SQLiteStateStore,
    boundary: Boundary,
    *,
    scope: CoreHostScope | None = None,
    evidence: LocalDurableEvidenceRegistry | None = None,
    secret_port: Any | None = None,
    migrations: MigrationRegistry | None = None,
    physical_schemas: Any | None = None,
) -> CoreHostStartupRecoveryCoordinator:
    evidence = evidence or LocalDurableEvidenceRegistry()
    protected = ProtectedFreshnessHandoffCoordinator(store, evidence, boundary)
    migrations = migrations or MigrationRegistry()
    lifecycles = DurableMigrationLifecycleCoordinator(store, migrations, protected)
    execution = MigrationExecutionCoordinator(store, migrations, protected)
    completion = DurableMigrationCompletionCoordinator(store, execution, lifecycles, protected)
    secret_lifecycles = DurableSecretHandoffLifecycleCoordinator(store, protected)
    secrets = DurableSecretHandoffExecutionCoordinator(
        store, secret_lifecycles, protected, secret_port or SecretPort()
    )
    return CoreHostStartupRecoveryCoordinator(
        scope or _scope(path),
        store,
        protected,
        migrations,
        lifecycles,
        completion,
        secrets,
        evidence,
        physical_schemas,
    )


def test_initialized_real_pipeline_reaches_recovery_complete(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    with SQLiteStateStore(path) as store:
        metadata = _commit(store, _metadata())
        boundary = Boundary(
            record(
                "COMMITTED",
                committed_generation=metadata.protected_freshness_generation,
                committed_state_fingerprint_sha256=metadata.state_fingerprint_sha256,
            )
        )
        result = _coordinator(path, store, boundary).recover()
        assert (
            result.classification
            is StartupSubsystemRecoveryClassification.INITIALIZED_DURABLE_RECOVERY_RESOLVED
        )
        assert not {"prepare", "abort", "finalize"} & set(boundary.calls)


def test_empty_store_skips_every_authority(tmp_path: Path) -> None:
    path = tmp_path / "empty.db"
    with SQLiteStateStore(path) as store:
        boundary = Boundary(record("UNINITIALIZED"))
        result = _coordinator(path, store, boundary).recover()
        assert result.classification is StartupSubsystemRecoveryClassification.EMPTY_UNINITIALIZED
        assert boundary.calls == []


@pytest.mark.parametrize(
    ("scope_factory", "message"),
    [
        (lambda path: _scope(path.with_name("other.db")), "path"),
        (lambda path: _scope(path, account="acct_01890f4c-7b9a-7cc1-8a2b-123456789abd"), "account"),
        (lambda path: _scope(path, device="dev_01890f4c-7b9a-7cc1-8a2b-123456789abd"), "device"),
    ],
)
def test_local_binding_fails_before_m0_3(tmp_path: Path, scope_factory: Any, message: str) -> None:
    path = tmp_path / "state.db"
    with SQLiteStateStore(path) as store:
        _commit(store, _metadata())
        boundary = Boundary(record("UNINITIALIZED"))
        with pytest.raises(CoreHostStartupRecoveryError, match=message):
            _coordinator(path, store, boundary, scope=scope_factory(path)).recover()
        assert boundary.calls == []


def test_physical_registry_is_exact_and_unknown_fails_closed() -> None:
    registry = StateStorePhysicalSchemaRegistry()
    assert registry.current_version == 1
    assert (
        registry.expected_fingerprint(1)
        == "18f9bac7640b66fb1051d5e1bcfe7345c79a8dcb33f417b40009fb049547c680"
    )
    with pytest.raises(StateStorePhysicalSchemaError, match="unknown"):
        registry.expected_fingerprint(2)


def test_core_host_retains_result_runs_once_and_clears_on_close(tmp_path: Path) -> None:
    events: list[str] = []

    class Recovery:
        def recover(self) -> StartupSubsystemRecoveryResult:
            events.append("recovery")
            return StartupSubsystemRecoveryResult(
                StartupSubsystemRecoveryClassification.INITIALIZED_DURABLE_RECOVERY_RESOLVED
            )

    host = CoreHost(
        _scope(tmp_path / "state.db"),
        runtime_session_factory=lambda: Session(events),
        state_store_factory=lambda: Session(events),
        startup_recovery_factory=lambda scope, store: Recovery(),
    )
    host.start()
    host.start()
    assert host.startup_recovery_result is not None
    assert (
        host.startup_recovery_result.classification
        is CoreHostRecoveryClassification.INITIALIZED_RECOVERY_COMPLETE
    )
    assert events == ["recovery"]
    host.close()
    assert host.startup_recovery_result is None


def test_recovery_failure_closes_store_session_and_releases_lock(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    events: list[str] = []

    class Recovery:
        def recover(self) -> StartupSubsystemRecoveryResult:
            raise RuntimeError("recovery failed")

    host = CoreHost(
        _scope(path),
        runtime_session_factory=lambda: Session(events),
        state_store_factory=lambda: Session(events),
        startup_recovery_factory=lambda scope, store: Recovery(),
    )
    with pytest.raises(RuntimeError, match="recovery failed"):
        host.start()
    assert events == ["session:close", "session:close"]
    assert host.startup_recovery_result is None
    assert not host.owns_process_lock
    with CoreHostProcessLock(_scope(path)):
        pass


@pytest.mark.parametrize("invalid", ["complete", True, object(), {"complete": True}])
def test_core_host_rejects_untyped_recovery_proof_and_cleans_up(
    tmp_path: Path, invalid: object
) -> None:
    events: list[str] = []

    class InvalidRecovery:
        def recover(self) -> Any:
            return invalid

    host = CoreHost(
        _scope(tmp_path / "invalid.db"),
        runtime_session_factory=lambda: Session(events),
        state_store_factory=lambda: Session(events),
        startup_recovery_factory=cast(Any, lambda scope, store: InvalidRecovery()),
    )
    with pytest.raises(RuntimeError, match="invalid result type"):
        host.start()
    assert events == ["session:close", "session:close"]
    assert not host.owns_process_lock
    assert host.startup_recovery_result is None


def test_core_host_rejects_typed_result_with_invalid_classification(tmp_path: Path) -> None:
    invalid = StartupSubsystemRecoveryResult(cast(Any, "INITIALIZED_RECOVERY_COMPLETE"))

    class InvalidRecovery:
        def recover(self) -> StartupSubsystemRecoveryResult:
            return invalid

    host = CoreHost(
        _scope(tmp_path / "classification.db"),
        runtime_session_factory=Session,
        state_store_factory=Session,
        startup_recovery_factory=lambda scope, store: InvalidRecovery(),
    )
    with pytest.raises(RuntimeError, match="invalid classification"):
        host.start()
    assert not host.owns_process_lock


def test_real_core_host_owns_initialized_recovery_complete(tmp_path: Path) -> None:
    path = tmp_path / "owned.db"
    with SQLiteStateStore(path) as initializer:
        metadata = _commit(initializer, _metadata())
    boundary = Boundary(
        record(
            "COMMITTED",
            committed_generation=metadata.protected_freshness_generation,
            committed_state_fingerprint_sha256=metadata.state_fingerprint_sha256,
        )
    )
    opened: list[SQLiteStateStore] = []

    def open_store() -> SQLiteStateStore:
        store = SQLiteStateStore(path)
        opened.append(store)
        return store

    host = CoreHost(
        _scope(path),
        runtime_session_factory=Session,
        state_store_factory=open_store,
        startup_recovery_factory=lambda scope, store: _coordinator(
            path, store, boundary, scope=scope
        ),
    )
    host.start()
    assert host.owns_process_lock
    assert host.startup_recovery_result is not None
    assert (
        host.startup_recovery_result.classification
        is CoreHostRecoveryClassification.INITIALIZED_RECOVERY_COMPLETE
    )
    host.close()
    assert len(opened) == 1


def test_real_core_host_empty_path_is_not_recovery_complete(tmp_path: Path) -> None:
    path = tmp_path / "empty-host.db"
    boundary = Boundary(record("UNINITIALIZED"))
    host = CoreHost(
        _scope(path),
        runtime_session_factory=Session,
        state_store_factory=lambda: SQLiteStateStore(path),
        startup_recovery_factory=lambda scope, store: _coordinator(
            path, store, boundary, scope=scope
        ),
    )
    host.start()
    assert host.startup_recovery_result is not None
    assert (
        host.startup_recovery_result.classification
        is CoreHostRecoveryClassification.EMPTY_UNINITIALIZED
    )
    host.close()


def test_same_path_substituted_store_is_rejected_by_m0_3_identity(tmp_path: Path) -> None:
    path = tmp_path / "substituted.db"
    with SQLiteStateStore(path) as store:
        _commit(store, _metadata(state_store_identity_fingerprint_sha256="2" * 64))
        boundary = Boundary(
            record(
                "COMMITTED",
                committed_generation=1,
                committed_state_fingerprint_sha256="3" * 64,
            )
        )
        with pytest.raises(Exception, match="missing|differ"):
            _coordinator(path, store, boundary).recover()


def test_zero_migration_wrong_actual_schema_fails_before_evidence(tmp_path: Path) -> None:
    path = tmp_path / "wrong-schema.db"
    with SQLiteStateStore(path) as store:
        metadata = _commit(store, _metadata())
        store._connection.execute("CREATE TABLE unauthorized_physical_table(value TEXT)")
        boundary = Boundary(
            record(
                "COMMITTED",
                committed_generation=1,
                committed_state_fingerprint_sha256=metadata.state_fingerprint_sha256,
            )
        )
        evidence = LocalDurableEvidenceRegistry()
        with pytest.raises(Exception, match="schema"):
            _coordinator(path, store, boundary, evidence=evidence).recover()


def test_unknown_metadata_schema_version_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "unknown-version.db"
    with SQLiteStateStore(path) as store:
        metadata = _commit(store, _metadata(state_store_schema_version=99))
        boundary = Boundary(
            record(
                "COMMITTED",
                committed_generation=1,
                committed_state_fingerprint_sha256=metadata.state_fingerprint_sha256,
            )
        )
        with pytest.raises(StateStorePhysicalSchemaError, match="unknown"):
            _coordinator(path, store, boundary).recover()


def test_evidence_failure_prevents_recovery_complete(tmp_path: Path) -> None:
    class FailingEvidence(LocalDurableEvidenceRegistry):
        def publish_verified_state(self, store: SQLiteStateStore) -> None:
            return None

    path = tmp_path / "evidence-failure.db"
    with SQLiteStateStore(path) as store:
        metadata = _commit(store, _metadata())
        boundary = Boundary(
            record(
                "COMMITTED",
                committed_generation=1,
                committed_state_fingerprint_sha256=metadata.state_fingerprint_sha256,
            )
        )
        with pytest.raises(CoreHostStartupRecoveryError, match="evidence"):
            _coordinator(path, store, boundary, evidence=FailingEvidence()).recover()


def test_corrupt_store_fails_closed_without_restore(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.db"
    with SQLiteStateStore(path) as store:
        _commit(store, _metadata())
    connection = sqlite3.connect(path)
    connection.execute("UPDATE state_store_metadata SET state_fingerprint_sha256 = ?", ("f" * 64,))
    connection.commit()
    connection.close()
    host = CoreHost(
        _scope(path),
        runtime_session_factory=Session,
        state_store_factory=lambda: SQLiteStateStore(path),
        startup_recovery_factory=lambda scope, store: _coordinator(
            path, store, Boundary(record("UNINITIALIZED")), scope=scope
        ),
    )
    with pytest.raises(Exception):
        host.start()
    assert not host.owns_process_lock


@pytest.mark.parametrize("iteration", range(10))
def test_initialized_success_stress_ten_fresh_hosts(tmp_path: Path, iteration: int) -> None:
    path = tmp_path / f"stress-{iteration}.db"
    with SQLiteStateStore(path) as initializer:
        metadata = _commit(initializer, _metadata())
    boundary = Boundary(
        record(
            "COMMITTED",
            committed_generation=1,
            committed_state_fingerprint_sha256=metadata.state_fingerprint_sha256,
        )
    )
    host = CoreHost(
        _scope(path),
        runtime_session_factory=Session,
        state_store_factory=lambda: SQLiteStateStore(path),
        startup_recovery_factory=lambda scope, store: _coordinator(
            path, store, boundary, scope=scope
        ),
    )
    host.start()
    assert host.startup_recovery_result is not None
    assert (
        host.startup_recovery_result.classification
        is CoreHostRecoveryClassification.INITIALIZED_RECOVERY_COMPLETE
    )
    host.close()


@pytest.mark.parametrize("mode", ["abort", "finalize"])
@pytest.mark.parametrize("_iteration", range(10))
def test_real_m0_3_prepared_recovery_through_core_host(
    tmp_path: Path, mode: str, _iteration: int
) -> None:
    path = tmp_path / f"prepared-{mode}.db"
    with SQLiteStateStore(path) as initializer:
        one = _commit(initializer, _metadata())
        candidate = initializer.derive_prepared_metadata(
            _metadata(2), expected_current_generation=1
        )
        if mode == "finalize":
            two = _commit(initializer, _metadata(2), expected=1)
            assert two == candidate
    boundary = Boundary(
        prepared(candidate, committed=1, committed_state=one.state_fingerprint_sha256)
    )
    host = CoreHost(
        _scope(path),
        runtime_session_factory=Session,
        state_store_factory=lambda: SQLiteStateStore(path),
        startup_recovery_factory=lambda scope, store: _coordinator(
            path, store, boundary, scope=scope
        ),
    )
    host.start()
    assert host.startup_recovery_result is not None
    assert boundary.calls.count(mode) == 1
    assert boundary.calls.count("prepare") == 0
    host.close()


@pytest.mark.parametrize("iteration", range(10))
def test_real_m0_3_failure_cleanup_stress_through_core_host(tmp_path: Path, iteration: int) -> None:
    path = tmp_path / f"m03-failure-{iteration}.db"
    with SQLiteStateStore(path) as initializer:
        _commit(initializer, _metadata())
    sessions: list[Session] = []
    stores: list[CountingSQLiteStateStore] = []

    def session_factory() -> Session:
        session = Session()
        sessions.append(session)
        return session

    def store_factory() -> CountingSQLiteStateStore:
        store = CountingSQLiteStateStore(path)
        stores.append(store)
        return store

    host = CoreHost(
        _scope(path),
        runtime_session_factory=session_factory,
        state_store_factory=store_factory,
        startup_recovery_factory=lambda scope, store: _coordinator(
            path, store, Boundary(record("UNINITIALIZED")), scope=scope
        ),
    )
    with pytest.raises(Exception, match="ahead"):
        host.start()
    assert host.startup_recovery_result is None
    assert not host.owns_process_lock
    assert len(sessions) == len(stores) == 1
    assert sessions[0].close_count == 1
    assert stores[0].close_count == 1
    with CoreHostProcessLock(_scope(path)):
        pass


@pytest.mark.parametrize("tamper", ["lock", "session", "store"])
def test_core_host_rejects_topology_tampering_before_final_result(
    tmp_path: Path, tamper: str
) -> None:
    path = tmp_path / f"tamper-{tamper}.db"
    host: CoreHost | None = None
    original_sessions: list[Session] = []
    original_stores: list[Session] = []
    replacements: list[Session] = []

    def session_factory() -> Session:
        resource = Session()
        original_sessions.append(resource)
        return resource

    def store_factory() -> Session:
        resource = Session()
        original_stores.append(resource)
        return resource

    class TamperingRecovery:
        def recover(self) -> StartupSubsystemRecoveryResult:
            assert host is not None
            if tamper == "lock":
                assert host._lock is not None
                cast(Any, host)._lock = None
            elif tamper == "session":
                replacement = Session()
                replacements.append(replacement)
                cast(Any, host)._runtime_session = replacement
            else:
                replacement = Session()
                replacements.append(replacement)
                cast(Any, host)._state_store = replacement
            return StartupSubsystemRecoveryResult(
                StartupSubsystemRecoveryClassification.INITIALIZED_DURABLE_RECOVERY_RESOLVED
            )

    host = CoreHost(
        _scope(path),
        runtime_session_factory=session_factory,
        state_store_factory=store_factory,
        startup_recovery_factory=lambda scope, store: TamperingRecovery(),
    )
    with pytest.raises(RuntimeError, match="topology changed"):
        host.start()
    assert host.startup_recovery_result is None
    assert not host.owns_process_lock
    assert original_sessions[0].close_count == 1
    assert original_stores[0].close_count == 1
    assert all(replacement.close_count == 0 for replacement in replacements)
    with CoreHostProcessLock(_scope(path)):
        pass


def test_held_foreign_lock_cannot_replace_exact_start_attempt_lock(tmp_path: Path) -> None:
    canonical = _scope(tmp_path / "canonical.db")
    foreign = CoreHostProcessLock(_scope(tmp_path / "foreign.db"))
    host: CoreHost | None = None
    original: list[CoreHostProcessLock] = []

    def lock_factory(scope: CoreHostScope) -> CoreHostProcessLock:
        lock = CoreHostProcessLock(scope)
        original.append(lock)
        return lock

    class SwapLockRecovery:
        def recover(self) -> StartupSubsystemRecoveryResult:
            assert host is not None and host._lock is original[0]
            original[0].release()
            foreign.acquire()
            cast(Any, host)._lock = foreign
            assert host.owns_process_lock
            return StartupSubsystemRecoveryResult(
                StartupSubsystemRecoveryClassification.INITIALIZED_DURABLE_RECOVERY_RESOLVED
            )

    host = CoreHost(
        canonical,
        runtime_session_factory=Session,
        state_store_factory=Session,
        startup_recovery_factory=lambda scope, store: SwapLockRecovery(),
        lock_factory=lock_factory,
    )
    try:
        with pytest.raises(RuntimeError, match="topology changed"):
            host.start()
        assert host.startup_recovery_result is None
        assert not original[0].held
        assert foreign.held
        with CoreHostProcessLock(canonical):
            pass
    finally:
        foreign.release()


def test_initialized_high_level_stage_order_and_evidence_last(tmp_path: Path) -> None:
    path = tmp_path / "ordered.db"
    events: list[str] = []
    with SQLiteStateStore(path) as initializer:
        metadata = _commit(initializer, _metadata())
    boundary = Boundary(
        record(
            "COMMITTED",
            committed_generation=1,
            committed_state_fingerprint_sha256=metadata.state_fingerprint_sha256,
        )
    )

    class RecordingLock(CoreHostProcessLock):
        def acquire(self) -> None:
            super().acquire()
            events.append("lock_acquired")

    class RecordingStore(SQLiteStateStore):
        def __init__(self, value: Path) -> None:
            super().__init__(value)
            events.append("store_opened")
            self.reads = 0

        def read_verified_snapshot(self) -> Any:
            self.reads += 1
            if self.reads == 1:
                events.append("initial_verified_read")
            return super().read_verified_snapshot()

    class RecordingEvidence(LocalDurableEvidenceRegistry):
        def publish_verified_state(self, store: SQLiteStateStore) -> str | None:
            events.append("evidence_publication")
            return cast(str | None, super().publish_verified_state(store))

    def session_factory() -> Session:
        events.append("session_created")
        return Session()

    def recovery_factory(
        scope: CoreHostScope, store: RecordingStore
    ) -> CoreHostStartupRecoveryCoordinator:
        evidence = RecordingEvidence()
        protected_calls = 0

        class RecordingProtected(ProtectedFreshnessHandoffCoordinator):
            def recover_protected_state(self, protected_scope: Any) -> Any:
                nonlocal protected_calls
                protected_calls += 1
                events.append("m0_3_recovery" if protected_calls == 1 else "final_m0_3_check")
                return super().recover_protected_state(protected_scope)

        protected = RecordingProtected(store, evidence, boundary)
        migrations = MigrationRegistry()
        lifecycles = DurableMigrationLifecycleCoordinator(store, migrations, protected)
        execution = MigrationExecutionCoordinator(store, migrations, protected)
        completion = DurableMigrationCompletionCoordinator(store, execution, lifecycles, protected)
        secrets = DurableSecretHandoffExecutionCoordinator(
            store,
            DurableSecretHandoffLifecycleCoordinator(store, protected),
            protected,
            SecretPort(),
        )

        class RecordingCoordinator(CoreHostStartupRecoveryCoordinator):
            bind_count = 0

            def _bind(self, snapshot: Any) -> tuple[str, str, str]:
                self.bind_count += 1
                events.append(
                    "initial_scope_binding" if self.bind_count == 1 else "final_scope_binding"
                )
                return cast(tuple[str, str, str], super()._bind(snapshot))

            def _required_snapshot(self, stage: str) -> Any:
                labels = {
                    "protected recovery": "fresh_read_after_m0_3",
                    "migration recovery": "fresh_read_after_migration",
                    "secret recovery": "final_verified_read",
                }
                events.append(labels[stage])
                return super()._required_snapshot(stage)

            def _recover_migrations(self, snapshot: Any) -> None:
                events.append("migration_stage")
                super()._recover_migrations(snapshot)

            def _recover_secrets(self, snapshot: Any) -> None:
                events.append("secret_stage")
                super()._recover_secrets(snapshot)

            def _verify_physical_schema(self, snapshot: Any) -> None:
                events.append("physical_schema_gate")
                super()._verify_physical_schema(snapshot)

            def _validate_final_migrations(self, snapshot: Any) -> None:
                events.append("final_migration_rediscovery")
                super()._validate_final_migrations(snapshot)

            def _validate_final_secrets(self, snapshot: Any) -> None:
                events.append("final_secret_rediscovery")
                super()._validate_final_secrets(snapshot)

        return RecordingCoordinator(
            scope,
            store,
            protected,
            migrations,
            lifecycles,
            completion,
            secrets,
            evidence,
            StateStorePhysicalSchemaRegistry(),
        )

    host = CoreHost(
        _scope(path),
        runtime_session_factory=session_factory,
        state_store_factory=lambda: RecordingStore(path),
        startup_recovery_factory=recovery_factory,
        lock_factory=RecordingLock,
    )
    host.start()
    assert events == [
        "lock_acquired",
        "session_created",
        "store_opened",
        "initial_verified_read",
        "initial_scope_binding",
        "m0_3_recovery",
        "fresh_read_after_m0_3",
        "migration_stage",
        "fresh_read_after_migration",
        "secret_stage",
        "final_verified_read",
        "final_scope_binding",
        "physical_schema_gate",
        "final_m0_3_check",
        "final_migration_rediscovery",
        "final_secret_rediscovery",
        "evidence_publication",
    ]
    assert host.startup_recovery_result is not None
    assert events.count("m0_3_recovery") == 1
    assert events.count("final_m0_3_check") == 1
    assert events[-1] == "evidence_publication"
    host.close()


def test_repeated_real_start_has_exact_factory_recovery_and_evidence_counts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "repeated-real.db"
    with SQLiteStateStore(path) as initializer:
        metadata = _commit(initializer, _metadata())
    boundary = Boundary(
        record(
            "COMMITTED",
            committed_generation=1,
            committed_state_fingerprint_sha256=metadata.state_fingerprint_sha256,
        )
    )
    counts = {"session": 0, "store": 0, "factory": 0, "recover": 0, "evidence": 0}

    class CountingEvidence(LocalDurableEvidenceRegistry):
        def publish_verified_state(self, store: SQLiteStateStore) -> str | None:
            counts["evidence"] += 1
            return cast(str | None, super().publish_verified_state(store))

    evidence = CountingEvidence()

    class CountingRecovery:
        def __init__(self, delegate: CoreHostStartupRecoveryCoordinator) -> None:
            self.delegate = delegate

        def recover(self) -> StartupSubsystemRecoveryResult:
            counts["recover"] += 1
            return self.delegate.recover()

    def session_factory() -> Session:
        counts["session"] += 1
        return Session()

    def store_factory() -> SQLiteStateStore:
        counts["store"] += 1
        return SQLiteStateStore(path)

    def recovery_factory(scope: CoreHostScope, store: SQLiteStateStore) -> CountingRecovery:
        counts["factory"] += 1
        return CountingRecovery(_coordinator(path, store, boundary, scope=scope, evidence=evidence))

    host = CoreHost(
        _scope(path),
        runtime_session_factory=session_factory,
        state_store_factory=store_factory,
        startup_recovery_factory=recovery_factory,
    )
    host.start()
    host.start()
    assert counts == {"session": 1, "store": 1, "factory": 1, "recover": 1, "evidence": 1}
    host.close()


@pytest.mark.parametrize(
    ("initial_state", "expected_calls", "succeeds"),
    [
        ("PREPARED", {"begin": 0, "reconcile": 1, "cleanup": 1}, True),
        ("COMMITTED", {"begin": 0, "reconcile": 0, "cleanup": 1}, True),
        ("CLEANUP_PENDING", {"begin": 0, "reconcile": 0, "cleanup": 0}, True),
        ("UNKNOWN_RECONCILIATION", {"begin": 0, "reconcile": 0, "cleanup": 0}, False),
    ],
)
@pytest.mark.parametrize("_iteration", range(10))
def test_real_secret_states_recover_through_core_host(
    tmp_path: Path,
    initial_state: str,
    expected_calls: dict[str, int],
    succeeds: bool,
    _iteration: int,
) -> None:
    path = tmp_path / f"secret-{initial_state}.db"
    boundary = Boundary(record("UNINITIALIZED"))
    port = Port(reconcile=ExternalOutcome.COMMITTED)
    handoff = descriptor(scope=(_metadata().account_id, _metadata().device_installation_id))
    with SQLiteStateStore(path) as initializer:
        ProtectedFreshnessHandoffCoordinator(
            initializer, LocalDurableEvidenceRegistry(), boundary
        ).advance_protected_state(_metadata())
        protected = ProtectedFreshnessHandoffCoordinator(
            initializer, LocalDurableEvidenceRegistry(), boundary
        )
        lifecycle = DurableSecretHandoffLifecycleCoordinator(initializer, protected)
        lifecycle.prepare(handoff)
        if initial_state == "PREPARED":
            port.initial_effects.add(handoff.handoff_id)
        elif initial_state in {"COMMITTED", "CLEANUP_PENDING"}:
            lifecycle.record_external_outcome(handoff.handoff_id, ExternalOutcome.COMMITTED)
            if initial_state == "CLEANUP_PENDING":
                lifecycle.mark_cleanup_pending(handoff.handoff_id)
        elif initial_state == "UNKNOWN_RECONCILIATION":
            lifecycle.record_external_outcome(handoff.handoff_id, ExternalOutcome.UNRESOLVED)

    host = CoreHost(
        _scope(path),
        runtime_session_factory=Session,
        state_store_factory=lambda: SQLiteStateStore(path),
        startup_recovery_factory=lambda scope, store: _coordinator(
            path, store, boundary, scope=scope, secret_port=port
        ),
    )
    if succeeds:
        host.start()
        assert host.startup_recovery_result is not None
        owned_store = cast(SQLiteStateStore, host._state_store)
        final_lifecycle = DurableSecretHandoffLifecycleCoordinator(
            owned_store,
            ProtectedFreshnessHandoffCoordinator(
                owned_store, LocalDurableEvidenceRegistry(), boundary
            ),
        ).discover(handoff.handoff_id)
        assert final_lifecycle.current is not None
        assert final_lifecycle.current["state"] == "CLEANUP_PENDING"
        host.close()
    else:
        with pytest.raises(CoreHostStartupRecoveryError, match="unknown"):
            host.start()
        assert host.startup_recovery_result is None
        assert not host.owns_process_lock
    assert {name: port.count(name) for name in expected_calls} == expected_calls


def test_three_real_secret_handoffs_process_in_canonical_id_order(tmp_path: Path) -> None:
    path = tmp_path / "three-secrets.db"
    boundary = Boundary(record("UNINITIALIZED"))
    port = Port(reconcile=ExternalOutcome.COMMITTED)
    creation_order = ("handoff-c", "handoff-a", "handoff-b")
    with SQLiteStateStore(path) as initializer:
        ProtectedFreshnessHandoffCoordinator(
            initializer, LocalDurableEvidenceRegistry(), boundary
        ).advance_protected_state(_metadata())
        protected = ProtectedFreshnessHandoffCoordinator(
            initializer, LocalDurableEvidenceRegistry(), boundary
        )
        lifecycle = DurableSecretHandoffLifecycleCoordinator(initializer, protected)
        for handoff_id in creation_order:
            item = descriptor(
                handoff_id=handoff_id,
                scope=(_metadata().account_id, _metadata().device_installation_id),
            )
            lifecycle.prepare(item)
            port.initial_effects.add(handoff_id)
    host = CoreHost(
        _scope(path),
        runtime_session_factory=Session,
        state_store_factory=lambda: SQLiteStateStore(path),
        startup_recovery_factory=lambda scope, store: _coordinator(
            path, store, boundary, scope=scope, secret_port=port
        ),
    )
    host.start()
    observed = [mapping["handoff_id"] for action, mapping in port.calls if action == "reconcile"]
    assert observed == ["handoff-a", "handoff-b", "handoff-c"]
    assert host.startup_recovery_result is not None
    host.close()


def test_final_secret_rediscovery_rejects_late_real_prepared_family(tmp_path: Path) -> None:
    path = tmp_path / "late-secret.db"
    boundary = Boundary(record("UNINITIALIZED"))
    with SQLiteStateStore(path) as store:
        metadata = _commit(store, _metadata())
        boundary = Boundary(
            record(
                "COMMITTED",
                committed_generation=1,
                committed_state_fingerprint_sha256=metadata.state_fingerprint_sha256,
            )
        )
        base = _coordinator(path, store, boundary)
        late = descriptor(
            handoff_id="handoff-late",
            scope=(_metadata().account_id, _metadata().device_installation_id),
        )

        class LateSecretCoordinator(CoreHostStartupRecoveryCoordinator):
            def _recover_secrets(self, snapshot: Any) -> None:
                super()._recover_secrets(snapshot)
                lifecycle = DurableSecretHandoffLifecycleCoordinator(store, self._protected)
                lifecycle.prepare(late)

        coordinator = LateSecretCoordinator(
            base._scope,
            store,
            base._protected,
            base._migrations,
            base._migration_lifecycles,
            base._migration_completion,
            base._secret_execution,
            base._evidence,
            base._physical_schemas,
        )
        with pytest.raises(CoreHostStartupRecoveryError, match="non-terminal secret"):
            coordinator.recover()


class FuturePhysicalSchemas:
    __slots__ = ("_fingerprints",)

    def __init__(self, source: str, target: str) -> None:
        self._fingerprints = (source, target)

    @property
    def current_version(self) -> int:
        return 2

    def expected_fingerprint(self, version: int) -> str:
        if version not in {1, 2}:
            raise StateStorePhysicalSchemaError("unknown test schema")
        return self._fingerprints[version - 1]


class ThreeVersionPhysicalSchemas:
    __slots__ = ("_fingerprints",)

    def __init__(self, one: str, two: str, three: str) -> None:
        self._fingerprints = (one, two, three)

    @property
    def current_version(self) -> int:
        return 3

    def expected_fingerprint(self, version: int) -> str:
        return self._fingerprints[version - 1]


class CurrentOneWithFutureSchema(FuturePhysicalSchemas):
    @property
    def current_version(self) -> int:
        return 1


@pytest.mark.parametrize("initial_state", ["PREPARED", "APPLYING"])
def test_real_pre_materialization_migration_completes_through_core_host(
    tmp_path: Path, initial_state: str
) -> None:
    path = tmp_path / f"migration-{initial_state}.db"
    boundary = Boundary(record("UNINITIALIZED"))
    operation = _operation()
    with SQLiteStateStore(path) as initializer:
        states = ("PREPARED",) if initial_state == "PREPARED" else ("PREPARED", "APPLYING")
        _initialize_with_migration(initializer, boundary, states, schema=1)
        source = initializer.sqlite_schema_fingerprint()
        target = _target_fingerprint(initializer, (operation,))
        definition, migrations, calls = _registry(initializer, (operation,), target=target)
    host = CoreHost(
        _scope(path),
        runtime_session_factory=Session,
        state_store_factory=lambda: SQLiteStateStore(path),
        startup_recovery_factory=lambda scope, store: _coordinator(
            path,
            store,
            boundary,
            scope=scope,
            migrations=migrations,
            physical_schemas=FuturePhysicalSchemas(source, target),
        ),
    )
    host.start()
    assert host.startup_recovery_result is not None
    assert len(calls) == 1
    owned_store = cast(SQLiteStateStore, host._state_store)
    protected = ProtectedFreshnessHandoffCoordinator(
        owned_store, LocalDurableEvidenceRegistry(), boundary
    )
    lifecycle = DurableMigrationLifecycleCoordinator(owned_store, migrations, protected).discover(
        definition.migration_id
    )
    assert lifecycle.current is not None and lifecycle.current["state"] == "COMPLETED"
    host.close()


@pytest.mark.parametrize(
    "initial_state", ["APPLYING_MATERIALIZED", "DURABLE_MIGRATED", "COMPLETED"]
)
@pytest.mark.parametrize("_iteration", range(10))
def test_real_materialized_migration_never_replays_sql_through_core_host(
    tmp_path: Path, initial_state: str, _iteration: int
) -> None:
    path = tmp_path / f"materialized-{initial_state}.db"
    boundary = Boundary(record("UNINITIALIZED"))
    operation = _operation()
    with SQLiteStateStore(path) as initializer:
        _initialize_with_migration(initializer, boundary, ("PREPARED", "APPLYING"), schema=1)
        source = initializer.sqlite_schema_fingerprint()
        target = _target_fingerprint(initializer, (operation,))
        definition, migrations, calls = _registry(initializer, (operation,), target=target)
        protected = ProtectedFreshnessHandoffCoordinator(
            initializer, LocalDurableEvidenceRegistry(), boundary
        )
        proof = MigrationExecutionCoordinator(initializer, migrations, protected).execute(
            definition.migration_id
        )
        lifecycles = DurableMigrationLifecycleCoordinator(initializer, migrations, protected)
        if initial_state in {"DURABLE_MIGRATED", "COMPLETED"}:
            lifecycles.record_durable_migrated(definition.migration_id, proof)
        if initial_state == "COMPLETED":
            lifecycles.complete(definition.migration_id)
        calls.clear()
    host = CoreHost(
        _scope(path),
        runtime_session_factory=Session,
        state_store_factory=lambda: SQLiteStateStore(path),
        startup_recovery_factory=lambda scope, store: _coordinator(
            path,
            store,
            boundary,
            scope=scope,
            migrations=migrations,
            physical_schemas=FuturePhysicalSchemas(source, target),
        ),
    )
    host.start()
    assert calls == []
    assert host.startup_recovery_result is not None
    host.close()


@pytest.mark.parametrize("authority_case", ["failed", "missing", "physical-mismatch"])
def test_real_migration_authority_failures_block_top_level_startup(
    tmp_path: Path, authority_case: str
) -> None:
    path = tmp_path / f"migration-failure-{authority_case}.db"
    boundary = Boundary(record("UNINITIALIZED"))
    with SQLiteStateStore(path) as initializer:
        states = ("PREPARED", "FAILED") if authority_case == "failed" else ("PREPARED",)
        _initialize_with_migration(initializer, boundary, states, schema=1)
        source = initializer.sqlite_schema_fingerprint()
        target = _target_fingerprint(initializer, (_operation(),))
        _definition, sealed, calls = _registry(initializer, (_operation(),), target=target)
    migrations = MigrationRegistry() if authority_case == "missing" else sealed
    schemas = FuturePhysicalSchemas(
        "f" * 64 if authority_case == "physical-mismatch" else source,
        target,
    )
    host = CoreHost(
        _scope(path),
        runtime_session_factory=Session,
        state_store_factory=lambda: SQLiteStateStore(path),
        startup_recovery_factory=lambda scope, store: _coordinator(
            path,
            store,
            boundary,
            scope=scope,
            migrations=migrations,
            physical_schemas=schemas,
        ),
    )
    with pytest.raises(Exception):
        host.start()
    assert calls == []
    assert host.startup_recovery_result is None
    assert not host.owns_process_lock


def test_two_real_migrations_follow_schema_chain_not_lexical_ids(tmp_path: Path) -> None:
    path = tmp_path / "two-migrations.db"
    boundary = Boundary(record("UNINITIALIZED"))
    first_id, second_id = "migration-z-v1-v2", "migration-a-v2-v3"
    first_op = MigrationSqlOperation(1, "create-first", "DDL", "CREATE TABLE first_v2(id INTEGER)")
    second_op = MigrationSqlOperation(
        1, "create-second", "DDL", "CREATE TABLE second_v3(id INTEGER)"
    )
    with SQLiteStateStore(path) as initializer:
        ProtectedFreshnessHandoffCoordinator(
            initializer, LocalDurableEvidenceRegistry(), boundary
        ).advance_protected_state(_metadata())
        one = initializer.sqlite_schema_fingerprint()
        two = _target_fingerprint(initializer, (first_op,))
        three = _target_fingerprint(initializer, (first_op, second_op))
        definitions = (
            MigrationDefinition(first_id, 1, 2, ("first",)),
            MigrationDefinition(second_id, 2, 3, ("second",)),
        )
        plans = (
            MigrationExecutionPlan((first_op,), one, two),
            MigrationExecutionPlan((second_op,), two, three),
        )
        entries = []
        for definition, plan in zip(definitions, plans):
            authority = MigrationExecutionAuthority(
                definition.migration_id,
                definition.source_schema_version,
                definition.target_schema_version,
                definition.ordered_path,
                definition.rollback_policy,
                definition.fingerprint(),
                plan.operation_plan_fingerprint_sha256,
                plan.pre_sqlite_schema_fingerprint_sha256,
                plan.target_sqlite_schema_fingerprint_sha256,
            )
            entries.append((definition, authority, lambda snapshot, value=plan: value))
        migrations = MigrationRegistry(tuple(entries))
        protected = ProtectedFreshnessHandoffCoordinator(
            initializer, LocalDurableEvidenceRegistry(), boundary
        )
        lifecycle = DurableMigrationLifecycleCoordinator(initializer, migrations, protected)
        lifecycle.prepare(first_id)
        lifecycle.begin_applying(first_id)
        proof = MigrationExecutionCoordinator(initializer, migrations, protected).execute(first_id)
        lifecycle.record_durable_migrated(first_id, proof)
        lifecycle.complete(first_id)
        lifecycle.prepare(second_id)

    observed: list[str] = []

    def recovery_factory(
        scope: CoreHostScope, store: SQLiteStateStore
    ) -> CoreHostStartupRecoveryCoordinator:
        evidence = LocalDurableEvidenceRegistry()
        protected = ProtectedFreshnessHandoffCoordinator(store, evidence, boundary)

        class RecordingLifecycle(DurableMigrationLifecycleCoordinator):
            def discover(self, migration_id: str) -> Any:
                observed.append(migration_id)
                return super().discover(migration_id)

        lifecycles = RecordingLifecycle(store, migrations, protected)
        execution = MigrationExecutionCoordinator(store, migrations, protected)
        completion = DurableMigrationCompletionCoordinator(store, execution, lifecycles, protected)
        secrets = DurableSecretHandoffExecutionCoordinator(
            store,
            DurableSecretHandoffLifecycleCoordinator(store, protected),
            protected,
            SecretPort(),
        )
        return CoreHostStartupRecoveryCoordinator(
            scope,
            store,
            protected,
            migrations,
            lifecycles,
            completion,
            secrets,
            evidence,
            cast(Any, ThreeVersionPhysicalSchemas(one, two, three)),
        )

    host = CoreHost(
        _scope(path),
        runtime_session_factory=Session,
        state_store_factory=lambda: SQLiteStateStore(path),
        startup_recovery_factory=recovery_factory,
    )
    host.start()
    first_distinct = list(dict.fromkeys(observed))
    assert first_distinct[:2] == [first_id, second_id]
    owned = cast(SQLiteStateStore, host._state_store)
    assert owned.read_metadata().state_store_schema_version == 3
    assert owned.sqlite_schema_fingerprint() == three
    assert host.startup_recovery_result is not None
    host.close()


def test_final_migration_rediscovery_rejects_late_real_prepared_family(tmp_path: Path) -> None:
    path = tmp_path / "late-migration.db"
    boundary = Boundary(record("UNINITIALIZED"))
    with SQLiteStateStore(path) as store:
        ProtectedFreshnessHandoffCoordinator(
            store, LocalDurableEvidenceRegistry(), boundary
        ).advance_protected_state(_metadata())
        source = store.sqlite_schema_fingerprint()
        target = _target_fingerprint(store, (_operation(),))
        definition, migrations, _calls = _registry(store, (_operation(),), target=target)
        evidence = LocalDurableEvidenceRegistry()
        protected = ProtectedFreshnessHandoffCoordinator(store, evidence, boundary)
        lifecycles = DurableMigrationLifecycleCoordinator(store, migrations, protected)
        execution = MigrationExecutionCoordinator(store, migrations, protected)
        completion = DurableMigrationCompletionCoordinator(store, execution, lifecycles, protected)
        secrets = DurableSecretHandoffExecutionCoordinator(
            store,
            DurableSecretHandoffLifecycleCoordinator(store, protected),
            protected,
            SecretPort(),
        )

        class LateMigrationCoordinator(CoreHostStartupRecoveryCoordinator):
            def _recover_secrets(self, snapshot: Any) -> None:
                super()._recover_secrets(snapshot)
                lifecycles.prepare(definition.migration_id)

        coordinator = LateMigrationCoordinator(
            _scope(path),
            store,
            protected,
            migrations,
            lifecycles,
            completion,
            secrets,
            evidence,
            cast(Any, CurrentOneWithFutureSchema(source, target)),
        )
        with pytest.raises(Exception, match="non-terminal migration"):
            coordinator.recover()


def _migration_shape_registry(
    definitions: tuple[MigrationDefinition, ...], fingerprint: str, calls: list[str]
) -> MigrationRegistry:
    entries = []
    for definition in definitions:
        plan = MigrationExecutionPlan(
            (MigrationSqlOperation(1, "shape-noop", "DML", "SELECT 1"),),
            fingerprint,
            fingerprint,
        )
        authority = MigrationExecutionAuthority(
            definition.migration_id,
            definition.source_schema_version,
            definition.target_schema_version,
            definition.ordered_path,
            definition.rollback_policy,
            definition.fingerprint(),
            plan.operation_plan_fingerprint_sha256,
            fingerprint,
            fingerprint,
        )

        def planner(
            snapshot: Any,
            *,
            value: MigrationExecutionPlan = plan,
            name: str = definition.migration_id,
        ) -> MigrationExecutionPlan:
            calls.append(name)
            return value

        entries.append((definition, authority, planner))
    return MigrationRegistry(tuple(entries))


@pytest.mark.parametrize("shape", ["branch", "cycle"])
def test_representable_invalid_migration_chain_fails_top_level_before_sql(
    tmp_path: Path, shape: str
) -> None:
    path = tmp_path / f"invalid-{shape}.db"
    boundary = Boundary(record("UNINITIALIZED"))
    definitions = (
        MigrationDefinition("shape-first", 1, 2, ("first",)),
        MigrationDefinition(
            "shape-second",
            1 if shape == "branch" else 2,
            3 if shape == "branch" else 1,
            ("second",),
        ),
    )
    calls: list[str] = []
    with SQLiteStateStore(path) as initializer:
        ProtectedFreshnessHandoffCoordinator(
            initializer, LocalDurableEvidenceRegistry(), boundary
        ).advance_protected_state(_metadata())
        fingerprint = initializer.sqlite_schema_fingerprint()
        migrations = _migration_shape_registry(definitions, fingerprint, calls)
        protected = ProtectedFreshnessHandoffCoordinator(
            initializer, LocalDurableEvidenceRegistry(), boundary
        )
        lifecycle = DurableMigrationLifecycleCoordinator(initializer, migrations, protected)
        lifecycle.prepare("shape-first")
        if shape == "branch":
            lifecycle.prepare("shape-second")
        else:
            lifecycle.begin_applying("shape-first")
            proof = MigrationExecutionCoordinator(initializer, migrations, protected).execute(
                "shape-first"
            )
            lifecycle.record_durable_migrated("shape-first", proof)
            lifecycle.complete("shape-first")
            lifecycle.prepare("shape-second")
            calls.clear()
    host = CoreHost(
        _scope(path),
        runtime_session_factory=Session,
        state_store_factory=lambda: SQLiteStateStore(path),
        startup_recovery_factory=lambda scope, store: _coordinator(
            path,
            store,
            boundary,
            scope=scope,
            migrations=migrations,
            physical_schemas=cast(Any, FuturePhysicalSchemas(fingerprint, fingerprint)),
        ),
    )
    with pytest.raises(Exception, match="branches|unique head|cyclic"):
        host.start()
    assert calls == []
    assert host.startup_recovery_result is None


def test_disconnected_family_is_rejected_by_public_lifecycle_source_gate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid-disconnected.db"
    boundary = Boundary(record("UNINITIALIZED"))
    definitions = (
        MigrationDefinition("connected", 1, 2, ("first",)),
        MigrationDefinition("disconnected", 3, 4, ("second",)),
    )
    calls: list[str] = []
    with SQLiteStateStore(path) as store:
        ProtectedFreshnessHandoffCoordinator(
            store, LocalDurableEvidenceRegistry(), boundary
        ).advance_protected_state(_metadata())
        migrations = _migration_shape_registry(
            definitions, store.sqlite_schema_fingerprint(), calls
        )
        lifecycle = DurableMigrationLifecycleCoordinator(
            store,
            migrations,
            ProtectedFreshnessHandoffCoordinator(store, LocalDurableEvidenceRegistry(), boundary),
        )
        with pytest.raises(Exception, match="PREPARED requires verified StateStore schema 3"):
            lifecycle.prepare("disconnected")
    assert calls == []
