from __future__ import annotations

import threading
from pathlib import Path

import pytest

from bot_core.persistence.local_durable_evidence import LocalDurableEvidenceRegistry
from bot_core.persistence.migration_completion import DurableMigrationCompletionCoordinator
from bot_core.persistence.migration_execution import MigrationSqlOperation
from bot_core.persistence.migration_execution_engine import MigrationExecutionCoordinator
from bot_core.persistence.migration_protocol import (
    DurableMigrationLifecycleCoordinator,
    MigrationError,
)
from bot_core.persistence.protected_freshness_handoff import (
    ProtectedFreshnessHandoffCoordinator,
)
from bot_core.persistence.state_store import SQLiteStateStore
from tests.persistence.test_durable_lifecycle_cas import _initialize_with_migration
from tests.persistence.test_migration_execution_engine import _registry, _target_fingerprint
from tests.persistence.test_protected_freshness_handoff import Boundary, record


def _coordinators(store: SQLiteStateStore, registry, boundary: Boundary):
    protected = ProtectedFreshnessHandoffCoordinator(
        store, LocalDurableEvidenceRegistry(), boundary
    )
    lifecycles = DurableMigrationLifecycleCoordinator(store, registry, protected)
    execution = MigrationExecutionCoordinator(store, registry, protected)
    return DurableMigrationCompletionCoordinator(
        store, execution, lifecycles, protected
    ), lifecycles


def _operation() -> MigrationSqlOperation:
    return MigrationSqlOperation(1, "create-a", "DDL", "CREATE TABLE a(id INTEGER)")


class _ForbiddenExecution:
    def execute(self, migration_id: str):
        raise AssertionError(f"structural executor called for {migration_id}")


def _declarations(snapshot):
    return tuple(
        item
        for item in snapshot.immutable_history
        if item.representation_name == "Migration execution declaration"
    )


def _transitions(lifecycle, state: str):
    return tuple(item for item in lifecycle.history if item["state"] == state)


def _synchronize_boundary(boundary: Boundary) -> None:
    lock = threading.RLock()
    for name in ("resolve_current", "prepare", "finalize", "abort"):
        original = getattr(boundary, name)

        def synchronized(*args, _original=original, **kwargs):
            with lock:
                return _original(*args, **kwargs)

        setattr(boundary, name, synchronized)


def test_full_completion_has_three_transactions_and_exact_retry_is_read_only(
    tmp_path: Path,
) -> None:
    boundary = Boundary(record("UNINITIALIZED"))
    with SQLiteStateStore(tmp_path / "state.db") as store:
        _initialize_with_migration(store, boundary, ("PREPARED", "APPLYING"), schema=1)
        definition, registry, calls = _registry(store, (_operation(),))
        coordinator, _ = _coordinators(store, registry, boundary)
        before = store.read_verified_snapshot()

        result = coordinator.resume_to_completion(definition.migration_id)
        after = store.read_verified_snapshot()

        assert result.current is not None and result.current["state"] == "COMPLETED"
        assert after.metadata.protected_freshness_generation == (
            before.metadata.protected_freshness_generation + 3
        )
        assert len(after.transaction_descriptors) == len(before.transaction_descriptors) + 3
        assert [item["state"] for item in result.history] == [
            "PREPARED",
            "APPLYING",
            "DURABLE_MIGRATED",
            "COMPLETED",
        ]
        assert [item["transition_revision"] for item in result.history] == [1, 2, 3, 4]
        assert calls == [before.metadata.protected_freshness_generation]
        declarations = _declarations(after)
        assert len(declarations) == 1
        durable_names = {
            item.representation_name for item in (*after.current_records, *after.immutable_history)
        }
        assert durable_names <= {
            "CryptoHunterAccount current record",
            "Migration current state/designation",
            "Migration transition/history revisions",
            "Migration execution declaration",
        }
        assert all(
            item.record_key != f"migration-record:{definition.migration_id}"
            for item in (*after.current_records, *after.immutable_history)
        )
        structural, durable, completed = after.transaction_descriptors[-3:]
        assert structural.immutable_history_appends == declarations
        assert declarations[0] not in durable.current_record_mutations
        assert declarations[0] not in completed.current_record_mutations
        assert all(
            item.representation_name == "Migration transition/history revisions"
            for descriptor in (durable, completed)
            for item in descriptor.immutable_history_appends
        )

        assert coordinator.resume_to_completion(definition.migration_id) == result
        assert store.read_verified_snapshot() == after
        assert calls == [before.metadata.protected_freshness_generation]


def test_structural_durable_applying_reopens_without_planner_or_sql(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    boundary = Boundary(record("UNINITIALIZED"))
    operation = _operation()
    with SQLiteStateStore(path) as store:
        _initialize_with_migration(store, boundary, ("PREPARED", "APPLYING"), schema=1)
        definition, registry, _ = _registry(store, (operation,))
        target = _target_fingerprint(store, (operation,))
        MigrationExecutionCoordinator(
            store,
            registry,
            ProtectedFreshnessHandoffCoordinator(store, LocalDurableEvidenceRegistry(), boundary),
        ).execute(definition.migration_id)
        generation = store.read_metadata().protected_freshness_generation

    with SQLiteStateStore(path) as reopened:
        definition, registry, calls = _registry(reopened, (operation,), target=target)
        coordinator, _ = _coordinators(reopened, registry, boundary)
        result = coordinator.resume_to_completion(definition.migration_id)
        assert result.current["state"] == "COMPLETED"
        assert calls == []
        assert reopened.read_metadata().protected_freshness_generation == generation + 2
        assert reopened._connection.execute("SELECT count(*) FROM a").fetchone() == (0,)


@pytest.mark.parametrize("ack_lost", [False, True])
def test_durable_migrated_finalize_failure_recovers_before_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ack_lost: bool
) -> None:
    path = tmp_path / f"state-{ack_lost}.db"
    boundary = Boundary(record("UNINITIALIZED"))
    operation = _operation()
    with SQLiteStateStore(path) as store:
        _initialize_with_migration(store, boundary, ("PREPARED", "APPLYING"), schema=1)
        definition, registry, _ = _registry(store, (operation,))
        target = _target_fingerprint(store, (operation,))
        protected = ProtectedFreshnessHandoffCoordinator(
            store, LocalDurableEvidenceRegistry(), boundary
        )
        proof = MigrationExecutionCoordinator(store, registry, protected).execute(
            definition.migration_id
        )
        if ack_lost:
            boundary.ack_lost = True
        else:
            boundary.fail_finalize = True
        with pytest.raises(MigrationError, match="protected migration CAS failed"):
            DurableMigrationLifecycleCoordinator(
                store, registry, protected
            ).record_durable_migrated(definition.migration_id, proof)
        generation = store.read_metadata().protected_freshness_generation
        assert boundary.value["lifecycle"] == ("COMMITTED" if ack_lost else "PREPARED")

    boundary.ack_lost = boundary.fail_finalize = False
    events: list[str] = []
    original_finalize, original_prepare = boundary.finalize, boundary.prepare

    def finalize(*args, **kwargs):
        events.append("external-finalize")
        return original_finalize(*args, **kwargs)

    def prepare(*args, **kwargs):
        events.append("prepare")
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(boundary, "finalize", finalize)
    monkeypatch.setattr(boundary, "prepare", prepare)
    with SQLiteStateStore(path) as reopened:
        definition, registry, calls = _registry(reopened, (operation,), target=target)
        protected = ProtectedFreshnessHandoffCoordinator(
            reopened, LocalDurableEvidenceRegistry(), boundary
        )
        original_recover = protected.recover_protected_state

        def recover(scope):
            events.append("recover")
            return original_recover(scope)

        monkeypatch.setattr(protected, "recover_protected_state", recover)
        coordinator = DurableMigrationCompletionCoordinator(
            reopened,
            MigrationExecutionCoordinator(reopened, registry, protected),
            DurableMigrationLifecycleCoordinator(reopened, registry, protected),
            protected,
        )
        result = coordinator.resume_to_completion(definition.migration_id)
        assert result.current["state"] == "COMPLETED"
        assert calls == []
        assert reopened.read_metadata().protected_freshness_generation == generation + 1
        assert [item["state"] for item in result.history].count("DURABLE_MIGRATED") == 1
        first_prepare = events.index("prepare")
        assert events.index("recover") < first_prepare
        if not ack_lost:
            assert events.index("external-finalize") < first_prepare


def test_completed_pending_finalize_is_recovered_before_terminal_return(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    boundary = Boundary(record("UNINITIALIZED"))
    with SQLiteStateStore(path) as store:
        _initialize_with_migration(store, boundary, ("PREPARED", "APPLYING"), schema=1)
        definition, registry, _ = _registry(store, (_operation(),))
        coordinator, lifecycles = _coordinators(store, registry, boundary)
        proof = coordinator._execution.execute(definition.migration_id)
        lifecycles.record_durable_migrated(definition.migration_id, proof)
        boundary.fail_finalize = True
        with pytest.raises(MigrationError):
            lifecycles.complete(definition.migration_id)
        generation = store.read_metadata().protected_freshness_generation
        assert boundary.value["lifecycle"] == "PREPARED"

    boundary.fail_finalize = False
    with SQLiteStateStore(path) as reopened:
        definition, registry, calls = _registry(
            reopened, (_operation(),), target=reopened.sqlite_schema_fingerprint()
        )
        coordinator, _ = _coordinators(reopened, registry, boundary)
        assert coordinator.resume_to_completion(definition.migration_id).current["state"] == (
            "COMPLETED"
        )
        assert reopened.read_metadata().protected_freshness_generation == generation
        assert calls == []


@pytest.mark.parametrize("state", ["PREPARED", "FAILED"])
def test_non_owned_and_failed_states_never_mutate(tmp_path: Path, state: str) -> None:
    boundary = Boundary(record("UNINITIALIZED"))
    schema = 1
    with SQLiteStateStore(tmp_path / f"{state}.db") as store:
        states = ("PREPARED",) if state == "PREPARED" else ("PREPARED", "FAILED")
        _initialize_with_migration(store, boundary, states, schema=schema)
        definition, registry, calls = _registry(store, (_operation(),))
        coordinator, _ = _coordinators(store, registry, boundary)
        before = store.read_verified_snapshot()
        if state == "PREPARED":
            with pytest.raises(MigrationError, match="does not own"):
                coordinator.resume_to_completion(definition.migration_id)
        else:
            assert (
                coordinator.resume_to_completion(definition.migration_id).current["state"] == state
            )
        assert store.read_verified_snapshot() == before
        assert calls == []
        assert store._connection.execute(
            "SELECT count(*) FROM sqlite_schema WHERE name='a'"
        ).fetchone() == (0,)


def test_two_connection_full_orchestration_race_has_one_durable_winner(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    boundary = Boundary(record("UNINITIALIZED"))
    _synchronize_boundary(boundary)
    operation = _operation()
    with SQLiteStateStore(path) as store:
        _initialize_with_migration(store, boundary, ("PREPARED", "APPLYING"), schema=1)
        target = _target_fingerprint(store, (operation,))
        before = store.read_verified_snapshot()

    start = threading.Barrier(2)
    outcomes: list[object] = []

    def contend() -> None:
        try:
            with SQLiteStateStore(path) as store:
                definition, registry, _ = _registry(store, (operation,), target=target)
                coordinator, _ = _coordinators(store, registry, boundary)
                start.wait()
                outcomes.append(coordinator.resume_to_completion(definition.migration_id))
        except Exception as exc:  # a protected/CAS race loser must fail closed
            outcomes.append(exc)

    threads = [threading.Thread(target=contend) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert len(outcomes) == 2
    with SQLiteStateStore(path) as store:
        definition, registry, calls = _registry(store, (operation,), target=target)
        coordinator, _ = _coordinators(store, registry, boundary)
        final = coordinator.resume_to_completion(definition.migration_id)
        snapshot = store.read_verified_snapshot()
        assert final.current["state"] == "COMPLETED"
        assert [item["state"] for item in final.history] == [
            "PREPARED",
            "APPLYING",
            "DURABLE_MIGRATED",
            "COMPLETED",
        ]
        assert snapshot.metadata.protected_freshness_generation == (
            before.metadata.protected_freshness_generation + 3
        )
        assert len(snapshot.transaction_descriptors) == len(before.transaction_descriptors) + 3
        assert calls == []
        assert (
            sum(
                item.representation_name == "Migration execution declaration"
                for item in snapshot.immutable_history
            )
            == 1
        )
        assert [item["transition_revision"] for item in final.history] == [1, 2, 3, 4]
        assert final.current["current_transition_revision"] == 4
        assert len(_transitions(final, "DURABLE_MIGRATED")) == 1
        assert len(_transitions(final, "COMPLETED")) == 1
        unchanged = store.read_verified_snapshot()
        assert coordinator.resume_to_completion(definition.migration_id) == final
        assert store.read_verified_snapshot() == unchanged


def test_durable_migrated_prepare_without_local_commit_recovers_before_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "dm-local-absent.db"
    boundary = Boundary(record("UNINITIALIZED"))
    operation = _operation()
    with SQLiteStateStore(path) as store:
        _initialize_with_migration(store, boundary, ("PREPARED", "APPLYING"), schema=1)
        definition, registry, _ = _registry(store, (operation,))
        target = _target_fingerprint(store, (operation,))
        protected = ProtectedFreshnessHandoffCoordinator(
            store, LocalDurableEvidenceRegistry(), boundary
        )
        proof = MigrationExecutionCoordinator(store, registry, protected).execute(
            definition.migration_id
        )
        store._connection.execute(
            "CREATE TRIGGER fail_dm BEFORE INSERT ON state_store_immutable_history "
            "BEGIN SELECT RAISE(ABORT, 'dm local absent'); END"
        )
        with pytest.raises(MigrationError):
            DurableMigrationLifecycleCoordinator(
                store, registry, protected
            ).record_durable_migrated(definition.migration_id, proof)
        store._connection.execute("DROP TRIGGER fail_dm")
        assert boundary.value["lifecycle"] == "PREPARED"
        assert (
            DurableMigrationLifecycleCoordinator(store, registry, protected)
            .discover(definition.migration_id)
            .current["state"]
            == "APPLYING"
        )

    events: list[str] = []
    original_abort, original_prepare = boundary.abort, boundary.prepare

    def abort(*args, **kwargs):
        events.append("recover-abort")
        return original_abort(*args, **kwargs)

    def prepare(*args, **kwargs):
        events.append("prepare")
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(boundary, "abort", abort)
    monkeypatch.setattr(boundary, "prepare", prepare)
    with SQLiteStateStore(path) as reopened:
        definition, registry, calls = _registry(reopened, (operation,), target=target)
        result = _coordinators(reopened, registry, boundary)[0].resume_to_completion(
            definition.migration_id
        )
        assert events[0:2] == ["recover-abort", "prepare"]
        assert calls == []
        assert len(_transitions(result, "DURABLE_MIGRATED")) == 1
        assert len(_transitions(result, "COMPLETED")) == 1


@pytest.mark.parametrize("ack_lost", [False, True])
def test_completed_finalize_windows_recover_before_terminal_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ack_lost: bool
) -> None:
    path = tmp_path / f"completed-finalize-{ack_lost}.db"
    boundary = Boundary(record("UNINITIALIZED"))
    operation = _operation()
    with SQLiteStateStore(path) as store:
        _initialize_with_migration(store, boundary, ("PREPARED", "APPLYING"), schema=1)
        definition, registry, _ = _registry(store, (operation,))
        target = _target_fingerprint(store, (operation,))
        coordinator, lifecycles = _coordinators(store, registry, boundary)
        proof = coordinator._execution.execute(definition.migration_id)
        lifecycles.record_durable_migrated(definition.migration_id, proof)
        boundary.ack_lost = ack_lost
        boundary.fail_finalize = not ack_lost
        with pytest.raises(MigrationError):
            lifecycles.complete(definition.migration_id)
        generation = store.read_metadata().protected_freshness_generation

    boundary.ack_lost = boundary.fail_finalize = False
    events: list[str] = []
    original_finalize, original_prepare = boundary.finalize, boundary.prepare

    def finalize(*args, **kwargs):
        events.append("recover-finalize")
        return original_finalize(*args, **kwargs)

    def prepare(*args, **kwargs):
        events.append("prepare")
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(boundary, "finalize", finalize)
    monkeypatch.setattr(boundary, "prepare", prepare)
    with SQLiteStateStore(path) as reopened:
        definition, registry, calls = _registry(reopened, (operation,), target=target)
        protected = ProtectedFreshnessHandoffCoordinator(
            reopened, LocalDurableEvidenceRegistry(), boundary
        )
        original_recover = protected.recover_protected_state

        def recover(scope):
            events.append("recover")
            return original_recover(scope)

        monkeypatch.setattr(protected, "recover_protected_state", recover)
        lifecycles = DurableMigrationLifecycleCoordinator(reopened, registry, protected)
        coordinator = DurableMigrationCompletionCoordinator(
            reopened, _ForbiddenExecution(), lifecycles, protected
        )
        result = coordinator.resume_to_completion(definition.migration_id)
        assert result.current["state"] == "COMPLETED"
        assert reopened.read_metadata().protected_freshness_generation == generation
        assert len(_transitions(result, "COMPLETED")) == 1
        assert calls == []
        assert events == (["recover"] if ack_lost else ["recover", "recover-finalize"])


def test_completed_prepare_without_local_commit_recovers_before_new_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "completed-local-absent.db"
    boundary = Boundary(record("UNINITIALIZED"))
    operation = _operation()
    with SQLiteStateStore(path) as store:
        _initialize_with_migration(store, boundary, ("PREPARED", "APPLYING"), schema=1)
        definition, registry, _ = _registry(store, (operation,))
        target = _target_fingerprint(store, (operation,))
        coordinator, lifecycles = _coordinators(store, registry, boundary)
        proof = coordinator._execution.execute(definition.migration_id)
        lifecycles.record_durable_migrated(definition.migration_id, proof)
        generation = store.read_metadata().protected_freshness_generation
        store._connection.execute(
            "CREATE TRIGGER fail_completed BEFORE INSERT ON state_store_immutable_history "
            "BEGIN SELECT RAISE(ABORT, 'completed local absent'); END"
        )
        with pytest.raises(MigrationError):
            lifecycles.complete(definition.migration_id)
        store._connection.execute("DROP TRIGGER fail_completed")
        assert boundary.value["lifecycle"] == "PREPARED"
        assert store.read_metadata().protected_freshness_generation == generation

    events: list[str] = []
    original_abort, original_prepare = boundary.abort, boundary.prepare

    def abort(*args, **kwargs):
        events.append("recover-abort")
        return original_abort(*args, **kwargs)

    def prepare(*args, **kwargs):
        events.append("prepare")
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(boundary, "abort", abort)
    monkeypatch.setattr(boundary, "prepare", prepare)
    with SQLiteStateStore(path) as reopened:
        definition, registry, calls = _registry(reopened, (operation,), target=target)
        protected = ProtectedFreshnessHandoffCoordinator(
            reopened, LocalDurableEvidenceRegistry(), boundary
        )
        result = DurableMigrationCompletionCoordinator(
            reopened,
            _ForbiddenExecution(),
            DurableMigrationLifecycleCoordinator(reopened, registry, protected),
            protected,
        ).resume_to_completion(definition.migration_id)
        assert events == ["recover-abort", "prepare"]
        assert calls == []
        assert reopened.read_metadata().protected_freshness_generation == generation + 1
        assert len(_transitions(result, "COMPLETED")) == 1


def test_reopened_completed_retry_uses_no_process_memory_or_executor(tmp_path: Path) -> None:
    path = tmp_path / "completed-retry.db"
    boundary = Boundary(record("UNINITIALIZED"))
    operation = _operation()
    with SQLiteStateStore(path) as store:
        _initialize_with_migration(store, boundary, ("PREPARED", "APPLYING"), schema=1)
        definition, registry, _ = _registry(store, (operation,))
        target = _target_fingerprint(store, (operation,))
        _coordinators(store, registry, boundary)[0].resume_to_completion(definition.migration_id)
        before = store.read_verified_snapshot()

    with SQLiteStateStore(path) as reopened:
        definition, registry, calls = _registry(reopened, (operation,), target=target)
        protected = ProtectedFreshnessHandoffCoordinator(
            reopened, LocalDurableEvidenceRegistry(), boundary
        )
        result = DurableMigrationCompletionCoordinator(
            reopened,
            _ForbiddenExecution(),
            DurableMigrationLifecycleCoordinator(reopened, registry, protected),
            protected,
        ).resume_to_completion(definition.migration_id)
        assert result.current["state"] == "COMPLETED"
        assert reopened.read_verified_snapshot() == before
        assert calls == []


def test_stale_coordinator_fresh_reads_completed_authority(tmp_path: Path) -> None:
    path = tmp_path / "stale.db"
    boundary = Boundary(record("UNINITIALIZED"))
    operation = _operation()
    with SQLiteStateStore(path) as store_a, SQLiteStateStore(path) as store_b:
        _initialize_with_migration(store_a, boundary, ("PREPARED", "APPLYING"), schema=1)
        definition_a, registry_a, _ = _registry(store_a, (operation,))
        target = _target_fingerprint(store_a, (operation,))
        definition_b, registry_b, calls_b = _registry(store_b, (operation,), target=target)
        stale, _ = _coordinators(store_b, registry_b, boundary)
        _coordinators(store_a, registry_a, boundary)[0].resume_to_completion(
            definition_a.migration_id
        )
        before = store_a.read_verified_snapshot()
        assert stale.resume_to_completion(definition_b.migration_id).current["state"] == "COMPLETED"
        assert stale.resume_to_completion(definition_b.migration_id).current["state"] == "COMPLETED"
        assert store_b.read_verified_snapshot() == before
        assert calls_b == []
        assert len(_declarations(before)) == 1


def test_corrupt_physical_materialization_never_advances_lifecycle(tmp_path: Path) -> None:
    boundary = Boundary(record("UNINITIALIZED"))
    with SQLiteStateStore(tmp_path / "corrupt-materialization.db") as store:
        _initialize_with_migration(store, boundary, ("PREPARED", "APPLYING"), schema=1)
        definition, registry, _ = _registry(store, (_operation(),))
        coordinator, _ = _coordinators(store, registry, boundary)
        coordinator._execution.execute(definition.migration_id)
        before = store.read_verified_snapshot()
        store._connection.execute("DROP TABLE a")
        with pytest.raises(MigrationError, match="durable migration declaration"):
            coordinator.resume_to_completion(definition.migration_id)
        after = store.read_verified_snapshot()
        assert after == before
        assert len(_declarations(after)) == 1


def test_corrupt_lifecycle_fails_at_verified_snapshot_without_repair(tmp_path: Path) -> None:
    boundary = Boundary(record("UNINITIALIZED"))
    with SQLiteStateStore(tmp_path / "corrupt-lifecycle.db") as store:
        _initialize_with_migration(store, boundary, ("PREPARED", "APPLYING"), schema=1)
        definition, registry, calls = _registry(store, (_operation(),))
        coordinator, _ = _coordinators(store, registry, boundary)
        generation = store.read_metadata().protected_freshness_generation
        store._connection.execute(
            "UPDATE state_store_current_records SET record_json='{}' WHERE record_key=?",
            (f"migration-current:{definition.migration_id}",),
        )
        with pytest.raises(Exception):
            coordinator.resume_to_completion(definition.migration_id)
        assert store.read_metadata().protected_freshness_generation == generation
        assert calls == []
