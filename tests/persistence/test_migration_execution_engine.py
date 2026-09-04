from __future__ import annotations

import inspect
import sqlite3
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from bot_core.persistence.local_durable_evidence import LocalDurableEvidenceRegistry
from bot_core.persistence.migration_execution import (
    MigrationExecutionAuthority,
    MigrationExecutionPlan,
    MigrationSqlOperation,
    sqlite_schema_fingerprint,
)
from bot_core.persistence.migration_execution_engine import (
    MigrationExecutionCoordinator,
)
from bot_core.persistence.migration_protocol import (
    MigrationDefinition,
    MigrationError,
    MigrationRegistry,
    migration_current_carrier,
    migration_target_materialized,
    migration_transition_carrier,
)
from bot_core.persistence.protected_freshness_handoff import (
    ProtectedFreshnessHandoffCoordinator,
)
from bot_core.persistence.state_store import SQLiteStateStore, StateStoreError
from tests.persistence.test_migration_protocol import lifecycle
from tests.persistence.test_protected_freshness_handoff import Boundary, record
from tests.persistence.test_state_store_records import _account, _metadata


def test_protected_advancement_paths_share_one_private_protocol_core() -> None:
    ordinary = inspect.getsource(
        ProtectedFreshnessHandoffCoordinator.advance_protected_state
    )
    migration = inspect.getsource(
        ProtectedFreshnessHandoffCoordinator._advance_migration_execution
    )
    shared = inspect.getsource(
        ProtectedFreshnessHandoffCoordinator._advance_protected_candidate
    )

    assert "_advance_protected_candidate" in ordinary
    assert "_advance_protected_candidate" in migration
    for protocol_operation in ("._authority.prepare", "._authority.finalize"):
        assert protocol_operation not in ordinary
        assert protocol_operation not in migration
        assert shared.count(protocol_operation) == 1
    assert "publish_verified_state" not in migration
    assert shared.count("publish_verified_state") == 1


def test_migration_execution_authority_has_no_public_statestore_commit() -> None:
    assert not hasattr(SQLiteStateStore, "commit_migration_execution")
    assert callable(SQLiteStateStore._commit_migration_execution)
    public_execution_methods = {
        name
        for name, value in inspect.getmembers(
            MigrationExecutionCoordinator, inspect.isfunction
        )
        if not name.startswith("_")
    }
    assert public_execution_methods == {"execute"}


def _protected(store: SQLiteStateStore, boundary: Boundary):
    return ProtectedFreshnessHandoffCoordinator(
        store, LocalDurableEvidenceRegistry(), boundary
    )


def _initialize_applying(
    store: SQLiteStateStore, boundary: Boundary, *, schema: int = 1
) -> None:
    history, current = lifecycle(("PREPARED", "APPLYING"))
    _protected(store, boundary).advance_protected_state(
        _metadata(state_store_schema_version=schema),
        current_records=(_account(), migration_current_carrier(current)),
        immutable_history=tuple(migration_transition_carrier(item) for item in history),
    )


def _target_fingerprint(store: SQLiteStateStore, operations) -> str:
    clone = sqlite3.connect(":memory:")
    rows = store._connection.execute(
        "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL "
        "AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\' ORDER BY type,name"
    ).fetchall()
    for (statement,) in rows:
        clone.execute(statement)
    for operation in operations:
        clone.execute(operation.statement, operation.parameters)
    return sqlite_schema_fingerprint(clone)


def _registry(
    store: SQLiteStateStore,
    operations,
    *,
    pre: str | None = None,
    target: str | None = None,
    barrier: threading.Barrier | None = None,
):
    definition = MigrationDefinition("migration-1", 1, 2, ("widgets",))
    durable = store.read_verified_snapshot()
    declaration = next(
        (
            record.payload
            for record in (() if durable is None else durable.immutable_history)
            if record.representation_name == "Migration execution declaration"
        ),
        None,
    )
    pre = pre or (
        str(declaration["pre_sqlite_schema_fingerprint_sha256"])
        if declaration is not None
        else store.sqlite_schema_fingerprint()
    )
    target = target or _target_fingerprint(store, operations)
    calls = []

    def planner(snapshot):
        calls.append(snapshot.metadata.protected_freshness_generation)
        if barrier is not None:
            barrier.wait()
        return MigrationExecutionPlan(tuple(operations), pre, target)

    sealed_plan = MigrationExecutionPlan(tuple(operations), pre, target)
    authority = MigrationExecutionAuthority(
        definition.migration_id,
        definition.source_schema_version,
        definition.target_schema_version,
        definition.ordered_path,
        definition.rollback_policy,
        definition.fingerprint(),
        sealed_plan.operation_plan_fingerprint_sha256,
        pre,
        target,
    )
    return definition, MigrationRegistry(((definition, authority, planner),)), calls


def test_protected_execution_persists_exact_declaration_and_materialization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    boundary = Boundary(record("UNINITIALIZED"))
    with SQLiteStateStore(path) as store:
        _initialize_applying(store, boundary)
        operations = (
            MigrationSqlOperation(
                1,
                "create-widget",
                "DDL",
                "CREATE TABLE widget(id INTEGER PRIMARY KEY, name TEXT)",
            ),
            MigrationSqlOperation(
                2,
                "seed-widget",
                "DML",
                "INSERT INTO widget(name) VALUES (?)",
                ("first",),
            ),
        )
        definition, registry, calls = _registry(store, operations)
        before = store.read_verified_snapshot()
        result = MigrationExecutionCoordinator(
            store, registry, _protected(store, boundary)
        ).execute(definition.migration_id)
        after = store.read_verified_snapshot()
        assert before is not None and after is not None
        assert calls == [before.metadata.protected_freshness_generation]
        assert store._connection.execute("SELECT name FROM widget").fetchall() == [
            ("first",)
        ]
        assert after.metadata.state_store_schema_version == 2
        assert (
            after.metadata.protected_freshness_generation
            == before.metadata.protected_freshness_generation + 1
        )
        declarations = tuple(
            item
            for item in after.immutable_history
            if item.representation_name == "Migration execution declaration"
        )
        assert len(declarations) == 1
        descriptor = after.transaction_descriptors[-1]
        assert descriptor.current_record_mutations == ()
        assert descriptor.immutable_history_appends == declarations
        assert migration_target_materialized(definition, result, after)
        current, _ = store.read_lifecycle_records(
            identity="migration-1",
            current_name="Migration current state/designation",
            history_name="Migration transition/history revisions",
            current_key="migration-current:migration-1",
            history_key_prefix="migration-transition:migration-1:",
        )
        assert current is not None and current.payload["state"] == "APPLYING"


def test_operation_failure_and_target_mismatch_roll_back_everything(
    tmp_path: Path,
) -> None:
    for target_mismatch in (False, True):
        boundary = Boundary(record("UNINITIALIZED"))
        with SQLiteStateStore(tmp_path / f"state-{target_mismatch}.db") as store:
            _initialize_applying(store, boundary)
            operations = (
                MigrationSqlOperation(
                    1, "create-a", "DDL", "CREATE TABLE a(id INTEGER)"
                ),
                MigrationSqlOperation(
                    2,
                    "second",
                    "DML" if target_mismatch else "DDL",
                    (
                        "INSERT INTO a VALUES (1)"
                        if target_mismatch
                        else "CREATE TABLE a(id INTEGER)"
                    ),
                ),
            )
            target = "f" * 64
            definition, registry, _ = _registry(store, operations, target=target)
            before = store.read_verified_snapshot()
            with pytest.raises(MigrationError, match="execution failed"):
                MigrationExecutionCoordinator(
                    store, registry, _protected(store, boundary)
                ).execute(definition.migration_id)
            after = store.read_verified_snapshot()
            assert after == before
            assert (
                after.metadata.protected_freshness_generation
                == before.metadata.protected_freshness_generation
            )
            assert (
                after.metadata.state_store_schema_version
                == before.metadata.state_store_schema_version
            )
            assert after.transaction_descriptors == before.transaction_descriptors
            assert not any(
                item.representation_name == "Migration execution declaration"
                for item in after.immutable_history
            )
            current = next(
                item
                for item in after.current_records
                if item.representation_name == "Migration current state/designation"
            )
            assert current.payload["state"] == "APPLYING"
            assert store._connection.execute(
                "SELECT count(*) FROM sqlite_schema WHERE name='a'"
            ).fetchone() == (0,)


def test_wrong_lifecycle_fails_before_prepare(tmp_path: Path) -> None:
    boundary = Boundary(record("UNINITIALIZED"))
    with SQLiteStateStore(tmp_path / "state.db") as store:
        _protected(store, boundary).advance_protected_state(
            _metadata(state_store_schema_version=1), current_records=(_account(),)
        )
        operation = MigrationSqlOperation(
            1, "create-a", "DDL", "CREATE TABLE a(id INTEGER)"
        )
        definition, registry, _ = _registry(store, (operation,))
        with pytest.raises(MigrationError, match="APPLYING"):
            MigrationExecutionCoordinator(
                store, registry, _protected(store, boundary)
            ).execute(definition.migration_id)
        assert boundary.calls.count("prepare") == 1  # initialization only


def test_wrong_logical_source_schema_fails_before_planning_or_prepare(
    tmp_path: Path,
) -> None:
    boundary = Boundary(record("UNINITIALIZED"))
    with SQLiteStateStore(tmp_path / "state.db") as store:
        _initialize_applying(store, boundary, schema=3)
        operation = MigrationSqlOperation(
            1, "create-a", "DDL", "CREATE TABLE a(id INTEGER)"
        )
        definition, registry, calls = _registry(store, (operation,), target="f" * 64)
        before = store.read_verified_snapshot()
        prepares = boundary.calls.count("prepare")
        with pytest.raises(MigrationError, match="source schema mismatch"):
            MigrationExecutionCoordinator(
                store, registry, _protected(store, boundary)
            ).execute(definition.migration_id)
        assert calls == []
        assert boundary.calls.count("prepare") == prepares
        assert store.read_verified_snapshot() == before
        assert store._connection.execute(
            "SELECT count(*) FROM sqlite_schema WHERE name='a'"
        ).fetchone() == (0,)


def test_wrong_physical_pre_schema_fails_before_prepare_or_sql(tmp_path: Path) -> None:
    boundary = Boundary(record("UNINITIALIZED"))
    with SQLiteStateStore(tmp_path / "state.db") as store:
        _initialize_applying(store, boundary)
        operation = MigrationSqlOperation(
            1, "create-a", "DDL", "CREATE TABLE a(id INTEGER)"
        )
        definition, registry, calls = _registry(
            store, (operation,), pre="f" * 64, target="e" * 64
        )
        before = store.read_verified_snapshot()
        physical_before = store.sqlite_schema_fingerprint()
        prepares = boundary.calls.count("prepare")
        with pytest.raises(MigrationError, match="pre-schema fingerprint mismatch"):
            MigrationExecutionCoordinator(
                store, registry, _protected(store, boundary)
            ).execute(definition.migration_id)
        assert calls == [before.metadata.protected_freshness_generation]
        assert boundary.calls.count("prepare") == prepares
        assert store.read_verified_snapshot() == before
        assert store.sqlite_schema_fingerprint() == physical_before


def test_duplicate_and_reopen_do_not_execute_sql_again(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    boundary = Boundary(record("UNINITIALIZED"))
    operation = MigrationSqlOperation(
        1, "create-a", "DDL", "CREATE TABLE a(id INTEGER)"
    )
    with SQLiteStateStore(path) as store:
        _initialize_applying(store, boundary)
        definition, registry, calls = _registry(store, (operation,))
        target = _target_fingerprint(store, (operation,))
        coordinator = MigrationExecutionCoordinator(
            store, registry, _protected(store, boundary)
        )
        first = coordinator.execute(definition.migration_id)
        generation = store.read_metadata().protected_freshness_generation
        assert coordinator.execute(definition.migration_id) == first
        assert calls == [generation - 1]
    with SQLiteStateStore(path) as reopened:
        definition, registry, calls = _registry(reopened, (operation,), target=target)
        result = MigrationExecutionCoordinator(
            reopened, registry, _protected(reopened, boundary)
        ).execute(definition.migration_id)
        assert calls == []
        assert reopened.read_metadata().protected_freshness_generation == generation
        assert migration_target_materialized(
            definition, result, reopened.read_verified_snapshot()
        )


def test_local_durable_external_prepared_recovers_after_restart(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    boundary = Boundary(record("UNINITIALIZED"))
    operation = MigrationSqlOperation(
        1, "create-a", "DDL", "CREATE TABLE a(id INTEGER)"
    )
    with SQLiteStateStore(path) as store:
        _initialize_applying(store, boundary)
        definition, registry, calls = _registry(store, (operation,))
        target = _target_fingerprint(store, (operation,))
        boundary.fail_finalize = True
        with pytest.raises(RuntimeError, match="finalize denied"):
            MigrationExecutionCoordinator(
                store, registry, _protected(store, boundary)
            ).execute(definition.migration_id)
        durable = store.read_verified_snapshot()
        assert durable.metadata.state_store_schema_version == 2
        assert boundary.value["lifecycle"] == "PREPARED"
        generation = durable.metadata.protected_freshness_generation
        assert calls == [generation - 1]
    boundary.fail_finalize = False
    with SQLiteStateStore(path) as reopened:
        definition, registry, calls = _registry(reopened, (operation,), target=target)
        result = MigrationExecutionCoordinator(
            reopened, registry, _protected(reopened, boundary)
        ).execute(definition.migration_id)
        snapshot = reopened.read_verified_snapshot()
        assert calls == []
        assert snapshot.metadata.protected_freshness_generation == generation
        assert boundary.value["lifecycle"] == "COMMITTED"
        assert (
            sum(
                item.representation_name == "Migration execution declaration"
                for item in snapshot.immutable_history
            )
            == 1
        )
        assert migration_target_materialized(definition, result, snapshot)


def test_source_local_external_prepared_recovers_before_new_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.db"
    boundary = Boundary(record("UNINITIALIZED"))
    operation = MigrationSqlOperation(
        1, "create-a", "DDL", "CREATE TABLE a(id INTEGER)"
    )
    with SQLiteStateStore(path) as store:
        _initialize_applying(store, boundary)
        definition, registry, _ = _registry(store, (operation,))
        target = _target_fingerprint(store, (operation,))
        source_generation = store.read_metadata().protected_freshness_generation

        def fail_local(*args, **kwargs):
            raise StateStoreError("injected pre-commit crash")

        monkeypatch.setattr(store, "_commit_migration_execution", fail_local)
        with pytest.raises(MigrationError, match="execution failed"):
            MigrationExecutionCoordinator(
                store, registry, _protected(store, boundary)
            ).execute(definition.migration_id)
        assert boundary.value["lifecycle"] == "PREPARED"
        assert store.read_metadata().protected_freshness_generation == source_generation
    events: list[str] = []
    original_abort, original_prepare = boundary.abort, boundary.prepare

    def abort(*args, **kwargs):
        events.append("recovery")
        return original_abort(*args, **kwargs)

    def prepare(*args, **kwargs):
        events.append("prepare")
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(boundary, "abort", abort)
    monkeypatch.setattr(boundary, "prepare", prepare)
    with SQLiteStateStore(path) as reopened:
        original_commit = reopened._commit_migration_execution

        def commit(*args, **kwargs):
            events.append("sql")
            return original_commit(*args, **kwargs)

        monkeypatch.setattr(reopened, "_commit_migration_execution", commit)
        definition, registry, calls = _registry(reopened, (operation,), target=target)
        MigrationExecutionCoordinator(
            reopened, registry, _protected(reopened, boundary)
        ).execute(definition.migration_id)
        snapshot = reopened.read_verified_snapshot()
        assert events == ["recovery", "prepare", "sql"]
        assert calls == [source_generation]
        assert snapshot.metadata.protected_freshness_generation == source_generation + 1
        assert (
            sum(
                item.representation_name == "Migration execution declaration"
                for item in snapshot.immutable_history
            )
            == 1
        )


def test_finalize_effect_then_ack_lost_recovers_without_sql(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    boundary = Boundary(record("UNINITIALIZED"))
    operation = MigrationSqlOperation(
        1, "create-a", "DDL", "CREATE TABLE a(id INTEGER)"
    )
    with SQLiteStateStore(path) as store:
        _initialize_applying(store, boundary)
        definition, registry, _ = _registry(store, (operation,))
        target = _target_fingerprint(store, (operation,))
        boundary.ack_lost = True
        with pytest.raises(RuntimeError, match="ack lost"):
            MigrationExecutionCoordinator(
                store, registry, _protected(store, boundary)
            ).execute(definition.migration_id)
        generation = store.read_metadata().protected_freshness_generation
        assert boundary.value["lifecycle"] == "COMMITTED"
    boundary.ack_lost = False
    with SQLiteStateStore(path) as reopened:
        definition, registry, calls = _registry(reopened, (operation,), target=target)
        result = MigrationExecutionCoordinator(
            reopened, registry, _protected(reopened, boundary)
        ).execute(definition.migration_id)
        snapshot = reopened.read_verified_snapshot()
        assert calls == []
        assert snapshot.metadata.protected_freshness_generation == generation
        assert migration_target_materialized(definition, result, snapshot)


def test_conflicting_trusted_definition_rejects_durable_declaration(
    tmp_path: Path,
) -> None:
    boundary = Boundary(record("UNINITIALIZED"))
    with SQLiteStateStore(tmp_path / "state.db") as store:
        _initialize_applying(store, boundary)
        operation = MigrationSqlOperation(
            1, "create-a", "DDL", "CREATE TABLE a(id INTEGER)"
        )
        definition, registry, _ = _registry(store, (operation,))
        MigrationExecutionCoordinator(
            store, registry, _protected(store, boundary)
        ).execute(definition.migration_id)
        before = store.read_verified_snapshot()
        conflict = replace(definition, ordered_path=("conflicting-valid-path",))
        calls: list[int] = []

        def planner(snapshot):
            calls.append(snapshot.metadata.protected_freshness_generation)
            return MigrationExecutionPlan((operation,), "a" * 64, "b" * 64)

        authority = MigrationExecutionAuthority(
            conflict.migration_id,
            conflict.source_schema_version,
            conflict.target_schema_version,
            conflict.ordered_path,
            conflict.rollback_policy,
            conflict.fingerprint(),
            MigrationExecutionPlan(
                (operation,), "a" * 64, "b" * 64
            ).operation_plan_fingerprint_sha256,
            "a" * 64,
            "b" * 64,
        )
        conflicting_registry = MigrationRegistry(((conflict, authority, planner),))
        with pytest.raises(MigrationError, match="declaration does not match"):
            MigrationExecutionCoordinator(
                store, conflicting_registry, _protected(store, boundary)
            ).execute(conflict.migration_id)
        assert calls == []
        assert store.read_verified_snapshot() == before


def _race(
    path: Path,
    boundary: Boundary,
    operation_sets: tuple[tuple[MigrationSqlOperation, ...], ...],
    targets: tuple[str, ...],
):
    barrier = threading.Barrier(2)
    outcomes: list[object] = []

    def contender(index: int) -> None:
        try:
            with SQLiteStateStore(path) as store:
                definition, registry, _ = _registry(
                    store,
                    operation_sets[index],
                    target=targets[index],
                    barrier=barrier,
                )
                outcomes.append(
                    MigrationExecutionCoordinator(
                        store, registry, _protected(store, boundary)
                    ).execute(definition.migration_id)
                )
        except Exception as exc:  # race loser is required to fail closed or reconcile
            outcomes.append(exc)

    threads = [threading.Thread(target=contender, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    return outcomes


def test_two_connection_same_execution_race_commits_structural_sql_once(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    boundary = Boundary(record("UNINITIALIZED"))
    operations = (
        MigrationSqlOperation(1, "create-a", "DDL", "CREATE TABLE a(id INTEGER)"),
        MigrationSqlOperation(2, "seed-a", "DML", "INSERT INTO a VALUES (1)"),
    )
    with SQLiteStateStore(path) as store:
        _initialize_applying(store, boundary)
        target = _target_fingerprint(store, operations)
        source_generation = store.read_metadata().protected_freshness_generation
    outcomes = _race(path, boundary, (operations, operations), (target, target))
    assert len(outcomes) == 2
    with SQLiteStateStore(path) as store:
        snapshot = store.read_verified_snapshot()
        assert store._connection.execute("SELECT * FROM a").fetchall() == [(1,)]
        assert snapshot.metadata.protected_freshness_generation == source_generation + 1
        assert len(snapshot.transaction_descriptors) == source_generation + 1
        assert (
            sum(
                item.representation_name == "Migration execution declaration"
                for item in snapshot.immutable_history
            )
            == 1
        )
        definition, registry, calls = _registry(store, operations, target=target)
        MigrationExecutionCoordinator(
            store, registry, _protected(store, boundary)
        ).execute(definition.migration_id)
        assert calls == []


def test_two_connection_conflicting_effective_plan_race_has_one_winner(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    boundary = Boundary(record("UNINITIALIZED"))
    left = (
        MigrationSqlOperation(
            1, "create-left", "DDL", "CREATE TABLE left_wins(id INTEGER)"
        ),
    )
    right = (
        MigrationSqlOperation(
            1, "create-right", "DDL", "CREATE TABLE right_wins(id INTEGER)"
        ),
    )
    with SQLiteStateStore(path) as store:
        _initialize_applying(store, boundary)
        targets = (_target_fingerprint(store, left), _target_fingerprint(store, right))
        source_generation = store.read_metadata().protected_freshness_generation
    outcomes = _race(path, boundary, (left, right), targets)
    assert len(outcomes) == 2
    with SQLiteStateStore(path) as store:
        snapshot = store.read_verified_snapshot()
        existing = {
            row[0]
            for row in store._connection.execute(
                "SELECT name FROM sqlite_schema WHERE name IN ('left_wins','right_wins')"
            )
        }
        assert existing in ({"left_wins"}, {"right_wins"})
        assert snapshot.metadata.protected_freshness_generation == source_generation + 1
        declarations = tuple(
            item
            for item in snapshot.immutable_history
            if item.representation_name == "Migration execution declaration"
        )
        assert len(declarations) == 1
        assert (
            declarations[0].payload["target_sqlite_schema_fingerprint_sha256"]
            in targets
        )
