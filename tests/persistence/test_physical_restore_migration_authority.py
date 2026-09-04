from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest

from bot_core.persistence.backup_authentication import (
    BackupArtifactAuthenticator,
    BackupArtifactVerifier,
)
from bot_core.persistence.local_durable_evidence import LocalDurableEvidenceRegistry
from bot_core.persistence.migration_completion import (
    DurableMigrationCompletionCoordinator,
)
from bot_core.persistence.physical_backup import (
    PhysicalBackupAdmissionValidator,
    PhysicalBackupCreator,
)
from bot_core.persistence.fingerprints import transaction_fingerprint_sha256
from bot_core.persistence.migration_execution import (
    MigrationExecutionAuthority,
    MigrationExecutionDeclaration,
    MigrationExecutionPlan,
    MigrationSqlOperation,
)
from bot_core.persistence.migration_execution_engine import (
    MigrationExecutionCoordinator,
)
from bot_core.persistence.migration_protocol import (
    DurableMigrationLifecycleCoordinator,
    MigrationDefinition,
    MigrationError,
    MigrationRegistry,
    migration_current_carrier,
    migration_transition_carrier,
)
from bot_core.persistence.restore_protocol import (
    RestoreDecision,
    RestoreLifecycleAuthorityBundle,
    SealedMigrationRestoreAuthority,
    TrustedPhysicalRestoreCoordinator,
)
from bot_core.persistence.state_store import SQLiteStateStore, StateStoreSnapshot
from tests.persistence.test_migration_execution_engine import (
    _initialize_applying,
    _protected,
    _registry,
    _target_fingerprint,
)
from tests.persistence.test_protected_freshness_handoff import Boundary, record
from tests.persistence.test_migration_protocol import lifecycle
from tests.persistence.physical_backup_helpers import authority_for
from tests.persistence.test_restore_protocol import Boundary as RestoreBoundary


def _materialized(tmp_path):  # type: ignore[no-untyped-def]
    store = SQLiteStateStore(tmp_path / "migration-restore.sqlite")
    boundary = Boundary(record("UNINITIALIZED"))
    _initialize_applying(store, boundary)
    operations = (
        MigrationSqlOperation(
            1, "restore-table", "DDL", "CREATE TABLE restored(id INTEGER)"
        ),
    )
    definition, registry, _calls = _registry(store, operations)
    MigrationExecutionCoordinator(store, registry, _protected(store, boundary)).execute(
        definition.migration_id
    )
    snapshot = store.read_verified_snapshot()
    assert snapshot is not None
    return store, snapshot, registry


def test_two_real_sequential_migrations_restore_revalidate(tmp_path):  # type: ignore[no-untyped-def]
    store = SQLiteStateStore(tmp_path / "two-real-migrations.sqlite")
    boundary = Boundary(record("UNINITIALIZED"))
    try:
        _initialize_applying(store, boundary)
        operation_a = MigrationSqlOperation(
            1, "create-a", "DDL", "CREATE TABLE migration_a(id INTEGER)"
        )
        operation_b = MigrationSqlOperation(
            1, "create-b", "DDL", "CREATE TABLE migration_b(id INTEGER)"
        )
        pre_a = store.sqlite_schema_fingerprint()
        target_a = _target_fingerprint(store, (operation_a,))
        target_b = _target_fingerprint(store, (operation_a, operation_b))
        definition_a = MigrationDefinition("migration-1", 1, 2, ("a",))
        definition_b = MigrationDefinition("migration-2", 2, 3, ("b",))

        def entry(definition, operation, pre, target):  # type: ignore[no-untyped-def]
            plan = MigrationExecutionPlan((operation,), pre, target)
            authority = MigrationExecutionAuthority(
                definition.migration_id,
                definition.source_schema_version,
                definition.target_schema_version,
                definition.ordered_path,
                definition.rollback_policy,
                definition.fingerprint(),
                plan.operation_plan_fingerprint_sha256,
                pre,
                target,
            )
            return definition, authority, lambda _snapshot: plan

        registry = MigrationRegistry(
            (
                entry(definition_a, operation_a, pre_a, target_a),
                entry(definition_b, operation_b, target_a, target_b),
            )
        )
        protected = _protected(store, boundary)
        execution = MigrationExecutionCoordinator(store, registry, protected)
        materialized_a = execution.execute("migration-1")
        lifecycles = DurableMigrationLifecycleCoordinator(store, registry, protected)
        lifecycles.record_durable_migrated("migration-1", materialized_a)
        lifecycles.complete("migration-1")
        lifecycles.prepare("migration-2")
        lifecycles.begin_applying("migration-2")
        execution.execute("migration-2")

        snapshot = store.read_verified_snapshot()
        assert snapshot is not None
        declarations = tuple(
            item
            for item in snapshot.immutable_history
            if item.representation_name == "Migration execution declaration"
        )
        assert len(declarations) == 2
        assert (
            declarations[0].payload["target_generation"]
            < declarations[1].payload["target_generation"]
        )
        assert snapshot.metadata.state_store_schema_version == 3
        assert target_a != store.sqlite_schema_fingerprint() == target_b
        SealedMigrationRestoreAuthority(registry).revalidate(snapshot, target_b)
    finally:
        store.close()


def test_sealed_restore_accepts_exact_historical_materialization(tmp_path):  # type: ignore[no-untyped-def]
    store, snapshot, registry = _materialized(tmp_path)
    try:
        SealedMigrationRestoreAuthority(registry).revalidate(
            snapshot, store.sqlite_schema_fingerprint()
        )
    finally:
        store.close()


@pytest.mark.parametrize("bucket", ["current", "history"])
def test_sealed_restore_rejects_extra_structural_migration_mutation(
    tmp_path, bucket
):  # type: ignore[no-untyped-def]
    store, snapshot, registry = _materialized(tmp_path)
    try:
        descriptor = snapshot.transaction_descriptors[-1]
        if bucket == "current":
            changed = replace(
                descriptor,
                current_record_mutations=(snapshot.current_records[0],),
            )
        else:
            declaration = descriptor.immutable_history_appends[0]
            extra = next(
                item
                for item in snapshot.immutable_history
                if item.representation_name == "Migration transition/history revisions"
            )
            changed = replace(
                descriptor,
                immutable_history_appends=tuple(
                    sorted(
                        (declaration, extra),
                        key=lambda item: (item.representation_name, item.record_key),
                    )
                ),
            )
        changed = replace(
            changed,
            transaction_fingerprint_sha256=transaction_fingerprint_sha256(changed),
        )
        assert changed.has_valid_transaction_fingerprint()
        broken = StateStoreSnapshot(
            snapshot.metadata,
            snapshot.current_records,
            snapshot.immutable_history,
            (*snapshot.transaction_descriptors[:-1], changed),
        )
        with pytest.raises(MigrationError, match="materialized"):
            SealedMigrationRestoreAuthority(registry).revalidate(
                broken, store.sqlite_schema_fingerprint()
            )
    finally:
        store.close()


@pytest.mark.parametrize(
    "field",
    ["pre_state_fingerprint_sha256", "pre_history_tail_fingerprint_sha256"],
)
def test_sealed_restore_rejects_declaration_descriptor_edge_mismatch(
    tmp_path, field
):  # type: ignore[no-untyped-def]
    store, snapshot, registry = _materialized(tmp_path)
    try:
        broken_descriptor = replace(
            snapshot.transaction_descriptors[-1], **{field: "f" * 64}
        )
        broken = StateStoreSnapshot(
            snapshot.metadata,
            snapshot.current_records,
            snapshot.immutable_history,
            (*snapshot.transaction_descriptors[:-1], broken_descriptor),
        )
        with pytest.raises(MigrationError, match="materialized"):
            SealedMigrationRestoreAuthority(registry).revalidate(
                broken, store.sqlite_schema_fingerprint()
            )
    finally:
        store.close()


def test_sealed_restore_rejects_numeric_schema_version_mismatch(tmp_path):  # type: ignore[no-untyped-def]
    store, snapshot, registry = _materialized(tmp_path)
    try:
        broken = StateStoreSnapshot(
            replace(snapshot.metadata, state_store_schema_version=3),
            snapshot.current_records,
            snapshot.immutable_history,
            snapshot.transaction_descriptors,
        )
        with pytest.raises(MigrationError, match="migration head"):
            SealedMigrationRestoreAuthority(registry).revalidate(
                broken, store.sqlite_schema_fingerprint()
            )
    finally:
        store.close()


def _schema_entry(migration_id, source, target, pre, post, generation):  # type: ignore[no-untyped-def]
    definition = MigrationDefinition(migration_id, source, target, (migration_id,))
    authority = MigrationExecutionAuthority(
        migration_id,
        source,
        target,
        definition.ordered_path,
        definition.rollback_policy,
        definition.fingerprint(),
        "a" * 64,
        pre,
        post,
    )
    declaration = (
        None
        if generation is None
        else cast(
            MigrationExecutionDeclaration,
            SimpleNamespace(
                expected_current_generation=generation - 1, target_generation=generation
            ),
        )
    )
    return definition, authority, declaration


def _schema_snapshot(snapshot, version):  # type: ignore[no-untyped-def]
    return StateStoreSnapshot(
        replace(snapshot.metadata, state_store_schema_version=version), (), (), ()
    )


def test_two_sequential_materializations_select_only_terminal_head(tmp_path):  # type: ignore[no-untyped-def]
    store, snapshot, _registry = _materialized(tmp_path)
    try:
        entries = [
            _schema_entry("migration-a", 1, 2, "1" * 64, "2" * 64, 2),
            _schema_entry("migration-b", 2, 3, "2" * 64, "3" * 64, 4),
        ]
        SealedMigrationRestoreAuthority._assert_schema_chain(
            _schema_snapshot(snapshot, 3), "3" * 64, entries
        )
    finally:
        store.close()


def test_schema_chain_rejects_broken_adjacent_sealed_fingerprint(tmp_path):  # type: ignore[no-untyped-def]
    store, snapshot, _registry = _materialized(tmp_path)
    try:
        entries = [
            _schema_entry("migration-a", 1, 2, "1" * 64, "2" * 64, 2),
            _schema_entry("migration-b", 2, 3, "f" * 64, "3" * 64, 4),
        ]
        with pytest.raises(MigrationError, match="sealed schema chain"):
            SealedMigrationRestoreAuthority._assert_schema_chain(
                _schema_snapshot(snapshot, 3), "3" * 64, entries
            )
    finally:
        store.close()


@pytest.mark.parametrize(
    "entries",
    [
        [
            _schema_entry("migration-a", 1, 2, "1" * 64, "2" * 64, None),
            _schema_entry("migration-b", 2, 3, "2" * 64, "3" * 64, 4),
        ],
        [
            _schema_entry("migration-a", 1, 2, "1" * 64, "2" * 64, 5),
            _schema_entry("migration-b", 2, 3, "2" * 64, "3" * 64, 4),
        ],
    ],
)
def test_schema_chain_rejects_materialization_gap_or_reverse_chronology(
    tmp_path, entries
):  # type: ignore[no-untyped-def]
    store, snapshot, _registry = _materialized(tmp_path)
    try:
        with pytest.raises(MigrationError):
            SealedMigrationRestoreAuthority._assert_schema_chain(
                _schema_snapshot(snapshot, 3), "3" * 64, entries
            )
    finally:
        store.close()


def test_restore_authority_rejects_two_declarations_for_one_migration(tmp_path):  # type: ignore[no-untyped-def]
    store, snapshot, registry = _materialized(tmp_path)
    try:
        declaration = next(
            item
            for item in snapshot.immutable_history
            if item.representation_name == "Migration execution declaration"
        )
        broken = StateStoreSnapshot(
            snapshot.metadata,
            snapshot.current_records,
            (*snapshot.immutable_history, declaration),
            snapshot.transaction_descriptors,
        )
        with pytest.raises(MigrationError, match="cardinality"):
            SealedMigrationRestoreAuthority(registry).revalidate(
                broken, store.sqlite_schema_fingerprint()
            )
    finally:
        store.close()


def test_restore_authority_rejects_declaration_without_lifecycle(tmp_path):  # type: ignore[no-untyped-def]
    store, snapshot, registry = _materialized(tmp_path)
    try:
        broken = StateStoreSnapshot(
            snapshot.metadata,
            tuple(
                item
                for item in snapshot.current_records
                if item.representation_name != "Migration current state/designation"
            ),
            tuple(
                item
                for item in snapshot.immutable_history
                if item.representation_name != "Migration transition/history revisions"
            ),
            snapshot.transaction_descriptors,
        )
        with pytest.raises(MigrationError, match="current designation"):
            SealedMigrationRestoreAuthority(registry).revalidate(
                broken, store.sqlite_schema_fingerprint()
            )
    finally:
        store.close()


def _snapshot_with_migration_state(snapshot, states, *, keep_declaration):  # type: ignore[no-untyped-def]
    history, current = lifecycle(states)
    current_records = tuple(
        item
        for item in snapshot.current_records
        if item.representation_name != "Migration current state/designation"
    ) + (migration_current_carrier(current),)
    immutable = tuple(
        item
        for item in snapshot.immutable_history
        if item.representation_name != "Migration transition/history revisions"
        and (
            keep_declaration
            or item.representation_name != "Migration execution declaration"
        )
    ) + tuple(migration_transition_carrier(item) for item in history)
    return StateStoreSnapshot(
        snapshot.metadata, current_records, immutable, snapshot.transaction_descriptors
    )


def test_restore_authority_rejects_prepared_with_declaration(tmp_path):  # type: ignore[no-untyped-def]
    store, snapshot, registry = _materialized(tmp_path)
    try:
        prepared = _snapshot_with_migration_state(
            snapshot, ("PREPARED",), keep_declaration=True
        )
        with pytest.raises(MigrationError, match="cardinality"):
            SealedMigrationRestoreAuthority(registry).revalidate(
                prepared, store.sqlite_schema_fingerprint()
            )
    finally:
        store.close()


def test_restore_authority_rejects_completed_without_declaration(tmp_path):  # type: ignore[no-untyped-def]
    store, snapshot, registry = _materialized(tmp_path)
    try:
        completed = _snapshot_with_migration_state(
            snapshot,
            ("PREPARED", "APPLYING", "DURABLE_MIGRATED", "COMPLETED"),
            keep_declaration=False,
        )
        with pytest.raises(MigrationError, match="cardinality"):
            SealedMigrationRestoreAuthority(registry).revalidate(
                completed, store.sqlite_schema_fingerprint()
            )
    finally:
        store.close()


def _real_ab_store(path):  # type: ignore[no-untyped-def]
    store = SQLiteStateStore(path)
    boundary = Boundary(record("UNINITIALIZED"))
    _initialize_applying(store, boundary)
    operation_a = MigrationSqlOperation(
        1, "create-a", "DDL", "CREATE TABLE migration_a(id INTEGER)"
    )
    operation_b = MigrationSqlOperation(
        1, "create-b", "DDL", "CREATE TABLE migration_b(id INTEGER)"
    )
    pre_a = store.sqlite_schema_fingerprint()
    target_a = _target_fingerprint(store, (operation_a,))
    target_b = _target_fingerprint(store, (operation_a, operation_b))
    definition_a = MigrationDefinition("migration-1", 1, 2, ("a",))
    definition_b = MigrationDefinition("migration-2", 2, 3, ("b",))

    def entry(definition, operation, pre, target):  # type: ignore[no-untyped-def]
        plan = MigrationExecutionPlan((operation,), pre, target)
        authority = MigrationExecutionAuthority(
            definition.migration_id,
            definition.source_schema_version,
            definition.target_schema_version,
            definition.ordered_path,
            definition.rollback_policy,
            definition.fingerprint(),
            plan.operation_plan_fingerprint_sha256,
            pre,
            target,
        )
        return definition, authority, lambda _snapshot: plan

    registry = MigrationRegistry(
        (
            entry(definition_a, operation_a, pre_a, target_a),
            entry(definition_b, operation_b, target_a, target_b),
        )
    )
    protected = _protected(store, boundary)
    execution = MigrationExecutionCoordinator(store, registry, protected)
    materialized_a = execution.execute("migration-1")
    lifecycles = DurableMigrationLifecycleCoordinator(store, registry, protected)
    lifecycles.record_durable_migrated("migration-1", materialized_a)
    lifecycles.complete("migration-1")
    lifecycles.prepare("migration-2")
    lifecycles.begin_applying("migration-2")
    execution.execute("migration-2")
    snapshot = store.read_verified_snapshot()
    assert snapshot is not None
    return store, registry, snapshot, target_b


def test_real_ab_full_trusted_physical_restore_never_executes_migration(
    tmp_path, monkeypatch
):  # type: ignore[no-untyped-def]
    store, registry, snapshot, target_b = _real_ab_store(tmp_path / "ab-source.sqlite")
    authority = authority_for(tmp_path, store)
    artifact = PhysicalBackupCreator(BackupArtifactAuthenticator(authority)).create(
        store, tmp_path / "ab-backup.sqlite"
    )
    store.close()
    execution_calls = 0
    completion_calls = 0

    def forbidden_execution(*_args, **_kwargs):
        nonlocal execution_calls
        execution_calls += 1
        raise AssertionError("restore executed migration SQL")

    def forbidden_completion(*_args, **_kwargs):
        nonlocal completion_calls
        completion_calls += 1
        raise AssertionError("restore resumed migration completion")

    monkeypatch.setattr(MigrationExecutionCoordinator, "execute", forbidden_execution)
    monkeypatch.setattr(
        DurableMigrationCompletionCoordinator,
        "resume_to_completion",
        forbidden_completion,
    )
    live = tmp_path / "ab-restored.sqlite"
    result = TrustedPhysicalRestoreCoordinator(
        live,
        LocalDurableEvidenceRegistry(),
        RestoreBoundary(artifact.backup_envelope),
        PhysicalBackupAdmissionValidator(BackupArtifactVerifier(authority)),
        RestoreLifecycleAuthorityBundle.from_migration_registry(registry),
    ).restore_trusted_artifact(artifact)

    assert result.decision is RestoreDecision.RESTORE_EXTERNAL_COMMITTED_CURRENT
    assert execution_calls == completion_calls == 0
    with SQLiteStateStore(live) as restored:
        assert restored.read_verified_snapshot() == snapshot
        assert restored.sqlite_schema_fingerprint() == target_b
