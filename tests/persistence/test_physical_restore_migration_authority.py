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
    PRODUCTION_MIGRATION_REGISTRY,
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
from bot_core.persistence.state_store_v2_migration import MIGRATION_ID
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
from tests.persistence.test_state_store_v2_migration import _applying_v1, _coordinator


@pytest.fixture(autouse=True)
def _restore_production_schema_edge_authority():
    from bot_core.persistence import state_store_v2_migration

    authority = state_store_v2_migration.STATE_STORE_V2_MIGRATION_AUTHORITY
    yield
    state_store_v2_migration.STATE_STORE_V2_MIGRATION_AUTHORITY = authority


def _materialized(tmp_path):  # type: ignore[no-untyped-def]
    store = SQLiteStateStore(tmp_path / "migration-restore.sqlite")
    boundary = Boundary(record("UNINITIALIZED"))
    _initialize_applying(store, boundary)
    operations = (
        MigrationSqlOperation(1, "restore-table", "DDL", "CREATE TABLE restored(id INTEGER)"),
    )
    definition, registry, _calls = _registry(store, operations)
    MigrationExecutionCoordinator(store, registry, _protected(store, boundary)).execute(
        definition.migration_id
    )
    snapshot = store.read_verified_snapshot()
    assert snapshot is not None
    return store, snapshot, registry


def _production_v2_store(path):  # type: ignore[no-untyped-def]
    store, boundary, _ = _applying_v1(path)
    protected = _protected(store, boundary)
    materialized = _coordinator(store, boundary).execute(MIGRATION_ID)
    lifecycles = DurableMigrationLifecycleCoordinator(
        store, PRODUCTION_MIGRATION_REGISTRY, protected
    )
    lifecycles.record_durable_migrated(MIGRATION_ID, materialized)
    lifecycles.complete(MIGRATION_ID)
    snapshot = store.read_verified_snapshot()
    assert snapshot is not None and snapshot.metadata.state_store_schema_version == 2
    return store, boundary, snapshot


def test_unfrozen_schema_three_migration_fails_closed_after_production_v2(tmp_path):  # type: ignore[no-untyped-def]
    store, boundary, _ = _production_v2_store(tmp_path / "unsupported-v3.sqlite")
    try:
        operation = MigrationSqlOperation(
            1, "unsupported-v3", "DDL", "CREATE TABLE unfrozen_v3(id INTEGER)"
        )
        pre = store.sqlite_schema_fingerprint()
        target = _target_fingerprint(store, (operation,))
        definition = MigrationDefinition("unfrozen-v2-to-v3", 2, 3, ("unsupported-v3",))
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
        registry = MigrationRegistry(((definition, authority, lambda _snapshot: plan),))
        protected = _protected(store, boundary)
        lifecycles = DurableMigrationLifecycleCoordinator(store, registry, protected)
        lifecycles.prepare(definition.migration_id)
        lifecycles.begin_applying(definition.migration_id)
        before = store.read_verified_snapshot()
        assert before is not None
        physical_before = store.sqlite_schema_fingerprint()

        with pytest.raises(MigrationError):
            MigrationExecutionCoordinator(store, registry, protected).execute(
                definition.migration_id
            )

        assert store.read_verified_snapshot() == before
        assert before.metadata.state_store_schema_version == 2
        assert store.sqlite_schema_fingerprint() == physical_before == pre
        assert not store._connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='unfrozen_v3'"
        ).fetchone()
        assert all(
            descriptor.state_store_schema_version != 3
            for descriptor in before.transaction_descriptors
        )
        assert all(
            record.payload.get("target_schema_version") != 3
            for record in before.immutable_history
            if record.representation_name == "Migration execution declaration"
        )
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
def test_sealed_restore_rejects_extra_structural_migration_mutation(tmp_path, bucket):  # type: ignore[no-untyped-def]
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
def test_sealed_restore_rejects_declaration_descriptor_edge_mismatch(tmp_path, field):  # type: ignore[no-untyped-def]
    store, snapshot, registry = _materialized(tmp_path)
    try:
        broken_descriptor = replace(snapshot.transaction_descriptors[-1], **{field: "f" * 64})
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
def test_schema_chain_rejects_materialization_gap_or_reverse_chronology(tmp_path, entries):  # type: ignore[no-untyped-def]
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
        and (keep_declaration or item.representation_name != "Migration execution declaration")
    ) + tuple(migration_transition_carrier(item) for item in history)
    return StateStoreSnapshot(
        snapshot.metadata, current_records, immutable, snapshot.transaction_descriptors
    )


def test_restore_authority_rejects_prepared_with_declaration(tmp_path):  # type: ignore[no-untyped-def]
    store, snapshot, registry = _materialized(tmp_path)
    try:
        prepared = _snapshot_with_migration_state(snapshot, ("PREPARED",), keep_declaration=True)
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


def test_production_v2_full_trusted_physical_restore_never_executes_migration(
    tmp_path, monkeypatch
):  # type: ignore[no-untyped-def]
    store, _boundary, snapshot = _production_v2_store(tmp_path / "v2-source.sqlite")
    source_physical = store.sqlite_schema_fingerprint()
    authority = authority_for(tmp_path, store)
    artifact = PhysicalBackupCreator(BackupArtifactAuthenticator(authority)).create(
        store, tmp_path / "v2-backup.sqlite"
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
    live = tmp_path / "v2-restored.sqlite"
    result = TrustedPhysicalRestoreCoordinator(
        live,
        LocalDurableEvidenceRegistry(),
        RestoreBoundary(artifact.backup_envelope),
        PhysicalBackupAdmissionValidator(BackupArtifactVerifier(authority)),
        RestoreLifecycleAuthorityBundle.from_migration_registry(PRODUCTION_MIGRATION_REGISTRY),
    ).restore_trusted_artifact(artifact)

    assert result.decision is RestoreDecision.RESTORE_EXTERNAL_COMMITTED_CURRENT
    assert execution_calls == completion_calls == 0
    with SQLiteStateStore(live) as restored:
        assert restored.read_verified_snapshot() == snapshot
        assert restored.read_verified_snapshot().metadata.state_store_schema_version == 2
        assert restored.sqlite_schema_fingerprint() == source_physical
