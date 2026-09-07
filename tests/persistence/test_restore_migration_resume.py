from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from bot_core.persistence.local_durable_evidence import LocalDurableEvidenceRegistry
from bot_core.persistence.migration_execution import MigrationExecutionDeclaration
from bot_core.persistence.migration_execution_contract import (
    migration_definition_fingerprint,
    thaw_json,
)
from bot_core.persistence.migration_execution_engine import MigrationExecutionCoordinator
from bot_core.persistence.migration_protocol import (
    PRODUCTION_MIGRATION_REGISTRY,
    DurableMigrationLifecycleCoordinator,
    MigrationRegistry,
    migration_current,
    migration_current_carrier,
    migration_transition,
    migration_transition_carrier,
)
from bot_core.persistence.state_store_v2_migration import MIGRATION_ID
from bot_core.persistence.protected_freshness_handoff import (
    ProtectedFreshnessHandoffCoordinator,
)
from bot_core.persistence.restore_migration_resume import (
    RestoreMigrationCompletedDisposition,
    RestoreMigrationResumeCoordinator,
    RestoreMigrationResumeError,
)
from bot_core.persistence.restore_migration_staging import prepare_legacy_restore_staging
from bot_core.persistence.state_store import SQLiteStateStore
from tests.persistence.test_protected_freshness_handoff import Boundary, record
from tests.persistence.test_restore_migration_staging import _legacy_backup


def _boundary_for(backup):  # type: ignore[no-untyped-def]
    return Boundary(
        record(
            "COMMITTED",
            committed_generation=backup.local_protected_freshness_generation,
            committed_state_fingerprint_sha256=backup.state_fingerprint_sha256,
        )
    )


def test_source_ready_reaches_completed_without_install_and_retry_is_read_only(
    tmp_path: Path,
) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live = tmp_path / "live.sqlite3"
    live.write_bytes(b"live-is-not-an-install-target-yet")
    live_before = live.read_bytes()
    boundary = _boundary_for(backup)
    coordinator = RestoreMigrationResumeCoordinator(live, LocalDurableEvidenceRegistry(), boundary)

    result = coordinator.resume(backup)
    first = result.completed_snapshot
    retry = RestoreMigrationResumeCoordinator(
        live, LocalDurableEvidenceRegistry(), boundary
    ).resume(backup)

    assert result.disposition is RestoreMigrationCompletedDisposition.COMPLETED_READY_FOR_INSTALL
    assert first.metadata.state_store_schema_version == 2
    assert retry.completed_snapshot == first
    assert result.sqlite_path.exists()
    assert result.sqlite_path.parent.resolve() == live.parent.resolve()
    assert result.sqlite_path.name.startswith(f".{live.name}.restore-migration-")
    manifest_path = live.parent / f"{result.sqlite_path.name[: -len('.sqlite3')]}.manifest.json"
    unpublished_path = (
        live.parent / f"{result.sqlite_path.name[: -len('.sqlite3')]}.unpublished.sqlite3"
    )
    assert manifest_path.exists()
    assert not unpublished_path.exists()
    assert not (live.parent / ".cryptohunter-restore-staging").exists()
    assert live.read_bytes() == live_before
    with SQLiteStateStore(result.sqlite_path) as reopened:
        assert reopened.read_verified_snapshot() == first
    assert boundary.value["lifecycle"] == "COMMITTED"
    assert boundary.value["committed_generation"] == first.metadata.protected_freshness_generation
    assert boundary.value["committed_state_fingerprint_sha256"] == (
        first.metadata.state_fingerprint_sha256
    )


def test_missing_sealed_path_fails_before_first_protected_mutation(tmp_path: Path) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    boundary = _boundary_for(backup)
    before = dict(boundary.value)

    with pytest.raises(RestoreMigrationResumeError, match="NO_SEALED_PATH"):
        RestoreMigrationResumeCoordinator(
            tmp_path / "live.sqlite3",
            LocalDurableEvidenceRegistry(),
            boundary,
            MigrationRegistry(),
        )

    assert boundary.value == before
    assert boundary.calls == []


@pytest.mark.parametrize(
    "phase", ["PREPARED", "APPLYING_SOURCE", "APPLYING_MATERIALIZED", "DURABLE_MIGRATED"]
)
def test_fresh_process_shape_resumes_every_intermediate_phase(tmp_path: Path, phase: str) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live = tmp_path / "live.sqlite3"
    artifact = prepare_legacy_restore_staging(backup, live_state_store_path=live)
    boundary = _boundary_for(backup)
    with SQLiteStateStore(artifact.sqlite_path) as store:
        protected = ProtectedFreshnessHandoffCoordinator(
            store, LocalDurableEvidenceRegistry(), boundary
        )
        lifecycle = DurableMigrationLifecycleCoordinator(
            store, PRODUCTION_MIGRATION_REGISTRY, protected
        )
        lifecycle.prepare(MIGRATION_ID)
        if phase != "PREPARED":
            lifecycle.begin_applying(MIGRATION_ID)
        if phase in {"APPLYING_MATERIALIZED", "DURABLE_MIGRATED"}:
            materialization = MigrationExecutionCoordinator(
                store, PRODUCTION_MIGRATION_REGISTRY, protected
            ).execute(MIGRATION_ID)
            if phase == "DURABLE_MIGRATED":
                lifecycle.record_durable_migrated(MIGRATION_ID, materialization)

    result = RestoreMigrationResumeCoordinator(
        live, LocalDurableEvidenceRegistry(), boundary
    ).resume(backup)
    migration = tuple(
        record
        for record in result.completed_snapshot.immutable_history
        if record.representation_name == "Migration execution declaration"
    )
    transitions = tuple(
        record
        for record in result.completed_snapshot.immutable_history
        if record.representation_name == "Migration transition/history revisions"
    )

    assert len(migration) == 1
    assert len(transitions) == 4
    assert result.completed_snapshot.metadata.state_store_schema_version == 2


def _foreign_lifecycle_carriers():  # type: ignore[no-untyped-def]
    transition = migration_transition(
        migration_id="other-migration",
        transition_revision=1,
        previous_state=None,
        state="PREPARED",
        transaction_fingerprint_sha256="a" * 64,
        state_fingerprint_sha256="b" * 64,
        protected_freshness_generation=2,
    )
    current = migration_current(
        migration_id="other-migration",
        current_transition_revision=1,
        state="PREPARED",
        authoritative_state_fingerprint_sha256="b" * 64,
        protected_freshness_generation=2,
    )
    return migration_current_carrier(current), migration_transition_carrier(transition)


def test_foreign_migration_lifecycle_suffix_is_rejected(tmp_path: Path) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    completed = RestoreMigrationResumeCoordinator(
        tmp_path / "live.sqlite3", LocalDurableEvidenceRegistry(), _boundary_for(backup)
    ).resume(backup)
    current, transition = _foreign_lifecycle_carriers()
    foreign = replace(
        completed.completed_snapshot.transaction_descriptors[-1],
        expected_current_generation=completed.completed_snapshot.metadata.protected_freshness_generation,
        target_generation=completed.completed_snapshot.metadata.protected_freshness_generation + 1,
        current_record_mutations=(current,),
        immutable_history_appends=(transition,),
    )
    candidate = replace(
        completed.completed_snapshot,
        transaction_descriptors=(*completed.completed_snapshot.transaction_descriptors, foreign),
    )

    with pytest.raises(RestoreMigrationResumeError, match="LINEAGE_CONFLICT"):
        RestoreMigrationResumeCoordinator._assert_lineage(backup, completed.manifest, candidate)


def test_completed_target_does_not_mask_foreign_family_suffix(tmp_path: Path) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live = tmp_path / "live.sqlite3"
    boundary = _boundary_for(backup)
    completed = RestoreMigrationResumeCoordinator(
        live, LocalDurableEvidenceRegistry(), boundary
    ).resume(backup)
    current, transition = _foreign_lifecycle_carriers()
    foreign = replace(
        completed.completed_snapshot.transaction_descriptors[-1],
        expected_current_generation=(
            completed.completed_snapshot.metadata.protected_freshness_generation
        ),
        target_generation=completed.completed_snapshot.metadata.protected_freshness_generation + 1,
        current_record_mutations=(current,),
        immutable_history_appends=(transition,),
    )
    candidate = replace(
        completed.completed_snapshot,
        transaction_descriptors=(
            *completed.completed_snapshot.transaction_descriptors,
            foreign,
        ),
    )

    with pytest.raises(RestoreMigrationResumeError, match="LINEAGE_CONFLICT"):
        RestoreMigrationResumeCoordinator._assert_lineage(backup, completed.manifest, candidate)


def test_unrelated_domain_suffix_is_rejected(tmp_path: Path) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    completed = RestoreMigrationResumeCoordinator(
        tmp_path / "live.sqlite3", LocalDurableEvidenceRegistry(), _boundary_for(backup)
    ).resume(backup)
    unrelated = replace(
        completed.completed_snapshot.transaction_descriptors[-1],
        expected_current_generation=completed.completed_snapshot.metadata.protected_freshness_generation,
        target_generation=completed.completed_snapshot.metadata.protected_freshness_generation + 1,
        current_record_mutations=(completed.completed_snapshot.current_records[0],),
        immutable_history_appends=(),
    )
    candidate = replace(
        completed.completed_snapshot,
        transaction_descriptors=(*completed.completed_snapshot.transaction_descriptors, unrelated),
    )
    with pytest.raises(RestoreMigrationResumeError, match="LINEAGE_CONFLICT"):
        RestoreMigrationResumeCoordinator._assert_lineage(backup, completed.manifest, candidate)


def test_external_same_generation_different_state_fails_closed(tmp_path: Path) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live = tmp_path / "live.sqlite3"
    artifact = prepare_legacy_restore_staging(backup, live_state_store_path=live)
    boundary = Boundary(
        record(
            "COMMITTED",
            committed_generation=backup.local_protected_freshness_generation,
            committed_state_fingerprint_sha256="f" * 64,
        )
    )

    with pytest.raises(RestoreMigrationResumeError, match="EXTERNAL_MISMATCH"):
        RestoreMigrationResumeCoordinator(live, LocalDurableEvidenceRegistry(), boundary).resume(
            backup
        )
    assert artifact.sqlite_path.exists()
    assert artifact.manifest_path.exists()


def test_declaration_selector_counts_only_target_and_rejects_malformed(tmp_path: Path) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    completed = RestoreMigrationResumeCoordinator(
        tmp_path / "live.sqlite3", LocalDurableEvidenceRegistry(), _boundary_for(backup)
    ).resume(backup)
    target = next(
        item
        for item in completed.completed_snapshot.immutable_history
        if item.representation_name == "Migration execution declaration"
    )
    declaration = MigrationExecutionDeclaration.from_mapping(thaw_json(target.payload))
    foreign = replace(
        declaration,
        migration_id="other-migration",
        migration_definition_fingerprint_sha256=migration_definition_fingerprint(
            migration_id="other-migration",
            source_schema_version=declaration.source_schema_version,
            target_schema_version=declaration.target_schema_version,
            ordered_path=declaration.ordered_path,
            rollback_policy=declaration.rollback_policy,
        ),
    ).carrier()
    with_foreign = replace(
        completed.completed_snapshot,
        immutable_history=(
            *(item for item in completed.completed_snapshot.immutable_history if item != target),
            foreign,
        ),
    )
    assert RestoreMigrationResumeCoordinator._exact_migration_declarations(with_foreign) == ()

    malformed = replace(foreign, payload={"value": 1})
    with_malformed = replace(
        completed.completed_snapshot,
        immutable_history=(
            *(item for item in completed.completed_snapshot.immutable_history if item != target),
            malformed,
        ),
    )
    with pytest.raises(RestoreMigrationResumeError, match="LINEAGE_CONFLICT"):
        RestoreMigrationResumeCoordinator._exact_migration_declarations(with_malformed)


def test_verified_same_family_current_only_suffix_is_rejected_without_rollback(
    tmp_path: Path,
) -> None:
    backup = _legacy_backup(tmp_path / "source.sqlite3")
    live = tmp_path / "live.sqlite3"
    live.write_bytes(b"still-not-an-install-target")
    live_before = live.read_bytes()
    boundary = _boundary_for(backup)
    completed = RestoreMigrationResumeCoordinator(
        live, LocalDurableEvidenceRegistry(), boundary
    ).resume(backup)

    with SQLiteStateStore(completed.sqlite_path) as store:
        protected = ProtectedFreshnessHandoffCoordinator(
            store, LocalDurableEvidenceRegistry(), boundary
        )
        lifecycle = DurableMigrationLifecycleCoordinator(
            store, PRODUCTION_MIGRATION_REGISTRY, protected
        )
        before = store.read_verified_snapshot()
        assert before is not None
        current = next(
            item
            for item in before.current_records
            if item.representation_name == "Migration current state/designation"
        )

        protected.advance_protected_mutation(
            lambda source: (
                replace(
                    source.metadata,
                    protected_freshness_generation=(
                        source.metadata.protected_freshness_generation + 1
                    ),
                ),
                (current,),
                (),
            )
        )
        verified_noop = store.read_verified_snapshot()
        assert verified_noop is not None
        discovered = lifecycle.discover(MIGRATION_ID)
        assert discovered.current is not None
        assert discovered.current["state"] == "COMPLETED"

    external_before_retry = dict(boundary.value)
    with pytest.raises(RestoreMigrationResumeError, match="LINEAGE_CONFLICT"):
        RestoreMigrationResumeCoordinator(live, LocalDurableEvidenceRegistry(), boundary).resume(
            backup
        )

    assert boundary.value == external_before_retry
    assert boundary.value["committed_generation"] == (
        verified_noop.metadata.protected_freshness_generation
    )
    assert boundary.value["committed_state_fingerprint_sha256"] == (
        verified_noop.metadata.state_fingerprint_sha256
    )
    assert completed.sqlite_path.exists()
    assert live.read_bytes() == live_before
