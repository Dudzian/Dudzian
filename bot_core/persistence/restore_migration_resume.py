"""Restart-safe orchestration of a source-seeded legacy restore migration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from .backup_envelope import BackupEnvelope
from .local_durable_evidence import LocalDurableEvidenceRegistry
from .migration_completion import DurableMigrationCompletionCoordinator
from .migration_execution_engine import MigrationExecutionCoordinator
from .migration_protocol import (
    MIGRATION_FAMILY_REPRESENTATIONS,
    DurableMigrationLifecycle,
    DurableMigrationLifecycleCoordinator,
    MigrationError,
    MigrationRegistry,
    PRODUCTION_MIGRATION_REGISTRY,
    migration_mapping_payload,
)
from .protected_freshness_handoff import (
    ProtectedFreshnessAuthorityPort,
    ProtectedFreshnessAuthorityRecord,
    ProtectedFreshnessHandoffCoordinator,
    ProtectedFreshnessHandoffError,
)
from .restore_migration_staging import (
    RestoreMigrationStagingManifest,
    prepare_legacy_restore_staging,
)
from .restore_protocol import SealedMigrationRestoreAuthority
from .records import PersistenceRecord
from .state_store import SQLiteStateStore, StateStoreSnapshot
from .state_store_v2_migration import (
    MIGRATION_ID,
    STATE_STORE_V2_MIGRATION_AUTHORITY,
    STATE_STORE_V2_MIGRATION_DEFINITION,
)
from .transaction_descriptor import StateStoreTransactionDescriptor

_ALLOWED_SUFFIX_ROLE_PREFIXES = frozenset(
    {
        (),
        ("PREPARED",),
        ("PREPARED", "APPLYING"),
        ("PREPARED", "APPLYING", "STRUCTURAL_MIGRATION"),
        ("PREPARED", "APPLYING", "STRUCTURAL_MIGRATION", "DURABLE_MIGRATED"),
        (
            "PREPARED",
            "APPLYING",
            "STRUCTURAL_MIGRATION",
            "DURABLE_MIGRATED",
            "COMPLETED",
        ),
    }
)


class _LineageBinding(Protocol):
    @property
    def source_generation(self) -> int: ...

    @property
    def migration_id(self) -> str: ...


class RestoreMigrationResumeError(RuntimeError):
    """The durable staging workspace cannot safely continue to completion."""


class RestoreMigrationCompletedDisposition(str, Enum):
    COMPLETED_READY_FOR_INSTALL = "COMPLETED_READY_FOR_INSTALL"


@dataclass(frozen=True, slots=True)
class RestoreMigrationCompletedStaging:
    manifest: RestoreMigrationStagingManifest
    sqlite_path: Path
    completed_snapshot: StateStoreSnapshot
    sqlite_schema_fingerprint_sha256: str
    disposition: RestoreMigrationCompletedDisposition


class RestoreMigrationResumeCoordinator:
    """Compose the sealed existing owners without acquiring install authority."""

    def __init__(
        self,
        live_state_store_path: str | Path,
        local_evidence_registry: LocalDurableEvidenceRegistry,
        external_authority: ProtectedFreshnessAuthorityPort,
        registry: MigrationRegistry = PRODUCTION_MIGRATION_REGISTRY,
    ) -> None:
        self._live_path = Path(live_state_store_path)
        self._evidence = local_evidence_registry
        self._external = external_authority
        self._registry = registry
        self._assert_production_seal()

    def _assert_production_seal(self) -> None:
        try:
            definition = self._registry.definition_for(MIGRATION_ID)
            authority = self._registry.execution_authority_for(MIGRATION_ID)
        except MigrationError as exc:
            raise RestoreMigrationResumeError("NO_SEALED_PATH_TO_CURRENT_SCHEMA") from exc
        if definition != STATE_STORE_V2_MIGRATION_DEFINITION or authority != (
            STATE_STORE_V2_MIGRATION_AUTHORITY
        ):
            raise RestoreMigrationResumeError("NO_SEALED_PATH_TO_CURRENT_SCHEMA")
        try:
            authority.assert_definition(definition)
        except Exception as exc:
            raise RestoreMigrationResumeError("NO_SEALED_PATH_TO_CURRENT_SCHEMA") from exc

    def resume(self, source_backup: BackupEnvelope) -> RestoreMigrationCompletedStaging:
        artifact = prepare_legacy_restore_staging(
            source_backup, live_state_store_path=self._live_path
        )
        manifest = artifact.manifest
        with SQLiteStateStore(artifact.sqlite_path) as store:
            protected = ProtectedFreshnessHandoffCoordinator(store, self._evidence, self._external)
            lifecycles = DurableMigrationLifecycleCoordinator(store, self._registry, protected)
            execution = MigrationExecutionCoordinator(store, self._registry, protected)
            completion = DurableMigrationCompletionCoordinator(
                store, execution, lifecycles, protected
            )
            scope = (
                manifest.account_id,
                manifest.device_installation_id,
                manifest.state_store_identity_fingerprint_sha256,
            )

            # Recovery is deliberately the first action after opening the mutable store.
            try:
                protected.recover_protected_state(scope)
            except ProtectedFreshnessHandoffError as exc:
                raise RestoreMigrationResumeError("RESTORE_MIGRATION_EXTERNAL_MISMATCH") from exc
            while True:
                snapshot = self._required_snapshot(store)
                lifecycle = lifecycles.discover(MIGRATION_ID)
                roles = self._assert_lineage(source_backup, manifest, snapshot)
                phase = self._phase(store, manifest, snapshot, lifecycle, roles)
                if phase == "SOURCE_READY":
                    self._assert_external_matches(scope, snapshot)
                    lifecycles.prepare(MIGRATION_ID)
                elif phase == "PREPARED":
                    lifecycles.begin_applying(MIGRATION_ID)
                elif phase in {
                    "APPLYING_SOURCE",
                    "APPLYING_MATERIALIZED",
                    "DURABLE_MIGRATED",
                }:
                    result = completion.resume_to_completion(MIGRATION_ID)
                    if result.current is not None and result.current["state"] == "FAILED":
                        raise RestoreMigrationResumeError("RESTORE_MIGRATION_FAILED")
                elif phase == "COMPLETED":
                    break
                else:  # pragma: no cover - closed classifier
                    raise RestoreMigrationResumeError("RESTORE_MIGRATION_PHASE_INCONSISTENT")
                protected.recover_protected_state(scope)

            completed = self._required_snapshot(store)
            self._assert_lineage(source_backup, manifest, completed)
            schema_fingerprint = store.sqlite_schema_fingerprint()
            SealedMigrationRestoreAuthority(self._registry).revalidate(
                completed, schema_fingerprint
            )
            protected.recover_protected_state(scope)
            completed = self._required_snapshot(store)
            self._assert_external_matches(scope, completed)
            return RestoreMigrationCompletedStaging(
                manifest,
                artifact.sqlite_path,
                completed,
                schema_fingerprint,
                RestoreMigrationCompletedDisposition.COMPLETED_READY_FOR_INSTALL,
            )

    @staticmethod
    def _required_snapshot(store: SQLiteStateStore) -> StateStoreSnapshot:
        snapshot = store.read_verified_snapshot()
        if snapshot is None:
            raise RestoreMigrationResumeError("RESTORE_MIGRATION_PHASE_INCONSISTENT")
        return snapshot

    def _phase(
        self,
        store: SQLiteStateStore,
        manifest: RestoreMigrationStagingManifest,
        snapshot: StateStoreSnapshot,
        lifecycle: DurableMigrationLifecycle,
        roles: tuple[str, ...],
    ) -> str:
        state = None if lifecycle.current is None else str(lifecycle.current["state"])
        declarations = self._exact_migration_declarations(snapshot)
        schema = snapshot.metadata.state_store_schema_version
        generation = snapshot.metadata.protected_freshness_generation
        if state == "FAILED":
            raise RestoreMigrationResumeError("RESTORE_MIGRATION_FAILED")
        if (
            state is None
            and schema == 1
            and not declarations
            and generation == manifest.source_generation
        ):
            phase = "SOURCE_READY"
        elif state == "PREPARED" and schema == 1 and not declarations:
            phase = "PREPARED"
        elif state == "APPLYING" and schema == 1 and not declarations:
            phase = "APPLYING_SOURCE"
        elif state == "APPLYING" and schema == 2 and len(declarations) == 1:
            SealedMigrationRestoreAuthority(self._registry).revalidate(
                snapshot, store.sqlite_schema_fingerprint()
            )
            phase = "APPLYING_MATERIALIZED"
        elif state == "DURABLE_MIGRATED" and schema == 2 and len(declarations) == 1:
            SealedMigrationRestoreAuthority(self._registry).revalidate(
                snapshot, store.sqlite_schema_fingerprint()
            )
            phase = "DURABLE_MIGRATED"
        elif state == "COMPLETED" and schema == 2 and len(declarations) == 1:
            SealedMigrationRestoreAuthority(self._registry).revalidate(
                snapshot, store.sqlite_schema_fingerprint()
            )
            phase = "COMPLETED"
        else:
            raise RestoreMigrationResumeError("RESTORE_MIGRATION_PHASE_INCONSISTENT")
        expected_roles = {
            "SOURCE_READY": (),
            "PREPARED": ("PREPARED",),
            "APPLYING_SOURCE": ("PREPARED", "APPLYING"),
            "APPLYING_MATERIALIZED": (
                "PREPARED",
                "APPLYING",
                "STRUCTURAL_MIGRATION",
            ),
            "DURABLE_MIGRATED": (
                "PREPARED",
                "APPLYING",
                "STRUCTURAL_MIGRATION",
                "DURABLE_MIGRATED",
            ),
            "COMPLETED": (
                "PREPARED",
                "APPLYING",
                "STRUCTURAL_MIGRATION",
                "DURABLE_MIGRATED",
                "COMPLETED",
            ),
        }
        if roles != expected_roles[phase]:
            raise RestoreMigrationResumeError("RESTORE_MIGRATION_PHASE_INCONSISTENT")
        return phase

    @staticmethod
    def _exact_migration_declarations(
        snapshot: StateStoreSnapshot,
    ) -> tuple[PersistenceRecord, ...]:
        exact: list[PersistenceRecord] = []
        for record in snapshot.immutable_history:
            if record.representation_name != "Migration execution declaration":
                continue
            try:
                migration_id = migration_mapping_payload(record.payload)["migration_id"]
            except (MigrationError, KeyError) as exc:
                raise RestoreMigrationResumeError("RESTORE_MIGRATION_LINEAGE_CONFLICT") from exc
            if not isinstance(migration_id, str) or not migration_id:
                raise RestoreMigrationResumeError("RESTORE_MIGRATION_LINEAGE_CONFLICT")
            if migration_id == MIGRATION_ID:
                exact.append(record)
        return tuple(exact)

    @staticmethod
    def _assert_exact_migration_carrier(record: PersistenceRecord) -> None:
        if record.representation_name not in MIGRATION_FAMILY_REPRESENTATIONS:
            raise RestoreMigrationResumeError("RESTORE_MIGRATION_LINEAGE_CONFLICT")
        try:
            migration_id = migration_mapping_payload(record.payload)["migration_id"]
        except (MigrationError, KeyError) as exc:
            raise RestoreMigrationResumeError("RESTORE_MIGRATION_LINEAGE_CONFLICT") from exc
        if not isinstance(migration_id, str) or not migration_id or migration_id != MIGRATION_ID:
            raise RestoreMigrationResumeError("RESTORE_MIGRATION_LINEAGE_CONFLICT")

    @staticmethod
    def _assert_lineage(
        source: BackupEnvelope,
        manifest: _LineageBinding,
        snapshot: StateStoreSnapshot,
    ) -> tuple[str, ...]:
        prefix = tuple(
            item
            for item in snapshot.transaction_descriptors
            if item.target_generation <= manifest.source_generation
        )
        if prefix != source.integrity_metadata.state_store_transaction_descriptors:
            raise RestoreMigrationResumeError("RESTORE_MIGRATION_LINEAGE_CONFLICT")
        if manifest.migration_id != MIGRATION_ID:
            raise RestoreMigrationResumeError("RESTORE_MIGRATION_LINEAGE_CONFLICT")
        suffix = tuple(
            item
            for item in snapshot.transaction_descriptors
            if item.target_generation > manifest.source_generation
        )
        predecessors = () if not suffix else (prefix[-1], *suffix[:-1])
        roles = tuple(
            RestoreMigrationResumeCoordinator._classify_suffix_descriptor(previous, descriptor)
            for previous, descriptor in zip(predecessors, suffix, strict=True)
        )
        if roles not in _ALLOWED_SUFFIX_ROLE_PREFIXES:
            raise RestoreMigrationResumeError("RESTORE_MIGRATION_LINEAGE_CONFLICT")
        return roles

    @staticmethod
    def _classify_suffix_descriptor(
        previous: StateStoreTransactionDescriptor,
        descriptor: StateStoreTransactionDescriptor,
    ) -> str:
        current, history = descriptor.current_record_mutations, descriptor.immutable_history_appends
        if not current and len(history) == 1:
            declaration = history[0]
            RestoreMigrationResumeCoordinator._assert_exact_migration_carrier(declaration)
            if (
                declaration.representation_name != "Migration execution declaration"
                or previous.state_store_schema_version != 1
                or descriptor.state_store_schema_version != 2
            ):
                raise RestoreMigrationResumeError("RESTORE_MIGRATION_LINEAGE_CONFLICT")
            return "STRUCTURAL_MIGRATION"
        if len(current) != 1 or len(history) != 1:
            raise RestoreMigrationResumeError("RESTORE_MIGRATION_LINEAGE_CONFLICT")
        current_record, transition_record = current[0], history[0]
        for record in (current_record, transition_record):
            RestoreMigrationResumeCoordinator._assert_exact_migration_carrier(record)
        if (
            current_record.representation_name != "Migration current state/designation"
            or transition_record.representation_name != "Migration transition/history revisions"
        ):
            raise RestoreMigrationResumeError("RESTORE_MIGRATION_LINEAGE_CONFLICT")
        try:
            current_payload = migration_mapping_payload(current_record.payload)
            transition = migration_mapping_payload(transition_record.payload)
            state = transition["state"]
            observation_generation = transition["protected_freshness_generation"]
            matches = (
                current_payload["current_transition_revision"] == transition["transition_revision"]
                and current_payload["state"] == state
                and current_payload["authoritative_state_fingerprint_sha256"]
                == transition["state_fingerprint_sha256"]
                and current_payload["protected_freshness_generation"] == observation_generation
                and descriptor.expected_current_generation == observation_generation
                and descriptor.target_generation == observation_generation + 1
                and descriptor.pre_state_fingerprint_sha256
                == transition["state_fingerprint_sha256"]
                and transition["transaction_fingerprint_sha256"]
                == previous.transaction_fingerprint_sha256
            )
        except (KeyError, TypeError) as exc:
            raise RestoreMigrationResumeError("RESTORE_MIGRATION_LINEAGE_CONFLICT") from exc
        if state not in {"PREPARED", "APPLYING", "DURABLE_MIGRATED", "COMPLETED"}:
            raise RestoreMigrationResumeError("RESTORE_MIGRATION_LINEAGE_CONFLICT")
        expected_schema = 1 if state in {"PREPARED", "APPLYING"} else 2
        if not matches or descriptor.state_store_schema_version != expected_schema:
            raise RestoreMigrationResumeError("RESTORE_MIGRATION_LINEAGE_CONFLICT")
        return str(state)

    def _assert_external_matches(
        self, scope: tuple[str, str, str], snapshot: StateStoreSnapshot
    ) -> None:
        resolved = self._external.resolve_current(scope)
        if resolved is None:
            raise RestoreMigrationResumeError("RESTORE_MIGRATION_EXTERNAL_MISMATCH")
        record = ProtectedFreshnessAuthorityRecord.from_mapping(resolved[1])
        metadata = snapshot.metadata
        if (
            record.scope != scope
            or record.lifecycle != "COMMITTED"
            or record.committed_generation != metadata.protected_freshness_generation
            or record.committed_state_fingerprint_sha256 != metadata.state_fingerprint_sha256
        ):
            raise RestoreMigrationResumeError("RESTORE_MIGRATION_EXTERNAL_MISMATCH")


__all__ = [
    "RestoreMigrationCompletedDisposition",
    "RestoreMigrationCompletedStaging",
    "RestoreMigrationResumeCoordinator",
    "RestoreMigrationResumeError",
]
