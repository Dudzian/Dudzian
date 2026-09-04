"""S7C externally-authorized, isolated durable StateStore restore protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import hashlib
import os
from pathlib import Path
import shutil
import sqlite3
from tempfile import NamedTemporaryFile
from threading import RLock
from typing import Any, Protocol, cast

from .backup_envelope import (
    BackupEnvelope,
    BackupEnvelopeError,
    validate_backup_envelope,
)
from .local_durable_evidence import EvidenceScope, LocalDurableEvidenceRegistry
from .protected_freshness_handoff import (
    ProtectedFreshnessAuthorityPort,
    ProtectedFreshnessAuthorityRecord,
    ProtectedFreshnessHandoffCoordinator,
    ProtectedFreshnessHandoffError,
)
from .state_store import (
    SQLiteStateStore,
    StateStoreError,
    StateStoreMetadata,
    StateStoreSnapshot,
)
from .physical_backup import (
    AuthenticatedPhysicalBackupCandidate,
    PhysicalBackupAdmissionError,
    PhysicalBackupAdmissionValidator,
    TrustedPhysicalBackupArtifact,
)
from .secret_handoff import (
    SecretHandoffError,
    SecretHandoffRecord,
    validate_secret_handoff_snapshot,
)
from .lifecycle_records import LifecycleIntegrityError, validate_chain
from .migration_execution import (
    MigrationExecutionDeclaration,
    MigrationExecutionAuthority,
    MigrationExecutionError,
)
from .migration_execution_contract import thaw_json
from .migration_protocol import (
    MIGRATION_TRANSITIONS,
    MigrationDefinition,
    MigrationError,
    MigrationRegistry,
)


class RestoreDecision(str, Enum):
    NOOP_ALREADY_CURRENT = "NOOP_ALREADY_CURRENT"
    RESTORE_EXTERNAL_COMMITTED_CURRENT = "RESTORE_EXTERNAL_COMMITTED_CURRENT"
    RESTORE_EXACT_PROTECTED_PENDING_AND_FINALIZE = (
        "RESTORE_EXACT_PROTECTED_PENDING_AND_FINALIZE"
    )
    DENY = "DENY"


class LocalRestoreClassification(str, Enum):
    EMPTY = "EMPTY"
    NO_TRUSTED_LOCAL_OBSERVATION = "NO_TRUSTED_LOCAL_OBSERVATION"
    CORRUPT_OR_UNREADABLE = "CORRUPT_OR_UNREADABLE"
    BEHIND = "BEHIND"
    EXACT = "EXACT"
    SAME_GENERATION_DIFFERENT_STATE = "SAME_GENERATION_DIFFERENT_STATE"
    SAME_GENERATION_STATE_MATCH_LINEAGE_MISMATCH = (
        "SAME_GENERATION_STATE_MATCH_LINEAGE_MISMATCH"
    )
    AHEAD = "AHEAD"
    SCOPE_CONFLICT = "SCOPE_CONFLICT"
    ENVIRONMENT_CONFLICT = "ENVIRONMENT_CONFLICT"


class _InstallOutcome(str, Enum):
    INSTALLED = "INSTALLED"
    ALREADY_EXACT = "ALREADY_EXACT"


@dataclass(frozen=True, slots=True)
class RestoreResult:
    decision: RestoreDecision
    reason: str


class SecretHandoffRestoreFence(str, Enum):
    """The closed set of read-only restore observation fences."""

    INITIAL_STAGE_2 = "INITIAL_STAGE_2"
    PRE_INSTALL = "PRE_INSTALL"
    FINAL_PROMOTION = "FINAL_PROMOTION"


@dataclass(frozen=True, slots=True)
class SecretHandoffRestoreObservation:
    handoff_id: str
    scope: tuple[str, str]
    operation_fingerprint_sha256: str
    metadata_fingerprint_sha256: str
    external_state: str


class SecretHandoffRestoreAuthorityPort(Protocol):
    def observe(
        self,
        descriptor: SecretHandoffRecord,
    ) -> SecretHandoffRestoreObservation: ...


class MigrationRestoreAuthorityPort(Protocol):
    """Read-only sealed migration authority; implementations must never execute SQL."""

    def revalidate(
        self, snapshot: StateStoreSnapshot, sqlite_schema_fingerprint_sha256: str
    ) -> None: ...


class SealedMigrationRestoreAuthority:
    """Read-only restore verifier rooted exclusively in a sealed registry."""

    _NAMES = frozenset(
        {
            "Migration transition/history revisions",
            "Migration current state/designation",
            "Migration execution declaration",
        }
    )

    def __init__(self, registry: MigrationRegistry) -> None:
        if not isinstance(registry, MigrationRegistry):
            raise TypeError("migration restore authority requires MigrationRegistry")
        self._registry = registry

    @staticmethod
    def _ids(snapshot: StateStoreSnapshot) -> tuple[str, ...]:
        ids: set[str] = set()
        for carrier in (*snapshot.current_records, *snapshot.immutable_history):
            if carrier.representation_name in SealedMigrationRestoreAuthority._NAMES:
                migration_id = carrier.payload.get("migration_id")
                if not isinstance(migration_id, str) or not migration_id:
                    raise MigrationError("migration carrier identity is invalid")
                ids.add(migration_id)
        return tuple(sorted(ids))

    def revalidate(
        self, snapshot: StateStoreSnapshot, sqlite_schema_fingerprint_sha256: str
    ) -> None:
        chain: list[
            tuple[
                MigrationDefinition,
                MigrationExecutionAuthority,
                MigrationExecutionDeclaration | None,
            ]
        ] = []
        for migration_id in self._ids(snapshot):
            definition = self._registry.definition_for(migration_id)
            authority = self._registry.execution_authority_for(migration_id)
            authority.assert_definition(definition)
            current_record, history_records = SQLiteStateStore._select_lifecycle(
                snapshot,
                identity=migration_id,
                current_name="Migration current state/designation",
                history_name="Migration transition/history revisions",
                current_key=f"migration-current:{migration_id}",
                history_key_prefix=f"migration-transition:{migration_id}:",
            )
            history = tuple(item.payload for item in history_records)
            current = None if current_record is None else current_record.payload
            if current is None:
                raise MigrationError("migration has no current designation")
            validate_chain(
                history,
                current,
                identity_field="migration_id",
                allowed=MIGRATION_TRANSITIONS,
                transition_hash_field="transition_fingerprint_sha256",
                current_hash_field="designation_fingerprint_sha256",
            )
            declarations = tuple(
                item
                for item in snapshot.immutable_history
                if item.representation_name == "Migration execution declaration"
                and item.payload.get("migration_id") == migration_id
            )
            state = str(current["state"])
            predecessor = (
                str(history[-2]["state"])
                if state == "FAILED" and len(history) > 1
                else None
            )
            allowed = {
                "PREPARED": {0},
                "APPLYING": {0, 1},
                "DURABLE_MIGRATED": {1},
                "COMPLETED": {1},
                "FAILED": (
                    {0, 1}
                    if predecessor == "APPLYING"
                    else ({1} if predecessor == "DURABLE_MIGRATED" else {0})
                ),
            }.get(state, set())
            if len(declarations) not in allowed:
                raise MigrationError("migration declaration cardinality is invalid")
            declaration = None
            if declarations:
                declaration = MigrationExecutionDeclaration.from_mapping(
                    thaw_json(declarations[0].payload)
                )
                if declarations[0] != declaration.carrier():
                    raise MigrationError("migration declaration carrier mismatch")
                authority.assert_declaration(declaration)
                metadata = snapshot.metadata
                if (
                    declaration.account_id,
                    declaration.device_installation_id,
                    declaration.environment,
                ) != (
                    metadata.account_id,
                    metadata.device_installation_id,
                    metadata.environment,
                ) or declaration.state_store_identity_fingerprint_sha256 != metadata.state_store_identity_fingerprint_sha256:
                    raise MigrationError("migration declaration scope mismatch")
                self._assert_materialization(snapshot, definition, declaration)
            chain.append((definition, authority, declaration))
        self._assert_schema_chain(snapshot, sqlite_schema_fingerprint_sha256, chain)

    @staticmethod
    def _assert_materialization(
        snapshot: StateStoreSnapshot,
        definition: MigrationDefinition,
        declaration: MigrationExecutionDeclaration,
    ) -> None:
        descriptors = tuple(
            item
            for item in snapshot.transaction_descriptors
            if item.target_generation == declaration.target_generation
        )
        predecessors = tuple(
            item
            for item in snapshot.transaction_descriptors
            if item.target_generation == declaration.expected_current_generation
        )
        if len(descriptors) != 1 or len(predecessors) != 1:
            raise MigrationError("migration transaction edge is unavailable")
        descriptor = descriptors[0]
        predecessor = predecessors[0]
        metadata = snapshot.metadata
        carrier = declaration.carrier()
        if (
            declaration.target_generation != declaration.expected_current_generation + 1
            or descriptor.expected_current_generation
            != declaration.expected_current_generation
            or predecessor.target_generation != declaration.expected_current_generation
            or predecessor.post_state_fingerprint_sha256
            != declaration.pre_state_fingerprint_sha256
            or predecessor.post_history_tail_fingerprint_sha256
            != declaration.pre_history_tail_fingerprint_sha256
            or descriptor.pre_state_fingerprint_sha256
            != declaration.pre_state_fingerprint_sha256
            or descriptor.pre_history_tail_fingerprint_sha256
            != declaration.pre_history_tail_fingerprint_sha256
            or descriptor.immutable_history_appends != (carrier,)
            or descriptor.current_record_mutations != ()
            or descriptor.state_store_schema_version != definition.target_schema_version
            or predecessor.state_store_schema_version
            != definition.source_schema_version
            or (
                descriptor.account_id,
                descriptor.device_installation_id,
                descriptor.environment,
                descriptor.state_store_identity_fingerprint_sha256,
            )
            != (
                metadata.account_id,
                metadata.device_installation_id,
                metadata.environment,
                metadata.state_store_identity_fingerprint_sha256,
            )
            or (
                predecessor.account_id,
                predecessor.device_installation_id,
                predecessor.environment,
                predecessor.state_store_identity_fingerprint_sha256,
            )
            != (
                metadata.account_id,
                metadata.device_installation_id,
                metadata.environment,
                metadata.state_store_identity_fingerprint_sha256,
            )
            or not descriptor.has_valid_transaction_fingerprint()
            or not predecessor.has_valid_transaction_fingerprint()
        ):
            raise MigrationError("migration declaration is not exactly materialized")

    @staticmethod
    def _assert_schema_chain(
        snapshot: StateStoreSnapshot,
        physical_schema: str,
        entries: list[
            tuple[
                MigrationDefinition,
                MigrationExecutionAuthority,
                MigrationExecutionDeclaration | None,
            ]
        ],
    ) -> None:
        if not entries:
            return
        by_source: dict[
            int,
            tuple[
                MigrationDefinition,
                MigrationExecutionAuthority,
                MigrationExecutionDeclaration | None,
            ],
        ] = {}
        targets: set[int] = set()
        for entry in entries:
            definition = entry[0]
            source = definition.source_schema_version
            target = definition.target_schema_version
            if source in by_source or target in targets:
                raise MigrationError("migration schema chain branches")
            by_source[source] = entry
            targets.add(target)
        starts = tuple(source for source in by_source if source not in targets)
        if len(starts) != 1:
            raise MigrationError("migration schema chain has no unique head")
        ordered: list[
            tuple[
                MigrationDefinition,
                MigrationExecutionAuthority,
                MigrationExecutionDeclaration | None,
            ]
        ] = []
        version = starts[0]
        while version in by_source:
            entry = by_source[version]
            if entry in ordered:
                raise MigrationError("migration schema chain cycles")
            ordered.append(entry)
            version = entry[0].target_schema_version
        if len(ordered) != len(entries):
            raise MigrationError("migration schema chain is disconnected")
        for left, right in zip(ordered, ordered[1:]):
            if (
                left[1].target_sqlite_schema_fingerprint_sha256
                != right[1].pre_sqlite_schema_fingerprint_sha256
            ):
                raise MigrationError("migration sealed schema chain mismatch")
        materialized = [entry for entry in ordered if entry[2] is not None]
        materialized_count = len(materialized)
        if any(entry[2] is None for entry in ordered[:materialized_count]) or any(
            entry[2] is not None for entry in ordered[materialized_count:]
        ):
            raise MigrationError("materialized migrations do not form a prefix")
        if len(ordered) - materialized_count > 1:
            raise MigrationError("migration chain has an impossible active tail")
        declarations = [entry[2] for entry in materialized]
        if any(
            declaration is None
            or declaration.expected_current_generation >= declaration.target_generation
            for declaration in declarations
        ) or any(
            left.target_generation >= right.target_generation
            for left, right in zip(declarations, declarations[1:])
            if left is not None and right is not None
        ):
            raise MigrationError("migration materialization chronology is invalid")
        terminal = materialized[-1] if materialized else ordered[0]
        expected_version = (
            terminal[0].target_schema_version
            if materialized
            else terminal[0].source_schema_version
        )
        expected_schema = (
            terminal[1].target_sqlite_schema_fingerprint_sha256
            if materialized
            else terminal[1].pre_sqlite_schema_fingerprint_sha256
        )
        if (
            snapshot.metadata.state_store_schema_version != expected_version
            or physical_schema != expected_schema
        ):
            raise MigrationError("current schema does not match migration head")


@dataclass(frozen=True, slots=True)
class RestoreLifecycleAuthorityBundle:
    migration_restore_authority: MigrationRestoreAuthorityPort | None = None
    secret_handoff_restore_authority: SecretHandoffRestoreAuthorityPort | None = None

    @classmethod
    def from_migration_registry(
        cls,
        registry: MigrationRegistry,
        secret_handoff_restore_authority: (
            SecretHandoffRestoreAuthorityPort | None
        ) = None,
    ) -> "RestoreLifecycleAuthorityBundle":
        return cls(
            SealedMigrationRestoreAuthority(registry),
            secret_handoff_restore_authority,
        )


@dataclass(frozen=True, slots=True)
class _ExternalTarget:
    ref: object
    record: ProtectedFreshnessAuthorityRecord


class S7CRestoreCoordinator:
    """Execute M0.11 restore without creating any new authority surface."""

    def __init__(
        self,
        live_path: str | Path,
        registry: LocalDurableEvidenceRegistry,
        authority: ProtectedFreshnessAuthorityPort,
    ) -> None:
        self._live_path = Path(live_path)
        self._registry = registry
        self._authority = authority
        self._lock = RLock()

    @staticmethod
    def _scope(candidate: BackupEnvelope) -> EvidenceScope:
        return (
            candidate.account_id,
            candidate.device_installation_id,
            candidate.state_store_identity_fingerprint_sha256,
        )

    def _external(self, scope: EvidenceScope) -> _ExternalTarget:
        resolved = self._authority.resolve_current(scope)
        if resolved is None:
            raise ProtectedFreshnessHandoffError(
                "external current membership is missing"
            )
        ref, raw = resolved
        record = ProtectedFreshnessAuthorityRecord.from_mapping(raw)
        if record.scope != scope:
            raise ProtectedFreshnessHandoffError("external current scope mismatch")
        return _ExternalTarget(ref, record)

    @staticmethod
    def _eligible(
        candidate: BackupEnvelope, external: ProtectedFreshnessAuthorityRecord
    ) -> bool:
        generation = candidate.local_protected_freshness_generation
        if external.lifecycle == "COMMITTED":
            return bool(
                generation == external.committed_generation
                and candidate.state_fingerprint_sha256
                == external.committed_state_fingerprint_sha256
            )
        if external.lifecycle == "PREPARED":
            return bool(
                generation == external.prepared_generation
                and candidate.state_fingerprint_sha256
                == external.prepared_state_fingerprint_sha256
                and candidate.transaction_fingerprint_sha256
                == external.prepared_transaction_fingerprint_sha256
            )
        return False

    def _classify(
        self, candidate: BackupEnvelope
    ) -> tuple[LocalRestoreClassification, StateStoreSnapshot | None]:
        if not self._live_path.exists():
            return LocalRestoreClassification.NO_TRUSTED_LOCAL_OBSERVATION, None
        try:
            with SQLiteStateStore(self._live_path) as store:
                snapshot = store.read_verified_snapshot()
        except (OSError, sqlite3.Error, StateStoreError):
            return LocalRestoreClassification.CORRUPT_OR_UNREADABLE, None
        if snapshot is None:
            return LocalRestoreClassification.EMPTY, None
        metadata = snapshot.metadata
        if (
            metadata.account_id,
            metadata.device_installation_id,
            metadata.state_store_identity_fingerprint_sha256,
        ) != self._scope(candidate):
            return LocalRestoreClassification.SCOPE_CONFLICT, snapshot
        if metadata.environment != candidate.environment:
            return LocalRestoreClassification.ENVIRONMENT_CONFLICT, snapshot
        generation = metadata.protected_freshness_generation
        target = candidate.local_protected_freshness_generation
        if generation < target:
            return LocalRestoreClassification.BEHIND, snapshot
        if generation > target:
            return LocalRestoreClassification.AHEAD, snapshot
        if metadata.state_fingerprint_sha256 != candidate.state_fingerprint_sha256:
            return LocalRestoreClassification.SAME_GENERATION_DIFFERENT_STATE, snapshot
        if (
            metadata.transaction_fingerprint_sha256
            != candidate.transaction_fingerprint_sha256
            or metadata.history_tail_fingerprint_sha256
            != candidate.history_tail_fingerprint_sha256
        ):
            return (
                LocalRestoreClassification.SAME_GENERATION_STATE_MATCH_LINEAGE_MISMATCH,
                snapshot,
            )
        return LocalRestoreClassification.EXACT, snapshot

    @staticmethod
    def _candidate_snapshot(candidate: BackupEnvelope) -> StateStoreSnapshot:
        return StateStoreSnapshot(
            StateStoreMetadata(
                account_id=candidate.account_id,
                device_installation_id=candidate.device_installation_id,
                state_store_schema_version=candidate.state_store_schema_version,
                state_store_identity_fingerprint_sha256=(
                    candidate.state_store_identity_fingerprint_sha256
                ),
                environment=candidate.environment,
                protected_freshness_generation=(
                    candidate.local_protected_freshness_generation
                ),
                state_fingerprint_sha256=candidate.state_fingerprint_sha256,
                transaction_fingerprint_sha256=candidate.transaction_fingerprint_sha256,
                history_tail_fingerprint_sha256=candidate.history_tail_fingerprint_sha256,
            ),
            candidate.canonical_durable_records,
            candidate.immutable_recovery_history,
            candidate.integrity_metadata.state_store_transaction_descriptors,
        )

    @staticmethod
    def _exact(snapshot: StateStoreSnapshot | None, candidate: BackupEnvelope) -> bool:
        if snapshot is None:
            return False
        metadata = snapshot.metadata
        return bool(
            metadata.account_id == candidate.account_id
            and metadata.device_installation_id == candidate.device_installation_id
            and metadata.state_store_identity_fingerprint_sha256
            == candidate.state_store_identity_fingerprint_sha256
            and metadata.environment == candidate.environment
            and metadata.protected_freshness_generation
            == candidate.local_protected_freshness_generation
            and metadata.state_fingerprint_sha256 == candidate.state_fingerprint_sha256
            and metadata.transaction_fingerprint_sha256
            == candidate.transaction_fingerprint_sha256
            and metadata.history_tail_fingerprint_sha256
            == candidate.history_tail_fingerprint_sha256
            and snapshot.transaction_descriptors
            == candidate.integrity_metadata.state_store_transaction_descriptors
        )

    def _restore_isolated(
        self,
        candidate: BackupEnvelope,
        expected: _ExternalTarget,
    ) -> _InstallOutcome:
        self._live_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            prefix=f".{self._live_path.name}.s7c-",
            suffix=".sqlite3",
            dir=self._live_path.parent,
            delete=False,
        ) as handle:
            isolated_path = Path(handle.name)
        isolated_path.unlink()
        isolated: SQLiteStateStore | None = None
        try:
            isolated = SQLiteStateStore(isolated_path)
            isolated.write_restored_snapshot(self._candidate_snapshot(candidate))
            if not self._exact(isolated.read_verified_snapshot(), candidate):
                raise StateStoreError("isolated restore verification mismatch")
            isolated.prepare_for_atomic_install()
            latest = self._external(self._scope(candidate))
            if latest != expected or not self._eligible(candidate, latest.record):
                raise StateStoreError(
                    "external authority changed before atomic install"
                )
            with SQLiteStateStore.installation_gate(self._live_path):
                fresh_classification, fresh_snapshot = self._classify(candidate)
                if fresh_classification is LocalRestoreClassification.EXACT:
                    return _InstallOutcome.ALREADY_EXACT
                if fresh_classification in {
                    LocalRestoreClassification.SAME_GENERATION_DIFFERENT_STATE,
                    LocalRestoreClassification.SAME_GENERATION_STATE_MATCH_LINEAGE_MISMATCH,
                    LocalRestoreClassification.AHEAD,
                    LocalRestoreClassification.SCOPE_CONFLICT,
                    LocalRestoreClassification.ENVIRONMENT_CONFLICT,
                }:
                    raise StateStoreError(
                        "fresh local classification forbids replacement"
                    )
                if fresh_classification not in {
                    LocalRestoreClassification.NO_TRUSTED_LOCAL_OBSERVATION,
                    LocalRestoreClassification.CORRUPT_OR_UNREADABLE,
                }:
                    live = SQLiteStateStore(self._live_path)
                    try:
                        if live.read_verified_snapshot() != fresh_snapshot:
                            raise StateStoreError(
                                "live StateStore changed before checkpoint"
                            )
                        live.prepare_for_atomic_install()
                    except BaseException:
                        live.close()
                        raise
                final_external = self._external(self._scope(candidate))
                if final_external != expected or not self._eligible(
                    candidate, final_external.record
                ):
                    raise StateStoreError(
                        "external authority changed at final install fence"
                    )
                SQLiteStateStore.atomic_replace(isolated_path, self._live_path)
            with SQLiteStateStore(self._live_path) as installed:
                if not self._exact(installed.read_verified_snapshot(), candidate):
                    raise StateStoreError("installed restore verification mismatch")
            return _InstallOutcome.INSTALLED
        finally:
            if isolated is not None:
                isolated.close()
            isolated_path.unlink(missing_ok=True)
            Path(f"{isolated_path}-wal").unlink(missing_ok=True)
            Path(f"{isolated_path}-shm").unlink(missing_ok=True)

    def restore(
        self, raw_candidate: Mapping[str, object] | BackupEnvelope
    ) -> RestoreResult:
        """Validate, authorize, restore if needed, and reconcile exactly once."""

        with self._lock:
            try:
                candidate = validate_backup_envelope(raw_candidate)
                scope = self._scope(candidate)
                external = self._external(scope)
                if not self._eligible(candidate, external.record):
                    return RestoreResult(
                        RestoreDecision.DENY, "candidate is not externally current"
                    )
                classification, _ = self._classify(candidate)
                denied = {
                    LocalRestoreClassification.SAME_GENERATION_DIFFERENT_STATE,
                    LocalRestoreClassification.SAME_GENERATION_STATE_MATCH_LINEAGE_MISMATCH,
                    LocalRestoreClassification.AHEAD,
                    LocalRestoreClassification.SCOPE_CONFLICT,
                    LocalRestoreClassification.ENVIRONMENT_CONFLICT,
                }
                if classification in denied:
                    return RestoreResult(RestoreDecision.DENY, classification.value)
                # Restore/resume owns evidence freshness for this exact scope.
                # Invalidation happens only after full validation, external
                # authorization, and local conflict fencing.
                self._registry.invalidate_scope(scope)
                installed = False
                if classification is not LocalRestoreClassification.EXACT:
                    installed = (
                        self._restore_isolated(candidate, external)
                        is _InstallOutcome.INSTALLED
                    )

                if external.record.lifecycle == "COMMITTED":
                    # EXACT controls durable idempotency, never evidence
                    # idempotency. Every authorized invocation rebuilds S4
                    # from a fresh verified live observation.
                    with SQLiteStateStore(self._live_path) as evidence_store:
                        if (
                            self._registry.publish_verified_state(evidence_store)
                            is None
                        ):
                            return RestoreResult(
                                RestoreDecision.DENY,
                                "fresh durable evidence unavailable",
                            )
                    latest = self._external(scope)
                    if latest != external or not self._eligible(
                        candidate, latest.record
                    ):
                        return RestoreResult(
                            RestoreDecision.DENY,
                            "external changed during reconciliation",
                        )
                    with SQLiteStateStore(self._live_path) as live:
                        if not self._exact(live.read_verified_snapshot(), candidate):
                            return RestoreResult(
                                RestoreDecision.DENY,
                                "local changed during reconciliation",
                            )
                    decision = (
                        RestoreDecision.RESTORE_EXTERNAL_COMMITTED_CURRENT
                        if installed
                        else RestoreDecision.NOOP_ALREADY_CURRENT
                    )
                    return RestoreResult(decision, classification.value)

                latest = self._external(scope)
                if latest != external or latest.record.lifecycle != "PREPARED":
                    return RestoreResult(
                        RestoreDecision.DENY, "prepared authority changed"
                    )
                # Re-run the full production validator at the protected-action
                # fence, including the ACK-loss reconciliation precondition.
                candidate = validate_backup_envelope(candidate)
                with SQLiteStateStore(self._live_path) as live:
                    coordinator = ProtectedFreshnessHandoffCoordinator(
                        live, self._registry, self._authority
                    )
                    coordinator.recover_protected_state(scope)
                    if not self._exact(live.read_verified_snapshot(), candidate):
                        return RestoreResult(
                            RestoreDecision.DENY, "local changed after FINALIZE"
                        )
                return RestoreResult(
                    RestoreDecision.RESTORE_EXACT_PROTECTED_PENDING_AND_FINALIZE,
                    classification.value,
                )
            except (
                BackupEnvelopeError,
                OSError,
                StateStoreError,
                ProtectedFreshnessHandoffError,
            ):
                return RestoreResult(RestoreDecision.DENY, "fail-closed restore denial")


class TrustedPhysicalRestoreCoordinator(S7CRestoreCoordinator):
    """C2D-admitted physical restore through the frozen S7C authority boundary.

    ``restore_trusted_artifact`` owns and always closes the admission lease.  The
    lower-level ``restore_admitted`` borrows its candidate and never closes it.
    """

    _SECRET_STATES = {
        "PREPARED": {"NOT_STARTED", "COMMITTED", "UNRESOLVED"},
        "COMMITTED": {"COMMITTED", "CLEANUP_ACCEPTED_OR_SATISFIED"},
        "CLEANUP_PENDING": {"CLEANUP_ACCEPTED_OR_SATISFIED"},
        "UNKNOWN_RECONCILIATION": {"UNRESOLVED"},
    }

    def __init__(
        self,
        live_path: str | Path,
        registry: LocalDurableEvidenceRegistry,
        authority: ProtectedFreshnessAuthorityPort,
        admission: PhysicalBackupAdmissionValidator,
        lifecycle_authority: RestoreLifecycleAuthorityBundle,
    ) -> None:
        super().__init__(live_path, registry, authority)
        self._admission = admission
        self._lifecycle_authority = lifecycle_authority

    @staticmethod
    def _secret_lifecycles(
        snapshot: StateStoreSnapshot,
    ) -> tuple[tuple[SecretHandoffRecord, str], ...]:
        return cast(
            tuple[tuple[SecretHandoffRecord, str], ...],
            validate_secret_handoff_snapshot(snapshot),
        )

    @staticmethod
    def _migration_relations(snapshot: StateStoreSnapshot) -> None:
        """Validate local Migration current/history relations without authority calls."""

        for migration_id in SealedMigrationRestoreAuthority._ids(snapshot):
            current_record, history_records = SQLiteStateStore._select_lifecycle(
                snapshot,
                identity=migration_id,
                current_name="Migration current state/designation",
                history_name="Migration transition/history revisions",
                current_key=f"migration-current:{migration_id}",
                history_key_prefix=f"migration-transition:{migration_id}:",
            )
            if current_record is None:
                raise MigrationError("migration has no current designation")
            history = tuple(item.payload for item in history_records)
            current = current_record.payload
            validate_chain(
                history,
                current,
                identity_field="migration_id",
                allowed=MIGRATION_TRANSITIONS,
                transition_hash_field="transition_fingerprint_sha256",
                current_hash_field="designation_fingerprint_sha256",
            )
            if (
                current["authoritative_state_fingerprint_sha256"]
                != history[-1]["state_fingerprint_sha256"]
                or current["protected_freshness_generation"]
                != history[-1]["protected_freshness_generation"]
            ):
                raise MigrationError(
                    "migration current designation is not bound to latest observation"
                )

    def _observe(
        self,
        lifecycles: tuple[tuple[SecretHandoffRecord, str], ...],
        fence: SecretHandoffRestoreFence,
    ) -> None:
        if not lifecycles:
            return
        authority = self._lifecycle_authority.secret_handoff_restore_authority
        if authority is None:
            raise SecretHandoffError("secret restore authority unavailable")
        for descriptor, lifecycle in lifecycles:
            # The fence is coordinator timing context, never external authority input.
            observation = authority.observe(descriptor)
            expected = (
                descriptor.handoff_id,
                descriptor.scope,
                descriptor.operation_fingerprint_sha256,
                descriptor.metadata_fingerprint_sha256,
            )
            actual = (
                observation.handoff_id,
                observation.scope,
                observation.operation_fingerprint_sha256,
                observation.metadata_fingerprint_sha256,
            )
            if (
                actual != expected
                or observation.external_state
                not in self._SECRET_STATES.get(lifecycle, set())
            ):
                raise SecretHandoffError("secret restore authority rejected candidate")

    @staticmethod
    def _copy_for_install(
        candidate: AuthenticatedPhysicalBackupCandidate, target: Path
    ) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        with (
            NamedTemporaryFile(
                prefix=f".{target.name}.physical-",
                suffix=".sqlite3",
                dir=target.parent,
                delete=False,
            ) as output,
            candidate.private_path.open("rb") as source,
        ):
            staged = Path(output.name)
            shutil.copyfileobj(source, output)
            output.flush()
            os.fsync(output.fileno())
        if not TrustedPhysicalRestoreCoordinator._staged_matches(candidate, staged):
            staged.unlink(missing_ok=True)
            raise StateStoreError("physical install staging continuity mismatch")
        return staged

    @staticmethod
    def _staged_matches(
        candidate: AuthenticatedPhysicalBackupCandidate, staged: Path
    ) -> bool:
        digest = hashlib.sha256()
        length = 0
        with staged.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                length += len(chunk)
        return bool(
            length == candidate.physical_artifact_byte_length
            and digest.hexdigest() == candidate.physical_artifact_sha256
        )

    def _restore_exact_without_admission(
        self, envelope: BackupEnvelope
    ) -> RestoreResult:
        """Authorize an exact live store without consulting the physical artifact."""

        with SQLiteStateStore(self._live_path) as live:
            snapshot = live.read_verified_snapshot()
            if snapshot is None or not self._exact(snapshot, envelope):
                raise StateStoreError("exact live snapshot became unavailable")
            schema = live.sqlite_schema_fingerprint()
        lifecycles = self._secret_lifecycles(snapshot)
        self._migration_relations(snapshot)  # Stage 1, local and intrinsic only
        migration = self._lifecycle_authority.migration_restore_authority
        has_migration = any(
            record.representation_name in SealedMigrationRestoreAuthority._NAMES
            for record in (*snapshot.current_records, *snapshot.immutable_history)
        )
        if has_migration and not isinstance(migration, SealedMigrationRestoreAuthority):
            raise MigrationError("sealed migration restore authority unavailable")
        if migration is not None and has_migration:
            migration.revalidate(snapshot, schema)
        self._observe(lifecycles, SecretHandoffRestoreFence.INITIAL_STAGE_2)
        self._migration_relations(snapshot)
        self._secret_lifecycles(snapshot)  # explicit Stage 3 relational revalidation
        scope = self._scope(envelope)
        external = self._external(scope)
        if not self._eligible(envelope, external.record):
            raise StateStoreError("candidate is not externally current")
        self._registry.invalidate_scope(scope)
        if external.record.lifecycle == "PREPARED":
            self._observe(lifecycles, SecretHandoffRestoreFence.FINAL_PROMOTION)
            with SQLiteStateStore(self._live_path) as live:
                ProtectedFreshnessHandoffCoordinator(
                    live, self._registry, self._authority
                ).recover_protected_state(scope)
        with SQLiteStateStore(self._live_path) as live:
            if not self._exact(live.read_verified_snapshot(), envelope):
                raise StateStoreError("live changed during no-op validation")
        self._observe(lifecycles, SecretHandoffRestoreFence.FINAL_PROMOTION)
        with SQLiteStateStore(self._live_path) as evidence_store:
            if self._registry.publish_verified_state(evidence_store) is None:
                raise StateStoreError("fresh durable evidence unavailable")
        return RestoreResult(
            (
                RestoreDecision.RESTORE_EXACT_PROTECTED_PENDING_AND_FINALIZE
                if external.record.lifecycle == "PREPARED"
                else RestoreDecision.NOOP_ALREADY_CURRENT
            ),
            LocalRestoreClassification.EXACT.value,
        )

    def restore_trusted_artifact(
        self, artifact: TrustedPhysicalBackupArtifact
    ) -> RestoreResult:
        """Admit a trusted bundle and own its private lease through completion."""

        with self._lock:
            try:
                envelope = validate_backup_envelope(artifact.backup_envelope)
                classification, _ = self._classify(envelope)
                if classification is LocalRestoreClassification.EXACT:
                    return self._restore_exact_without_admission(envelope)
                candidate = self._admission.admit(artifact)
            except (
                BackupEnvelopeError,
                MigrationError,
                MigrationExecutionError,
                OSError,
                sqlite3.Error,
                StateStoreError,
                ProtectedFreshnessHandoffError,
                PhysicalBackupAdmissionError,
                SecretHandoffError,
                LifecycleIntegrityError,
            ):
                return RestoreResult(
                    RestoreDecision.DENY, "physical artifact admission denied"
                )
            try:
                return self.restore_admitted(candidate)
            finally:
                candidate.close()

    def restore_admitted(
        self, candidate: AuthenticatedPhysicalBackupCandidate
    ) -> RestoreResult:
        """Borrow an authenticated C2D candidate; raw filesystem paths are not accepted."""

        with self._lock:
            staged: Path | None = None
            try:
                if not isinstance(candidate, AuthenticatedPhysicalBackupCandidate) or (
                    candidate.classification
                    != "ELIGIBLE_FOR_FURTHER_RESTORE_VALIDATION_ONLY"
                ):
                    raise StateStoreError(
                        "physical restore requires an admitted candidate"
                    )
                # Admission continuity is the prerequisite, not lifecycle Stage 1.
                candidate.verify_physical_continuity()
                envelope = validate_backup_envelope(
                    candidate.backup_envelope
                )  # Stage 1
                snapshot = candidate.state_store_snapshot
                lifecycles = self._secret_lifecycles(snapshot)
                self._migration_relations(snapshot)  # Stage 1

                migration = self._lifecycle_authority.migration_restore_authority
                has_migration = any(
                    record.representation_name in SealedMigrationRestoreAuthority._NAMES
                    for record in (
                        *snapshot.current_records,
                        *snapshot.immutable_history,
                    )
                )
                if has_migration and not isinstance(
                    migration, SealedMigrationRestoreAuthority
                ):
                    raise StateStoreError(
                        "sealed migration restore authority unavailable"
                    )
                if migration is not None and has_migration:
                    migration.revalidate(
                        snapshot, candidate.sqlite_schema_fingerprint_sha256
                    )
                self._observe(lifecycles, SecretHandoffRestoreFence.INITIAL_STAGE_2)
                # Stage 3: repeat relational validation after all Stage-2 authority succeeds.
                self._migration_relations(snapshot)
                self._secret_lifecycles(snapshot)

                scope = self._scope(envelope)
                external = self._external(scope)  # Stage 4 assessment
                if not self._eligible(envelope, external.record):
                    raise StateStoreError("candidate is not externally current")
                classification, _ = self._classify(envelope)
                if classification in {
                    LocalRestoreClassification.SAME_GENERATION_DIFFERENT_STATE,
                    LocalRestoreClassification.SAME_GENERATION_STATE_MATCH_LINEAGE_MISMATCH,
                    LocalRestoreClassification.AHEAD,
                    LocalRestoreClassification.SCOPE_CONFLICT,
                    LocalRestoreClassification.ENVIRONMENT_CONFLICT,
                }:
                    raise StateStoreError("local classification forbids replacement")
                self._registry.invalidate_scope(scope)

                installed = False
                if classification is not LocalRestoreClassification.EXACT:
                    candidate.verify_physical_continuity()
                    staged = self._copy_for_install(candidate, self._live_path)
                    with SQLiteStateStore.installation_gate(self._live_path):
                        fresh, fresh_snapshot = self._classify(envelope)
                        if fresh is LocalRestoreClassification.EXACT:
                            staged.unlink(missing_ok=True)
                            staged = None
                        else:
                            if fresh in {
                                LocalRestoreClassification.SAME_GENERATION_DIFFERENT_STATE,
                                LocalRestoreClassification.SAME_GENERATION_STATE_MATCH_LINEAGE_MISMATCH,
                                LocalRestoreClassification.AHEAD,
                                LocalRestoreClassification.SCOPE_CONFLICT,
                                LocalRestoreClassification.ENVIRONMENT_CONFLICT,
                            }:
                                raise StateStoreError(
                                    "fresh local classification forbids replacement"
                                )
                            if fresh not in {
                                LocalRestoreClassification.NO_TRUSTED_LOCAL_OBSERVATION,
                                LocalRestoreClassification.CORRUPT_OR_UNREADABLE,
                            }:
                                with SQLiteStateStore(self._live_path) as live:
                                    if live.read_verified_snapshot() != fresh_snapshot:
                                        raise StateStoreError(
                                            "live StateStore changed before checkpoint"
                                        )
                                    live.prepare_for_atomic_install()
                            candidate.verify_physical_continuity()
                            latest = self._external(scope)
                            if latest != external or not self._eligible(
                                envelope, latest.record
                            ):
                                raise StateStoreError(
                                    "external authority changed before install"
                                )
                            if not self._staged_matches(candidate, staged):
                                raise StateStoreError(
                                    "final staged continuity mismatch"
                                )
                            self._observe(
                                lifecycles, SecretHandoffRestoreFence.PRE_INSTALL
                            )
                            SQLiteStateStore.atomic_replace(staged, self._live_path)
                            staged = None
                            installed = True

                with SQLiteStateStore(self._live_path) as live:
                    if not self._exact(live.read_verified_snapshot(), envelope):
                        raise StateStoreError("installed restore verification mismatch")
                    if external.record.lifecycle == "PREPARED":
                        self._observe(
                            lifecycles, SecretHandoffRestoreFence.FINAL_PROMOTION
                        )
                        ProtectedFreshnessHandoffCoordinator(
                            live, self._registry, self._authority
                        ).recover_protected_state(scope)
                        if not self._exact(live.read_verified_snapshot(), envelope):
                            raise StateStoreError("local changed after FINALIZE")

                latest = self._external(scope)
                if external.record.lifecycle == "COMMITTED" and (
                    latest != external or not self._eligible(envelope, latest.record)
                ):
                    raise StateStoreError("external authority changed before promotion")
                self._observe(lifecycles, SecretHandoffRestoreFence.FINAL_PROMOTION)
                with SQLiteStateStore(self._live_path) as evidence_store:
                    if self._registry.publish_verified_state(evidence_store) is None:
                        raise StateStoreError("fresh durable evidence unavailable")
                if external.record.lifecycle == "PREPARED":
                    decision = (
                        RestoreDecision.RESTORE_EXACT_PROTECTED_PENDING_AND_FINALIZE
                    )
                elif installed:
                    decision = RestoreDecision.RESTORE_EXTERNAL_COMMITTED_CURRENT
                else:
                    decision = RestoreDecision.NOOP_ALREADY_CURRENT
                return RestoreResult(decision, classification.value)
            except (
                BackupEnvelopeError,
                OSError,
                StateStoreError,
                ProtectedFreshnessHandoffError,
                PhysicalBackupAdmissionError,
                SecretHandoffError,
                LifecycleIntegrityError,
                MigrationError,
                MigrationExecutionError,
            ):
                return RestoreResult(
                    RestoreDecision.DENY, "fail-closed physical restore denial"
                )
            finally:
                if staged is not None:
                    staged.unlink(missing_ok=True)
