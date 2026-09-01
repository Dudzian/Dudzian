"""S7C externally-authorized, isolated durable StateStore restore protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sqlite3
from tempfile import NamedTemporaryFile
from threading import RLock
from typing import Any

from .backup_envelope import BackupEnvelope, BackupEnvelopeError, validate_backup_envelope
from .local_durable_evidence import EvidenceScope, LocalDurableEvidenceRegistry
from .protected_freshness_handoff import (
    ProtectedFreshnessAuthorityPort,
    ProtectedFreshnessAuthorityRecord,
    ProtectedFreshnessHandoffCoordinator,
    ProtectedFreshnessHandoffError,
)
from .state_store import SQLiteStateStore, StateStoreError, StateStoreMetadata, StateStoreSnapshot


class RestoreDecision(str, Enum):
    NOOP_ALREADY_CURRENT = "NOOP_ALREADY_CURRENT"
    RESTORE_EXTERNAL_COMMITTED_CURRENT = "RESTORE_EXTERNAL_COMMITTED_CURRENT"
    RESTORE_EXACT_PROTECTED_PENDING_AND_FINALIZE = "RESTORE_EXACT_PROTECTED_PENDING_AND_FINALIZE"
    DENY = "DENY"


class LocalRestoreClassification(str, Enum):
    EMPTY = "EMPTY"
    NO_TRUSTED_LOCAL_OBSERVATION = "NO_TRUSTED_LOCAL_OBSERVATION"
    CORRUPT_OR_UNREADABLE = "CORRUPT_OR_UNREADABLE"
    BEHIND = "BEHIND"
    EXACT = "EXACT"
    SAME_GENERATION_DIFFERENT_STATE = "SAME_GENERATION_DIFFERENT_STATE"
    SAME_GENERATION_STATE_MATCH_LINEAGE_MISMATCH = "SAME_GENERATION_STATE_MATCH_LINEAGE_MISMATCH"
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
            raise ProtectedFreshnessHandoffError("external current membership is missing")
        ref, raw = resolved
        record = ProtectedFreshnessAuthorityRecord.from_mapping(raw)
        if record.scope != scope:
            raise ProtectedFreshnessHandoffError("external current scope mismatch")
        return _ExternalTarget(ref, record)

    @staticmethod
    def _eligible(candidate: BackupEnvelope, external: ProtectedFreshnessAuthorityRecord) -> bool:
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
                and candidate.state_fingerprint_sha256 == external.prepared_state_fingerprint_sha256
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
            metadata.transaction_fingerprint_sha256 != candidate.transaction_fingerprint_sha256
            or metadata.history_tail_fingerprint_sha256 != candidate.history_tail_fingerprint_sha256
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
                protected_freshness_generation=(candidate.local_protected_freshness_generation),
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
            and metadata.transaction_fingerprint_sha256 == candidate.transaction_fingerprint_sha256
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
                raise StateStoreError("external authority changed before atomic install")
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
                    raise StateStoreError("fresh local classification forbids replacement")
                if fresh_classification not in {
                    LocalRestoreClassification.NO_TRUSTED_LOCAL_OBSERVATION,
                    LocalRestoreClassification.CORRUPT_OR_UNREADABLE,
                }:
                    live = SQLiteStateStore(self._live_path)
                    try:
                        if live.read_verified_snapshot() != fresh_snapshot:
                            raise StateStoreError("live StateStore changed before checkpoint")
                        live.prepare_for_atomic_install()
                    except BaseException:
                        live.close()
                        raise
                final_external = self._external(self._scope(candidate))
                if final_external != expected or not self._eligible(
                    candidate, final_external.record
                ):
                    raise StateStoreError("external authority changed at final install fence")
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

    def restore(self, raw_candidate: Mapping[str, object] | BackupEnvelope) -> RestoreResult:
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
                        self._restore_isolated(candidate, external) is _InstallOutcome.INSTALLED
                    )

                if external.record.lifecycle == "COMMITTED":
                    # EXACT controls durable idempotency, never evidence
                    # idempotency. Every authorized invocation rebuilds S4
                    # from a fresh verified live observation.
                    with SQLiteStateStore(self._live_path) as evidence_store:
                        if self._registry.publish_verified_state(evidence_store) is None:
                            return RestoreResult(
                                RestoreDecision.DENY, "fresh durable evidence unavailable"
                            )
                    latest = self._external(scope)
                    if latest != external or not self._eligible(candidate, latest.record):
                        return RestoreResult(
                            RestoreDecision.DENY, "external changed during reconciliation"
                        )
                    with SQLiteStateStore(self._live_path) as live:
                        if not self._exact(live.read_verified_snapshot(), candidate):
                            return RestoreResult(
                                RestoreDecision.DENY, "local changed during reconciliation"
                            )
                    decision = (
                        RestoreDecision.RESTORE_EXTERNAL_COMMITTED_CURRENT
                        if installed
                        else RestoreDecision.NOOP_ALREADY_CURRENT
                    )
                    return RestoreResult(decision, classification.value)

                latest = self._external(scope)
                if latest != external or latest.record.lifecycle != "PREPARED":
                    return RestoreResult(RestoreDecision.DENY, "prepared authority changed")
                # Re-run the full production validator at the protected-action
                # fence, including the ACK-loss reconciliation precondition.
                candidate = validate_backup_envelope(candidate)
                with SQLiteStateStore(self._live_path) as live:
                    coordinator = ProtectedFreshnessHandoffCoordinator(
                        live, self._registry, self._authority
                    )
                    coordinator.recover_protected_state(scope)
                    if not self._exact(live.read_verified_snapshot(), candidate):
                        return RestoreResult(RestoreDecision.DENY, "local changed after FINALIZE")
                return RestoreResult(
                    RestoreDecision.RESTORE_EXACT_PROTECTED_PENDING_AND_FINALIZE,
                    classification.value,
                )
            except (BackupEnvelopeError, OSError, StateStoreError, ProtectedFreshnessHandoffError):
                return RestoreResult(RestoreDecision.DENY, "fail-closed restore denial")
