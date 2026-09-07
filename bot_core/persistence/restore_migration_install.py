"""Internal dual-gate installer for completed legacy restore migrations."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sqlite3

from .backup_envelope import BackupEnvelope, validate_backup_envelope
from .local_durable_evidence import LocalDurableEvidenceRegistry
from .migration_protocol import MigrationRegistry, PRODUCTION_MIGRATION_REGISTRY
from .protected_freshness_handoff import (
    ProtectedFreshnessAuthorityPort,
    ProtectedFreshnessAuthorityRecord,
    ProtectedFreshnessHandoffCoordinator,
    ProtectedFreshnessHandoffError,
)
from .restore_migration_resume import (
    RestoreMigrationResumeCoordinator,
    RestoreMigrationResumeError,
)
from .restore_migration_staging import (
    RestoreMigrationStagingManifest,
    _read_manifest,
    restore_migration_manifest_path,
    restore_migration_staged_sqlite_path,
    restore_migration_staging_id,
    restore_migration_unpublished_sqlite_path,
)
from .restore_protocol import (
    RestoreLifecycleAuthorityBundle,
    SealedMigrationRestoreAuthority,
    SecretHandoffRestoreFence,
    TrustedPhysicalRestoreCoordinator,
)
from .state_store import SQLiteStateStore, StateStoreError, StateStoreSnapshot
from .state_store_v2_migration import MIGRATION_ID


class RestoreMigrationInstallError(RuntimeError):
    """A completed staging database cannot safely be installed or recovered."""


class RestoreMigrationInstallDisposition(str, Enum):
    INSTALLED = "INSTALLED"
    ALREADY_INSTALLED = "ALREADY_INSTALLED"


@dataclass(frozen=True, slots=True)
class RestoreMigrationInstallResult:
    installed_snapshot: StateStoreSnapshot
    live_path: Path
    disposition: RestoreMigrationInstallDisposition


@dataclass(frozen=True, slots=True)
class _RuntimeLineageBinding:
    source_generation: int
    migration_id: str = MIGRATION_ID


@dataclass(frozen=True, slots=True)
class _ExternalObservation:
    ref: object
    record: ProtectedFreshnessAuthorityRecord


class RestoreMigrationInstallCoordinator:
    """Promote P2A output while retaining both SQLite path gates."""

    def __init__(
        self,
        live_state_store_path: str | Path,
        local_evidence_registry: LocalDurableEvidenceRegistry,
        external_authority: ProtectedFreshnessAuthorityPort,
        registry: MigrationRegistry = PRODUCTION_MIGRATION_REGISTRY,
        lifecycle_authority: RestoreLifecycleAuthorityBundle | None = None,
    ) -> None:
        self._live_path = Path(live_state_store_path).resolve()
        self._evidence = local_evidence_registry
        self._external = external_authority
        self._registry = registry
        self._lifecycles = (
            lifecycle_authority or RestoreLifecycleAuthorityBundle.from_migration_registry(registry)
        )
        # Construction must fail before filesystem or protected-state mutation.
        RestoreMigrationResumeCoordinator(
            self._live_path, self._evidence, self._external, self._registry
        )

    @staticmethod
    def _scope(source: BackupEnvelope) -> tuple[str, str, str]:
        return (
            source.account_id,
            source.device_installation_id,
            source.state_store_identity_fingerprint_sha256,
        )

    def _resolve_external_exact(
        self, source: BackupEnvelope, snapshot: StateStoreSnapshot
    ) -> _ExternalObservation:
        scope = self._scope(source)
        try:
            resolved = self._external.resolve_current(scope)
            if resolved is None:
                raise RestoreMigrationInstallError("RESTORE_MIGRATION_EXTERNAL_MISMATCH")
            record = ProtectedFreshnessAuthorityRecord.from_mapping(resolved[1])
        except ProtectedFreshnessHandoffError as exc:
            raise RestoreMigrationInstallError("RESTORE_MIGRATION_EXTERNAL_MISMATCH") from exc
        metadata = snapshot.metadata
        if (
            record.scope != scope
            or record.lifecycle != "COMMITTED"
            or record.committed_generation != metadata.protected_freshness_generation
            or record.committed_state_fingerprint_sha256 != metadata.state_fingerprint_sha256
        ):
            raise RestoreMigrationInstallError("RESTORE_MIGRATION_EXTERNAL_MISMATCH")
        return _ExternalObservation(resolved[0], record)

    def _recover_external_if_prepared(self, source: BackupEnvelope) -> None:
        scope = self._scope(source)
        try:
            resolved = self._external.resolve_current(scope)
            if resolved is None:
                return
            record = ProtectedFreshnessAuthorityRecord.from_mapping(resolved[1])
            if record.lifecycle == "PREPARED":
                with SQLiteStateStore(self._live_path) as live:
                    ProtectedFreshnessHandoffCoordinator(
                        live, self._evidence, self._external
                    ).recover_protected_state(scope)
        except ProtectedFreshnessHandoffError as exc:
            raise RestoreMigrationInstallError("RESTORE_MIGRATION_EXTERNAL_MISMATCH") from exc

    @staticmethod
    def _assert_manifest_source(
        source: BackupEnvelope,
        manifest: RestoreMigrationStagingManifest,
        staging_id: str,
    ) -> None:
        if (
            manifest.staging_id != staging_id
            or manifest.account_id != source.account_id
            or manifest.device_installation_id != source.device_installation_id
            or manifest.environment != source.environment
            or manifest.state_store_identity_fingerprint_sha256
            != source.state_store_identity_fingerprint_sha256
            or manifest.source_backup_envelope_fingerprint_sha256
            != source.envelope_fingerprint_sha256
            or manifest.source_generation != source.local_protected_freshness_generation
            or manifest.source_state_fingerprint_sha256 != source.state_fingerprint_sha256
            or manifest.source_transaction_fingerprint_sha256
            != source.transaction_fingerprint_sha256
            or manifest.source_history_tail_fingerprint_sha256
            != source.history_tail_fingerprint_sha256
        ):
            raise RestoreMigrationInstallError("STAGING_CONFLICT: manifest does not match source")

    def _observe_secrets(
        self, snapshot: StateStoreSnapshot, fence: SecretHandoffRestoreFence
    ) -> None:
        lifecycles = TrustedPhysicalRestoreCoordinator._secret_lifecycles(snapshot)
        if not lifecycles:
            return
        authority = self._lifecycles.secret_handoff_restore_authority
        if authority is None:
            raise RestoreMigrationInstallError("SECRET_HANDOFF_RESTORE_AUTHORITY_UNAVAILABLE")
        # Reuse the frozen physical-restore observer, including its closed state map.
        observer = object.__new__(TrustedPhysicalRestoreCoordinator)
        observer._lifecycle_authority = self._lifecycles
        observer._observe(lifecycles, fence)

    def _prove_target(
        self,
        source: BackupEnvelope,
        snapshot: StateStoreSnapshot,
        schema_fingerprint: str,
        manifest: RestoreMigrationStagingManifest | None,
    ) -> None:
        if snapshot.metadata.state_store_schema_version != 2:
            raise RestoreMigrationInstallError("RESTORE_MIGRATION_TARGET_MISMATCH")
        if manifest is not None and manifest.migration_id != MIGRATION_ID:
            raise RestoreMigrationInstallError("RESTORE_MIGRATION_TARGET_MISMATCH")
        # This is P2A's single canonical lineage classifier, deliberately reused.
        if manifest is None:
            # Only the fields consumed by the read-only lineage proof are needed
            # for manifest-free idempotence; no durable carrier is synthesized.
            lineage_binding = _RuntimeLineageBinding(source.local_protected_freshness_generation)
        else:
            lineage_binding = manifest
        try:
            roles = RestoreMigrationResumeCoordinator._assert_lineage(
                source, lineage_binding, snapshot
            )
        except RestoreMigrationResumeError as exc:
            raise RestoreMigrationInstallError("RESTORE_MIGRATION_TARGET_MISMATCH") from exc
        if (
            roles
            != (
                "PREPARED",
                "APPLYING",
                "STRUCTURAL_MIGRATION",
                "DURABLE_MIGRATED",
                "COMPLETED",
            )
            or len(RestoreMigrationResumeCoordinator._exact_migration_declarations(snapshot)) != 1
        ):
            raise RestoreMigrationInstallError("RESTORE_MIGRATION_TARGET_MISMATCH")
        try:
            SealedMigrationRestoreAuthority(self._registry).revalidate(snapshot, schema_fingerprint)
        except Exception as exc:
            raise RestoreMigrationInstallError("RESTORE_MIGRATION_TARGET_MISMATCH") from exc

    def _read_and_prove(
        self,
        path: Path,
        source: BackupEnvelope,
        manifest: RestoreMigrationStagingManifest | None,
    ) -> StateStoreSnapshot:
        try:
            with SQLiteStateStore(path) as store:
                snapshot = store.read_verified_snapshot()
                if snapshot is None:
                    raise RestoreMigrationInstallError("RESTORE_MIGRATION_TARGET_MISMATCH")
                schema = store.sqlite_schema_fingerprint()
        except RestoreMigrationInstallError:
            raise
        except Exception as exc:
            raise RestoreMigrationInstallError("RESTORE_MIGRATION_TARGET_MISMATCH") from exc
        self._prove_target(source, snapshot, schema, manifest)
        return snapshot

    def _read_and_prove_isolated(
        self,
        path: Path,
        source: BackupEnvelope,
        manifest: RestoreMigrationStagingManifest | None,
    ) -> StateStoreSnapshot:
        """Prove one closed main without opening a writable/registered handle."""

        try:
            snapshot, schema = SQLiteStateStore.read_isolated_verified_snapshot_and_schema(path)
            if snapshot is None:
                raise RestoreMigrationInstallError("RESTORE_MIGRATION_TARGET_MISMATCH")
            self._prove_target(source, snapshot, schema, manifest)
        except RestoreMigrationInstallError:
            raise
        except Exception as exc:
            raise RestoreMigrationInstallError("RESTORE_MIGRATION_TARGET_MISMATCH") from exc
        return snapshot

    @staticmethod
    @contextmanager
    def _path_gate(path: Path, error: str):
        try:
            with SQLiteStateStore.installation_gate(path):
                yield
        except StateStoreError as exc:
            if "pre-existing process-local handles" in str(exc):
                raise RestoreMigrationInstallError(error) from exc
            raise

    def _classify_live(self, target: StateStoreSnapshot) -> tuple[str, StateStoreSnapshot | None]:
        if not self._live_path.exists():
            return "ABSENT", None
        try:
            snapshot = SQLiteStateStore.read_isolated_verified_snapshot(self._live_path)
        except (OSError, sqlite3.Error, StateStoreError):
            return "UNREADABLE", None
        if snapshot is None:
            return "EMPTY", None
        tm, lm = target.metadata, snapshot.metadata
        if (
            lm.account_id,
            lm.device_installation_id,
            lm.state_store_identity_fingerprint_sha256,
        ) != (tm.account_id, tm.device_installation_id, tm.state_store_identity_fingerprint_sha256):
            return "SCOPE_CONFLICT", snapshot
        if lm.environment != tm.environment:
            return "ENVIRONMENT_CONFLICT", snapshot
        if lm.protected_freshness_generation < tm.protected_freshness_generation:
            return "BEHIND", snapshot
        if lm.protected_freshness_generation > tm.protected_freshness_generation:
            return "AHEAD", snapshot
        if lm.state_fingerprint_sha256 != tm.state_fingerprint_sha256:
            return "SAME_GENERATION_DIFFERENT_STATE", snapshot
        if snapshot != target:
            return "SAME_GENERATION_STATE_MATCH_LINEAGE_MISMATCH", snapshot
        return "EXACT", snapshot

    @staticmethod
    def _cleanup(manifest_path: Path, staged_path: Path, unpublished_path: Path) -> None:
        if unpublished_path.exists():
            raise RestoreMigrationInstallError("STAGING_CONFLICT: unpublished SQLite exists")
        for path in (
            Path(f"{staged_path}-wal"),
            Path(f"{staged_path}-shm"),
            Path(f"{unpublished_path}-wal"),
            Path(f"{unpublished_path}-shm"),
        ):
            path.unlink(missing_ok=True)
        # The manifest is the recovery marker and therefore the final unlink.
        manifest_path.unlink(missing_ok=True)

    def _publish_evidence(self) -> None:
        self._evidence.invalidate_scope(self._scope_from_live())
        with SQLiteStateStore(self._live_path) as live:
            if self._evidence.publish_verified_state(live) is None:
                raise RestoreMigrationInstallError("fresh durable evidence unavailable")

    def _scope_from_live(self) -> tuple[str, str, str]:
        snapshot = SQLiteStateStore.read_isolated_verified_snapshot(self._live_path)
        if snapshot is None:
            raise RestoreMigrationInstallError("RESTORE_MIGRATION_POST_INSTALL_MISMATCH")
        metadata = snapshot.metadata
        return (
            metadata.account_id,
            metadata.device_installation_id,
            metadata.state_store_identity_fingerprint_sha256,
        )

    def _finish_cleanup_expect_staged_absent(
        self,
        manifest_path: Path,
        staged_path: Path,
        unpublished_path: Path,
    ) -> None:
        """Clean only when atomic promotion left the staged pathname absent."""

        with self._path_gate(staged_path, "STAGING_PATH_NOT_QUIESCENT: CLEANUP_PENDING"):
            if staged_path.exists():
                raise RestoreMigrationInstallError(
                    "STAGING_CONFLICT: CLEANUP_PENDING: staged path reappeared"
                )
            self._cleanup(manifest_path, staged_path, unpublished_path)

    def _finish_cleanup_exact_staged(
        self,
        source: BackupEnvelope,
        expected_snapshot: StateStoreSnapshot,
        expected_manifest: RestoreMigrationStagingManifest,
        manifest_path: Path,
        staged_path: Path,
        unpublished_path: Path,
    ) -> None:
        """Re-prove the current pathname before deleting an exact no-op target."""

        with self._path_gate(staged_path, "STAGING_PATH_NOT_QUIESCENT: CLEANUP_PENDING"):
            try:
                if not manifest_path.exists() or _read_manifest(manifest_path) != expected_manifest:
                    raise RestoreMigrationInstallError("staging manifest changed before cleanup")
                if unpublished_path.exists():
                    raise RestoreMigrationInstallError("unpublished SQLite appeared before cleanup")
                if staged_path.exists():
                    cleanup_snapshot = self._read_and_prove_isolated(
                        staged_path, source, expected_manifest
                    )
                    if cleanup_snapshot != expected_snapshot:
                        raise RestoreMigrationInstallError("staged target changed before cleanup")
                    staged_path.unlink()
            except Exception as exc:
                raise RestoreMigrationInstallError("STAGING_CONFLICT: CLEANUP_PENDING") from exc
            self._cleanup(manifest_path, staged_path, unpublished_path)

    def _already_installed_result(
        self, installed: StateStoreSnapshot
    ) -> RestoreMigrationInstallResult:
        self._publish_evidence()
        return RestoreMigrationInstallResult(
            installed,
            self._live_path,
            RestoreMigrationInstallDisposition.ALREADY_INSTALLED,
        )

    def install(self, source_backup: BackupEnvelope) -> RestoreMigrationInstallResult:
        source = validate_backup_envelope(source_backup)
        if source.state_store_schema_version != 1:
            raise RestoreMigrationInstallError("legacy migration install is not applicable")
        staging_id = restore_migration_staging_id(
            state_store_identity_fingerprint_sha256=source.state_store_identity_fingerprint_sha256,
            source_backup_envelope_fingerprint_sha256=source.envelope_fingerprint_sha256,
        )
        staged_path = restore_migration_staged_sqlite_path(self._live_path, staging_id)
        manifest_path = restore_migration_manifest_path(self._live_path, staging_id)
        unpublished_path = restore_migration_unpublished_sqlite_path(self._live_path, staging_id)

        manifest = _read_manifest(manifest_path) if manifest_path.exists() else None
        if manifest is not None:
            self._assert_manifest_source(source, manifest, staging_id)
        if manifest is not None and unpublished_path.exists():
            raise RestoreMigrationInstallError("STAGING_CONFLICT: unpublished SQLite exists")
        # C7 and post-cleanup idempotence are classified before P1/P2A.
        early_installed: StateStoreSnapshot | None = None
        if not staged_path.exists() and self._live_path.exists():
            with self._path_gate(self._live_path, "LIVE_PATH_NOT_QUIESCENT"):
                try:
                    # Pure immutable proof must precede any PREPARED recovery:
                    # recovery may perform an externally visible ABORT/FINALIZE.
                    installed = self._read_and_prove_isolated(self._live_path, source, manifest)
                except RestoreMigrationInstallError as exc:
                    if manifest is not None:
                        raise RestoreMigrationInstallError(
                            "RESTORE_MIGRATION_STAGING_LOST: MANUAL_RECOVERY_REQUIRED"
                        ) from exc
                else:
                    self._recover_external_if_prepared(source)
                    installed = self._read_and_prove_isolated(self._live_path, source, manifest)
                    before_secret = self._resolve_external_exact(source, installed)
                    self._observe_secrets(installed, SecretHandoffRestoreFence.FINAL_PROMOTION)
                    after_secret = self._resolve_external_exact(source, installed)
                    if after_secret != before_secret:
                        raise RestoreMigrationInstallError("RESTORE_MIGRATION_EXTERNAL_CHANGED")
                    early_installed = installed
            if early_installed is not None:
                self._finish_cleanup_expect_staged_absent(
                    manifest_path, staged_path, unpublished_path
                )
                return self._already_installed_result(early_installed)
        if manifest is not None and not staged_path.exists():
            raise RestoreMigrationInstallError(
                "RESTORE_MIGRATION_STAGING_LOST: MANUAL_RECOVERY_REQUIRED"
            )

        completed = RestoreMigrationResumeCoordinator(
            self._live_path, self._evidence, self._external, self._registry
        ).resume(source)
        if completed.sqlite_path.resolve() != staged_path or completed.manifest != _read_manifest(
            manifest_path
        ):
            raise RestoreMigrationInstallError("RESTORE_MIGRATION_TARGET_MISMATCH")
        if unpublished_path.exists():
            raise RestoreMigrationInstallError("STAGING_CONFLICT: unpublished SQLite exists")

        # F1: fresh mutable read, semantic/physical proof, and PRE_INSTALL fence.
        staged = SQLiteStateStore(staged_path)
        try:
            snapshot = staged.read_verified_snapshot()
            if snapshot is None:
                raise RestoreMigrationInstallError("RESTORE_MIGRATION_TARGET_MISMATCH")
            schema = staged.sqlite_schema_fingerprint()
            self._prove_target(source, snapshot, schema, completed.manifest)
            expected_external = self._resolve_external_exact(source, snapshot)
            self._observe_secrets(snapshot, SecretHandoffRestoreFence.PRE_INSTALL)
            # F2 closes this exact invoking handle.
            staged.prepare_for_atomic_install()
        finally:
            staged.close()

        disposition = RestoreMigrationInstallDisposition.INSTALLED
        # F2B/F3/F4: gate order is staging then live, held through replace.
        with self._path_gate(staged_path, "STAGING_PATH_NOT_QUIESCENT"):
            closed, closed_schema = SQLiteStateStore.read_isolated_verified_snapshot_and_schema(
                staged_path
            )
            if closed is None:
                raise RestoreMigrationInstallError("RESTORE_MIGRATION_CLOSED_MAIN_MISMATCH")
            if closed != snapshot:
                raise RestoreMigrationInstallError("RESTORE_MIGRATION_CLOSED_MAIN_MISMATCH")
            self._prove_target(source, closed, closed_schema, completed.manifest)
            with self._path_gate(self._live_path, "LIVE_PATH_NOT_QUIESCENT"):
                classification, live_snapshot = self._classify_live(snapshot)
                if classification == "EXACT":
                    disposition = RestoreMigrationInstallDisposition.ALREADY_INSTALLED
                elif classification in {"ABSENT", "UNREADABLE", "EMPTY", "BEHIND"}:
                    if classification in {"EMPTY", "BEHIND"}:
                        live = SQLiteStateStore(self._live_path)
                        try:
                            if live.read_verified_snapshot() != live_snapshot:
                                raise RestoreMigrationInstallError("live changed before checkpoint")
                            live.prepare_for_atomic_install()
                        finally:
                            live.close()
                else:
                    raise RestoreMigrationInstallError(
                        f"RESTORE_MIGRATION_LIVE_CONFLICT: {classification}"
                    )
                # This is the true final protected fence: all potentially
                # blocking local live preparation is complete and both path
                # gates remain held.
                final_external = self._resolve_external_exact(source, snapshot)
                if final_external != expected_external:
                    raise RestoreMigrationInstallError("RESTORE_MIGRATION_EXTERNAL_CHANGED")
                self._observe_secrets(snapshot, SecretHandoffRestoreFence.FINAL_PROMOTION)
                if classification != "EXACT":
                    SQLiteStateStore.atomic_replace(staged_path, self._live_path)

        try:
            installed = self._read_and_prove(self._live_path, source, completed.manifest)
            if installed != snapshot:
                raise RestoreMigrationInstallError("RESTORE_MIGRATION_POST_INSTALL_MISMATCH")
        except Exception as exc:
            raise RestoreMigrationInstallError(
                "RESTORE_MIGRATION_POST_INSTALL_MISMATCH: MANUAL_RECOVERY_REQUIRED"
            ) from exc
        post_external = self._resolve_external_exact(source, installed)
        if post_external != final_external:
            raise RestoreMigrationInstallError("RESTORE_MIGRATION_EXTERNAL_CHANGED")
        if disposition is RestoreMigrationInstallDisposition.ALREADY_INSTALLED:
            self._finish_cleanup_exact_staged(
                source,
                installed,
                completed.manifest,
                manifest_path,
                staged_path,
                unpublished_path,
            )
            return self._already_installed_result(installed)
        self._finish_cleanup_expect_staged_absent(manifest_path, staged_path, unpublished_path)
        self._publish_evidence()
        return RestoreMigrationInstallResult(installed, self._live_path, disposition)


__all__ = [
    "RestoreMigrationInstallCoordinator",
    "RestoreMigrationInstallDisposition",
    "RestoreMigrationInstallError",
    "RestoreMigrationInstallResult",
]
