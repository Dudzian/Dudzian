"""Durable source-seeded workspace for authenticated legacy restores.

This module intentionally stops before the migration lifecycle.  Its manifest is
an immutable integrity/recovery carrier, never freshness or install authority.
"""

from __future__ import annotations

import json
import re
import shutil
from threading import RLock
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, cast

from .backup_envelope import BackupEnvelope, validate_backup_envelope
from .fingerprints import canonical_json, canonical_json_sha256
from .migration_protocol import MIGRATION_FAMILY_REPRESENTATIONS
from .physical_durability import (
    atomic_write_bytes_durably,
    flush_created_directory_metadata,
    fsync_file,
    reinforce_published_file_durability,
)
from .physical_backup import PhysicalSQLiteArtifact, physical_sqlite_artifact_fingerprint
from .state_store import SQLiteStateStore, StateStoreMetadata, StateStoreSnapshot
from .state_store_v2_migration import (
    MIGRATION_ID,
    STATE_STORE_V2_MIGRATION_AUTHORITY,
    STATE_STORE_V2_MIGRATION_DEFINITION,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_NAME = "manifest.json"
_SQLITE_NAME = "state_store.sqlite3"
_PREPUBLICATION_LOCK = RLock()


class RestoreMigrationStagingError(RuntimeError):
    """The requested staging workspace is inapplicable, conflicting, or invalid."""


class RestoreMigrationStagingDisposition(str, Enum):
    SOURCE_READY = "SOURCE_READY"
    EXISTING_STAGING_REQUIRES_PHASE_RESUME = "EXISTING_STAGING_REQUIRES_PHASE_RESUME"


@dataclass(frozen=True, slots=True)
class RestoreMigrationStagingManifest:
    staging_manifest_schema_version: int
    staging_id: str
    account_id: str
    device_installation_id: str
    environment: str
    state_store_identity_fingerprint_sha256: str
    source_backup_envelope_fingerprint_sha256: str
    source_state_store_schema_version: int
    source_generation: int
    source_state_fingerprint_sha256: str
    source_transaction_fingerprint_sha256: str
    source_history_tail_fingerprint_sha256: str
    source_staged_sqlite_artifact_fingerprint_sha256: str
    migration_id: str
    migration_definition_fingerprint_sha256: str
    operation_plan_fingerprint_sha256: str
    target_state_store_schema_version: int
    manifest_fingerprint_sha256: str

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)

    def fingerprint_mapping(self) -> dict[str, object]:
        value = self.to_mapping()
        del value["manifest_fingerprint_sha256"]
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> RestoreMigrationStagingManifest:
        expected = tuple(field.name for field in fields(cls))
        if not isinstance(value, Mapping) or set(value) != set(expected):
            raise RestoreMigrationStagingError("staging manifest requires exact fields")
        try:
            candidate = cls(**{name: value[name] for name in expected})  # type: ignore[arg-type]
            candidate.validate()
            return candidate
        except (TypeError, ValueError) as exc:
            raise RestoreMigrationStagingError("invalid staging manifest") from exc

    def validate(self) -> None:
        if self.staging_manifest_schema_version != 1 or isinstance(
            self.staging_manifest_schema_version, bool
        ):
            raise RestoreMigrationStagingError("unsupported staging manifest version")
        if self.source_state_store_schema_version != 1 or isinstance(
            self.source_state_store_schema_version, bool
        ):
            raise RestoreMigrationStagingError("staging source must be schema version 1")
        if self.target_state_store_schema_version != 2 or isinstance(
            self.target_state_store_schema_version, bool
        ):
            raise RestoreMigrationStagingError("staging target must be schema version 2")
        if (
            isinstance(self.source_generation, bool)
            or not isinstance(self.source_generation, int)
            or self.source_generation < 1
        ):
            raise RestoreMigrationStagingError("source generation must be positive")
        StateStoreMetadata(
            self.account_id,
            self.device_installation_id,
            1,
            self.state_store_identity_fingerprint_sha256,
            self.environment,
            self.source_generation,
            self.source_state_fingerprint_sha256,
            self.source_transaction_fingerprint_sha256,
            self.source_history_tail_fingerprint_sha256,
        )
        for name in (
            "staging_id",
            "source_backup_envelope_fingerprint_sha256",
            "source_staged_sqlite_artifact_fingerprint_sha256",
            "migration_definition_fingerprint_sha256",
            "operation_plan_fingerprint_sha256",
            "manifest_fingerprint_sha256",
        ):
            if (
                not isinstance(getattr(self, name), str)
                or _SHA256.fullmatch(getattr(self, name)) is None
            ):
                raise RestoreMigrationStagingError(f"{name} must be lowercase SHA-256")
        authority = STATE_STORE_V2_MIGRATION_AUTHORITY
        authority.assert_definition(STATE_STORE_V2_MIGRATION_DEFINITION)
        if (
            self.migration_id,
            self.migration_definition_fingerprint_sha256,
            self.operation_plan_fingerprint_sha256,
        ) != (
            authority.migration_id,
            authority.migration_definition_fingerprint_sha256,
            authority.operation_plan_fingerprint_sha256,
        ):
            raise RestoreMigrationStagingError("staging migration binding is not sealed")
        if self.staging_id != restore_migration_staging_id(
            state_store_identity_fingerprint_sha256=self.state_store_identity_fingerprint_sha256,
            source_backup_envelope_fingerprint_sha256=self.source_backup_envelope_fingerprint_sha256,
        ):
            raise RestoreMigrationStagingError("staging ID mismatch")
        if self.manifest_fingerprint_sha256 != restore_migration_staging_manifest_fingerprint(self):
            raise RestoreMigrationStagingError("staging manifest fingerprint mismatch")


@dataclass(frozen=True, slots=True)
class RestoreMigrationStagingArtifact:
    directory: Path
    sqlite_path: Path
    manifest_path: Path
    manifest: RestoreMigrationStagingManifest
    snapshot: StateStoreSnapshot
    disposition: RestoreMigrationStagingDisposition


def restore_migration_staging_manifest_fingerprint(
    manifest: RestoreMigrationStagingManifest,
) -> str:
    return cast(str, canonical_json_sha256(manifest.fingerprint_mapping()))


def restore_migration_staging_id(
    *,
    state_store_identity_fingerprint_sha256: str,
    source_backup_envelope_fingerprint_sha256: str,
) -> str:
    return cast(
        str,
        canonical_json_sha256(
            {
                "state_store_identity_fingerprint_sha256": state_store_identity_fingerprint_sha256,
                "source_backup_envelope_fingerprint_sha256": source_backup_envelope_fingerprint_sha256,
                "migration_id": MIGRATION_ID,
                "target_state_store_schema_version": 2,
            }
        ),
    )


def restore_migration_staging_path(live_state_store_path: str | Path, staging_id: str) -> Path:
    live = Path(live_state_store_path).resolve()
    return live.parent / ".cryptohunter-restore-staging" / staging_id


def _source_snapshot(envelope: BackupEnvelope) -> StateStoreSnapshot:
    return StateStoreSnapshot(
        StateStoreMetadata(
            envelope.account_id,
            envelope.device_installation_id,
            envelope.state_store_schema_version,
            envelope.state_store_identity_fingerprint_sha256,
            envelope.environment,
            envelope.local_protected_freshness_generation,
            envelope.state_fingerprint_sha256,
            envelope.transaction_fingerprint_sha256,
            envelope.history_tail_fingerprint_sha256,
        ),
        envelope.canonical_durable_records,
        envelope.immutable_recovery_history,
        envelope.integrity_metadata.state_store_transaction_descriptors,
    )


def _assert_source_anchor(
    snapshot: StateStoreSnapshot, manifest: RestoreMigrationStagingManifest
) -> None:
    anchors = tuple(
        descriptor
        for descriptor in snapshot.transaction_descriptors
        if (
            descriptor.target_generation == manifest.source_generation
            and descriptor.post_state_fingerprint_sha256 == manifest.source_state_fingerprint_sha256
            and descriptor.transaction_fingerprint_sha256
            == manifest.source_transaction_fingerprint_sha256
            and descriptor.post_history_tail_fingerprint_sha256
            == manifest.source_history_tail_fingerprint_sha256
            and descriptor.account_id == manifest.account_id
            and descriptor.device_installation_id == manifest.device_installation_id
            and descriptor.environment == manifest.environment
            and descriptor.state_store_identity_fingerprint_sha256
            == manifest.state_store_identity_fingerprint_sha256
        )
    )
    if len(anchors) != 1:
        raise RestoreMigrationStagingError("STAGING_CONFLICT: source descriptor anchor mismatch")


def _read_manifest(path: Path) -> RestoreMigrationStagingManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RestoreMigrationStagingError("STAGING_CONFLICT: unreadable manifest") from exc
    if not isinstance(raw, dict):
        raise RestoreMigrationStagingError("STAGING_CONFLICT: invalid manifest")
    return RestoreMigrationStagingManifest.from_mapping(raw)


def _verify_existing(
    directory: Path,
    expected: RestoreMigrationStagingManifest,
    source_snapshot: StateStoreSnapshot,
) -> RestoreMigrationStagingArtifact:
    manifest_path, sqlite_path = directory / _MANIFEST_NAME, directory / _SQLITE_NAME
    manifest = _read_manifest(manifest_path)
    if manifest != expected:
        raise RestoreMigrationStagingError("STAGING_CONFLICT: manifest does not match source")
    try:
        with SQLiteStateStore(sqlite_path) as reopened:
            snapshot = reopened.read_verified_snapshot()
    except Exception as exc:
        raise RestoreMigrationStagingError("STAGING_CONFLICT: staged SQLite is invalid") from exc
    if snapshot is None:
        raise RestoreMigrationStagingError("STAGING_CONFLICT: staged SQLite is empty")
    _assert_source_anchor(snapshot, manifest)
    has_migration = any(
        record.representation_name in MIGRATION_FAMILY_REPRESENTATIONS
        for record in (*snapshot.current_records, *snapshot.immutable_history)
    )
    if snapshot.metadata.state_store_schema_version != 1 or has_migration:
        reinforce_published_file_durability(
            manifest_path, canonical_json(manifest.to_mapping()).encode("utf-8")
        )
        return RestoreMigrationStagingArtifact(
            directory,
            sqlite_path,
            manifest_path,
            manifest,
            snapshot,
            RestoreMigrationStagingDisposition.EXISTING_STAGING_REQUIRES_PHASE_RESUME,
        )
    _source_snapshot_from_manifest_source(expected, snapshot)
    if snapshot != source_snapshot:
        raise RestoreMigrationStagingError("STAGING_CONFLICT: staged source semantics mismatch")
    if physical_sqlite_artifact_fingerprint(PhysicalSQLiteArtifact(sqlite_path)) != (
        manifest.source_staged_sqlite_artifact_fingerprint_sha256
    ):
        raise RestoreMigrationStagingError("STAGING_CONFLICT: staged source bytes mismatch")
    _reinforce_existing_durability(sqlite_path, manifest_path, manifest)
    return RestoreMigrationStagingArtifact(
        directory,
        sqlite_path,
        manifest_path,
        manifest,
        snapshot,
        RestoreMigrationStagingDisposition.SOURCE_READY,
    )


def _source_snapshot_from_manifest_source(
    manifest: RestoreMigrationStagingManifest, snapshot: StateStoreSnapshot
) -> StateStoreSnapshot:
    # At SOURCE_READY, verified metadata and the exact anchor bind all source
    # semantics; return the snapshot only when those frozen values agree.
    metadata = snapshot.metadata
    if (
        metadata.protected_freshness_generation,
        metadata.state_fingerprint_sha256,
        metadata.transaction_fingerprint_sha256,
        metadata.history_tail_fingerprint_sha256,
    ) != (
        manifest.source_generation,
        manifest.source_state_fingerprint_sha256,
        manifest.source_transaction_fingerprint_sha256,
        manifest.source_history_tail_fingerprint_sha256,
    ):
        raise RestoreMigrationStagingError("STAGING_CONFLICT: staged source metadata mismatch")
    return snapshot


def _write_manifest_durably(path: Path, manifest: RestoreMigrationStagingManifest) -> None:
    atomic_write_bytes_durably(path, canonical_json(manifest.to_mapping()).encode("utf-8"))


def _reinforce_existing_durability(
    sqlite_path: Path, manifest_path: Path, manifest: RestoreMigrationStagingManifest
) -> None:
    fsync_file(sqlite_path)
    reinforce_published_file_durability(
        manifest_path, canonical_json(manifest.to_mapping()).encode("utf-8")
    )


def _published_manifest_path(directory: Path) -> Path | None:
    manifest_path = directory / _MANIFEST_NAME
    return manifest_path if manifest_path.exists() else None


def _remove_unpublished_debris(directory: Path) -> None:
    """Remove only a deterministic workspace with no final publication marker."""

    if _published_manifest_path(directory) is not None:
        raise RestoreMigrationStagingError("STAGING_CONFLICT: manifest publication raced cleanup")
    shutil.rmtree(directory)


def prepare_legacy_restore_staging(
    source: BackupEnvelope | Mapping[str, object], *, live_state_store_path: str | Path
) -> RestoreMigrationStagingArtifact:
    """Create or revalidate the sole deterministic source-ready workspace."""

    envelope = validate_backup_envelope(source)
    if envelope.state_store_schema_version != 1:
        raise RestoreMigrationStagingError("legacy migration staging is not applicable")
    staging_id = restore_migration_staging_id(
        state_store_identity_fingerprint_sha256=envelope.state_store_identity_fingerprint_sha256,
        source_backup_envelope_fingerprint_sha256=envelope.envelope_fingerprint_sha256,
    )
    with _PREPUBLICATION_LOCK:
        directory = restore_migration_staging_path(live_state_store_path, staging_id)
        sqlite_path, manifest_path = directory / _SQLITE_NAME, directory / _MANIFEST_NAME
        if directory.exists():
            if _published_manifest_path(directory) is not None:
                existing = _read_manifest(manifest_path)
                expected = _manifest(
                    envelope,
                    staging_id,
                    existing.source_staged_sqlite_artifact_fingerprint_sha256,
                )
                return _verify_existing(directory, expected, _source_snapshot(envelope))
            _remove_unpublished_debris(directory)

        staging_root = directory.parent
        if not staging_root.exists():
            staging_root.mkdir()
            flush_created_directory_metadata(staging_root.parent)
        try:
            directory.mkdir()
        except FileExistsError:
            # Never delete after an atomic-create race: the winner may be about
            # to publish. Reclassify a published winner, otherwise fail closed.
            if _published_manifest_path(directory) is not None:
                existing = _read_manifest(manifest_path)
                expected = _manifest(
                    envelope,
                    staging_id,
                    existing.source_staged_sqlite_artifact_fingerprint_sha256,
                )
                return _verify_existing(directory, expected, _source_snapshot(envelope))
            raise RestoreMigrationStagingError("STAGING_CONFLICT: concurrent staging creation")
        flush_created_directory_metadata(staging_root)
        try:
            staged = SQLiteStateStore(sqlite_path)
            try:
                staged.write_restored_snapshot(_source_snapshot(envelope))
                verified = staged.read_verified_snapshot()
                if verified is None:
                    raise RestoreMigrationStagingError("source materialization is empty")
                staged.prepare_for_atomic_install()
            finally:
                staged.close()
            fingerprint = physical_sqlite_artifact_fingerprint(PhysicalSQLiteArtifact(sqlite_path))
            fsync_file(sqlite_path)
            manifest = _manifest(envelope, staging_id, fingerprint)
            _assert_source_anchor(verified, manifest)
            _write_manifest_durably(manifest_path, manifest)
            return _verify_existing(directory, manifest, _source_snapshot(envelope))
        finally:
            # A final manifest is the irrevocable publication boundary, even if
            # its trailing durability fence failed. Never silently recreate it.
            if directory.exists() and _published_manifest_path(directory) is None:
                shutil.rmtree(directory, ignore_errors=True)


def _manifest(
    envelope: BackupEnvelope, staging_id: str, physical_fingerprint: str
) -> RestoreMigrationStagingManifest:
    authority = STATE_STORE_V2_MIGRATION_AUTHORITY
    values: dict[str, Any] = {
        "staging_manifest_schema_version": 1,
        "staging_id": staging_id,
        "account_id": envelope.account_id,
        "device_installation_id": envelope.device_installation_id,
        "environment": envelope.environment,
        "state_store_identity_fingerprint_sha256": envelope.state_store_identity_fingerprint_sha256,
        "source_backup_envelope_fingerprint_sha256": envelope.envelope_fingerprint_sha256,
        "source_state_store_schema_version": 1,
        "source_generation": envelope.local_protected_freshness_generation,
        "source_state_fingerprint_sha256": envelope.state_fingerprint_sha256,
        "source_transaction_fingerprint_sha256": envelope.transaction_fingerprint_sha256,
        "source_history_tail_fingerprint_sha256": envelope.history_tail_fingerprint_sha256,
        "source_staged_sqlite_artifact_fingerprint_sha256": physical_fingerprint,
        "migration_id": authority.migration_id,
        "migration_definition_fingerprint_sha256": authority.migration_definition_fingerprint_sha256,
        "operation_plan_fingerprint_sha256": authority.operation_plan_fingerprint_sha256,
        "target_state_store_schema_version": authority.target_schema_version,
    }
    candidate = RestoreMigrationStagingManifest(**values, manifest_fingerprint_sha256="0" * 64)
    return RestoreMigrationStagingManifest(
        **values,
        manifest_fingerprint_sha256=restore_migration_staging_manifest_fingerprint(candidate),
    )
