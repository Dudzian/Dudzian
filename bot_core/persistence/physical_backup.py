"""Trusted physical SQLite backup creation and pre-lifecycle outer admission."""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, cast

from .backup_authentication import (
    BackupArtifactAuthenticationScope,
    BackupArtifactAuthenticator,
    BackupArtifactVerificationResult,
    BackupArtifactVerifier,
    PhysicalArtifactAuthenticationProof,
    PhysicalSQLiteArtifactManifest,
)
from .backup_envelope import (
    BackupEnvelope,
    create_backup_envelope,
    validate_backup_envelope,
)
from .migration_execution import sqlite_schema_fingerprint
from .state_store import SQLiteStateStore, StateStoreSnapshot


class PhysicalBackupError(RuntimeError):
    """Backup production failed; no trusted artifact was published."""


class PhysicalBackupAdmissionReason(str, Enum):
    MALFORMED_BUNDLE = "MALFORMED_BUNDLE"
    MANIFEST_MISMATCH = "MANIFEST_MISMATCH"
    PHYSICAL_HASH_MISMATCH = "PHYSICAL_HASH_MISMATCH"
    ENVELOPE_MISMATCH = "ENVELOPE_MISMATCH"
    AUTHENTICATION_REJECTED = "AUTHENTICATION_REJECTED"
    AUTHORITY_UNAVAILABLE = "AUTHORITY_UNAVAILABLE"
    SQLITE_INTEGRITY_FAILURE = "SQLITE_INTEGRITY_FAILURE"
    SQLITE_SCHEMA_MISMATCH = "SQLITE_SCHEMA_MISMATCH"
    CANDIDATE_STATE_MISMATCH = "CANDIDATE_STATE_MISMATCH"


class PhysicalBackupAdmissionError(RuntimeError):
    def __init__(self, reason: PhysicalBackupAdmissionReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


@dataclass(frozen=True, slots=True)
class PhysicalSQLiteArtifact:
    path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("physical artifact path must be a pathlib.Path")
        object.__setattr__(self, "path", self.path.resolve())


@dataclass(frozen=True, slots=True)
class TrustedPhysicalBackupArtifact:
    """The exact four candidate-carried components; it contains no authority."""

    backup_envelope: BackupEnvelope
    physical_artifact: PhysicalSQLiteArtifact
    manifest: PhysicalSQLiteArtifactManifest
    authentication_proof: PhysicalArtifactAuthenticationProof

    def __post_init__(self) -> None:
        expected = (
            (self.backup_envelope, BackupEnvelope),
            (self.physical_artifact, PhysicalSQLiteArtifact),
            (self.manifest, PhysicalSQLiteArtifactManifest),
            (self.authentication_proof, PhysicalArtifactAuthenticationProof),
        )
        if any(not isinstance(value, kind) for value, kind in expected):
            raise TypeError("trusted physical backup contains an invalid component")


def _hash_file(path: Path, destination: Path | None = None) -> tuple[str, int]:
    digest = hashlib.sha256()
    length = 0
    output = destination.open("xb") if destination is not None else None
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                length += len(chunk)
                if output is not None:
                    output.write(chunk)
        if output is not None:
            output.flush()
            os.fsync(output.fileno())
    finally:
        if output is not None:
            output.close()
    return digest.hexdigest(), length


def _validate_sqlite(path: Path) -> str:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro&immutable=1", uri=True)
    try:
        if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise PhysicalBackupAdmissionError(
                PhysicalBackupAdmissionReason.SQLITE_INTEGRITY_FAILURE
            )
        databases = connection.execute("PRAGMA database_list").fetchall()
        if (
            not databases
            or databases[0][1] != "main"
            or any(row[1] not in {"main", "temp"} for row in databases)
        ):
            raise PhysicalBackupAdmissionError(
                PhysicalBackupAdmissionReason.SQLITE_INTEGRITY_FAILURE
            )
        return cast(str, sqlite_schema_fingerprint(connection))
    except sqlite3.Error as exc:
        raise PhysicalBackupAdmissionError(
            PhysicalBackupAdmissionReason.SQLITE_INTEGRITY_FAILURE
        ) from exc
    finally:
        connection.close()


def _envelope_from_snapshot(snapshot: StateStoreSnapshot) -> BackupEnvelope:
    class _PinnedStore:
        def read_verified_snapshot(self) -> StateStoreSnapshot:
            return snapshot

    envelope = create_backup_envelope(_PinnedStore())  # type: ignore[arg-type]
    if envelope is None:  # pragma: no cover - snapshot is necessarily initialized
        raise PhysicalBackupError("BACKUP_CREATION_FAILED")
    return envelope


class PhysicalBackupCreator:
    def __init__(
        self,
        authenticator: BackupArtifactAuthenticator,
        *,
        _before_publication_hook: Callable[[Path], None] | None = None,
    ) -> None:
        self._authenticator = authenticator
        self._before_publication_hook = _before_publication_hook

    def create(
        self, store: SQLiteStateStore, output_path: str | Path
    ) -> TrustedPhysicalBackupArtifact:
        output = Path(output_path).resolve()
        if output == store.path:
            raise PhysicalBackupError("BACKUP_CREATION_FAILED")
        output.parent.mkdir(parents=True, exist_ok=True)
        work_directory = Path(
            tempfile.mkdtemp(prefix=".physical-backup-", dir=output.parent)
        )
        staged = work_directory / "backup.sqlite"
        try:
            snapshot = store.capture_verified_physical_snapshot(staged)
            envelope = _envelope_from_snapshot(snapshot)
            schema = _validate_sqlite(staged)
            physical_hash, byte_length = _hash_file(staged)
            metadata = snapshot.metadata
            manifest = PhysicalSQLiteArtifactManifest.create(
                artifact_format_version=1,
                account_id=metadata.account_id,
                device_installation_id=metadata.device_installation_id,
                environment=metadata.environment,
                state_store_identity_fingerprint_sha256=metadata.state_store_identity_fingerprint_sha256,
                state_store_schema_version=metadata.state_store_schema_version,
                local_protected_freshness_generation=metadata.protected_freshness_generation,
                state_fingerprint_sha256=metadata.state_fingerprint_sha256,
                transaction_fingerprint_sha256=metadata.transaction_fingerprint_sha256,
                backup_envelope_fingerprint_sha256=envelope.envelope_fingerprint_sha256,
                physical_artifact_sha256=physical_hash,
                physical_artifact_byte_length=byte_length,
                sqlite_schema_fingerprint_sha256=schema,
            )
            scope = BackupArtifactAuthenticationScope(
                metadata.account_id,
                metadata.device_installation_id,
                metadata.environment,
            )
            proof = self._authenticator.authenticate(scope, manifest)
            # Permissions and every other expected fallible operation precede
            # replace(), which is the sole publication commit point.
            os.chmod(staged, 0o600)
            if self._before_publication_hook is not None:
                self._before_publication_hook(staged)
            if _hash_file(staged) != (
                manifest.physical_artifact_sha256,
                manifest.physical_artifact_byte_length,
            ):
                raise PhysicalBackupError("BACKUP_CREATION_FAILED")
            published = TrustedPhysicalBackupArtifact(
                envelope, PhysicalSQLiteArtifact(output), manifest, proof
            )
            os.replace(staged, output)
            return published
        except Exception as exc:
            if isinstance(exc, PhysicalBackupError):
                raise
            raise PhysicalBackupError("BACKUP_CREATION_FAILED") from exc
        finally:
            shutil.rmtree(work_directory, ignore_errors=True)


@dataclass(frozen=True, slots=True)
class AuthenticatedPhysicalBackupCandidate:
    private_path: Path
    backup_envelope: BackupEnvelope
    manifest: PhysicalSQLiteArtifactManifest
    authentication_proof: PhysicalArtifactAuthenticationProof
    physical_artifact_sha256: str
    physical_artifact_byte_length: int
    sqlite_schema_fingerprint_sha256: str
    state_store_snapshot: StateStoreSnapshot
    classification: str = "ELIGIBLE_FOR_FURTHER_RESTORE_VALIDATION_ONLY"
    _directory: Path | None = None

    def close(self) -> None:
        if self._directory is not None:
            shutil.rmtree(self._directory, ignore_errors=True)
            object.__setattr__(self, "_directory", None)

    def verify_physical_continuity(self) -> None:
        """Fail closed if the leased private bytes changed after admission."""
        if self._directory is None or _hash_file(self.private_path) != (
            self.physical_artifact_sha256,
            self.physical_artifact_byte_length,
        ):
            raise PhysicalBackupAdmissionError(
                PhysicalBackupAdmissionReason.PHYSICAL_HASH_MISMATCH
            )

    def __enter__(self) -> AuthenticatedPhysicalBackupCandidate:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class PhysicalBackupAdmissionValidator:
    def __init__(
        self,
        verifier: BackupArtifactVerifier,
        *,
        authenticated_hook: Callable[[Path], None] | None = None,
    ) -> None:
        self._verifier = verifier
        self._authenticated_hook = authenticated_hook

    def admit(
        self, artifact: TrustedPhysicalBackupArtifact
    ) -> AuthenticatedPhysicalBackupCandidate:
        directory = Path(tempfile.mkdtemp(prefix="physical-backup-admission-"))
        os.chmod(directory, 0o700)
        staged = directory / "candidate.sqlite"
        try:
            envelope = validate_backup_envelope(artifact.backup_envelope)
            manifest = PhysicalSQLiteArtifactManifest.from_mapping(
                artifact.manifest.projection()
            )
            proof = PhysicalArtifactAuthenticationProof.from_mapping(
                artifact.authentication_proof.to_mapping()
            )
            physical_hash, byte_length = _hash_file(
                artifact.physical_artifact.path, staged
            )
            if (physical_hash, byte_length) != (
                manifest.physical_artifact_sha256,
                manifest.physical_artifact_byte_length,
            ):
                raise PhysicalBackupAdmissionError(
                    PhysicalBackupAdmissionReason.PHYSICAL_HASH_MISMATCH
                )
            if (
                envelope.envelope_fingerprint_sha256
                != manifest.backup_envelope_fingerprint_sha256
            ):
                raise PhysicalBackupAdmissionError(
                    PhysicalBackupAdmissionReason.ENVELOPE_MISMATCH
                )
            if (
                proof.manifest_fingerprint_sha256
                != manifest.manifest_fingerprint_sha256
            ):
                raise PhysicalBackupAdmissionError(
                    PhysicalBackupAdmissionReason.MANIFEST_MISMATCH
                )
            scope = BackupArtifactAuthenticationScope(
                manifest.account_id,
                manifest.device_installation_id,
                manifest.environment,
            )
            result = self._verifier.verify(scope, manifest, proof)
            if result is not BackupArtifactVerificationResult.VERIFIED:
                reason = (
                    PhysicalBackupAdmissionReason.AUTHORITY_UNAVAILABLE
                    if result
                    in {
                        BackupArtifactVerificationResult.AUTHORITY_UNAVAILABLE,
                        BackupArtifactVerificationResult.AUTHORITY_NOT_PROVISIONED,
                    }
                    else PhysicalBackupAdmissionReason.AUTHENTICATION_REJECTED
                )
                raise PhysicalBackupAdmissionError(reason)
            if self._authenticated_hook is not None:
                self._authenticated_hook(staged)
            # Rehash after the testable post-auth boundary so even private-file
            # corruption cannot swap the authenticated bytes before SQLite open.
            if _hash_file(staged) != (physical_hash, byte_length):
                raise PhysicalBackupAdmissionError(
                    PhysicalBackupAdmissionReason.PHYSICAL_HASH_MISMATCH
                )
            schema = _validate_sqlite(staged)
            if schema != manifest.sqlite_schema_fingerprint_sha256:
                raise PhysicalBackupAdmissionError(
                    PhysicalBackupAdmissionReason.SQLITE_SCHEMA_MISMATCH
                )
            snapshot = SQLiteStateStore.read_isolated_verified_snapshot(staged)
            if snapshot is None or not _candidate_matches(snapshot, envelope, manifest):
                raise PhysicalBackupAdmissionError(
                    PhysicalBackupAdmissionReason.CANDIDATE_STATE_MISMATCH
                )
            return AuthenticatedPhysicalBackupCandidate(
                staged,
                envelope,
                manifest,
                proof,
                physical_hash,
                byte_length,
                schema,
                snapshot,
                _directory=directory,
            )
        except PhysicalBackupAdmissionError:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(directory, ignore_errors=True)
            raise PhysicalBackupAdmissionError(
                PhysicalBackupAdmissionReason.MALFORMED_BUNDLE
            ) from exc


def _candidate_matches(
    snapshot: StateStoreSnapshot,
    envelope: BackupEnvelope,
    manifest: PhysicalSQLiteArtifactManifest,
) -> bool:
    metadata = snapshot.metadata
    facts = (
        metadata.account_id,
        metadata.device_installation_id,
        metadata.environment,
        metadata.state_store_identity_fingerprint_sha256,
        metadata.state_store_schema_version,
        metadata.protected_freshness_generation,
        metadata.state_fingerprint_sha256,
        metadata.transaction_fingerprint_sha256,
    )
    return (
        facts
        == (
            manifest.account_id,
            manifest.device_installation_id,
            manifest.environment,
            manifest.state_store_identity_fingerprint_sha256,
            manifest.state_store_schema_version,
            manifest.local_protected_freshness_generation,
            manifest.state_fingerprint_sha256,
            manifest.transaction_fingerprint_sha256,
        )
        == (
            envelope.account_id,
            envelope.device_installation_id,
            envelope.environment,
            envelope.state_store_identity_fingerprint_sha256,
            envelope.state_store_schema_version,
            envelope.local_protected_freshness_generation,
            envelope.state_fingerprint_sha256,
            envelope.transaction_fingerprint_sha256,
        )
    )
