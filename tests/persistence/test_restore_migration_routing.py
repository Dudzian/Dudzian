from __future__ import annotations

from pathlib import Path
import shutil
import sqlite3
from unittest.mock import Mock

from bot_core.persistence.local_durable_evidence import LocalDurableEvidenceRegistry
from bot_core.persistence.backup_authentication import (
    BackupArtifactAuthenticator,
    BackupArtifactVerifier,
)
from bot_core.persistence.physical_backup import (
    AuthenticatedPhysicalBackupCandidate,
    PhysicalBackupAdmissionValidator,
    PhysicalBackupCreator,
)
from bot_core.persistence.migration_protocol import (
    MigrationRegistry,
    PRODUCTION_MIGRATION_REGISTRY,
)
from bot_core.persistence.restore_migration_install import (
    RestoreMigrationInstallCoordinator,
    RestoreMigrationInstallError,
)
from bot_core.persistence.restore_migration_resume import RestoreMigrationResumeCoordinator
from bot_core.persistence.restore_migration_staging import (
    restore_migration_manifest_path,
    restore_migration_staged_sqlite_path,
    restore_migration_staging_id,
)
from bot_core.persistence.restore_protocol import (
    RestoreDecision,
    RestoreLifecycleAuthorityBundle,
    RestoreResult,
    SealedMigrationRestoreAuthority,
    TrustedPhysicalRestoreCoordinator,
)
from bot_core.persistence.state_store import SQLiteStateStore, StateStoreError
from tests.persistence.physical_backup_helpers import authority_for
from tests.persistence.test_restore_migration_resume import _boundary_for
from tests.persistence.test_restore_migration_staging import _legacy_backup


def _legacy_artifact(tmp_path: Path):  # type: ignore[no-untyped-def]
    source_path = tmp_path / "legacy-source.sqlite3"
    _legacy_backup(source_path)
    source = SQLiteStateStore(source_path)
    authority = authority_for(tmp_path, source)
    artifact = PhysicalBackupCreator(BackupArtifactAuthenticator(authority)).create(
        source, tmp_path / "legacy-backup.sqlite3"
    )
    source.close()
    return artifact, authority


def _coordinator(tmp_path: Path, artifact, authority):  # type: ignore[no-untyped-def]
    evidence = LocalDurableEvidenceRegistry()
    return TrustedPhysicalRestoreCoordinator(
        tmp_path / "live.sqlite3",
        evidence,
        _boundary_for(artifact.backup_envelope),
        PhysicalBackupAdmissionValidator(BackupArtifactVerifier(authority)),
        RestoreLifecycleAuthorityBundle.from_migration_registry(PRODUCTION_MIGRATION_REGISTRY),
    )


def _migration_paths(artifact, live: Path) -> tuple[Path, Path]:  # type: ignore[no-untyped-def]
    envelope = artifact.backup_envelope
    staging_id = restore_migration_staging_id(
        state_store_identity_fingerprint_sha256=(envelope.state_store_identity_fingerprint_sha256),
        source_backup_envelope_fingerprint_sha256=envelope.envelope_fingerprint_sha256,
    )
    return (
        restore_migration_staged_sqlite_path(live, staging_id),
        restore_migration_manifest_path(live, staging_id),
    )


def test_authenticated_v1_routes_to_migration_install_and_retry_is_noop(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    artifact, authority = _legacy_artifact(tmp_path)
    coordinator = _coordinator(tmp_path, artifact, authority)
    original_install = RestoreMigrationInstallCoordinator.install
    calls: list[object] = []

    def counted_install(self, source):  # type: ignore[no-untyped-def]
        calls.append(source)
        return original_install(self, source)

    monkeypatch.setattr(RestoreMigrationInstallCoordinator, "install", counted_install)

    first = coordinator.restore_trusted_artifact(artifact)
    second = coordinator.restore_trusted_artifact(artifact)

    assert first.decision is RestoreDecision.RESTORE_EXTERNAL_COMMITTED_CURRENT
    assert second.decision is RestoreDecision.NOOP_ALREADY_CURRENT
    assert len(calls) == 2
    with SQLiteStateStore(tmp_path / "live.sqlite3") as live:
        snapshot = live.read_verified_snapshot()
    assert snapshot is not None
    assert snapshot.metadata.state_store_schema_version == 2
    assert snapshot.metadata.protected_freshness_generation > (
        artifact.backup_envelope.local_protected_freshness_generation
    )


def test_custom_read_only_registry_cannot_steer_legacy_execution(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    artifact, authentication_authority = _legacy_artifact(tmp_path)
    custom_registry = MigrationRegistry(())
    bundle = RestoreLifecycleAuthorityBundle.from_migration_registry(custom_registry)
    captured: dict[str, object] = {}
    original_init = RestoreMigrationInstallCoordinator.__init__

    def capturing_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        captured["registry"] = kwargs["registry"]
        captured["lifecycle_authority"] = kwargs["lifecycle_authority"]
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(RestoreMigrationInstallCoordinator, "__init__", capturing_init)
    coordinator = TrustedPhysicalRestoreCoordinator(
        tmp_path / "live.sqlite3",
        LocalDurableEvidenceRegistry(),
        _boundary_for(artifact.backup_envelope),
        PhysicalBackupAdmissionValidator(BackupArtifactVerifier(authentication_authority)),
        bundle,
    )

    result = coordinator.restore_trusted_artifact(artifact)

    assert result.decision is RestoreDecision.RESTORE_EXTERNAL_COMMITTED_CURRENT
    assert captured["registry"] is PRODUCTION_MIGRATION_REGISTRY
    assert captured["lifecycle_authority"] is bundle
    with SQLiteStateStore(tmp_path / "live.sqlite3") as live:
        snapshot = live.read_verified_snapshot()
    assert snapshot is not None and snapshot.metadata.state_store_schema_version == 2


def test_sealed_migration_restore_authority_does_not_export_registry() -> None:
    authority = SealedMigrationRestoreAuthority(PRODUCTION_MIGRATION_REGISTRY)

    assert not hasattr(authority, "registry")


def test_authenticated_v2_does_not_enter_migration_installer(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    legacy, legacy_authority = _legacy_artifact(tmp_path / "legacy")
    migrated_path = tmp_path / "migrated.sqlite3"
    migrated = TrustedPhysicalRestoreCoordinator(
        migrated_path,
        LocalDurableEvidenceRegistry(),
        _boundary_for(legacy.backup_envelope),
        PhysicalBackupAdmissionValidator(BackupArtifactVerifier(legacy_authority)),
        RestoreLifecycleAuthorityBundle.from_migration_registry(PRODUCTION_MIGRATION_REGISTRY),
    ).restore_trusted_artifact(legacy)
    assert migrated.decision is RestoreDecision.RESTORE_EXTERNAL_COMMITTED_CURRENT
    source = SQLiteStateStore(migrated_path)
    authority = authority_for(tmp_path / "current", source)
    artifact = PhysicalBackupCreator(BackupArtifactAuthenticator(authority)).create(
        source, tmp_path / "current-backup.sqlite3"
    )
    source.close()
    install = Mock(side_effect=AssertionError("v2 must not migrate"))
    monkeypatch.setattr(RestoreMigrationInstallCoordinator, "install", install)
    coordinator = TrustedPhysicalRestoreCoordinator(
        tmp_path / "v2-live.sqlite3",
        LocalDurableEvidenceRegistry(),
        _boundary_for(artifact.backup_envelope),
        PhysicalBackupAdmissionValidator(BackupArtifactVerifier(authority)),
        RestoreLifecycleAuthorityBundle.from_migration_registry(PRODUCTION_MIGRATION_REGISTRY),
    )

    result = coordinator.restore_trusted_artifact(artifact)

    assert result.decision is RestoreDecision.RESTORE_EXTERNAL_COMMITTED_CURRENT
    install.assert_not_called()


def test_unauthenticated_v1_never_enters_migration_installer(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    artifact, authority = _legacy_artifact(tmp_path)
    artifact.physical_artifact.path.write_bytes(b"tampered")
    install = Mock(side_effect=AssertionError("unauthenticated v1 must not migrate"))
    monkeypatch.setattr(RestoreMigrationInstallCoordinator, "install", install)

    result = _coordinator(tmp_path, artifact, authority).restore_trusted_artifact(artifact)

    assert result.decision is RestoreDecision.DENY
    install.assert_not_called()


def test_unauthenticated_v1_never_classifies_live_before_admission(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    artifact, authority = _legacy_artifact(tmp_path)
    artifact.physical_artifact.path.write_bytes(b"tampered")
    coordinator = _coordinator(tmp_path, artifact, authority)
    classify = Mock(side_effect=AssertionError("legacy must not classify before admission"))
    install = Mock(side_effect=AssertionError("unauthenticated legacy must not migrate"))
    monkeypatch.setattr(coordinator, "_classify", classify)
    monkeypatch.setattr(RestoreMigrationInstallCoordinator, "install", install)

    result = coordinator.restore_trusted_artifact(artifact)

    assert result.decision is RestoreDecision.DENY
    classify.assert_not_called()
    install.assert_not_called()


def test_unauthenticated_v1_does_not_touch_foreign_live_sqlite(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    artifact, authority = _legacy_artifact(tmp_path)
    artifact.physical_artifact.path.write_bytes(b"tampered")
    live = tmp_path / "live.sqlite3"
    with sqlite3.connect(live) as connection:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("CREATE TABLE marker(value TEXT)")
        connection.execute("INSERT INTO marker VALUES ('unchanged')")
    before_bytes = live.read_bytes()
    with sqlite3.connect(f"file:{live}?mode=ro", uri=True) as connection:
        before_schema = connection.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"
        ).fetchall()
        before_marker = connection.execute("SELECT value FROM marker").fetchall()
    boundary = _boundary_for(artifact.backup_envelope)
    before_external = boundary.ref, dict(boundary.value)
    install = Mock(side_effect=AssertionError("unauthenticated legacy must not migrate"))
    monkeypatch.setattr(RestoreMigrationInstallCoordinator, "install", install)
    coordinator = TrustedPhysicalRestoreCoordinator(
        live,
        LocalDurableEvidenceRegistry(),
        boundary,
        PhysicalBackupAdmissionValidator(BackupArtifactVerifier(authority)),
        RestoreLifecycleAuthorityBundle.from_migration_registry(PRODUCTION_MIGRATION_REGISTRY),
    )

    result = coordinator.restore_trusted_artifact(artifact)

    with sqlite3.connect(f"file:{live}?mode=ro", uri=True) as connection:
        after_schema = connection.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"
        ).fetchall()
        after_marker = connection.execute("SELECT value FROM marker").fetchall()
    assert result.decision is RestoreDecision.DENY
    assert live.read_bytes() == before_bytes
    assert after_schema == before_schema and after_marker == before_marker
    assert not any(name.startswith("state_store_") for _, name, _, _ in after_schema)
    assert not Path(f"{live}-wal").exists() and not Path(f"{live}-shm").exists()
    assert boundary.ref is before_external[0] and boundary.value == before_external[1]
    install.assert_not_called()


def test_v1_success_admission_lease_is_closed_once(tmp_path: Path, monkeypatch) -> None:
    artifact, authority = _legacy_artifact(tmp_path)
    original_close = AuthenticatedPhysicalBackupCandidate.close
    closed: list[object] = []

    def counted_close(self):  # type: ignore[no-untyped-def]
        closed.append(self)
        return original_close(self)

    monkeypatch.setattr(AuthenticatedPhysicalBackupCandidate, "close", counted_close)

    result = _coordinator(tmp_path, artifact, authority).restore_trusted_artifact(artifact)

    assert result.decision is RestoreDecision.RESTORE_EXTERNAL_COMMITTED_CURRENT
    assert len(closed) == 1


def test_v1_controlled_failure_denies_and_closes_admission_lease_once(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    artifact, authority = _legacy_artifact(tmp_path)
    original_close = AuthenticatedPhysicalBackupCandidate.close
    closed: list[object] = []

    def counted_close(self):  # type: ignore[no-untyped-def]
        closed.append(self)
        return original_close(self)

    monkeypatch.setattr(AuthenticatedPhysicalBackupCandidate, "close", counted_close)
    monkeypatch.setattr(
        RestoreMigrationInstallCoordinator,
        "install",
        Mock(side_effect=RestoreMigrationInstallError("controlled legacy failure")),
    )

    result = _coordinator(tmp_path, artifact, authority).restore_trusted_artifact(artifact)

    assert result == RestoreResult(RestoreDecision.DENY, "controlled legacy failure")
    assert len(closed) == 1


def test_public_c7_retry_recovers_without_migration_resume_replay(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    artifact, authority = _legacy_artifact(tmp_path)
    coordinator = _coordinator(tmp_path, artifact, authority)
    live = tmp_path / "live.sqlite3"
    staged, manifest = _migration_paths(artifact, live)
    original_replace = SQLiteStateStore.atomic_replace
    original_resume = RestoreMigrationResumeCoordinator.resume
    resume_calls: list[object] = []

    def counted_resume(self, source):  # type: ignore[no-untyped-def]
        resume_calls.append(source)
        return original_resume(self, source)

    def crash_after_replace(source, target):  # type: ignore[no-untyped-def]
        original_replace(source, target)
        raise StateStoreError("injected crash after rename")

    monkeypatch.setattr(RestoreMigrationResumeCoordinator, "resume", counted_resume)
    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", crash_after_replace)

    first = coordinator.restore_trusted_artifact(artifact)

    assert first.decision is RestoreDecision.DENY
    assert live.exists() and not staged.exists() and manifest.exists()
    with SQLiteStateStore(live) as installed:
        first_snapshot = installed.read_verified_snapshot()
    assert first_snapshot is not None
    assert first_snapshot.metadata.state_store_schema_version == 2
    calls_after_first = len(resume_calls)

    monkeypatch.setattr(SQLiteStateStore, "atomic_replace", original_replace)
    second = coordinator.restore_trusted_artifact(artifact)

    assert second.decision is RestoreDecision.NOOP_ALREADY_CURRENT
    assert len(resume_calls) == calls_after_first
    assert not staged.exists() and not manifest.exists()
    with SQLiteStateStore(live) as installed:
        second_snapshot = installed.read_verified_snapshot()
    assert second_snapshot == first_snapshot


def test_public_staging_lost_denies_without_replay_or_durable_mutation(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    artifact, authority = _legacy_artifact(tmp_path)
    live = tmp_path / "live.sqlite3"
    boundary = _boundary_for(artifact.backup_envelope)
    evidence = LocalDurableEvidenceRegistry()
    completed = RestoreMigrationResumeCoordinator(live, evidence, boundary).resume(
        artifact.backup_envelope
    )
    staged, manifest = _migration_paths(artifact, live)
    assert completed.sqlite_path == staged and manifest.exists()
    staged.unlink()
    shutil.copyfile(tmp_path / "legacy-source.sqlite3", live)
    live_before = live.read_bytes()
    external_before = boundary.ref, dict(boundary.value)
    original_close = AuthenticatedPhysicalBackupCandidate.close
    closed: list[object] = []
    resume = Mock(side_effect=AssertionError("staging-lost must not resume migration"))

    def counted_close(self):  # type: ignore[no-untyped-def]
        closed.append(self)
        return original_close(self)

    monkeypatch.setattr(RestoreMigrationResumeCoordinator, "resume", resume)
    monkeypatch.setattr(AuthenticatedPhysicalBackupCandidate, "close", counted_close)
    coordinator = TrustedPhysicalRestoreCoordinator(
        live,
        evidence,
        boundary,
        PhysicalBackupAdmissionValidator(BackupArtifactVerifier(authority)),
        RestoreLifecycleAuthorityBundle.from_migration_registry(PRODUCTION_MIGRATION_REGISTRY),
    )

    result = coordinator.restore_trusted_artifact(artifact)

    assert result.decision is RestoreDecision.DENY
    resume.assert_not_called()
    assert live.read_bytes() == live_before
    assert manifest.exists() and not staged.exists()
    assert boundary.ref is external_before[0] and boundary.value == external_before[1]
    assert len(closed) == 1
