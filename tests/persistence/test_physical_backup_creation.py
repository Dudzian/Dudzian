from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import fields

import pytest

from bot_core.persistence.backup_authentication import BackupArtifactVerificationResult
from bot_core.persistence.backup_authentication import (
    BackupArtifactAuthenticationAuthority,
    BackupArtifactAuthenticator,
    BackupArtifactAuthenticationScope,
    SQLiteBackupAuthenticationMetadataStore,
    SecretStorageBackupAuthenticationSecureCustody,
)
from bot_core.persistence.physical_backup import (
    PhysicalBackupAdmissionValidator,
    PhysicalBackupCreator,
    PhysicalBackupError,
    TrustedPhysicalBackupArtifact,
)
from tests.persistence.physical_backup_helpers import (
    artifact_fixture,
    authority_for,
    MemoryStorage,
)
from tests.persistence.test_backup_envelope import _store
from tests.persistence.test_state_store_records import _commit, _metadata, _runtime


def test_creates_exact_authenticated_self_contained_online_backup(tmp_path):
    store, _authority, artifact, verifier = artifact_fixture(tmp_path)
    try:
        assert isinstance(artifact, TrustedPhysicalBackupArtifact)
        assert len(fields(TrustedPhysicalBackupArtifact)) == 4
        payload = artifact.physical_artifact.path.read_bytes()
        assert (
            hashlib.sha256(payload).hexdigest()
            == artifact.manifest.physical_artifact_sha256
        )
        assert len(payload) == artifact.manifest.physical_artifact_byte_length
        assert not artifact.physical_artifact.path.with_name(
            "backup.sqlite-wal"
        ).exists()
        assert not artifact.physical_artifact.path.with_name(
            "backup.sqlite-shm"
        ).exists()
        with sqlite3.connect(artifact.physical_artifact.path) as connection:
            assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        metadata = store.read_metadata()
        assert metadata is not None
        assert artifact.manifest.local_protected_freshness_generation == (
            metadata.protected_freshness_generation
        )
        assert artifact.backup_envelope.state_fingerprint_sha256 == (
            artifact.manifest.state_fingerprint_sha256
        )
        scope = (
            verifier._authority.read_snapshot
        )  # prove no capability is bundle-carried
        del scope
        from bot_core.persistence.backup_authentication import (
            BackupArtifactAuthenticationScope,
        )

        auth_scope = BackupArtifactAuthenticationScope(
            metadata.account_id, metadata.device_installation_id, metadata.environment
        )
        assert verifier.verify(
            auth_scope, artifact.manifest, artifact.authentication_proof
        ) is (BackupArtifactVerificationResult.VERIFIED)
    finally:
        store.close()


def test_backup_does_not_change_live_bytes_or_generation(tmp_path):
    store = _store(tmp_path / "live.sqlite")
    authority = authority_for(tmp_path, store)
    try:
        before = store.read_verified_snapshot()
        PhysicalBackupCreator(BackupArtifactAuthenticator(authority)).create(
            store, tmp_path / "backup.sqlite"
        )
        after = store.read_verified_snapshot()
        assert before == after
    finally:
        store.close()


def test_repeated_destination_replaces_with_later_complete_backup(tmp_path):
    store = _store(tmp_path / "live.sqlite")
    authority = authority_for(tmp_path, store)
    creator = PhysicalBackupCreator(BackupArtifactAuthenticator(authority))
    output = tmp_path / "backup.sqlite"
    try:
        first = creator.create(store, output)
        _commit(
            store,
            _metadata(2),
            history=(_runtime("run_01890f4c-7b9a-7cc2-8a2b-123456789abc"),),
            expected=1,
        )
        second = creator.create(store, output)
        assert (
            first.manifest.physical_artifact_sha256
            != second.manifest.physical_artifact_sha256
        )
        assert second.manifest.local_protected_freshness_generation == 2
    finally:
        store.close()


def test_failure_before_publication_preserves_previous_good_output(tmp_path):
    store = _store(tmp_path / "live.sqlite")
    authority = authority_for(tmp_path, store)
    output = tmp_path / "backup.sqlite"
    creator = PhysicalBackupCreator(BackupArtifactAuthenticator(authority))
    try:
        first = creator.create(store, output)
        original = output.read_bytes()
        failing = PhysicalBackupCreator(
            BackupArtifactAuthenticator(authority),
            _before_publication_hook=lambda _path: (_ for _ in ()).throw(
                RuntimeError()
            ),
        )
        with pytest.raises(PhysicalBackupError):
            failing.create(store, output)
        assert output.read_bytes() == original
        assert (
            hashlib.sha256(original).hexdigest()
            == first.manifest.physical_artifact_sha256
        )
        from bot_core.persistence.backup_authentication import BackupArtifactVerifier

        with PhysicalBackupAdmissionValidator(BackupArtifactVerifier(authority)).admit(
            first
        ) as preserved:
            assert (
                preserved.classification
                == "ELIGIBLE_FOR_FURTHER_RESTORE_VALIDATION_ONLY"
            )
        assert not list(tmp_path.glob(".physical-backup-*"))
    finally:
        store.close()


def test_mutating_internal_failpoint_cannot_publish_unauthenticated_bytes(tmp_path):
    store = _store(tmp_path / "live.sqlite")
    authority = authority_for(tmp_path, store)
    output = tmp_path / "backup.sqlite"
    creator = PhysicalBackupCreator(BackupArtifactAuthenticator(authority))
    try:
        first = creator.create(store, output)
        original = output.read_bytes()
        _commit(
            store,
            _metadata(2),
            history=(_runtime("run_01890f4c-7b9a-7cc2-8a2b-123456789abc"),),
            expected=1,
        )

        def mutate_and_return(path):
            path.write_bytes(b"not the authenticated candidate")

        with pytest.raises(PhysicalBackupError):
            PhysicalBackupCreator(
                BackupArtifactAuthenticator(authority),
                _before_publication_hook=mutate_and_return,
            ).create(store, output)
        assert output.read_bytes() == original
        from bot_core.persistence.backup_authentication import BackupArtifactVerifier

        with PhysicalBackupAdmissionValidator(BackupArtifactVerifier(authority)).admit(
            first
        ):
            pass
        assert not list(tmp_path.glob(".physical-backup-*"))
    finally:
        store.close()


def test_wal_snapshot_is_standalone_and_contains_committed_state(tmp_path):
    store = _store(tmp_path / "live.sqlite", generations=2)
    authority = authority_for(tmp_path, store)
    try:
        expected = store.read_verified_snapshot()
        artifact = PhysicalBackupCreator(BackupArtifactAuthenticator(authority)).create(
            store, tmp_path / "backup.sqlite"
        )
        actual = store.read_isolated_verified_snapshot(artifact.physical_artifact.path)
        assert actual == expected
        assert (
            actual is not None and actual.metadata.protected_freshness_generation == 2
        )
        with sqlite3.connect(artifact.physical_artifact.path) as connection:
            assert connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    finally:
        store.close()


def test_output_may_not_alias_live_store(tmp_path):
    store = _store(tmp_path / "live.sqlite")
    authority = authority_for(tmp_path, store)
    try:
        with pytest.raises(PhysicalBackupError):
            PhysicalBackupCreator(BackupArtifactAuthenticator(authority)).create(
                store, store.path
            )
    finally:
        store.close()


@pytest.mark.parametrize("writer_first", [False, True])
def test_writer_and_backup_forced_orders_are_never_torn(
    tmp_path, monkeypatch, writer_first
):
    store = _store(tmp_path / "live.sqlite")
    writer = type(store)(store.path)
    authority = authority_for(tmp_path, store)

    def commit_next():
        _commit(
            writer,
            _metadata(2),
            history=(_runtime("run_01890f4c-7b9a-7cc2-8a2b-123456789abc"),),
            expected=1,
        )

    try:
        if writer_first:
            commit_next()
        else:
            original = store.capture_verified_physical_snapshot

            def capture_then_write(path):
                snapshot = original(path)
                commit_next()
                return snapshot

            monkeypatch.setattr(
                store, "capture_verified_physical_snapshot", capture_then_write
            )
        artifact = PhysicalBackupCreator(BackupArtifactAuthenticator(authority)).create(
            store, tmp_path / "backup.sqlite"
        )
        physical = store.read_isolated_verified_snapshot(
            artifact.physical_artifact.path
        )
        assert physical is not None
        point = (
            physical.metadata.protected_freshness_generation,
            physical.metadata.state_fingerprint_sha256,
            physical.metadata.transaction_fingerprint_sha256,
        )
        assert (
            point
            == (
                artifact.manifest.local_protected_freshness_generation,
                artifact.manifest.state_fingerprint_sha256,
                artifact.manifest.transaction_fingerprint_sha256,
            )
            == (
                artifact.backup_envelope.local_protected_freshness_generation,
                artifact.backup_envelope.state_fingerprint_sha256,
                artifact.backup_envelope.transaction_fingerprint_sha256,
            )
        )
        assert point[0] == (2 if writer_first else 1)
    finally:
        writer.close()
        store.close()


@pytest.mark.parametrize("provision_then_revoke", [False, True])
def test_authority_creation_failure_publishes_nothing(tmp_path, provision_then_revoke):
    store = _store(tmp_path / "live.sqlite")
    metadata = store.read_metadata()
    assert metadata is not None
    scope = BackupArtifactAuthenticationScope(
        metadata.account_id, metadata.device_installation_id, metadata.environment
    )
    authority = BackupArtifactAuthenticationAuthority(
        SQLiteBackupAuthenticationMetadataStore(tmp_path / "authority.sqlite"),
        SecretStorageBackupAuthenticationSecureCustody(MemoryStorage()),
    )
    if provision_then_revoke:
        snapshot = authority.provision(scope)
        authority.revoke(
            scope, snapshot.keys[0].authority_key_id, snapshot.authority_revision
        )
    output = tmp_path / "backup.sqlite"
    try:
        with pytest.raises(PhysicalBackupError):
            PhysicalBackupCreator(BackupArtifactAuthenticator(authority)).create(
                store, output
            )
        assert not output.exists()
        assert not list(tmp_path.glob(".physical-backup-*"))
    finally:
        store.close()
