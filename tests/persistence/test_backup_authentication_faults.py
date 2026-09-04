from __future__ import annotations

import sqlite3
from dataclasses import fields

import pytest
import bot_core.persistence.backup_authentication as backup_authentication

from bot_core.persistence.backup_authentication import (
    AuthorityKeyState,
    AuthorityUnavailableError,
    BackupArtifactAuthenticationAuthority,
    BackupArtifactAuthenticationScope,
    BackupArtifactAuthenticator,
    BackupArtifactVerificationResult,
    BackupArtifactVerifier,
    SQLiteBackupAuthenticationMetadataStore,
    SecretStorageBackupAuthenticationSecureCustody,
    UnknownAdminMutationOutcomeError,
)
from tests.persistence.test_backup_authentication import MemoryStorage, ident, manifest


def build(tmp_path, *, hook=None, storage=None):
    backend = storage or MemoryStorage()
    metadata = SQLiteBackupAuthenticationMetadataStore(tmp_path / "authority.sqlite")
    authority = BackupArtifactAuthenticationAuthority(
        metadata,
        SecretStorageBackupAuthenticationSecureCustody(backend),
        _fault_hook=hook,
    )
    return authority, metadata, backend


def new_scope():
    return BackupArtifactAuthenticationScope(ident("acct"), ident("dev"), "PAPER")


@pytest.mark.parametrize("operation", ["provision", "rotate", "rekey", "revoke"])
def test_response_loss_is_unknown_but_commit_survives(tmp_path, operation):
    scope = new_scope()
    base, metadata, storage = build(tmp_path)
    expected_revision = 0
    target = ""
    if operation != "provision":
        first = base.provision(scope)
        expected_revision = first.authority_revision
        target = first.keys[0].authority_key_id
    if operation == "rekey":
        base.revoke(scope, target, expected_revision)
        expected_revision += 1
    authority = BackupArtifactAuthenticationAuthority(
        metadata,
        SecretStorageBackupAuthenticationSecureCustody(storage),
        _fault_hook=lambda point: (
            (_ for _ in ()).throw(OSError("secret detail"))
            if point == "AFTER_METADATA_COMMIT_BEFORE_RETURN"
            else None
        ),
    )
    with pytest.raises(UnknownAdminMutationOutcomeError) as caught:
        if operation == "provision":
            authority.provision(scope)
        elif operation == "rotate":
            authority.rotate(scope, expected_revision)
        elif operation == "rekey":
            authority.rekey(scope, expected_revision)
        else:
            authority.revoke(scope, target, expected_revision)
    assert str(caught.value) == "UNKNOWN_ADMIN_MUTATION_OUTCOME"
    snapshot = base.read_snapshot(scope)
    assert snapshot is not None and snapshot.authority_revision == expected_revision + 1


@pytest.mark.parametrize("operation", ["provision", "rotate", "rekey"])
def test_failure_after_staging_leaves_metadata_unchanged_and_orphan_unusable(
    tmp_path, operation
):
    scope = new_scope()
    base, metadata, storage = build(tmp_path)
    revision = 0
    if operation != "provision":
        first = base.provision(scope)
        revision = 1
        if operation == "rekey":
            base.revoke(scope, first.keys[0].authority_key_id, 1)
            revision = 2
    before = base.read_snapshot(scope)
    authority = BackupArtifactAuthenticationAuthority(
        metadata,
        SecretStorageBackupAuthenticationSecureCustody(storage),
        _fault_hook=lambda point: (
            (_ for _ in ()).throw(RuntimeError("raw secret"))
            if point == "AFTER_KEY_DURABLE_PERSIST_BEFORE_METADATA_COMMIT"
            else None
        ),
    )
    with pytest.raises(AuthorityUnavailableError) as caught:
        (
            getattr(authority, operation)(scope, revision)
            if operation != "provision"
            else authority.provision(scope)
        )
    assert str(caught.value) == "AUTHORITY_UNAVAILABLE"
    after = base.read_snapshot(scope)
    assert after == before
    assert len(storage.values) == (1 if operation == "provision" else 2)


@pytest.mark.parametrize("operation", ["provision", "rotate", "rekey", "revoke"])
def test_true_before_commit_failure_rolls_back_transactional_dml(tmp_path, operation):
    scope = new_scope()
    base, metadata, storage = build(tmp_path)
    revision = 0
    target = ""
    proof = None
    if operation != "provision":
        first = base.provision(scope)
        revision = 1
        target = first.keys[0].authority_key_id
        proof = base.authenticate(scope, manifest(scope))
        if operation == "rekey":
            base.revoke(scope, target, revision)
            revision = 2
    before = base.read_snapshot(scope)
    authority = BackupArtifactAuthenticationAuthority(
        metadata,
        SecretStorageBackupAuthenticationSecureCustody(storage),
        _fault_hook=lambda point: (
            (_ for _ in ()).throw(RuntimeError("pre-commit"))
            if point == "BEFORE_METADATA_COMMIT"
            else None
        ),
    )
    with pytest.raises(AuthorityUnavailableError):
        if operation == "provision":
            authority.provision(scope)
        elif operation == "rotate":
            authority.rotate(scope, revision)
        elif operation == "rekey":
            authority.rekey(scope, revision)
        else:
            authority.revoke(scope, target, revision)
    assert base.read_snapshot(scope) == before
    if operation == "revoke":
        assert (
            base.verify(scope, manifest(scope), proof)
            is BackupArtifactVerificationResult.VERIFIED
        )


class BrokenStorage(MemoryStorage):
    fail_get = False
    fail_set = False

    def get_secret(self, key):
        if self.fail_get:
            raise RuntimeError("backend secret text")
        return super().get_secret(key)

    def set_secret(self, key, value):
        if self.fail_set:
            raise RuntimeError("backend secret text")
        super().set_secret(key, value)


def test_backend_failures_are_normalized(tmp_path):
    storage = BrokenStorage()
    authority, _, _ = build(tmp_path, storage=storage)
    scope = new_scope()
    storage.fail_set = True
    with pytest.raises(AuthorityUnavailableError) as caught:
        authority.provision(scope)
    assert str(caught.value) == "AUTHORITY_UNAVAILABLE"
    storage.fail_set = False
    authority.provision(scope)
    proof = authority.authenticate(scope, manifest(scope))
    storage.fail_get = True
    with pytest.raises(AuthorityUnavailableError):
        authority.authenticate(scope, manifest(scope))
    assert (
        authority.verify(scope, manifest(scope), proof)
        is BackupArtifactVerificationResult.AUTHORITY_UNAVAILABLE
    )


def test_malformed_committed_secret_fails_closed(tmp_path):
    authority, _, storage = build(tmp_path)
    scope = new_scope()
    authority.provision(scope)
    proof = authority.authenticate(scope, manifest(scope))
    storage.values[next(iter(storage.values))] = "not-base64!"
    with pytest.raises(AuthorityUnavailableError):
        authority.authenticate(scope, manifest(scope))
    assert (
        authority.verify(scope, manifest(scope), proof)
        is BackupArtifactVerificationResult.AUTHORITY_UNAVAILABLE
    )


@pytest.mark.parametrize("kind", ["revision", "lifecycle", "multiple_active"])
def test_corrupt_metadata_fails_closed(tmp_path, kind):
    authority, metadata, _ = build(tmp_path)
    scope = new_scope()
    authority.provision(scope)
    connection = sqlite3.connect(metadata.path)
    connection.execute("PRAGMA ignore_check_constraints=ON")
    if kind == "revision":
        connection.execute("UPDATE authority_scope SET revision=0")
    elif kind == "lifecycle":
        connection.execute("UPDATE authority_key SET lifecycle_state='BROKEN'")
    else:
        connection.execute("DROP INDEX one_active_per_scope")
        row = connection.execute(
            "SELECT account_id,device_installation_id,environment,purpose FROM authority_scope"
        ).fetchone()
        connection.execute(
            "INSERT INTO authority_key VALUES (?,?,?,?,?,?,?)",
            (*row, "other", "ACTIVE", "other-handle"),
        )
    connection.commit()
    connection.close()
    with pytest.raises(AuthorityUnavailableError):
        authority.read_snapshot(scope)


def test_public_objects_and_capabilities_do_not_leak_secrets(tmp_path):
    authority, _, storage = build(tmp_path)
    scope = new_scope()
    snapshot = authority.provision(scope)
    proof = authority.authenticate(scope, manifest(scope))
    raw = next(iter(storage.values.values()))
    assert raw not in repr(scope) + repr(snapshot) + repr(proof)
    assert "custody_handle" not in {field.name for field in fields(snapshot.keys[0])}
    authenticator, verifier = BackupArtifactAuthenticator(
        authority
    ), BackupArtifactVerifier(authority)
    for capability in (authenticator, verifier):
        assert not any(
            hasattr(capability, name)
            for name in ("provision", "rotate", "rekey", "revoke")
        )


def test_detected_custody_handle_collision_never_overwrites(tmp_path, monkeypatch):
    authority, _, storage = build(tmp_path)
    scope = new_scope()
    handle = "fixed-collision"
    storage.values[SecretStorageBackupAuthenticationSecureCustody._PREFIX + handle] = (
        "trusted"
    )
    monkeypatch.setattr(
        BackupArtifactAuthenticationAuthority,
        "_new_key",
        staticmethod(
            lambda: (
                backup_authentication._StoredKey(
                    "new-key", AuthorityKeyState.ACTIVE, handle
                ),
                b"x" * 32,
            )
        ),
    )
    with pytest.raises(AuthorityUnavailableError):
        authority.provision(scope)
    assert (
        storage.values[SecretStorageBackupAuthenticationSecureCustody._PREFIX + handle]
        == "trusted"
    )
    assert authority.read_snapshot(scope) is None


def test_authority_metadata_path_cannot_alias_state_store(tmp_path):
    path = tmp_path / "state.sqlite"
    with pytest.raises(ValueError):
        SQLiteBackupAuthenticationMetadataStore(path, state_store_path=path)
    assert not path.exists()
