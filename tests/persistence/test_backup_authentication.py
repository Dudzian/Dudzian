from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from uuid import uuid4

import pytest

from bot_core.persistence.backup_authentication import (
    BACKUP_AUTHENTICATION_ALGORITHM,
    BACKUP_AUTHENTICATION_PURPOSE,
    AlreadyProvisionedError,
    AuthorityKeyState,
    AuthorityKeyMetadata,
    AuthorityNotProvisionedError,
    AuthorityScopeCondition,
    BackupArtifactAuthenticationAuthority,
    BackupArtifactAuthenticationScope,
    BackupAuthenticationAuthorityScopeSnapshot,
    BackupArtifactVerificationResult,
    InvalidAuthorityOperationError,
    NoActiveKeyError,
    PhysicalArtifactAuthenticationProof,
    PhysicalSQLiteArtifactManifest,
    SQLiteBackupAuthenticationMetadataStore,
    SecretStorageBackupAuthenticationSecureCustody,
    StaleAuthorityRevisionError,
    canonical_authentication_payload,
)
from bot_core.security.base import SecretStorage


class MemoryStorage(SecretStorage):
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get_secret(self, key: str) -> str | None:
        return self.values.get(key)

    def set_secret(self, key: str, value: str) -> None:
        self.values[key] = value

    def delete_secret(self, key: str) -> None:
        self.values.pop(key, None)


def ident(prefix: str) -> str:
    value = list(str(uuid4()))
    value[14] = "7"
    return f"{prefix}_{''.join(value)}"


@pytest.fixture
def scope() -> BackupArtifactAuthenticationScope:
    return BackupArtifactAuthenticationScope(ident("acct"), ident("dev"), "PAPER")


def manifest(
    scope: BackupArtifactAuthenticationScope, *, artifact: str = "1" * 64
) -> PhysicalSQLiteArtifactManifest:
    data: dict[str, object] = {
        "artifact_format_version": 1,
        "account_id": scope.account_id,
        "device_installation_id": scope.device_installation_id,
        "environment": scope.environment,
        "state_store_identity_fingerprint_sha256": "2" * 64,
        "state_store_schema_version": 1,
        "local_protected_freshness_generation": 1,
        "state_fingerprint_sha256": "3" * 64,
        "transaction_fingerprint_sha256": "4" * 64,
        "backup_envelope_fingerprint_sha256": "5" * 64,
        "physical_artifact_sha256": artifact,
        "physical_artifact_byte_length": 42,
        "sqlite_schema_fingerprint_sha256": "6" * 64,
    }
    canonical = json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    data["manifest_fingerprint_sha256"] = hashlib.sha256(canonical).hexdigest()
    return PhysicalSQLiteArtifactManifest(**data)  # type: ignore[arg-type]


@pytest.fixture
def authority(tmp_path):
    storage = MemoryStorage()
    value = BackupArtifactAuthenticationAuthority(
        SQLiteBackupAuthenticationMetadataStore(tmp_path / "authority.sqlite"),
        SecretStorageBackupAuthenticationSecureCustody(storage),
    )
    return value, storage


def test_scope_and_proof_are_exact(scope):
    assert scope.purpose == BACKUP_AUTHENTICATION_PURPOSE
    with pytest.raises(ValueError):
        replace(scope, purpose="other")
    proof = PhysicalArtifactAuthenticationProof(
        1,
        BACKUP_AUTHENTICATION_ALGORITHM,
        BACKUP_AUTHENTICATION_PURPOSE,
        "key",
        "a" * 64,
        "b" * 64,
    )
    assert {field.name for field in fields(proof)} == {
        "proof_version",
        "algorithm",
        "purpose",
        "authority_key_id",
        "manifest_fingerprint_sha256",
        "authentication_tag_hex",
    }
    with pytest.raises(ValueError):
        PhysicalArtifactAuthenticationProof.from_mapping({**proof.to_mapping(), "extra": 1})
    for change in (
        {"proof_version": 2},
        {"algorithm": "SHA-256"},
        {"purpose": "other"},
        {"authentication_tag_hex": "B" * 64},
        {"manifest_fingerprint_sha256": "x"},
    ):
        with pytest.raises(ValueError):
            replace(proof, **change)


def test_canonical_payload_is_exact(scope):
    item = manifest(scope)
    expected = (
        BACKUP_AUTHENTICATION_PURPOSE.encode()
        + b"\0"
        + json.dumps(
            item.projection(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    )
    assert canonical_authentication_payload(item) == expected


def test_lifecycle_authentication_and_cross_scope(authority, scope):
    service, storage = authority
    item = manifest(scope)
    with pytest.raises(AuthorityNotProvisionedError):
        service.authenticate(scope, item)
    assert (
        service.verify(
            scope,
            item,
            PhysicalArtifactAuthenticationProof(
                1,
                BACKUP_AUTHENTICATION_ALGORITHM,
                BACKUP_AUTHENTICATION_PURPOSE,
                "unknown",
                item.manifest_fingerprint_sha256,
                "0" * 64,
            ),
        )
        is BackupArtifactVerificationResult.AUTHORITY_NOT_PROVISIONED
    )
    first = service.provision(scope)
    assert first.authority_revision == 1
    assert first.condition is AuthorityScopeCondition.PROVISIONED_WITH_ACTIVE
    assert len(storage.values) == 1
    proof = service.authenticate(scope, item)
    assert service.verify(scope, item, proof) is BackupArtifactVerificationResult.VERIFIED
    with pytest.raises(AlreadyProvisionedError):
        service.provision(scope)
    rotated = service.rotate(scope, 1)
    assert rotated.authority_revision == 2
    assert [key.state for key in rotated.keys].count(AuthorityKeyState.ACTIVE) == 1
    assert [key.state for key in rotated.keys].count(AuthorityKeyState.VERIFY_ONLY) == 1
    assert service.verify(scope, item, proof) is BackupArtifactVerificationResult.VERIFIED
    old = next(key for key in rotated.keys if key.state is AuthorityKeyState.VERIFY_ONLY)
    revoked = service.revoke(scope, old.authority_key_id, 2)
    assert service.verify(scope, item, proof) is BackupArtifactVerificationResult.REVOKED_KEY
    with pytest.raises(InvalidAuthorityOperationError):
        service.revoke(scope, old.authority_key_id, revoked.authority_revision)
    active = next(key for key in revoked.keys if key.state is AuthorityKeyState.ACTIVE)
    zero = service.revoke(scope, active.authority_key_id, 3)
    with pytest.raises(NoActiveKeyError):
        service.authenticate(scope, item)
    rekeyed = service.rekey(scope, zero.authority_revision)
    assert rekeyed.authority_revision == 5
    assert [key.state for key in rekeyed.keys].count(AuthorityKeyState.ACTIVE) == 1
    wrong = replace(scope, account_id=ident("acct"))
    assert (
        service.verify(wrong, manifest(wrong), proof)
        is BackupArtifactVerificationResult.INVALID_PROOF
    )


def test_changed_manifest_plain_hash_unknown_and_missing_material(authority, scope):
    service, storage = authority
    service.provision(scope)
    item = manifest(scope)
    proof = service.authenticate(scope, item)
    changed = manifest(scope, artifact="9" * 64)
    assert service.verify(scope, changed, proof) is BackupArtifactVerificationResult.INVALID_PROOF
    assert (
        service.verify(scope, item, replace(proof, authority_key_id="unknown"))
        is BackupArtifactVerificationResult.UNKNOWN_KEY
    )
    assert (
        service.verify(
            scope,
            item,
            replace(proof, authentication_tag_hex=item.physical_artifact_sha256),
        )
        is BackupArtifactVerificationResult.INVALID_PROOF
    )
    storage.values.clear()
    assert (
        service.verify(scope, item, proof) is BackupArtifactVerificationResult.AUTHORITY_UNAVAILABLE
    )


def test_stale_revision_fences_mutations(authority, scope):
    service, _ = authority
    service.provision(scope)
    service.rotate(scope, 1)
    with pytest.raises(StaleAuthorityRevisionError):
        service.rotate(scope, 1)


@pytest.mark.parametrize("revision", [True, False, 0, -1, 1.0, "1", None])
def test_snapshot_rejects_non_exact_positive_revision(scope, revision):
    with pytest.raises(ValueError):
        BackupAuthenticationAuthorityScopeSnapshot(scope, revision, ())


def test_snapshot_validates_exact_public_shape(scope):
    active = AuthorityKeyMetadata("key", AuthorityKeyState.ACTIVE)
    assert BackupAuthenticationAuthorityScopeSnapshot(scope, 1, (active,)).authority_revision == 1
    with pytest.raises(ValueError):
        AuthorityKeyMetadata("", AuthorityKeyState.ACTIVE)
    with pytest.raises(ValueError):
        BackupAuthenticationAuthorityScopeSnapshot(scope, 1, [active])
    with pytest.raises(ValueError):
        BackupAuthenticationAuthorityScopeSnapshot(scope, 1, (active, active))


def test_admin_and_store_reject_boolean_revision(authority, scope):
    service, _ = authority
    first = service.provision(scope)
    with pytest.raises(InvalidAuthorityOperationError):
        service.rotate(scope, True)
    with pytest.raises(InvalidAuthorityOperationError):
        service.revoke(scope, first.keys[0].authority_key_id, True)
    service.revoke(scope, first.keys[0].authority_key_id, 1)
    with pytest.raises(InvalidAuthorityOperationError):
        service.rekey(scope, True)
    with pytest.raises(ValueError):
        service._metadata.replace(scope, True, ())


@pytest.mark.parametrize("iteration", range(10))
def test_concurrent_provision_has_one_winner(tmp_path, scope, iteration):
    storage = MemoryStorage()
    path = tmp_path / f"authority-{iteration}.sqlite"
    services = [
        BackupArtifactAuthenticationAuthority(
            SQLiteBackupAuthenticationMetadataStore(path),
            SecretStorageBackupAuthenticationSecureCustody(storage),
        )
        for _ in range(2)
    ]

    def run(service):
        try:
            service.provision(scope)
            return True
        except AlreadyProvisionedError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(run, services))
    assert outcomes.count(True) == 1
    snapshot = services[0].read_snapshot(scope)
    assert snapshot is not None and snapshot.authority_revision == 1


@pytest.mark.parametrize("iteration", range(10))
def test_concurrent_rotate_has_one_revision_winner(tmp_path, scope, iteration):
    storage = MemoryStorage()
    path = tmp_path / f"rotate-{iteration}.sqlite"
    services = [
        BackupArtifactAuthenticationAuthority(
            SQLiteBackupAuthenticationMetadataStore(path),
            SecretStorageBackupAuthenticationSecureCustody(storage),
        )
        for _ in range(2)
    ]
    services[0].provision(scope)

    def run(service):
        try:
            service.rotate(scope, 1)
            return True
        except StaleAuthorityRevisionError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(run, services))
    assert outcomes.count(True) == 1
    snapshot = services[0].read_snapshot(scope)
    assert snapshot is not None and snapshot.authority_revision == 2
    assert sum(key.state is AuthorityKeyState.ACTIVE for key in snapshot.keys) == 1
