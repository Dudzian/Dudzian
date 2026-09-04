from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace

import pytest

from bot_core.persistence.backup_authentication import (
    BackupArtifactAuthenticator,
    BackupArtifactAuthenticationScope,
    BackupArtifactVerificationResult,
    PhysicalSQLiteArtifactManifest,
)

from bot_core.persistence.physical_backup import (
    PhysicalBackupAdmissionError,
    PhysicalBackupAdmissionReason,
    PhysicalBackupAdmissionValidator,
    PhysicalBackupCreator,
)
from bot_core.persistence.state_store import SQLiteStateStore
from tests.persistence.physical_backup_helpers import artifact_fixture
from tests.persistence.test_state_store_records import _commit, _metadata, _runtime


def _manifest_with(authority, artifact, **changes):
    values = artifact.manifest.projection(include_fingerprint=False)
    values.update(changes)
    manifest = PhysicalSQLiteArtifactManifest.create(**values)
    envelope = artifact.backup_envelope
    scope = BackupArtifactAuthenticationScope(
        envelope.account_id, envelope.device_installation_id, envelope.environment
    )
    proof = BackupArtifactAuthenticator(authority).authenticate(scope, manifest)
    return replace(artifact, manifest=manifest, authentication_proof=proof)


@pytest.mark.parametrize(
    "change",
    [
        {"artifact_format_version": True},
        {"physical_artifact_sha256": "bad"},
    ],
)
def test_manifest_parser_rejects_wrong_security_types(tmp_path, change):
    _store, _authority, artifact, _verifier = artifact_fixture(tmp_path)
    mapping = artifact.manifest.projection()
    mapping.update(change)
    with pytest.raises(ValueError):
        PhysicalSQLiteArtifactManifest.from_mapping(mapping)
    _store.close()


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_manifest_parser_requires_exact_fields(tmp_path, mutation):
    store, _authority, artifact, _verifier = artifact_fixture(tmp_path)
    mapping = artifact.manifest.projection()
    if mutation == "extra":
        mapping["extra"] = "forbidden"
    else:
        del mapping["environment"]
    try:
        with pytest.raises(ValueError):
            PhysicalSQLiteArtifactManifest.from_mapping(mapping)
    finally:
        store.close()


def test_admits_only_for_further_validation_and_cleans_up(tmp_path):
    store, _authority, artifact, verifier = artifact_fixture(tmp_path)
    try:
        candidate = PhysicalBackupAdmissionValidator(verifier).admit(artifact)
        path = candidate.private_path
        assert (
            candidate.classification == "ELIGIBLE_FOR_FURTHER_RESTORE_VALIDATION_ONLY"
        )
        assert candidate.state_store_snapshot.metadata.state_fingerprint_sha256 == (
            artifact.manifest.state_fingerprint_sha256
        )
        candidate.close()
        assert not path.exists()
        candidate.close()
    finally:
        store.close()


def test_hash_mismatch_rejects_before_sqlite_open(tmp_path, monkeypatch):
    store, _authority, artifact, verifier = artifact_fixture(tmp_path)
    opened = 0

    def forbidden(_path):
        nonlocal opened
        opened += 1
        raise AssertionError

    monkeypatch.setattr(
        "bot_core.persistence.physical_backup._validate_sqlite", forbidden
    )
    monkeypatch.setattr(
        "bot_core.persistence.physical_backup.SQLiteStateStore.read_isolated_verified_snapshot",
        forbidden,
    )
    artifact.physical_artifact.path.chmod(0o600)
    artifact.physical_artifact.path.write_bytes(b"not authenticated")
    try:
        with pytest.raises(PhysicalBackupAdmissionError) as raised:
            PhysicalBackupAdmissionValidator(verifier).admit(artifact)
        assert (
            raised.value.reason is PhysicalBackupAdmissionReason.PHYSICAL_HASH_MISMATCH
        )
        assert opened == 0
    finally:
        store.close()


def test_invalid_proof_rejects_before_sqlite_open(tmp_path, monkeypatch):
    store, _authority, artifact, verifier = artifact_fixture(tmp_path)
    opened = 0

    def forbidden(_path):
        nonlocal opened
        opened += 1
        raise AssertionError

    monkeypatch.setattr(
        "bot_core.persistence.physical_backup._validate_sqlite", forbidden
    )
    monkeypatch.setattr(
        "bot_core.persistence.physical_backup.SQLiteStateStore.read_isolated_verified_snapshot",
        forbidden,
    )
    bad = replace(
        artifact,
        authentication_proof=replace(
            artifact.authentication_proof, authentication_tag_hex="0" * 64
        ),
    )
    try:
        with pytest.raises(PhysicalBackupAdmissionError) as raised:
            PhysicalBackupAdmissionValidator(verifier).admit(bad)
        assert (
            raised.value.reason is PhysicalBackupAdmissionReason.AUTHENTICATION_REJECTED
        )
        assert opened == 0
    finally:
        store.close()


def test_post_auth_corruption_is_rehashed_before_sqlite_open(tmp_path, monkeypatch):
    store, _authority, artifact, verifier = artifact_fixture(tmp_path)
    opened = 0

    def corrupt(path):
        path.write_bytes(b"substituted")

    def forbidden(_path):
        nonlocal opened
        opened += 1

    monkeypatch.setattr(
        "bot_core.persistence.physical_backup._validate_sqlite", forbidden
    )
    try:
        with pytest.raises(PhysicalBackupAdmissionError) as raised:
            PhysicalBackupAdmissionValidator(
                verifier, authenticated_hook=corrupt
            ).admit(artifact)
        assert (
            raised.value.reason is PhysicalBackupAdmissionReason.PHYSICAL_HASH_MISMATCH
        )
        assert opened == 0
    finally:
        store.close()


def test_adjacent_sidecars_are_never_consumed(tmp_path):
    store, _authority, artifact, verifier = artifact_fixture(tmp_path)
    artifact.physical_artifact.path.with_name("backup.sqlite-wal").write_bytes(
        b"attacker"
    )
    artifact.physical_artifact.path.with_name("backup.sqlite-shm").write_bytes(
        b"attacker"
    )
    try:
        with PhysicalBackupAdmissionValidator(verifier).admit(artifact) as candidate:
            assert not candidate.private_path.with_name("candidate.sqlite-wal").exists()
            assert not candidate.private_path.with_name("candidate.sqlite-shm").exists()
            assert candidate.state_store_snapshot.metadata.state_fingerprint_sha256 == (
                artifact.manifest.state_fingerprint_sha256
            )
    finally:
        store.close()


def test_admitted_public_facts_are_immutable_but_cleanup_remains_idempotent(tmp_path):
    store, _authority, artifact, verifier = artifact_fixture(tmp_path)
    candidate = PhysicalBackupAdmissionValidator(verifier).admit(artifact)
    try:
        replacements = {
            "private_path": tmp_path / "other",
            "backup_envelope": artifact.backup_envelope,
            "manifest": artifact.manifest,
            "authentication_proof": artifact.authentication_proof,
            "physical_artifact_sha256": "0" * 64,
            "physical_artifact_byte_length": 1,
            "sqlite_schema_fingerprint_sha256": "0" * 64,
            "state_store_snapshot": candidate.state_store_snapshot,
            "classification": "LIVE",
        }
        assert set(replacements) <= {field.name for field in fields(candidate)}
        for name, value in replacements.items():
            with pytest.raises(FrozenInstanceError):
                setattr(candidate, name, value)
    finally:
        candidate.close()
        candidate.close()
        store.close()


def test_post_admission_mutation_fails_continuity(tmp_path):
    store, _authority, artifact, verifier = artifact_fixture(tmp_path)
    candidate = PhysicalBackupAdmissionValidator(verifier).admit(artifact)
    try:
        candidate.verify_physical_continuity()
        candidate.private_path.write_bytes(b"changed after admission")
        with pytest.raises(PhysicalBackupAdmissionError) as raised:
            candidate.verify_physical_continuity()
        assert (
            raised.value.reason is PhysicalBackupAdmissionReason.PHYSICAL_HASH_MISMATCH
        )
    finally:
        candidate.close()
        store.close()


@pytest.mark.parametrize(
    "result",
    [
        BackupArtifactVerificationResult.AUTHORITY_UNAVAILABLE,
        BackupArtifactVerificationResult.AUTHORITY_NOT_PROVISIONED,
    ],
)
def test_unavailable_authority_rejects_before_both_sqlite_openers(
    tmp_path, monkeypatch, result
):
    store, _authority, artifact, _verifier = artifact_fixture(tmp_path)
    opened = []

    class Verifier:
        def verify(self, *_args):
            return result

    monkeypatch.setattr(
        "bot_core.persistence.physical_backup._validate_sqlite",
        lambda path: opened.append(("sqlite", path)),
    )
    monkeypatch.setattr(
        "bot_core.persistence.physical_backup.SQLiteStateStore.read_isolated_verified_snapshot",
        lambda path: opened.append(("store", path)),
    )
    try:
        with pytest.raises(PhysicalBackupAdmissionError) as raised:
            PhysicalBackupAdmissionValidator(Verifier()).admit(artifact)  # type: ignore[arg-type]
        assert (
            raised.value.reason is PhysicalBackupAdmissionReason.AUTHORITY_UNAVAILABLE
        )
        assert opened == []
    finally:
        store.close()


def test_unknown_key_rejects_before_both_sqlite_openers(tmp_path, monkeypatch):
    store, _authority, artifact, verifier = artifact_fixture(tmp_path)
    opened = []
    bad = replace(
        artifact,
        authentication_proof=replace(
            artifact.authentication_proof, authority_key_id="unknown"
        ),
    )
    monkeypatch.setattr(
        "bot_core.persistence.physical_backup._validate_sqlite",
        lambda path: opened.append(("sqlite", path)),
    )
    monkeypatch.setattr(
        "bot_core.persistence.physical_backup.SQLiteStateStore.read_isolated_verified_snapshot",
        lambda path: opened.append(("store", path)),
    )
    try:
        with pytest.raises(PhysicalBackupAdmissionError):
            PhysicalBackupAdmissionValidator(verifier).admit(bad)
        assert opened == []
    finally:
        store.close()


def test_rotated_proof_admits_then_revoked_proof_rejects_before_open(
    tmp_path, monkeypatch
):
    store, authority, artifact, verifier = artifact_fixture(tmp_path)
    metadata = store.read_metadata()
    assert metadata is not None
    scope = BackupArtifactAuthenticationScope(
        metadata.account_id, metadata.device_installation_id, metadata.environment
    )
    rotated = authority.rotate(scope, 1)
    try:
        with PhysicalBackupAdmissionValidator(verifier).admit(artifact):
            pass
        old_key = artifact.authentication_proof.authority_key_id
        authority.revoke(scope, old_key, rotated.authority_revision)
        opened = []
        monkeypatch.setattr(
            "bot_core.persistence.physical_backup._validate_sqlite",
            lambda path: opened.append(path),
        )
        monkeypatch.setattr(
            "bot_core.persistence.physical_backup.SQLiteStateStore.read_isolated_verified_snapshot",
            lambda path: opened.append(path),
        )
        with pytest.raises(PhysicalBackupAdmissionError) as raised:
            PhysicalBackupAdmissionValidator(verifier).admit(artifact)
        assert (
            raised.value.reason is PhysicalBackupAdmissionReason.AUTHENTICATION_REJECTED
        )
        assert opened == []
    finally:
        store.close()


def test_valid_hmac_with_wrong_schema_rejects_after_sqlite_open(tmp_path, monkeypatch):
    store, authority, artifact, verifier = artifact_fixture(tmp_path)
    bad = _manifest_with(authority, artifact, sqlite_schema_fingerprint_sha256="0" * 64)
    opened = 0
    from bot_core.persistence import physical_backup

    original = physical_backup._validate_sqlite

    def counted(path):
        nonlocal opened
        opened += 1
        return original(path)

    monkeypatch.setattr(physical_backup, "_validate_sqlite", counted)
    try:
        with pytest.raises(PhysicalBackupAdmissionError) as raised:
            PhysicalBackupAdmissionValidator(verifier).admit(bad)
        assert (
            raised.value.reason is PhysicalBackupAdmissionReason.SQLITE_SCHEMA_MISMATCH
        )
        assert opened == 1
    finally:
        store.close()


def test_authenticated_non_sqlite_bytes_fail_integrity_after_authentication(tmp_path):
    import hashlib

    store, authority, artifact, verifier = artifact_fixture(tmp_path)
    payload = b"authenticated but not sqlite"
    artifact.physical_artifact.path.write_bytes(payload)
    bad = _manifest_with(
        authority,
        artifact,
        physical_artifact_sha256=hashlib.sha256(payload).hexdigest(),
        physical_artifact_byte_length=len(payload),
    )
    try:
        with pytest.raises(PhysicalBackupAdmissionError) as raised:
            PhysicalBackupAdmissionValidator(verifier).admit(bad)
        assert (
            raised.value.reason
            is PhysicalBackupAdmissionReason.SQLITE_INTEGRITY_FAILURE
        )
    finally:
        store.close()


def test_authenticated_same_schema_different_dml_rejects_candidate_state(tmp_path):
    store, authority, first, verifier = artifact_fixture(tmp_path)
    try:
        _commit(
            store,
            _metadata(2),
            history=(_runtime("run_01890f4c-7b9a-7cc2-8a2b-123456789abc"),),
            expected=1,
        )
        second = PhysicalBackupCreator(BackupArtifactAuthenticator(authority)).create(
            store, tmp_path / "second.sqlite"
        )
        mixed = replace(first, physical_artifact=second.physical_artifact)
        mixed = _manifest_with(
            authority,
            mixed,
            physical_artifact_sha256=second.manifest.physical_artifact_sha256,
            physical_artifact_byte_length=second.manifest.physical_artifact_byte_length,
            sqlite_schema_fingerprint_sha256=second.manifest.sqlite_schema_fingerprint_sha256,
        )
        with pytest.raises(PhysicalBackupAdmissionError) as raised:
            PhysicalBackupAdmissionValidator(verifier).admit(mixed)
        assert (
            raised.value.reason
            is PhysicalBackupAdmissionReason.CANDIDATE_STATE_MISMATCH
        )
    finally:
        store.close()


def test_structurally_valid_envelope_mismatch_rejects_before_both_openers(
    tmp_path, monkeypatch
):
    store, authority, first, verifier = artifact_fixture(tmp_path)
    try:
        _commit(
            store,
            _metadata(2),
            history=(_runtime("run_01890f4c-7b9a-7cc2-8a2b-123456789abc"),),
            expected=1,
        )
        second = PhysicalBackupCreator(BackupArtifactAuthenticator(authority)).create(
            store, tmp_path / "second.sqlite"
        )
        mismatched = replace(first, backup_envelope=second.backup_envelope)
        opened = []
        monkeypatch.setattr(
            "bot_core.persistence.physical_backup._validate_sqlite",
            lambda path: opened.append(("sqlite", path)),
        )
        monkeypatch.setattr(
            "bot_core.persistence.physical_backup.SQLiteStateStore.read_isolated_verified_snapshot",
            lambda path: opened.append(("store", path)),
        )
        with pytest.raises(PhysicalBackupAdmissionError) as raised:
            PhysicalBackupAdmissionValidator(verifier).admit(mismatched)
        assert raised.value.reason is PhysicalBackupAdmissionReason.ENVELOPE_MISMATCH
        assert opened == []
    finally:
        store.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state_store_identity_fingerprint_sha256", "0" * 64),
        ("state_store_schema_version", 2),
        ("local_protected_freshness_generation", 2),
        ("state_fingerprint_sha256", "0" * 64),
        ("transaction_fingerprint_sha256", "0" * 64),
    ],
)
def test_authenticated_outer_candidate_fact_mismatch_rejects_after_sqlite_validation(
    tmp_path, monkeypatch, field, value
):
    store, authority, artifact, verifier = artifact_fixture(tmp_path)
    bad = _manifest_with(authority, artifact, **{field: value})
    opened = []
    from bot_core.persistence import physical_backup

    validate = physical_backup._validate_sqlite

    def counted(path):
        opened.append("sqlite")
        return validate(path)

    monkeypatch.setattr(physical_backup, "_validate_sqlite", counted)
    try:
        with pytest.raises(PhysicalBackupAdmissionError) as raised:
            PhysicalBackupAdmissionValidator(verifier).admit(bad)
        assert (
            raised.value.reason
            is PhysicalBackupAdmissionReason.CANDIDATE_STATE_MISMATCH
        )
        assert opened == ["sqlite"]
    finally:
        store.close()


def test_admission_does_not_mutate_live_store(tmp_path):
    store, _authority, artifact, verifier = artifact_fixture(tmp_path)
    before = store.read_verified_snapshot()
    try:
        with PhysicalBackupAdmissionValidator(verifier).admit(artifact) as candidate:
            candidate.verify_physical_continuity()
        after = store.read_verified_snapshot()
        assert before == after
    finally:
        store.close()


def test_authenticated_candidate_environment_mismatch_rejects_after_sqlite(tmp_path):
    store, authority, outer, verifier = artifact_fixture(tmp_path)
    other = SQLiteStateStore(tmp_path / "other.sqlite")
    try:
        other_metadata = _metadata(environment="TESTNET")
        from tests.persistence.test_state_store_records import _account, _runtime

        _commit(other, other_metadata, current=(_account(),), history=(_runtime(),))
        other_scope = BackupArtifactAuthenticationScope(
            other_metadata.account_id,
            other_metadata.device_installation_id,
            other_metadata.environment,
        )
        authority.provision(other_scope)
        physical = PhysicalBackupCreator(BackupArtifactAuthenticator(authority)).create(
            other, tmp_path / "other-backup.sqlite"
        )
        mixed = replace(outer, physical_artifact=physical.physical_artifact)
        mixed = _manifest_with(
            authority,
            mixed,
            physical_artifact_sha256=physical.manifest.physical_artifact_sha256,
            physical_artifact_byte_length=physical.manifest.physical_artifact_byte_length,
            sqlite_schema_fingerprint_sha256=physical.manifest.sqlite_schema_fingerprint_sha256,
        )
        with pytest.raises(PhysicalBackupAdmissionError) as raised:
            PhysicalBackupAdmissionValidator(verifier).admit(mixed)
        assert (
            raised.value.reason
            is PhysicalBackupAdmissionReason.CANDIDATE_STATE_MISMATCH
        )
    finally:
        other.close()
        store.close()
