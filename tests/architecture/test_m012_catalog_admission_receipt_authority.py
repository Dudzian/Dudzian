from __future__ import annotations

from dataclasses import replace
import hashlib
import hmac
import json
import sqlite3

import pytest

from bot_core.instruments.catalog_admission_receipt import (
    CATALOG_ADMISSION_RECEIPT_ALGORITHM,
    CATALOG_ADMISSION_RECEIPT_DOMAIN,
    CATALOG_ADMISSION_RECEIPT_PURPOSE,
    CATALOG_RECEIPT_AUTHORITY_ANCHOR_PURPOSE,
    CATALOG_RECEIPT_KEY_LIFECYCLE_PURPOSE,
    CatalogAdmissionReceipt,
    CatalogAdmissionReceiptAuthority,
    CatalogAdmissionReceiptAuthorityUnavailable,
    SecretStorageCatalogAdmissionReceiptSecureCustody,
    SQLiteCatalogAdmissionReceiptMetadataStore,
    canonical_catalog_admission_receipt_payload,
)
from bot_core.instruments.testing_catalog_admission_receipt import (
    TestCatalogAdmissionReceiptAuthority,
)
from bot_core.persistence.backup_authentication import BACKUP_AUTHENTICATION_PURPOSE
from bot_core.security.base import SecretStorage
from bot_core.security.keyring_storage import KeyringSecretStorage


CATALOG = "a" * 64
MEMBERSHIP = "b" * 64
WHEN = "2030-01-02T03:04:05.123456Z"


class _MemorySecretStorage(SecretStorage):
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get_secret(self, key: str) -> str | None:
        return self.values.get(key)

    def set_secret(self, key: str, value: str) -> None:
        self.values[key] = value

    def delete_secret(self, key: str) -> None:
        self.values.pop(key, None)


class ExfiltratingSecretStorage(_MemorySecretStorage):
    """A caller-controlled backend that must never enter production custody."""


@pytest.fixture
def production(tmp_path, monkeypatch):
    path = tmp_path / "catalog-receipts.sqlite3"
    secrets: dict[str, str] = {}

    def initialize(storage, **_kwargs):
        storage._catalog_test_values = secrets

    monkeypatch.setattr(KeyringSecretStorage, "__init__", initialize)
    monkeypatch.setattr(
        KeyringSecretStorage,
        "get_secret",
        lambda storage, key: storage._catalog_test_values.get(key),
    )
    monkeypatch.setattr(
        KeyringSecretStorage,
        "set_secret",
        lambda storage, key, value: storage._catalog_test_values.__setitem__(key, value),
    )
    store = SQLiteCatalogAdmissionReceiptMetadataStore(path)
    authority = CatalogAdmissionReceiptAuthority(store)
    return authority, path, secrets


def _issue(authority) -> CatalogAdmissionReceipt:
    authority.provision()
    return authority.issue(
        catalog_commitment_sha256=CATALOG,
        membership_commitment_sha256=MEMBERSHIP,
        accepted_at_utc=WHEN,
    )


def test_production_constructor_has_no_key_injection_and_test_type_is_distinct(
    production, tmp_path
) -> None:
    authority, _path, _secrets = production
    assert type(authority) is CatalogAdmissionReceiptAuthority
    assert "secret" not in CatalogAdmissionReceiptAuthority.__init__.__annotations__
    test = TestCatalogAdmissionReceiptAuthority(
        tmp_path / "distinct-test-authority.sqlite3",
        deterministic_seed=b"test seed".ljust(32, b"!"),
    )
    assert type(test) is TestCatalogAdmissionReceiptAuthority
    assert type(test) is not CatalogAdmissionReceiptAuthority


def test_frozen_purpose_and_catalog_custody_namespace_are_separate_from_backup() -> None:
    assert CATALOG_ADMISSION_RECEIPT_PURPOSE == "CRYPTOHUNTER_M0_12_CATALOG_ADMISSION_RECEIPT_V1"
    assert CATALOG_ADMISSION_RECEIPT_PURPOSE != BACKUP_AUTHENTICATION_PURPOSE
    assert CATALOG_RECEIPT_KEY_LIFECYCLE_PURPOSE == "CRYPTOHUNTER_M0_12_CATALOG_RECEIPT_KEY_LIFECYCLE_V1"
    assert CATALOG_RECEIPT_AUTHORITY_ANCHOR_PURPOSE == "CRYPTOHUNTER_M0_12_CATALOG_RECEIPT_AUTHORITY_ANCHOR_V1"
    assert SecretStorageCatalogAdmissionReceiptSecureCustody._PREFIX == "dudzian.catalog-admission-receipt.v1:"


def test_production_custody_rejects_caller_controlled_secret_storage() -> None:
    with pytest.raises(TypeError):
        SecretStorageCatalogAdmissionReceiptSecureCustody(ExfiltratingSecretStorage())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("authority_domain", "cryptohunter.backup.v1"),
        ("purpose", BACKUP_AUTHENTICATION_PURPOSE),
        ("algorithm", "HMAC-SHA-384"),
        ("key_id", "bak_wrong"),
        ("accepted_at_utc", "2030-01-02T03:04:06Z"),
        ("catalog_commitment_sha256", "c" * 64),
        ("membership_commitment_sha256", "d" * 64),
    ],
)
def test_purpose_domain_algorithm_key_and_payload_confusion_fail_closed(
    production, field, value
) -> None:
    authority, _path, _secrets = production
    receipt = _issue(authority)
    assert authority.verify(receipt)
    assert not authority.verify(replace(receipt, **{field: value}))


def test_backup_mac_cannot_verify_as_catalog_receipt(production) -> None:
    authority, _path, _secrets = production
    receipt = _issue(authority)
    mapping = receipt.to_mapping()
    backup_mac = hmac.digest(
        b"backup-only-key".ljust(32, b"!"),
        canonical_catalog_admission_receipt_payload(mapping),
        "sha256",
    ).hex()
    assert not authority.verify(replace(receipt, receipt_mac=backup_mac))


def test_restart_preserves_receipt_key_sequence_and_mac(production) -> None:
    authority, path, _secrets = production
    receipt = _issue(authority)
    restarted = CatalogAdmissionReceiptAuthority(
        SQLiteCatalogAdmissionReceiptMetadataStore(path),
    )
    assert restarted.receipts() == (receipt,)
    assert restarted.verify(receipt)
    assert restarted.receipts()[0].key_id == receipt.key_id
    assert restarted.receipts()[0].receipt_mac == receipt.receipt_mac


def test_secret_bytes_are_not_stored_in_catalog_sqlite(production) -> None:
    authority, path, secrets = production
    _issue(authority)
    for encoded_secret in secrets.values():
        assert encoded_secret.encode() not in path.read_bytes()
    with sqlite3.connect(path) as db:
        all_sql = " ".join(row[0] or "" for row in db.execute("SELECT sql FROM sqlite_master"))
    assert "secret" not in all_sql.lower()


def test_public_sha_reseal_cannot_forge_receipt(production) -> None:
    authority, _path, _secrets = production
    receipt = _issue(authority)
    forged = replace(
        receipt,
        catalog_commitment_sha256="e" * 64,
        membership_commitment_sha256="f" * 64,
    )
    public_reseal = hashlib.sha256(
        canonical_catalog_admission_receipt_payload(forged.to_mapping())
    ).hexdigest()
    assert not authority.verify(replace(forged, receipt_mac=public_reseal))


def test_direct_sql_mint_with_valid_public_chain_fails_closed_on_restart(production) -> None:
    authority, path, _secrets = production
    first = _issue(authority)
    with sqlite3.connect(path) as db:
        previous = db.execute(
            "SELECT committed_digest FROM catalog_receipt_authority_head WHERE singleton=1"
        ).fetchone()[0]
        forged = CatalogAdmissionReceipt(
            authority_domain=CATALOG_ADMISSION_RECEIPT_DOMAIN,
            receipt_id="car_00000000000000000002",
            receipt_sequence=2,
            previous_receipt_digest=previous,
            catalog_commitment_sha256="c" * 64,
            membership_commitment_sha256="d" * 64,
            accepted_at_utc="2030-01-02T03:04:06Z",
            key_id=first.key_id,
            algorithm=CATALOG_ADMISSION_RECEIPT_ALGORITHM,
            purpose=CATALOG_ADMISSION_RECEIPT_PURPOSE,
            receipt_mac="0" * 64,
        )
        canonical = json.dumps(
            forged.to_mapping(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        digest = authority._record_digest(2, canonical, previous)
        db.execute(
            "INSERT INTO catalog_admission_receipts VALUES(?,?,?,?,?,?,?,?)",
            (2, forged.receipt_id, forged.key_id, forged.purpose, canonical, forged.receipt_mac, previous, digest),
        )
        db.execute(
            "UPDATE catalog_receipt_authority_head SET committed_sequence=2,committed_digest=?,last_receipt_id=? WHERE singleton=1",
            (digest, forged.receipt_id),
        )
        db.commit()
    with pytest.raises(CatalogAdmissionReceiptAuthorityUnavailable):
        CatalogAdmissionReceiptAuthority(
            SQLiteCatalogAdmissionReceiptMetadataStore(path),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"catalog_commitment_sha256": True, "membership_commitment_sha256": MEMBERSHIP, "accepted_at_utc": WHEN},
        {"catalog_commitment_sha256": CATALOG.upper(), "membership_commitment_sha256": MEMBERSHIP, "accepted_at_utc": WHEN},
        {"catalog_commitment_sha256": CATALOG, "membership_commitment_sha256": 1, "accepted_at_utc": WHEN},
        {"catalog_commitment_sha256": CATALOG, "membership_commitment_sha256": MEMBERSHIP, "accepted_at_utc": "2030-01-02T03:04:05+00:00"},
    ],
)
def test_issue_rejects_non_exact_commitments_and_noncanonical_time(production, kwargs) -> None:
    authority, _path, _secrets = production
    authority.provision()
    with pytest.raises(ValueError):
        authority.issue(**kwargs)


def test_rotation_keeps_old_receipt_verifiable_and_revocation_fails_closed(production) -> None:
    authority, _path, _secrets = production
    receipt = _issue(authority)
    new_key = authority.rotate()
    assert new_key != receipt.key_id
    assert authority.verify(receipt)
    authority.revoke(receipt.key_id)
    assert not authority.verify(receipt)


def test_revoked_key_cannot_be_resurrected_by_direct_sql(production) -> None:
    authority, path, _secrets = production
    receipt = _issue(authority)
    authority.rotate()
    authority.revoke(receipt.key_id)
    assert not authority.verify(receipt)
    with sqlite3.connect(path) as db:
        db.execute(
            "UPDATE catalog_receipt_authority_keys SET lifecycle_state='VERIFY_ONLY' WHERE key_id=?",
            (receipt.key_id,),
        )
        db.commit()
    with pytest.raises(CatalogAdmissionReceiptAuthorityUnavailable):
        CatalogAdmissionReceiptAuthority(SQLiteCatalogAdmissionReceiptMetadataStore(path))


def test_old_genuine_lifecycle_state_and_mac_rollback_fails_against_anchor(production) -> None:
    authority, path, _secrets = production
    receipt = _issue(authority)
    authority.rotate()
    with sqlite3.connect(path) as db:
        lifecycle_two_keys = db.execute(
            "SELECT * FROM catalog_receipt_authority_keys ORDER BY key_id"
        ).fetchall()
        lifecycle_two_metadata = db.execute(
            "SELECT * FROM catalog_receipt_authority_metadata WHERE singleton=1"
        ).fetchone()
    authority.revoke(receipt.key_id)
    assert not authority.verify(receipt)
    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM catalog_receipt_authority_keys")
        db.executemany(
            "INSERT INTO catalog_receipt_authority_keys VALUES(?,?,?)",
            lifecycle_two_keys,
        )
        db.execute("DELETE FROM catalog_receipt_authority_metadata")
        db.execute(
            "INSERT INTO catalog_receipt_authority_metadata VALUES(?,?,?,?,?,?,?)",
            lifecycle_two_metadata,
        )
        db.commit()
    with pytest.raises(CatalogAdmissionReceiptAuthorityUnavailable):
        CatalogAdmissionReceiptAuthority(SQLiteCatalogAdmissionReceiptMetadataStore(path))


def test_valid_genuine_receipt_prefix_and_old_head_rollback_fails_against_anchor(
    production,
) -> None:
    authority, path, _secrets = production
    first = _issue(authority)
    second = authority.issue(
        catalog_commitment_sha256="c" * 64,
        membership_commitment_sha256="d" * 64,
        accepted_at_utc="2030-01-02T03:04:06Z",
    )
    assert authority.verify(second)
    with sqlite3.connect(path) as db:
        first_digest = db.execute(
            "SELECT record_digest FROM catalog_admission_receipts WHERE receipt_sequence=1"
        ).fetchone()[0]
        db.execute("DELETE FROM catalog_admission_receipts WHERE receipt_sequence=2")
        db.execute(
            "UPDATE catalog_receipt_authority_head SET committed_sequence=1,"
            "committed_digest=?,last_receipt_id=? WHERE singleton=1",
            (first_digest, first.receipt_id),
        )
        db.commit()
    with pytest.raises(CatalogAdmissionReceiptAuthorityUnavailable):
        CatalogAdmissionReceiptAuthority(SQLiteCatalogAdmissionReceiptMetadataStore(path))


def test_full_sqlite_reset_cannot_bootstrap_while_external_anchor_remains(production) -> None:
    authority, path, secrets = production
    _issue(authority)
    assert any(key.endswith("authority-anchor-v1") for key in secrets)
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA foreign_keys=OFF")
        db.execute("DELETE FROM catalog_admission_receipts")
        db.execute("DELETE FROM catalog_receipt_authority_keys")
        db.execute("DELETE FROM catalog_receipt_authority_metadata")
        db.execute("DELETE FROM catalog_receipt_authority_head")
        db.commit()
    rebuilt = SQLiteCatalogAdmissionReceiptMetadataStore(path)
    with pytest.raises(CatalogAdmissionReceiptAuthorityUnavailable):
        CatalogAdmissionReceiptAuthority(rebuilt)


def test_external_anchor_tracks_lifecycle_and_receipt_head(production) -> None:
    authority, path, secrets = production
    receipt = _issue(authority)
    anchor_key = next(key for key in secrets if key.endswith("authority-anchor-v1"))
    first_anchor = json.loads(secrets[anchor_key])
    assert first_anchor["enrolled"] is True
    assert first_anchor["key_revision"] == 1
    assert first_anchor["receipt_committed_sequence"] == 1
    assert first_anchor["last_receipt_id"] == receipt.receipt_id
    authority.rotate()
    rotated_anchor = json.loads(secrets[anchor_key])
    assert rotated_anchor["key_revision"] == 2
    authority.revoke(receipt.key_id)
    revoked_anchor = json.loads(secrets[anchor_key])
    assert revoked_anchor["key_revision"] == 3
    with sqlite3.connect(path) as db:
        head = db.execute("SELECT * FROM catalog_receipt_authority_head").fetchone()
        lifecycle_mac = db.execute(
            "SELECT lifecycle_state_mac FROM catalog_receipt_authority_metadata"
        ).fetchone()[0]
    assert revoked_anchor["receipt_committed_sequence"] == head[1]
    assert revoked_anchor["receipt_committed_digest"] == head[2]
    assert revoked_anchor["last_receipt_id"] == head[3]
    assert revoked_anchor["lifecycle_state_mac"] == lifecycle_mac


@pytest.mark.parametrize("attack", ["missing", "malformed", "wrong_domain", "wrong_head"])
def test_enrolled_external_anchor_tampering_or_unavailability_fails_closed(
    production, attack
) -> None:
    authority, path, secrets = production
    _issue(authority)
    anchor_key = next(key for key in secrets if key.endswith("authority-anchor-v1"))
    if attack == "missing":
        del secrets[anchor_key]
    elif attack == "malformed":
        secrets[anchor_key] = "not-json"
    else:
        anchor = json.loads(secrets[anchor_key])
        if attack == "wrong_domain":
            anchor["authority_domain"] = "attacker"
        else:
            anchor["receipt_committed_sequence"] = 0
        secrets[anchor_key] = json.dumps(anchor, sort_keys=True, separators=(",", ":"))
    with pytest.raises(CatalogAdmissionReceiptAuthorityUnavailable):
        CatalogAdmissionReceiptAuthority(SQLiteCatalogAdmissionReceiptMetadataStore(path))


@pytest.mark.parametrize(
    "attack",
    [
        "revoked_to_verify_only",
        "revoked_to_active",
        "verify_only_to_active",
        "active_to_revoked",
        "custody_handle",
        "revision_increase",
        "revision_rollback",
        "lifecycle_mac",
        "root_handle",
    ],
)
def test_all_lifecycle_shadow_tampering_fails_closed_on_restart(production, attack) -> None:
    authority, path, _secrets = production
    first = _issue(authority)
    second_key = authority.rotate()
    if attack in {"revoked_to_verify_only", "revoked_to_active"}:
        authority.revoke(first.key_id)
    with sqlite3.connect(path) as db:
        if attack == "revoked_to_verify_only":
            db.execute("UPDATE catalog_receipt_authority_keys SET lifecycle_state='VERIFY_ONLY' WHERE key_id=?", (first.key_id,))
        elif attack == "revoked_to_active":
            db.execute("UPDATE catalog_receipt_authority_keys SET lifecycle_state='VERIFY_ONLY' WHERE key_id=?", (second_key,))
            db.execute("UPDATE catalog_receipt_authority_keys SET lifecycle_state='ACTIVE' WHERE key_id=?", (first.key_id,))
        elif attack == "verify_only_to_active":
            db.execute("UPDATE catalog_receipt_authority_keys SET lifecycle_state='VERIFY_ONLY' WHERE key_id=?", (second_key,))
            db.execute("UPDATE catalog_receipt_authority_keys SET lifecycle_state='ACTIVE' WHERE key_id=?", (first.key_id,))
        elif attack == "active_to_revoked":
            db.execute("UPDATE catalog_receipt_authority_keys SET lifecycle_state='REVOKED' WHERE key_id=?", (second_key,))
        elif attack == "custody_handle":
            db.execute("UPDATE catalog_receipt_authority_keys SET custody_handle='catalog-material-attacker' WHERE key_id=?", (first.key_id,))
        elif attack == "revision_increase":
            db.execute("UPDATE catalog_receipt_authority_metadata SET key_revision=key_revision+1")
        elif attack == "revision_rollback":
            db.execute("UPDATE catalog_receipt_authority_metadata SET key_revision=1")
        elif attack == "lifecycle_mac":
            db.execute("UPDATE catalog_receipt_authority_metadata SET lifecycle_state_mac=?", ("0" * 64,))
        else:
            db.execute("UPDATE catalog_receipt_authority_metadata SET lifecycle_root_handle='catalog-lifecycle-root-attacker'")
        db.commit()
    with pytest.raises(CatalogAdmissionReceiptAuthorityUnavailable):
        CatalogAdmissionReceiptAuthority(SQLiteCatalogAdmissionReceiptMetadataStore(path))


@pytest.mark.parametrize("operation", ["rotate", "revoke"])
def test_corrupt_receipt_journal_cannot_receive_lifecycle_ack(production, operation) -> None:
    authority, path, _secrets = production
    receipt = _issue(authority)
    with sqlite3.connect(path) as db:
        before_metadata = db.execute("SELECT * FROM catalog_receipt_authority_metadata").fetchall()
        before_keys = db.execute("SELECT * FROM catalog_receipt_authority_keys ORDER BY key_id").fetchall()
        db.execute("UPDATE catalog_admission_receipts SET receipt_mac=?", ("0" * 64,))
        db.commit()
    with pytest.raises(CatalogAdmissionReceiptAuthorityUnavailable):
        authority.rotate() if operation == "rotate" else authority.revoke(receipt.key_id)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT * FROM catalog_receipt_authority_metadata").fetchall() == before_metadata
        assert db.execute("SELECT * FROM catalog_receipt_authority_keys ORDER BY key_id").fetchall() == before_keys


def test_test_authority_is_deterministic_and_restartable(tmp_path) -> None:
    path = tmp_path / "test-receipts.sqlite3"
    seed = b"deterministic-test-seed".ljust(32, b"!")
    authority = TestCatalogAdmissionReceiptAuthority(path, deterministic_seed=seed)
    receipt = _issue(authority)
    restarted = TestCatalogAdmissionReceiptAuthority(path, deterministic_seed=seed)
    assert restarted.verify(receipt)
    assert restarted.receipts() == (receipt,)
