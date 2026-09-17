"""Purpose-separated authentication authority for Catalog admission receipts.

This module is intentionally not integrated with Catalog snapshot persistence.  It owns
only a key lifecycle and an authenticated, append-only receipt journal.  Public hashes
are integrity aids; only a MAC produced by opaque Core custody proves receipt admission.
SQLite rollback without a matching external keyring-anchor rollback fails closed.  A
coordinated rollback of both SQLite and the historical external anchor remains outside
the locally detectable boundary without a TPM or remote monotonic anchor.
"""
from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from enum import Enum
import hashlib
import hmac
import json
from pathlib import Path
import re
import secrets
import sqlite3
from typing import Protocol
from uuid import uuid4

from bot_core.security.keyring_storage import KeyringSecretStorage


CATALOG_ADMISSION_RECEIPT_PURPOSE = "CRYPTOHUNTER_M0_12_CATALOG_ADMISSION_RECEIPT_V1"
CATALOG_RECEIPT_KEY_LIFECYCLE_PURPOSE = "CRYPTOHUNTER_M0_12_CATALOG_RECEIPT_KEY_LIFECYCLE_V1"
CATALOG_RECEIPT_AUTHORITY_ANCHOR_PURPOSE = "CRYPTOHUNTER_M0_12_CATALOG_RECEIPT_AUTHORITY_ANCHOR_V1"
CATALOG_ADMISSION_RECEIPT_ALGORITHM = "HMAC-SHA-256"
CATALOG_ADMISSION_RECEIPT_DOMAIN = "cryptohunter.catalog-admission-receipt.production.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z\Z")


class CatalogAdmissionReceiptError(RuntimeError):
    """Fail-closed receipt authority error."""


class CatalogAdmissionReceiptAuthorityUnavailable(CatalogAdmissionReceiptError):
    pass


class CatalogAdmissionReceiptNotProvisioned(CatalogAdmissionReceiptError):
    pass


class CatalogAdmissionReceiptAlreadyProvisioned(CatalogAdmissionReceiptError):
    pass


class CatalogAdmissionReceiptKeyState(str, Enum):
    ACTIVE = "ACTIVE"
    VERIFY_ONLY = "VERIFY_ONLY"
    REVOKED = "REVOKED"


def _canonical(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _valid_sha(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _valid_utc(value: object) -> bool:
    if type(value) is not str or _UTC.fullmatch(value) is None:
        return False
    try:
        parsed = datetime.strptime(
            value,
            "%Y-%m-%dT%H:%M:%SZ" if "." not in value else "%Y-%m-%dT%H:%M:%S.%fZ",
        ).replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    return parsed.isoformat().endswith("+00:00")


@dataclass(frozen=True, slots=True)
class CatalogAdmissionReceipt:
    authority_domain: str
    receipt_id: str
    receipt_sequence: int
    previous_receipt_digest: str
    catalog_commitment_sha256: str
    membership_commitment_sha256: str
    accepted_at_utc: str
    key_id: str
    algorithm: str
    purpose: str
    receipt_mac: str

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


_RECEIPT_FIELDS = frozenset(field.name for field in fields(CatalogAdmissionReceipt))


def _receipt_payload(value: Mapping[str, object]) -> dict[str, object]:
    return {name: value[name] for name in sorted(_RECEIPT_FIELDS - {"receipt_mac"})}


def _valid_receipt_shape(value: object, *, domain: str) -> bool:
    try:
        return bool(
            isinstance(value, Mapping)
            and set(value) == _RECEIPT_FIELDS
            and value.get("authority_domain") == domain
            and type(value.get("receipt_id")) is str
            and value["receipt_id"].startswith("car_")
            and type(value.get("receipt_sequence")) is int
            and value["receipt_sequence"] > 0
            and _valid_sha(value.get("previous_receipt_digest"))
            and _valid_sha(value.get("catalog_commitment_sha256"))
            and _valid_sha(value.get("membership_commitment_sha256"))
            and _valid_utc(value.get("accepted_at_utc"))
            and type(value.get("key_id")) is str
            and value["key_id"].startswith("cark_")
            and value.get("algorithm") == CATALOG_ADMISSION_RECEIPT_ALGORITHM
            and value.get("purpose") == CATALOG_ADMISSION_RECEIPT_PURPOSE
            and _valid_sha(value.get("receipt_mac"))
        )
    except (KeyError, TypeError, ValueError):
        return False


def canonical_catalog_admission_receipt_payload(value: Mapping[str, object]) -> bytes:
    """Return the exact purpose/domain-separated bytes authenticated by custody."""

    return (
        CATALOG_ADMISSION_RECEIPT_PURPOSE.encode("ascii")
        + b"\x00"
        + str(value["authority_domain"]).encode("utf-8")
        + b"\x00"
        + _canonical(_receipt_payload(value))
    )


class CatalogAdmissionReceiptSecureCustody(Protocol):
    """Opaque custody: key bytes can be persisted and used, never retrieved."""

    def persist(self, custody_handle: str, key_material: bytes) -> None: ...
    def digest(self, custody_handle: str, payload: bytes) -> bytes | None: ...
    def read_anchor(self) -> str | None: ...
    def write_anchor(self, canonical_anchor: str) -> None: ...


class SecretStorageCatalogAdmissionReceiptSecureCustody:
    """Catalog-only custody backed by the exact production OS keyring implementation."""

    _PREFIX = "dudzian.catalog-admission-receipt.v1:"
    _ANCHOR_SLOT = "authority-anchor-v1"

    def __init__(self) -> None:
        self._storage = KeyringSecretStorage(service_name="dudzian.catalog-admission-receipt")

    def persist(self, custody_handle: str, key_material: bytes) -> None:
        if type(key_material) is not bytes or len(key_material) != 32:
            raise CatalogAdmissionReceiptAuthorityUnavailable
        storage_key = self._PREFIX + custody_handle
        try:
            if self._storage.get_secret(storage_key) is not None:
                raise CatalogAdmissionReceiptAuthorityUnavailable
            self._storage.set_secret(storage_key, base64.b64encode(key_material).decode("ascii"))
        except CatalogAdmissionReceiptError:
            raise
        except Exception:
            raise CatalogAdmissionReceiptAuthorityUnavailable from None

    def digest(self, custody_handle: str, payload: bytes) -> bytes | None:
        try:
            encoded = self._storage.get_secret(self._PREFIX + custody_handle)
            if encoded is None:
                return None
            key = base64.b64decode(encoded, validate=True)
        except Exception:
            return None
        if len(key) != 32:
            return None
        return hmac.digest(key, payload, "sha256")

    def read_anchor(self) -> str | None:
        try:
            return self._storage.get_secret(self._PREFIX + self._ANCHOR_SLOT)
        except Exception:
            raise CatalogAdmissionReceiptAuthorityUnavailable from None

    def write_anchor(self, canonical_anchor: str) -> None:
        if type(canonical_anchor) is not str or not canonical_anchor:
            raise CatalogAdmissionReceiptAuthorityUnavailable
        try:
            self._storage.set_secret(self._PREFIX + self._ANCHOR_SLOT, canonical_anchor)
        except Exception:
            raise CatalogAdmissionReceiptAuthorityUnavailable from None


@dataclass(frozen=True, slots=True)
class _StoredCatalogReceiptKey:
    key_id: str
    lifecycle_state: CatalogAdmissionReceiptKeyState
    custody_handle: str


class _SQLiteCatalogAdmissionReceiptMetadataStoreBase:
    _AUTHORITY_DOMAIN = CATALOG_ADMISSION_RECEIPT_DOMAIN

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        return db

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS catalog_receipt_authority_metadata(
                  singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                  authority_domain TEXT NOT NULL, purpose TEXT NOT NULL,
                  algorithm TEXT NOT NULL, key_revision INTEGER NOT NULL CHECK(key_revision >= 0),
                  lifecycle_root_handle TEXT, lifecycle_state_mac TEXT);
                CREATE TABLE IF NOT EXISTS catalog_receipt_authority_keys(
                  key_id TEXT PRIMARY KEY, lifecycle_state TEXT NOT NULL
                    CHECK(lifecycle_state IN ('ACTIVE','VERIFY_ONLY','REVOKED')),
                  custody_handle TEXT NOT NULL UNIQUE);
                CREATE UNIQUE INDEX IF NOT EXISTS catalog_receipt_one_active_key
                  ON catalog_receipt_authority_keys(lifecycle_state) WHERE lifecycle_state='ACTIVE';
                CREATE TABLE IF NOT EXISTS catalog_admission_receipts(
                  receipt_sequence INTEGER PRIMARY KEY, receipt_id TEXT UNIQUE NOT NULL,
                  key_id TEXT NOT NULL, purpose TEXT NOT NULL, canonical_receipt TEXT NOT NULL,
                  receipt_mac TEXT NOT NULL, previous_receipt_digest TEXT NOT NULL,
                  record_digest TEXT UNIQUE NOT NULL,
                  FOREIGN KEY(key_id) REFERENCES catalog_receipt_authority_keys(key_id));
                CREATE TABLE IF NOT EXISTS catalog_receipt_authority_head(
                  singleton INTEGER PRIMARY KEY CHECK(singleton=1), committed_sequence INTEGER NOT NULL,
                  committed_digest TEXT NOT NULL, last_receipt_id TEXT);
            """)
            row = db.execute("SELECT COUNT(*) FROM catalog_receipt_authority_metadata").fetchone()
            if row == (0,):
                db.execute(
                    "INSERT INTO catalog_receipt_authority_metadata VALUES(1,?,?,?,0,NULL,NULL)",
                    (self._AUTHORITY_DOMAIN, CATALOG_ADMISSION_RECEIPT_PURPOSE, CATALOG_ADMISSION_RECEIPT_ALGORITHM),
                )
            row = db.execute("SELECT COUNT(*) FROM catalog_receipt_authority_head").fetchone()
            if row == (0,):
                db.execute("INSERT INTO catalog_receipt_authority_head VALUES(1,0,?,NULL)", ("0" * 64,))


class SQLiteCatalogAdmissionReceiptMetadataStore(_SQLiteCatalogAdmissionReceiptMetadataStoreBase):
    """Core-owned production metadata; it never contains secret key bytes."""


class _CatalogAdmissionReceiptAuthorityBase:
    _AUTHORITY_DOMAIN = CATALOG_ADMISSION_RECEIPT_DOMAIN

    def __init__(
        self,
        metadata: _SQLiteCatalogAdmissionReceiptMetadataStoreBase,
        custody: CatalogAdmissionReceiptSecureCustody,
    ) -> None:
        self._metadata = metadata
        self._custody = custody
        with metadata._connect() as db:
            self._replay(db)

    def _anchor_mapping(
        self,
        db: sqlite3.Connection,
        revision: int,
        root_handle: str | None,
    ) -> dict[str, object] | None:
        metadata = db.execute(
            "SELECT lifecycle_state_mac FROM catalog_receipt_authority_metadata WHERE singleton=1"
        ).fetchone()
        head = db.execute(
            "SELECT committed_sequence,committed_digest,last_receipt_id "
            "FROM catalog_receipt_authority_head WHERE singleton=1"
        ).fetchone()
        if metadata is None or head is None:
            raise CatalogAdmissionReceiptAuthorityUnavailable
        lifecycle_mac = metadata[0]
        sequence, digest, last_receipt_id = head
        if revision == 0 and root_handle is None and sequence == 0:
            if lifecycle_mac is not None or digest != "0" * 64 or last_receipt_id is not None:
                raise CatalogAdmissionReceiptAuthorityUnavailable
            return None
        if (
            root_handle is None
            or not _valid_sha(lifecycle_mac)
            or type(sequence) is not int
            or sequence < 0
            or not _valid_sha(digest)
            or (last_receipt_id is not None and type(last_receipt_id) is not str)
        ):
            raise CatalogAdmissionReceiptAuthorityUnavailable
        return {
            "anchor_schema_version": 1,
            "anchor_purpose": CATALOG_RECEIPT_AUTHORITY_ANCHOR_PURPOSE,
            "authority_domain": self._AUTHORITY_DOMAIN,
            "receipt_purpose": CATALOG_ADMISSION_RECEIPT_PURPOSE,
            "algorithm": CATALOG_ADMISSION_RECEIPT_ALGORITHM,
            "enrolled": True,
            "key_revision": revision,
            "lifecycle_state_mac": lifecycle_mac,
            "receipt_committed_sequence": sequence,
            "receipt_committed_digest": digest,
            "last_receipt_id": last_receipt_id,
        }

    def _read_anchor(self) -> dict[str, object] | None:
        raw = self._custody.read_anchor()
        if raw is None:
            return None
        try:
            value = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            raise CatalogAdmissionReceiptAuthorityUnavailable from None
        if not isinstance(value, dict):
            raise CatalogAdmissionReceiptAuthorityUnavailable
        return value

    def _validate_anchor(
        self, db: sqlite3.Connection, revision: int, root_handle: str | None
    ) -> None:
        expected = self._anchor_mapping(db, revision, root_handle)
        actual = self._read_anchor()
        if expected is None:
            if actual is not None:
                raise CatalogAdmissionReceiptAuthorityUnavailable
            return
        if actual is None or not hmac.compare_digest(_canonical(actual), _canonical(expected)):
            raise CatalogAdmissionReceiptAuthorityUnavailable

    def _publish_anchor(self, db: sqlite3.Connection) -> None:
        revision, _keys, root_handle = self._metadata_and_keys(db)
        value = self._anchor_mapping(db, revision, root_handle)
        if value is None:
            raise CatalogAdmissionReceiptAuthorityUnavailable
        self._custody.write_anchor(_canonical(value).decode("utf-8"))

    def _record_digest(self, sequence: int, canonical: str, previous: str) -> str:
        return hashlib.sha256(
            (
                f"{CATALOG_ADMISSION_RECEIPT_PURPOSE}\n{self._AUTHORITY_DOMAIN}\n"
                f"{sequence}\n{previous}\n{canonical}"
            ).encode()
        ).hexdigest()

    def _lifecycle_payload(
        self, revision: int, keys: tuple[_StoredCatalogReceiptKey, ...]
    ) -> bytes:
        value: dict[str, object] = {
            "authority_domain": self._AUTHORITY_DOMAIN,
            "receipt_purpose": CATALOG_ADMISSION_RECEIPT_PURPOSE,
            "algorithm": CATALOG_ADMISSION_RECEIPT_ALGORITHM,
            "key_revision": revision,
            "keys": [
                {
                    "key_id": key.key_id,
                    "lifecycle_state": key.lifecycle_state.value,
                    "custody_handle": key.custody_handle,
                }
                for key in keys
            ],
        }
        return (
            CATALOG_RECEIPT_KEY_LIFECYCLE_PURPOSE.encode("ascii")
            + b"\x00"
            + self._AUTHORITY_DOMAIN.encode("utf-8")
            + b"\x00"
            + _canonical(value)
        )

    def _metadata_and_keys(
        self, db: sqlite3.Connection
    ) -> tuple[int, tuple[_StoredCatalogReceiptKey, ...], str | None]:
        rows = db.execute(
            "SELECT authority_domain,purpose,algorithm,key_revision,lifecycle_root_handle,lifecycle_state_mac "
            "FROM catalog_receipt_authority_metadata WHERE singleton=1"
        ).fetchall()
        if len(rows) != 1:
            raise CatalogAdmissionReceiptAuthorityUnavailable
        domain, purpose, algorithm, revision, root_handle, lifecycle_mac = rows[0]
        if (
            domain != self._AUTHORITY_DOMAIN
            or purpose != CATALOG_ADMISSION_RECEIPT_PURPOSE
            or algorithm != CATALOG_ADMISSION_RECEIPT_ALGORITHM
        ):
            raise CatalogAdmissionReceiptAuthorityUnavailable
        if type(revision) is not int or revision < 0:
            raise CatalogAdmissionReceiptAuthorityUnavailable
        try:
            keys = tuple(
                _StoredCatalogReceiptKey(key_id, CatalogAdmissionReceiptKeyState(state), handle)
                for key_id, state, handle in db.execute(
                    "SELECT key_id,lifecycle_state,custody_handle FROM catalog_receipt_authority_keys ORDER BY key_id"
                )
            )
        except (TypeError, ValueError):
            raise CatalogAdmissionReceiptAuthorityUnavailable from None
        if revision < len(keys) or sum(key.lifecycle_state is CatalogAdmissionReceiptKeyState.ACTIVE for key in keys) > 1:
            raise CatalogAdmissionReceiptAuthorityUnavailable
        if any(not key.key_id.startswith("cark_") or not key.custody_handle for key in keys):
            raise CatalogAdmissionReceiptAuthorityUnavailable
        if revision == 0 and not keys:
            if root_handle is not None or lifecycle_mac is not None:
                raise CatalogAdmissionReceiptAuthorityUnavailable
            return revision, keys, None
        if (
            type(root_handle) is not str
            or not root_handle.startswith("catalog-lifecycle-root-")
            or not _valid_sha(lifecycle_mac)
        ):
            raise CatalogAdmissionReceiptAuthorityUnavailable
        expected = self._custody.digest(root_handle, self._lifecycle_payload(revision, keys))
        if expected is None or not hmac.compare_digest(expected.hex(), lifecycle_mac):
            raise CatalogAdmissionReceiptAuthorityUnavailable
        return revision, keys, root_handle

    def _seal_lifecycle(
        self,
        db: sqlite3.Connection,
        revision: int,
        keys: tuple[_StoredCatalogReceiptKey, ...],
        root_handle: str,
    ) -> None:
        tag = self._custody.digest(root_handle, self._lifecycle_payload(revision, keys))
        if tag is None:
            raise CatalogAdmissionReceiptAuthorityUnavailable
        changed = db.execute(
            "UPDATE catalog_receipt_authority_metadata SET key_revision=?,"
            "lifecycle_root_handle=?,lifecycle_state_mac=? WHERE singleton=1",
            (revision, root_handle, tag.hex()),
        ).rowcount
        if changed != 1:
            raise CatalogAdmissionReceiptAuthorityUnavailable

    def _replay(
        self, db: sqlite3.Connection, *, check_anchor: bool = True
    ) -> tuple[CatalogAdmissionReceipt, ...]:
        _revision, keys, _root_handle = self._metadata_and_keys(db)
        by_id = {key.key_id: key for key in keys}
        head = db.execute(
            "SELECT committed_sequence,committed_digest,last_receipt_id FROM catalog_receipt_authority_head WHERE singleton=1"
        ).fetchone()
        rows = db.execute(
            "SELECT receipt_sequence,receipt_id,key_id,purpose,canonical_receipt,receipt_mac,"
            "previous_receipt_digest,record_digest FROM catalog_admission_receipts ORDER BY receipt_sequence"
        ).fetchall()
        if head is None or head[0] != len(rows):
            raise CatalogAdmissionReceiptAuthorityUnavailable
        previous = "0" * 64
        receipts: list[CatalogAdmissionReceipt] = []
        for expected_sequence, row in enumerate(rows, 1):
            sequence, receipt_id, key_id, purpose, canonical, mac, row_previous, digest = row
            try:
                raw = json.loads(canonical)
            except (TypeError, json.JSONDecodeError):
                raise CatalogAdmissionReceiptAuthorityUnavailable from None
            key = by_id.get(key_id)
            if (
                sequence != expected_sequence
                or row_previous != previous
                or digest != self._record_digest(sequence, canonical, previous)
                or not _valid_receipt_shape(raw, domain=self._AUTHORITY_DOMAIN)
                or raw["receipt_sequence"] != sequence
                or raw["receipt_id"] != receipt_id
                or raw["key_id"] != key_id
                or raw["purpose"] != purpose
                or raw["receipt_mac"] != mac
                or raw["previous_receipt_digest"] != previous
                or key is None
            ):
                raise CatalogAdmissionReceiptAuthorityUnavailable
            expected = self._custody.digest(
                key.custody_handle, canonical_catalog_admission_receipt_payload(raw)
            )
            if expected is None or not hmac.compare_digest(expected.hex(), mac):
                raise CatalogAdmissionReceiptAuthorityUnavailable
            receipts.append(CatalogAdmissionReceipt(**raw))
            previous = digest
        expected_last = receipts[-1].receipt_id if receipts else None
        if head[1:] != (previous, expected_last):
            raise CatalogAdmissionReceiptAuthorityUnavailable
        if check_anchor:
            self._validate_anchor(db, _revision, _root_handle)
        return tuple(receipts)

    def _new_key(self) -> tuple[_StoredCatalogReceiptKey, bytes]:
        return (
            _StoredCatalogReceiptKey(
                "cark_" + uuid4().hex,
                CatalogAdmissionReceiptKeyState.ACTIVE,
                "catalog-material-" + uuid4().hex,
            ),
            self._key_material(),
        )

    def _key_material(self) -> bytes:
        return secrets.token_bytes(32)

    def provision(self) -> str:
        with self._metadata._connect() as db:
            self._replay(db)
            revision, keys, _root_handle = self._metadata_and_keys(db)
        if revision != 0 or keys:
            raise CatalogAdmissionReceiptAlreadyProvisioned
        key, material = self._new_key()
        root_handle = "catalog-lifecycle-root-" + uuid4().hex
        root_material = secrets.token_bytes(32)
        self._custody.persist(key.custody_handle, material)
        self._custody.persist(root_handle, root_material)
        with self._metadata._connect() as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                self._replay(db)
                revision, keys, _current_root = self._metadata_and_keys(db)
                if revision != 0 or keys:
                    raise CatalogAdmissionReceiptAlreadyProvisioned
                db.execute(
                    "INSERT INTO catalog_receipt_authority_keys VALUES(?,?,?)",
                    (key.key_id, key.lifecycle_state.value, key.custody_handle),
                )
                updated = (key,)
                self._seal_lifecycle(db, 1, updated, root_handle)
                self._replay(db, check_anchor=False)
                db.commit()
            except Exception:
                db.rollback()
                raise
        with self._metadata._connect() as db:
            self._publish_anchor(db)
            self._replay(db)
        return key.key_id

    def rotate(self) -> str:
        with self._metadata._connect() as db:
            self._replay(db)
            _revision, keys, _root_handle = self._metadata_and_keys(db)
        if sum(
            key.lifecycle_state is CatalogAdmissionReceiptKeyState.ACTIVE for key in keys
        ) != 1:
            raise CatalogAdmissionReceiptNotProvisioned
        key, material = self._new_key()
        self._custody.persist(key.custody_handle, material)
        with self._metadata._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._replay(db)
            revision, keys, root_handle = self._metadata_and_keys(db)
            active = [item for item in keys if item.lifecycle_state is CatalogAdmissionReceiptKeyState.ACTIVE]
            if len(active) != 1:
                db.rollback()
                raise CatalogAdmissionReceiptNotProvisioned
            if root_handle is None:
                raise CatalogAdmissionReceiptAuthorityUnavailable
            db.execute("UPDATE catalog_receipt_authority_keys SET lifecycle_state='VERIFY_ONLY' WHERE key_id=?", (active[0].key_id,))
            db.execute("INSERT INTO catalog_receipt_authority_keys VALUES(?,?,?)", (key.key_id, key.lifecycle_state.value, key.custody_handle))
            updated = tuple(
                _StoredCatalogReceiptKey(
                    item.key_id,
                    CatalogAdmissionReceiptKeyState.VERIFY_ONLY
                    if item.key_id == active[0].key_id
                    else item.lifecycle_state,
                    item.custody_handle,
                )
                for item in keys
            ) + (key,)
            updated = tuple(sorted(updated, key=lambda item: item.key_id))
            self._seal_lifecycle(db, revision + 1, updated, root_handle)
            self._replay(db, check_anchor=False)
            db.commit()
        with self._metadata._connect() as db:
            self._publish_anchor(db)
            self._replay(db)
        return key.key_id

    def revoke(self, key_id: object) -> None:
        if type(key_id) is not str:
            raise CatalogAdmissionReceiptAuthorityUnavailable
        with self._metadata._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._replay(db)
            revision, keys, root_handle = self._metadata_and_keys(db)
            if root_handle is None:
                raise CatalogAdmissionReceiptAuthorityUnavailable
            changed = db.execute(
                "UPDATE catalog_receipt_authority_keys SET lifecycle_state='REVOKED' "
                "WHERE key_id=? AND lifecycle_state!='REVOKED'",
                (key_id,),
            ).rowcount
            if changed != 1:
                db.rollback()
                raise CatalogAdmissionReceiptAuthorityUnavailable
            updated = tuple(
                _StoredCatalogReceiptKey(
                    item.key_id,
                    CatalogAdmissionReceiptKeyState.REVOKED
                    if item.key_id == key_id
                    else item.lifecycle_state,
                    item.custody_handle,
                )
                for item in keys
            )
            self._seal_lifecycle(db, revision + 1, updated, root_handle)
            self._replay(db, check_anchor=False)
            db.commit()
        with self._metadata._connect() as db:
            self._publish_anchor(db)
            self._replay(db)

    def issue(
        self,
        *,
        catalog_commitment_sha256: object,
        membership_commitment_sha256: object,
        accepted_at_utc: object,
    ) -> CatalogAdmissionReceipt:
        if not _valid_sha(catalog_commitment_sha256) or not _valid_sha(membership_commitment_sha256) or not _valid_utc(accepted_at_utc):
            raise ValueError("exact commitments and canonical UTC timestamp required")
        with self._metadata._connect() as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                current = self._replay(db)
                _revision, keys, _root_handle = self._metadata_and_keys(db)
                active = [key for key in keys if key.lifecycle_state is CatalogAdmissionReceiptKeyState.ACTIVE]
                if len(active) != 1:
                    raise CatalogAdmissionReceiptNotProvisioned
                head = db.execute("SELECT committed_sequence,committed_digest FROM catalog_receipt_authority_head WHERE singleton=1").fetchone()
                sequence, previous = head[0] + 1, head[1]
                material: dict[str, object] = {
                    "authority_domain": self._AUTHORITY_DOMAIN,
                    "receipt_id": f"car_{sequence:020d}",
                    "receipt_sequence": sequence,
                    "previous_receipt_digest": previous,
                    "catalog_commitment_sha256": catalog_commitment_sha256,
                    "membership_commitment_sha256": membership_commitment_sha256,
                    "accepted_at_utc": accepted_at_utc,
                    "key_id": active[0].key_id,
                    "algorithm": CATALOG_ADMISSION_RECEIPT_ALGORITHM,
                    "purpose": CATALOG_ADMISSION_RECEIPT_PURPOSE,
                }
                tag = self._custody.digest(active[0].custody_handle, canonical_catalog_admission_receipt_payload(material))
                if tag is None:
                    raise CatalogAdmissionReceiptAuthorityUnavailable
                receipt = CatalogAdmissionReceipt(**material, receipt_mac=tag.hex())
                canonical = _canonical(receipt.to_mapping()).decode("utf-8")
                digest = self._record_digest(sequence, canonical, previous)
                db.execute(
                    "INSERT INTO catalog_admission_receipts VALUES(?,?,?,?,?,?,?,?)",
                    (sequence, receipt.receipt_id, receipt.key_id, receipt.purpose, canonical, receipt.receipt_mac, previous, digest),
                )
                db.execute(
                    "UPDATE catalog_receipt_authority_head SET committed_sequence=?,committed_digest=?,last_receipt_id=? WHERE singleton=1",
                    (sequence, digest, receipt.receipt_id),
                )
                if len(self._replay(db, check_anchor=False)) != len(current) + 1:
                    raise CatalogAdmissionReceiptAuthorityUnavailable
                db.commit()
            except Exception:
                db.rollback()
                raise
        with self._metadata._connect() as db:
            self._publish_anchor(db)
            self._replay(db)
        return receipt

    def receipts(self) -> tuple[CatalogAdmissionReceipt, ...]:
        with self._metadata._connect() as db:
            return self._replay(db)

    def verify(self, receipt: object) -> bool:
        if type(receipt) is not CatalogAdmissionReceipt:
            return False
        try:
            with self._metadata._connect() as db:
                accepted = self._replay(db)
                _revision, keys, _root_handle = self._metadata_and_keys(db)
            matches = [item for item in accepted if item.receipt_id == receipt.receipt_id]
            key = next((item for item in keys if item.key_id == receipt.key_id), None)
            return (
                key is not None
                and key.lifecycle_state is not CatalogAdmissionReceiptKeyState.REVOKED
                and len(matches) == 1
                and hmac.compare_digest(
                    _canonical(matches[0].to_mapping()),
                    _canonical(receipt.to_mapping()),
                )
            )
        except (CatalogAdmissionReceiptError, sqlite3.Error, TypeError, ValueError):
            return False


class CatalogAdmissionReceiptAuthority(_CatalogAdmissionReceiptAuthorityBase):
    """Production trust type; caller cannot inject key bytes or a key provider."""

    def __init__(
        self,
        metadata: SQLiteCatalogAdmissionReceiptMetadataStore,
    ) -> None:
        if type(metadata) is not SQLiteCatalogAdmissionReceiptMetadataStore:
            raise TypeError("exact production Catalog receipt metadata store required")
        super().__init__(metadata, SecretStorageCatalogAdmissionReceiptSecureCustody())


__all__ = [
    "CATALOG_ADMISSION_RECEIPT_ALGORITHM",
    "CATALOG_ADMISSION_RECEIPT_DOMAIN",
    "CATALOG_ADMISSION_RECEIPT_PURPOSE",
    "CATALOG_RECEIPT_KEY_LIFECYCLE_PURPOSE",
    "CATALOG_RECEIPT_AUTHORITY_ANCHOR_PURPOSE",
    "CatalogAdmissionReceipt",
    "CatalogAdmissionReceiptAuthority",
    "CatalogAdmissionReceiptSecureCustody",
    "SecretStorageCatalogAdmissionReceiptSecureCustody",
    "SQLiteCatalogAdmissionReceiptMetadataStore",
]
