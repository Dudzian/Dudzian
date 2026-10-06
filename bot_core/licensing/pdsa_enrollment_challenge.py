"""Frozen production PDSA challenge and issuer-owned retained state.

This store belongs to the trusted off-host PDSA service. Selecting a local SQLite
path does not establish that service's authority. Production composition must
use its retained issuer store, the canonical Production Trust loader, and the
independent production pre-enrollment authentication capability before consume.
No private authority keys, legal enrollment packages or subjects are generated.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator
from weakref import WeakKeyDictionary

from cryptography.exceptions import InvalidSignature

from deployment.windows_stage9_production_trust import (
    PDSA_KEY_SET_DIGEST,
    ProductionTrustContext,
    require_current_production_trust_context,
)

from .canonical import canonical_json_bytes, parse_canonical
from .pre_enrollment import MAX_EXACT_INTEGER, PDSA_TRUST_DOMAIN, PreEnrollmentRequestV1
from .product_profile import PRODUCT_NAME, PRODUCTION_PRODUCT_PROFILE

SCHEMA_VERSION = "PDSAEnrollmentChallengeV1"
SIGNATURE_PROFILE = "Ed25519-SHA256-DIGEST-CH-STAGE9-PDSA-CHALLENGE-V1"
SIGNATURE_DOMAIN = b"CryptoHunter.Stage9.PDSAEnrollmentChallenge.v1\x00"
VALIDITY_SECONDS = 604_800
PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "environment",
        "product",
        "product_profile",
        "pdsa_trust_domain",
        "challenge_id",
        "nonce_hex",
        "release_policy_digest_sha256",
        "release_policy_generation",
        "issued_at_utc",
        "expires_at_utc",
        "pdsa_key_set_digest",
        "signature_algorithm_profile",
    }
)
SIGNATURE_FIELDS = frozenset({"key_id", "algorithm", "signature_hex"})
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX128 = re.compile(r"[0-9a-f]{128}\Z")
_KEY_ID = re.compile(r"[A-Za-z0-9._-]{1,128}\Z")
_CHALLENGE_ID = re.compile(
    r"pchal_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
)
_UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")


class PDSAChallengeError(ValueError):
    """Stable fail-closed challenge reason."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: object) -> datetime:
    if type(value) is not str or _UTC.fullmatch(value) is None:
        raise PDSAChallengeError("INVALID_CHALLENGE_TIMESTAMP")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise PDSAChallengeError("INVALID_CHALLENGE_TIMESTAMP") from exc


def _validate_payload(value: object) -> dict[str, Any]:
    if type(value) is not dict or set(value) != PAYLOAD_FIELDS:
        raise PDSAChallengeError("CHALLENGE_PAYLOAD_SCHEMA_MISMATCH")
    literals = {
        "schema_version": SCHEMA_VERSION,
        "environment": "PRODUCTION",
        "product": PRODUCT_NAME,
        "product_profile": PRODUCTION_PRODUCT_PROFILE,
        "pdsa_trust_domain": PDSA_TRUST_DOMAIN,
        "pdsa_key_set_digest": PDSA_KEY_SET_DIGEST,
        "signature_algorithm_profile": SIGNATURE_PROFILE,
    }
    for field, expected in literals.items():
        if type(value[field]) is not str or value[field] != expected:
            raise PDSAChallengeError(f"INVALID_CHALLENGE_{field.upper()}")
    for field in ("nonce_hex", "release_policy_digest_sha256"):
        if type(value[field]) is not str or _HEX64.fullmatch(value[field]) is None:
            raise PDSAChallengeError(f"INVALID_CHALLENGE_{field.upper()}")
    generation = value["release_policy_generation"]
    if type(generation) is not int or not 1 <= generation <= MAX_EXACT_INTEGER:
        raise PDSAChallengeError("INVALID_CHALLENGE_RELEASE_POLICY_GENERATION")
    issued = _timestamp(value["issued_at_utc"])
    expires = _timestamp(value["expires_at_utc"])
    if expires - issued != timedelta(seconds=VALIDITY_SECONDS):
        raise PDSAChallengeError("INVALID_CHALLENGE_VALIDITY_WINDOW")
    challenge_id = value["challenge_id"]
    if type(challenge_id) is not str or _CHALLENGE_ID.fullmatch(challenge_id) is None:
        raise PDSAChallengeError("INVALID_PDSA_CHALLENGE_ID")
    parsed = uuid.UUID(challenge_id[6:])
    millis = int(issued.timestamp()) * 1000
    if parsed.version != 7 or str(parsed) != challenge_id[6:] or parsed.int >> 80 != millis:
        raise PDSAChallengeError("INVALID_PDSA_CHALLENGE_ID")
    return value


def _parse(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or len(raw) > 16_384:
        raise PDSAChallengeError("INVALID_CHALLENGE_CANONICAL_BYTES")
    try:
        value: dict[str, Any] = parse_canonical(raw)
    except (ValueError, TypeError, RecursionError) as exc:
        raise PDSAChallengeError("NONCANONICAL_PDSA_CHALLENGE") from exc
    if set(value) != {"payload", "signatures"}:
        raise PDSAChallengeError("CHALLENGE_SCHEMA_MISMATCH")
    _validate_payload(value["payload"])
    signatures = value["signatures"]
    if type(signatures) is not list or not 2 <= len(signatures) <= 3:
        raise PDSAChallengeError("CHALLENGE_PDSA_THRESHOLD_NOT_MET")
    ids: list[str] = []
    for record in signatures:
        if type(record) is not dict or set(record) != SIGNATURE_FIELDS:
            raise PDSAChallengeError("CHALLENGE_SIGNATURE_SCHEMA_MISMATCH")
        key_id = record["key_id"]
        if type(key_id) is not str or _KEY_ID.fullmatch(key_id) is None:
            raise PDSAChallengeError("INVALID_CHALLENGE_SIGNER_ID")
        if record["algorithm"] != "Ed25519":
            raise PDSAChallengeError("INVALID_CHALLENGE_SIGNATURE_ALGORITHM")
        signature = record["signature_hex"]
        if type(signature) is not str or _HEX128.fullmatch(signature) is None:
            raise PDSAChallengeError("INVALID_CHALLENGE_SIGNATURE_ENCODING")
        ids.append(key_id)
    if ids != sorted(set(ids)):
        raise PDSAChallengeError("UNKNOWN_OR_DUPLICATE_CHALLENGE_SIGNER")
    return value


@dataclass(frozen=True, init=False)
class PDSAEnrollmentChallengeV1:
    """Canonical public syntax only; parsing never confers issuer authority."""

    canonical_bytes: bytes

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("use from_canonical_bytes()")

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> PDSAEnrollmentChallengeV1:
        _parse(raw)
        result = object.__new__(cls)
        object.__setattr__(result, "canonical_bytes", raw)
        return result

    @property
    def document(self) -> dict[str, Any]:
        return _parse(self.canonical_bytes)

    @property
    def digest_sha256(self) -> str:
        _parse(self.canonical_bytes)
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @property
    def nonce_digest_sha256(self) -> str:
        return hashlib.sha256(bytes.fromhex(self.document["payload"]["nonce_hex"])).hexdigest()


def _verify_signatures(raw: bytes, context: object) -> PDSAEnrollmentChallengeV1:
    trusted = require_current_production_trust_context(context)
    challenge = PDSAEnrollmentChallengeV1.from_canonical_bytes(raw)
    document = challenge.document
    payload = document["payload"]
    if (
        payload["release_policy_digest_sha256"] != trusted.release_payload_digest
        or payload["release_policy_generation"] != trusted.release_version
    ):
        raise PDSAChallengeError("CHALLENGE_PRODUCTION_TRUST_BINDING_MISMATCH")
    message = SIGNATURE_DOMAIN + hashlib.sha256(canonical_json_bytes(payload)).digest()
    for record in document["signatures"]:
        key = trusted.pdsa_keys.get(record["key_id"])
        if key is None:
            raise PDSAChallengeError("UNKNOWN_OR_DUPLICATE_CHALLENGE_SIGNER")
        try:
            key.verify(bytes.fromhex(record["signature_hex"]), message)
        except (InvalidSignature, ValueError) as exc:
            raise PDSAChallengeError("INVALID_CHALLENGE_PDSA_SIGNATURE") from exc
    return challenge


def verify_signed_production_pdsa_challenge(
    raw: bytes, context: object
) -> PDSAEnrollmentChallengeV1:
    """Client-side signed freshness check; returns data, never retained authority.

    The Windows preparation boundary has no access to the off-host issuer store.
    Only verify_issued can grant the separate retained ISSUED capability.
    """
    challenge = _verify_signatures(raw, context)
    payload = challenge.document["payload"]
    now = _utc_now()
    if now < _timestamp(payload["issued_at_utc"]):
        raise PDSAChallengeError("CHALLENGE_NOT_YET_VALID")
    if now >= _timestamp(payload["expires_at_utc"]):
        raise PDSAChallengeError("CHALLENGE_EXPIRED")
    return challenge


class VerifiedIssuedPDSAChallenge:
    """Verifier-issued immutable capability for one exact retained issuer store."""

    __slots__ = ("canonical_bytes", "digest_sha256", "nonce_digest_sha256", "__weakref__")
    canonical_bytes: bytes
    digest_sha256: str
    nonce_digest_sha256: str

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("VerifiedIssuedPDSAChallenge comes only from verify_issued")

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("VerifiedIssuedPDSAChallenge is immutable")

    @property
    def document(self) -> dict[str, Any]:
        return _parse(self.canonical_bytes)

    def require_request_binding(self, request: PreEnrollmentRequestV1) -> None:
        require_verified_issued_challenge(self)
        payload = self.document["payload"]
        value = request.document
        expected = {
            "pdsa_challenge_id": payload["challenge_id"],
            "pdsa_challenge_digest_sha256": self.digest_sha256,
            "pdsa_challenge_nonce_digest_sha256": self.nonce_digest_sha256,
            "release_policy_digest_sha256": payload["release_policy_digest_sha256"],
            "release_policy_generation": payload["release_policy_generation"],
            "pdsa_trust_domain": payload["pdsa_trust_domain"],
            "product": payload["product"],
            "product_profile": payload["product_profile"],
            "environment": payload["environment"],
        }
        if any(value[field] != expected_value for field, expected_value in expected.items()):
            raise PDSAChallengeError("PRE_ENROLLMENT_CHALLENGE_BINDING_MISMATCH")


@dataclass(frozen=True)
class _IssuedSnapshot:
    raw: bytes
    digest: str
    nonce_digest: str
    store: PDSAChallengeStore
    store_path: Path
    context: ProductionTrustContext


_ISSUED: WeakKeyDictionary[VerifiedIssuedPDSAChallenge, _IssuedSnapshot] = WeakKeyDictionary()


def require_verified_issued_challenge(
    value: object,
    *,
    store: PDSAChallengeStore | None = None,
    context: object | None = None,
) -> VerifiedIssuedPDSAChallenge:
    """Recheck provenance, unchanged bytes, retained ISSUED state and current UTC."""
    if type(value) is not VerifiedIssuedPDSAChallenge:
        raise PDSAChallengeError("VERIFIED_ISSUED_PDSA_CHALLENGE_REQUIRED")
    snapshot = _ISSUED.get(value)
    if snapshot is None:
        raise PDSAChallengeError("VERIFIED_ISSUED_PDSA_CHALLENGE_REQUIRED")
    try:
        valid = (
            value.canonical_bytes == snapshot.raw
            and value.digest_sha256 == snapshot.digest
            and value.nonce_digest_sha256 == snapshot.nonce_digest
            and snapshot.store.path == snapshot.store_path
            and (store is None or store is snapshot.store)
            and (context is None or context is snapshot.context)
        )
    except AttributeError:
        valid = False
    if not valid:
        raise PDSAChallengeError("VERIFIED_ISSUED_PDSA_CHALLENGE_REQUIRED")
    _verify_signatures(snapshot.raw, snapshot.context)
    snapshot.store._require_issued(snapshot.raw)
    return value


class PDSAChallengeStore:
    """Trusted PDSA service state; its custody is a production deployment prerequisite.

    The database must remain under issuer custody. Its path is never accepted from
    the enrollment transport caller. SQLite FULL synchronization and one immediate
    transaction protect publication/consume atomicity, not arbitrary file rollback.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        with self._connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS pdsa_challenges (
                    challenge_id TEXT PRIMARY KEY,
                    challenge_raw BLOB NOT NULL,
                    challenge_digest TEXT NOT NULL UNIQUE,
                    expires_at_utc TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('ISSUED','CONSUMED','EXPIRED')),
                    request_raw BLOB,
                    request_digest TEXT,
                    receipt_raw BLOB,
                    CHECK((state='CONSUMED' AND request_raw IS NOT NULL
                        AND request_digest IS NOT NULL AND receipt_raw IS NOT NULL)
                        OR (state!='CONSUMED' AND request_raw IS NULL
                        AND request_digest IS NULL AND receipt_raw IS NULL))
                )"""
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            yield db
        finally:
            db.close()

    def issue(
        self,
        context: object,
        signature_provider: Callable[[bytes], list[dict[str, str]]],
    ) -> bytes:
        """Generate issuer freshness, verify external quorum, commit before return."""
        trusted = require_current_production_trust_context(context)
        issued = _utc_now().replace(microsecond=0)
        millis = int(issued.timestamp()) * 1000
        if not 0 <= millis < 1 << 48:
            raise PDSAChallengeError("INVALID_CHALLENGE_TIMESTAMP")
        raw_id = (millis << 80) | (7 << 76) | (secrets.randbits(12) << 64)
        raw_id |= (2 << 62) | secrets.randbits(62)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "environment": "PRODUCTION",
            "product": PRODUCT_NAME,
            "product_profile": PRODUCTION_PRODUCT_PROFILE,
            "pdsa_trust_domain": PDSA_TRUST_DOMAIN,
            "challenge_id": "pchal_" + str(uuid.UUID(int=raw_id)),
            "nonce_hex": secrets.token_bytes(32).hex(),
            "release_policy_digest_sha256": trusted.release_payload_digest,
            "release_policy_generation": trusted.release_version,
            "issued_at_utc": issued.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "expires_at_utc": (issued + timedelta(seconds=VALIDITY_SECONDS)).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "pdsa_key_set_digest": PDSA_KEY_SET_DIGEST,
            "signature_algorithm_profile": SIGNATURE_PROFILE,
        }
        message = SIGNATURE_DOMAIN + hashlib.sha256(canonical_json_bytes(payload)).digest()
        signatures = signature_provider(message)
        raw: bytes = canonical_json_bytes({"payload": payload, "signatures": signatures})
        challenge = _verify_signatures(raw, trusted)
        # A signing service can be slow or lose responses. Never publish an
        # already-expired artifact.
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            require_current_production_trust_context(trusted)
            # Acquiring the writer lock can wait. Check the deadline only after
            # acquiring it, so delayed signing/lock contention cannot publish an
            # already-expired ISSUED challenge.
            now = _utc_now()
            if not issued <= now < _timestamp(payload["expires_at_utc"]):
                raise PDSAChallengeError("CHALLENGE_EXPIRED")
            try:
                db.execute(
                    "INSERT INTO pdsa_challenges VALUES (?,?,?,?, 'ISSUED', NULL,NULL,NULL)",
                    (
                        payload["challenge_id"],
                        raw,
                        challenge.digest_sha256,
                        payload["expires_at_utc"],
                    ),
                )
                db.commit()
            except sqlite3.IntegrityError as exc:
                db.rollback()
                raise PDSAChallengeError("CHALLENGE_ISSUANCE_CONFLICT") from exc
        return raw

    @staticmethod
    def _exact_row(db: sqlite3.Connection, raw: bytes) -> sqlite3.Row:
        challenge = PDSAEnrollmentChallengeV1.from_canonical_bytes(raw)
        row = db.execute(
            "SELECT * FROM pdsa_challenges WHERE challenge_id=?",
            (challenge.document["payload"]["challenge_id"],),
        ).fetchone()
        if not isinstance(row, sqlite3.Row):
            raise PDSAChallengeError("UNKNOWN_RETAINED_PDSA_CHALLENGE")
        if row["challenge_raw"] != raw or row["challenge_digest"] != challenge.digest_sha256:
            raise PDSAChallengeError("RETAINED_CHALLENGE_BYTES_MISMATCH")
        return row

    def _require_issued(self, raw: bytes) -> None:
        payload = PDSAEnrollmentChallengeV1.from_canonical_bytes(raw).document["payload"]
        expired = False
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._exact_row(db, raw)
            if row["state"] == "EXPIRED":
                raise PDSAChallengeError("CHALLENGE_EXPIRED")
            if row["state"] != "ISSUED":
                raise PDSAChallengeError("CHALLENGE_ALREADY_CONSUMED")
            now = _utc_now()
            if now < _timestamp(payload["issued_at_utc"]):
                raise PDSAChallengeError("CHALLENGE_NOT_YET_VALID")
            if now >= _timestamp(payload["expires_at_utc"]):
                db.execute(
                    "UPDATE pdsa_challenges SET state='EXPIRED' "
                    "WHERE challenge_id=? AND state='ISSUED'",
                    (payload["challenge_id"],),
                )
                expired = True
            db.commit()
        if expired:
            raise PDSAChallengeError("CHALLENGE_EXPIRED")

    def verify_issued(self, raw: bytes, context: object) -> VerifiedIssuedPDSAChallenge:
        trusted = require_current_production_trust_context(context)
        challenge = _verify_signatures(raw, trusted)
        self._require_issued(raw)
        result = object.__new__(VerifiedIssuedPDSAChallenge)
        object.__setattr__(result, "canonical_bytes", raw)
        object.__setattr__(result, "digest_sha256", challenge.digest_sha256)
        object.__setattr__(result, "nonce_digest_sha256", challenge.nonce_digest_sha256)
        _ISSUED[result] = _IssuedSnapshot(
            raw, challenge.digest_sha256, challenge.nonce_digest_sha256, self, self.path, trusted
        )
        return result

    def retry_exact_accepted(self, *, request_raw: bytes, challenge_raw: bytes) -> bytes | None:
        """Return only a previously stored result; no new authority is established."""
        request = PreEnrollmentRequestV1.from_canonical_bytes(request_raw)
        with self._connect() as db:
            row = self._exact_row(db, challenge_raw)
            if row["state"] == "CONSUMED":
                if (
                    row["request_digest"] != request.digest_sha256
                    or row["request_raw"] != request_raw
                ):
                    raise PDSAChallengeError("CHALLENGE_REPLAY_CONFLICT")
                return bytes(row["receipt_raw"])
            if row["state"] == "EXPIRED":
                raise PDSAChallengeError("CHALLENGE_EXPIRED")
            return None

    def consume_authenticated_request(self, value: object) -> bytes:
        """Atomic acceptance boundary; package issuance and subject minting stay separate."""
        from .production_pre_enrollment import require_authenticated_pre_enrollment

        accepted = require_authenticated_pre_enrollment(value, challenge_store=self)
        request_raw, challenge_raw = accepted.request_raw, accepted.challenge_raw
        request = PreEnrollmentRequestV1.from_canonical_bytes(request_raw)
        challenge = PDSAEnrollmentChallengeV1.from_canonical_bytes(challenge_raw)
        payload = challenge.document["payload"]
        if (
            request.document["pdsa_challenge_id"] != payload["challenge_id"]
            or request.document["pdsa_challenge_digest_sha256"] != challenge.digest_sha256
            or request.document["pdsa_challenge_nonce_digest_sha256"]
            != challenge.nonce_digest_sha256
        ):
            raise PDSAChallengeError("PRE_ENROLLMENT_CHALLENGE_BINDING_MISMATCH")
        receipt: bytes = canonical_json_bytes(
            {
                "schema_version": "PDSAPreEnrollmentAcceptanceReceiptV1",
                "environment": "PRODUCTION",
                "purpose": "PRE_ENROLLMENT_AUTHENTICATION_ACCEPTED_ONLY",
                "pdsa_challenge_id": payload["challenge_id"],
                "pdsa_challenge_digest_sha256": challenge.digest_sha256,
                "pre_enrollment_request_digest_sha256": request.digest_sha256,
                "legal_enrollment": "NOT_PERFORMED",
            }
        )
        expired = False
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._exact_row(db, challenge_raw)
            if row["state"] == "CONSUMED":
                if (
                    row["request_digest"] != request.digest_sha256
                    or row["request_raw"] != request_raw
                ):
                    raise PDSAChallengeError("CHALLENGE_REPLAY_CONFLICT")
                return bytes(row["receipt_raw"])
            if row["state"] == "EXPIRED":
                raise PDSAChallengeError("CHALLENGE_EXPIRED")
            # The TPM challenge or EK endorsement can expire earlier than the
            # PDSA challenge while acquiring this writer lock. Recheck the full
            # composed capability at the acceptance boundary, without renewing it.
            require_authenticated_pre_enrollment(value, challenge_store=self)
            require_current_production_trust_context(accepted.context)
            now = _utc_now()
            if now < _timestamp(payload["issued_at_utc"]):
                raise PDSAChallengeError("CHALLENGE_NOT_YET_VALID")
            if now >= _timestamp(payload["expires_at_utc"]):
                db.execute(
                    "UPDATE pdsa_challenges SET state='EXPIRED' "
                    "WHERE challenge_id=? AND state='ISSUED'",
                    (payload["challenge_id"],),
                )
                expired = True
            else:
                db.execute(
                    """UPDATE pdsa_challenges SET state='CONSUMED',request_raw=?,
                        request_digest=?,receipt_raw=? WHERE challenge_id=? AND state='ISSUED'""",
                    (request_raw, request.digest_sha256, receipt, payload["challenge_id"]),
                )
            db.commit()
        if expired:
            raise PDSAChallengeError("CHALLENGE_EXPIRED")
        return receipt
