"""Initial production PDSA authorization with durable reserve/sign/finalize.

Only a verifier-issued pre-enrollment capability can reserve new identity. The
registered issuer owns both the retained SQLite state and the off-host quorum
signing boundary. Public parsing and terminal lookups never confer authority.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization

from bot_core.uuid7 import (
    UUID7Error,
    mint_uuid7,
    reservation_epoch_milliseconds,
    secrets as secrets,
)

from deployment.production_enrollment_issuer import (
    require_production_enrollment_issuer,
    require_production_pdsa_store,
)
from deployment.windows_stage9_production_trust import (
    require_current_production_trust_context,
)

from .canonical import canonical_json_bytes, parse_canonical
from .external_provisioning import PACKAGE_FIELDS, PACKAGE_SCHEMA, PDSA_DOMAIN
from .pdsa_enrollment_challenge import (
    PDSAEnrollmentChallengeV1,
    require_verified_issued_challenge,
)
from .pre_enrollment import (
    ALGORITHM_PROFILE,
    MAX_EXACT_INTEGER,
    PDSA_TRUST_DOMAIN,
    PreEnrollmentRequestV1,
)
from .product_profile import PRODUCTION_PRODUCT_PROFILE
from .production_pre_enrollment import (
    AuthenticatedProductionPreEnrollment,
    _snapshot,
    authenticated_package_expiry,
    require_authenticated_pre_enrollment,
)

SIGNATURE_FIELDS = frozenset({"key_id", "algorithm", "signature_hex"})
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX128 = re.compile(r"[0-9a-f]{128}\Z")
_KEY_ID = re.compile(r"[A-Za-z0-9._-]{1,128}\Z")
_UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
_UUID7 = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z")


class PDSAAuthorizationError(ValueError):
    """Stable fail-closed production authorization reason."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: object) -> datetime:
    if type(value) is not str or _UTC.fullmatch(value) is None:
        raise PDSAAuthorizationError("INVALID_PACKAGE_TIMESTAMP")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise PDSAAuthorizationError("INVALID_PACKAGE_TIMESTAMP") from exc


def _identity(value: object, prefix: str) -> None:
    if type(value) is not str or not value.startswith(prefix):
        raise PDSAAuthorizationError("INVALID_PACKAGE_IDENTITY")
    suffix = value[len(prefix) :]
    if _UUID7.fullmatch(suffix) is None:
        raise PDSAAuthorizationError("INVALID_PACKAGE_IDENTITY")
    parsed = uuid.UUID(suffix)
    if parsed.version != 7 or str(parsed) != suffix:
        raise PDSAAuthorizationError("INVALID_PACKAGE_IDENTITY")


def validate_initial_production_payload(value: object) -> dict[str, Any]:
    """Validate the frozen initial production syntax without granting authority."""
    if type(value) is not dict or set(value) != PACKAGE_FIELDS:
        raise PDSAAuthorizationError("PACKAGE_PAYLOAD_SCHEMA_MISMATCH")
    literals = {
        "schema_version": PACKAGE_SCHEMA,
        "environment": "PRODUCTION",
        "pdsa_trust_domain": PDSA_TRUST_DOMAIN,
        "product_profile": PRODUCTION_PRODUCT_PROFILE,
        "pre_enrollment_public_key_algorithm_profile": ALGORITHM_PROFILE,
    }
    for field, expected in literals.items():
        if type(value[field]) is not str or value[field] != expected:
            raise PDSAAuthorizationError(f"INVALID_PACKAGE_{field.upper()}")
    _identity(value["provisioning_subject_id"], "psub_")
    _identity(value["enrollment_reference"], "penr_")
    _identity(value["pdsa_challenge_id"], "pchal_")
    for field in (
        "pdsa_challenge_digest_sha256",
        "pre_enrollment_request_digest_sha256",
        "verified_tpm_exchange_reference",
        "verified_tpm_public_projection_id",
        "target_tpm_ek_public_digest",
        "target_tpm_ak_public_digest",
        "pre_enrollment_public_key_fingerprint_sha256",
        "release_policy_digest_sha256",
    ):
        if type(value[field]) is not str or _HEX64.fullmatch(value[field]) is None:
            raise PDSAAuthorizationError(f"INVALID_PACKAGE_{field.upper()}")
    for field in ("authorization_generation", "authorization_version", "lineage_generation"):
        if type(value[field]) is not int or value[field] != 1:
            raise PDSAAuthorizationError("INITIAL_PACKAGE_LINEAGE_REQUIRED")
    generation = value["release_policy_generation"]
    if type(generation) is not int or not 1 <= generation <= MAX_EXACT_INTEGER:
        raise PDSAAuthorizationError("INVALID_PACKAGE_RELEASE_POLICY_GENERATION")
    if value["predecessor_package_digest_or_null"] is not None:
        raise PDSAAuthorizationError("INITIAL_PACKAGE_LINEAGE_REQUIRED")
    if _timestamp(value["issued_at_utc"]) >= _timestamp(value["expires_at_utc"]):
        raise PDSAAuthorizationError("INVALID_PACKAGE_VALIDITY_WINDOW")
    return value


def _parse(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or len(raw) > 32_768:
        raise PDSAAuthorizationError("INVALID_PACKAGE_CANONICAL_BYTES")
    try:
        value: dict[str, Any] = parse_canonical(raw)
    except (ValueError, TypeError, RecursionError) as exc:
        raise PDSAAuthorizationError("NONCANONICAL_PDSA_AUTHORIZATION_PACKAGE") from exc
    if set(value) != {"payload", "signatures"}:
        raise PDSAAuthorizationError("PACKAGE_SCHEMA_MISMATCH")
    validate_initial_production_payload(value["payload"])
    signatures = value["signatures"]
    if type(signatures) is not list or len(signatures) != 2:
        raise PDSAAuthorizationError("PACKAGE_PDSA_THRESHOLD_NOT_MET")
    ids: list[str] = []
    for record in signatures:
        if type(record) is not dict or set(record) != SIGNATURE_FIELDS:
            raise PDSAAuthorizationError("PACKAGE_SIGNATURE_SCHEMA_MISMATCH")
        key_id = record["key_id"]
        if type(key_id) is not str or _KEY_ID.fullmatch(key_id) is None:
            raise PDSAAuthorizationError("INVALID_PACKAGE_SIGNER_ID")
        if type(record["algorithm"]) is not str or record["algorithm"] != "Ed25519":
            raise PDSAAuthorizationError("INVALID_PACKAGE_SIGNATURE_ALGORITHM")
        signature = record["signature_hex"]
        if type(signature) is not str or _HEX128.fullmatch(signature) is None:
            raise PDSAAuthorizationError("INVALID_PACKAGE_SIGNATURE_ENCODING")
        ids.append(key_id)
    if ids != sorted(set(ids)):
        raise PDSAAuthorizationError("UNKNOWN_OR_DUPLICATE_PACKAGE_SIGNER")
    return value


@dataclass(frozen=True, init=False)
class PDSAEnrollmentAuthorizationPackageV1:
    """Exact canonical public package data; construction is never authority."""

    canonical_bytes: bytes

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("use from_canonical_bytes()")

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> PDSAEnrollmentAuthorizationPackageV1:
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


def _verify_package_signatures(raw: bytes, context: object) -> PDSAEnrollmentAuthorizationPackageV1:
    trusted = require_current_production_trust_context(context)
    package = PDSAEnrollmentAuthorizationPackageV1.from_canonical_bytes(raw)
    payload = package.document["payload"]
    if (
        payload["release_policy_digest_sha256"] != trusted.release_payload_digest
        or payload["release_policy_generation"] != trusted.release_version
    ):
        raise PDSAAuthorizationError("REJECT_PACKAGE_TARGET_MISMATCH")
    message = PDSA_DOMAIN + hashlib.sha256(canonical_json_bytes(payload)).digest()
    for record in package.document["signatures"]:
        key = trusted.pdsa_keys.get(record["key_id"])
        if key is None:
            raise PDSAAuthorizationError("UNKNOWN_OR_DUPLICATE_PACKAGE_SIGNER")
        try:
            key.verify(bytes.fromhex(record["signature_hex"]), message)
        except (InvalidSignature, ValueError) as exc:
            raise PDSAAuthorizationError("INVALID_PACKAGE_PDSA_SIGNATURE") from exc
    return package


def _reservation_epoch_milliseconds(reservation_now: datetime) -> int:
    try:
        return reservation_epoch_milliseconds(reservation_now)
    except UUID7Error as exc:
        raise PDSAAuthorizationError("INVALID_PACKAGE_TIMESTAMP") from exc


def _mint_uuidv7(prefix: str, millis: int) -> str:
    try:
        return mint_uuid7(prefix, millis)
    except UUID7Error as exc:
        raise PDSAAuthorizationError("INVALID_PACKAGE_TIMESTAMP") from exc


def _payload(
    accepted: AuthenticatedProductionPreEnrollment,
    *,
    subject: str,
    reference: str,
    issued: str,
    expires: str,
) -> dict[str, Any]:
    request = PreEnrollmentRequestV1.from_canonical_bytes(accepted.request_raw)
    source = request.document
    payload = {
        "schema_version": PACKAGE_SCHEMA,
        "environment": "PRODUCTION",
        "pdsa_trust_domain": source["pdsa_trust_domain"],
        "provisioning_subject_id": subject,
        "enrollment_reference": reference,
        "pdsa_challenge_id": source["pdsa_challenge_id"],
        "pdsa_challenge_digest_sha256": source["pdsa_challenge_digest_sha256"],
        "pre_enrollment_request_digest_sha256": request.digest_sha256,
        "verified_tpm_exchange_reference": source["verified_tpm_exchange_reference"],
        "verified_tpm_public_projection_id": source["verified_tpm_public_projection_id"],
        "target_tpm_ek_public_digest": source["ek_public_digest"],
        "target_tpm_ak_public_digest": source["ak_public_digest"],
        "pre_enrollment_public_key_algorithm_profile": source[
            "pre_enrollment_public_key_algorithm_profile"
        ],
        "pre_enrollment_public_key_fingerprint_sha256": source[
            "pre_enrollment_public_key_fingerprint_sha256"
        ],
        "authorization_generation": 1,
        "authorization_version": 1,
        "issued_at_utc": issued,
        "expires_at_utc": expires,
        "release_policy_digest_sha256": source["release_policy_digest_sha256"],
        "release_policy_generation": source["release_policy_generation"],
        "product_profile": source["product_profile"],
        "predecessor_package_digest_or_null": None,
        "lineage_generation": 1,
    }
    return validate_initial_production_payload(payload)


@dataclass(frozen=True)
class _Reservation:
    payload_raw: bytes
    state: str
    package_raw: bytes | None


def _issuance_row(db: sqlite3.Connection, challenge_id: str) -> sqlite3.Row | None:
    row = db.execute(
        "SELECT * FROM pdsa_authorization_issuances WHERE pdsa_challenge_id=?",
        (challenge_id,),
    ).fetchone()
    return row if isinstance(row, sqlite3.Row) else None


def _check_exact_request(row: sqlite3.Row, request_raw: bytes, challenge_raw: bytes) -> None:
    request = PreEnrollmentRequestV1.from_canonical_bytes(request_raw)
    challenge = PDSAEnrollmentChallengeV1.from_canonical_bytes(challenge_raw)
    if (
        row["request_raw"] != request_raw
        or row["pre_enrollment_request_digest_sha256"] != request.digest_sha256
        or row["pdsa_challenge_digest_sha256"] != challenge.digest_sha256
        or request.document["pdsa_challenge_id"] != row["pdsa_challenge_id"]
        or request.document["pdsa_challenge_digest_sha256"] != challenge.digest_sha256
    ):
        raise PDSAAuthorizationError("REJECT_CHALLENGE_REPLAY_CONFLICT")


def _check_reservation(row: sqlite3.Row, accepted: AuthenticatedProductionPreEnrollment) -> None:
    _check_exact_request(row, accepted.request_raw, accepted.challenge_raw)
    _check_quorum_reservation(row, accepted.context)
    expected = _payload(
        accepted,
        subject=row["provisioning_subject_id"],
        reference=row["enrollment_reference"],
        issued=row["issued_at_utc"],
        expires=authenticated_package_expiry(accepted),
    )
    if (
        row["payload_raw"] != canonical_json_bytes(expected)
        or row["expires_at_utc"] != expected["expires_at_utc"]
    ):
        raise PDSAAuthorizationError("REJECT_PACKAGE_TARGET_MISMATCH")


def _production_trust_snapshot(context: object) -> bytes:
    trusted = require_current_production_trust_context(context)
    raw: bytes = canonical_json_bytes(
        {
            "ceremony_id": trusted.ceremony_id,
            "release_policy_digest_sha256": trusted.release_payload_digest,
            "release_policy_generation": trusted.release_version,
            "pdsa_public_keys": [
                {
                    "key_id": key_id,
                    "public_key_hex": key.public_bytes(
                        serialization.Encoding.Raw, serialization.PublicFormat.Raw
                    ).hex(),
                }
                for key_id, key in sorted(trusted.pdsa_keys.items())
            ],
        }
    )
    return raw


def _retained_quorum_metadata(row: sqlite3.Row) -> list[str]:
    """Check frozen metadata without renewing terminal package authority."""
    try:
        authorized = _retained_signer_ids(row["authorized_signer_ids_raw"], 3)
        trust = parse_canonical(row["production_trust_raw"])
        valid = (
            type(row["required_threshold"]) is int
            and row["required_threshold"] == 2
            and type(trust) is dict
            and set(trust)
            == {
                "ceremony_id",
                "release_policy_digest_sha256",
                "release_policy_generation",
                "pdsa_public_keys",
            }
            and type(trust["ceremony_id"]) is str
            and bool(trust["ceremony_id"])
            and type(trust["release_policy_digest_sha256"]) is str
            and _HEX64.fullmatch(trust["release_policy_digest_sha256"]) is not None
            and type(trust["release_policy_generation"]) is int
            and 1 <= trust["release_policy_generation"] <= MAX_EXACT_INTEGER
            and type(trust["pdsa_public_keys"]) is list
            and len(trust["pdsa_public_keys"]) == 3
        )
        if valid:
            valid = all(
                type(record) is dict
                and set(record) == {"key_id", "public_key_hex"}
                and record["key_id"] == key_id
                and type(record["public_key_hex"]) is str
                and _HEX64.fullmatch(record["public_key_hex"]) is not None
                for key_id, record in zip(authorized, trust["pdsa_public_keys"], strict=True)
            )
        if valid and row["state"] == "RESERVED":
            valid = (
                row["signer_ids_raw"] is None
                and row["package_raw"] is None
                and row["pdsa_package_digest_sha256"] is None
            )
        elif valid and row["state"] in ("SIGNED", "COMMITTED"):
            selected = _retained_signer_ids(row["signer_ids_raw"], 2)
            valid = (
                all(key_id in authorized for key_id in selected)
                and type(row["package_raw"]) is bytes
                and type(row["pdsa_package_digest_sha256"]) is str
            )
        else:
            valid = False
    except (ValueError, TypeError, KeyError, RecursionError):
        valid = False
    if not valid:
        raise PDSAAuthorizationError("INVALID_RETAINED_QUORUM_RESERVATION")
    return authorized


def _retained_signer_ids(raw: object, count: int) -> list[str]:
    if type(raw) is not bytes:
        raise PDSAAuthorizationError("INVALID_RETAINED_QUORUM_RESERVATION")
    ids = json.loads(raw)
    if (
        type(ids) is not list
        or len(ids) != count
        or any(type(key_id) is not str or _KEY_ID.fullmatch(key_id) is None for key_id in ids)
        or ids != sorted(set(ids))
        or canonical_json_bytes(ids) != raw
    ):
        raise PDSAAuthorizationError("INVALID_RETAINED_QUORUM_RESERVATION")
    result: list[str] = ids
    return result


def _check_quorum_reservation(row: sqlite3.Row, context: object) -> None:
    authorized = _retained_quorum_metadata(row)
    trusted = require_current_production_trust_context(context)
    if authorized != sorted(trusted.pdsa_keys) or row[
        "production_trust_raw"
    ] != _production_trust_snapshot(trusted):
        raise PDSAAuthorizationError("REJECT_PACKAGE_TARGET_MISMATCH")


def _require_live_challenge(row: sqlite3.Row, raw: bytes) -> None:
    payload = PDSAEnrollmentChallengeV1.from_canonical_bytes(raw).document["payload"]
    if row["state"] != "ISSUED":
        raise PDSAAuthorizationError("CHALLENGE_NOT_AVAILABLE_FOR_PACKAGE_ISSUANCE")
    now = _utc_now()
    if now < _timestamp(payload["issued_at_utc"]):
        raise PDSAAuthorizationError("CHALLENGE_NOT_YET_VALID")
    if now >= _timestamp(payload["expires_at_utc"]):
        raise PDSAAuthorizationError("CHALLENGE_EXPIRED")


def _retained_package(row: sqlite3.Row) -> bytes:
    _retained_quorum_metadata(row)
    raw = bytes(row["package_raw"])
    package = PDSAEnrollmentAuthorizationPackageV1.from_canonical_bytes(raw)
    payload = package.document["payload"]
    trust = parse_canonical(row["production_trust_raw"])
    if (
        package.digest_sha256 != row["pdsa_package_digest_sha256"]
        or canonical_json_bytes(payload) != row["payload_raw"]
        or payload["provisioning_subject_id"] != row["provisioning_subject_id"]
        or payload["enrollment_reference"] != row["enrollment_reference"]
        or payload["pdsa_challenge_id"] != row["pdsa_challenge_id"]
        or payload["pdsa_challenge_digest_sha256"] != row["pdsa_challenge_digest_sha256"]
        or payload["pre_enrollment_request_digest_sha256"]
        != row["pre_enrollment_request_digest_sha256"]
        or canonical_json_bytes([entry["key_id"] for entry in package.document["signatures"]])
        != row["signer_ids_raw"]
        or payload["release_policy_digest_sha256"] != trust["release_policy_digest_sha256"]
        or payload["release_policy_generation"] != trust["release_policy_generation"]
    ):
        raise PDSAAuthorizationError("RETAINED_PACKAGE_BYTES_MISMATCH")
    return raw


def _reserve_issuance(accepted: AuthenticatedProductionPreEnrollment) -> _Reservation:
    store = accepted.challenge_store
    request = PreEnrollmentRequestV1.from_canonical_bytes(accepted.request_raw)
    challenge_id = request.document["pdsa_challenge_id"]
    with store._connect() as db:
        db.execute("BEGIN IMMEDIATE")
        require_production_pdsa_store(store, context=accepted.context)
        challenge_row = store._exact_row(db, accepted.challenge_raw)
        row = _issuance_row(db, challenge_id)
        if row is not None:
            _check_exact_request(row, accepted.request_raw, accepted.challenge_raw)
            if row["state"] == "COMMITTED":
                _check_committed_challenge(db, row)
                return _Reservation(bytes(row["payload_raw"]), "COMMITTED", _retained_package(row))
            require_authenticated_pre_enrollment(accepted, challenge_store=store)
            _require_live_challenge(challenge_row, accepted.challenge_raw)
            _check_reservation(row, accepted)
            return _Reservation(
                bytes(row["payload_raw"]),
                row["state"],
                None if row["package_raw"] is None else _retained_package(row),
            )
        # A legacy acceptance receipt has no reserved identity or legal package.
        # It cannot be upgraded after its challenge has already been consumed.
        _require_live_challenge(challenge_row, accepted.challenge_raw)
        require_authenticated_pre_enrollment(accepted, challenge_store=store)
        expires = authenticated_package_expiry(accepted)
        reservation_now = _utc_now()
        reservation_ms = _reservation_epoch_milliseconds(reservation_now)
        issued = reservation_now.replace(microsecond=0)
        if not issued < _timestamp(expires):
            raise PDSAAuthorizationError("PACKAGE_EXPIRED")
        subject = _mint_uuidv7("psub_", reservation_ms)
        reference = _mint_uuidv7("penr_", reservation_ms)
        issued_text = issued.strftime("%Y-%m-%dT%H:%M:%SZ")
        payload_raw = canonical_json_bytes(
            _payload(
                accepted, subject=subject, reference=reference, issued=issued_text, expires=expires
            )
        )
        try:
            db.execute(
                """INSERT INTO pdsa_authorization_issuances (
                    pdsa_challenge_id,pdsa_challenge_digest_sha256,
                    pre_enrollment_request_digest_sha256,request_raw,payload_raw,
                    provisioning_subject_id,enrollment_reference,issued_at_utc,
                    expires_at_utc,production_trust_raw,authorized_signer_ids_raw,
                    required_threshold,state) VALUES (?,?,?,?,?,?,?,?,?,?,?,?, 'RESERVED')""",
                (
                    challenge_id,
                    request.document["pdsa_challenge_digest_sha256"],
                    request.digest_sha256,
                    accepted.request_raw,
                    payload_raw,
                    subject,
                    reference,
                    issued_text,
                    expires,
                    _production_trust_snapshot(accepted.context),
                    canonical_json_bytes(sorted(accepted.context.pdsa_keys)),
                    2,
                ),
            )
            db.commit()
        except sqlite3.IntegrityError as exc:
            db.rollback()
            raise PDSAAuthorizationError("PACKAGE_ISSUANCE_RESERVATION_CONFLICT") from exc
    return _Reservation(payload_raw, "RESERVED", None)


def _persist_signed_package(
    accepted: AuthenticatedProductionPreEnrollment, package_raw: bytes
) -> None:
    package = _verify_package_signatures(package_raw, accepted.context)
    store = accepted.challenge_store
    request = PreEnrollmentRequestV1.from_canonical_bytes(accepted.request_raw)
    with store._connect() as db:
        db.execute("BEGIN IMMEDIATE")
        require_production_pdsa_store(store, context=accepted.context)
        row = _issuance_row(db, request.document["pdsa_challenge_id"])
        if row is None:
            raise PDSAAuthorizationError("UNKNOWN_PACKAGE_ISSUANCE_RESERVATION")
        _check_exact_request(row, accepted.request_raw, accepted.challenge_raw)
        _check_quorum_reservation(row, accepted.context)
        if canonical_json_bytes(package.document["payload"]) != row["payload_raw"]:
            raise PDSAAuthorizationError("REJECT_PACKAGE_TARGET_MISMATCH")
        selected_signer_ids_raw = canonical_json_bytes(
            [entry["key_id"] for entry in package.document["signatures"]]
        )
        if row["state"] in ("SIGNED", "COMMITTED"):
            if (
                row["signer_ids_raw"] != selected_signer_ids_raw
                or _retained_package(row) != package_raw
            ):
                raise PDSAAuthorizationError("PACKAGE_SIGNATURE_RETRY_CONFLICT")
            return
        require_authenticated_pre_enrollment(accepted, challenge_store=store)
        _check_reservation(row, accepted)
        _require_live_challenge(
            store._exact_row(db, accepted.challenge_raw), accepted.challenge_raw
        )
        if _utc_now() >= _timestamp(row["expires_at_utc"]):
            raise PDSAAuthorizationError("PACKAGE_EXPIRED")
        db.execute(
            """UPDATE pdsa_authorization_issuances SET state='SIGNED',package_raw=?,
                pdsa_package_digest_sha256=?,signer_ids_raw=?
                WHERE pdsa_challenge_id=? AND state='RESERVED'""",
            (package_raw, package.digest_sha256, selected_signer_ids_raw, row["pdsa_challenge_id"]),
        )
        db.commit()


def _receipt(request_raw: bytes, challenge_raw: bytes) -> bytes:
    request = PreEnrollmentRequestV1.from_canonical_bytes(request_raw)
    challenge = PDSAEnrollmentChallengeV1.from_canonical_bytes(challenge_raw)
    raw: bytes = canonical_json_bytes(
        {
            "schema_version": "PDSAPreEnrollmentAcceptanceReceiptV1",
            "environment": "PRODUCTION",
            "purpose": "PRE_ENROLLMENT_AUTHENTICATION_ACCEPTED_ONLY",
            "pdsa_challenge_id": challenge.document["payload"]["challenge_id"],
            "pdsa_challenge_digest_sha256": challenge.digest_sha256,
            "pre_enrollment_request_digest_sha256": request.digest_sha256,
            "legal_enrollment": "NOT_PERFORMED",
        }
    )
    return raw


def _check_committed_challenge(db: sqlite3.Connection, row: sqlite3.Row) -> None:
    challenge = db.execute(
        "SELECT * FROM pdsa_challenges WHERE challenge_id=?", (row["pdsa_challenge_id"],)
    ).fetchone()
    if (
        not isinstance(challenge, sqlite3.Row)
        or challenge["state"] != "CONSUMED"
        or challenge["challenge_digest"] != row["pdsa_challenge_digest_sha256"]
        or challenge["request_raw"] != row["request_raw"]
        or challenge["request_digest"] != row["pre_enrollment_request_digest_sha256"]
        or challenge["receipt_raw"] != _receipt(row["request_raw"], challenge["challenge_raw"])
    ):
        raise PDSAAuthorizationError("INCONSISTENT_PACKAGE_COMMIT")


def _finalize_issuance(accepted: AuthenticatedProductionPreEnrollment) -> bytes:
    store = accepted.challenge_store
    request = PreEnrollmentRequestV1.from_canonical_bytes(accepted.request_raw)
    with store._connect() as db:
        db.execute("BEGIN IMMEDIATE")
        require_production_pdsa_store(store, context=accepted.context)
        row = _issuance_row(db, request.document["pdsa_challenge_id"])
        if row is None:
            raise PDSAAuthorizationError("UNKNOWN_PACKAGE_ISSUANCE_RESERVATION")
        _check_exact_request(row, accepted.request_raw, accepted.challenge_raw)
        if row["state"] == "COMMITTED":
            _check_committed_challenge(db, row)
            return _retained_package(row)
        if row["state"] != "SIGNED":
            raise PDSAAuthorizationError("SIGNED_PACKAGE_ISSUANCE_REQUIRED")
        # Recheck the actual registered request PoP, exchange, endorsement and
        # custody after writer-lock acquisition, immediately before final commit.
        require_authenticated_pre_enrollment(accepted, challenge_store=store)
        _check_reservation(row, accepted)
        challenge_row = store._exact_row(db, accepted.challenge_raw)
        _require_live_challenge(challenge_row, accepted.challenge_raw)
        raw = _retained_package(row)
        _verify_package_signatures(raw, accepted.context)
        now = _utc_now()
        if now < _timestamp(row["issued_at_utc"]):
            raise PDSAAuthorizationError("PACKAGE_NOT_YET_VALID")
        if now >= _timestamp(row["expires_at_utc"]):
            raise PDSAAuthorizationError("PACKAGE_EXPIRED")
        db.execute(
            """UPDATE pdsa_challenges SET state='CONSUMED',request_raw=?,
                request_digest=?,receipt_raw=? WHERE challenge_id=? AND state='ISSUED'""",
            (
                accepted.request_raw,
                request.digest_sha256,
                _receipt(accepted.request_raw, accepted.challenge_raw),
                row["pdsa_challenge_id"],
            ),
        )
        db.execute(
            "UPDATE pdsa_authorization_issuances SET state='COMMITTED' "
            "WHERE pdsa_challenge_id=? AND state='SIGNED'",
            (row["pdsa_challenge_id"],),
        )
        db.commit()
    return raw


def _require_signing_reservation(issuer: object, payload_raw: bytes) -> None:
    """Restrict the issuer signing boundary to its exact unpublished reservation."""
    trusted_issuer = require_production_enrollment_issuer(issuer)
    trusted = require_current_production_trust_context(trusted_issuer.trust)
    if type(payload_raw) is not bytes:
        raise PDSAAuthorizationError("INVALID_PACKAGE_CANONICAL_BYTES")
    try:
        payload = validate_initial_production_payload(parse_canonical(payload_raw))
    except (ValueError, TypeError, RecursionError) as exc:
        raise PDSAAuthorizationError("INVALID_SIGNING_RESERVATION_PAYLOAD") from exc
    store = trusted_issuer.pdsa_store
    with store._connect() as db:
        require_production_pdsa_store(store, context=trusted, issuer=trusted_issuer)
        row = _issuance_row(db, payload["pdsa_challenge_id"])
        if (
            row is None
            or row["state"] not in ("RESERVED", "SIGNED")
            or row["payload_raw"] != payload_raw
            or payload["release_policy_digest_sha256"] != trusted.release_payload_digest
            or payload["release_policy_generation"] != trusted.release_version
        ):
            raise PDSAAuthorizationError("EXACT_SIGNING_RESERVATION_REQUIRED")
        try:
            _check_quorum_reservation(row, trusted)
        except PDSAAuthorizationError as exc:
            raise PDSAAuthorizationError("EXACT_SIGNING_RESERVATION_REQUIRED") from exc
        challenge = db.execute(
            "SELECT * FROM pdsa_challenges WHERE challenge_id=?", (payload["pdsa_challenge_id"],)
        ).fetchone()
        if not isinstance(challenge, sqlite3.Row):
            raise PDSAAuthorizationError("EXACT_SIGNING_RESERVATION_REQUIRED")
        _check_exact_request(row, row["request_raw"], challenge["challenge_raw"])
        _require_live_challenge(challenge, challenge["challenge_raw"])
        now = _utc_now()
        if now < _timestamp(payload["issued_at_utc"]):
            raise PDSAAuthorizationError("PACKAGE_NOT_YET_VALID")
        if now >= _timestamp(payload["expires_at_utc"]):
            raise PDSAAuthorizationError("PACKAGE_EXPIRED")


def retry_production_pdsa_enrollment_authorization(
    issuer: object, *, request_raw: bytes, challenge_raw: bytes
) -> bytes | None:
    """Retrieve exact terminal bytes without renewing an expired authorization."""
    trusted_issuer = require_production_enrollment_issuer(issuer)
    store = trusted_issuer.pdsa_store
    require_production_pdsa_store(store, issuer=trusted_issuer)
    request = PreEnrollmentRequestV1.from_canonical_bytes(request_raw)
    with store._connect() as db:
        require_production_pdsa_store(store, issuer=trusted_issuer)
        challenge = store._exact_row(db, challenge_raw)
        parsed_challenge = PDSAEnrollmentChallengeV1.from_canonical_bytes(challenge_raw)
        if (
            request.document["pdsa_challenge_id"] != challenge["challenge_id"]
            or request.document["pdsa_challenge_digest_sha256"] != parsed_challenge.digest_sha256
            or request.document["pdsa_challenge_nonce_digest_sha256"]
            != parsed_challenge.nonce_digest_sha256
        ):
            raise PDSAAuthorizationError("REJECT_CHALLENGE_REPLAY_CONFLICT")
        row = _issuance_row(db, request.document["pdsa_challenge_id"])
        if row is not None:
            _check_exact_request(row, request_raw, challenge_raw)
            if row["state"] == "COMMITTED":
                _check_committed_challenge(db, row)
                return _retained_package(row)
            return None
        if challenge["state"] == "CONSUMED":
            if (
                challenge["request_raw"] != request_raw
                or challenge["request_digest"] != request.digest_sha256
            ):
                raise PDSAAuthorizationError("REJECT_CHALLENGE_REPLAY_CONFLICT")
            raise PDSAAuthorizationError("CONSUMED_RECEIPT_HAS_NO_AUTHORIZATION_PACKAGE")
        return None


def lookup_production_pdsa_enrollment_authorization(
    issuer: object, *, enrollment_reference: str
) -> bytes | None:
    """Issuer-side LPPI lookup exposes only an atomically committed package."""
    trusted_issuer = require_production_enrollment_issuer(issuer)
    _identity(enrollment_reference, "penr_")
    store = trusted_issuer.pdsa_store
    with store._connect() as db:
        require_production_pdsa_store(store, issuer=trusted_issuer)
        row = db.execute(
            "SELECT * FROM pdsa_authorization_issuances WHERE enrollment_reference=?",
            (enrollment_reference,),
        ).fetchone()
        if not isinstance(row, sqlite3.Row) or row["state"] != "COMMITTED":
            return None
        _check_committed_challenge(db, row)
        return _retained_package(row)


def issue_production_pdsa_enrollment_authorization(value: object) -> bytes:
    """Issue only from genuine authentication; identities/signers are issuer-owned."""
    snapshot = _snapshot(value)
    issuer = require_production_pdsa_store(
        snapshot.challenge_store, context=snapshot.context, issuer=snapshot.issuer
    )
    retained = retry_production_pdsa_enrollment_authorization(
        issuer, request_raw=snapshot.request_raw, challenge_raw=snapshot.challenge_raw
    )
    if retained is not None:
        return retained
    try:
        accepted = require_authenticated_pre_enrollment(value)
        require_verified_issued_challenge(
            accepted.challenge,
            store=accepted.challenge_store,
            context=accepted.context,
            issuer=issuer,
        )
        reservation = _reserve_issuance(accepted)
        if reservation.state == "COMMITTED":
            if reservation.package_raw is None:
                raise PDSAAuthorizationError("INCONSISTENT_PACKAGE_COMMIT")
            return reservation.package_raw
        if reservation.state == "RESERVED":
            # No SQLite write lock is held across the external quorum signing call.
            require_authenticated_pre_enrollment(accepted)
            signatures = issuer.sign_enrollment_authorization(reservation.payload_raw)
            raw = canonical_json_bytes(
                {"payload": parse_canonical(reservation.payload_raw), "signatures": signatures}
            )
            _persist_signed_package(accepted, raw)
        return _finalize_issuance(accepted)
    except ValueError:
        # An exact concurrent request can commit after the initial lookup and
        # before a live ISSUED/signing-reservation guard. Only a fully validated
        # terminal commit for these exact raw bytes can satisfy that retry.
        retained = retry_production_pdsa_enrollment_authorization(
            issuer, request_raw=snapshot.request_raw, challenge_raw=snapshot.challenge_raw
        )
        if retained is not None:
            return retained
        raise
