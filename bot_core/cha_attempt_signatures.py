"""Schema v6 append-only request, invocation latch and signature decisions.

An invocation latch without a signature blocks continuation permanently. It is
never erased on rollback or inferred to mean 'not signed'. This deliberately
trades availability for the frozen no-resigning requirement.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING

from bot_core.cha_issuance_request import (
    IssuanceSignerIdentity,
    IssuanceSigningRole,
    request_bytes,
    request_reference,
)

if TYPE_CHECKING:
    from bot_core.cha_attempt_store import AttemptAuthorization, AttemptIdentity

TABLES = ("issuance_requests", "signing_intents", "signature_checkpoints")
CREATE_STATEMENTS = (
    "CREATE TABLE issuance_requests("
    "attempt_id TEXT PRIMARY KEY REFERENCES reservations(attempt_id), "
    "reference TEXT NOT NULL UNIQUE, digest TEXT NOT NULL UNIQUE, canonical_bytes BLOB NOT NULL)",
    "CREATE TABLE signing_intents(attempt_id TEXT NOT NULL REFERENCES reservations(attempt_id), "
    "role TEXT NOT NULL, request_reference TEXT NOT NULL REFERENCES issuance_requests(reference), "
    "signer_json BLOB NOT NULL, PRIMARY KEY(attempt_id,role))",
    "CREATE TABLE signature_checkpoints(attempt_id TEXT NOT NULL, role TEXT NOT NULL, "
    "request_reference TEXT NOT NULL REFERENCES issuance_requests(reference), "
    "signer_json BLOB NOT NULL, signature TEXT NOT NULL, decision_key TEXT NOT NULL UNIQUE, "
    "PRIMARY KEY(attempt_id,role), "
    "FOREIGN KEY(attempt_id,role) REFERENCES signing_intents(attempt_id,role))",
) + tuple(
    f"CREATE TRIGGER {table}_immutable_{operation} BEFORE {operation.upper()} ON {table} "
    f"BEGIN SELECT RAISE(ABORT,'immutable {table}'); END"
    for table in TABLES
    for operation in ("update", "delete")
)


def create_schema(db: sqlite3.Connection) -> None:
    for statement in CREATE_STATEMENTS:
        db.execute(statement)


@dataclass(frozen=True, slots=True)
class RetainedIssuanceRequest:
    attempt_id: str
    canonical_bytes: bytes
    reference: str
    digest: str
    requester: tuple[IssuanceSignerIdentity, str] | None
    claimant: tuple[IssuanceSignerIdentity, str] | None


def checkpoint_decision(
    attempt_id: str, signer: IssuanceSignerIdentity, reference: str, signature: str
) -> str:
    from bot_core.cha_attempt_store import _decision_key

    return _decision_key(
        "SIGNATURE_CHECKPOINT",
        {
            "issuance_attempt_id": attempt_id,
            "signer": signer.payload(),
            "request_reference": reference,
            "signature": signature,
        },
    )


def load_request(
    db: sqlite3.Connection, auth: AttemptAuthorization, attempt_id: str
) -> RetainedIssuanceRequest | None:
    from bot_core.cha_attempt_store import AttemptCorruptError

    row = db.execute(
        "SELECT reference,digest,canonical_bytes FROM issuance_requests WHERE attempt_id=?",
        (attempt_id,),
    ).fetchone()
    if row is None:
        return None
    reference, digest, raw = row
    try:
        if (
            type(raw) is not bytes
            or raw != request_bytes(auth, attempt_id)
            or reference != request_reference(raw)
            or digest != hashlib.sha256(raw).hexdigest()
        ):
            raise ValueError("request bytes/reference/digest mismatch")
        signed = {}
        for role, retained_reference, signer_raw, signature, decision in db.execute(
            "SELECT role,request_reference,signer_json,signature,decision_key "
            "FROM signature_checkpoints WHERE attempt_id=?",
            (attempt_id,),
        ):
            signer = IssuanceSignerIdentity.from_bytes(signer_raw)
            signer.require_authorization(auth)
            signer.verify(raw, signature)
            intent = db.execute(
                "SELECT request_reference,signer_json FROM signing_intents "
                "WHERE attempt_id=? AND role=?",
                (attempt_id, role),
            ).fetchone()
            if (
                signer.role.value != role
                or retained_reference != reference
                or intent != (reference, signer_raw)
                or checkpoint_decision(attempt_id, signer, reference, signature) != decision
            ):
                raise ValueError("signature checkpoint identity mismatch")
            signed[signer.role] = (signer, signature)
        requester, claimant = (
            signed.get(IssuanceSigningRole.REQUESTER),
            signed.get(IssuanceSigningRole.CLAIMANT),
        )
        if claimant is not None and (
            requester is None or requester[0].public_key_hex == claimant[0].public_key_hex
        ):
            raise ValueError("claimant without distinct requester authorization")
        return RetainedIssuanceRequest(attempt_id, raw, reference, digest, requester, claimant)
    except (ValueError, TypeError, KeyError, UnicodeError) as exc:
        raise AttemptCorruptError("retained signing request is corrupt") from exc


def validate_history(
    db: sqlite3.Connection, reservations: dict, identities: dict, expected: dict, kinds: dict
) -> None:
    from bot_core.cha_attempt_store import AttemptCorruptError, AttemptState

    if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise AttemptCorruptError("signing history contains an orphaned authority record")
    retained = {}
    for (attempt_id,) in db.execute("SELECT attempt_id FROM issuance_requests"):
        if attempt_id not in reservations:
            raise AttemptCorruptError("request has no reservation")
        value = load_request(db, reservations[attempt_id][0], attempt_id)
        if value is None:
            raise AttemptCorruptError("missing request")
        retained[attempt_id] = value
        if (
            db.execute(
                "SELECT 1 FROM signing_intents WHERE attempt_id=? AND role=?",
                (attempt_id, IssuanceSigningRole.REQUESTER.value),
            ).fetchone()
            is None
        ):
            raise AttemptCorruptError("durable request has no requester invocation latch")
        for checkpoint, state in (
            (value.requester, AttemptState.REQUEST_SIGNED_BY_REQUESTER),
            (value.claimant, AttemptState.CLAIMANT_AUTHORIZED),
        ):
            if checkpoint is not None:
                signer, signature = checkpoint
                key = checkpoint_decision(attempt_id, signer, value.reference, signature)
                expected[key] = (attempt_id, state.value, value.reference, value.digest, key)
                kinds[key] = state.value
    for attempt_id, role, reference, signer_raw in db.execute(
        "SELECT attempt_id,role,request_reference,signer_json FROM signing_intents"
    ):
        try:
            signer = IssuanceSignerIdentity.from_bytes(signer_raw)
            signer.require_authorization(reservations[attempt_id][0])
            value = retained[attempt_id]
            if (
                signer.role.value != role
                or reference != value.reference
                or (signer.role is IssuanceSigningRole.CLAIMANT and value.requester is None)
            ):
                raise ValueError("inconsistent intent")
            if (
                signer.role is IssuanceSigningRole.CLAIMANT
                and value.requester is not None
                and value.requester[0].public_key_hex == signer.public_key_hex
            ):
                raise ValueError("requester/claimant alias")
        except (KeyError, TypeError, ValueError, UnicodeError) as exc:
            raise AttemptCorruptError("signing invocation intent is corrupt") from exc
    for attempt_id, identity in identities.items():
        if identity.authorization.reservation_identity is not None:
            verify_final_identity(identity, retained.get(attempt_id))


def verify_final_identity(
    identity: AttemptIdentity, retained: RetainedIssuanceRequest | None
) -> None:
    from bot_core.cha_attempt_store import AttemptCorruptError

    if (
        retained is None
        or retained.requester is None
        or retained.claimant is None
        or retained.attempt_id != identity.issuance_attempt_id
        or retained.digest != identity.root_proof_issuance_request_signed_payload_digest_sha256
        or retained.reference != identity.root_proof_issuance_request_canonical_bytes_reference
        or retained.requester[1] != identity.requester_signature_base64url
        or retained.claimant[1] != identity.claimant_authorization_signature_base64url
    ):
        raise AttemptCorruptError(
            "immutable identity differs from exact durable signing checkpoints"
        )
