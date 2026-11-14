"""Bounded APT V4c contracts and synthetic signature checks, without enrollment.

Only canonical JSON encoding is shared with other Stage 9 contracts. Their keys,
acceptance domains and authority are deliberately not inherited. Explicit caller
TEST authorities can exercise cryptography; none can authorize production.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, NoReturn, cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from deployment.canonical_json import canonical_json_bytes

MAX_DOCUMENT_BYTES = 65536
MAX_DEPTH = 8
MAX_SIGNATURES = 8
MAX_SIGNERS = 32
MAX_INTEGER = 2**53 - 1
MAX_TIMESTAMP = 253402300799
ALGORITHM = "ed25519-sha256"
DOMAINS = {
    "acceptance": "stage9-apt-context-acceptance-v1",
    "checkpoint": "stage9-apt-checkpoint-v1",
    "time": "stage9-apt-time-evidence-v1",
}


class V4cRefusal(ValueError):
    """A fixed diagnostic code; untrusted document contents are never echoed."""

    def __init__(self, code: str = "V4C_FORMAT_REJECTED") -> None:
        super().__init__(code)
        self.code = code


def _deny(code: str = "V4C_FORMAT_REJECTED") -> NoReturn:
    raise V4cRefusal(code)


def _require(condition: bool, code: str = "V4C_FORMAT_REJECTED") -> None:
    if not condition:
        _deny(code)


def _record(value: Any, fields: set[str]) -> dict[str, Any]:
    _require(type(value) is dict and set(value) == fields)
    return cast(dict[str, Any], value)


def _integer(value: Any, minimum: int = 0, maximum: int = MAX_INTEGER) -> int:
    _require(type(value) is int and minimum <= value <= maximum)
    return cast(int, value)


def _label(value: Any) -> str:
    _require(
        type(value) is str
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/+#=-]{0,255}", value) is not None
    )
    return cast(str, value)


def _hex(value: Any, size: int) -> str:
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{%d}" % size, value) is not None)
    return cast(str, value)


def _digest(value: Any) -> str:
    result = _hex(value, 64)
    _require(result != "0" * 64)
    return result


def _optional_digest(value: Any) -> None:
    if value is not None:
        _digest(value)


def _interval(valid_from: Any, valid_until: Any) -> None:
    _integer(valid_from, maximum=MAX_TIMESTAMP)
    _integer(valid_until, maximum=MAX_TIMESTAMP)
    _require(valid_from < valid_until)


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in items:
        _require(key not in value)
        value[key] = item
    return value


def _number(value: str) -> NoReturn:
    _deny()


def _parse_integer(value: str) -> int:
    _require(len(value) <= 16, "V4C_LIMIT_EXCEEDED")
    return _integer(int(value))


def _decode(raw: bytes) -> dict[str, Any]:
    _require(type(raw) is bytes and bool(raw))
    _require(len(raw) <= MAX_DOCUMENT_BYTES, "V4C_LIMIT_EXCEEDED")
    # Bound nesting before entering the recursive JSON decoder. Syntax, matching
    # delimiters and malformed escapes are subsequently checked by json.loads.
    depth = 0
    in_string = False
    escaped = False
    for char in raw:
        if in_string:
            if escaped:
                escaped = False
            elif char == 92:
                escaped = True
            elif char == 34:
                in_string = False
        elif char == 34:
            in_string = True
        elif char in (91, 123):
            depth += 1
            _require(depth <= MAX_DEPTH, "V4C_LIMIT_EXCEEDED")
        elif char in (93, 125):
            depth -= 1
            _require(depth >= 0)
    _require(depth == 0 and not in_string)
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_pairs,
            parse_int=_parse_integer,
            parse_float=_number,
            parse_constant=_number,
        )
        pending = [value]
        while pending:
            item = pending.pop()
            if type(item) is str:
                _require(not any(0xD800 <= ord(char) <= 0xDFFF for char in item))
            elif type(item) is dict:
                pending.extend(item.keys())
                pending.extend(item.values())
            elif type(item) is list:
                pending.extend(item)
        _require(raw == canonical_json_bytes(value), "V4C_NONCANONICAL")
    except V4cRefusal:
        raise
    except (ValueError, UnicodeError, RecursionError):
        raise V4cRefusal() from None
    _require(type(value) is dict)
    return cast(dict[str, Any], value)


@dataclass(frozen=True, slots=True)
class TestAuthorityKeyV4c:
    """A caller TEST key, never an enrolled APT root."""

    signer_id: str
    public_key_hex: str
    valid_from: int
    valid_until: int
    revoked: bool = False


@dataclass(frozen=True, slots=True)
class TestAuthorityV4c:
    """Explicit out-of-material TEST anchor; there is no production counterpart."""

    authority_id: str
    policy_version: int
    keys: tuple[TestAuthorityKeyV4c, ...]
    threshold: int
    valid_from: int
    valid_until: int
    revoked: bool = False
    scope: Literal["SYNTHETIC_TEST_ONLY"] = field(default="SYNTHETIC_TEST_ONLY", init=False)


@dataclass(frozen=True, slots=True)
class SignatureV4c:
    signer_id: str
    signature_hex: str


@dataclass(frozen=True, slots=True)
class SignedRecordV4c:
    """Immutable bounded wire record; inspecting it never establishes authority."""

    raw: bytes
    kind: str
    payload_bytes: bytes
    signatures: tuple[SignatureV4c, ...]
    authorization: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)

    @property
    def payload(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.payload_bytes))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.raw).hexdigest()


_COMMON = {
    "schema",
    "authority_id",
    "policy_version",
    "algorithm",
    "domain",
    "valid_from",
    "valid_until",
}
_ACCEPTANCE = {
    "context_sha256",
    "keyring_sha256",
    "allowed_signers",
    "repository_id",
    "snapshot_id",
    "snapshot_sha256",
    "sequence",
    "issued_at",
    "checkpoint_id",
    "checkpoint_sha256",
    "rotation_policy",
    "revocation_policy",
    "previous_context_sha256",
}
_CHECKPOINT = {
    "checkpoint_id",
    "checkpoint_source_id",
    "repository_id",
    "context_sha256",
    "snapshot_id",
    "sequence",
    "snapshot_sha256",
    "successor_policy",
    "rotation_policy",
    "revoked",
    "previous_checkpoint_id",
    "previous_snapshot_sha256",
    "previous_context_sha256",
    "permitted_context_successors",
}
_TIME = {
    "proof_id",
    "time_source_id",
    "repository_id",
    "snapshot_id",
    "snapshot_sha256",
    "context_sha256",
    "checkpoint_id",
    "evaluation_time",
}


def _validate_payload(kind: str, candidate: Any) -> dict[str, Any]:
    _require(type(kind) is str and kind in DOMAINS)
    fields = {"acceptance": _ACCEPTANCE, "checkpoint": _CHECKPOINT, "time": _TIME}[kind]
    value = _record(candidate, _COMMON | fields)
    _require(value["schema"] == DOMAINS[kind] and value["domain"] == DOMAINS[kind])
    _require(value["algorithm"] == ALGORITHM)
    _label(value["authority_id"])
    _integer(value["policy_version"], 1, 2147483647)
    _interval(value["valid_from"], value["valid_until"])
    for name in ("repository_id", "snapshot_id", "checkpoint_id"):
        _label(value[name])
    for name in ("context_sha256", "snapshot_sha256"):
        _digest(value[name])
    if kind in {"acceptance", "checkpoint"}:
        _integer(value["sequence"])
        _require(value["rotation_policy"] == "EXPLICIT_CONTEXT_LINK")
        _optional_digest(value["previous_context_sha256"])
    if kind == "acceptance":
        for name in ("keyring_sha256", "checkpoint_sha256"):
            _digest(value[name])
        _integer(value["issued_at"], maximum=MAX_TIMESTAMP)
        _require(value["valid_from"] <= value["issued_at"] < value["valid_until"])
        _require(value["revocation_policy"] == "DENY_REVOKED")
        signers = value["allowed_signers"]
        _require(type(signers) is list and 1 <= len(signers) <= MAX_SIGNERS, "V4C_LIMIT_EXCEEDED")
        pairs = []
        for signer in signers:
            signer = _record(signer, {"signing_fingerprint", "primary_fingerprint"})
            for fingerprint in signer.values():
                _require(
                    type(fingerprint) is str
                    and re.fullmatch(r"[0-9A-F]{40}", fingerprint) is not None
                )
                _require(fingerprint != "0" * 40)
            pairs.append((signer["signing_fingerprint"], signer["primary_fingerprint"]))
        _require(pairs == sorted(set(pairs)), "V4C_SIGNERS_REJECTED")
    elif kind == "checkpoint":
        _label(value["checkpoint_source_id"])
        _require(value["successor_policy"] == "EXACT_NEXT_SEQUENCE")
        _require(type(value["revoked"]) is bool)
        previous_id = value["previous_checkpoint_id"]
        if previous_id is not None:
            _label(previous_id)
        _optional_digest(value["previous_snapshot_sha256"])
        _require(
            (previous_id is None)
            == (value["previous_snapshot_sha256"] is None)
            == (value["previous_context_sha256"] is None),
            "V4C_CHECKPOINT_LINK_REJECTED",
        )
        successors = value["permitted_context_successors"]
        _require(type(successors) is list and len(successors) <= MAX_SIGNERS, "V4C_LIMIT_EXCEEDED")
        for successor in successors:
            _digest(successor)
        _require(successors == sorted(set(successors)))
    else:
        _label(value["proof_id"])
        _label(value["time_source_id"])
        _integer(value["evaluation_time"], maximum=MAX_TIMESTAMP)
        _require(value["valid_from"] <= value["evaluation_time"] < value["valid_until"])
    return value


def _parse(kind: str, raw: bytes) -> SignedRecordV4c:
    envelope = _record(_decode(raw), {"payload", "signatures"})
    payload = _validate_payload(kind, envelope["payload"])
    signatures = envelope["signatures"]
    _require(
        type(signatures) is list and 1 <= len(signatures) <= MAX_SIGNATURES, "V4C_LIMIT_EXCEEDED"
    )
    result = []
    for signature in signatures:
        signature = _record(signature, {"signer_id", "signature_hex"})
        result.append(
            SignatureV4c(_label(signature["signer_id"]), _hex(signature["signature_hex"], 128))
        )
    signers = [signature.signer_id for signature in result]
    _require(signers == sorted(set(signers)), "V4C_DUPLICATE_SIGNATURE")
    return SignedRecordV4c(raw, kind, canonical_json_bytes(payload), tuple(result))


def parse_acceptance_v4c(raw: bytes) -> SignedRecordV4c:
    return _parse("acceptance", raw)


def parse_checkpoint_v4c(raw: bytes) -> SignedRecordV4c:
    return _parse("checkpoint", raw)


def parse_time_evidence_v4c(raw: bytes) -> SignedRecordV4c:
    return _parse("time", raw)


def signing_message_v4c(kind: str, payload: dict[str, Any]) -> bytes:
    """APT-only domain, one NUL, then raw SHA-256 of canonical payload bytes."""
    _validate_payload(kind, payload)
    return (
        DOMAINS[kind].encode("ascii")
        + b"\x00"
        + hashlib.sha256(canonical_json_bytes(payload)).digest()
    )


def _validate_authority(
    authority: TestAuthorityV4c | None, evaluation_time: int
) -> TestAuthorityV4c:
    _require(type(authority) is TestAuthorityV4c, "V4C_TEST_AUTHORITY_MISSING")
    assert authority is not None
    _label(authority.authority_id)
    _integer(authority.policy_version, 1, 2147483647)
    _interval(authority.valid_from, authority.valid_until)
    _require(type(authority.revoked) is bool and not authority.revoked, "V4C_AUTHORITY_REVOKED")
    _require(authority.scope == "SYNTHETIC_TEST_ONLY")
    _require(
        type(authority.keys) is tuple and 1 <= len(authority.keys) <= MAX_SIGNATURES,
        "V4C_LIMIT_EXCEEDED",
    )
    _integer(authority.threshold, 1, len(authority.keys))
    identifiers: set[str] = set()
    public_keys: set[str] = set()
    for key in authority.keys:
        _require(type(key) is TestAuthorityKeyV4c)
        _label(key.signer_id)
        _hex(key.public_key_hex, 64)
        _require(key.public_key_hex != "0" * 64)
        _interval(key.valid_from, key.valid_until)
        _require(type(key.revoked) is bool)
        _require(
            key.signer_id not in identifiers and key.public_key_hex not in public_keys,
            "V4C_AUTHORITY_KEYS_REJECTED",
        )
        identifiers.add(key.signer_id)
        public_keys.add(key.public_key_hex)
    _integer(evaluation_time, maximum=MAX_TIMESTAMP)
    _require(
        authority.valid_from <= evaluation_time < authority.valid_until, "V4C_AUTHORITY_EXPIRED"
    )
    return authority


def verify_record_v4c(
    record: SignedRecordV4c,
    authority: TestAuthorityV4c | None,
    evaluation_time: int,
) -> None:
    """Verify every signature under an explicit TEST anchor; authorize nothing.

    Threshold and public keys come exclusively from the separate caller argument.
    Neither an envelope nor a signature can add a key or lower the threshold.
    All extra signatures must also verify; there is no partial threshold success.
    """
    _require(type(record) is SignedRecordV4c)
    parsed = _parse(record.kind, record.raw)
    _require(parsed == record, "V4C_RECORD_CHANGED")
    authority = _validate_authority(authority, evaluation_time)
    payload = parsed.payload
    _require(
        (payload["authority_id"], payload["policy_version"])
        == (authority.authority_id, authority.policy_version),
        "V4C_AUTHORITY_MISMATCH",
    )
    _require(
        payload["valid_from"] <= evaluation_time < payload["valid_until"], "V4C_POLICY_EXPIRED"
    )
    _require(
        authority.valid_from
        <= payload["valid_from"]
        < payload["valid_until"]
        <= authority.valid_until,
        "V4C_POLICY_INTERVAL_REJECTED",
    )
    _require(len(parsed.signatures) >= authority.threshold, "V4C_THRESHOLD_REJECTED")
    keys = {key.signer_id: key for key in authority.keys}
    if "issued_at" in payload:
        _require(payload["issued_at"] <= evaluation_time, "V4C_SIGNING_TIME_REJECTED")
    message = signing_message_v4c(parsed.kind, payload)
    for signature in parsed.signatures:
        key = keys.get(signature.signer_id)
        _require(key is not None, "V4C_UNKNOWN_SIGNER")
        assert key is not None
        _require(not key.revoked, "V4C_AUTHORITY_KEY_REVOKED")
        _require(key.valid_from <= evaluation_time < key.valid_until, "V4C_AUTHORITY_KEY_EXPIRED")
        if "issued_at" in payload:
            _require(
                key.valid_from <= payload["issued_at"] < key.valid_until,
                "V4C_SIGNING_TIME_REJECTED",
            )
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(key.public_key_hex)).verify(
                bytes.fromhex(signature.signature_hex), message
            )
        except (InvalidSignature, ValueError):
            raise V4cRefusal("V4C_SIGNATURE_INVALID") from None
