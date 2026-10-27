"""Bounded Stage 9 observer data contracts; parsing never establishes trust.

This module performs no I/O and never executes inspected code. Parsed contract
records hold immutable bytes. Structural acceptance is always UNVERIFIED.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, NoReturn, cast

from deployment.canonical_json import canonical_json_bytes

PURPOSE = "STAGE9_OBSERVER_TEST_ONLY"
ACCEPTANCE_DOMAIN = b"CryptoHunter.Stage9ObserverAcceptanceV1\x00"
MAX_INTEGER = 2**53 - 1
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_DOCUMENT_BYTES = 65536
MAX_CONTEXT_BYTES = 16384
MAX_DEPTH = 16
MAX_FILES = 32768
MAX_SOURCES = 128
MAX_REASON_CODES = 64
MAX_PATH_BYTES = 4096
REASON_CODES = frozenset(
    {
        "FORMAT_REJECTED",
        "LIMIT_EXCEEDED",
        "TRUST_CONTEXT_MISSING",
        "ACCEPTANCE_REJECTED",
        "LAUNCH_POLICY_MISMATCH",
        "RUNTIME_LAUNCH_UNVERIFIED",
        "HOST_SCOPE_UNVERIFIED",
        "IMPORT_POLICY_REJECTED",
        "CLOSURE_INCOMPLETE",
        "ROLLBACK_REJECTED",
        "ARTIFACT_MISMATCH",
        "UNSUPPORTED_SOURCE",
        "PROVENANCE_INCOMPLETE",
        "FILE_POLICY_REJECTED",
        "FILE_CHANGED",
        "REPORT_REJECTED",
        "CHANNEL_REJECTED",
    }
)


class ContractRefusal(ValueError):
    """An input-independent diagnostic; no input or exception text is exposed."""

    def __init__(self, code: str = "FORMAT_REJECTED") -> None:
        self.code = code if type(code) is str and code in REASON_CODES else "FORMAT_REJECTED"
        super().__init__(self.code)


def _deny(code: str = "FORMAT_REJECTED") -> NoReturn:
    raise ContractRefusal(code)


def _require(condition: bool, code: str = "FORMAT_REJECTED") -> None:
    if not condition:
        _deny(code)


def _string(value: Any, maximum: int = MAX_MANIFEST_BYTES) -> str:
    if type(value) is not str:
        _deny()
    if any(ord(c) == 0 or 0xD800 <= ord(c) <= 0xDFFF for c in value):
        _deny()
    if len(value.encode("utf-8")) > maximum:
        _deny("LIMIT_EXCEEDED")
    return cast(str, value)


def _integer(value: Any, minimum: int = 0, maximum: int = MAX_INTEGER) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _deny()
    return cast(int, value)


def _record(value: Any, fields: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        _deny()
    return cast(dict[str, Any], value)


def _list(value: Any, maximum: int = MAX_FILES, minimum: int = 0) -> list[Any]:
    if type(value) is not list:
        _deny()
    if len(value) > maximum:
        _deny("LIMIT_EXCEEDED")
    if len(value) < minimum:
        _deny()
    return cast(list[Any], value)


def _identifier(value: Any) -> str:
    value = _string(value, 128)
    _require(bool(value) and value.isascii())
    return cast(str, value)


def _hex(value: Any, length: int) -> str:
    value = _string(value)
    _require(re.fullmatch(r"[0-9a-f]{%d}" % length, value) is not None)
    return cast(str, value)


def _digest(value: Any) -> str:
    value = _hex(value, 64)
    _require(value != "0" * 64)
    return cast(str, value)


def _enum(value: Any, allowed: set[str] | frozenset[str]) -> str:
    value = _string(value)
    _require(value in allowed)
    return cast(str, value)


def _equal(value: Any, expected: Any) -> None:
    _require(type(value) is type(expected))
    if type(expected) is dict:
        _require(set(value) == set(expected))
        for key in expected:
            _equal(value[key], expected[key])
    elif type(expected) is list:
        _require(len(value) == len(expected))
        for actual, wanted in zip(value, expected):
            _equal(actual, wanted)
    else:
        _require(value == expected)


def _path(value: Any, absolute: bool = True) -> str:
    value = _string(value, MAX_PATH_BYTES)
    _require(bool(value) and "\\" not in value and all(ord(c) >= 32 for c in value))
    _require(value.startswith("/") is absolute)
    if absolute and value == "/":
        return cast(str, value)
    components = value[1:].split("/") if absolute else value.split("/")
    _require(all(c not in {"", ".", ".."} for c in components))
    return cast(str, value)


def _timestamp(value: Any) -> datetime:
    value = _string(value, 20)
    _require(
        re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value) is not None
    )
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        raise ContractRefusal() from None


def _schema(value: dict[str, Any], name: str, purpose: bool = False) -> None:
    _equal(value["schema"], name)
    _equal(value["version"], 1)
    if purpose:
        _equal(value["purpose"], PURPOSE)


def _reason_codes(value: Any) -> list[str]:
    result = [_enum(x, REASON_CODES) for x in _list(value, MAX_REASON_CODES)]
    _require(len(set(result)) == len(result))
    return result


def _blocked_fields() -> dict[str, Any]:
    return {
        "qualification": "BLOCKED",
        "full_confirmation": False,
        "coexistence_authorized": False,
        "preflight_exception": False,
        "physical_test_status": "PHYSICAL_TEST_PENDING",
        "status": "IN_PROGRESS",
        "production_status": "PRODUCTION_NO_GO",
        "runtime_attestation": "MISSING",
        "host_pid_namespace": "UNVERIFIED",
    }


def _preflight(raw: bytes) -> None:
    """Bound depth, integer tokens and container cardinality before the JSON decoder.

    This lexical pass has a stack of at most 16 integers, never recurses and
    allocates no input-sized token lists. JSON syntax is checked by json.loads.
    """
    stack: list[int] = []
    index = 0
    while index < len(raw):
        char = raw[index]
        if char in b" \t\r\n:":
            index += 1
            continue
        if char == 44:
            if stack:
                stack[-1] += 1
                _require(stack[-1] <= MAX_FILES, "LIMIT_EXCEEDED")
            index += 1
            continue
        if char in (93, 125):
            _require(bool(stack))
            stack.pop()
            index += 1
            continue
        if char in (91, 123):
            _require(len(stack) < MAX_DEPTH, "LIMIT_EXCEEDED")
            stack.append(1)
            index += 1
        elif char == 34:
            start = index
            index += 1
            while index < len(raw) and raw[index] != 34:
                index += 2 if raw[index] == 92 else 1
            _require(index < len(raw))
            index += 1
        else:
            start = index
            while index < len(raw) and raw[index] not in b" \t\r\n,]}:":
                index += 1
                _require(index - start <= 16, "LIMIT_EXCEEDED")
            token = raw[start:index]
            _require(
                token in (b"true", b"false") or re.fullmatch(rb"0|[1-9][0-9]*", token) is not None
            )
    _require(not stack)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _string(key)
        _require(key not in result)
        result[key] = value
    return result


def _reject_number(value: str) -> NoReturn:
    _deny()


def _parse_integer(value: str) -> int:
    _require(len(value) <= 16, "LIMIT_EXCEEDED")
    return _integer(int(value))


def _decode(raw: bytes, maximum: int = MAX_DOCUMENT_BYTES) -> Any:
    _require(type(raw) is bytes)
    _require(bool(raw))
    _require(len(raw) <= maximum, "LIMIT_EXCEEDED")
    _require(not raw.startswith(b"\xef\xbb\xbf"))
    _preflight(raw)
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_pairs,
            parse_int=_parse_integer,
            parse_float=_reject_number,
            parse_constant=_reject_number,
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, ContractRefusal):
            raise ContractRefusal(exc.code) from None
        raise ContractRefusal() from None
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is str:
            _string(item)
        elif type(item) is dict:
            pending.extend(item.values())
        elif type(item) is list:
            pending.extend(item)
        elif type(item) is int:
            _integer(item)
        elif type(item) is not bool:
            _deny()
    return value


@dataclass(frozen=True, slots=True)
class ParsedContract:
    """Immutable structural data, explicitly carrying no authorization."""

    schema: str
    raw: bytes
    canonical: bytes
    sha256: str
    authorization: Literal["UNVERIFIED"] = field(default="UNVERIFIED", init=False)

    def as_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self.canonical))


def decode_json(raw: bytes) -> Any:
    """Strict bounded JSON data only; does not validate a contract or trust."""
    return _decode(raw, MAX_MANIFEST_BYTES)


def _checkpoint(value: Any) -> None:
    value = _record(value, {"sequence", "state_sha256"})
    _integer(value["sequence"])
    _digest(value["state_sha256"])


def _validate_acceptance(value: Any) -> None:
    value = _record(value, {"payload", "payload_digest", "threshold", "signatures"})
    payload = _record(
        value["payload"],
        {
            "schema",
            "version",
            "purpose",
            "manifest_sha256",
            "policy_sha256",
            "root_set_id",
            "sequence",
            "revocation_checkpoint",
            "issued_at",
            "expires_at",
        },
    )
    _schema(payload, "CryptoHunter.Stage9ObserverAcceptanceV1", purpose=True)
    _digest(payload["manifest_sha256"])
    _digest(payload["policy_sha256"])
    _identifier(payload["root_set_id"])
    _integer(payload["sequence"])
    _checkpoint(payload["revocation_checkpoint"])
    _require(
        _timestamp(payload["issued_at"]) < _timestamp(payload["expires_at"]), "ACCEPTANCE_REJECTED"
    )
    _equal(value["threshold"], 2)
    _digest(value["payload_digest"])
    _require(
        value["payload_digest"] == hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        "ACCEPTANCE_REJECTED",
    )
    signers: list[str] = []
    for signature in _list(value["signatures"], maximum=MAX_FILES, minimum=2):
        signature = _record(signature, {"signer_id", "signature_hex"})
        signers.append(_identifier(signature["signer_id"]))
        _hex(signature["signature_hex"], 128)
    _require(signers == sorted(set(signers)), "ACCEPTANCE_REJECTED")


def acceptance_signed_message(raw: bytes) -> bytes:
    """Return the specified signing preimage, without verifying any signature."""
    value = _decode(raw)
    _validate_acceptance(value)
    return ACCEPTANCE_DOMAIN + bytes.fromhex(value["payload_digest"])
