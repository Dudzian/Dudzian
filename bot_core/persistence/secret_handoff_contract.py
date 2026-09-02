"""Dependency-safe intrinsic contract for durable SecretHandoff descriptors."""

from __future__ import annotations

from collections.abc import Mapping
import math
import re
from typing import Any

from .fingerprints import canonical_json_sha256

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_ID_RE = re.compile(
    r"^(?P<prefix>[a-z][a-z0-9]*)_"
    r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
SECRET_HANDOFF_FIELDS = frozenset(
    {
        "handoff_id",
        "scope",
        "operation",
        "old_reference",
        "new_reference",
        "metadata_fingerprint_sha256",
        "operation_fingerprint_sha256",
        "reconciliation_metadata",
    }
)


class SecretHandoffContractError(ValueError):
    """A raw descriptor does not satisfy its intrinsic Stage-1 contract."""


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SecretHandoffContractError("reconciliation metadata numbers must be finite")
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise SecretHandoffContractError("reconciliation metadata keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    raise SecretHandoffContractError("reconciliation metadata must be exact JSON")


def secret_metadata_fingerprint_value(metadata: Mapping[str, Any]) -> str:
    return canonical_json_sha256(_json_value(metadata))


def secret_operation_fingerprint_value(
    *,
    scope: tuple[str, str] | list[str],
    operation: str,
    old_reference: str | None,
    new_reference: str | None,
    metadata_fingerprint_sha256: str,
) -> str:
    return canonical_json_sha256(
        {
            "scope": list(scope),
            "operation": operation,
            "old_reference": old_reference,
            "new_reference": new_reference,
            "metadata_fingerprint_sha256": metadata_fingerprint_sha256,
        }
    )


def _canonical_id(value: object, prefix: str) -> bool:
    match = _CANONICAL_ID_RE.fullmatch(value) if isinstance(value, str) else None
    return match is not None and match.group("prefix") == prefix


def validate_raw_secret_handoff_record(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != SECRET_HANDOFF_FIELDS:
        raise SecretHandoffContractError("SecretHandoffRecord requires its exact field set")
    if not isinstance(value["handoff_id"], str) or not value["handoff_id"]:
        raise SecretHandoffContractError("handoff_id must be non-empty")
    scope = value["scope"]
    if (
        not isinstance(scope, (list, tuple))
        or len(scope) != 2
        or not _canonical_id(scope[0], "acct")
        or not _canonical_id(scope[1], "dev")
    ):
        raise SecretHandoffContractError("scope must be the canonical account/device pair")
    if not isinstance(value["operation"], str) or not value["operation"]:
        raise SecretHandoffContractError("operation must be non-empty")
    for field in ("old_reference", "new_reference"):
        if value[field] is not None and not isinstance(value[field], str):
            raise SecretHandoffContractError(f"{field} must be a string or null")
    metadata = value["reconciliation_metadata"]
    if not isinstance(metadata, Mapping):
        raise SecretHandoffContractError("reconciliation_metadata must be an object")
    for field in ("metadata_fingerprint_sha256", "operation_fingerprint_sha256"):
        if not isinstance(value[field], str) or _SHA256_RE.fullmatch(value[field]) is None:
            raise SecretHandoffContractError(f"{field} must be lowercase SHA-256")
    metadata_fingerprint = secret_metadata_fingerprint_value(metadata)
    if value["metadata_fingerprint_sha256"] != metadata_fingerprint:
        raise SecretHandoffContractError("metadata fingerprint mismatch")
    if value["operation_fingerprint_sha256"] != secret_operation_fingerprint_value(
        scope=scope,
        operation=value["operation"],
        old_reference=value["old_reference"],
        new_reference=value["new_reference"],
        metadata_fingerprint_sha256=metadata_fingerprint,
    ):
        raise SecretHandoffContractError("operation fingerprint mismatch")


__all__ = [
    "SECRET_HANDOFF_FIELDS",
    "SecretHandoffContractError",
    "secret_metadata_fingerprint_value",
    "secret_operation_fingerprint_value",
    "validate_raw_secret_handoff_record",
]
