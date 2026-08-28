"""Intrinsic validation of candidate persistence records.

The types in this module are integrity carriers only.  Validation performed here
does not confer domain authority, accepted/current membership, or restore status.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import re
from types import MappingProxyType
from typing import Any, ClassVar

from .fingerprints import canonical_json_sha256


class PersistenceRecordError(ValueError):
    """Raised when a persistence carrier fails intrinsic validation."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_ID_RE = re.compile(
    r"^(?P<prefix>[a-z][a-z0-9]*)_"
    r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)

_REGISTRY = {
    "CryptoHunterAccount current record": {
        "representation_category": "M011_ENTITY_IDENTITY_PROJECTION",
        "semantic_owner_milestone": "M0.2",
        "semantic_artifact": "canonical_domain_vocabulary.json",
        "semantic_json_pointer": "/entity_kinds",
        "semantic_contract_fingerprint_sha256": (
            "1913a18c7e7479d9c20850a690ee81b9f311a7d08cf2458374c91f6332d277f1"
        ),
        "record_key_strategy": "CANONICAL_ENTITY_ID",
        "record_key_source_field": "entity_id",
        "record_key_source_location": "payload",
    },
    "RuntimeSession canonical identity/history": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.2",
        "semantic_artifact": "canonical_domain_vocabulary.json",
        "semantic_json_pointer": "/entity_kinds",
        "semantic_contract_fingerprint_sha256": (
            "1913a18c7e7479d9c20850a690ee81b9f311a7d08cf2458374c91f6332d277f1"
        ),
        "record_key_strategy": "CANONICAL_ENTITY_ID",
        "record_key_source_field": "runtime_session_id",
        "record_key_source_location": "upstream_payload",
    },
}


def _freeze_json(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PersistenceRecordError("JSON numbers must be finite")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PersistenceRecordError("JSON object keys must be strings")
            frozen[key] = _freeze_json(item)
        return MappingProxyType(frozen)
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    raise PersistenceRecordError(f"value of type {type(value).__name__} is not in JSON domain")


def _thaw_json(value: object) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _fingerprint(value: object) -> str:
    return canonical_json_sha256(_thaw_json(value))


@dataclass(frozen=True, slots=True)
class PersistenceRecord:
    """A deeply immutable candidate/integrity persistence carrier."""

    representation_name: str
    representation_category: str
    semantic_owner_milestone: str
    semantic_artifact: str
    semantic_json_pointer: str
    semantic_contract_fingerprint_sha256: str
    record_key: str
    payload: object
    payload_fingerprint_sha256: str

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "representation_name",
        "representation_category",
        "semantic_owner_milestone",
        "semantic_artifact",
        "semantic_json_pointer",
        "semantic_contract_fingerprint_sha256",
        "record_key",
        "payload",
        "payload_fingerprint_sha256",
    )

    def __post_init__(self) -> None:
        for field in self._FIELDS[:7]:
            value = getattr(self, field)
            if not isinstance(value, str):
                raise PersistenceRecordError(f"{field} must be a string")
        if not self.record_key:
            raise PersistenceRecordError("record_key must be non-empty")
        for field in (
            "semantic_contract_fingerprint_sha256",
            "payload_fingerprint_sha256",
        ):
            if _SHA256_RE.fullmatch(getattr(self, field)) is None:
                raise PersistenceRecordError(f"{field} must be lowercase SHA-256 hexadecimal")
        object.__setattr__(self, "payload", _freeze_json(self.payload))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PersistenceRecord:
        """Parse an exact carrier without deriving or repairing caller values."""

        if not isinstance(value, Mapping):
            raise PersistenceRecordError("PersistenceRecord input must be a Mapping")
        if set(value) != set(cls._FIELDS):
            raise PersistenceRecordError("PersistenceRecord requires its exact field set")
        return cls(**{field: value[field] for field in cls._FIELDS})  # type: ignore[arg-type]

    def to_mapping(self) -> dict[str, object]:
        """Return a fresh mutable, JSON-compatible carrier mapping."""

        return {
            "representation_name": self.representation_name,
            "representation_category": self.representation_category,
            "semantic_owner_milestone": self.semantic_owner_milestone,
            "semantic_artifact": self.semantic_artifact,
            "semantic_json_pointer": self.semantic_json_pointer,
            "semantic_contract_fingerprint_sha256": self.semantic_contract_fingerprint_sha256,
            "record_key": self.record_key,
            "payload": _thaw_json(self.payload),
            "payload_fingerprint_sha256": self.payload_fingerprint_sha256,
        }


def _require_exact_mapping(value: object, fields: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise PersistenceRecordError(f"{label} must be an object with its exact field set")
    return value


def _require_canonical_id(value: object, prefix: str, label: str) -> str:
    if not isinstance(value, str):
        raise PersistenceRecordError(f"{label} must be a canonical ID string")
    match = _CANONICAL_ID_RE.fullmatch(value)
    if match is None or match.group("prefix") != prefix:
        raise PersistenceRecordError(f"{label} must be a canonical {prefix} UUIDv7 ID")
    return value


def _validate_account(record: PersistenceRecord) -> None:
    payload = _require_exact_mapping(
        record.payload, {"entity_kind", "entity_id", "parent_scope_bindings"}, "payload"
    )
    if payload["entity_kind"] != "CryptoHunterAccount":
        raise PersistenceRecordError("entity_kind must be CryptoHunterAccount")
    entity_id = _require_canonical_id(payload["entity_id"], "acct", "entity_id")
    if (
        not isinstance(payload["parent_scope_bindings"], Mapping)
        or payload["parent_scope_bindings"]
    ):
        raise PersistenceRecordError("root CryptoHunterAccount parent_scope_bindings must be {}")
    if record.record_key != entity_id:
        raise PersistenceRecordError("record_key must equal payload.entity_id")


def _validate_runtime_session(record: PersistenceRecord) -> None:
    payload = _require_exact_mapping(
        record.payload,
        {"fact_kind", "upstream_payload", "upstream_payload_fingerprint_sha256"},
        "payload",
    )
    if payload["fact_kind"] != "RuntimeSession":
        raise PersistenceRecordError("fact_kind must be RuntimeSession")
    upstream = _require_exact_mapping(
        payload["upstream_payload"],
        {"runtime_session_id", "device_installation_id"},
        "upstream_payload",
    )
    runtime_session_id = _require_canonical_id(
        upstream["runtime_session_id"], "run", "runtime_session_id"
    )
    _require_canonical_id(upstream["device_installation_id"], "dev", "device_installation_id")
    upstream_sha = payload["upstream_payload_fingerprint_sha256"]
    if not isinstance(upstream_sha, str) or _SHA256_RE.fullmatch(upstream_sha) is None:
        raise PersistenceRecordError(
            "upstream_payload_fingerprint_sha256 must be lowercase SHA-256"
        )
    if upstream_sha != _fingerprint(upstream):
        raise PersistenceRecordError("upstream payload fingerprint mismatch")
    if record.record_key != runtime_session_id:
        raise PersistenceRecordError("record_key must equal upstream_payload.runtime_session_id")


def validate_persistence_record(record: PersistenceRecord) -> None:
    """Validate Stage 1 intrinsic carrier integrity, without establishing authority."""

    if not isinstance(record, PersistenceRecord):
        raise PersistenceRecordError("record must be a PersistenceRecord")
    registry = _REGISTRY.get(record.representation_name)
    if registry is None:
        raise PersistenceRecordError(
            "representation unsupported by current production implementation"
        )
    for field in (
        "representation_category",
        "semantic_owner_milestone",
        "semantic_artifact",
        "semantic_json_pointer",
        "semantic_contract_fingerprint_sha256",
    ):
        if getattr(record, field) != registry[field]:
            raise PersistenceRecordError(f"registry binding mismatch: {field}")
    if record.representation_name == "CryptoHunterAccount current record":
        _validate_account(record)
    else:
        _validate_runtime_session(record)
    if record.payload_fingerprint_sha256 != _fingerprint(record.payload):
        raise PersistenceRecordError("payload fingerprint mismatch")
