"""Intrinsic validation of candidate persistence records.

The types in this module are integrity carriers only.  Validation performed here
does not confer domain authority, accepted/current membership, or restore status.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from datetime import datetime
from hashlib import sha256
from types import MappingProxyType
from typing import Any, ClassVar

from .fingerprints import canonical_json, canonical_json_sha256
from .record_registry import (
    DIRECT_SEMANTIC_CONSTRAINTS,
    DIRECT_UPSTREAM_VALIDATORS,
    LOCAL_SCHEMA_CONTRACTS,
    PERSISTENCE_RECORD_REGISTRY,
)


class PersistenceRecordError(ValueError):
    """Raised when a persistence carrier fails intrinsic validation."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_ID_RE = re.compile(
    r"^(?P<prefix>[a-z][a-z0-9]*)_"
    r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)

_LEGACY_REGISTRY = {
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

# Public-to-the-persistence-kernel static registry used by StateStore and tests.
_REGISTRY = PERSISTENCE_RECORD_REGISTRY

FIELD_VALIDATOR_CAPABILITIES = frozenset(
    {
        "array",
        "array_of_canonical_id",
        "asset_reference",
        "boolean",
        "canonical_id",
        "canonical_scope_id",
        "compound_scope",
        "constant",
        "decimal",
        "enum",
        "event_safe_payload",
        "id",
        "integer",
        "non_empty_string",
        "non_negative_integer",
        "nullable_canonical_id",
        "nullable_timestamp",
        "object",
        "positive_decimal",
        "positive_integer",
        "risk_limits",
        "sha256_hex",
        "string",
        "timestamp",
    }
)
SEMANTIC_FINGERPRINT_SHAPE_CAPABILITIES = frozenset(
    {
        "JSON_OBJECT",
        "CANONICAL_NFC_JSON_OBJECT",
        "ORDERED_CANONICAL_JSON_ARRAY",
        "DOMAIN_SEPARATOR_NEWLINE_CANONICAL_JSON_OBJECT",
        "DOMAIN_SEPARATOR_BYTES_PLUS_CANONICAL_JSON_VALUE",
    }
)


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
    if isinstance(value, list):
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


def _valid_field(value: object, contract: Mapping[str, object]) -> bool:
    kind = contract.get("type")
    if kind == "constant":
        return value == contract.get("value")
    if kind in {"canonical_id", "id"}:
        prefix = contract.get("id_prefix", contract.get("prefix"))
        return (
            isinstance(prefix, str)
            and isinstance(value, str)
            and bool(
                (match := _CANONICAL_ID_RE.fullmatch(value)) and match.group("prefix") == prefix
            )
        )
    if kind == "positive_integer":
        return isinstance(value, int) and not isinstance(value, bool) and value > 0
    if kind == "non_negative_integer":
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "sha256_hex":
        return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None
    if kind == "enum":
        return value in contract.get("values", ())
    if kind in {"decimal", "positive_decimal", "non_negative_decimal"}:
        if (
            not isinstance(value, str)
            or re.fullmatch(r"^(0|[1-9][0-9]*)(\.[0-9]*[1-9])?$", value) is None
        ):
            return False
        try:
            number = Decimal(value)
        except InvalidOperation:
            return False
        constraint = contract.get("constraint", kind.removesuffix("_decimal"))
        return (constraint != "positive" or number > 0) and (
            constraint != "non_negative" or number >= 0
        )
    if kind == "timestamp":
        if (
            not isinstance(value, str)
            or re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]{1,9})?Z",
                value,
            )
            is None
        ):
            return False
        try:
            datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
        except ValueError:
            return False
        return True
    if kind == "nullable_timestamp":
        return value is None or _valid_field(value, {"type": "timestamp"})
    if kind == "nullable_canonical_id":
        return value is None or _valid_field(
            value, {"type": "canonical_id", "id_prefix": contract.get("id_prefix")}
        )
    if kind == "array":
        return isinstance(value, (list, tuple))
    if kind == "array_of_canonical_id":
        return isinstance(value, (list, tuple)) and all(
            _valid_field(item, {"type": "canonical_id", "id_prefix": contract.get("id_prefix")})
            for item in value
        )
    if kind in {"object", "asset_reference"}:
        fields = contract.get("fields")
        schemas = contract.get("field_schemas")
        return (
            isinstance(value, Mapping)
            and isinstance(fields, list)
            and isinstance(schemas, Mapping)
            and set(value) == set(fields)
            and all(_valid_field(value[name], schemas[name]) for name in fields)
        )
    if kind in {"non_empty_object", "event_safe_payload"}:
        return isinstance(value, Mapping) and bool(value)
    if kind in {"non_empty_string", "string"}:
        return isinstance(value, str) and bool(value)
    return False


_SCOPE_PREFIXES = {
    "PRODUCT_SYSTEM": None,
    "WORKSPACE": "ws",
    "PORTFOLIO": "port",
    "EXCHANGE_ACCOUNT": "xacc",
    "STRATEGY_INSTANCE": "sinst",
    "INSTRUMENT": "instr",
    "EXECUTION_ROUTE": "xroute",
}


def _validate_event_safe_payload(event_type: object, safe: object) -> bool:
    schemas = DIRECT_SEMANTIC_CONSTRAINTS["Event"]["safe_payload"]
    event_schema = schemas.get(event_type) if isinstance(event_type, str) else None
    if not isinstance(event_schema, Mapping) or not isinstance(safe, Mapping):
        return False
    fields = event_schema["safe_payload_fields"]
    nullable = set(event_schema["nullable_fields"])
    return set(safe) == set(fields) and all(
        (safe[field] is None and field in nullable)
        or (
            safe[field] is not None
            and _valid_field(safe[field], event_schema["field_schemas"][field])
        )
        for field in fields
    )


def _validate_risk_limits(value: object, contract: Mapping[str, object]) -> bool:
    if not isinstance(value, (list, tuple)) or not value:
        return False
    names = contract["supported_limit_names"]
    asset_fields = set(contract["asset_reference_fields"])
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 3 or item[0] not in names:
            return False
        fraction = item[1]
        if (
            not isinstance(fraction, str)
            or re.fullmatch(r"-?(0|[1-9][0-9]*)/[1-9][0-9]*", fraction) is None
        ):
            return False
        numerator, denominator = map(int, fraction.split("/"))
        if denominator <= 0 or __import__("math").gcd(abs(numerator), denominator) != 1:
            return False
        asset = item[2]
        if not isinstance(asset, Mapping) or set(asset) != asset_fields:
            return False
        if not all(isinstance(asset[field], str) and asset[field] for field in asset_fields):
            return False
        if asset["mapping_status"] not in {"EXACT", "EXPLICIT_ALIAS"}:
            return False
    return True


def _valid_context_field(
    field: str,
    value: object,
    contract: Mapping[str, object],
    payload: Mapping[str, object],
) -> bool:
    kind = contract.get("type")
    if kind == "canonical_scope_id":
        scope_type = payload.get(str(contract["scope_type_field"]))
        prefix = _SCOPE_PREFIXES.get(scope_type)
        if scope_type == "PRODUCT_SYSTEM":
            return value == "product"
        return (
            isinstance(prefix, str)
            and isinstance(value, str)
            and bool(
                (match := _CANONICAL_ID_RE.fullmatch(value)) and match.group("prefix") == prefix
            )
        )
    if kind == "compound_scope":
        return value == "|".join(str(payload[name]) for name in contract["components"])
    if kind == "risk_limits":
        return _validate_risk_limits(value, contract)
    if kind == "event_safe_payload":
        return _validate_event_safe_payload(payload.get("event_type"), value)
    return _valid_field(value, contract)


def _nfc_json(value: object) -> object:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized:
                raise PersistenceRecordError("NFC object-key collision")
            normalized[normalized_key] = _nfc_json(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_nfc_json(item) for item in value]
    return value


def _semantic_fingerprint(semantic: Mapping[str, Any], upstream: Mapping[str, object]) -> str:
    derivation = semantic["semantic_fingerprint_derivation"]
    shape = derivation["input_shape"]
    if shape not in SEMANTIC_FINGERPRINT_SHAPE_CAPABILITIES:
        raise PersistenceRecordError("unsupported semantic fingerprint input_shape")
    fields = derivation["input_fields"]
    if shape == "JSON_OBJECT":
        return _fingerprint({field: upstream[field] for field in fields})
    if shape == "CANONICAL_NFC_JSON_OBJECT":
        return _fingerprint(_nfc_json({field: upstream[field] for field in fields}))
    if shape == "ORDERED_CANONICAL_JSON_ARRAY":
        return _fingerprint([upstream[field] for field in fields])
    if shape == "DOMAIN_SEPARATOR_NEWLINE_CANONICAL_JSON_OBJECT":
        projected = {field: upstream[field] for field in fields}
        for field in ("instrument_ids", "source_catalog_snapshot_ids"):
            if isinstance(projected.get(field), tuple):
                projected[field] = sorted(projected[field])
        return sha256(
            f"{derivation['domain_separator']}\n{canonical_json(_thaw_json(projected))}".encode()
        ).hexdigest()
    if shape == "DOMAIN_SEPARATOR_BYTES_PLUS_CANONICAL_JSON_VALUE":
        return sha256(
            derivation["domain_separator"].encode()
            + canonical_json(_thaw_json(upstream["configuration"])).encode()
        ).hexdigest()
    raise AssertionError("unreachable semantic fingerprint shape")


def _validate_local_schema(value: object, schema: Mapping[str, object]) -> bool:
    if isinstance(schema, str):
        if schema == "canonical M0.2 AccountId":
            return isinstance(value, str) and bool(
                (match := _CANONICAL_ID_RE.fullmatch(value)) and match.group("prefix") == "acct"
            )
        if schema == "canonical M0.2 DeviceInstallationId":
            return isinstance(value, str) and bool(
                (match := _CANONICAL_ID_RE.fullmatch(value)) and match.group("prefix") == "dev"
            )
        return False
    if "enum" in schema:
        return value in schema["enum"]
    kind = schema.get("type")
    if kind == "object":
        required = schema.get("required", ())
        properties = schema.get("properties", {})
        return (
            isinstance(value, Mapping)
            and set(value) == set(required)
            and isinstance(properties, Mapping)
            and all(_validate_local_schema(value[name], properties[name]) for name in required)
        )
    if kind == "integer":
        return (
            isinstance(value, int)
            and (schema.get("boolean_allowed", True) or not isinstance(value, bool))
            and value >= schema.get("minimum", value)
        )
    if kind == "string":
        return isinstance(value, str) and (
            "pattern" not in schema or re.fullmatch(str(schema["pattern"]), value) is not None
        )
    return False


def _validate_direct_semantics(name: str, payload: Mapping[str, object]) -> None:
    constraints = DIRECT_SEMANTIC_CONSTRAINTS.get(name, {})
    for field, expected in constraints.items():
        if field in {"safe_payload", "fee_semantics", "secret_reference"}:
            continue
        if isinstance(expected, (list, tuple)) and payload[field] not in expected:
            raise PersistenceRecordError(f"direct upstream semantic constraint failed: {field}")
    if name == "Event":
        if not _validate_event_safe_payload(payload["event_type"], payload["safe_payload"]):
            raise PersistenceRecordError("Event safe_payload violates event registry")
    elif name == "Fill":
        fee_kind = payload["fee_kind"]
        quantity = payload["fee_quantity"]
        reference = payload["fee_asset_reference"]
        if fee_kind == "NONE" and (quantity != "0" or reference is not None):
            raise PersistenceRecordError("Fill NONE fee semantics violated")
        if fee_kind == "CHARGE":
            if not isinstance(reference, Mapping) or Decimal(str(quantity)) <= 0:
                raise PersistenceRecordError("Fill CHARGE fee semantics violated")
            if reference.get("mapping_status") not in {"EXACT", "EXPLICIT_ALIAS"}:
                raise PersistenceRecordError("Fill fee mapping_status invalid")
            if reference.get("asset_namespace") != payload["exchange_id"]:
                raise PersistenceRecordError("Fill fee asset namespace mismatch")
    elif name == "SecretMetadataProjection":
        policy = constraints["secret_reference"]
        reference = payload["secret_reference"]
        if (
            not isinstance(reference, str)
            or not reference.startswith(policy["prefix"])
            or not reference.removeprefix(policy["prefix"])
            or any(character.isspace() or character in "?#=" for character in reference)
            or any(marker in reference.casefold() for marker in policy["forbidden_payload_markers"])
        ):
            raise PersistenceRecordError("secret_reference violates opaque-reference policy")


def _derive_record_key(name: str, entry: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    strategy = entry["record_key_strategy"]
    if strategy == "DIRECT_UPSTREAM_KEY_FIELDS":
        fields = DIRECT_UPSTREAM_VALIDATORS[name]["record_key_fields"]
        return f"direct:{name}:" + ":".join(str(payload[field]) for field in fields)
    if strategy == "IMMUTABLE_PAYLOAD_IDENTITY_REVISION":
        binding = entry["immutable_fact_binding"]
        upstream = payload["upstream_payload"]
        fields = binding["canonical_object_identity_fields"] + binding["revision_generation_fields"]
        return f"immutable:{payload['fact_kind']}:" + ":".join(
            str(upstream[field]) for field in fields
        )
    if strategy == "CANONICAL_ENTITY_ID":
        location = entry["record_key_source_location"]
        source = payload if location == "payload" else payload["upstream_payload"]
        return str(source[entry["record_key_source_field"]])
    if strategy == "SCOPE_CURRENT_REFERENCE_REVISION_GENERATION":
        return f"current:{payload['scope_key']}:{payload['current_reference']}:{payload['current_revision']}:{payload['current_generation']}"
    if strategy == "FACT_SCOPE_OBJECT_GENERATION":
        facts = payload["facts"]
        fields = entry["fact_binding"]["record_key_object_fields"]
        return f"facts:{payload['fact_kind']}:" + ":".join(str(facts[field]) for field in fields)
    templates = {
        "BOOTSTRAP_SCOPE_STATE_REVISION": "bootstrap-current:{account_id}:{device_installation_id}:{state_revision}",
        "BOOTSTRAP_SCOPE_GENERATION_REVISION_CLAIM": "bootstrap-history:{account_id}:{device_installation_id}:{bootstrap_generation}:{bootstrap_revision}:{claim_fingerprint_sha256}",
        "STATE_STORE_SCOPE_GENERATION": "state-store:{account_id}:{device_installation_id}:{state_store_identity_fingerprint_sha256}:{protected_freshness_generation}",
        "MIGRATION_ID_TRANSITION_REVISION": "migration-transition:{migration_id}:{transition_revision}",
        "MIGRATION_ID_CURRENT_REVISION": "migration-current:{migration_id}:{current_transition_revision}",
        "HANDOFF_ID_TRANSITION_REVISION": "handoff-transition:{handoff_id}:{transition_revision}",
        "HANDOFF_ID_CURRENT_REVISION": "handoff-current:{handoff_id}:{current_transition_revision}",
    }
    if strategy == "CANONICAL_OBJECT_ID_REVISION":
        return f"object:{payload['semantic_object']}:{payload['object_id']}:{payload['revision']}"
    return templates[strategy].format(**payload)


def _validate_immutable(name: str, entry: Mapping[str, Any], payload: object) -> None:
    binding = entry["immutable_fact_binding"]
    wrapper_fields = binding["wrapper_fields"]
    value = _require_exact_mapping(payload, set(wrapper_fields), "payload")
    if value["fact_kind"] != binding.get(
        "fact_kind_literal", entry["semantic_object_or_invariant"]
    ):
        raise PersistenceRecordError("fact_kind binding mismatch")
    upstream = value["upstream_payload"]
    if not isinstance(upstream, Mapping):
        raise PersistenceRecordError("upstream_payload must be an object")
    semantic = binding
    if "upstream_payload_variants" in binding:
        discriminator = binding["upstream_payload_discriminator"]
        semantic = binding["upstream_payload_variants"].get(upstream.get(discriminator))
        if not isinstance(semantic, Mapping):
            raise PersistenceRecordError("unknown immutable payload variant")
    fields = semantic["persisted_payload_fields"]
    contracts = semantic["field_contracts"]
    if set(upstream) != set(fields) or not all(
        _valid_context_field(field, upstream[field], contracts[field], upstream) for field in fields
    ):
        raise PersistenceRecordError("upstream payload violates its exact frozen schema")
    terminal = semantic.get("semantic_fingerprint_field")
    if isinstance(terminal, str):
        expected = _semantic_fingerprint(semantic, upstream)
        if upstream[terminal] != expected:
            raise PersistenceRecordError("semantic content fingerprint mismatch")
    if value["upstream_payload_fingerprint_sha256"] != _fingerprint(upstream):
        raise PersistenceRecordError("upstream payload fingerprint mismatch")


def _validate_generic(record: PersistenceRecord, entry: Mapping[str, Any]) -> None:
    payload = record.payload
    category = entry["representation_category"]
    if record.representation_name == "bootstrap consumed fence":
        value = _require_exact_mapping(
            payload,
            {
                "state_fingerprint_sha256",
                "account_id",
                "device_installation_id",
                "intended_operator_id",
                "startup_readiness",
                "initial_security_lifecycle",
                "first_operator_presence",
                "expected_generation",
                "expected_revision",
                "consumed_authorities",
                "state_revision",
            },
            "payload",
        )
        _require_canonical_id(value["account_id"], "acct", "account_id")
        _require_canonical_id(value["device_installation_id"], "dev", "device_installation_id")
    elif record.representation_name == "bootstrap accepted/consumption history":
        value = _require_exact_mapping(
            payload,
            {
                "account_id",
                "device_installation_id",
                "bootstrap_generation",
                "bootstrap_revision",
                "claim_fingerprint_sha256",
                "challenge_fingerprint_sha256",
            },
            "payload",
        )
        _require_canonical_id(value["account_id"], "acct", "account_id")
        _require_canonical_id(value["device_installation_id"], "dev", "device_installation_id")
    elif category == "M011_ENTITY_IDENTITY_PROJECTION":
        value = _require_exact_mapping(
            payload, {"entity_kind", "entity_id", "parent_scope_bindings"}, "payload"
        )
        expected_kind = (
            "CryptoHunterAccount"
            if record.representation_name.startswith("CryptoHunterAccount")
            else "Workspace"
        )
        prefix = "acct" if expected_kind == "CryptoHunterAccount" else "ws"
        if value["entity_kind"] != expected_kind:
            raise PersistenceRecordError("entity_kind binding mismatch")
        _require_canonical_id(value["entity_id"], prefix, "entity_id")
    elif category == "M011_CURRENT_DESIGNATION_PROJECTION":
        value = _require_exact_mapping(
            payload,
            {
                "scope_key",
                "current_reference",
                "current_revision",
                "current_generation",
                "content_fingerprint_sha256",
            },
            "payload",
        )
        if not all(
            isinstance(value[field], str) and value[field]
            for field in ("scope_key", "current_reference")
        ):
            raise PersistenceRecordError("designation strings must be non-empty")
        for field in ("current_revision", "current_generation"):
            if value[field] is not None and not _valid_field(
                value[field], {"type": "positive_integer"}
            ):
                raise PersistenceRecordError("designation revision/generation invalid")
        if not _valid_field(value["content_fingerprint_sha256"], {"type": "sha256_hex"}):
            raise PersistenceRecordError("designation content fingerprint invalid")
    elif category == "M011_PERSISTENCE_PROJECTION_OF_UPSTREAM_FACTS":
        value = _require_exact_mapping(
            payload, {"fact_kind", "scope_key", "facts", "source_fingerprint_sha256"}, "payload"
        )
        binding = entry["fact_binding"]
        facts = value["facts"]
        if not isinstance(facts, Mapping) or set(facts) != set(binding["required_fact_fields"]):
            raise PersistenceRecordError("facts require their exact field set")
        if (
            value["fact_kind"] != record.representation_name
            or value["source_fingerprint_sha256"] != entry["semantic_contract_fingerprint_sha256"]
        ):
            raise PersistenceRecordError("facts source binding mismatch")
        if value["scope_key"] != facts[binding["scope_binding"]] or not all(
            _valid_context_field(field, facts[field], binding["field_contracts"][field], facts)
            for field in facts
        ):
            raise PersistenceRecordError("facts scope or fields invalid")
    elif category == "M011_IMMUTABLE_HISTORY_WRAPPER":
        _validate_immutable(record.representation_name, entry, payload)
    elif category == "DIRECT_UPSTREAM_SCHEMA":
        contract = DIRECT_UPSTREAM_VALIDATORS[record.representation_name]
        value = _require_exact_mapping(payload, set(contract["exact_fields"]), "payload")
        nullable = set(contract["nullable_fields"])
        if not all(
            (value[field] is None and field in nullable)
            or (
                value[field] is not None
                and _valid_field(value[field], contract["upstream_field_schemas"][field])
            )
            for field in value
        ):
            raise PersistenceRecordError("direct upstream payload invalid")
        _validate_direct_semantics(record.representation_name, value)
        terminal = {
            "Event": "event_fingerprint_sha256",
            "Fill": "fill_fingerprint_sha256",
            "kill-switch state/generation": "record_fingerprint_sha256",
            "RiskDecision": "decision_fingerprint_sha256",
            "ExecutionLease immutable record": "lease_fingerprint_sha256",
            "SessionSecurityState current generation/state": "content_fingerprint_sha256",
            "SecretMetadataProjection": "content_fingerprint_sha256",
        }.get(record.representation_name)
        if terminal and value[terminal] != _fingerprint(
            {key: item for key, item in value.items() if key != terminal}
        ):
            raise PersistenceRecordError("direct upstream terminal fingerprint mismatch")
    elif category == "M011_LOCAL_SCHEMA":
        schema = LOCAL_SCHEMA_CONTRACTS[entry["projection_schema_if_any"]]
        if not _validate_local_schema(payload, schema):
            raise PersistenceRecordError("local payload violates exact frozen schema")
    else:
        raise PersistenceRecordError("unsupported representation category")
    if record.record_key != _derive_record_key(record.representation_name, entry, payload):
        raise PersistenceRecordError("record_key derivation mismatch")


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
    elif record.representation_name == "RuntimeSession canonical identity/history":
        _validate_runtime_session(record)
    else:
        _validate_generic(record, registry)
    if record.payload_fingerprint_sha256 != _fingerprint(record.payload):
        raise PersistenceRecordError("payload fingerprint mismatch")


def record_durability_class(record: PersistenceRecord) -> str:
    """Return the exact frozen StateStore bucket for a Stage-1 representation."""

    try:
        return str(_REGISTRY[record.representation_name]["durability_class"])
    except KeyError as exc:
        raise PersistenceRecordError("record representation has no durable bucket") from exc


def validate_record_bucket(record: PersistenceRecord, expected: str) -> None:
    validate_persistence_record(record)
    if record_durability_class(record) != expected:
        raise PersistenceRecordError("record is in the wrong durable bucket")
