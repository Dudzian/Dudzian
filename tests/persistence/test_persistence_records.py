from __future__ import annotations

from dataclasses import fields
from hashlib import sha256
import inspect
import json
import math
import unicodedata
from pathlib import Path
from types import MappingProxyType

import pytest

from bot_core.persistence import (
    PersistenceRecord,
    PersistenceRecordError,
    validate_persistence_record,
)
from bot_core.persistence import records as production
from bot_core.persistence.record_registry import (
    DIRECT_SEMANTIC_CONSTRAINTS,
    DIRECT_UPSTREAM_VALIDATORS,
    LOCAL_SCHEMA_CONTRACTS,
    PERSISTENCE_RECORD_REGISTRY,
    STATE_STORE_SCOPE_BINDINGS,
)
from tests.architecture import (
    test_cryptohunter_persistence_versioning_migrations_backup_and_recovery as frozen_oracle,
)


ARCHITECTURE = Path("docs/architecture/cryptohunter_product_architecture")
M011 = json.loads(
    (ARCHITECTURE / "persistence_versioning_migrations_backup_and_recovery.json").read_text()
)
M02 = json.loads((ARCHITECTURE / "canonical_domain_vocabulary.json").read_text())
REQUIRED = M011["executable_boundary_schemas"]["PersistenceRecord"]["required"]
CANONICAL_REGISTRY = M011["backup_contract"]["representation_registry"]
ACCOUNT_ID = "acct_01890f4c-7b9a-7cc1-8a2b-123456789abc"
OTHER_ACCOUNT_ID = "acct_01890f4c-7b9a-7cc1-8a2b-123456789abd"
SESSION_ID = "run_01890f4c-7b9a-7cc1-8a2b-123456789abc"
DEVICE_ID = "dev_01890f4c-7b9a-7cc1-8a2b-123456789abc"
ScopePath = tuple[str | int, ...]
_INDEXED_ACCOUNT_DEVICE_SCOPE_RULE = (
    "scope is exactly [canonical account_id, canonical device_installation_id]"
)


def fingerprint(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def account_mapping(**changes: object) -> dict[str, object]:
    payload = {
        "entity_kind": "CryptoHunterAccount",
        "entity_id": ACCOUNT_ID,
        "parent_scope_bindings": {},
    }
    result: dict[str, object] = {
        "representation_name": "CryptoHunterAccount current record",
        "representation_category": "M011_ENTITY_IDENTITY_PROJECTION",
        "semantic_owner_milestone": "M0.2",
        "semantic_artifact": "canonical_domain_vocabulary.json",
        "semantic_json_pointer": "/entity_kinds",
        "semantic_contract_fingerprint_sha256": "1913a18c7e7479d9c20850a690ee81b9f311a7d08cf2458374c91f6332d277f1",
        "record_key": ACCOUNT_ID,
        "payload": payload,
        "payload_fingerprint_sha256": fingerprint(payload),
    }
    result.update(changes)
    return result


def runtime_mapping(**changes: object) -> dict[str, object]:
    upstream = {"runtime_session_id": SESSION_ID, "device_installation_id": DEVICE_ID}
    payload = {
        "fact_kind": "RuntimeSession",
        "upstream_payload": upstream,
        "upstream_payload_fingerprint_sha256": fingerprint(upstream),
    }
    result: dict[str, object] = {
        "representation_name": "RuntimeSession canonical identity/history",
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.2",
        "semantic_artifact": "canonical_domain_vocabulary.json",
        "semantic_json_pointer": "/entity_kinds",
        "semantic_contract_fingerprint_sha256": "1913a18c7e7479d9c20850a690ee81b9f311a7d08cf2458374c91f6332d277f1",
        "record_key": SESSION_ID,
        "payload": payload,
        "payload_fingerprint_sha256": fingerprint(payload),
    }
    result.update(changes)
    return result


def parsed(mapping: dict[str, object]) -> PersistenceRecord:
    return PersistenceRecord.from_mapping(mapping)


def self_consistent(mapping: dict[str, object], payload: dict[str, object]) -> PersistenceRecord:
    mapping["payload"] = payload
    mapping["payload_fingerprint_sha256"] = fingerprint(payload)
    return parsed(mapping)


def test_valid_account_and_runtime_records_pass_stage_1() -> None:
    assert validate_persistence_record(parsed(account_mapping())) is None
    assert validate_persistence_record(parsed(runtime_mapping())) is None


FROZEN_PERSISTENCE_NAMES = tuple(
    name
    for name, entry in CANONICAL_REGISTRY.items()
    if entry.get("carrier_strategy") == "PERSISTENCE_RECORD"
)


def test_production_registry_exactly_covers_frozen_persistence_records() -> None:
    assert set(production._REGISTRY) == set(FROZEN_PERSISTENCE_NAMES)


def test_static_registry_has_deep_exact_parity_with_frozen_registry() -> None:
    expected = {
        name: entry
        for name, entry in CANONICAL_REGISTRY.items()
        if entry.get("carrier_strategy") == "PERSISTENCE_RECORD"
    }
    assert PERSISTENCE_RECORD_REGISTRY == expected
    assert (
        DIRECT_UPSTREAM_VALIDATORS == M011["backup_contract"]["direct_upstream_validator_registry"]
    )
    assert LOCAL_SCHEMA_CONTRACTS == {
        name: M011["executable_boundary_schemas"][name] for name in LOCAL_SCHEMA_CONTRACTS
    }
    assert set(STATE_STORE_SCOPE_BINDINGS) == set(expected)


def test_resolved_direct_constraints_are_deeply_source_derived() -> None:
    assert (
        DIRECT_UPSTREAM_VALIDATORS == M011["backup_contract"]["direct_upstream_validator_registry"]
    )


def test_scope_binding_table_has_exact_source_derived_paths() -> None:
    expected: dict[str, dict[str, tuple[ScopePath, ...]]] = {}
    direct = M011["backup_contract"]["direct_upstream_validator_registry"]
    for name in FROZEN_PERSISTENCE_NAMES:
        entry = CANONICAL_REGISTRY[name]
        account: list[ScopePath] = []
        device: list[ScopePath] = []
        category = entry["representation_category"]
        if name == "CryptoHunterAccount current record":
            account.append(("entity_id",))
        elif category == "M011_IMMUTABLE_HISTORY_WRAPPER" and entry.get("immutable_fact_binding"):
            for field in entry["immutable_fact_binding"].get("scope_fields", []):
                if field == "account_id":
                    account.append(("upstream_payload", field))
                elif field == "device_installation_id":
                    device.append(("upstream_payload", field))
        elif category == "M011_PERSISTENCE_PROJECTION_OF_UPSTREAM_FACTS":
            required = entry["fact_binding"]["required_fact_fields"]
            if "account_id" in required:
                account.append(("facts", "account_id"))
            if "device_installation_id" in required:
                device.append(("facts", "device_installation_id"))
        elif category == "DIRECT_UPSTREAM_SCHEMA":
            exact = direct[name]["exact_fields"]
            if "account_id" in exact:
                account.append(("account_id",))
            if "device_installation_id" in exact:
                device.append(("device_installation_id",))
        elif category == "M011_LOCAL_SCHEMA":
            schema = LOCAL_SCHEMA_CONTRACTS[entry["projection_schema_if_any"]]
            properties = schema.get("properties", {})
            if "account_id" in properties:
                account.append(("account_id",))
            if "device_installation_id" in properties:
                device.append(("device_installation_id",))
            if _INDEXED_ACCOUNT_DEVICE_SCOPE_RULE in schema.get("intrinsic_constraints", ()):
                account.append(("scope", 0))
                device.append(("scope", 1))
        if name in {"bootstrap consumed fence", "bootstrap accepted/consumption history"}:
            account = [("account_id",)]
            device = [("device_installation_id",)]
        expected[name] = {"account_paths": tuple(account), "device_paths": tuple(device)}
    assert expected["SecretHandoff immutable descriptor"] == {
        "account_paths": (("scope", 0),),
        "device_paths": (("scope", 1),),
    }
    assert (
        STATE_STORE_SCOPE_BINDINGS["SecretHandoff immutable descriptor"]
        == expected["SecretHandoff immutable descriptor"]
    )
    assert STATE_STORE_SCOPE_BINDINGS == expected


@pytest.mark.parametrize("name", FROZEN_PERSISTENCE_NAMES)
def test_every_frozen_persistence_record_has_a_valid_stage_1_fixture(name: str) -> None:
    mapping = frozen_oracle._persistence_record(name)
    if name == "RiskPolicy accepted revisions":
        payload = mapping["payload"]
        upstream = payload["upstream_payload"]
        upstream["limits"][0][2]["mapping_status"] = "EXACT"
        binding = CANONICAL_REGISTRY[name]["immutable_fact_binding"]
        upstream["semantic_fingerprint_sha256"] = frozen_oracle._semantic_fingerprint(
            binding, upstream
        )
        payload["upstream_payload_fingerprint_sha256"] = fingerprint(upstream)
        mapping["payload_fingerprint_sha256"] = fingerprint(payload)
    validate_persistence_record(PersistenceRecord.from_mapping(mapping))


@pytest.mark.parametrize("name", FROZEN_PERSISTENCE_NAMES)
def test_every_frozen_persistence_record_rejects_wrong_semantic_binding(name: str) -> None:
    mapping = frozen_oracle._persistence_record(name)
    mapping["semantic_contract_fingerprint_sha256"] = "0" * 64
    with pytest.raises(PersistenceRecordError, match="registry binding mismatch"):
        validate_persistence_record(PersistenceRecord.from_mapping(mapping))


LOCAL_NAMES = tuple(
    name
    for name, entry in CANONICAL_REGISTRY.items()
    if entry.get("representation_category") == "M011_LOCAL_SCHEMA"
)


def _self_consistent_record(name: str, payload: dict[str, object]) -> PersistenceRecord:
    mapping = frozen_oracle._persistence_record(name, payload)
    mapping["record_key"] = frozen_oracle._derive_record_key(
        name, CANONICAL_REGISTRY[name], payload
    )
    mapping["payload_fingerprint_sha256"] = fingerprint(payload)
    return PersistenceRecord.from_mapping(mapping)


def _rehash_immutable(name: str, mapping: dict[str, object]) -> PersistenceRecord:
    payload = mapping["payload"]
    upstream = payload["upstream_payload"]
    binding = CANONICAL_REGISTRY[name]["immutable_fact_binding"]
    semantic = binding
    if "upstream_payload_variants" in binding:
        semantic = binding["upstream_payload_variants"][
            upstream[binding["upstream_payload_discriminator"]]
        ]
    terminal = semantic.get("semantic_fingerprint_field")
    if terminal:
        upstream[terminal] = production._semantic_fingerprint(semantic, upstream)
    payload["upstream_payload_fingerprint_sha256"] = fingerprint(upstream)
    mapping["record_key"] = frozen_oracle._derive_record_key(
        name, CANONICAL_REGISTRY[name], payload
    )
    mapping["payload_fingerprint_sha256"] = fingerprint(payload)
    return PersistenceRecord.from_mapping(mapping)


def _inventory() -> tuple[set[str], set[str]]:
    types: set[str] = set()
    shapes: set[str] = set()

    def walk(value: object) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("type"), str):
                types.add(value["type"])
            if isinstance(value.get("input_shape"), str):
                shapes.add(value["input_shape"])
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(PERSISTENCE_RECORD_REGISTRY)
    walk(DIRECT_UPSTREAM_VALIDATORS)
    walk(LOCAL_SCHEMA_CONTRACTS)
    return types, shapes


def test_every_frozen_field_type_and_fingerprint_shape_has_executor_capability() -> None:
    types, shapes = _inventory()
    assert types == set(production.FIELD_VALIDATOR_CAPABILITIES)
    assert shapes == set(production.SEMANTIC_FINGERPRINT_SHAPE_CAPABILITIES)


@pytest.mark.parametrize(
    ("name", "scope_type", "scope_id"),
    [
        ("RiskPolicy accepted revisions", "WORKSPACE", "port_01890f3a-2b4c-7abc-8def-0123456789ab"),
        ("RiskPolicy accepted revisions", "PRODUCT_SYSTEM", "not-product"),
        ("kill-switch transition history", "WORKSPACE", "garbage"),
    ],
)
def test_canonical_scope_id_rejects_self_consistent_wrong_scope(
    name: str, scope_type: str, scope_id: str
) -> None:
    mapping = frozen_oracle._persistence_record(name)
    upstream = mapping["payload"]["upstream_payload"]
    upstream["scope_type"] = scope_type
    upstream["scope_id"] = scope_id
    with pytest.raises(PersistenceRecordError, match="frozen schema"):
        validate_persistence_record(_rehash_immutable(name, mapping))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda limits: limits.__setitem__(0, ["UNKNOWN_LIMIT", "1/1", limits[0][2]]),
        lambda limits: limits.__setitem__(0, ["MAX_ORDER_QUANTITY", "1/0", limits[0][2]]),
        lambda limits: limits.__setitem__(0, ["MAX_ORDER_QUANTITY", "2/2", limits[0][2]]),
        lambda limits: limits[0][2].pop("asset_namespace"),
        lambda limits: limits[0][2].update(extra="x"),
        lambda limits: limits[0][2].update(mapping_status="UNKNOWN"),
        lambda limits: limits[0][2].update(venue_asset_code=1),
        lambda limits: limits.__setitem__(0, ["MAX_ORDER_QUANTITY"]),
    ],
)
def test_risk_limits_reject_semantic_attacks_after_full_rehash(mutate) -> None:  # type: ignore[no-untyped-def]
    name = "RiskPolicy accepted revisions"
    mapping = frozen_oracle._persistence_record(name)
    limits = mapping["payload"]["upstream_payload"]["limits"]
    limits[0][2]["mapping_status"] = "EXACT"
    mutate(limits)
    with pytest.raises(PersistenceRecordError, match="frozen schema"):
        validate_persistence_record(_rehash_immutable(name, mapping))


def test_compound_scope_binds_risk_budget_components_after_full_rehash() -> None:
    name = "RiskBudget current state"
    payload = frozen_oracle._payload_for(name)
    payload["facts"]["risk_scope_key"] = "different|but|non-empty"
    with pytest.raises(PersistenceRecordError, match="facts scope or fields"):
        validate_persistence_record(_self_consistent_record(name, payload))


def _reconciliation_event(name: str) -> PersistenceRecord:
    mapping = frozen_oracle._persistence_record(name)
    upstream = mapping["payload"]["upstream_payload"] if name != "Event" else mapping["payload"]
    upstream["event_type"] = "ORDER_RECONCILIATION_OBSERVED"
    upstream["safe_payload"] = {"trusted_fact_kind": "REJECTED", "venue_order_id": None}
    if name == "Event":
        upstream["event_fingerprint_sha256"] = fingerprint(
            {key: value for key, value in upstream.items() if key != "event_fingerprint_sha256"}
        )
        return _self_consistent_record(name, upstream)
    return _rehash_immutable(name, mapping)


@pytest.mark.parametrize("name", ["Event", "Order lifecycle events/history"])
def test_nullable_reconciliation_venue_order_id_is_accepted(name: str) -> None:
    validate_persistence_record(_reconciliation_event(name))


def test_immutable_event_reuses_exact_safe_payload_registry() -> None:
    name = "Order lifecycle events/history"
    mapping = frozen_oracle._persistence_record(name)
    upstream = mapping["payload"]["upstream_payload"]
    upstream["event_type"] = "ORDER_FILLED"
    upstream["safe_payload"] = {"garbage": "still non-empty"}
    with pytest.raises(PersistenceRecordError, match="frozen schema"):
        validate_persistence_record(_rehash_immutable(name, mapping))


def test_nfc_semantic_fingerprint_normalizes_nested_unicode_values() -> None:
    name = "Order lifecycle events/history"
    mapping = frozen_oracle._persistence_record(name)
    upstream = mapping["payload"]["upstream_payload"]
    upstream["event_type"] = "ORDER_REJECTED"
    decomposed = "odrzucone-e\u0301"
    upstream["safe_payload"] = {"reason_code": decomposed}
    binding = CANONICAL_REGISTRY[name]["immutable_fact_binding"]
    expected = production._semantic_fingerprint(binding, upstream)
    generic = fingerprint(
        {
            field: upstream[field]
            for field in binding["semantic_fingerprint_derivation"]["input_fields"]
        }
    )
    assert unicodedata.normalize("NFC", decomposed) != decomposed
    assert expected != generic
    validate_persistence_record(_rehash_immutable(name, mapping))


@pytest.mark.parametrize("name", LOCAL_NAMES)
def test_every_local_schema_rejects_self_consistent_invalid_state_enum(name: str) -> None:
    payload = frozen_oracle._payload_for(name)
    payload["state"] = "BANANA"
    with pytest.raises(PersistenceRecordError, match="local payload"):
        validate_persistence_record(_self_consistent_record(name, payload))


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("current_transition_revision", True),
        ("current_transition_revision", 0),
        ("designation_fingerprint_sha256", "x" * 64),
        ("migration_id", 123),
    ],
)
def test_local_schema_rejects_self_consistent_scalar_attacks(mutation: str, value: object) -> None:
    name = "Migration current state/designation"
    payload = frozen_oracle._payload_for(name)
    payload[mutation] = value
    with pytest.raises(PersistenceRecordError, match="local payload"):
        validate_persistence_record(_self_consistent_record(name, payload))


@pytest.mark.parametrize("shape", ["extra", "missing"])
def test_local_schema_remains_closed_after_full_rehash(shape: str) -> None:
    name = "SecretHandoff current state/designation"
    payload = frozen_oracle._payload_for(name)
    if shape == "extra":
        payload["extra"] = True
    else:
        del payload["operation_fingerprint_sha256"]
    with pytest.raises((PersistenceRecordError, KeyError)):
        validate_persistence_record(_self_consistent_record(name, payload))


@pytest.mark.parametrize(
    ("name", "field", "invalid"),
    [
        ("Event", "event_type", "UNKNOWN_EVENT"),
        ("LedgerEntry", "direction", "SIDEWAYS"),
        ("LedgerEntry", "posting_role", "UNKNOWN_ROLE"),
        ("RiskDecision", "decision", "MAYBE"),
        ("SessionSecurityState current generation/state", "state", "PAUSED"),
        ("SecretMetadataProjection", "state", "HIDDEN"),
        ("SecretMetadataProjection", "secret_reference", "not-secure-store"),
    ],
)
def test_direct_semantic_constraints_reject_self_consistent_values(
    name: str, field: str, invalid: object
) -> None:
    payload = frozen_oracle._payload_for(name)
    payload[field] = invalid
    terminal = frozen_oracle._terminal_fingerprint_field(name)
    if terminal:
        payload[terminal] = fingerprint({k: v for k, v in payload.items() if k != terminal})
    with pytest.raises(PersistenceRecordError):
        validate_persistence_record(_self_consistent_record(name, payload))


def test_event_safe_payload_registry_is_exact_after_full_rehash() -> None:
    name = "Event"
    payload = frozen_oracle._payload_for(name)
    payload["safe_payload"] = {"arbitrary": "non-empty"}
    payload["event_fingerprint_sha256"] = fingerprint(
        {k: v for k, v in payload.items() if k != "event_fingerprint_sha256"}
    )
    with pytest.raises(PersistenceRecordError, match="safe_payload"):
        validate_persistence_record(_self_consistent_record(name, payload))


def test_fill_conditional_fee_semantics_survive_full_rehash() -> None:
    name = "Fill"
    payload = frozen_oracle._payload_for(name)
    payload["fee_kind"] = "NONE"
    payload["fee_quantity"] = "1"
    payload["fill_fingerprint_sha256"] = fingerprint(
        {k: v for k, v in payload.items() if k != "fill_fingerprint_sha256"}
    )
    with pytest.raises(PersistenceRecordError, match="fee semantics"):
        validate_persistence_record(_self_consistent_record(name, payload))


def test_carrier_fields_bind_exactly_to_canonical_schema() -> None:
    schema = M011["executable_boundary_schemas"]["PersistenceRecord"]
    assert [field.name for field in fields(PersistenceRecord)] == REQUIRED
    assert schema["additionalProperties"] is False
    assert (
        schema["properties"]["semantic_contract_fingerprint_sha256"]["pattern"] == "^[0-9a-f]{64}$"
    )
    assert schema["properties"]["payload_fingerprint_sha256"]["pattern"] == "^[0-9a-f]{64}$"
    assert schema["properties"]["payload"] == {}


def test_generic_fingerprint_contract_binding() -> None:
    derivation = M011["backup_contract"]["persistence_record_contract"][
        "payload_fingerprint_sha256_derivation"
    ]
    assert derivation == {
        "algorithm": "SHA-256",
        "input": "exact PersistenceRecord.payload JSON value",
        "encoding": "UTF-8",
        "canonical_json": {
            "sort_keys": True,
            "separators": [",", ":"],
            "ensure_ascii": False,
            "allow_nan": False,
        },
        "digest_encoding": "lowercase hexadecimal",
        "domain_separator": None,
        "excluded_inputs": [
            "PersistenceRecord fields other than payload",
            "timestamp",
            "representation_name",
            "record_key",
        ],
        "meaning": "INTEGRITY_ONLY",
        "establishes_domain_authority": False,
        "establishes_accepted_or_current_membership": False,
    }


def test_event_shadow_support_fragment_exactly_attests_canonical_upstream() -> None:
    upstream = json.loads(
        (ARCHITECTURE / "commands_events_order_lifecycle_and_idempotency.json").read_text()
    )
    assert (
        DIRECT_SEMANTIC_CONSTRAINTS["Event"]["safe_payload"]
        == upstream["event_contract"]["event_schema_registry"]
    )


@pytest.mark.parametrize(
    "name", ["CryptoHunterAccount current record", "RuntimeSession canonical identity/history"]
)
def test_production_registry_literals_bind_to_canonical_registry(name: str) -> None:
    canonical = CANONICAL_REGISTRY[name]
    literal = production._REGISTRY[name]
    for key in literal:
        assert literal[key] == canonical[key]


def test_account_contract_binds_to_m02() -> None:
    registry = CANONICAL_REGISTRY["CryptoHunterAccount current record"]
    entity = next(
        item for item in M02["entity_kinds"] if item["canonical_name"] == "CryptoHunterAccount"
    )
    projection = M011["executable_boundary_schemas"]["PersistentEntityIdentityProjection"]
    assert (
        registry["payload_contract"]["required_fields"]
        == projection["required"]
        == ["entity_kind", "entity_id", "parent_scope_bindings"]
    )
    assert entity["id_prefix"] == "acct"
    assert entity["parent"] == "none"
    assert registry["record_key_strategy"] == "CANONICAL_ENTITY_ID"
    assert registry["record_key_source_field"] == "entity_id"
    assert registry["record_key_source_location"] == "payload"


def test_runtime_contract_binds_to_m02_and_m011_amendment() -> None:
    registry = CANONICAL_REGISTRY["RuntimeSession canonical identity/history"]
    binding = registry["immutable_fact_binding"]
    runtime = next(
        item for item in M02["entity_kinds"] if item["canonical_name"] == "RuntimeSession"
    )
    device = next(
        item for item in M02["entity_kinds"] if item["canonical_name"] == "DeviceInstallation"
    )
    assert registry["payload_contract"]["required_fields"] == [
        "fact_kind",
        "upstream_payload",
        "upstream_payload_fingerprint_sha256",
    ]
    assert (
        registry["payload_contract"]["fact_kind_literal"]
        == binding["fact_kind_literal"]
        == "RuntimeSession"
    )
    assert binding["persisted_payload_fields"] == ["runtime_session_id", "device_installation_id"]
    assert runtime["id_prefix"] == "run" and device["id_prefix"] == "dev"
    assert {
        "from": "DeviceInstallation",
        "to": "RuntimeSession",
        "cardinality": "one_to_many",
    } in M02["relationships"]
    assert binding["upstream_fingerprint_literal"] is False
    derivation = binding["upstream_payload_fingerprint_sha256_derivation"]
    assert derivation["origin"] == "M0.11 wrapper integrity"
    assert derivation["canonical_json"] == {
        "sort_keys": True,
        "separators": [",", ":"],
        "ensure_ascii": False,
        "allow_nan": False,
    }
    assert derivation["authority"] is False and derivation["establishes_membership"] is False
    assert registry["record_key_strategy"] == "CANONICAL_ENTITY_ID"
    assert registry["record_key_source_field"] == "runtime_session_id"
    assert registry["record_key_source_location"] == "upstream_payload"
    assert binding["revision_generation_fields"] == []


def test_input_and_output_mutations_cannot_change_snapshot() -> None:
    source = runtime_mapping()
    record = parsed(source)
    payload = source["payload"]
    assert isinstance(payload, dict)
    upstream = payload["upstream_payload"]
    assert isinstance(upstream, dict)
    payload["fact_kind"] = "changed"
    upstream["runtime_session_id"] = "changed"
    assert record.to_mapping()["payload"] != payload
    output = record.to_mapping()
    output_payload = output["payload"]
    assert isinstance(output_payload, dict)
    output_upstream = output_payload["upstream_payload"]
    assert isinstance(output_upstream, dict)
    output_upstream["runtime_session_id"] = "again"
    assert record.to_mapping()["payload"] != output_payload
    assert isinstance(record.payload, MappingProxyType)
    with pytest.raises(TypeError):
        record.payload["fact_kind"] = "no"  # type: ignore[index]


@pytest.mark.parametrize("value", [None, [], "record", 1])
def test_from_mapping_requires_mapping(value: object) -> None:
    with pytest.raises(PersistenceRecordError):
        PersistenceRecord.from_mapping(value)  # type: ignore[arg-type]


def test_missing_and_extra_top_level_fields_fail() -> None:
    missing = account_mapping()
    missing.pop("record_key")
    extra = account_mapping(extra="value")
    with pytest.raises(PersistenceRecordError):
        parsed(missing)
    with pytest.raises(PersistenceRecordError):
        parsed(extra)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("record_key", ""),
        ("semantic_contract_fingerprint_sha256", "x" * 64),
        ("semantic_contract_fingerprint_sha256", "A" * 64),
        ("payload_fingerprint_sha256", "x" * 64),
        ("payload_fingerprint_sha256", "A" * 64),
        ("representation_category", 1),
    ],
)
def test_carrier_shape_rejects_invalid_text_and_sha(field: str, value: object) -> None:
    with pytest.raises(PersistenceRecordError):
        parsed(account_mapping(**{field: value}))


@pytest.mark.parametrize(
    "bad",
    [
        {1: "value"},
        {"x": object()},
        {"x": (1, 2)},
        {"x": {1, 2}},
        {"x": float("nan")},
        {"x": float("inf")},
        {"x": -float("inf")},
    ],
)
def test_strict_json_domain_rejects_non_json_values(bad: object) -> None:
    with pytest.raises(PersistenceRecordError):
        parsed(account_mapping(payload=bad))


def test_json_semantics_do_not_coerce_bool_integer_or_string() -> None:
    record = parsed(account_mapping(payload={"one": 1, "true": True, "text": "1", "array": [1]}))
    assert record.to_mapping()["payload"] == {"one": 1, "true": True, "text": "1", "array": [1]}
    assert math.isfinite(1.0)


def test_workspace_name_does_not_relax_its_exact_entity_binding() -> None:
    mapping = account_mapping(representation_name="Workspace")
    with pytest.raises(PersistenceRecordError, match="entity_kind binding mismatch"):
        validate_persistence_record(parsed(mapping))


@pytest.mark.parametrize(
    "field",
    [
        "representation_category",
        "semantic_owner_milestone",
        "semantic_artifact",
        "semantic_json_pointer",
    ],
)
def test_exact_registry_binding_rejects_changed_metadata(field: str) -> None:
    with pytest.raises(PersistenceRecordError, match="registry binding"):
        validate_persistence_record(parsed(account_mapping(**{field: "changed"})))


def test_registry_and_outer_fingerprint_reject_syntactically_valid_wrong_sha() -> None:
    for field in ("semantic_contract_fingerprint_sha256", "payload_fingerprint_sha256"):
        with pytest.raises(PersistenceRecordError):
            validate_persistence_record(parsed(account_mapping(**{field: "0" * 64})))


@pytest.mark.parametrize(
    ("payload_change", "key"),
    [
        ({"entity_kind": "Workspace"}, ACCOUNT_ID),
        ({"entity_id": "invalid"}, "invalid"),
        ({"entity_id": SESSION_ID}, SESSION_ID),
        ({"parent_scope_bindings": {"account_id": ACCOUNT_ID}}, ACCOUNT_ID),
    ],
)
def test_account_semantic_failures_are_not_masked_by_self_consistency(
    payload_change: dict[str, object], key: str
) -> None:
    payload = account_mapping()["payload"]
    assert isinstance(payload, dict)
    payload.update(payload_change)
    with pytest.raises(PersistenceRecordError):
        validate_persistence_record(self_consistent(account_mapping(record_key=key), payload))


def test_account_requires_exact_payload_fields() -> None:
    for mutate in (lambda p: p.pop("entity_kind"), lambda p: p.update(extra=True)):
        payload = account_mapping()["payload"]
        assert isinstance(payload, dict)
        mutate(payload)
        with pytest.raises(PersistenceRecordError):
            validate_persistence_record(self_consistent(account_mapping(), payload))


@pytest.mark.parametrize(
    "key", [f"entity:CryptoHunterAccount:{ACCOUNT_ID}", "arbitrary", OTHER_ACCOUNT_ID]
)
def test_account_record_key_is_exact_payload_entity_id(key: str) -> None:
    with pytest.raises(PersistenceRecordError):
        validate_persistence_record(parsed(account_mapping(record_key=key)))


@pytest.mark.parametrize(
    ("change", "key"),
    [
        ({"fact_kind": "RuntimeSession canonical identity/history"}, SESSION_ID),
        ({"runtime_session_id": "invalid"}, "invalid"),
        ({"runtime_session_id": ACCOUNT_ID}, ACCOUNT_ID),
        ({"device_installation_id": "invalid"}, SESSION_ID),
        ({"device_installation_id": ACCOUNT_ID}, SESSION_ID),
    ],
)
def test_runtime_semantic_failures_are_not_masked_by_self_consistency(
    change: dict[str, object], key: str
) -> None:
    payload = runtime_mapping()["payload"]
    assert isinstance(payload, dict)
    if "fact_kind" in change:
        payload.update(change)
    else:
        upstream = payload["upstream_payload"]
        assert isinstance(upstream, dict)
        upstream.update(change)
        payload["upstream_payload_fingerprint_sha256"] = fingerprint(upstream)
    with pytest.raises(PersistenceRecordError):
        validate_persistence_record(self_consistent(runtime_mapping(record_key=key), payload))


def test_runtime_requires_exact_wrapper_and_upstream_fields() -> None:
    mutations = [
        lambda p: p.pop("fact_kind"),
        lambda p: p.update(extra=True),
        lambda p: p["upstream_payload"].pop("runtime_session_id"),
        lambda p: p["upstream_payload"].update(extra=True),
    ]
    for mutate in mutations:
        payload = runtime_mapping()["payload"]
        assert isinstance(payload, dict)
        mutate(payload)
        upstream = payload.get("upstream_payload")
        if isinstance(upstream, dict):
            payload["upstream_payload_fingerprint_sha256"] = fingerprint(upstream)
        with pytest.raises(PersistenceRecordError):
            validate_persistence_record(self_consistent(runtime_mapping(), payload))


def test_runtime_recomputes_both_fingerprints() -> None:
    payload = runtime_mapping()["payload"]
    assert isinstance(payload, dict)
    for bad in ("0" * 64, "1" * 64):
        payload["upstream_payload_fingerprint_sha256"] = bad
        with pytest.raises(PersistenceRecordError):
            validate_persistence_record(self_consistent(runtime_mapping(), payload))
    with pytest.raises(PersistenceRecordError):
        validate_persistence_record(parsed(runtime_mapping(payload_fingerprint_sha256="0" * 64)))


@pytest.mark.parametrize(
    "key", ["arbitrary", f"immutable:RuntimeSession:{SESSION_ID}", f"{SESSION_ID}:{DEVICE_ID}"]
)
def test_runtime_record_key_is_exact_session_id(key: str) -> None:
    with pytest.raises(PersistenceRecordError):
        validate_persistence_record(parsed(runtime_mapping(record_key=key)))


def test_production_does_not_load_architecture_docs() -> None:
    source = Path(production.__file__).read_text()
    assert "docs/architecture" not in source
    assert "Path(" not in source


@pytest.mark.parametrize(
    ("created", "retired", "valid"),
    [
        ("2026-01-01T00:00:00.123456700Z", "2026-01-01T00:00:00.123456701Z", True),
        ("2026-01-01T00:00:00.123456701Z", "2026-01-01T00:00:00.123456700Z", False),
        ("2026-01-01T00:00:00.123456789Z", "2026-01-01T00:00:00.123456789Z", True),
        ("2026-01-01T00:00:00.1234567Z", "2026-01-01T00:00:00.123456700Z", True),
        ("2026-01-01T00:00:00.000000002Z", "2026-01-01T00:00:00.000000001Z", False),
    ],
)
def test_exchange_account_nanosecond_retirement_ordering(
    created: str, retired: str, valid: bool
) -> None:
    payload = frozen_oracle._payload_for("ExchangeAccount")
    payload.update(
        lifecycle_state="RETIRED", created_at_utc=created, retired_at_utc=retired, display_name=""
    )
    record = _self_consistent_record("ExchangeAccount", payload)
    if valid:
        validate_persistence_record(record)
    else:
        with pytest.raises(PersistenceRecordError):
            validate_persistence_record(record)


@pytest.mark.parametrize(
    ("created", "retired", "valid"),
    [
        ("2026-01-01T00:00:00.123456700Z", "2026-01-01T00:00:00.123456701Z", True),
        ("2026-01-01T00:00:00.123456701Z", "2026-01-01T00:00:00.123456700Z", False),
        ("2026-01-01T00:00:00.123456789Z", "2026-01-01T00:00:00.123456789Z", True),
        ("2026-01-01T00:00:00.000000002Z", "2026-01-01T00:00:00.000000001Z", False),
    ],
)
def test_credential_profile_nanosecond_retirement_ordering(
    created: str, retired: str, valid: bool
) -> None:
    name = "CredentialProfile metadata/reference"
    payload = frozen_oracle._payload_for(name)
    payload.update(lifecycle_state="RETIRED", created_at_utc=created, retired_at_utc=retired)
    record = _self_consistent_record(name, payload)
    if valid:
        validate_persistence_record(record)
    else:
        with pytest.raises(PersistenceRecordError):
            validate_persistence_record(record)


@pytest.mark.parametrize("mutation", ["missing", "true"])
def test_credential_profile_requires_exact_false_saas_sync_candidate(mutation: str) -> None:
    name = "CredentialProfile metadata/reference"
    payload = frozen_oracle._payload_for(name)
    if mutation == "missing":
        payload.pop("saas_sync_candidate")
    else:
        payload["saas_sync_candidate"] = True
    with pytest.raises(PersistenceRecordError):
        validate_persistence_record(_self_consistent_record(name, payload))


@pytest.mark.parametrize(
    "updates",
    [
        {"exchange_id": "paper_simulated_venue", "environment": "PAPER", "market_type": "SPOT"},
        {"exchange_id": "generic_testnet_venue", "environment": "TESTNET", "market_type": "SPOT"},
        {
            "exchange_id": "generic_testnet_venue",
            "environment": "TESTNET",
            "market_type": "PERPETUAL",
        },
        {"display_name": ""},
        {"external_account_identity_state": "VERIFIED"},
        {"lifecycle_state": "DRAFT", "retired_at_utc": None},
        {"lifecycle_state": "ACTIVE", "retired_at_utc": None},
        {"lifecycle_state": "DISABLED", "retired_at_utc": None},
        {
            "lifecycle_state": "RETIRED",
            "created_at_utc": "2026-01-01T00:00:00Z",
            "retired_at_utc": "2026-01-01T00:00:00.1Z",
        },
    ],
)
def test_exchange_account_corrected_m05_positive_matrix(updates: dict[str, object]) -> None:
    payload = frozen_oracle._payload_for("ExchangeAccount")
    payload.update(updates)
    validate_persistence_record(_self_consistent_record("ExchangeAccount", payload))


@pytest.mark.parametrize(
    "updates",
    [
        {"exchange_id": "unknown_exchange"},
        {"exchange_id": "generic_testnet_venue", "environment": "PAPER", "market_type": "SPOT"},
        {"exchange_id": "generic_testnet_venue", "environment": "TESTNET", "market_type": "MARGIN"},
        {"market_type": "UNKNOWN_MARKET"},
        {"external_account_identity_state": {"canonical_reference": "invented"}},
        {"external_account_identity_state": "UNKNOWN_STATE"},
        {"lifecycle_state": "DRAFT", "retired_at_utc": "2026-01-01T00:00:00Z"},
        {"lifecycle_state": "ACTIVE", "retired_at_utc": "2026-01-01T00:00:00Z"},
        {"lifecycle_state": "DISABLED", "retired_at_utc": "2026-01-01T00:00:00Z"},
        {"lifecycle_state": "RETIRED", "retired_at_utc": None},
    ],
)
def test_exchange_account_corrected_m05_adversarial_matrix(updates: dict[str, object]) -> None:
    payload = frozen_oracle._payload_for("ExchangeAccount")
    payload.update(updates)
    with pytest.raises(PersistenceRecordError):
        validate_persistence_record(_self_consistent_record("ExchangeAccount", payload))


def test_exchange_account_unknown_lifecycle_rule_fails_closed() -> None:
    assert not production._validate_exchange_account_lifecycle_rule(
        "INVENTED_RULE", created="2026-01-01T00:00:00Z", retired=None
    )


def test_exchange_account_lifecycle_executor_has_no_state_name_semantic_group() -> None:
    source = inspect.getsource(production._validate_direct_semantics)
    assert 'state in {"DRAFT", "ACTIVE", "DISABLED"}' not in source


@pytest.mark.parametrize(
    "updates",
    [
        {"rotated_from_credential_profile_id": "acct_01890f3a-2b4c-7abc-8def-0123456789ab"},
        {
            "credential_profile_id": "cred_01890f3a-2b4c-7abc-8def-0123456789ab",
            "rotated_from_credential_profile_id": "cred_01890f3a-2b4c-7abc-8def-0123456789ab",
        },
        {"permission_snapshot": ["READ_ACCOUNT", "READ_ACCOUNT"]},
        {"permission_snapshot": ["UNKNOWN_PERMISSION"]},
        {"secure_store_reference": "keyring://wrong"},
        {"secure_store_reference": "secure-store://secret-token"},
        {"lifecycle_state": "ACTIVE", "retired_at_utc": "2026-01-01T00:00:00Z"},
        {"lifecycle_state": "RETIRED", "retired_at_utc": None},
    ],
)
def test_credential_profile_corrected_m05_adversarial_matrix(updates: dict[str, object]) -> None:
    name = "CredentialProfile metadata/reference"
    payload = frozen_oracle._payload_for(name)
    payload.update(updates)
    with pytest.raises(PersistenceRecordError):
        validate_persistence_record(_self_consistent_record(name, payload))
