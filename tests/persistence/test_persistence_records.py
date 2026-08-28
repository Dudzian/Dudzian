from __future__ import annotations

from dataclasses import fields
from hashlib import sha256
import json
import math
from pathlib import Path
from types import MappingProxyType

import pytest

from bot_core.persistence import (
    PersistenceRecord,
    PersistenceRecordError,
    validate_persistence_record,
)
from bot_core.persistence import records as production


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


def test_workspace_is_architecturally_valid_but_unsupported_in_s2a() -> None:
    mapping = account_mapping(representation_name="Workspace")
    with pytest.raises(
        PersistenceRecordError, match="unsupported by current production implementation"
    ):
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
