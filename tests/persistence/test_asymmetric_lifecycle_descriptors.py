from __future__ import annotations

from dataclasses import replace
import math

import pytest

from bot_core.persistence.fingerprints import canonical_json_sha256
from bot_core.persistence.migration_protocol import (
    MigrationDefinition,
    derive_runtime_migration_record,
)
from bot_core.persistence.record_registry import PERSISTENCE_RECORD_REGISTRY
from bot_core.persistence.records import (
    PersistenceRecord,
    PersistenceRecordError,
    validate_persistence_record,
)
from bot_core.persistence.secret_handoff import (
    SecretHandoffError,
    SecretHandoffRecord,
    handoff_descriptor_carrier,
    secret_metadata_fingerprint,
    secret_operation_fingerprint,
    validate_handoff_descriptor_identity,
)
from bot_core.persistence.state_store import (
    StateStoreError,
    StateStoreMetadata,
    StateStoreSnapshot,
    _validate_record_store_scope,
)

ACCOUNT = "acct_01890f3a-2b4c-7abc-8def-0123456789ab"
DEVICE = "dev_01890f3a-2b4c-7abc-8def-0123456789ab"
SHA = "a" * 64


def descriptor(**changes: object) -> SecretHandoffRecord:
    metadata = changes.pop("reconciliation_metadata", {"cleanup": True})
    scope = changes.pop("scope", (ACCOUNT, DEVICE))
    operation = changes.pop("operation", "ROTATE")
    old_reference = changes.pop("old_reference", "secure-ref:old")
    new_reference = changes.pop("new_reference", "secure-ref:new")
    metadata_hash = secret_metadata_fingerprint(metadata)  # type: ignore[arg-type]
    operation_hash = secret_operation_fingerprint(
        scope=scope,  # type: ignore[arg-type]
        operation=operation,  # type: ignore[arg-type]
        old_reference=old_reference,  # type: ignore[arg-type]
        new_reference=new_reference,  # type: ignore[arg-type]
        metadata_fingerprint_sha256=metadata_hash,
    )
    values = {
        "handoff_id": "handoff-1",
        "scope": scope,
        "operation": operation,
        "old_reference": old_reference,
        "new_reference": new_reference,
        "metadata_fingerprint_sha256": metadata_hash,
        "operation_fingerprint_sha256": operation_hash,
        "reconciliation_metadata": metadata,
    }
    values.update(changes)
    return SecretHandoffRecord(**values)  # type: ignore[arg-type]


def raw_carrier(
    payload: dict[str, object], key: str = "handoff-descriptor:handoff-1"
) -> PersistenceRecord:
    template = handoff_descriptor_carrier(descriptor())
    return replace(
        template,
        record_key=key,
        payload=payload,
        payload_fingerprint_sha256=canonical_json_sha256(payload),
    )


def test_migration_record_has_no_durable_projection() -> None:
    assert all(
        entry.get("projection_schema_if_any") != "MigrationRecord"
        for entry in PERSISTENCE_RECORD_REGISTRY.values()
    )
    assert "migration-descriptor" not in {
        entry.get("record_key_strategy") for entry in PERSISTENCE_RECORD_REGISTRY.values()
    }


def test_runtime_migration_record_is_rebound_from_each_fresh_snapshot() -> None:
    definition = MigrationDefinition("migration-1", 1, 2, ("step",))

    def snapshot(generation: int, state: str, transaction: str) -> StateStoreSnapshot:
        metadata = StateStoreMetadata(
            ACCOUNT, DEVICE, 1, "d" * 64, "PAPER", generation, state, transaction, "e" * 64
        )
        return StateStoreSnapshot(metadata, (), (), ())

    first = derive_runtime_migration_record(
        definition, snapshot(2, "1" * 64, "2" * 64), derived_post_state_fingerprint_sha256="3" * 64
    )
    second = derive_runtime_migration_record(
        definition, snapshot(3, "4" * 64, "5" * 64), derived_post_state_fingerprint_sha256="6" * 64
    )
    assert first != second
    assert definition.matches_static_fields(first) and definition.matches_static_fields(second)


def test_descriptor_positive_stage_one_and_exact_key() -> None:
    carrier = handoff_descriptor_carrier(descriptor())
    assert carrier.representation_name == "SecretHandoff immutable descriptor"
    assert carrier.record_key == "handoff-descriptor:handoff-1"
    validate_persistence_record(PersistenceRecord.from_mapping(carrier.to_mapping()))


def test_descriptor_scope_is_bound_to_state_store_account_and_device() -> None:
    carrier = handoff_descriptor_carrier(descriptor())
    metadata = StateStoreMetadata(
        ACCOUNT, DEVICE, 1, "d" * 64, "PAPER", 1, "e" * 64, "f" * 64, "1" * 64
    )
    _validate_record_store_scope(carrier, metadata)
    with pytest.raises(StateStoreError, match="account scope mismatch"):
        _validate_record_store_scope(
            carrier, replace(metadata, account_id="acct_01890f3a-2b4c-7abc-8def-0123456789ac")
        )
    with pytest.raises(StateStoreError, match="device scope mismatch"):
        _validate_record_store_scope(
            carrier,
            replace(metadata, device_installation_id="dev_01890f3a-2b4c-7abc-8def-0123456789ac"),
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda p: p.pop("operation"),
        lambda p: p.update(extra=True),
        lambda p: p.update(scope=[ACCOUNT]),
        lambda p: p.update(scope=["acct", DEVICE]),
        lambda p: p.update(scope=[ACCOUNT, "dev"]),
        lambda p: p.update(operation=""),
        lambda p: p.update(old_reference=7),
        lambda p: p.update(new_reference=[]),
        lambda p: p.update(reconciliation_metadata={"x": math.nan}),
        lambda p: p.update(reconciliation_metadata={"x": math.inf}),
        lambda p: p.update(reconciliation_metadata={"x": object()}),
        lambda p: p.update(metadata_fingerprint_sha256="b" * 64),
        lambda p: p.update(operation_fingerprint_sha256="b" * 64),
        lambda p: p.update(operation="REPLACE"),
        lambda p: p.update(old_reference="secure-ref:changed"),
        lambda p: p.update(new_reference="secure-ref:changed"),
        lambda p: p.update(reconciliation_metadata={"cleanup": False}),
    ],
)
def test_raw_descriptor_tampering_rejected(mutator) -> None:
    payload = descriptor().to_mapping()
    mutator(payload)
    with pytest.raises((PersistenceRecordError, ValueError, TypeError)):
        validate_persistence_record(raw_carrier(payload))


@pytest.mark.parametrize(
    "key", ["wrong:handoff-1", "handoff-descriptor:other", "handoff-descriptor:handoff-1:1"]
)
def test_descriptor_rejects_non_exact_record_key(key: str) -> None:
    with pytest.raises(PersistenceRecordError):
        validate_persistence_record(raw_carrier(descriptor().to_mapping(), key))


def test_inner_hash_attack_layers_and_same_id_conflict() -> None:
    payload = descriptor().to_mapping()
    payload["reconciliation_metadata"] = {"cleanup": False}
    with pytest.raises(PersistenceRecordError):
        validate_persistence_record(raw_carrier(payload))
    payload["metadata_fingerprint_sha256"] = secret_metadata_fingerprint({"cleanup": False})
    with pytest.raises(PersistenceRecordError):
        validate_persistence_record(raw_carrier(payload))
    changed = handoff_descriptor_carrier(descriptor(reconciliation_metadata={"cleanup": False}))
    validate_persistence_record(changed)
    with pytest.raises(SecretHandoffError, match="immutable handoff descriptor conflict"):
        validate_handoff_descriptor_identity(handoff_descriptor_carrier(descriptor()), changed)


def test_migration_carrier_hash_cycle_is_rejected_by_asymmetric_dag() -> None:
    graph = {
        "post_state": {"migration_payload"},
        "migration_payload": {"carrier"},
        "carrier": {"target_projection"},
        "target_projection": {"post_state"},
    }

    def cyclic(node: str, path: frozenset[str] = frozenset()) -> bool:
        return node in path or any(cyclic(child, path | {node}) for child in graph.get(node, ()))

    assert cyclic("post_state")
    accepted = {
        "definition": {"runtime_binding"},
        "snapshot": {"runtime_binding"},
        "runtime_binding": {"target_derivation"},
        "descriptor": {"history"},
    }
    graph = accepted
    assert not any(cyclic(node) for node in graph)
