"""Executable authority proofs for the acyclic M0.11 lifecycle corrective."""

from __future__ import annotations

import json
from pathlib import Path

from bot_core.persistence.fingerprints import (
    canonical_json_sha256,
    history_tail_fingerprint_sha256,
    state_fingerprint_sha256,
    transaction_fingerprint_sha256,
)
from bot_core.persistence.migration_protocol import (
    migration_current,
    migration_current_carrier,
    migration_transition,
    migration_transition_carrier,
)
from bot_core.persistence.secret_handoff import handoff_current_carrier, handoff_transition_carrier

DOC = Path(
    "docs/architecture/cryptohunter_product_architecture/persistence_versioning_migrations_backup_and_recovery.json"
)
MACHINE = json.loads(DOC.read_text())
ACCOUNT = "acct_01890f3a-2b4c-7abc-8def-0123456789ab"
DEVICE = "dev_01890f3a-2b4c-7abc-8def-0123456789ab"
IDENTITY = canonical_json_sha256({"store": "corrective-ii"})


def _has_cycle(graph: dict[str, set[str]]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(visit(child) for child in graph.get(node, set())):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in graph)


def test_historical_target_hash_bindings_are_structural_cycles() -> None:
    transaction_cycle = {
        "target_transaction_fingerprint": {"descriptor_projection"},
        "descriptor_projection": {"transition_carrier"},
        "transition_carrier": {"target_transaction_fingerprint"},
    }
    state_cycle = {
        "target_state_fingerprint": {"state_or_history_projection"},
        "state_or_history_projection": {"transition_carrier"},
        "transition_carrier": {"target_state_fingerprint"},
    }
    assert _has_cycle(transaction_cycle) and _has_cycle(state_cycle)
    assert MACHINE["migration_protocol"]["no_persisted_fingerprint_self_reference"][
        "failure"
    ].startswith("CONTRACT INVALID")


def test_corrective_dependency_graph_is_a_dag() -> None:
    order = MACHINE["migration_protocol"]["hash_dependency_dag"]["topological_order"]
    graph = {item: {order[index + 1]} for index, item in enumerate(order[:-1])}
    assert len(order) == len(set(order)) and not _has_cycle(graph)
    assert MACHINE["migration_protocol"]["hash_dependency_dag"]["back_edges_allowed"] is False


def test_real_hash_construction_is_strictly_forward() -> None:
    source_state = canonical_json_sha256({"verified": "source-state", "records": []})
    source_transaction = canonical_json_sha256({"verified": "source-transaction", "generation": 4})
    prepared = migration_transition(
        migration_id="migration-proof",
        transition_revision=1,
        previous_state=None,
        state="PREPARED",
        transaction_fingerprint_sha256=source_transaction,
        state_fingerprint_sha256=source_state,
        protected_freshness_generation=4,
    )
    prepared_record = migration_transition_carrier(prepared)
    prepared_current = migration_current_carrier(
        migration_current(
            migration_id="migration-proof",
            current_transition_revision=1,
            state="PREPARED",
            authoritative_state_fingerprint_sha256=source_state,
            protected_freshness_generation=4,
        )
    )
    history_1 = history_tail_fingerprint_sha256((prepared_record,))
    state_1 = state_fingerprint_sha256(
        account_id=ACCOUNT,
        device_installation_id=DEVICE,
        state_store_schema_version=1,
        state_store_identity_fingerprint_sha256=IDENTITY,
        environment="PAPER",
        protected_freshness_generation=5,
        current_records=(prepared_current,),
        history_tail_fingerprint_sha256=history_1,
    )
    projection_1 = {
        "account_id": ACCOUNT,
        "device_installation_id": DEVICE,
        "state_store_identity_fingerprint_sha256": IDENTITY,
        "state_store_schema_version": 1,
        "environment": "PAPER",
        "expected_current_generation": 4,
        "target_generation": 5,
        "pre_state_fingerprint_sha256": source_state,
        "pre_history_tail_fingerprint_sha256": canonical_json_sha256([]),
        "post_state_fingerprint_sha256": state_1,
        "post_history_tail_fingerprint_sha256": history_1,
        "current_record_mutations": [prepared_current.to_mapping()],
        "immutable_history_appends": [prepared_record.to_mapping()],
    }
    transaction_1 = transaction_fingerprint_sha256(projection_1)
    applying = migration_transition(
        migration_id="migration-proof",
        transition_revision=2,
        previous_state="PREPARED",
        state="APPLYING",
        transaction_fingerprint_sha256=transaction_1,
        state_fingerprint_sha256=state_1,
        protected_freshness_generation=5,
    )
    applying_record = migration_transition_carrier(applying)
    assert source_transaction != transaction_1
    assert applying["transaction_fingerprint_sha256"] == transaction_1
    assert transaction_1 not in prepared_record.payload_fingerprint_sha256
    assert applying_record.payload_fingerprint_sha256 == canonical_json_sha256(applying)


def test_identity_stable_current_keys_and_descriptor_shape_remain_frozen() -> None:
    migration_1 = migration_current_carrier(
        migration_current(
            migration_id="M",
            current_transition_revision=1,
            state="PREPARED",
            authoritative_state_fingerprint_sha256="a" * 64,
            protected_freshness_generation=1,
        )
    )
    migration_2 = migration_current_carrier(
        migration_current(
            migration_id="M",
            current_transition_revision=2,
            state="APPLYING",
            authoritative_state_fingerprint_sha256="b" * 64,
            protected_freshness_generation=2,
        )
    )
    assert migration_1.record_key == migration_2.record_key == "migration-current:M"
    fields = MACHINE["executable_boundary_schemas"]["StateStoreTransactionDescriptor"]["required"]
    assert len(fields) == 14 and "current_record_deletions" not in fields
    assert (
        handoff_current_carrier(
            {
                "handoff_id": "H",
                "current_transition_revision": 1,
                "state": "PREPARED",
                "operation_fingerprint_sha256": "a" * 64,
                "designation_fingerprint_sha256": "0" * 64,
            }
        ).record_key
        == "handoff-current:H"
    )
    assert (
        handoff_transition_carrier(
            {
                "handoff_id": "H",
                "transition_revision": 1,
                "previous_state": None,
                "state": "PREPARED",
                "operation_fingerprint_sha256": "a" * 64,
                "metadata_fingerprint_sha256": "b" * 64,
                "transition_fingerprint_sha256": "0" * 64,
            }
        ).record_key
        != "handoff-transition:H:2"
    )


def test_corrective_iii_static_definition_is_runtime_fact_free() -> None:
    contract = MACHINE["migration_protocol"]["static_intent_authority"]
    assert contract["definition_exact_fields"] == [
        "migration_id",
        "source_schema_version",
        "target_schema_version",
        "ordered_path",
        "rollback_policy",
    ]
    assert set(contract["forbidden_definition_fields"]) == {
        "scope",
        "environment",
        "pre_state_fingerprint_sha256",
        "post_state_fingerprint_sha256",
        "transaction_fingerprint_sha256",
        "protected_freshness_generation",
    }
    assert contract["fallback"] is False
    assert contract["path_only_execution_authority"] is False
    assert contract["empty_registry"] == "NO MIGRATIONS AUTHORIZED"


def test_corrective_iii_classifies_every_runtime_record_field_once() -> None:
    classes = MACHINE["migration_protocol"]["MigrationRecord_field_authority_classification"]
    classified = [field for values in classes.values() for field in values]
    schema = MACHINE["migration_protocol"]["schema"]
    assert len(classified) == len(set(classified))
    assert set(classified) == set(schema["required"])
    assert classes["RUNTIME_DERIVED_TARGET_BOUND"] == ["post_state_fingerprint_sha256"]


def test_corrective_iii_check_order_places_static_authority_first() -> None:
    order = MACHINE["migration_protocol"]["production_check_order"]
    assert order[0].startswith("resolve migration_id")
    assert "lifecycle" in order[3]
    assert order.index("compare exact static candidate fields") < order.index(
        "validate durable lifecycle relation"
    )
