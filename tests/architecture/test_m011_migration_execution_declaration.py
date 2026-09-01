from __future__ import annotations

import json
from pathlib import Path

from bot_core.persistence.fingerprints import transaction_fingerprint_sha256
from bot_core.persistence.transaction_descriptor import StateStoreTransactionDescriptor

ROOT = Path(__file__).parents[2]
CANONICAL = json.loads(
    (
        ROOT
        / "docs/architecture/cryptohunter_product_architecture/persistence_versioning_migrations_backup_and_recovery.json"
    ).read_text()
)


def test_descriptor_shape_and_ordinary_hash_are_backward_compatible() -> None:
    required = CANONICAL["executable_boundary_schemas"]["StateStoreTransactionDescriptor"][
        "required"
    ]
    assert len(required) == 14
    assert [
        field.name for field in __import__("dataclasses").fields(StateStoreTransactionDescriptor)
    ] == required
    projection = {
        "account_id": "acct_01890f3a-2b4c-7abc-8def-0123456789ab",
        "device_installation_id": "dev_01890f3a-2b4c-7abc-8def-0123456789ab",
        "state_store_identity_fingerprint_sha256": "a" * 64,
        "state_store_schema_version": 1,
        "environment": "PAPER",
        "expected_current_generation": 1,
        "target_generation": 2,
        "pre_state_fingerprint_sha256": "b" * 64,
        "pre_history_tail_fingerprint_sha256": "c" * 64,
        "post_state_fingerprint_sha256": "d" * 64,
        "post_history_tail_fingerprint_sha256": "e" * 64,
        "current_record_mutations": [],
        "immutable_history_appends": [],
    }
    assert (
        transaction_fingerprint_sha256(projection)
        == "d2812c6d2d3e40c1e100f146d3d7f4963885f0f9dad73b4bf9b101849bd14dbf"
    )


def test_execution_declaration_extends_dag_without_back_edges() -> None:
    contract = CANONICAL["migration_protocol"]["migration_execution_declaration"]
    dag = contract["hash_dependency_dag"]
    assert dag.index("MigrationExecutionDeclaration payload") < dag.index(
        "declaration PersistenceRecord carrier"
    )
    assert dag.index("declaration PersistenceRecord carrier") < dag.index(
        "post StateStore state fingerprint"
    )
    assert dag.index("post StateStore state fingerprint") < dag.index(
        "unchanged 14-field descriptor"
    )
    assert dag.index("unchanged 14-field descriptor") < dag.index("transaction fingerprint")
    schema = CANONICAL["executable_boundary_schemas"]["MigrationExecutionDeclaration"]
    forbidden = {"transaction_fingerprint_sha256", "post_state_fingerprint_sha256"}
    assert not forbidden.intersection(schema["required"])
    assert contract["forbidden_dependency_edges"] == [
        "declaration -> enclosing transaction fingerprint",
        "declaration -> resulting post-StateStore fingerprint",
        "operation plan -> descriptor or descriptor-derived fingerprint",
    ]


def test_execution_declaration_is_history_evidence_not_authority() -> None:
    entry = CANONICAL["backup_contract"]["representation_registry"][
        "Migration execution declaration"
    ]
    assert entry["durability_class"] == "DURABLE IMMUTABLE / APPEND-ONLY HISTORY"
    assert entry["restorable_authority"] is False
    contract = CANONICAL["migration_protocol"]["migration_execution_declaration"]
    assert (
        contract["descriptor_shape_changed"] is False
        if "descriptor_shape_changed" in contract
        else contract["transaction_binding"]["descriptor_shape_changed"] is False
    )
    assert "restore authority" in contract["not_authority"]
