"""Executable freeze for the M0.11 StateStore v1-to-v2 representation lineage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from bot_core.persistence.migration_execution import (
    MigrationExecutionAuthority,
    MigrationExecutionPlan,
    MigrationSqlOperation,
)
from bot_core.persistence.migration_execution_contract import (
    migration_definition_fingerprint,
    operation_plan_fingerprint_from_mappings,
    validate_and_normalize_operation,
)
from bot_core.persistence.migration_protocol import MigrationDefinition


ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
M011_PATH = DOCS / "persistence_versioning_migrations_backup_and_recovery.json"
M010_PATH = DOCS / "identity_device_authentication_and_secrets.json"
MACHINE = json.loads(M011_PATH.read_text(encoding="utf-8"))
LINEAGE = MACHINE["state_store_representation_lineage_v2"]
PHYSICAL_FINGERPRINT = "18f9bac7640b66fb1051d5e1bcfe7345c79a8dcb33f417b40009fb049547c680"
M010_FROZEN_SHA256 = "76e76f676720c9ae347c4396f1e6e2bb80f9a69c476c1d659b7bb9512ffc1313"


def test_sealed_migration_contains_complete_definition_authority() -> None:
    migration = LINEAGE["sealed_production_migrations"][0]
    assert set(migration["definition"]) == {
        "migration_id",
        "source_schema_version",
        "target_schema_version",
        "ordered_path",
        "rollback_policy",
        "migration_definition_fingerprint_sha256",
    }


def test_sealed_migration_contains_complete_execution_authority() -> None:
    migration = LINEAGE["sealed_production_migrations"][0]
    assert migration["operations"]
    assert set(migration["execution_authority"]) == {
        "migration_id",
        "source_schema_version",
        "target_schema_version",
        "ordered_path",
        "rollback_policy",
        "migration_definition_fingerprint_sha256",
        "operation_plan_fingerprint_sha256",
        "pre_sqlite_schema_fingerprint_sha256",
        "target_sqlite_schema_fingerprint_sha256",
    }


def _sealed_migration() -> dict[str, object]:
    return LINEAGE["sealed_production_migrations"][0]


def test_definition_and_execution_authority_reconstruct_with_existing_dtos() -> None:
    migration = _sealed_migration()
    raw_definition = migration["definition"]
    definition_fields = {
        key: raw_definition[key]
        for key in (
            "migration_id",
            "source_schema_version",
            "target_schema_version",
            "ordered_path",
            "rollback_policy",
        )
    }
    assert set(definition_fields).isdisjoint(
        {"scope", "environment", "account_id", "device_installation_id", "operations"}
    )
    definition = MigrationDefinition(
        migration_id=definition_fields["migration_id"],
        source_schema_version=definition_fields["source_schema_version"],
        target_schema_version=definition_fields["target_schema_version"],
        ordered_path=tuple(definition_fields["ordered_path"]),
        rollback_policy=definition_fields["rollback_policy"],
    )
    assert (
        definition.fingerprint()
        == raw_definition["migration_definition_fingerprint_sha256"]
        == migration_definition_fingerprint(**definition_fields)
    )

    raw_operations = migration["operations"]
    assert raw_operations
    assert [validate_and_normalize_operation(item) for item in raw_operations] == raw_operations
    operations = tuple(MigrationSqlOperation.from_mapping(item) for item in raw_operations)
    registry = {
        entry["state_store_schema_version"]: entry["sqlite_schema_fingerprint_sha256"]
        for entry in MACHINE["state_store_physical_schema_registry"]["entries"]
    }
    plan = MigrationExecutionPlan(operations, registry[1], registry[2])
    assert (
        plan.operation_plan_fingerprint_sha256
        == migration["operation_plan_fingerprint_sha256"]
        == operation_plan_fingerprint_from_mappings(raw_operations)
    )
    authority_raw = migration["execution_authority"]
    authority = MigrationExecutionAuthority(
        **{
            **authority_raw,
            "ordered_path": tuple(authority_raw["ordered_path"]),
        }
    )
    authority.assert_definition(definition)
    authority.assert_plan(plan)
    assert authority_raw["pre_sqlite_schema_fingerprint_sha256"] == registry[1]
    assert authority_raw["target_sqlite_schema_fingerprint_sha256"] == registry[2]
    assert operations and all(operation.operation_kind == "DML" for operation in operations)
    assert tuple(operation.ordinal for operation in operations) == tuple(
        range(1, len(operations) + 1)
    )
    assert len({operation.operation_id for operation in operations}) == len(operations)
    assert migration["authority_source"].startswith("exact operations")
    assert "complete canonical post-operation v2 rows" in migration["descriptor_poststate_binding"]


def test_sealed_sqlite_json_capability_is_available_without_an_extension() -> None:
    capability = _sealed_migration()["sqlite_runtime_capability"]
    assert capability == {
        "minimum_sqlite_version": "3.38.0",
        "required_builtin_functions": ["json_extract", "json_set"],
        "extension_loading": False,
        "reason": (
            "JSON functions are SQLite built-ins from 3.38.0; no loadable extension or caller "
            "function is authority"
        ),
    }
    assert tuple(map(int, sqlite3.sqlite_version.split("."))) >= (3, 38, 0)
    connection = sqlite3.connect(":memory:")
    assert connection.execute("SELECT json_extract('{\"a\":1}', '$.a')").fetchone() == (1,)
    assert connection.execute("SELECT json_set('{\"a\":1}', '$.a', 2)").fetchone() == ('{"a":2}',)


def _legacy_record(key: str, payload: dict[str, object]) -> str:
    return json.dumps(
        {
            "representation_name": payload.pop("representation_name"),
            "representation_category": "test",
            "semantic_owner_milestone": "M0.10",
            "semantic_artifact": "identity_device_authentication_and_secrets.json",
            "semantic_json_pointer": "/executable_boundary_schemas/PinVerifierRecord",
            "semantic_contract_fingerprint_sha256": "a" * 64,
            "record_key": key,
            "payload": payload,
            "payload_fingerprint_sha256": "b" * 64,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


@pytest.mark.parametrize("history_cardinality", [0, 1, 4])
@pytest.mark.parametrize("repetition", range(3))
def test_exact_sql_plan_rekeys_arbitrary_history_and_preserves_unrelated_rows(
    history_cardinality: int, repetition: int
) -> None:
    del repetition
    connection = sqlite3.connect(":memory:", isolation_level=None)
    for table in ("state_store_current_records", "state_store_immutable_history"):
        connection.execute(
            f"CREATE TABLE {table} (record_key TEXT PRIMARY KEY, "
            "representation_name TEXT NOT NULL, record_json TEXT NOT NULL)"
        )
    expected_history: dict[str, dict[str, object]] = {}
    for index in range(history_cardinality):
        operator = f"op_{index}"
        fingerprint = f"{index + 1:064x}"
        upstream = {
            "operator_id": operator,
            "pin_revision": index + 1,
            "security_generation": 1,
            "content_fingerprint_sha256": fingerprint,
        }
        payload = {
            "representation_name": "PinVerifierRecord accepted revisions",
            "fact_kind": "PinVerifierRecord accepted revisions",
            "upstream_payload": upstream,
            "upstream_payload_fingerprint_sha256": "c" * 64,
        }
        old_key = f"immutable:PinVerifierRecord accepted revisions:{operator}:{index + 1}:1"
        new_key = f"{old_key}:{fingerprint}"
        record_json = _legacy_record(old_key, payload)
        connection.execute(
            "INSERT INTO state_store_immutable_history VALUES (?,?,?)",
            (old_key, "PinVerifierRecord accepted revisions", record_json),
        )
        expected_history[new_key] = upstream

    scope, reference = "acct_a:op_a:dev_a", "d" * 64
    current_payload = {
        "representation_name": "PinVerifierRecord current designation",
        "scope_key": scope,
        "current_reference": reference,
        "current_revision": 1,
        "current_generation": 1,
        "content_fingerprint_sha256": "e" * 64,
    }
    old_current = f"current:{scope}:{reference}:1:1"
    connection.execute(
        "INSERT INTO state_store_current_records VALUES (?,?,?)",
        (
            old_current,
            "PinVerifierRecord current designation",
            _legacy_record(old_current, current_payload),
        ),
    )
    unrelated_json = '{"record_key":"unrelated"}'
    connection.execute(
        "INSERT INTO state_store_current_records VALUES (?,?,?)",
        ("unrelated", "Other current", unrelated_json),
    )

    before_upstreams = tuple(expected_history.values())
    connection.execute("BEGIN")
    for raw in _sealed_migration()["operations"]:
        operation = MigrationSqlOperation.from_mapping(raw)
        connection.execute(operation.statement, operation.parameters)
    connection.execute("COMMIT")

    rows = connection.execute(
        "SELECT record_key,record_json FROM state_store_immutable_history ORDER BY record_key"
    ).fetchall()
    assert {key for key, _ in rows} == set(expected_history)
    assert (
        tuple(json.loads(record_json)["payload"]["upstream_payload"] for _, record_json in rows)
        == before_upstreams
    )
    assert all(json.loads(record_json)["record_key"] == key for key, record_json in rows)
    current_rows = connection.execute(
        "SELECT record_key,record_json FROM state_store_current_records ORDER BY record_key"
    ).fetchall()
    migrated = next(row for row in current_rows if row[0] == f"current:{scope}")
    assert json.loads(migrated[1])["record_key"] == migrated[0]
    assert ("unrelated", unrelated_json) in current_rows


def test_exact_sql_plan_collision_rolls_back_without_partial_rekey() -> None:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    connection.execute(
        "CREATE TABLE state_store_immutable_history "
        "(record_key TEXT PRIMARY KEY, representation_name TEXT NOT NULL, "
        "record_json TEXT NOT NULL)"
    )
    operator, fingerprint = "op_a", "1" * 64
    old_key = "immutable:PinVerifierRecord accepted revisions:op_a:1:1"
    new_key = f"{old_key}:{fingerprint}"
    payload = {
        "representation_name": "PinVerifierRecord accepted revisions",
        "fact_kind": "PinVerifierRecord accepted revisions",
        "upstream_payload": {
            "operator_id": operator,
            "pin_revision": 1,
            "security_generation": 1,
            "content_fingerprint_sha256": fingerprint,
        },
        "upstream_payload_fingerprint_sha256": "c" * 64,
    }
    legacy_json = _legacy_record(old_key, payload)
    connection.execute(
        "INSERT INTO state_store_immutable_history VALUES (?,?,?)",
        (old_key, "PinVerifierRecord accepted revisions", legacy_json),
    )
    connection.execute(
        "INSERT INTO state_store_immutable_history VALUES (?,?,?)",
        (new_key, "unrelated collision sentinel", '{"record_key":"sentinel"}'),
    )
    operation = MigrationSqlOperation.from_mapping(_sealed_migration()["operations"][0])
    connection.execute("BEGIN")
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(operation.statement, operation.parameters)
    connection.execute("ROLLBACK")
    assert connection.execute(
        "SELECT record_key,record_json FROM state_store_immutable_history ORDER BY record_key"
    ).fetchall() == [
        (old_key, legacy_json),
        (new_key, '{"record_key":"sentinel"}'),
    ]


def test_v2_is_current_and_v1_remains_known_with_the_same_physical_schema() -> None:
    registry = MACHINE["state_store_physical_schema_registry"]
    assert registry["current_state_store_schema_version"] == 2
    assert {
        entry["state_store_schema_version"]: entry["sqlite_schema_fingerprint_sha256"]
        for entry in registry["entries"]
    } == {1: PHYSICAL_FINGERPRINT, 2: PHYSICAL_FINGERPRINT}
    invariants = registry["composition_invariants"]
    assert invariants["one_entry_per_schema_version"] is True
    assert invariants["sqlite_schema_fingerprint_unique_across_versions"] is False
    assert (
        "logical durable representation contract differs"
        in invariants["same_physical_fingerprint_rule"]
    )


def test_record_contract_is_selected_only_by_container_schema_version() -> None:
    selector = LINEAGE["version_selector"]
    assert "no component" in selector["sole_selector"]
    assert selector["top_level_records"].endswith("StateStoreMetadata.state_store_schema_version")
    assert "descriptor.state_store_schema_version" in selector["descriptor_embedded_records"]
    assert "never the current product version" in selector["descriptor_embedded_records"]


def test_pin_record_key_contracts_are_exact_versioned_overrides() -> None:
    contracts = LINEAGE["record_contracts"]
    assert contracts["unchanged_representations"] == (
        "inherit the base registry contract in both versions"
    )
    assert contracts["version_overrides"] == {
        "1": {
            "PinVerifierRecord accepted revisions": "IMMUTABLE_PAYLOAD_IDENTITY_REVISION",
            "PinVerifierRecord current designation": (
                "SCOPE_CURRENT_REFERENCE_REVISION_GENERATION"
            ),
        },
        "2": {
            "PinVerifierRecord accepted revisions": (
                "IMMUTABLE_PAYLOAD_IDENTITY_REVISION_CONTENT_FINGERPRINT"
            ),
            "PinVerifierRecord current designation": "SCOPE_CURRENT_STABLE",
        },
    }
    assert contracts["v1_exact_key_shapes"] == {
        "history": "immutable:PinVerifierRecord accepted revisions:"
        "{operator_id}:{pin_revision}:{security_generation}",
        "current": "current:{scope_key}:{current_reference}:"
        "{current_revision}:{current_generation}",
    }
    assert contracts["v2_exact_key_shapes"] == {
        "history": "immutable:PinVerifierRecord accepted revisions:"
        "{operator_id}:{pin_revision}:{security_generation}:"
        "{content_fingerprint_sha256}",
        "current": "current:{account_id}:{operator_id}:{device_installation_id}",
    }


def test_historical_v1_descriptors_are_never_rewritten() -> None:
    history = LINEAGE["historical_descriptors"]
    assert history["v1_immutable_after_migration"] is True
    assert history["rewritten_fields"] == []
    assert set(history["preserved_exact"]) == {
        "descriptor JSON",
        "embedded v1 record keys",
        "transaction_fingerprint_sha256",
        "target_generation",
        "pre/post fingerprints",
        "historical evidence chain",
    }


def test_exact_single_sealed_migration_is_representation_only_and_has_no_ddl() -> None:
    migrations = LINEAGE["sealed_production_migrations"]
    assert len(migrations) == 1
    migration = migrations[0]
    assert migration["name"] == "StateStore PIN representation migration v1 to v2"
    assert migration["framework"] == "existing MigrationDefinition and MigrationExecutionAuthority"
    assert (migration["source_schema_version"], migration["target_schema_version"]) == (1, 2)
    assert migration["rollback_policy"] == "FORWARD_ONLY"
    assert migration["ddl_effects"] == []
    assert migration["dml_effects"] == [
        "recognize exact v1 PIN history carriers",
        "intrinsic/source-validate every exact upstream PinVerifierRecord and wrapper fingerprint",
        "derive canonical v2 PIN history keys",
        "replace each v1 top-level PIN history carrier row with its v2 carrier row while preserving exact wrapper payload",
        "recognize exact v1 PIN current designation and prove its reference selects an existing accepted PIN fact",
        "replace the v1 dynamic current row with exactly one v2 stable current scope row",
        "publish required existing migration lifecycle/evidence",
        "publish the migration StateStoreTransactionDescriptor under schema version 2",
    ]
    assert "upstream PinVerifierRecord" in migration["forbidden_semantic_changes"]
    assert "credential material" in migration["forbidden_semantic_changes"]


def test_carrier_rekey_preserves_every_semantic_fact_and_p4_idempotence() -> None:
    rekey = LINEAGE["carrier_rekey"]
    assert rekey["classification"] == (
        "physical carrier replacement, not semantic immutable-history deletion"
    )
    assert set(rekey["forbidden"]) == {
        "delete an upstream fact",
        "change payload",
        "merge distinct content fingerprints",
    }
    assert "one canonical v2 carrier" in rekey["p4_equals_p0"]
    assert "no event ID" in rekey["p4_equals_p0"]


def test_migration_descriptor_and_cross_version_chain_are_exact() -> None:
    publication = LINEAGE["migration_publication_descriptor"]
    assert publication["state_store_schema_version"] == 2
    assert publication["embedded_record_contract_version"] == 2
    assert publication["pre_binding"].startswith("exact verified v1")
    assert publication["post_binding"].startswith("canonical v2")
    assert publication["predecessor_descriptors_rewritten"] is False
    chain = LINEAGE["cross_version_descriptor_chain"]
    assert chain["legal_shape"] == [
        "G1..Gn descriptors use schema version 1",
        "G(n+1) sealed migration publication uses schema version 2",
        "G(n+2).. descriptors use schema version 2",
    ]
    assert chain["ordinary_transaction_version_change"] == "FAIL_CLOSED"
    assert chain["version_change_authority"].startswith("ONLY sealed Migration execution")


def test_v2_top_level_poststate_has_no_legacy_pin_carriers() -> None:
    poststate = LINEAGE["top_level_v2_poststate"]
    assert "no v1 top-level PIN history row remains" in poststate["pin_history"]
    assert "exactly one stable current scope row" in poststate["pin_current"]
    assert "no v1 dynamic current row remains" in poststate["pin_current"]


def test_known_v1_backup_is_stage1_valid_but_not_recovery_complete() -> None:
    backup = LINEAGE["legacy_backup"]
    assert "known exact version 1 may pass" in backup["stage_1_v1"]
    assert backup["classification_after_stage_1"] == (
        "AUTHENTICATED LEGACY CANDIDATE; NOT RECOVERY-COMPLETE; NOT CURRENT RESTORE AUTHORITY"
    )
    assert backup["restore_owns_migration_sql"] is False
    assert backup["missing_sealed_path"] == "NO_SEALED_PATH_TO_CURRENT_SCHEMA / FAIL_CLOSED"
    assert "mismatch fails closed" in backup["tampering"]


def test_restore_and_migration_authorities_remain_separate() -> None:
    assert LINEAGE["legacy_backup"]["composition"] == [
        "authenticate exact v1 artifact",
        "materialize exact isolated v1 StateStore candidate",
        "hand candidate to ordinary M0.11 recovery/migration phase",
        "execute sealed migration 1 → 2 only through migration executor",
        "verify canonical isolated v2 result",
        "only then permit final restore/install authority completion",
    ]
    assert LINEAGE["backup_envelope_schema_version"] == {
        "value": 1,
        "unchanged_reason": (
            "outer envelope shape is unchanged and is independent of StateStore schema version"
        ),
    }


def test_record_key_policy_separates_immutable_facts_from_current_slots() -> None:
    policy = MACHINE["backup_contract"]["record_key_policy"]
    assert "collision_requirement" not in policy
    assert (
        "distinct canonical immutable semantic facts" in policy["immutable_collision_requirement"]
    )
    assert "exact same immutable semantic fact" in policy["immutable_collision_requirement"]
    assert "distinct current scopes" in policy["current_slot_requirement"]
    assert "identity-stable current-slot strategy" in policy["current_slot_requirement"]


def test_non_durable_authorities_and_m010_hash_remain_frozen() -> None:
    assert set(LINEAGE["non_durable_authority_preserved"]) == {
        "AuthenticationProof",
        "CoreIssuedAuthenticationProofBinding",
        "PlatformBiometricAssertion",
        "CoreAcceptedPlatformBiometricAssertionBinding",
        "raw PIN",
        "biometric material",
    }
    assert hashlib.sha256(M010_PATH.read_bytes()).hexdigest() == M010_FROZEN_SHA256
    assert LINEAGE["status"] == "CLOSED — REPRESENTATION LINEAGE V1_TO_V2 FROZEN"
