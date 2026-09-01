from __future__ import annotations

import sqlite3
from dataclasses import replace

import pytest

from bot_core.persistence.fingerprints import transaction_fingerprint_sha256
from bot_core.persistence.migration_execution import (
    MigrationExecutionDeclaration,
    MigrationExecutionError,
    MigrationExecutionPlan,
    MigrationSqlOperation,
    sqlite_schema_fingerprint,
)
from bot_core.persistence.migration_protocol import MigrationDefinition, MigrationError
from bot_core.persistence.records import PersistenceRecord

SHA = "a" * 64
ACCOUNT = "acct_01890f3a-2b4c-7abc-8def-0123456789ab"
DEVICE = "dev_01890f3a-2b4c-7abc-8def-0123456789ab"


def definition() -> MigrationDefinition:
    return MigrationDefinition("migration-1", 1, 2, ("create-widget",))


def plan(connection: sqlite3.Connection) -> MigrationExecutionPlan:
    before = sqlite_schema_fingerprint(connection)
    operation = MigrationSqlOperation(
        1, "create-widget", "DDL", "CREATE TABLE widget(id INTEGER PRIMARY KEY)"
    )
    connection.execute(operation.statement, operation.parameters)
    after = sqlite_schema_fingerprint(connection)
    connection.execute("DROP TABLE widget")
    return MigrationExecutionPlan((operation,), before, after)


def declaration(value: MigrationExecutionPlan) -> MigrationExecutionDeclaration:
    static = definition()
    return MigrationExecutionDeclaration(
        static.migration_id,
        static.source_schema_version,
        static.target_schema_version,
        static.ordered_path,
        static.rollback_policy,
        static.fingerprint(),
        ACCOUNT,
        DEVICE,
        "PAPER",
        SHA,
        1,
        2,
        "b" * 64,
        "c" * 64,
        value.pre_sqlite_schema_fingerprint_sha256,
        value.target_sqlite_schema_fingerprint_sha256,
        value.operations,
        value.operation_plan_fingerprint_sha256,
    )


def test_sqlite_schema_fingerprint_and_undeclared_ddl_detection() -> None:
    first, second = sqlite3.connect(":memory:"), sqlite3.connect(":memory:")
    assert sqlite_schema_fingerprint(first) == sqlite_schema_fingerprint(second)
    value = plan(first)
    first.execute(value.operations[0].statement)
    assert sqlite_schema_fingerprint(first) == value.target_sqlite_schema_fingerprint_sha256
    first.execute("CREATE INDEX widget_id ON widget(id)")
    assert sqlite_schema_fingerprint(first) != value.target_sqlite_schema_fingerprint_sha256


def test_declaration_is_stage1_history_and_descriptor_transitively_binds_it() -> None:
    connection = sqlite3.connect(":memory:")
    carrier = declaration(plan(connection)).carrier()
    assert carrier.record_key == "migration-execution:migration-1:2"
    projection = {
        "account_id": ACCOUNT,
        "device_installation_id": DEVICE,
        "state_store_identity_fingerprint_sha256": SHA,
        "state_store_schema_version": 2,
        "environment": "PAPER",
        "expected_current_generation": 1,
        "target_generation": 2,
        "pre_state_fingerprint_sha256": "b" * 64,
        "pre_history_tail_fingerprint_sha256": "c" * 64,
        "post_state_fingerprint_sha256": "d" * 64,
        "post_history_tail_fingerprint_sha256": "e" * 64,
        "current_record_mutations": [],
        "immutable_history_appends": [carrier.to_mapping()],
    }
    bound = transaction_fingerprint_sha256(projection)
    changed = {
        **projection,
        "immutable_history_appends": [
            {**carrier.to_mapping(), "record_key": "migration-execution:other:2"}
        ],
    }
    assert bound != transaction_fingerprint_sha256(changed)
    assert "transaction_fingerprint_sha256" not in carrier.payload
    assert "post_state_fingerprint_sha256" not in carrier.payload


def test_operation_order_and_every_plan_field_are_exact() -> None:
    a = MigrationSqlOperation(1, "a", "DDL", "CREATE TABLE a(id INTEGER)")
    b = MigrationSqlOperation(2, "b", "DML", "INSERT INTO a(id) VALUES (?)", (1,))
    one = MigrationExecutionPlan((a, b), SHA, "b" * 64)
    reordered = MigrationExecutionPlan(
        (replace(b, ordinal=1), replace(a, ordinal=2)), SHA, "b" * 64
    )
    assert one.operation_plan_fingerprint_sha256 != reordered.operation_plan_fingerprint_sha256
    declaration(one).assert_matches(definition(), one)
    for changed in (
        replace(definition(), source_schema_version=2),
        replace(definition(), target_schema_version=3),
        replace(definition(), ordered_path=("other",)),
    ):
        with pytest.raises((MigrationExecutionError, ValueError)):
            declaration(one).assert_matches(changed, one)
    with pytest.raises(MigrationError):
        replace(definition(), rollback_policy="ROLLBACK")


def test_ambiguous_sql_and_noncanonical_parameters_fail_closed() -> None:
    with pytest.raises(MigrationExecutionError):
        MigrationSqlOperation(1, "bad", "DDL", "CREATE TABLE a(x); DROP TABLE a")
    with pytest.raises(MigrationExecutionError):
        MigrationSqlOperation(1, "bad", "DML", "SELECT ?", (float("nan"),))
    with pytest.raises(MigrationExecutionError):
        MigrationSqlOperation(1, "bad", "PYTHON", "SELECT 1")


def _raw_mapping() -> dict[str, object]:
    import json

    connection = sqlite3.connect(":memory:")
    return json.loads(json.dumps(declaration(plan(connection)).carrier().to_mapping()))


def _rehashed_raw(mapping: dict[str, object]) -> PersistenceRecord:
    from bot_core.persistence.fingerprints import canonical_json_sha256

    value = dict(mapping)
    value["payload_fingerprint_sha256"] = canonical_json_sha256(value["payload"])
    return PersistenceRecord.from_mapping(value)


def _assert_raw_rejected(mutator: object, *, refresh_plan: bool = False) -> None:
    from bot_core.persistence.records import PersistenceRecordError, validate_persistence_record
    from bot_core.persistence.migration_execution_contract import (
        operation_plan_fingerprint_from_mappings,
    )

    mapping = _raw_mapping()
    payload = mapping["payload"]
    assert isinstance(payload, dict)
    mutator(payload)  # type: ignore[operator]
    with pytest.raises((PersistenceRecordError, MigrationExecutionError, TypeError, ValueError)):
        if refresh_plan:
            payload["operation_plan_fingerprint_sha256"] = operation_plan_fingerprint_from_mappings(
                payload["operations"]  # type: ignore[arg-type]
            )
        validate_persistence_record(_rehashed_raw(mapping))


def test_valid_raw_json_carrier_round_trips_through_production_stage1() -> None:
    from bot_core.persistence.records import PersistenceRecord, validate_persistence_record

    mapping = _raw_mapping()
    restored = PersistenceRecord.from_mapping(mapping)
    validate_persistence_record(restored)
    assert restored.record_key == mapping["record_key"]
    assert restored.payload_fingerprint_sha256 == mapping["payload_fingerprint_sha256"]
    assert restored.to_mapping() == mapping


@pytest.mark.parametrize(
    "mutator",
    [
        lambda p: p.update(target_schema_version=p["source_schema_version"]),
        lambda p: p.update(target_generation=p["expected_current_generation"] + 2),
        lambda p: p["operations"][0].update(ordinal=2),
        lambda p: p["operations"].append({**p["operations"][0]}),
        lambda p: p["operations"][0].update(ordinal=True),
        lambda p: p.update(migration_definition_fingerprint_sha256="0" * 64),
        lambda p: p.update(operation_plan_fingerprint_sha256="0" * 64),
    ],
)
def test_raw_intrinsic_relation_substitutions_fail_stage1(mutator: object) -> None:
    _assert_raw_rejected(mutator)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda p: p["operations"][0].update(statement="CREATE TABLE changed(id INTEGER)"),
        lambda p: p["operations"][0].update(parameters=[1]),
        lambda p: p["operations"][0].update(operation_kind="DML"),
        lambda p: p["operations"][0].update(operation_id="changed"),
        lambda p: p["operations"].insert(0, {**p["operations"][0], "ordinal": 1}),
    ],
)
def test_raw_operation_substitutions_with_stale_plan_hash_fail_stage1(mutator: object) -> None:
    _assert_raw_rejected(mutator)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda p: p["operations"][0].update(statement="CREATE TABLE a(x); DROP TABLE a"),
        lambda p: p["operations"][0].update(statement="CREATE TABLE a(x)\x00"),
        lambda p: p["operations"][0].update(parameters=[float("nan")]),
        lambda p: p["operations"][0].update(parameters=[float("inf")]),
        lambda p: p["operations"][0].update(parameters=[object()]),
    ],
)
def test_raw_malformed_effects_fail_even_when_hashes_are_refreshed(mutator: object) -> None:
    _assert_raw_rejected(mutator, refresh_plan=True)
