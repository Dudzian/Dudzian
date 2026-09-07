"""Build-time sealed StateStore PIN carrier migration from schema v1 to v2."""

from .migration_execution import (
    MigrationExecutionAuthority,
    MigrationExecutionPlan,
    MigrationSqlOperation,
)
from .migration_protocol import MigrationDefinition, MigrationError

MIGRATION_ID = "state-store-pin-representation-v1-to-v2"
SQLITE_PHYSICAL_FINGERPRINT_SHA256 = (
    "18f9bac7640b66fb1051d5e1bcfe7345c79a8dcb33f417b40009fb049547c680"
)

STATE_STORE_V2_MIGRATION_DEFINITION = MigrationDefinition(
    MIGRATION_ID, 1, 2, ("rekey-pin-history-carriers", "stabilize-pin-current-slot")
)
STATE_STORE_V2_MIGRATION_OPERATIONS = (
    MigrationSqlOperation(
        1,
        "rekey-pin-history-carriers",
        "DML",
        "UPDATE state_store_immutable_history SET record_key = 'immutable:PinVerifierRecord accepted revisions:' || json_extract(record_json, '$.payload.upstream_payload.operator_id') || ':' || json_extract(record_json, '$.payload.upstream_payload.pin_revision') || ':' || json_extract(record_json, '$.payload.upstream_payload.security_generation') || ':' || json_extract(record_json, '$.payload.upstream_payload.content_fingerprint_sha256'), record_json = json_set(record_json, '$.record_key', 'immutable:PinVerifierRecord accepted revisions:' || json_extract(record_json, '$.payload.upstream_payload.operator_id') || ':' || json_extract(record_json, '$.payload.upstream_payload.pin_revision') || ':' || json_extract(record_json, '$.payload.upstream_payload.security_generation') || ':' || json_extract(record_json, '$.payload.upstream_payload.content_fingerprint_sha256')) WHERE representation_name = 'PinVerifierRecord accepted revisions' AND record_key = 'immutable:PinVerifierRecord accepted revisions:' || json_extract(record_json, '$.payload.upstream_payload.operator_id') || ':' || json_extract(record_json, '$.payload.upstream_payload.pin_revision') || ':' || json_extract(record_json, '$.payload.upstream_payload.security_generation')",
    ),
    MigrationSqlOperation(
        2,
        "stabilize-pin-current-slot",
        "DML",
        "UPDATE state_store_current_records SET record_key = 'current:' || json_extract(record_json, '$.payload.scope_key'), record_json = json_set(record_json, '$.record_key', 'current:' || json_extract(record_json, '$.payload.scope_key')) WHERE representation_name = 'PinVerifierRecord current designation' AND record_key = 'current:' || json_extract(record_json, '$.payload.scope_key') || ':' || json_extract(record_json, '$.payload.current_reference') || ':' || json_extract(record_json, '$.payload.current_revision') || ':' || json_extract(record_json, '$.payload.current_generation')",
    ),
)
STATE_STORE_V2_MIGRATION_PLAN = MigrationExecutionPlan(
    STATE_STORE_V2_MIGRATION_OPERATIONS,
    SQLITE_PHYSICAL_FINGERPRINT_SHA256,
    SQLITE_PHYSICAL_FINGERPRINT_SHA256,
)
STATE_STORE_V2_MIGRATION_AUTHORITY = MigrationExecutionAuthority(
    MIGRATION_ID,
    1,
    2,
    STATE_STORE_V2_MIGRATION_DEFINITION.ordered_path,
    "FORWARD_ONLY",
    "526d21da17e5492624de7458e2708afbb7ac04bc0e978821038c51e4b0855785",
    "bc9664c8c3f053d151ab78e770f1e6ef8c9d8adefb1fbec9a644c52367741c8b",
    SQLITE_PHYSICAL_FINGERPRINT_SHA256,
    SQLITE_PHYSICAL_FINGERPRINT_SHA256,
)


def plan_state_store_v2_migration(snapshot):
    if snapshot.metadata.state_store_schema_version != 1:
        raise MigrationError("StateStore v2 migration planner requires schema version 1")
    return STATE_STORE_V2_MIGRATION_PLAN


STATE_STORE_V2_MIGRATION_AUTHORITY.assert_definition(STATE_STORE_V2_MIGRATION_DEFINITION)
STATE_STORE_V2_MIGRATION_AUTHORITY.assert_plan(STATE_STORE_V2_MIGRATION_PLAN)
