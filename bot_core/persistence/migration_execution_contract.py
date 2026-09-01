"""Dependency-safe intrinsic semantics for M0.11 migration declarations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from .fingerprints import canonical_json_sha256

_KINDS = frozenset({"DDL", "DML"})
_SHA_FIELDS = (
    "migration_definition_fingerprint_sha256",
    "state_store_identity_fingerprint_sha256",
    "pre_state_fingerprint_sha256",
    "pre_history_tail_fingerprint_sha256",
    "pre_sqlite_schema_fingerprint_sha256",
    "target_sqlite_schema_fingerprint_sha256",
    "operation_plan_fingerprint_sha256",
)
_DECLARATION_FIELDS = frozenset(
    {
        "migration_id",
        "source_schema_version",
        "target_schema_version",
        "ordered_path",
        "rollback_policy",
        "migration_definition_fingerprint_sha256",
        "account_id",
        "device_installation_id",
        "environment",
        "state_store_identity_fingerprint_sha256",
        "expected_current_generation",
        "target_generation",
        "pre_state_fingerprint_sha256",
        "pre_history_tail_fingerprint_sha256",
        "pre_sqlite_schema_fingerprint_sha256",
        "target_sqlite_schema_fingerprint_sha256",
        "operations",
        "operation_plan_fingerprint_sha256",
    }
)
_OPERATION_FIELDS = frozenset(
    {"ordinal", "operation_id", "operation_kind", "statement", "parameters"}
)


class MigrationExecutionContractError(ValueError):
    """A locally derivable declaration invariant is false."""


def freeze_json(value: Any) -> Any:
    """Return the immutable exact JSON value, rejecting coercion and non-finite numbers."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MigrationExecutionContractError("SQL parameters must be finite JSON values")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise MigrationExecutionContractError("SQL parameter object keys must be strings")
        return MappingProxyType({key: freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    raise MigrationExecutionContractError("SQL parameters must be exact JSON values")


def thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def is_one_sqlite_statement(statement: str) -> bool:
    """Conservatively reject NUL, unclosed quotes, and non-final unquoted semicolons."""

    if not isinstance(statement, str) or not statement.strip() or "\x00" in statement:
        return False
    quote: str | None = None
    semicolons: list[int] = []
    index = 0
    while index < len(statement):
        char = statement[index]
        if quote is None and char in {"'", '"', "`", "["}:
            quote = "]" if char == "[" else char
        elif quote is not None and char == quote:
            if quote != "]" and index + 1 < len(statement) and statement[index + 1] == quote:
                index += 1
            else:
                quote = None
        elif quote is None and char == ";":
            semicolons.append(index)
        index += 1
    return quote is None and (not semicolons or semicolons == [len(statement.rstrip()) - 1])


def _positive(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_and_normalize_operation(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _OPERATION_FIELDS:
        raise MigrationExecutionContractError("MigrationSqlOperation requires its exact field set")
    ordinal = value["ordinal"]
    if not _positive(ordinal):
        raise MigrationExecutionContractError("operation ordinal must be a positive integer")
    operation_id = value["operation_id"]
    if not isinstance(operation_id, str) or not operation_id:
        raise MigrationExecutionContractError("operation_id must be non-empty")
    if value["operation_kind"] not in _KINDS:
        raise MigrationExecutionContractError("operation_kind must be DDL or DML")
    statement = value["statement"]
    if not is_one_sqlite_statement(statement):
        raise MigrationExecutionContractError(
            "exactly one unambiguous SQLite statement is required"
        )
    parameters = value["parameters"]
    if not isinstance(parameters, (list, tuple)):
        raise MigrationExecutionContractError("parameters must be an exact JSON array")
    return {
        "ordinal": ordinal,
        "operation_id": operation_id,
        "operation_kind": value["operation_kind"],
        "statement": statement,
        "parameters": [thaw_json(freeze_json(item)) for item in parameters],
    }


def operation_plan_fingerprint_from_mappings(operations: Sequence[Mapping[str, Any]]) -> str:
    normalized = [validate_and_normalize_operation(operation) for operation in operations]
    return canonical_json_sha256(normalized)


def migration_definition_fingerprint(
    *,
    migration_id: object,
    source_schema_version: object,
    target_schema_version: object,
    ordered_path: object,
    rollback_policy: object,
) -> str:
    return canonical_json_sha256(
        {
            "migration_id": migration_id,
            "source_schema_version": source_schema_version,
            "target_schema_version": target_schema_version,
            "ordered_path": thaw_json(ordered_path),
            "rollback_policy": rollback_policy,
        }
    )


def validate_raw_migration_execution_declaration(payload: object) -> None:
    """Validate only facts intrinsically derivable from one durable payload."""

    if not isinstance(payload, Mapping) or set(payload) != _DECLARATION_FIELDS:
        raise MigrationExecutionContractError("declaration requires its exact field set")
    if not all(
        isinstance(payload[name], str) and payload[name]
        for name in ("migration_id", "account_id", "device_installation_id")
    ):
        raise MigrationExecutionContractError(
            "migration and StateStore identities must be non-empty"
        )
    for name in (
        "source_schema_version",
        "target_schema_version",
        "expected_current_generation",
        "target_generation",
    ):
        if not _positive(payload[name]):
            raise MigrationExecutionContractError(f"{name} must be a positive integer")
    if payload["target_schema_version"] <= payload["source_schema_version"]:
        raise MigrationExecutionContractError("migration declaration must be forward-only")
    if payload["target_generation"] != payload["expected_current_generation"] + 1:
        raise MigrationExecutionContractError("migration declaration must bind exact G to G+1")
    if payload["environment"] not in {"PAPER", "TESTNET", "LIVE"}:
        raise MigrationExecutionContractError("invalid environment")
    if payload["rollback_policy"] != "FORWARD_ONLY":
        raise MigrationExecutionContractError("rollback policy must be FORWARD_ONLY")
    path = payload["ordered_path"]
    if not isinstance(path, (list, tuple)) or not all(
        isinstance(item, str) and item for item in path
    ):
        raise MigrationExecutionContractError("ordered_path must be an exact string array")
    if any(not _sha(payload[name]) for name in _SHA_FIELDS):
        raise MigrationExecutionContractError("declaration fingerprints must be lowercase SHA-256")
    operations = payload["operations"]
    if not isinstance(operations, (list, tuple)) or not operations:
        raise MigrationExecutionContractError("operations must be a non-empty array")
    normalized = [validate_and_normalize_operation(operation) for operation in operations]
    if [operation["ordinal"] for operation in normalized] != list(range(1, len(normalized) + 1)):
        raise MigrationExecutionContractError("operation ordinals must be contiguous 1..N")
    if payload["operation_plan_fingerprint_sha256"] != canonical_json_sha256(normalized):
        raise MigrationExecutionContractError("operation plan fingerprint mismatch")
    expected_definition = migration_definition_fingerprint(
        migration_id=payload["migration_id"],
        source_schema_version=payload["source_schema_version"],
        target_schema_version=payload["target_schema_version"],
        ordered_path=path,
        rollback_policy=payload["rollback_policy"],
    )
    if payload["migration_definition_fingerprint_sha256"] != expected_definition:
        raise MigrationExecutionContractError("migration definition fingerprint mismatch")


__all__ = [
    "MigrationExecutionContractError",
    "freeze_json",
    "is_one_sqlite_statement",
    "migration_definition_fingerprint",
    "operation_plan_fingerprint_from_mappings",
    "thaw_json",
    "validate_and_normalize_operation",
    "validate_raw_migration_execution_declaration",
]
