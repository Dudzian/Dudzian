"""Exact, acyclic declarations of trusted SQLite migration effects."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

from .fingerprints import canonical_json_sha256
from .lifecycle_records import persistence_record
from .migration_execution_contract import (
    MigrationExecutionContractError,
    freeze_json,
    operation_plan_fingerprint_from_mappings,
    thaw_json,
    validate_and_normalize_operation,
    validate_raw_migration_execution_declaration,
)
from .records import PersistenceRecord

MigrationExecutionError = MigrationExecutionContractError


@dataclass(frozen=True, slots=True)
class MigrationSqlOperation:
    ordinal: int
    operation_id: str
    operation_kind: str
    statement: str
    parameters: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, tuple):
            raise MigrationExecutionError("parameters must be an immutable tuple")
        normalized = validate_and_normalize_operation(
            {
                "ordinal": self.ordinal,
                "operation_id": self.operation_id,
                "operation_kind": self.operation_kind,
                "statement": self.statement,
                "parameters": self.parameters,
            }
        )
        object.__setattr__(
            self, "parameters", tuple(freeze_json(item) for item in normalized["parameters"])
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "operation_id": self.operation_id,
            "operation_kind": self.operation_kind,
            "statement": self.statement,
            "parameters": [thaw_json(item) for item in self.parameters],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MigrationSqlOperation:
        if set(value) != {field.name for field in fields(cls)}:
            raise MigrationExecutionError("MigrationSqlOperation requires its exact field set")
        parameters = value["parameters"]
        if not isinstance(parameters, list):
            raise MigrationExecutionError("JSON parameters must be an array")
        return cls(
            ordinal=value["ordinal"],
            operation_id=value["operation_id"],
            operation_kind=value["operation_kind"],
            statement=value["statement"],
            parameters=tuple(parameters),
        )


def operation_plan_fingerprint(operations: Sequence[MigrationSqlOperation]) -> str:
    return operation_plan_fingerprint_from_mappings(
        [operation.to_mapping() for operation in operations]
    )


@dataclass(frozen=True, slots=True)
class MigrationExecutionPlan:
    operations: tuple[MigrationSqlOperation, ...]
    pre_sqlite_schema_fingerprint_sha256: str
    target_sqlite_schema_fingerprint_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.operations, tuple) or not self.operations:
            raise MigrationExecutionError("migration plan requires an immutable operation tuple")
        if tuple(operation.ordinal for operation in self.operations) != tuple(
            range(1, len(self.operations) + 1)
        ):
            raise MigrationExecutionError("operation ordinals must be positive and contiguous")
        for value in (
            self.pre_sqlite_schema_fingerprint_sha256,
            self.target_sqlite_schema_fingerprint_sha256,
        ):
            if not _is_sha(value):
                raise MigrationExecutionError("schema fingerprints must be lowercase SHA-256")

    @property
    def operation_plan_fingerprint_sha256(self) -> str:
        return operation_plan_fingerprint(self.operations)


@dataclass(frozen=True, slots=True)
class MigrationExecutionDeclaration:
    migration_id: str
    source_schema_version: int
    target_schema_version: int
    ordered_path: tuple[str, ...]
    rollback_policy: str
    migration_definition_fingerprint_sha256: str
    account_id: str
    device_installation_id: str
    environment: str
    state_store_identity_fingerprint_sha256: str
    expected_current_generation: int
    target_generation: int
    pre_state_fingerprint_sha256: str
    pre_history_tail_fingerprint_sha256: str
    pre_sqlite_schema_fingerprint_sha256: str
    target_sqlite_schema_fingerprint_sha256: str
    operations: tuple[MigrationSqlOperation, ...]
    operation_plan_fingerprint_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.ordered_path, tuple) or not isinstance(self.operations, tuple):
            raise MigrationExecutionError("DTO arrays must use immutable tuple representation")
        validate_raw_migration_execution_declaration(self.to_mapping())

    def to_mapping(self) -> dict[str, Any]:
        result = {field.name: getattr(self, field.name) for field in fields(self)}
        result["ordered_path"] = list(self.ordered_path)
        result["operations"] = [operation.to_mapping() for operation in self.operations]
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MigrationExecutionDeclaration:
        if set(value) != {field.name for field in fields(cls)}:
            raise MigrationExecutionError("MigrationExecutionDeclaration requires its exact fields")
        if not isinstance(value["ordered_path"], list) or not isinstance(value["operations"], list):
            raise MigrationExecutionError("JSON ordered_path and operations must be arrays")
        parsed = dict(value)
        parsed["ordered_path"] = tuple(value["ordered_path"])
        parsed["operations"] = tuple(
            MigrationSqlOperation.from_mapping(item) for item in value["operations"]
        )
        return cls(**parsed)

    def carrier(self) -> PersistenceRecord:
        return persistence_record(
            "Migration execution declaration",
            f"migration-execution:{self.migration_id}:{self.target_generation}",
            self.to_mapping(),
        )

    def assert_matches(self, definition: Any, plan: MigrationExecutionPlan) -> None:
        """Rebind durable evidence to the current trusted definition and exact plan."""

        static = (
            self.migration_id,
            self.source_schema_version,
            self.target_schema_version,
            self.ordered_path,
            self.rollback_policy,
            self.migration_definition_fingerprint_sha256,
        )
        expected = (
            definition.migration_id,
            definition.source_schema_version,
            definition.target_schema_version,
            definition.ordered_path,
            definition.rollback_policy,
            definition.fingerprint(),
        )
        if static != expected or (
            self.operations != plan.operations
            or self.operation_plan_fingerprint_sha256 != plan.operation_plan_fingerprint_sha256
            or self.pre_sqlite_schema_fingerprint_sha256
            != plan.pre_sqlite_schema_fingerprint_sha256
            or self.target_sqlite_schema_fingerprint_sha256
            != plan.target_sqlite_schema_fingerprint_sha256
        ):
            raise MigrationExecutionError("declaration does not match trusted definition and plan")


def sqlite_schema_fingerprint(connection: sqlite3.Connection) -> str:
    """Hash the explicit user schema; SQLite internal/implicit objects are excluded."""

    rows = connection.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_schema "
        "WHERE type IN ('table','index','view','trigger') AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\' "
        "ORDER BY type,name,tbl_name,sql"
    ).fetchall()
    projection = [
        {"type": row[0], "name": row[1], "tbl_name": row[2], "sql": row[3]} for row in rows
    ]
    return canonical_json_sha256(projection)


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


__all__ = [
    "MigrationExecutionDeclaration",
    "MigrationExecutionError",
    "MigrationExecutionPlan",
    "MigrationSqlOperation",
    "operation_plan_fingerprint",
    "sqlite_schema_fingerprint",
]
