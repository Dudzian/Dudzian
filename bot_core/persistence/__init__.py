"""Produkcyjne fundamenty trwałego StateStore."""

from .records import PersistenceRecord, PersistenceRecordError, validate_persistence_record
from .state_store import StateStoreError, StateStoreMetadata, StateStoreSnapshot, SQLiteStateStore
from .restore_protocol import RestoreDecision, RestoreResult, S7CRestoreCoordinator
from .transaction_descriptor import (
    StateStoreTransactionDescriptor,
    TransactionDescriptorError,
)
from .migration_protocol import (
    MigrationCoordinator,
    MigrationDefinition,
    MigrationRecord,
    MigrationRegistry,
)
from .migration_execution import (
    MigrationExecutionDeclaration,
    MigrationExecutionError,
    MigrationExecutionPlan,
    MigrationSqlOperation,
    sqlite_schema_fingerprint,
)
from .secret_handoff import SecretHandoffCoordinator, SecretHandoffRecord

__all__ = [
    "PersistenceRecord",
    "PersistenceRecordError",
    "RestoreDecision",
    "RestoreResult",
    "S7CRestoreCoordinator",
    "SQLiteStateStore",
    "StateStoreError",
    "StateStoreMetadata",
    "StateStoreSnapshot",
    "StateStoreTransactionDescriptor",
    "TransactionDescriptorError",
    "validate_persistence_record",
    "MigrationCoordinator",
    "MigrationDefinition",
    "MigrationRecord",
    "MigrationRegistry",
    "MigrationExecutionDeclaration",
    "MigrationExecutionError",
    "MigrationExecutionPlan",
    "MigrationSqlOperation",
    "sqlite_schema_fingerprint",
    "SecretHandoffCoordinator",
    "SecretHandoffRecord",
]
