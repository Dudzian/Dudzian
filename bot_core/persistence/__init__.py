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
from .migration_execution_engine import MigrationExecutionCoordinator
from .migration_completion import DurableMigrationCompletionCoordinator
from .secret_handoff import (
    DurableSecretHandoffExecutionCoordinator,
    SecretHandoffCoordinator,
    SecretHandoffRecord,
)
from .physical_backup import (
    AuthenticatedPhysicalBackupCandidate,
    PhysicalBackupAdmissionError,
    PhysicalBackupAdmissionValidator,
    PhysicalBackupCreator,
    PhysicalBackupError,
    PhysicalSQLiteArtifact,
    TrustedPhysicalBackupArtifact,
)

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
    "DurableMigrationCompletionCoordinator",
    "MigrationExecutionCoordinator",
    "MigrationExecutionError",
    "MigrationExecutionPlan",
    "MigrationSqlOperation",
    "sqlite_schema_fingerprint",
    "SecretHandoffCoordinator",
    "DurableSecretHandoffExecutionCoordinator",
    "SecretHandoffRecord",
    "AuthenticatedPhysicalBackupCandidate",
    "PhysicalBackupAdmissionError",
    "PhysicalBackupAdmissionValidator",
    "PhysicalBackupCreator",
    "PhysicalBackupError",
    "PhysicalSQLiteArtifact",
    "TrustedPhysicalBackupArtifact",
]
