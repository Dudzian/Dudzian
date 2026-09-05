"""Produkcyjne fundamenty trwałego StateStore."""

from .records import (
    PersistenceRecord,
    PersistenceRecordError,
    validate_persistence_record,
)
from .state_store import (
    StateStoreError,
    StateStoreMetadata,
    StateStoreSnapshot,
    SQLiteStateStore,
)
from .restore_protocol import (
    MigrationRestoreAuthorityPort,
    RestoreDecision,
    RestoreLifecycleAuthorityBundle,
    RestoreResult,
    S7CRestoreCoordinator,
    SealedMigrationRestoreAuthority,
    SecretHandoffRestoreAuthorityPort,
    SecretHandoffRestoreFence,
    SecretHandoffRestoreObservation,
    TrustedPhysicalRestoreCoordinator,
)
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
    MigrationExecutionAuthority,
    MigrationExecutionDeclaration,
    MigrationExecutionError,
    MigrationExecutionPlan,
    MigrationSqlOperation,
    sqlite_schema_fingerprint,
)
from .migration_execution_engine import MigrationExecutionCoordinator
from .migration_completion import DurableMigrationCompletionCoordinator
from .physical_schema_registry import (
    StateStorePhysicalSchemaError,
    StateStorePhysicalSchemaRegistry,
)
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
    "SealedMigrationRestoreAuthority",
    "TrustedPhysicalRestoreCoordinator",
    "RestoreLifecycleAuthorityBundle",
    "MigrationRestoreAuthorityPort",
    "SecretHandoffRestoreAuthorityPort",
    "SecretHandoffRestoreFence",
    "SecretHandoffRestoreObservation",
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
    "MigrationExecutionAuthority",
    "DurableMigrationCompletionCoordinator",
    "MigrationExecutionCoordinator",
    "StateStorePhysicalSchemaError",
    "StateStorePhysicalSchemaRegistry",
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
