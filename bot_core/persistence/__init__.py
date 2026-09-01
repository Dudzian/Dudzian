"""Produkcyjne fundamenty trwałego StateStore."""

from .records import PersistenceRecord, PersistenceRecordError, validate_persistence_record
from .state_store import StateStoreError, StateStoreMetadata, StateStoreSnapshot, SQLiteStateStore
from .restore_protocol import RestoreDecision, RestoreResult, S7CRestoreCoordinator
from .transaction_descriptor import (
    StateStoreTransactionDescriptor,
    TransactionDescriptorError,
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
]
