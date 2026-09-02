"""Restart-safe semantic completion of an already-started migration."""

from __future__ import annotations

from .migration_execution_engine import MigrationExecutionCoordinator
from .migration_protocol import (
    DurableMigrationLifecycle,
    DurableMigrationLifecycleCoordinator,
    MigrationError,
)
from .protected_freshness_handoff import (
    ProtectedFreshnessHandoffCoordinator,
    ProtectedFreshnessHandoffError,
)
from .state_store import SQLiteStateStore, StateStoreSnapshot


class DurableMigrationCompletionCoordinator:
    """Resume an APPLYING migration through its two durable terminal facts."""

    def __init__(
        self,
        store: SQLiteStateStore,
        execution: MigrationExecutionCoordinator,
        lifecycles: DurableMigrationLifecycleCoordinator,
        protected: ProtectedFreshnessHandoffCoordinator,
    ) -> None:
        self._store = store
        self._execution = execution
        self._lifecycles = lifecycles
        self._protected = protected

    def resume_to_completion(self, migration_id: str) -> DurableMigrationLifecycle:
        """Reach or rediscover the terminal state using fresh durable authority only."""

        while True:
            lifecycle = self._recover_then_discover(migration_id)
            if lifecycle.current is None:
                raise MigrationError("migration completion requires an existing lifecycle")
            state = str(lifecycle.current["state"])
            if state == "APPLYING":
                materialization = self._execution.execute(migration_id)
                fresh = self._lifecycles.discover(migration_id)
                if fresh.current is None or fresh.current["state"] != "APPLYING":
                    raise MigrationError("migration lifecycle changed during structural execution")
                self._lifecycles.record_durable_migrated(migration_id, materialization)
                continue
            if state == "DURABLE_MIGRATED":
                self._lifecycles.complete(migration_id)
                continue
            if state in {"COMPLETED", "FAILED"}:
                return lifecycle
            if state == "PREPARED":
                raise MigrationError("migration completion does not own PREPARED to APPLYING")
            raise MigrationError(f"unsupported migration lifecycle state {state!r}")

    def _recover_then_discover(self, migration_id: str) -> DurableMigrationLifecycle:
        source = self._required_snapshot()
        metadata = source.metadata
        scope = (
            metadata.account_id,
            metadata.device_installation_id,
            metadata.state_store_identity_fingerprint_sha256,
        )
        try:
            recovered = self._protected.recover_protected_state(scope)
        except ProtectedFreshnessHandoffError as exc:
            raise MigrationError("protected migration completion recovery failed") from exc
        if recovered != metadata:
            raise MigrationError("protected recovery changed migration completion source")
        # Discovery performs the mandated fresh, verified reread after recovery.
        return self._lifecycles.discover(migration_id)

    def _required_snapshot(self) -> StateStoreSnapshot:
        snapshot = self._store.read_verified_snapshot()
        if snapshot is None:
            raise MigrationError("migration completion requires initialized StateStore")
        return snapshot


__all__ = ["DurableMigrationCompletionCoordinator"]
