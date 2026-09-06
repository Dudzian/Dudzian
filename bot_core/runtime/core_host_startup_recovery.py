"""Production orchestration of the frozen CoreHost startup recovery prefix."""

from __future__ import annotations

from bot_core.persistence.local_durable_evidence import LocalDurableEvidenceRegistry
from bot_core.persistence.migration_completion import DurableMigrationCompletionCoordinator
from bot_core.persistence.migration_protocol import (
    DurableMigrationLifecycleCoordinator,
    MigrationError,
    MigrationRegistry,
    migration_family_ids,
)
from bot_core.persistence.physical_schema_registry import StateStorePhysicalSchemaRegistry
from bot_core.persistence.protected_freshness_handoff import (
    ProtectedFreshnessHandoffCoordinator,
)
from bot_core.persistence.restore_protocol import SealedMigrationRestoreAuthority
from bot_core.persistence.secret_handoff import (
    DurableSecretHandoffExecutionCoordinator,
    validate_secret_handoff_snapshot,
)
from bot_core.persistence.state_store import SQLiteStateStore, StateStoreSnapshot
from bot_core.persistence.runtime_session_history import RuntimeSessionHistoryPublisher

from .core_host import CoreHostScope
from .core_host_recovery_types import (
    StartupSubsystemRecoveryClassification,
    StartupSubsystemRecoveryResult,
)


class CoreHostStartupRecoveryError(RuntimeError):
    """Raised when startup cannot resolve all durable recovery gates."""


class CoreHostStartupRecoveryCoordinator:
    """Sequence existing authority owners without duplicating their semantics."""

    def __init__(
        self,
        scope: CoreHostScope,
        store: SQLiteStateStore,
        protected: ProtectedFreshnessHandoffCoordinator,
        migrations: MigrationRegistry,
        migration_lifecycles: DurableMigrationLifecycleCoordinator,
        migration_completion: DurableMigrationCompletionCoordinator,
        secret_execution: DurableSecretHandoffExecutionCoordinator,
        evidence: LocalDurableEvidenceRegistry,
        physical_schemas: StateStorePhysicalSchemaRegistry | None = None,
    ) -> None:
        self._scope = scope
        self._store = store
        self._protected = protected
        self._migrations = migrations
        self._migration_lifecycles = migration_lifecycles
        self._migration_completion = migration_completion
        self._secret_execution = secret_execution
        self._evidence = evidence
        self._physical_schemas = physical_schemas or StateStorePhysicalSchemaRegistry()
        self._migration_validator = SealedMigrationRestoreAuthority(migrations)

    def runtime_session_history_publisher(self) -> RuntimeSessionHistoryPublisher:
        """Return the P1C owner bound to the same recovered store and M0.3 port."""

        return RuntimeSessionHistoryPublisher(self._store, self._protected)

    def recover(self) -> StartupSubsystemRecoveryResult:
        initial = self._store.read_verified_snapshot()
        if initial is None:
            return StartupSubsystemRecoveryResult(
                StartupSubsystemRecoveryClassification.EMPTY_UNINITIALIZED
            )
        protected_scope = self._bind(initial)
        if self._protected.recover_protected_state(protected_scope) != initial.metadata:
            raise CoreHostStartupRecoveryError("protected recovery did not preserve local state")

        after_protected = self._required_snapshot("protected recovery")
        self._recover_migrations(after_protected)
        after_migrations = self._required_snapshot("migration recovery")
        self._recover_secrets(after_migrations)

        final = self._required_snapshot("secret recovery")
        final_scope = self._bind(final)
        self._verify_physical_schema(final)
        if self._protected.recover_protected_state(final_scope) != final.metadata:
            raise CoreHostStartupRecoveryError("final protected state is not exact/current")
        self._validate_final_migrations(final)
        self._validate_final_secrets(final)
        if self._evidence.publish_verified_state(self._store) is None:
            raise CoreHostStartupRecoveryError("fresh durable evidence publication failed")
        return StartupSubsystemRecoveryResult(
            StartupSubsystemRecoveryClassification.INITIALIZED_DURABLE_RECOVERY_RESOLVED
        )

    def _bind(self, snapshot: StateStoreSnapshot) -> tuple[str, str, str]:
        metadata = snapshot.metadata
        if self._store.path.resolve() != self._scope.state_store_path:
            raise CoreHostStartupRecoveryError(
                "opened StateStore path does not match CoreHost scope"
            )
        if metadata.account_id != self._scope.account_id:
            raise CoreHostStartupRecoveryError("StateStore account does not match CoreHost scope")
        if metadata.device_installation_id != self._scope.device_installation_id:
            raise CoreHostStartupRecoveryError("StateStore device does not match CoreHost scope")
        return (
            metadata.account_id,
            metadata.device_installation_id,
            metadata.state_store_identity_fingerprint_sha256,
        )

    def _required_snapshot(self, stage: str) -> StateStoreSnapshot:
        snapshot = self._store.read_verified_snapshot()
        if snapshot is None:
            raise CoreHostStartupRecoveryError(f"StateStore became empty after {stage}")
        return snapshot

    def _migration_ids(self, snapshot: StateStoreSnapshot) -> tuple[str, ...]:
        identities = migration_family_ids(snapshot)
        definitions = [self._migrations.definition_for(identity) for identity in identities]
        by_source = {definition.source_schema_version: definition for definition in definitions}
        if len(by_source) != len(definitions):
            raise MigrationError("migration schema chain branches")
        targets = {definition.target_schema_version for definition in definitions}
        starts = [source for source in by_source if source not in targets]
        if definitions and len(starts) != 1:
            raise MigrationError("migration schema chain has no unique head")
        ordered: list[str] = []
        version = starts[0] if starts else -1
        while version in by_source:
            definition = by_source[version]
            ordered.append(definition.migration_id)
            version = definition.target_schema_version
        if len(ordered) != len(definitions):
            raise MigrationError("migration schema chain is disconnected or cyclic")
        return tuple(ordered)

    def _validate_registry_binding(self, migration_id: str) -> None:
        definition = self._migrations.definition_for(migration_id)
        authority = self._migrations.execution_authority_for(migration_id)
        if (
            authority.pre_sqlite_schema_fingerprint_sha256
            != self._physical_schemas.expected_fingerprint(definition.source_schema_version)
            or authority.target_sqlite_schema_fingerprint_sha256
            != self._physical_schemas.expected_fingerprint(definition.target_schema_version)
        ):
            raise CoreHostStartupRecoveryError(
                "migration authority does not match sealed physical schema registry"
            )

    def _recover_migrations(self, snapshot: StateStoreSnapshot) -> None:
        self._migration_validator.revalidate(snapshot, self._store.sqlite_schema_fingerprint())
        for migration_id in self._migration_ids(snapshot):
            self._validate_registry_binding(migration_id)
            lifecycle = self._migration_lifecycles.discover(migration_id)
            if lifecycle.current is None:
                raise MigrationError("migration has no current designation")
            state = str(lifecycle.current["state"])
            if state == "FAILED":
                raise MigrationError("FAILED migration blocks startup recovery")
            if state == "COMPLETED":
                continue
            if state == "PREPARED":
                self._migration_lifecycles.begin_applying(migration_id)
            terminal = self._migration_completion.resume_to_completion(migration_id)
            if terminal.current is None or terminal.current["state"] != "COMPLETED":
                raise MigrationError("migration recovery did not reach COMPLETED")

    def _recover_secrets(self, snapshot: StateStoreSnapshot) -> None:
        for descriptor, state in validate_secret_handoff_snapshot(snapshot):
            if state == "UNKNOWN_RECONCILIATION":
                raise CoreHostStartupRecoveryError("unknown secret reconciliation blocks startup")
            if state in {"PREPARED", "COMMITTED"}:
                terminal = self._secret_execution.resume(descriptor.handoff_id)
                if terminal.current is None or terminal.current["state"] != "CLEANUP_PENDING":
                    raise CoreHostStartupRecoveryError(
                        "secret recovery did not reach terminal state"
                    )

    def _verify_physical_schema(self, snapshot: StateStoreSnapshot) -> None:
        version = snapshot.metadata.state_store_schema_version
        expected = self._physical_schemas.expected_fingerprint(version)
        if version != self._physical_schemas.current_version:
            raise CoreHostStartupRecoveryError(
                "StateStore schema is not the product current version"
            )
        if self._store.sqlite_schema_fingerprint() != expected:
            raise CoreHostStartupRecoveryError("StateStore physical schema fingerprint mismatch")

    def _validate_final_migrations(self, snapshot: StateStoreSnapshot) -> None:
        self._migration_validator.revalidate(snapshot, self._store.sqlite_schema_fingerprint())
        for migration_id in self._migration_ids(snapshot):
            self._validate_registry_binding(migration_id)
            lifecycle = self._migration_lifecycles.view_verified_snapshot(snapshot, migration_id)
            if lifecycle.current is None or lifecycle.current["state"] != "COMPLETED":
                raise MigrationError("non-terminal migration remains after recovery")

    @staticmethod
    def _validate_final_secrets(snapshot: StateStoreSnapshot) -> None:
        for _descriptor, state in validate_secret_handoff_snapshot(snapshot):
            if state != "CLEANUP_PENDING":
                raise CoreHostStartupRecoveryError("non-terminal secret handoff remains")


__all__ = [
    "CoreHostStartupRecoveryCoordinator",
    "CoreHostStartupRecoveryError",
]
