"""Protected, declaration-bound production SQLite migration execution."""

from __future__ import annotations

import sqlite3
from dataclasses import replace

from .migration_execution import (
    MigrationExecutionDeclaration,
    MigrationExecutionError,
)
from .migration_execution_contract import thaw_json
from .migration_protocol import (
    DurableMigrationLifecycleCoordinator,
    MigrationError,
    MigrationRecord,
    MigrationRegistry,
    migration_mapping_payload,
)
from .protected_freshness_handoff import (
    ProtectedFreshnessHandoffCoordinator,
    ProtectedFreshnessHandoffError,
)
from .state_store import SQLiteStateStore, StateStoreError, StateStoreSnapshot


class MigrationExecutionCoordinator:
    """Execute only a migration ID resolved through one sealed registry."""

    def __init__(
        self,
        store: SQLiteStateStore,
        registry: MigrationRegistry,
        protected: ProtectedFreshnessHandoffCoordinator,
    ) -> None:
        self._store = store
        self._registry = registry
        self._protected = protected
        self._lifecycles = DurableMigrationLifecycleCoordinator(store, registry, protected)

    def execute(self, migration_id: str) -> MigrationRecord:
        definition = self._registry.definition_for(migration_id)
        source = self._required_snapshot()
        scope = self._scope(source)
        try:
            recovered = self._protected.recover_protected_state(scope)
        except ProtectedFreshnessHandoffError as exc:
            raise MigrationError("protected migration preflight recovery failed") from exc
        if recovered != source.metadata:
            raise MigrationError("protected recovery changed migration source")
        source = self._required_snapshot()
        lifecycle = self._lifecycles._view(source, migration_id)
        if lifecycle.current is None or lifecycle.current["state"] != "APPLYING":
            raise MigrationError("migration execution requires APPLYING lifecycle")
        if source.metadata.state_store_schema_version == definition.target_schema_version:
            return self._recover_durable(definition, source)
        if source.metadata.state_store_schema_version != definition.source_schema_version:
            raise MigrationError("migration execution source schema mismatch")

        plan = self._registry.authorized_plan_for(definition, source)
        if self._store.sqlite_schema_fingerprint() != plan.pre_sqlite_schema_fingerprint_sha256:
            raise MigrationError("migration pre-schema fingerprint mismatch")
        declaration = self._declaration(definition, plan, source)
        declaration.assert_matches(definition, plan)
        self._registry.execution_authority_for(migration_id).assert_declaration(declaration)
        carrier = declaration.carrier()
        target_seed = replace(
            source.metadata,
            state_store_schema_version=definition.target_schema_version,
            protected_freshness_generation=declaration.target_generation,
        )
        candidate = self._store.derive_prepared_metadata(
            target_seed,
            current_records=(),
            immutable_history=(carrier,),
            expected_current_generation=declaration.expected_current_generation,
        )
        try:
            self._protected._advance_migration_execution(
                candidate, declaration, expected_source=source.metadata
            )
        except (ProtectedFreshnessHandoffError, StateStoreError, sqlite3.Error) as exc:
            latest = self._required_snapshot()
            if latest.metadata.state_store_schema_version == definition.target_schema_version:
                return self._recover_durable(definition, latest)
            raise MigrationError("protected atomic migration execution failed") from exc
        return self._recover_durable(definition, self._required_snapshot())

    @staticmethod
    def _scope(snapshot: StateStoreSnapshot) -> tuple[str, str, str]:
        metadata = snapshot.metadata
        return (
            metadata.account_id,
            metadata.device_installation_id,
            metadata.state_store_identity_fingerprint_sha256,
        )

    def _required_snapshot(self) -> StateStoreSnapshot:
        snapshot = self._store.read_verified_snapshot()
        if snapshot is None:
            raise MigrationError("migration execution requires initialized StateStore")
        return snapshot

    @staticmethod
    def _declaration(definition, plan, source) -> MigrationExecutionDeclaration:
        metadata = source.metadata
        return MigrationExecutionDeclaration(
            migration_id=definition.migration_id,
            source_schema_version=definition.source_schema_version,
            target_schema_version=definition.target_schema_version,
            ordered_path=definition.ordered_path,
            rollback_policy=definition.rollback_policy,
            migration_definition_fingerprint_sha256=definition.fingerprint(),
            account_id=metadata.account_id,
            device_installation_id=metadata.device_installation_id,
            environment=metadata.environment,
            state_store_identity_fingerprint_sha256=metadata.state_store_identity_fingerprint_sha256,
            expected_current_generation=metadata.protected_freshness_generation,
            target_generation=metadata.protected_freshness_generation + 1,
            pre_state_fingerprint_sha256=metadata.state_fingerprint_sha256,
            pre_history_tail_fingerprint_sha256=metadata.history_tail_fingerprint_sha256,
            pre_sqlite_schema_fingerprint_sha256=plan.pre_sqlite_schema_fingerprint_sha256,
            target_sqlite_schema_fingerprint_sha256=plan.target_sqlite_schema_fingerprint_sha256,
            operations=plan.operations,
            operation_plan_fingerprint_sha256=plan.operation_plan_fingerprint_sha256,
        )

    def _recover_durable(self, definition, snapshot: StateStoreSnapshot) -> MigrationRecord:
        lifecycle = self._lifecycles._view(snapshot, definition.migration_id)
        if lifecycle.current is None or lifecycle.current["state"] != "APPLYING":
            raise MigrationError("durable execution lifecycle is not APPLYING")
        matching = tuple(
            record
            for record in snapshot.immutable_history
            if record.representation_name == "Migration execution declaration"
            and migration_mapping_payload(record.payload).get("migration_id")
            == definition.migration_id
        )
        if len(matching) != 1:
            raise MigrationError("exactly one durable migration declaration is required")
        try:
            declaration = MigrationExecutionDeclaration.from_mapping(thaw_json(matching[0].payload))
        except (TypeError, ValueError) as exc:
            raise MigrationError("durable migration declaration is malformed") from exc
        try:
            authority = self._registry.execution_authority_for(definition.migration_id)
            authority.assert_definition(definition)
            authority.assert_declaration(declaration)
        except MigrationExecutionError as exc:
            raise MigrationError("durable declaration does not match trusted authority") from exc
        metadata = snapshot.metadata
        if (
            matching[0] != declaration.carrier()
            or metadata.state_store_schema_version != definition.target_schema_version
            or declaration.target_generation != metadata.protected_freshness_generation
            or (
                declaration.account_id,
                declaration.device_installation_id,
                declaration.environment,
            )
            != (
                metadata.account_id,
                metadata.device_installation_id,
                metadata.environment,
            )
            or declaration.state_store_identity_fingerprint_sha256
            != metadata.state_store_identity_fingerprint_sha256
            or self._store.sqlite_schema_fingerprint()
            != declaration.target_sqlite_schema_fingerprint_sha256
        ):
            raise MigrationError("durable migration declaration does not match current store")
        descriptors = tuple(
            item
            for item in snapshot.transaction_descriptors
            if item.target_generation == declaration.target_generation
        )
        previous = tuple(
            item
            for item in snapshot.transaction_descriptors
            if item.target_generation == declaration.expected_current_generation
        )
        if len(descriptors) != 1 or len(previous) != 1:
            raise MigrationError("migration descriptor edge is unavailable")
        descriptor = descriptors[0]
        if (
            descriptor.immutable_history_appends != (matching[0],)
            or descriptor.current_record_mutations
            or descriptor.pre_state_fingerprint_sha256 != declaration.pre_state_fingerprint_sha256
            or descriptor.pre_history_tail_fingerprint_sha256
            != declaration.pre_history_tail_fingerprint_sha256
            or descriptor.post_state_fingerprint_sha256 != metadata.state_fingerprint_sha256
        ):
            raise MigrationError("migration descriptor does not bind exact declaration")

        before_recovery = metadata
        try:
            recovered = self._protected.recover_protected_state(self._scope(snapshot))
        except ProtectedFreshnessHandoffError as exc:
            raise MigrationError("durable migration protected recovery failed") from exc
        if recovered != before_recovery:
            raise MigrationError("durable migration recovery changed local metadata")
        fresh = self._required_snapshot()
        if fresh.metadata != before_recovery:
            raise MigrationError("durable migration changed during proof reconstruction")
        return MigrationRecord(
            migration_id=definition.migration_id,
            source_schema_version=definition.source_schema_version,
            target_schema_version=definition.target_schema_version,
            ordered_path=definition.ordered_path,
            scope=(declaration.account_id, declaration.device_installation_id),
            environment=declaration.environment,
            pre_state_fingerprint_sha256=declaration.pre_state_fingerprint_sha256,
            post_state_fingerprint_sha256=metadata.state_fingerprint_sha256,
            transaction_fingerprint_sha256=previous[0].transaction_fingerprint_sha256,
            protected_freshness_generation=declaration.expected_current_generation,
            rollback_policy=definition.rollback_policy,
        )


__all__ = ["MigrationExecutionCoordinator"]
