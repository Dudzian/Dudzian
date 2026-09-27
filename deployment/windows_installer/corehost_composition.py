"""Windows production composition over canonical durable Core authorities."""

from __future__ import annotations
from pathlib import Path
from typing import Protocol

from bot_core.persistence.first_run_bootstrap import (
    DurableFirstRunBootstrapCoordinator,
    DurableFirstRunBootstrapRegistry,
)
from bot_core.persistence.local_durable_evidence import LocalDurableEvidenceRegistry
from bot_core.persistence.migration_completion import DurableMigrationCompletionCoordinator
from bot_core.persistence.migration_execution_engine import MigrationExecutionCoordinator
from bot_core.persistence.migration_protocol import (
    DurableMigrationLifecycleCoordinator,
    MigrationRegistry,
)
from bot_core.persistence.physical_schema_registry import StateStorePhysicalSchemaRegistry
from bot_core.persistence.protected_freshness_handoff import (
    ProtectedFreshnessAuthorityPort,
    ProtectedFreshnessHandoffCoordinator,
)
from bot_core.persistence.secret_handoff import (
    DurableSecretHandoffExecutionCoordinator,
    DurableSecretHandoffLifecycleCoordinator,
    SecretExternalResourcePort,
)
from bot_core.persistence.state_store import SQLiteStateStore, StateStoreMetadata
from bot_core.runtime.core_host import CoreHost, CoreHostScope
from bot_core.runtime.core_host_startup_recovery import CoreHostStartupRecoveryCoordinator
from bot_core.runtime.first_run_bootstrap import FirstRunBootstrapClaim, ProvisioningBoundary


class WindowsExternalProvisioningHandoff(ProvisioningBoundary, Protocol):
    """Missing product adapter: accepted external authority, never caller JSON."""

    def claim_reference(self) -> str: ...
    def initial_state_store_metadata(self) -> StateStoreMetadata: ...
    def protected_freshness_authority(self) -> ProtectedFreshnessAuthorityPort: ...
    def secret_resource_port(self) -> SecretExternalResourcePort: ...


class WindowsProvisioningAdapterUnavailable(RuntimeError):
    pass


def load_windows_external_provisioning_handoff() -> WindowsExternalProvisioningHandoff:
    """Fail closed until the product supplies its external provisioning adapter."""
    raise WindowsProvisioningAdapterUnavailable(
        "production external_product_provisioning_boundary adapter is not configured"
    )


def materialize_canonical_pre_state(
    path: Path,
    provisioning: WindowsExternalProvisioningHandoff,
) -> CoreHostScope:
    """Initialize metadata and PRE only through the frozen provisioning boundary."""
    reference = provisioning.claim_reference()
    claim = provisioning.resolve_accepted_claim(reference)
    if not isinstance(claim, FirstRunBootstrapClaim):
        raise TypeError("external provisioning returned a non-canonical claim")
    metadata = provisioning.initial_state_store_metadata()
    if (metadata.account_id, metadata.device_installation_id) != (
        claim.account_id,
        claim.device_installation_id,
    ):
        raise ValueError("StateStore metadata scope differs from accepted claim")
    with SQLiteStateStore(path) as store:
        if store.read_metadata() is not None:
            raise ValueError("canonical StateStore metadata already exists")
        prepared = store.derive_prepared_metadata(metadata, expected_current_generation=None)
        store.commit_prepared_metadata(prepared, expected_current_generation=None)
        DurableFirstRunBootstrapCoordinator(store, provisioning).materialize_initial_state(
            reference
        )
        current = DurableFirstRunBootstrapRegistry(store).current_pre_state()
    return CoreHostScope(current.account_id, current.device_installation_id, path)


def resolve_durable_scope(path: Path) -> CoreHostScope:
    """Resolve service scope only from the verified durable PRE registry."""
    with SQLiteStateStore(path) as store:
        current = DurableFirstRunBootstrapRegistry(store).current_pre_state()
    return CoreHostScope(current.account_id, current.device_installation_id, path)


def build_production_core_host(
    path: Path,
    provisioning: WindowsExternalProvisioningHandoff,
) -> CoreHost:
    """Compose CoreHost with the real production startup-recovery coordinator."""
    scope = resolve_durable_scope(path)

    def state_store_factory() -> SQLiteStateStore:
        return SQLiteStateStore(path)

    def recovery_factory(
        bound_scope: CoreHostScope, store: SQLiteStateStore
    ) -> CoreHostStartupRecoveryCoordinator:
        evidence = LocalDurableEvidenceRegistry()
        protected = ProtectedFreshnessHandoffCoordinator(
            store, evidence, provisioning.protected_freshness_authority()
        )
        migrations = MigrationRegistry()
        lifecycles = DurableMigrationLifecycleCoordinator(store, migrations, protected)
        execution = MigrationExecutionCoordinator(store, migrations, protected)
        completion = DurableMigrationCompletionCoordinator(store, execution, lifecycles, protected)
        secret_lifecycles = DurableSecretHandoffLifecycleCoordinator(store, protected)
        secrets = DurableSecretHandoffExecutionCoordinator(
            store, secret_lifecycles, protected, provisioning.secret_resource_port()
        )
        return CoreHostStartupRecoveryCoordinator(
            bound_scope,
            store,
            protected,
            migrations,
            lifecycles,
            completion,
            secrets,
            evidence,
            StateStorePhysicalSchemaRegistry(),
        )

    return CoreHost(
        scope,
        state_store_factory=state_store_factory,
        startup_recovery_factory=recovery_factory,
    )
