"""Windows production composition over canonical durable Core authorities."""

from __future__ import annotations
from pathlib import Path
from typing import Protocol

from bot_core.persistence.first_run_bootstrap import (
    DurableFirstRunBootstrapCoordinator,
    DurableFirstRunBootstrapRegistry,
)
from bot_core.persistence.local_durable_evidence import LocalDurableEvidenceRegistry
from bot_core.persistence.migration_completion import (
    DurableMigrationCompletionCoordinator,
)
from bot_core.persistence.migration_execution_engine import (
    MigrationExecutionCoordinator,
)
from bot_core.persistence.migration_protocol import (
    DurableMigrationLifecycleCoordinator,
    MigrationRegistry,
)
from bot_core.persistence.physical_schema_registry import (
    StateStorePhysicalSchemaRegistry,
)
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
from bot_core.runtime.core_host_startup_recovery import (
    CoreHostStartupRecoveryCoordinator,
)
from bot_core.runtime.first_run_bootstrap import (
    FirstRunBootstrapClaim,
    ProvisioningBoundary,
)
from bot_core.runtime.first_run_bootstrap import (
    AUTHORITY_SOURCE,
    ProvisioningMembershipBinding,
    claim_content_fingerprint,
)


class WindowsExternalProvisioningHandoff(ProvisioningBoundary, Protocol):
    """Missing product adapter: accepted external authority, never caller JSON."""

    def claim_reference(self) -> str: ...
    def initial_state_store_metadata(self) -> StateStoreMetadata: ...
    def protected_freshness_authority(self) -> ProtectedFreshnessAuthorityPort: ...
    def secret_resource_port(self) -> SecretExternalResourcePort: ...


class WindowsProvisioningAdapterUnavailable(RuntimeError):
    pass


def load_windows_external_provisioning_handoff() -> WindowsExternalProvisioningHandoff:
    """Distinguish unavailable trust from the still-unfinished legal enrollment."""
    from deployment.platforms.windows import production_trust_package_path
    from deployment.windows_stage9_production_trust import (
        CEREMONY_ID,
        load_production_trust,
    )

    try:
        load_production_trust(production_trust_package_path(CEREMONY_ID))
    except Exception as exc:
        raise WindowsProvisioningAdapterUnavailable(
            "PRODUCTION_TRUST_UNAVAILABLE"
        ) from exc
    # Ceremony trust does not mint claims, membership keys, freshness, or secret providers.
    raise WindowsProvisioningAdapterUnavailable(
        "LEGAL_PRODUCTION_ENROLLMENT_NOT_COMPLETED: accepted provisioning handoff unavailable"
    )


def materialize_canonical_pre_state(
    path: Path,
    provisioning: WindowsExternalProvisioningHandoff,
) -> CoreHostScope:
    """Initialize metadata and PRE only through the frozen provisioning boundary."""
    reference, claim, metadata = resolve_accepted_handoff(provisioning)
    with SQLiteStateStore(path) as store:
        existing = store.read_metadata()
        if existing is None:
            prepared = store.derive_prepared_metadata(
                metadata, expected_current_generation=None
            )
            store.commit_prepared_metadata(prepared, expected_current_generation=None)
        elif any(
            getattr(existing, field) != getattr(metadata, field)
            for field in (
                "account_id",
                "device_installation_id",
                "state_store_schema_version",
                "state_store_identity_fingerprint_sha256",
                "environment",
            )
        ):
            raise ValueError(
                "canonical StateStore metadata conflicts with accepted claim"
            )
        try:
            current = DurableFirstRunBootstrapRegistry(store).current_pre_state()
        except Exception:
            snapshot = store.read_verified_snapshot()
            if snapshot is None or any(
                record.representation_name
                in {
                    "bootstrap consumed fence",
                    "bootstrap accepted/consumption history",
                }
                for record in (*snapshot.current_records, *snapshot.immutable_history)
            ):
                raise
            DurableFirstRunBootstrapCoordinator(
                store, provisioning
            ).materialize_initial_state(reference)
            current = DurableFirstRunBootstrapRegistry(store).current_pre_state()
        if (
            current.account_id,
            current.device_installation_id,
            current.intended_operator_id,
            current.expected_generation,
            current.expected_revision,
        ) != (
            claim.account_id,
            claim.device_installation_id,
            claim.intended_operator_id,
            claim.bootstrap_generation,
            claim.bootstrap_revision,
        ):
            raise ValueError("canonical PRE conflicts with accepted claim")
    return CoreHostScope(current.account_id, current.device_installation_id, path)


def resolve_accepted_handoff(
    provisioning: WindowsExternalProvisioningHandoff,
) -> tuple[str, FirstRunBootstrapClaim, StateStoreMetadata]:
    """Validate the accepted external identity without materializing local state."""
    reference = provisioning.claim_reference()
    claim = provisioning.resolve_accepted_claim(reference)
    membership = provisioning.resolve_membership(reference)
    metadata = provisioning.initial_state_store_metadata()
    if not isinstance(claim, FirstRunBootstrapClaim) or not isinstance(
        membership, ProvisioningMembershipBinding
    ):
        raise TypeError("external provisioning returned a non-canonical authority")
    complete = claim_content_fingerprint(claim)
    if (
        reference != claim.claim_fingerprint_sha256
        or membership.claim_fingerprint_sha256 != reference
        or membership.complete_claim_content_fingerprint_sha256 != complete
        or membership.authority_source != AUTHORITY_SOURCE
        or membership.provisioning_context_fingerprint_sha256
        != claim.provisioning_context_fingerprint_sha256
        or (metadata.account_id, metadata.device_installation_id)
        != (claim.account_id, claim.device_installation_id)
    ):
        raise ValueError("external provisioning authority binding differs")
    return reference, claim, metadata


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
        completion = DurableMigrationCompletionCoordinator(
            store, execution, lifecycles, protected
        )
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
