from __future__ import annotations
from pathlib import Path
from dataclasses import replace
import pytest
from bot_core.persistence.first_run_bootstrap import DurableFirstRunBootstrapRegistry
from bot_core.persistence.state_store import SQLiteStateStore
from bot_core.runtime.core_host_startup_recovery import CoreHostStartupRecoveryCoordinator
from deployment.windows_installer.corehost_composition import (
    WindowsProvisioningAdapterUnavailable,
    build_production_core_host,
    load_windows_external_provisioning_handoff,
    materialize_canonical_pre_state,
)
from deployment.windows_stage9_production_trust import ProductionTrustUnavailable
from tests.persistence.test_durable_first_run_bootstrap import (
    Provisioning,
    claim,
    initialize_metadata,
)
from tests.persistence.test_protected_freshness_handoff import Boundary, record
from tests.runtime.test_core_host_startup_recovery import SecretPort


class AcceptanceProvisioning(Provisioning):
    """Test-only producer which still enters through the frozen boundary."""

    def __init__(self, path: Path):
        self.path = path
        self.item = claim()
        super().__init__(self.item)
        with SQLiteStateStore(path) as store:
            self.metadata = initialize_metadata(store)
        path.unlink()
        wal = path.with_name(path.name + "-wal")
        shm = path.with_name(path.name + "-shm")
        wal.unlink(missing_ok=True)
        shm.unlink(missing_ok=True)

    def claim_reference(self):
        return self.item.claim_fingerprint_sha256

    def initial_state_store_metadata(self):
        return self.metadata

    def protected_freshness_authority(self):
        return Boundary(record("COMMITTED", committed_generation=2))

    def secret_resource_port(self):
        return SecretPort()


def test_missing_production_provisioning_adapter_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "deployment.platforms.windows.production_trust_package_path",
        lambda ceremony_id: tmp_path / ceremony_id / "production-trust.json",
    )
    with pytest.raises(WindowsProvisioningAdapterUnavailable) as raised:
        load_windows_external_provisioning_handoff()
    assert isinstance(raised.value.__cause__, ProductionTrustUnavailable)


def test_acceptance_handoff_materializes_pre_through_durable_coordinator(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite"
    provider = AcceptanceProvisioning(tmp_path / "seed.sqlite")
    scope = materialize_canonical_pre_state(path, provider)
    with SQLiteStateStore(path) as store:
        current = DurableFirstRunBootstrapRegistry(store).current_pre_state()
    assert (scope.account_id, scope.device_installation_id) == (
        current.account_id,
        current.device_installation_id,
    )
    host = build_production_core_host(path, provider)
    with SQLiteStateStore(path) as store:
        recovery = host._startup_recovery_factory(scope, store)
        assert isinstance(recovery, CoreHostStartupRecoveryCoordinator)


def test_same_accepted_handoff_resumes_existing_canonical_pre(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite"
    provider = AcceptanceProvisioning(tmp_path / "seed.sqlite")
    first = materialize_canonical_pre_state(path, provider)
    second = materialize_canonical_pre_state(path, provider)
    assert second == first


def test_metadata_only_interruption_resumes_pre_materialization(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite"
    provider = AcceptanceProvisioning(tmp_path / "seed.sqlite")
    with SQLiteStateStore(path) as store:
        prepared = store.derive_prepared_metadata(
            provider.metadata, expected_current_generation=None
        )
        store.commit_prepared_metadata(prepared, expected_current_generation=None)
    materialize_canonical_pre_state(path, provider)
    with SQLiteStateStore(path) as store:
        assert DurableFirstRunBootstrapRegistry(store).current_pre_state().account_id == (
            provider.item.account_id
        )


def test_conflicting_existing_state_store_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite"
    provider = AcceptanceProvisioning(tmp_path / "seed.sqlite")
    materialize_canonical_pre_state(path, provider)
    provider.metadata = replace(provider.metadata, state_store_identity_fingerprint_sha256="f" * 64)
    with pytest.raises(ValueError, match="conflicts"):
        materialize_canonical_pre_state(path, provider)
