"""Executable acceptance evidence for the frozen S9D-C25 CoreHost lifecycle."""
from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from bot_core.instruments.catalog_admission_receipt import (
    CatalogAdmissionReceiptAuthority,
    SQLiteCatalogAdmissionReceiptMetadataStore,
)
from bot_core.instruments.catalog_runtime_acceptance import _BinanceSpotCatalogProducer
from bot_core.instruments.source_producer_membership import (
    SQLiteMembershipCarrier,
    SourceProducerMembershipAuthority,
)
from bot_core.runtime.core_host import CoreHostProcessLock
from bot_core.runtime.core_host_catalog_runtime import (
    CatalogRuntimeDeploymentConfiguration,
    CoreHostCatalogRuntimeLifecycle,
)
import bot_core.runtime.core_host_catalog_runtime as catalog_lifecycle_module
from bot_core.runtime.core_host_startup_recovery import CoreHostStartupRecoveryCoordinator
from bot_core.security.keyring_storage import KeyringSecretStorage
from tests.runtime.test_core_host_runtime_session_readiness import _initialize, _real_host


@pytest.fixture
def keyring(monkeypatch):
    values: dict[str, str] = {}
    monkeypatch.setattr(
        KeyringSecretStorage,
        "__init__",
        lambda storage, **kwargs: setattr(storage, "_catalog_test_values", values),
    )
    monkeypatch.setattr(
        KeyringSecretStorage,
        "get_secret",
        lambda storage, key: storage._catalog_test_values.get(key),
    )
    monkeypatch.setattr(
        KeyringSecretStorage,
        "set_secret",
        lambda storage, key, value: storage._catalog_test_values.__setitem__(key, value),
    )
    return values


def _offline_storage(tmp_path: Path, *, custody: bool, admission: bool):
    tmp_path.chmod(0o700)
    catalog = tmp_path / "catalog.sqlite3"
    receipts = tmp_path / "receipts.sqlite3"
    membership = SourceProducerMembershipAuthority(SQLiteMembershipCarrier(catalog))
    receipt_authority = CatalogAdmissionReceiptAuthority(
        SQLiteCatalogAdmissionReceiptMetadataStore(receipts)
    )
    if custody:
        receipt_authority.provision()
    if admission:
        assert membership.admit_release_grant("core_release_1_45_binance_spot") is not None
    catalog.chmod(0o600)
    receipts.chmod(0o600)
    config = CatalogRuntimeDeploymentConfiguration.from_protected_mapping({
        "catalog_state_path": str(catalog.absolute()),
        "receipt_metadata_path": str(receipts.absolute()),
    })
    return config, catalog, receipts


def test_protected_configuration_has_exact_keys_no_defaults_and_absolute_paths(tmp_path) -> None:
    for value in (None, {}, {"catalog_state_path": "/x"}, {
        "catalog_state_path": "/x", "receipt_metadata_path": "/y", "extra": "/z"
    }):
        with pytest.raises(ValueError, match="CONFIGURATION_INVALID"):
            CatalogRuntimeDeploymentConfiguration.from_protected_mapping(value)
    with pytest.raises(ValueError, match="PATH_NOT_ABSOLUTE"):
        CatalogRuntimeDeploymentConfiguration.from_protected_mapping({
            "catalog_state_path": "catalog.db", "receipt_metadata_path": "receipts.db"
        })


@pytest.mark.parametrize("bad_target", ["directory", "file"])
def test_permission_qualification_blocks_without_repair(tmp_path, keyring, bad_target) -> None:
    config, catalog, _receipts = _offline_storage(tmp_path, custody=True, admission=True)
    target = tmp_path if bad_target == "directory" else catalog
    target.chmod(0o755 if bad_target == "directory" else 0o644)
    before = target.stat().st_mode

    result = CoreHostCatalogRuntimeLifecycle(config).start_after_recovery()

    assert result.reason == "CATALOG_RUNTIME_PERMISSIONS_INVALID"
    assert target.stat().st_mode == before


def test_non_posix_permission_proof_fails_closed(tmp_path, keyring, monkeypatch) -> None:
    config, _catalog, _receipts = _offline_storage(tmp_path, custody=True, admission=True)
    monkeypatch.setattr(catalog_lifecycle_module, "_POSIX_PERMISSION_MODEL", False)

    result = CoreHostCatalogRuntimeLifecycle(config).start_after_recovery()

    assert result.reason == "CATALOG_RUNTIME_PLATFORM_PERMISSION_PROOF_UNAVAILABLE"


def test_missing_custody_blocks_without_key_generation_or_fetch(tmp_path, keyring, monkeypatch) -> None:
    config, _catalog, _receipts = _offline_storage(tmp_path, custody=False, admission=True)
    monkeypatch.setattr(_BinanceSpotCatalogProducer, "fetch", lambda self: pytest.fail("fetch"))
    lifecycle = CoreHostCatalogRuntimeLifecycle(config)

    assert lifecycle.start_after_recovery().reason == "CATALOG_RUNTIME_RECEIPT_CUSTODY_MISSING"
    assert lifecycle.fetch_catalog_once() is None
    assert keyring == {}


def test_missing_admission_blocks_without_grant_or_fetch(tmp_path, keyring, monkeypatch) -> None:
    config, catalog, _receipts = _offline_storage(tmp_path, custody=True, admission=False)
    monkeypatch.setattr(_BinanceSpotCatalogProducer, "fetch", lambda self: pytest.fail("fetch"))
    lifecycle = CoreHostCatalogRuntimeLifecycle(config)

    assert lifecycle.start_after_recovery().reason == "CATALOG_RUNTIME_PRODUCER_ADMISSION_MISSING"
    assert lifecycle.fetch_catalog_once() is None
    assert SourceProducerMembershipAuthority(SQLiteMembershipCarrier(catalog))._carrier.read() == ()


def test_corrupt_authority_blocks_without_repair_or_fetch(tmp_path, keyring, monkeypatch) -> None:
    config, catalog, _receipts = _offline_storage(tmp_path, custody=True, admission=True)
    with sqlite3.connect(catalog) as db:
        db.execute("UPDATE membership_authority_store_metadata SET authority_domain='corrupt'")
    before = catalog.read_bytes()
    monkeypatch.setattr(_BinanceSpotCatalogProducer, "fetch", lambda self: pytest.fail("fetch"))

    result = CoreHostCatalogRuntimeLifecycle(config).start_after_recovery()

    assert result.reason == "CATALOG_RUNTIME_AUTHORITY_REPLAY_FAILED"
    assert catalog.read_bytes() == before


def test_restart_and_pre_fetch_crash_reopen_without_side_effects(tmp_path, keyring, monkeypatch) -> None:
    config, catalog, receipts = _offline_storage(tmp_path, custody=True, admission=True)
    monkeypatch.setattr(_BinanceSpotCatalogProducer, "fetch", lambda self: pytest.fail("fetch"))
    before = (catalog.read_bytes(), receipts.read_bytes(), dict(keyring))

    first = CoreHostCatalogRuntimeLifecycle(config)
    assert first.start_after_recovery().ready
    del first  # process crash before explicit fetch
    second = CoreHostCatalogRuntimeLifecycle(config)
    assert second.start_after_recovery().ready
    assert second.start_after_recovery().ready

    assert (catalog.read_bytes(), receipts.read_bytes(), keyring) == before


def test_corehost_orders_catalog_gate_after_lock_and_recovery_before_publication(
    tmp_path, monkeypatch
) -> None:
    events: list[str] = []
    state = tmp_path / "state.sqlite3"
    boundary, _ = _initialize(state)
    host, _stores, _registries = _real_host(
        state, boundary, hook=lambda stage, session: events.append(f"publication:{stage}")
    )
    original_lock = CoreHostProcessLock.acquire
    original_recovery = CoreHostStartupRecoveryCoordinator.recover

    def acquire(lock):
        original_lock(lock)
        events.append("lock")

    def recover(coordinator):
        result = original_recovery(coordinator)
        events.append("recovery")
        return result

    def catalog_start(lifecycle):
        events.append("catalog_gate")
        return lifecycle._block("CATALOG_RUNTIME_CONFIGURATION_MISSING")

    monkeypatch.setattr(CoreHostProcessLock, "acquire", acquire)
    monkeypatch.setattr(CoreHostStartupRecoveryCoordinator, "recover", recover)
    monkeypatch.setattr(CoreHostCatalogRuntimeLifecycle, "start_after_recovery", catalog_start)

    host.start()
    try:
        assert events == ["lock", "recovery", "catalog_gate", "publication:before", "publication:after"]
    finally:
        host.close()


def test_corehost_alone_owns_explicit_one_shot_fetch(tmp_path, keyring, monkeypatch) -> None:
    storage = tmp_path / "catalog-authority"
    storage.mkdir(mode=0o700)
    config, _catalog, _receipts = _offline_storage(
        storage, custody=True, admission=True
    )
    state = tmp_path / "state.sqlite3"
    boundary, _ = _initialize(state)
    host, _stores, _registries = _real_host(
        state, boundary, catalog_runtime_configuration=config
    )
    calls: list[str] = []
    monkeypatch.setattr(
        _BinanceSpotCatalogProducer,
        "fetch",
        lambda self: calls.append("fetch") or {"symbols": []},
    )

    host.start()
    try:
        assert host.catalog_runtime_startup_result is not None
        assert host.catalog_runtime_startup_result.ready
        assert calls == []
        assert host.fetch_source_catalog_once() is None
        assert calls == ["fetch"]
    finally:
        host.close()
