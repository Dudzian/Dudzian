"""Executable acceptance evidence for the frozen S9D-C25 CoreHost lifecycle."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from bot_core.instruments.catalog_admission_receipt import (
    CatalogAdmissionReceiptAuthority,
    SQLiteCatalogAdmissionReceiptMetadataStore,
)
from bot_core.instruments.catalog_runtime_acceptance import (
    CatalogRuntimeAcceptanceAuthority,
    _BinanceSpotCatalogProducer,
)
from bot_core.instruments.source_producer_membership import (
    SQLiteMembershipCarrier,
    SourceProducerMembershipAuthority,
)
from bot_core.runtime.core_host import CoreHostProcessLock
from bot_core.runtime.core_host_catalog_runtime import (
    CatalogPermissionQualification,
    CatalogPermissionQualifier,
    CatalogRuntimeDeploymentConfiguration,
    CoreHostCatalogRuntimeLifecycle,
    PosixCatalogPermissionQualifier,
    UnavailableCatalogPermissionQualifier,
    native_catalog_permission_qualifier,
)
from bot_core.runtime.core_host_startup_recovery import CoreHostStartupRecoveryCoordinator
from bot_core.security.keyring_storage import KeyringSecretStorage
from tests.runtime.test_core_host_runtime_session_readiness import _initialize, _real_host


class _FixedQualifier(CatalogPermissionQualifier):
    def __init__(self, result: CatalogPermissionQualification) -> None:
        self._result = result

    def qualify(self, paths):  # type: ignore[no-untyped-def]
        del paths
        return self._result


QUALIFIED = _FixedQualifier(CatalogPermissionQualification.QUALIFIED)

_RECEIPT_TABLES = (
    "catalog_receipt_authority_metadata",
    "catalog_receipt_authority_keys",
    "catalog_admission_receipts",
    "catalog_receipt_authority_head",
    "catalog_admission_receipt_finalizations",
    "catalog_receipt_finalization_head",
)


def _receipt_semantic_snapshot(path: Path) -> tuple[tuple, tuple]:
    """Read schema and authority rows without invoking provisioning or repair."""
    uri = f"{path.absolute().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        schema = tuple(
            db.execute(
                "SELECT type,name,tbl_name,sql FROM sqlite_schema "
                "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name,tbl_name"
            )
        )
        rows = tuple(
            (table, tuple(db.execute(f'SELECT * FROM "{table}" ORDER BY rowid')))
            for table in _RECEIPT_TABLES
        )
    return schema, rows


def _sqlite_semantic_snapshot(path: Path) -> tuple[tuple, tuple]:
    """Capture every application table and schema object through a read-only connection."""
    uri = f"{path.absolute().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        schema = tuple(
            db.execute(
                "SELECT type,name,tbl_name,sql FROM sqlite_schema "
                "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name,tbl_name"
            )
        )
        tables = tuple(
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_schema "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        )
        rows = tuple(
            (table, tuple(db.execute(f'SELECT * FROM "{table}" ORDER BY rowid')))
            for table in tables
        )
    return schema, rows


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
    CatalogRuntimeAcceptanceAuthority(membership, receipt_authority)
    catalog.chmod(0o600)
    receipts.chmod(0o600)
    config = CatalogRuntimeDeploymentConfiguration.from_protected_mapping(
        {
            "catalog_state_path": str(catalog.absolute()),
            "receipt_metadata_path": str(receipts.absolute()),
        }
    )
    return config, catalog, receipts


def test_protected_configuration_has_exact_keys_no_defaults_and_absolute_paths(tmp_path) -> None:
    for value in (
        None,
        {},
        {"catalog_state_path": "/x"},
        {"catalog_state_path": "/x", "receipt_metadata_path": "/y", "extra": "/z"},
    ):
        with pytest.raises(ValueError, match="CONFIGURATION_INVALID"):
            CatalogRuntimeDeploymentConfiguration.from_protected_mapping(value)
    with pytest.raises(ValueError, match="PATH_NOT_ABSOLUTE"):
        CatalogRuntimeDeploymentConfiguration.from_protected_mapping(
            {"catalog_state_path": "catalog.db", "receipt_metadata_path": "receipts.db"}
        )


@pytest.mark.parametrize("bad_target", ["directory", "file"])
def test_permission_qualification_blocks_without_repair(tmp_path, keyring, bad_target) -> None:
    config, catalog, _receipts = _offline_storage(tmp_path, custody=True, admission=True)
    target = tmp_path if bad_target == "directory" else catalog
    before = target.read_bytes() if target.is_file() else tuple(target.iterdir())

    result = CoreHostCatalogRuntimeLifecycle(
        config,
        permission_qualifier=_FixedQualifier(CatalogPermissionQualification.INVALID),
    ).start_after_recovery()

    assert result.reason == "CATALOG_RUNTIME_PERMISSIONS_INVALID"
    after = target.read_bytes() if target.is_file() else tuple(target.iterdir())
    assert after == before


def test_non_posix_permission_proof_fails_closed(tmp_path, keyring) -> None:
    config, _catalog, _receipts = _offline_storage(tmp_path, custody=True, admission=True)

    result = CoreHostCatalogRuntimeLifecycle(
        config, permission_qualifier=UnavailableCatalogPermissionQualifier()
    ).start_after_recovery()

    assert result.reason == "CATALOG_RUNTIME_PLATFORM_PERMISSION_PROOF_UNAVAILABLE"


def test_missing_custody_blocks_without_key_generation_or_fetch(
    tmp_path, keyring, monkeypatch
) -> None:
    config, _catalog, _receipts = _offline_storage(tmp_path, custody=False, admission=True)
    monkeypatch.setattr(_BinanceSpotCatalogProducer, "fetch", lambda self: pytest.fail("fetch"))
    lifecycle = CoreHostCatalogRuntimeLifecycle(config, permission_qualifier=QUALIFIED)

    assert lifecycle.start_after_recovery().reason == "CATALOG_RUNTIME_RECEIPT_CUSTODY_MISSING"
    assert lifecycle.fetch_catalog_once() is None
    assert keyring == {}


def test_missing_admission_blocks_without_grant_or_fetch(tmp_path, keyring, monkeypatch) -> None:
    config, catalog, _receipts = _offline_storage(tmp_path, custody=True, admission=False)
    monkeypatch.setattr(_BinanceSpotCatalogProducer, "fetch", lambda self: pytest.fail("fetch"))
    lifecycle = CoreHostCatalogRuntimeLifecycle(config, permission_qualifier=QUALIFIED)

    assert lifecycle.start_after_recovery().reason == "CATALOG_RUNTIME_PRODUCER_ADMISSION_MISSING"
    assert lifecycle.fetch_catalog_once() is None
    assert SourceProducerMembershipAuthority(SQLiteMembershipCarrier(catalog))._carrier.read() == ()


def test_corrupt_authority_blocks_without_repair_or_fetch(tmp_path, keyring, monkeypatch) -> None:
    config, catalog, _receipts = _offline_storage(tmp_path, custody=True, admission=True)
    with sqlite3.connect(catalog) as db:
        db.execute("UPDATE membership_authority_store_metadata SET authority_domain='corrupt'")
    before = catalog.read_bytes()
    monkeypatch.setattr(_BinanceSpotCatalogProducer, "fetch", lambda self: pytest.fail("fetch"))

    result = CoreHostCatalogRuntimeLifecycle(
        config, permission_qualifier=QUALIFIED
    ).start_after_recovery()

    assert result.reason == "CATALOG_RUNTIME_AUTHORITY_REPLAY_FAILED"
    assert catalog.read_bytes() == before


def test_restart_and_pre_fetch_crash_reopen_without_side_effects(
    tmp_path, keyring, monkeypatch
) -> None:
    config, catalog, receipts = _offline_storage(tmp_path, custody=True, admission=True)
    monkeypatch.setattr(_BinanceSpotCatalogProducer, "fetch", lambda self: pytest.fail("fetch"))
    before = (
        _sqlite_semantic_snapshot(catalog),
        _receipt_semantic_snapshot(receipts),
        dict(keyring),
    )

    first = CoreHostCatalogRuntimeLifecycle(config, permission_qualifier=QUALIFIED)
    assert first.start_after_recovery().ready
    del first  # process crash before explicit fetch
    second = CoreHostCatalogRuntimeLifecycle(config, permission_qualifier=QUALIFIED)
    assert second.start_after_recovery().ready
    assert second.start_after_recovery().ready

    assert (
        _sqlite_semantic_snapshot(catalog),
        _receipt_semantic_snapshot(receipts),
        keyring,
    ) == before


def test_receipt_reopen_rejects_missing_head_without_repair_or_fetch(
    tmp_path, keyring, monkeypatch
) -> None:
    config, _catalog, receipts = _offline_storage(tmp_path, custody=True, admission=True)
    with sqlite3.connect(receipts) as db:
        db.execute("DELETE FROM catalog_receipt_authority_head")
    before = (_receipt_semantic_snapshot(receipts), dict(keyring))
    monkeypatch.setattr(_BinanceSpotCatalogProducer, "fetch", lambda self: pytest.fail("fetch"))

    result = CoreHostCatalogRuntimeLifecycle(
        config, permission_qualifier=QUALIFIED
    ).start_after_recovery()

    assert not result.ready
    assert result.reason == "CATALOG_RUNTIME_AUTHORITY_REPLAY_FAILED"
    assert (_receipt_semantic_snapshot(receipts), keyring) == before


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
        assert events == [
            "lock",
            "recovery",
            "catalog_gate",
            "publication:before",
            "publication:after",
        ]
    finally:
        host.close()


def test_corehost_alone_owns_explicit_one_shot_fetch(tmp_path, keyring, monkeypatch) -> None:
    storage = tmp_path / "catalog-authority"
    storage.mkdir(mode=0o700)
    config, _catalog, _receipts = _offline_storage(storage, custody=True, admission=True)
    state = tmp_path / "state.sqlite3"
    boundary, _ = _initialize(state)
    host, _stores, _registries = _real_host(
        state,
        boundary,
        catalog_runtime_configuration=config,
        catalog_permission_qualifier_factory=lambda: QUALIFIED,
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


def test_native_windows_selection_remains_unavailable() -> None:
    assert isinstance(
        native_catalog_permission_qualifier("nt"),
        UnavailableCatalogPermissionQualifier,
    )


def test_posix_qualifier_preserves_frozen_owner_and_mode_rules(tmp_path, keyring) -> None:
    config, _catalog, _receipts = _offline_storage(tmp_path, custody=False, admission=False)
    modes = {
        config.catalog_state_path.parent: 0o700,
        config.receipt_metadata_path.parent: 0o700,
        config.catalog_state_path: 0o600,
        config.receipt_metadata_path: 0o600,
    }

    def fake_stat(path):  # type: ignore[no-untyped-def]
        return SimpleNamespace(st_uid=42, st_mode=modes[path])

    qualifier = PosixCatalogPermissionQualifier(stat_reader=fake_stat, uid_reader=lambda: 42)
    assert qualifier.qualify(config.paths) is CatalogPermissionQualification.QUALIFIED
    modes[config.catalog_state_path] = 0o644
    assert qualifier.qualify(config.paths) is CatalogPermissionQualification.INVALID


def test_permission_contract_rejects_untyped_qualifier_and_result(tmp_path, keyring) -> None:
    config, _catalog, _receipts = _offline_storage(tmp_path, custody=False, admission=False)
    with pytest.raises(TypeError, match="CatalogPermissionQualifier required"):
        CoreHostCatalogRuntimeLifecycle(config, permission_qualifier=lambda: "QUALIFIED")  # type: ignore[arg-type]

    class BadQualifier(CatalogPermissionQualifier):
        def qualify(self, paths):  # type: ignore[no-untyped-def]
            del paths
            return "QUALIFIED"

    lifecycle = CoreHostCatalogRuntimeLifecycle(config, permission_qualifier=BadQualifier())
    with pytest.raises(TypeError, match="invalid result type"):
        lifecycle.start_after_recovery()

    state = tmp_path / "state.sqlite3"
    boundary, _ = _initialize(state)
    host, _stores, _registries = _real_host(
        state,
        boundary,
        catalog_runtime_configuration=config,
        catalog_permission_qualifier_factory=lambda: "QUALIFIED",  # type: ignore[return-value]
    )
    with pytest.raises(TypeError, match="CatalogPermissionQualifier required"):
        host.start()
