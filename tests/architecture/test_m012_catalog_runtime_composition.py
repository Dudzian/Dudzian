"""S9D-C25 production Catalog runtime composition regressions."""
import json
import os
from pathlib import Path
import sqlite3

import pytest

from bot_core.instruments.catalog_admission_receipt import (
    CatalogAdmissionReceiptAuthority,
    SQLiteCatalogAdmissionReceiptMetadataStore,
)
from bot_core.instruments.catalog_runtime_acceptance import (
    CatalogRuntimeAcceptanceAuthority,
    _BinanceSpotCatalogProducer,
)
from bot_core.instruments.catalog_runtime_composition import (
    CatalogRuntimeAuthorityPaths,
    compose_catalog_runtime_acceptance,
)
from bot_core.instruments.source_producer_membership import (
    SQLiteMembershipCarrier,
    SourceProducerMembershipAuthority,
)
from bot_core.security.keyring_storage import KeyringSecretStorage


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


def test_composition_reopens_exact_production_authorities_without_mutating_policy(
    tmp_path: Path, keyring: dict[str, str]
) -> None:
    paths = CatalogRuntimeAuthorityPaths(
        (tmp_path / "catalog.sqlite3").absolute(),
        (tmp_path / "receipts.sqlite3").absolute(),
    )
    receipts = CatalogAdmissionReceiptAuthority(
        SQLiteCatalogAdmissionReceiptMetadataStore(paths.receipt_metadata)
    )
    key_id = receipts.provision()
    membership = SourceProducerMembershipAuthority(SQLiteMembershipCarrier(paths.catalog_state))
    grant = membership.admit_release_grant("core_release_1_45_binance_spot")

    runtime = compose_catalog_runtime_acceptance(paths)

    assert type(runtime) is CatalogRuntimeAcceptanceAuthority
    assert runtime._receipts.receipts() == ()
    assert runtime._membership.resolve_current(
        _BinanceSpotCatalogProducer.identity,
        _BinanceSpotCatalogProducer.generation,
        "9999-01-01T00:00:00Z",
    ) == grant
    assert runtime._receipts.rotate() != key_id


def test_composition_does_not_provision_receipt_or_admit_producer(
    tmp_path: Path, keyring: dict[str, str], monkeypatch
) -> None:
    paths = CatalogRuntimeAuthorityPaths(
        (tmp_path / "catalog.sqlite3").absolute(),
        (tmp_path / "receipts.sqlite3").absolute(),
    )
    SQLiteMembershipCarrier(paths.catalog_state)
    SQLiteCatalogAdmissionReceiptMetadataStore(paths.receipt_metadata)
    monkeypatch.setattr(_BinanceSpotCatalogProducer, "fetch", lambda self: {"symbols": []})

    runtime = compose_catalog_runtime_acceptance(paths)

    assert runtime._membership._carrier.read() == ()
    assert runtime._receipts.receipts() == ()
    assert runtime.fetch_catalog_once("core_release_1_46_binance_spot") is None
    assert keyring == {}


@pytest.mark.parametrize("existing", [None, "catalog", "receipts"])
def test_composition_missing_storage_fails_atomically_without_creating_files(
    tmp_path: Path, existing: str | None
) -> None:
    catalog = (tmp_path / "catalog.sqlite3").absolute()
    receipts = (tmp_path / "receipts.sqlite3").absolute()
    if existing == "catalog":
        catalog.write_bytes(b"offline-existing-catalog")
    elif existing == "receipts":
        receipts.write_bytes(b"offline-existing-receipts")
    before = {
        path: path.read_bytes()
        for path in (catalog, receipts)
        if path.exists()
    }

    with pytest.raises(ValueError, match="CATALOG_RUNTIME_STORAGE_MISSING"):
        compose_catalog_runtime_acceptance(
            CatalogRuntimeAuthorityPaths(catalog, receipts)
        )

    assert {path: path.read_bytes() for path in before} == before
    missing = {catalog, receipts} - set(before)
    assert all(not path.exists() for path in missing)
    assert all(
        not Path(f"{path}{suffix}").exists()
        for path in (catalog, receipts)
        for suffix in ("-wal", "-shm")
    )


@pytest.mark.parametrize("relative", ["catalog.sqlite3", Path("state/catalog.sqlite3")])
def test_runtime_paths_reject_relative_locations(tmp_path: Path, relative: str | Path) -> None:
    with pytest.raises(ValueError, match="PATH_NOT_ABSOLUTE"):
        CatalogRuntimeAuthorityPaths(Path(relative), (tmp_path / "receipts.sqlite3").absolute())


def test_runtime_paths_reject_storage_alias_and_non_file_targets(tmp_path: Path) -> None:
    state = (tmp_path / "authority.sqlite3").absolute()
    with pytest.raises(ValueError, match="PATH_ALIAS"):
        CatalogRuntimeAuthorityPaths(state, state)
    with pytest.raises(ValueError, match="STATE_NOT_FILE"):
        CatalogRuntimeAuthorityPaths(tmp_path.absolute(), state)
    with pytest.raises(ValueError, match="PARENT_UNAVAILABLE"):
        CatalogRuntimeAuthorityPaths(
            (tmp_path / "missing" / "catalog.sqlite3").absolute(), state
        )


def test_runtime_paths_reject_distinct_hardlinks_to_the_same_physical_file(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "catalog.sqlite3"
    receipts = tmp_path / "receipts.sqlite3"
    catalog.touch()
    os.link(catalog, receipts)

    assert catalog != receipts
    assert os.path.samefile(catalog, receipts)
    with pytest.raises(ValueError, match="CATALOG_RUNTIME_PHYSICAL_ALIAS"):
        CatalogRuntimeAuthorityPaths(catalog.absolute(), receipts.absolute())


@pytest.mark.parametrize("link_side", ["receipts", "catalog"])
def test_runtime_paths_reject_direct_symlink_alias(
    tmp_path: Path, link_side: str
) -> None:
    target = tmp_path / "authority.sqlite3"
    target.touch()
    catalog = target if link_side == "receipts" else tmp_path / "catalog-link.sqlite3"
    receipts = target if link_side == "catalog" else tmp_path / "receipts-link.sqlite3"
    (receipts if link_side == "receipts" else catalog).symlink_to(target)

    assert catalog != receipts
    assert os.path.samefile(catalog, receipts)
    with pytest.raises(ValueError, match="CATALOG_RUNTIME_PATH_ALIAS"):
        CatalogRuntimeAuthorityPaths(catalog.absolute(), receipts.absolute())


def test_runtime_paths_reject_parent_symlink_alias(tmp_path: Path) -> None:
    physical_parent = tmp_path / "physical"
    physical_parent.mkdir()
    parent_alias = tmp_path / "parent-alias"
    parent_alias.symlink_to(physical_parent, target_is_directory=True)
    catalog = physical_parent / "authority.sqlite3"
    receipts = parent_alias / "authority.sqlite3"

    assert catalog != receipts
    with pytest.raises(ValueError, match="CATALOG_RUNTIME_PATH_ALIAS"):
        CatalogRuntimeAuthorityPaths(catalog.absolute(), receipts.absolute())


@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
@pytest.mark.parametrize("side", ["catalog", "receipts"])
def test_runtime_paths_reserve_both_sqlite_wal_sidecar_namespaces(
    tmp_path: Path, suffix: str, side: str
) -> None:
    main = (tmp_path / "authority.sqlite3").absolute()
    sidecar = Path(f"{main}{suffix}")
    catalog, receipts = (main, sidecar) if side == "catalog" else (sidecar, main)

    with pytest.raises(ValueError, match="CATALOG_RUNTIME_SQLITE_SIDECAR_ALIAS"):
        CatalogRuntimeAuthorityPaths(catalog, receipts)


@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
@pytest.mark.parametrize("side", ["catalog", "receipts"])
def test_runtime_paths_reject_main_hardlinked_to_opposite_sqlite_sidecar(
    tmp_path: Path, suffix: str, side: str
) -> None:
    catalog = tmp_path / "catalog.sqlite3"
    receipts = tmp_path / "receipts.sqlite3"
    sidecar_owner = catalog if side == "catalog" else receipts
    opposite_main = receipts if side == "catalog" else catalog
    sidecar_owner.touch()
    sidecar = Path(f"{sidecar_owner}{suffix}")
    sidecar.touch()
    os.link(sidecar, opposite_main)

    assert catalog != receipts
    assert not os.path.samefile(catalog, receipts)
    assert os.path.samefile(sidecar, opposite_main)
    with pytest.raises(
        ValueError, match="CATALOG_RUNTIME_SQLITE_SIDECAR_PHYSICAL_ALIAS"
    ):
        CatalogRuntimeAuthorityPaths(catalog.absolute(), receipts.absolute())


@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
@pytest.mark.parametrize("side", ["catalog", "receipts"])
def test_runtime_paths_reject_dangling_sidecar_symlink_to_future_opposite_main(
    tmp_path: Path, suffix: str, side: str
) -> None:
    catalog = tmp_path / "catalog.sqlite3"
    receipts = tmp_path / "receipts.sqlite3"
    existing_main = catalog if side == "catalog" else receipts
    future_opposite_main = receipts if side == "catalog" else catalog
    existing_main.touch()
    sidecar = Path(f"{existing_main}{suffix}")
    sidecar.symlink_to(future_opposite_main)

    assert sidecar.is_symlink()
    assert not sidecar.exists()
    assert not future_opposite_main.exists()
    with pytest.raises(ValueError, match="CATALOG_RUNTIME_SQLITE_SIDECAR_SYMLINK"):
        CatalogRuntimeAuthorityPaths(catalog.absolute(), receipts.absolute())


@pytest.mark.parametrize(
    ("source_suffix", "target_suffix"),
    [("-wal", "-wal"), ("-wal", "-shm"), ("-shm", "-wal"), ("-shm", "-shm")],
)
@pytest.mark.parametrize("source_side", ["catalog", "receipts"])
def test_runtime_paths_reject_dangling_symlink_across_sidecar_namespaces(
    tmp_path: Path, source_suffix: str, target_suffix: str, source_side: str
) -> None:
    catalog = tmp_path / "catalog.sqlite3"
    receipts = tmp_path / "receipts.sqlite3"
    source_main = catalog if source_side == "catalog" else receipts
    target_main = receipts if source_side == "catalog" else catalog
    source = Path(f"{source_main}{source_suffix}")
    target = Path(f"{target_main}{target_suffix}")
    source.symlink_to(target)

    assert source.is_symlink()
    assert not source.exists()
    assert not target.exists()
    with pytest.raises(ValueError, match="CATALOG_RUNTIME_SQLITE_SIDECAR_SYMLINK"):
        CatalogRuntimeAuthorityPaths(catalog.absolute(), receipts.absolute())


@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
@pytest.mark.parametrize("side", ["catalog", "receipts"])
def test_runtime_paths_reject_opposite_main_reached_through_sidecar_symlink(
    tmp_path: Path, suffix: str, side: str
) -> None:
    catalog = tmp_path / "catalog.sqlite3"
    receipts = tmp_path / "receipts.sqlite3"
    catalog.touch()
    receipts.touch()
    sidecar_owner = catalog if side == "catalog" else receipts
    opposite_main = receipts if side == "catalog" else catalog
    sidecar = Path(f"{sidecar_owner}{suffix}")
    sidecar.symlink_to(opposite_main)

    assert os.path.samefile(sidecar, opposite_main)
    with pytest.raises(
        ValueError, match="CATALOG_RUNTIME_SQLITE_SIDECAR_SYMLINK"
    ):
        CatalogRuntimeAuthorityPaths(catalog.absolute(), receipts.absolute())


@pytest.mark.parametrize(
    ("catalog_suffix", "receipt_suffix"),
    [("-wal", "-wal"), ("-wal", "-shm"), ("-shm", "-wal"), ("-shm", "-shm")],
)
def test_runtime_paths_reject_physical_alias_across_sidecar_namespaces(
    tmp_path: Path, catalog_suffix: str, receipt_suffix: str
) -> None:
    catalog = tmp_path / "catalog.sqlite3"
    receipts = tmp_path / "receipts.sqlite3"
    catalog.touch()
    receipts.touch()
    catalog_sidecar = Path(f"{catalog}{catalog_suffix}")
    receipt_sidecar = Path(f"{receipts}{receipt_suffix}")
    catalog_sidecar.touch()
    os.link(catalog_sidecar, receipt_sidecar)

    assert os.path.samefile(catalog_sidecar, receipt_sidecar)
    with pytest.raises(
        ValueError, match="CATALOG_RUNTIME_SQLITE_SIDECAR_PHYSICAL_ALIAS"
    ):
        CatalogRuntimeAuthorityPaths(catalog.absolute(), receipts.absolute())


def test_runtime_paths_preserve_distinct_existing_sidecar_objects(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.sqlite3"
    receipts = tmp_path / "receipts.sqlite3"
    paths = (catalog, Path(f"{catalog}-wal"), Path(f"{catalog}-shm"),
             receipts, Path(f"{receipts}-wal"), Path(f"{receipts}-shm"))
    for path in paths:
        path.touch()

    validated = CatalogRuntimeAuthorityPaths(catalog.absolute(), receipts.absolute())

    assert validated.catalog_state == catalog
    assert validated.receipt_metadata == receipts
    assert all(path.is_file() for path in paths)


def test_runtime_paths_reject_real_wal_hardlinked_as_receipt_main(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.sqlite3"
    receipts = tmp_path / "receipts.sqlite3"
    with sqlite3.connect(catalog) as connection:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        connection.execute("CREATE TABLE durable_fact(value TEXT NOT NULL)")
        connection.execute("INSERT INTO durable_fact VALUES('accepted')")
        connection.commit()
        catalog_wal = Path(f"{catalog}-wal")
        assert catalog_wal.is_file()
        os.link(catalog_wal, receipts)

        assert not os.path.samefile(catalog, receipts)
        assert os.path.samefile(catalog_wal, receipts)
        with pytest.raises(
            ValueError, match="CATALOG_RUNTIME_SQLITE_SIDECAR_PHYSICAL_ALIAS"
        ):
            CatalogRuntimeAuthorityPaths(catalog.absolute(), receipts.absolute())


def test_composition_rejects_subclassed_path_policy(tmp_path: Path) -> None:
    class SubstitutePaths(CatalogRuntimeAuthorityPaths):
        pass

    paths = SubstitutePaths(
        (tmp_path / "catalog.sqlite3").absolute(),
        (tmp_path / "receipts.sqlite3").absolute(),
    )
    with pytest.raises(TypeError, match="exact CatalogRuntimeAuthorityPaths"):
        compose_catalog_runtime_acceptance(paths)


def test_machine_contract_records_composition_without_closing_s9d() -> None:
    root = Path(__file__).resolve().parents[2]
    contract = json.loads(
        (root / "docs/architecture/cryptohunter_product_architecture/"
         "audit_observability_alerts_and_updater.json").read_text(encoding="utf-8")
    )
    membership = contract["release_artifact_authority"][
        "accepted_source_producer_membership_authority"
    ]
    composition = membership["catalog_runtime_acceptance_authority"]

    assert composition["runtime_composition_symbol"].endswith(
        ".compose_catalog_runtime_acceptance"
    )
    assert composition["runtime_composition_status"] == (
        "IMPLEMENTED_CURRENT_TREE_NOT_FINAL_ACCEPTANCE"
    )
    assert composition["physical_location_separation"] == (
        "CANONICAL_PATH_AND_EXISTING_CROSS_NAMESPACE_FILE_IDENTITY_AND_SQLITE_WAL_NAMESPACE"
    )
    assert composition["sqlite_sidecar_symlink_policy"] == (
        "DENY_ANY_PREEXISTING_WAL_OR_SHM_SYMLINK_ENTRY_INCLUDING_DANGLING"
    )
    assert composition["parent_directory_race_boundary"] == (
        "CORE_OWNED_NOT_WRITABLE_BY_UNPRIVILEGED_ONLINE_PRINCIPALS"
    )
    assert membership["catalog_runtime_acceptance"] == (
        "IMPLEMENTED_CURRENT_TREE_PENDING_DEPLOYMENT_ACCEPTANCE"
    )
    assert membership["C25"] == "BLOCKED"
    assert membership["S9D"] == "OPEN"


def test_next_c25_lifecycle_deployment_contract_is_implemented_current_tree() -> None:
    root = Path(__file__).resolve().parents[2]
    contract = json.loads(
        (root / "docs/architecture/cryptohunter_product_architecture/"
         "audit_observability_alerts_and_updater.json").read_text(encoding="utf-8")
    )
    membership = contract["release_artifact_authority"][
        "accepted_source_producer_membership_authority"
    ]
    lifecycle = membership["catalog_runtime_acceptance_authority"][
        "next_c25_lifecycle_deployment_contract"
    ]

    assert lifecycle["status"] == "IMPLEMENTED_CURRENT_TREE_PENDING_DEPLOYMENT_ACCEPTANCE"
    assert lifecycle["contract_id"] == (
        "M0.12-S9D-C25-CATALOG-RUNTIME-LIFECYCLE-DEPLOYMENT-V1"
    )
    assert set(lifecycle) == {
        "status", "contract_id", "owner", "composition_hook", "configuration_source",
        "path_default_policy", "directory_ownership", "file_ownership",
        "posix_directory_mode", "posix_file_mode", "non_posix_acl_policy",
        "storage_creation_authority", "provisioning_authority", "runtime_reopen_authority",
        "startup_missing_database", "startup_missing_receipt_custody",
        "startup_missing_producer_admission", "startup_corrupt_authority",
        "restart_behavior", "crash_behavior", "repeated_start_idempotency",
        "authority_lifetime", "one_shot_fetch_owner", "background_scheduler_dependency",
        "trading_universe_auto_selection_dependency", "acceptance_evidence",
        "status_transition", "c25_unblocking_condition", "s9d_advancement_condition",
    }
    assert lifecycle["path_default_policy"].startswith("NO_DEFAULTS")
    assert lifecycle["posix_directory_mode"] == "0700"
    assert lifecycle["posix_file_mode"] == "0600"
    assert "never CoreHost startup" in lifecycle["provisioning_authority"]
    assert lifecycle["background_scheduler_dependency"].startswith(
        "NOT_REQUIRED_FOR_C25_COMPLETION"
    )
    assert lifecycle["trading_universe_auto_selection_dependency"].startswith(
        "NOT_REQUIRED_FOR_C25_COMPLETION"
    )
    assert len(lifecycle["acceptance_evidence"]) == 6
    assert lifecycle["status_transition"].startswith(
        "PARTIAL_PROPOSED_UNACCEPTED_1.46.0 -> "
        "IMPLEMENTED_CURRENT_TREE_PENDING_DEPLOYMENT_ACCEPTANCE"
    )
    assert membership["C25"] == "BLOCKED"
    assert membership["S9D"] == "OPEN"
