from __future__ import annotations

from datetime import datetime, timezone
import json
import sqlite3
from threading import Event, Thread

import pytest

from bot_core.instruments.catalog_runtime_acceptance import (
    CatalogRuntimeAcceptanceAuthority,
    _BinanceSpotCatalogProducer,
    _METADATA_FP_DOMAIN,
    _canonical,
    _digest,
    _normalize_binance,
)
from bot_core.instruments.catalog_admission_receipt import (
    CatalogAdmissionReceiptAuthority,
    SQLiteCatalogAdmissionReceiptMetadataStore,
)
from bot_core.instruments.testing_catalog_admission_receipt import TestCatalogAdmissionReceiptAuthority
from bot_core.security.keyring_storage import KeyringSecretStorage
from bot_core.instruments.catalog_projection_oracle import (
    SOURCE_FIELDS,
    SOURCE_FINGERPRINT_FIELDS,
    SOURCE_MEMBER_FIELDS,
    _fingerprint,
    validate_accepted_source_catalog_snapshot,
    validate_accepted_source_snapshot_membership_evidence,
    validate_canonical_catalog_context_graph,
)
from bot_core.instruments.source_producer_membership import (
    SQLiteMembershipCarrier,
    SourceProducerMembershipAuthority,
)


def _payload(*, server_time: int | None = None, tick: str = "0.01", extra: object = None) -> dict[str, object]:
    now = server_time or int(datetime.now(timezone.utc).timestamp() * 1000) - 1000
    product: dict[str, object] = {
        "symbol": "BTCUSDT", "status": "TRADING", "baseAsset": "BTC",
        "baseAssetPrecision": 8, "quoteAsset": "USDT", "quotePrecision": 8,
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": tick},
            {"filterType": "LOT_SIZE", "stepSize": "0.00001", "minQty": "0.00001", "maxQty": "9000"},
            {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
        ],
    }
    if extra is not None:
        product["ignoredReleaseOwnedField"] = extra
    return {"timezone": "UTC", "serverTime": now, "rateLimits": [], "exchangeFilters": [], "symbols": [product]}


@pytest.fixture
def authority(tmp_path, monkeypatch):
    secrets: dict[str, str] = {}
    monkeypatch.setattr(KeyringSecretStorage, "__init__", lambda storage, **kwargs: setattr(storage, "_catalog_test_values", secrets))
    monkeypatch.setattr(KeyringSecretStorage, "get_secret", lambda storage, key: storage._catalog_test_values.get(key))
    monkeypatch.setattr(KeyringSecretStorage, "set_secret", lambda storage, key, value: storage._catalog_test_values.__setitem__(key, value))
    receipts = CatalogAdmissionReceiptAuthority(
        SQLiteCatalogAdmissionReceiptMetadataStore(tmp_path / "receipts.sqlite3")
    )
    receipts.provision()
    carrier = SQLiteMembershipCarrier(tmp_path / "authority.sqlite3")
    membership = SourceProducerMembershipAuthority(carrier)
    assert membership.admit_release_grant("core_release_1_45_binance_spot") is not None
    values = [_payload()]
    monkeypatch.setattr(_BinanceSpotCatalogProducer, "fetch", lambda self: values[-1])
    return CatalogRuntimeAcceptanceAuthority(membership, receipts), membership, values, carrier


def test_core_invokes_release_producer_and_restart_preserves_history(authority) -> None:
    runtime, membership, values, carrier = authority
    first = runtime.fetch_catalog_once("core_release_1_46_binance_spot")
    assert first is not None
    assert set(first.to_mapping()) == SOURCE_FIELDS
    assert set(first.member_source_product_metadata_versions[0]) == SOURCE_MEMBER_FIELDS
    assert validate_accepted_source_catalog_snapshot(first.to_mapping())
    assert validate_accepted_source_snapshot_membership_evidence(first.to_mapping(), membership)
    assert validate_canonical_catalog_context_graph(
        workspace_catalog_projections_by_id={},
        accepted_source_catalog_snapshots_by_id={first.accepted_source_catalog_snapshot_id: first.to_mapping()},
        instruments_by_id={}, instrument_history_by_id={},
        source_producer_membership_authority=membership,
    )
    grant = membership.resolve_historical(first.accepted_source_producer_membership_id, first.source_producer_generation, first.source_producer_membership_fingerprint, _BinanceSpotCatalogProducer.identity, first.effective_at_utc)
    assert grant is not None
    assert first.acceptance_status == "VALID"
    assert first.completeness_status == "COMPLETE"
    assert first.completeness_evidence["policy"] == "BINANCE_EXCHANGE_INFO_ATOMIC_V1"
    assert first.stale_after_utc > first.effective_at_utc >= first.observed_at_utc
    assert runtime.fetch_catalog_once("core_release_1_46_binance_spot") == first

    restarted = CatalogRuntimeAcceptanceAuthority(SourceProducerMembershipAuthority(carrier), runtime._receipts)
    assert restarted.snapshots() == (first,)
    old_metadata = restarted.metadata_history("binance", "SPOT", "BTCUSDT")
    assert len(old_metadata) == 1

    values.append(_payload(server_time=int(datetime.now(timezone.utc).timestamp() * 1000) - 500, tick="0.1"))
    second = restarted.fetch_catalog_once("core_release_1_46_binance_spot")
    assert second is not None and second.previous_snapshot_id == first.accepted_source_catalog_snapshot_id
    history = restarted.metadata_history("binance", "SPOT", "BTCUSDT")
    assert len(history) == 2
    assert history[1].previous_source_metadata_version_id == history[0].source_metadata_version_id


def test_production_rejects_test_receipt_authority(authority, tmp_path) -> None:
    _runtime, membership, _values, _carrier = authority
    test_receipts = TestCatalogAdmissionReceiptAuthority(
        tmp_path / "test-receipts.sqlite3", deterministic_seed=b"x" * 32
    )
    with pytest.raises(TypeError, match="exact production Catalog"):
        CatalogRuntimeAcceptanceAuthority(membership, test_receipts)  # type: ignore[arg-type]


def test_receipt_relation_is_required_exact_and_idempotently_reused(authority) -> None:
    runtime, membership, _values, carrier = authority
    first = runtime.fetch_catalog_once("core_release_1_46_binance_spot")
    assert first is not None
    receipt_count = len(runtime._receipts.receipts())
    finalization_count = len(runtime._receipts.finalizations())
    with _sqlite(carrier) as db:
        relation = db.execute(
            "SELECT catalog_admission_receipt_id FROM catalog_snapshot_admission_receipts"
        ).fetchone()
    assert runtime.fetch_catalog_once("core_release_1_46_binance_spot") == first
    assert len(runtime._receipts.receipts()) == receipt_count
    assert len(runtime._receipts.finalizations()) == finalization_count
    with _sqlite(carrier) as db:
        assert db.execute(
            "SELECT catalog_admission_receipt_id FROM catalog_snapshot_admission_receipts"
        ).fetchone() == relation
        db.execute("DELETE FROM catalog_snapshot_admission_receipts")
    with pytest.raises(ValueError, match="CORRUPT_CATALOG_AUTHORITY"):
        CatalogRuntimeAcceptanceAuthority(membership, runtime._receipts)


def test_orphan_genuine_receipt_does_not_create_catalog_fact(authority) -> None:
    runtime, membership, _values, carrier = authority
    receipt = runtime._receipts.prepare(
        catalog_commitment_sha256="a" * 64,
        membership_commitment_sha256="b" * 64,
        accepted_at_utc="2030-01-02T03:04:05Z",
    )
    assert not runtime._receipts.verify(receipt)
    restarted = CatalogRuntimeAcceptanceAuthority(membership, runtime._receipts)
    assert restarted.snapshots() == ()
    with _sqlite(carrier) as db:
        assert db.execute("SELECT COUNT(*) FROM catalog_snapshot_admission_receipts").fetchone() == (0,)


def test_catalog_commit_without_finalization_fails_closed(authority) -> None:
    runtime, membership, values, _carrier = authority
    observation = _normalize_binance(values[-1])
    assert observation is not None
    with runtime._connect() as db:
        db.execute("BEGIN IMMEDIATE")
        snapshot, receipt, _storage = runtime._accept_locked(
            db, _BinanceSpotCatalogProducer(), observation
        )
        assert snapshot is not None and receipt is not None
        db.commit()
    assert not runtime._receipts.verify(receipt)
    with pytest.raises(ValueError, match="CORRUPT_CATALOG_AUTHORITY"):
        CatalogRuntimeAcceptanceAuthority(membership, runtime._receipts)


def test_prepared_orphan_cannot_authorize_coherent_sql_catalog_mint(authority) -> None:
    runtime, membership, values, carrier = authority
    observation = _normalize_binance(values[-1])
    assert observation is not None
    with runtime._connect() as db:
        db.execute("BEGIN IMMEDIATE")
        _snapshot, receipt, _storage = runtime._accept_locked(
            db, _BinanceSpotCatalogProducer(), observation
        )
        captured = {
            table: db.execute(f"SELECT * FROM {table}").fetchall()
            for table in (
                "source_metadata_versions", "source_metadata_authority_head",
                "accepted_catalog_snapshots", "catalog_snapshot_admission_receipts",
                "catalog_authority_head",
            )
        }
        db.rollback()
    assert receipt is not None and not runtime._receipts.verify(receipt)
    with _sqlite(carrier) as db:
        for table in ("source_metadata_versions", "accepted_catalog_snapshots",
                      "catalog_snapshot_admission_receipts"):
            placeholders = ",".join("?" for _ in captured[table][0])
            db.executemany(f"INSERT INTO {table} VALUES({placeholders})", captured[table])
        for table in ("source_metadata_authority_head", "catalog_authority_head"):
            db.execute(f"DELETE FROM {table}")
            placeholders = ",".join("?" for _ in captured[table][0])
            db.executemany(f"INSERT INTO {table} VALUES({placeholders})", captured[table])
    with pytest.raises(ValueError, match="CORRUPT_CATALOG_AUTHORITY"):
        CatalogRuntimeAcceptanceAuthority(membership, runtime._receipts)


def test_receipt_key_rotation_and_revocation_are_current_trust(authority) -> None:
    runtime, _membership, values, _carrier = authority
    first = runtime.fetch_catalog_once("core_release_1_46_binance_spot")
    assert first is not None
    first_receipt = runtime._receipts.receipts()[0]
    new_key = runtime._receipts.rotate()
    assert runtime.snapshots() == (first,)  # VERIFY_ONLY remains trusted.
    values.append(_payload(server_time=int(datetime.now(timezone.utc).timestamp() * 1000) - 100))
    second = runtime.fetch_catalog_once("core_release_1_46_binance_spot")
    assert second is not None
    assert runtime._receipts.receipts()[-1].key_id == new_key
    runtime._receipts.revoke(first_receipt.key_id)
    with pytest.raises(ValueError, match="CORRUPT_CATALOG_AUTHORITY"):
        runtime.snapshots()


def test_catalog_valid_prefix_rollback_is_rejected_by_finalization_closure(authority) -> None:
    runtime, membership, values, carrier = authority
    first = runtime.fetch_catalog_once("core_release_1_46_binance_spot")
    assert first is not None
    with _sqlite(carrier) as db:
        catalog_head_one = db.execute(
            "SELECT * FROM catalog_authority_head WHERE singleton=1"
        ).fetchone()
        metadata_head_one = db.execute(
            "SELECT * FROM source_metadata_authority_head WHERE singleton=1"
        ).fetchone()
    values.append(_payload(
        server_time=int(datetime.now(timezone.utc).timestamp() * 1000) - 100,
        tick="0.1",
    ))
    second = runtime.fetch_catalog_once("core_release_1_46_binance_spot")
    assert second is not None
    assert len(runtime.snapshots()) == len(runtime._receipts.finalizations()) == 2
    with _sqlite(carrier) as db:
        db.execute(
            "DELETE FROM catalog_snapshot_admission_receipts "
            "WHERE accepted_source_catalog_snapshot_id=?", (second.accepted_source_catalog_snapshot_id,)
        )
        db.execute("DELETE FROM accepted_catalog_snapshots WHERE snapshot_sequence=2")
        db.execute("DELETE FROM source_metadata_versions WHERE source_metadata_sequence>1")
        db.execute("DELETE FROM catalog_authority_head")
        db.execute("INSERT INTO catalog_authority_head VALUES(?,?,?,?)", catalog_head_one)
        db.execute("DELETE FROM source_metadata_authority_head")
        db.execute(
            "INSERT INTO source_metadata_authority_head VALUES(?,?,?,?)", metadata_head_one
        )
    assert len(runtime._receipts.finalizations()) == 2
    with pytest.raises(ValueError, match="CORRUPT_CATALOG_AUTHORITY"):
        CatalogRuntimeAcceptanceAuthority(membership, runtime._receipts)


def test_complete_catalog_substore_reset_with_finalization_is_rejected(authority) -> None:
    runtime, membership, _values, carrier = authority
    assert runtime.fetch_catalog_once("core_release_1_46_binance_spot") is not None
    assert len(runtime._receipts.finalizations()) == 1
    with _sqlite(carrier) as db:
        for table in (
            "catalog_snapshot_admission_receipts", "accepted_catalog_snapshots",
            "catalog_authority_head", "source_metadata_versions",
            "source_metadata_authority_head", "catalog_authority_metadata",
        ):
            db.execute(f"DROP TABLE {table}")
    with pytest.raises(ValueError, match="CORRUPT_CATALOG_AUTHORITY"):
        CatalogRuntimeAcceptanceAuthority(membership, runtime._receipts)


def test_public_boundary_denies_factory_payload_time_and_identity_substitution(authority) -> None:
    runtime, _membership, _values, _carrier = authority
    for attack in (None, True, 1, [], {}, _BinanceSpotCatalogProducer, "binance_spot", "fake"):
        assert runtime.fetch_catalog_once(attack) is None
    assert not hasattr(runtime.fetch_catalog_once, "factory")


@pytest.mark.parametrize("bad", [None, True, 1, [], {}, {"symbols": []}, {"serverTime": float("nan"), "symbols": []}])
def test_malformed_observations_fail_closed(authority, monkeypatch, bad) -> None:
    runtime, _membership, _values, _carrier = authority
    monkeypatch.setattr(_BinanceSpotCatalogProducer, "fetch", lambda self: bad)
    assert runtime.fetch_catalog_once("core_release_1_46_binance_spot") is None


def test_future_observation_duplicate_symbol_and_bad_decimal_deny(authority, monkeypatch) -> None:
    runtime, _membership, _values, _carrier = authority
    future = int(datetime.now(timezone.utc).timestamp() * 1000) + 60_000
    attacks = [_payload(server_time=future), _payload(tick="0"), _payload(tick="NaN")]
    duplicate = _payload()
    duplicate["symbols"] = [duplicate["symbols"][0], duplicate["symbols"][0]]
    attacks.append(duplicate)
    negative_precision = _payload()
    negative_precision["symbols"][0]["baseAssetPrecision"] = -1
    attacks.append(negative_precision)
    for bad in attacks:
        monkeypatch.setattr(_BinanceSpotCatalogProducer, "fetch", lambda self, value=bad: value)
        assert runtime.fetch_catalog_once("core_release_1_46_binance_spot") is None


def test_revoked_membership_cannot_accept_new_snapshot_but_old_replays(authority) -> None:
    runtime, membership, values, _carrier = authority
    old = runtime.fetch_catalog_once("core_release_1_46_binance_spot")
    assert old is not None
    assert membership.admit_release_event("core_release_1_45_revoke_binance_spot") is not None
    values.append(_payload(server_time=int(datetime.now(timezone.utc).timestamp() * 1000) - 100))
    assert runtime.fetch_catalog_once("core_release_1_46_binance_spot") is None
    assert runtime.snapshots() == (old,)


def test_same_retrieval_id_with_different_content_denies(authority) -> None:
    runtime, _membership, values, _carrier = authority
    assert runtime.fetch_catalog_once("core_release_1_46_binance_spot") is not None
    before = (len(runtime._receipts.receipts()), len(runtime._receipts.finalizations()))
    original_time = values[-1]["serverTime"]
    values.append(_payload(server_time=original_time, tick="0.1"))
    assert runtime.fetch_catalog_once("core_release_1_46_binance_spot") is None
    assert len(runtime.snapshots()) == 1
    assert (len(runtime._receipts.receipts()), len(runtime._receipts.finalizations())) == before


def test_concurrent_revoke_committed_before_acceptance_fence_wins(authority, monkeypatch) -> None:
    runtime, membership, _values, _carrier = authority
    entered = Event()
    release = Event()
    result: list[object] = []

    def delayed_fetch(_self):
        entered.set()
        assert release.wait(5)
        return _payload()

    monkeypatch.setattr(_BinanceSpotCatalogProducer, "fetch", delayed_fetch)
    worker = Thread(target=lambda: result.append(runtime.fetch_catalog_once("core_release_1_46_binance_spot")))
    worker.start()
    assert entered.wait(5)
    assert membership.admit_release_event("core_release_1_45_revoke_binance_spot") is not None
    release.set()
    worker.join(5)
    assert not worker.is_alive()
    assert result == [None]
    assert runtime.snapshots() == ()


def _sqlite(carrier: SQLiteMembershipCarrier) -> sqlite3.Connection:
    return sqlite3.connect(carrier.path)


def test_orphan_valid_fingerprint_metadata_is_corruption(authority) -> None:
    runtime, _membership, _values, carrier = authority
    assert runtime.fetch_catalog_once("core_release_1_46_binance_spot") is not None
    old = runtime.metadata_history("binance", "SPOT", "BTCUSDT")[0]
    material = {
        "source_exchange_id": "binance", "market_type": "SPOT", "venue_symbol": "BTCUSDT",
        "metadata_version": 2, "previous_source_metadata_version_id": old.source_metadata_version_id,
        "normalized_metadata": {**old.normalized_metadata, "tick_size": "1"},
        "effective_at_utc": old.effective_at_utc,
    }
    record = {"source_metadata_version_id": "smeta_orphan", **material,
              "content_fingerprint": _digest(_METADATA_FP_DOMAIN, material, set())}
    with _sqlite(carrier) as db:
        db.execute("INSERT INTO source_metadata_versions VALUES(2,?,?,?,?,?,?,?,?,?)",
                   ("smeta_orphan", "binance", "SPOT", "BTCUSDT", 2,
                    "ascat_00000000000000000001", _canonical(record), "0" * 64, "1" * 64))
    with pytest.raises(ValueError, match="CORRUPT_CATALOG_AUTHORITY"):
        CatalogRuntimeAcceptanceAuthority(SourceProducerMembershipAuthority(carrier), runtime._receipts)


def test_metadata_mutation_with_recomputed_public_fingerprint_is_corruption(authority) -> None:
    runtime, _membership, _values, carrier = authority
    assert runtime.fetch_catalog_once("core_release_1_46_binance_spot") is not None
    with _sqlite(carrier) as db:
        canonical, = db.execute("SELECT canonical_record FROM source_metadata_versions").fetchone()
        record = json.loads(canonical)
        record["normalized_metadata"]["tick_size"] = "9"
        record["content_fingerprint"] = _digest(
            _METADATA_FP_DOMAIN, record,
            {"source_metadata_version_id", "content_fingerprint"},
        )
        db.execute("UPDATE source_metadata_versions SET canonical_record=?", (_canonical(record),))
    with pytest.raises(ValueError, match="CORRUPT_CATALOG_AUTHORITY"):
        runtime.metadata_history("binance", "SPOT", "BTCUSDT")


@pytest.mark.parametrize("mutation", ["wrong_tuple", "future_effective"])
def test_metadata_exact_relation_and_introduction_time_are_enforced(authority, mutation) -> None:
    runtime, _membership, _values, carrier = authority
    snapshot = runtime.fetch_catalog_once("core_release_1_46_binance_spot")
    assert snapshot is not None
    with _sqlite(carrier) as db:
        row = db.execute("SELECT source_metadata_sequence,introduced_by_snapshot_id,canonical_record,previous_metadata_digest FROM source_metadata_versions").fetchone()
        sequence, introduced_id, canonical, previous = row
        record = json.loads(canonical)
        if mutation == "wrong_tuple":
            record["venue_symbol"] = "ETHUSDT"
        else:
            record["effective_at_utc"] = "9999-01-01T00:00:00Z"
        record["content_fingerprint"] = _digest(
            _METADATA_FP_DOMAIN, record,
            {"source_metadata_version_id", "content_fingerprint"},
        )
        canonical = _canonical(record)
        digest = runtime._metadata_digest(sequence, introduced_id, canonical, previous)
        if mutation == "wrong_tuple":
            db.execute("UPDATE source_metadata_versions SET venue_symbol=?,canonical_record=?,metadata_record_digest=?", ("ETHUSDT", canonical, digest))
        else:
            db.execute("UPDATE source_metadata_versions SET canonical_record=?,metadata_record_digest=?", (canonical, digest))
        db.execute("UPDATE source_metadata_authority_head SET committed_digest=?", (digest,))
    with pytest.raises(ValueError, match="CORRUPT_CATALOG_AUTHORITY"):
        runtime.snapshots()


@pytest.mark.parametrize("mutation", [
    "DROP TABLE catalog_authority_metadata",
    "DELETE FROM catalog_authority_metadata",
    "UPDATE catalog_authority_metadata SET authority_domain='evil'",
    "UPDATE catalog_authority_metadata SET schema_version=999",
])
def test_catalog_domain_corruption_fails_live_and_on_restart(authority, mutation) -> None:
    runtime, membership, _values, carrier = authority
    assert runtime.fetch_catalog_once("core_release_1_46_binance_spot") is not None
    with _sqlite(carrier) as db:
        db.execute(mutation)
    with pytest.raises(ValueError, match="CATALOG_AUTHORITY_DOMAIN"):
        runtime.snapshots()
    with pytest.raises(ValueError, match="CATALOG_AUTHORITY_DOMAIN"):
        runtime.metadata_history("binance", "SPOT", "BTCUSDT")
    assert runtime.fetch_catalog_once("core_release_1_46_binance_spot") is None
    with pytest.raises(ValueError, match="CATALOG_AUTHORITY_DOMAIN"):
        CatalogRuntimeAcceptanceAuthority(membership, runtime._receipts)


def test_wrong_alias_and_wrong_fingerprint_domain_are_not_canonical(authority) -> None:
    runtime, _membership, _values, _carrier = authority
    snapshot = runtime.fetch_catalog_once("core_release_1_46_binance_spot")
    assert snapshot is not None
    alias = snapshot.to_mapping()
    alias["members"] = alias.pop("member_source_product_metadata_versions")
    assert not validate_accepted_source_catalog_snapshot(alias)
    wrong = snapshot.to_mapping()
    wrong["content_fingerprint"] = _fingerprint(
        "cryptohunter.m0.5.accepted-source-catalog-snapshot.v1",
        SOURCE_FINGERPRINT_FIELDS, wrong,
    )
    assert not validate_accepted_source_catalog_snapshot(wrong)


@pytest.mark.parametrize(("column", "value"), [
    ("upstream_snapshot_or_retrieval_id", "shadow-R2"),
    ("normalized_content_sha256", "f" * 64),
    ("accepted_source_catalog_snapshot_id", "ascat_shadow"),
    ("source_exchange_id", "shadow_exchange"),
    ("market_type", "MARGIN"),
])
def test_snapshot_shadow_columns_are_exact_bound(authority, column, value) -> None:
    runtime, membership, _values, carrier = authority
    assert runtime.fetch_catalog_once("core_release_1_46_binance_spot") is not None
    with _sqlite(carrier) as db:
        db.execute(f"UPDATE accepted_catalog_snapshots SET {column}=? WHERE snapshot_sequence=1", (value,))
    with pytest.raises(ValueError, match="CORRUPT_CATALOG_AUTHORITY"):
        runtime.snapshots()
    with pytest.raises(ValueError, match="CORRUPT_CATALOG_AUTHORITY"):
        CatalogRuntimeAcceptanceAuthority(membership, runtime._receipts)


def _rewrite_snapshot_row(runtime, db, sequence: int, mutate) -> dict[str, object]:
    canonical, previous = db.execute(
        "SELECT canonical_record,previous_digest FROM accepted_catalog_snapshots WHERE snapshot_sequence=?",
        (sequence,),
    ).fetchone()
    record = json.loads(canonical)
    mutate(record)
    record["content_fingerprint"] = _fingerprint(
        "cryptohunter.m0.5.accepted_source_catalog_snapshot.v2",
        SOURCE_FINGERPRINT_FIELDS, record,
    )
    canonical = _canonical(record)
    digest = runtime._snapshot_digest(sequence, canonical, previous)
    db.execute(
        "UPDATE accepted_catalog_snapshots SET canonical_record=?,record_digest=? WHERE snapshot_sequence=?",
        (canonical, digest, sequence),
    )
    db.execute("UPDATE catalog_authority_head SET committed_digest=?", (digest,))
    return record


def test_runtime_completeness_product_count_is_replayed(authority) -> None:
    runtime, _membership, _values, carrier = authority
    assert runtime.fetch_catalog_once("core_release_1_46_binance_spot") is not None
    with _sqlite(carrier) as db:
        _rewrite_snapshot_row(
            runtime, db, 1,
            lambda record: record["completeness_evidence"].__setitem__("product_count", 2),
        )
    with pytest.raises(ValueError, match="CORRUPT_CATALOG_AUTHORITY"):
        runtime.snapshots()


def test_canonical_retrieval_uniqueness_is_replayed_not_delegated_to_sql(authority) -> None:
    runtime, _membership, values, carrier = authority
    first = runtime.fetch_catalog_once("core_release_1_46_binance_spot")
    assert first is not None
    values.append(_payload(server_time=int(datetime.now(timezone.utc).timestamp() * 1000) - 50))
    assert runtime.fetch_catalog_once("core_release_1_46_binance_spot") is not None
    with _sqlite(carrier) as db:
        db.executescript("""
        ALTER TABLE accepted_catalog_snapshots RENAME TO accepted_catalog_snapshots_old;
        CREATE TABLE accepted_catalog_snapshots(
          snapshot_sequence INTEGER PRIMARY KEY,
          accepted_source_catalog_snapshot_id TEXT UNIQUE NOT NULL,
          source_exchange_id TEXT NOT NULL, market_type TEXT NOT NULL,
          upstream_snapshot_or_retrieval_id TEXT NOT NULL,
          normalized_content_sha256 TEXT NOT NULL, canonical_record TEXT NOT NULL,
          previous_digest TEXT NOT NULL, record_digest TEXT UNIQUE NOT NULL);
        INSERT INTO accepted_catalog_snapshots SELECT * FROM accepted_catalog_snapshots_old;
        DROP TABLE accepted_catalog_snapshots_old;
        """)
        duplicate = _rewrite_snapshot_row(
            runtime, db, 2,
            lambda record: record.__setitem__(
                "upstream_snapshot_or_retrieval_id",
                first.upstream_snapshot_or_retrieval_id,
            ),
        )
        db.execute(
            "UPDATE accepted_catalog_snapshots SET upstream_snapshot_or_retrieval_id=? WHERE snapshot_sequence=2",
            (duplicate["upstream_snapshot_or_retrieval_id"],),
        )
    with pytest.raises(ValueError, match="CORRUPT_CATALOG_AUTHORITY"):
        runtime.snapshots()


def test_coherent_metadata_rewrite_is_caught_by_snapshot_content_commitment(authority) -> None:
    runtime, _membership, _values, carrier = authority
    assert runtime.fetch_catalog_once("core_release_1_46_binance_spot") is not None
    with _sqlite(carrier) as db:
        sequence, introduced_id, canonical, previous = db.execute(
            "SELECT source_metadata_sequence,introduced_by_snapshot_id,canonical_record,previous_metadata_digest FROM source_metadata_versions"
        ).fetchone()
        record = json.loads(canonical)
        record["normalized_metadata"]["tick_size"] = "9"
        record["content_fingerprint"] = _digest(
            _METADATA_FP_DOMAIN, record,
            {"source_metadata_version_id", "content_fingerprint"},
        )
        canonical = _canonical(record)
        digest = runtime._metadata_digest(sequence, introduced_id, canonical, previous)
        db.execute(
            "UPDATE source_metadata_versions SET canonical_record=?,metadata_record_digest=?",
            (canonical, digest),
        )
        db.execute("UPDATE source_metadata_authority_head SET committed_digest=?", (digest,))
    with pytest.raises(ValueError, match="CORRUPT_CATALOG_AUTHORITY"):
        runtime.snapshots()


@pytest.mark.parametrize("mutation", ["extra", "bool_precision", "bad_decimal"])
def test_metadata_closed_schema_and_normalized_structure_are_replayed(authority, mutation) -> None:
    runtime, _membership, _values, carrier = authority
    assert runtime.fetch_catalog_once("core_release_1_46_binance_spot") is not None
    with _sqlite(carrier) as db:
        sequence, introduced_id, canonical, previous = db.execute(
            "SELECT source_metadata_sequence,introduced_by_snapshot_id,canonical_record,previous_metadata_digest FROM source_metadata_versions"
        ).fetchone()
        record = json.loads(canonical)
        if mutation == "extra":
            record["normalized_metadata"]["trusted"] = True
        elif mutation == "bool_precision":
            record["normalized_metadata"]["price_precision"] = True
        else:
            record["normalized_metadata"]["tick_size"] = "banana"
        record["content_fingerprint"] = _digest(
            _METADATA_FP_DOMAIN, record,
            {"source_metadata_version_id", "content_fingerprint"},
        )
        canonical = _canonical(record)
        digest = runtime._metadata_digest(sequence, introduced_id, canonical, previous)
        db.execute(
            "UPDATE source_metadata_versions SET canonical_record=?,metadata_record_digest=?",
            (canonical, digest),
        )
        db.execute("UPDATE source_metadata_authority_head SET committed_digest=?", (digest,))
    with pytest.raises(ValueError, match="CORRUPT_CATALOG_AUTHORITY"):
        runtime.snapshots()


def test_introduced_by_shadow_is_digest_bound_even_if_later_snapshot_references_metadata(authority) -> None:
    runtime, _membership, values, carrier = authority
    first = runtime.fetch_catalog_once("core_release_1_46_binance_spot")
    assert first is not None
    values.append(_payload(server_time=int(datetime.now(timezone.utc).timestamp() * 1000) - 50))
    second = runtime.fetch_catalog_once("core_release_1_46_binance_spot")
    assert second is not None
    with _sqlite(carrier) as db:
        db.execute(
            "UPDATE source_metadata_versions SET introduced_by_snapshot_id=?",
            (second.accepted_source_catalog_snapshot_id,),
        )
    with pytest.raises(ValueError, match="CORRUPT_CATALOG_AUTHORITY"):
        runtime.snapshots()


@pytest.mark.parametrize("tables", [
    ("accepted_catalog_snapshots",),
    ("catalog_authority_head",),
    ("source_metadata_versions",),
    ("source_metadata_authority_head",),
    ("accepted_catalog_snapshots", "catalog_authority_head", "source_metadata_versions", "source_metadata_authority_head"),
])
def test_initialized_store_never_repairs_missing_required_tables(authority, tables) -> None:
    runtime, membership, _values, carrier = authority
    first = runtime.fetch_catalog_once("core_release_1_46_binance_spot")
    assert first is not None
    with _sqlite(carrier) as db:
        for table in tables:
            db.execute(f"DROP TABLE {table}")
    with pytest.raises(ValueError, match="CATALOG_AUTHORITY_STORAGE_MISSING"):
        CatalogRuntimeAcceptanceAuthority(membership, runtime._receipts)


def test_supersession_before_acceptance_denies_and_old_snapshot_stays_historical(authority, monkeypatch) -> None:
    runtime, membership, values, _carrier = authority
    old = runtime.fetch_catalog_once("core_release_1_46_binance_spot")
    assert old is not None
    assert membership.admit_release_grant("core_release_1_45_binance_spot_generation_2") is not None
    assert membership.admit_release_event("core_release_1_45_supersede_binance_spot") is not None
    values.append(_payload(server_time=int(datetime.now(timezone.utc).timestamp() * 1000) - 50))
    assert runtime.fetch_catalog_once("core_release_1_46_binance_spot") is None
    assert validate_accepted_source_snapshot_membership_evidence(old.to_mapping(), membership)


def test_acceptance_writer_lock_commits_before_concurrent_supersession(authority, monkeypatch) -> None:
    runtime, membership, _values, _carrier = authority
    assert membership.admit_release_grant("core_release_1_45_binance_spot_generation_2") is not None
    locked = Event()
    release = Event()
    original = runtime._accept_locked

    def fenced(_self, db, producer, observation):
        locked.set()
        assert release.wait(5)
        return original(db, producer, observation)

    monkeypatch.setattr(CatalogRuntimeAcceptanceAuthority, "_accept_locked", fenced)
    accepted: list[object] = []
    terminal: list[object] = []
    accept_thread = Thread(target=lambda: accepted.append(runtime.fetch_catalog_once("core_release_1_46_binance_spot")))
    accept_thread.start()
    assert locked.wait(5)
    supersede_thread = Thread(target=lambda: terminal.append(membership.admit_release_event("core_release_1_45_supersede_binance_spot")))
    supersede_thread.start()
    release.set()
    accept_thread.join(5); supersede_thread.join(5)
    assert accepted[0] is not None
    assert terminal[0] is not None
    assert validate_accepted_source_snapshot_membership_evidence(accepted[0].to_mapping(), membership)


def test_retrieval_identity_cannot_fork_across_generation(authority, monkeypatch) -> None:
    runtime, membership, values, _carrier = authority
    old = runtime.fetch_catalog_once("core_release_1_46_binance_spot")
    assert old is not None
    assert membership.admit_release_grant("core_release_1_45_binance_spot_generation_2") is not None
    assert membership.admit_release_event("core_release_1_45_supersede_binance_spot") is not None
    monkeypatch.setattr(_BinanceSpotCatalogProducer, "generation", 2)
    same_retrieval = values[-1]["serverTime"]
    values.append(_payload(server_time=same_retrieval, tick="0.1"))
    assert runtime.fetch_catalog_once("core_release_1_46_binance_spot") is None
    values[-1] = _payload(server_time=same_retrieval)
    assert runtime.fetch_catalog_once("core_release_1_46_binance_spot") == old
