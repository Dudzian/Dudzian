from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from bot_core.persistence.state_store import SQLiteStateStore, StateStoreError
from tests.persistence.test_state_store_records import _account, _commit, _metadata, _runtime


def _two_generations(path: Path) -> SQLiteStateStore:
    store = SQLiteStateStore(path)
    _commit(store, _metadata(), current=(_account(),), history=(_runtime(),))
    _commit(
        store,
        _metadata(2),
        history=(_runtime("run_01890f4c-7b9a-7cc1-8a2b-123456789abd"),),
        expected=1,
    )
    return store


def test_verified_snapshot_contains_one_coherent_complete_generation(tmp_path: Path) -> None:
    with _two_generations(tmp_path / "state.sqlite3") as store:
        snapshot = store.read_verified_snapshot()
        assert snapshot is not None
        assert snapshot.metadata.protected_freshness_generation == 2
        assert len(snapshot.current_records) == 1
        assert len(snapshot.immutable_history) == 2
        assert [item.target_generation for item in snapshot.transaction_descriptors] == [1, 2]


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE state_store_metadata SET state_fingerprint_sha256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'",
        "UPDATE state_store_metadata SET history_tail_fingerprint_sha256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'",
        "UPDATE state_store_metadata SET transaction_fingerprint_sha256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'",
        "DELETE FROM state_store_transaction_descriptors WHERE target_generation=1",
        "UPDATE state_store_transaction_descriptors SET descriptor_json='{}' WHERE target_generation=1",
    ],
)
def test_raw_sqlite_tamper_fails_verified_restart(tmp_path: Path, statement: str) -> None:
    path = tmp_path / "state.sqlite3"
    with _two_generations(path) as store:
        store._connection.execute(statement)
    with SQLiteStateStore(path) as reopened:
        with pytest.raises(StateStoreError):
            reopened.read_verified_snapshot()


def test_previous_chain_is_required_before_next_commit(tmp_path: Path) -> None:
    with _two_generations(tmp_path / "state.sqlite3") as store:
        store._connection.execute(
            "DELETE FROM state_store_transaction_descriptors WHERE target_generation=1"
        )
        with pytest.raises(StateStoreError):
            _commit(store, _metadata(3), expected=2)
        assert store.read_metadata() is not None
        assert store.read_metadata().protected_freshness_generation == 2


def test_wrong_caller_hashes_fail_without_durable_content(tmp_path: Path) -> None:
    for field in (
        "history_tail_fingerprint_sha256",
        "state_fingerprint_sha256",
        "transaction_fingerprint_sha256",
    ):
        path = tmp_path / f"{field}.sqlite3"
        with SQLiteStateStore(path) as store:
            prepared = store.derive_prepared_metadata(
                _metadata(), current_records=(_account(),), expected_current_generation=None
            )
            candidate = replace(prepared, **{field: "a" * 64})
            with pytest.raises(StateStoreError):
                store.commit_prepared_state(
                    candidate,
                    current_records=(_account(),),
                    immutable_history=(),
                    expected_current_generation=None,
                )
            assert store.read_snapshot() is None


def _four_generations(path: Path) -> SQLiteStateStore:
    store = _two_generations(path)
    _commit(store, _metadata(3), expected=2)
    _commit(store, _metadata(4), expected=3)
    return store


def _rewrite_descriptor(
    store: SQLiteStateStore, generation: int, **changes: object
) -> dict[str, object]:
    from bot_core.persistence.fingerprints import canonical_json, transaction_fingerprint_sha256

    encoded = store._connection.execute(
        "SELECT descriptor_json FROM state_store_transaction_descriptors WHERE target_generation=?",
        (generation,),
    ).fetchone()[0]
    import json

    mapping = json.loads(encoded)
    mapping.update(changes)
    projection = {
        key: value for key, value in mapping.items() if key != "transaction_fingerprint_sha256"
    }
    mapping["transaction_fingerprint_sha256"] = transaction_fingerprint_sha256(projection)
    store._connection.execute(
        "UPDATE state_store_transaction_descriptors SET descriptor_json=? WHERE target_generation=?",
        (canonical_json(mapping), generation),
    )
    return mapping


def test_shuffled_descriptor_reads_verify_and_do_not_block_next_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.sqlite3"
    with _four_generations(path) as store:
        original = SQLiteStateStore._read_descriptors
        calls = 0

        def shuffled(self: SQLiteStateStore):  # type: ignore[no-untyped-def]
            nonlocal calls
            descriptors = original(self)
            calls += 1
            order = (3, 0, 2, 1) if calls % 2 else (1, 3, 0, 2)
            if len(descriptors) == 4:
                return tuple(descriptors[index] for index in order)
            return tuple(reversed(descriptors))

        monkeypatch.setattr(SQLiteStateStore, "_read_descriptors", shuffled)
        snapshot = store.read_verified_snapshot()
        assert snapshot is not None
        assert sorted(item.target_generation for item in snapshot.transaction_descriptors) == [
            1,
            2,
            3,
            4,
        ]
        _commit(store, _metadata(5), expected=4)
        assert store.read_verified_snapshot().metadata.protected_freshness_generation == 5


def test_real_two_connection_wal_snapshot_is_generation_pinned(tmp_path: Path) -> None:
    from bot_core.persistence.state_store import StateStoreSnapshot

    path = tmp_path / "state.sqlite3"
    with SQLiteStateStore(path) as reader, SQLiteStateStore(path) as writer:
        _commit(writer, _metadata(), current=(_account(),), history=(_runtime(),))
        reader._connection.execute("BEGIN")
        pinned_metadata = reader.read_metadata()  # first SELECT pins the WAL snapshot at G1
        assert pinned_metadata is not None
        _commit(
            writer,
            _metadata(2),
            history=(_runtime("run_01890f4c-7b9a-7cc1-8a2b-123456789abd"),),
            expected=1,
        )
        pinned = StateStoreSnapshot(
            pinned_metadata,
            reader._read_records(
                "state_store_current_records", "CryptoHunterAccount current record", pinned_metadata
            ),
            reader._read_records(
                "state_store_immutable_history",
                "RuntimeSession canonical identity/history",
                pinned_metadata,
            ),
            reader._read_descriptors(),
        )
        reader.verify_snapshot(pinned)
        assert pinned.metadata.protected_freshness_generation == 1
        assert len(pinned.immutable_history) == len(pinned.transaction_descriptors) == 1
        reader._connection.execute("COMMIT")
        current = reader.read_verified_snapshot()
        assert current is not None
        assert current.metadata.protected_freshness_generation == 2
        assert len(current.immutable_history) == len(current.transaction_descriptors) == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("account_id", "acct_01890f4c-7b9a-7cc1-8a2b-123456789abd", "account scope"),
        ("device_installation_id", "dev_01890f4c-7b9a-7cc1-8a2b-123456789abd", "device scope"),
        ("pre_state_fingerprint_sha256", "a" * 64, "fingerprint edge"),
        ("pre_history_tail_fingerprint_sha256", "b" * 64, "fingerprint edge"),
    ],
)
def test_self_consistent_historical_tamper_fails_full_chain(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    from bot_core.persistence.transaction_descriptor import StateStoreTransactionDescriptor

    with _four_generations(tmp_path / "state.sqlite3") as store:
        mapping = _rewrite_descriptor(store, 2, **{field: value})
        descriptor = StateStoreTransactionDescriptor.from_mapping(mapping)
        assert descriptor.has_valid_transaction_fingerprint()
        with pytest.raises(StateStoreError, match=message):
            store.read_verified_snapshot()


def test_recomputed_hash_does_not_hide_bad_historical_environment(tmp_path: Path) -> None:
    from bot_core.persistence.fingerprints import transaction_fingerprint_sha256

    with _four_generations(tmp_path / "state.sqlite3") as store:
        mapping = _rewrite_descriptor(store, 2, environment="HACKED")
        assert len(mapping["transaction_fingerprint_sha256"]) == 64
        assert mapping["transaction_fingerprint_sha256"] == transaction_fingerprint_sha256(
            {
                key: value
                for key, value in mapping.items()
                if key != "transaction_fingerprint_sha256"
            }
        )
        with pytest.raises(StateStoreError, match="malformed"):
            store.read_verified_snapshot()


def test_recomputed_outer_hash_does_not_hide_invalid_nested_record(tmp_path: Path) -> None:
    with _four_generations(tmp_path / "state.sqlite3") as store:
        descriptor = store.read_transaction_descriptors()[0].to_mapping()
        nested = descriptor["current_record_mutations"][0]
        nested["payload_fingerprint_sha256"] = "a" * 64
        mapping = _rewrite_descriptor(
            store, 1, current_record_mutations=descriptor["current_record_mutations"]
        )
        assert len(mapping["transaction_fingerprint_sha256"]) == 64
        with pytest.raises(StateStoreError, match="malformed"):
            store.read_verified_snapshot()


def test_descriptor_carrier_and_collection_tamper_matrix(tmp_path: Path) -> None:
    from bot_core.persistence.fingerprints import canonical_json, transaction_fingerprint_sha256

    statements = (
        "UPDATE state_store_transaction_descriptors SET target_generation=9 WHERE target_generation=1",
        "UPDATE state_store_transaction_descriptors SET state_store_identity_fingerprint_sha256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' WHERE target_generation=1",
    )
    for index, statement in enumerate(statements):
        with _two_generations(tmp_path / f"carrier-{index}.sqlite3") as store:
            store._connection.execute(statement)
            with pytest.raises(StateStoreError, match="carrier mismatch"):
                store.read_verified_snapshot()

    with _two_generations(tmp_path / "hash.sqlite3") as store:
        import json

        mapping = json.loads(
            store._connection.execute(
                "SELECT descriptor_json FROM state_store_transaction_descriptors WHERE target_generation=1"
            ).fetchone()[0]
        )
        mapping["transaction_fingerprint_sha256"] = "a" * 64
        store._connection.execute(
            "UPDATE state_store_transaction_descriptors SET descriptor_json=? WHERE target_generation=1",
            (canonical_json(mapping),),
        )
        with pytest.raises(StateStoreError, match="transaction fingerprint"):
            store.read_verified_snapshot()

    for name, identity in (("future", "1" * 64), ("unrelated", "a" * 64)):
        with _two_generations(tmp_path / f"{name}.sqlite3") as store:
            base = store.read_transaction_descriptors()[-1].to_mapping()
            base.update(
                state_store_identity_fingerprint_sha256=identity,
                expected_current_generation=2,
                target_generation=3,
                pre_state_fingerprint_sha256=base["post_state_fingerprint_sha256"],
                pre_history_tail_fingerprint_sha256=base["post_history_tail_fingerprint_sha256"],
            )
            base["transaction_fingerprint_sha256"] = transaction_fingerprint_sha256(
                {
                    key: value
                    for key, value in base.items()
                    if key != "transaction_fingerprint_sha256"
                }
            )
            store._connection.execute(
                "INSERT INTO state_store_transaction_descriptors VALUES(?,?,?)",
                (identity, 3, canonical_json(base)),
            )
            with pytest.raises(StateStoreError, match="cardinality"):
                store.read_verified_snapshot()


def test_complete_chain_matrix_is_order_independent_and_rejects_bad_collections(
    tmp_path: Path,
) -> None:
    from bot_core.persistence.state_store import StateStoreSnapshot

    with _four_generations(tmp_path / "state.sqlite3") as store:
        snapshot = store.read_verified_snapshot()
        assert snapshot is not None
        chain = snapshot.transaction_descriptors
        for order in ((0, 1, 2, 3), (3, 1, 0, 2), (2, 3, 1, 0)):
            store.verify_snapshot(
                replace(snapshot, transaction_descriptors=tuple(chain[i] for i in order))
            )
        bad = ((0, 2, 3), (1, 2, 3), (0, 1, 3), (0, 1, 2), (0, 1, 2, 3, 3), (0, 1, 1, 2, 3))
        for order in bad:
            candidate = StateStoreSnapshot(
                snapshot.metadata,
                snapshot.current_records,
                snapshot.immutable_history,
                tuple(chain[i] for i in order),
            )
            with pytest.raises(StateStoreError):
                store.verify_snapshot(candidate)
