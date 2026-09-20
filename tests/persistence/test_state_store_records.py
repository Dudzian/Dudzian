from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pytest

from bot_core.persistence.records import PersistenceRecord
from bot_core.persistence.record_registry import STATE_STORE_SCOPE_BINDINGS
from bot_core.persistence.state_store import (
    SQLiteStateStore,
    StateStoreError,
    StateStoreMetadata,
    _validate_record_store_scope,
)
from tests.architecture import (
    test_cryptohunter_persistence_versioning_migrations_backup_and_recovery as frozen_oracle,
)

ACCOUNT_ID = "acct_01890f4c-7b9a-7cc1-8a2b-123456789abc"
DEVICE_ID = "dev_01890f4c-7b9a-7cc1-8a2b-123456789abc"
OTHER_ACCOUNT_ID = "acct_01890f4c-7b9a-7cc1-8a2b-123456789abd"
OTHER_DEVICE_ID = "dev_01890f4c-7b9a-7cc1-8a2b-123456789abd"
SESSION_1 = "run_01890f4c-7b9a-7cc1-8a2b-123456789abc"
SESSION_2 = "run_01890f4c-7b9a-7cc1-8a2b-123456789abd"
SESSION_3 = "run_01890f4c-7b9a-7cc1-8a2b-123456789abe"

ACCOUNT_SCOPED = tuple(
    name for name, binding in STATE_STORE_SCOPE_BINDINGS.items() if binding["account_paths"]
)
DEVICE_SCOPED = tuple(
    name for name, binding in STATE_STORE_SCOPE_BINDINGS.items() if binding["device_paths"]
)


def _fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return sha256(encoded).hexdigest()


def _oracle_scope_metadata(**changes: object) -> StateStoreMetadata:
    account_id, device_id, _ = frozen_oracle.SCOPE
    values = {"account_id": account_id, "device_installation_id": device_id, **changes}
    return _metadata(**values)


@pytest.mark.parametrize("name", ACCOUNT_SCOPED)
def test_explicit_account_scope_paths_accept_exact_and_reject_wrong_store(name: str) -> None:
    record = PersistenceRecord.from_mapping(frozen_oracle._persistence_record(name))
    _validate_record_store_scope(record, _oracle_scope_metadata())
    with pytest.raises(StateStoreError, match="scope"):
        _validate_record_store_scope(
            record,
            _oracle_scope_metadata(account_id="acct_01890f3a-2b4c-7abc-8def-0123456789ac"),
        )


@pytest.mark.parametrize("name", DEVICE_SCOPED)
def test_explicit_device_scope_paths_accept_exact_and_reject_wrong_store(name: str) -> None:
    record = PersistenceRecord.from_mapping(frozen_oracle._persistence_record(name))
    _validate_record_store_scope(record, _oracle_scope_metadata())
    with pytest.raises(StateStoreError, match="scope"):
        _validate_record_store_scope(
            record,
            _oracle_scope_metadata(
                device_installation_id="dev_01890f3a-2b4c-7abc-8def-0123456789ac"
            ),
        )


def _metadata(generation: int = 1, **changes: Any) -> StateStoreMetadata:
    values: dict[str, Any] = {
        "account_id": ACCOUNT_ID,
        "device_installation_id": DEVICE_ID,
        "state_store_schema_version": 1,
        "state_store_identity_fingerprint_sha256": "1" * 64,
        "environment": "PAPER",
        "protected_freshness_generation": generation,
        "state_fingerprint_sha256": "2" * 64,
        "transaction_fingerprint_sha256": "3" * 64,
        "history_tail_fingerprint_sha256": "4" * 64,
    }
    values.update(changes)
    return StateStoreMetadata.from_mapping(values)


def _account(
    account_id: str = ACCOUNT_ID, *, payload_fingerprint: str | None = None
) -> PersistenceRecord:
    payload = {
        "entity_kind": "CryptoHunterAccount",
        "entity_id": account_id,
        "parent_scope_bindings": {},
    }
    return PersistenceRecord(
        representation_name="CryptoHunterAccount current record",
        representation_category="M011_ENTITY_IDENTITY_PROJECTION",
        semantic_owner_milestone="M0.2",
        semantic_artifact="canonical_domain_vocabulary.json",
        semantic_json_pointer="/entity_kinds",
        semantic_contract_fingerprint_sha256=(
            "bb92436a67abe42975d763b06717962009d3f4bc8a6066b227f34c8e7d178e9b"
        ),
        record_key=account_id,
        payload=payload,
        payload_fingerprint_sha256=payload_fingerprint or _fingerprint(payload),
    )


def _runtime(
    session_id: str = SESSION_1,
    *,
    device_id: str = DEVICE_ID,
    **payload_changes: object,
) -> PersistenceRecord:
    upstream = {"runtime_session_id": session_id, "device_installation_id": device_id}
    payload: dict[str, object] = {
        "fact_kind": "RuntimeSession",
        "upstream_payload": upstream,
        "upstream_payload_fingerprint_sha256": _fingerprint(upstream),
    }
    payload.update(payload_changes)
    return PersistenceRecord(
        representation_name="RuntimeSession canonical identity/history",
        representation_category="M011_IMMUTABLE_HISTORY_WRAPPER",
        semantic_owner_milestone="M0.2",
        semantic_artifact="canonical_domain_vocabulary.json",
        semantic_json_pointer="/entity_kinds",
        semantic_contract_fingerprint_sha256=(
            "bb92436a67abe42975d763b06717962009d3f4bc8a6066b227f34c8e7d178e9b"
        ),
        record_key=session_id,
        payload=payload,
        payload_fingerprint_sha256=_fingerprint(payload),
    )


def _commit(
    store: SQLiteStateStore,
    metadata: StateStoreMetadata,
    *,
    current: tuple[PersistenceRecord, ...] = (),
    history: tuple[PersistenceRecord, ...] = (),
    expected: int | None = None,
) -> StateStoreMetadata:
    prepared = store.derive_prepared_metadata(
        metadata,
        current_records=current,
        immutable_history=history,
        expected_current_generation=expected,
    )
    store.commit_prepared_state(
        prepared,
        current_records=current,
        immutable_history=history,
        expected_current_generation=expected,
    )
    return prepared


def _encode(record: PersistenceRecord) -> str:
    return json.dumps(
        record.to_mapping(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _replace_raw_record(store: SQLiteStateStore, table: str, record: PersistenceRecord) -> None:
    store._connection.execute(f"DELETE FROM {table}")
    store._connection.execute(
        f"""
        INSERT INTO {table} (record_key, representation_name, record_json)
        VALUES (?, ?, ?)
        """,
        (record.record_key, record.representation_name, _encode(record)),
    )


def test_fresh_atomic_commit_survives_reopen_and_preserves_prepared_fingerprints(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.sqlite3"
    metadata, account, runtime = _metadata(), _account(), _runtime()
    with SQLiteStateStore(path) as store:
        metadata = _commit(store, metadata, current=(account,), history=(runtime,))
        assert store.read_metadata() == metadata
        assert store.read_current_records() == (account,)
        assert store.read_immutable_history() == (runtime,)
    with SQLiteStateStore(path) as store:
        assert store.read_metadata() == metadata
        assert store.read_current_records() == (account,)
        assert store.read_immutable_history() == (runtime,)


@pytest.mark.parametrize(
    ("current", "history"),
    [
        ((_account(OTHER_ACCOUNT_ID),), ()),
        ((), (_runtime(device_id=OTHER_DEVICE_ID),)),
    ],
)
def test_self_consistent_foreign_incoming_scope_fails_before_fresh_commit(
    tmp_path: Path,
    current: tuple[PersistenceRecord, ...],
    history: tuple[PersistenceRecord, ...],
) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        with pytest.raises(StateStoreError, match="outside StateStore scope"):
            _commit(store, _metadata(), current=current, history=history)
        assert store.read_metadata() is None
        assert store.read_current_records() == ()
        assert store.read_immutable_history() == ()


@pytest.mark.parametrize(
    ("table", "foreign", "reader"),
    [
        (
            "state_store_current_records",
            _account(OTHER_ACCOUNT_ID),
            "read_current_records",
        ),
        (
            "state_store_immutable_history",
            _runtime(device_id=OTHER_DEVICE_ID),
            "read_immutable_history",
        ),
    ],
)
def test_self_consistent_persisted_foreign_scope_fails_public_read(
    tmp_path: Path,
    table: str,
    foreign: PersistenceRecord,
    reader: str,
) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        _commit(store, _metadata(), current=(_account(),), history=(_runtime(),))
        _replace_raw_record(store, table, foreign)
        with pytest.raises(StateStoreError, match="outside StateStore scope"):
            getattr(store, reader)()


@pytest.mark.parametrize(
    ("table", "record", "reader"),
    [
        ("state_store_current_records", _account(), "read_current_records"),
        ("state_store_immutable_history", _runtime(), "read_immutable_history"),
    ],
)
def test_valid_rows_without_metadata_fail_closed(
    tmp_path: Path,
    table: str,
    record: PersistenceRecord,
    reader: str,
) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        _replace_raw_record(store, table, record)
        with pytest.raises(StateStoreError, match="without StateStoreMetadata"):
            getattr(store, reader)()


def test_foreign_scope_corruption_blocks_next_commit_and_is_not_repaired(
    tmp_path: Path,
) -> None:
    foreign = _account(OTHER_ACCOUNT_ID)
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        _commit(store, _metadata(), current=(_account(),), history=(_runtime(),))
        _replace_raw_record(store, "state_store_current_records", foreign)
        with pytest.raises(StateStoreError, match="outside StateStore scope"):
            _commit(store, _metadata(2), history=(_runtime(SESSION_2),), expected=1)
        assert store.read_metadata() is not None
        assert store.read_metadata().protected_freshness_generation == 1
        assert store._connection.execute(
            "SELECT record_key FROM state_store_current_records"
        ).fetchone() == (OTHER_ACCOUNT_ID,)
        assert store._connection.execute(
            "SELECT COUNT(*) FROM state_store_immutable_history"
        ).fetchone() == (1,)


def test_metadata_only_commit_cannot_bypass_foreign_scope_validation(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        _commit(store, _metadata(), current=(_account(),), history=(_runtime(),))
        _commit(store, _metadata(2), expected=1)
        assert store.read_metadata() is not None
        assert store.read_metadata().protected_freshness_generation == 2
        _replace_raw_record(
            store, "state_store_immutable_history", _runtime(device_id=OTHER_DEVICE_ID)
        )
        with pytest.raises(StateStoreError, match="outside StateStore scope"):
            store.commit_prepared_metadata(_metadata(3), expected_current_generation=2)
        assert store.read_metadata() is not None
        assert store.read_metadata().protected_freshness_generation == 2


def test_local_scope_coherence_does_not_add_authority_api(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        _commit(store, _metadata(), current=(_account(),), history=(_runtime(),))
        for authority_api in ("accept", "authorize", "set_current", "grant", "promote"):
            assert not hasattr(store, authority_api)


def test_generation_update_replaces_current_and_preserves_old_history(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        first, second = _runtime(), _runtime(SESSION_2)
        _commit(store, _metadata(), current=(_account(),), history=(first,))
        _commit(store, _metadata(2), current=(_account(),), history=(second,), expected=1)
        assert store.read_metadata() is not None
        assert store.read_metadata().protected_freshness_generation == 2
        assert store.read_current_records() == (_account(),)
        assert store.read_immutable_history() == (first, second)


def test_reads_are_ordered_by_record_key_not_row_order(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        second, first = _runtime(SESSION_2), _runtime(SESSION_1)
        _commit(store, _metadata(), history=(second, first))
        assert store.read_immutable_history() == (first, second)


def test_two_writers_reject_stale_record_commit_without_partial_rows(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    with SQLiteStateStore(path) as first, SQLiteStateStore(path) as second:
        _commit(first, _metadata(), current=(_account(),), history=(_runtime(),))
        assert second.read_metadata() is not None
        assert second.read_metadata().protected_freshness_generation == 1
        _commit(first, _metadata(2), history=(_runtime(SESSION_2),), expected=1)
        with pytest.raises(StateStoreError):
            _commit(second, _metadata(2), history=(_runtime(SESSION_3),), expected=1)
    with SQLiteStateStore(path) as store:
        assert store.read_metadata() is not None
        assert store.read_metadata().protected_freshness_generation == 2
        assert store.read_immutable_history() == (_runtime(), _runtime(SESSION_2))
        assert [item.target_generation for item in store.read_transaction_descriptors()] == [1, 2]


@pytest.mark.parametrize(
    ("current", "history"),
    [
        ((_account(payload_fingerprint="f" * 64),), ()),
        ((), (_runtime(upstream_payload_fingerprint_sha256="f" * 64),)),
        ((_runtime(),), ()),
        ((), (_account(),)),
        ((_account(), _account()), ()),
        ((), (_runtime(), _runtime())),
    ],
)
def test_invalid_or_ambiguous_input_fails_before_any_mutation(
    tmp_path: Path,
    current: tuple[PersistenceRecord, ...],
    history: tuple[PersistenceRecord, ...],
) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        with pytest.raises(StateStoreError):
            _commit(store, _metadata(), current=current, history=history)
        assert store.read_metadata() is None
        assert store.read_current_records() == ()
        assert store.read_immutable_history() == ()


def test_unsupported_workspace_input_fails_closed(tmp_path: Path) -> None:
    unsupported = replace(_account(), representation_name="Workspace")
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        with pytest.raises(StateStoreError):
            _commit(store, _metadata(), current=(unsupported,))
        assert store.read_metadata() is None


def test_existing_history_duplicate_rolls_back_current_and_metadata(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        account, runtime = _account(), _runtime()
        _commit(store, _metadata(), current=(account,), history=(runtime,))
        with pytest.raises(StateStoreError):
            _commit(store, _metadata(2), current=(account,), history=(runtime,), expected=1)
        assert store.read_metadata() is not None
        assert store.read_metadata().protected_freshness_generation == 1
        assert store.read_current_records() == (account,)
        assert store.read_immutable_history() == (runtime,)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"account_id": "acct_01890f4c-7b9a-7cc1-8a2b-123456789abd"}, 1),
        ({"device_installation_id": "dev_01890f4c-7b9a-7cc1-8a2b-123456789abd"}, 1),
        ({"state_store_identity_fingerprint_sha256": "a" * 64}, 1),
        ({"state_store_schema_version": 2}, 1),
        ({}, 0),
        ({"protected_freshness_generation": 3}, 1),
    ],
)
def test_all_metadata_fences_rollback_requested_records(
    tmp_path: Path, changes: dict[str, object], expected: int
) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        account, runtime = _account(), _runtime()
        _commit(store, _metadata(), current=(account,), history=(runtime,))
        candidate = replace(_metadata(2), **changes)
        with pytest.raises(StateStoreError):
            _commit(store, candidate, history=(_runtime(SESSION_2),), expected=expected)
        assert store.read_metadata() is not None
        assert store.read_metadata().protected_freshness_generation == 1
        assert store.read_current_records() == (account,)
        assert store.read_immutable_history() == (runtime,)


def test_history_insert_failure_rolls_back_prior_current_mutation(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        _commit(store, _metadata(), current=(_account(),), history=(_runtime(),))
        before = store.read_current_records()
        store._connection.execute(
            """
            CREATE TRIGGER fail_history BEFORE INSERT ON state_store_immutable_history
            BEGIN SELECT RAISE(ABORT, 'injected history failure'); END
            """
        )
        with pytest.raises(StateStoreError):
            _commit(
                store,
                _metadata(2),
                current=(_account(),),
                history=(_runtime(SESSION_2),),
                expected=1,
            )
        assert store.read_metadata() is not None
        assert store.read_metadata().protected_freshness_generation == 1
        assert store.read_current_records() == before
        assert store.read_immutable_history() == (_runtime(),)


def test_metadata_write_failure_rolls_back_all_record_mutations(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        _commit(store, _metadata(), current=(_account(),), history=(_runtime(),))
        store._connection.execute(
            """
            CREATE TRIGGER fail_metadata BEFORE UPDATE ON state_store_metadata
            BEGIN SELECT RAISE(ABORT, 'injected metadata failure'); END
            """
        )
        with pytest.raises(StateStoreError):
            _commit(
                store,
                _metadata(2),
                current=(_account(),),
                history=(_runtime(SESSION_2),),
                expected=1,
            )
        assert store.read_metadata() is not None
        assert store.read_metadata().protected_freshness_generation == 1
        assert store.read_current_records() == (_account(),)
        assert store.read_immutable_history() == (_runtime(),)
        assert [item.target_generation for item in store.read_transaction_descriptors()] == [1]


def test_descriptor_insert_failure_rolls_back_metadata_current_and_history(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        first_metadata = _commit(store, _metadata(), current=(_account(),), history=(_runtime(),))
        first_current = store.read_current_records()
        first_history = store.read_immutable_history()
        first_descriptors = store.read_transaction_descriptors()
        store._connection.execute(
            """
            CREATE TRIGGER fail_descriptor BEFORE INSERT ON state_store_transaction_descriptors
            BEGIN SELECT RAISE(ABORT, 'injected descriptor failure'); END
            """
        )
        with pytest.raises(StateStoreError):
            _commit(
                store,
                _metadata(2),
                current=(_account(),),
                history=(_runtime(SESSION_2),),
                expected=1,
            )
        assert store.read_metadata() == first_metadata
        assert store.read_current_records() == first_current
        assert store.read_immutable_history() == first_history
        assert store.read_transaction_descriptors() == first_descriptors


def test_genesis_descriptor_insert_failure_leaves_store_empty(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        store._connection.execute(
            """
            CREATE TRIGGER fail_descriptor BEFORE INSERT ON state_store_transaction_descriptors
            BEGIN SELECT RAISE(ABORT, 'injected descriptor failure'); END
            """
        )
        with pytest.raises(StateStoreError):
            _commit(store, _metadata(), current=(_account(),), history=(_runtime(),))
        assert store.read_snapshot() is None


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE state_store_current_records SET record_json = '{'",
        "UPDATE state_store_current_records SET record_key = 'acct_01890f4c-7b9a-7cc1-8a2b-123456789abd'",
        "UPDATE state_store_current_records SET record_json = replace(record_json, 'payload_fingerprint_sha256\":\"', 'payload_fingerprint_sha256\":\"f')",
    ],
)
def test_corrupt_current_row_makes_whole_read_fail_closed(tmp_path: Path, statement: str) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        _commit(store, _metadata(), current=(_account(),))
        store._connection.execute(statement)
        with pytest.raises(StateStoreError):
            store.read_current_records()


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE state_store_immutable_history SET representation_name = 'wrong'",
        "UPDATE state_store_immutable_history SET record_json = replace(record_json, 'device_installation_id', 'wrong_field')",
        "UPDATE state_store_immutable_history SET record_json = replace(record_json, 'payload_fingerprint_sha256\":\"', 'payload_fingerprint_sha256\":\"f')",
    ],
)
def test_corrupt_history_row_makes_whole_read_fail_closed(tmp_path: Path, statement: str) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        _commit(store, _metadata(), history=(_runtime(),))
        store._connection.execute(statement)
        with pytest.raises(StateStoreError):
            store.read_immutable_history()


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("state_store_current_records", "record_json"),
        ("state_store_immutable_history", "record_json"),
    ],
)
def test_existing_corruption_blocks_next_atomic_commit(
    tmp_path: Path, table: str, column: str
) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        _commit(store, _metadata(), current=(_account(),), history=(_runtime(),))
        store._connection.execute(f"UPDATE {table} SET {column} = '{{'")
        with pytest.raises(StateStoreError):
            _commit(store, _metadata(2), history=(_runtime(SESSION_2),), expected=1)
        assert store.read_metadata() is not None
        assert store.read_metadata().protected_freshness_generation == 1
        assert (
            store._connection.execute(
                "SELECT COUNT(*) FROM state_store_immutable_history"
            ).fetchone()[0]
            == 1
        )
        assert store._connection.execute(f"SELECT {column} FROM {table}").fetchone()[0] == "{"


def test_metadata_only_api_uses_same_kernel_without_creating_records(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        _commit(store, _metadata())
        _commit(store, _metadata(2), expected=1)
        assert store.read_metadata() is not None
        assert store.read_metadata().protected_freshness_generation == 2
        assert store.read_current_records() == ()
        assert store.read_immutable_history() == ()


def test_schema_contains_exact_record_tables_and_storage_columns(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        tables = {
            row[0]
            for row in store._connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = 'table'"
            )
        }
        assert tables == {
            "state_store_metadata",
            "state_store_current_records",
            "state_store_immutable_history",
            "state_store_transaction_descriptors",
        }
        for table in ("state_store_current_records", "state_store_immutable_history"):
            assert [row[1] for row in store._connection.execute(f"PRAGMA table_info({table})")] == [
                "record_key",
                "representation_name",
                "record_json",
            ]
