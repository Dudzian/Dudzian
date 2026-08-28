from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from bot_core.persistence.durable_observation import (
    DurableStateObservation,
    observe_verified_durable_state,
)
from bot_core.persistence.state_store import SQLiteStateStore, StateStoreError
from tests.persistence.test_state_store_records import _account, _commit, _metadata, _runtime


OBSERVATION_FIELDS = {
    "account_id",
    "device_installation_id",
    "state_store_schema_version",
    "state_store_identity_fingerprint_sha256",
    "environment",
    "protected_freshness_generation",
    "state_fingerprint_sha256",
    "transaction_fingerprint_sha256",
    "history_tail_fingerprint_sha256",
    "durable_confirmed",
    "authoritative_history_integrity",
    "current_commit",
}


def _generation_one(store: SQLiteStateStore) -> None:
    _commit(store, _metadata(), current=(_account(),), history=(_runtime(),))


def test_valid_generation_is_an_exact_verified_metadata_observation(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        _generation_one(store)
        snapshot = store.read_verified_snapshot()
        observation = observe_verified_durable_state(store)

    assert snapshot is not None and observation is not None
    assert {field.name for field in fields(observation)} == OBSERVATION_FIELDS
    for field in fields(snapshot.metadata):
        assert getattr(observation, field.name) == getattr(snapshot.metadata, field.name)
    assert observation.durable_confirmed is True
    assert observation.authoritative_history_integrity is True
    assert observation.current_commit is True


def test_empty_store_has_no_observation(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "empty.sqlite3") as store:
        assert observe_verified_durable_state(store) is None


@pytest.mark.parametrize(
    "field",
    [
        "account_id",
        "protected_freshness_generation",
        "state_fingerprint_sha256",
        "transaction_fingerprint_sha256",
        "durable_confirmed",
        "authoritative_history_integrity",
        "current_commit",
    ],
)
def test_observation_fields_are_immutable(tmp_path: Path, field: str) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        _generation_one(store)
        observation = observe_verified_durable_state(store)
    assert observation is not None
    with pytest.raises((FrozenInstanceError, AttributeError)):
        setattr(observation, field, "caller mutation")


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE state_store_metadata SET state_fingerprint_sha256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'",
        "UPDATE state_store_metadata SET transaction_fingerprint_sha256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'",
        "UPDATE state_store_metadata SET history_tail_fingerprint_sha256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'",
        "DELETE FROM state_store_transaction_descriptors",
        "UPDATE state_store_current_records SET record_json='{}'",
        "UPDATE state_store_immutable_history SET record_json='{}'",
    ],
)
def test_s2c_corruption_fails_closed_without_observation(tmp_path: Path, statement: str) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        _generation_one(store)
        store._connection.execute(statement)
        with pytest.raises(StateStoreError):
            observe_verified_durable_state(store)


def test_observation_is_rebuilt_after_store_reopen(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    with SQLiteStateStore(path) as store:
        _generation_one(store)
        first = observe_verified_durable_state(store)
    with SQLiteStateStore(path) as reopened:
        second = observe_verified_durable_state(reopened)

    assert first is not None and second is not None
    assert first is not second
    assert first == second


def test_new_generation_does_not_mutate_old_snapshot_bound_observation(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        _generation_one(store)
        first = observe_verified_durable_state(store)
        _commit(
            store,
            _metadata(2),
            history=(_runtime("run_01890f4c-7b9a-7cc1-8a2b-123456789abd"),),
            expected=1,
        )
        second = observe_verified_durable_state(store)

    assert first is not None and second is not None
    assert first.protected_freshness_generation == 1
    assert second.protected_freshness_generation == 2
    assert first.state_fingerprint_sha256 != second.state_fingerprint_sha256
    assert first.transaction_fingerprint_sha256 != second.transaction_fingerprint_sha256


def test_observer_acquires_exactly_one_verified_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        _generation_one(store)
        original = SQLiteStateStore.read_verified_snapshot
        calls = 0

        def counted(instance: SQLiteStateStore):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            return original(instance)

        monkeypatch.setattr(SQLiteStateStore, "read_verified_snapshot", counted)
        assert observe_verified_durable_state(store) is not None
        assert calls == 1


def test_observation_is_read_only_and_does_not_change_s2c_state(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        _generation_one(store)
        before = store.read_verified_snapshot()
        for _ in range(3):
            assert observe_verified_durable_state(store) is not None
        after = store.read_verified_snapshot()
        tables = {
            row[0]
            for row in store._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

    assert before == after
    assert tables == {
        "state_store_metadata",
        "state_store_current_records",
        "state_store_immutable_history",
        "state_store_transaction_descriptors",
    }


def test_no_caller_mapping_or_authority_api_exists() -> None:
    forbidden = {
        "from_mapping",
        "accept",
        "authorize",
        "grant",
        "promote",
        "set_current",
        "designate_current",
        "enable_live",
    }
    assert forbidden.isdisjoint(dir(DurableStateObservation))
    with pytest.raises((AttributeError, TypeError)):
        observe_verified_durable_state(  # type: ignore[arg-type]
            {
                "durable_confirmed": True,
                "authoritative_history_integrity": True,
                "current_commit": True,
            }
        )
