from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import pytest

from bot_core.persistence.state_store import (
    SQLiteStateStore,
    StateStoreError,
    StateStoreMetadata,
)

ARCHITECTURE = Path("docs/architecture/cryptohunter_product_architecture")
ACCOUNT_ID = "acct_01890f3a-8b4c-7def-8abc-0123456789ab"
DEVICE_ID = "dev_01890f3a-8b4c-7def-9abc-0123456789ab"


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


def _canonical() -> tuple[dict[str, Any], dict[str, Any]]:
    vocabulary = json.loads((ARCHITECTURE / "canonical_domain_vocabulary.json").read_text())
    persistence = json.loads(
        (ARCHITECTURE / "persistence_versioning_migrations_backup_and_recovery.json").read_text()
    )
    return vocabulary, persistence["state_store_contract"]["schema"]


def test_representation_is_bound_to_frozen_contract() -> None:
    vocabulary, schema = _canonical()
    assert [field.name for field in fields(StateStoreMetadata)] == schema["required"]
    assert set(schema["properties"]) == set(schema["required"])
    assert schema["additionalProperties"] is False
    assert schema["properties"]["environment"]["enum"] == ["PAPER", "TESTNET", "LIVE"]
    assert {item["name"].upper() for item in vocabulary["public_trading_environments"]} == {
        "PAPER",
        "TESTNET",
        "LIVE",
    }
    for name in ("state_store_schema_version", "protected_freshness_generation"):
        assert schema["properties"][name] == {
            "type": "integer",
            "minimum": 1,
            "boolean_allowed": False,
        }
    for name in (
        "state_store_identity_fingerprint_sha256",
        "state_fingerprint_sha256",
        "transaction_fingerprint_sha256",
        "history_tail_fingerprint_sha256",
    ):
        assert schema["properties"][name]["pattern"] == "^[0-9a-f]{64}$"


def test_canonical_identifier_policy_and_entity_prefixes_are_enforced() -> None:
    vocabulary, _ = _canonical()
    assert vocabulary["identifier_policy"]["regex"] == (
        "^[a-z][a-z0-9]*_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
    )
    prefixes = {item["id_field"]: item["id_prefix"] for item in vocabulary["entity_kinds"]}
    assert prefixes["account_id"] == "acct"
    assert prefixes["device_installation_id"] == "dev"
    assert _metadata().account_id == ACCOUNT_ID


def test_mapping_round_trip_is_exact_and_deterministic() -> None:
    metadata = _metadata()
    assert StateStoreMetadata.from_mapping(metadata.to_mapping()) == metadata
    assert tuple(metadata.to_mapping()) == tuple(field.name for field in fields(StateStoreMetadata))


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("missing", None),
        ("extra", "unexpected"),
        ("state_store_schema_version", True),
        ("protected_freshness_generation", True),
        ("state_store_schema_version", 0),
        ("protected_freshness_generation", 0),
        ("account_id", "acct_invalid"),
        ("device_installation_id", ACCOUNT_ID),
        ("state_fingerprint_sha256", "a" * 63),
        ("state_fingerprint_sha256", "A" * 64),
        ("state_fingerprint_sha256", "z" * 64),
        ("environment", "SANDBOX"),
    ],
)
def test_malformed_metadata_fails_closed(mutation: str, value: object) -> None:
    mapping = _metadata().to_mapping()
    if mutation == "missing":
        mapping.pop("account_id")
    elif mutation == "extra":
        mapping["extra"] = value
    else:
        mapping[mutation] = value  # type: ignore[assignment]
    with pytest.raises((TypeError, ValueError)):
        StateStoreMetadata.from_mapping(mapping)


def test_fresh_store_is_empty_and_commit_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "state" / "store.sqlite3"
    metadata = _metadata()
    with SQLiteStateStore(path) as store:
        assert store.read_metadata() is None
        store.commit_prepared_metadata(metadata, expected_current_generation=None)
    with SQLiteStateStore(path) as reopened:
        assert reopened.read_metadata() == metadata


def test_generation_fences_and_failures_preserve_durable_state(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "store.sqlite3") as store:
        generation_1 = _metadata(1)
        generation_2 = replace(generation_1, protected_freshness_generation=2)
        store.commit_prepared_metadata(generation_1, expected_current_generation=None)
        store.commit_prepared_metadata(generation_2, expected_current_generation=1)
        assert store.read_metadata() == generation_2

        attempts = [
            (replace(generation_2, protected_freshness_generation=3), 1),
            (replace(generation_2, protected_freshness_generation=4), 2),
            (generation_1, 2),
        ]
        for prepared, expected in attempts:
            with pytest.raises(StateStoreError):
                store.commit_prepared_metadata(
                    prepared,
                    expected_current_generation=expected,
                )
            assert store.read_metadata() == generation_2


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("account_id", "acct_01890f3a-8b4c-7def-aabc-0123456789ac"),
        ("device_installation_id", "dev_01890f3a-8b4c-7def-aabc-0123456789ac"),
        ("state_store_identity_fingerprint_sha256", "a" * 64),
    ],
)
def test_identity_fence_preserves_previous_state(
    tmp_path: Path, field_name: str, value: str
) -> None:
    with SQLiteStateStore(tmp_path / "store.sqlite3") as store:
        current = _metadata()
        store.commit_prepared_metadata(current, expected_current_generation=None)
        prepared = replace(
            current,
            protected_freshness_generation=2,
            **{field_name: value},
        )
        with pytest.raises(StateStoreError):
            store.commit_prepared_metadata(prepared, expected_current_generation=1)
        assert store.read_metadata() == current


def test_schema_version_drift_requires_future_migration_engine(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "store.sqlite3") as store:
        current = _metadata()
        store.commit_prepared_metadata(current, expected_current_generation=None)
        prepared = replace(
            current,
            protected_freshness_generation=2,
            state_store_schema_version=2,
        )
        with pytest.raises(StateStoreError):
            store.commit_prepared_metadata(prepared, expected_current_generation=1)
        assert store.read_metadata() == current


def test_two_instances_reject_deterministic_stale_write(tmp_path: Path) -> None:
    path = tmp_path / "store.sqlite3"
    with SQLiteStateStore(path) as first, SQLiteStateStore(path) as second:
        generation_1 = _metadata()
        first.commit_prepared_metadata(generation_1, expected_current_generation=None)
        assert second.read_metadata() == generation_1
        generation_2 = replace(generation_1, protected_freshness_generation=2)
        first.commit_prepared_metadata(generation_2, expected_current_generation=1)
        stale_candidate = replace(
            generation_2,
            protected_freshness_generation=2,
            state_fingerprint_sha256="a" * 64,
        )
        with pytest.raises(StateStoreError):
            second.commit_prepared_metadata(stale_candidate, expected_current_generation=1)
        assert first.read_metadata() == generation_2


def test_sqlite_durability_settings_are_active(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "store.sqlite3") as store:
        connection = store._connection
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_malformed_persisted_state_fails_closed(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "store.sqlite3") as store:
        metadata = _metadata()
        store.commit_prepared_metadata(metadata, expected_current_generation=None)
        store._connection.execute(
            "UPDATE state_store_metadata SET state_fingerprint_sha256 = ?",
            ("CORRUPT",),
        )
        with pytest.raises(StateStoreError):
            store.read_metadata()
