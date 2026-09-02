from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from bot_core.persistence.backup_envelope import (
    BackupEnvelope,
    BackupEnvelopeError,
    BackupIntegrityMetadata,
    create_backup_envelope,
    validate_backup_envelope,
)
from bot_core.persistence.fingerprints import (
    canonical_json_sha256,
    history_tail_fingerprint_sha256,
    state_fingerprint_sha256,
    transaction_fingerprint_sha256,
)
from bot_core.persistence.records import PersistenceRecord, validate_persistence_record
from bot_core.persistence.secret_handoff import (
    SecretHandoffRecord,
    handoff_descriptor_carrier,
    secret_metadata_fingerprint,
    secret_operation_fingerprint,
)
from bot_core.persistence.state_store import SQLiteStateStore, StateStoreError
from tests.persistence.test_state_store_records import _account, _commit, _metadata, _runtime
from tests.architecture import (
    test_cryptohunter_persistence_versioning_migrations_backup_and_recovery as frozen_oracle,
)


def _store(path: Path, generations: int = 1) -> SQLiteStateStore:
    store = SQLiteStateStore(path)
    _commit(store, _metadata(), current=(_account(),), history=(_runtime(),))
    for generation in range(2, generations + 1):
        history = (_runtime(f"run_01890f4c-7b9a-7cc{generation}-8a2b-123456789abc"),)
        _commit(store, _metadata(generation), history=history, expected=generation - 1)
    return store


def _candidate(path: Path, generations: int = 1) -> BackupEnvelope:
    with _store(path, generations) as store:
        result = create_backup_envelope(store)
    assert result is not None
    return result


def _rehash(mapping: dict[str, object]) -> None:
    projection = {
        key: value for key, value in mapping.items() if key != "envelope_fingerprint_sha256"
    }
    mapping["envelope_fingerprint_sha256"] = canonical_json_sha256(projection)


def _rehash_descriptor(descriptor: dict[str, object]) -> None:
    projection = {
        key: value for key, value in descriptor.items() if key != "transaction_fingerprint_sha256"
    }
    descriptor["transaction_fingerprint_sha256"] = transaction_fingerprint_sha256(projection)


def _pin_record() -> PersistenceRecord:
    record = PersistenceRecord.from_mapping(
        frozen_oracle._persistence_record("PinVerifierRecord accepted revisions")
    )
    validate_persistence_record(record)
    return record


def _record_with_nested_forbidden_field(field: str) -> dict[str, object]:
    name = "Order lifecycle events/history"
    mapping = frozen_oracle._persistence_record(name)
    payload = mapping["payload"]
    upstream = payload["upstream_payload"]
    upstream["safe_payload"][field] = "forbidden"
    entry = frozen_oracle.MACHINE["backup_contract"]["representation_registry"][name]
    semantic = entry["immutable_fact_binding"]
    terminal = semantic["semantic_fingerprint_field"]
    upstream[terminal] = frozen_oracle._semantic_fingerprint(semantic, upstream)
    payload["upstream_payload_fingerprint_sha256"] = canonical_json_sha256(upstream)
    mapping["payload_fingerprint_sha256"] = canonical_json_sha256(payload)
    return mapping


def test_g1_creation_exact_projection_and_fingerprint(tmp_path: Path) -> None:
    backup = _candidate(tmp_path / "g1.sqlite3")
    mapping = backup.to_mapping()
    assert [field.name for field in fields(BackupEnvelope)] == list(BackupEnvelope._FIELDS)
    assert len(mapping) == 14
    assert backup.backup_envelope_schema_version == 1
    assert backup.local_protected_freshness_generation == 1
    assert len(backup.canonical_durable_records) == 1
    assert len(backup.immutable_recovery_history) == 1
    assert [
        item.target_generation
        for item in backup.integrity_metadata.state_store_transaction_descriptors
    ] == [1]
    projection = {
        key: value for key, value in mapping.items() if key != "envelope_fingerprint_sha256"
    }
    assert backup.envelope_fingerprint_sha256 == canonical_json_sha256(projection)


def test_g2_contains_complete_chain_and_full_state(tmp_path: Path) -> None:
    backup = _candidate(tmp_path / "g2.sqlite3", 2)
    assert backup.local_protected_freshness_generation == 2
    assert len(backup.immutable_recovery_history) == 2
    assert [
        item.target_generation
        for item in backup.integrity_metadata.state_store_transaction_descriptors
    ] == [1, 2]
    last = backup.integrity_metadata.state_store_transaction_descriptors[-1]
    assert last.post_state_fingerprint_sha256 == backup.state_fingerprint_sha256
    assert last.transaction_fingerprint_sha256 == backup.transaction_fingerprint_sha256


def test_secret_handoff_descriptor_round_trips_as_immutable_recovery_history(
    tmp_path: Path,
) -> None:
    state_store_metadata = _metadata()
    account = state_store_metadata.account_id
    device = state_store_metadata.device_installation_id
    metadata = {"cleanup": True}
    metadata_hash = secret_metadata_fingerprint(metadata)
    operation_hash = secret_operation_fingerprint(
        scope=(account, device),
        operation="ROTATE",
        old_reference="secure-ref:old",
        new_reference="secure-ref:new",
        metadata_fingerprint_sha256=metadata_hash,
    )
    carrier = handoff_descriptor_carrier(
        SecretHandoffRecord(
            "handoff-1",
            (account, device),
            "ROTATE",
            "secure-ref:old",
            "secure-ref:new",
            metadata_hash,
            operation_hash,
            metadata,
        )
    )
    with SQLiteStateStore(tmp_path / "handoff.sqlite3") as store:
        _commit(store, state_store_metadata, current=(_account(),), history=(carrier,))
        backup = create_backup_envelope(store)
    assert backup is not None and backup.immutable_recovery_history == (carrier,)
    restored = validate_backup_envelope(backup.to_mapping())
    assert restored.immutable_recovery_history == (carrier,)


def test_real_store_and_backup_preserve_pin_verifier_revision(tmp_path: Path) -> None:
    account_id, device_id, _ = frozen_oracle.SCOPE
    with SQLiteStateStore(tmp_path / "pin.sqlite3") as store:
        _commit(
            store,
            _metadata(account_id=account_id, device_installation_id=device_id),
            current=(_account(account_id),),
            history=(_pin_record(),),
        )
        backup = create_backup_envelope(store)
    assert backup is not None
    assert backup.immutable_recovery_history == (_pin_record(),)
    assert validate_backup_envelope(backup.to_mapping()) == backup


def test_empty_store_has_no_generation_zero_candidate(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "empty.sqlite3") as store:
        assert create_backup_envelope(store) is None


def test_creator_reads_exactly_one_verified_snapshot(tmp_path: Path) -> None:
    with _store(tmp_path / "one-read.sqlite3") as store:
        snapshot = store.read_verified_snapshot()
        assert snapshot is not None

    class Spy:
        calls = 0

        def read_verified_snapshot(self):  # type: ignore[no-untyped-def]
            self.calls += 1
            return snapshot

    spy = Spy()
    assert create_backup_envelope(spy) is not None  # type: ignore[arg-type]
    assert spy.calls == 1


def test_creation_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    with _store(tmp_path / "deterministic.sqlite3", 2) as store:
        first = create_backup_envelope(store)
        second = create_backup_envelope(store)
    assert first is not None and first == second
    assert first.to_mapping() == second.to_mapping()
    assert validate_backup_envelope(first.to_mapping()) == first


def test_candidate_and_nested_state_are_deeply_immutable(tmp_path: Path) -> None:
    backup = _candidate(tmp_path / "immutable.sqlite3")
    with pytest.raises(FrozenInstanceError):
        backup.environment = "LIVE"  # type: ignore[misc]
    with pytest.raises(TypeError):
        backup.canonical_durable_records[0].payload["entity_id"] = "changed"  # type: ignore[index]
    mapping = backup.to_mapping()
    mapping["environment"] = "LIVE"
    mapping["canonical_durable_records"][0]["payload"]["entity_id"] = "changed"  # type: ignore[index]
    assert backup.environment == "PAPER"
    assert backup.to_mapping() != mapping


@pytest.mark.parametrize("extra", ["authorized", "restore_allowed", "live_allowed"])
def test_top_level_schema_rejects_extras_even_when_rehashed(tmp_path: Path, extra: str) -> None:
    mapping = _candidate(tmp_path / f"{extra}.sqlite3").to_mapping()
    mapping[extra] = True
    _rehash(mapping)
    with pytest.raises(BackupEnvelopeError):
        validate_backup_envelope(mapping)


def test_top_level_schema_rejects_missing_field(tmp_path: Path) -> None:
    mapping = _candidate(tmp_path / "missing.sqlite3").to_mapping()
    del mapping["environment"]
    with pytest.raises(BackupEnvelopeError):
        validate_backup_envelope(mapping)


@pytest.mark.parametrize("metadata", [{}, {"extra": []}, [], None])
def test_integrity_metadata_is_closed(tmp_path: Path, metadata: object) -> None:
    mapping = _candidate(tmp_path / f"integrity-{type(metadata).__name__}.sqlite3").to_mapping()
    mapping["integrity_metadata"] = metadata
    with pytest.raises(BackupEnvelopeError):
        validate_backup_envelope(mapping)


def test_envelope_fingerprint_tamper_fails(tmp_path: Path) -> None:
    mapping = _candidate(tmp_path / "hash.sqlite3").to_mapping()
    mapping["envelope_fingerprint_sha256"] = "a" * 64
    with pytest.raises(BackupEnvelopeError, match="fingerprint mismatch"):
        validate_backup_envelope(mapping)


def test_stale_top_level_record_payload_hash_fails(tmp_path: Path) -> None:
    mapping = _candidate(tmp_path / "stale-record.sqlite3").to_mapping()
    mapping["canonical_durable_records"][0]["payload"]["entity_id"] = (  # type: ignore[index]
        "acct_01890f4c-7b9a-7cc1-8a2b-123456789abd"
    )
    _rehash(mapping)
    with pytest.raises(BackupEnvelopeError):
        validate_backup_envelope(mapping)


def test_descriptor_permutation_is_canonicalized_without_fingerprint_change(tmp_path: Path) -> None:
    backup = _candidate(tmp_path / "permutation.sqlite3", 4)
    mapping = backup.to_mapping()
    descriptors = mapping["integrity_metadata"]["state_store_transaction_descriptors"]  # type: ignore[index]
    mapping["integrity_metadata"]["state_store_transaction_descriptors"] = [  # type: ignore[index]
        descriptors[index] for index in (3, 0, 2, 1)
    ]
    parsed = validate_backup_envelope(mapping)
    assert parsed.envelope_fingerprint_sha256 == backup.envelope_fingerprint_sha256
    assert [
        item.target_generation
        for item in parsed.integrity_metadata.state_store_transaction_descriptors
    ] == [1, 2, 3, 4]


@pytest.mark.parametrize("generations", [[1, 2, 2, 4], [1, 3, 4], [1, 2, 3, 4, 5]])
def test_invalid_descriptor_multisets_fail(tmp_path: Path, generations: list[int]) -> None:
    mapping = _candidate(tmp_path / f"multiset-{len(generations)}.sqlite3", 4).to_mapping()
    source = mapping["integrity_metadata"]["state_store_transaction_descriptors"]  # type: ignore[index]
    by_generation = {item["target_generation"]: item for item in source}
    mapping["integrity_metadata"]["state_store_transaction_descriptors"] = [  # type: ignore[index]
        by_generation.get(generation, source[-1]) for generation in generations
    ]
    _rehash(mapping)
    with pytest.raises(BackupEnvelopeError):
        validate_backup_envelope(mapping)


def test_true_future_descriptor_fails_for_g4_envelope(tmp_path: Path) -> None:
    mapping = _candidate(tmp_path / "future.sqlite3", 4).to_mapping()
    descriptors = mapping["integrity_metadata"]["state_store_transaction_descriptors"]  # type: ignore[index]
    future = dict(descriptors[-1])
    future["expected_current_generation"] = 4
    future["target_generation"] = 5
    future["pre_state_fingerprint_sha256"] = future["post_state_fingerprint_sha256"]
    future["pre_history_tail_fingerprint_sha256"] = future["post_history_tail_fingerprint_sha256"]
    _rehash_descriptor(future)
    descriptors.append(future)
    _rehash(mapping)
    with pytest.raises(BackupEnvelopeError):
        validate_backup_envelope(mapping)


def test_self_consistent_descriptor_chain_discontinuity_fails(tmp_path: Path) -> None:
    mapping = _candidate(tmp_path / "chain-tamper.sqlite3", 2).to_mapping()
    descriptors = mapping["integrity_metadata"]["state_store_transaction_descriptors"]  # type: ignore[index]
    descriptors[1]["pre_state_fingerprint_sha256"] = "e" * 64
    _rehash_descriptor(descriptors[1])
    mapping["transaction_fingerprint_sha256"] = descriptors[1]["transaction_fingerprint_sha256"]
    _rehash(mapping)
    with pytest.raises(BackupEnvelopeError):
        validate_backup_envelope(mapping)


@pytest.mark.parametrize("field", ["api_secret", "verifier"])
def test_fully_rehashed_nested_forbidden_field_fails_security_scan(
    tmp_path: Path, field: str
) -> None:
    mapping = _candidate(tmp_path / f"nested-{field}.sqlite3").to_mapping()
    descriptor = mapping["integrity_metadata"]["state_store_transaction_descriptors"][0]  # type: ignore[index]
    descriptor["immutable_history_appends"] = [_record_with_nested_forbidden_field(field)]
    _rehash_descriptor(descriptor)
    mapping["transaction_fingerprint_sha256"] = descriptor["transaction_fingerprint_sha256"]
    _rehash(mapping)
    with pytest.raises(BackupEnvelopeError, match="forbidden payload field"):
        validate_backup_envelope(mapping)


@pytest.mark.parametrize(
    "kind",
    ["LocalDurableStateEvidence", "M0.3ProtectedMembership", "AuthenticationProof"],
)
def test_forbidden_record_kinds_fail_closed_at_stage_1(tmp_path: Path, kind: str) -> None:
    mapping = _candidate(tmp_path / f"forbidden-{kind}.sqlite3").to_mapping()
    record = mapping["canonical_durable_records"][0]  # type: ignore[index]
    record["representation_name"] = kind
    record["payload_fingerprint_sha256"] = canonical_json_sha256(record["payload"])
    _rehash(mapping)
    with pytest.raises(BackupEnvelopeError, match="contains an invalid record"):
        validate_backup_envelope(mapping)


@pytest.mark.parametrize(
    ("field", "generation"),
    [("target_generation", 0), ("expected_current_generation", 1)],
)
def test_descriptor_boolean_integer_attacks_fail(
    tmp_path: Path, field: str, generation: int
) -> None:
    mapping = _candidate(tmp_path / f"descriptor-bool-{field}.sqlite3", 2).to_mapping()
    descriptor = mapping["integrity_metadata"]["state_store_transaction_descriptors"][generation]  # type: ignore[index]
    descriptor[field] = True
    _rehash_descriptor(descriptor)
    _rehash(mapping)
    with pytest.raises(BackupEnvelopeError):
        validate_backup_envelope(mapping)


def test_self_consistent_top_level_wrong_bucket_fails(tmp_path: Path) -> None:
    backup = _candidate(tmp_path / "top-wrong-bucket.sqlite3")
    mapping = backup.to_mapping()
    current_record = mapping["canonical_durable_records"][0]  # type: ignore[index]
    history_record = mapping["immutable_recovery_history"][0]  # type: ignore[index]
    mapping["canonical_durable_records"] = [history_record]
    mapping["immutable_recovery_history"] = [current_record]
    parsed_current = (PersistenceRecord.from_mapping(history_record),)
    parsed_history = (PersistenceRecord.from_mapping(current_record),)
    history_hash = history_tail_fingerprint_sha256(parsed_history)
    state_hash = state_fingerprint_sha256(
        account_id=backup.account_id,
        device_installation_id=backup.device_installation_id,
        state_store_schema_version=backup.state_store_schema_version,
        state_store_identity_fingerprint_sha256=backup.state_store_identity_fingerprint_sha256,
        environment=backup.environment,
        protected_freshness_generation=backup.local_protected_freshness_generation,
        current_records=parsed_current,
        history_tail_fingerprint_sha256=history_hash,
    )
    descriptor = mapping["integrity_metadata"]["state_store_transaction_descriptors"][0]  # type: ignore[index]
    descriptor["post_state_fingerprint_sha256"] = state_hash
    descriptor["post_history_tail_fingerprint_sha256"] = history_hash
    _rehash_descriptor(descriptor)
    mapping["state_fingerprint_sha256"] = state_hash
    mapping["history_tail_fingerprint_sha256"] = history_hash
    mapping["transaction_fingerprint_sha256"] = descriptor["transaction_fingerprint_sha256"]
    _rehash(mapping)
    with pytest.raises(BackupEnvelopeError, match="verified StateStore snapshot"):
        validate_backup_envelope(mapping)


@pytest.mark.parametrize(
    "field",
    [
        "backup_envelope_schema_version",
        "state_store_schema_version",
        "local_protected_freshness_generation",
    ],
)
def test_boolean_integer_attacks_fail(tmp_path: Path, field: str) -> None:
    mapping = _candidate(tmp_path / f"bool-{field}.sqlite3").to_mapping()
    mapping[field] = True
    _rehash(mapping)
    with pytest.raises(BackupEnvelopeError):
        validate_backup_envelope(mapping)


def test_non_finite_payload_fails_closed(tmp_path: Path) -> None:
    mapping = _candidate(tmp_path / "nan.sqlite3").to_mapping()
    mapping["canonical_durable_records"][0]["payload"]["entity_id"] = float("nan")  # type: ignore[index]
    with pytest.raises(BackupEnvelopeError):
        validate_backup_envelope(mapping)


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM state_store_transaction_descriptors WHERE target_generation=1",
        "UPDATE state_store_metadata SET state_fingerprint_sha256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'",
        "UPDATE state_store_current_records SET record_json='{}'",
    ],
)
def test_corrupt_store_cannot_create_candidate(tmp_path: Path, statement: str) -> None:
    with _store(tmp_path / "corrupt.sqlite3") as store:
        store._connection.execute(statement)
        with pytest.raises(StateStoreError):
            create_backup_envelope(store)


def test_creation_is_read_only_and_schema_remains_exact(tmp_path: Path) -> None:
    with _store(tmp_path / "readonly.sqlite3") as store:
        before = store.read_verified_snapshot()
        assert create_backup_envelope(store) is not None
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


def test_nested_integrity_carrier_has_exact_field_set() -> None:
    assert [field.name for field in fields(BackupIntegrityMetadata)] == [
        "state_store_transaction_descriptors"
    ]
