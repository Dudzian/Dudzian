from __future__ import annotations

from dataclasses import fields

import pytest

from bot_core.persistence.fingerprints import transaction_fingerprint_sha256
from bot_core.persistence.transaction_descriptor import (
    StateStoreTransactionDescriptor,
    TransactionDescriptorError,
)
from tests.persistence.test_state_store_records import ACCOUNT_ID, DEVICE_ID, _account, _runtime


def _mapping(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "account_id": ACCOUNT_ID,
        "device_installation_id": DEVICE_ID,
        "state_store_identity_fingerprint_sha256": "1" * 64,
        "state_store_schema_version": 1,
        "environment": "PAPER",
        "expected_current_generation": None,
        "target_generation": 1,
        "pre_state_fingerprint_sha256": None,
        "pre_history_tail_fingerprint_sha256": None,
        "post_state_fingerprint_sha256": "2" * 64,
        "post_history_tail_fingerprint_sha256": "3" * 64,
        "current_record_mutations": [],
        "immutable_history_appends": [],
        "transaction_fingerprint_sha256": "0" * 64,
    }
    value.update(changes)
    value["transaction_fingerprint_sha256"] = transaction_fingerprint_sha256(
        {key: item for key, item in value.items() if key != "transaction_fingerprint_sha256"}
    )
    return value


def test_exact_14_fields_round_trip_and_metadata_only_arrays() -> None:
    descriptor = StateStoreTransactionDescriptor.from_mapping(_mapping())
    assert len(fields(descriptor)) == 14
    assert descriptor.to_mapping() == _mapping()
    assert descriptor.current_record_mutations == descriptor.immutable_history_appends == ()
    assert descriptor.has_valid_transaction_fingerprint()


@pytest.mark.parametrize("extra", ["authorized", "accepted", "live_allowed"])
def test_missing_extra_and_authority_fields_fail(extra: str) -> None:
    missing = _mapping()
    missing.pop("account_id")
    with pytest.raises(TransactionDescriptorError):
        StateStoreTransactionDescriptor.from_mapping(missing)
    with pytest.raises(TransactionDescriptorError):
        StateStoreTransactionDescriptor.from_mapping({**_mapping(), extra: True})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("account_id", "bad"),
        ("device_installation_id", ACCOUNT_ID),
        ("state_store_identity_fingerprint_sha256", "A" * 64),
        ("post_state_fingerprint_sha256", "z" * 64),
        ("state_store_schema_version", True),
        ("state_store_schema_version", 0),
        ("target_generation", True),
        ("target_generation", 0),
        ("environment", "HACKED"),
    ],
)
def test_intrinsic_scalar_validation(field: str, value: object) -> None:
    with pytest.raises(TransactionDescriptorError):
        StateStoreTransactionDescriptor.from_mapping(_mapping(**{field: value}))


def test_genesis_and_non_genesis_null_rules() -> None:
    with pytest.raises(TransactionDescriptorError):
        StateStoreTransactionDescriptor.from_mapping(_mapping(expected_current_generation=1))
    valid = _mapping(
        target_generation=2,
        expected_current_generation=1,
        pre_state_fingerprint_sha256="4" * 64,
        pre_history_tail_fingerprint_sha256="5" * 64,
    )
    assert StateStoreTransactionDescriptor.from_mapping(valid).target_generation == 2
    with pytest.raises(TransactionDescriptorError):
        StateStoreTransactionDescriptor.from_mapping(_mapping(target_generation=2))


def test_nested_records_are_validated_and_arrays_must_already_be_canonical() -> None:
    invalid = _account().to_mapping()
    invalid["payload_fingerprint_sha256"] = "a" * 64
    with pytest.raises(TransactionDescriptorError):
        StateStoreTransactionDescriptor.from_mapping(_mapping(current_record_mutations=[invalid]))
    records = [_runtime().to_mapping(), _account().to_mapping()]
    with pytest.raises(TransactionDescriptorError, match="canonically sorted"):
        StateStoreTransactionDescriptor.from_mapping(_mapping(current_record_mutations=records))
