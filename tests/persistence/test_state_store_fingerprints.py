from __future__ import annotations

from dataclasses import replace
import math

import pytest

from bot_core.persistence.fingerprints import (
    canonical_json_sha256,
    history_tail_fingerprint_sha256,
    state_fingerprint_sha256,
    transaction_fingerprint_sha256,
)
from tests.persistence.test_state_store_records import ACCOUNT_ID, DEVICE_ID, _account, _runtime


def test_canonical_json_is_key_order_unicode_and_nested_order_invariant() -> None:
    left = {"ż": {"b": [2, {"y": 1, "x": 0}], "a": "łódź"}, "a": True}
    right = {"a": True, "ż": {"a": "łódź", "b": [2, {"x": 0, "y": 1}]}}
    assert canonical_json_sha256(left) == canonical_json_sha256(right)
    assert canonical_json_sha256(left).isalnum()
    assert canonical_json_sha256(left) == canonical_json_sha256(left).lower()


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_canonical_json_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(ValueError):
        canonical_json_sha256({"number": value})


def test_record_and_empty_history_fingerprints_are_deterministic() -> None:
    first, second = _runtime(), _runtime("run_01890f4c-7b9a-7cc1-8a2b-123456789abd")
    assert history_tail_fingerprint_sha256(()) == canonical_json_sha256([])
    assert history_tail_fingerprint_sha256((first, second)) == history_tail_fingerprint_sha256(
        (second, first)
    )
    assert history_tail_fingerprint_sha256((first,)) != history_tail_fingerprint_sha256(
        (first, second)
    )
    changed = replace(first, payload_fingerprint_sha256="a" * 64)
    assert history_tail_fingerprint_sha256((first,)) != history_tail_fingerprint_sha256((changed,))


def _state(**changes: object) -> str:
    values = {
        "account_id": ACCOUNT_ID,
        "device_installation_id": DEVICE_ID,
        "state_store_schema_version": 1,
        "state_store_identity_fingerprint_sha256": "1" * 64,
        "environment": "PAPER",
        "protected_freshness_generation": 1,
        "current_records": (_account(),),
        "history_tail_fingerprint_sha256": "2" * 64,
    }
    values.update(changes)
    return state_fingerprint_sha256(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("account_id", "acct_01890f4c-7b9a-7cc1-8a2b-123456789abd"),
        ("device_installation_id", "dev_01890f4c-7b9a-7cc1-8a2b-123456789abd"),
        ("state_store_schema_version", 2),
        ("state_store_identity_fingerprint_sha256", "3" * 64),
        ("environment", "TESTNET"),
        ("protected_freshness_generation", 2),
        ("current_records", ()),
        ("history_tail_fingerprint_sha256", "4" * 64),
    ],
)
def test_every_state_projection_field_affects_hash(field: str, value: object) -> None:
    assert _state() != _state(**{field: value})


def test_every_transaction_projection_field_affects_hash_and_stored_hash_is_excluded() -> None:
    base = {
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
    }
    variants = [
        "acct_01890f4c-7b9a-7cc1-8a2b-123456789abd",
        "dev_01890f4c-7b9a-7cc1-8a2b-123456789abd",
        "4" * 64,
        2,
        "TESTNET",
        1,
        2,
        "5" * 64,
        "6" * 64,
        "7" * 64,
        "8" * 64,
        [_account().to_mapping()],
        [_runtime().to_mapping()],
    ]
    for field, value in zip(base, variants, strict=True):
        changed = {**base, field: value}
        assert transaction_fingerprint_sha256(base) != transaction_fingerprint_sha256(changed)
