"""Focused Stage-1 proofs for the frozen InitialSecurityState carriers."""

from dataclasses import replace

import pytest

from bot_core.persistence.fingerprints import canonical_json_sha256
from bot_core.persistence.lifecycle_records import persistence_record
from bot_core.persistence.records import PersistenceRecordError, validate_persistence_record

ACCOUNT = "acct_01890f4c-7b9a-7cc1-8a2b-123456789abc"
OPERATOR = "op_01890f4c-7b9a-7cc1-8a2b-123456789abc"
DEVICE = "dev_01890f4c-7b9a-7cc1-8a2b-123456789abc"


def _state(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "account_id": ACCOUNT,
        "operator_id": OPERATOR,
        "device_installation_id": DEVICE,
        "security_generation": 1,
        "session_generation": 1,
        "state": "ESTABLISHED",
        "bootstrap_claim_fingerprint_sha256": "a" * 64,
    }
    value.update(changes)
    value["content_fingerprint_sha256"] = canonical_json_sha256(value)
    return value


def test_exact_current_and_accepted_carriers_are_intrinsically_valid() -> None:
    upstream = _state()
    current = persistence_record(
        "InitialSecurityState current state",
        f"direct:InitialSecurityState current state:{ACCOUNT}:{DEVICE}",
        upstream,
    )
    history_payload = {
        "fact_kind": "InitialSecurityState accepted history",
        "upstream_payload": upstream,
        "upstream_payload_fingerprint_sha256": canonical_json_sha256(upstream),
    }
    history = persistence_record(
        "InitialSecurityState accepted history",
        f"immutable:InitialSecurityState accepted history:{ACCOUNT}:{DEVICE}:1:1",
        history_payload,
    )
    validate_persistence_record(current)
    validate_persistence_record(history)
    # Stage 1 proves carrier integrity only; it exposes no accepted/current resolver.
    assert current.payload == history.payload["upstream_payload"]


@pytest.mark.parametrize(
    "change",
    [
        {"account_id": "acct_wrong"},
        {"device_installation_id": "dev_wrong"},
        {"operator_id": "op_wrong"},
        {"security_generation": 0},
        {"session_generation": True},
        {"bootstrap_claim_fingerprint_sha256": "A" * 64},
        {"state": "READY"},
    ],
)
def test_malformed_upstream_initial_security_facts_are_rejected(change: dict[str, object]) -> None:
    payload = _state(**change)
    record = persistence_record(
        "InitialSecurityState current state",
        f"direct:InitialSecurityState current state:{ACCOUNT}:{DEVICE}",
        _state(),
    )
    candidate = replace(
        record,
        payload=payload,
        payload_fingerprint_sha256=canonical_json_sha256(payload),
    )
    with pytest.raises(PersistenceRecordError):
        validate_persistence_record(candidate)


def test_current_record_key_is_exact_account_device_scope() -> None:
    payload = _state()
    with pytest.raises(PersistenceRecordError):
        persistence_record(
            "InitialSecurityState current state",
            f"direct:InitialSecurityState current state:{ACCOUNT}:dev_01890f4c-7b9a-7cc1-8a2b-123456789abd",
            payload,
        )
