from dataclasses import replace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from bot_core.authenticated_issuer_history import (
    NO_PREDECESSOR,
    AttestedHistoryHead,
    HistoryContractError,
    HistoryEventIdentity,
    HistoryStreamIdentity,
    LocalCheckpointProvider,
    ReconciliationOutcome,
    ReferenceAuthenticatedHistory,
    attest_head,
    canonical_json_bytes,
    reconcile_checkpoint,
    verify_head,
)
from bot_core.root_proof_issuer_substrate import (
    CredentialRoleIdentity,
    CredentialSemanticRole,
)


class Signer:
    def __init__(self, *, role=CredentialSemanticRole.HISTORY_ATTESTATION_SIGNING):
        self.key = Ed25519PrivateKey.generate()
        self.role = role

    def active_credential_identity(self):
        return CredentialRoleIdentity(self.role, "history-credential", "history-provider", "7", "lifecycle", "material")

    def sign_history_head(self, payload):
        return self.key.sign(payload)

    def public(self):
        return self.key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


@pytest.fixture
def stream():
    return HistoryStreamIdentity("account-genesis-history", "IndependentAccountGenesisRootProofIssuer", "PRODUCTION_LOCAL", "production", "tenant-a", "product-a", 1)


def identity(n=1):
    return HistoryEventIdentity(f"operation-{n}", f"attempt-{n}", f"proof-{n}")


def committed(stream, count=1):
    history = ReferenceAuthenticatedHistory(stream)
    records = []
    expected = NO_PREDECESSOR
    for n in range(1, count + 1):
        record, replay = history.append(expected_digest=expected, event_identity=identity(n), payload={"account_id": "ą", "generation": n, "enabled": True, "optional": None})
        assert not replay
        records.append(record)
        expected = record.authenticated_digest
    return history, records


def test_canonicalization_is_order_independent_and_exact_typed():
    assert canonical_json_bytes({"b": 1, "a": "ą"}) == canonical_json_bytes({"a": "ą", "b": 1})
    with pytest.raises(TypeError):
        canonical_json_bytes({"bad": 1.0})
    with pytest.raises(TypeError):
        canonical_json_bytes({"bad": {1: "not text"}})


@pytest.mark.parametrize("field,value", [
    ("environment", "test"), ("trust_domain", "tenant-b"),
    ("product_scope", "product-b"), ("stream_id", "other-history"),
    ("security_epoch", 2),
])
def test_cross_boundary_head_replay_fails(stream, field, value):
    _, records = committed(stream)
    signer = Signer()
    head = attest_head(records[0], signer)
    with pytest.raises(HistoryContractError, match="wrong stream"):
        verify_head(head, signer.public(), expected_stream=replace(stream, **{field: value}),
                    expected_credential_id="history-credential", expected_key_version="7")


def test_exact_replay_conflict_and_lost_response(stream):
    history, records = committed(stream)
    replay, was_replay = history.append(expected_digest="ignored-on-replay", event_identity=identity(), payload={"optional": None, "enabled": True, "generation": 1, "account_id": "ą"})
    assert was_replay and replay == records[0]
    with pytest.raises(HistoryContractError, match="conflicting"):
        history.append(expected_digest=records[0].authenticated_digest, event_identity=identity(), payload={"generation": 2})


def test_gap_fork_predecessor_and_cross_stream_fail_closed(stream):
    history, records = committed(stream)
    with pytest.raises(HistoryContractError, match="stale predecessor"):
        history.append(expected_digest=NO_PREDECESSOR, event_identity=identity(2), payload={"x": 1})
    with pytest.raises(TypeError):
        records[0].canonical_event_payload["x"] = 1  # type: ignore[index]


def test_payload_predecessor_and_sequence_tamper_detected(stream):
    history, records = committed(stream, 2)
    object.__setattr__(records[0], "event_digest", "sha256:" + "0" * 64)
    with pytest.raises(HistoryContractError):
        history.verify()
    history, records = committed(stream, 2)
    object.__setattr__(records[1], "predecessor_authenticated_digest", "sha256:" + "1" * 64)
    with pytest.raises(HistoryContractError):
        history.verify()
    history, records = committed(stream, 2)
    object.__setattr__(records[1], "sequence", 4)
    with pytest.raises(HistoryContractError):
        history.verify()


def test_attestation_signature_role_stream_credential_and_version(stream):
    _, records = committed(stream)
    signer = Signer()
    head = attest_head(records[0], signer)
    verify_head(head, signer.public(), expected_stream=stream, expected_credential_id="history-credential", expected_key_version="7")
    for changed in (replace(head, signature=b"x" * 64), replace(head, signing_credential_id="wrong"), replace(head, signing_key_version="8")):
        with pytest.raises(HistoryContractError):
            verify_head(changed, signer.public(), expected_stream=stream, expected_credential_id="history-credential", expected_key_version="7")
    wrong_stream = replace(stream, trust_domain="tenant-b")
    with pytest.raises(HistoryContractError):
        verify_head(head, signer.public(), expected_stream=wrong_stream, expected_credential_id="history-credential", expected_key_version="7")
    with pytest.raises(HistoryContractError, match="HISTORY"):
        attest_head(records[0], Signer(role=CredentialSemanticRole.ROOT_PROOF_ISSUER_SIGNING))


def test_checkpoint_replay_behind_ahead_rewind_and_split_brain(stream):
    _, records = committed(stream, 2)
    signer = Signer()
    h1, h2 = attest_head(records[0], signer), attest_head(records[1], signer)
    checkpoints = LocalCheckpointProvider("local-checkpoint", stream)
    cp1, replay = checkpoints.advance(expected_revision=0, head=h1)
    assert not replay
    assert reconcile_checkpoint(h2, cp1) is ReconciliationOutcome.STALE
    cp1_again, replay = checkpoints.advance(expected_revision=1, head=h1)
    assert replay and cp1_again == cp1
    cp2, _ = checkpoints.advance(expected_revision=1, head=h2)
    assert reconcile_checkpoint(h1, cp2) is ReconciliationOutcome.CORRUPT
    with pytest.raises(HistoryContractError, match="rewind"):
        checkpoints.advance(expected_revision=2, head=h1)
    split = replace(h2, record_digest="sha256:" + "f" * 64)
    with pytest.raises(HistoryContractError, match="split brain"):
        checkpoints.advance(expected_revision=2, head=split)


def test_checkpoint_conflict_and_absence_are_not_not_found(stream):
    _, records = committed(stream)
    signer = Signer()
    head = attest_head(records[0], signer)
    checkpoints = LocalCheckpointProvider("local-checkpoint", stream)
    cp, _ = checkpoints.advance(expected_revision=0, head=head)
    assert reconcile_checkpoint(None, cp) is ReconciliationOutcome.CORRUPT
    assert reconcile_checkpoint(None, None) is ReconciliationOutcome.NOT_FOUND
    assert reconcile_checkpoint(replace(head, record_digest="sha256:" + "e" * 64), cp) is ReconciliationOutcome.CONFLICT


def test_restart_identity_is_value_based(stream):
    history, records = committed(stream)
    restarted = ReferenceAuthenticatedHistory(replace(stream))
    restarted._records = list(records)
    restarted._events = {records[0].event_identity: records[0]}
    restarted.verify()
