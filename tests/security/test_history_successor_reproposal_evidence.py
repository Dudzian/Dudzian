"""Fail-closed normalization of already signed historical authorization facts."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import replace

import psycopg
import pytest
from psycopg.types.json import Jsonb

from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.postgresql_atomic_root_proof_commit import AtomicRootProofCommitConflict
from bot_core.postgresql_atomic_root_proof_schema import PostgreSQLAtomicRootProofProvisioning
from bot_core.postgresql_history_successor_reproposal_evidence import (
    _same_original_authorization_evidence,
)
from bot_core.postgresql_history_successor_reproposal_schema import (
    reviewed_reproposal_function_sources,
)
from tests.licensing.test_cha_attempt_store_stage9 import authorization
from tests.security.test_postgresql_checkpoint_authority import BASE_DSN


def _evidence(cas):
    return {
        "schema_version": "RootProofOperationScopedAuthorizationEvidenceV1",
        "scope": "EXACT_OPERATION_SCOPED",
        "entitlement": {"authoritative_state_revision": 1, "binding": {"kind": "UNBOUND"}},
        "requester": {"registry_revision": 1, "lifecycle": "ACTIVE"},
        "providers": {
            "entitlement": {
                "identity": {"provider_namespace": "postgresql:entitlement"},
                "capabilities": {
                    "implemented": True,
                    "authoritative_reads": True,
                    "durable_state": True,
                    "compare_and_swap": cas,
                    "signing": None,
                    "checkpoint": None,
                },
            }
        },
    }


def _auth(raw):
    return replace(authorization(), authorization_evidence_sha256=hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize("historical_cas", [False, True])
def test_only_legacy_cas_revocation_can_change_historical_authorization_digest(historical_cas):
    original = canonical_json_bytes(_evidence(historical_cas))
    fresh = canonical_json_bytes(_evidence(False))
    auth = _auth(original)
    _same_original_authorization_evidence(
        auth, original, _auth(fresh), fresh, historical_entitlement_cas=historical_cas
    )
    assert original == canonical_json_bytes(_evidence(historical_cas))
    assert auth.authorization_evidence_sha256 == hashlib.sha256(original).hexdigest()


@pytest.mark.parametrize(
    "path,value",
    [
        (("entitlement", "authoritative_state_revision"), 2),
        (("entitlement", "binding", "kind"), "BOUND"),
        (("requester", "registry_revision"), 2),
        (("requester", "lifecycle"), "REVOKED"),
        (("providers", "entitlement", "identity", "provider_namespace"), "postgresql:other"),
        (("providers", "entitlement", "capabilities", "authoritative_reads"), False),
        (("providers", "entitlement", "capabilities", "durable_state"), False),
        (("providers", "entitlement", "capabilities", "compare_and_swap"), True),
        (("providers", "entitlement", "capabilities", "compare_and_swap"), 0),
        (("providers", "entitlement", "capabilities", "signing"), {}),
    ],
)
def test_any_other_live_fact_change_refuses_historical_signed_request(path, value):
    original = canonical_json_bytes(_evidence(True))
    changed = deepcopy(_evidence(False))
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    fresh = canonical_json_bytes(changed)
    with pytest.raises(AtomicRootProofCommitConflict):
        _same_original_authorization_evidence(
            _auth(original), original, _auth(fresh), fresh, historical_entitlement_cas=True
        )


def test_authorization_tuple_change_cannot_be_hidden_by_evidence_hash_normalization():
    original = canonical_json_bytes(_evidence(True))
    fresh = canonical_json_bytes(_evidence(False))
    changed = replace(_auth(fresh), requester_key_version=2)
    with pytest.raises(AtomicRootProofCommitConflict):
        _same_original_authorization_evidence(
            _auth(original), original, changed, fresh, historical_entitlement_cas=True
        )


@pytest.mark.parametrize("mismatched", ["original", "fresh"])
def test_original_and_fresh_evidence_must_match_their_own_exact_hash(mismatched):
    original = canonical_json_bytes(_evidence(True))
    fresh = canonical_json_bytes(_evidence(False))
    old_auth, fresh_auth = _auth(original), _auth(fresh)
    if mismatched == "original":
        old_auth = replace(old_auth, authorization_evidence_sha256="a" * 64)
    else:
        fresh_auth = replace(fresh_auth, authorization_evidence_sha256="a" * 64)
    with pytest.raises(AtomicRootProofCommitConflict):
        _same_original_authorization_evidence(
            old_auth, original, fresh_auth, fresh, historical_entitlement_cas=True
        )


def test_read_only_historical_provider_cannot_claim_former_standalone_cas_authority():
    original = canonical_json_bytes(_evidence(True))
    fresh = canonical_json_bytes(_evidence(False))
    with pytest.raises(AtomicRootProofCommitConflict):
        _same_original_authorization_evidence(
            _auth(original), original, _auth(fresh), fresh, historical_entitlement_cas=False
        )


@pytest.mark.external_postgresql
@pytest.mark.parametrize("changed", [None, "payload_hex", "event_payload", "history_credential"])
def test_postgresql_immutable_projection_preserves_original_fields(changed):
    """Execute the actual admission predicate; these JSON objects authorize nothing."""
    config = PostgreSQLAtomicRootProofProvisioning(
        "atomic",
        "atomic_owner",
        "atomic_writer",
        "atomic_admin",
        "entitlement",
        "intent",
        "retention",
        "history",
        "trust",
        "isolated_instance",
        "requester",
        "claimant",
    )
    source = reviewed_reproposal_function_sources(config)[
        "approve_history_proposal(jsonb,bigint,text)"
    ]
    predicate = (
        "(p_package - "
        + source.split(" IF (p_package - ", 1)[1].split(
            "\n THEN RAISE EXCEPTION 'reproposal must preserve original proof and event bytes'", 1
        )[0]
    )
    predicate = predicate.replace("p_package", "%(candidate)s::jsonb").replace(
        "base", "%(original)s::jsonb"
    )
    original = {
        "payload_hex": "original-root-proof",
        "history_record": {
            "sequence": 1,
            "predecessor_authenticated_digest": "NO_PREDECESSOR",
            "authenticated_digest": "record-one",
            "canonical_payload_hex": "immutable-event",
        },
        "history_head": {"sequence": 1, "canonical_credential_hex": "original-history-signer"},
        "expected_history_digest": "NO_PREDECESSOR",
        "history_record_hex": "record-one",
        "history_head_hex": "head-one",
        "history_signature_hex": "signature-one",
    }
    candidate = deepcopy(original)
    candidate["history_record"].update(
        sequence=2,
        predecessor_authenticated_digest="authoritative-predecessor",
        authenticated_digest="record-two",
    )
    candidate.update(
        expected_history_digest="authoritative-predecessor",
        history_record_hex="record-two",
        history_head_hex="head-two",
        history_signature_hex="signature-two",
    )
    candidate["history_head"]["sequence"] = 2
    if changed == "payload_hex":
        candidate[changed] = "changed-root-proof"
    elif changed == "event_payload":
        candidate["history_record"]["canonical_payload_hex"] = "changed-event"
    elif changed == "history_credential":
        candidate["history_head"]["canonical_credential_hex"] = "another-history-signer"
    with psycopg.connect(BASE_DSN) as observer:
        observer.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE READ ONLY")
        row = observer.execute(
            "SELECT " + predicate,
            {"candidate": Jsonb(candidate), "original": Jsonb(original)},
        ).fetchone()
    assert row == (changed is not None,)
