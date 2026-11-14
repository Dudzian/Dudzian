"""Independent TEST authority signatures exercise mechanics without enrollment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from deployment import stage9_observer_trust_v4c as trust
from deployment.canonical_json import canonical_json_bytes


@pytest.fixture
def material():
    private_keys = tuple(Ed25519PrivateKey.generate() for _ in range(3))
    keys = tuple(
        trust.TestAuthorityKeyV4c(
            f"reviewer-{index}",
            private.public_key()
            .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            .hex(),
            0,
            1000,
        )
        for index, private in enumerate(private_keys)
    )
    authority = trust.TestAuthorityV4c("test-authority", 1, keys, 2, 0, 1000)
    payload = {
        "schema": trust.DOMAINS["time"],
        "authority_id": authority.authority_id,
        "policy_version": 1,
        "algorithm": trust.ALGORITHM,
        "domain": trust.DOMAINS["time"],
        "valid_from": 10,
        "valid_until": 900,
        "proof_id": "test-proof",
        "time_source_id": "test-clock",
        "repository_id": "test-repository",
        "snapshot_id": "test-snapshot",
        "snapshot_sha256": "a1" * 32,
        "context_sha256": "b2" * 32,
        "checkpoint_id": "test-checkpoint",
        "evaluation_time": 100,
    }

    def wire(document=None, count=2):
        document = payload if document is None else document
        message = trust.signing_message_v4c("time", document)
        return canonical_json_bytes(
            {
                "payload": document,
                "signatures": [
                    {"signer_id": key.signer_id, "signature_hex": private.sign(message).hex()}
                    for key, private in zip(keys[:count], private_keys[:count], strict=True)
                ],
            }
        )

    return payload, authority, wire


def test_real_threshold_signatures_leave_authorization_unverified(material):
    _, authority, wire = material
    record = trust.parse_time_evidence_v4c(wire())
    assert trust.verify_record_v4c(record, authority, 100) is None
    assert record.authorization == "UNVERIFIED"
    assert authority.scope == "SYNTHETIC_TEST_ONLY"
    record.payload["snapshot_id"] = "changed-copy"
    assert record.payload["snapshot_id"] == "test-snapshot"


def test_signing_preimage_is_apt_specific_domain_nul_and_raw_digest(material):
    payload, _, _ = material
    message = trust.signing_message_v4c("time", payload)
    assert (
        message
        == b"stage9-apt-time-evidence-v1\0" + hashlib.sha256(canonical_json_bytes(payload)).digest()
    )
    assert message.count(b"\0") >= 1
    assert not message.startswith(b"CryptoHunter.Stage9ObserverAcceptanceV1")


@pytest.mark.parametrize("authority", [None, {}, True])
def test_material_cannot_supply_its_own_anchor(material, authority):
    _, _, wire = material
    with pytest.raises(trust.V4cRefusal, match="V4C_TEST_AUTHORITY_MISSING"):
        trust.verify_record_v4c(trust.parse_time_evidence_v4c(wire()), authority, 100)


@pytest.mark.parametrize("threshold", [True, False, 0, -1, 4, 2.0, None, "2"])
def test_threshold_is_exact_integer_bounded_by_distinct_keys(material, threshold):
    _, authority, wire = material
    with pytest.raises(trust.V4cRefusal):
        trust.verify_record_v4c(
            trust.parse_time_evidence_v4c(wire()), replace(authority, threshold=threshold), 100
        )


def test_missing_threshold_signature_refuses(material):
    _, authority, wire = material
    with pytest.raises(trust.V4cRefusal, match="V4C_THRESHOLD_REJECTED"):
        trust.verify_record_v4c(trust.parse_time_evidence_v4c(wire(count=1)), authority, 100)


@pytest.mark.parametrize(
    "mutation", ["authority", "key", "substituted-key", "same-key-two-ids", "same-id"]
)
def test_revocation_and_key_substitution_fail_closed(material, mutation):
    _, authority, wire = material
    keys = list(authority.keys)
    if mutation == "authority":
        authority = replace(authority, revoked=True)
    elif mutation == "key":
        keys[0] = replace(keys[0], revoked=True)
    elif mutation == "substituted-key":
        keys[0] = replace(keys[0], public_key_hex=keys[2].public_key_hex)
        keys = keys[:2]
    elif mutation == "same-key-two-ids":
        keys[1] = replace(keys[1], public_key_hex=keys[0].public_key_hex)
    else:
        keys[1] = replace(keys[1], signer_id=keys[0].signer_id)
    authority = replace(authority, keys=tuple(keys))
    with pytest.raises(trust.V4cRefusal):
        trust.verify_record_v4c(trust.parse_time_evidence_v4c(wire()), authority, 100)


@pytest.mark.parametrize(
    "mutation", ["unknown", "invalid", "duplicate", "reordered", "embedded-key", "threshold"]
)
def test_extra_signature_and_envelope_fields_never_partially_succeed(material, mutation):
    _, authority, wire = material
    envelope = json.loads(wire(count=3))
    if mutation == "unknown":
        envelope["signatures"][2]["signer_id"] = "unknown"
    elif mutation == "invalid":
        envelope["signatures"][2]["signature_hex"] = "0" * 128
    elif mutation == "duplicate":
        envelope["signatures"][2] = envelope["signatures"][0]
    elif mutation == "reordered":
        envelope["signatures"].reverse()
    elif mutation == "embedded-key":
        envelope["signatures"][0]["public_key_hex"] = authority.keys[0].public_key_hex
    else:
        envelope["threshold"] = 1
    with pytest.raises(trust.V4cRefusal):
        record = trust.parse_time_evidence_v4c(canonical_json_bytes(envelope))
        trust.verify_record_v4c(record, authority, 100)


@pytest.mark.parametrize("mutation", ["payload-bytes", "signatures", "kind", "wire"])
def test_constructed_record_cannot_bypass_validated_wire(material, mutation):
    _, authority, wire = material
    record = trust.parse_time_evidence_v4c(wire())
    changes = {
        "payload-bytes": {"payload_bytes": b"{}"},
        "signatures": {"signatures": ()},
        "kind": {"kind": "acceptance"},
        "wire": {"raw": b"{}"},
    }
    with pytest.raises(trust.V4cRefusal):
        trust.verify_record_v4c(replace(record, **changes[mutation]), authority, 100)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "CryptoHunter.Stage9ObserverAcceptanceV1"),
        ("domain", "stage9-apt-checkpoint-v1"),
        ("algorithm", "rsa"),
        ("evaluation_time", True),
        ("evaluation_time", 900),
        ("policy_version", 0),
        ("policy_version", False),
        ("snapshot_sha256", "0" * 64),
        ("snapshot_sha256", "AA" * 32),
        ("checkpoint_id", "bad\x00id"),
        ("time_source_id", None),
    ],
)
def test_closed_payload_values_refuse(material, field, value):
    _, _, wire = material
    envelope = json.loads(wire())
    envelope["payload"][field] = value
    with pytest.raises(trust.V4cRefusal):
        trust.parse_time_evidence_v4c(canonical_json_bytes(envelope))


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"{}",
        b"[]",
        b'{"x":1,"x":1}',
        b"[" * 9 + b"]" * 9,
        b'"\\ud800"',
        b'{"\\ud800":0}',
        b'{"x":NaN}',
        b'{"x":1.0}',
        b'{"x":-1}',
        b'{"x":999999999999999999999}',
        b'{"x":"unterminated}',
        b"{}" + b" " * 65535,
        b"\xef\xbb\xbf{}",
    ],
)
def test_decoder_rejects_malformed_oversized_or_unbounded_values(raw):
    with pytest.raises(trust.V4cRefusal):
        trust.parse_time_evidence_v4c(raw)


def test_noncanonical_json_is_refused_without_reformatting(material):
    _, _, wire = material
    with pytest.raises(trust.V4cRefusal, match="V4C_NONCANONICAL"):
        trust.parse_time_evidence_v4c(json.dumps(json.loads(wire()), indent=2).encode())


@pytest.mark.parametrize(
    "mutation",
    [
        "expired-authority",
        "expired-key",
        "different-policy",
        "different-authority",
        "interval-widening",
    ],
)
def test_anchor_policy_and_validity_are_bound(material, mutation):
    _, authority, wire = material
    if mutation == "expired-authority":
        authority = replace(authority, valid_until=100)
    elif mutation == "expired-key":
        authority = replace(
            authority, keys=(replace(authority.keys[0], valid_until=100), *authority.keys[1:])
        )
    elif mutation == "different-policy":
        authority = replace(authority, policy_version=2)
    elif mutation == "different-authority":
        authority = replace(authority, authority_id="other-authority")
    else:
        authority = replace(authority, valid_from=50)
    with pytest.raises(trust.V4cRefusal):
        trust.verify_record_v4c(trust.parse_time_evidence_v4c(wire()), authority, 100)


@pytest.mark.parametrize("issued_at", [99, 100, 101])
def test_acceptance_creation_time_cannot_be_after_signed_evaluation(material, issued_at):
    base, _, _ = material
    payload = {
        key: value
        for key, value in base.items()
        if key not in {"proof_id", "time_source_id", "evaluation_time"}
    }
    payload.update(
        schema=trust.DOMAINS["acceptance"],
        domain=trust.DOMAINS["acceptance"],
        authority_id="test-reviewer",
        keyring_sha256="c3" * 32,
        allowed_signers=[{"signing_fingerprint": "A1" * 20, "primary_fingerprint": "B2" * 20}],
        sequence=1,
        issued_at=issued_at,
        checkpoint_sha256="d4" * 32,
        rotation_policy="EXPLICIT_CONTEXT_LINK",
        revocation_policy="DENY_REVOKED",
        previous_context_sha256=None,
    )
    private = Ed25519PrivateKey.generate()
    key = trust.TestAuthorityKeyV4c(
        "reviewer",
        private.public_key()
        .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        .hex(),
        0,
        1000,
    )
    authority = trust.TestAuthorityV4c("test-reviewer", 1, (key,), 1, 0, 1000)
    raw = canonical_json_bytes(
        {
            "payload": payload,
            "signatures": [
                {
                    "signer_id": "reviewer",
                    "signature_hex": private.sign(
                        trust.signing_message_v4c("acceptance", payload)
                    ).hex(),
                }
            ],
        }
    )
    record = trust.parse_acceptance_v4c(raw)
    if issued_at <= 100:
        trust.verify_record_v4c(record, authority, 100)
    else:
        with pytest.raises(trust.V4cRefusal, match="V4C_SIGNING_TIME_REJECTED"):
            trust.verify_record_v4c(record, authority, 100)


def _checkpoint_payload(material):
    base, _, _ = material
    payload = {
        key: value
        for key, value in base.items()
        if key not in {"proof_id", "time_source_id", "evaluation_time"}
    }
    payload.update(
        schema=trust.DOMAINS["checkpoint"],
        domain=trust.DOMAINS["checkpoint"],
        checkpoint_source_id="checkpoint-source",
        sequence=1,
        successor_policy="EXACT_NEXT_SEQUENCE",
        rotation_policy="EXPLICIT_CONTEXT_LINK",
        revoked=False,
        previous_checkpoint_id=None,
        previous_snapshot_sha256=None,
        previous_context_sha256=None,
        permitted_context_successors=[],
    )
    return payload


def _unverified_wire(payload):
    return canonical_json_bytes(
        {
            "payload": payload,
            "signatures": [{"signer_id": "test", "signature_hex": "0" * 128}],
        }
    )


def test_null_predecessor_is_parseable_but_never_an_authorization(material):
    record = trust.parse_checkpoint_v4c(_unverified_wire(_checkpoint_payload(material)))
    assert record.payload["previous_checkpoint_id"] is None
    assert record.authorization == "UNVERIFIED"


@pytest.mark.parametrize(
    "field", ["previous_checkpoint_id", "previous_snapshot_sha256", "previous_context_sha256"]
)
def test_partial_predecessor_links_refuse(material, field):
    payload = _checkpoint_payload(material)
    payload[field] = "predecessor" if field == "previous_checkpoint_id" else "ab" * 32
    with pytest.raises(trust.V4cRefusal, match="V4C_CHECKPOINT_LINK_REJECTED"):
        trust.parse_checkpoint_v4c(_unverified_wire(payload))


@pytest.mark.parametrize(
    "successors", [["ab" * 32] * 2, ["ff" * 32, "aa" * 32], ["ab" * 32] * 33, None]
)
def test_context_successor_allowlist_is_bounded_unique_and_ordered(material, successors):
    payload = _checkpoint_payload(material)
    payload["permitted_context_successors"] = successors
    with pytest.raises(trust.V4cRefusal):
        trust.parse_checkpoint_v4c(_unverified_wire(payload))
