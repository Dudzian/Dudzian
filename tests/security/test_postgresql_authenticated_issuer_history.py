"""Canonical storage envelopes fail closed before becoming history evidence.

Real PostgreSQL durability and privilege qualification live in the integration
suite; these tests exercise the frozen serializer and strict decoding boundary.
"""

from dataclasses import replace

import pytest

from bot_core.authenticated_issuer_history import (
    ATTESTATION_DOMAIN,
    NO_PREDECESSOR,
    HistoryContractError,
    HistoryEventIdentity,
    HistoryStreamIdentity,
    _credential_material,
    build_record,
    canonical_json_bytes,
)
from bot_core.postgresql_authenticated_issuer_history import (
    PostgreSQLAuthenticatedIssuerHistory,
    _head_from_row,
    _parse_canonical,
    _record_from_row,
    _unhex,
)
from bot_core.root_proof_issuer_runtime import RootProofIssuancePreflight
from bot_core.root_proof_issuer_substrate import CredentialRoleIdentity, CredentialSemanticRole


@pytest.fixture
def stream():
    return HistoryStreamIdentity(
        "stream", "authority", "PRODUCTION_LOCAL", "production", "tenant", "product", 1
    )


@pytest.fixture
def record(stream):
    return build_record(
        stream,
        1,
        NO_PREDECESSOR,
        HistoryEventIdentity("operation", "attempt", "proof"),
        {"nested": [None, True, 2**100, "\u0000", {"unicode": "żółć"}]},
    )


def record_row(record):
    return {
        "sequence": record.sequence,
        "predecessor_authenticated_digest": record.predecessor_authenticated_digest,
        "event_identity_hex": canonical_json_bytes(record.event_identity.material()).hex(),
        "event_digest": record.event_digest,
        "authenticated_digest": record.authenticated_digest,
        "canonical_record": canonical_json_bytes(record.unsigned_material()).hex(),
    }


def test_record_envelope_preserves_canonical_contract(record, stream):
    retained = _record_from_row(record_row(record), stream)
    assert retained == record
    assert retained is not record
    assert retained.stream is not stream
    assert retained.event_identity is not record.event_identity
    assert retained.canonical_event_payload["nested"][3] == "\u0000"
    with pytest.raises(TypeError):
        retained.canonical_event_payload["tamper"] = "mutable"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sequence", True),
        ("sequence", 2),
        ("predecessor_authenticated_digest", "substitution"),
        ("event_digest", "sha256:" + "0" * 64),
        ("authenticated_digest", "sha256:" + "0" * 64),
        ("event_identity_hex", b'{"forged":true}'.hex()),
        ("canonical_record", "00"),
    ],
)
def test_record_envelope_corruption_is_rejected(record, stream, field, value):
    row = {**record_row(record), field: value}
    with pytest.raises(HistoryContractError):
        _record_from_row(row, stream)


@pytest.mark.parametrize(
    "field", ["environment", "trust_domain", "product_scope", "security_epoch"]
)
def test_cross_scope_record_replay_is_rejected(record, stream, field):
    other = replace(stream, **{field: 2 if field == "security_epoch" else "other"})
    with pytest.raises(HistoryContractError, match="contradicts"):
        _record_from_row(record_row(record), other)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":1}',
        b'{"a":1.0}',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"a":-Infinity}',
        b'{ "a":1}',
        b'{"z":1,"a":2}',
        b"[]",
        b'{"a":"\\u0078"}',
        b'{"a":"\xff"}',
    ],
)
def test_noncanonical_retained_json_is_rejected(raw):
    with pytest.raises(HistoryContractError):
        _parse_canonical(raw)


@pytest.mark.parametrize("value", [None, b"ab", "AB", "a b", "a", "zz"])
def test_noncanonical_hex_envelope_is_rejected(value):
    with pytest.raises(HistoryContractError):
        _unhex(value)


def test_stored_head_copies_stream_and_binds_complete_credential(stream, record):
    credential = CredentialRoleIdentity(
        CredentialSemanticRole.HISTORY_ATTESTATION_SIGNING,
        "credential\u0000id",
        "provider",
        "version",
        "lifecycle",
        "sha256:" + "1" * 64,
    )
    material = {
        "stream": stream.material(),
        "sequence": record.sequence,
        "record_digest": record.authenticated_digest,
        "signing_credential_identity": _credential_material(credential),
    }
    row = {
        "sequence": record.sequence,
        "record_digest": record.authenticated_digest,
        "canonical_credential_hex": canonical_json_bytes(_credential_material(credential)).hex(),
        "canonical_head": (ATTESTATION_DOMAIN + canonical_json_bytes(material)).hex(),
        "signature": (b"s" * 64).hex(),
    }
    retained = _head_from_row(row, stream)
    assert retained.stream == stream
    assert retained.stream is not stream
    assert retained.signing_credential_identity == credential
    object.__setattr__(retained.stream, "product_scope", "tampered")
    assert stream.product_scope == "product"
    row["record_digest"] = "sha256:" + "0" * 64
    with pytest.raises(HistoryContractError):
        _head_from_row(row, stream)


def test_preflight_report_is_not_a_history_successor_or_expected_head(record):
    # The operation rejects foreign objects before touching database authority.
    # Runtime preflight uses a report object, never the exact HistoryRecord type.
    provider = object.__new__(PostgreSQLAuthenticatedIssuerHistory)
    provider._stream = record.stream
    report = RootProofIssuancePreflight(
        disposition="VERIFIED_NOT_AUTHORIZED_TO_ISSUE",
        security_profile="PRODUCTION_LOCAL",
        environment="PRODUCTION",
        trust_domain="tenant",
        logical_operation_id="operation",
        account_id="reserved-candidate-account",
        issuance_attempt_id="attempt",
        request_reference="request-reference",
        request_digest_sha256="1" * 64,
        initial_binding_reference="initial-binding-reference",
        initial_binding_digest_sha256="2" * 64,
        entitlement_id="entitlement",
        entitlement_generation=1,
        entitlement_registry_revision=1,
        requester_registry_revision=1,
        claimant_registry_revision=1,
        attempt_identity_digest_sha256="3" * 64,
    )
    with pytest.raises(TypeError):
        provider.append_exact_successor(report, record)
    with pytest.raises(TypeError):
        provider.append_exact_successor(None, report)
    with pytest.raises(TypeError):
        provider.append(expected_digest=NO_PREDECESSOR, event_identity=report, payload={})
    with pytest.raises(TypeError):
        provider.append(
            expected_digest=NO_PREDECESSOR,
            event_identity=record.event_identity,
            payload=report,
        )


def test_provider_cannot_be_subclassed_to_forge_qualified_authority():
    with pytest.raises(TypeError, match="cannot be subclassed"):

        class ForgedAuthority(PostgreSQLAuthenticatedIssuerHistory):
            pass
