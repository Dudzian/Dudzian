"""Public delivery tests using isolated TEST_ONLY cryptographic protocol fixtures."""

from __future__ import annotations

import hashlib

import pytest

from bot_core.licensing.canonical import canonical_json_bytes, parse_canonical
from bot_core.licensing.enrollment_transport import (
    MAX_ENVELOPE_BYTES,
    ArtifactEnvelope,
    EnrollmentTransportError,
    OfflineFileDelivery,
)
from bot_core.licensing.production_tpm_custody import parse_production_tpm_endorsement
from tests.licensing import test_pdsa_enrollment_authorization as authorization_tests

challenge_harness = authorization_tests.challenge_harness
integration = authorization_tests.integration
issuance = authorization_tests.issuance


def envelopes(item):
    args = item.arguments
    device = ArtifactEnvelope.create(
        "DEVICE_REQUEST",
        {
            "activation_request": args["activation_request_raw"],
            "tpm_request": args["tpm_request_raw"],
        },
    )
    challenge = ArtifactEnvelope.create(
        "TPM_CHALLENGE",
        device.artifacts
        | {
            "pdsa_challenge": args["challenge_raw"],
            "endorsement": args["endorsement_raw"],
            "tpm_challenge": args["tpm_challenge_raw"],
        },
        parent_id=device.request_id,
    )
    request = ArtifactEnvelope.create(
        "PRE_ENROLLMENT",
        challenge.artifacts
        | {
            "tpm_response": args["tpm_response_raw"],
            "request": args["request_raw"],
            "request_signature": args["signature"],
            "custody_evidence": args["custody_evidence_raw"],
        },
        parent_id=challenge.request_id,
    )
    return device, challenge, request


def test_exact_four_stage_delivery(issuance, tmp_path):
    stages = list(envelopes(issuance))
    package = authorization_tests._issue(issuance)
    stages.append(
        ArtifactEnvelope.create(
            "PDSA_PACKAGE", {"package": package}, parent_id=stages[-1].request_id
        )
    )
    delivery = OfflineFileDelivery(tmp_path)
    for envelope in stages:
        identity = delivery.deliver(envelope)
        assert identity == hashlib.sha256(envelope.canonical_bytes).hexdigest()
        assert delivery.receive(identity).canonical_bytes == envelope.canonical_bytes
        assert delivery.deliver(envelope) == identity
        assert envelope.preview()["authorization"] == "NOT_ESTABLISHED_BY_TRANSPORT"
    assert stages[-1].artifacts["package"] == package
    assert len(issuance.sign_calls) == 1


@pytest.mark.parametrize(
    "change",
    [
        lambda value: value.update(extra=True),
        lambda value: value.update(schema_version="ONLINE_API_UNTRUSTED_V2"),
        lambda value: value.update(kind="MEMBERSHIP"),
        lambda value: value.update(kind=[]),
        lambda value: value.update(parent_id="00" * 32),
        lambda value: value["artifacts"].update(private_key="00"),
        lambda value: value["artifacts"].update(tpm_request="00"),
        lambda value: value["artifacts"].update(tpm_request="AB"),
        lambda value: value["artifacts"].update(tpm_request="a"),
        lambda value: value["artifacts"].update(tpm_request=""),
        lambda value: value["artifacts"].update(tpm_request={}),
        lambda value: value["artifacts"].update(tpm_request="00" * 131_073),
    ],
)
def test_strict_schema_and_encoding(integration, change):
    device, _, _ = envelopes(integration)
    value = parse_canonical(device.canonical_bytes)
    change(value)
    with pytest.raises(EnrollmentTransportError):
        ArtifactEnvelope.from_canonical_bytes(canonical_json_bytes(value))


@pytest.mark.parametrize("raw", [b"{}", b"[]", b"null", b"\xff", b"x" * (MAX_ENVELOPE_BYTES + 1)])
def test_corrupt_oversize_and_wrong_document_type(raw):
    with pytest.raises(EnrollmentTransportError):
        ArtifactEnvelope.from_canonical_bytes(raw)


def test_noncanonical_inner_and_outer_documents(integration):
    device, _, _ = envelopes(integration)
    with pytest.raises(EnrollmentTransportError):
        ArtifactEnvelope.from_canonical_bytes(device.canonical_bytes + b"\n")
    value = parse_canonical(device.canonical_bytes)
    # Duplicate outer keys are rejected even when their last value agrees.
    duplicate = b'{"kind":"DEVICE_REQUEST",' + device.canonical_bytes[1:]
    with pytest.raises(EnrollmentTransportError):
        ArtifactEnvelope.from_canonical_bytes(duplicate)
    value["artifacts"]["activation_request"] += "0a"
    with pytest.raises(EnrollmentTransportError):
        ArtifactEnvelope.from_canonical_bytes(canonical_json_bytes(value))


def test_parent_mixing_and_wrong_signature_rejected(integration):
    _, challenge, request = envelopes(integration)
    with pytest.raises(EnrollmentTransportError, match="PARENT"):
        ArtifactEnvelope.create("TPM_CHALLENGE", challenge.artifacts, parent_id="00" * 32)
    with pytest.raises(EnrollmentTransportError, match="PARENT"):
        ArtifactEnvelope.create("PRE_ENROLLMENT", request.artifacts, parent_id="00" * 32)
    with pytest.raises(EnrollmentTransportError):
        ArtifactEnvelope.create(
            "PRE_ENROLLMENT",
            request.artifacts | {"request_signature": b"\x00" * 64},
            parent_id=challenge.request_id,
        )


def test_inner_unknown_field_and_test_only_projection_rejected(integration):
    device, _, _ = envelopes(integration)
    artifacts = device.artifacts
    activation = parse_canonical(artifacts["activation_request"])
    activation["private_key"] = "forbidden"
    with pytest.raises(EnrollmentTransportError):
        ArtifactEnvelope.create(
            "DEVICE_REQUEST", artifacts | {"activation_request": canonical_json_bytes(activation)}
        )
    tpm_request = parse_canonical(artifacts["tpm_request"])
    tpm_request["public_projection"]["evidence_profile"] = "TEST_ONLY"
    with pytest.raises(EnrollmentTransportError):
        ArtifactEnvelope.create(
            "DEVICE_REQUEST", artifacts | {"tpm_request": canonical_json_bytes(tpm_request)}
        )


def test_receive_rejects_digest_substitution_and_path_in_id(integration, tmp_path):
    device, _, _ = envelopes(integration)
    delivery = OfflineFileDelivery(tmp_path)
    (tmp_path / ("00" * 32 + ".json")).write_bytes(device.canonical_bytes)
    with pytest.raises(EnrollmentTransportError, match="ID_MISMATCH"):
        delivery.receive("00" * 32)
    with pytest.raises(EnrollmentTransportError, match="INVALID_DELIVERY_ID"):
        delivery.receive("../request")


def test_no_mutable_artifact_alias(integration):
    device, _, _ = envelopes(integration)
    artifact_view = device.artifacts
    artifact_view["tpm_request"] = b"corrupt"
    assert device.artifacts["tpm_request"] == integration.arguments["tpm_request_raw"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", []),
        ("environment", True),
        ("product", "Other"),
        ("product_profile", "TEST_ONLY"),
        ("pdsa_trust_domain", None),
        ("endorsement_profile", "TEST_ONLY"),
        ("hardware_origin_verification", False),
        ("signature_algorithm_profile", "Ed25519"),
        ("release_policy_generation", True),
        ("release_policy_generation", "1"),
        ("release_policy_generation", 0),
        ("release_policy_generation", -1),
        ("endorsement_id", 1),
        ("endorsement_id", "ptpm_invalid"),
        ("ek_public_digest", None),
        ("ek_public_digest", "AB" * 32),
        ("ek_certificate_digest_sha256", []),
        ("manufacturer_chain_digest_sha256", "a" * 63),
        ("manufacturer_chain_verification_record_digest_sha256", "gg" * 32),
        ("release_policy_digest_sha256", 1),
        ("ek_name", "000b" + "00" * 32),
        ("issued_at_utc", 1),
        ("issued_at_utc", "2026-10-06T12:00:00+00:00"),
        ("expires_at_utc", "2026-10-06T12:00:00Z"),
        ("expires_at_utc", "2026-10-14T12:00:00Z"),
        ("signer_key_ids", "signer"),
        ("signer_key_ids", [True, 1]),
        ("signer_key_ids", ["TEST_ONLY_CHALLENGE_1", "TEST_ONLY_CHALLENGE_1"]),
        ("signer_key_ids", ["TEST_ONLY_CHALLENGE_2", "TEST_ONLY_CHALLENGE_1"]),
        ("signer_key_ids", ["", "TEST_ONLY_CHALLENGE_2"]),
        ("signer_key_ids", ["a" * 129, "b" * 129]),
    ],
)
def test_endorsement_primitives_profiles_and_bindings_are_strict(integration, field, value):
    device, challenge, _ = envelopes(integration)
    endorsement = parse_canonical(challenge.artifacts["endorsement"])
    endorsement["payload"][field] = value
    # Digest consistency cannot make malformed public fields valid.
    endorsement["payload_digest_sha256"] = hashlib.sha256(
        canonical_json_bytes(endorsement["payload"])
    ).hexdigest()
    with pytest.raises(EnrollmentTransportError):
        ArtifactEnvelope.create(
            "TPM_CHALLENGE",
            challenge.artifacts | {"endorsement": canonical_json_bytes(endorsement)},
            parent_id=device.request_id,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("algorithm", True),
        ("algorithm", "ed25519"),
        ("key_id", []),
        ("key_id", "CALLER_SELECTED_KEY"),
        ("authority_key_set_digest", 1),
        ("authority_key_set_digest", "00" * 32),
        ("signature_hex", None),
        ("signature_hex", ""),
        ("signature_hex", "00"),
        ("signature_hex", "AA" * 64),
        ("signature_hex", "a" * 127),
        ("signature_hex", "gg" * 64),
    ],
)
def test_endorsement_signer_records_are_strict(integration, field, value):
    device, challenge, _ = envelopes(integration)
    endorsement = parse_canonical(challenge.artifacts["endorsement"])
    endorsement["signatures"][0][field] = value
    with pytest.raises(EnrollmentTransportError):
        ArtifactEnvelope.create(
            "TPM_CHALLENGE",
            challenge.artifacts | {"endorsement": canonical_json_bytes(endorsement)},
            parent_id=device.request_id,
        )


@pytest.mark.parametrize("location", ["envelope", "payload", "signature"])
def test_endorsement_unknown_fields_rejected_at_every_level(integration, location):
    device, challenge, _ = envelopes(integration)
    endorsement = parse_canonical(challenge.artifacts["endorsement"])
    target = {
        "envelope": endorsement,
        "payload": endorsement["payload"],
        "signature": endorsement["signatures"][0],
    }[location]
    target["unknown"] = True
    endorsement["payload_digest_sha256"] = hashlib.sha256(
        canonical_json_bytes(endorsement["payload"])
    ).hexdigest()
    with pytest.raises(EnrollmentTransportError):
        ArtifactEnvelope.create(
            "TPM_CHALLENGE",
            challenge.artifacts | {"endorsement": canonical_json_bytes(endorsement)},
            parent_id=device.request_id,
        )


def test_endorsement_public_parser_does_not_grant_current_authority(integration, monkeypatch):
    import bot_core.licensing.production_tpm_custody as custody

    document = parse_canonical(integration.endorsement_raw)
    document["payload"].update(
        release_policy_generation=2,
        release_policy_digest_sha256="ff" * 32,
        issued_at_utc="2020-01-01T12:00:00Z",
        expires_at_utc="2020-01-02T12:00:00Z",
    )
    document["payload_digest_sha256"] = hashlib.sha256(
        canonical_json_bytes(document["payload"])
    ).hexdigest()

    def forbidden(*args, **kwargs):
        raise AssertionError("public syntax must not consult production trust or current time")

    monkeypatch.setattr(custody, "require_verified_production_trust_context", forbidden)
    monkeypatch.setattr(custody, "_utc_now", forbidden)
    assert parse_production_tpm_endorsement(canonical_json_bytes(document)) == document


@pytest.mark.parametrize("generation", [b"1.0", b"9007199254740992"])
def test_endorsement_noncanonical_integer_representation_rejected(integration, generation):
    device, challenge, _ = envelopes(integration)
    raw = challenge.artifacts["endorsement"]
    assert b'"release_policy_generation":1' in raw
    raw = raw.replace(
        b'"release_policy_generation":1', b'"release_policy_generation":' + generation
    )
    with pytest.raises(EnrollmentTransportError):
        ArtifactEnvelope.create(
            "TPM_CHALLENGE",
            challenge.artifacts | {"endorsement": raw},
            parent_id=device.request_id,
        )


def test_endorsement_excessive_json_nesting_is_a_public_parse_rejection():
    raw = b'{"payload":' + b"[" * 1100 + b"0" + b"]" * 1100 + b"}"
    with pytest.raises(ValueError, match="NONCANONICAL_PRODUCTION_TPM_ARTIFACT"):
        parse_production_tpm_endorsement(raw)


@pytest.mark.parametrize("version", [b"true", b"1.0"])
def test_embedded_activation_version_requires_exact_integer(integration, version):
    device, _, _ = envelopes(integration)
    activation_raw = device.artifacts["activation_request"]
    assert b'"version":1' in activation_raw
    activation_raw = activation_raw.replace(b'"version":1', b'"version":' + version)
    envelope = parse_canonical(device.canonical_bytes)
    envelope["artifacts"]["activation_request"] = activation_raw.hex()
    with pytest.raises(EnrollmentTransportError):
        ArtifactEnvelope.from_canonical_bytes(canonical_json_bytes(envelope))
