"""TEST_ONLY software fixtures exercise payload mechanics, never TPM authority."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa, utils

from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.licensing.device_enrollment import build_activation_request, make_evidence
from bot_core.licensing.pre_enrollment import (
    ALGORITHM_PROFILE,
    P256_ORDER,
    PAYLOAD_FIELDS,
    PDSA_TRUST_DOMAIN,
    SIGNATURE_DOMAIN,
    PreEnrollmentError,
    PreEnrollmentRequestV1,
    public_key_fingerprint,
    validate_public_key,
)
from bot_core.licensing.tpm_attestation import (
    EXCHANGE_REFERENCE_DOMAIN,
    TPMEnrollmentChallengeResponseV1,
    TPMEnrollmentChallengeV1,
    TPMEnrollmentRequestV1,
)
from deployment.windows_stage9_production_trust import (
    ProductionTrustContext,
    ProductionTrustUnavailable,
    load_production_trust,
)

ROOT = Path(__file__).resolve().parents[2]


def test_only_key() -> ec.EllipticCurvePrivateKey:
    """Software key is a TEST_ONLY fixture, not a production key factory."""
    return ec.derive_private_key(1, ec.SECP256R1())


test_only_key.__test__ = False


def test_only_payload() -> dict:
    public = (
        test_only_key()
        .public_key()
        .public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    )
    payload = {field: "ab" * 32 for field in PAYLOAD_FIELDS}
    payload.update(
        schema_version="PreEnrollmentRequestV1",
        environment="PRODUCTION",
        product="CryptoHunter",
        product_profile="CryptoHunter",
        pdsa_trust_domain=PDSA_TRUST_DOMAIN,
        pdsa_challenge_id="pchal_01930000-0000-7000-8000-000000000001",
        pre_enrollment_public_key_algorithm_profile=ALGORITHM_PROFILE,
        pre_enrollment_public_key_canonical_bytes=public.hex(),
        pre_enrollment_public_key_fingerprint_sha256=hashlib.sha256(public).hexdigest(),
        release_policy_generation=1,
    )
    return payload


test_only_payload.__test__ = False


def low_s_signature(request: PreEnrollmentRequestV1, scalar: int = 1) -> bytes:
    key = ec.derive_private_key(scalar, ec.SECP256R1())
    r, s = utils.decode_dss_signature(key.sign(request.signing_bytes, ec.ECDSA(hashes.SHA256())))
    return utils.encode_dss_signature(r, min(s, P256_ORDER - s))


def test_exact_frozen_fields_and_stable_canonical_roundtrip():
    frozen = json.loads(
        (
            ROOT / "docs/architecture/cryptohunter_product_architecture/"
            "stage9_external_provisioning_architecture_contract.json"
        ).read_bytes()
    )
    assert PAYLOAD_FIELDS == set(frozen["pre_enrollment_request"]["canonical_payload_fields"])
    assert len(PAYLOAD_FIELDS) == 22
    value = test_only_payload()
    request = PreEnrollmentRequestV1.from_mapping(value)
    reverse_order = dict(reversed(list(value.items())))
    assert (
        request.canonical_bytes
        == PreEnrollmentRequestV1.from_mapping(reverse_order).canonical_bytes
    )
    assert request.canonical_bytes == canonical_json_bytes(value)
    assert PreEnrollmentRequestV1.from_canonical_bytes(request.canonical_bytes) == request
    assert request.digest_sha256 == hashlib.sha256(request.canonical_bytes).hexdigest()
    assert request.signing_bytes == SIGNATURE_DOMAIN + bytes.fromhex(request.digest_sha256)
    request.document["product"] = "mutable copy"
    value["product"] = "mutable source"
    assert request.document["product"] == "CryptoHunter"
    with pytest.raises(FrozenInstanceError):
        request.canonical_bytes = b"changed"
    with pytest.raises(TypeError):
        PreEnrollmentRequestV1()


@pytest.mark.parametrize("field", sorted(PAYLOAD_FIELDS))
def test_every_missing_field_is_rejected(field):
    value = test_only_payload()
    value.pop(field)
    with pytest.raises(PreEnrollmentError, match="SCHEMA_MISMATCH"):
        PreEnrollmentRequestV1.from_mapping(value)


@pytest.mark.parametrize("field", ["unknown", "issued_at_utc", "private_blob", "signature"])
def test_no_extra_fields_or_private_material(field):
    value = {**test_only_payload(), field: "forbidden"}
    with pytest.raises(PreEnrollmentError, match="SCHEMA_MISMATCH"):
        PreEnrollmentRequestV1.from_mapping(value)


@pytest.mark.parametrize("value", [None, [], "text", 1])
def test_mapping_constructor_rejects_non_mappings(value):
    with pytest.raises(PreEnrollmentError, match="SCHEMA_MISMATCH"):
        PreEnrollmentRequestV1.from_mapping(value)


@pytest.mark.parametrize("value", [None, True, 1, b"04" + b"00" * 64])
def test_public_key_wire_field_requires_hex_string(value):
    with pytest.raises(PreEnrollmentError, match="PUBLIC_KEY"):
        PreEnrollmentRequestV1.from_mapping(
            {**test_only_payload(), "pre_enrollment_public_key_canonical_bytes": value}
        )


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("schema_version", "PreEnrollmentRequestV2"),
        ("schema_version", 1),
        ("environment", "TEST_ONLY"),
        ("environment", "production"),
        ("product", "OtherProduct"),
        ("product_profile", "TEST_ONLY"),
        ("pdsa_trust_domain", "TEST_ONLY_2_OF_3_ED25519"),
        ("pdsa_trust_domain", "caller-selected"),
        ("pre_enrollment_public_key_algorithm_profile", "Ed25519"),
        ("pre_enrollment_public_key_algorithm_profile", "ECDSA-P384-SHA256"),
        ("pre_enrollment_public_key_algorithm_profile", "RSA"),
        ("release_policy_generation", True),
        ("release_policy_generation", False),
        ("release_policy_generation", 0),
        ("release_policy_generation", -1),
        ("release_policy_generation", 1.0),
        ("release_policy_generation", "1"),
        ("release_policy_generation", 9_007_199_254_740_992),
        ("pdsa_challenge_id", "pchal_01930000-0000-4000-8000-000000000001"),
        ("pdsa_challenge_id", "pchal_01930000-0000-7000-0000-000000000001"),
        ("pdsa_challenge_id", "PCHAL_01930000-0000-7000-8000-000000000001"),
        ("pdsa_challenge_id", None),
    ],
)
def test_wrong_profile_schema_and_types_are_rejected(field, wrong):
    value = {**test_only_payload(), field: wrong}
    with pytest.raises(PreEnrollmentError):
        PreEnrollmentRequestV1.from_mapping(value)


@pytest.mark.parametrize(
    "field",
    sorted(
        PAYLOAD_FIELDS
        - {
            "schema_version",
            "environment",
            "product",
            "product_profile",
            "pdsa_trust_domain",
            "pdsa_challenge_id",
            "pre_enrollment_public_key_algorithm_profile",
            "pre_enrollment_public_key_canonical_bytes",
            "release_policy_generation",
        }
    ),
)
@pytest.mark.parametrize("wrong", ["AB" * 32, "ab" * 31, "zz" * 32, "ab" * 32 + "\n", None, 1])
def test_digest_nonce_and_reference_formats_are_strict(field, wrong):
    with pytest.raises(PreEnrollmentError):
        PreEnrollmentRequestV1.from_mapping({**test_only_payload(), field: wrong})


@pytest.mark.parametrize(
    "raw", [b"[]", b"null", b"{}\n", b"not JSON", b"\xff", "{}", b" " * 16_385]
)
def test_malformed_or_non_object_canonical_input_rejected(raw):
    with pytest.raises(PreEnrollmentError):
        PreEnrollmentRequestV1.from_canonical_bytes(raw)


def test_whitespace_order_duplicate_keys_and_noncanonical_json_rejected():
    request = PreEnrollmentRequestV1.from_mapping(test_only_payload())
    duplicate = request.canonical_bytes.replace(
        b'"environment":"PRODUCTION"', b'"environment":"TEST_ONLY","environment":"PRODUCTION"'
    )
    for raw in (
        request.canonical_bytes + b"\n",
        json.dumps(request.document, indent=2).encode(),
        json.dumps(dict(reversed(list(request.document.items()))), separators=(",", ":")).encode(),
        duplicate,
    ):
        with pytest.raises(PreEnrollmentError, match="NONCANONICAL"):
            PreEnrollmentRequestV1.from_canonical_bytes(raw)


def test_public_key_roundtrip_exact_sec1_fingerprint():
    raw = bytes.fromhex(test_only_payload()["pre_enrollment_public_key_canonical_bytes"])
    loaded = validate_public_key(raw)
    assert loaded.curve.name == "secp256r1"
    assert (
        loaded.public_bytes(
            serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
        )
        == raw
    )
    assert public_key_fingerprint(raw) == hashlib.sha256(raw).hexdigest()


def test_alternative_encodings_algorithms_curves_and_invalid_points_are_rejected():
    p256 = test_only_key().public_key()
    p384 = ec.generate_private_key(ec.SECP384R1()).public_key()
    raw = bytes.fromhex(test_only_payload()["pre_enrollment_public_key_canonical_bytes"])
    alternatives = (
        p256.public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        ),
        p256.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.CompressedPoint),
        p384.public_bytes(
            serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
        ),
        ed25519.Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw),
        rsa.generate_private_key(65537, 2048)
        .public_key()
        .public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo),
        bytes([3]) + raw[1:],
        b"\x04" + bytes(64),
        raw[:-1],
        bytearray(raw),
    )
    for invalid in alternatives:
        with pytest.raises(PreEnrollmentError, match="PUBLIC_KEY"):
            public_key_fingerprint(invalid)


@pytest.mark.parametrize(
    "case", ["uppercase", "fingerprint", "altered_key", "wrong_prefix", "off_curve"]
)
def test_request_key_encoding_or_fingerprint_tampering_rejected(case):
    value = test_only_payload()
    if case == "uppercase":
        value["pre_enrollment_public_key_canonical_bytes"] = value[
            "pre_enrollment_public_key_canonical_bytes"
        ].upper()
    elif case == "fingerprint":
        value["pre_enrollment_public_key_fingerprint_sha256"] = "00" * 32
    elif case == "altered_key":
        changed = (
            ec.derive_private_key(2, ec.SECP256R1())
            .public_key()
            .public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
        )
        value["pre_enrollment_public_key_canonical_bytes"] = changed.hex()
    elif case == "wrong_prefix":
        value["pre_enrollment_public_key_canonical_bytes"] = (
            "03" + value["pre_enrollment_public_key_canonical_bytes"][2:]
        )
    else:
        value["pre_enrollment_public_key_canonical_bytes"] = "04" + "00" * 64
    with pytest.raises(PreEnrollmentError):
        PreEnrollmentRequestV1.from_mapping(value)


def test_request_signature_verifies_only_exact_key_and_domain_payload():
    request = PreEnrollmentRequestV1.from_mapping(test_only_payload())
    signature = low_s_signature(request)
    request.verify_signature(signature)
    with pytest.raises(PreEnrollmentError, match="SIGNATURE"):
        request.verify_signature(low_s_signature(request, scalar=2))
    changed = PreEnrollmentRequestV1.from_mapping(
        {**request.document, "request_nonce_hex": "cd" * 32}
    )
    with pytest.raises(PreEnrollmentError, match="SIGNATURE"):
        changed.verify_signature(signature)
    wrong_domain = test_only_key().sign(
        bytes.fromhex(request.digest_sha256), ec.ECDSA(hashes.SHA256())
    )
    r, s = utils.decode_dss_signature(wrong_domain)
    with pytest.raises(PreEnrollmentError, match="SIGNATURE"):
        request.verify_signature(utils.encode_dss_signature(r, min(s, P256_ORDER - s)))


def test_nonminimal_der_high_s_trailing_bytes_and_invalid_scalars_rejected():
    request = PreEnrollmentRequestV1.from_mapping(test_only_payload())
    valid = low_s_signature(request)
    r, s = utils.decode_dss_signature(valid)
    # INTEGER 1 encoded with a redundant zero is nonminimal DER.
    nonminimal = bytes.fromhex("300702020001020101")
    for invalid in (
        valid + b"\x00",
        valid[:-1],
        nonminimal,
        utils.encode_dss_signature(r, P256_ORDER - s),
        utils.encode_dss_signature(0, 1),
        utils.encode_dss_signature(P256_ORDER, 1),
        utils.encode_dss_signature(1, 0),
        utils.encode_dss_signature(1, P256_ORDER),
        b"",
        "not bytes",
        bytearray(valid),
    ):
        with pytest.raises(PreEnrollmentError, match="SIGNATURE"):
            request.verify_signature(invalid)


@pytest.mark.parametrize("case", ["fingerprint", "environment", "unknown", "noncanonical"])
def test_forged_or_mutated_model_cannot_bypass_consequential_validation(case):
    original = PreEnrollmentRequestV1.from_mapping(test_only_payload())
    valid_signature = low_s_signature(original)
    altered = original.document
    if case == "fingerprint":
        altered["pre_enrollment_public_key_fingerprint_sha256"] = "00" * 32
    elif case == "environment":
        altered["environment"] = "TEST_ONLY"
    elif case == "unknown":
        altered["private_blob"] = "forbidden"
    forged_bytes = canonical_json_bytes(altered)
    if case == "noncanonical":
        forged_bytes += b"\n"
    forged = object.__new__(PreEnrollmentRequestV1)
    object.__setattr__(forged, "canonical_bytes", forged_bytes)
    object.__setattr__(original, "canonical_bytes", forged_bytes)
    for invalid in (forged, original):
        with pytest.raises(PreEnrollmentError):
            invalid.verify_signature(valid_signature)
        with pytest.raises(PreEnrollmentError):
            _ = invalid.document
        with pytest.raises(PreEnrollmentError):
            _ = invalid.signing_bytes


def test_arbitrary_and_forged_production_trust_objects_are_rejected():
    request = PreEnrollmentRequestV1.from_mapping(test_only_payload())
    arbitrary = SimpleNamespace(
        release_payload_digest=request.document["release_policy_digest_sha256"], release_version=1
    )
    forged = object.__new__(ProductionTrustContext)
    object.__setattr__(forged, "release_payload_digest", arbitrary.release_payload_digest)
    object.__setattr__(forged, "release_version", 1)
    for invalid in (None, object(), arbitrary, forged):
        with pytest.raises(
            ProductionTrustUnavailable, match="VERIFIED_PRODUCTION_TRUST_CONTEXT_REQUIRED"
        ):
            request.require_production_trust_binding(invalid)


def test_real_public_production_trust_exact_release_binding_when_available():
    supplied = os.environ.get("CRYPTOHUNTER_STAGE9_FINAL_PACKAGE")
    if not supplied:
        pytest.skip("operator-supplied verified public production package required")
    context = load_production_trust(Path(supplied))
    payload = test_only_payload()
    payload.update(
        release_policy_digest_sha256=context.release_payload_digest,
        release_policy_generation=context.release_version,
    )
    PreEnrollmentRequestV1.from_mapping(payload).require_production_trust_binding(context)
    for field, changed in (
        ("release_policy_digest_sha256", "00" * 32),
        ("release_policy_generation", context.release_version + 1),
    ):
        with pytest.raises(PreEnrollmentError, match="TRUST_BINDING_MISMATCH"):
            PreEnrollmentRequestV1.from_mapping(
                {**payload, field: changed}
            ).require_production_trust_binding(context)


def test_only_exchange():
    """Fabricated bytes are valid for comparison only; signatures are deliberately fake."""
    fixture = json.loads(
        (ROOT / "tests/fixtures/windows_stage9_policy_vector_v1.json").read_bytes()
    )
    key = fixture["enrollment_policy_material"]["k_psa"]
    projection = make_evidence(
        public_area_hex=key["public_hex"],
        returned_name=fixture["policy_vector"]["k_psa_name"],
        creation_hash="77" * 32,
        ek_public_area_hex=key["public_hex"],
        ek_name=fixture["policy_vector"]["k_psa_name"],
        ak_public_area_hex=key["public_hex"],
        ak_name=fixture["policy_vector"]["k_psa_name"],
        algorithm_profile="Stage9.K_PSA.ECC_P256_SHA256.TEST_ONLY",
        evidence_profile="WindowsTPM2-TBS-Stage9-v1",
        substrate_profile="TEST_ONLY-software-fabricated-binding-test",
    )
    activation = build_activation_request(
        evidence=projection,
        release_policy_digest="ab" * 32,
        release_policy_version=1,
        requested_entitlements={
            "product": "CryptoHunter",
            "edition": "pro",
            "requested_features": [],
        },
        installation_id="TEST_ONLY-installation",
        architecture="AMD64",
        environment="TEST_ONLY",
        created_at_utc="2026-01-01T00:00:00Z",
        nonce="44" * 32,
    )
    request = TPMEnrollmentRequestV1.create(
        activation_request=activation, public_projection=projection, client_nonce=b"\x11" * 32
    )
    challenge = TPMEnrollmentChallengeV1.create(
        request,
        expires_at_utc="2099-01-01T00:00:00Z",
        credential_blob_hex="01",
        encrypted_secret_hex="02",
    )
    response = TPMEnrollmentChallengeResponseV1.create(
        challenge,
        activated_credential_digest="03",
        credential_activation_proof_hex="04",
        certify_creation_attest_hex="05",
        certify_creation_signature_hex="06",
        k_psa_pop_signature_der_hex="07",
    )
    raw = tuple(item.canonical_bytes for item in (activation, request, challenge, response))
    digests = tuple(hashlib.sha256(item).hexdigest() for item in raw)
    payload = test_only_payload()
    payload.update(
        tpm_enrollment_request_digest_sha256=digests[1],
        tpm_enrollment_challenge_digest_sha256=digests[2],
        tpm_enrollment_response_digest_sha256=digests[3],
        verified_tpm_exchange_reference=hashlib.sha256(
            EXCHANGE_REFERENCE_DOMAIN + b"".join(bytes.fromhex(item) for item in digests)
        ).hexdigest(),
        verified_tpm_public_projection_id=projection.evidence_reference,
        ek_public_digest=projection.document["ek"]["public_digest"],
        ak_public_digest=projection.document["ak"]["public_digest"],
        tpm_attestation_evidence_reference=projection.evidence_reference,
    )
    parameters = dict(
        zip(
            ("activation_request_raw", "request_raw", "challenge_raw", "response_raw"),
            raw,
            strict=True,
        )
    )
    return payload, parameters


test_only_exchange.__test__ = False


def test_exact_tpm_exchange_binding_comparison_is_not_authentication():
    payload, parameters = test_only_exchange()
    request = PreEnrollmentRequestV1.from_mapping(payload)
    assert request.compare_tpm_exchange_bindings(**parameters) is None
    # Deliberately invalid attestation signature bytes were accepted only for binding comparison.
    response = json.loads(parameters["response_raw"])
    assert response["certify_creation_signature_hex"] == "06"


@pytest.mark.parametrize(
    "field",
    [
        "tpm_enrollment_request_digest_sha256",
        "tpm_enrollment_challenge_digest_sha256",
        "tpm_enrollment_response_digest_sha256",
        "verified_tpm_exchange_reference",
        "verified_tpm_public_projection_id",
        "ek_public_digest",
        "ak_public_digest",
        "tpm_attestation_evidence_reference",
        "release_policy_digest_sha256",
        "release_policy_generation",
    ],
)
def test_altered_tpm_device_release_digest_bindings_fail(field):
    payload, parameters = test_only_exchange()
    wrong = 2 if field == "release_policy_generation" else "00" * 32
    request = PreEnrollmentRequestV1.from_mapping({**payload, field: wrong})
    with pytest.raises(PreEnrollmentError, match="EXCHANGE_BINDING_MISMATCH"):
        request.compare_tpm_exchange_bindings(**parameters)


@pytest.mark.parametrize(
    "raw_name", ["activation_request_raw", "request_raw", "challenge_raw", "response_raw"]
)
def test_noncanonical_or_altered_retained_exchange_bytes_fail(raw_name):
    payload, parameters = test_only_exchange()
    request = PreEnrollmentRequestV1.from_mapping(payload)
    parameters[raw_name] += b"\n"
    with pytest.raises(PreEnrollmentError, match="INVALID_TPM_EXCHANGE_BYTES"):
        request.compare_tpm_exchange_bindings(**parameters)


def test_altered_response_is_not_accepted_despite_recomputed_digest_and_exchange_reference():
    payload, parameters = test_only_exchange()
    response = json.loads(parameters["response_raw"])
    response["issuer_nonce_hex"] = "00" * 32
    parameters["response_raw"] = canonical_json_bytes(response)
    digests = [hashlib.sha256(raw).hexdigest() for raw in parameters.values()]
    payload.update(
        tpm_enrollment_response_digest_sha256=digests[3],
        verified_tpm_exchange_reference=hashlib.sha256(
            EXCHANGE_REFERENCE_DOMAIN + b"".join(bytes.fromhex(item) for item in digests)
        ).hexdigest(),
    )
    with pytest.raises(PreEnrollmentError, match="EXCHANGE_BINDING_MISMATCH"):
        PreEnrollmentRequestV1.from_mapping(payload).compare_tpm_exchange_bindings(**parameters)
