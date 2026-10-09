from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils

from bot_core.licensing.activation_request import EnrollmentDecisionV1
from bot_core.licensing.authority import (
    AttestedEnrollmentIssuer,
    TestOnlyAttestedEnrollmentIssuer,
    TestOnlyPDSAAuthority,
)
from bot_core.licensing.device_enrollment import build_activation_request, make_evidence
from bot_core.licensing.canonical import digest
from bot_core.licensing.tpm_attestation import (
    PendingChallengeStore,
    ProductionTPMAttestationVerifier,
    TPMEnrollmentChallengeResponseV1,
    TPMEnrollmentChallengeV1,
    TPMEnrollmentRequestV1,
    TestOnlyTPMAttestationVerifier,
    certify_qualifying_data,
    create_issuer_challenge,
    create_test_only_issuer_challenge,
    credential_activation_proof,
    make_credential_ecc,
    proof_of_possession_digest,
    test_only_sign_policy_digest as sign_policy_digest,
    verify_k_psa_proof_of_possession,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/windows_stage9_policy_vector_v1.json"
RELEASE = "aa" * 32


def projection():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    key = fixture["enrollment_policy_material"]["k_psa"]
    return make_evidence(
        public_area_hex=key["public_hex"],
        returned_name=fixture["policy_vector"]["k_psa_name"],
        creation_hash="77" * 32,
        ek_public_area_hex=key["public_hex"],
        ek_name=fixture["policy_vector"]["k_psa_name"],
        ak_public_area_hex=key["public_hex"],
        ak_name=fixture["policy_vector"]["k_psa_name"],
        algorithm_profile="Stage9.K_PSA.ECC_P256_SHA256.TEST_ONLY",
        evidence_profile="WindowsTPM2-TBS-Stage9-v1",
        substrate_profile="software-fabricated",
    )


def signature(nonce: bytes, scalar: int = 1) -> bytes:
    return ec.derive_private_key(scalar, ec.SECP256R1()).sign(
        proof_of_possession_digest(nonce), ec.ECDSA(utils.Prehashed(hashes.SHA256()))
    )


def activation(public=None, **changes):
    public = public or projection()
    values = {
        "installation_id": "installation-a",
        "release_policy_digest": RELEASE,
        "requested_entitlements": {
            "product": "CryptoHunter",
            "edition": "pro",
            "requested_features": ["core_bot"],
        },
    }
    values.update(changes)
    return build_activation_request(
        evidence=public,
        release_policy_digest=values["release_policy_digest"],
        release_policy_version=1,
        requested_entitlements=values["requested_entitlements"],
        installation_id=values["installation_id"],
        architecture="AMD64",
        environment="TEST_ONLY",
        created_at_utc="2026-01-01T00:00:00Z",
        nonce="44" * 32,
    )


def exchange(candidate=None):
    candidate = candidate or activation()
    request = TPMEnrollmentRequestV1.create(
        activation_request=candidate,
        public_projection=projection(),
        client_nonce=b"\x10" * 32,
    )
    pending = PendingChallengeStore()
    credential_secret = b"\x33" * 32
    challenge = create_test_only_issuer_challenge(
        request,
        pending,
        expires_at_utc="2099-01-01T00:00:00Z",
        credential_secret=credential_secret,
        ephemeral_scalar=2,
    )
    issuer_nonce = bytes.fromhex(challenge.document["issuer_nonce_hex"])
    qualifying = certify_qualifying_data(
        issuer_nonce, hashlib.sha256(candidate.canonical_bytes).digest()
    )
    public = request.document["public_projection"]
    attest = (
        bytes.fromhex("ff544347801a")
        + b"\x00\x00"
        + len(qualifying).to_bytes(2, "big")
        + qualifying
        + bytes(25)
        + len(bytes.fromhex(public["k_psa"]["name"])).to_bytes(2, "big")
        + bytes.fromhex(public["k_psa"]["name"])
        + len(bytes.fromhex(public["k_psa"]["creation_hash"])).to_bytes(2, "big")
        + bytes.fromhex(public["k_psa"]["creation_hash"])
    )
    ak_der = ec.derive_private_key(1, ec.SECP256R1()).sign(
        hashlib.sha256(attest).digest(), ec.ECDSA(utils.Prehashed(hashes.SHA256()))
    )
    r, s = utils.decode_dss_signature(ak_der)
    r_raw, s_raw = r.to_bytes(32, "big"), s.to_bytes(32, "big")
    ak_signature = bytes.fromhex("0018000b") + b"\x00\x20" + r_raw + b"\x00\x20" + s_raw
    response = TPMEnrollmentChallengeResponseV1.create(
        challenge,
        activated_credential_digest=hashlib.sha256(credential_secret).hexdigest(),
        credential_activation_proof_hex=credential_activation_proof(
            credential_secret, challenge.document["challenge_id"]
        ).hex(),
        certify_creation_attest_hex=attest.hex(),
        certify_creation_signature_hex=ak_signature.hex(),
        k_psa_pop_signature_der_hex=signature(
            bytes.fromhex(challenge.document["issuer_nonce_hex"])
        ).hex(),
    )
    return candidate, request, challenge, response, pending


def decision(candidate):
    item = candidate.document
    return EnrollmentDecisionV1.from_mapping(
        {
            "schema": "EnrollmentDecisionV1",
            "version": 1,
            "request_id": item["request_id"],
            "license_id": "license-a",
            "product": "CryptoHunter",
            "edition": "pro",
            "features": ["core_bot"],
            "issued_at": "2026-01-01T00:00:00Z",
            "expires_at": None,
            "renewal_after": None,
            "approval_status": "APPROVED",
            "operator_note": None,
        }
    )


def mutate_activation(candidate, mutation):
    changed = deepcopy(candidate.document)
    mutation(changed)
    identity = deepcopy(changed)
    identity.pop("request_id")
    changed["request_id"] = digest(identity)
    from bot_core.licensing.activation_request import ActivationRequestV1

    return ActivationRequestV1.from_mapping(changed)


def test_test_only_policy_and_pop_are_preserved():
    expected = hashlib.sha256(bytes(32) + bytes.fromhex("0000016c0000015d")).digest()
    assert sign_policy_digest() == expected
    verify_k_psa_proof_of_possession(projection(), b"\x22" * 32, signature(b"\x22" * 32))


def test_make_credential_deterministic_vector_and_ak_name_binding():
    public = projection().document
    kwargs = {
        "credential_secret": b"\x33" * 32,
        "test_only_ephemeral_scalar": 2,
    }
    first = make_credential_ecc(
        public["ek"]["public_area"]["hex"], bytes.fromhex(public["ak"]["name"]), **kwargs
    )
    second = make_credential_ecc(
        public["ek"]["public_area"]["hex"], bytes.fromhex(public["ak"]["name"]), **kwargs
    )
    wrong_name = make_credential_ecc(
        public["ek"]["public_area"]["hex"], b"\x00\x0b" + b"\x99" * 32, **kwargs
    )
    assert first == second
    assert first.credential_blob != wrong_name.credential_blob
    assert first.credential_secret == b"\x33" * 32


@pytest.mark.parametrize("case", ["wrong_nonce", "wrong_key", "tampered", "ak_key"])
def test_negative_proof_of_possession(case: str):
    nonce = b"\x22" * 32
    pop = signature(nonce, 2 if case in {"wrong_key", "ak_key"} else 1)
    checked_nonce = b"\x23" * 32 if case == "wrong_nonce" else nonce
    if case == "tampered":
        pop = pop[:-1] + bytes([pop[-1] ^ 1])
    with pytest.raises(ValueError, match="K_PSA_PROOF_OF_POSSESSION_FAILED"):
        verify_k_psa_proof_of_possession(projection(), checked_nonce, pop)


def test_canonical_request_challenge_response_bindings_and_transport_roundtrip():
    candidate, request, challenge, response, _ = exchange()
    assert (
        TPMEnrollmentRequestV1.from_canonical_bytes(request.canonical_bytes).canonical_bytes
        == request.canonical_bytes
    )
    assert (
        TPMEnrollmentChallengeV1.from_canonical_bytes(challenge.canonical_bytes).document[
            "request_id"
        ]
        == request.document["request_id"]
    )
    assert (
        TPMEnrollmentChallengeResponseV1.from_canonical_bytes(response.canonical_bytes).document[
            "challenge_id"
        ]
        == challenge.document["challenge_id"]
    )


def test_issuer_challenge_is_bound_to_projection_and_random_by_default():
    candidate, request, _, _, _ = exchange()
    first = create_issuer_challenge(
        request, PendingChallengeStore(), expires_at_utc="2099-01-01T00:00:00Z"
    )
    second = create_issuer_challenge(
        request, PendingChallengeStore(), expires_at_utc="2099-01-01T00:00:00Z"
    )
    assert first.document["issuer_nonce_hex"] != second.document["issuer_nonce_hex"]
    assert first.document["public_projection_id"] == projection().evidence_reference


def test_test_only_harness_verifies_without_consuming_then_store_rejects_replay():
    candidate, request, challenge, response, pending = exchange()
    harness = TestOnlyTPMAttestationVerifier()
    verified = harness.verify_for_tests(
        candidate.canonical_bytes,
        request.canonical_bytes,
        challenge.canonical_bytes,
        response.canonical_bytes,
        pending=pending,
        expected_release_policy_digest=RELEASE,
    )
    assert len(verified.exchange_reference) == 64
    assert (
        verified.activation_request_digest == hashlib.sha256(candidate.canonical_bytes).hexdigest()
    )
    pending.require_pending(challenge)
    pending.consume(challenge)
    with pytest.raises(ValueError, match="REPLAY"):
        pending.require_pending(challenge)


def test_concrete_issuer_has_no_arbitrary_verifier_callback_and_verifies_exchange():
    candidate, request, challenge, response, pending = exchange()
    verifier = ProductionTPMAttestationVerifier()
    with pytest.raises(TypeError):
        verifier.verify(
            candidate.canonical_bytes,
            request.canonical_bytes,
            challenge.canonical_bytes,
            response.canonical_bytes,
            pending=pending,
            expected_release_policy_digest=RELEASE,
            hardware_chain_verifier=lambda *_: True,
        )
    verified = verifier.verify(
        candidate.canonical_bytes,
        request.canonical_bytes,
        challenge.canonical_bytes,
        response.canonical_bytes,
        pending=pending,
        expected_release_policy_digest=RELEASE,
    )
    assert len(verified.exchange_reference) == 64
    pending.require_pending(challenge)


def test_exchange_rejects_changed_response_binding_before_hardware_verification():
    candidate, request, challenge, response, pending = exchange()
    changed = response.document
    changed["request_id"] = "bb" * 32
    changed_raw = json.dumps(changed, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="challenge/response binding mismatch"):
        ProductionTPMAttestationVerifier().verify(
            candidate.canonical_bytes,
            request.canonical_bytes,
            challenge.canonical_bytes,
            changed_raw,
            pending=pending,
            expected_release_policy_digest=RELEASE,
        )


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("activated_credential_digest", lambda value: "00" * 32),
        ("credential_activation_proof_hex", lambda value: "00" * 32),
        (
            "k_psa_pop_signature_der_hex",
            lambda value: value[:-2] + f"{int(value[-2:], 16) ^ 1:02x}",
        ),
        (
            "certify_creation_signature_hex",
            lambda value: value[:-2] + f"{int(value[-2:], 16) ^ 1:02x}",
        ),
        ("certify_creation_attest_hex", lambda value: "00" * 4 + value[8:]),
        ("certify_creation_attest_hex", lambda value: value[:8] + "0000" + value[12:]),
        (
            "certify_creation_attest_hex",
            lambda value: value[:20] + ("00" * 32) + value[84:],
        ),
        (
            "certify_creation_attest_hex",
            lambda value: value[:138] + ("00" * 34) + value[206:],
        ),
        (
            "certify_creation_attest_hex",
            lambda value: value[:210] + ("00" * 32),
        ),
    ],
)
def test_concrete_verifier_negative_attestation_matrix_preserves_pending(field, mutation):
    candidate, request, challenge, response, pending = exchange()
    changed = response.document
    changed[field] = mutation(changed[field])
    raw = json.dumps(changed, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError):
        ProductionTPMAttestationVerifier().verify(
            candidate.canonical_bytes,
            request.canonical_bytes,
            challenge.canonical_bytes,
            raw,
            pending=pending,
            expected_release_policy_digest=RELEASE,
        )
    pending.require_pending(challenge)


def test_forged_nominal_objects_are_irrelevant_to_consequential_issuer_boundary(monkeypatch):
    candidate, request, challenge, response, pending = exchange()
    forged = object.__new__(type("IssuerVerifiedTPMEvidenceV1", (), {}))
    assert forged is not None  # exploit setup: arbitrary nominal state is constructible

    signer = TestOnlyPDSAAuthority.deterministic_fixture()
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("private signing operation was reached")

    monkeypatch.setattr(signer, "_sign_unsigned_payload", forbidden)
    issuer = AttestedEnrollmentIssuer(signer)
    with pytest.raises(AssertionError, match="private signing operation was reached"):
        issuer.issue(
            activation_request_raw=candidate.canonical_bytes,
            decision=decision(candidate),
            enrollment_request_raw=request.canonical_bytes,
            challenge_raw=challenge.canonical_bytes,
            response_raw=response.canonical_bytes,
            pending=pending,
            expected_release_policy_digest=RELEASE,
        )
    assert called is True
    pending.require_pending(challenge)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("installation_id", "installation-b"),
        lambda value: value["release"].__setitem__("release_policy_digest", "bb" * 32),
        lambda value: value["requested_entitlements"].__setitem__(
            "requested_features", ["core_bot", "other"]
        ),
        lambda value: value["k_psa"].__setitem__("name", "000b" + "bb" * 32),
        lambda value: value["tpm"].__setitem__("ek_public_digest", "bb" * 32),
        lambda value: value["tpm"].__setitem__("ak_public_digest", "bb" * 32),
        lambda value: value["device"].__setitem__("device_id", "device-b"),
    ],
)
def test_confused_deputy_exchange_a_cannot_sign_modified_activation_b(mutation, monkeypatch):
    candidate_a, request, challenge, response, pending = exchange()
    candidate_b = mutate_activation(candidate_a, mutation)
    signer = TestOnlyPDSAAuthority.deterministic_fixture()
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("private signing operation was reached")

    monkeypatch.setattr(signer, "_sign_unsigned_payload", forbidden)
    issuer = TestOnlyAttestedEnrollmentIssuer(signer)
    with pytest.raises(ValueError, match="ACTIVATION_REQUEST_EXCHANGE_BINDING_MISMATCH"):
        issuer.issue(
            activation_request_raw=candidate_b.canonical_bytes,
            decision=decision(candidate_b),
            enrollment_request_raw=request.canonical_bytes,
            challenge_raw=challenge.canonical_bytes,
            response_raw=response.canonical_bytes,
            pending=pending,
            expected_release_policy_digest=RELEASE,
        )
    assert called is False
    pending.require_pending(challenge)


def test_decision_mismatch_rejected_before_exchange_or_signing(monkeypatch):
    candidate, request, challenge, response, pending = exchange()
    other = activation(installation_id="installation-b")
    signer = TestOnlyPDSAAuthority.deterministic_fixture()
    monkeypatch.setattr(
        signer,
        "_sign_unsigned_payload",
        lambda *_: (_ for _ in ()).throw(AssertionError("signer called")),
    )
    with pytest.raises(ValueError, match="DECISION_ACTIVATION_REQUEST_MISMATCH"):
        TestOnlyAttestedEnrollmentIssuer(signer).issue(
            activation_request_raw=candidate.canonical_bytes,
            decision=decision(other),
            enrollment_request_raw=request.canonical_bytes,
            challenge_raw=challenge.canonical_bytes,
            response_raw=response.canonical_bytes,
            pending=pending,
            expected_release_policy_digest=RELEASE,
        )


def test_test_only_attested_issuer_signs_only_after_exact_exchange_and_consumes_once():
    candidate, request, challenge, response, pending = exchange()
    issuer = TestOnlyAttestedEnrollmentIssuer(TestOnlyPDSAAuthority.deterministic_fixture())
    package = issuer.issue(
        activation_request_raw=candidate.canonical_bytes,
        decision=decision(candidate),
        enrollment_request_raw=request.canonical_bytes,
        challenge_raw=challenge.canonical_bytes,
        response_raw=response.canonical_bytes,
        pending=pending,
        expected_release_policy_digest=RELEASE,
    )
    assert package["request_binding"]["request_id"] == candidate.document["request_id"]
    assert issuer.last_exchange_reference is not None
    assert len(issuer.last_exchange_reference) == 64
    with pytest.raises(ValueError, match="REPLAY"):
        pending.require_pending(challenge)


def test_direct_private_helper_and_old_capability_api_do_not_exist():
    import bot_core.licensing.tpm_attestation as module

    assert not hasattr(module, "IssuerVerifiedTPMEvidenceV1")
    assert not hasattr(module, "verify_issuer_grade_attestation")
    assert not hasattr(TestOnlyPDSAAuthority, "_sign")


def test_physical_profile_cannot_use_unattested_authority_entrypoint():
    candidate = activation()
    with pytest.raises(ValueError, match="ATTESTED_EXCHANGE_ISSUER_REQUIRED"):
        TestOnlyPDSAAuthority.deterministic_fixture().issue(candidate, decision(candidate))
