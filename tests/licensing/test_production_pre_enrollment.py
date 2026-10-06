"""TEST_ONLY composition simulations; no production authority or physical TPM.

The public trust-package loader/current-runtime predicate and NCrypt DLL boundary
are explicit test harnesses. Challenge quorum, MakeCredential/activation proof,
AK CertifyCreation, custody, request PoP and issuer consumption use real code.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import struct
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils

from bot_core.licensing import (
    pdsa_enrollment_challenge as challenge,
    production_pre_enrollment as production,
    production_tpm_custody as custody,
)
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.licensing.device_enrollment import build_activation_request, make_evidence
from bot_core.licensing.pre_enrollment import PreEnrollmentError, PreEnrollmentRequestV1
from bot_core.licensing.tpm_attestation import (
    TPMEnrollmentChallengeResponseV1,
    TPMEnrollmentRequestV1,
    credential_activation_proof,
)
from deployment import windows_cng_pre_enrollment as cng, windows_stage9_production_trust as trust
from tests.deployment.test_windows_cng_pre_enrollment import TestOnlyNCryptDLL
from tests.licensing import test_pdsa_enrollment_challenge as test_challenge
from tests.licensing.test_production_tpm_custody import _attest, _public, _signature

NOW = test_challenge.NOW
challenge_harness = test_challenge.harness


@pytest.fixture
def integration(challenge_harness, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Real cryptographic composition around clearly isolated authority/hardware mocks."""
    authority = challenge_harness
    monkeypatch.setattr(
        production,
        "require_current_production_trust_context",
        trust.require_verified_production_trust_context,
    )
    monkeypatch.setattr(custody, "_utc_now", lambda: NOW)
    monkeypatch.setattr(
        custody,
        "require_verified_production_trust_context",
        trust.require_verified_production_trust_context,
    )
    pdsa_raw = authority.store.issue(authority.context, authority.sign)
    pdsa = authority.store.verify_issued(pdsa_raw, authority.context)
    dll = TestOnlyNCryptDLL()
    native = object.__new__(cng._NCryptAPI)
    native.dll = dll
    monkeypatch.setattr(cng, "_load_native", lambda: native)
    key = cng.WindowsCNGPreEnrollmentKey.open_or_create(tmp_path / "TEST_ONLY_CNG")
    keys = {role: ec.generate_private_key(ec.SECP256R1()) for role in ("ek", "ak", "k_psa")}
    keys["pre_enrollment"] = dll.private
    publics = {role: _public(private, role) for role, private in keys.items()}
    names = {role: b"\x00\x0b" + hashlib.sha256(raw).digest() for role, raw in publics.items()}
    projection = make_evidence(
        public_area_hex=publics["k_psa"].hex(),
        returned_name=names["k_psa"].hex(),
        creation_hash="77" * 32,
        ek_public_area_hex=publics["ek"].hex(),
        ek_name=names["ek"].hex(),
        ak_public_area_hex=publics["ak"].hex(),
        ak_name=names["ak"].hex(),
        algorithm_profile=custody.K_PSA_PROFILE,
        evidence_profile=custody.PROJECTION_PROFILE,
        substrate_profile=custody.PROJECTION_SOURCE_PROFILE,
        ek_certificate_digest="cc" * 32,
    )
    signer_ids = sorted(authority.keys)[:2]
    endorsement_payload = {
        "schema_version": "ProductionTPMEndorsementV1",
        "environment": "PRODUCTION",
        "product": "CryptoHunter",
        "product_profile": "CryptoHunter",
        "pdsa_trust_domain": "PDSA_PRODUCTION_2_OF_3_ED25519",
        "endorsement_profile": custody.ENDORSEMENT_PROFILE,
        "endorsement_id": "ptpm_019ba13c-5c00-7000-8000-000000000001",
        "ek_public_digest": hashlib.sha256(publics["ek"]).hexdigest(),
        "ek_name": names["ek"].hex(),
        "ek_certificate_digest_sha256": "cc" * 32,
        "manufacturer_chain_digest_sha256": "dd" * 32,
        "manufacturer_chain_verification_record_digest_sha256": "ee" * 32,
        "hardware_origin_verification": "MANUFACTURER_CHAIN_AND_EK_CERTIFICATE_BINDING_VERIFIED",
        "release_policy_digest_sha256": authority.context.release_payload_digest,
        "release_policy_generation": authority.context.release_version,
        "issued_at_utc": NOW.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expires_at_utc": (NOW + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "signature_algorithm_profile": "PDSA-2-OF-3-ED25519",
        "signer_key_ids": signer_ids,
    }
    digest = hashlib.sha256(canonical_json_bytes(endorsement_payload)).digest()
    endorsement_raw = canonical_json_bytes(
        {
            "payload": endorsement_payload,
            "payload_digest_sha256": digest.hex(),
            "signatures": [
                {
                    "algorithm": "Ed25519",
                    "key_id": signer,
                    "authority_key_set_digest": custody.PDSA_KEY_SET_DIGEST,
                    "signature_hex": authority.keys[signer]
                    .sign(custody.ENDORSEMENT_DOMAIN + digest)
                    .hex(),
                }
                for signer in signer_ids
            ],
        }
    )
    endorsement = custody.verify_production_tpm_endorsement(
        endorsement_raw, context=authority.context
    )
    activation = build_activation_request(
        evidence=projection,
        release_policy_digest=authority.context.release_payload_digest,
        release_policy_version=authority.context.release_version,
        requested_entitlements={
            "product": "CryptoHunter",
            "edition": "pro",
            "requested_features": ["core_bot"],
        },
        installation_id="TEST_ONLY_COMPOSITION_SIMULATION",
        architecture="AMD64",
        environment="TEST_ONLY",
        created_at_utc=NOW.strftime("%Y-%m-%dT%H:%M:%SZ"),
        nonce="33" * 32,
    )
    tpm_request = TPMEnrollmentRequestV1.create(
        activation_request=activation, public_projection=projection
    )
    pending = custody.ProductionTPMChallengeStore(tmp_path / "TEST_ONLY_TPM_PENDING.sqlite")
    tpm_challenge = pending.issue(
        activation.canonical_bytes,
        tpm_request.canonical_bytes,
        pdsa_challenge=pdsa,
        pdsa_store=authority.store,
        endorsement=endorsement,
        context=authority.context,
    )
    # TEST_ONLY emulation of the hardware activation output reads issuer-private
    # retention. The production verifier still verifies the exact generated proof.
    secret = pending._record(tpm_challenge)[2]
    nonce = bytes.fromhex(tpm_challenge.document["issuer_nonce_hex"])
    pdsa_digest = bytes.fromhex(pdsa.digest_sha256)
    attest = _attest(
        names["k_psa"],
        b"\x77" * 32,
        custody.production_exchange_qualifying_data(
            nonce, hashlib.sha256(activation.canonical_bytes).digest(), pdsa_digest
        ),
    )
    pop = keys["k_psa"].sign(
        custody.production_k_psa_pop_digest(nonce, pdsa_digest),
        ec.ECDSA(utils.Prehashed(hashes.SHA256())),
    )
    tpm_response = TPMEnrollmentChallengeResponseV1.create(
        tpm_challenge,
        activated_credential_digest=hashlib.sha256(secret).hexdigest(),
        credential_activation_proof_hex=credential_activation_proof(
            secret, tpm_challenge.document["challenge_id"]
        ).hex(),
        certify_creation_attest_hex=attest.hex(),
        certify_creation_signature_hex=_signature(keys["ak"], attest).hex(),
        k_psa_pop_signature_der_hex=pop.hex(),
    )
    raw_items = (
        activation.canonical_bytes,
        tpm_request.canonical_bytes,
        tpm_challenge.canonical_bytes,
        tpm_response.canonical_bytes,
    )
    exchange = custody.ProductionTPMEnrollmentVerifier().verify(
        *raw_items,
        pending=pending,
        pdsa_challenge=pdsa,
        pdsa_store=authority.store,
        endorsement=endorsement,
        context=authority.context,
    )
    build_args = {
        "context": authority.context,
        "challenge": pdsa,
        "challenge_store": authority.store,
        "exchange": exchange,
        "pending": pending,
        "key": key,
    }
    request = production.build_production_pre_enrollment_request(**build_args)
    subject_attest = _attest(
        names["pre_enrollment"],
        b"\x66" * 32,
        custody.pre_enrollment_custody_qualifying_data(request),
    )
    evidence = custody.make_pre_enrollment_custody_evidence(
        request=request,
        target_tpm_projection=projection,
        subject_tpmt_public=publics["pre_enrollment"],
        subject_name=names["pre_enrollment"],
        creation_hash=b"\x66" * 32,
        attest=subject_attest,
        signature=_signature(keys["ak"], subject_attest),
    )
    arguments = {
        "request_raw": request.canonical_bytes,
        "signature": key.sign_request(request, production_trust_context=authority.context),
        "challenge_raw": pdsa_raw,
        "challenge_store": authority.store,
        "activation_request_raw": raw_items[0],
        "tpm_request_raw": raw_items[1],
        "tpm_challenge_raw": raw_items[2],
        "tpm_response_raw": raw_items[3],
        "pending": pending,
        "endorsement_raw": endorsement_raw,
        "custody_evidence_raw": evidence.canonical_bytes,
        "context": authority.context,
    }
    yield SimpleNamespace(**locals())
    key.close()


def _authenticate(item, **changes):
    return production.authenticate_production_pre_enrollment(**(item.arguments | changes))


def _state(item):
    with sqlite3.connect(item.authority.store.path) as db:
        return db.execute("SELECT state FROM pdsa_challenges").fetchone()[0]


def _changed_request(item, **changes):
    return PreEnrollmentRequestV1.from_mapping(item.request.document | changes)


def test_test_only_real_crypto_composition_and_atomic_exact_retry(integration):
    item = integration
    accepted = _authenticate(item)
    assert production.require_authenticated_pre_enrollment(accepted) is accepted
    assert accepted.request_raw == item.request.canonical_bytes
    assert accepted.challenge_raw == item.pdsa_raw
    assert accepted.challenge_store is item.authority.store
    assert accepted.context is item.authority.context
    assert accepted.challenge.canonical_bytes == item.pdsa_raw
    assert _state(item) == "ISSUED"
    receipt_raw = item.authority.store.consume_authenticated_request(accepted)
    receipt = json.loads(receipt_raw)
    assert receipt["purpose"] == "PRE_ENROLLMENT_AUTHENTICATION_ACCEPTED_ONLY"
    assert receipt["legal_enrollment"] == "NOT_PERFORMED"
    assert receipt["pre_enrollment_request_digest_sha256"] == item.request.digest_sha256
    assert "provisioning_subject_id" not in receipt
    assert _state(item) == "CONSUMED"
    assert item.authority.store.consume_authenticated_request(accepted) == receipt_raw
    assert (
        item.authority.store.retry_exact_accepted(
            request_raw=item.request.canonical_bytes, challenge_raw=item.pdsa_raw
        )
        == receipt_raw
    )
    with pytest.raises(challenge.PDSAChallengeError, match="CONSUMED"):
        _authenticate(item)


def test_builder_derives_bindings_and_generates_fresh_request_nonce(integration):
    item = integration
    second = production.build_production_pre_enrollment_request(**item.build_args)
    left, right = item.request.document, second.document
    assert left.pop("request_nonce_hex") != right.pop("request_nonce_hex")
    assert left == right
    assert (
        item.request.document["verified_tpm_exchange_reference"] == item.exchange.exchange_reference
    )
    assert (
        item.request.document["pre_enrollment_public_key_canonical_bytes"]
        == item.key.public_key_bytes.hex()
    )
    with pytest.raises(TypeError):
        production.build_production_pre_enrollment_request(
            **item.build_args, request_nonce_hex="00" * 32
        )


def test_real_current_guard_rejects_audit_only_loader_context(integration, monkeypatch):
    item = integration
    monkeypatch.setattr(
        production,
        "require_current_production_trust_context",
        trust.require_current_production_trust_context,
    )
    with pytest.raises(trust.ProductionTrustUnavailable, match="CURRENT_RUNTIME"):
        _authenticate(item)
    with pytest.raises(trust.ProductionTrustUnavailable, match="CURRENT_RUNTIME"):
        production.build_production_pre_enrollment_request(**item.build_args)
    assert _state(item) == "ISSUED"


@pytest.mark.parametrize("kind", ["none", "shape", "clone", "copied_capability"])
def test_forged_or_copied_context_cannot_authenticate(integration, kind):
    item = integration
    value = None
    if kind == "shape":
        value = SimpleNamespace(
            ceremony_id=item.authority.context.ceremony_id,
            release_payload_digest=item.authority.context.release_payload_digest,
            release_version=item.authority.context.release_version,
            pdsa_keys=item.authority.context.pdsa_keys,
        )
    elif kind == "clone":
        value = object.__new__(trust.ProductionTrustContext)
        for field in trust.ProductionTrustContext.__slots__:
            if field != "__weakref__":
                object.__setattr__(value, field, getattr(item.authority.context, field))
    elif kind == "copied_capability":
        value = object.__new__(trust.ProductionTrustContext)
        object.__setattr__(value, "_capability", item.authority.context._capability)
    with pytest.raises(trust.ProductionTrustUnavailable):
        _authenticate(item, context=value)
    assert _state(item) == "ISSUED"


@pytest.mark.parametrize("field", ["challenge_store", "pending"])
def test_arbitrary_issuer_store_shape_rejected(integration, field):
    with pytest.raises(
        production.ProductionPreEnrollmentError, match="EXACT_PRODUCTION_ISSUER_STORES"
    ):
        _authenticate(
            integration, **{field: SimpleNamespace(path=integration.authority.store.path)}
        )
    assert _state(integration) == "ISSUED"


@pytest.mark.parametrize("field", ["challenge", "exchange", "key"])
def test_builder_rejects_copied_verified_capabilities(integration, field):
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        value = copy.copy(integration.build_args[field])
        production.build_production_pre_enrollment_request(
            **(integration.build_args | {field: value})
        )
    assert _state(integration) == "ISSUED"


@pytest.mark.parametrize("field", ["challenge_store", "pending"])
def test_builder_requires_exact_retaining_store_identity(integration, field):
    item = integration
    value = (
        challenge.PDSAChallengeStore(item.authority.store.path)
        if field == "challenge_store"
        else custody.ProductionTPMChallengeStore(item.pending._path)
    )
    with pytest.raises((ValueError, RuntimeError), match="REQUIRED|STORE_MISMATCH"):
        production.build_production_pre_enrollment_request(**(item.build_args | {field: value}))


@pytest.mark.parametrize(
    "field",
    [
        "request_raw",
        "challenge_raw",
        "activation_request_raw",
        "tpm_request_raw",
        "tpm_challenge_raw",
        "tpm_response_raw",
        "endorsement_raw",
        "custody_evidence_raw",
    ],
)
def test_noncanonical_or_malformed_raw_artifacts_rejected(integration, field):
    item = integration
    with pytest.raises(ValueError):
        _authenticate(item, **{field: item.arguments[field] + b" "})
    assert _state(item) == "ISSUED"


@pytest.mark.parametrize("signature", [b"", b"bad-DER", utils.encode_dss_signature(1, 1)])
def test_invalid_request_pop_cannot_authenticate_or_consume(integration, signature):
    with pytest.raises(PreEnrollmentError, match="INVALID_PRE_ENROLLMENT_SIGNATURE"):
        _authenticate(integration, signature=signature)
    assert _state(integration) == "ISSUED"


@pytest.mark.parametrize(
    ("field", "value"),
    [("release_policy_digest_sha256", "00" * 32), ("release_policy_generation", 2)],
)
def test_request_release_binding_rejected_before_consumption(integration, field, value):
    changed = _changed_request(integration, **{field: value})
    with pytest.raises(PreEnrollmentError, match="PRODUCTION_TRUST_BINDING_MISMATCH"):
        _authenticate(integration, request_raw=changed.canonical_bytes)
    assert _state(integration) == "ISSUED"


def test_changed_request_nonce_with_valid_pop_cannot_reuse_custody(integration):
    item = integration
    changed = _changed_request(item, request_nonce_hex="bb" * 32)
    signature = item.key.sign_request(changed, production_trust_context=item.authority.context)
    with pytest.raises(custody.ProductionTPMCustodyError, match="REQUEST_BINDING_MISMATCH"):
        _authenticate(item, request_raw=changed.canonical_bytes, signature=signature)
    assert _state(item) == "ISSUED"


def test_changed_pre_enrollment_key_with_valid_fingerprint_rejected(integration):
    item = integration
    public = (
        ec.generate_private_key(ec.SECP256R1())
        .public_key()
        .public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    )
    changed = _changed_request(
        item,
        pre_enrollment_public_key_canonical_bytes=public.hex(),
        pre_enrollment_public_key_fingerprint_sha256=hashlib.sha256(public).hexdigest(),
    )
    with pytest.raises(PreEnrollmentError, match="INVALID_PRE_ENROLLMENT_SIGNATURE"):
        _authenticate(item, request_raw=changed.canonical_bytes)


@pytest.mark.parametrize("property_name", ["Impl Type", "PCP_EXPORT_ALLOWED"])
def test_software_or_exportable_local_key_rejected(integration, property_name):
    item = integration
    handle = 11 if property_name == "Impl Type" else 22
    item.dll.properties[(handle, property_name)] = (
        struct.pack("<I", cng.SOFTWARE) if property_name == "Impl Type" else b"\x01"
    )
    with pytest.raises(
        cng.WindowsCNGPreEnrollmentError, match="VERIFIED_PRODUCTION_CNG_KEY_REQUIRED"
    ):
        production.build_production_pre_enrollment_request(**item.build_args)
    assert _state(item) == "ISSUED"


def test_closed_native_key_revokes_local_builder_and_signing(integration):
    item = integration
    item.key.close()
    with pytest.raises(
        cng.WindowsCNGPreEnrollmentError, match="VERIFIED_PRODUCTION_CNG_KEY_REQUIRED"
    ):
        production.build_production_pre_enrollment_request(**item.build_args)
    with pytest.raises(
        cng.WindowsCNGPreEnrollmentError, match="VERIFIED_PRODUCTION_CNG_KEY_REQUIRED"
    ):
        item.key.sign_request(item.request, production_trust_context=item.authority.context)
    assert _state(item) == "ISSUED"


def test_offhost_authentication_and_consume_require_no_live_local_key(integration):
    item = integration
    item.key.close()
    accepted = _authenticate(item)
    assert item.authority.store.consume_authenticated_request(accepted)
    assert _state(item) == "CONSUMED"


@pytest.mark.parametrize("kind", ["plain", "copied", "new", "subclass"])
def test_forged_acceptance_cannot_consume_retained_challenge(integration, kind):
    item = integration
    accepted = _authenticate(item)
    if kind == "plain":
        fake = SimpleNamespace(
            request_raw=accepted.request_raw, challenge_raw=accepted.challenge_raw
        )
    elif kind == "copied":
        fake = copy.copy(accepted)
    elif kind == "new":
        fake = object.__new__(production.AuthenticatedProductionPreEnrollment)
    else:

        class Forged(production.AuthenticatedProductionPreEnrollment):
            pass

        fake = object.__new__(Forged)
    with pytest.raises(
        production.ProductionPreEnrollmentError,
        match="AUTHENTICATED_PRODUCTION_PRE_ENROLLMENT_REQUIRED",
    ):
        item.authority.store.consume_authenticated_request(fake)
    assert _state(item) == "ISSUED"
    assert (
        item.authority.store.retry_exact_accepted(
            request_raw=item.request.canonical_bytes, challenge_raw=item.pdsa_raw
        )
        is None
    )


def test_acceptance_is_immutable_and_cannot_be_constructed(integration):
    with pytest.raises(TypeError, match="only"):
        production.AuthenticatedProductionPreEnrollment()
    accepted = _authenticate(integration)
    with pytest.raises(TypeError, match="immutable"):
        accepted.request_raw = b"forged"
    with pytest.raises(TypeError, match="immutable"):
        accepted.signature = b"forged"


def test_consumption_requires_original_store_object(integration):
    item = integration
    accepted = _authenticate(item)
    copied_store = challenge.PDSAChallengeStore(item.authority.store.path)
    with pytest.raises(production.ProductionPreEnrollmentError, match="ISSUER_STORE_MISMATCH"):
        copied_store.consume_authenticated_request(accepted)
    assert _state(item) == "ISSUED"


def test_changed_issuer_store_path_revokes_acceptance(integration, tmp_path):
    item = integration
    accepted = _authenticate(item)
    item.authority.store.path = tmp_path / "different.sqlite"
    with pytest.raises(production.ProductionPreEnrollmentError, match="ISSUER_STORE_MISMATCH"):
        production.require_authenticated_pre_enrollment(accepted)


def test_expiry_after_authentication_is_terminal_before_atomic_consume(integration, monkeypatch):
    item = integration
    accepted = _authenticate(item)
    monkeypatch.setattr(challenge, "_utc_now", lambda: NOW + timedelta(days=7))
    with pytest.raises(challenge.PDSAChallengeError, match="EXPIRED"):
        item.authority.store.consume_authenticated_request(accepted)
    assert _state(item) == "EXPIRED"


def test_shorter_endorsement_expiry_during_writer_lock_prevents_consumption(
    integration, monkeypatch
):
    item = integration
    accepted = _authenticate(item)
    original_connect = item.authority.store._connect

    @contextmanager
    def delayed_database():
        with original_connect() as db:
            # PDSA remains live for seven days, but the hardware approval and
            # its derived TPM challenge expire after one day in this harness.
            monkeypatch.setattr(challenge, "_utc_now", lambda: NOW + timedelta(days=1))
            monkeypatch.setattr(custody, "_utc_now", lambda: NOW + timedelta(days=1))
            yield db

    monkeypatch.setattr(item.authority.store, "_connect", delayed_database)
    with pytest.raises(custody.ProductionTPMCustodyError, match="ENDORSEMENT_EXPIRED"):
        item.authority.store.consume_authenticated_request(accepted)
    assert _state(item) == "ISSUED"


def test_same_challenge_other_request_is_permanent_replay_conflict(integration):
    item = integration
    accepted = _authenticate(item)
    item.authority.store.consume_authenticated_request(accepted)
    other = _changed_request(item, request_nonce_hex="dd" * 32)
    with pytest.raises(challenge.PDSAChallengeError, match="CHALLENGE_REPLAY_CONFLICT"):
        item.authority.store.retry_exact_accepted(
            request_raw=other.canonical_bytes, challenge_raw=item.pdsa_raw
        )
    assert _state(item) == "CONSUMED"


def test_software_substitute_with_valid_request_pop_lacks_exact_tpm_custody(integration):
    item = integration
    substitute = ec.generate_private_key(ec.SECP256R1())
    public = substitute.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    changed = _changed_request(
        item,
        pre_enrollment_public_key_canonical_bytes=public.hex(),
        pre_enrollment_public_key_fingerprint_sha256=hashlib.sha256(public).hexdigest(),
    )
    r, s = utils.decode_dss_signature(
        substitute.sign(changed.signing_bytes, ec.ECDSA(hashes.SHA256()))
    )
    signature = utils.encode_dss_signature(r, min(s, cng.P256_ORDER - s))
    changed.verify_signature(signature)
    with pytest.raises(custody.ProductionTPMCustodyError, match="REQUEST_BINDING_MISMATCH"):
        _authenticate(item, request_raw=changed.canonical_bytes, signature=signature)
    assert _state(item) == "ISSUED"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("environment", "TEST_ONLY"),
        ("custody_profile", "TEST_ONLY"),
        ("provider_name", "Microsoft Software Key Storage Provider"),
        ("subject_tpmt_public_hex", "00"),
        ("subject_tpm_name", "000b" + "00" * 32),
        ("subject_creation_hash", "00" * 32),
        ("certify_creation_signature_hex", "0018000b000100000100"),
        ("pre_enrollment_request_digest_sha256", "00" * 32),
        ("pdsa_challenge_digest_sha256", "00" * 32),
        ("release_policy_digest_sha256", "00" * 32),
        ("verified_tpm_public_projection_id", "00" * 32),
        ("ak_public_digest", "00" * 32),
        ("ek_public_digest", "00" * 32),
    ],
)
def test_changed_canonical_tpm_custody_cannot_authenticate(integration, field, value):
    item = integration
    payload = item.evidence.document | {field: value}
    with pytest.raises(ValueError):
        _authenticate(item, custody_evidence_raw=canonical_json_bytes(payload))
    assert _state(item) == "ISSUED"


@pytest.mark.parametrize(
    "field", ["pdsa_challenge_digest_sha256", "pdsa_challenge_nonce_digest_sha256"]
)
def test_changed_request_challenge_bindings_rejected(integration, field):
    changed = _changed_request(integration, **{field: "00" * 32})
    with pytest.raises(challenge.PDSAChallengeError, match="CHALLENGE_BINDING_MISMATCH"):
        _authenticate(integration, request_raw=changed.canonical_bytes)
    assert _state(integration) == "ISSUED"


def test_changed_challenge_quorum_signature_rejected_in_composition(integration):
    item = integration
    payload = json.loads(item.pdsa_raw)
    payload["signatures"][0]["signature_hex"] = "00" * 64
    with pytest.raises(challenge.PDSAChallengeError, match="PDSA_SIGNATURE"):
        _authenticate(item, challenge_raw=canonical_json_bytes(payload))
    assert _state(item) == "ISSUED"


def test_unknown_retained_challenge_rejected_in_composition(integration, tmp_path):
    item = integration
    foreign_store = challenge.PDSAChallengeStore(tmp_path / "unknown-issuer.sqlite")
    with pytest.raises(challenge.PDSAChallengeError, match="UNKNOWN_RETAINED"):
        _authenticate(item, challenge_store=foreign_store)
    assert _state(item) == "ISSUED"


def test_changed_endorsement_quorum_signature_rejected_in_composition(integration):
    item = integration
    payload = json.loads(item.endorsement_raw)
    payload["signatures"][0]["signature_hex"] = "00" * 64
    with pytest.raises(custody.ProductionTPMCustodyError, match="ENDORSEMENT_SIGNATURE"):
        _authenticate(item, endorsement_raw=canonical_json_bytes(payload))
    assert _state(item) == "ISSUED"


def test_changed_retained_tpm_response_bytes_rejected_in_composition(integration):
    item = integration
    payload = item.tpm_response.document | {"credential_activation_proof_hex": "00" * 32}
    with pytest.raises(custody.ProductionTPMCustodyError, match="TPM_EXCHANGE_REPLAY_CONFLICT"):
        _authenticate(item, tpm_response_raw=canonical_json_bytes(payload))
    assert _state(item) == "ISSUED"


def test_authenticated_capability_rejects_changed_pending_store_path(integration, tmp_path):
    item = integration
    accepted = _authenticate(item)
    copied_path = tmp_path / "copied-pending.sqlite"
    copied_path.write_bytes(item.pending._path.read_bytes())
    object.__setattr__(item.pending, "_path", copied_path)
    with pytest.raises(custody.ProductionTPMCustodyError, match="STORE|CONFIGURATION"):
        item.authority.store.consume_authenticated_request(accepted)
    assert _state(item) == "ISSUED"


@pytest.mark.parametrize(
    "mutation",
    ["state", "exchange_reference", "response"],
)
def test_authenticated_capability_rejects_changed_retained_tpm_source(integration, mutation):
    item = integration
    accepted = _authenticate(item)
    with sqlite3.connect(item.pending._path) as db:
        if mutation == "state":
            db.execute("UPDATE tpm_challenges SET state='EXPIRED'")
        elif mutation == "exchange_reference":
            db.execute("UPDATE tpm_challenges SET exchange_reference=?", ("00" * 32,))
        else:
            db.execute("UPDATE tpm_challenges SET response=?", (b"changed-response",))
    with pytest.raises(custody.ProductionTPMCustodyError):
        item.authority.store.consume_authenticated_request(accepted)
    assert _state(item) == "ISSUED"
