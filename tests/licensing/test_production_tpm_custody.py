"""TEST_ONLY algorithm simulations; no OEM approval, production authority or physical TPM."""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils

from bot_core.licensing import production_tpm_custody as module
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.licensing.device_enrollment import build_activation_request, make_evidence
from bot_core.licensing.pre_enrollment import (
    ALGORITHM_PROFILE,
    PDSA_TRUST_DOMAIN,
    PreEnrollmentRequestV1,
    public_key_fingerprint,
)
from bot_core.licensing.product_profile import PRODUCT_NAME, PRODUCTION_PRODUCT_PROFILE
from bot_core.licensing.tpm_attestation import (
    TPMEnrollmentChallengeResponseV1,
    TPMEnrollmentRequestV1,
    credential_activation_proof,
)
from deployment import windows_stage9_production_trust as trust_module
from deployment.production_enrollment_issuer import (
    ProductionEnrollmentIssuerError,
    require_production_tpm_store,
)
from tests.licensing import test_pdsa_enrollment_challenge as test_challenge

challenge_harness = test_challenge.harness
NOW = test_challenge.NOW


def _b(raw):
    return len(raw).to_bytes(2, "big") + raw


def _public(key, role):
    point = key.public_key().public_numbers()
    attributes = {"pre_enrollment": 0x40072, "ak": 0x50072, "k_psa": 0x400B2, "ek": 0x300B2}[role]
    policy = (
        bytes.fromhex("837197674484b3f81a90cc8d46a5d724fd52d76e06520b64f2a1da1b331469aa")
        if role == "ek"
        else b"\x11" * 32
        if role == "k_psa"
        else b""
    )
    symmetric = bytes.fromhex("000600800043") if role == "ek" else bytes.fromhex("0010")
    scheme = bytes.fromhex("0010") if role == "ek" else bytes.fromhex("0018000b")
    return (
        bytes.fromhex("0023000b")
        + attributes.to_bytes(4, "big")
        + _b(policy)
        + symmetric
        + scheme
        + bytes.fromhex("00030010")
        + _b(point.x.to_bytes(32, "big"))
        + _b(point.y.to_bytes(32, "big"))
    )


def _attest(name, creation_hash, qualifier):
    return (
        bytes.fromhex("ff544347801a")
        + _b(b"\x00\x0b" + b"\x99" * 32)
        + _b(qualifier)
        + bytes(16)
        + b"\x01"
        + bytes(8)
        + _b(name)
        + _b(creation_hash)
    )


def _signature(key, attest):
    r, s = utils.decode_dss_signature(key.sign(attest, ec.ECDSA(hashes.SHA256())))
    return bytes.fromhex("0018000b") + _b(r.to_bytes(32, "big")) + _b(s.to_bytes(32, "big"))


@pytest.fixture
def simulation(challenge_harness, monkeypatch):
    """Real store provenance around explicit TEST_ONLY trust-package/hardware boundaries."""
    harness = challenge_harness
    authority, context = harness.keys, harness.context
    monkeypatch.setattr(
        module,
        "require_verified_production_trust_context",
        trust_module.require_verified_production_trust_context,
    )
    monkeypatch.setattr(module, "_utc_now", lambda: NOW)
    keys = {
        role: ec.generate_private_key(ec.SECP256R1())
        for role in ("ek", "ak", "k_psa", "pre_enrollment")
    }
    publics = {role: _public(key, role) for role, key in keys.items()}
    names = {role: b"\x00\x0b" + hashlib.sha256(raw).digest() for role, raw in publics.items()}
    projection = make_evidence(
        public_area_hex=publics["k_psa"].hex(),
        returned_name=names["k_psa"].hex(),
        creation_hash="77" * 32,
        ek_public_area_hex=publics["ek"].hex(),
        ek_name=names["ek"].hex(),
        ak_public_area_hex=publics["ak"].hex(),
        ak_name=names["ak"].hex(),
        algorithm_profile=module.K_PSA_PROFILE,
        evidence_profile=module.PROJECTION_PROFILE,
        substrate_profile=module.PROJECTION_SOURCE_PROFILE,
        ek_certificate_digest="cc" * 32,
    )
    payload = {
        "schema_version": "ProductionTPMEndorsementV1",
        "environment": "PRODUCTION",
        "product": PRODUCT_NAME,
        "product_profile": PRODUCTION_PRODUCT_PROFILE,
        "pdsa_trust_domain": PDSA_TRUST_DOMAIN,
        "endorsement_profile": module.ENDORSEMENT_PROFILE,
        "endorsement_id": "ptpm_019ba13c-5c00-7000-8000-000000000001",
        "ek_public_digest": hashlib.sha256(publics["ek"]).hexdigest(),
        "ek_name": names["ek"].hex(),
        "ek_certificate_digest_sha256": "cc" * 32,
        "manufacturer_chain_digest_sha256": "dd" * 32,
        "manufacturer_chain_verification_record_digest_sha256": "ee" * 32,
        "hardware_origin_verification": "MANUFACTURER_CHAIN_AND_EK_CERTIFICATE_BINDING_VERIFIED",
        "release_policy_digest_sha256": context.release_payload_digest,
        "release_policy_generation": 1,
        "issued_at_utc": "2026-10-06T12:00:00Z",
        "expires_at_utc": "2026-10-13T12:00:00Z",
        "signature_algorithm_profile": "PDSA-2-OF-3-ED25519",
        "signer_key_ids": sorted(authority)[:2],
    }

    def signed_endorsement(candidate):
        digest = hashlib.sha256(canonical_json_bytes(candidate)).digest()
        return canonical_json_bytes(
            {
                "payload": candidate,
                "payload_digest_sha256": digest.hex(),
                "signatures": [
                    {
                        "algorithm": "Ed25519",
                        "key_id": name,
                        "authority_key_set_digest": module.PDSA_KEY_SET_DIGEST,
                        "signature_hex": authority[name]
                        .sign(module.ENDORSEMENT_DOMAIN + digest)
                        .hex(),
                    }
                    for name in candidate["signer_key_ids"]
                    if name in authority
                ],
            }
        )

    endorsement_raw = signed_endorsement(payload)
    endorsement = module.verify_production_tpm_endorsement(endorsement_raw, context=context)
    pdsa_store = harness.store
    pdsa_raw = pdsa_store.issue(context, harness.sign)
    pdsa = pdsa_store.verify_issued(pdsa_raw, context)
    activation = build_activation_request(
        evidence=projection,
        release_policy_digest=context.release_payload_digest,
        release_policy_version=1,
        requested_entitlements={
            "product": "CryptoHunter",
            "edition": "pro",
            "requested_features": ["core_bot"],
        },
        installation_id="TEST_ONLY_ALGORITHM_SIMULATION",
        architecture="AMD64",
        environment="TEST_ONLY",
        created_at_utc="2026-01-01T00:00:00Z",
        nonce="33" * 32,
    )
    tpm_request = TPMEnrollmentRequestV1.create(
        activation_request=activation, public_projection=projection
    )
    pending = harness.issuer.tpm_store
    challenge = pending.issue(
        activation.canonical_bytes,
        tpm_request.canonical_bytes,
        pdsa_challenge=pdsa,
        pdsa_store=pdsa_store,
        endorsement=endorsement,
        context=context,
    )
    # TEST_ONLY hardware emulation reads issuer state; callers cannot supply this secret.
    issued_row = pending._record(challenge)
    secret = issued_row[2]
    nonce = bytes.fromhex(challenge.document["issuer_nonce_hex"])
    qualify = module.production_exchange_qualifying_data(
        nonce,
        hashlib.sha256(activation.canonical_bytes).digest(),
        bytes.fromhex(pdsa.digest_sha256),
    )
    attest = _attest(names["k_psa"], b"\x77" * 32, qualify)
    pop = keys["k_psa"].sign(
        module.production_k_psa_pop_digest(nonce, bytes.fromhex(pdsa.digest_sha256)),
        ec.ECDSA(utils.Prehashed(hashes.SHA256())),
    )
    response = TPMEnrollmentChallengeResponseV1.create(
        challenge,
        activated_credential_digest=hashlib.sha256(secret).hexdigest(),
        credential_activation_proof_hex=credential_activation_proof(
            secret, challenge.document["challenge_id"]
        ).hex(),
        certify_creation_attest_hex=attest.hex(),
        certify_creation_signature_hex=_signature(keys["ak"], attest).hex(),
        k_psa_pop_signature_der_hex=pop.hex(),
    )
    raw_items = (
        activation.canonical_bytes,
        tpm_request.canonical_bytes,
        challenge.canonical_bytes,
        response.canonical_bytes,
    )
    arguments = {
        "pending": pending,
        "pdsa_challenge": pdsa,
        "pdsa_store": pdsa_store,
        "endorsement": endorsement,
        "context": context,
    }
    exchange = module.ProductionTPMEnrollmentVerifier().verify(*raw_items, **arguments)
    sec1 = (
        keys["pre_enrollment"]
        .public_key()
        .public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    )
    request = PreEnrollmentRequestV1.from_mapping(
        {
            "schema_version": "PreEnrollmentRequestV1",
            "environment": "PRODUCTION",
            "product": PRODUCT_NAME,
            "product_profile": PRODUCTION_PRODUCT_PROFILE,
            "pdsa_trust_domain": PDSA_TRUST_DOMAIN,
            "pdsa_challenge_id": pdsa.document["payload"]["challenge_id"],
            "pdsa_challenge_digest_sha256": pdsa.digest_sha256,
            "pdsa_challenge_nonce_digest_sha256": pdsa.nonce_digest_sha256,
            "tpm_enrollment_request_digest_sha256": hashlib.sha256(raw_items[1]).hexdigest(),
            "tpm_enrollment_challenge_digest_sha256": hashlib.sha256(raw_items[2]).hexdigest(),
            "tpm_enrollment_response_digest_sha256": hashlib.sha256(raw_items[3]).hexdigest(),
            "verified_tpm_exchange_reference": exchange.exchange_reference,
            "verified_tpm_public_projection_id": projection.evidence_reference,
            "ek_public_digest": projection.document["ek"]["public_digest"],
            "ak_public_digest": projection.document["ak"]["public_digest"],
            "tpm_attestation_evidence_reference": projection.evidence_reference,
            "pre_enrollment_public_key_algorithm_profile": ALGORITHM_PROFILE,
            "pre_enrollment_public_key_canonical_bytes": sec1.hex(),
            "pre_enrollment_public_key_fingerprint_sha256": public_key_fingerprint(sec1),
            "release_policy_digest_sha256": context.release_payload_digest,
            "release_policy_generation": 1,
            "request_nonce_hex": "44" * 32,
        }
    )
    subject_attest = _attest(
        names["pre_enrollment"],
        b"\x66" * 32,
        module.pre_enrollment_custody_qualifying_data(request),
    )
    evidence = module.make_pre_enrollment_custody_evidence(
        request=request,
        target_tpm_projection=projection,
        subject_tpmt_public=publics["pre_enrollment"],
        subject_name=names["pre_enrollment"],
        creation_hash=b"\x66" * 32,
        attest=subject_attest,
        signature=_signature(keys["ak"], subject_attest),
    )
    return SimpleNamespace(**locals())


def test_test_only_simulated_production_algorithms_and_exact_retry(simulation):
    item = simulation
    verifier = module.ProductionPreEnrollmentKeyCustodyVerifier()
    verified = verifier.verify(
        item.evidence.canonical_bytes,
        request=item.request,
        exchange=item.exchange,
        endorsement=item.endorsement,
        context=item.context,
    )
    assert (
        module.require_verified_pre_enrollment_custody(
            verified, context=item.context, request=item.request, exchange=item.exchange
        )
        is verified
    )
    assert (
        module.ProductionTPMEnrollmentVerifier()
        .verify(*item.raw_items, **item.arguments)
        .exchange_reference
        == item.exchange.exchange_reference
    )
    reference = item.exchange.exchange_reference
    item.harness.issuer.close()
    with pytest.raises(ProductionEnrollmentIssuerError, match="STORE_REQUIRED"):
        module.require_verified_production_tpm_exchange(item.exchange, context=item.context)
    reopened = item.harness.reopen()
    assert reopened.tpm_store._record(item.challenge)[9] == "VERIFIED"
    pdsa = reopened.pdsa_store.verify_issued(item.pdsa_raw, item.context)
    retried = module.ProductionTPMEnrollmentVerifier().verify(
        *item.raw_items,
        **{
            **item.arguments,
            "pending": reopened.tpm_store,
            "pdsa_store": reopened.pdsa_store,
            "pdsa_challenge": pdsa,
        },
    )
    assert retried.exchange_reference == reference


@pytest.mark.parametrize(
    "field",
    [
        "pre_enrollment_request_digest_sha256",
        "verified_tpm_public_projection_id",
        "verified_tpm_exchange_reference",
        "ek_public_digest",
        "ak_public_digest",
        "pdsa_challenge_digest_sha256",
        "release_policy_digest_sha256",
        "release_policy_generation",
        "subject_tpm_name",
        "subject_creation_hash",
        "creation_attestation_digest_sha256",
        "pre_enrollment_public_key_fingerprint_sha256",
        "pre_enrollment_public_key_canonical_bytes",
        "subject_tpmt_public_hex",
        "certify_creation_attest_hex",
        "certify_creation_signature_hex",
    ],
)
def test_mutated_custody_is_rejected(simulation, field):
    item, changed = simulation, simulation.evidence.document
    changed[field] = (
        2 if field == "release_policy_generation" else "00" * (len(changed[field]) // 2)
    )
    with pytest.raises(ValueError):
        module.ProductionPreEnrollmentKeyCustodyVerifier().verify(
            canonical_json_bytes(changed),
            request=item.request,
            exchange=item.exchange,
            endorsement=item.endorsement,
            context=item.context,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("environment", "TEST_ONLY"),
        ("endorsement_profile", "TEST_ONLY"),
        ("product", "Other"),
        ("release_policy_generation", True),
        ("release_policy_digest_sha256", "ff" * 32),
        ("ek_name", "000b" + "ff" * 32),
        ("expires_at_utc", "2026-10-06T12:00:00Z"),
        ("issued_at_utc", "2026-10-07T12:00:00Z"),
        ("signer_key_ids", ["CALLER_SELECTED_KEY"]),
        ("signer_key_ids", ["TEST_ONLY_PDSA_0"]),
    ],
)
def test_endorsement_profile_authority_window_rejections(simulation, field, value):
    payload = dict(simulation.payload)
    payload[field] = value
    with pytest.raises(ValueError):
        module.verify_production_tpm_endorsement(
            simulation.signed_endorsement(payload), context=simulation.context
        )


def test_software_fabricated_evidence_without_genuine_loader_context_is_rejected():
    with pytest.raises(RuntimeError, match="CONTEXT_REQUIRED"):
        module.verify_production_tpm_endorsement(b"{}", context=SimpleNamespace())


@pytest.mark.parametrize(
    "kind",
    [
        module.VerifiedProductionTPMEndorsement,
        module.VerifiedProductionTPMExchange,
        module.VerifiedProductionPreEnrollmentKeyCustody,
    ],
)
def test_forged_and_copied_capability_fields_do_not_transfer_provenance(simulation, kind):
    with pytest.raises(TypeError):
        kind()
    forged = object.__new__(kind)
    with pytest.raises(ValueError, match="CAPABILITY_REQUIRED"):
        module._snapshot(forged)
    with pytest.raises(AttributeError):
        object.__setattr__(forged, "canonical_bytes", simulation.evidence.canonical_bytes)


@pytest.mark.parametrize("role", ["ek", "ak", "k_psa", "pre_enrollment"])
def test_public_parser_rejects_imported_wrong_attributes_point_and_trailing(simulation, role):
    raw = simulation.publics[role]
    assert module.parse_production_ecc_public(raw, role=role).name == simulation.names[role]
    for bit in (0x2, 0x10, 0x20):
        attributes = int.from_bytes(raw[4:8], "big") & ~bit
        with pytest.raises(ValueError):
            module.parse_production_ecc_public(
                raw[:4] + attributes.to_bytes(4, "big") + raw[8:], role=role
            )
    for malformed in (raw[:-1], raw + b"\x00", raw[:-32] + bytes(32)):
        with pytest.raises(ValueError):
            module.parse_production_ecc_public(malformed, role=role)


def test_custody_rejects_second_request_and_production_creation_domain_relabel(simulation):
    item = simulation
    changed = item.request.document
    changed["request_nonce_hex"] = "88" * 32
    other = PreEnrollmentRequestV1.from_mapping(changed)
    with pytest.raises(ValueError, match="REQUEST_BINDING"):
        module.ProductionPreEnrollmentKeyCustodyVerifier().verify(
            item.evidence.canonical_bytes,
            request=other,
            exchange=item.exchange,
            endorsement=item.endorsement,
            context=item.context,
        )
    attest = _attest(
        item.names["pre_enrollment"], b"\x66" * 32, hashlib.sha256(b"TEST_ONLY").digest()
    )
    changed_evidence = item.evidence.document
    changed_evidence.update(
        certify_creation_attest_hex=attest.hex(),
        certify_creation_signature_hex=_signature(item.keys["ak"], attest).hex(),
        creation_attestation_digest_sha256=hashlib.sha256(attest).hexdigest(),
    )
    with pytest.raises(ValueError, match="CREATION_BINDING"):
        module.ProductionPreEnrollmentKeyCustodyVerifier().verify(
            canonical_json_bytes(changed_evidence),
            request=item.request,
            exchange=item.exchange,
            endorsement=item.endorsement,
            context=item.context,
        )


def test_exchange_conflict_unknown_store_and_expiration(simulation, monkeypatch, tmp_path):
    item = simulation
    changed = item.response.document
    changed["activated_credential_digest"] = "00" * 32
    with pytest.raises(ValueError, match="REPLAY_CONFLICT"):
        module.ProductionTPMEnrollmentVerifier().verify(
            *item.raw_items[:3], canonical_json_bytes(changed), **item.arguments
        )
    unknown = module.ProductionTPMChallengeStore(tmp_path / "unknown.sqlite")
    with pytest.raises(ValueError, match="STORE_MISMATCH"):
        module.require_verified_production_tpm_exchange(
            item.exchange, context=item.context, pending=unknown
        )
    with sqlite3.connect(item.pending._path) as database:
        database.execute("UPDATE tpm_challenges SET response=?", (b"modified",))
    with pytest.raises(ValueError, match="RETAINED"):
        module.require_verified_production_tpm_exchange(item.exchange, context=item.context)
    monkeypatch.setattr(module, "_utc_now", lambda: datetime(2026, 10, 14, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="EXPIRED"):
        module._require_endorsement(item.endorsement, item.context)


def test_noncanonical_extra_fields_and_wrong_signatures(simulation):
    item = simulation
    for raw in (
        item.endorsement_raw + b"\n",
        canonical_json_bytes({**item.evidence.document, "extra": True}),
    ):
        with pytest.raises(ValueError):
            module.ProductionPreEnrollmentKeyCustodyEvidenceV1.from_canonical_bytes(raw)
    envelope = item.endorsement.document
    envelope["signatures"][0]["signature_hex"] = "00" * 64
    with pytest.raises(ValueError, match="SIGNATURE"):
        module.verify_production_tpm_endorsement(
            canonical_json_bytes(envelope), context=item.context
        )
    envelope["signatures"][0]["signature_hex"] = "00"
    with pytest.raises(ValueError):
        module.verify_production_tpm_endorsement(
            canonical_json_bytes(envelope), context=item.context
        )


def test_malformed_creation_structure_and_certify_substitution(simulation):
    raw = bytes.fromhex(simulation.evidence.document["certify_creation_attest_hex"])
    for malformed in (raw[:-1], raw + b"\x00", bytes(4) + raw[4:], raw[:4] + b"\x80\x17" + raw[6:]):
        with pytest.raises(ValueError):
            module.parse_production_creation_attestation(malformed)


@pytest.mark.parametrize(
    "attack", ["empty", "copied_database", "reconstructed_verified", "stale_issued", "same_path"]
)
def test_unissued_store_cannot_verify_copied_or_reconstructed_pending_state(
    simulation, tmp_path, attack
):
    item = simulation
    target = tmp_path / f"{attack}.sqlite"
    if attack == "copied_database":
        # All retained bytes, including the valid response and credential secret,
        # are present. A bit-for-bit SQLite copy still has no issuer issuance.
        with sqlite3.connect(item.pending._path) as database:
            database.execute("PRAGMA wal_checkpoint(FULL)")
        shutil.copyfile(item.pending._path, target)
        assert target.read_bytes() == item.pending._path.read_bytes()
    if attack == "same_path":
        target = item.pending._path
    counterfeit = module.ProductionTPMChallengeStore(target)
    if attack in {"reconstructed_verified", "stale_issued"}:
        with sqlite3.connect(item.pending._path) as database:
            retained = database.execute("SELECT * FROM tpm_challenges").fetchone()
        assert retained[9] == "VERIFIED"
        row = item.issued_row if attack == "stale_issued" else retained
        with sqlite3.connect(target) as database:
            database.execute("INSERT INTO tpm_challenges VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", row)
    with pytest.raises(ProductionEnrollmentIssuerError, match="STORE_REQUIRED"):
        module.ProductionTPMEnrollmentVerifier().verify(
            *item.raw_items, **{**item.arguments, "pending": counterfeit}
        )
    with pytest.raises(ProductionEnrollmentIssuerError, match="STORE_REQUIRED"):
        counterfeit._record(item.challenge)
    with pytest.raises(ProductionEnrollmentIssuerError, match="STORE_REQUIRED"):
        counterfeit.issue(
            *item.raw_items[:2],
            pdsa_challenge=item.pdsa,
            pdsa_store=item.pdsa_store,
            endorsement=item.endorsement,
            context=item.context,
        )
    with pytest.raises(ProductionEnrollmentIssuerError, match="STORE_REQUIRED"):
        counterfeit._retain_verified(
            item.challenge,
            item.raw_items[3],
            item.exchange.exchange_reference,
            expected_record=item.issued_row,
            pdsa_challenge=item.pdsa,
            pdsa_store=item.pdsa_store,
            endorsement=item.endorsement,
            context=item.context,
        )
    assert (
        module.require_verified_production_tpm_exchange(item.exchange, context=item.context)
        is item.exchange
    )


def test_copied_python_store_fields_never_register_issuer_provenance(simulation):
    item = simulation
    counterfeit = object.__new__(module.ProductionTPMChallengeStore)
    object.__setattr__(counterfeit, "_path", item.pending._path)
    with pytest.raises(TypeError, match="immutable"):
        counterfeit._path = item.pending._path
    with pytest.raises(ProductionEnrollmentIssuerError, match="STORE_REQUIRED"):
        require_production_tpm_store(counterfeit, context=item.context)
    with pytest.raises(ProductionEnrollmentIssuerError, match="STORE_REQUIRED"):
        module.ProductionTPMEnrollmentVerifier().verify(
            *item.raw_items, **{**item.arguments, "pending": counterfeit}
        )


def test_genuine_stores_from_different_issuer_contexts_cannot_be_mixed(simulation):
    item = simulation
    other = item.harness.reopen()
    assert other is not item.harness.issuer
    assert require_production_tpm_store(other.tpm_store, context=item.context) is other
    with pytest.raises(ProductionEnrollmentIssuerError, match="CONTEXT_MISMATCH"):
        module.ProductionTPMEnrollmentVerifier().verify(
            *item.raw_items, **{**item.arguments, "pending": other.tpm_store}
        )
    with pytest.raises(ProductionEnrollmentIssuerError, match="CONTEXT_MISMATCH"):
        item.pending.issue(
            *item.raw_items[:2],
            pdsa_challenge=item.pdsa,
            pdsa_store=other.pdsa_store,
            endorsement=item.endorsement,
            context=item.context,
        )
    with pytest.raises(ProductionEnrollmentIssuerError, match="CONTEXT_MISMATCH"):
        module.ProductionTPMEnrollmentVerifier().verify(
            *item.raw_items, **{**item.arguments, "context": object()}
        )
    with pytest.raises(ValueError, match="ISSUER_MISMATCH"):
        module.require_verified_production_tpm_exchange(
            item.exchange, context=item.context, issuer=other
        )


def test_pending_mechanics_preserve_wal_full_and_close_connections(simulation):
    with simulation.pending._connect() as database:
        assert database.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert database.execute("PRAGMA synchronous").fetchone()[0] == 2
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        database.execute("SELECT 1")


def test_terminal_expired_pending_cannot_be_republished_after_clock_rollback(
    simulation, monkeypatch
):
    item = simulation
    # Model the originally retained ISSUED state before any response acceptance.
    with sqlite3.connect(item.pending._path) as database:
        database.execute(
            "UPDATE tpm_challenges SET state='ISSUED',exchange_reference=NULL,response=NULL"
        )
    monkeypatch.setattr(module, "_utc_now", lambda: NOW + timedelta(days=8))
    with pytest.raises(ValueError, match="EXPIRED"):
        item.pending._record(item.challenge)
    monkeypatch.setattr(module, "_utc_now", lambda: NOW)
    with pytest.raises(ValueError, match="EXPIRED"):
        item.pending.issue(
            *item.raw_items[:2],
            pdsa_challenge=item.pdsa,
            pdsa_store=item.pdsa_store,
            endorsement=item.endorsement,
            context=item.context,
        )
    with sqlite3.connect(item.pending._path) as database:
        assert database.execute("SELECT state FROM tpm_challenges").fetchone()[0] == "EXPIRED"


@pytest.mark.parametrize("operation", ["issue", "retain"])
@pytest.mark.parametrize("change", ["close_issuer", "expire_endorsement"])
def test_writer_lock_rechecks_issuer_and_clock_before_tpm_publication(
    simulation, monkeypatch, operation, change
):
    item = simulation
    original_connect = module.ProductionTPMChallengeStore._connect
    with sqlite3.connect(item.pending._path) as database:
        before = database.execute("SELECT * FROM tpm_challenges").fetchone()

    class TestOnlyWriterLock:
        def __init__(self, database):
            self.database = database

        def execute(self, statement, *parameters):
            result = self.database.execute(statement, *parameters)
            if statement == "BEGIN IMMEDIATE":
                # Model a writer-lock wait that ends after the authority/time
                # changed. The real SQLite writer lock has been acquired here.
                if change == "close_issuer":
                    item.harness.issuer.close()
                else:
                    monkeypatch.setattr(module, "_utc_now", lambda: NOW + timedelta(days=8))
            return result

    @contextmanager
    def delayed_connect(store):
        with original_connect(store) as database:
            yield TestOnlyWriterLock(database) if store is item.pending else database

    monkeypatch.setattr(module.ProductionTPMChallengeStore, "_connect", delayed_connect)
    expected = "STORE_REQUIRED" if change == "close_issuer" else "EXPIRED"
    with pytest.raises(ValueError, match=expected):
        if operation == "issue":
            item.pending.issue(
                *item.raw_items[:2],
                pdsa_challenge=item.pdsa,
                pdsa_store=item.pdsa_store,
                endorsement=item.endorsement,
                context=item.context,
            )
        else:
            module.ProductionTPMEnrollmentVerifier().verify(*item.raw_items, **item.arguments)
    with sqlite3.connect(item.pending._path) as database:
        assert database.execute("SELECT * FROM tpm_challenges").fetchone() == before
