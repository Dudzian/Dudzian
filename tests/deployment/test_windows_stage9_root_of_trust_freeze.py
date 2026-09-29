from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from deployment.windows_stage9_policy_material import (
    PolicyVectorError,
    canonical_digest,
    canonical_json_bytes,
)
from deployment.windows_stage9_root_of_trust_freeze import (
    ENROLLMENT_DOMAIN,
    RELEASE_DOMAIN,
    REVOCATION_DOMAIN,
    TrustedRetainedRevocationHead,
    VerifiedReleasePolicyV1,
    VerifiedRevocationStateV1,
    build_frozen_manifest,
    build_unprovisioned_freeze_manifest,
    migration_allowed,
    load_pinned_product_release_root_v1,
    load_current_trusted_retained_revocation_head,
    make_test_only_retained_revocation_head,
    ProductionRevocationHeadCustodyReader,
    source_revision,
    stage9_revocation_genesis_v1,
    validate_tpmt_public,
    verify_freeze_manifest,
    verify_pdsa_enrollment_package,
    verify_revocation_state,
    verify_signed_release_policy,
    verify_test_only_signed_release_policy,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/windows_stage9_policy_vector_v1.json"
NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)
GENESIS_REVOCATION_HEAD = stage9_revocation_genesis_v1()


def _key(seed: int):
    private = Ed25519PrivateKey.from_private_bytes(bytes([seed]) * 32)
    public = (
        private.public_key()
        .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        .hex()
    )
    return private, public


def _pinned_root(roots, *, purpose="TEST_ONLY", environment="UNIT_TEST_ONLY"):
    return load_pinned_product_release_root_v1(
        [
            {
                "key_id": name,
                "algorithm": "Ed25519",
                "encoding": "RFC8032_RAW_32_BYTES_LOWER_HEX",
                "public_key_hex": pair[1],
            }
            for name, pair in roots.items()
        ],
        purpose=purpose,
        environment=environment,
    )


class _TestProtectedCustodyReader(ProductionRevocationHeadCustodyReader):
    def __init__(self, serialized_record):
        self.serialized_record = serialized_record

    def read_current_authenticated_record(self):
        return self.serialized_record


def _release_material():
    release = deepcopy(json.loads(FIXTURE.read_text())["release_policy"])
    roots = {name: _key(seed) for name, seed in (("root-a", 1), ("root-b", 2), ("root-c", 3))}
    pdsa = {name: _key(seed) for name, seed in (("pdsa-a", 11), ("pdsa-b", 12), ("pdsa-c", 13))}
    release["product_release_root"].update(
        encoding="RFC8032_RAW_32_BYTES_LOWER_HEX",
        key_ids=list(roots),
        public_keys_hex=[pair[1] for pair in roots.values()],
    )
    release["pdsa_verification_key_set"] = [
        {
            "key_id": name,
            "algorithm": "Ed25519",
            "encoding": "RFC8032_RAW_32_BYTES_LOWER_HEX",
            "public_key_hex": pair[1],
        }
        for name, pair in pdsa.items()
    ]
    release["pdsa_verification_keys"] = [pair[1] for pair in pdsa.values()]
    release["production_contract"] = {
        "release_version": 1,
        "pdsa_threshold": 2,
        "pdsa_key_ids": list(pdsa),
        "valid_from": "2026-01-01T00:00:00Z",
        "valid_until": "2027-01-01T00:00:00Z",
        "rotation": "NEWER_ROOT_QUORUM_AND_MONOTONIC_VERSION",
        "revocation": "SIGNED_APPEND_ONLY_DENY_LIST",
        "compromise_recovery": "OFFLINE_BREAK_GLASS_QUORUM_NO_DOWNGRADE",
    }
    return release, roots, pdsa


def _signed_release(release=None, roots=None, signer_ids=("root-a", "root-b"), threshold=2):
    if release is None:
        release, roots, pdsa = _release_material()
    else:
        pdsa = None
    digest = canonical_digest(release)
    envelope = {
        "schema": "CryptoHunter.SignedReleasePolicyV1",
        "version": 1,
        "serialization_profile": "CRYPTOHUNTER_CANONICAL_JSON_V1",
        "payload": release,
        "payload_digest": digest.hex(),
        "signature_profile": "Ed25519-SHA256-DIGEST-CH-STAGE9-RELEASE-V1",
        "threshold": threshold,
        "signatures": [
            {"signer_id": name, "signature_hex": roots[name][0].sign(RELEASE_DOMAIN + digest).hex()}
            for name in signer_ids
        ],
    }
    return envelope, roots, pdsa


def _revocation_envelope(
    roots,
    *,
    retained=GENESIS_REVOCATION_HEAD,
    revoked_root=(),
    revoked_pdsa=(),
    signer_ids=("root-a", "root-b"),
    sequence=None,
    previous_digest=None,
):
    payload = {
        "schema": "CryptoHunter.Stage9RevocationPayloadV1",
        "version": 1,
        "sequence": retained.sequence + 1 if sequence is None else sequence,
        "previous_state_digest": retained.state_digest
        if previous_digest is None
        else previous_digest,
        "revoked_root_signer_ids": sorted(revoked_root),
        "revoked_pdsa_signer_ids": sorted(revoked_pdsa),
        "effective_at": "2026-01-01T00:00:00Z",
        "authority": "PRODUCT_RELEASE_ROOT_QUORUM",
    }
    digest = canonical_digest(payload)
    return {
        "schema": "CryptoHunter.Stage9RevocationStateV1",
        "version": 1,
        "serialization_profile": "CRYPTOHUNTER_CANONICAL_JSON_V1",
        "payload": payload,
        "payload_digest": digest.hex(),
        "signature_profile": "Ed25519-SHA256-DIGEST-CH-STAGE9-REVOCATION-V1",
        "threshold": 2,
        "signatures": [
            {
                "signer_id": name,
                "signature_hex": roots[name][0].sign(REVOCATION_DOMAIN + digest).hex(),
            }
            for name in signer_ids
        ],
    }


def _verified_revocation(roots, **kwargs):
    retained = kwargs.get("retained", GENESIS_REVOCATION_HEAD)
    envelope = _revocation_envelope(roots, **kwargs)
    return verify_revocation_state(
        canonical_json_bytes(envelope),
        _pinned_root(roots),
        retained_head=retained,
        verification_time=NOW,
    )


def _retained_head(state):
    record = {
        "schema": "CryptoHunter.Stage9RevocationRetainedHeadV1",
        "version": 1,
        "sequence": state.sequence,
        "state_digest": state.state_digest,
        "revoked_root_signer_ids": sorted(state.revoked_root_signer_ids),
        "revoked_pdsa_signer_ids": sorted(state.revoked_pdsa_signer_ids),
        "source": "TEST_ONLY_CUSTODY_SIMULATION",
    }
    serialized = canonical_json_bytes(record)
    return make_test_only_retained_revocation_head(serialized)


def _verified(release=None, roots=None, *, verification_time=NOW, revocations=None, **revoked):
    envelope, roots, pdsa = _signed_release(release, roots)
    if revocations is None:
        revocations = _verified_revocation(roots, **revoked)
    context = verify_test_only_signed_release_policy(
        canonical_json_bytes(envelope),
        _pinned_root(roots),
        verification_time=verification_time,
        revocations=revocations,
    )
    return context, pdsa


def _package(context, pdsa, signer_ids=("pdsa-a", "pdsa-b")):
    source = json.loads(FIXTURE.read_text())["enrollment_policy_material"]["k_psa"]
    parsed = validate_tpmt_public(source["public_hex"], context.k_psa_profile)
    payload = {
        "schema": "CryptoHunter.PDSAEnrollmentPayloadV1",
        "version": 1,
        "purpose": "TEST_ONLY",
        "release_policy_digest": context.payload_digest,
        "enrollment_id": "enroll-1",
        "device_identity_binding": "22" * 32,
        "target_tpm": {
            "ek_certificate_sha256": "31" * 32,
            "ak_public_sha256": "32" * 32,
            "ak_attestation_sha256": "33" * 32,
        },
        "k_psa": {
            "key_id": "psa-device-1",
            "tpmt_public_hex": source["public_hex"],
            "name_hex": parsed.name,
            "creation_attestation_sha256": "41" * 32,
            "provenance": "TEST_FIXTURE",
        },
        "generation": "13",
        "state_enrollment_digest": "51" * 32,
    }
    digest = canonical_digest(payload)
    return {
        "schema": "CryptoHunter.PDSAEnrollmentPackageV1",
        "version": 1,
        "serialization_profile": "CRYPTOHUNTER_CANONICAL_JSON_V1",
        "payload": payload,
        "payload_digest": digest.hex(),
        "signature_profile": "Ed25519-SHA256-DIGEST-CH-STAGE9-PDSA-ENROLLMENT-V1",
        "threshold": context.pdsa_threshold,
        "signatures": [
            {
                "signer_id": name,
                "signature_hex": pdsa[name][0].sign(ENROLLMENT_DOMAIN + digest).hex(),
            }
            for name in signer_ids
        ],
    }


def _resign_package(package, pdsa, signer_ids=("pdsa-a", "pdsa-b")):
    digest = canonical_digest(package["payload"])
    package["payload_digest"] = digest.hex()
    package["signatures"] = [
        {"signer_id": name, "signature_hex": pdsa[name][0].sign(ENROLLMENT_DOMAIN + digest).hex()}
        for name in signer_ids
    ]


def test_release_threshold_is_only_from_signed_payload_and_cannot_be_overridden():
    release, roots, _ = _release_material()
    envelope, _, _ = _signed_release(release, roots, signer_ids=("root-a",), threshold=1)
    assert release["product_release_root"]["threshold"] == 2
    with pytest.raises(PolicyVectorError, match="threshold"):
        verify_test_only_signed_release_policy(
            canonical_json_bytes(envelope),
            _pinned_root(roots),
            verification_time=NOW,
            revocations=_verified_revocation(roots),
        )
    assert (
        "expected_threshold"
        not in __import__("inspect").signature(verify_signed_release_policy).parameters
    )


def test_production_api_has_fixed_purpose_and_rejects_test_ceremony_fixture():
    envelope, roots, _ = _signed_release()
    with pytest.raises(PolicyVectorError, match="purpose"):
        verify_signed_release_policy(
            canonical_json_bytes(envelope),
            _pinned_root(roots),
            verification_time=NOW,
            revocations=_verified_revocation(roots),
        )
    assert (
        "expected_purpose"
        not in __import__("inspect").signature(verify_signed_release_policy).parameters
    )


@pytest.mark.parametrize(
    "mutation", ["payload_digest", "duplicate_signer", "unordered_signer", "signature", "version"]
)
def test_release_envelope_remains_fail_closed(mutation):
    envelope, roots, _ = _signed_release()
    if mutation == "payload_digest":
        envelope["payload_digest"] = "00" * 32
    elif mutation == "duplicate_signer":
        envelope["signatures"][1]["signer_id"] = "root-a"
    elif mutation == "unordered_signer":
        envelope["signatures"].reverse()
    elif mutation == "signature":
        envelope["signatures"][0]["signature_hex"] = "00" * 64
    else:
        envelope["version"] = 2
    with pytest.raises(PolicyVectorError):
        verify_test_only_signed_release_policy(
            canonical_json_bytes(envelope),
            _pinned_root(roots),
            verification_time=NOW,
            revocations=_verified_revocation(roots),
        )


def test_release_rejects_noncanonical_serialization():
    envelope, roots, _ = _signed_release()
    with pytest.raises(PolicyVectorError, match="noncanonical"):
        verify_test_only_signed_release_policy(
            json.dumps(envelope, indent=2).encode(),
            _pinned_root(roots),
            verification_time=NOW,
            revocations=_verified_revocation(roots),
        )


def test_release_exact_root_set_and_recovery_bytes_are_enforced():
    release, roots, _ = _release_material()
    envelope, _, _ = _signed_release(release, roots)
    wrong_roots = dict(roots)
    wrong_roots["root-a"] = _key(41)
    with pytest.raises(PolicyVectorError, match="payload-bound"):
        verify_test_only_signed_release_policy(
            canonical_json_bytes(envelope),
            _pinned_root(wrong_roots),
            verification_time=NOW,
            revocations=_verified_revocation(roots),
        )
    bad = deepcopy(release)
    raw = bytearray.fromhex(bad["k_recovery"]["public_hex"])
    raw[16:18] = b"\x00\x04"
    bad["k_recovery"]["public_hex"] = raw.hex()
    signed, _, _ = _signed_release(bad, roots)
    with pytest.raises(PolicyVectorError, match="TPMT_PUBLIC"):
        verify_test_only_signed_release_policy(
            canonical_json_bytes(signed),
            _pinned_root(roots),
            verification_time=NOW,
            revocations=_verified_revocation(roots),
        )


@pytest.mark.parametrize(
    "when", [datetime(2025, 12, 31, tzinfo=timezone.utc), datetime(2027, 1, 1, tzinfo=timezone.utc)]
)
def test_release_rejects_outside_validity_window(when):
    envelope, roots, _ = _signed_release()
    with pytest.raises(PolicyVectorError, match="validity"):
        verify_test_only_signed_release_policy(
            canonical_json_bytes(envelope),
            _pinned_root(roots),
            verification_time=when,
            revocations=_verified_revocation(roots),
        )


@pytest.mark.parametrize(
    "start,end",
    [
        ("2027-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        ("2026-01-01 00:00:00Z", "2027-01-01T00:00:00Z"),
    ],
)
def test_release_rejects_invalid_interval_or_timestamp(start, end):
    release, roots, _ = _release_material()
    release["production_contract"].update(valid_from=start, valid_until=end)
    envelope, _, _ = _signed_release(release, roots)
    with pytest.raises(PolicyVectorError):
        verify_test_only_signed_release_policy(
            canonical_json_bytes(envelope),
            _pinned_root(roots),
            verification_time=NOW,
            revocations=_verified_revocation(roots),
        )


def test_release_accepts_inside_window_and_rejects_revoked_root():
    context, _ = _verified()
    assert context.valid_from <= NOW < context.valid_until
    envelope, roots, _ = _signed_release()
    revoked = _verified_revocation(roots, revoked_root=("root-a",))
    with pytest.raises(PolicyVectorError, match="revoked"):
        verify_test_only_signed_release_policy(
            canonical_json_bytes(envelope),
            _pinned_root(roots),
            verification_time=NOW,
            revocations=revoked,
        )


def test_revocation_signature_failure_and_unauthenticated_object_are_rejected():
    envelope, roots, _ = _signed_release()
    revocation = _revocation_envelope(roots)
    revocation["signatures"][0]["signature_hex"] = "00" * 64
    with pytest.raises(PolicyVectorError, match="signature"):
        verify_revocation_state(
            canonical_json_bytes(revocation),
            _pinned_root(roots),
            retained_head=GENESIS_REVOCATION_HEAD,
            verification_time=NOW,
        )
    with pytest.raises(TypeError, match="only be created"):
        VerifiedRevocationStateV1()
    forged = object.__new__(VerifiedRevocationStateV1)
    with pytest.raises(PolicyVectorError, match="forged"):
        verify_test_only_signed_release_policy(
            canonical_json_bytes(envelope),
            _pinned_root(roots),
            verification_time=NOW,
            revocations=forged,
        )


def test_revocation_rollback_and_unrevocation_are_rejected():
    _, roots, _ = _signed_release()
    first = _verified_revocation(roots, revoked_pdsa=("pdsa-a",))
    retained = _retained_head(first)
    rollback = _revocation_envelope(roots, retained=retained, sequence=first.sequence)
    with pytest.raises(PolicyVectorError, match="sequence"):
        verify_revocation_state(
            canonical_json_bytes(rollback),
            _pinned_root(roots),
            retained_head=retained,
            verification_time=NOW,
        )
    removal = _revocation_envelope(roots, retained=retained, revoked_pdsa=())
    with pytest.raises(PolicyVectorError, match="unrevocation"):
        verify_revocation_state(
            canonical_json_bytes(removal),
            _pinned_root(roots),
            retained_head=retained,
            verification_time=NOW,
        )


def test_revocation_signed_by_different_keys_under_same_ids_cannot_bind_release():
    release, real_roots, _ = _release_material()
    evil_roots = {
        name: _key(seed) for name, seed in (("root-a", 41), ("root-b", 42), ("root-c", 43))
    }
    evil_revocation = _verified_revocation(evil_roots)
    envelope, _, _ = _signed_release(release, real_roots)
    with pytest.raises(PolicyVectorError, match="identity mismatch"):
        verify_test_only_signed_release_policy(
            canonical_json_bytes(envelope),
            _pinned_root(real_roots),
            verification_time=NOW,
            revocations=evil_revocation,
        )


def test_fake_caller_created_retained_head_and_seal_forgery_are_rejected():
    from deployment import windows_stage9_root_of_trust_freeze as freeze

    with pytest.raises(TypeError, match="genesis or custody"):
        TrustedRetainedRevocationHead(0, "00" * 32, frozenset(), frozenset())
    fake = object.__new__(TrustedRetainedRevocationHead)
    object.__setattr__(fake, "sequence", 0)
    object.__setattr__(fake, "state_digest", "00" * 32)
    object.__setattr__(fake, "revoked_root_signer_ids", frozenset())
    object.__setattr__(fake, "revoked_pdsa_signer_ids", frozenset())
    object.__setattr__(fake, "serialized_record", b"{}")
    object.__setattr__(fake, "trust_domain", "PINNED_GENESIS")
    object.__setattr__(fake, "custody_reader", None)
    _, roots, _ = _signed_release()
    with pytest.raises(PolicyVectorError, match="fake revocation genesis"):
        verify_revocation_state(
            canonical_json_bytes(_revocation_envelope(roots)),
            _pinned_root(roots),
            retained_head=fake,
            verification_time=NOW,
        )
    assert not hasattr(freeze, "_VERIFIED_SEAL")
    forged = object.__new__(VerifiedReleasePolicyV1)
    with pytest.raises(AttributeError):
        object.__setattr__(forged, "_seal", object())
    with pytest.raises(PolicyVectorError, match="forged"):
        build_frozen_manifest(forged)


def test_caller_computed_self_hash_cannot_authenticate_forged_old_head():
    fake_record = {
        "schema": "CryptoHunter.Stage9RevocationRetainedHeadV1",
        "version": 1,
        "sequence": 4,
        "state_digest": "44" * 32,
        "revoked_root_signer_ids": [],
        "revoked_pdsa_signer_ids": [],
        "source": "CUSTODY_AUTHENTICATED_STORE",
    }
    serialized = canonical_json_bytes(fake_record)
    caller_computed_digest = hashlib.sha256(serialized).hexdigest()
    fake_business_input = type(
        "CallerProvidedRecordAndDigest",
        (),
        {"serialized_record": serialized, "expected_digest": caller_computed_digest},
    )()
    with pytest.raises(PolicyVectorError, match="configured custody reader"):
        load_current_trusted_retained_revocation_head(fake_business_input)
    assert caller_computed_digest == hashlib.sha256(serialized).hexdigest()


def test_actual_custody_head_prevents_revoked_root_rollback_successor():
    _, roots, _ = _signed_release()
    actual_record = {
        "schema": "CryptoHunter.Stage9RevocationRetainedHeadV1",
        "version": 1,
        "sequence": 8,
        "state_digest": "88" * 32,
        "revoked_root_signer_ids": ["root-a"],
        "revoked_pdsa_signer_ids": [],
        "source": "CUSTODY_AUTHENTICATED_STORE",
    }
    reader = _TestProtectedCustodyReader(canonical_json_bytes(actual_record))
    actual_head = load_current_trusted_retained_revocation_head(reader)
    successor = _revocation_envelope(
        roots,
        retained=actual_head,
        revoked_root=("root-a",),
        signer_ids=("root-a", "root-b"),
    )
    with pytest.raises(PolicyVectorError, match="revoked signer"):
        verify_revocation_state(
            canonical_json_bytes(successor),
            _pinned_root(roots, purpose="PRODUCTION", environment="PRODUCTION_CEREMONY"),
            retained_head=actual_head,
            verification_time=NOW,
        )


@pytest.mark.parametrize(
    "change", ["cardinality", "order", "duplicate_id", "duplicate_public", "threshold"]
)
def test_pdsa_key_mapping_and_threshold_are_executable(change):
    release, roots, _ = _release_material()
    if change == "cardinality":
        release["pdsa_verification_keys"].pop()
    elif change == "order":
        release["pdsa_verification_key_set"].reverse()
    elif change == "duplicate_id":
        release["pdsa_verification_key_set"][1]["key_id"] = "pdsa-a"
    elif change == "duplicate_public":
        release["pdsa_verification_key_set"][1]["public_key_hex"] = release[
            "pdsa_verification_key_set"
        ][0]["public_key_hex"]
    else:
        release["production_contract"]["pdsa_threshold"] = 4
    envelope, _, _ = _signed_release(release, roots)
    with pytest.raises(PolicyVectorError):
        verify_test_only_signed_release_policy(
            canonical_json_bytes(envelope),
            _pinned_root(roots),
            verification_time=NOW,
            revocations=_verified_revocation(roots),
        )


def test_evil_pdsa_map_cannot_be_supplied_to_production_verifier():
    context, _ = _verified()
    evil = {name: _key(seed) for name, seed in (("evil-a", 21), ("evil-b", 22))}
    package = _package(context, evil, tuple(evil))
    with pytest.raises(PolicyVectorError, match="unknown signer"):
        verify_pdsa_enrollment_package(
            canonical_json_bytes(package), context, device_identity_binding="22" * 32
        )
    assert (
        "pdsa_keys"
        not in __import__("inspect").signature(verify_pdsa_enrollment_package).parameters
    )


def test_verified_release_authority_projection_is_deeply_immutable():
    context, _ = _verified()
    evil_public = _key(31)[1]
    with pytest.raises(TypeError):
        context.pdsa_keys["pdsa-a"] = evil_public
    with pytest.raises(TypeError):
        context.k_psa_profile["curve"] = "ATTACKER_CURVE"
    with pytest.raises(TypeError):
        context.policy_refs["normal_hex"] = "00" * 32
    with pytest.raises(TypeError):
        context.branch_order[0] = "BOOTSTRAP_RECOVERY"
    with pytest.raises(TypeError):
        context.release_version = 999
    with pytest.raises(TypeError):
        context.purpose = "PRODUCTION"
    assert not hasattr(context, "payload")
    assert not hasattr(context, "envelope")


def test_post_verification_pdsa_replacement_cannot_authorize_attacker_signature():
    context, pdsa = _verified()
    package = _package(context, pdsa)
    evil_private, evil_public = _key(31)
    with pytest.raises(TypeError):
        context.pdsa_keys["pdsa-a"] = evil_public
    digest = canonical_digest(package["payload"])
    package["signatures"] = [
        {
            "signer_id": "pdsa-a",
            "signature_hex": evil_private.sign(ENROLLMENT_DOMAIN + digest).hex(),
        },
        {
            "signer_id": "pdsa-b",
            "signature_hex": pdsa["pdsa-b"][0].sign(ENROLLMENT_DOMAIN + digest).hex(),
        },
    ]
    with pytest.raises(PolicyVectorError, match="signature"):
        verify_pdsa_enrollment_package(
            canonical_json_bytes(package), context, device_identity_binding="22" * 32
        )


def test_manually_forged_verified_release_context_is_rejected_downstream():
    with pytest.raises(TypeError, match="only be created"):
        VerifiedReleasePolicyV1()
    forged = object.__new__(VerifiedReleasePolicyV1)
    with pytest.raises(PolicyVectorError, match="forged"):
        build_frozen_manifest(forged)
    with pytest.raises(PolicyVectorError, match="forged"):
        verify_pdsa_enrollment_package(b"{}", forged, device_identity_binding="22" * 32)


def test_pdsa_threshold_downgrade_and_revocation_are_rejected():
    context, pdsa = _verified()
    package = _package(context, pdsa, ("pdsa-a",))
    package["threshold"] = 1
    with pytest.raises(PolicyVectorError, match="threshold"):
        verify_pdsa_enrollment_package(
            canonical_json_bytes(package), context, device_identity_binding="22" * 32
        )
    revoked, pdsa = _verified(revoked_pdsa=("pdsa-a",))
    with pytest.raises(PolicyVectorError, match="revoked"):
        verify_pdsa_enrollment_package(
            canonical_json_bytes(_package(revoked, pdsa)),
            revoked,
            device_identity_binding="22" * 32,
        )


@pytest.mark.parametrize("mutation", ["duplicate", "unordered", "tampered", "version"])
def test_pdsa_signature_envelope_remains_fail_closed(mutation):
    context, pdsa = _verified()
    package = _package(context, pdsa)
    if mutation == "duplicate":
        package["signatures"][1]["signer_id"] = "pdsa-a"
    elif mutation == "unordered":
        package["signatures"].reverse()
    elif mutation == "tampered":
        package["signatures"][0]["signature_hex"] = "00" * 64
    else:
        package["version"] = 2
    with pytest.raises(PolicyVectorError):
        verify_pdsa_enrollment_package(
            canonical_json_bytes(package), context, device_identity_binding="22" * 32
        )


@pytest.mark.parametrize(
    "mutation", ["curve", "attributes", "auth_size", "scheme", "name_alg", "trailing", "arbitrary"]
)
def test_k_psa_tpmt_public_profile_is_fail_closed(mutation):
    context, pdsa = _verified()
    package = _package(context, pdsa)
    raw = bytearray.fromhex(package["payload"]["k_psa"]["tpmt_public_hex"])
    if mutation == "curve":
        raw[48:50] = b"\x00\x04"
    elif mutation == "attributes":
        raw[4:8] = b"\x00\x04\x00\xb3"
    elif mutation == "auth_size":
        raw[8:10] = b"\x00\x1f"
    elif mutation == "scheme":
        raw[44:46] = b"\x00\x14"
    elif mutation == "name_alg":
        raw[2:4] = b"\x00\x04"
    elif mutation == "trailing":
        raw += b"\x00"
    else:
        raw = bytearray(b"not-a-tpmt-public")
    value = raw.hex()
    package["payload"]["k_psa"]["tpmt_public_hex"] = value
    package["payload"]["k_psa"]["name_hex"] = (b"\x00\x0b" + hashlib.sha256(raw).digest()).hex()
    _resign_package(package, pdsa)
    with pytest.raises(PolicyVectorError):
        verify_pdsa_enrollment_package(
            canonical_json_bytes(package), context, device_identity_binding="22" * 32
        )


def test_enrollment_release_device_name_and_canonical_bindings():
    context, pdsa = _verified()
    package = _package(context, pdsa)
    verify_pdsa_enrollment_package(
        canonical_json_bytes(package), context, device_identity_binding="22" * 32
    )
    for field in ("release_policy_digest", "device_identity_binding", "name_hex"):
        changed = deepcopy(package)
        target = changed["payload"]["k_psa"] if field == "name_hex" else changed["payload"]
        target[field] = ("000b" + "00" * 32) if field == "name_hex" else "ff" * 32
        _resign_package(changed, pdsa)
        with pytest.raises(PolicyVectorError):
            verify_pdsa_enrollment_package(
                canonical_json_bytes(changed), context, device_identity_binding="22" * 32
            )
    with pytest.raises(PolicyVectorError, match="noncanonical"):
        verify_pdsa_enrollment_package(
            json.dumps(package, indent=2).encode(), context, device_identity_binding="22" * 32
        )


def test_unprovisioned_and_complete_test_only_frozen_manifests():
    release = json.loads(FIXTURE.read_text())["release_policy"]
    pending = build_unprovisioned_freeze_manifest(release)
    assert pending["status"] == "PRODUCTION_ROOT_MATERIAL_NOT_PROVISIONED"
    assert pending["product_release_root"]["public_keys_hex"] == []
    verify_freeze_manifest(pending, release_template=release)
    context, _ = _verified()
    frozen = build_frozen_manifest(context)
    assert frozen["status"] == "TEST_ONLY_ROOT_OF_TRUST_FROZEN"
    assert "PENDING_CEREMONY" not in json.dumps(frozen)
    verify_freeze_manifest(frozen, verified_release=context)
    assert source_revision() != "UNKNOWN"


@pytest.mark.parametrize("mutation", ["digest", "material_status", "root_key"])
def test_unprovisioned_manifest_rejects_partial_material(mutation):
    release = json.loads(FIXTURE.read_text())["release_policy"]
    manifest = build_unprovisioned_freeze_manifest(release)
    if mutation == "digest":
        manifest["pdsa_key_set_digest"] = "00" * 32
    elif mutation == "material_status":
        manifest["k_recovery"]["material_status"] = "PROVISIONED"
    else:
        manifest["product_release_root"]["key_ids"] = ["substitute"]
    with pytest.raises(PolicyVectorError):
        verify_freeze_manifest(manifest, release_template=release)


@pytest.mark.parametrize(
    "mutation",
    [
        "partial",
        "pending",
        "root",
        "root_digest",
        "pdsa",
        "recovery",
        "revocation_digest",
        "revocation_sequence",
    ],
)
def test_frozen_manifest_recomputes_every_authority_value(mutation):
    context, _ = _verified()
    manifest = build_frozen_manifest(context)
    if mutation == "partial":
        manifest.pop("pdsa_key_set_digest")
    elif mutation == "pending":
        manifest["signed_release_policy_digest"] = "PENDING_CEREMONY"
    elif mutation == "root":
        manifest["product_release_root"]["public_keys_hex"][0] = "00" * 32
    elif mutation == "root_digest":
        manifest["product_root_key_set_digest"] = "00" * 32
    elif mutation == "pdsa":
        manifest["pdsa_key_set_digest"] = "00" * 32
    elif mutation == "recovery":
        manifest["k_recovery"]["name"] = "000b" + "00" * 32
    elif mutation == "revocation_digest":
        manifest["revocation_state_digest"] = "00" * 32
    else:
        manifest["revocation_sequence"] += 1
    with pytest.raises(PolicyVectorError):
        verify_freeze_manifest(manifest, verified_release=context)


def test_source_revision_is_creation_provenance_not_current_checkout(monkeypatch):
    from deployment import windows_stage9_root_of_trust_freeze as freeze

    context, _ = _verified()
    revision_a = "a1" * 20
    manifest = build_frozen_manifest(context, artifact_source_revision=revision_a)
    monkeypatch.setattr(freeze, "source_revision", lambda repo=None: "UNKNOWN")
    verify_freeze_manifest(manifest, verified_release=context)
    assert manifest["source_revision"] == revision_a


ALLOWED_MIGRATIONS = {
    "RELEASE_POLICY_UPDATE": (
        "PRODUCT_RELEASE_ROOT_QUORUM",
        "RELEASE_AND_NV_HEAD_ADVANCE",
        "SIGNED_SUCCESSOR_RELEASE",
        "OFFLINE_RELEASE_CEREMONY",
    ),
    "PRODUCT_ROOT_ROTATION": (
        "CURRENT_PRODUCT_RELEASE_ROOT_QUORUM",
        "RELEASE_AND_REVOCATION_HEAD_ADVANCE",
        "DUAL_ROOT_CEREMONY_RECORD",
        "OFFLINE_BREAK_GLASS_CEREMONY",
    ),
    "PDSA_ROTATION": (
        "PRODUCT_RELEASE_ROOT_QUORUM",
        "RELEASE_VERSION_ADVANCE",
        "SIGNED_RELEASE_AND_REVOCATIONS",
        "OFFLINE_RELEASE_CEREMONY",
    ),
    "K_RECOVERY_ROTATION": (
        "PRODUCT_RELEASE_ROOT_QUORUM",
        "RELEASE_VERSION_ADVANCE",
        "TPMT_PUBLIC_NAMES_AND_CEREMONY",
        "OFFLINE_RECOVERY_KEY_CEREMONY",
    ),
    "K_PSA_REPLACEMENT": (
        "PDSA_THRESHOLD_RECOVERY",
        "DEVICE_GENERATION_ADVANCE",
        "EK_AK_CREATION_AND_PACKAGE",
        "PDSA_RECOVERY_ENROLLMENT",
    ),
    "TPM_REPLACEMENT": (
        "PDSA_THRESHOLD_AND_OPERATOR_RECOVERY",
        "DEVICE_LINEAGE_ADVANCE",
        "LOSS_REPLACEMENT_AND_PACKAGE",
        "DEVICE_REPROVISION",
    ),
    "NV_RECREATION": (
        "RECOVERY_AUTHORITY_AND_PDSA_POLICY",
        "RETAINED_HEAD_ADVANCE",
        "NV_NAMES_COUNTER_CPHASH_APPROVALS",
        "OFFLINE_NV_RECOVERY",
    ),
    "DEVICE_REPROVISION": (
        "PDSA_THRESHOLD",
        "ENROLLMENT_LINEAGE_ADVANCE",
        "SIGNED_RECOVERY_LINEAGE",
        "PDSA_RECOVERY_ENROLLMENT",
    ),
}


def _migration(case, values):
    authority, monotonic, audit, recovery = values
    return {
        "case": case,
        "authority": authority,
        "old_version": 4,
        "new_version": 5,
        "monotonic_evidence": monotonic,
        "audit_evidence": audit,
        "recovery_path": recovery,
    }


@pytest.mark.parametrize("case,values", ALLOWED_MIGRATIONS.items())
def test_each_allowed_migration_has_exact_positive_and_negative_contract(case, values):
    record = _migration(case, values)
    assert migration_allowed(record)
    assert not migration_allowed({**record, "authority": "ATTACKER"})
    assert not migration_allowed({**record, "new_version": 6})
    assert not migration_allowed({**record, "audit_evidence": "SOMETHING"})


DENIED_MIGRATIONS = {
    "SERIALIZATION_PROFILE_CHANGE": (
        "PRODUCT_RELEASE_ROOT_QUORUM",
        "SUPPORTED_PROFILE_SUCCESSOR",
        "COMPATIBILITY_APPROVAL_AND_VECTORS",
        "UPGRADE_VERIFIER_OUT_OF_BAND",
    ),
    "ROLLBACK_ATTEMPT": (
        "NONE",
        "NEVER",
        "ROLLBACK_REJECTION_EVENT",
        "RESTORE_CURRENT_EVIDENCE",
    ),
    "UNKNOWN_FUTURE_VERSION": (
        "NONE",
        "NEVER",
        "UNKNOWN_VERSION_REJECTION_EVENT",
        "UPGRADE_VERIFIER_OUT_OF_BAND",
    ),
}


@pytest.mark.parametrize("case,values", DENIED_MIGRATIONS.items())
def test_denied_migration_cases_are_always_denied(case, values):
    record = _migration(case, values)
    assert not migration_allowed(record)
    attacker = {
        **record,
        "authority": "ATTACKER",
        "old_version": 4,
        "new_version": 5,
        "monotonic_evidence": "SOMETHING",
        "audit_evidence": "SOMETHING",
        "recovery_path": "SOMETHING",
    }
    assert not migration_allowed(attacker)
