"""TEST_ONLY cryptographic simulations of the production package verifier.

The existing #3067 harness explicitly substitutes the public trust loader and
native hardware boundary. Its request PoP, exchange, custody and capability
provenance checks remain active; no production private key is retained here.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.licensing.external_provisioning import (
    PACKAGE_FIELDS,
    PACKAGE_SCHEMA,
    PDSA_DOMAIN,
    ProductionProvisioningPackageVerifier,
    ProductionVerifierUnavailable,
    ProvisioningError,
)
from deployment import windows_stage9_production_trust as trust
from tests.licensing import test_production_pre_enrollment as pre_enrollment_tests
from tests.licensing.test_pdsa_enrollment_challenge import NOW

challenge_harness = pre_enrollment_tests.challenge_harness
integration = pre_enrollment_tests.integration
_DEFAULT_PACKAGE = object()


def _signed(item, payload=None):
    value = dict(item.payload if payload is None else payload)
    message = PDSA_DOMAIN + hashlib.sha256(canonical_json_bytes(value)).digest()
    return canonical_json_bytes({"payload": value, "signatures": item.authority.sign(message)})


@pytest.fixture
def package(integration, monkeypatch):
    item = integration
    # Explicit TEST_ONLY current-trust substitution, matching #3067's harness.
    monkeypatch.setattr(
        trust,
        "require_current_production_trust_context",
        trust.require_verified_production_trust_context,
    )
    authenticated = pre_enrollment_tests._authenticate(item)
    request = item.request.document
    payload = {
        "schema_version": PACKAGE_SCHEMA,
        "environment": "PRODUCTION",
        "pdsa_trust_domain": request["pdsa_trust_domain"],
        "provisioning_subject_id": "psub_019ba13c-5c00-7000-8000-000000000001",
        "enrollment_reference": "penr_019ba13c-5c00-7000-8000-000000000002",
        "pdsa_challenge_id": request["pdsa_challenge_id"],
        "pdsa_challenge_digest_sha256": request["pdsa_challenge_digest_sha256"],
        "pre_enrollment_request_digest_sha256": item.request.digest_sha256,
        "verified_tpm_exchange_reference": request["verified_tpm_exchange_reference"],
        "verified_tpm_public_projection_id": request["verified_tpm_public_projection_id"],
        "target_tpm_ek_public_digest": request["ek_public_digest"],
        "target_tpm_ak_public_digest": request["ak_public_digest"],
        "pre_enrollment_public_key_algorithm_profile": request[
            "pre_enrollment_public_key_algorithm_profile"
        ],
        "pre_enrollment_public_key_fingerprint_sha256": request[
            "pre_enrollment_public_key_fingerprint_sha256"
        ],
        "authorization_generation": 1,
        "authorization_version": 1,
        "issued_at_utc": NOW.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expires_at_utc": item.tpm_challenge.document["expires_at_utc"],
        "release_policy_digest_sha256": request["release_policy_digest_sha256"],
        "release_policy_generation": request["release_policy_generation"],
        "product_profile": request["product_profile"],
        "predecessor_package_digest_or_null": None,
        "lineage_generation": 1,
    }
    result = SimpleNamespace(
        authority=item.authority,
        authenticated=authenticated,
        pending=item.pending,
        payload=payload,
        device=request["pre_enrollment_public_key_fingerprint_sha256"],
        verifier=ProductionProvisioningPackageVerifier(item.authority.context),
    )
    result.raw = _signed(result)
    return result


def _verify(item, raw=_DEFAULT_PACKAGE, *, now=NOW):
    return item.verifier.verify(
        item.raw if raw is _DEFAULT_PACKAGE else raw, expected_device_key=item.device, now=now
    )


@pytest.mark.parametrize("pair", [(0, 1), (0, 2), (1, 2)], ids=["K1+K2", "K1+K3", "K2+K3"])
def test_all_distinct_authorized_quorum_pairs_verify(package, pair):
    item = package
    key_ids = sorted(item.authority.keys)
    message = PDSA_DOMAIN + hashlib.sha256(canonical_json_bytes(item.payload)).digest()
    signatures = [
        {
            "key_id": key_ids[index],
            "algorithm": "Ed25519",
            "signature_hex": item.authority.keys[key_ids[index]].sign(message).hex(),
        }
        for index in pair
    ]
    raw = canonical_json_bytes({"payload": item.payload, "signatures": signatures})
    assert _verify(item, raw).canonical_package == raw
    assert (
        item.verifier.verify_authenticated(raw, item.authenticated, now=NOW).canonical_package
        == raw
    )


def test_exact_frozen_initial_package_and_authenticated_targets_are_verified(package):
    verified = _verify(package)
    assert set(verified.payload) == PACKAGE_FIELDS
    assert len(verified.payload) == 23
    assert verified.canonical_package == package.raw
    assert verified.package_digest_sha256 == hashlib.sha256(package.raw).hexdigest()
    assert (
        package.verifier.verify_authenticated(package.raw, package.authenticated, now=NOW)
        == verified
    )


@pytest.mark.parametrize("field", sorted(PACKAGE_FIELDS))
def test_every_missing_payload_field_is_rejected(package, field):
    value = json.loads(package.raw)
    del value["payload"][field]
    with pytest.raises(ProvisioningError, match="PACKAGE_PAYLOAD_SCHEMA_MISMATCH"):
        _verify(package, canonical_json_bytes(value))


@pytest.mark.parametrize(
    "field",
    ["prvop", "provisioning_operation_id", "account_id", "logical_operation_id", "membership"],
)
def test_unknown_operation_or_membership_fields_are_rejected(package, field):
    value = json.loads(package.raw)
    value["payload"][field] = "caller-controlled"
    with pytest.raises(ProvisioningError, match="PACKAGE_PAYLOAD_SCHEMA_MISMATCH"):
        _verify(package, canonical_json_bytes(value))


@pytest.mark.parametrize(
    ("field", "replacement", "reason"),
    [
        ("environment", "TEST_ONLY", "INVALID_ENVIRONMENT"),
        ("pdsa_trust_domain", "TEST_ONLY_2_OF_3_ED25519", "INVALID_PDSA_TRUST_DOMAIN"),
        ("product_profile", "TEST_ONLY", "INVALID_PRODUCT_PROFILE"),
        ("release_policy_digest_sha256", "00" * 32, "RELEASE_POLICY_DIGEST_MISMATCH"),
        ("release_policy_generation", 2, "RELEASE_POLICY_GENERATION_MISMATCH"),
        ("authorization_generation", 2, "INVALID_PRODUCTION_PACKAGE_PROFILE"),
        ("authorization_version", 2, "INVALID_PRODUCTION_PACKAGE_PROFILE"),
        ("lineage_generation", 2, "INVALID_PRODUCTION_PACKAGE_PROFILE"),
        ("predecessor_package_digest_or_null", "00" * 32, "INVALID_PRODUCTION_PACKAGE_PROFILE"),
        ("provisioning_subject_id", "psub_caller", "INVALID_PRODUCTION_PACKAGE_PROFILE"),
        ("enrollment_reference", "auth-caller", "INVALID_PRODUCTION_PACKAGE_PROFILE"),
        ("pdsa_challenge_id", "pchal_caller", "INVALID_PRODUCTION_PACKAGE_PROFILE"),
        (
            "verified_tpm_exchange_reference",
            "exchange-caller",
            "INVALID_PRODUCTION_PACKAGE_PROFILE",
        ),
        (
            "verified_tpm_public_projection_id",
            "projection-caller",
            "INVALID_PRODUCTION_PACKAGE_PROFILE",
        ),
        ("issued_at_utc", "2026-10-06T11:59:59.000001Z", "INVALID_PRODUCTION_PACKAGE_PROFILE"),
    ],
)
def test_signed_wrong_production_profile_is_rejected(package, field, replacement, reason):
    changed = package.payload | {field: replacement}
    with pytest.raises(ProvisioningError, match=reason):
        _verify(package, _signed(package, changed))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("pdsa_challenge_id", "pchal_019ba13c-5c00-7000-8000-000000000099"),
        ("pdsa_challenge_digest_sha256", "00" * 32),
        ("pre_enrollment_request_digest_sha256", "00" * 32),
        ("verified_tpm_exchange_reference", "00" * 32),
        ("verified_tpm_public_projection_id", "00" * 32),
        ("target_tpm_ek_public_digest", "00" * 32),
        ("target_tpm_ak_public_digest", "00" * 32),
        ("pre_enrollment_public_key_algorithm_profile", "ECDSA-P384-SHA384"),
        ("pre_enrollment_public_key_fingerprint_sha256", "00" * 32),
        ("release_policy_digest_sha256", "00" * 32),
        ("release_policy_generation", 2),
        ("expires_at_utc", (NOW + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")),
    ],
)
def test_each_independently_resigned_authenticated_target_mutation_is_rejected(
    package, field, replacement
):
    raw = _signed(package, package.payload | {field: replacement})
    with pytest.raises(ProvisioningError):
        package.verifier.verify_authenticated(raw, package.authenticated, now=NOW)


@pytest.mark.parametrize(
    "mutation", ["unknown", "test_only", "duplicate", "invalid", "reordered", "one", "three"]
)
def test_signature_block_mutations_are_rejected_deterministically(package, mutation):
    value = json.loads(package.raw)
    signatures = value["signatures"]
    if mutation in {"unknown", "test_only"}:
        signatures[0]["key_id"] = "TEST_ONLY_UNKNOWN" if mutation == "test_only" else "unknown"
    elif mutation == "duplicate":
        signatures[1] = dict(signatures[0])
    elif mutation == "invalid":
        signatures[0]["signature_hex"] = "00" * 64
    elif mutation == "reordered":
        signatures.reverse()
    elif mutation == "one":
        signatures.pop()
    else:
        key_id = sorted(package.authority.keys)[2]
        message = PDSA_DOMAIN + hashlib.sha256(canonical_json_bytes(value["payload"])).digest()
        signatures.append(
            {
                "algorithm": "Ed25519",
                "key_id": key_id,
                "signature_hex": package.authority.keys[key_id].sign(message).hex(),
            }
        )
    raw = canonical_json_bytes(value)
    reasons = []
    for _ in range(2):
        with pytest.raises(ProvisioningError) as rejected:
            _verify(package, raw)
        reasons.append(str(rejected.value))
    assert reasons[0] == reasons[1]


@pytest.mark.parametrize("signature", [None, 7, [], "AA" * 64, "00 " * 64, "00" * 63, "gg" * 64])
def test_malformed_signature_encoding_fails_closed(package, signature):
    value = json.loads(package.raw)
    value["signatures"][0]["signature_hex"] = signature
    with pytest.raises(ProvisioningError, match="INVALID_PDSA_SIGNATURE"):
        _verify(package, canonical_json_bytes(value))


@pytest.mark.parametrize("key_id", [None, 7, [], {}, "", "x" * 257])
def test_malformed_signer_identifier_fails_closed(package, key_id):
    value = json.loads(package.raw)
    value["signatures"][0]["key_id"] = key_id
    with pytest.raises(ProvisioningError, match="INVALID_PDSA_SIGNER_ID"):
        _verify(package, canonical_json_bytes(value))


def test_wrong_signature_domain_does_not_verify(package):
    value = json.loads(package.raw)
    value["signatures"] = package.authority.sign(
        hashlib.sha256(canonical_json_bytes(value["payload"])).digest()
    )
    with pytest.raises(ProvisioningError, match="INVALID_PDSA_SIGNATURE"):
        _verify(package, canonical_json_bytes(value))


@pytest.mark.parametrize(
    "mutation", ["newline", "spacing", "byte", "float", "nan", "huge_integer", "surrogate"]
)
def test_noncanonical_and_invalid_canonical_input_fails_closed(package, mutation):
    if mutation == "newline":
        raw = package.raw + b"\n"
    elif mutation == "spacing":
        raw = json.dumps(json.loads(package.raw)).encode()
    elif mutation == "byte":
        raw = package.raw.replace(b'"authorization_version":1', b'"authorization_version":2')
    else:
        replacement = {
            "float": b"1.0",
            "nan": b"NaN",
            "huge_integer": b"9007199254740992",
            "surrogate": b'"\\ud800"',
        }[mutation]
        raw = package.raw.replace(
            b'"authorization_version":1', b'"authorization_version":' + replacement
        )
    with pytest.raises(ProvisioningError):
        _verify(package, raw)


@pytest.mark.parametrize("raw", [None, "{}", bytearray(b"{}"), b"x" * 32769, b"[" * 2000])
def test_untrusted_raw_type_size_and_depth_fail_closed(package, raw):
    with pytest.raises(ProvisioningError):
        _verify(package, raw)


def test_expiry_boundary_and_not_yet_valid(package):
    expiry = datetime.fromisoformat(package.payload["expires_at_utc"].replace("Z", "+00:00"))
    assert _verify(package, now=expiry - timedelta(microseconds=1))
    with pytest.raises(ProvisioningError, match="PACKAGE_EXPIRED"):
        _verify(package, now=expiry)
    with pytest.raises(ProvisioningError, match="PACKAGE_NOT_YET_VALID"):
        _verify(package, now=NOW - timedelta(microseconds=1))


@pytest.mark.parametrize("now", [None, "2026-10-06T12:00:00Z", datetime(2026, 10, 6)])
def test_malformed_verification_time_fails_closed(package, now):
    with pytest.raises(ProvisioningError, match="INVALID_VERIFICATION_TIME"):
        _verify(package, now=now)


@pytest.mark.parametrize("attack", ["public_dict", "new_instance", "copy"])
def test_public_or_copied_authentication_shape_cannot_verify_targets(package, attack):
    from bot_core.licensing.production_pre_enrollment import AuthenticatedProductionPreEnrollment

    if attack == "public_dict":
        value = {"request_raw": package.authenticated.request_raw}
    elif attack == "new_instance":
        value = object.__new__(AuthenticatedProductionPreEnrollment)
    else:
        value = copy.copy(package.authenticated)
    with pytest.raises(ProvisioningError, match="REJECT_PACKAGE_TARGET_MISMATCH"):
        package.verifier.verify_authenticated(package.raw, value, now=NOW)


def test_package_verifier_rechecks_trust_provenance_after_construction(package):
    object.__setattr__(package.authority.context, "release_version", 2)
    with pytest.raises(
        ProductionVerifierUnavailable, match="CURRENT_PRODUCTION_TRUST_CONTEXT_REQUIRED"
    ):
        _verify(package)


def test_authenticated_target_cannot_cross_independently_registered_trust_context(package):
    other = trust.verify_production_trust_for_audit(
        package.authority.store.path.parent, verification_time=NOW
    )
    assert other is not package.authority.context
    verifier = ProductionProvisioningPackageVerifier(other)
    assert verifier.verify(package.raw, expected_device_key=package.device, now=NOW)
    with pytest.raises(ProvisioningError, match="REJECT_PACKAGE_TARGET_MISMATCH"):
        verifier.verify_authenticated(package.raw, package.authenticated, now=NOW)


def test_authenticated_verification_rechecks_retained_live_tpm_exchange(package):
    with package.pending._connect() as database:
        database.execute("UPDATE tpm_challenges SET exchange_reference=?", ("00" * 32,))
    with pytest.raises(ProvisioningError, match="REJECT_PACKAGE_TARGET_MISMATCH"):
        package.verifier.verify_authenticated(package.raw, package.authenticated, now=NOW)


def test_mutable_mechanics_attributes_cannot_select_production_authority(package):
    ids = sorted(package.authority.keys)
    attacker_keys = {key_id: ed25519.Ed25519PrivateKey.generate() for key_id in ids}
    package.verifier._keys = {key_id: key.public_key() for key_id, key in attacker_keys.items()}
    package.verifier._environment = "TEST_ONLY"
    package.verifier._trust_domain = "TEST_ONLY_2_OF_3_ED25519"
    package.verifier._product_profile = "TEST_ONLY"
    package.verifier._release_digest = "00" * 32
    package.verifier._release_generation = 2
    package.verifier._production_trust = object()
    assert _verify(package).canonical_package == package.raw
    message = PDSA_DOMAIN + hashlib.sha256(canonical_json_bytes(package.payload)).digest()
    malicious = canonical_json_bytes(
        {
            "payload": package.payload,
            "signatures": [
                {
                    "algorithm": "Ed25519",
                    "key_id": key_id,
                    "signature_hex": attacker_keys[key_id].sign(message).hex(),
                }
                for key_id in ids[:2]
            ],
        }
    )
    with pytest.raises(ProvisioningError, match="INVALID_PDSA_SIGNATURE"):
        _verify(package, malicious)


@pytest.mark.parametrize("kind", ["new_instance", "copied"])
def test_unregistered_verifier_cannot_reuse_public_mechanics_attributes(package, kind):
    verifier = (
        object.__new__(ProductionProvisioningPackageVerifier)
        if kind == "new_instance"
        else copy.copy(package.verifier)
    )
    with pytest.raises(
        ProductionVerifierUnavailable, match="CURRENT_PRODUCTION_TRUST_CONTEXT_REQUIRED"
    ):
        verifier.verify(package.raw, expected_device_key=package.device, now=NOW)


def test_historical_audit_context_cannot_verify_new_production_package(package, monkeypatch):
    # Restore the actual runtime guard: the TEST_ONLY loader registered only audit provenance.
    monkeypatch.undo()
    with pytest.raises(
        ProductionVerifierUnavailable, match="CURRENT_PRODUCTION_TRUST_CONTEXT_REQUIRED"
    ):
        _verify(package)
