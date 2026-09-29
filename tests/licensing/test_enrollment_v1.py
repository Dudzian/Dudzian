from __future__ import annotations
import copy
from datetime import datetime, timezone
import pytest
from bot_core.licensing.activation_request import ActivationRequestV1, EnrollmentDecisionV1
from bot_core.licensing.authority import (
    MockOnlineEnrollmentAuthority,
    OfflineEnrollmentAuthority,
    TestOnlyPDSAAuthority,
)
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.licensing.local_state import resolve_startup
from bot_core.licensing.verification import (
    EnrollmentVerificationError,
    LocalIdentityV1,
    verify_pdsa_enrollment_package,
)

D = "a" * 64


def material():
    r = ActivationRequestV1.create(
        created_at_utc="2026-01-01T00:00:00Z",
        installation_id="install-a",
        nonce="b" * 64,
        device={"device_id": "device-a", "platform": "Windows", "architecture": "AMD64"},
        tpm={
            "evidence_profile": "TPM2",
            "ek_public_digest": D,
            "ak_public_digest": "b" * 64,
            "evidence_reference": "attestation-1",
            "manufacturer": None,
            "model": None,
        },
        k_psa={
            "public_area": {"hex": "0102"},
            "name": "000b" + "c" * 64,
            "algorithm_profile": "Stage9",
        },
        release={"release_policy_digest": "d" * 64, "release_policy_version": 1},
        requested_entitlements={
            "product": "CryptoHunter",
            "edition": "pro",
            "requested_features": ["core_bot", "multi_exchange"],
        },
    )
    d = EnrollmentDecisionV1.from_mapping(
        {
            "schema": "EnrollmentDecisionV1",
            "version": 1,
            "request_id": r.document["request_id"],
            "license_id": "license-a",
            "product": "CryptoHunter",
            "edition": "pro",
            "features": ["core_bot", "multi_exchange"],
            "issued_at": "2026-01-01T00:00:00Z",
            "expires_at": None,
            "renewal_after": None,
            "approval_status": "APPROVED",
            "operator_note": None,
        }
    )
    a = TestOnlyPDSAAuthority.deterministic_fixture()
    p = OfflineEnrollmentAuthority(a).issue(r, d)
    ident = LocalIdentityV1(
        "install-a",
        "device-a",
        r.document["k_psa"]["name"],
        r.document["k_psa"]["public_area"],
        D,
        "b" * 64,
        "TPM2",
        "d" * 64,
        1,
    )
    return r, d, a, p, ident


def verify(p=None, ident=None):
    r, d, a, base, i = material()
    p = base if p is None else p
    ident = i if ident is None else ident
    return verify_pdsa_enrollment_package(
        canonical_json_bytes(p),
        request=r,
        identity=ident,
        trusted_public_keys=a.public_keys,
        expected_key_set_digest=a.key_set_digest,
    )


def test_transport_independence_and_offline_roundtrip():
    r, d, a, p, i = material()
    assert canonical_json_bytes(p) == canonical_json_bytes(
        MockOnlineEnrollmentAuthority(a).issue(r, d)
    )
    assert verify(p, i).features == ("core_bot", "multi_exchange")


def test_outage_does_not_deactivate_valid_local_enrollment():
    state = resolve_startup(lambda: verify(), lambda: (_ for _ in ()).throw(TimeoutError()))
    assert (
        state.may_run
        and state.activation_status.value == "ACTIVATED"
        and state.service_status == "ONLINE_SERVICE_UNAVAILABLE"
    )
    state = resolve_startup(
        lambda: None,
        lambda: (_ for _ in ()).throw(ConnectionError()),
    )
    assert not state.may_run and state.activation_status.value == "OFFLINE_ACTIVATION_REQUIRED"


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("k_psa_name", "K_PSA_NAME_MISMATCH"),
        ("device_id", "DEVICE_MISMATCH"),
        ("release_policy_digest", "RELEASE_POLICY_MISMATCH"),
    ],
)
def test_local_binding_mismatches(field, code):
    *_, i = material()
    values = i.__dict__.copy()
    values[field] = "wrong"
    with pytest.raises(EnrollmentVerificationError, match=code):
        verify(ident=LocalIdentityV1(**values))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["license"].__setitem__("license_id", "changed"),
        lambda p: p["k_psa"].__setitem__("name", "changed"),
        lambda p: p["license"].__setitem__("features", ["core_bot"]),
        lambda p: p["license"].__setitem__("expires_at", "2030-01-01T00:00:00Z"),
        lambda p: p["release_binding"].__setitem__("release_policy_digest", "e" * 64),
        lambda p: p["request_binding"].__setitem__("activation_request_digest", "e" * 64),
    ],
)
def test_signed_fields_are_tamper_evident(mutation):
    *_, p, i = material()
    mutation(p)
    with pytest.raises(EnrollmentVerificationError):
        verify(p, i)


def test_unknown_insufficient_and_duplicate_signers_rejected():
    *_, p, i = material()
    p["signature_block"][0]["key_id"] = "unknown"
    with pytest.raises(EnrollmentVerificationError, match="SIGNER_IDS_MISMATCH"):
        verify(p, i)
    _, _, authority, p, i = material()
    resign(p, authority, ["TEST_ONLY_PDSA_1"])
    with pytest.raises(EnrollmentVerificationError, match="THRESHOLD_NOT_MET"):
        verify(p, i)
    *_, p, i = material()
    p["signature_block"][1]["key_id"] = p["signature_block"][0]["key_id"]
    with pytest.raises(EnrollmentVerificationError, match="DUPLICATE_SIGNER_ID"):
        verify(p, i)


def test_expiry_requires_trusted_freshness_and_can_expire():
    r, d, a, _, i = material()
    dd = copy.deepcopy(d.document)
    dd["expires_at"] = "2025-01-01T00:00:00Z"
    p = a.issue(r, EnrollmentDecisionV1.from_mapping(dd))
    with pytest.raises(EnrollmentVerificationError, match="TRUSTED_FRESHNESS_REQUIRED"):
        verify(p, i)
    with pytest.raises(EnrollmentVerificationError, match="EXPIRED"):
        verify_pdsa_enrollment_package(
            canonical_json_bytes(p),
            request=r,
            identity=i,
            trusted_public_keys=a.public_keys,
            expected_key_set_digest=a.key_set_digest,
            trusted_time=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
        )


def test_production_path_guard():
    with pytest.raises(ValueError, match="PRODUCTION PDSA"):
        TestOnlyPDSAAuthority.from_test_material(
            "TEST_ONLY",
            TestOnlyPDSAAuthority.deterministic_fixture()._keys,
            material_path=r"C:\CryptoHunter-Production-Authority\keys",
        )


def resign(package, authority, signer_ids=None, authority_digest=None):
    import hashlib
    from bot_core.licensing.enrollment import DOMAIN

    signer_ids = signer_ids or list(authority._keys)[:2]
    authority_digest = authority_digest or authority.key_set_digest
    package["authority"]["pdsa_key_set_digest"] = authority_digest
    package["authority"]["signer_key_ids"] = sorted(signer_ids)
    unsigned = copy.deepcopy(package)
    unsigned.pop("signature_block", None)
    message = DOMAIN + hashlib.sha256(canonical_json_bytes(unsigned)).digest()
    package["signature_block"] = [
        {
            "algorithm": "Ed25519",
            "key_id": key_id,
            "authority_key_set_digest": authority_digest,
            "signature": authority._keys[key_id].sign(message).hex(),
        }
        for key_id in signer_ids
    ]
    return package


def test_request_and_decision_are_deeply_immutable():
    request, decision, authority, _, _ = material()
    request_mapping = request.document
    reconstructed = ActivationRequestV1.from_mapping(request_mapping)
    request_mapping["device"]["device_id"] = "attacker"
    projection = reconstructed.document
    projection["device"]["device_id"] = "also-attacker"
    assert reconstructed.document["device"]["device_id"] == "device-a"

    decision_mapping = decision.document
    reconstructed_decision = EnrollmentDecisionV1.from_mapping(decision_mapping)
    decision_mapping["features"].append("attacker")
    decision_projection = reconstructed_decision.document
    decision_projection["features"].append("also-attacker")
    assert reconstructed_decision.document["features"] == ["core_bot", "multi_exchange"]
    assert (
        authority.issue(reconstructed, reconstructed_decision)["subject"]["device_id"] == "device-a"
    )


def test_stale_request_id_is_rejected_before_signing():
    request, decision, authority, _, _ = material()
    stale = request.document
    stale["device"]["device_id"] = "changed"
    with pytest.raises(ValueError, match="request_id mismatch"):
        ActivationRequestV1.from_mapping(stale)
    with pytest.raises(TypeError, match="use create"):
        ActivationRequestV1(canonical_json_bytes(stale))


@pytest.mark.parametrize("target", ["nested_unknown", "signature_unknown", "nested_missing"])
def test_strict_nested_package_schema(target):
    *_, package, identity = material()
    if target == "nested_unknown":
        package["subject"]["unknown"] = True
    elif target == "signature_unknown":
        package["signature_block"][0]["unknown"] = True
    else:
        package["tpm"].pop("ak_public_digest")
    with pytest.raises(EnrollmentVerificationError, match="INVALID_SCHEMA"):
        verify(package, identity)


def test_derived_enrollment_id_and_issued_at_are_independently_checked():
    request, _, authority, package, identity = material()
    package["enrollment_id"] = "e" * 64
    resign(package, authority)
    with pytest.raises(EnrollmentVerificationError, match="ENROLLMENT_ID_MISMATCH"):
        verify(package, identity)
    _, _, authority, package, identity = material()
    package["issued_at_utc"] = "2026-01-02T00:00:00Z"
    resign(package, authority)
    with pytest.raises(EnrollmentVerificationError, match="ISSUED_AT_MISMATCH"):
        verify(package, identity)


def test_actual_signers_and_trusted_key_set_are_bound():
    *_, package, identity = material()
    package["authority"]["signer_key_ids"].append("TEST_ONLY_PDSA_3")
    with pytest.raises(EnrollmentVerificationError, match="SIGNER_IDS_MISMATCH"):
        verify(package, identity)
    request, _, authority, package, identity = material()
    package["authority"]["signer_key_ids"] = ["TEST_ONLY_PDSA_1"]
    with pytest.raises(EnrollmentVerificationError, match="SIGNER_IDS_MISMATCH"):
        verify(package, identity)
    with pytest.raises(EnrollmentVerificationError, match="AUTHORITY_MISMATCH"):
        verify_pdsa_enrollment_package(
            canonical_json_bytes(material()[3]),
            request=request,
            identity=identity,
            trusted_public_keys=authority.public_keys,
            expected_key_set_digest="f" * 64,
        )


@pytest.mark.parametrize("expiry", ["bad", "2026-01-01T00:00:00", "+12", 123])
def test_malformed_expiry_has_structured_failure(expiry):
    _, _, authority, package, identity = material()
    package["license"]["expires_at"] = expiry
    resign(package, authority)
    with pytest.raises(EnrollmentVerificationError, match="INVALID_EXPIRY"):
        verify(package, identity)


def test_verified_package_is_immutable_and_real_import_roundtrip(tmp_path):
    request, _, authority, package, identity = material()
    raw = canonical_json_bytes(package)
    verifier = lambda value: verify_pdsa_enrollment_package(
        value,
        request=request,
        identity=identity,
        trusted_public_keys=authority.public_keys,
        expected_key_set_digest=authority.key_set_digest,
    )
    from bot_core.licensing.local_state import LocalEnrollmentStore

    store = LocalEnrollmentStore(tmp_path / "license.json", verifier=verifier)
    verified = store.import_and_verify(raw)
    projection = verified.package
    projection["license"]["features"].append("attacker")
    assert store.load() == raw
    reverified = verifier(store.load())
    state = resolve_startup(lambda: reverified, lambda: (_ for _ in ()).throw(TimeoutError()))
    assert state.may_run and state.activation_status.value == "ACTIVATED"
    forged = type(verified)._create("e" * 64, "forged", ("attacker",), None, b"{}")
    assert forged.canonical_bytes == b"{}"
    with pytest.raises(EnrollmentVerificationError):
        store.import_and_verify(forged.canonical_bytes)
    assert store.load() == raw
    with pytest.raises(EnrollmentVerificationError):
        store.import_and_verify(canonical_json_bytes({"attacker": True}))
    assert store.load() == raw


def test_startup_classification_is_orthogonal_to_network():
    def down():
        raise TimeoutError

    for code, expected in [
        ("SIGNATURE_INVALID", "LICENSE_INVALID"),
        ("DEVICE_MISMATCH", "DEVICE_MISMATCH"),
        ("K_PSA_NAME_MISMATCH", "DEVICE_MISMATCH"),
        ("EXPIRED", "EXPIRED"),
    ]:
        state = resolve_startup(
            lambda code=code: (_ for _ in ()).throw(EnrollmentVerificationError(code)), down
        )
        assert state.activation_status.value == expected
        assert state.service_status == "ONLINE_SERVICE_UNAVAILABLE"
        assert not state.may_run


def test_pdsa_authority_rejects_two_key_profile_even_with_valid_two_signatures():
    from bot_core.licensing.authority import public_authority_projection
    from bot_core.licensing.canonical import digest

    request, _, authority, package, identity = material()
    trusted = dict(list(authority.public_keys.items())[:2])
    two_key_digest = digest(public_authority_projection(trusted))
    resign(package, authority, authority_digest=two_key_digest)
    with pytest.raises(EnrollmentVerificationError, match="AUTHORITY_PROFILE_MISMATCH"):
        verify_pdsa_enrollment_package(
            canonical_json_bytes(package),
            request=request,
            identity=identity,
            trusted_public_keys=trusted,
            expected_key_set_digest=two_key_digest,
        )


def test_pdsa_authority_rejects_four_keys_malformed_unused_and_duplicate_bytes():
    request, _, authority, package, identity = material()

    four = dict(authority.public_keys)
    four["TEST_ONLY_PDSA_4"] = "11" * 32
    with pytest.raises(EnrollmentVerificationError, match="AUTHORITY_PROFILE_MISMATCH"):
        verify_pdsa_enrollment_package(
            canonical_json_bytes(package),
            request=request,
            identity=identity,
            trusted_public_keys=four,
            expected_key_set_digest=authority.key_set_digest,
        )

    malformed = dict(authority.public_keys)
    malformed["TEST_ONLY_PDSA_3"] = "not-an-ed25519-key"
    with pytest.raises(EnrollmentVerificationError, match="AUTHORITY_PROFILE_MISMATCH"):
        verify_pdsa_enrollment_package(
            canonical_json_bytes(package),
            request=request,
            identity=identity,
            trusted_public_keys=malformed,
            expected_key_set_digest=authority.key_set_digest,
        )

    duplicate = dict(authority.public_keys)
    duplicate["TEST_ONLY_PDSA_3"] = duplicate["TEST_ONLY_PDSA_1"]
    with pytest.raises(EnrollmentVerificationError, match="AUTHORITY_PROFILE_MISMATCH"):
        verify_pdsa_enrollment_package(
            canonical_json_bytes(package),
            request=request,
            identity=identity,
            trusted_public_keys=duplicate,
            expected_key_set_digest=authority.key_set_digest,
        )
