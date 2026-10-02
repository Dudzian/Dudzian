from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from bot_core.licensing.canonical import canonical_json_bytes, digest
from bot_core.licensing.device_enrollment import (
    NAME_STOP,
    PRODUCTION_STOP,
    TPMPublicProjectionV1,
    build_activation_request,
    derive_device_id,
    export_bundle,
    make_evidence,
    verify_activation_request_bundle,
)

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/windows_stage9_policy_vector_v1.json"
)


def material():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    key = fixture["enrollment_policy_material"]["k_psa"]
    evidence = make_evidence(
        public_area_hex=key["public_hex"],
        returned_name=fixture["policy_vector"]["k_psa_name"],
        creation_hash="77" * 32,
        ek_public_area_hex=key["public_hex"],
        ek_name=fixture["policy_vector"]["k_psa_name"],
        ak_public_area_hex=key["public_hex"],
        ak_name=fixture["policy_vector"]["k_psa_name"],
        algorithm_profile="Stage9.K_PSA.ECC_P256_SHA256.TEST_ONLY",
        evidence_profile="WindowsTPM2-TBS-Stage9-v1",
        substrate_profile="Stage9-TBS-test-fake-v1",
    )
    release = digest(fixture["release_policy"]), 1
    request = build_activation_request(
        evidence=evidence,
        release_policy_digest=release[0],
        release_policy_version=release[1],
        requested_entitlements={
            "product": "CryptoHunter",
            "edition": "pro",
            "requested_features": ["core_bot"],
        },
        installation_id="11" * 32,
        architecture="AMD64",
        environment="TEST_ONLY",
        created_at_utc="2026-01-01T00:00:00Z",
        nonce="22" * 32,
    )
    return evidence, request, release


def test_verified_evidence_builds_and_roundtrips_public_bundle(tmp_path):
    evidence, request, release = material()
    folder = export_bundle(tmp_path, request, evidence, allowed_release=release)
    verified, verified_evidence = verify_activation_request_bundle(
        (folder / "activation-request.json").read_bytes(),
        (folder / "tpm-evidence.json").read_bytes(),
        allowed_release=release,
    )
    assert verified.canonical_bytes == request.canonical_bytes
    assert verified_evidence.canonical_bytes == evidence.canonical_bytes
    assert not any(
        token in b"".join(path.read_bytes() for path in folder.iterdir()).lower()
        for token in (b"private key", b"private_blob", b"password", b"passphrase")
    )


@pytest.mark.parametrize("field", ["k_psa", "ek", "ak"])
def test_tampered_evidence_and_caller_key_overrides_are_rejected(field):
    evidence, request, release = material()
    changed = evidence.document
    if field == "k_psa":
        changed[field]["name"] = "000b" + "00" * 32
    else:
        changed[field]["public_digest"] = "00" * 32
    changed["evidence_id"] = digest(
        {k: v for k, v in changed.items() if k != "evidence_id"}
    )
    with pytest.raises(ValueError):
        verify_activation_request_bundle(
            request.canonical_bytes,
            canonical_json_bytes(changed),
            allowed_release=release,
        )
    with pytest.raises(TypeError):
        build_activation_request(
            evidence=evidence,
            release_policy_digest=release[0],
            release_policy_version=1,
            requested_entitlements={},
            installation_id="x",
            architecture="x",
            environment="TEST_ONLY",
            **{field: "override"},
        )


def test_wrong_returned_name_and_public_area_mismatch_stop_before_request():
    evidence, _, _ = material()
    value = evidence.document
    with pytest.raises(ValueError, match=NAME_STOP):
        make_evidence(
            public_area_hex=value["k_psa"]["public_area"]["hex"],
            returned_name="000b" + "00" * 32,
            creation_hash="77" * 32,
            ek_public_area_hex=value["ek"]["public_area"]["hex"],
            ek_name=value["ek"]["name"],
            ak_public_area_hex=value["ak"]["public_area"]["hex"],
            ak_name=value["ak"]["name"],
            algorithm_profile="x",
            evidence_profile="x",
            substrate_profile="x",
        )
    value["k_psa"]["public_area"]["hex"] = (
        value["k_psa"]["public_area"]["hex"][:-2] + "00"
    )
    value["evidence_id"] = digest(
        {k: v for k, v in value.items() if k != "evidence_id"}
    )
    with pytest.raises(ValueError, match=NAME_STOP):
        TPMPublicProjectionV1.verify(value)


def test_wrong_reference_device_release_and_noncanonical_inputs_rejected():
    evidence, request, release = material()
    for section, key, value in (
        ("tpm", "evidence_reference", "00" * 32),
        ("device", "device_id", "00" * 32),
        ("release", "release_policy_digest", "00" * 32),
    ):
        changed = request.document
        changed[section][key] = value
        identity = deepcopy(changed)
        identity.pop("request_id")
        changed["request_id"] = digest(identity)
        with pytest.raises(ValueError, match="binding mismatch"):
            verify_activation_request_bundle(
                canonical_json_bytes(changed),
                evidence.canonical_bytes,
                allowed_release=release,
            )
    with pytest.raises(ValueError, match="noncanonical evidence"):
        verify_activation_request_bundle(
            request.canonical_bytes,
            b" " + evidence.canonical_bytes,
            allowed_release=release,
        )


def test_evidence_id_tamper_and_production_fail_closed():
    evidence, _, release = material()
    changed = evidence.document
    changed["evidence_id"] = "00" * 32
    with pytest.raises(ValueError, match="evidence_id mismatch"):
        TPMPublicProjectionV1.verify(changed)
    with pytest.raises(RuntimeError, match=PRODUCTION_STOP):
        build_activation_request(
            evidence=evidence,
            release_policy_digest=release[0],
            release_policy_version=1,
            requested_entitlements={
                "product": "CryptoHunter",
                "edition": "pro",
                "requested_features": [],
            },
            installation_id="x",
            architecture="AMD64",
            environment="PRODUCTION",
        )


def test_forged_production_context_is_rejected():
    from deployment.windows_stage9_production_trust import ProductionTrustContext

    evidence, _, release = material()
    context = object.__new__(ProductionTrustContext)
    object.__setattr__(context, "release_payload_digest", release[0])
    object.__setattr__(context, "release_version", release[1])

    with pytest.raises(RuntimeError, match=PRODUCTION_STOP):
        build_activation_request(
            evidence=evidence,
            release_policy_digest=release[0],
            release_policy_version=release[1],
            requested_entitlements={
                "product": "CryptoHunter",
                "edition": "pro",
                "requested_features": [],
            },
            installation_id="x",
            architecture="AMD64",
            environment="PRODUCTION",
            production_trust_context=context,
        )


def test_device_id_is_only_deterministic_verified_public_identity():
    evidence, request, _ = material()
    assert request.document["device"]["device_id"] == derive_device_id(evidence)
    assert request.document["device"]["device_id"] not in {
        evidence.document["ek"]["public_digest"],
        evidence.document["ak"]["public_digest"],
    }
