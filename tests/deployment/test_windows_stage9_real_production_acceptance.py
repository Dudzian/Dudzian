"""Opt-in public-only acceptance against the operator-supplied final package."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from bot_core.licensing.device_enrollment import (
    build_activation_request,
    export_bundle,
    make_evidence,
    verify_activation_request_bundle,
)
from deployment.windows_stage9_production_trust import (
    CEREMONY_ID,
    PDSA_KEY_SET_DIGEST,
    RECOVERY_PUBLIC_DIGEST,
    RELEASE_PAYLOAD_DIGEST,
    ROOT_KEY_SET_DIGEST,
    ProductionTrustUnavailable,
    load_production_trust,
)

PACKAGE_ENV = "CRYPTOHUNTER_STAGE9_FINAL_PACKAGE"
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/windows_stage9_policy_vector_v1.json"


def _real_package() -> Path:
    value = os.environ.get(PACKAGE_ENV)
    if not value:
        pytest.skip(f"set {PACKAGE_ENV} to run the public production acceptance")
    return Path(value)


def _evidence():
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
        algorithm_profile="Stage9.K_PSA.ECC_P256_SHA256.PRODUCTION",
        evidence_profile="WindowsTPM2-TBS-Stage9-v1",
        substrate_profile="Stage9-TBS-public-acceptance-v1",
    )


def test_real_public_final_package_activation_roundtrip(tmp_path):
    package = _real_package()
    context = load_production_trust(package)
    assert context.ceremony_id == CEREMONY_ID
    assert context.release_payload_digest == RELEASE_PAYLOAD_DIGEST
    assert len(context.pdsa_keys) == 3
    root = json.loads((package / "root_anchor_bundle.json").read_text())
    pdsa = json.loads((package / "pdsa_public_bundle.json").read_text())
    recovery = json.loads((package / "recovery_public_bundle.json").read_text())
    assert root["canonical_key_set_digest"] == ROOT_KEY_SET_DIGEST
    assert pdsa["canonical_key_set_digest"] == PDSA_KEY_SET_DIGEST
    assert recovery["tpmt_public_sha256"] == RECOVERY_PUBLIC_DIGEST

    evidence = _evidence()
    request = build_activation_request(
        evidence=evidence,
        release_policy_digest=context.release_payload_digest,
        release_policy_version=context.release_version,
        requested_entitlements={
            "product": "CryptoHunter",
            "edition": "pro",
            "requested_features": ["core_bot"],
        },
        installation_id="11" * 32,
        architecture="AMD64",
        environment="PRODUCTION",
        production_trust_context=context,
    )
    verify_activation_request_bundle(
        request.canonical_bytes,
        evidence.canonical_bytes,
        environment="PRODUCTION",
        production_trust_context=context,
    )
    folder = export_bundle(
        tmp_path,
        request,
        evidence,
        environment="PRODUCTION",
        production_trust_context=context,
    )
    verify_activation_request_bundle(
        (folder / "activation-request.json").read_bytes(),
        (folder / "tpm-evidence.json").read_bytes(),
        environment="PRODUCTION",
        production_trust_context=context,
    )
    with pytest.raises((RuntimeError, ProductionTrustUnavailable)):
        verify_activation_request_bundle(
            request.canonical_bytes, evidence.canonical_bytes, environment="PRODUCTION"
        )

    altered = tmp_path / CEREMONY_ID
    shutil.copytree(package, altered)
    target = altered / "package_manifest.json"
    target.write_bytes(target.read_bytes().replace(b'"version":1', b'"version":2', 1))
    with pytest.raises(ProductionTrustUnavailable):
        load_production_trust(altered)
