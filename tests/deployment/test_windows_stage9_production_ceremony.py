from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from deployment.windows_stage9_policy_material import PolicyVectorError, canonical_digest
from deployment.windows_stage9_production_ceremony import (
    RELEASE_SIGNATURE_PROFILE,
    REVOCATION_SIGNATURE_PROFILE,
    assemble_signed_artifact,
    build_audit_transcript,
    build_initial_revocation_payload,
    build_pdsa_public_bundle,
    build_recovery_public_bundle,
    build_root_anchor_bundle,
    build_signing_request,
    build_unsigned_release_policy,
    ceremony_id,
    import_detached_signatures,
    publish_final,
    verify_ceremony,
    verify_pdsa_public_bundle,
    verify_recovery_public_bundle,
    verify_root_anchor_bundle,
)
from deployment.windows_stage9_root_of_trust_freeze import (
    build_frozen_manifest,
    verify_freeze_manifest,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/windows_stage9_policy_vector_v1.json"
MODULE = Path(__file__).resolve().parents[2] / "deployment/windows_stage9_production_ceremony.py"
NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _keys(prefix: str, seeds: tuple[int, ...]):
    result = {}
    for number, seed in enumerate(seeds, 1):
        private = Ed25519PrivateKey.from_private_bytes(bytes([seed]) * 32)
        public = (
            private.public_key()
            .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            .hex()
        )
        result[f"{prefix}-{number}"] = (private, public)
    return result


def _records(keys):
    return [
        {
            "key_id": key_id,
            "algorithm": "Ed25519",
            "encoding": "RFC8032_RAW_32_BYTES_LOWER_HEX",
            "public_key_hex": pair[1],
        }
        for key_id, pair in keys.items()
    ]


def _signature(request, signer_id, private):
    return {
        "schema": "CryptoHunter.Stage9DetachedSignatureV1",
        "version": 1,
        "ceremony_id": request["ceremony_id"],
        "artifact_type": request["artifact_type"],
        "payload_digest": request["payload_digest"],
        "signer_id": signer_id,
        "signature_profile": RELEASE_SIGNATURE_PROFILE
        if request["artifact_type"] == "RELEASE_POLICY"
        else REVOCATION_SIGNATURE_PROFILE,
        "signature_hex": private.sign(bytes.fromhex(request["message_to_sign_hex"])).hex(),
    }


@pytest.fixture
def ceremony_material():
    fixture = json.loads(FIXTURE.read_text())["release_policy"]
    roots, pdsa = _keys("root", (1, 2, 3)), _keys("pdsa", (11, 12, 13))
    root_bundle, pinned = build_root_anchor_bundle(
        _records(roots), purpose="TEST_ONLY", environment="UNIT_TEST_ONLY"
    )
    pdsa_bundle = build_pdsa_public_bundle(_records(pdsa), threshold=2, purpose="TEST_ONLY")
    recovery = build_recovery_public_bundle(
        key_id="TEST_K_RECOVERY",
        tpmt_public_hex=fixture["k_recovery"]["public_hex"],
        purpose="TEST_ONLY",
        provenance="TEST_FIXTURE",
    )
    payload = build_unsigned_release_policy(
        root_bundle=root_bundle,
        pdsa_bundle=pdsa_bundle,
        recovery_bundle=recovery,
        release_policy_id="TEST_CEREMONY_V1",
        release_version=1,
        valid_from="2026-01-01T00:00:00Z",
        valid_until="2027-01-01T00:00:00Z",
        k_psa_profile=fixture["k_psa_profile"],
        policy_refs=fixture["policy_refs"],
        nv_template=fixture["nv_template"],
        branch_order=fixture["branch_order"],
    )
    cid = ceremony_id(pinned.key_set_digest, payload, pinned.environment)
    return roots, root_bundle, pinned, pdsa_bundle, recovery, payload, cid


def test_complete_test_only_offline_style_ceremony(ceremony_material, tmp_path):
    roots, root_bundle, pinned, pdsa, recovery, payload, cid = ceremony_material
    release_request = build_signing_request(
        artifact_type="RELEASE_POLICY", payload=payload, pinned_root=pinned, ceremony=cid
    )
    release_signatures = [
        _signature(release_request, name, roots[name][0]) for name in ("root-1", "root-2")
    ]
    signed_release = assemble_signed_artifact(
        request=release_request, payload=payload, signatures=release_signatures, pinned_root=pinned
    )
    revocation_payload = build_initial_revocation_payload(effective_at="2026-01-01T00:00:00Z")
    revocation_request = build_signing_request(
        artifact_type="INITIAL_REVOCATION",
        payload=revocation_payload,
        pinned_root=pinned,
        ceremony=cid,
    )
    revocation_signatures = [
        _signature(revocation_request, name, roots[name][0]) for name in ("root-1", "root-2")
    ]
    signed_revocation = assemble_signed_artifact(
        request=revocation_request,
        payload=revocation_payload,
        signatures=revocation_signatures,
        pinned_root=pinned,
    )
    verified = verify_ceremony(
        signed_release=signed_release,
        signed_revocation=signed_revocation,
        root_bundle=root_bundle,
        verification_time=NOW,
    )
    manifest = build_frozen_manifest(verified, artifact_source_revision="a" * 40)
    verify_freeze_manifest(manifest, verified_release=verified)
    audit = build_audit_transcript(
        ceremony=cid,
        release=verified,
        manifest=manifest,
        source_revision_value="a" * 40,
        environment="UNIT_TEST_ONLY",
    )
    final = publish_final(
        tmp_path, ceremony=cid, verified_release=verified, manifest=manifest, audit=audit
    )
    assert manifest["status"] == audit["final_status"] == "TEST_ONLY_ROOT_OF_TRUST_FROZEN"
    assert final.is_dir()
    with pytest.raises(PolicyVectorError, match="already exists"):
        publish_final(
            tmp_path, ceremony=cid, verified_release=verified, manifest=manifest, audit=audit
        )


def test_signature_import_failures(ceremony_material):
    roots, _, pinned, _, _, payload, cid = ceremony_material
    request = build_signing_request(
        artifact_type="RELEASE_POLICY", payload=payload, pinned_root=pinned, ceremony=cid
    )
    good = _signature(request, "root-1", roots["root-1"][0])
    with pytest.raises(PolicyVectorError, match="insufficient"):
        import_detached_signatures(request, [good], pinned)
    wrong = deepcopy(good)
    wrong["signer_id"] = "unknown"
    with pytest.raises(PolicyVectorError, match="unknown"):
        import_detached_signatures(request, [wrong, good], pinned)
    with pytest.raises(PolicyVectorError, match="duplicate"):
        import_detached_signatures(request, [good, good], pinned)
    other = deepcopy(good)
    other["ceremony_id"] = "0" * 64
    with pytest.raises(PolicyVectorError, match="another"):
        import_detached_signatures(request, [other, good], pinned)
    bad = deepcopy(good)
    bad["signature_hex"] = "00" * 64
    with pytest.raises(PolicyVectorError, match="wrong signature"):
        import_detached_signatures(
            request, [bad, _signature(request, "root-2", roots["root-2"][0])], pinned
        )
    changed = deepcopy(payload)
    changed["release_policy_id"] = "ANOTHER"
    with pytest.raises(PolicyVectorError, match="request"):
        assemble_signed_artifact(
            request=request, payload=changed, signatures=[], pinned_root=pinned
        )


def test_public_bundle_tampering_and_genesis_contract(ceremony_material):
    _, root, _, pdsa, recovery, _, _ = ceremony_material
    bad_pdsa = deepcopy(pdsa)
    bad_pdsa["canonical_key_set_digest"] = "00" * 32
    with pytest.raises(PolicyVectorError, match="digest"):
        verify_pdsa_public_bundle(bad_pdsa, expected_purpose="TEST_ONLY")
    bad_recovery = deepcopy(recovery)
    bad_recovery["derived_name"] = "000b" + "00" * 32
    with pytest.raises(PolicyVectorError, match="recovery"):
        verify_recovery_public_bundle(bad_recovery, expected_purpose="TEST_ONLY")
    malformed = deepcopy(recovery)
    malformed["tpmt_public_hex"] += "00"
    with pytest.raises(PolicyVectorError):
        verify_recovery_public_bundle(malformed, expected_purpose="TEST_ONLY")
    relabelled = deepcopy(root)
    relabelled["purpose"] = "PRODUCTION"
    with pytest.raises(PolicyVectorError, match="digest"):
        verify_root_anchor_bundle(relabelled)
    assert (
        build_initial_revocation_payload(effective_at="2026-01-01T00:00:00Z")[
            "previous_state_digest"
        ]
        == "00" * 32
    )


def test_partial_ceremony_never_publishes_final(ceremony_material, tmp_path):
    roots, _, pinned, _, _, payload, cid = ceremony_material
    request = build_signing_request(
        artifact_type="RELEASE_POLICY", payload=payload, pinned_root=pinned, ceremony=cid
    )
    with pytest.raises(PolicyVectorError):
        assemble_signed_artifact(
            request=request,
            payload=payload,
            signatures=[_signature(request, "root-1", roots["root-1"][0])],
            pinned_root=pinned,
        )
    assert not (tmp_path / "final").exists()


def test_production_module_has_no_private_key_operations():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    forbidden = {"Ed25519PrivateKey", "from_private_bytes", "private_bytes", "generate_private_key"}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    imports = {
        alias.name.rsplit(".", 1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not forbidden & (names | attributes | imports)
