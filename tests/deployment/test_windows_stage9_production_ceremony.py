from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from deployment.windows_stage9_policy_material import (
    PolicyVectorError,
    canonical_digest,
    canonical_json_bytes,
)
from deployment.windows_stage9_production_ceremony import (
    RELEASE_SIGNATURE_PROFILE,
    REVOCATION_SIGNATURE_PROFILE,
    assemble_signed_artifact,
    build_audit_transcript,
    build_ceremony_context,
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
    verify_final_package,
    verify_pdsa_public_bundle,
    verify_recovery_public_bundle,
    verify_root_anchor_bundle,
)
from deployment.windows_stage9_root_of_trust_freeze import (
    build_frozen_manifest,
    verify_freeze_manifest,
)
from deployment.windows_stage9_production_trust import (
    ProductionTrustUnavailable,
    verify_production_trust_for_audit,
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
        "signature_profile": (
            RELEASE_SIGNATURE_PROFILE
            if request["artifact_type"] == "RELEASE_POLICY"
            else REVOCATION_SIGNATURE_PROFILE
        ),
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
    context = build_ceremony_context(pinned_root=pinned, release_payload=payload)
    return roots, root_bundle, pinned, pdsa_bundle, recovery, payload, context


def test_complete_test_only_offline_style_ceremony(ceremony_material, tmp_path):
    roots, root_bundle, pinned, pdsa, recovery, payload, context = ceremony_material
    release_request = build_signing_request(
        artifact_type="RELEASE_POLICY",
        payload=payload,
        pinned_root=pinned,
        context=context,
    )
    release_signatures = [
        _signature(release_request, name, roots[name][0]) for name in ("root-1", "root-2")
    ]
    signed_release = assemble_signed_artifact(
        request=release_request,
        payload=payload,
        signatures=release_signatures,
        pinned_root=pinned,
        context=context,
    )
    revocation_payload = build_initial_revocation_payload(effective_at="2026-01-01T00:00:00Z")
    revocation_request = build_signing_request(
        artifact_type="INITIAL_REVOCATION",
        payload=revocation_payload,
        pinned_root=pinned,
        context=context,
    )
    revocation_signatures = [
        _signature(revocation_request, name, roots[name][0]) for name in ("root-1", "root-2")
    ]
    signed_revocation = assemble_signed_artifact(
        request=revocation_request,
        payload=revocation_payload,
        signatures=revocation_signatures,
        pinned_root=pinned,
        context=context,
    )
    verified = verify_ceremony(
        signed_release=signed_release,
        signed_revocation=signed_revocation,
        release_request=release_request,
        revocation_request=revocation_request,
        root_bundle=root_bundle,
        verification_time=NOW,
    )
    manifest = build_frozen_manifest(verified.verified_release, artifact_source_revision="a" * 40)
    verify_freeze_manifest(manifest, verified_release=verified.verified_release)
    audit = build_audit_transcript(
        ceremony=verified,
        manifest=manifest,
    )
    with pytest.raises(TypeError):
        verified.signed_release["signatures"][0]["signature_hex"] = "00" * 64
    with pytest.raises(TypeError):
        verified.signed_release["payload"]["release_policy_id"] = "MUTATED"
    with pytest.raises(TypeError):
        verified.signed_revocation["payload"]["sequence"] = 2
    with pytest.raises(TypeError):
        verified.release_request["ceremony_id"] = "0" * 64
    with pytest.raises(TypeError):
        verified.revocation_request["ceremony_id"] = "0" * 64
    with pytest.raises(TypeError):
        verified.root_bundle["environment"] = "MUTATED"

    mutation_cases = {
        "signed_release_bytes": ("signatures", 0, "signature_hex", "00" * 64),
        "signed_revocation_bytes": ("payload", "sequence", 2),
        "release_request_bytes": ("ceremony_id", "0" * 64),
        "revocation_request_bytes": ("ceremony_id", "0" * 64),
        "root_bundle_bytes": ("environment", "MUTATED"),
    }
    for field, path in mutation_cases.items():
        document = json.loads(getattr(verified, field))
        target = document
        for key in path[:-2]:
            target = target[key]
        target[path[-2]] = path[-1]
        corrupted = replace(verified, **{field: canonical_json_bytes(document)})
        with pytest.raises(PolicyVectorError):
            publish_final(
                tmp_path,
                ceremony=corrupted,
                pdsa_bundle=pdsa,
                recovery_bundle=recovery,
                manifest=manifest,
                audit=audit,
            )
        assert not (tmp_path / "final").exists()

    wrong_pdsa = deepcopy(pdsa)
    wrong_pdsa["threshold"] = 1
    with pytest.raises(PolicyVectorError, match="not bound"):
        publish_final(
            tmp_path,
            ceremony=verified,
            pdsa_bundle=wrong_pdsa,
            recovery_bundle=recovery,
            manifest=manifest,
            audit=audit,
        )
    wrong_recovery_id = deepcopy(recovery)
    wrong_recovery_id["key_id"] = "ATTACKER_LABEL"
    with pytest.raises(PolicyVectorError, match="not bound"):
        publish_final(
            tmp_path,
            ceremony=verified,
            pdsa_bundle=pdsa,
            recovery_bundle=wrong_recovery_id,
            manifest=manifest,
            audit=audit,
        )
    wrong_recovery_metadata = deepcopy(recovery)
    wrong_recovery_metadata["profile"]["role"] = "ATTACKER_ROLE"
    with pytest.raises(PolicyVectorError):
        publish_final(
            tmp_path,
            ceremony=verified,
            pdsa_bundle=pdsa,
            recovery_bundle=wrong_recovery_metadata,
            manifest=manifest,
            audit=audit,
        )
    wrong_audit = deepcopy(audit)
    wrong_audit["ceremony_id"] = "0" * 64
    with pytest.raises(PolicyVectorError, match="audit transcript"):
        publish_final(
            tmp_path,
            ceremony=verified,
            pdsa_bundle=pdsa,
            recovery_bundle=recovery,
            manifest=manifest,
            audit=wrong_audit,
        )
    assert not (tmp_path / "final").exists()
    wrong_revision_audit = deepcopy(audit)
    wrong_revision_audit["source_revision"] = "b" * 40
    with pytest.raises(PolicyVectorError, match="audit transcript"):
        publish_final(
            tmp_path,
            ceremony=verified,
            pdsa_bundle=pdsa,
            recovery_bundle=recovery,
            manifest=manifest,
            audit=wrong_revision_audit,
        )
    assert not (tmp_path / "final").exists()
    final = publish_final(
        tmp_path,
        ceremony=verified,
        pdsa_bundle=pdsa,
        recovery_bundle=recovery,
        manifest=manifest,
        audit=audit,
    )
    assert manifest["status"] == audit["final_status"] == "TEST_ONLY_ROOT_OF_TRUST_FROZEN"
    assert final.is_dir()
    assert verify_final_package(final, verification_time=NOW).ceremony_id == context.ceremony_id
    with pytest.raises(ProductionTrustUnavailable, match="frozen production authority"):
        verify_production_trust_for_audit(final, verification_time=NOW)
    assert (final / "package_manifest.json").is_file()
    assert json.loads((final / "ceremony_audit.json").read_text())["source_revision"] == "a" * 40
    broken = tmp_path / context.ceremony_id
    shutil.copytree(final, broken)
    altered = json.loads((broken / "unsigned_release_policy.json").read_text())
    altered["release_policy_id"] = "TAMPERED"
    (broken / "unsigned_release_policy.json").write_text(json.dumps(altered))
    with pytest.raises(PolicyVectorError, match="payload copy mismatch|digest mismatch"):
        verify_final_package(broken, verification_time=NOW)
    digest_broken = tmp_path / "digest-broken" / context.ceremony_id
    shutil.copytree(final, digest_broken)
    package_manifest = json.loads((digest_broken / "package_manifest.json").read_text())
    package_manifest["artifacts"][0]["sha256"] = "0" * 64
    (digest_broken / "package_manifest.json").write_text(json.dumps(package_manifest))
    with pytest.raises(PolicyVectorError, match="artifact digest mismatch"):
        verify_final_package(digest_broken, verification_time=NOW)
    for missing in (
        "signed_release_policy.json",
        "signed_initial_revocation.json",
        "ceremony_audit.json",
        "freeze_manifest.json",
    ):
        incomplete = tmp_path / f"missing-{missing}"
        shutil.copytree(final, incomplete)
        (incomplete / missing).unlink()
        with pytest.raises(PolicyVectorError, match="incomplete"):
            verify_final_package(incomplete, verification_time=NOW)
    with pytest.raises(PolicyVectorError, match="already exists"):
        publish_final(
            tmp_path,
            ceremony=verified,
            pdsa_bundle=pdsa,
            recovery_bundle=recovery,
            manifest=manifest,
            audit=audit,
        )


def test_production_trust_loader_fails_closed_when_package_is_missing(tmp_path):
    with pytest.raises(ProductionTrustUnavailable, match="PRODUCTION_TRUST_UNAVAILABLE"):
        verify_production_trust_for_audit(tmp_path / "missing", verification_time=NOW)


def test_runtime_trust_loader_uses_current_utc_not_caller_time(tmp_path, monkeypatch):
    import deployment.windows_stage9_production_trust as trust

    observed = {}

    def capture(path, *, verification_time):
        observed["path"] = path
        observed["time"] = verification_time
        return object()

    monkeypatch.setattr(trust, "verify_production_trust_for_audit", capture)
    before = datetime.now(timezone.utc)
    trust.load_production_trust(tmp_path)
    after = datetime.now(timezone.utc)
    assert observed["path"] == tmp_path
    assert before <= observed["time"] <= after


def test_public_trust_installer_copies_atomically_and_refuses_overwrite(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import deployment.windows_stage9_production_trust as trust

    source = tmp_path / trust.CEREMONY_ID
    source.mkdir()
    for name in trust.PUBLIC_PACKAGE_FILENAMES:
        (source / name).write_text('{"public":"value"}\n', encoding="utf-8")
    verified = SimpleNamespace(ceremony_id=trust.CEREMONY_ID)
    monkeypatch.setattr(trust, "load_production_trust", lambda path: verified)
    monkeypatch.setattr(
        trust,
        "verify_production_trust_for_audit",
        lambda path, verification_time: verified,
    )
    destination = tmp_path / "Config" / "ProductionTrust" / trust.CEREMONY_ID
    assert trust.install_public_production_trust(source, destination) == destination
    assert {item.name: item.read_bytes() for item in destination.iterdir()} == {
        item.name: item.read_bytes() for item in source.iterdir()
    }
    assert not list(destination.parent.glob(".production-trust-*"))
    with pytest.raises(trust.ProductionTrustUnavailable, match="OVERWRITE_FORBIDDEN"):
        trust.install_public_production_trust(source, destination)


def test_public_trust_installer_rejects_non_public_extra_before_mutation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import deployment.windows_stage9_production_trust as trust

    source = tmp_path / trust.CEREMONY_ID
    source.mkdir()
    for name in trust.PUBLIC_PACKAGE_FILENAMES:
        (source / name).write_text("{}\n", encoding="utf-8")
    (source / "private.pem").write_text("PRIVATE KEY", encoding="utf-8")
    monkeypatch.setattr(
        trust,
        "load_production_trust",
        lambda path: SimpleNamespace(ceremony_id=trust.CEREMONY_ID),
    )
    destination = tmp_path / "Config" / "ProductionTrust" / trust.CEREMONY_ID
    with pytest.raises(trust.ProductionTrustUnavailable, match="ALLOWLIST"):
        trust.install_public_production_trust(source, destination)
    assert not destination.exists()


def test_public_trust_failure_before_rename_removes_staging_and_owned_parent(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import deployment.windows_stage9_production_trust as trust

    source = tmp_path / trust.CEREMONY_ID
    source.mkdir()
    for name in trust.PUBLIC_PACKAGE_FILENAMES:
        (source / name).write_text("{}\n", encoding="utf-8")
    verified = SimpleNamespace(ceremony_id=trust.CEREMONY_ID)
    monkeypatch.setattr(trust, "load_production_trust", lambda path: verified)
    monkeypatch.setattr(
        trust,
        "verify_production_trust_for_audit",
        lambda path, verification_time: verified,
    )
    monkeypatch.setattr(
        trust.os,
        "rename",
        lambda source, destination: (_ for _ in ()).throw(OSError("before rename")),
    )
    destination = tmp_path / "Config" / "ProductionTrust" / trust.CEREMONY_ID
    with pytest.raises(OSError, match="before rename"):
        trust.install_public_production_trust(source, destination)
    assert not destination.exists()
    assert not destination.parent.exists()


def test_staging_verification_failure_has_no_publication_or_residue(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import deployment.windows_stage9_production_trust as trust

    source = tmp_path / trust.CEREMONY_ID
    source.mkdir()
    for name in trust.PUBLIC_PACKAGE_FILENAMES:
        (source / name).write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        trust,
        "load_production_trust",
        lambda path: SimpleNamespace(ceremony_id=trust.CEREMONY_ID),
    )
    monkeypatch.setattr(
        trust,
        "verify_production_trust_for_audit",
        lambda path, verification_time: (_ for _ in ()).throw(ValueError("staged invalid")),
    )
    destination = tmp_path / "Config" / "ProductionTrust" / trust.CEREMONY_ID
    with pytest.raises(ValueError, match="staged invalid"):
        trust.install_public_production_trust(source, destination)
    assert not destination.exists()
    assert not destination.parent.exists()


@pytest.mark.parametrize("failure_point", ["creation", "copy"])
def test_early_staging_failures_leave_no_publication_or_residue(
    tmp_path, monkeypatch, failure_point
):
    from types import SimpleNamespace
    import deployment.windows_stage9_production_trust as trust

    source = tmp_path / trust.CEREMONY_ID
    source.mkdir()
    for name in trust.PUBLIC_PACKAGE_FILENAMES:
        (source / name).write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        trust,
        "load_production_trust",
        lambda path: SimpleNamespace(ceremony_id=trust.CEREMONY_ID),
    )
    destination = tmp_path / "Config" / "ProductionTrust" / trust.CEREMONY_ID
    if failure_point == "creation":
        original_mkdir = Path.mkdir

        def fail_staging_mkdir(path, *args, **kwargs):
            if path.name == trust.CEREMONY_ID and path.parent.name.startswith(".production-trust-"):
                raise OSError("staging creation failed")
            return original_mkdir(path, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", fail_staging_mkdir)
    else:
        original_write_bytes = Path.write_bytes

        def fail_staging_copy(path, data):
            if any(part.startswith(".production-trust-") for part in path.parts):
                raise OSError("staging copy failed")
            return original_write_bytes(path, data)

        monkeypatch.setattr(Path, "write_bytes", fail_staging_copy)

    with pytest.raises(OSError, match=f"staging {failure_point} failed"):
        trust.install_public_production_trust(source, destination)
    assert not destination.exists()
    assert not destination.parent.exists()
    assert not list(tmp_path.rglob(".production-trust-*"))


def test_cleanup_failure_after_rename_is_committed_and_unambiguous(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import deployment.windows_stage9_production_trust as trust

    source = tmp_path / trust.CEREMONY_ID
    source.mkdir()
    for name in trust.PUBLIC_PACKAGE_FILENAMES:
        (source / name).write_text("{}\n", encoding="utf-8")
    verified = SimpleNamespace(ceremony_id=trust.CEREMONY_ID)
    monkeypatch.setattr(trust, "load_production_trust", lambda path: verified)
    monkeypatch.setattr(
        trust,
        "verify_production_trust_for_audit",
        lambda path, verification_time: verified,
    )
    original_rmdir = Path.rmdir

    def fail_staging_rmdir(path):
        if path.name.startswith(".production-trust-"):
            raise OSError("simulated cleanup failure")
        return original_rmdir(path)

    monkeypatch.setattr(Path, "rmdir", fail_staging_rmdir)
    destination = tmp_path / "Config" / "ProductionTrust" / trust.CEREMONY_ID
    assert trust.install_public_production_trust(source, destination) == destination
    assert destination.is_dir()
    assert not list(destination.parent.glob(".production-trust-*"))


def test_signature_import_failures(ceremony_material):
    roots, _, pinned, _, _, payload, context = ceremony_material
    request = build_signing_request(
        artifact_type="RELEASE_POLICY",
        payload=payload,
        pinned_root=pinned,
        context=context,
    )
    arbitrary = replace(context, ceremony_id="0" * 64)
    with pytest.raises(PolicyVectorError, match="context invariant"):
        build_signing_request(
            artifact_type="RELEASE_POLICY",
            payload=payload,
            pinned_root=pinned,
            context=arbitrary,
        )
    mismatched = deepcopy(request)
    mismatched["ceremony_id"] = "0" * 64
    with pytest.raises(PolicyVectorError, match="request invariant"):
        assemble_signed_artifact(
            request=mismatched,
            payload=payload,
            signatures=[],
            pinned_root=pinned,
            context=context,
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
    with pytest.raises(PolicyVectorError, match="invariant"):
        assemble_signed_artifact(
            request=request,
            payload=changed,
            signatures=[],
            pinned_root=pinned,
            context=context,
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


def test_release_and_revocation_from_different_ceremonies_are_rejected(
    ceremony_material,
):
    roots, root_bundle, pinned, _, _, payload, context_a = ceremony_material
    release_request = build_signing_request(
        artifact_type="RELEASE_POLICY",
        payload=payload,
        pinned_root=pinned,
        context=context_a,
    )
    signed_release = assemble_signed_artifact(
        request=release_request,
        payload=payload,
        signatures=[
            _signature(release_request, signer, roots[signer][0]) for signer in ("root-1", "root-2")
        ],
        pinned_root=pinned,
        context=context_a,
    )
    other_release = deepcopy(payload)
    other_release["release_policy_id"] = "OTHER_VALID_CEREMONY"
    context_b = build_ceremony_context(pinned_root=pinned, release_payload=other_release)
    revocation_payload = build_initial_revocation_payload(effective_at="2026-01-01T00:00:00Z")
    revocation_request = build_signing_request(
        artifact_type="INITIAL_REVOCATION",
        payload=revocation_payload,
        pinned_root=pinned,
        context=context_b,
    )
    signed_revocation = assemble_signed_artifact(
        request=revocation_request,
        payload=revocation_payload,
        signatures=[
            _signature(revocation_request, signer, roots[signer][0])
            for signer in ("root-1", "root-2")
        ],
        pinned_root=pinned,
        context=context_b,
    )
    with pytest.raises(PolicyVectorError, match="signing request invariant"):
        verify_ceremony(
            release_request=release_request,
            revocation_request=revocation_request,
            signed_release=signed_release,
            signed_revocation=signed_revocation,
            root_bundle=root_bundle,
            verification_time=NOW,
        )


def test_partial_ceremony_never_publishes_final(ceremony_material, tmp_path):
    roots, _, pinned, _, _, payload, context = ceremony_material
    request = build_signing_request(
        artifact_type="RELEASE_POLICY",
        payload=payload,
        pinned_root=pinned,
        context=context,
    )
    with pytest.raises(PolicyVectorError):
        assemble_signed_artifact(
            request=request,
            payload=payload,
            signatures=[_signature(request, "root-1", roots["root-1"][0])],
            pinned_root=pinned,
            context=context,
        )
    assert not (tmp_path / "final").exists()


def test_production_module_has_no_private_key_operations():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    forbidden = {
        "Ed25519PrivateKey",
        "from_private_bytes",
        "private_bytes",
        "generate_private_key",
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    imports = {
        alias.name.rsplit(".", 1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not forbidden & (names | attributes | imports)
