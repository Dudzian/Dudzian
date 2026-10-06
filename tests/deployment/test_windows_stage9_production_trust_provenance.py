"""Loader provenance tests with a mocked verifier and TEST_ONLY public projections.

These tests create neither production authority material nor enrollment evidence.
"""

from __future__ import annotations

import gc
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from weakref import ref

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

import deployment.windows_stage9_production_trust as trust

NOW = datetime(2026, 10, 6, tzinfo=timezone.utc)
CONTEXT_FIELDS = (
    "ceremony_id",
    "release_payload_digest",
    "release_version",
    "pdsa_keys",
)


@pytest.fixture
def verified_context_loader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    keys = {f"TEST_ONLY_PROVENANCE_{index}": bytes([index]) * 32 for index in range(1, 4)}
    release = SimpleNamespace(
        payload_digest=trust.RELEASE_PAYLOAD_DIGEST,
        release_version=1,
        pinned_root=SimpleNamespace(
            key_set_digest=trust.ROOT_KEY_SET_DIGEST,
            environment="PRODUCTION",
        ),
        pdsa_key_set_digest=trust.PDSA_KEY_SET_DIGEST,
        recovery_public_digest=trust.RECOVERY_PUBLIC_DIGEST,
        purpose="PRODUCTION",
        pdsa_threshold=2,
        pdsa_keys=keys,
    )
    ceremony = SimpleNamespace(ceremony_id=trust.CEREMONY_ID, verified_release=release)
    verified = []

    def verify_package(path: Path, *, verification_time: datetime):
        verified.append((path, verification_time))
        return ceremony

    def public_document(path: Path):
        if path.name == "pdsa_public_bundle.json":
            return {
                "threshold": 2,
                "keys": [
                    {"key_id": key_id, "public_key_hex": public.hex()}
                    for key_id, public in keys.items()
                ],
            }
        assert path.name == "freeze_manifest.json"
        return {"status": "PRODUCTION_ROOT_OF_TRUST_FROZEN"}

    monkeypatch.setattr(trust, "verify_final_package", verify_package)
    monkeypatch.setattr(trust, "_canonical_document", public_document)

    def load_context():
        before = len(verified)
        context = trust.verify_production_trust_for_audit(tmp_path, verification_time=NOW)
        assert len(verified) == before + 1
        assert verified[-1] == (tmp_path, NOW)
        return context

    return load_context


@pytest.fixture
def issued_context(verified_context_loader):
    return verified_context_loader()


def _copied_values(context: trust.ProductionTrustContext) -> dict[str, object]:
    return {name: getattr(context, name) for name in CONTEXT_FIELDS}


def _forged_projection(context: trust.ProductionTrustContext):
    forged = object.__new__(trust.ProductionTrustContext)
    for name, value in _copied_values(context).items():
        object.__setattr__(forged, name, value)
    object.__setattr__(forged, "_capability", context._capability)
    return forged


def test_verified_loader_issues_unchanged_context(issued_context):
    assert trust.require_verified_production_trust_context(issued_context) is issued_context
    with pytest.raises(TypeError, match="immutable"):
        issued_context.release_version = 2


@pytest.mark.parametrize(
    "forged", [None, SimpleNamespace(), object.__new__(trust.ProductionTrustContext)]
)
def test_synthetic_or_uninitialized_context_rejected(forged):
    with pytest.raises(trust.ProductionTrustUnavailable, match="VERIFIED_PRODUCTION"):
        trust.require_verified_production_trust_context(forged)


def test_copied_verified_fields_and_capability_do_not_transfer_provenance(issued_context):
    forged = _forged_projection(issued_context)
    with pytest.raises(trust.ProductionTrustUnavailable, match="VERIFIED_PRODUCTION"):
        trust.require_verified_production_trust_context(forged)


def test_constructor_token_does_not_register_context(issued_context):
    forged = trust.ProductionTrustContext(trust._CONTEXT_TOKEN, **_copied_values(issued_context))
    with pytest.raises(trust.ProductionTrustUnavailable, match="VERIFIED_PRODUCTION"):
        trust.require_verified_production_trust_context(forged)


def test_copied_capability_with_arbitrary_public_authorities_rejected(issued_context):
    forged = _forged_projection(issued_context)
    substituted = MappingProxyType(
        {
            f"TEST_ONLY_FORGED_{index}": Ed25519PublicKey.from_public_bytes(bytes([index]) * 32)
            for index in range(4, 7)
        }
    )
    object.__setattr__(forged, "pdsa_keys", substituted)
    with pytest.raises(trust.ProductionTrustUnavailable, match="VERIFIED_PRODUCTION"):
        trust.require_verified_production_trust_context(forged)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ceremony_id", "TEST_ONLY_FORGED_CEREMONY"),
        ("release_payload_digest", "0" * 64),
        ("release_version", 2),
        ("release_version", True),
        ("_capability", object()),
        ("pdsa_keys", MappingProxyType({})),
    ],
)
def test_mutated_issued_context_rejected(issued_context, field, value):
    object.__setattr__(issued_context, field, value)
    with pytest.raises(trust.ProductionTrustUnavailable, match="VERIFIED_PRODUCTION"):
        trust.require_verified_production_trust_context(issued_context)


def test_equal_public_key_replacement_is_not_an_unchanged_verified_projection(issued_context):
    keys = {
        key_id: Ed25519PublicKey.from_public_bytes(
            key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        )
        for key_id, key in issued_context.pdsa_keys.items()
    }
    object.__setattr__(issued_context, "pdsa_keys", MappingProxyType(keys))
    with pytest.raises(trust.ProductionTrustUnavailable, match="VERIFIED_PRODUCTION"):
        trust.require_verified_production_trust_context(issued_context)


def test_mutated_public_key_backing_mapping_rejected(issued_context):
    backing = next(
        value for value in gc.get_referents(issued_context.pdsa_keys) if isinstance(value, dict)
    )
    key_id = next(iter(backing))
    backing[key_id] = Ed25519PublicKey.from_public_bytes(b"\xff" * 32)
    with pytest.raises(trust.ProductionTrustUnavailable, match="VERIFIED_PRODUCTION"):
        trust.require_verified_production_trust_context(issued_context)


def test_subclass_cannot_transfer_verified_provenance(issued_context):
    class ForgedContext(trust.ProductionTrustContext):
        pass

    forged = object.__new__(ForgedContext)
    for name, value in _copied_values(issued_context).items():
        object.__setattr__(forged, name, value)
    object.__setattr__(forged, "_capability", issued_context._capability)
    with pytest.raises(trust.ProductionTrustUnavailable, match="VERIFIED_PRODUCTION"):
        trust.require_verified_production_trust_context(forged)


def test_context_provenance_does_not_retain_unused_contexts(verified_context_loader):
    disposable = verified_context_loader()
    weak_context = ref(disposable)
    before = len(trust._ISSUED_CONTEXTS)
    del disposable
    gc.collect()
    assert weak_context() is None
    assert len(trust._ISSUED_CONTEXTS) == before - 1


def test_missing_production_package_never_issues_context(tmp_path: Path):
    before = len(trust._ISSUED_CONTEXTS)
    with pytest.raises(trust.ProductionTrustUnavailable, match="PRODUCTION_TRUST_UNAVAILABLE"):
        trust.load_production_trust(tmp_path / "missing-production-package")
    assert len(trust._ISSUED_CONTEXTS) == before


def test_public_package_artifact_must_be_a_canonical_object(tmp_path: Path):
    artifact = tmp_path / "TEST_ONLY_public_document.json"
    artifact.write_bytes(b"[]\n")
    with pytest.raises(trust.PolicyVectorError, match="not an object"):
        trust._canonical_document(artifact)


def test_failed_package_verification_never_issues_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    def rejected_package(path: Path, *, verification_time: datetime):
        raise trust.PolicyVectorError("TEST_ONLY rejected package")

    monkeypatch.setattr(trust, "verify_final_package", rejected_package)
    before = len(trust._ISSUED_CONTEXTS)
    with pytest.raises(trust.ProductionTrustUnavailable, match="PRODUCTION_TRUST_UNAVAILABLE"):
        trust.verify_production_trust_for_audit(tmp_path, verification_time=NOW)
    assert len(trust._ISSUED_CONTEXTS) == before
