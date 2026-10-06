"""Canonical product-side activation of the public Stage-9 production trust package."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping
from weakref import WeakKeyDictionary

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from deployment.windows_stage9_evidence_contract import PRODUCTION_TRUST_CEREMONY_ID
from deployment.windows_stage9_policy_material import (
    PolicyVectorError,
    canonical_json_bytes,
)
from deployment.windows_stage9_production_ceremony import verify_final_package

CEREMONY_ID = PRODUCTION_TRUST_CEREMONY_ID
PUBLIC_PACKAGE_FILENAMES = frozenset(
    {
        "root_anchor_bundle.json",
        "pdsa_public_bundle.json",
        "recovery_public_bundle.json",
        "unsigned_release_policy.json",
        "release_signing_request.json",
        "signed_release_policy.json",
        "initial_revocation_payload.json",
        "revocation_signing_request.json",
        "signed_initial_revocation.json",
        "freeze_manifest.json",
        "ceremony_audit.json",
        "package_manifest.json",
    }
)
RELEASE_PAYLOAD_DIGEST = "7ade98f8f1d3573b243556b6404d119a40429f6e1d738ebbab6698d64293c5a9"
# Public authority-set digests, not key material. Keep their fixed pairing explicit.
ROOT_KEY_SET_DIGEST, PDSA_KEY_SET_DIGEST = (
    "d0e2ff6672da5979688433fe5f9cac9bee0844baa1a307b52b36c0e37e6a5e7a",
    "3203b496f67d571787e59bb74d62f69fc1bda9debda74cab3294b546f5884c14",
)
RECOVERY_PUBLIC_DIGEST = "06e957b575380ee019ee311d9f149c8b888a2f3b4fa1ae7e4168bc3dfe2584b7"


class ProductionTrustUnavailable(RuntimeError):
    """The installed public package is absent or does not match frozen production trust."""


class ProductionTrustContext:
    """Opaque immutable projection produced only after full final-package verification."""

    __slots__ = (
        "ceremony_id",
        "release_payload_digest",
        "release_version",
        "pdsa_keys",
        "_capability",
        "__weakref__",
    )

    ceremony_id: str
    release_payload_digest: str
    release_version: int
    pdsa_keys: Mapping[str, Ed25519PublicKey]
    _capability: object

    def __init__(self, token: object, **values: object) -> None:
        if token is not _CONTEXT_TOKEN:
            raise TypeError("ProductionTrustContext comes only from load_production_trust")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_capability", _CONTEXT_CAPABILITY)

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("ProductionTrustContext is immutable")


_CONTEXT_TOKEN = object()
_CONTEXT_CAPABILITY = object()


@dataclass(frozen=True, slots=True)
class _VerifiedTrustSnapshot:
    ceremony_id: str
    release_payload_digest: str
    release_version: int
    pdsa_keys: Mapping[str, Ed25519PublicKey]
    pdsa_public_bytes: tuple[tuple[str, bytes], ...]


# Registration belongs to the package verifier, never to the public constructor.
# Weak keys retain provenance only while the verified projection is in use.
_ISSUED_CONTEXTS: WeakKeyDictionary[ProductionTrustContext, _VerifiedTrustSnapshot] = (
    WeakKeyDictionary()
)
_RUNTIME_CONTEXTS: WeakKeyDictionary[ProductionTrustContext, Path] = WeakKeyDictionary()


def _pdsa_public_snapshot(
    keys: Mapping[str, Ed25519PublicKey],
) -> tuple[tuple[str, bytes], ...]:
    if any(type(key_id) is not str for key_id in keys):
        raise ValueError("invalid PDSA key identifier")
    return tuple(
        (
            key_id,
            key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw),
        )
        for key_id, key in sorted(keys.items())
    )


def require_verified_production_trust_context(
    context: object,
) -> ProductionTrustContext:
    """Require loader-issued provenance and its unchanged verified projection."""
    if type(context) is not ProductionTrustContext:
        raise ProductionTrustUnavailable("VERIFIED_PRODUCTION_TRUST_CONTEXT_REQUIRED")
    try:
        snapshot = _ISSUED_CONTEXTS.get(context)
        valid = (
            snapshot is not None
            and context._capability is _CONTEXT_CAPABILITY
            and type(context.ceremony_id) is str
            and context.ceremony_id == CEREMONY_ID
            and type(context.release_payload_digest) is str
            and context.release_payload_digest == RELEASE_PAYLOAD_DIGEST
            and type(context.release_version) is int
            and context.release_version >= 1
            and isinstance(context.pdsa_keys, MappingProxyType)
            and len(context.pdsa_keys) == 3
            and all(isinstance(key, Ed25519PublicKey) for key in context.pdsa_keys.values())
        )
        if valid and snapshot is not None:
            valid = (
                context.ceremony_id == snapshot.ceremony_id
                and context.release_payload_digest == snapshot.release_payload_digest
                and context.release_version == snapshot.release_version
                and context.pdsa_keys is snapshot.pdsa_keys
                and _pdsa_public_snapshot(context.pdsa_keys) == snapshot.pdsa_public_bytes
            )
    except (AttributeError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ProductionTrustUnavailable("VERIFIED_PRODUCTION_TRUST_CONTEXT_REQUIRED")
    return context


def _canonical_document(path: Path) -> dict:
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise PolicyVectorError("final package artifact lacks canonical newline")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise PolicyVectorError("final package artifact is not an object")
    if raw != canonical_json_bytes(value) + b"\n":
        raise PolicyVectorError("final package artifact is noncanonical")
    return value


def verify_production_trust_for_audit(
    path: Path, *, verification_time: datetime
) -> ProductionTrustContext:
    """Audit/test API with an explicit historical verification instant."""
    try:
        for artifact in path.glob("*.json"):
            _canonical_document(artifact)
        ceremony = verify_final_package(path, verification_time=verification_time)
        release = ceremony.verified_release
        pdsa = _canonical_document(path / "pdsa_public_bundle.json")
        freeze = _canonical_document(path / "freeze_manifest.json")
        bindings = (
            ceremony.ceremony_id,
            release.payload_digest,
            release.pinned_root.key_set_digest,
            release.pdsa_key_set_digest,
            release.recovery_public_digest,
            release.purpose,
            release.pinned_root.environment,
            release.pdsa_threshold,
            freeze.get("status"),
        )
        expected = (
            CEREMONY_ID,
            RELEASE_PAYLOAD_DIGEST,
            ROOT_KEY_SET_DIGEST,
            PDSA_KEY_SET_DIGEST,
            RECOVERY_PUBLIC_DIGEST,
            "PRODUCTION",
            "PRODUCTION",
            2,
            "PRODUCTION_ROOT_OF_TRUST_FROZEN",
        )
        if bindings != expected or pdsa.get("threshold") != 2:
            raise PolicyVectorError("final package does not match frozen production authority")
        keys = {
            item["key_id"]: Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(item["public_key_hex"])
            )
            for item in pdsa["keys"]
        }
        if tuple(keys) != tuple(release.pdsa_keys) or len(keys) != 3:
            raise PolicyVectorError("PDSA projection differs from signed release")
        context = ProductionTrustContext(
            _CONTEXT_TOKEN,
            ceremony_id=ceremony.ceremony_id,
            release_payload_digest=release.payload_digest,
            release_version=release.release_version,
            pdsa_keys=MappingProxyType(keys),
        )
        _ISSUED_CONTEXTS[context] = _VerifiedTrustSnapshot(
            context.ceremony_id,
            context.release_payload_digest,
            context.release_version,
            context.pdsa_keys,
            _pdsa_public_snapshot(context.pdsa_keys),
        )
        return require_verified_production_trust_context(context)
    except (OSError, ValueError, KeyError, TypeError, PolicyVectorError) as exc:
        raise ProductionTrustUnavailable(f"PRODUCTION_TRUST_UNAVAILABLE: {exc}") from exc


def load_production_trust(path: Path) -> ProductionTrustContext:
    """Runtime API: verify production trust only at the current UTC instant."""
    context = verify_production_trust_for_audit(path, verification_time=datetime.now(timezone.utc))
    _RUNTIME_CONTEXTS[context] = path.resolve()
    return context


def require_current_production_trust_context(context: object) -> ProductionTrustContext:
    """Require runtime provenance and reverify installed authority at current UTC.

    Historical audit capabilities cannot authorize new production challenges or
    enrollment proofs. A runtime capability also ceases to authorize use when its
    package disappears, expires, changes authority, or changes release identity.
    """
    trusted = require_verified_production_trust_context(context)
    path = _RUNTIME_CONTEXTS.get(trusted)
    if path is None:
        raise ProductionTrustUnavailable("CURRENT_RUNTIME_PRODUCTION_TRUST_CONTEXT_REQUIRED")
    current = verify_production_trust_for_audit(path, verification_time=datetime.now(timezone.utc))
    if (
        current.ceremony_id != trusted.ceremony_id
        or current.release_payload_digest != trusted.release_payload_digest
        or current.release_version != trusted.release_version
        or _pdsa_public_snapshot(current.pdsa_keys) != _pdsa_public_snapshot(trusted.pdsa_keys)
    ):
        raise ProductionTrustUnavailable("CURRENT_PRODUCTION_TRUST_IDENTITY_CHANGED")
    return trusted


def validate_public_package_layout(path: Path) -> None:
    """Enforce the operator artifact's directory identity and exact flat layout."""
    if path.name != CEREMONY_ID:
        raise ProductionTrustUnavailable("PUBLIC_FINAL_PACKAGE_CEREMONY_DIRECTORY_REQUIRED")
    if path.is_symlink() or not path.is_dir():
        raise ProductionTrustUnavailable("PUBLIC_FINAL_PACKAGE_DIRECTORY_REQUIRED")
    entries = tuple(path.iterdir())
    if {item.name for item in entries} != PUBLIC_PACKAGE_FILENAMES or any(
        item.is_symlink() or not item.is_file() for item in entries
    ):
        raise ProductionTrustUnavailable("PUBLIC_FINAL_PACKAGE_ALLOWLIST_VIOLATION")


def install_public_production_trust(
    source: Path,
    destination: Path,
    report_stage: Callable[[str], None] | None = None,
) -> Path:
    """Verify, stage, byte-qualify and atomically publish the public package once."""
    report = report_stage or (lambda stage: None)
    if destination.exists():
        raise ProductionTrustUnavailable("PRODUCTION_TRUST_OVERWRITE_FORBIDDEN")
    validate_public_package_layout(source)
    # These labels are deliberately stable and contain no caller-controlled data.
    report("SOURCE_CRYPTOGRAPHIC_VERIFICATION")
    context = load_production_trust(source)
    required = PUBLIC_PACKAGE_FILENAMES
    parent_preexisted = destination.parent.exists()
    staging_root: Path | None = None
    preexisting_staging = (
        set(destination.parent.glob(".production-trust-*")) if parent_preexisted else set()
    )
    try:
        report("DESTINATION_PARENT_CREATION")
        destination.parent.mkdir(parents=True, exist_ok=True)
        report("STAGING_CREATION")
        staging_root = Path(tempfile.mkdtemp(prefix=".production-trust-", dir=destination.parent))
        staging = staging_root / CEREMONY_ID
        staging.mkdir()
        report("STAGING_COPY")
        for name in required:
            raw = (source / name).read_bytes()
            if any(marker in raw.lower() for marker in (b"private key", b"password", b"seed")):
                raise ProductionTrustUnavailable("PRIVATE_MATERIAL_IN_PUBLIC_PACKAGE")
            (staging / name).write_bytes(raw)
            if (staging / name).read_bytes() != raw:
                raise ProductionTrustUnavailable("PRODUCTION_TRUST_COPY_MISMATCH")
        report("STAGING_VERIFICATION")
        staged = verify_production_trust_for_audit(
            staging, verification_time=datetime.now(timezone.utc)
        )
        if staged.ceremony_id != context.ceremony_id:
            raise ProductionTrustUnavailable("PRODUCTION_TRUST_COPY_MISMATCH")
        report("ATOMIC_PUBLISH")
        os.rename(staging, destination)
    except Exception:
        if staging_root is not None:
            shutil.rmtree(staging_root, ignore_errors=True)
        if destination.parent.is_dir():
            for residue in destination.parent.glob(".production-trust-*"):
                if residue not in preexisting_staging:
                    shutil.rmtree(residue, ignore_errors=True)
        if not parent_preexisted:
            try:
                destination.parent.rmdir()
            except OSError:
                pass
        raise
    # The rename is the commit point.  Failure to remove an already-empty
    # staging container must never turn a committed publication into an
    # ambiguous exception visible to the caller.
    try:
        staging_root.rmdir()
    except OSError:
        shutil.rmtree(staging_root, ignore_errors=True)
    return destination
