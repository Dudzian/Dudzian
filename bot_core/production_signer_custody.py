"""Read-only observations of independently installed Linux signer custody.

An existing native secret can establish a local seed/public-key match. It does
not independently establish exclusive private-key control or another service's
inability to read that key. These diagnostics therefore always remain BLOCKED;
neither a public metadata document nor its service's assertion is attestation.
No signing, provisioning, lifecycle mutation, or enrollment port is exposed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys
import re
from dataclasses import dataclass, field
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from bot_core.local_signing_custody import (
    NativeKeyringSigningSecretReader,
    SigningCredentialSnapshot,
    _decode_record,
    _canonical_signing_identity,
    _provisioning_unique_from_credential_id,
    _SEMANTIC_ROLE,
)
from bot_core.production_root_lifecycle import (
    _installed_boundary,
    _object,
    _protected_bytes,
)
from bot_core.root_proof_issuer_substrate import (
    ProviderIdentity,
    ProviderRole,
    SecurityProfile,
    SecurityProfileIdentity,
    public_key_material_identity,
)

ROOT_SIGNER_CUSTODY_INSTALLATION_FILE = Path("/etc/cryptohunter/root-proof-signer/custody.json")
HISTORY_SIGNER_CUSTODY_INSTALLATION_FILE = Path(
    "/etc/cryptohunter/history-attestation-signer/custody.json"
)
_ROLE = {
    "root_signer": (ProviderRole.ROOT_PROOF_SIGNING, "root-proof-signing.json"),
    "history_signer": (
        ProviderRole.HISTORY_ATTESTATION_SIGNING,
        "history-attestation-signing.json",
    ),
}
_BLOCKERS = (
    "INDEPENDENT_PRIVATE_CONTROL_ATTESTATION_UNAVAILABLE",
    "CROSS_ROLE_NATIVE_KEYRING_ACCESS_DENIAL_UNPROVEN",
)


class ProductionSignerCustodyError(RuntimeError):
    """Installed custody is missing, changed, malformed, or unavailable."""


@dataclass(frozen=True, slots=True)
class ProductionSignerCustodyObservation:
    """Public diagnostic facts, never an issuance or private-control capability."""

    service_role: str
    snapshot: SigningCredentialSnapshot
    metadata_sha256: str
    installation_sha256: str
    local_private_seed_matches_public: bool
    private_control_status: str = field(default="BLOCKED", init=False)
    blockers: tuple[str, ...] = field(default=_BLOCKERS, init=False)

    def public_facts(self) -> dict[str, Any]:
        snapshot = self.snapshot
        identity = snapshot.credential_identity
        return {
            "service_role": self.service_role,
            "provider_role": snapshot.provider_identity.role.value,
            "semantic_role": snapshot.provider_identity.role.value,
            "credential_semantic_role": identity.semantic_role.value,
            "trust_domain": snapshot.provider_identity.security.trust_domain,
            "authority_identity": identity.provider_namespace,
            "identity": identity.credential_identity,
            "key_handle_or_version": identity.key_handle_or_version,
            "custody_lifecycle_namespace": identity.custody_lifecycle_namespace,
            "public_material_identity": snapshot.key_material_identity,
            "key_version": snapshot.key_version,
            "lifecycle_generation": snapshot.lifecycle_generation,
            "state": snapshot.lifecycle_state.value,
            "public_key_b64": base64.b64encode(snapshot.public_key).decode("ascii"),
            "metadata_sha256": self.metadata_sha256,
            "installation_sha256": self.installation_sha256,
            "local_private_seed_matches_public": self.local_private_seed_matches_public,
            "private_control_status": self.private_control_status,
        }


def validate_signer_custody_facts(role: str, facts: object) -> dict[str, Any]:
    """Snapshot untrusted IPC facts; validation cannot establish their truth.

    Only the established local custody metadata V1 identity and lifecycle
    contract is supported. Future key-version or generation schemes fail
    closed here rather than being inferred from an opaque credential handle.
    """
    try:
        if type(role) is not str or role not in _ROLE or type(facts) is not dict:
            raise ValueError("exact signer facts required")
        keys = {
            "service_role",
            "provider_role",
            "semantic_role",
            "credential_semantic_role",
            "trust_domain",
            "authority_identity",
            "identity",
            "key_handle_or_version",
            "custody_lifecycle_namespace",
            "public_material_identity",
            "key_version",
            "lifecycle_generation",
            "state",
            "public_key_b64",
            "metadata_sha256",
            "installation_sha256",
            "local_private_seed_matches_public",
            "private_control_status",
        }
        if set(facts) != keys:
            raise ValueError("exact signer facts fields required")
        value = dict(facts)
        for key in keys - {
            "key_version",
            "lifecycle_generation",
            "local_private_seed_matches_public",
        }:
            if (
                type(value[key]) is not str
                or not value[key]
                or "\0" in value[key]
                or len(value[key]) > 4096
            ):
                raise ValueError("bounded exact signer text required")
        if (
            type(value["key_version"]) is not int
            or value["key_version"] != 1
            or type(value["lifecycle_generation"]) is not int
            or type(value["local_private_seed_matches_public"]) is not bool
            or value["local_private_seed_matches_public"] is not True
            or value["private_control_status"] != "BLOCKED"
            or len(value["trust_domain"]) > 512
            or not value["trust_domain"].strip()
        ):
            raise ValueError("exact non-authorizing signer facts required")
        provider_role = _ROLE[role][0]
        security = SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, value["trust_domain"])
        unique = _provisioning_unique_from_credential_id(value["identity"], provider_role)
        canonical = _canonical_signing_identity(security, provider_role, unique)
        expected = {
            "service_role": role,
            "provider_role": provider_role.value,
            "semantic_role": provider_role.value,
            "credential_semantic_role": _SEMANTIC_ROLE[provider_role].value,
            "authority_identity": canonical.provider_namespace,
            "identity": canonical.credential_id,
            "key_handle_or_version": canonical.key_handle + ":v1",
            "custody_lifecycle_namespace": canonical.lifecycle_namespace,
        }
        if any(value[key] != item for key, item in expected.items()):
            raise ValueError("canonical role and credential binding required")
        legal_generation = {"ACTIVE": {1}, "VERIFY_ONLY": {2}, "REVOKED": {2, 3}}
        if value["lifecycle_generation"] not in legal_generation.get(value["state"], set()):
            raise ValueError("exact lifecycle state/generation required")
        encoded = value["public_key_b64"]
        if len(encoded) != 44:
            raise ValueError("canonical public key required")
        public_key = base64.b64decode(encoded, validate=True)
        if (
            base64.b64encode(public_key).decode("ascii") != encoded
            or value["public_material_identity"] != public_key_material_identity(public_key)
            or any(
                re.fullmatch(r"[0-9a-f]{64}", value[key]) is None
                for key in ("metadata_sha256", "installation_sha256")
            )
        ):
            raise ValueError("public key identity and bounded digests required")
        return value
    except Exception:
        raise ProductionSignerCustodyError("INVALID_UNTRUSTED_SIGNER_CUSTODY_FACTS") from None


def _no_acl(descriptor: int) -> None:
    if sys.platform != "linux":
        raise ProductionSignerCustodyError("EXACT_INSTALLED_LINUX_SIGNER_REQUIRED")
    if any(name.startswith("system.posix_acl_") for name in os.listxattr(descriptor)):
        raise ProductionSignerCustodyError("SIGNER_CUSTODY_ACL_FORBIDDEN")


def _private_bytes(
    path: Path,
    *,
    uid: int,
    gid: int,
    directory_owner: int,
    directory_mode: int,
    file_mode: int,
    pending_rotation: str | None = None,
) -> bytes:
    """Pin root-owned ancestors and an exact role-owned bounded regular file."""
    descriptors: list[int] = []
    try:
        directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(directory)
        for component in path.parts[1:-1]:
            info = os.fstat(directory)
            _no_acl(directory)
            if info.st_uid != 0 or info.st_mode & 0o022:
                raise ProductionSignerCustodyError("PROTECTED_SIGNER_CUSTODY_REQUIRED")
            directory = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
            )
            descriptors.append(directory)
        parent = os.fstat(directory)
        _no_acl(directory)
        if (
            parent.st_uid != directory_owner
            or parent.st_gid != gid
            or stat.S_IMODE(parent.st_mode) != directory_mode
        ):
            raise ProductionSignerCustodyError("EXCLUSIVE_SIGNER_DIRECTORY_REQUIRED")
        if pending_rotation is not None:
            try:
                rotation = os.open(
                    ".signing-rotation",
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory,
                )
            except FileNotFoundError:
                rotation = None
            if rotation is not None:
                descriptors.append(rotation)
                try:
                    os.stat(pending_rotation, dir_fd=rotation, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise ProductionSignerCustodyError("SIGNER_ROTATION_RECOVERY_REQUIRED")
        descriptor = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        _no_acl(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != uid
            or before.st_gid != gid
            or stat.S_IMODE(before.st_mode) != file_mode
            or before.st_nlink != 1
            or not 0 < before.st_size <= 65536
        ):
            raise ProductionSignerCustodyError("EXACT_SIGNER_CUSTODY_FILE_REQUIRED")
        raw = b""
        while len(raw) <= 65536:
            part = os.read(descriptor, min(8192, 65537 - len(raw)))
            if not part:
                break
            raw += part
        after = os.fstat(descriptor)
        fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_gid",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if len(raw) != before.st_size or any(
            getattr(before, key) != getattr(after, key) for key in fields
        ):
            raise ProductionSignerCustodyError("SIGNER_CUSTODY_CHANGED_DURING_READ")
        _no_acl(descriptor)
        return raw
    except OSError:
        raise ProductionSignerCustodyError("PROTECTED_SIGNER_CUSTODY_REQUIRED") from None
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _absolute_path(value: object) -> Path:
    if type(value) is not str or not value or "\0" in value or len(value) > 4096:
        raise ValueError("exact absolute custody path required")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError("canonical absolute custody path required")
    return path


def qualify_installed_signer_custody(role: str) -> ProductionSignerCustodyObservation:
    """Observe only the current installed signer; no caller path or reader injection.

    SQL lifecycle matching belongs to the independently qualified issuer. This
    process cannot turn its own keyring access into independent private-control
    or cross-service isolation evidence.
    """
    if sys.platform != "linux" or type(role) is not str or role not in _ROLE:
        raise ProductionSignerCustodyError("EXACT_INSTALLED_LINUX_SIGNER_REQUIRED")
    try:
        boundary = _installed_boundary()
        _, uid, gid, _ = boundary.principal(role)
        if (os.getuid(), os.geteuid(), os.getgid(), os.getegid()) != (uid, uid, gid, gid):
            raise ProductionSignerCustodyError("EXACT_INSTALLED_SIGNER_PROCESS_REQUIRED")
        if not set(os.getgroups()).issubset({gid}):
            raise ProductionSignerCustodyError("EXCLUSIVE_SIGNER_GROUP_REQUIRED")
        config_path = (
            ROOT_SIGNER_CUSTODY_INSTALLATION_FILE
            if role == "root_signer"
            else HISTORY_SIGNER_CUSTODY_INSTALLATION_FILE
        )
        raw_config = _protected_bytes(config_path)
        protected_config = _private_bytes(
            config_path, uid=0, gid=gid, directory_owner=0, directory_mode=0o750, file_mode=0o640
        )
        if protected_config != raw_config:
            raise ProductionSignerCustodyError("SIGNER_INSTALLATION_CHANGED")
        value = json.loads(raw_config, object_pairs_hook=_object)
        if type(value) is not dict or set(value) != {
            "schema_version",
            "profile",
            "trust_domain",
            "service_role",
            "custody_directory",
            "keyring_index_path",
        }:
            raise ValueError("exact signer installation required")
        if (
            type(value["schema_version"]) is not int
            or value["schema_version"] != 1
            or value["profile"] != "PRODUCTION_LOCAL"
            or value["service_role"] != role
            or type(value["trust_domain"]) is not str
            or value["trust_domain"] != boundary.pin.provisioning.trust_domain
        ):
            raise ValueError("exact production signer installation required")
        directory = _absolute_path(value["custody_directory"])
        index = _absolute_path(value["keyring_index_path"])
        if index.parent != directory or index.name in {item[1] for item in _ROLE.values()}:
            raise ValueError("role-scoped native keyring index required")
        provider_role, filename = _ROLE[role]
        pending_rotation = provider_role.value.lower().replace("_", "-") + ".intent.json"
        raw_metadata = _private_bytes(
            directory / filename,
            uid=uid,
            gid=gid,
            directory_owner=uid,
            directory_mode=0o700,
            file_mode=0o600,
            pending_rotation=pending_rotation,
        )
        # Reject duplicate JSON keys before applying the established custody
        # identity/role/lifecycle decoder; it intentionally retains its contract.
        json.loads(raw_metadata, object_pairs_hook=_object)
        security = SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, value["trust_domain"])
        metadata = _decode_record(
            raw_metadata, expected_security=security, expected_role=provider_role
        )
        reader = NativeKeyringSigningSecretReader(
            index_path=index, security=security, role=provider_role
        )
        encoded = reader.read_existing_authority_secret(
            metadata.protected_private_material_reference
        )
        if type(encoded) is not str or len(encoded) != 44:
            raise ValueError("existing native signing seed required")
        seed = base64.b64decode(encoded, validate=True)
        if len(seed) != 32 or base64.b64encode(seed).decode("ascii") != encoded:
            raise ValueError("canonical existing signing seed required")
        private_key = Ed25519PrivateKey.from_private_bytes(seed)
        derived = private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        if not secrets.compare_digest(derived, metadata.public_key):
            raise ProductionSignerCustodyError("NATIVE_PRIVATE_SEED_PUBLIC_KEY_MISMATCH")
        if (
            raw_metadata
            != _private_bytes(
                directory / filename,
                uid=uid,
                gid=gid,
                directory_owner=uid,
                directory_mode=0o700,
                file_mode=0o600,
                pending_rotation=pending_rotation,
            )
            or raw_config != _protected_bytes(config_path)
            or boundary != _installed_boundary()
        ):
            raise ProductionSignerCustodyError("SIGNER_CUSTODY_CHANGED_DURING_QUALIFICATION")
        snapshot = SigningCredentialSnapshot(
            ProviderIdentity(provider_role, security, metadata.provider_namespace),
            metadata.credential_identity(),
            metadata.key_version,
            metadata.lifecycle_generation,
            metadata.lifecycle_state,
            bytes(metadata.public_key),
            metadata.key_material_identity,
        )
        return ProductionSignerCustodyObservation(
            role,
            snapshot,
            hashlib.sha256(raw_metadata).hexdigest(),
            boundary.source_sha256,
            True,
        )
    except ProductionSignerCustodyError:
        raise
    except Exception:
        raise ProductionSignerCustodyError("INSTALLED_SIGNER_CUSTODY_UNAVAILABLE") from None


__all__ = [
    "ProductionSignerCustodyError",
    "ProductionSignerCustodyObservation",
    "qualify_installed_signer_custody",
    "validate_signer_custody_facts",
]
