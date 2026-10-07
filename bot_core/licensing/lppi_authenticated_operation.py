"""Canonical initial operation binding and verifier-issued ACTIVE authority.

Transport parsing grants no authority. Only the installed durable owner can
publish a capability, and every consumption requalifies the retained ACTIVE
successor, the exact singleton record and its purpose-specific signature.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from weakref import WeakKeyDictionary

from .canonical import canonical_json_bytes, parse_canonical
from .lppi_authority_key import LPPIAuthorityKeyBindingV1, verify_low_s_signature
from .pdsa_enrollment_authorization import _reservation_epoch_milliseconds
from .pre_enrollment import PDSA_TRUST_DOMAIN

PURPOSE_DOMAIN = "CryptoHunter.Stage9.ProvisioningOperation.v1"
DOMAIN = b"CryptoHunter.Stage9.LPPIAuthenticatedProvisioningOperationBinding.v1\x00"
BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "environment",
        "pdsa_trust_domain",
        "pdsa_package_digest_sha256",
        "provisioning_subject_id",
        "enrollment_reference",
        "provisioning_operation_id",
        "binding_generation",
        "created_at_utc",
    }
)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_UUID7 = r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
_UTC = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z\Z")


class LPPIAuthenticatedOperationError(ValueError):
    """Stable fail-closed initial authenticated operation error."""


def _timestamp(value: object) -> datetime:
    if type(value) is not str or _UTC.fullmatch(value) is None:
        raise LPPIAuthenticatedOperationError("INVALID_LPPI_OPERATION_CREATED_AT")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
        _reservation_epoch_milliseconds(parsed)
        return parsed
    except ValueError as exc:
        raise LPPIAuthenticatedOperationError("INVALID_LPPI_OPERATION_CREATED_AT") from exc


def _identity(value: object, prefix: str) -> uuid.UUID:
    if type(value) is not str or re.fullmatch(prefix + _UUID7, value) is None:
        raise LPPIAuthenticatedOperationError("INVALID_LPPI_OPERATION_IDENTITY")
    return uuid.UUID(value[len(prefix) :])


@dataclass(frozen=True, slots=True)
class LPPIAuthenticatedProvisioningOperationBindingV1:
    """Public canonical transport; a parsed or copied binding is never authority."""

    canonical_bytes: bytes

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> LPPIAuthenticatedProvisioningOperationBindingV1:
        if type(raw) is not bytes or len(raw) > 8192:
            raise LPPIAuthenticatedOperationError("INVALID_LPPI_AUTHENTICATED_OPERATION_BINDING")
        try:
            payload = parse_canonical(raw)
            if type(payload) is not dict or set(payload) != BINDING_FIELDS:
                raise ValueError("operation schema mismatch")
            literals = {
                "schema_version": 1,
                "environment": "PRODUCTION",
                "pdsa_trust_domain": PDSA_TRUST_DOMAIN,
                "binding_generation": 1,
            }
            if any(
                type(payload[field]) is not type(expected) or payload[field] != expected
                for field, expected in literals.items()
            ):
                raise ValueError("operation required value mismatch")
            if (
                type(payload["pdsa_package_digest_sha256"]) is not str
                or _HEX64.fullmatch(payload["pdsa_package_digest_sha256"]) is None
            ):
                raise ValueError("operation package digest mismatch")
            _identity(payload["provisioning_subject_id"], "psub_")
            _identity(payload["enrollment_reference"], "penr_")
            operation = _identity(payload["provisioning_operation_id"], "prvop_")
            created = _timestamp(payload["created_at_utc"])
            if operation.int >> 80 != _reservation_epoch_milliseconds(created):
                raise ValueError("operation reservation instant mismatch")
        except (ValueError, TypeError, KeyError, RecursionError) as exc:
            raise LPPIAuthenticatedOperationError(
                "INVALID_LPPI_AUTHENTICATED_OPERATION_BINDING"
            ) from exc
        return cls(raw)

    @classmethod
    def from_mapping(
        cls, payload: dict[str, Any]
    ) -> LPPIAuthenticatedProvisioningOperationBindingV1:
        return cls.from_canonical_bytes(canonical_json_bytes(payload))

    @property
    def document(self) -> dict[str, Any]:
        validated = self.from_canonical_bytes(self.canonical_bytes)
        return cast(dict[str, Any], parse_canonical(validated.canonical_bytes))

    @property
    def digest_sha256(self) -> str:
        validated = self.from_canonical_bytes(self.canonical_bytes)
        return hashlib.sha256(validated.canonical_bytes).hexdigest()


def authenticated_operation_signed_bytes(binding_raw: bytes) -> bytes:
    binding = LPPIAuthenticatedProvisioningOperationBindingV1.from_canonical_bytes(binding_raw)
    return DOMAIN + hashlib.sha256(binding.canonical_bytes).digest()


def _active_operation_material(
    active_authority: object,
) -> tuple[dict[str, Any], bytes, bytes]:
    """Read one immutable snapshot immediately after full current qualification.

    Each boundary invokes the production guard independently. This snapshot is
    local to the invocation; reading it never caches or skips a future guard.
    """
    from deployment import windows_production_lppi_authority as lifecycle

    active = lifecycle.require_verified_active_lppi_authority_key(active_authority)
    snapshot = lifecycle._ACTIVE.get(active)
    if snapshot is None:
        raise LPPIAuthenticatedOperationError("VERIFIED_ACTIVE_LPPI_AUTHORITY_KEY_REQUIRED")
    state = parse_canonical(snapshot.state_raw)
    binding = LPPIAuthorityKeyBindingV1.from_canonical_bytes(
        bytes.fromhex(state["binding_raw_hex"])
    )
    package = snapshot.accepted.package
    payload = package.payload
    source = {
        "pdsa_trust_domain": payload["pdsa_trust_domain"],
        "pdsa_package_digest_sha256": package.package_digest_sha256,
        "provisioning_subject_id": payload["provisioning_subject_id"],
        "enrollment_reference": payload["enrollment_reference"],
        "lppi_authority_key_binding_digest_sha256": binding.digest_sha256,
        **{
            field: binding.document[field]
            for field in (
                "lppi_authority_public_key_algorithm_profile",
                "lppi_authority_public_key_fingerprint_sha256",
                "custody_profile",
            )
        },
    }
    return source, binding.canonical_bytes, bytes.fromhex(state["authority_sec1_hex"])


def operation_authority_tuple(active_authority: object) -> dict[str, Any]:
    """Derive the exact request identity solely from the current ACTIVE lineage."""
    source, _, _ = _active_operation_material(active_authority)
    return source


def reservation_digest(authority_tuple: dict[str, Any]) -> str:
    """Purpose-separated conflict identity; the domain does not derive UUID bits."""
    return hashlib.sha256(
        PURPOSE_DOMAIN.encode("utf-8") + b"\x00" + canonical_json_bytes(authority_tuple)
    ).hexdigest()


def build_reserved_operation_binding(
    active_authority: object, provisioning_operation_id: str, created_at_utc: str
) -> LPPIAuthenticatedProvisioningOperationBindingV1:
    """Build transport for the durable owner; this helper confers no capability."""
    source = operation_authority_tuple(active_authority)
    return LPPIAuthenticatedProvisioningOperationBindingV1.from_mapping(
        {
            "schema_version": 1,
            "environment": "PRODUCTION",
            **{
                field: source[field]
                for field in (
                    "pdsa_trust_domain",
                    "pdsa_package_digest_sha256",
                    "provisioning_subject_id",
                    "enrollment_reference",
                )
            },
            "provisioning_operation_id": provisioning_operation_id,
            "binding_generation": 1,
            "created_at_utc": created_at_utc,
        }
    )


def verify_authenticated_operation_binding(
    binding_raw: bytes, signature: bytes, active_authority: object
) -> LPPIAuthenticatedProvisioningOperationBindingV1:
    """Requalify ACTIVE and verify exact source and strict low-S DER signature."""
    source, _, public = _active_operation_material(active_authority)
    binding = LPPIAuthenticatedProvisioningOperationBindingV1.from_canonical_bytes(binding_raw)
    if any(binding.document[field] != source[field] for field in BINDING_FIELDS & source.keys()):
        raise LPPIAuthenticatedOperationError("LPPI_AUTHENTICATED_OPERATION_CONFLICT")
    verify_low_s_signature(
        public,
        signature,
        authenticated_operation_signed_bytes(binding.canonical_bytes),
        error="REJECT_LPPI_AUTHENTICATED_OPERATION_SIGNATURE",
    )
    return binding


@dataclass(frozen=True, slots=True)
class _OperationSnapshot:
    active_authority: object
    path: Path
    state_raw: bytes


class VerifiedLPPIAuthenticatedProvisioningOperation:
    """Opaque committed authority; every access revalidates its private snapshot."""

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("authenticated LPPI operation comes only from the installed verifier")

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("authenticated LPPI operation is immutable")

    @property
    def binding(self) -> LPPIAuthenticatedProvisioningOperationBindingV1:
        snapshot = _operation_snapshot(self)
        document = parse_canonical(snapshot.state_raw)
        return LPPIAuthenticatedProvisioningOperationBindingV1.from_canonical_bytes(
            bytes.fromhex(document["binding_raw_hex"])
        )

    @property
    def signature(self) -> bytes:
        snapshot = _operation_snapshot(self)
        return bytes.fromhex(parse_canonical(snapshot.state_raw)["signature_hex"])

    @property
    def provisioning_operation_id(self) -> str:
        return cast(str, self.binding.document["provisioning_operation_id"])


_ISSUED: WeakKeyDictionary[VerifiedLPPIAuthenticatedProvisioningOperation, _OperationSnapshot] = (
    WeakKeyDictionary()
)


def _operation_snapshot(value: object) -> _OperationSnapshot:
    from deployment.windows_production_lppi_operation import validate_retained_operation

    if type(value) is not VerifiedLPPIAuthenticatedProvisioningOperation:
        raise LPPIAuthenticatedOperationError("VERIFIED_LPPI_AUTHENTICATED_OPERATION_REQUIRED")
    snapshot = _ISSUED.get(value)
    if snapshot is None:
        raise LPPIAuthenticatedOperationError("VERIFIED_LPPI_AUTHENTICATED_OPERATION_REQUIRED")
    validate_retained_operation(snapshot.active_authority, snapshot.path, snapshot.state_raw)
    return snapshot


def require_verified_lppi_authenticated_operation(
    value: object,
) -> VerifiedLPPIAuthenticatedProvisioningOperation:
    _operation_snapshot(value)
    return cast(VerifiedLPPIAuthenticatedProvisioningOperation, value)


def _issue_verified_operation(
    active_authority: object, path: Path, state_raw: bytes
) -> VerifiedLPPIAuthenticatedProvisioningOperation:
    from deployment.windows_production_lppi_operation import validate_retained_operation

    validate_retained_operation(active_authority, path, state_raw)
    result = object.__new__(VerifiedLPPIAuthenticatedProvisioningOperation)
    _ISSUED[result] = _OperationSnapshot(active_authority, path, state_raw)
    return result
