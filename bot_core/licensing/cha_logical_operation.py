"""Verifier-issued CHA ownership of one durably committed logical operation.

Neither transport nor retained JSON grants authority. Each access requalifies
the exact upstream LPPI capability and the immutable installed CHA mapping.
This boundary stops before account reservation or account genesis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from weakref import WeakKeyDictionary

from .canonical import parse_canonical
from .lppi_authenticated_operation import LPPIAuthenticatedProvisioningOperationBindingV1


class CHALogicalOperationError(ValueError):
    """The caller did not supply an exact, currently qualified CHA capability."""


@dataclass(frozen=True, slots=True)
class _LogicalOperationSnapshot:
    upstream_operation: object
    path: Path
    state_raw: bytes


class VerifiedCHALogicalOperation:
    """Opaque CHA capability issued only after the immutable mapping is durable."""

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("CHA logical operation comes only from the installed verifier")

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("CHA logical operation is immutable")

    @property
    def provisioning_operation_id(self) -> str:
        snapshot = _logical_operation_snapshot(self)
        return cast(str, parse_canonical(snapshot.state_raw)["provisioning_operation_id"])

    @property
    def logical_operation_id(self) -> str:
        snapshot = _logical_operation_snapshot(self)
        return cast(str, parse_canonical(snapshot.state_raw)["logical_operation_id"])

    @property
    def source_binding(self) -> LPPIAuthenticatedProvisioningOperationBindingV1:
        """Return exact validated source transport; the transport grants no authority."""
        snapshot = _logical_operation_snapshot(self)
        state = parse_canonical(snapshot.state_raw)
        upstream = parse_canonical(bytes.fromhex(state["lppi_operation_state_raw_hex"]))
        return LPPIAuthenticatedProvisioningOperationBindingV1.from_canonical_bytes(
            bytes.fromhex(upstream["binding_raw_hex"])
        )

    @property
    def source_tuple(self) -> dict[str, Any]:
        """Return a fresh copy of the frozen source identity after current verification."""
        binding = self.source_binding
        document = binding.document
        return {
            **{
                field: document[field]
                for field in (
                    "environment",
                    "pdsa_trust_domain",
                    "pdsa_package_digest_sha256",
                    "provisioning_subject_id",
                    "enrollment_reference",
                    "provisioning_operation_id",
                    "binding_generation",
                )
            },
            "lppi_authenticated_operation_binding_digest_sha256": binding.digest_sha256,
        }


_ISSUED: WeakKeyDictionary[VerifiedCHALogicalOperation, _LogicalOperationSnapshot] = (
    WeakKeyDictionary()
)


def _logical_operation_snapshot(value: object) -> _LogicalOperationSnapshot:
    from deployment.windows_production_cha_operation import validate_retained_cha_operation

    if type(value) is not VerifiedCHALogicalOperation:
        raise CHALogicalOperationError("VERIFIED_CHA_LOGICAL_OPERATION_REQUIRED")
    snapshot = _ISSUED.get(value)
    if snapshot is None:
        raise CHALogicalOperationError("VERIFIED_CHA_LOGICAL_OPERATION_REQUIRED")
    validate_retained_cha_operation(snapshot.upstream_operation, snapshot.path, snapshot.state_raw)
    return snapshot


def require_verified_cha_logical_operation(value: object) -> VerifiedCHALogicalOperation:
    """Reverify provenance, upstream authority, exact retained bytes and the bijection."""
    _logical_operation_snapshot(value)
    return cast(VerifiedCHALogicalOperation, value)


def _issue_verified_cha_operation(
    upstream_operation: object, path: Path, state_raw: bytes
) -> VerifiedCHALogicalOperation:
    from deployment.windows_production_cha_operation import validate_retained_cha_operation

    validate_retained_cha_operation(upstream_operation, path, state_raw)
    result = object.__new__(VerifiedCHALogicalOperation)
    _ISSUED[result] = _LogicalOperationSnapshot(upstream_operation, path, state_raw)
    return result
