"""Verifier-issued proof of durable INITIAL_BINDING, solely for a candidate account."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast
from weakref import WeakKeyDictionary

from .canonical import parse_canonical


class AccountInitialBindingError(ValueError):
    """An exact currently verified reservation capability is required."""


@dataclass(frozen=True, slots=True)
class _InitialBindingSnapshot:
    upstream: object
    path: Path
    state_raw: bytes


class VerifiedAccountGenesisInitialBinding:
    """Opaque reservation proof; neither authorization nor genuine account authority."""

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("initial binding comes only from the installed verifier")

    def __init_subclass__(cls) -> None:
        raise TypeError("initial binding capability cannot be subclassed")

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("initial binding capability is immutable")

    def __copy__(self) -> VerifiedAccountGenesisInitialBinding:
        raise TypeError("initial binding capability cannot be copied")

    def __deepcopy__(self, memo: object) -> VerifiedAccountGenesisInitialBinding:
        raise TypeError("initial binding capability cannot be copied")

    @property
    def account_id(self) -> str:
        snapshot = _initial_binding_snapshot(self)
        return cast(str, parse_canonical(snapshot.state_raw)["account_id"])

    @property
    def logical_operation_id(self) -> str:
        snapshot = _initial_binding_snapshot(self)
        return cast(str, parse_canonical(snapshot.state_raw)["logical_operation_id"])

    @property
    def provisioning_operation_id(self) -> str:
        snapshot = _initial_binding_snapshot(self)
        return cast(str, parse_canonical(snapshot.state_raw)["provisioning_operation_id"])


_ISSUED: WeakKeyDictionary[VerifiedAccountGenesisInitialBinding, _InitialBindingSnapshot] = (
    WeakKeyDictionary()
)


def _initial_binding_snapshot(value: object) -> _InitialBindingSnapshot:
    from deployment.windows_production_cha_account_reservation import (
        validate_retained_initial_binding,
    )

    if type(value) is not VerifiedAccountGenesisInitialBinding:
        raise AccountInitialBindingError("VERIFIED_ACCOUNT_INITIAL_BINDING_REQUIRED")
    snapshot = _ISSUED.get(value)
    if snapshot is None:
        raise AccountInitialBindingError("VERIFIED_ACCOUNT_INITIAL_BINDING_REQUIRED")
    validate_retained_initial_binding(snapshot.upstream, snapshot.path, snapshot.state_raw)
    return snapshot


def require_verified_account_initial_binding(value: object) -> VerifiedAccountGenesisInitialBinding:
    """Reverify provenance, current CHA authority and exact retained INITIAL_BINDING."""
    _initial_binding_snapshot(value)
    return cast(VerifiedAccountGenesisInitialBinding, value)


def _issue_verified_initial_binding(
    upstream: object, path: Path, state_raw: bytes
) -> VerifiedAccountGenesisInitialBinding:
    from deployment.windows_production_cha_account_reservation import (
        validate_retained_initial_binding,
    )

    validate_retained_initial_binding(upstream, path, state_raw)
    result = object.__new__(VerifiedAccountGenesisInitialBinding)
    _ISSUED[result] = _InitialBindingSnapshot(upstream, path, state_raw)
    return result
