"""Sealed product authority for StateStore physical SQLite schemas."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final


class StateStorePhysicalSchemaError(RuntimeError):
    """Raised when no exact sealed physical-schema decision exists."""


def _compose(entries: tuple[tuple[int, str], ...]) -> MappingProxyType[int, str]:
    composed: dict[int, str] = {}
    for version, fingerprint in entries:
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise StateStorePhysicalSchemaError("schema version must be a positive integer")
        if version in composed:
            raise StateStorePhysicalSchemaError("duplicate physical schema version")
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise StateStorePhysicalSchemaError("schema fingerprint must be lowercase SHA-256")
        composed[version] = fingerprint
    return MappingProxyType(composed)


# This tuple is a build-time product decision frozen by canonical A3.  It is not
# loaded from a StateStore, a backup, runtime configuration, or caller input.
_SEALED_ENTRIES: Final = ((1, "18f9bac7640b66fb1051d5e1bcfe7345c79a8dcb33f417b40009fb049547c680"),)
_SEALED_SCHEMA_FINGERPRINTS: Final = _compose(_SEALED_ENTRIES)


class StateStorePhysicalSchemaRegistry:
    """Read-only facade over the package-sealed physical schema composition."""

    __slots__ = ()

    @property
    def current_version(self) -> int:
        return 1

    def expected_fingerprint(self, version: int) -> str:
        if isinstance(version, bool) or not isinstance(version, int):
            raise StateStorePhysicalSchemaError("schema version must be an integer")
        try:
            return _SEALED_SCHEMA_FINGERPRINTS[version]
        except KeyError as exc:
            raise StateStorePhysicalSchemaError("unknown StateStore schema version") from exc


__all__ = [
    "StateStorePhysicalSchemaError",
    "StateStorePhysicalSchemaRegistry",
]
