"""Canonical operating-system identity for machine execution evidence."""

from __future__ import annotations

import platform


class UnsupportedHostOSError(OSError):
    """Raised when deployment evidence is requested on an unsupported OS."""

    pass


def canonical_host_os(native_name: str | None = None) -> str:
    """Map native Python host names to the canonical evidence labels."""
    detected = platform.system() if native_name is None else native_name
    mapping = {"Linux": "Linux", "Windows": "Windows", "Darwin": "macOS"}
    try:
        return mapping[detected]
    except KeyError as exc:
        raise UnsupportedHostOSError(f"unsupported host OS: {detected!r}") from exc
