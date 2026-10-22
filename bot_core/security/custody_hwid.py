"""Versioned Linux installation identity for isolated production custody only.

The licensing fingerprint remains unchanged.  This identity deliberately has no
MAC, UUID-node, environment, caller-supplied value, or alternate-path fallback.
The host administrator, firmware and installed root filesystem are trust
boundaries. Machine-id plus mandatory DMI UUID is not hardware attestation.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import sys
from contextlib import ExitStack

from bot_core.security.hwid import HwIdProviderError

CUSTODY_HWID_SCHEME = "linux-host-v1"
CUSTODY_HWID_PREFIX = "custody:linux-host-v1:"
_DOMAIN = b"cryptohunter.production-signing-custody\x00linux-host-v1\x00"
_SIGNER_UNITS = frozenset(
    {
        "cryptohunter-root-proof-signer.service",
        "cryptohunter-history-attestation-signer.service",
    }
)


def _require_protected(
    metadata: os.stat_result, *, directory: bool, credential: bool = False
) -> None:
    if sys.platform != "linux":
        raise HwIdProviderError("Linux custody identity is required.")
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if (
        not expected_type(metadata.st_mode)
        or metadata.st_uid not in ({0, os.geteuid()} if credential else {0})
        or metadata.st_mode & (0o222 if credential else 0o022)
    ):
        raise HwIdProviderError("Protected root-owned custody identity source required.")


def _snapshot(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_protected_source(
    parts: tuple[str, ...], *, credential: bool = False, exact_size: bool = True
) -> bytes:
    if sys.platform != "linux":
        raise HwIdProviderError("Linux custody identity is required.")
    # Pin every component, reject symlinks, and check write permissions on the
    # ancestry too. POSIX ACL write access is bounded by these group mode bits.
    with ExitStack() as descriptors:
        directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
        root_fd = os.open("/", directory_flags)
        descriptors.callback(os.close, root_fd)
        _require_protected(os.fstat(root_fd), directory=True)
        parent_fd = root_fd
        for index, part in enumerate(parts[:-1]):
            parent_fd = os.open(part, directory_flags, dir_fd=parent_fd)
            descriptors.callback(os.close, parent_fd)
            is_credential_directory = credential and index == len(parts) - 2
            _require_protected(
                os.fstat(parent_fd), directory=True, credential=is_credential_directory
            )
            if is_credential_directory and not os.fstatvfs(parent_fd).f_flag & os.ST_RDONLY:
                raise HwIdProviderError("Read-only systemd credential mount required.")
        identity_fd = os.open(
            parts[-1],
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent_fd,
        )
        descriptors.callback(os.close, identity_fd)
        before = os.fstat(identity_fd)
        _require_protected(before, directory=False, credential=credential)
        if credential and not os.fstatvfs(identity_fd).f_flag & os.ST_RDONLY:
            raise HwIdProviderError("Read-only systemd credential mount required.")
        value = os.read(identity_fd, 64)
        after = os.fstat(identity_fd)
        if _snapshot(before) != _snapshot(after) or (exact_size and len(value) != before.st_size):
            raise HwIdProviderError("Custody identity source changed during read.")
    return value


def _read_machine_id() -> bytes:
    value = _read_protected_source(("etc", "machine-id"))
    if re.fullmatch(rb"[0-9a-f]{32}\n?", value) is None:
        raise HwIdProviderError("Invalid custody machine identity; no fallback permitted.")
    identity = value.rstrip(b"\n")
    if identity in (b"0" * 32, b"f" * 32):
        raise HwIdProviderError("Uninitialized custody machine identity; no fallback permitted.")
    return identity


def _read_product_uuid(systemd_unit: str | None) -> bytes:
    # systemd PID 1 loads the fixed kernel DMI source for unprivileged signers.
    # Offline administration uses the same protected kernel source directly.
    # Context selects one fixed port, never a fallback or a supplied HWID/path.
    parts: tuple[str, ...]
    if systemd_unit is None:
        parts = ("sys", "devices", "virtual", "dmi", "id", "product_uuid")
    else:
        parts = ("run", "credentials", systemd_unit, "custody-product-uuid")
    value = _read_protected_source(
        parts, credential=systemd_unit is not None, exact_size=systemd_unit is not None
    )
    if re.fullmatch(rb"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\n?", value) is None:
        raise HwIdProviderError("Invalid custody hardware UUID; no fallback permitted.")
    identity = value.rstrip(b"\n").lower()
    compact = identity.replace(b"-", b"")
    if compact in (b"0" * 32, b"f" * 32):
        raise HwIdProviderError("Uninitialized custody hardware UUID; no fallback permitted.")
    return identity


class CustodyHwIdProvider:
    """Read the fixed protected host identity without a requester override port."""

    def __init__(self, *, systemd_unit: str | None = None) -> None:
        if systemd_unit is not None and (
            type(systemd_unit) is not str or systemd_unit not in _SIGNER_UNITS
        ):
            raise HwIdProviderError("Fixed installed signer identity context required.")
        self._systemd_unit = systemd_unit

    def read(self) -> str:
        if sys.platform != "linux":
            raise HwIdProviderError("Linux custody identity is required.")
        try:
            identity = _read_machine_id()
            hardware = _read_product_uuid(self._systemd_unit)
        except (OSError, ValueError) as exc:
            raise HwIdProviderError("Protected custody machine identity unavailable.") from exc
        return (
            CUSTODY_HWID_PREFIX
            + hashlib.sha384(_DOMAIN + identity + b"\x00" + hardware).hexdigest()
        )


__all__ = ["CUSTODY_HWID_SCHEME", "CUSTODY_HWID_PREFIX", "CustodyHwIdProvider"]
