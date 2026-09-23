"""Core-owned composition boundary for production source-Catalog acceptance.

Construction is deliberately separate from provisioning.  Startup may reopen already
provisioned authorities, but it must never create signing custody or admit a producer
as a side effect of composing the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from bot_core.instruments.catalog_admission_receipt import (
    CatalogAdmissionReceiptAuthority,
    SQLiteCatalogAdmissionReceiptMetadataStore,
)
from bot_core.instruments.catalog_runtime_acceptance import CatalogRuntimeAcceptanceAuthority
from bot_core.instruments.source_producer_membership import (
    SQLiteMembershipCarrier,
    SourceProducerMembershipAuthority,
)


_SQLITE_WAL_SIDECAR_SUFFIXES = ("-wal", "-shm")


def _is_reserved_sqlite_sidecar(left: Path, right: Path) -> bool:
    """Return whether either main database occupies the other's WAL namespace."""
    return any(
        right == Path(f"{left}{suffix}") or left == Path(f"{right}{suffix}")
        for suffix in _SQLITE_WAL_SIDECAR_SUFFIXES
    )


def _existing_sqlite_storage(path: Path) -> tuple[Path, ...]:
    """Return the existing main/WAL/SHM objects in one authority namespace."""
    candidates = (path, *(Path(f"{path}{suffix}") for suffix in _SQLITE_WAL_SIDECAR_SUFFIXES))
    return tuple(candidate for candidate in candidates if candidate.exists())


def _has_reserved_sidecar_symlink(path: Path) -> bool:
    """See live and dangling symlink entries without following their targets."""
    return any(Path(f"{path}{suffix}").is_symlink() for suffix in _SQLITE_WAL_SIDECAR_SUFFIXES)


def _has_physical_cross_authority_alias(left: Path, right: Path) -> bool:
    """Compare every existing object across the two SQLite storage namespaces."""
    return any(
        os.path.samefile(left_object, right_object)
        for left_object in _existing_sqlite_storage(left)
        for right_object in _existing_sqlite_storage(right)
        if left_object != left or right_object != right
    )


@dataclass(frozen=True, slots=True)
class CatalogRuntimeAuthorityPaths:
    """Explicit, non-aliasing durable locations owned by the Core runtime."""

    catalog_state: Path
    receipt_metadata: Path

    def __post_init__(self) -> None:
        catalog = Path(self.catalog_state)
        receipts = Path(self.receipt_metadata)
        if not catalog.is_absolute() or not receipts.is_absolute():
            raise ValueError("CATALOG_RUNTIME_PATH_NOT_ABSOLUTE")
        catalog = catalog.resolve(strict=False)
        receipts = receipts.resolve(strict=False)
        if catalog == receipts:
            raise ValueError("CATALOG_RUNTIME_PATH_ALIAS")
        if catalog.exists() and receipts.exists() and os.path.samefile(catalog, receipts):
            raise ValueError("CATALOG_RUNTIME_PHYSICAL_ALIAS")
        if _is_reserved_sqlite_sidecar(catalog, receipts):
            raise ValueError("CATALOG_RUNTIME_SQLITE_SIDECAR_ALIAS")
        if _has_reserved_sidecar_symlink(catalog) or _has_reserved_sidecar_symlink(receipts):
            raise ValueError("CATALOG_RUNTIME_SQLITE_SIDECAR_SYMLINK")
        if _has_physical_cross_authority_alias(catalog, receipts):
            raise ValueError("CATALOG_RUNTIME_SQLITE_SIDECAR_PHYSICAL_ALIAS")
        if catalog.exists() and not catalog.is_file():
            raise ValueError("CATALOG_RUNTIME_STATE_NOT_FILE")
        if receipts.exists() and not receipts.is_file():
            raise ValueError("CATALOG_RECEIPT_METADATA_NOT_FILE")
        if not catalog.parent.is_dir() or not receipts.parent.is_dir():
            raise ValueError("CATALOG_RUNTIME_PARENT_UNAVAILABLE")
        object.__setattr__(self, "catalog_state", catalog)
        object.__setattr__(self, "receipt_metadata", receipts)


def compose_catalog_runtime_acceptance(
    paths: CatalogRuntimeAuthorityPaths,
) -> CatalogRuntimeAcceptanceAuthority:
    """Reopen and compose the exact production authorities without provisioning.

    Missing producer admission or receipt-key provisioning therefore remains a
    fail-closed operational state at the ingestion boundary.
    """
    if type(paths) is not CatalogRuntimeAuthorityPaths:
        raise TypeError("exact CatalogRuntimeAuthorityPaths required")
    if not paths.catalog_state.is_file() or not paths.receipt_metadata.is_file():
        raise ValueError("CATALOG_RUNTIME_STORAGE_MISSING")
    membership = SourceProducerMembershipAuthority(SQLiteMembershipCarrier(paths.catalog_state))
    receipts = CatalogAdmissionReceiptAuthority(
        SQLiteCatalogAdmissionReceiptMetadataStore(paths.receipt_metadata)
    )
    return CatalogRuntimeAcceptanceAuthority(membership, receipts)


__all__ = ["CatalogRuntimeAuthorityPaths", "compose_catalog_runtime_acceptance"]
