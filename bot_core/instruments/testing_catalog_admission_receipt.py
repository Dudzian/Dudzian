"""Deterministic Catalog receipt authority available only through an explicit test path."""

from __future__ import annotations

import hmac
from pathlib import Path

from bot_core.instruments.catalog_admission_receipt import (
    CATALOG_ADMISSION_RECEIPT_DOMAIN,
    CatalogAdmissionReceiptSecureCustody,
    _CatalogAdmissionReceiptAuthorityBase,
    _SQLiteCatalogAdmissionReceiptMetadataStoreBase,
)


class TestSQLiteCatalogAdmissionReceiptMetadataStore(
    _SQLiteCatalogAdmissionReceiptMetadataStoreBase
):
    __test__ = False
    _AUTHORITY_DOMAIN = "cryptohunter.catalog-admission-receipt.test.v1"


class _DeterministicCatalogAdmissionReceiptSecureCustody:
    __slots__ = ("_seed", "_anchor_path")

    def __init__(self, seed: bytes, anchor_path: Path) -> None:
        if type(seed) is not bytes or len(seed) < 32:
            raise ValueError("test-only seed must contain at least 32 bytes")
        self._seed = seed
        self._anchor_path = anchor_path

    def persist(self, custody_handle: str, key_material: bytes) -> None:
        if not custody_handle or len(key_material) != 32:
            raise ValueError("invalid test custody persistence")

    def digest(self, custody_handle: str, payload: bytes) -> bytes | None:
        if not custody_handle:
            return None
        key = hmac.digest(self._seed, custody_handle.encode("utf-8"), "sha256")
        return hmac.digest(key, payload, "sha256")

    def read_anchor(self) -> str | None:
        return self._anchor_path.read_text(encoding="utf-8") if self._anchor_path.exists() else None

    def write_anchor(self, canonical_anchor: str) -> None:
        self._anchor_path.write_text(canonical_anchor, encoding="utf-8")


class TestCatalogAdmissionReceiptAuthority(_CatalogAdmissionReceiptAuthorityBase):
    """Non-production runtime type with deterministic, test-owned key derivation."""

    __test__ = False
    _AUTHORITY_DOMAIN = "cryptohunter.catalog-admission-receipt.test.v1"

    def __init__(self, path: str | Path, *, deterministic_seed: bytes) -> None:
        self._test_seed = deterministic_seed
        metadata_path = Path(path)
        metadata = TestSQLiteCatalogAdmissionReceiptMetadataStore(metadata_path)
        custody: CatalogAdmissionReceiptSecureCustody = (
            _DeterministicCatalogAdmissionReceiptSecureCustody(
                deterministic_seed, metadata_path.with_suffix(metadata_path.suffix + ".anchor")
            )
        )
        super().__init__(metadata, custody)

    def _key_material(self) -> bytes:
        return hmac.digest(self._test_seed, b"catalog-admission-receipt-test-key", "sha256")


assert TestCatalogAdmissionReceiptAuthority._AUTHORITY_DOMAIN != CATALOG_ADMISSION_RECEIPT_DOMAIN

__all__ = ["TestCatalogAdmissionReceiptAuthority", "TestSQLiteCatalogAdmissionReceiptMetadataStore"]
