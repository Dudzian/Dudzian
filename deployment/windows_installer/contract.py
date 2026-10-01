"""Frozen, reviewable constants for the Stage-9 MSI."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class InstallerContract:
    product_name: str = "CryptoHunter"
    manufacturer: str = "CryptoHunter"
    architecture: str = "x64"
    scope: str = "perMachine"
    upgrade_code: str = "D79EA891-1C35-4CBC-B761-C5A5A46D8C70"
    wix_version: str = "7.0.0"
    postgresql_version: str = "17.11"
    postgresql_packaging_revision: str = "4"
    # Product-owned, deterministic, loopback-only endpoint.  5432 is not used.
    postgresql_host: str = "127.0.0.1"
    postgresql_port: int = 55432
    backend_service: str = "CryptoHunterBackend"
    verifier_service: str = "CryptoHunterFreshnessVerifier"
    postgresql_service: str = "CryptoHunterPostgreSQL"
    postgresql_display_name: str = "CryptoHunter Private PostgreSQL"

    @property
    def backend_identity(self) -> str:
        return rf"NT SERVICE\{self.backend_service}"

    @property
    def verifier_identity(self) -> str:
        return rf"NT SERVICE\{self.verifier_service}"

    @property
    def postgresql_identity(self) -> str:
        return rf"NT SERVICE\{self.postgresql_service}"


CONTRACT = InstallerContract()


def is_reviewed_wix_version(observed: object) -> bool:
    """Return whether WiX reports the pinned release, with optional build metadata."""
    return isinstance(observed, str) and re.fullmatch(
        rf"{re.escape(CONTRACT.wix_version)}(?:\+[0-9A-Za-z.-]+)?", observed.strip()
    ) is not None
