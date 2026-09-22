"""Canonical CoreHost lifecycle gate for M0.12 Catalog runtime acceptance."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat
from typing import Mapping

from bot_core.instruments.catalog_runtime_acceptance import (
    AcceptedSourceCatalogSnapshot,
    CatalogRuntimeAcceptanceAuthority,
    _BinanceSpotCatalogProducer,
)
from bot_core.instruments.catalog_runtime_composition import (
    CatalogRuntimeAuthorityPaths,
    compose_catalog_runtime_acceptance,
)
from bot_core.instruments.core_time import PRODUCTION_CORE_CLOCK

_POSIX_PERMISSION_MODEL = os.name == "posix"


@dataclass(frozen=True, slots=True)
class CatalogRuntimeDeploymentConfiguration:
    catalog_state_path: Path
    receipt_metadata_path: Path

    @classmethod
    def from_protected_mapping(
        cls, value: object
    ) -> "CatalogRuntimeDeploymentConfiguration":
        if type(value) is not dict or set(value) != {
            "catalog_state_path", "receipt_metadata_path"
        }:
            raise ValueError("CATALOG_RUNTIME_CONFIGURATION_INVALID")
        if any(type(value[key]) is not str for key in value):
            raise ValueError("CATALOG_RUNTIME_CONFIGURATION_INVALID")
        paths = CatalogRuntimeAuthorityPaths(
            Path(value["catalog_state_path"]), Path(value["receipt_metadata_path"])
        )
        return cls(paths.catalog_state, paths.receipt_metadata)

    @property
    def paths(self) -> CatalogRuntimeAuthorityPaths:
        return CatalogRuntimeAuthorityPaths(
            self.catalog_state_path, self.receipt_metadata_path
        )


@dataclass(frozen=True, slots=True)
class CatalogRuntimeStartupResult:
    ready: bool
    reason: str


class CoreHostCatalogRuntimeLifecycle:
    """CoreHost-owned, reopen-only gate; it never repairs or provisions storage."""

    __slots__ = ("_configuration", "_runtime", "_result")

    def __init__(self, configuration: CatalogRuntimeDeploymentConfiguration | None) -> None:
        if configuration is not None and type(configuration) is not CatalogRuntimeDeploymentConfiguration:
            raise TypeError("exact CatalogRuntimeDeploymentConfiguration required")
        self._configuration = configuration
        self._runtime: CatalogRuntimeAcceptanceAuthority | None = None
        self._result = CatalogRuntimeStartupResult(False, "NOT_STARTED")

    @property
    def result(self) -> CatalogRuntimeStartupResult:
        return self._result

    def start_after_recovery(self) -> CatalogRuntimeStartupResult:
        if self._result.reason != "NOT_STARTED":
            return self._result
        if self._configuration is None:
            return self._block("CATALOG_RUNTIME_CONFIGURATION_MISSING")
        paths = self._configuration.paths
        if not paths.catalog_state.is_file() or not paths.receipt_metadata.is_file():
            return self._block("CATALOG_RUNTIME_STORAGE_MISSING")
        if not _POSIX_PERMISSION_MODEL:
            return self._block("CATALOG_RUNTIME_PLATFORM_PERMISSION_PROOF_UNAVAILABLE")
        if not self._permissions_qualified(paths):
            return self._block("CATALOG_RUNTIME_PERMISSIONS_INVALID")
        try:
            runtime = compose_catalog_runtime_acceptance(paths)
        except Exception:
            return self._block("CATALOG_RUNTIME_AUTHORITY_REPLAY_FAILED")
        if not runtime._receipts.qualified_for_runtime():
            return self._block("CATALOG_RUNTIME_RECEIPT_CUSTODY_MISSING")
        grant = runtime._membership.resolve_current(
            _BinanceSpotCatalogProducer.identity,
            _BinanceSpotCatalogProducer.generation,
            PRODUCTION_CORE_CLOCK.now_utc(),
        )
        if grant is None:
            return self._block("CATALOG_RUNTIME_PRODUCER_ADMISSION_MISSING")
        self._runtime = runtime
        self._result = CatalogRuntimeStartupResult(True, "READY_FOR_EXPLICIT_ONE_SHOT_FETCH")
        return self._result

    @staticmethod
    def _permissions_qualified(paths: CatalogRuntimeAuthorityPaths) -> bool:
        uid = os.geteuid()
        directories = {paths.catalog_state.parent, paths.receipt_metadata.parent}
        return all(
            path.stat().st_uid == uid and stat.S_IMODE(path.stat().st_mode) == 0o700
            for path in directories
        ) and all(
            path.stat().st_uid == uid and stat.S_IMODE(path.stat().st_mode) == 0o600
            for path in (paths.catalog_state, paths.receipt_metadata)
        )

    def _block(self, reason: str) -> CatalogRuntimeStartupResult:
        self._runtime = None
        self._result = CatalogRuntimeStartupResult(False, reason)
        return self._result

    def fetch_catalog_once(self) -> AcceptedSourceCatalogSnapshot | None:
        if not self._result.ready or self._runtime is None:
            return None
        return self._runtime.fetch_catalog_once("core_release_1_46_binance_spot")


__all__ = [
    "CatalogRuntimeDeploymentConfiguration",
    "CatalogRuntimeStartupResult",
    "CoreHostCatalogRuntimeLifecycle",
]
