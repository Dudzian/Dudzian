"""Instalator presetów Marketplace wraz z walidacją licencji."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from bot_core.marketplace import PresetDocument, PresetRepository, parse_preset_document
from bot_core.security.hwid import HwIdProvider, HwIdProviderError

from .marketplace import MarketplaceCatalog, MarketplaceCatalogError, MarketplacePreset, load_catalog

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class MarketplaceInstallResult:
    """Raport z instalacji presetu Marketplace."""

    preset_id: str
    version: str | None
    success: bool
    installed_path: Path | None
    signature_verified: bool
    fingerprint_verified: bool | None
    issues: tuple[str, ...] = field(default_factory=tuple)
    license_payload: Mapping[str, Any] | None = None


@dataclass(slots=True)
class _EvaluationResult:
    success: bool
    signature_verified: bool
    fingerprint_verified: bool | None
    issues: tuple[str, ...]
    license_payload: Mapping[str, Any] | None


class MarketplacePresetInstaller:
    """Instaluje presety Marketplace z weryfikacją podpisu i licencji."""

    def __init__(
        self,
        repository: PresetRepository,
        *,
        catalog: MarketplaceCatalog | None = None,
        catalog_path: str | Path | None = None,
        licenses_dir: str | Path | None = None,
        signing_keys: Mapping[str, bytes | str] | None = None,
        hwid_provider: HwIdProvider | None = None,
    ) -> None:
        self._repository = repository
        if catalog is not None:
            self._catalog = catalog
        else:
            base_path = (
                Path(catalog_path).expanduser().resolve()
                if catalog_path is not None
                else Path(__file__).resolve().parent
            )
            self._catalog = load_catalog(base_path)
        self._licenses_dir = Path(licenses_dir).expanduser().resolve() if licenses_dir else None
        self._signing_keys = signing_keys
        self._hwid_provider = hwid_provider or HwIdProvider()

    # ------------------------------------------------------------------
    def list_available(self) -> Sequence[MarketplacePreset]:
        """Zwraca listę presetów dostępnych w katalogu Marketplace."""

        return self._catalog.presets

    # ------------------------------------------------------------------
    def install_from_catalog(self, preset_id: str) -> MarketplaceInstallResult:
        """Instaluje preset na podstawie wpisu w katalogu."""

        preset = self._catalog.find(preset_id)
        if preset is None:
            raise MarketplaceCatalogError(f"Preset {preset_id!r} nie istnieje w katalogu Marketplace.")
        return self.install_from_path(preset.artifact_path)

    # ------------------------------------------------------------------
    def install_from_path(self, path: str | Path) -> MarketplaceInstallResult:
        """Instaluje preset z lokalnego pliku, weryfikując podpis i licencję."""

        path = Path(path).expanduser().resolve()
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"Nie udało się odczytać pliku presetu {path}: {exc}") from exc

        document = parse_preset_document(payload, source=path, signing_keys=self._signing_keys)
        evaluation = self._evaluate_document(document)

        installed_path: Path | None = None
        if evaluation.success:
            installed_doc = self._repository.import_payload(
                payload,
                filename=path.name,
                signing_keys=self._signing_keys,
                require_signature=False,
            )
            installed_path = installed_doc.path
        else:
            LOGGER.warning(
                "Instalacja presetu %s zakończyła się problemami: issues=%s",
                document.preset_id,
                ",".join(evaluation.issues) or "brak",
            )

        return MarketplaceInstallResult(
            preset_id=document.preset_id,
            version=document.version,
            success=evaluation.success,
            installed_path=installed_path,
            signature_verified=evaluation.signature_verified,
            fingerprint_verified=evaluation.fingerprint_verified,
            issues=evaluation.issues,
            license_payload=evaluation.license_payload,
        )

    # ------------------------------------------------------------------
    def preview_installation(self, preset_id: str) -> MarketplaceInstallResult:
        """Waliduje preset bez instalacji (na potrzeby UI/raportów)."""

        preset = self._catalog.find(preset_id)
        if preset is None:
            raise MarketplaceCatalogError(f"Preset {preset_id!r} nie istnieje w katalogu Marketplace.")

        payload = preset.artifact_path.read_bytes()
        document = parse_preset_document(payload, source=preset.artifact_path, signing_keys=self._signing_keys)
        evaluation = self._evaluate_document(document)
        return MarketplaceInstallResult(
            preset_id=document.preset_id,
            version=document.version,
            success=evaluation.success,
            installed_path=None,
            signature_verified=evaluation.signature_verified,
            fingerprint_verified=evaluation.fingerprint_verified,
            issues=evaluation.issues,
            license_payload=evaluation.license_payload,
        )

    # ------------------------------------------------------------------
    def _evaluate_document(self, document: PresetDocument) -> _EvaluationResult:
        issues: list[str] = []
        signature_verified = document.verification.verified
        if not signature_verified:
            issues.append("signature-unverified")

        license_payload = self._load_license_payload(document.preset_id)
        license_ok, fingerprint_verified, license_issues = self._validate_license(
            document.preset_id,
            document.version,
            license_payload,
        )
        issues.extend(license_issues)

        success = signature_verified and license_ok and not license_issues
        return _EvaluationResult(
            success=success,
            signature_verified=signature_verified,
            fingerprint_verified=fingerprint_verified,
            issues=tuple(dict.fromkeys(issues)),
            license_payload=license_payload,
        )

    # ------------------------------------------------------------------
    def _load_license_payload(self, preset_id: str) -> Mapping[str, Any] | None:
        if self._licenses_dir is None:
            return None
        candidate = self._licenses_dir / f"{preset_id}.json"
        if not candidate.exists():
            return None
        try:
            raw = candidate.read_text(encoding="utf-8")
        except OSError as exc:
            LOGGER.error("Nie udało się odczytać pliku licencji %s: %s", candidate, exc)
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            LOGGER.error("Plik licencji %s zawiera niepoprawny JSON: %s", candidate, exc)
            return None
        if not isinstance(payload, Mapping):
            LOGGER.error("Plik licencji %s ma niepoprawny format (oczekiwano obiektu).", candidate)
            return None
        return payload

    # ------------------------------------------------------------------
    def _validate_license(
        self,
        preset_id: str,
        version: str | None,
        license_payload: Mapping[str, Any] | None,
    ) -> tuple[bool, bool | None, list[str]]:
        if license_payload is None:
            return False, None, ["license-missing"]

        issues: list[str] = []

        license_preset = str(license_payload.get("preset_id") or license_payload.get("id") or "").strip()
        if license_preset and license_preset != preset_id:
            issues.append("license-preset-mismatch")

        allowed_versions = license_payload.get("allowed_versions")
        if allowed_versions:
            if isinstance(allowed_versions, Sequence) and not isinstance(allowed_versions, (str, bytes)):
                normalized_versions = {str(item).strip() for item in allowed_versions}
                if version not in normalized_versions:
                    issues.append("license-version-mismatch")
            else:
                issues.append("license-version-invalid")

        expires_at = license_payload.get("expires_at")
        if isinstance(expires_at, str) and expires_at.strip():
            try:
                expiry = datetime.fromisoformat(expires_at.strip().replace("Z", "+00:00"))
            except ValueError:
                issues.append("license-expiry-invalid")
            else:
                if expiry < datetime.now(timezone.utc):
                    issues.append("license-expired")

        fingerprint_verified: bool | None = None
        fingerprints = self._normalize_fingerprints(license_payload.get("allowed_fingerprints"))
        if fingerprints:
            try:
                hwid = self._hwid_provider.read()
            except HwIdProviderError:
                issues.append("fingerprint-unavailable")
                fingerprint_verified = False
            else:
                fingerprint_verified = any(self._match_fingerprint(hwid, candidate) for candidate in fingerprints)
                if not fingerprint_verified:
                    issues.append("fingerprint-mismatch")

        return (len(issues) == 0), fingerprint_verified, issues

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_fingerprints(candidates: Any) -> tuple[str, ...]:
        if candidates in (None, ""):
            return tuple()
        if isinstance(candidates, str):
            items = [candidates]
        elif isinstance(candidates, Sequence):
            items = [str(item) for item in candidates]
        else:
            return tuple()
        cleaned = []
        for item in items:
            text = str(item).strip()
            if text:
                cleaned.append(text)
        return tuple(dict.fromkeys(cleaned))

    # ------------------------------------------------------------------
    @staticmethod
    def _match_fingerprint(candidate: str, pattern: str) -> bool:
        normalized_candidate = candidate.strip().upper()
        normalized_pattern = pattern.strip().upper()
        if normalized_pattern.endswith("*"):
            prefix = normalized_pattern[:-1]
            return normalized_candidate.startswith(prefix)
        return normalized_candidate == normalized_pattern


__all__ = [
    "MarketplaceInstallResult",
    "MarketplacePresetInstaller",
]
