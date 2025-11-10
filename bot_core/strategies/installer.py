"""Instalator presetów Marketplace wraz z walidacją licencji."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from bot_core.marketplace import PresetDocument, PresetRepository, parse_preset_document
from bot_core.security.hwid import HwIdProvider, HwIdProviderError
from bot_core.security.license import (
    _parse_seat_policy,
    _parse_subscription_section,
)
from bot_core.security.messages import ValidationMessage

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
    warnings: tuple[str, ...] = field(default_factory=tuple)
    license_payload: Mapping[str, Any] | None = None


@dataclass(slots=True)
class _EvaluationResult:
    success: bool
    signature_verified: bool
    fingerprint_verified: bool | None
    issues: tuple[str, ...]
    warnings: tuple[str, ...]
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
            warnings=evaluation.warnings,
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
            warnings=evaluation.warnings,
            license_payload=evaluation.license_payload,
        )

    # ------------------------------------------------------------------
    def load_catalog_document(self, preset_id: str) -> PresetDocument:
        """Zwraca dokument presetu z katalogu (do prezentacji w UI)."""

        preset = self._catalog.find(preset_id)
        if preset is None:
            raise MarketplaceCatalogError(
                f"Preset {preset_id!r} nie istnieje w katalogu Marketplace."
            )
        payload = preset.artifact_path.read_bytes()
        return parse_preset_document(payload, source=preset.artifact_path, signing_keys=self._signing_keys)

    # ------------------------------------------------------------------
    def _evaluate_document(self, document: PresetDocument) -> _EvaluationResult:
        issues: list[str] = []
        signature_verified = document.verification.verified
        if not signature_verified:
            issues.append("signature-unverified")

        license_payload = self._load_license_payload(document.preset_id)
        (
            license_ok,
            fingerprint_verified,
            license_errors,
            license_warnings,
            normalized_license,
        ) = self._validate_license(
            document.preset_id,
            document.version,
            license_payload,
        )
        issues.extend(license_errors)

        success = signature_verified and license_ok and not license_errors
        return _EvaluationResult(
            success=success,
            signature_verified=signature_verified,
            fingerprint_verified=fingerprint_verified,
            issues=tuple(dict.fromkeys(issues)),
            warnings=tuple(dict.fromkeys(license_warnings)),
            license_payload=normalized_license,
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
    ) -> tuple[bool, bool | None, list[str], list[str], Mapping[str, Any] | None]:
        if license_payload is None:
            return False, None, ["license-missing"], [], None

        issues: list[str] = []
        warnings: list[str] = []
        normalized: Mapping[str, Any] | None = dict(license_payload)

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
        expiry: datetime | None = None
        if isinstance(expires_at, str) and expires_at.strip():
            try:
                expiry = datetime.fromisoformat(expires_at.strip().replace("Z", "+00:00"))
            except ValueError:
                issues.append("license-expiry-invalid")
            else:
                now = datetime.now(timezone.utc)
                if expiry < now:
                    issues.append("license-expired")
                elif expiry - now <= timedelta(days=30):
                    warnings.append("license-expiring-soon")

        fingerprint_verified: bool | None = None
        device_fingerprint: str | None = None
        fingerprints = self._normalize_fingerprints(license_payload.get("allowed_fingerprints"))
        if fingerprints:
            try:
                device_fingerprint = self._hwid_provider.read()
            except HwIdProviderError:
                issues.append("fingerprint-unavailable")
                fingerprint_verified = False
            else:
                fingerprint_verified = any(
                    self._match_fingerprint(device_fingerprint, candidate) for candidate in fingerprints
                )
                if not fingerprint_verified:
                    issues.append("fingerprint-mismatch")

        seat_errors: list[ValidationMessage] = []
        seat_warnings: list[ValidationMessage] = []
        (
            seats_total,
            seats_in_use,
            seats_available,
            seat_assignments,
            seat_pending,
            seat_enforcement,
            seat_auto_assign,
        ) = _parse_seat_policy(
            license_payload,
            fingerprint_value=device_fingerprint,
            errors=seat_errors,
            warnings=seat_warnings,
        )

        subscription_errors: list[ValidationMessage] = []
        subscription_warnings: list[ValidationMessage] = []
        (
            subscription_status,
            subscription_renews_at,
            subscription_period_start,
            subscription_period_end,
            subscription_grace_expires_at,
        ) = _parse_subscription_section(
            license_payload,
            current_time=datetime.now(timezone.utc),
            errors=subscription_errors,
            warnings=subscription_warnings,
        )

        validation_errors = [*seat_errors, *subscription_errors]
        validation_warnings = [*seat_warnings, *subscription_warnings]

        if validation_errors:
            issues.extend(message.code for message in validation_errors)
        if validation_warnings:
            warnings.extend(message.code for message in validation_warnings)

        if normalized is not None:
            payload_copy: dict[str, Any] = dict(normalized)
            payload_copy["seat_summary"] = {
                "total": seats_total,
                "in_use": seats_in_use,
                "available": seats_available,
                "assignments": seat_assignments,
                "pending": seat_pending,
                "enforcement": seat_enforcement,
                "auto_assign": seat_auto_assign,
            }
            payload_copy["subscription_summary"] = {
                "status": subscription_status,
                "renews_at": subscription_renews_at,
                "period_start": subscription_period_start,
                "period_end": subscription_period_end,
                "grace_expires_at": subscription_grace_expires_at,
            }

            validation_payload: dict[str, Any] = {}
            if validation_errors:
                validation_payload["errors"] = [entry.to_dict() for entry in validation_errors]
                validation_payload["error_messages"] = [entry.message for entry in validation_errors]
                validation_payload["error_codes"] = [entry.code for entry in validation_errors]
            if validation_warnings:
                validation_payload["warnings"] = [entry.to_dict() for entry in validation_warnings]
                validation_payload["warning_messages"] = [entry.message for entry in validation_warnings]
                validation_payload["warning_codes"] = [entry.code for entry in validation_warnings]
            if validation_payload:
                existing_validation = payload_copy.get("validation")
                if isinstance(existing_validation, Mapping):
                    merged_validation = dict(existing_validation)
                    merged_validation.update(validation_payload)
                else:
                    merged_validation = validation_payload
                payload_copy["validation"] = merged_validation
            normalized = payload_copy

        success = len(issues) == 0
        return success, fingerprint_verified, issues, warnings, normalized

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
