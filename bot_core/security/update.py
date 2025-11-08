"""Verification and application helpers for signed update packages."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from bot_core.security.hwid import HwIdProvider
from bot_core.security.license import LicenseValidationResult
from bot_core.security.guards import CapabilityGuard, LicenseCapabilityError
from bot_core.security.signing import verify_hmac_signature
from bot_core.update.differential import DeltaManifestValidation, DifferentialUpdateManager

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class UpdateArtifact:
    """Single file declared in the update manifest."""

    path: str
    sha384: str | None
    size: int
    kind: str = "full"
    sha256: str | None = None
    base_id: str | None = None
    has_type: bool = False
    has_sha384: bool = False
    has_sha256: bool = False


@dataclass(slots=True)
class UpdateManifest:
    """Structured representation of ``manifest.json`` produced by bundlers."""

    version: str
    platform: str
    runtime: str
    artifacts: list[UpdateArtifact]
    generator: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    allowed_profiles: Sequence[str] | None = None
    raw_payload: Mapping[str, object] | None = None
    signature: Mapping[str, object] | None = None
    integrity_manifest: Mapping[str, object] | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "UpdateManifest":
        try:
            artifacts_raw = payload.get("artifacts")
            if not isinstance(artifacts_raw, Sequence):
                raise ValueError("Brak listy artefaktów w manifeście aktualizacji")
            artifacts: list[UpdateArtifact] = []
            for entry in artifacts_raw:
                if not isinstance(entry, Mapping):
                    raise ValueError("Niepoprawny wpis artefaktu w manifeście")
                path = entry.get("path")
                size = entry.get("size")
                sha384 = entry.get("sha384")
                sha256 = entry.get("sha256")
                if not isinstance(path, str) or not isinstance(size, int):
                    raise ValueError("Artefakt musi zawierać pola 'path' oraz 'size'")
                if sha384 is None and sha256 is None:
                    raise ValueError("Artefakt musi posiadać hash 'sha384' lub 'sha256'")
                if sha384 is not None and not isinstance(sha384, str):
                    raise ValueError("Pole 'sha384' musi być napisem")
                if sha256 is not None and not isinstance(sha256, str):
                    raise ValueError("Pole 'sha256' musi być napisem")
                type_field = entry.get("type")
                include_type = type_field is not None and str(type_field).strip() != ""
                kind = str(type_field).lower() if include_type else "full"
                base_id = entry.get("base_id") or entry.get("baseId")
                if base_id is not None:
                    base_id = str(base_id)
                artifacts.append(
                    UpdateArtifact(
                        path=path,
                        sha384=str(sha384) if sha384 is not None else None,
                        sha256=str(sha256) if sha256 is not None else None,
                        size=size,
                        kind=kind,
                        base_id=base_id,
                        has_type=include_type,
                        has_sha384="sha384" in entry,
                        has_sha256="sha256" in entry,
                    )
                )
        except Exception as exc:
            raise ValueError(f"Nie udało się zinterpretować artefaktów: {exc}") from exc

        metadata_raw = payload.get("metadata")
        metadata: Mapping[str, object]
        if metadata_raw is None:
            metadata = {}
        else:
            if not isinstance(metadata_raw, Mapping):
                raise ValueError("Pole 'metadata' w manifeście aktualizacji musi być mapą")
            metadata = metadata_raw

        allowed_profiles = payload.get("allowed_profiles")
        if allowed_profiles is not None and not isinstance(allowed_profiles, Sequence):
            raise ValueError("Pole 'allowed_profiles' musi być sekwencją profili")

        version = payload.get("version")
        if not isinstance(version, str):
            raise ValueError("Manifest aktualizacji musi zawierać pole 'version' typu string")

        platform = payload.get("platform")
        if not isinstance(platform, str):
            raise ValueError("Manifest aktualizacji musi zawierać pole 'platform' typu string")

        runtime = payload.get("runtime")
        if not isinstance(runtime, str):
            raise ValueError("Manifest aktualizacji musi zawierać pole 'runtime' typu string")

        raw_payload = dict(payload)
        raw_payload.pop("signature", None)
        raw_payload["artifacts"] = []
        for art in artifacts:
            entry_payload = {"path": art.path, "size": art.size}
            if art.has_sha384:
                entry_payload["sha384"] = art.sha384
            if art.has_sha256:
                entry_payload["sha256"] = art.sha256
            if art.has_type:
                entry_payload["type"] = art.kind
            if art.base_id:
                entry_payload["base_id"] = art.base_id
            raw_payload["artifacts"].append(entry_payload)
        if metadata_raw is not None:
            raw_payload["metadata"] = dict(metadata)
        if allowed_profiles is not None:
            raw_payload["allowed_profiles"] = list(allowed_profiles)

        signature = payload.get("signature") if isinstance(payload.get("signature"), Mapping) else None
        integrity_manifest = (
            payload.get("integrity_manifest")
            if isinstance(payload.get("integrity_manifest"), Mapping)
            else None
        )

        return cls(
            version=version,
            platform=platform,
            runtime=runtime,
            artifacts=artifacts,
            generator=str(payload.get("generator")) if payload.get("generator") else None,
            metadata=dict(metadata),
            allowed_profiles=list(allowed_profiles) if allowed_profiles is not None else None,
            raw_payload=raw_payload,
            signature=dict(signature) if signature else None,
            integrity_manifest=dict(integrity_manifest) if integrity_manifest else None,
        )


@dataclass(slots=True)
class UpdateVerificationResult:
    """Outcome of update validation."""

    manifest: UpdateManifest
    signature_valid: bool
    signature_checked: bool
    license_ok: bool
    artifact_checks: list[str]
    errors: list[str]
    warnings: list[str] = field(default_factory=list)

    @property
    def is_successful(self) -> bool:
        return (self.signature_valid or not self.signature_checked) and self.license_ok and not self.errors


class UpdateVerificationError(RuntimeError):
    """Raised when update verification cannot be completed."""


@dataclass(slots=True)
class DifferentialUpdateReport:
    """Combined status of differential manifest and update verification."""

    delta_manifest: DeltaManifestValidation
    update: UpdateVerificationResult

    @property
    def is_successful(self) -> bool:
        return (
            self.delta_manifest.signature_valid is not False
            and self.delta_manifest.fingerprint_ok
            and not self.delta_manifest.issues
            and self.update.is_successful
        )


def _hash_file_sha384(path: Path) -> str:
    digest = hashlib.sha384()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> UpdateManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise UpdateVerificationError(f"Manifest {path} zawiera niepoprawny JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise UpdateVerificationError("Manifest aktualizacji musi być obiektem JSON")
    return UpdateManifest.from_mapping(payload)


def _verify_artifacts(manifest: UpdateManifest, base_dir: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    audit: list[str] = []
    for artifact in manifest.artifacts:
        candidate = base_dir / artifact.path
        if not candidate.exists():
            errors.append(f"Brak artefaktu {artifact.path} w katalogu aktualizacji")
            continue
        if artifact.sha384 is not None:
            actual_hash = _hash_file_sha384(candidate)
            if actual_hash != artifact.sha384:
                errors.append(f"Artefakt {artifact.path} posiada niepoprawny hash SHA-384")
                continue
        if artifact.sha256 is not None:
            actual_hash256 = _hash_file_sha256(candidate)
            if actual_hash256 != artifact.sha256:
                errors.append(f"Artefakt {artifact.path} posiada niepoprawny hash SHA-256")
                continue
        audit.append(f"{artifact.path}: OK")
    return audit, errors


def verify_update_bundle(
    *,
    manifest_path: Path,
    base_dir: Path,
    signature_path: Path | None = None,
    hmac_key: bytes | None = None,
    license_result: LicenseValidationResult | None = None,
) -> UpdateVerificationResult:
    """Validate signed update bundle and optionally enforce license policy."""

    manifest = _load_manifest(manifest_path)

    signature_valid = True
    signature_checked = False
    signature_payload: Mapping[str, object] | None = None
    signature_required = hmac_key is not None
    if signature_path is not None:
        signature_payload = json.loads(signature_path.read_text(encoding="utf-8"))
        if not isinstance(signature_payload, Mapping):
            raise UpdateVerificationError("Plik podpisu manifestu musi być obiektem JSON")
    elif manifest.signature is not None:
        signature_payload = manifest.signature

    errors: list[str] = []
    warnings: list[str] = []

    if signature_required:
        if signature_payload is None:
            signature_valid = False
            signature_checked = True
            errors.append(
                "Manifest aktualizacji nie zawiera podpisu, mimo że oczekiwano go w konfiguracji."
            )
        else:
            if manifest.raw_payload is None:
                raise UpdateVerificationError(
                    "Manifest aktualizacji nie zawiera surowych danych do weryfikacji"
                )
            signature_checked = True
            signature_valid = verify_hmac_signature(
                payload=manifest.raw_payload,
                signature=signature_payload,
                key=hmac_key,
            )
            if not signature_valid:
                LOGGER.warning("Podpis manifestu aktualizacji jest niepoprawny")
                errors.append("Podpis manifestu aktualizacji jest niepoprawny.")
    else:
        if signature_payload is not None:
            warnings.append(
                "Podpis manifestu aktualizacji nie został zweryfikowany (brak klucza HMAC)."
            )

    audit, artifact_errors = _verify_artifacts(manifest, base_dir)
    errors.extend(artifact_errors)

    license_ok = True
    if license_result is not None:
        if manifest.allowed_profiles:
            allowed = {profile.lower() for profile in manifest.allowed_profiles}
            profile = (license_result.profile or "").lower()
            if profile not in allowed:
                license_ok = False
                errors.append(
                    "Licencja OEM nie jest uprawniona do aktualizacji (profil %s, dozwolone: %s)"
                    % (license_result.profile or "<unknown>", ", ".join(sorted(manifest.allowed_profiles)))
                )
            elif not license_result.is_valid:
                license_ok = False
                errors.append("Licencja OEM nie przeszła walidacji i nie może otrzymać aktualizacji")
            else:
                warnings.append(
                    f"Licencja {license_result.license_path} potwierdzona dla profilu {license_result.profile}"
                )

        capabilities = getattr(license_result, "capabilities", None)
        guard: CapabilityGuard | None = getattr(license_result, "capability_guard", None)
        if capabilities is None:
            license_ok = False
            errors.append("Licencja offline nie dostarczyła capabilities – aktualizacja jest zablokowana.")
        else:
            if guard is None:
                guard = CapabilityGuard(capabilities)

            try:
                guard.require_module(
                    "oem_updater",
                    message="Moduł OEM Updater jest wymagany do zastosowania aktualizacji.",
                )
            except LicenseCapabilityError as exc:
                license_ok = False
                errors.append(str(exc))

            if not capabilities.is_maintenance_active():
                license_ok = False
                errors.append("Licencja utrzymaniowa wygasła – aktualizacja została przerwana.")

            metadata = manifest.metadata or {}
            required_modules = metadata.get("required_modules")
            if required_modules:
                if isinstance(required_modules, Sequence) and not isinstance(required_modules, (str, bytes, bytearray)):
                    missing = [
                        str(module)
                        for module in required_modules
                        if not capabilities.is_module_enabled(str(module))
                    ]
                    if missing:
                        license_ok = False
                        errors.append(
                            "Licencja nie zawiera modułów wymaganych przez aktualizację: %s"
                            % ", ".join(sorted(missing))
                        )
                else:
                    license_ok = False
                    errors.append("Pole metadata.required_modules musi być listą nazw modułów.")

            min_edition = metadata.get("min_edition")
            if min_edition:
                if isinstance(min_edition, str):
                    try:
                        guard.require_edition(
                            min_edition,
                            message=(
                                "Edycja licencji jest zbyt niska dla tej aktualizacji (wymagana: %s)."
                                % min_edition
                            ),
                        )
                    except LicenseCapabilityError as exc:
                        license_ok = False
                        errors.append(str(exc))
                else:
                    license_ok = False
                    errors.append("Pole metadata.min_edition musi być ciągiem znaków.")
    else:
        warnings.append("Nie dostarczono wyniku walidacji licencji – pomijam kontrolę profilu")

    return UpdateVerificationResult(
        manifest=manifest,
        signature_valid=signature_valid,
        signature_checked=signature_checked,
        license_ok=license_ok,
        artifact_checks=audit,
        errors=errors,
        warnings=warnings,
    )


def verify_differential_update(
    *,
    manifest_path: Path,
    package_dir: Path,
    delta_signature_path: Path | None = None,
    manifest_key: bytes | None = None,
    package_key: bytes | None = None,
    license_result: LicenseValidationResult | None = None,
    hwid_provider: HwIdProvider | None = None,
) -> DifferentialUpdateReport:
    """Validate delta manifest and package integrity in a single step."""

    manager = DifferentialUpdateManager(
        storage_dir=package_dir.parent,
        manifest_key=manifest_key,
        package_key=package_key,
        hwid_provider=hwid_provider,
    )
    manifest_validation = manager.validate_manifest(manifest_path, signature_path=delta_signature_path)
    update_result = manager.verify_package(
        package_dir,
        license_result=license_result,
    )
    return DifferentialUpdateReport(delta_manifest=manifest_validation, update=update_result)


__all__ = [
    "UpdateArtifact",
    "UpdateManifest",
    "UpdateVerificationResult",
    "UpdateVerificationError",
    "verify_update_bundle",
    "DifferentialUpdateReport",
    "verify_differential_update",
]
