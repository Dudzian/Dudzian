"""Walidator podpisów i fingerprintu paczek Marketplace."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

from bot_core.config_marketplace.schema import (
    DistributionArtifact,
    MarketplaceCatalog,
    MarketplacePackageMetadata,
)
from bot_core.security.hwid import HwIdProvider, HwIdProviderError
from bot_core.security.signing import verify_hmac_signature


@dataclass(slots=True)
class VerificationResult:
    """Raport z walidacji pojedynczego artefaktu."""

    package_id: str
    artifact: str
    verified: bool
    errors: list[str]
    warnings: list[str]

    def raise_for_status(self) -> None:
        if self.errors:
            raise MarketplaceVerificationError("\n".join(self.errors))


class MarketplaceVerificationError(RuntimeError):
    """Wyjątek rzucany przy krytycznych błędach walidacji Marketplace."""


class MarketplaceValidator:
    """Weryfikuje integralność paczek Marketplace."""

    def __init__(
        self,
        *,
        signing_keys: Mapping[str, bytes] | None = None,
        hwid_provider: HwIdProvider | None = None,
    ) -> None:
        self._signing_keys = dict(signing_keys or {})
        self._hwid_provider = hwid_provider or HwIdProvider()

    # ------------------------------------------------------------------
    # Weryfikacja katalogu / paczki
    # ------------------------------------------------------------------

    def verify_catalog(
        self,
        catalog: MarketplaceCatalog,
        *,
        repository_root: Path | None = None,
    ) -> list[VerificationResult]:
        """Waliduje wszystkie paczki z katalogu."""

        results: list[VerificationResult] = []
        for metadata in catalog.packages:
            for artifact in metadata.distribution:
                result = self.verify_package(
                    metadata,
                    artifact=artifact,
                    repository_root=repository_root,
                )
                results.append(result)
        return results

    def verify_package(
        self,
        metadata: MarketplacePackageMetadata,
        *,
        artifact: DistributionArtifact,
        repository_root: Path | None = None,
    ) -> VerificationResult:
        """Sprawdza podpis, skrót oraz fingerprint paczki."""

        errors: list[str] = []
        warnings: list[str] = []

        artifact_path = self._resolve_artifact_path(artifact, repository_root)
        if artifact_path is None:
            warnings.append(
                f"Artefakt '{artifact.name}' pakietu '{metadata.package_id}' nie jest lokalnym plikiem – pomijam kontrolę skrótu i podpisu."
            )
        else:
            if artifact.integrity:
                digest_ok, message = self._verify_integrity(artifact_path, artifact.integrity.digest, artifact.integrity.algorithm)
                if not digest_ok:
                    errors.append(message)
            if artifact.signature:
                signature_errors = self._verify_signature(metadata, artifact)
                errors.extend(signature_errors)
            elif artifact.integrity:
                warnings.append(
                    f"Artefakt '{artifact.name}' ma zadeklarowany skrót, ale brak podpisu kryptograficznego."
                )

        fingerprint_errors = self._verify_fingerprint(metadata)
        errors.extend(fingerprint_errors)

        return VerificationResult(
            package_id=metadata.package_id,
            artifact=artifact.name,
            verified=not errors,
            errors=errors,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Helpery
    # ------------------------------------------------------------------

    def _resolve_artifact_path(
        self,
        artifact: DistributionArtifact,
        repository_root: Path | None,
    ) -> Path | None:
        uri_text = str(artifact.uri)
        parsed = urlparse(uri_text)
        if parsed.scheme in {"http", "https"}:
            return None
        if parsed.scheme == "file":
            path = Path(parsed.path)
        else:
            path = Path(uri_text)
            if not path.is_absolute() and repository_root is not None:
                path = repository_root / path
        return path

    def _verify_integrity(self, path: Path, expected_digest: str, algorithm: str) -> tuple[bool, str]:
        normalized = algorithm.strip().lower()
        try:
            hasher = hashlib.new(normalized)
        except ValueError:
            return False, f"Nieobsługiwany algorytm skrótu '{algorithm}' dla pliku {path}"
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
        except FileNotFoundError:
            return False, f"Plik artefaktu nie istnieje: {path}"
        digest = hasher.hexdigest()
        if digest != expected_digest.lower():
            return False, (
                "Skrót artefaktu nie zgadza się: oczekiwano {} otrzymano {}".format(
                    expected_digest,
                    digest,
                )
            )
        return True, ""

    def _verify_signature(
        self,
        metadata: MarketplacePackageMetadata,
        artifact: DistributionArtifact,
    ) -> list[str]:
        signature = artifact.signature
        if signature is None:
            return []
        key_bytes = self._signing_keys.get(signature.key_id)
        if not key_bytes:
            return [
                f"Brak klucza podpisu '{signature.key_id}' dla paczki '{metadata.package_id}'.",
            ]
        payload = metadata.signed_payload(artifact)
        signed_fields = list(signature.signed_fields or [])
        if signed_fields:
            filtered: dict[str, object] = {}
            for field in signed_fields:
                if field in payload:
                    filtered[field] = payload[field]
            if filtered:
                payload = filtered
        signature_doc = {
            "algorithm": signature.algorithm,
            "value": signature.value,
            "key_id": signature.key_id,
        }
        if not verify_hmac_signature(payload, signature_doc, key=key_bytes, algorithm=signature.algorithm):
            serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            return [
                "Niepoprawny podpis artefaktu '{}': payload={}, key_id={}".format(
                    artifact.name,
                    serialized,
                    signature.key_id,
                )
            ]
        return []

    def _verify_fingerprint(
        self,
        metadata: MarketplacePackageMetadata,
    ) -> list[str]:
        policy = metadata.security
        if not policy.allowed_fingerprints or policy.mode == "none":
            return []
        try:
            hwid = self._hwid_provider.read()
        except HwIdProviderError as exc:
            return [f"Nie udało się odczytać fingerprintu urządzenia: {exc}"]

        def _matches(candidate: str) -> bool:
            if policy.require_strict_match or policy.mode == "allowlist":
                if policy.require_strict_match:
                    return hwid == candidate
                return hwid.startswith(candidate) or hwid == candidate
            if policy.mode == "prefix":
                return hwid.startswith(candidate)
            return False

        if not any(_matches(candidate) for candidate in policy.allowed_fingerprints):
            message = policy.audit_message or "Fingerprint urządzenia nie znajduje się na liście dopuszczonych."
            return [f"[fingerprint] {message}"]
        return []


__all__ = [
    "MarketplaceValidator",
    "MarketplaceVerificationError",
    "VerificationResult",
]
