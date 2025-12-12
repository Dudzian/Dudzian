"""Workflow publikacji presetów Marketplace."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from bot_core.config_marketplace.schema import (
    MarketplaceCatalog,
    MarketplacePackageMetadata,
    load_catalog,
)
from bot_core.marketplace import verify_preset_signature


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(slots=True)
class PresetReviewSummary:
    """Reprezentuje zweryfikowaną recenzję presetu."""

    preset_id: str
    rating: int
    author: str
    comment: str
    signature_valid: bool
    review_id: str | None = None
    submitted_at: str | None = None
    warnings: Sequence[str] = ()

    def to_mapping(self) -> dict[str, Any]:
        return {
            "presetId": self.preset_id,
            "rating": self.rating,
            "author": self.author,
            "comment": self.comment,
            "signatureValid": self.signature_valid,
            "reviewId": self.review_id,
            "submittedAt": self.submitted_at,
            "warnings": list(self.warnings),
        }


class PresetPublicationWorkflow:
    """Waliduje presetowe artefakty i przygotowuje payload dla UI."""

    def __init__(
        self,
        catalog: MarketplaceCatalog,
        reviews: Mapping[str, Sequence[Mapping[str, Any]]],
        signing_keys: Mapping[str, bytes] | None = None,
    ) -> None:
        self._catalog = catalog
        self._reviews = reviews
        self._signing_keys = dict(signing_keys or {})
        self._ready_packages = [
            pkg
            for pkg in catalog.packages
            if pkg.release.review_status.lower() == "approved"
        ]

    @property
    def ready_packages(self) -> Sequence[MarketplacePackageMetadata]:
        return tuple(self._ready_packages)

    @classmethod
    def from_paths(
        cls,
        *,
        catalog_path: Path,
        reviews_dir: Path | None = None,
        signing_keys: Mapping[str, bytes] | None = None,
    ) -> "PresetPublicationWorkflow":
        catalog = load_catalog(catalog_path)
        reviews = cls._load_review_documents(reviews_dir)
        return cls(catalog, reviews, signing_keys)

    @staticmethod
    def _load_review_documents(
        reviews_dir: Path | None,
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        documents: dict[str, list[Mapping[str, Any]]] = {}
        if reviews_dir is None or not reviews_dir.exists():
            return documents
        for path in sorted(reviews_dir.glob("*.json")):
            payload = _load_json(path)
            preset_id = str(payload.get("preset_id") or path.stem)
            entries = payload.get("reviews")
            if isinstance(entries, Sequence):
                normalized = [entry for entry in entries if isinstance(entry, Mapping)]
                documents.setdefault(preset_id, []).extend(normalized)
        return documents

    def _verify_signature(
        self,
        payload: Mapping[str, Any],
        signature: Mapping[str, Any] | None,
    ) -> bool:
        verification, _ = verify_preset_signature(
            payload,
            signature,
            signing_keys=self._signing_keys,
        )
        return verification.verified

    def _collect_reviews(self, preset: MarketplacePackageMetadata) -> list[PresetReviewSummary]:
        entries = self._reviews.get(preset.package_id) or []
        summaries: list[PresetReviewSummary] = []
        for entry in entries:
            payload = {k: v for k, v in entry.items() if k != "signature"}
            signature_valid = self._verify_signature(payload, entry.get("signature"))
            summaries.append(
                PresetReviewSummary(
                    preset_id=preset.package_id,
                    rating=int(entry.get("rating", 0)),
                    author=str(entry.get("author") or "unknown"),
                    comment=str(entry.get("comment") or ""),
                    signature_valid=signature_valid,
                    review_id=str(entry.get("review_id") or entry.get("reviewId") or ""),
                    submitted_at=str(entry.get("submitted_at") or entry.get("submittedAt") or ""),
                    warnings=tuple(entry.get("warnings") or ()),
                )
            )
        return summaries

    def _artifact_payloads(
        self, preset: MarketplacePackageMetadata
    ) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for artifact in preset.distribution:
            normalized = {
                "name": artifact.name,
                "uri": str(artifact.uri),
                "kind": artifact.kind,
                "size_bytes": artifact.size_bytes,
            }
            if artifact.integrity:
                normalized["integrity"] = artifact.integrity.model_dump()
            if artifact.signature:
                signed_payload = preset.signed_payload(artifact)
                normalized["signature"] = {
                    **artifact.signature.model_dump(),
                    "valid": self._verify_signature(
                        signed_payload,
                        artifact.signature.model_dump(),
                    ),
                }
            payloads.append(normalized)
        return payloads

    def validate(self, minimum_ready: int = 15) -> None:
        if len(self._ready_packages) < minimum_ready:
            raise ValueError(
                f"Brakuje gotowych presetów: wymagane {minimum_ready}, dostępne {len(self._ready_packages)}"
            )
        for preset in self._ready_packages:
            for artifact in preset.distribution:
                if artifact.signature is None:
                    raise ValueError(f"Preset {preset.package_id} zawiera artefakt bez podpisu")

    def build_ui_payload(self) -> dict[str, Any]:
        presets_payload: list[dict[str, Any]] = []
        for preset in self._ready_packages:
            presets_payload.append(
                {
                    "packageId": preset.package_id,
                    "displayName": preset.display_name,
                    "version": preset.version,
                    "releaseDate": preset.release_date.isoformat() if preset.release_date else None,
                    "artifacts": self._artifact_payloads(preset),
                    "reviews": [review.to_mapping() for review in self._collect_reviews(preset)],
                    "exchangeCompatibility": [entry.model_dump() for entry in preset.exchange_compatibility],
                    "wizard": {
                        "importable": True,
                        "source": preset.versioning.source,
                        "channels": preset.versioning.channel,
                        "reviewStatus": preset.release.review_status,
                    },
                }
            )
        return {
            "generatedAt": self._catalog.generated_at.isoformat(),
            "total": len(presets_payload),
            "presets": presets_payload,
        }

    def to_report_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for preset in self._ready_packages:
            rows.append(
                {
                    "package": preset.package_id,
                    "name": preset.display_name,
                    "status": preset.release.review_status,
                    "approved_at": preset.release.approved_at.isoformat() if preset.release.approved_at else "",
                    "reviewers": [review.name for review in preset.release.reviewers],
                    "signed_artifacts": all(artifact.signature for artifact in preset.distribution),
                    "review_count": len(self._collect_reviews(preset)),
                }
            )
        return rows

