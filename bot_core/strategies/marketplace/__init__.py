"""Manifesty presetów strategii oraz helpery Marketplace."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import yaml

CATALOG_FILENAME = "catalog.yaml"
PRESETS_DIRNAME = "presets"


@dataclass(slots=True)
class MarketplaceAuthor:
    """Metadane autora presetu Marketplace."""

    name: str
    email: str | None = None
    organization: str | None = None


@dataclass(slots=True)
class MarketplacePreset:
    """Opis pojedynczego presetu dostępnego w Marketplace."""

    preset_id: str
    name: str
    version: str
    author: MarketplaceAuthor
    required_exchanges: tuple[str, ...]
    tags: tuple[str, ...]
    summary: str | None
    license_tier: str | None
    artifact_path: Path
    signature: Mapping[str, Any] | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "presetId": self.preset_id,
            "name": self.name,
            "version": self.version,
            "summary": self.summary,
            "requiredExchanges": list(self.required_exchanges),
            "tags": list(self.tags),
            "licenseTier": self.license_tier,
        }
        if self.signature:
            payload["signature"] = dict(self.signature)
        if self.extras:
            payload["metadata"] = dict(self.extras)
        return payload


@dataclass(slots=True)
class MarketplaceCatalog:
    """Struktura katalogu presetów Marketplace."""

    schema_version: str
    generated_at: datetime
    presets: tuple[MarketplacePreset, ...]
    base_path: Path

    def find(self, preset_id: str) -> MarketplacePreset | None:
        preset_id = preset_id.strip()
        for preset in self.presets:
            if preset.preset_id == preset_id:
                return preset
        return None

    def __iter__(self):  # type: ignore[override]
        return iter(self.presets)


class MarketplaceCatalogError(RuntimeError):
    """Błąd podczas ładowania katalogu Marketplace."""


def _normalize_sequence(values: Iterable[str] | None) -> tuple[str, ...]:
    if not values:
        return tuple()
    cleaned = []
    for item in values:
        text = str(item).strip()
        if text:
            cleaned.append(text)
    return tuple(dict.fromkeys(cleaned))


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MarketplaceCatalogError(f"Nie udało się odczytać pliku manifestu {path}: {exc}") from exc
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:  # pragma: no cover - biblioteka raportuje błąd
        raise MarketplaceCatalogError(f"Manifest Marketplace ma niepoprawny format YAML ({path}).") from exc
    if not isinstance(data, Mapping):
        raise MarketplaceCatalogError("Manifest Marketplace musi być słownikiem.")
    return data


def _parse_author(payload: Mapping[str, Any]) -> MarketplaceAuthor:
    name = str(payload.get("name") or "").strip()
    if not name:
        raise MarketplaceCatalogError("Pole author.name w manifeście presetu jest wymagane.")
    email = payload.get("email")
    organization = payload.get("organization")
    return MarketplaceAuthor(
        name=name,
        email=str(email).strip() if isinstance(email, str) and email.strip() else None,
        organization=(
            str(organization).strip()
            if isinstance(organization, str) and organization.strip()
            else None
        ),
    )


def _parse_preset(base_path: Path, payload: Mapping[str, Any]) -> MarketplacePreset:
    preset_id = str(payload.get("id") or "").strip()
    if not preset_id:
        raise MarketplaceCatalogError("Preset w katalogu Marketplace musi posiadać pole 'id'.")
    name = str(payload.get("name") or "").strip() or preset_id
    version = str(payload.get("version") or "").strip() or "0"
    summary = payload.get("summary")
    summary_text = str(summary).strip() if isinstance(summary, str) else None
    license_tier = payload.get("license_tier")
    license_tier_text = (
        str(license_tier).strip() if isinstance(license_tier, str) and license_tier.strip() else None
    )

    author_payload = payload.get("author")
    if not isinstance(author_payload, Mapping):
        raise MarketplaceCatalogError(
            f"Preset {preset_id} musi zawierać sekcję author z informacjami o wydawcy."
        )
    author = _parse_author(author_payload)

    required_exchanges = _normalize_sequence(payload.get("required_exchanges"))
    tags = _normalize_sequence(payload.get("tags"))

    artifact = payload.get("artifact")
    if not isinstance(artifact, str) or not artifact.strip():
        raise MarketplaceCatalogError(
            f"Preset {preset_id} musi wskazywać ścieżkę artefaktu w polu 'artifact'."
        )
    artifact_path = (base_path / artifact).resolve()

    signature = payload.get("signature") if isinstance(payload.get("signature"), Mapping) else None

    extras: MutableMapping[str, Any] = {}
    for key, value in payload.items():
        if key in {
            "id",
            "name",
            "version",
            "author",
            "summary",
            "required_exchanges",
            "tags",
            "license_tier",
            "artifact",
            "signature",
        }:
            continue
        extras[str(key)] = value

    return MarketplacePreset(
        preset_id=preset_id,
        name=name,
        version=version,
        author=author,
        required_exchanges=required_exchanges,
        tags=tags,
        summary=summary_text,
        license_tier=license_tier_text,
        artifact_path=artifact_path,
        signature=signature,
        extras=dict(extras),
    )


def load_catalog(base_path: str | Path | None = None) -> MarketplaceCatalog:
    """Ładuje manifest Marketplace i zwraca obiekt katalogu."""

    if base_path is None:
        base_path = Path(__file__).resolve().parent
    else:
        base_path = Path(base_path).expanduser().resolve()

    manifest_path = base_path / CATALOG_FILENAME
    data = _load_yaml(manifest_path)

    schema_version = str(data.get("schema_version") or "1.0").strip()
    generated_raw = data.get("generated_at")
    if isinstance(generated_raw, datetime):
        generated_at = generated_raw
    elif isinstance(generated_raw, str) and generated_raw.strip():
        text = generated_raw.strip().replace("Z", "+00:00")
        try:
            generated_at = datetime.fromisoformat(text)
        except ValueError as exc:
            raise MarketplaceCatalogError("Pole generated_at w katalogu Marketplace ma niepoprawny format.") from exc
    else:
        generated_at = datetime.utcnow()

    presets_payload = data.get("presets")
    if not isinstance(presets_payload, Sequence):
        raise MarketplaceCatalogError("Manifest Marketplace musi zawierać listę presetów w polu 'presets'.")

    presets: list[MarketplacePreset] = []
    for item in presets_payload:
        if not isinstance(item, Mapping):
            raise MarketplaceCatalogError("Każdy preset w katalogu musi być obiektem mapującym.")
        preset = _parse_preset(base_path, item)
        presets.append(preset)

    return MarketplaceCatalog(
        schema_version=schema_version,
        generated_at=generated_at,
        presets=tuple(presets),
        base_path=base_path,
    )


__all__ = [
    "MarketplaceAuthor",
    "MarketplaceCatalog",
    "MarketplaceCatalogError",
    "MarketplacePreset",
    "load_catalog",
    "CATALOG_FILENAME",
    "PRESETS_DIRNAME",
]
