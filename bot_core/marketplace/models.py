from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from bot_core.security.signing import canonical_json_bytes


def canonical_preset_bytes(payload: Mapping[str, Any]) -> bytes:
    """Zwraca kanoniczną reprezentację JSON dla payloadu presetu."""

    return canonical_json_bytes(payload)


class PresetPayloadModel(BaseModel):
    """Model Pydantic dla sekcji payload presetu."""

    name: str | None = None
    metadata: MutableMapping[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    @field_validator("metadata", mode="before")
    @classmethod
    def ensure_metadata_mapping(cls, value: object) -> MutableMapping[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        return {}


class SignatureDocumentModel(BaseModel):
    """Model Pydantic opisujący sekcję podpisu dokumentu presetu."""

    algorithm: str = Field(default="ed25519")
    value: str = Field(alias="signature")
    key_id: str | None = None
    public_key: str | None = None
    signed_at: str | None = None
    issuer: str | None = None

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @field_validator("algorithm", mode="before")
    @classmethod
    def normalize_algorithm(cls, value: object) -> str:
        text = str(value).strip().lower() if isinstance(value, str) else ""
        return text or "ed25519"

    @field_validator("value", mode="before")
    @classmethod
    def normalize_value(cls, value: object) -> str:
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
        raise ValueError("signature value is required")


def normalize_preset_document(
    document: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
    """Normalizuje surowy dokument presetu do payloadu i sekcji podpisu."""

    if "preset" in document:
        payload_raw = document.get("preset")
        signature_doc = document.get("signature")
    else:
        payload_raw = document
        signature_doc = None

    try:
        payload_model = PresetPayloadModel.model_validate(payload_raw)
    except ValidationError as exc:  # pragma: no cover - błąd formatu wejściowego
        raise ValueError("Preset musi być obiektem JSON/YAML") from exc
    payload = payload_model.model_dump()
    normalized_signature = dict(signature_doc) if isinstance(signature_doc, Mapping) else None
    return payload, normalized_signature


@dataclass(slots=True)
class PresetSignatureVerification:
    """Informacje o wynikach weryfikacji podpisu presetu."""

    verified: bool
    issues: tuple[str, ...] = field(default_factory=tuple)
    algorithm: str | None = None
    key_id: str | None = None


@dataclass(slots=True)
class PresetSignature:
    """Metadane podpisu presetu Marketplace."""

    algorithm: str
    value: str
    key_id: str | None = None
    public_key: str | None = None
    signed_at: str | None = None
    issuer: str | None = None

    def as_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "algorithm": self.algorithm,
            "value": self.value,
        }
        if self.key_id:
            document["key_id"] = self.key_id
        if self.public_key:
            document["public_key"] = self.public_key
        if self.signed_at:
            document["signed_at"] = self.signed_at
        if self.issuer:
            document["issuer"] = self.issuer
        return document


@dataclass(slots=True)
class PresetDocument:
    """Reprezentuje plik presetu Marketplace wraz z podpisem."""

    payload: Mapping[str, Any]
    signature: PresetSignature | None
    verification: PresetSignatureVerification
    fmt: str
    path: Any | None = None
    issues: tuple[str, ...] = field(default_factory=tuple)

    @property
    def metadata(self) -> Mapping[str, Any]:
        raw = self.payload.get("metadata", {})
        if isinstance(raw, Mapping):
            return raw
        return {}

    @property
    def preset_id(self) -> str:
        meta = self.metadata
        value = meta.get("id") if isinstance(meta, Mapping) else None
        return str(value).strip() if value not in (None, "") else ""

    @property
    def version(self) -> str | None:
        meta = self.metadata
        value = meta.get("version") if isinstance(meta, Mapping) else None
        if value in (None, ""):
            return None
        return str(value).strip()

    @property
    def name(self) -> str | None:
        value = self.payload.get("name")
        if value in (None, ""):
            return None
        return str(value)

    @property
    def tags(self) -> tuple[str, ...]:
        tags = self.metadata.get("tags") if isinstance(self.metadata, Mapping) else None
        if isinstance(tags, Mapping):
            return tuple(str(tag) for tag in tags.values())
        if isinstance(tags, list):
            return tuple(str(tag) for tag in tags)
        if isinstance(tags, tuple):
            return tuple(str(tag) for tag in tags)
        return tuple()


__all__ = [
    "PresetDocument",
    "PresetPayloadModel",
    "PresetSignature",
    "PresetSignatureVerification",
    "SignatureDocumentModel",
    "canonical_preset_bytes",
    "normalize_preset_document",
]
