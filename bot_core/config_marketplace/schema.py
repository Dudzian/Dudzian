"""Schemat metadanych konfiguracji dla marketplace."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import List, Optional, Sequence

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    ValidationError,
    field_validator,
    model_validator,
)

SEMVER_PATTERN = r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"


class MarketplaceModel(BaseModel):
    """Bazowa klasa modeli marketplace z restrykcyjną konfiguracją."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


class LicenseInfo(MarketplaceModel):
    """Informacje o licencji dla paczki konfiguracji."""

    name: str = Field(..., min_length=1, max_length=200)
    url: HttpUrl
    spdx_id: Optional[str] = Field(
        None,
        description="Opcjonalny identyfikator SPDX pozwalający na jednoznaczną identyfikację licencji.",
    )


class DataRequirement(MarketplaceModel):
    """Opis wymaganych danych wejściowych/wyjściowych."""

    name: str = Field(..., min_length=1, max_length=120)
    description: Optional[str] = Field(None, max_length=500)
    data_format: str = Field(..., min_length=1, max_length=120, description="Format lub protokół danych.")
    required: bool = Field(True, description="Czy dane są wymagane do działania konfiguracji.")
    schema_uri: Optional[HttpUrl] = Field(
        None,
        description="Opcjonalny odnośnik do formalnego schematu danych (JSONSchema, Avro, itp.).",
    )


class ComponentDependency(MarketplaceModel):
    """Informacja o zależnościach od innych komponentów ekosystemu bota."""

    component: str = Field(..., min_length=1, max_length=120)
    min_version: str = Field(..., pattern=SEMVER_PATTERN)
    max_version: Optional[str] = Field(
        None,
        pattern=SEMVER_PATTERN,
        description="Opcjonalna maksymalna wspierana wersja komponentu.",
    )

    @model_validator(mode="after")
    def validate_version_range(self) -> "ComponentDependency":
        """Upewnia się, że wersje minimalna i maksymalna są w poprawnej relacji."""

        if self.max_version is None:
            return self

        min_parts = [int(part) for part in self.min_version.split("-")[0].split("+")[0].split(".")]
        max_parts = [int(part) for part in self.max_version.split("-")[0].split("+")[0].split(".")]

        if min_parts > max_parts:
            raise ValueError("min_version nie może być większa niż max_version")
        return self


class IntegrityInfo(MarketplaceModel):
    """Informacje integracyjne zapewniające spójność paczki."""

    checksum: str = Field(..., min_length=32, max_length=128, description="Suma kontrolna paczki (np. SHA256).")
    signature: Optional[str] = Field(
        None,
        min_length=32,
        max_length=1024,
        description="Opcjonalny podpis kryptograficzny metadanych/paczki w formacie base64.",
    )
    signing_key_id: Optional[str] = Field(
        None,
        min_length=3,
        max_length=120,
        pattern=r"^[a-z0-9_\-\.]+$",
        description="Identyfikator klucza publicznego wymaganego do weryfikacji podpisu.",
    )
    signature_algorithm: Optional[str] = Field(
        None,
        min_length=3,
        max_length=50,
        description="Nazwa algorytmu podpisu (np. ed25519, rsa-pss-sha256).",
    )
    fingerprint_whitelist: Optional[List[str]] = Field(
        None,
        description="Lista dozwolonych fingerprintów sprzętowych dla paczki.",
    )

    @model_validator(mode="after")
    def validate_signature_dependencies(self) -> "IntegrityInfo":
        """Zapewnia spójność pól związanych z podpisem i fingerprintami."""

        if self.signature is None:
            if self.signing_key_id is not None or self.signature_algorithm is not None:
                raise ValueError(
                    "Podpis nie został dostarczony – usuń signing_key_id i signature_algorithm lub uzupełnij signature."
                )
        else:
            missing: List[str] = []
            if self.signing_key_id is None:
                missing.append("signing_key_id")
            if self.signature_algorithm is None:
                missing.append("signature_algorithm")
            if missing:
                raise ValueError(
                    "Brakujące pola wymagane przy podpisie: " + ", ".join(sorted(set(missing)))
                )

        if self.fingerprint_whitelist is not None:
            duplicates = [fingerprint for fingerprint, occurrences in Counter(self.fingerprint_whitelist).items() if occurrences > 1]
            if duplicates:
                raise ValueError(
                    "Fingerprinty muszą być unikalne. Zduplikowane wartości: "
                    + ", ".join(sorted(duplicates))
                )

        return self


class ConfigurationMetadata(MarketplaceModel):
    """Główny schemat metadanych dla paczek konfiguracyjnych marketplace."""

    schema_version: str = Field(..., pattern=SEMVER_PATTERN, description="Wersja schematu metadanych.")
    config_id: str = Field(..., min_length=3, max_length=120, pattern=r"^[a-z0-9_\-\.]+$")
    config_version: str = Field(..., pattern=SEMVER_PATTERN)
    title: str = Field(..., min_length=3, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    author: str = Field(..., min_length=3, max_length=120)
    author_contact: Optional[str] = Field(None, max_length=200)
    license: LicenseInfo
    data_requirements: List[DataRequirement] = Field(default_factory=list)
    component_dependencies: List[ComponentDependency] = Field(default_factory=list)
    integrity: IntegrityInfo
    created_at: datetime = Field(..., description="Znacznik czasu utworzenia metadanych.")
    updated_at: datetime = Field(..., description="Znacznik czasu ostatniej aktualizacji metadanych.")
    tags: List[str] = Field(default_factory=list, max_length=30)

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: Sequence[str]) -> Sequence[str]:
        if len(value) != len(set(value)):
            raise ValueError("Tagi muszą być unikalne.")
        for tag in value:
            if len(tag) < 2 or len(tag) > 50:
                raise ValueError("Tagi muszą mieć długość od 2 do 50 znaków.")
            if not tag.replace("-", "").replace("_", "").isalnum():
                raise ValueError("Tagi mogą zawierać litery, cyfry oraz znaki '-' i '_'.")
        return value

    @model_validator(mode="after")
    def validate_timestamps(self) -> "ConfigurationMetadata":
        if self.updated_at < self.created_at:
            raise ValueError("updated_at nie może być wcześniejszy niż created_at")
        return self


__all__ = [
    "ConfigurationMetadata",
    "LicenseInfo",
    "DataRequirement",
    "ComponentDependency",
    "IntegrityInfo",
    "ValidationError",
]
