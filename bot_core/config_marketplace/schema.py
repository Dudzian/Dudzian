"""Schemat metadanych konfiguracji Marketplace."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from pydantic import AnyUrl, BaseModel, Field, HttpUrl, ValidationError, model_validator

_PACKAGE_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]{2,63}$"
_SEMVER_PATTERN = r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
_HASH_PATTERN = r"^[A-Fa-f0-9]{32,128}$"


class Maintainer(BaseModel):
    """Informacje o autorach/opiekunach pakietu."""

    name: str = Field(..., description="Imię i nazwisko lub nazwa zespołu.")
    email: str | None = Field(
        None,
        pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
        description="Opcjonalny adres kontaktowy do zgłaszania problemów.",
    )
    organization: str | None = Field(
        None, description="Organizacja odpowiedzialna za utrzymanie pakietu."
    )
    role: str | None = Field(None, description="Rola w projekcie (autor, maintainer...).")
    url: HttpUrl | None = Field(None, description="Strona domowa autora." )


class LicenseInfo(BaseModel):
    """Opis licencji dystrybucyjnej konfiguracji."""

    name: str = Field(..., description="Nazwa licencji (np. Proprietary, MIT).")
    spdx_id: str | None = Field(
        None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9.+-]{1,63}$",
        description="Identyfikator SPDX jeżeli licencja jest zarejestrowana.",
    )
    url: HttpUrl | None = Field(None, description="Link do pełnej treści licencji.")
    terms_summary: str | None = Field(
        None, description="Skrót najważniejszych warunków licencji."
    )
    redistributable: bool = Field(
        True,
        description="Czy paczkę można przekazywać dalej w ramach instalacji.",
    )
    commercial_use: bool = Field(
        True,
        description="Czy licencja dopuszcza użycie komercyjne konfiguracji.",
    )


class DataAssetRequirement(BaseModel):
    """Wymagane strumienie danych do działania konfiguracji."""

    asset_type: str = Field(
        ...,
        description="Rodzaj zasobu danych (market, account, reference, custom...).",
    )
    provider: str = Field(..., description="Dostawca danych lub wewnętrzny serwis.")
    access_level: str = Field(
        ..., description="Klasyfikacja dostępu (public, restricted, confidential)."
    )
    retention_days: int | None = Field(
        None, description="Wymagana minimalna retencja historyczna danych."
    )
    min_history_days: int | None = Field(
        None, description="Minimalna długość historii wykorzystywana przez model."
    )
    sampling: str | None = Field(
        None,
        description="Granularność danych (np. 1m, tick, eod).",
    )
    notes: str | None = Field(None, description="Dodatkowe uwagi lub komentarze.")


class DataRequirements(BaseModel):
    """Zestaw wymagań danych dla konfiguracji Marketplace."""

    inputs: list[DataAssetRequirement] = Field(
        default_factory=list,
        description="Lista wymaganych strumieni danych.",
    )
    requires_personal_data: bool = Field(
        False, description="Czy konfiguracja przetwarza dane osobowe klientów."
    )
    requires_realtime_access: bool = Field(
        False, description="Czy wymagany jest dostęp do danych w czasie rzeczywistym."
    )
    pii_handling_notes: str | None = Field(
        None, description="Instrukcje obchodzenia się z danymi wrażliwymi."
    )


class ArtifactIntegrity(BaseModel):
    """Informacje o integralności pliku dystrybucyjnego."""

    algorithm: str = Field(
        "sha256",
        description="Algorytm skrótu użyty do wyliczenia digestu.",
    )
    digest: str = Field(
        ..., pattern=_HASH_PATTERN, description="Oczekiwany skrót artefaktu (hex)."
    )

    def normalized_algorithm(self) -> str:
        return self.algorithm.strip().lower()


class ArtifactSignature(BaseModel):
    """Podpis kryptograficzny artefaktu."""

    algorithm: str = Field(
        "HMAC-SHA256",
        description="Algorytm podpisu (np. HMAC-SHA256).",
    )
    key_id: str = Field(
        ...,
        description="Identyfikator klucza wymagany do weryfikacji podpisu.",
    )
    value: str = Field(
        ...,
        description="Wartość podpisu (najczęściej zakodowana base64).",
    )
    signed_fields: Sequence[str] = Field(
        default_factory=lambda: ("digest", "uri"),
        description="Pola ładunku objęte podpisem w celu walidacji po stronie klienta.",
    )


class DistributionArtifact(BaseModel):
    """Opis pojedynczego artefaktu dystrybucyjnego konfiguracji."""

    name: str = Field(..., description="Nazwa logiczna artefaktu (np. config-bundle).")
    uri: AnyUrl | str = Field(
        ...,
        description="Źródło paczki – może wskazywać na URL HTTP(S) lub ścieżkę lokalną.",
    )
    kind: str = Field(
        "config",
        description="Klasyfikacja artefaktu (config, dataset, docs ...).",
    )
    description: str | None = Field(
        None, description="Krótki opis zawartości artefaktu."
    )
    size_bytes: int | None = Field(
        None, description="Rozmiar oczekiwany po stronie klienta (opcjonalnie)."
    )
    integrity: ArtifactIntegrity | None = Field(
        None, description="Informacje potrzebne do weryfikacji integralności.",
    )
    signature: ArtifactSignature | None = Field(
        None, description="Podpis kryptograficzny artefaktu.",
    )


class HardwareFingerprintPolicy(BaseModel):
    """Zasady wiążące paczkę z fingerprintem sprzętu."""

    mode: str = Field(
        "none",
        description="Sposób egzekwowania (none, allowlist, prefix).",
    )
    allowed_fingerprints: Sequence[str] = Field(
        default_factory=list,
        description="Lista akceptowanych fingerprintów (pełne wartości lub prefiksy).",
    )
    require_strict_match: bool = Field(
        True,
        description="Czy wymagane jest dokładne dopasowanie fingerprintu.",
    )
    audit_message: str | None = Field(
        None,
        description="Wiadomość zapisywana w audycie przy naruszeniach polityki.",
    )


class VersionCompatibility(BaseModel):
    """Deklaracja kompatybilności z wersjami platformy."""

    component: str = Field(
        "core",
        description="Identyfikator komponentu (core, bot_core, ui, gateway ...).",
    )
    minimum: str | None = Field(
        None, description="Minimalna wspierana wersja komponentu."
    )
    maximum: str | None = Field(
        None, description="Maksymalna wspierana wersja komponentu."
    )
    recommended: str | None = Field(
        None, description="Rekomendowana wersja komponentu.",
    )
    channel: str | None = Field(
        None, description="Kanał dystrybucji (stable, beta, nightly...)."
    )
    notes: str | None = Field(None, description="Uwagi dotyczące zgodności.")


class ReleaseReviewer(BaseModel):
    """Informacje o recenzencie odpowiedzialnym za weryfikację paczki."""

    name: str = Field(..., description="Imię i nazwisko lub nazwa recenzenta.")
    email: str | None = Field(
        None,
        pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
        description="Adres e-mail recenzenta (opcjonalnie).",
    )
    role: str | None = Field(
        None, description="Rola recenzenta (QA, compliance, produkt...)."
    )


class ReleaseMetadata(BaseModel):
    """Status recenzji i kanał publikacji paczki Marketplace."""

    channel: str = Field(
        "internal",
        description="Kanał dystrybucji (internal, beta, public, deprecated).",
    )
    review_status: str = Field(
        "pending",
        description="Status procesu review (pending, in_review, approved, rejected).",
    )
    reviewers: list[ReleaseReviewer] = Field(
        default_factory=list,
        description="Lista osób zatwierdzających publikację.",
    )
    ticket: HttpUrl | None = Field(
        None,
        description="Link do zgłoszenia/artefaktu zatwierdzającego (np. Jira).",
    )
    approved_at: datetime | None = Field(
        None, description="Znacznik czasu zatwierdzenia publikacji.",
    )
    notes: str | None = Field(
        None, description="Uwagi do publikacji (np. warunki dodatkowe).",
    )

    @model_validator(mode="after")
    def _validate_review(cls, values: "ReleaseMetadata") -> "ReleaseMetadata":
        status = values.review_status.lower().strip()
        if status == "approved":
            if not values.reviewers:
                raise ValueError(
                    "Zatwierdzone wydanie musi mieć przynajmniej jednego recenzenta."
                )
            if values.approved_at is None:
                raise ValueError(
                    "Pole approved_at jest wymagane przy statusie review 'approved'."
                )
        return values


class ExchangeCompatibilityEntry(BaseModel):
    """Deklaracja kompatybilności paczki z konkretną giełdą."""

    exchange: str = Field(
        ...,
        pattern=r"^[A-Z0-9][A-Z0-9._-]{1,31}$",
        description="Kod giełdy (np. BINANCE, OKX, KRAKEN).",
    )
    environments: list[str] = Field(
        default_factory=list,
        description="Środowiska obsługiwane przez preset (paper, testnet, live...).",
    )
    trading_modes: list[str] = Field(
        default_factory=list,
        description="Tryby handlu (spot, margin, futures ...).",
    )
    status: str = Field(
        "beta",
        description="Status certyfikacji (beta, certified, deprecated...).",
    )
    last_verified_at: datetime | None = Field(
        None, description="Znacznik czasu ostatniej weryfikacji funkcjonalnej.",
    )
    notes: str | None = Field(
        None, description="Uwagi dotyczące ograniczeń lub rekomendacji.",
    )


class VersioningMetadata(BaseModel):
    """Informacje o wersjonowaniu i następstwie paczki."""

    channel: str = Field(
        "internal",
        description="Kanał dystrybucji (internal, beta, public, deprecated).",
    )
    iteration: str = Field(
        "minor",
        description="Typ wydania (major, minor, patch, hotfix).",
    )
    supersedes: list[str] = Field(
        default_factory=list,
        description="Lista referencji package@version zastąpionych przez bieżące wydanie.",
    )
    superseded_by: list[str] = Field(
        default_factory=list,
        description="Lista referencji package@version następnych wydań (jeśli dostępne).",
    )
    migration_required: bool = Field(
        False,
        description="Czy wdrożenie wymaga migracji ręcznej po stronie klienta.",
    )
    source: str | None = Field(
        None,
        description="Ścieżka presetu (względnie wobec config/marketplace/presets).",
    )

    @model_validator(mode="after")
    def _validate_references(cls, values: "VersioningMetadata") -> "VersioningMetadata":
        seen = {values.source} if values.source else set()
        for ref in (*values.supersedes, *values.superseded_by):
            if "@" not in ref:
                raise ValueError(
                    "Wpisy w supersedes/superseded_by muszą mieć format package@version."
                )
            if ref in seen:
                raise ValueError(f"Duplikat odniesienia wersji: {ref}")
            seen.add(ref)
        return values


class MarketplacePackageMetadata(BaseModel):
    """Metadane pojedynczej paczki Marketplace."""

    schema_version: str = Field(
        "1.0",
        description="Wersja schematu metadanych Marketplace.",
    )
    package_id: str = Field(
        ..., pattern=_PACKAGE_ID_PATTERN, description="Identyfikator paczki."
    )
    display_name: str = Field(..., description="Przyjazna nazwa paczki.")
    summary: str = Field(..., description="Krótki opis funkcjonalny.")
    description: str = Field(..., description="Pełen opis paczki.")
    version: str = Field(
        ..., pattern=_SEMVER_PATTERN, description="Wersja paczki zgodna z SemVer."
    )
    revision: str | None = Field(
        None, description="Dodatkowy identyfikator rewizji (build, commit SHA)."
    )
    release_date: datetime | None = Field(
        None, description="Data publikacji paczki w ISO-8601."
    )
    maintainers: list[Maintainer] = Field(
        default_factory=list, description="Lista osób odpowiedzialnych za paczkę."
    )
    license: LicenseInfo = Field(..., description="Informacje licencyjne.")
    tags: list[str] = Field(default_factory=list, description="Tagi ułatwiające wyszukiwanie.")
    data_requirements: DataRequirements = Field(
        default_factory=DataRequirements,
        description="Deklaracja wymagań danych.",
    )
    distribution: list[DistributionArtifact] = Field(
        default_factory=list, description="Artefakty publikowane w ramach paczki."
    )
    compatibility: list[VersionCompatibility] = Field(
        default_factory=list,
        description="Macierz zgodności z wersjami platformy.",
    )
    documentation_url: HttpUrl | None = Field(
        None, description="Link do zewnętrznej dokumentacji paczki."
    )
    release_notes: Sequence[str] = Field(
        default_factory=tuple, description="Lista istotnych zmian w wydaniu."
    )
    security: HardwareFingerprintPolicy = Field(
        default_factory=HardwareFingerprintPolicy,
        description="Zasady bezpieczeństwa (fingerprint).",
    )
    release: ReleaseMetadata = Field(
        default_factory=ReleaseMetadata,
        description="Status recenzji i kanał publikacji paczki.",
    )
    exchange_compatibility: list[ExchangeCompatibilityEntry] = Field(
        default_factory=list,
        description="Lista kompatybilnych giełd wraz z zakresem wsparcia.",
    )
    versioning: VersioningMetadata = Field(
        default_factory=VersioningMetadata,
        description="Informacje o wersjonowaniu i następstwie paczki.",
    )

    def signed_payload(self, artifact: DistributionArtifact) -> Mapping[str, object]:
        """Buduje ładunek podpisu na podstawie metadanych."""

        payload = {
            "package_id": self.package_id,
            "version": self.version,
            "artifact": artifact.name,
            "uri": str(artifact.uri),
        }
        if artifact.integrity:
            payload[artifact.integrity.normalized_algorithm()] = artifact.integrity.digest
        return payload


class MarketplaceCatalog(BaseModel):
    """Indeks paczek marketplace."""

    schema_version: str = Field(
        "1.0",
        description="Wersja schematu katalogu Marketplace.",
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.utcnow(),
        description="Znacznik czasu wygenerowania katalogu.",
    )
    packages: list[MarketplacePackageMetadata] = Field(
        default_factory=list,
        description="Lista paczek dostępnych w katalogu.",
    )

    def find(self, package_id: str) -> MarketplacePackageMetadata | None:
        normalized = package_id.strip().lower()
        for entry in self.packages:
            if entry.package_id.lower() == normalized:
                return entry
        return None


class MarketplaceRepositoryConfig(BaseModel):
    """Konfiguracja lokalnego repozytorium Marketplace."""

    version: int = Field(1, description="Wersja formatu konfiguracji repozytorium.")
    remote_index_url: AnyUrl | str | None = Field(
        None,
        description="Domyślne źródło indeksu (HTTPS/HTTPS lub ścieżka plikowa).",
    )
    last_sync_at: datetime | None = Field(
        None, description="Znacznik czasu ostatniej synchronizacji katalogu."
    )
    etag: str | None = Field(
        None, description="Wartość ETag ostatnio pobranego indeksu (jeśli dostępny)."
    )


def load_catalog(path: Path) -> MarketplaceCatalog:
    """Wczytuje katalog Marketplace z pliku JSON."""

    document = path.read_text(encoding="utf-8")
    try:
        return MarketplaceCatalog.model_validate_json(document)
    except ValidationError as exc:  # pragma: no cover - defensywne logowanie
        raise ValueError(f"Niepoprawny katalog Marketplace: {path}") from exc


def load_repository_config(path: Path) -> MarketplaceRepositoryConfig:
    """Wczytuje konfigurację repozytorium Marketplace."""

    document = path.read_text(encoding="utf-8")
    try:
        return MarketplaceRepositoryConfig.model_validate_json(document)
    except ValidationError as exc:  # pragma: no cover - defensywne logowanie
        raise ValueError(f"Niepoprawna konfiguracja repozytorium Marketplace: {path}") from exc


__all__ = [
    "ArtifactIntegrity",
    "ArtifactSignature",
    "DataAssetRequirement",
    "DataRequirements",
    "DistributionArtifact",
    "HardwareFingerprintPolicy",
    "LicenseInfo",
    "Maintainer",
    "MarketplaceCatalog",
    "MarketplacePackageMetadata",
    "MarketplaceRepositoryConfig",
    "ReleaseMetadata",
    "ReleaseReviewer",
    "ExchangeCompatibilityEntry",
    "VersioningMetadata",
    "VersionCompatibility",
    "load_catalog",
    "load_repository_config",
]
