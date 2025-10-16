"""Polityki walidacji paczek odporności Stage6."""

from __future__ import annotations

import json
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Mapping, Sequence


_ALLOWED_SEVERITIES = {"error", "warning"}


@dataclass(slots=True)
class PatternRequirement:
    """Wymóg dopasowania wzorców plików w paczce odpornościowej."""

    pattern: str
    description: str
    min_matches: int = 1
    severity: str = "error"

    def validate(self) -> None:
        if self.min_matches < 1:
            raise ValueError(
                f"Wymagane dopasowania muszą być >= 1 dla wzorca {self.pattern}"
            )
        if self.severity not in _ALLOWED_SEVERITIES:
            raise ValueError(
                f"Nieobsługiwany poziom istotności {self.severity} dla {self.pattern}"
            )


@dataclass(slots=True)
class MetadataRequirement:
    """Wymogi wobec metadanych manifestu paczki odpornościowej."""

    key: str
    description: str
    required: bool = True
    allowed_values: Sequence[str] | None = None
    severity: str = "error"

    def validate(self) -> None:
        if not self.key:
            raise ValueError("Klucz wymogu metadanych nie może być pusty")
        if self.severity not in _ALLOWED_SEVERITIES:
            raise ValueError(
                f"Nieobsługiwany poziom istotności {self.severity} dla klucza {self.key}"
            )


@dataclass(slots=True)
class ResiliencePolicy:
    """Zbiór wymogów weryfikowanych podczas audytu paczek."""

    pattern_requirements: tuple[PatternRequirement, ...]
    metadata_requirements: tuple[MetadataRequirement, ...]

    @classmethod
    def empty(cls) -> "ResiliencePolicy":
        return cls(pattern_requirements=(), metadata_requirements=())


def _load_json_policy(path: Path) -> Mapping[str, object]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:  # noqa: BLE001 - komunikaty konfiguracyjne
        raise ValueError(f"Nie można odczytać polityki {path}: {exc}") from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:  # noqa: BLE001 - komunikaty konfiguracyjne
        raise ValueError(f"Błąd parsowania polityki {path}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ValueError("Polityka musi być obiektem JSON")
    return document


def load_policy(path: Path) -> ResiliencePolicy:
    """Ładuje politykę wymagań z pliku JSON."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Plik polityki nie istnieje: {path}")

    document = _load_json_policy(path)

    pattern_items = document.get("required_patterns", [])
    if not isinstance(pattern_items, Sequence):
        raise ValueError("Pole required_patterns powinno być listą")

    patterns: list[PatternRequirement] = []
    for index, item in enumerate(pattern_items, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"Pozycja {index} w required_patterns musi być obiektem")
        pattern = str(item.get("pattern", "")).strip()
        description = str(item.get("description", "")).strip()
        min_matches = int(item.get("min_matches", 1))
        severity = str(item.get("severity", "error")).strip().lower()
        requirement = PatternRequirement(
            pattern=pattern,
            description=description or pattern,
            min_matches=min_matches,
            severity=severity,
        )
        requirement.validate()
        patterns.append(requirement)

    metadata_items = document.get("metadata", [])
    if not isinstance(metadata_items, Sequence):
        raise ValueError("Pole metadata powinno być listą")

    metadata_requirements: list[MetadataRequirement] = []
    for index, item in enumerate(metadata_items, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"Pozycja {index} w metadata musi być obiektem")
        key = str(item.get("key", "")).strip()
        description = str(item.get("description", "")).strip() or key
        required = bool(item.get("required", True))
        severity = str(item.get("severity", "error")).strip().lower()
        allowed = item.get("allowed_values")
        if allowed is not None:
            if not isinstance(allowed, Sequence) or isinstance(allowed, (str, bytes)):
                raise ValueError("allowed_values musi być listą wartości dozwolonych")
            allowed_values = tuple(str(value) for value in allowed)
        else:
            allowed_values = None
        requirement = MetadataRequirement(
            key=key,
            description=description,
            required=required,
            allowed_values=allowed_values,
            severity=severity,
        )
        requirement.validate()
        metadata_requirements.append(requirement)

    return ResiliencePolicy(
        pattern_requirements=tuple(patterns),
        metadata_requirements=tuple(metadata_requirements),
    )


def evaluate_policy(
    manifest: Mapping[str, object], policy: ResiliencePolicy
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Sprawdza manifest paczki względem polityki i zwraca listy błędów/ostrzeżeń."""

    if not policy.pattern_requirements and not policy.metadata_requirements:
        return (), ()

    files_field = manifest.get("files", [])
    paths: list[str] = []
    if isinstance(files_field, Sequence):
        for item in files_field:
            if isinstance(item, Mapping):
                value = item.get("path")
                if isinstance(value, str):
                    paths.append(value)

    metadata = manifest.get("metadata")
    metadata_map: Mapping[str, object]
    if isinstance(metadata, Mapping):
        metadata_map = metadata
    else:
        metadata_map = {}

    errors: list[str] = []
    warnings: list[str] = []

    for requirement in policy.pattern_requirements:
        matches = sum(1 for path in paths if fnmatch(path, requirement.pattern))
        if matches >= requirement.min_matches:
            continue
        message = (
            f"Wymóg '{requirement.description}' niespełniony: "
            f"znaleziono {matches}, wymagane >= {requirement.min_matches} (wzorzec {requirement.pattern})"
        )
        target = warnings if requirement.severity == "warning" else errors
        target.append(message)

    for requirement in policy.metadata_requirements:
        if requirement.required and requirement.key not in metadata_map:
            message = (
                f"Brak wymaganego klucza metadanych '{requirement.key}' ({requirement.description})"
            )
            target = warnings if requirement.severity == "warning" else errors
            target.append(message)
            continue

        if requirement.allowed_values is not None and requirement.key in metadata_map:
            value = metadata_map.get(requirement.key)
            if str(value) not in requirement.allowed_values:
                allowed = ", ".join(requirement.allowed_values)
                message = (
                    f"Wartość metadanych '{requirement.key}'={value!r} nie jest jedną z: {allowed}"
                )
                target = warnings if requirement.severity == "warning" else errors
                target.append(message)

    return tuple(errors), tuple(warnings)


__all__ = [
    "PatternRequirement",
    "MetadataRequirement",
    "ResiliencePolicy",
    "load_policy",
    "evaluate_policy",
]

