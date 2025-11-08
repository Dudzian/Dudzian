"""Modele manifestów pluginów strategii."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


@dataclass(slots=True)
class PluginAuthor:
    """Autor pakietu strategii dostępnego w marketplace."""

    name: str
    email: str | None = None
    website: str | None = None

    def to_dict(self) -> Mapping[str, Any]:
        payload: dict[str, Any] = {"name": self.name}
        if self.email:
            payload["email"] = self.email
        if self.website:
            payload["website"] = self.website
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PluginAuthor":
        return cls(
            name=str(payload.get("name") or "").strip(),
            email=_normalize_optional_str(payload.get("email")),
            website=_normalize_optional_str(payload.get("website")),
        )


@dataclass(slots=True)
class StrategyPluginManifest:
    """Manifest pluginu opisujący strategie i wymagania."""

    identifier: str
    version: str
    title: str
    description: str
    author: PluginAuthor
    strategies: Sequence[str] = field(default_factory=tuple)
    capabilities: Sequence[str] = field(default_factory=tuple)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "id": self.identifier,
            "version": self.version,
            "title": self.title,
            "description": self.description,
            "author": dict(self.author.to_dict()),
            "strategies": list(self.strategies),
            "capabilities": list(self.capabilities),
            "created_at": _format_timestamp(self.created_at),
            "metadata": dict(self.metadata),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False) + "\n"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StrategyPluginManifest":
        identifier = str(payload.get("id") or payload.get("identifier") or "").strip()
        version = str(payload.get("version") or "").strip()
        title = str(payload.get("title") or "").strip()
        description = str(payload.get("description") or "").strip()
        author_payload = payload.get("author")
        if not isinstance(author_payload, Mapping):
            raise ValueError("Manifest wymaga sekcji 'author'")
        author = PluginAuthor.from_dict(author_payload)

        def _normalize_seq(values: Iterable[str] | None) -> tuple[str, ...]:
            if not values:
                return ()
            seen: MutableMapping[str, None] = {}
            normalized: list[str] = []
            for raw in values:
                text = str(raw).strip()
                if not text or text in seen:
                    continue
                seen[text] = None
                normalized.append(text)
            return tuple(normalized)

        strategies = _normalize_seq(_coerce_iterable(payload.get("strategies")))
        capabilities = _normalize_seq(_coerce_iterable(payload.get("capabilities")))

        created_at = _parse_timestamp(payload.get("created_at"))

        metadata = dict(payload.get("metadata") or {})

        return cls(
            identifier=identifier,
            version=version,
            title=title,
            description=description,
            author=author,
            strategies=strategies,
            capabilities=capabilities,
            created_at=created_at,
            metadata=metadata,
        )

    @classmethod
    def from_json(cls, data: str) -> "StrategyPluginManifest":
        payload = json.loads(data)
        if not isinstance(payload, Mapping):
            raise TypeError("Manifest JSON powinien być obiektem")
        return cls.from_dict(payload)


@dataclass(slots=True)
class PluginSignature:
    """Informacje o podpisie cyfrowym manifestu."""

    algorithm: str
    key_id: str | None
    value: str

    def to_dict(self) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "algorithm": self.algorithm,
            "value": self.value,
        }
        if self.key_id:
            payload["key_id"] = self.key_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PluginSignature":
        return cls(
            algorithm=str(payload.get("algorithm") or "").strip(),
            key_id=_normalize_optional_str(payload.get("key_id")),
            value=str(payload.get("value") or "").strip(),
        )


@dataclass(slots=True)
class SignedStrategyPlugin:
    """Manifest wraz z podpisem i dodatkowymi metadanymi review."""

    manifest: StrategyPluginManifest
    signature: PluginSignature
    review_notes: Sequence[str] = field(default_factory=tuple)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "manifest": dict(self.manifest.to_dict()),
            "signature": dict(self.signature.to_dict()),
            "review_notes": list(self.review_notes),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False) + "\n"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SignedStrategyPlugin":
        manifest_payload = payload.get("manifest")
        signature_payload = payload.get("signature")
        if not isinstance(manifest_payload, Mapping):
            raise ValueError("Package wymaga sekcji 'manifest'")
        if not isinstance(signature_payload, Mapping):
            raise ValueError("Package wymaga sekcji 'signature'")
        review_notes = tuple(
            str(note).strip()
            for note in _coerce_iterable(payload.get("review_notes"))
            if str(note).strip()
        )
        return cls(
            manifest=StrategyPluginManifest.from_dict(manifest_payload),
            signature=PluginSignature.from_dict(signature_payload),
            review_notes=review_notes,
        )

    @classmethod
    def from_json(cls, data: str) -> "SignedStrategyPlugin":
        payload = json.loads(data)
        if not isinstance(payload, Mapping):
            raise TypeError("Package JSON powinien być obiektem")
        return cls.from_dict(payload)


def _normalize_optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def _coerce_iterable(values: Any) -> Iterable[str]:
    if values is None:
        return ()
    if isinstance(values, (list, tuple, set)):
        return tuple(values)
    if isinstance(values, str):
        return (values,)
    if isinstance(values, Iterable):
        return tuple(values)
    raise TypeError("Oczekiwano sekwencji lub pojedynczego łańcucha znaków")


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            timestamp = datetime.utcnow().replace(tzinfo=timezone.utc)
        else:
            if candidate.endswith("Z"):
                candidate = candidate[:-1] + "+00:00"
            timestamp = datetime.fromisoformat(candidate)
    elif value is None:
        timestamp = datetime.utcnow().replace(tzinfo=timezone.utc)
    else:
        raise TypeError("created_at wymaga wartości datetime lub ISO8601")
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    normalized = value
    if normalized.tzinfo is None:
        normalized = normalized.replace(tzinfo=timezone.utc)
    return (
        normalized.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

