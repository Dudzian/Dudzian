"""Fabryka backendów ML zarządzająca fallbackami i priorytetami."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence, Tuple, TypeVar

from bot_core.ai import backends

T = TypeVar("T")


@dataclass(frozen=True)
class BackendEntry:
    """Opis pojedynczego backendu ML."""

    name: str
    builder: Callable[..., T]
    requires_backend: str | None = None
    description: str | None = None

    def is_available(self, *, config_path: Path | None = None) -> bool:
        if self.requires_backend is None:
            return True
        return backends.is_backend_available(self.requires_backend, config_path=config_path)


class BackendFactory:
    """Rejestr i logika wyboru backendów ML."""

    _registry: MutableMapping[str, BackendEntry] = {}

    @classmethod
    def register(
        cls,
        name: str,
        builder: Callable[..., T],
        *,
        requires_backend: str | None = None,
        description: str | None = None,
        replace: bool = False,
    ) -> None:
        key = name.strip().lower()
        if not key:
            raise ValueError("Nazwa backendu nie może być pusta")
        if not replace and key in cls._registry:
            raise ValueError(f"Backend '{key}' jest już zarejestrowany")
        cls._registry[key] = BackendEntry(
            name=key,
            builder=builder,
            requires_backend=requires_backend,
            description=description,
        )

    @classmethod
    def registry(cls) -> Mapping[str, BackendEntry]:
        return dict(cls._registry)

    @classmethod
    def _candidate_order(
        cls,
        preferred: Sequence[str] | None,
        *,
        config_path: Path | None = None,
    ) -> Iterator[str]:
        seen: set[str] = set()
        if preferred:
            for candidate in preferred:
                key = candidate.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    yield key
        for candidate in backends.get_backend_priority(config_path=config_path):
            key = candidate.strip().lower()
            if key and key not in seen:
                seen.add(key)
                yield key
        for key in cls._registry.keys():
            if key not in seen:
                yield key

    @classmethod
    def build(
        cls,
        *,
        preferred: Sequence[str] | None = None,
        config_path: Path | None = None,
        **kwargs: object,
    ) -> tuple[str, T]:
        last_error: Exception | None = None
        for candidate in cls._candidate_order(preferred, config_path=config_path):
            entry = cls._registry.get(candidate)
            if entry is None:
                continue
            if not entry.is_available(config_path=config_path):
                continue
            try:
                instance = entry.builder(**kwargs)
            except backends.BackendUnavailableError as exc:
                last_error = exc
                continue
            except ModuleNotFoundError as exc:
                last_error = backends.BackendUnavailableError(candidate, exc.name)
                continue
            except Exception as exc:  # pragma: no cover - defensywne logowanie
                last_error = exc
                continue
            return entry.name, instance

        if last_error:
            raise last_error
        raise backends.BackendUnavailableError(
            " lub ".join(cls._registry.keys()) or "brak dostępnych backendów",
            None,
        )


def register_backend(
    name: str,
    builder: Callable[..., T],
    *,
    requires_backend: str | None = None,
    description: str | None = None,
    replace: bool = False,
) -> None:
    BackendFactory.register(
        name,
        builder,
        requires_backend=requires_backend,
        description=description,
        replace=replace,
    )


def list_available_backends(*, config_path: Path | None = None) -> Dict[str, BackendEntry]:
    return {
        name: entry
        for name, entry in BackendFactory.registry().items()
        if entry.is_available(config_path=config_path)
    }


def build_backend(
    *,
    preferred: Sequence[str] | None = None,
    config_path: Path | None = None,
    **kwargs: object,
) -> tuple[str, T]:
    return BackendFactory.build(preferred=preferred, config_path=config_path, **kwargs)


# Rejestracja backendów wbudowanych.
from .backends import reference  # noqa: E402

register_backend(
    "reference",
    reference.build_reference_regressor,
    description="Referencyjny regresor liniowy w czystym Pythonie",
)

__all__ = [
    "BackendEntry",
    "BackendFactory",
    "register_backend",
    "list_available_backends",
    "build_backend",
]
