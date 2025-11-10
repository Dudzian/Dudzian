"""Utilities for normalising strategy names and generating alias sequences."""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


MIGRATION_FALLBACK_SUFFIX = "_migration_fallback"
"""Suffix used to mark migration fallback strategy aliases."""


def canonical_alias_map(alias_map: Mapping[str, Any] | None) -> dict[str, str]:
    """Return canonical alias mapping with normalised keys and values."""

    if not alias_map:
        return {}

    def _iter_alias_values(value: object) -> Iterable[str]:
        if isinstance(value, str):
            yield value
            return
        if isinstance(value, Mapping):
            for nested in value.values():
                yield from _iter_alias_values(nested)
            return
        if isinstance(value, Iterable):
            for item in value:
                yield from _iter_alias_values(item)

    canonical: dict[str, str] = {}
    for raw_key, raw_target in alias_map.items():
        key = str(raw_key or "").strip()
        if not key:
            continue
        if isinstance(raw_target, str):
            target = raw_target.strip()
            if target:
                canonical[key] = target
            continue
        for alias in _iter_alias_values(raw_target):
            alias_key = str(alias or "").strip()
            if not alias_key:
                continue
            canonical[alias_key] = key
        # Ensure canonical key maps to itself if nested aliases were provided.
        if key not in canonical:
            canonical[key] = key
    return canonical


def normalise_suffixes(suffixes: Iterable[str] | None) -> tuple[str, ...]:
    """Return tuple of unique, normalised suffixes preserving order."""

    if not suffixes:
        return ()

    seen: set[str] = set()
    ordered: list[str] = []
    for raw_suffix in suffixes:
        suffix = str(raw_suffix or "").strip()
        if not suffix or suffix in seen:
            continue
        seen.add(suffix)
        ordered.append(suffix)
    return tuple(ordered)


def normalise_alias_map(alias_map: Mapping[str, str] | None) -> dict[str, str]:
    """Return mapping expanded with normalised aliases for lookup."""

    if not alias_map:
        return {}

    normalised: dict[str, str] = {}
    for raw_key, raw_target in alias_map.items():
        target = str(raw_target).strip()
        if not target:
            continue
        for variant in strategy_key_aliases(raw_key):
            if variant in normalised:
                continue
            normalised[variant] = target
        key = str(raw_key).strip()
        if key and key not in normalised:
            normalised[key] = target
    return normalised


def strategy_key_aliases(name: str) -> tuple[str, ...]:
    """Return normalised variants for ``name`` used in cache keys."""

    base = str(name or "")
    stripped = base.strip()
    variants: list[str] = []
    seen: set[str] = set()
    for candidate in (
        base,
        stripped,
        stripped.lower(),
        stripped.replace(" ", "_"),
        stripped.replace(" ", "_").lower(),
        stripped.replace("-", "_"),
        stripped.replace("-", "_").lower(),
        stripped.replace("-", "_").replace(" ", "_"),
        stripped.replace("-", "_").replace(" ", "_").lower(),
        stripped.replace("-", " "),
        stripped.replace("-", " ").lower(),
        stripped.replace("_", " "),
        stripped.replace("_", " ").lower(),
    ):
        normalized = candidate.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        variants.append(normalized)
    return tuple(variants)


def strategy_name_candidates(
    name: str,
    alias_map: Mapping[str, str] | None = None,
    suffixes: Iterable[str] | None = None,
    *,
    normalised: bool = False,
) -> tuple[str, ...]:
    """Return ordered lookup candidates for strategy ``name``."""

    raw = str(name or "").strip()
    if not raw:
        return ()

    queue: deque[str] = deque([raw])
    ordered: list[str] = []
    seen: set[str] = set()
    suffixes = tuple(suffixes or ())
    if alias_map and not normalised:
        alias_map = normalise_alias_map(alias_map)
    elif not alias_map:
        alias_map = {}

    while queue:
        candidate = queue.popleft()
        if not candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
        for variant in strategy_key_aliases(candidate):
            if variant not in seen:
                queue.append(variant)
        alias = alias_map.get(candidate) if alias_map else None
        if alias and alias not in seen:
            queue.append(alias)
        for suffix in suffixes:
            if suffix and candidate.endswith(suffix):
                trimmed = candidate[: -len(suffix)]
                if trimmed and trimmed not in seen:
                    queue.append(trimmed)
    return tuple(ordered)


@dataclass(slots=True)
class StrategyAliasResolver:
    """Small helper bundling normalised alias map and suffix handling."""

    base_alias_map: Mapping[str, str] | None = None
    base_suffixes: Iterable[str] | None = None
    _alias_map: Mapping[str, str] = field(init=False, repr=False)
    _suffixes: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._alias_map = MappingProxyType(normalise_alias_map(self.base_alias_map))
        self._suffixes = tuple(self.base_suffixes or ())

    @property
    def alias_map(self) -> Mapping[str, str]:
        """Return the cached normalised alias mapping."""

        return self._alias_map

    @property
    def suffixes(self) -> tuple[str, ...]:
        """Return suffixes considered when generating candidates."""

        return self._suffixes

    def candidates(self, name: str) -> tuple[str, ...]:
        """Return candidate sequence for ``name`` using cached data."""

        return strategy_name_candidates(
            name,
            self._alias_map,
            self._suffixes,
            normalised=True,
        )

    def derive(
        self,
        *,
        alias_map: Mapping[str, str] | None = None,
        suffixes: Iterable[str] | None = None,
    ) -> "StrategyAliasResolver":
        """Return a resolver based on overrides keeping cached semantics."""

        return type(self)(
            alias_map if alias_map is not None else self.base_alias_map,
            suffixes if suffixes is not None else self._suffixes,
        )

    def extend(
        self,
        *,
        alias_map: Mapping[str, str] | None = None,
        suffixes: Iterable[str] | None = None,
    ) -> "StrategyAliasResolver":
        """Return resolver with overrides merged into the current base."""

        base_map = canonical_alias_map(self.base_alias_map)
        override_map = canonical_alias_map(alias_map)
        merged_map: dict[str, str] = {}
        if base_map:
            merged_map.update(base_map)
        if override_map:
            merged_map.update(override_map)

        base_suffixes = tuple(self.base_suffixes or self._suffixes)
        if suffixes is not None:
            combined_suffixes = normalise_suffixes(
                base_suffixes + tuple(suffixes)
            )
        else:
            combined_suffixes = base_suffixes

        return type(self)(
            merged_map or None,
            combined_suffixes if combined_suffixes else None,
        )
