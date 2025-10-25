"""Utilities for normalising strategy names and generating alias sequences."""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


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
