"""Utilities for normalising strategy names and generating alias sequences."""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


def canonical_alias_map(alias_map: Mapping[str, object] | None) -> dict[str, str]:
    """Return mapping coercing keys and targets to trimmed strings.

    Values can be provided either as direct target strings (``{"alias": "target"}``)
    or as iterables of aliases keyed by the canonical strategy name (``{"target":
    ["alias-a", "alias-b"]}``).  Nested iterables and mappings are flattened and
    empty values are ignored.
    """

    if not alias_map:
        return {}

    def _flatten_aliases(source: object) -> list[str]:
        queue: deque[object] = deque([source])
        flattened: list[str] = []
        while queue:
            item = queue.popleft()
            if isinstance(item, Mapping):
                queue.extend(item.values())
                continue
            if isinstance(item, Iterable) and not isinstance(
                item, (str, bytes, bytearray)
            ):
                queue.extend(item)
                continue
            text = str(item or "").strip()
            if text:
                flattened.append(text)
        return flattened

    cleaned: dict[str, str] = {}
    for raw_key, raw_target in alias_map.items():
        key = str(raw_key or "").strip()
        if not key:
            continue

        if isinstance(raw_target, Mapping) or (
            isinstance(raw_target, Iterable)
            and not isinstance(raw_target, (str, bytes, bytearray))
        ):
            aliases = _flatten_aliases(raw_target)
            for alias in aliases:
                cleaned[alias] = key
            continue

        target = str(raw_target or "").strip()
        if not target:
            continue
        cleaned[key] = target
    return cleaned


def normalise_alias_map(alias_map: Mapping[str, str] | None) -> dict[str, str]:
    """Return mapping expanded with normalised aliases for lookup."""

    cleaned = canonical_alias_map(alias_map)
    if not cleaned:
        return {}

    normalised: dict[str, str] = {}
    for raw_key, target in cleaned.items():
        for variant in strategy_key_aliases(raw_key):
            if variant in normalised:
                continue
            normalised[variant] = target
        key = str(raw_key).strip()
        if key and key not in normalised:
            normalised[key] = target
    return normalised


def normalise_suffixes(suffixes: Iterable[str] | None) -> tuple[str, ...]:
    """Return unique, trimmed suffixes preserving original order."""

    if suffixes is None:
        return ()

    ordered: list[str] = []
    seen: set[str] = set()
    for raw in suffixes:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return tuple(ordered)


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
        self._suffixes = normalise_suffixes(self.base_suffixes)

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
        alias_map: Mapping[str, str] | None = None,
        suffixes: Iterable[str] | None = None,
    ) -> "StrategyAliasResolver":
        """Return resolver with alias map merged and suffixes extended."""

        merged_map: Mapping[str, str] | None = self.base_alias_map
        if alias_map:
            base: dict[str, str] = dict(canonical_alias_map(self.base_alias_map))
            for key, target in canonical_alias_map(alias_map).items():
                base[key] = target
            merged_map = base

        merged_suffixes = self._suffixes
        if suffixes is not None:
            merged_suffixes = tuple(
                dict.fromkeys((*merged_suffixes, *normalise_suffixes(suffixes)))
            )

        if merged_map is self.base_alias_map and merged_suffixes == self._suffixes:
            return self

        return type(self)(merged_map, merged_suffixes)
