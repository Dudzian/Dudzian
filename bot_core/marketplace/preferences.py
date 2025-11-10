"""Persistent storage for per-portfolio Marketplace preference overrides."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def _normalize_id(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Identifier value cannot be empty")
    return text


class PresetPreferenceStore:
    """Stores user preference payloads and parameter overrides per preset."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser()
        self._entries: dict[str, dict[str, dict[str, object]]] = {}
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> None:
        if not self._path.exists():
            self._entries = {}
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._entries = {}
            return
        preferences = payload.get("preferences") if isinstance(payload, Mapping) else None
        normalized: dict[str, dict[str, dict[str, object]]] = {}
        if isinstance(preferences, Mapping):
            for preset_id, portfolio_map in preferences.items():
                if not isinstance(preset_id, str):
                    continue
                if not isinstance(portfolio_map, Mapping):
                    continue
                bucket: dict[str, dict[str, object]] = {}
                for portfolio_id, entry in portfolio_map.items():
                    if not isinstance(portfolio_id, str):
                        continue
                    if not isinstance(entry, Mapping):
                        continue
                    bucket[portfolio_id] = {
                        "preferences": (
                            dict(entry.get("preferences"))
                            if isinstance(entry.get("preferences"), Mapping)
                            else {}
                        ),
                        "overrides": (
                            {
                                str(strategy): {
                                    str(param): value
                                    for param, value in mapping.items()
                                    if isinstance(param, str)
                                }
                                for strategy, mapping in entry.get("overrides", {}).items()
                                if isinstance(strategy, str) and isinstance(mapping, Mapping)
                            }
                            if isinstance(entry.get("overrides"), Mapping)
                            else {}
                        ),
                    }
                if bucket:
                    normalized[preset_id] = bucket
        self._entries = normalized

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "preferences": {
                preset_id: {
                    portfolio_id: {
                        "preferences": dict(entry.get("preferences", {})),
                        "overrides": {
                            strategy: dict(params)
                            for strategy, params in (entry.get("overrides") or {}).items()
                        },
                    }
                    for portfolio_id, entry in portfolios.items()
                }
                for preset_id, portfolios in sorted(self._entries.items())
            }
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        self._path.write_text(serialized, encoding="utf-8")

    def set_entry(
        self,
        preset_id: str,
        portfolio_id: str,
        *,
        preferences: Mapping[str, object] | None = None,
        overrides: Mapping[str, Mapping[str, object]] | None = None,
    ) -> Mapping[str, object]:
        preset_key = _normalize_id(preset_id)
        portfolio_key = _normalize_id(portfolio_id)
        payload: dict[str, object] = {}
        if preferences:
            payload["preferences"] = {str(key): value for key, value in preferences.items()}
        else:
            payload["preferences"] = {}
        if overrides:
            normalized_overrides: dict[str, dict[str, object]] = {}
            for strategy, mapping in overrides.items():
                if not isinstance(mapping, Mapping):
                    continue
                normalized_overrides[strategy] = {
                    str(param): value for param, value in mapping.items()
                }
            payload["overrides"] = normalized_overrides
        else:
            payload["overrides"] = {}

        bucket = self._entries.setdefault(preset_key, {})
        bucket[portfolio_key] = payload
        self._save()
        return bucket[portfolio_key]

    def clear_entry(self, preset_id: str, portfolio_id: str) -> bool:
        preset_key = _normalize_id(preset_id)
        portfolio_key = _normalize_id(portfolio_id)
        portfolio_bucket = self._entries.get(preset_key)
        if not portfolio_bucket or portfolio_key not in portfolio_bucket:
            return False
        removed = portfolio_bucket.pop(portfolio_key, None) is not None
        if not portfolio_bucket:
            self._entries.pop(preset_key, None)
        self._save()
        return removed

    def entry(self, preset_id: str, portfolio_id: str) -> Mapping[str, object] | None:
        preset_key = _normalize_id(preset_id)
        portfolio_key = _normalize_id(portfolio_id)
        portfolio_bucket = self._entries.get(preset_key)
        if not portfolio_bucket:
            return None
        return portfolio_bucket.get(portfolio_key)

    def preferences_for(self, preset_id: str) -> Mapping[str, Mapping[str, object]]:
        preset_key = _normalize_id(preset_id)
        return dict(self._entries.get(preset_key, {}))

    def all_preferences(self) -> Mapping[str, Mapping[str, Mapping[str, object]]]:
        return {preset: dict(entries) for preset, entries in self._entries.items()}


__all__ = ["PresetPreferenceStore"]

