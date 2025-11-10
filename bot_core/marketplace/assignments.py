"""Persistent storage for Marketplace preset to portfolio assignments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


class PresetAssignmentStore:
    """Manages preset to portfolio mappings persisted as JSON."""

    def __init__(self, root: str | Path) -> None:
        self._path = Path(root).expanduser()
        self._data: dict[str, set[str]] = {}
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> None:
        if not self._path.exists():
            self._data = {}
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._data = {}
            return
        assignments = payload.get("assignments") if isinstance(payload, Mapping) else None
        normalized: dict[str, set[str]] = {}
        if isinstance(assignments, Mapping):
            for preset_id, portfolios in assignments.items():
                if not isinstance(preset_id, str):
                    continue
                bucket: set[str] = set()
                if isinstance(portfolios, list):
                    for portfolio in portfolios:
                        if isinstance(portfolio, str) and portfolio.strip():
                            bucket.add(portfolio.strip())
                normalized[preset_id] = bucket
        self._data = normalized

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "assignments": {
                preset_id: sorted(portfolios)
                for preset_id, portfolios in sorted(self._data.items())
            }
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        self._path.write_text(serialized, encoding="utf-8")

    def assigned_portfolios(self, preset_id: str) -> tuple[str, ...]:
        return tuple(sorted(self._data.get(preset_id, set())))

    def all_assignments(self) -> Mapping[str, tuple[str, ...]]:
        return {key: tuple(sorted(value)) for key, value in self._data.items()}

    def assign(self, preset_id: str, portfolio_id: str) -> tuple[str, ...]:
        preset = preset_id.strip()
        portfolio = portfolio_id.strip()
        if not preset or not portfolio:
            return self.assigned_portfolios(preset_id)
        bucket = self._data.setdefault(preset, set())
        bucket.add(portfolio)
        self._save()
        return self.assigned_portfolios(preset)

    def unassign(self, preset_id: str, portfolio_id: str) -> tuple[str, ...]:
        preset = preset_id.strip()
        portfolio = portfolio_id.strip()
        if not preset or not portfolio:
            return self.assigned_portfolios(preset_id)
        bucket = self._data.get(preset)
        if bucket and portfolio in bucket:
            bucket.remove(portfolio)
            if not bucket:
                self._data.pop(preset, None)
            self._save()
        return self.assigned_portfolios(preset)


__all__ = ["PresetAssignmentStore"]

