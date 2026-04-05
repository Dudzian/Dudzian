"""Jawne profile bot modes mapowane na istniejące elementy runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Mapping

_REQUIRED_FIELDS = ("id", "label", "execution_mode", "description")
_EXPECTED_BUILTIN_FILES = (
    "signal_grid.json",
    "paper_monitoring.json",
    "rule_auto_router.json",
)


@dataclass(frozen=True)
class BotModeProfile:
    id: str
    label: str
    strategy_engine: str | None
    execution_mode: str
    description: str
    strategy_params: Mapping[str, Any]


def _load_payloads_from_directory(config_dir: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(config_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Profil bot mode musi być obiektem JSON: {path}")
        payloads.append(payload)
    return payloads


def _load_payloads_from_builtin_resources() -> list[dict[str, Any]]:
    profiles_dir = resources.files("bot_core.product").joinpath("profiles")
    names = sorted(entry.name for entry in profiles_dir.iterdir() if entry.name.endswith(".json"))
    missing = sorted(set(_EXPECTED_BUILTIN_FILES).difference(names))
    if not names or missing:
        missing_text = ", ".join(missing) if missing else "brak plików"
        raise RuntimeError(
            f"Brak kompletnego zestawu wbudowanych bot modes w zasobach pakietu: {missing_text}"
        )

    payloads: list[dict[str, Any]] = []
    for name in names:
        payload = json.loads(profiles_dir.joinpath(name).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Wbudowany profil bot mode musi być obiektem JSON: {name}")
        payloads.append(payload)
    return payloads


def _coerce_profiles(payloads: Iterable[dict[str, Any]]) -> tuple[BotModeProfile, ...]:
    profiles: list[BotModeProfile] = []
    seen_ids: set[str] = set()

    for payload in payloads:
        missing_fields = [field for field in _REQUIRED_FIELDS if not payload.get(field)]
        if missing_fields:
            raise ValueError(
                "Profil bot mode ma braki w polach obowiązkowych: " + ", ".join(missing_fields)
            )

        profile_id = str(payload["id"])
        if profile_id in seen_ids:
            raise ValueError(f"Zduplikowany identyfikator profilu bot mode: {profile_id}")
        seen_ids.add(profile_id)

        profiles.append(
            BotModeProfile(
                id=profile_id,
                label=str(payload["label"]),
                strategy_engine=(
                    str(payload.get("strategy_engine"))
                    if payload.get("strategy_engine") is not None
                    else None
                ),
                execution_mode=str(payload["execution_mode"]),
                description=str(payload["description"]),
                strategy_params=dict(payload.get("strategy_params") or {}),
            )
        )
    return tuple(profiles)


def load_bot_mode_profiles(config_dir: Path | None = None) -> tuple[BotModeProfile, ...]:
    payloads = (
        _load_payloads_from_directory(config_dir)
        if config_dir is not None
        else _load_payloads_from_builtin_resources()
    )
    return _coerce_profiles(payloads)


def builtin_bot_modes() -> tuple[BotModeProfile, ...]:
    return load_bot_mode_profiles()
