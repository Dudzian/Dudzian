"""Wspólne presety profili ryzyka dla telemetrii UI.

Moduł udostępnia niewielkie API pozwalające na ponowne wykorzystanie
konfiguracji profili ryzyka zarówno w watcherze (`watch_metrics_stream`),
jak i w weryfikatorze decision logów (`verify_decision_log`).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


# Zestaw predefiniowanych profili ryzyka.  Klucze to nazwy dostępne w CLI.
_RISK_PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "conservative": {
        "expect_summary_enabled": True,
        "require_screen_info": True,
        "severity_min": "warning",
        "max_event_counts": {
            "overlay_budget": 0,
            "jank": 0,
            "reduce_motion": 3,
        },
        "min_event_counts": {
            "reduce_motion": 1,
        },
    },
    "balanced": {
        "expect_summary_enabled": True,
        "require_screen_info": True,
        "severity_min": "notice",
        "max_event_counts": {
            "overlay_budget": 2,
            "jank": 1,
            "reduce_motion": 5,
        },
    },
    "aggressive": {
        "expect_summary_enabled": True,
        "require_screen_info": True,
        "severity_min": "info",
        "max_event_counts": {
            "overlay_budget": 4,
            "jank": 2,
            "reduce_motion": 8,
        },
    },
    "manual": {},
}


def list_risk_profile_names() -> list[str]:
    """Zwraca posortowaną listę dostępnych profili ryzyka."""

    return sorted(_RISK_PROFILE_PRESETS)


def get_risk_profile(name: str) -> Mapping[str, Any]:
    """Zwraca kopię konfiguracji profilu o podanej nazwie.

    Raises:
        KeyError: jeśli profil nie istnieje.
    """

    normalized = name.strip().lower()
    try:
        preset = _RISK_PROFILE_PRESETS[normalized]
    except KeyError as exc:  # pragma: no cover - defensywne sprawdzenie
        raise KeyError(f"Nieznany profil ryzyka: {name!r}") from exc
    return deepcopy(preset)


def risk_profile_metadata(name: str) -> dict[str, Any]:
    """Buduje strukturę metadanych dla decision logów i raportów."""

    config = get_risk_profile(name)
    config_with_name = dict(config)
    config_with_name["name"] = name.strip().lower()
    return config_with_name
