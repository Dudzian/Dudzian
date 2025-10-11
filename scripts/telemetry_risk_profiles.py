"""Wspólne presety profili ryzyka dla telemetrii UI (z fallbackiem).

- Jeśli dostępny jest moduł `bot_core.runtime.telemetry_risk_profiles`, re-eksportujemy jego API.
- W przeciwnym razie używamy lokalnych presetów i minimalnych implementacji, tak aby
  `run_trading_stub_server` mógł korzystać z:
    * get_metrics_service_overrides(profile_name)
    * list_risk_profile_names()
    * load_risk_profiles_with_metadata(path, origin_label=...)
    * risk_profile_metadata(name)
  oraz pokrewnych funkcji pomocniczych.

Moduł może być współdzielony przez watcher (`watch_metrics_stream`) i weryfikator
decision logów (`verify_decision_log`).
"""

from __future__ import annotations

# --- PRÓBA UŻYCIA PEŁNEGO API Z bot_core -------------------------------------
try:  # pragma: no cover
    from bot_core.runtime.telemetry_risk_profiles import (  # type: ignore
        MetricsRiskProfileResolver,
        get_metrics_service_config_overrides,
        get_metrics_service_overrides,
        get_risk_profile,
        load_risk_profiles_with_metadata,
        list_risk_profile_names,
        load_risk_profiles_from_file,
        register_risk_profiles,
        risk_profile_metadata,
    )

    __all__ = [
        "MetricsRiskProfileResolver",
        "get_metrics_service_config_overrides",
        "get_metrics_service_overrides",
        "get_risk_profile",
        "load_risk_profiles_with_metadata",
        "list_risk_profile_names",
        "load_risk_profiles_from_file",
        "register_risk_profiles",
        "risk_profile_metadata",
    ]
except Exception:  # pragma: no cover - fallback lokalny
    import json
    import os
    from copy import deepcopy
    from pathlib import Path
    from typing import Any, Iterable, Mapping, Tuple

    # --- PRESETY LOKALNE -----------------------------------------------------
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
            "min_event_counts": {"reduce_motion": 1},
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

    # --- API PUBLICZNE (fallback) --------------------------------------------
    def list_risk_profile_names() -> list[str]:
        """Zwraca posortowaną listę dostępnych profili ryzyka."""
        return sorted(_RISK_PROFILE_PRESETS)

    def get_risk_profile(name: str) -> Mapping[str, Any]:
        """Zwraca kopię konfiguracji profilu o podanej nazwie."""
        normalized = name.strip().lower()
        try:
            preset = _RISK_PROFILE_PRESETS[normalized]
        except KeyError as exc:
            raise KeyError(f"Nieznany profil ryzyka: {name!r}") from exc
        return deepcopy(preset)

    def risk_profile_metadata(name: str) -> dict[str, Any]:
        """Buduje metadane profilu (do decision logów/raportów)."""
        config = get_risk_profile(name)
        out = dict(config)
        out["name"] = name.strip().lower()
        return out

    def register_risk_profiles(profiles: Mapping[str, Mapping[str, Any]] | Iterable[tuple[str, Mapping[str, Any]]]) -> list[str]:
        """Rejestruje/aktualizuje profile ryzyka. Zwraca listę zarejestrowanych nazw."""
        registered: list[str] = []
        if isinstance(profiles, Mapping):
            items = profiles.items()
        else:
            items = list(profiles)

        for name, cfg in items:
            normalized = name.strip().lower()
            _RISK_PROFILE_PRESETS[normalized] = deepcopy(dict(cfg))
            registered.append(normalized)
        return registered

    def _detect_format_from_suffix(path: Path) -> str:
        suf = path.suffix.lower()
        if suf in {".yml", ".yaml"}:
            return "yaml"
        return "json"

    def load_risk_profiles_from_file(path: Path | str) -> dict[str, dict[str, Any]]:
        """Ładuje profile z pliku JSON/YAML. Akceptuje formaty:
        - dict { name: { ...config... }, ... }
        - dict { "profiles": { name: { ... }, ... } }
        - list [{"name": "x", ...config...}, ...]
        """
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Nie znaleziono pliku profili ryzyka: {p}")

        fmt = _detect_format_from_suffix(p)
        text = p.read_text(encoding="utf-8")

        data: Any
        if fmt == "yaml":
            try:
                import yaml  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "Do wczytania profili w YAML wymagany jest PyYAML (pip install pyyaml)."
                ) from exc
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)

        profiles: dict[str, dict[str, Any]] = {}
        if isinstance(data, dict) and "profiles" in data and isinstance(data["profiles"], dict):
            for k, v in data["profiles"].items():
                profiles[str(k).strip().lower()] = dict(v or {})
        elif isinstance(data, dict):
            for k, v in data.items():
                profiles[str(k).strip().lower()] = dict(v or {})
        elif isinstance(data, list):
            for item in data:
                if not isinstance(item, dict) or "name" not in item:
                    continue
                name = str(item["name"]).strip().lower()
                cfg = {k: v for k, v in item.items() if k != "name"}
                profiles[name] = cfg
        else:
            raise ValueError("Nieobsługiwany format pliku profili ryzyka.")

        return profiles

    def load_risk_profiles_with_metadata(path: Path | str, *, origin_label: str | None = None) -> Tuple[list[str], Mapping[str, Any]]:
        """Ładuje profile z pliku, rejestruje i zwraca (registered_names, metadata)."""
        p = Path(path).expanduser()
        profiles = load_risk_profiles_from_file(p)
        registered = register_risk_profiles(profiles)

        stat = p.stat()
        metadata = {
            "path": str(p),
            "exists": True,
            "bytes": stat.st_size,
            "mtime": stat.st_mtime,
            "format": _detect_format_from_suffix(p),
        }
        if origin_label:
            metadata["origin"] = origin_label
        return registered, metadata

    def _severity_for(profile: Mapping[str, Any]) -> str | None:
        return str(profile.get("severity_min")).lower() if "severity_min" in profile else None

    def get_metrics_service_overrides(profile_name: str) -> dict[str, Any]:
        """Mapuje profil ryzyka na override’y opcji `metrics_*` (bez prefiksu)."""
        profile = get_risk_profile(profile_name)
        severity = _severity_for(profile) or "warning"
        max_counts = dict(profile.get("max_event_counts", {}))

        # Heurystyki mapowania:
        overlay_threshold = max(0, int(max_counts.get("overlay_budget", 0))) if isinstance(max_counts.get("overlay_budget", 0), (int, float)) else 0

        overrides: dict[str, Any] = {
            # tryby alertów
            "ui_alerts_reduce_mode": "enable",
            "ui_alerts_overlay_mode": "enable",
            "ui_alerts_jank_mode": "enable",
            # severities wynikające z minimalnej akceptowalnej głośności
            "ui_alerts_reduce_active_severity": severity,
            "ui_alerts_overlay_exceeded_severity": severity,
            "ui_alerts_jank_spike_severity": severity,
            # progi
            "ui_alerts_overlay_critical_threshold": overlay_threshold,
        }

        # Jeśli profil bardzo restrykcyjny dla jank (0), zdefiniuj krytyczne severity.
        jank_cap = max_counts.get("jank", None)
        if isinstance(jank_cap, (int, float)) and jank_cap <= 0:
            overrides["ui_alerts_jank_critical_severity"] = "critical"

        return overrides

    # w niektórych miejscach API może oczekiwać tej nazwy
    def get_metrics_service_config_overrides(profile_name: str) -> dict[str, Any]:
        return get_metrics_service_overrides(profile_name)

    class MetricsRiskProfileResolver:
        """Minimalny resolver profili ryzyka."""
        def __init__(self) -> None:
            self._names = set(list_risk_profile_names())

        def names(self) -> list[str]:
            return list_risk_profile_names()

        def resolve_overrides(self, name: str) -> dict[str, Any]:
            return get_metrics_service_overrides(name)

        def register_file(self, path: Path | str) -> list[str]:
            regs, _meta = load_risk_profiles_with_metadata(path, origin_label=f"resolver:{path}")
            self._names.update(regs)
            return regs

    __all__ = [
        "MetricsRiskProfileResolver",
        "get_metrics_service_config_overrides",
        "get_metrics_service_overrides",
        "get_risk_profile",
        "load_risk_profiles_with_metadata",
        "list_risk_profile_names",
        "load_risk_profiles_from_file",
        "register_risk_profiles",
        "risk_profile_metadata",
    ]
