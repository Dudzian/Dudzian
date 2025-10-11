"""Presety profili ryzyka dla telemetrii UI i ich aplikacja.

Moduł udostępnia znormalizowane profile, które można stosować
zarówno w narzędziach CLI (`watch_metrics_stream`,
`verify_decision_log`), jak i wewnątrz runtime'u (`MetricsService`).
"""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, Mapping, MutableMapping

# --- Presety profili -------------------------------------------------------

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
        "metrics_service_overrides": {
            "ui_alerts_reduce_mode": "enable",
            "ui_alerts_overlay_mode": "enable",
            "ui_alerts_jank_mode": "enable",
            "ui_alerts_reduce_active_severity": "critical",
            "ui_alerts_reduce_recovered_severity": "notice",
            "ui_alerts_overlay_exceeded_severity": "critical",
            "ui_alerts_overlay_recovered_severity": "notice",
            "ui_alerts_overlay_critical_severity": "critical",
            "ui_alerts_overlay_critical_threshold": 1,
            "ui_alerts_jank_spike_severity": "warning",
            "ui_alerts_jank_critical_severity": "error",
            "ui_alerts_jank_critical_over_ms": 12.0,
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
        "metrics_service_overrides": {
            "ui_alerts_reduce_mode": "enable",
            "ui_alerts_overlay_mode": "enable",
            "ui_alerts_jank_mode": "enable",
            "ui_alerts_reduce_active_severity": "warning",
            "ui_alerts_reduce_recovered_severity": "info",
            "ui_alerts_overlay_exceeded_severity": "warning",
            "ui_alerts_overlay_recovered_severity": "info",
            "ui_alerts_overlay_critical_severity": "error",
            "ui_alerts_overlay_critical_threshold": 2,
            "ui_alerts_jank_spike_severity": "notice",
            "ui_alerts_jank_critical_severity": "warning",
            "ui_alerts_jank_critical_over_ms": 18.0,
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
        "metrics_service_overrides": {
            "ui_alerts_reduce_mode": "enable",
            "ui_alerts_overlay_mode": "enable",
            "ui_alerts_jank_mode": "enable",
            "ui_alerts_reduce_active_severity": "info",
            "ui_alerts_reduce_recovered_severity": "info",
            "ui_alerts_overlay_exceeded_severity": "info",
            "ui_alerts_overlay_recovered_severity": "info",
            "ui_alerts_overlay_critical_severity": "warning",
            "ui_alerts_overlay_critical_threshold": 3,
            "ui_alerts_jank_spike_severity": "info",
            "ui_alerts_jank_critical_severity": "warning",
            "ui_alerts_jank_critical_over_ms": 22.0,
        },
    },
    "manual": {},
}

# Mapowanie nazw opcji CLI na pola konfiguracji MetricsServiceConfig
_METRICS_CLI_TO_CONFIG: Mapping[str, str] = {
    "ui_alerts_reduce_mode": "reduce_motion_mode",
    "ui_alerts_overlay_mode": "overlay_alert_mode",
    "ui_alerts_jank_mode": "jank_alert_mode",
    "ui_alerts_reduce_active_severity": "reduce_motion_severity_active",
    "ui_alerts_reduce_recovered_severity": "reduce_motion_severity_recovered",
    "ui_alerts_overlay_exceeded_severity": "overlay_alert_severity_exceeded",
    "ui_alerts_overlay_recovered_severity": "overlay_alert_severity_recovered",
    "ui_alerts_overlay_critical_severity": "overlay_alert_severity_critical",
    "ui_alerts_overlay_critical_threshold": "overlay_alert_critical_threshold",
    "ui_alerts_jank_spike_severity": "jank_alert_severity_spike",
    "ui_alerts_jank_critical_severity": "jank_alert_severity_critical",
    "ui_alerts_jank_critical_over_ms": "jank_alert_critical_over_ms",
}

# Mapowanie nazw opcji CLI na odpowiadające zmienne środowiskowe narzędzi CLI.
_METRICS_CLI_TO_ENV: Mapping[str, str] = {
    "ui_alerts_reduce_mode": "RUN_METRICS_SERVICE_UI_ALERTS_REDUCE_MODE",
    "ui_alerts_overlay_mode": "RUN_METRICS_SERVICE_UI_ALERTS_OVERLAY_MODE",
    "ui_alerts_jank_mode": "RUN_METRICS_SERVICE_UI_ALERTS_JANK_MODE",
    "ui_alerts_reduce_category": "RUN_METRICS_SERVICE_UI_ALERTS_REDUCE_CATEGORY",
    "ui_alerts_reduce_active_severity": "RUN_METRICS_SERVICE_UI_ALERTS_REDUCE_ACTIVE_SEVERITY",
    "ui_alerts_reduce_recovered_severity": "RUN_METRICS_SERVICE_UI_ALERTS_REDUCE_RECOVERED_SEVERITY",
    "ui_alerts_overlay_category": "RUN_METRICS_SERVICE_UI_ALERTS_OVERLAY_CATEGORY",
    "ui_alerts_overlay_exceeded_severity": "RUN_METRICS_SERVICE_UI_ALERTS_OVERLAY_EXCEEDED_SEVERITY",
    "ui_alerts_overlay_recovered_severity": "RUN_METRICS_SERVICE_UI_ALERTS_OVERLAY_RECOVERED_SEVERITY",
    "ui_alerts_overlay_critical_severity": "RUN_METRICS_SERVICE_UI_ALERTS_OVERLAY_CRITICAL_SEVERITY",
    "ui_alerts_overlay_critical_threshold": "RUN_METRICS_SERVICE_UI_ALERTS_OVERLAY_CRITICAL_THRESHOLD",
    "ui_alerts_jank_category": "RUN_METRICS_SERVICE_UI_ALERTS_JANK_CATEGORY",
    "ui_alerts_jank_spike_severity": "RUN_METRICS_SERVICE_UI_ALERTS_JANK_SPIKE_SEVERITY",
    "ui_alerts_jank_critical_severity": "RUN_METRICS_SERVICE_UI_ALERTS_JANK_CRITICAL_SEVERITY",
    "ui_alerts_jank_critical_over_ms": "RUN_METRICS_SERVICE_UI_ALERTS_JANK_CRITICAL_OVER_MS",
    "ui_alerts_audit_dir": "RUN_METRICS_SERVICE_UI_ALERTS_AUDIT_DIR",
    "ui_alerts_audit_backend": "RUN_METRICS_SERVICE_UI_ALERTS_AUDIT_BACKEND",
    "ui_alerts_audit_pattern": "RUN_METRICS_SERVICE_UI_ALERTS_AUDIT_PATTERN",
    "ui_alerts_audit_retention_days": "RUN_METRICS_SERVICE_UI_ALERTS_AUDIT_RETENTION_DAYS",
    "ui_alerts_audit_fsync": "RUN_METRICS_SERVICE_UI_ALERTS_AUDIT_FSYNC",
}

# Przy zastosowaniu trybu "enable" włączamy odpowiadające flagi boolowskie.
_REQUIRED_FLAG_FIELDS: Mapping[str, tuple[str, Any]] = {
    "reduce_motion_mode": ("reduce_motion_alerts", True),
    "overlay_alert_mode": ("overlay_alerts", True),
    "jank_alert_mode": ("jank_alerts", True),
}

_DEFAULT_SENTINEL = object()
_MISSING_OVERRIDE = object()
_METRICS_DEFAULT_CACHE: dict[str, Any] | None = None

_PROFILE_STORE: dict[str, dict[str, Any]] = {}
_PROFILE_ORIGINS: dict[str, str] = {}

_SUPPORTED_SUFFIXES = {".json", ".yaml", ".yml"}


def _initialize_store() -> None:
    if _PROFILE_STORE:
        return
    for name, data in _RISK_PROFILE_PRESETS.items():
        normalized = name.strip().lower()
        _PROFILE_STORE[normalized] = deepcopy(data)
        _PROFILE_ORIGINS[normalized] = "builtin"


def list_risk_profile_names() -> list[str]:
    """Zwraca posortowaną listę dostępnych profili."""

    _initialize_store()
    return sorted(_PROFILE_STORE)


def _merge_profile_dicts(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Zwraca nowy słownik powstały z głębokiego połączenia struktur."""

    result = deepcopy(base)
    for key, value in overrides.items():
        if key == "extends":
            # Dziedziczenie obsługujemy osobno – tu pomijamy wskaźnik rodzica.
            continue
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge_profile_dicts(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def register_risk_profiles(
    profiles: Mapping[str, Mapping[str, Any]] | None,
    *,
    origin: str = "external",
) -> list[str]:
    """Rejestruje dodatkowe profile lub nadpisuje istniejące."""

    if not profiles:
        return []

    _initialize_store()

    normalized_profiles: dict[str, dict[str, Any]] = {}
    for name, profile in profiles.items():
        if name is None:
            continue
        normalized = str(name).strip().lower()
        if not normalized:
            continue
        if profile is None:
            normalized_profiles[normalized] = {}
            continue
        if not isinstance(profile, Mapping):
            raise ValueError(
                f"Profil ryzyka '{name}' musi być mapą, otrzymano {type(profile)!r}"
            )
        normalized_profiles[normalized] = deepcopy(dict(profile))

    resolved: dict[str, dict[str, Any]] = {}
    visiting: set[str] = set()

    def resolve(name: str) -> dict[str, Any]:
        if name in resolved:
            return resolved[name]
        if name in visiting:
            raise ValueError(
                f"Wykryto cykliczne dziedziczenie profili ryzyka przy '{name}'"
            )
        try:
            entry = normalized_profiles[name]
        except KeyError as exc:
            # Powinno być niemożliwe – resolve wywoływany jest tylko dla znanych nazw.
            raise KeyError(f"Nieznany profil do zarejestrowania: {name}") from exc

        visiting.add(name)
        extends_raw = entry.get("extends")
        if extends_raw:
            base_name = str(extends_raw).strip().lower()
            if not base_name:
                raise ValueError(
                    f"Profil ryzyka '{name}' posiada nieprawidłowe pole extends"
                )
            if base_name == name:
                raise ValueError(
                    f"Profil ryzyka '{name}' nie może dziedziczyć z samego siebie"
                )
            if base_name in normalized_profiles:
                base_profile = resolve(base_name)
            else:
                try:
                    base_profile = get_risk_profile(base_name)
                except KeyError as exc:
                    raise ValueError(
                        f"Profil ryzyka '{name}' dziedziczy z nieznanego profilu '{extends_raw}'"
                    ) from exc
            merged = _merge_profile_dicts(base_profile, entry)
            chain: list[str] = []
            base_chain = base_profile.get("extends_chain")
            if isinstance(base_chain, list):
                chain.extend(base_chain)
            elif base_profile.get("extends"):
                chain.append(str(base_profile["extends"]))
            chain.append(base_name)
            merged["extends"] = base_name
            if chain:
                merged["extends_chain"] = chain
        else:
            merged = deepcopy(entry)
            if "extends_chain" in merged and not isinstance(
                merged.get("extends_chain"), list
            ):
                merged["extends_chain"] = list(
                    merged.get("extends_chain") or []
                )

        visiting.remove(name)
        resolved[name] = merged
        return merged

    registered: list[str] = []
    for normalized in normalized_profiles:
        profile_data = resolve(normalized)
        _PROFILE_STORE[normalized] = deepcopy(profile_data)
        _PROFILE_ORIGINS[normalized] = origin
        registered.append(normalized)
    return registered


def _load_risk_profiles_from_single_file(
    source: Path, *, origin: str | None = None
) -> list[str]:
    if not source.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku profili ryzyka: {source}")
    if source.is_dir():
        raise IsADirectoryError(
            f"Ścieżka {source} wskazuje katalog – użyj load_risk_profiles_from_file do obsługi katalogów"
        )

    text = source.read_text(encoding="utf-8")
    suffix = source.suffix.lower()
    data: Any
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - brak PyYAML
            raise RuntimeError("Do wczytania pliku YAML wymagany jest pakiet PyYAML") from exc
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text or "{}")

    if isinstance(data, Mapping) and "risk_profiles" in data:
        data = data["risk_profiles"]
    if not isinstance(data, Mapping):
        raise ValueError("Plik profili ryzyka musi zawierać mapę risk_profiles")

    origin_label = f"file:{source}" if origin is None else origin
    return register_risk_profiles(data, origin=origin_label)


def list_risk_profile_files(directory: Path) -> list[Path]:
    """Zwraca posortowaną listę plików profili w katalogu."""

    files = [
        entry
        for entry in directory.iterdir()
        if entry.is_file() and entry.suffix.lower() in _SUPPORTED_SUFFIXES
    ]
    files.sort()
    return files


def load_risk_profiles_from_file(path: str | Path, *, origin: str | None = None) -> list[str]:
    """Ładuje profile z pliku lub katalogu JSON/YAML i rejestruje w store."""

    target = Path(path).expanduser()
    if target.is_dir():
        files = list_risk_profile_files(target)
        if not files:
            raise ValueError(
                f"Katalog {target} nie zawiera żadnych plików JSON/YAML z profilami ryzyka"
            )

        registered: list[str] = []
        base_origin = origin or f"dir:{target}"
        for entry in files:
            entry_origin = f"{base_origin}#{entry.name}" if base_origin else None
            registered.extend(
                _load_risk_profiles_from_single_file(entry, origin=entry_origin)
            )
        return registered

    return _load_risk_profiles_from_single_file(target, origin=origin)


def load_risk_profiles_with_metadata(
    path: str | Path, *, origin_label: str | None = None
) -> tuple[list[str], dict[str, Any]]:
    """Ładuje profile i zwraca listę nazw oraz metadane artefaktu."""

    target = Path(path).expanduser()
    if target.is_dir():
        files = list_risk_profile_files(target)
        if not files:
            raise ValueError(
                f"Katalog {target} nie zawiera żadnych plików JSON/YAML z profilami ryzyka"
            )

        registered: list[str] = []
        files_meta: list[dict[str, Any]] = []
        base_origin = origin_label or f"dir:{target}"
        for entry in files:
            entry_origin = f"{base_origin}#{entry.name}" if base_origin else None
            entry_registered = _load_risk_profiles_from_single_file(
                entry, origin=entry_origin
            )
            files_meta.append(
                {
                    "path": str(entry),
                    "registered_profiles": list(entry_registered),
                }
            )
            registered.extend(entry_registered)

        metadata: dict[str, Any] = {
            "path": str(target),
            "type": "directory",
            "files": files_meta,
            "registered_profiles": list(registered),
        }
        if origin_label:
            metadata["origin"] = origin_label
        return registered, metadata

    registered = _load_risk_profiles_from_single_file(
        target, origin=origin_label or f"file:{target}"
    )
    metadata = {
        "path": str(target),
        "type": "file",
        "registered_profiles": list(registered),
    }
    if origin_label:
        metadata["origin"] = origin_label
    return registered, metadata


def get_risk_profile(name: str) -> Mapping[str, Any]:
    """Zwraca kopię konfiguracji profilu."""

    _initialize_store()
    normalized = name.strip().lower()
    try:
        preset = _PROFILE_STORE[normalized]
    except KeyError as exc:  # pragma: no cover - defensywne
        raise KeyError(f"Nieznany profil ryzyka: {name!r}") from exc
    return deepcopy(preset)


def get_metrics_service_overrides(name: str) -> Mapping[str, Any]:
    """Zwraca mapowanie opcji CLI dla serwera MetricsService."""

    profile = get_risk_profile(name)
    return deepcopy(profile.get("metrics_service_overrides", {}))


def get_metrics_service_config_overrides(name: str) -> Mapping[str, Any]:
    """Zwraca nadpisania odpowiadające polom MetricsServiceConfig."""

    overrides = {}
    cli_overrides = get_metrics_service_overrides(name)
    for option, value in cli_overrides.items():
        target = _METRICS_CLI_TO_CONFIG.get(option)
        if target is not None:
            overrides[target] = value
    return overrides


def get_metrics_service_env_overrides(name: str) -> Mapping[str, Any]:
    """Zwraca mapowanie zmiennych środowiskowych dla presetów MetricsService."""

    env_overrides: dict[str, Any] = {}
    for option, value in get_metrics_service_overrides(name).items():
        env_name = _METRICS_CLI_TO_ENV.get(option)
        if env_name is not None:
            env_overrides[env_name] = value
    return env_overrides


def risk_profile_metadata(name: str) -> dict[str, Any]:
    """Buduje strukturę metadanych do logów i raportów."""

    profile = get_risk_profile(name)
    metadata = dict(profile)
    normalized = name.strip().lower()
    metadata["name"] = normalized
    origin = _PROFILE_ORIGINS.get(normalized)
    if origin:
        metadata.setdefault("origin", origin)
    return metadata


def summarize_risk_profile(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Zwraca skrócone podsumowanie profilu ryzyka dla raportów audytowych."""

    summary: dict[str, Any] = {}

    for key in (
        "name",
        "origin",
        "extends",
        "severity_min",
        "expect_summary_enabled",
        "require_screen_info",
    ):
        value = metadata.get(key)
        if value is not None:
            summary[key] = deepcopy(value)

    extends_chain = metadata.get("extends_chain")
    if extends_chain is None:
        if metadata.get("extends"):
            extends_chain = [metadata.get("extends")]  # type: ignore[list-item]
        else:
            extends_chain = []
    summary["extends_chain"] = deepcopy(extends_chain)

    for key in ("max_event_counts", "min_event_counts"):
        value = metadata.get(key)
        if value:
            summary[key] = deepcopy(value)

    return summary


def get_risk_profile_summary(name: str) -> dict[str, Any]:
    """Buduje podsumowanie profilu ryzyka na podstawie jego nazwy."""

    metadata = risk_profile_metadata(name)
    return summarize_risk_profile(metadata)


def _metrics_config_defaults() -> Mapping[str, Any]:
    global _METRICS_DEFAULT_CACHE
    if _METRICS_DEFAULT_CACHE is not None:
        return _METRICS_DEFAULT_CACHE

    try:
        from bot_core.config.models import MetricsServiceConfig  # type: ignore
    except Exception:  # pragma: no cover - starsze gałęzie bez konfiguracji
        _METRICS_DEFAULT_CACHE = {}
        return _METRICS_DEFAULT_CACHE

    defaults: dict[str, Any] = {}
    for field in fields(MetricsServiceConfig):
        default_value = _DEFAULT_SENTINEL
        if field.default is not MISSING:
            default_value = field.default
        elif field.default_factory is not MISSING:  # type: ignore[attr-defined]
            try:
                default_value = field.default_factory()  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensywne
                default_value = _DEFAULT_SENTINEL
        defaults[field.name] = default_value
    _METRICS_DEFAULT_CACHE = defaults
    return defaults


def _should_apply_override(current_value: Any, default_value: Any) -> bool:
    if current_value is None:
        return True
    if isinstance(current_value, str):
        if not current_value.strip():
            return True
    if default_value is not _DEFAULT_SENTINEL and current_value == default_value:
        return True
    return False


class MetricsRiskProfileResolver:
    """Pomocnik stosujący presety profilu do konfiguracji MetricsService."""

    def __init__(self, profile_name: str, config: Any | None = None) -> None:
        normalized = profile_name.strip().lower()
        self._profile_name = normalized
        self._config = config
        self._overrides_config = dict(get_metrics_service_config_overrides(normalized))
        self._metadata_base = risk_profile_metadata(normalized)
        self._summary = summarize_risk_profile(self._metadata_base)
        self._defaults = _metrics_config_defaults()
        self._applied: MutableMapping[str, dict[str, Any]] = {}
        self._skipped: MutableMapping[str, dict[str, Any]] = {}

    @property
    def profile_name(self) -> str:
        return self._profile_name

    def override(self, field_name: str, current_value: Any) -> Any:
        override_value = self._overrides_config.get(field_name, _MISSING_OVERRIDE)
        if override_value is _MISSING_OVERRIDE:
            return current_value

        if current_value == override_value:
            self._applied.setdefault(
                field_name,
                {"status": "already", "value": override_value},
            )
            return override_value

        default_value = self._defaults.get(field_name, _DEFAULT_SENTINEL)
        if _should_apply_override(current_value, default_value):
            self._applied[field_name] = {
                "status": "applied",
                "previous": current_value,
                "value": override_value,
            }
            self._set_on_config(field_name, override_value)
            self._propagate_flag(field_name, override_value)
            return override_value

        self._skipped[field_name] = {
            "reason": "explicit_value",
            "current": current_value,
            "override": override_value,
        }
        return current_value

    def metadata(self) -> dict[str, Any]:
        metadata = dict(self._metadata_base)
        if self._summary:
            metadata.setdefault("summary", dict(self._summary))
        if self._applied:
            metadata["applied_overrides"] = {
                name: entry.get("value")
                for name, entry in self._applied.items()
                if "value" in entry
            }
        if self._skipped:
            metadata["skipped_overrides"] = {
                name: {"current": entry.get("current"), "reason": entry.get("reason")}
                for name, entry in self._skipped.items()
            }
        return metadata

    # --- pomocnicze -------------------------------------------------------

    def _set_on_config(self, field_name: str, value: Any) -> None:
        if self._config is None:
            return
        if hasattr(self._config, field_name):
            try:
                setattr(self._config, field_name, value)
            except Exception:  # pragma: no cover - defensywne
                self._skipped[field_name] = {
                    "reason": "set_failed",
                    "current": getattr(self._config, field_name, None),
                    "override": value,
                }

    def _propagate_flag(self, field_name: str, override_value: Any) -> None:
        flag_info = _REQUIRED_FLAG_FIELDS.get(field_name)
        if not flag_info or self._config is None:
            return
        flag_name, expected_value = flag_info
        if override_value == "disable":
            return
        if not hasattr(self._config, flag_name):
            return
        current_flag = getattr(self._config, flag_name)
        if current_flag == expected_value:
            return
        try:
            setattr(self._config, flag_name, expected_value)
        except Exception:  # pragma: no cover - defensywne
            self._skipped[flag_name] = {
                "reason": "flag_set_failed",
                "current": current_flag,
                "override": expected_value,
            }
        else:
            self._applied.setdefault(
                flag_name,
                {"status": "applied", "previous": current_flag, "value": expected_value},
            )


__all__ = [
    "MetricsRiskProfileResolver",
    "get_metrics_service_config_overrides",
    "get_metrics_service_env_overrides",
    "get_metrics_service_overrides",
    "get_risk_profile",
    "load_risk_profiles_with_metadata",
    "list_risk_profile_names",
    "load_risk_profiles_from_file",
    "register_risk_profiles",
    "risk_profile_metadata",
    "summarize_risk_profile",
    "get_risk_profile_summary",
]
