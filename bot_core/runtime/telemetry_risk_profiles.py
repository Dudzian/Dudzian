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
            "ui_alerts_performance_mode": "enable",
            "ui_alerts_performance_category": "ui.performance",
            "ui_alerts_performance_warning_severity": "warning",
            "ui_alerts_performance_critical_severity": "critical",
            "ui_alerts_performance_recovered_severity": "notice",
            "ui_alerts_performance_event_to_frame_warning_ms": 45.0,
            "ui_alerts_performance_event_to_frame_critical_ms": 60.0,
            "ui_alerts_performance_cpu_warning_percent": 75.0,
            "ui_alerts_performance_cpu_critical_percent": 90.0,
            "ui_alerts_performance_gpu_warning_percent": 65.0,
            "ui_alerts_performance_gpu_critical_percent": 80.0,
            "ui_alerts_performance_ram_warning_megabytes": 4096.0,
            "ui_alerts_performance_ram_critical_megabytes": 6144.0,
        },
        "observability": {
            "scheduler_metrics": {
                "loop_latency_ms": {"warning": 120.0, "critical": 200.0},
                "max_drift_seconds": {"warning": 5, "critical": 10},
                "signals_per_run": {"warning": 12, "critical": 16},
            },
            "strategy_metrics": {
                "mean_reversion": {
                    "avg_abs_zscore": {"warning": 2.4, "critical": 2.9},
                    "avg_realized_volatility": {"warning": 0.06, "critical": 0.08},
                },
                "volatility_targeting": {
                    "allocation_error_pct": {"warning": 12.0, "critical": 18.0},
                    "realized_vs_target_vol_pct": {"warning": 25.0, "critical": 35.0},
                },
                "cross_exchange_arbitrage": {
                    "secondary_delay_ms": {"warning": 250.0, "critical": 400.0},
                    "spread_capture_bps": {"warning": 8.0, "critical": 4.0},
                },
            },
            "alert_policies": {
                "pnl_drawdown_pct": {"warning": 1.5, "critical": 2.5, "window": "15m"},
                "risk_exposure_deviation_pct": {"warning": 5.0, "critical": 7.5, "window": "10m"},
                "scheduler_latency_ms": {"warning": 180.0, "critical": 220.0, "window": "5m"},
            },
            "decision_log": {
                "required_fields": [
                    "schedule",
                    "strategy",
                    "confidence",
                    "latency_ms",
                    "risk_profile",
                    "telemetry_namespace",
                ],
                "hmac_required": True,
                "retention_days": 730,
            },
        },
        "data_quality": {
            "mean_reversion": {
                "expected_symbols": ["BTC_USDT", "ETH_USDT"],
                "max_gap_minutes": 15,
            },
            "volatility_targeting": {
                "expected_symbols": ["BTC_USDT"],
                "max_gap_minutes": 10,
            },
            "cross_exchange_arbitrage": {
                "expected_symbol_pairs": ["BTC_USDT@binance/binance_futures"],
                "max_delay_ms": 400,
            },
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
            "ui_alerts_performance_mode": "enable",
            "ui_alerts_performance_category": "ui.performance",
            "ui_alerts_performance_warning_severity": "warning",
            "ui_alerts_performance_critical_severity": "critical",
            "ui_alerts_performance_recovered_severity": "info",
            "ui_alerts_performance_event_to_frame_warning_ms": 55.0,
            "ui_alerts_performance_event_to_frame_critical_ms": 75.0,
            "ui_alerts_performance_cpu_warning_percent": 80.0,
            "ui_alerts_performance_cpu_critical_percent": 92.0,
            "ui_alerts_performance_gpu_warning_percent": 70.0,
            "ui_alerts_performance_gpu_critical_percent": 88.0,
            "ui_alerts_performance_ram_warning_megabytes": 6144.0,
            "ui_alerts_performance_ram_critical_megabytes": 9216.0,
        },
        "observability": {
            "scheduler_metrics": {
                "loop_latency_ms": {"warning": 150.0, "critical": 220.0},
                "max_drift_seconds": {"warning": 8, "critical": 14},
                "signals_per_run": {"warning": 18, "critical": 24},
            },
            "strategy_metrics": {
                "mean_reversion": {
                    "avg_abs_zscore": {"warning": 2.8, "critical": 3.3},
                    "avg_realized_volatility": {"warning": 0.07, "critical": 0.09},
                },
                "volatility_targeting": {
                    "allocation_error_pct": {"warning": 16.0, "critical": 22.0},
                    "realized_vs_target_vol_pct": {"warning": 30.0, "critical": 40.0},
                },
                "cross_exchange_arbitrage": {
                    "secondary_delay_ms": {"warning": 300.0, "critical": 450.0},
                    "spread_capture_bps": {"warning": 6.0, "critical": 3.0},
                },
            },
            "alert_policies": {
                "pnl_drawdown_pct": {"warning": 2.5, "critical": 4.0, "window": "30m"},
                "risk_exposure_deviation_pct": {"warning": 7.5, "critical": 10.0, "window": "15m"},
                "scheduler_latency_ms": {"warning": 220.0, "critical": 260.0, "window": "5m"},
            },
            "decision_log": {
                "required_fields": [
                    "schedule",
                    "strategy",
                    "confidence",
                    "latency_ms",
                    "risk_profile",
                    "telemetry_namespace",
                ],
                "hmac_required": True,
                "retention_days": 730,
            },
        },
        "data_quality": {
            "mean_reversion": {
                "expected_symbols": ["BTC_USDT", "ETH_USDT", "BNB_USDT"],
                "max_gap_minutes": 20,
            },
            "volatility_targeting": {
                "expected_symbols": ["BTC_USDT", "ETH_USDT"],
                "max_gap_minutes": 15,
            },
            "cross_exchange_arbitrage": {
                "expected_symbol_pairs": [
                    "BTC_USDT@binance/binance_futures",
                    "ETH_USDT@kraken/binance",
                ],
                "max_delay_ms": 450,
            },
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
            "ui_alerts_performance_mode": "enable",
            "ui_alerts_performance_category": "ui.performance",
            "ui_alerts_performance_warning_severity": "warning",
            "ui_alerts_performance_critical_severity": "critical",
            "ui_alerts_performance_recovered_severity": "info",
            "ui_alerts_performance_event_to_frame_warning_ms": 65.0,
            "ui_alerts_performance_event_to_frame_critical_ms": 90.0,
            "ui_alerts_performance_cpu_warning_percent": 85.0,
            "ui_alerts_performance_cpu_critical_percent": 96.0,
            "ui_alerts_performance_gpu_warning_percent": 80.0,
            "ui_alerts_performance_gpu_critical_percent": 96.0,
            "ui_alerts_performance_ram_warning_megabytes": 8192.0,
            "ui_alerts_performance_ram_critical_megabytes": 12288.0,
        },
        "observability": {
            "scheduler_metrics": {
                "loop_latency_ms": {"warning": 180.0, "critical": 260.0},
                "max_drift_seconds": {"warning": 10, "critical": 18},
                "signals_per_run": {"warning": 24, "critical": 32},
            },
            "strategy_metrics": {
                "mean_reversion": {
                    "avg_abs_zscore": {"warning": 3.1, "critical": 3.8},
                    "avg_realized_volatility": {"warning": 0.08, "critical": 0.1},
                },
                "volatility_targeting": {
                    "allocation_error_pct": {"warning": 20.0, "critical": 28.0},
                    "realized_vs_target_vol_pct": {"warning": 40.0, "critical": 55.0},
                },
                "cross_exchange_arbitrage": {
                    "secondary_delay_ms": {"warning": 350.0, "critical": 500.0},
                    "spread_capture_bps": {"warning": 5.0, "critical": 2.5},
                },
            },
            "alert_policies": {
                "pnl_drawdown_pct": {"warning": 3.0, "critical": 5.0, "window": "30m"},
                "risk_exposure_deviation_pct": {"warning": 10.0, "critical": 15.0, "window": "15m"},
                "scheduler_latency_ms": {"warning": 250.0, "critical": 320.0, "window": "5m"},
            },
            "decision_log": {
                "required_fields": [
                    "schedule",
                    "strategy",
                    "confidence",
                    "latency_ms",
                    "risk_profile",
                    "telemetry_namespace",
                ],
                "hmac_required": True,
                "retention_days": 730,
            },
        },
        "data_quality": {
            "mean_reversion": {
                "expected_symbols": ["BTC_USDT", "ETH_USDT", "SOL_USDT"],
                "max_gap_minutes": 25,
            },
            "volatility_targeting": {
                "expected_symbols": ["BTC_USDT", "ETH_USDT", "SOL_USDT"],
                "max_gap_minutes": 20,
            },
            "cross_exchange_arbitrage": {
                "expected_symbol_pairs": [
                    "BTC_USDT@binance/binance_futures",
                    "ETH_USDT@kraken/binance",
                    "SOL_USDT@binance/okx",
                ],
                "max_delay_ms": 520,
            },
        },
    },
    "manual": {
        "observability": {
            "scheduler_metrics": {},
            "strategy_metrics": {},
            "alert_policies": {},
            "decision_log": {
                "required_fields": ["schedule", "strategy", "risk_profile"],
                "hmac_required": False,
            },
        },
        "data_quality": {},
    },
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
    "ui_alerts_performance_mode": "performance_alert_mode",
    "ui_alerts_performance_category": "performance_category",
    "ui_alerts_performance_warning_severity": "performance_severity_warning",
    "ui_alerts_performance_critical_severity": "performance_severity_critical",
    "ui_alerts_performance_recovered_severity": "performance_severity_recovered",
    "ui_alerts_performance_event_to_frame_warning_ms": "performance_event_to_frame_warning_ms",
    "ui_alerts_performance_event_to_frame_critical_ms": "performance_event_to_frame_critical_ms",
    "ui_alerts_performance_cpu_warning_percent": "cpu_utilization_warning_percent",
    "ui_alerts_performance_cpu_critical_percent": "cpu_utilization_critical_percent",
    "ui_alerts_performance_gpu_warning_percent": "gpu_utilization_warning_percent",
    "ui_alerts_performance_gpu_critical_percent": "gpu_utilization_critical_percent",
    "ui_alerts_performance_ram_warning_megabytes": "ram_usage_warning_megabytes",
    "ui_alerts_performance_ram_critical_megabytes": "ram_usage_critical_megabytes",
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
    "ui_alerts_performance_mode": "RUN_METRICS_SERVICE_UI_ALERTS_PERFORMANCE_MODE",
    "ui_alerts_performance_category": "RUN_METRICS_SERVICE_UI_ALERTS_PERFORMANCE_CATEGORY",
    "ui_alerts_performance_warning_severity": "RUN_METRICS_SERVICE_UI_ALERTS_PERFORMANCE_WARNING_SEVERITY",
    "ui_alerts_performance_critical_severity": "RUN_METRICS_SERVICE_UI_ALERTS_PERFORMANCE_CRITICAL_SEVERITY",
    "ui_alerts_performance_recovered_severity": "RUN_METRICS_SERVICE_UI_ALERTS_PERFORMANCE_RECOVERED_SEVERITY",
    "ui_alerts_performance_event_to_frame_warning_ms": "RUN_METRICS_SERVICE_UI_ALERTS_PERFORMANCE_EVENT_TO_FRAME_WARNING_MS",
    "ui_alerts_performance_event_to_frame_critical_ms": "RUN_METRICS_SERVICE_UI_ALERTS_PERFORMANCE_EVENT_TO_FRAME_CRITICAL_MS",
    "ui_alerts_performance_cpu_warning_percent": "RUN_METRICS_SERVICE_UI_ALERTS_PERFORMANCE_CPU_WARNING_PERCENT",
    "ui_alerts_performance_cpu_critical_percent": "RUN_METRICS_SERVICE_UI_ALERTS_PERFORMANCE_CPU_CRITICAL_PERCENT",
    "ui_alerts_performance_gpu_warning_percent": "RUN_METRICS_SERVICE_UI_ALERTS_PERFORMANCE_GPU_WARNING_PERCENT",
    "ui_alerts_performance_gpu_critical_percent": "RUN_METRICS_SERVICE_UI_ALERTS_PERFORMANCE_GPU_CRITICAL_PERCENT",
    "ui_alerts_performance_ram_warning_megabytes": "RUN_METRICS_SERVICE_UI_ALERTS_PERFORMANCE_RAM_WARNING_MEGABYTES",
    "ui_alerts_performance_ram_critical_megabytes": "RUN_METRICS_SERVICE_UI_ALERTS_PERFORMANCE_RAM_CRITICAL_MEGABYTES",
}

# Przy zastosowaniu trybu "enable" włączamy odpowiadające flagi boolowskie.
_REQUIRED_FLAG_FIELDS: Mapping[str, tuple[str, Any]] = {
    "reduce_motion_mode": ("reduce_motion_alerts", True),
    "overlay_alert_mode": ("overlay_alerts", True),
    "jank_alert_mode": ("jank_alerts", True),
    "performance_alert_mode": ("performance_alerts", True),
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


def reset_risk_profile_store() -> None:
    """Przywraca wbudowane presety profili ryzyka."""

    _PROFILE_STORE.clear()
    _PROFILE_ORIGINS.clear()
    _initialize_store()


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
            base_profile = _PROFILE_STORE.get(name)
            if base_profile:
                merged = _merge_profile_dicts(base_profile, entry)
                if "extends" not in merged and base_profile.get("extends"):
                    merged["extends"] = base_profile.get("extends")
                base_chain = base_profile.get("extends_chain")
                if (
                    base_chain
                    and "extends_chain" not in merged
                    and isinstance(base_chain, list)
                ):
                    merged["extends_chain"] = list(base_chain)
            else:
                merged = deepcopy(entry)
                if "extends_chain" in merged and not isinstance(
                    merged.get("extends_chain"), list
                ):
                    merged["extends_chain"] = list(merged.get("extends_chain") or [])

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
            registered.extend(_load_risk_profiles_from_single_file(entry, origin=entry_origin))
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
            entry_registered = _load_risk_profiles_from_single_file(entry, origin=entry_origin)
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

    registered = _load_risk_profiles_from_single_file(target, origin=origin_label or f"file:{target}")
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


def get_scheduler_metrics(name: str) -> Mapping[str, Any]:
    """Zwraca progi metryk scheduler-a powiązane z profilem."""

    profile = get_risk_profile(name)
    observability = profile.get("observability", {})
    scheduler_metrics = observability.get("scheduler_metrics", {})
    return deepcopy(scheduler_metrics)


def get_strategy_metrics(name: str) -> Mapping[str, Any]:
    """Zwraca mapę progów metryk strategii dla profilu."""

    profile = get_risk_profile(name)
    observability = profile.get("observability", {})
    strategy_metrics = observability.get("strategy_metrics", {})
    return deepcopy(strategy_metrics)


def get_alert_policies(name: str) -> Mapping[str, Any]:
    """Zwraca polityki alertów operacyjnych przypisane do profilu."""

    profile = get_risk_profile(name)
    observability = profile.get("observability", {})
    policies = observability.get("alert_policies", {})
    return deepcopy(policies)


def get_observability_composite_expectations(name: str) -> Mapping[str, Any]:
    """Zwraca oczekiwane kompozyty SLO2 dla profilu obserwowalności."""

    profile = get_risk_profile(name)
    observability = profile.get("observability", {})

    scheduler_metrics: list[str] = []
    strategy_metrics: dict[str, list[str]] = {}
    alert_policies: list[str] = []
    composites: list[str] = []

    if isinstance(observability, Mapping):
        scheduler_section = observability.get("scheduler_metrics", {})
        if isinstance(scheduler_section, Mapping):
            scheduler_metrics = sorted(str(metric) for metric in scheduler_section.keys())
            composites.extend(f"scheduler::{metric}" for metric in scheduler_metrics)

        strategy_section = observability.get("strategy_metrics", {})
        if isinstance(strategy_section, Mapping):
            for strategy_name, metrics in strategy_section.items():
                if not isinstance(metrics, Mapping):
                    continue
                names = sorted(str(metric_name) for metric_name in metrics.keys())
                normalized_name = str(strategy_name)
                strategy_metrics[normalized_name] = names
                composites.extend(
                    f"strategy::{normalized_name}.{metric_name}" for metric_name in names
                )

        alerts_section = observability.get("alert_policies", {})
        if isinstance(alerts_section, Mapping):
            alert_policies = sorted(str(name) for name in alerts_section.keys())
            composites.extend(f"alert::{policy}" for policy in alert_policies)

    deduplicated = sorted(dict.fromkeys(composites))
    return {
        "scheduler": scheduler_metrics,
        "strategy": strategy_metrics,
        "alert_policies": alert_policies,
        "composites": deduplicated,
    }


def get_decision_log_requirements(name: str) -> Mapping[str, Any]:
    """Zwraca wymagania dot. decision logu dla profilu ryzyka."""

    profile = get_risk_profile(name)
    observability = profile.get("observability", {})
    requirements = observability.get("decision_log", {})
    return deepcopy(requirements)


def get_data_quality_expectations(name: str) -> Mapping[str, Any]:
    """Zwraca oczekiwania jakości danych powiązane z profilem."""

    profile = get_risk_profile(name)
    expectations = profile.get("data_quality", {})
    return deepcopy(expectations)


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

    observability = metadata.get("observability")
    if isinstance(observability, Mapping):
        scheduler_metrics = observability.get("scheduler_metrics")
        if isinstance(scheduler_metrics, Mapping):
            summary["scheduler_metrics"] = sorted(scheduler_metrics.keys())
        strategy_metrics = observability.get("strategy_metrics")
        if isinstance(strategy_metrics, Mapping):
            summary["strategy_metrics"] = {
                str(name): sorted(metrics.keys())
                for name, metrics in strategy_metrics.items()
                if isinstance(metrics, Mapping)
            }
        alert_policies = observability.get("alert_policies")
        if isinstance(alert_policies, Mapping):
            summary["alert_policies"] = sorted(alert_policies.keys())
        decision_requirements = observability.get("decision_log")
        if isinstance(decision_requirements, Mapping):
            summary["decision_log"] = {
                "required_fields": list(decision_requirements.get("required_fields", [])),
                "hmac_required": bool(decision_requirements.get("hmac_required", False)),
            }

    data_quality = metadata.get("data_quality")
    if isinstance(data_quality, Mapping):
        summary["data_quality"] = {
            str(name): {
                key: deepcopy(value)
                for key, value in payload.items()
                if key in {"max_gap_minutes", "max_delay_ms", "expected_symbols", "expected_symbol_pairs"}
            }
            for name, payload in data_quality.items()
            if isinstance(payload, Mapping)
        }

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
    "get_scheduler_metrics",
    "get_strategy_metrics",
    "get_alert_policies",
    "get_observability_composite_expectations",
    "get_decision_log_requirements",
    "get_data_quality_expectations",
    "load_risk_profiles_with_metadata",
    "list_risk_profile_names",
    "load_risk_profiles_from_file",
    "register_risk_profiles",
    "risk_profile_metadata",
    "summarize_risk_profile",
    "get_risk_profile_summary",
]
