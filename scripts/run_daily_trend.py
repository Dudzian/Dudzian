"""CLI do uruchamiania pipeline'u strategii Daily Trend w trybie paper/testnet."""
from __future__ import annotations

import argparse
import hashlib
import json
import importlib
import os
import logging
import re
import signal
import subprocess
import sys
import shutil
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import deque
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Callable, TYPE_CHECKING, Type


_RAW_OUTPUT_MAX_LEN = 4096
_CONTEXT_SNIPPET_MAX_LEN = 240

from bot_core.alerts import AlertMessage
from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeAdapterFactory,
    ExchangeCredentials,
    OrderResult,
)
from bot_core.data.intervals import (
    interval_to_milliseconds as _interval_to_milliseconds,
    normalize_interval_token as _normalize_interval_token,
)
from bot_core.config.loader import load_core_config
from bot_core.reporting.audit import (
    PaperSmokeJsonSynchronizer,
    PaperSmokeJsonSyncResult,
)
from bot_core.reporting.environment_storage import store_environment_report
from bot_core.reporting.upload import (
    SmokeArchiveUploader,
    SmokeArchiveUploadResult,
)
from bot_core.data.ohlcv import evaluate_coverage
from bot_core.security import SecretManager, SecretStorageError, create_default_secret_storage
from bot_core.runtime.file_metadata import (
    directory_metadata as _directory_metadata,
    file_reference_metadata as _file_reference_metadata,
    log_security_warnings as _log_security_warnings,
    permissions_from_mode as _permissions_from_mode,
    security_flags_from_mode as _security_flags_from_mode,
)
from scripts import paper_precheck as paper_precheck_cli

if TYPE_CHECKING:
    from bot_core.config.models import CoreConfig, EnvironmentConfig, RiskProfileConfig
    from bot_core.runtime.pipeline import (
        build_daily_trend_pipeline as _BuildDailyTrendPipelineProto,
        create_trading_controller as _CreateTradingControllerProto,
    )
    from bot_core.runtime.realtime import (
        DailyTrendRealtimeRunner as _DailyTrendRealtimeRunnerProto,
    )
else:  # pragma: no cover - wykorzystywane tylko w czasie wykonywania
    _BuildDailyTrendPipelineProto = Callable[..., Any]
    _CreateTradingControllerProto = Callable[..., Any]
    _DailyTrendRealtimeRunnerProto = Type[Any]


build_daily_trend_pipeline: _BuildDailyTrendPipelineProto | None
create_trading_controller: _CreateTradingControllerProto | None
DailyTrendRealtimeRunner: _DailyTrendRealtimeRunnerProto | None


try:
    from bot_core.runtime.pipeline import (
        build_daily_trend_pipeline,
        create_trading_controller,
    )
except (ModuleNotFoundError, ImportError):  # pragma: no cover - zależy od instalacji
    build_daily_trend_pipeline = None  # type: ignore[assignment]
    create_trading_controller = None  # type: ignore[assignment]

try:
    from bot_core.runtime.realtime import (
        DailyTrendRealtimeRunner,
    )
except (ModuleNotFoundError, ImportError):  # pragma: no cover - zależy od instalacji
    DailyTrendRealtimeRunner = None  # type: ignore[assignment]

try:  # pragma: no cover - opcjonalny moduł telemetrii UI
    from bot_core.runtime.metrics_alerts import DEFAULT_UI_ALERTS_JSONL_PATH
    _UI_TELEMETRY_ALERT_SINK_AVAILABLE = True
except Exception:  # pragma: no cover - brak telemetrii UI
    DEFAULT_UI_ALERTS_JSONL_PATH = Path("logs/ui_telemetry_alerts.jsonl")
    _UI_TELEMETRY_ALERT_SINK_AVAILABLE = False


_FALLBACK_RUNTIME_MODULE = "bot_core.runtime"
_DEFAULT_PIPELINE_SOURCES: tuple[str, ...] = ("bot_core.runtime.pipeline",)
_DEFAULT_REALTIME_SOURCES: tuple[str, ...] = ("bot_core.runtime.realtime",)
_ENV_PIPELINE_MODULES = "RUN_DAILY_TREND_PIPELINE_MODULES"
_ENV_REALTIME_MODULES = "RUN_DAILY_TREND_REALTIME_MODULES"
_ENV_FAIL_ON_SECURITY_WARNINGS = "RUN_DAILY_TREND_FAIL_ON_SECURITY_WARNINGS"
_MODULE_ENV_SPLIT_PATTERN = re.compile(r"[;:,\s]+")

_DEFAULT_PIPELINE_ORIGIN = "domyślne moduły pipeline (bot_core.runtime.pipeline)"
_DEFAULT_REALTIME_ORIGIN = "domyślne moduły realtime (bot_core.runtime.realtime)"
_INTERNAL_OVERRIDE_ORIGIN = "wewnętrzne wywołanie _apply_runtime_overrides"


@dataclass(frozen=True)
class RuntimeModuleSnapshot:
    """Stan modułów runtime wykorzystywanych przez CLI Daily Trend."""

    pipeline_modules: tuple[str, ...]
    realtime_modules: tuple[str, ...]
    pipeline_origin: str | None
    realtime_origin: str | None
    pipeline_resolved_from: str | None
    realtime_resolved_from: str | None
    pipeline_fallback_used: bool
    realtime_fallback_used: bool

    def to_json_payload(self) -> dict[str, object]:
        """Reprezentacja JSON z zachowaniem kompatybilności wstecznej."""

        pipeline_list = list(self.pipeline_modules)
        realtime_list = list(self.realtime_modules)
        return {
            "pipeline_modules": pipeline_list,
            "realtime_modules": realtime_list,
            "pipeline_resolved_from": self.pipeline_resolved_from,
            "realtime_resolved_from": self.realtime_resolved_from,
            "pipeline_fallback_used": self.pipeline_fallback_used,
            "realtime_fallback_used": self.realtime_fallback_used,
            "pipeline": {
                "candidates": pipeline_list,
                "origin": self.pipeline_origin,
                "resolved_from": self.pipeline_resolved_from,
                "fallback_used": self.pipeline_fallback_used,
            },
            "realtime": {
                "candidates": realtime_list,
                "origin": self.realtime_origin,
                "resolved_from": self.realtime_resolved_from,
                "fallback_used": self.realtime_fallback_used,
            },
        }


@dataclass(frozen=True)
class _ValidatedRuntimeConfig:
    """Zawiera zweryfikowaną konfigurację CoreConfig oraz wybrane komponenty runtime."""

    config: "CoreConfig"
    environment: "EnvironmentConfig"
    strategy_name: str
    controller_name: str
    risk_profile_name: str


def _compact_mapping(values: Mapping[str, object] | dict[str, object]) -> dict[str, object]:
    """Usuwa pary, w których wartość to ``None``."""

    return {key: value for key, value in values.items() if value is not None}


def _controller_details(pipeline: Any) -> Mapping[str, object] | None:
    """Buduje podsumowanie kontrolera runtime wraz z parametrami egzekucji."""

    controller = getattr(pipeline, "controller", None)
    if controller is None:
        return None

    details: dict[str, object] = {
        "name": getattr(pipeline, "controller_name", None),
    }

    interval_token = getattr(controller, "interval", None)
    if interval_token is not None:
        interval_str = str(interval_token)
        details["interval"] = interval_str
        try:
            details["interval_normalized"] = _normalize_interval_token(interval_str)
        except Exception:  # noqa: BLE001 - diagnostyka pomocnicza
            _LOGGER.debug("Nie udało się znormalizować interwału kontrolera", exc_info=True)
        try:
            details["interval_ms"] = int(_interval_to_milliseconds(interval_str))
        except Exception:  # noqa: BLE001 - diagnostyka pomocnicza
            _LOGGER.debug("Nie udało się przeliczyć interwału na milisekundy", exc_info=True)

    tick_value = getattr(controller, "tick_seconds", None)
    if isinstance(tick_value, (int, float)):
        details["tick_seconds"] = float(tick_value)

    symbols = getattr(controller, "symbols", None)
    if isinstance(symbols, Sequence) and not isinstance(symbols, (str, bytes)):
        symbol_list = [str(item) for item in symbols]
        details["symbols"] = symbol_list
        details["symbol_count"] = len(symbol_list)

    position_size = getattr(controller, "position_size", None)
    if isinstance(position_size, (int, float)):
        details["position_size"] = float(position_size)

    execution_context = getattr(controller, "execution_context", None)
    if execution_context is not None:
        exec_payload: dict[str, object] = {
            "portfolio_id": getattr(execution_context, "portfolio_id", None),
            "risk_profile": getattr(execution_context, "risk_profile", None),
            "environment": getattr(execution_context, "environment", None),
        }
        metadata = getattr(execution_context, "metadata", None)
        if isinstance(metadata, Mapping):
            exec_payload["metadata"] = dict(metadata)
        details["execution_context"] = _compact_mapping(exec_payload)

    return _compact_mapping(details)


def _strategy_details(pipeline: Any) -> Mapping[str, object] | None:
    """Zwraca szczegóły strategii wykorzystanej w pipeline."""

    strategy = getattr(pipeline, "strategy", None)
    if strategy is None:
        return None

    payload: dict[str, object] = {
        "name": getattr(pipeline, "strategy_name", None),
        "class": strategy.__class__.__name__,
        "module": strategy.__class__.__module__,
    }

    settings = getattr(strategy, "_settings", None)
    if settings is not None:
        settings_payload: Mapping[str, object] | None
        try:
            settings_payload = asdict(settings)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001 - fallback gdy ustawienia nie są dataclass
            try:
                settings_payload = dict(vars(settings))
            except Exception:  # noqa: BLE001 - ostateczny fallback
                settings_payload = None
        if settings_payload:
            payload["settings"] = dict(settings_payload)
        max_history = getattr(settings, "max_history", None)
        if callable(max_history):
            try:
                payload["max_history_bars"] = int(max_history())
            except Exception:  # noqa: BLE001 - diagnostyka pomocnicza
                _LOGGER.debug("Nie udało się odczytać max_history ze strategii", exc_info=True)

    required_intervals = getattr(strategy, "required_intervals", None)
    if callable(required_intervals):
        try:
            payload["required_intervals"] = list(required_intervals())
        except Exception:  # noqa: BLE001 - strategia może nie implementować metody
            _LOGGER.debug("Nie udało się pobrać required_intervals strategii", exc_info=True)

    return _compact_mapping(payload)


def _environment_details(pipeline: Any) -> Mapping[str, object] | None:
    """Tworzy streszczenie środowiska operacyjnego pipeline'u."""

    bootstrap = getattr(pipeline, "bootstrap", None)
    if bootstrap is None:
        return None

    environment_cfg = getattr(bootstrap, "environment", None)
    if environment_cfg is None:
        return None

    environment_value = getattr(environment_cfg, "environment", None)
    if hasattr(environment_value, "value"):
        environment_value = getattr(environment_value, "value")

    payload: dict[str, object] = {
        "name": getattr(environment_cfg, "name", None),
        "exchange": getattr(environment_cfg, "exchange", None),
        "environment": environment_value,
        "risk_profile": getattr(environment_cfg, "risk_profile", None),
        "instrument_universe": getattr(environment_cfg, "instrument_universe", None),
        "data_cache_path": getattr(environment_cfg, "data_cache_path", None),
        "default_strategy": getattr(environment_cfg, "default_strategy", None),
        "default_controller": getattr(environment_cfg, "default_controller", None),
    }

    alert_channels = getattr(environment_cfg, "alert_channels", None)
    if isinstance(alert_channels, Sequence) and not isinstance(alert_channels, (str, bytes)):
        payload["alert_channels"] = list(alert_channels)

    ip_allowlist = getattr(environment_cfg, "ip_allowlist", None)
    if isinstance(ip_allowlist, Sequence) and not isinstance(ip_allowlist, (str, bytes)):
        payload["ip_allowlist_count"] = len(tuple(ip_allowlist))

    adapter = getattr(bootstrap, "adapter", None)
    if adapter is not None:
        payload["adapter"] = {
            "class": adapter.__class__.__name__,
            "module": adapter.__class__.__module__,
        }

    metrics_server = getattr(bootstrap, "metrics_server", None)
    payload["metrics_service_active"] = metrics_server is not None
    if metrics_server is not None:
        address = getattr(metrics_server, "address", None)
        if address:
            payload["metrics_service_address"] = address

    decision_journal = getattr(bootstrap, "decision_journal", None)
    payload["decision_journal_enabled"] = decision_journal is not None
    if decision_journal is not None:
        payload["decision_journal"] = {
            "class": decision_journal.__class__.__name__,
            "module": decision_journal.__class__.__module__,
        }

    return _compact_mapping(payload)


def _build_module_candidates(
    user_modules: Sequence[str] | None,
    defaults: Sequence[str],
) -> tuple[str, ...]:
    """Łączy moduły użytkownika, domyślne i fallback, zapewniając unikalność kolejności."""

    ordered: list[str] = []
    seen: set[str] = set()

    def _add(module_name: str | None) -> None:
        if not module_name:
            return
        if module_name in seen:
            return
        ordered.append(module_name)
        seen.add(module_name)

    if user_modules:
        for candidate in user_modules:
            _add(candidate.strip())

    for candidate in defaults:
        _add(candidate)

    _add(_FALLBACK_RUNTIME_MODULE)
    return tuple(ordered)


def _parse_modules_env_value(
    raw_value: str | None,
    *,
    env_var: str,
) -> tuple[tuple[str, ...] | None, str | None]:
    """Parsuje wartość modułów ze zmiennej środowiskowej."""

    if raw_value is None:
        return None, None

    stripped = raw_value.strip()
    if not stripped:
        _LOGGER.warning(
            "Zmienna środowiskowa %s jest ustawiona, ale nie zawiera żadnych modułów – ignoruję",
            env_var,
        )
        return None, "empty_value"

    modules = [candidate.strip() for candidate in _MODULE_ENV_SPLIT_PATTERN.split(raw_value) if candidate.strip()]
    if not modules:
        _LOGGER.warning(
            "Zmienna środowiskowa %s nie zawiera poprawnych nazw modułów (wartość: %r)",
            env_var,
            raw_value,
        )
        return None, "invalid_value"

    return tuple(modules), None


def _modules_from_environment(env_var: str) -> tuple[str, ...] | None:
    """Odczytuje listę modułów z zmiennej środowiskowej."""

    modules, _ = _parse_modules_env_value(os.environ.get(env_var), env_var=env_var)
    return modules


def _parse_env_bool(value: str, *, variable: str) -> bool:
    """Paruje wartości boolowskie ze zmiennych środowiskowych."""

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Zmienna {variable} oczekuje wartości bool (true/false, 1/0, yes/no) – otrzymano '{value}'."
    )


def _classify_risk_profile(name: str) -> str:
    normalized = name.strip().lower()
    if normalized in {"conservative", "balanced", "aggressive", "manual"}:
        return normalized
    return "custom"


def _risk_profile_notes(classification: str) -> list[str]:
    notes = [
        "Ścieżka demo→paper→live jest obowiązkowa przed uruchomieniem w produkcji.",
        "Monitoruj alerty reduce motion oraz overlay guard w powłoce Qt w trakcie testów.",
    ]
    if classification == "manual":
        notes.append(
            "Profil manualny wymaga ręcznej akceptacji sygnałów i rozszerzonego audytu RBAC/decision log.",
        )
    elif classification == "aggressive":
        notes.append(
            "Profil agresywny wymaga dodatkowych testów symulacyjnych i ścisłego nadzoru alertów ryzyka.",
        )
    elif classification == "conservative":
        notes.append("Profil konserwatywny zalecany przy wdrożeniach pilotażowych lub testach integracyjnych.")
    elif classification == "balanced":
        notes.append("Profil zbalansowany używany w kampanii referencyjnej paper trading.")
    return notes


def _risk_profile_entry(
    profile: "RiskProfileConfig", associated_envs: Sequence[str],
) -> dict[str, object]:
    classification = _classify_risk_profile(profile.name)
    payload: dict[str, object] = {
        "classification": classification,
        "limits": {
            "max_daily_loss_pct": profile.max_daily_loss_pct,
            "max_position_pct": profile.max_position_pct,
            "target_volatility": profile.target_volatility,
            "max_leverage": profile.max_leverage,
            "stop_loss_atr_multiple": profile.stop_loss_atr_multiple,
            "max_open_positions": profile.max_open_positions,
            "hard_drawdown_pct": profile.hard_drawdown_pct,
        },
        "associated_environments": list(associated_envs),
        "requires_manual_controls": classification == "manual",
        "deployment_pipeline": "demo→paper→live",
        "notes": _risk_profile_notes(classification),
    }
    if profile.data_quality is not None:
        payload["data_quality"] = {
            "max_gap_minutes": profile.data_quality.max_gap_minutes,
            "min_ok_ratio": profile.data_quality.min_ok_ratio,
        }
    return payload


def _risk_profiles_payload(
    config: "CoreConfig", environment_name: str | None,
) -> dict[str, object]:
    environment_profiles = {
        name: env.risk_profile for name, env in sorted(config.environments.items())
    }

    env_config = None
    if environment_name:
        env_config = config.environments.get(environment_name)

    profiles_section = {
        name: _risk_profile_entry(
            profile,
            [env for env, profile_name in environment_profiles.items() if profile_name == name],
        )
        for name, profile in sorted(config.risk_profiles.items())
    }

    available_profiles = sorted(profiles_section.keys())

    return {
        "environment": environment_name,
        "environment_found": env_config is not None,
        "environment_profile": env_config.risk_profile if env_config else None,
        "available_profiles": available_profiles,
        "profiles": profiles_section,
        "environments": environment_profiles,
    }


def _print_risk_profiles(config_path: Path, environment_name: str | None) -> int:
    if not config_path.exists():
        _LOGGER.error("Plik konfiguracyjny %s nie istnieje", config_path)
        return 1

    try:
        config = load_core_config(config_path)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.error("Nie udało się wczytać konfiguracji %s: %s", config_path, exc)
        return 2

    payload = _risk_profiles_payload(config, environment_name)
    profile_names = ", ".join(payload["available_profiles"])
    if profile_names:
        _LOGGER.info("Dostępne profile ryzyka: %s", profile_names)
    else:
        _LOGGER.warning("Konfiguracja %s nie definiuje żadnych profili ryzyka", config_path)

    env_profile = payload.get("environment_profile")
    if env_profile:
        _LOGGER.info(
            "Środowisko %s korzysta z profilu %s (audyt demo→paper→live wymagany).",
            payload["environment"],
            env_profile,
        )
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


def _precheck_summary(payload: Mapping[str, object] | None) -> Mapping[str, object] | None:
    """Redukuje wynik paper_precheck do metadanych audytowych."""

    if not isinstance(payload, Mapping):
        return None

    summary: dict[str, object] = {
        "status": payload.get("status"),
        "coverage_status": payload.get("coverage_status"),
        "risk_status": payload.get("risk_status"),
    }

    manifest_path = payload.get("manifest_path")
    if manifest_path is not None:
        summary["manifest_path"] = str(manifest_path)

    coverage_warnings = payload.get("coverage_warnings")
    if isinstance(coverage_warnings, Sequence) and coverage_warnings:
        summary["coverage_warnings"] = [str(item) for item in coverage_warnings]

    risk_payload = payload.get("risk")
    if isinstance(risk_payload, Mapping):
        risk_summary: dict[str, object] = {}
        warnings = risk_payload.get("warnings")
        if isinstance(warnings, Sequence) and warnings:
            risk_summary["warnings"] = [str(item) for item in warnings]
        if risk_summary:
            summary["risk"] = risk_summary

    return summary


def _sanitize_precheck_audit_metadata(
    metadata: Mapping[str, object] | None,
) -> Mapping[str, object] | None:
    """Zwraca tylko podstawowe informacje audytowe z raportu precheck."""

    if not isinstance(metadata, Mapping):
        return None

    allowed_keys = {
        "path",
        "sha256",
        "created_at",
        "environment",
        "status",
        "size_bytes",
    }
    sanitized: dict[str, object] = {}
    for key in sorted(allowed_keys):
        if key not in metadata:
            continue
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, (str, int, float)):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    return sanitized or None


def _collect_git_metadata(base_path: Path | None = None) -> Mapping[str, object] | None:
    """Zwraca podstawowe metadane repozytorium Git na potrzeby audytu."""

    git_binary = shutil.which("git")
    if not git_binary:
        _LOGGER.debug("Polecenie 'git' nie jest dostępne w PATH – pomijam metadane repozytorium")
        return None

    search_path = base_path or Path(__file__).resolve()
    if search_path.is_file():
        search_path = search_path.parent

    try:
        root_result = subprocess.run(
            [git_binary, "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            cwd=search_path,
        )
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - zależy od środowiska
        _LOGGER.debug(
            "Nie udało się ustalić katalogu głównego repozytorium Git", exc_info=True
        )
        return None

    repo_root = Path(root_result.stdout.strip())

    def _git(*args: str) -> str:
        result = subprocess.run(
            [git_binary, *args],
            check=True,
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        return result.stdout.strip()

    metadata: dict[str, object] = {"root": str(repo_root)}

    try:
        metadata["commit"] = _git("rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        _LOGGER.debug("Nie udało się pobrać identyfikatora commita Git", exc_info=True)
        return metadata

    try:
        metadata["branch"] = _git("rev-parse", "--abbrev-ref", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        _LOGGER.debug("Nie udało się ustalić bieżącej gałęzi Git", exc_info=True)

    try:
        metadata["tag"] = _git("describe", "--tags", "--always")
    except (OSError, subprocess.CalledProcessError):
        _LOGGER.debug("Polecenie git describe nie powiodło się", exc_info=True)

    try:
        metadata["commit_timestamp"] = _git("log", "-1", "--format=%cI")
    except (OSError, subprocess.CalledProcessError):
        _LOGGER.debug("Nie udało się pobrać znacznika czasu commita", exc_info=True)

    try:
        status_output = subprocess.run(
            [git_binary, "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repo_root,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        _LOGGER.debug("Nie udało się ustalić stanu roboczego repozytorium", exc_info=True)
    else:
        is_dirty = bool(status_output.strip())
        metadata["is_dirty"] = is_dirty
        if is_dirty:
            metadata["dirty_entries"] = len(status_output.strip().splitlines())

    return metadata


def _resolve_path_relative_to(base_dir: Path | None, value: Path | str) -> Path:
    """Zwraca ścieżkę uwzględniającą katalog bazowy konfiguracji."""

    candidate = Path(value).expanduser()
    if candidate.is_absolute() or base_dir is None:
        return candidate

    try:
        normalized_base = base_dir.expanduser().resolve(strict=False)
    except Exception:  # noqa: BLE001 - zachowujemy najlepsze przybliżenie
        normalized_base = base_dir.expanduser().absolute()

    return normalized_base / candidate


def _config_file_metadata(path: Path) -> Mapping[str, object] | None:
    """Zwraca metadane audytowe pliku konfiguracyjnego, w tym sumę SHA-256."""

    metadata = _file_reference_metadata(path, role="core_config")
    if not metadata.get("exists"):
        # Brak pliku konfiguracyjnego – komunikat został już zarejestrowany.
        return None

    metadata.pop("path", None)
    return metadata


def _metrics_service_details_from_config(
    config: "CoreConfig",
    *,
    base_dir: Path | None = None,
    runtime_ui_alert_path: Path | str | None = None,
    runtime_jsonl_path: Path | str | None = None,
    ui_alert_sink_active: bool | None = None,
    runtime_service_enabled: bool | None = None,
    runtime_ui_alert_metadata: Mapping[str, object] | None = None,
    runtime_jsonl_metadata: Mapping[str, object] | None = None,
    runtime_security_warnings: Sequence[str] | None = None,
) -> Mapping[str, object]:
    """Buduje sekcję audytową opisującą konfigurację MetricsService."""

    runtime_payload: dict[str, object] = {}
    if runtime_jsonl_metadata is not None:
        runtime_payload["jsonl_file"] = runtime_jsonl_metadata
        runtime_payload["jsonl_path"] = str(
            runtime_jsonl_metadata.get("path")
            or runtime_jsonl_metadata.get("absolute_path")
            or runtime_jsonl_path
            or ""
        )
    elif runtime_jsonl_path:
        runtime_jsonl = Path(runtime_jsonl_path).expanduser()
        runtime_payload["jsonl_path"] = str(runtime_jsonl)
        runtime_payload["jsonl_file"] = _file_reference_metadata(
            runtime_jsonl, role="jsonl"
        )
    if runtime_ui_alert_metadata is not None:
        runtime_payload["ui_alerts_file"] = runtime_ui_alert_metadata
        runtime_payload["ui_alerts_jsonl_path"] = str(
            runtime_ui_alert_metadata.get("path")
            or runtime_ui_alert_metadata.get("absolute_path")
            or runtime_ui_alert_path
            or ""
        )
    elif runtime_ui_alert_path:
        runtime_ui_alert = Path(runtime_ui_alert_path).expanduser()
        runtime_payload["ui_alerts_jsonl_path"] = str(runtime_ui_alert)
        runtime_payload["ui_alerts_file"] = _file_reference_metadata(
            runtime_ui_alert, role="ui_alerts_jsonl"
        )
    if ui_alert_sink_active is not None:
        runtime_payload["ui_alert_sink_active"] = bool(ui_alert_sink_active)
    if runtime_service_enabled is not None:
        runtime_payload["service_enabled"] = bool(runtime_service_enabled)
    if runtime_security_warnings:
        runtime_payload["security_warnings"] = list(runtime_security_warnings)
    runtime_payload = _compact_mapping(runtime_payload) if runtime_payload else {}

    metrics_cfg = getattr(config, "metrics_service", None)
    if metrics_cfg is None:
        payload: dict[str, object] = {
            "configured": False,
            "ui_alert_sink_available": _UI_TELEMETRY_ALERT_SINK_AVAILABLE,
        }
        if runtime_payload:
            payload["runtime_state"] = runtime_payload
        return payload

    payload: dict[str, object] = {
        "configured": True,
        "enabled": bool(getattr(metrics_cfg, "enabled", False)),
        "host": getattr(metrics_cfg, "host", None),
        "port": getattr(metrics_cfg, "port", None),
        "history_size": getattr(metrics_cfg, "history_size", None),
        "log_sink": bool(getattr(metrics_cfg, "log_sink", False)),
        "jsonl_fsync": bool(getattr(metrics_cfg, "jsonl_fsync", False)),
        "ui_alert_sink_available": _UI_TELEMETRY_ALERT_SINK_AVAILABLE,
    }

    jsonl_path = getattr(metrics_cfg, "jsonl_path", None)
    if jsonl_path:
        resolved_jsonl = _resolve_path_relative_to(base_dir, jsonl_path)
        jsonl_metadata = _file_reference_metadata(resolved_jsonl, role="jsonl")
        payload["jsonl_path"] = jsonl_metadata["path"]
        payload["jsonl_file"] = jsonl_metadata

    ui_alerts_path_raw = getattr(metrics_cfg, "ui_alerts_jsonl_path", None)
    ui_alerts_source = "config" if ui_alerts_path_raw else "default"
    if ui_alerts_path_raw:
        ui_alerts_path = _resolve_path_relative_to(base_dir, ui_alerts_path_raw)
    else:
        ui_alerts_path = DEFAULT_UI_ALERTS_JSONL_PATH.expanduser()
    ui_alerts_metadata = _file_reference_metadata(ui_alerts_path, role="ui_alerts_jsonl")
    payload["ui_alerts_source"] = ui_alerts_source
    payload["ui_alerts_jsonl_path"] = ui_alerts_metadata["path"]
    payload["ui_alerts_file"] = ui_alerts_metadata

    tls_cfg = getattr(metrics_cfg, "tls", None)
    if tls_cfg is not None:
        tls_payload: dict[str, object] = {
            "configured": True,
            "enabled": bool(getattr(tls_cfg, "enabled", False)),
            "require_client_auth": bool(getattr(tls_cfg, "require_client_auth", False)),
        }

        certificate_path = getattr(tls_cfg, "certificate_path", None)
        if certificate_path:
            tls_payload["certificate"] = _file_reference_metadata(
                _resolve_path_relative_to(base_dir, certificate_path), role="tls_cert"
            )

        private_key_path = getattr(tls_cfg, "private_key_path", None)
        if private_key_path:
            tls_payload["private_key"] = _file_reference_metadata(
                _resolve_path_relative_to(base_dir, private_key_path), role="tls_key"
            )

        client_ca_path = getattr(tls_cfg, "client_ca_path", None)
        if client_ca_path:
            tls_payload["client_ca"] = _file_reference_metadata(
                _resolve_path_relative_to(base_dir, client_ca_path), role="tls_client_ca"
            )

        payload["tls"] = tls_payload
    else:
        payload["tls"] = {"configured": False}

    if runtime_payload:
        payload["runtime_state"] = runtime_payload

    return _compact_mapping(payload)


def _load_validated_core_config(
    config_path: Path,
    *,
    environment: str,
    strategy: str | None,
    controller: str | None,
    risk_profile: str | None,
) -> _ValidatedRuntimeConfig | None:
    """Wczytuje CoreConfig i waliduje podstawowe parametry wejściowe CLI."""

    try:
        config = load_core_config(config_path)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.error("Nie udało się wczytać konfiguracji %s: %s", config_path, exc)
        return None

    environment_cfg = config.environments.get(environment)
    if environment_cfg is None:
        available = ", ".join(sorted(config.environments)) or "brak"
        _LOGGER.error(
            "Środowisko %s nie istnieje w konfiguracji %s. Dostępne środowiska: %s",
            environment,
            config_path,
            available,
        )
        return None

    resolved_strategy = strategy or environment_cfg.default_strategy
    if not resolved_strategy:
        _LOGGER.error(
            "Środowisko %s nie ma zdefiniowanej domyślnej strategii, a parametr --strategy nie został użyty.",
            environment,
        )
        return None

    if resolved_strategy not in config.strategies:
        available = ", ".join(sorted(config.strategies)) or "brak"
        if strategy:
            _LOGGER.error(
                "Strategia %s nie jest zdefiniowana w konfiguracji. Dostępne strategie: %s",
                resolved_strategy,
                available,
            )
        else:
            _LOGGER.error(
                "Środowisko %s odwołuje się do strategii %s, której nie znaleziono. Dostępne strategie: %s",
                environment,
                resolved_strategy,
                available,
            )
        return None

    resolved_controller = controller or environment_cfg.default_controller
    if not resolved_controller:
        _LOGGER.error(
            "Środowisko %s nie ma zdefiniowanego domyślnego kontrolera runtime, a parametr --controller nie został użyty.",
            environment,
        )
        return None

    if resolved_controller not in config.runtime_controllers:
        available = ", ".join(sorted(config.runtime_controllers)) or "brak"
        if controller:
            _LOGGER.error(
                "Kontroler runtime %s nie jest zdefiniowany w konfiguracji. Dostępne kontrolery: %s",
                resolved_controller,
                available,
            )
        else:
            _LOGGER.error(
                "Środowisko %s odwołuje się do kontrolera runtime %s, którego nie znaleziono. Dostępne kontrolery: %s",
                environment,
                resolved_controller,
                available,
            )
        return None

    requested_profile = risk_profile or environment_cfg.risk_profile
    if requested_profile not in config.risk_profiles:
        available = ", ".join(sorted(config.risk_profiles)) or "brak"
        if risk_profile:
            _LOGGER.error(
                "Profil ryzyka %s nie istnieje w konfiguracji. Dostępne profile: %s",
                requested_profile,
                available,
            )
        else:
            _LOGGER.error(
                "Środowisko %s odwołuje się do profilu ryzyka %s, którego nie znaleziono. Dostępne profile: %s",
                environment,
                requested_profile,
                available,
            )
        return None

    return _ValidatedRuntimeConfig(
        config=config,
        environment=environment_cfg,
        strategy_name=resolved_strategy,
        controller_name=resolved_controller,
        risk_profile_name=requested_profile,
    )


def _build_runtime_plan_payload(
    *,
    args: argparse.Namespace,
    snapshot: RuntimeModuleSnapshot,
    pipeline: Any,
    config: "CoreConfig",
    environment_name: str,
    cli_pipeline_modules: Sequence[str] | None,
    cli_realtime_modules: Sequence[str] | None,
    env_pipeline_modules: Sequence[str] | None,
    env_pipeline_raw: str | None,
    env_pipeline_applied: bool,
    env_pipeline_reason: str | None,
    env_realtime_modules: Sequence[str] | None,
    env_realtime_raw: str | None,
    env_realtime_applied: bool,
    env_realtime_reason: str | None,
    cli_fail_on_security_flag: bool,
    env_fail_on_security_raw: str | None,
    env_fail_on_security_applied: bool,
    env_fail_on_security_value: bool | None,
    fail_on_security_source: str,
    precheck_payload: Mapping[str, object] | None,
    precheck_audit_metadata: Mapping[str, object] | None,
    operator_name: str | None,
) -> Mapping[str, object]:
    """Buduje wpis audytowy opisujący konfigurację runtime."""

    runtime_overview = snapshot.to_json_payload()
    risk_payload = _risk_profiles_payload(config, environment_name)
    risk_profile_name = getattr(pipeline, "risk_profile_name", None)

    risk_details = None
    if risk_profile_name:
        profiles = risk_payload.get("profiles")
        if isinstance(profiles, Mapping):
            candidate = profiles.get(risk_profile_name)
            if isinstance(candidate, Mapping):
                risk_details = candidate

    environment_cfg = getattr(pipeline.bootstrap, "environment", None)
    environment_type = None
    if environment_cfg is not None and hasattr(environment_cfg, "environment"):
        environment_type = getattr(environment_cfg.environment, "value", environment_cfg.environment)

    precheck_summary = _precheck_summary(precheck_payload)

    config_path_arg = Path(args.config).expanduser()
    config_source_value = getattr(config, "source_path", None)
    config_source_path = (
        Path(config_source_value).expanduser()
        if config_source_value
        else config_path_arg
    )
    config_file_section: dict[str, object] = {"path": str(config_path_arg)}
    config_file_metadata = _config_file_metadata(config_source_path)
    if config_file_metadata:
        config_file_section.update(config_file_metadata)
    elif config_source_value:
        config_file_section["absolute_path"] = str(config_source_path)
    else:
        try:
            config_file_section["absolute_path"] = str(
                config_path_arg.resolve(strict=False)
            )
        except Exception:  # noqa: BLE001 - zachowujemy najlepsze przybliżenie
            config_file_section["absolute_path"] = str(config_path_arg.absolute())

    config_base_dir = config_source_path.parent

    environment_entries: list[dict[str, object]] = []

    if env_pipeline_raw is not None:
        pipeline_entry: dict[str, object] = {
            "option": "pipeline_modules",
            "variable": _ENV_PIPELINE_MODULES,
            "raw_value": env_pipeline_raw,
            "applied": bool(env_pipeline_applied and env_pipeline_modules is not None),
        }
        if env_pipeline_modules is not None:
            pipeline_entry["parsed_value"] = list(env_pipeline_modules)
        if env_pipeline_reason is not None:
            pipeline_entry["reason"] = env_pipeline_reason
        environment_entries.append(_compact_mapping(pipeline_entry))

    if env_realtime_raw is not None:
        realtime_entry: dict[str, object] = {
            "option": "realtime_modules",
            "variable": _ENV_REALTIME_MODULES,
            "raw_value": env_realtime_raw,
            "applied": bool(env_realtime_applied and env_realtime_modules is not None),
        }
        if env_realtime_modules is not None:
            realtime_entry["parsed_value"] = list(env_realtime_modules)
        if env_realtime_reason is not None:
            realtime_entry["reason"] = env_realtime_reason
        environment_entries.append(_compact_mapping(realtime_entry))

    env_fail_reason = None
    if (
        env_fail_on_security_raw is not None
        and not env_fail_on_security_applied
        and cli_fail_on_security_flag
    ):
        env_fail_reason = "cli_override"

    if env_fail_on_security_raw is not None:
        fail_entry: dict[str, object] = {
            "option": "fail_on_security_warnings",
            "variable": _ENV_FAIL_ON_SECURITY_WARNINGS,
            "raw_value": env_fail_on_security_raw,
            "applied": env_fail_on_security_applied,
        }
        if env_fail_on_security_value is not None:
            fail_entry["parsed_value"] = bool(env_fail_on_security_value)
        if env_fail_reason is not None:
            fail_entry["reason"] = env_fail_reason
        environment_entries.append(_compact_mapping(fail_entry))

    environment_entries_list = [entry for entry in environment_entries if entry]

    plan: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(args.config)),
        "config_file": config_file_section,
        "environment": environment_name,
        "environment_type": environment_type,
        "runtime_modules": runtime_overview,
        "pipeline_details": {
            "class": pipeline.__class__.__name__,
            "module": pipeline.__class__.__module__,
        },
        "overrides": {
            "cli": _compact_mapping(
                {
                    "pipeline_modules": list(cli_pipeline_modules or []),
                    "realtime_modules": list(cli_realtime_modules or []),
                    "fail_on_security_warnings": bool(cli_fail_on_security_flag),
                }
            ),
            "environment": _compact_mapping(
                {
                    "entries": environment_entries_list or None,
                    "pipeline_modules": list(env_pipeline_modules or [])
                    if env_pipeline_applied and env_pipeline_modules is not None
                    else None,
                    "realtime_modules": list(env_realtime_modules or [])
                    if env_realtime_applied and env_realtime_modules is not None
                    else None,
                    "fail_on_security_warnings": _compact_mapping(
                        {
                            "variable": _ENV_FAIL_ON_SECURITY_WARNINGS,
                            "raw_value": env_fail_on_security_raw,
                            "applied": env_fail_on_security_applied,
                            "parsed_value": bool(env_fail_on_security_value)
                            if env_fail_on_security_value is not None
                            else None,
                            "reason": env_fail_reason,
                        }
                    )
                    if env_fail_on_security_raw is not None
                    else None,
                }
            ),
        },
        "strategy": getattr(pipeline, "strategy_name", args.strategy),
        "controller": getattr(pipeline, "controller_name", args.controller),
        "risk_profile": risk_profile_name,
        "risk_profile_details": risk_details,
        "risk_profiles_overview": risk_payload,
        "paper_smoke": bool(args.paper_smoke),
        "paper_smoke_operator": operator_name,
        "notes": [
            "Profile ryzyka: konserwatywny, zbalansowany, agresywny, manualny.",
            "Pipeline środowiskowy: demo→paper→live (audyt wymagany przed produkcją).",
        ],
    }

    if precheck_summary:
        plan["paper_precheck"] = precheck_summary

    audit_metadata = _sanitize_precheck_audit_metadata(precheck_audit_metadata)
    if audit_metadata:
        plan["paper_precheck_audit"] = audit_metadata

    try:
        git_metadata = _collect_git_metadata()
    except Exception:  # noqa: BLE001 - diagnostyka pomocnicza, nie przerywamy wykonywania
        _LOGGER.debug("Nie udało się zebrać metadanych Git", exc_info=True)
    else:
        if git_metadata:
            plan["git"] = git_metadata

    controller_payload = _controller_details(pipeline)
    if controller_payload:
        plan["controller_details"] = controller_payload

    strategy_payload = _strategy_details(pipeline)
    if strategy_payload:
        plan["strategy_details"] = strategy_payload

    environment_payload = _environment_details(pipeline)
    if environment_payload:
        plan["environment_details"] = environment_payload

    bootstrap_ctx = getattr(pipeline, "bootstrap", None)
    runtime_ui_alert_path = None
    runtime_jsonl_path = None
    runtime_sink_active = None
    runtime_service_enabled = None
    runtime_ui_alert_metadata = None
    runtime_jsonl_metadata = None
    runtime_security_warnings = None
    if bootstrap_ctx is not None:
        runtime_ui_alert_path = getattr(bootstrap_ctx, "metrics_ui_alerts_path", None)
        runtime_jsonl_path = getattr(bootstrap_ctx, "metrics_jsonl_path", None)
        runtime_sink_active = getattr(bootstrap_ctx, "metrics_ui_alert_sink_active", None)
        runtime_service_enabled = getattr(bootstrap_ctx, "metrics_service_enabled", None)
        runtime_ui_alert_metadata = getattr(
            bootstrap_ctx, "metrics_ui_alerts_metadata", None
        )
        runtime_jsonl_metadata = getattr(
            bootstrap_ctx, "metrics_jsonl_metadata", None
        )
        runtime_security_warnings = getattr(
            bootstrap_ctx, "metrics_security_warnings", None
        )

    metrics_details = _metrics_service_details_from_config(
        config,
        base_dir=config_base_dir,
        runtime_ui_alert_path=runtime_ui_alert_path,
        runtime_jsonl_path=runtime_jsonl_path,
        ui_alert_sink_active=runtime_sink_active,
        runtime_service_enabled=runtime_service_enabled,
        runtime_ui_alert_metadata=runtime_ui_alert_metadata,
        runtime_jsonl_metadata=runtime_jsonl_metadata,
        runtime_security_warnings=runtime_security_warnings,
    )
    if metrics_details:
        plan["metrics_service_details"] = metrics_details

    fail_on_security_parameter_source = (
        "env" if fail_on_security_source.startswith("env:") else fail_on_security_source
    )

    plan["security"] = {
        "fail_on_security_warnings": _compact_mapping(
            {
                "enabled": bool(args.fail_on_security_warnings),
                "source": fail_on_security_source,
                "parameter_source": fail_on_security_parameter_source,
                "environment_variable": _ENV_FAIL_ON_SECURITY_WARNINGS
                if env_fail_on_security_raw is not None
                else None,
                "environment_raw_value": env_fail_on_security_raw,
                "environment_applied": env_fail_on_security_applied,
                "environment_reason": env_fail_reason,
            }
        ),
        "parameter_sources": {
            "fail_on_security_warnings": fail_on_security_parameter_source,
        },
    }

    return plan
def _append_runtime_plan_jsonl(path: Path, payload: Mapping[str, object]) -> Path:
    """Dopisuje wpis audytowy planu runtime do pliku JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")
    return path


@dataclass(frozen=True)
class _ResolvedRuntimeSymbols:
    """Wynik importu symboli runtime wraz z modułem źródłowym."""

    symbols: tuple[Any, ...]
    module_name: str


def _resolve_runtime_symbols(
    module_candidates: Iterable[str],
    symbol_names: Iterable[str],
    *,
    component_hint: str,
) -> _ResolvedRuntimeSymbols:
    """Zwraca symbole runtime z listy modułów z komunikatami diagnostycznymi."""

    diagnostics: list[str] = []
    names = tuple(symbol_names)
    candidates = tuple(module_candidates)

    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name == module_name:
                diagnostics.append(f"{module_name}: moduł nie znaleziony")
                continue
            raise

        missing = [name for name in names if not hasattr(module, name)]
        if missing:
            diagnostics.append(
                f"{module_name}: brak symboli {', '.join(missing)}"
            )
            continue

        resolved = tuple(getattr(module, name) for name in names)
        return _ResolvedRuntimeSymbols(resolved, module_name)

    candidates_display = ", ".join(candidates)
    symbols = ", ".join(names)
    details = "; ".join(diagnostics) or "brak dodatkowej diagnostyki"
    message = (
        f"Nie znaleziono {component_hint}: wymagane symbole ({symbols}) nie są dostępne w modułach: {candidates_display}. "
        f"Diagnostyka: {details}"
    )
    raise ImportError(message)


def _import_daily_trend_pipeline_symbols(
    module_candidates: Sequence[str],
) -> tuple[_BuildDailyTrendPipelineProto, _CreateTradingControllerProto, str]:
    """Importuje wymagane symbole pipeline'u zgodnie z kolejnością modułów."""

    symbols = ("build_daily_trend_pipeline", "create_trading_controller")
    resolved = _resolve_runtime_symbols(module_candidates, symbols, component_hint="runtime pipeline")
    return resolved.symbols[0], resolved.symbols[1], resolved.module_name


def _import_daily_trend_realtime_runner(
    module_candidates: Sequence[str],
) -> tuple[_DailyTrendRealtimeRunnerProto, str]:
    """Importuje klasę realtime z kolejnością modułów i walidacją typu."""

    resolved = _resolve_runtime_symbols(
        module_candidates,
        ("DailyTrendRealtimeRunner",),
        component_hint="realtime runner",
    )
    runner = resolved.symbols[0]
    if not isinstance(runner, type):
        raise ImportError(
            "DailyTrendRealtimeRunner nie jest klasą – sprawdź implementację bot_core.runtime"
        )
    return runner, resolved.module_name


_PIPELINE_MODULE_CANDIDATES = _build_module_candidates(None, _DEFAULT_PIPELINE_SOURCES)
_REALTIME_MODULE_CANDIDATES = _build_module_candidates(None, _DEFAULT_REALTIME_SOURCES)
_PIPELINE_MODULE_ORIGIN: str | None = _DEFAULT_PIPELINE_ORIGIN
_REALTIME_MODULE_ORIGIN: str | None = _DEFAULT_REALTIME_ORIGIN
_PIPELINE_RESOLVED_FROM: str | None = None
_REALTIME_RESOLVED_FROM: str | None = None
_PIPELINE_FALLBACK_USED: bool = False
_REALTIME_FALLBACK_USED: bool = False

if build_daily_trend_pipeline is None or create_trading_controller is None:
    (
        build_daily_trend_pipeline,
        create_trading_controller,
        _PIPELINE_RESOLVED_FROM,
    ) = _import_daily_trend_pipeline_symbols(
        _PIPELINE_MODULE_CANDIDATES
    )
else:
    _PIPELINE_RESOLVED_FROM = build_daily_trend_pipeline.__module__

_PIPELINE_FALLBACK_USED = _PIPELINE_RESOLVED_FROM == _FALLBACK_RUNTIME_MODULE

if DailyTrendRealtimeRunner is None:
    (
        DailyTrendRealtimeRunner,
        _REALTIME_RESOLVED_FROM,
    ) = _import_daily_trend_realtime_runner(_REALTIME_MODULE_CANDIDATES)
else:
    _REALTIME_RESOLVED_FROM = DailyTrendRealtimeRunner.__module__

_REALTIME_FALLBACK_USED = _REALTIME_RESOLVED_FROM == _FALLBACK_RUNTIME_MODULE


def _apply_runtime_overrides(
    pipeline_modules: Sequence[str] | None,
    realtime_modules: Sequence[str] | None,
    *,
    pipeline_origin: str | None = None,
    realtime_origin: str | None = None,
) -> None:
    """Aktualizuje globalne symbole runtime zgodnie z modułami przekazanymi w CLI."""

    global build_daily_trend_pipeline
    global create_trading_controller
    global DailyTrendRealtimeRunner
    global _PIPELINE_MODULE_CANDIDATES
    global _REALTIME_MODULE_CANDIDATES
    global _PIPELINE_MODULE_ORIGIN
    global _REALTIME_MODULE_ORIGIN
    global _PIPELINE_RESOLVED_FROM
    global _REALTIME_RESOLVED_FROM
    global _PIPELINE_FALLBACK_USED
    global _REALTIME_FALLBACK_USED

    if pipeline_modules is not None:
        candidates = _build_module_candidates(pipeline_modules, _DEFAULT_PIPELINE_SOURCES)
        try:
            (
                build_daily_trend_pipeline,
                create_trading_controller,
                pipeline_source_module,
            ) = _import_daily_trend_pipeline_symbols(candidates)
        except ImportError as exc:  # pragma: no cover - ścieżka obsługi błędów testowana przez main()
            source = pipeline_origin or "override pipeline"
            raise ImportError(f"{source}: {exc}") from exc
        _PIPELINE_MODULE_CANDIDATES = candidates
        _PIPELINE_MODULE_ORIGIN = pipeline_origin or _INTERNAL_OVERRIDE_ORIGIN
        _PIPELINE_RESOLVED_FROM = pipeline_source_module
        _PIPELINE_FALLBACK_USED = pipeline_source_module == _FALLBACK_RUNTIME_MODULE
        source_suffix = f" (źródło: {pipeline_origin})" if pipeline_origin else ""
        _LOGGER.info("Zastosowano moduły pipeline%s: %s", source_suffix, ", ".join(candidates))
        _LOGGER.debug(
            "Symbole pipeline pochodzą z modułu: %s", pipeline_source_module
        )

    if realtime_modules is not None:
        candidates = _build_module_candidates(realtime_modules, _DEFAULT_REALTIME_SOURCES)
        try:
            (
                DailyTrendRealtimeRunner,
                realtime_source_module,
            ) = _import_daily_trend_realtime_runner(candidates)
        except ImportError as exc:  # pragma: no cover - ścieżka obsługi błędów testowana przez main()
            source = realtime_origin or "override realtime"
            raise ImportError(f"{source}: {exc}") from exc
        _REALTIME_MODULE_CANDIDATES = candidates
        _REALTIME_MODULE_ORIGIN = realtime_origin or _INTERNAL_OVERRIDE_ORIGIN
        _REALTIME_RESOLVED_FROM = realtime_source_module
        _REALTIME_FALLBACK_USED = realtime_source_module == _FALLBACK_RUNTIME_MODULE
        source_suffix = f" (źródło: {realtime_origin})" if realtime_origin else ""
        _LOGGER.info("Zastosowano moduły realtime%s: %s", source_suffix, ", ".join(candidates))
        _LOGGER.debug(
            "Klasę realtime załadowano z modułu: %s", realtime_source_module
        )


def get_runtime_module_candidates() -> RuntimeModuleSnapshot:
    """Zwraca aktualną listę modułów runtime używanych przez CLI."""

    return RuntimeModuleSnapshot(
        tuple(_PIPELINE_MODULE_CANDIDATES),
        tuple(_REALTIME_MODULE_CANDIDATES),
        _PIPELINE_MODULE_ORIGIN,
        _REALTIME_MODULE_ORIGIN,
        _PIPELINE_RESOLVED_FROM,
        _REALTIME_RESOLVED_FROM,
        _PIPELINE_FALLBACK_USED,
        _REALTIME_FALLBACK_USED,
    )

_LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Argumenty CLI
# --------------------------------------------------------------------------------------
def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Uruchamia strategię trend-following D1 w trybie paper/testnet."
    )
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do CoreConfig")
    parser.add_argument(
        "--environment",
        default="binance_paper",
        help="Nazwa środowiska z pliku konfiguracyjnego (np. binance_paper)",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help="Nazwa strategii z sekcji strategies (domyślnie pobierana z konfiguracji środowiska)",
    )
    parser.add_argument(
        "--controller",
        default=None,
        help="Nazwa kontrolera runtime (domyślnie pobierana z konfiguracji środowiska)",
    )
    parser.add_argument(
        "--pipeline-module",
        action="append",
        dest="pipeline_modules",
        metavar="MODULE",
        help="Dodatkowy moduł z implementacją pipeline'u (można podać wielokrotnie)",
    )
    parser.add_argument(
        "--realtime-module",
        action="append",
        dest="realtime_modules",
        metavar="MODULE",
        help="Dodatkowy moduł z klasą DailyTrendRealtimeRunner (można podać wielokrotnie)",
    )
    parser.add_argument(
        "--print-runtime-modules",
        action="store_true",
        help="Wypisz aktualne moduły pipeline/realtime po zastosowaniu override'ów i zakończ",
    )
    parser.add_argument(
        "--print-risk-profiles",
        action="store_true",
        help="Wypisz zdefiniowane profile ryzyka wraz z limitami i zakończ",
    )
    parser.add_argument(
        "--runtime-plan-jsonl",
        default=None,
        help=(
            "Ścieżka pliku JSONL z wpisami planu runtime (snapshot modułów, profili ryzyka, "
            "override'ów). Wpis dodawany jest przed startem pipeline'u."
        ),
    )
    parser.add_argument(
        "--print-runtime-plan",
        action="store_true",
        help=(
            "Wypisz na stdout bieżący plan runtime (moduły, profil ryzyka, metadane audytu) "
            "i zakończ przed bootstrapem pipeline'u."
        ),
    )
    parser.add_argument(
        "--fail-on-security-warnings",
        action="store_true",
        help=(
            "Zakończ działanie, jeśli plan runtime zawiera ostrzeżenia bezpieczeństwa dotyczące plików "
            "telemetrii lub materiałów TLS."
        ),
    )
    parser.add_argument(
        "--risk-profile",
        default=None,
        help="Nazwa profilu ryzyka z sekcji risk_profiles (domyślnie używany profil przypisany do środowiska)",
    )
    parser.add_argument(
        "--history-bars",
        type=int,
        default=180,
        help="Liczba świec wykorzystywanych do analizy na starcie każdej iteracji",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=900.0,
        help="Jak często sprawdzać nowe sygnały (sekundy) w trybie ciągłym",
    )
    parser.add_argument(
        "--health-interval",
        type=float,
        default=3600.0,
        help="Interwał raportów health-check (sekundy)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Poziom logowania",
    )
    parser.add_argument(
        "--secret-namespace",
        default="dudzian.trading",
        help="Namespace używany przy zapisie sekretów w systemowym keychainie",
    )
    parser.add_argument(
        "--headless-passphrase",
        default=None,
        help="Hasło do szyfrowania magazynu sekretów w środowisku headless (Linux)",
    )
    parser.add_argument(
        "--headless-storage",
        default=None,
        help="Ścieżka pliku magazynu sekretów dla trybu headless",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Uruchom pojedynczą iterację i zakończ (np. do harmonogramu cron)",
    )
    parser.add_argument(
        "--paper-smoke",
        action="store_true",
        help="Uruchom test dymny strategii paper trading (backfill + pojedyncza iteracja)",
    )
    parser.add_argument(
        "--skip-paper-precheck",
        action="store_true",
        help="Pomiń automatyczny paper_precheck przed smoke testem (tylko debug)",
    )
    parser.add_argument(
        "--paper-precheck-fail-on-warnings",
        action="store_true",
        help="Traktuj ostrzeżenia paper_precheck jako błąd podczas uruchamiania smoke testu",
    )
    parser.add_argument(
        "--paper-precheck-audit-dir",
        default="audit/paper_precheck_reports",
        help=(
            "Katalog, do którego zapisujemy raport JSON z paper_precheck; "
            "podaj pusty napis, aby wyłączyć automatyczne logowanie."
        ),
    )
    parser.add_argument(
        "--paper-smoke-audit-log",
        default="docs/audit/paper_trading_log.md",
        help=(
            "Ścieżka pliku logu audytu paper trading; podaj pusty napis, aby pominąć wpis."
        ),
    )
    parser.add_argument(
        "--paper-smoke-operator",
        default=None,
        help=(
            "Nazwa operatora zapisywana w logu audytu (domyślnie PAPER_SMOKE_OPERATOR lub 'CI Agent')."
        ),
    )
    parser.add_argument(
        "--paper-smoke-json-log",
        default="docs/audit/paper_trading_log.jsonl",
        help=(
            "Ścieżka pliku JSONL z wpisami smoke testów; podaj pusty napis, aby wyłączyć logowanie."
        ),
    )
    parser.add_argument(
        "--paper-smoke-summary-json",
        default=None,
        help=(
            "Jeśli ustawione, zapisuje podsumowanie smoke testu w formacie JSON (np. do użytku w CI)."
        ),
    )
    parser.add_argument(
        "--paper-smoke-auto-publish",
        action="store_true",
        help=(
            "Po udanym smoke teście automatycznie opublikuj artefakty JSONL/ZIP"
            " z wykorzystaniem publish_paper_smoke_artifacts.py."
        ),
    )
    parser.add_argument(
        "--paper-smoke-auto-publish-required",
        action="store_true",
        help=(
            "Wymagaj powodzenia auto-publikacji artefaktów smoke (niepowodzenie lub pominięcie kończy run błędem)."
        ),
    )
    parser.add_argument(
        "--archive-smoke",
        action="store_true",
        help="Po zakończeniu smoke testu spakuj raport do archiwum ZIP z instrukcją audytu",
    )
    parser.add_argument(
        "--smoke-output",
        default=None,
        help="Opcjonalny katalog bazowy na raporty smoke testu; w środku powstanie podkatalog daily_trend_smoke_*.",
    )
    parser.add_argument(
        "--smoke-min-free-mb",
        type=float,
        default=None,
        help=(
            "Minimalna ilość wolnego miejsca (w MB) wymagana w katalogu raportu smoke; "
            "przy niższej wartości zgłosimy ostrzeżenie i oznaczymy raport."
        ),
    )
    parser.add_argument(
        "--smoke-fail-on-low-space",
        action="store_true",
        help=(
            "Traktuj ostrzeżenie o niskim wolnym miejscu jako błąd – po zapisaniu raportu "
            "zakończ proces kodem != 0."
        ),
    )
    parser.add_argument(
        "--date-window",
        default=None,
        help="Zakres dat w formacie START:END (np. 2024-01-01:2024-02-15) dla trybu --paper-smoke",
    )
    parser.add_argument(
        "--allow-live",
        action="store_true",
        help="Zezwól na uruchomienie na środowisku LIVE (domyślnie blokowane)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Zbuduj pipeline bez wykonywania iteracji (walidacja konfiguracji)",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------------------
# Pomocnicze I/O, formaty i konwersje
# --------------------------------------------------------------------------------------
def _create_secret_manager(args: argparse.Namespace) -> SecretManager:
    storage = create_default_secret_storage(
        namespace=args.secret_namespace,
        headless_passphrase=args.headless_passphrase,
        headless_path=args.headless_storage,
    )
    return SecretManager(storage, namespace=args.secret_namespace)


def _persist_precheck_audit(
    payload: Mapping[str, object],
    *,
    environment: str,
    audit_dir: Path | None,
    audit_clock: Callable[[], datetime] | None,
) -> Mapping[str, object] | None:
    """Zapisuje raport paper_precheck w katalogu audytu (jeśli skonfigurowano)."""

    if audit_dir is None:
        return None

    try:
        timestamp = audit_clock() if audit_clock else datetime.now(timezone.utc)
        metadata = paper_precheck_cli.persist_precheck_report(
            payload,
            environment_name=environment,
            base_dir=audit_dir,
            created_at=timestamp,
        )
    except Exception:  # noqa: BLE001
        _LOGGER.exception(
            "Nie udało się zapisać raportu paper_precheck do katalogu %s", audit_dir
        )
        return None

    if isinstance(payload, MutableMapping):
        payload.setdefault("audit_record", metadata)

    _LOGGER.info(
        "Raport paper_precheck zapisany do %s (sha256=%s)",
        metadata.get("path"),
        metadata.get("sha256"),
    )
    return metadata


def _run_paper_precheck_for_smoke(
    *,
    config_path: Path,
    environment: str,
    fail_on_warnings: bool,
    skip: bool,
    audit_dir: Path | None = None,
    audit_clock: Callable[[], datetime] | None = None,
) -> tuple[Mapping[str, object] | None, int, Mapping[str, object] | None]:
    """Uruchamia paper_precheck przed smoke testem i loguje wynik."""

    if skip:
        _LOGGER.warning(
            "Pomijam automatyczny paper_precheck dla %s (--skip-paper-precheck)", environment
        )
        payload: dict[str, object] = {
            "status": "skipped",
            "coverage_status": "skipped",
            "risk_status": "skipped",
            "skip_reason": "cli_flag",
        }
        metadata = _persist_precheck_audit(
            payload,
            environment=environment,
            audit_dir=audit_dir,
            audit_clock=audit_clock,
        )
        return payload, 0, metadata

    payload, exit_code = paper_precheck_cli.run_precheck(
        environment_name=environment,
        config_path=config_path,
        fail_on_warnings=fail_on_warnings,
    )

    status = str(payload.get("status", "unknown"))
    coverage_status = str(payload.get("coverage_status", "unknown"))
    risk_status = str(payload.get("risk_status", "unknown"))

    if payload.get("error_reason") == "environment_not_found":
        _LOGGER.error(
            "Paper pre-check: środowisko %s nie istnieje w konfiguracji", environment
        )
    elif payload.get("error_reason") == "invalid_min_ok_ratio":
        _LOGGER.error("Paper pre-check: parametr min_ok_ratio spoza zakresu 0-1")

    if exit_code != 0:
        _LOGGER.error(
            "Paper pre-check zakończony niepowodzeniem: status=%s coverage=%s risk=%s",  # noqa: G004
            status,
            coverage_status,
            risk_status,
        )
        return payload, exit_code, None

    _LOGGER.info(
        "Paper pre-check zakończony statusem %s (coverage=%s, risk=%s)",
        status,
        coverage_status,
        risk_status,
    )

    warning_sources: list[str] = []
    coverage_warnings = list(payload.get("coverage_warnings", []) or [])
    if coverage_status == "warning" or coverage_warnings:
        warning_sources.append("coverage")
    risk_payload = payload.get("risk")
    risk_warnings: Sequence[str] = ()
    if isinstance(risk_payload, Mapping):
        risk_warnings = risk_payload.get("warnings", []) or []
    if risk_status == "warning" or risk_warnings:
        warning_sources.append("risk")

    config_payload = payload.get("config") if isinstance(payload, Mapping) else None
    config_warnings = []
    if isinstance(config_payload, Mapping):
        config_warnings = list(config_payload.get("warnings", []) or [])
        if config_warnings:
            warning_sources.append("config")

    if warning_sources:
        _LOGGER.warning(
            "Paper pre-check zgłosił ostrzeżenia (%s) – sprawdź raport JSON przed kontynuacją",
            ", ".join(sorted(set(warning_sources))),
        )

    metadata = _persist_precheck_audit(
        payload,
        environment=environment,
        audit_dir=audit_dir,
        audit_clock=audit_clock,
    )

    return payload, exit_code, metadata


def _resolve_operator_name(raw: str | None) -> str:
    """Zwraca nazwę operatora do logu audytu."""

    candidates = [raw, os.environ.get("PAPER_SMOKE_OPERATOR"), "CI Agent"]
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            return text
    return "CI Agent"


def _prepare_precheck_audit_details(
    precheck_metadata: Mapping[str, object] | None,
    *,
    precheck_status: str | None,
    precheck_coverage_status: str | None,
    precheck_risk_status: str | None,
) -> tuple[list[str], dict[str, str]]:
    """Buduje listę notatek i słownik metadanych do logów audytowych."""

    notes: list[str] = []
    metadata: dict[str, str] = {}

    source = precheck_metadata if isinstance(precheck_metadata, Mapping) else {}

    report_path = source.get("path")
    if report_path:
        report_path_text = str(report_path)
        metadata["report_path"] = report_path_text
        notes.append(f"paper_precheck_report=`{report_path_text}`")

    report_hash = source.get("sha256")
    if report_hash:
        report_hash_text = str(report_hash)
        metadata["report_sha256"] = report_hash_text
        notes.append(f"paper_precheck_sha256=`{report_hash_text}`")

    created_at = source.get("created_at")
    if created_at:
        created_at_text = str(created_at)
        metadata["created_at"] = created_at_text
        notes.append(f"paper_precheck_created_at={created_at_text}")

    if precheck_status:
        status_text = str(precheck_status)
        metadata["status"] = status_text
        notes.append(f"paper_precheck_status={status_text}")

    if precheck_coverage_status:
        coverage_text = str(precheck_coverage_status)
        metadata["coverage_status"] = coverage_text
        notes.append(f"paper_precheck_coverage_status={coverage_text}")

    if precheck_risk_status:
        risk_text = str(precheck_risk_status)
        metadata["risk_status"] = risk_text
        notes.append(f"paper_precheck_risk_status={risk_text}")

    return notes, metadata


def _append_smoke_audit_entry(
    *,
    log_path: Path | None,
    timestamp: datetime,
    operator: str,
    environment: str,
    window: Mapping[str, str],
    summary_path: Path,
    summary_sha256: str,
    severity: str,
    precheck_metadata: Mapping[str, object] | None,
    precheck_status: str | None,
    precheck_coverage_status: str | None,
    precheck_risk_status: str | None,
) -> str | None:
    """Dopisuje nowy wiersz do logu audytu smoke testów paper tradingu."""

    if log_path is None:
        return None

    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        _LOGGER.warning(
            "Plik logu audytu %s nie istnieje – pomijam automatyczny wpis smoke testu.", log_path
        )
        return None
    except Exception:  # noqa: BLE001
        _LOGGER.exception("Nie udało się odczytać logu audytu %s", log_path)
        return None

    section_header = "## Sekcja B1 – Smoke testy paper tradingu"
    try:
        section_index = lines.index(section_header)
    except ValueError:
        _LOGGER.error("Nie znaleziono sekcji smoke testów w logu audytu %s", log_path)
        return None

    header_index = None
    for idx in range(section_index + 1, len(lines)):
        if lines[idx].startswith("|----"):
            header_index = idx
            break
    if header_index is None:
        _LOGGER.error("Nie znaleziono nagłówka tabeli smoke testów w %s", log_path)
        return None

    table_end_index = len(lines)
    id_pattern = re.compile(r"^\|\s*(S-(\d+))\b")
    max_id = 0
    for idx in range(header_index + 1, len(lines)):
        line = lines[idx]
        if not line.startswith("|"):
            table_end_index = idx
            break
        match = id_pattern.match(line)
        if match:
            try:
                number = int(match.group(2))
            except ValueError:
                continue
            max_id = max(max_id, number)

    new_id = f"S-{max_id + 1:04d}"
    window_start = str(window.get("start", "?"))
    window_end = str(window.get("end", "?"))
    window_text = f"{window_start} → {window_end}"
    summary_display = f"`{summary_path}`"
    summary_hash_display = f"`{summary_sha256}`"
    severity_text = severity.upper()

    notes, _ = _prepare_precheck_audit_details(
        precheck_metadata,
        precheck_status=precheck_status,
        precheck_coverage_status=precheck_coverage_status,
        precheck_risk_status=precheck_risk_status,
    )
    note_text = "; ".join(notes) if notes else "-"

    new_row = (
        f"| {new_id} | {timestamp.isoformat()} | {operator} | {environment} | {window_text} | "
        f"{summary_display} | {summary_hash_display} | {severity_text} | {note_text} |"
    )

    lines.insert(table_end_index, new_row)
    updated_content = "\n".join(lines) + "\n"
    try:
        log_path.write_text(updated_content, encoding="utf-8")
    except Exception:  # noqa: BLE001
        _LOGGER.exception("Nie udało się zapisać wpisu smoke testu do logu %s", log_path)
        return None

    _LOGGER.info(
        "Dodano wpis %s do logu audytu smoke testów (%s)", new_id, log_path
    )
    return new_id


def _append_smoke_json_log_entry(
    *,
    json_path: Path | None,
    timestamp: datetime,
    operator: str,
    environment: str,
    window: Mapping[str, str],
    summary_path: Path,
    summary_sha256: str,
    severity: str,
    precheck_metadata: Mapping[str, object] | None,
    precheck_payload: Mapping[str, object] | None,
    precheck_status: str | None,
    precheck_coverage_status: str | None,
    precheck_risk_status: str | None,
    markdown_entry_id: str | None,
) -> Mapping[str, object] | None:
    """Zapisuje wpis smoke testu w dzienniku JSONL (jeśli skonfigurowano)."""

    if json_path is None:
        return None

    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        _LOGGER.exception("Nie udało się utworzyć katalogu dla %s", json_path)
        return None

    timestamp_utc = timestamp.astimezone(timezone.utc)
    record_id = f"J-{timestamp_utc.strftime('%Y%m%dT%H%M%S')}-{summary_sha256[:8]}"

    notes, metadata_details = _prepare_precheck_audit_details(
        precheck_metadata,
        precheck_status=precheck_status,
        precheck_coverage_status=precheck_coverage_status,
        precheck_risk_status=precheck_risk_status,
    )

    if isinstance(precheck_payload, Mapping):
        try:
            sanitized_payload = json.loads(json.dumps(precheck_payload))
        except TypeError:
            sanitized_payload = None
    else:
        sanitized_payload = None

    record = {
        "record_id": record_id,
        "markdown_entry_id": markdown_entry_id,
        "timestamp": timestamp_utc.isoformat(),
        "operator": operator,
        "environment": environment,
        "window_start": str(window.get("start", "?")),
        "window_end": str(window.get("end", "?")),
        "summary_path": str(summary_path),
        "summary_sha256": summary_sha256,
        "severity": severity.upper(),
        "precheck_metadata": metadata_details,
        "precheck_payload": sanitized_payload,
        "notes": notes,
    }

    try:
        with json_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        _LOGGER.exception("Nie udało się zapisać wpisu JSONL do %s", json_path)
        return None

    _LOGGER.info(
        "Dodano wpis JSON smoke testu (%s) do %s", record_id, json_path
    )
    return record


def _build_smoke_summary_payload(
    *,
    environment: str,
    timestamp: datetime,
    operator: str,
    window: Mapping[str, str],
    report_dir: Path,
    summary_path: Path,
    summary_sha256: str,
    severity: str,
    storage_context: Mapping[str, str] | None,
    precheck_status: str | None,
    precheck_coverage_status: str | None,
    precheck_risk_status: str | None,
    precheck_payload: Mapping[str, object] | None,
    json_log_path: Path | None,
    json_record: Mapping[str, object] | None,
    json_sync_result: PaperSmokeJsonSyncResult | None,
    archive_path: Path | None,
    archive_upload_result: SmokeArchiveUploadResult | None,
    publish_result: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Buduje podsumowanie smoke testu dla pipeline'u CI."""

    timestamp_utc = timestamp.astimezone(timezone.utc)
    payload: MutableMapping[str, object] = {
        "environment": environment,
        "timestamp": timestamp_utc.isoformat(),
        "operator": operator,
        "severity": severity.upper(),
        "window": dict(window),
        "report": {
            "directory": str(report_dir),
            "summary_path": str(summary_path),
            "summary_sha256": summary_sha256,
        },
    }

    if storage_context:
        payload["storage"] = dict(storage_context)

    precheck_info: MutableMapping[str, object] = {
        "status": precheck_status or "unknown",
        "coverage_status": precheck_coverage_status or "unknown",
        "risk_status": precheck_risk_status or "unknown",
    }

    if isinstance(precheck_payload, Mapping):
        try:
            sanitized_precheck = json.loads(json.dumps(precheck_payload))
        except TypeError:
            sanitized_precheck = None
        if sanitized_precheck is not None:
            precheck_info["payload"] = sanitized_precheck

    payload["precheck"] = precheck_info

    if json_log_path is not None or json_record is not None or json_sync_result is not None:
        json_info: MutableMapping[str, object] = {}
        if json_log_path is not None:
            json_info["path"] = str(json_log_path)
        if json_record is not None:
            try:
                sanitized_record = json.loads(json.dumps(json_record))
            except TypeError:
                sanitized_record = {str(key): str(value) for key, value in json_record.items()}
            json_info["record"] = sanitized_record
            record_id_value = json_record.get("record_id")
            if record_id_value is not None:
                json_info["record_id"] = str(record_id_value)
        if json_sync_result is not None:
            json_info["sync"] = {
                "backend": json_sync_result.backend,
                "location": json_sync_result.location,
                "metadata": dict(json_sync_result.metadata),
            }
        payload["json_log"] = json_info

    if archive_path is not None or archive_upload_result is not None:
        archive_info: MutableMapping[str, object] = {}
        if archive_path is not None:
            archive_info["path"] = str(archive_path)
        if archive_upload_result is not None:
            archive_info["upload"] = {
                "backend": archive_upload_result.backend,
                "location": archive_upload_result.location,
                "metadata": dict(archive_upload_result.metadata),
            }
        payload["archive"] = archive_info

    if publish_result is not None:
        try:
            payload["publish"] = json.loads(json.dumps(publish_result))
        except TypeError:
            payload["publish"] = {str(key): str(value) for key, value in publish_result.items()}

    return payload


def _write_smoke_summary_json(path: Path, payload: Mapping[str, object]) -> None:
    """Zapisuje podsumowanie smoke testu do pliku JSON."""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _coerce_exit_code(value: object) -> int | None:
    """Konwertuje wartość exit code na liczbę całkowitą, jeśli to możliwe."""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensywne
        return None


def _truncate_text(value: str, *, limit: int) -> str:
    """Zwraca tekst obcięty do limitu znaków z wielokropkiem."""

    if limit <= 0:
        return ""

    text = value.strip()
    if len(text) <= limit:
        return text

    return text[: max(limit - 1, 1)] + "…"


def _normalize_publish_result(
    raw_result: Mapping[str, object] | None,
    *,
    exit_code: int | None,
    required: bool,
) -> Mapping[str, object]:
    """Ujednolica wynik auto-publikacji na potrzeby raportów i alertów."""

    normalized: dict[str, object] = {}
    if isinstance(raw_result, Mapping):
        for key, value in raw_result.items():
            normalized[str(key)] = value

    if not normalized.get("status"):
        normalized["status"] = "unknown"

    coerced_exit = _coerce_exit_code(normalized.get("exit_code"))
    if coerced_exit is not None:
        normalized["exit_code"] = coerced_exit
    elif "exit_code" not in normalized:
        normalized["exit_code"] = exit_code
    else:
        normalized["exit_code"] = normalized.get("exit_code")

    normalized["required"] = bool(required)

    return normalized


def _is_publish_result_ok(publish_result: Mapping[str, object] | None) -> bool:
    """Sprawdza, czy auto-publikacja zakończyła się powodzeniem."""

    if not isinstance(publish_result, Mapping):
        return False

    exit_code = _coerce_exit_code(publish_result.get("exit_code"))
    if exit_code is not None and exit_code != 0:
        return False

    status = str(publish_result.get("status", "")).strip().lower()
    return status == "ok"


def _append_publish_context(
    context: MutableMapping[str, str], publish_result: Mapping[str, object] | None
) -> None:
    """Dodaje do kontekstu alertów/compliance informacje o auto-publikacji."""

    if not isinstance(publish_result, Mapping):
        return

    status = str(publish_result.get("status", "unknown"))
    context["paper_smoke_publish_status"] = status

    exit_code = _coerce_exit_code(publish_result.get("exit_code"))
    if exit_code is not None:
        context["paper_smoke_publish_exit_code"] = str(exit_code)

    required_value = publish_result.get("required")
    if required_value is not None:
        context["paper_smoke_publish_required"] = "true" if bool(required_value) else "false"

    reason = publish_result.get("reason")
    if reason:
        context["paper_smoke_publish_reason"] = str(reason)

    if publish_result.get("status") != "ok":
        stdout_raw = publish_result.get("raw_stdout")
        if stdout_raw:
            context["paper_smoke_publish_stdout_snippet"] = _truncate_text(
                str(stdout_raw), limit=_CONTEXT_SNIPPET_MAX_LEN
            )

        stderr_raw = publish_result.get("raw_stderr")
        if stderr_raw:
            context["paper_smoke_publish_stderr_snippet"] = _truncate_text(
                str(stderr_raw), limit=_CONTEXT_SNIPPET_MAX_LEN
            )

    json_step = publish_result.get("json_sync")
    if isinstance(json_step, Mapping):
        context["paper_smoke_publish_json_status"] = str(json_step.get("status", "unknown"))
        backend_value = json_step.get("backend")
        if backend_value:
            context["paper_smoke_publish_json_backend"] = str(backend_value)
        location_value = json_step.get("location")
        if location_value:
            context["paper_smoke_publish_json_location"] = str(location_value)
        metadata = json_step.get("metadata")
        if isinstance(metadata, Mapping):
            for meta_key, meta_val in metadata.items():
                context[f"paper_smoke_publish_json_{meta_key}"] = str(meta_val)

    archive_step = publish_result.get("archive_upload")
    if isinstance(archive_step, Mapping):
        context["paper_smoke_publish_archive_status"] = str(
            archive_step.get("status", "unknown")
        )
        backend_value = archive_step.get("backend")
        if backend_value:
            context["paper_smoke_publish_archive_backend"] = str(backend_value)
        location_value = archive_step.get("location")
        if location_value:
            context["paper_smoke_publish_archive_location"] = str(location_value)
        metadata = archive_step.get("metadata")
        if isinstance(metadata, Mapping):
            for meta_key, meta_val in metadata.items():
                context[f"paper_smoke_publish_archive_{meta_key}"] = str(meta_val)


def _sync_smoke_json_log(
    *,
    json_sync_cfg,
    json_log_path: Path | None,
    environment: str,
    record_id: str | None,
    timestamp: datetime,
    secret_manager: SecretManager | None,
):
    if json_sync_cfg is None or json_log_path is None:
        return None
    record_id = record_id or ""
    try:
        synchronizer = PaperSmokeJsonSynchronizer(
            json_sync_cfg,
            secret_manager=secret_manager,
        )
        result = synchronizer.sync(
            json_log_path,
            environment=environment,
            record_id=record_id,
            timestamp=timestamp,
        )
        metadata = dict(result.metadata)
        version_info = metadata.get("version_id")
        receipt = metadata.get("ack_request_id") or metadata.get("ack_mechanism")
        log_suffix = []
        if version_info:
            log_suffix.append(f"version_id={version_info}")
        if receipt:
            log_suffix.append(f"receipt={receipt}")
        suffix_text = ", ".join(log_suffix)
        if suffix_text:
            suffix_text = f" ({suffix_text})"
        _LOGGER.info(
            "Zsynchronizowano dziennik JSONL smoke testów: backend=%s, location=%s%s",
            result.backend,
            result.location,
            suffix_text,
        )
        return result
    except Exception:  # noqa: BLE001
        _LOGGER.exception("Nie udało się zsynchronizować dziennika JSONL smoke testów")
        return None


def _auto_publish_smoke_artifacts(
    *,
    config_path: Path,
    environment: str,
    report_dir: Path,
    json_log_path: Path | None,
    summary_json_path: Path | None,
    archive_path: Path | None,
    record_id: str | None,
    skip_json_sync: bool,
    skip_archive_upload: bool,
    dry_run: bool,
) -> tuple[int, Mapping[str, object] | None]:
    """Uruchamia publish_paper_smoke_artifacts.py i zwraca kod wyjścia oraz wynik."""

    script_path = Path(__file__).with_name("publish_paper_smoke_artifacts.py")
    if not script_path.exists():
        _LOGGER.error("Brak skryptu publish_paper_smoke_artifacts.py obok run_daily_trend.py")
        return 1, {"status": "error", "reason": "missing_script"}

    cmd: list[str] = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
        "--environment",
        environment,
        "--report-dir",
        str(report_dir),
        "--json",
    ]

    if json_log_path is not None:
        cmd.extend(["--json-log", str(json_log_path)])
    if summary_json_path is not None and summary_json_path.exists():
        cmd.extend(["--summary-json", str(summary_json_path)])
    if archive_path is not None and archive_path.exists():
        cmd.extend(["--archive", str(archive_path)])
    if record_id:
        cmd.extend(["--record-id", record_id])
    if skip_json_sync:
        cmd.append("--skip-json-sync")
    if skip_archive_upload:
        cmd.append("--skip-archive-upload")
    if dry_run:
        cmd.append("--dry-run")

    _LOGGER.info(
        "Publikuję artefakty smoke testu przy pomocy publish_paper_smoke_artifacts.py (auto)",
    )

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:  # pragma: no cover - uruchomienie interpretatora może zawieść na systemach CI
        _LOGGER.exception("Nie udało się uruchomić publish_paper_smoke_artifacts.py")
        return 1, {"status": "error", "reason": "exec_failed"}

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    trimmed_stdout = _truncate_text(stdout, limit=_RAW_OUTPUT_MAX_LEN) if stdout else ""
    trimmed_stderr = _truncate_text(stderr, limit=_RAW_OUTPUT_MAX_LEN) if stderr else ""
    if stderr:
        _LOGGER.debug("publish_paper_smoke_artifacts stderr: %s", stderr)

    payload: Mapping[str, object] | None = None
    if stdout:
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            _LOGGER.error(
                "Nie udało się sparsować wyjścia publish_paper_smoke_artifacts jako JSON: %s",
                stdout,
            )
            payload = {"status": "error", "reason": "invalid_json"}
        else:
            if isinstance(parsed, Mapping):
                payload = parsed
            else:
                payload = {"status": "error", "reason": "invalid_payload"}

    if payload is None:
        payload_dict: dict[str, object] = {"status": "unknown"}
    elif isinstance(payload, Mapping):
        payload_dict = dict(payload)
        payload_dict.setdefault("status", "unknown")
    else:  # pragma: no cover - defensywnie dla nietypowych wyników
        payload_dict = {"status": "unknown"}
    payload_dict["exit_code"] = completed.returncode

    if payload_dict.get("status") != "ok":
        if trimmed_stdout and "raw_stdout" not in payload_dict:
            payload_dict["raw_stdout"] = trimmed_stdout
        if trimmed_stderr and "raw_stderr" not in payload_dict:
            payload_dict["raw_stderr"] = trimmed_stderr

    if completed.returncode == 0:
        _LOGGER.info("Publikacja artefaktów smoke zakończona powodzeniem")
    else:
        _LOGGER.error(
            "Publikacja artefaktów smoke zakończona błędem (code=%s)",
            completed.returncode,
        )

    return completed.returncode, payload_dict


def _log_order_results(results: Iterable[OrderResult]) -> None:
    for result in results:
        _LOGGER.info(
            "Zlecenie zrealizowane: id=%s status=%s qty=%s avg_price=%s",
            result.order_id,
            result.status,
            result.filled_quantity,
            result.avg_price,
        )


def _parse_iso_date(value: str, *, is_end: bool) -> datetime:
    text = value.strip()
    if not text:
        raise ValueError("wartość daty nie może być pusta")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:  # pragma: no cover
        raise ValueError(f"nieprawidłowy format daty: {text}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    if "T" not in text and " " not in text:
        if is_end:
            parsed = parsed + timedelta(days=1) - timedelta(milliseconds=1)
    return parsed


def _resolve_date_window(arg: str | None, *, default_days: int = 30) -> tuple[int, int, Mapping[str, str]]:
    if not arg:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=default_days)
    else:
        parts = arg.split(":", maxsplit=1)
        if len(parts) != 2:
            raise ValueError("zakres musi mieć format START:END")
        start_dt = _parse_iso_date(parts[0], is_end=False)
        end_dt = _parse_iso_date(parts[1], is_end=True)
    if start_dt > end_dt:
        raise ValueError("data początkowa jest późniejsza niż końcowa")
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    return start_ms, end_ms, {"start": start_dt.isoformat(), "end": end_dt.isoformat()}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _as_int(value: object) -> int | None:
    float_value = _as_float(value)
    if float_value is None:
        return None
    try:
        return int(float_value)
    except (TypeError, ValueError):  # pragma: no cover
        return None


def _format_money(value: float, *, decimals: int = 2) -> str:
    formatted = f"{value:,.{decimals}f}"
    return formatted.replace(",", " ")


def _format_percentage(value: float | None, *, decimals: int = 2) -> str:
    if value is None:
        return "n/d"
    return f"{value * 100:.{decimals}f}%"


def _normalize_position_entry(symbol: str, payload: Mapping[str, object]) -> tuple[float, str] | None:
    """Buduje opis pojedynczej pozycji do raportu tekstowego."""
    notional = _as_float(payload.get("notional"))
    if notional is None or notional <= 0:
        return None
    side = str(payload.get("side", "")).strip().upper() or "?"
    description = f"{symbol}: {side} {_format_money(notional)}"
    return notional, description


# --------------------------------------------------------------------------------------
# Metryki ledger / PnL FIFO long/short
# --------------------------------------------------------------------------------------
def _compute_ledger_metrics(ledger_entries: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    counts: MutableMapping[str, int] = {"buy": 0, "sell": 0}
    other_counts: MutableMapping[str, int] = {}
    notionals: MutableMapping[str, float] = {"buy": 0.0, "sell": 0.0}
    other_notionals: MutableMapping[str, float] = {}
    total_fees = 0.0
    last_position_value: float | None = None
    per_symbol: dict[str, dict[str, float]] = {}
    pnl_trackers: dict[str, dict[str, object]] = {}
    realized_pnl_total = 0.0
    eps = 1e-9

    for entry in ledger_entries:
        if not isinstance(entry, Mapping):
            continue

        side = str(entry.get("side", "")).lower()
        quantity = _as_float(entry.get("quantity")) or 0.0
        price = _as_float(entry.get("price")) or 0.0
        notional_value = abs(quantity) * max(price, 0.0)
        abs_quantity = abs(quantity)

        if side in ("buy", "sell"):
            counts[side] += 1
            notionals[side] += notional_value
        else:
            side_key = side or "unknown"
            other_counts[side_key] = other_counts.get(side_key, 0) + 1
            other_notionals[side_key] = other_notionals.get(side_key, 0.0) + notional_value

        fee_value = _as_float(entry.get("fee"))
        if fee_value is not None:
            total_fees += fee_value

        position_value = _as_float(entry.get("position_value"))
        if position_value is not None:
            last_position_value = position_value

        symbol = entry.get("symbol")
        if symbol:
            symbol_key = str(symbol)
            stats = per_symbol.setdefault(
                symbol_key,
                {
                    "orders": 0,
                    "buy_orders": 0,
                    "sell_orders": 0,
                    "other_orders": 0,
                    "buy_quantity": 0.0,
                    "sell_quantity": 0.0,
                    "other_quantity": 0.0,
                    "buy_notional": 0.0,
                    "sell_notional": 0.0,
                    "other_notional": 0.0,
                    "total_notional": 0.0,
                    "net_quantity": 0.0,
                    "fees": 0.0,
                    "realized_pnl": 0.0,
                },
            )

            stats["orders"] += 1
            if side == "buy":
                stats["buy_orders"] += 1
                stats["buy_quantity"] += quantity
                stats["buy_notional"] += notional_value
                stats["net_quantity"] += quantity
            elif side == "sell":
                stats["sell_orders"] += 1
                stats["sell_quantity"] += quantity
                stats["sell_notional"] += notional_value
                stats["net_quantity"] -= quantity
            else:
                stats["other_orders"] += 1
                stats["other_quantity"] += quantity
                stats["other_notional"] += notional_value

            stats["total_notional"] = (
                stats["buy_notional"] + stats["sell_notional"] + stats["other_notional"]
            )

            if fee_value is not None:
                stats["fees"] += fee_value

            if position_value is not None:
                stats["last_position_value"] = position_value

            tracker = pnl_trackers.setdefault(
                symbol_key,
                {
                    "long_lots": deque(),
                    "short_lots": deque(),
                    "realized_pnl": 0.0,
                },
            )
            long_lots: deque[tuple[float, float]] = tracker["long_lots"]  # type: ignore[assignment]
            short_lots: deque[tuple[float, float]] = tracker["short_lots"]  # type: ignore[assignment]
            realized_symbol: float = tracker["realized_pnl"]  # type: ignore[assignment]

            remaining_qty = abs_quantity

            if side == "buy":
                while remaining_qty > eps and short_lots:
                    lot_qty, lot_price = short_lots[0]
                    matched = min(remaining_qty, lot_qty)
                    realized_symbol += (lot_price - price) * matched
                    lot_qty -= matched
                    remaining_qty -= matched
                    if lot_qty <= eps:
                        short_lots.popleft()
                    else:
                        short_lots[0] = (lot_qty, lot_price)
                if remaining_qty > eps:
                    long_lots.append((remaining_qty, price))
            elif side == "sell":
                while remaining_qty > eps and long_lots:
                    lot_qty, lot_price = long_lots[0]
                    matched = min(remaining_qty, lot_qty)
                    realized_symbol += (price - lot_price) * matched
                    lot_qty -= matched
                    remaining_qty -= matched
                    if lot_qty <= eps:
                        long_lots.popleft()
                    else:
                        long_lots[0] = (lot_qty, lot_price)
                if remaining_qty > eps:
                    short_lots.append((remaining_qty, price))

            tracker["realized_pnl"] = realized_symbol
            stats["realized_pnl"] = realized_symbol
            previous_realized = tracker.get("_realized_accumulator", 0.0)
            realized_pnl_total += realized_symbol - float(previous_realized)
            tracker["_realized_accumulator"] = realized_symbol

    total_notional = sum(notionals.values()) + sum(other_notionals.values())

    side_counts: MutableMapping[str, int] = {
        "buy": counts.get("buy", 0),
        "sell": counts.get("sell", 0),
    }
    for key, value in other_counts.items():
        if value:
            side_counts[key] = value

    notional_payload: MutableMapping[str, float] = {
        "buy": notionals.get("buy", 0.0),
        "sell": notionals.get("sell", 0.0),
    }
    for key, value in other_notionals.items():
        if value:
            notional_payload[key] = value
    notional_payload["total"] = total_notional

    metrics: dict[str, object] = {
        "side_counts": dict(side_counts),
        "notional": dict(notional_payload),
        "total_fees": total_fees,
    }
    metrics["realized_pnl_total"] = realized_pnl_total
    if last_position_value is not None:
        metrics["last_position_value"] = last_position_value
    if per_symbol:
        metrics["per_symbol"] = {
            symbol: {k: (float(v) if isinstance(v, float) else v) for k, v in stats.items()}
            for symbol, stats in per_symbol.items()
        }
    return metrics


# --------------------------------------------------------------------------------------
# Raport smoke
# --------------------------------------------------------------------------------------
def _export_smoke_report(
    *,
    report_dir: Path,
    results: Sequence[OrderResult],
    ledger: Iterable[Mapping[str, object]],
    window: Mapping[str, str],
    environment: str,
    alert_snapshot: Mapping[str, Mapping[str, str]],
    risk_state: Mapping[str, object] | None = None,
    data_checks: Mapping[str, object] | None = None,
    storage_info: Mapping[str, object] | None = None,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)

    # Zapis ledger.jsonl
    ledger_entries = list(ledger)
    ledger_path = report_dir / "ledger.jsonl"
    with ledger_path.open("w", encoding="utf-8") as handle:
        for entry in ledger_entries:
            json.dump(entry, handle, ensure_ascii=False)
            handle.write("\n")

    # Metryki
    metrics = _compute_ledger_metrics(ledger_entries)

    # Podsumowanie
    summary: dict[str, object] = {
        "environment": environment,
        "window": dict(window),
        "orders": [
            {
                "order_id": result.order_id,
                "status": result.status,
                "filled_quantity": result.filled_quantity,
                "avg_price": result.avg_price,
            }
            for result in results
        ],
        "ledger_entries": len(ledger_entries),
        "metrics": metrics,
        "alert_snapshot": {channel: dict(data) for channel, data in alert_snapshot.items()},
    }
    if risk_state:
        summary["risk_state"] = dict(risk_state)
    if data_checks:
        summary["data_checks"] = json.loads(json.dumps(data_checks))
    if storage_info:
        summary["storage"] = json.loads(json.dumps(storage_info))

    # Zapis summary.json
    summary_path = report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_path


def _write_smoke_readme(report_dir: Path) -> Path:
    readme_path = report_dir / "README.txt"
    readme_text = (
        "Daily Trend – smoke test paper trading\n"
        "======================================\n\n"
        "Ten katalog zawiera artefakty pojedynczego uruchomienia trybu --paper-smoke.\n"
        "Na potrzeby audytu:\n\n"
        "1. Zweryfikuj hash SHA-256 pliku summary.json zapisany w logu CLI oraz w alertach.\n"
        "2. Przepisz treść summary.txt do dziennika audytowego (docs/audit/paper_trading_log.md).\n"
        "3. Zabezpiecz ledger.jsonl (pełna historia decyzji) w repozytorium operacyjnym.\n"
        "4. Zarchiwizowany plik ZIP można przechowywać w sejfie audytu przez min. 24 miesiące.\n"
    )
    readme_path.write_text(readme_text + "\n", encoding="utf-8")
    return readme_path


def _archive_smoke_report(report_dir: Path) -> Path:
    archive_path_str = shutil.make_archive(str(report_dir), "zip", root_dir=report_dir)
    return Path(archive_path_str)


_MEGABYTE = 1024 * 1024


def _collect_storage_health(directory: Path, *, min_free_mb: float | None) -> Mapping[str, object]:
    """Zwraca informacje o stanie przestrzeni dyskowej dla raportu smoke."""
    info: dict[str, object] = {"directory": str(directory)}
    threshold_mb = float(min_free_mb) if min_free_mb is not None else None
    if threshold_mb is not None and threshold_mb < 0:
        threshold_mb = 0.0
    threshold_bytes = int(threshold_mb * _MEGABYTE) if threshold_mb is not None else None

    try:
        usage = shutil.disk_usage(directory)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("Nie udało się odczytać informacji o wolnym miejscu dla %s: %s", directory, exc)
        info.update({"status": "unknown", "error": str(exc)})
        if threshold_bytes is not None:
            info["threshold_bytes"] = threshold_bytes
            info["threshold_mb"] = threshold_bytes / _MEGABYTE
        return info

    free_bytes = int(usage.free)
    total_bytes = int(usage.total)
    info.update(
        {
            "status": "ok",
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
            "free_mb": free_bytes / _MEGABYTE,
            "total_mb": total_bytes / _MEGABYTE,
        }
    )

    if threshold_bytes is not None:
        info["threshold_bytes"] = threshold_bytes
        info["threshold_mb"] = threshold_bytes / _MEGABYTE
        if free_bytes < threshold_bytes:
            info["status"] = "low"
            _LOGGER.warning(
                "Wolne miejsce w katalogu raportu %s: %.2f MB (< %.2f MB)",
                directory,
                free_bytes / _MEGABYTE,
                threshold_bytes / _MEGABYTE,
            )

    return info


def _collect_required_intervals(
    pipeline: Any,
    *,
    symbols: Sequence[str],
) -> tuple[str, ...]:
    """Zwraca uporządkowaną listę interwałów wymaganych do smoke testu."""
    intervals: list[str] = []
    seen: set[str] = set()

    def _add_interval(value: str | None) -> None:
        normalized = _normalize_interval_token(value)
        if not normalized:
            return
        if normalized in seen:
            return
        seen.add(normalized)
        intervals.append(value or normalized)

    primary_interval = getattr(getattr(pipeline, "controller", None), "interval", None)
    if primary_interval:
        _add_interval(primary_interval)

    bootstrap = getattr(pipeline, "bootstrap", None)
    if bootstrap is None:
        return tuple(intervals)

    environment_cfg = getattr(bootstrap, "environment", None)
    core_config = getattr(bootstrap, "core_config", None)
    if environment_cfg is None or core_config is None:
        return tuple(intervals)

    universe_name = getattr(environment_cfg, "instrument_universe", None)
    exchange_name = getattr(environment_cfg, "exchange", None)
    if not universe_name or not exchange_name:
        return tuple(intervals)

    if not hasattr(core_config, "instrument_universes"):
        return tuple(intervals)

    try:
        universe = core_config.instrument_universes[universe_name]
    except Exception:  # noqa: BLE001
        return tuple(intervals)

    tracked_symbols = {str(symbol).lower() for symbol in symbols}
    for instrument in getattr(universe, "instruments", ()):  # type: ignore[attr-defined]
        symbol = instrument.exchange_symbols.get(exchange_name) if instrument else None
        if symbol and symbol.lower() in tracked_symbols:
            for window in getattr(instrument, "backfill_windows", ()):  # type: ignore[attr-defined]
                _add_interval(getattr(window, "interval", None))

    return tuple(intervals)


def _prepare_smoke_report_directory(target: str | None) -> Path:
    """Zwraca katalog na raport smoke testu, tworząc go jeśli potrzeba."""
    if target:
        base_dir = Path(target).expanduser()
        base_dir.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="daily_trend_smoke_", dir=str(base_dir)))
    return Path(tempfile.mkdtemp(prefix="daily_trend_smoke_"))


def _render_smoke_summary(*, summary: Mapping[str, object], summary_sha256: str) -> str:
    environment = str(summary.get("environment", "unknown"))
    window = summary.get("window", {})
    if isinstance(window, Mapping):
        start = str(window.get("start", "?"))
        end = str(window.get("end", "?"))
    else:  # pragma: no cover
        start = end = "?"

    orders = summary.get("orders", [])
    orders_count = len(orders) if isinstance(orders, Sequence) else 0
    ledger_entries = summary.get("ledger_entries", 0)
    try:
        ledger_entries = int(ledger_entries)
    except Exception:  # noqa: BLE001
        ledger_entries = 0

    alert_snapshot = summary.get("alert_snapshot", {})
    alert_lines: list[str] = []
    if isinstance(alert_snapshot, Mapping):
        for channel, data in alert_snapshot.items():
            status = "UNKNOWN"
            detail: str | None = None
            if isinstance(data, Mapping):
                raw_status = data.get("status")
                if raw_status is not None:
                    status = str(raw_status).upper()
                raw_detail = data.get("detail")
                if raw_detail:
                    detail = str(raw_detail)
            channel_name = str(channel)
            if detail:
                alert_lines.append(f"{channel_name}: {status} ({detail})")
            else:
                alert_lines.append(f"{channel_name}: {status}")
    if not alert_lines:
        alert_lines.append("brak danych o kanałach alertów")

    metrics_lines: list[str] = []
    metrics = summary.get("metrics")
    if isinstance(metrics, Mapping):
        side_counts = metrics.get("side_counts")
        if isinstance(side_counts, Mapping):
            buy_count = _as_int(side_counts.get("buy")) or 0
            sell_count = _as_int(side_counts.get("sell")) or 0
            if buy_count or sell_count:
                metrics_lines.append(f"Zlecenia BUY/SELL: {buy_count}/{sell_count}")
            other_sides = [
                f"{str(name).upper()}: {_as_int(value) or 0}"
                for name, value in side_counts.items()
                if str(name).lower() not in {"buy", "sell"}
            ]
            if other_sides:
                metrics_lines.append("Inne strony: " + ", ".join(other_sides))

        notionals = metrics.get("notional")
        if isinstance(notionals, Mapping) and notionals:
            buy_notional = _as_float(notionals.get("buy")) or 0.0
            sell_notional = _as_float(notionals.get("sell")) or 0.0
            total_notional = _as_float(notionals.get("total")) or (buy_notional + sell_notional)
            metrics_lines.append(
                "Wolumen BUY: {buy} | SELL: {sell} | Razem: {total}".format(
                    buy=_format_money(buy_notional),
                    sell=_format_money(sell_notional),
                    total=_format_money(total_notional),
                )
            )
            other_notional_lines = [
                f"{str(name).upper()}: {_format_money(_as_float(value) or 0.0)}"
                for name, value in notionals.items()
                if str(name).lower() not in {"buy", "sell", "total"}
            ]
            if other_notional_lines:
                metrics_lines.append("Wolumen inne: " + "; ".join(other_notional_lines))

        total_fees = _as_float(metrics.get("total_fees"))
        if total_fees is not None:
            metrics_lines.append(f"Łączne opłaty: {_format_money(total_fees, decimals=4)}")

        realized_total = _as_float(metrics.get("realized_pnl_total"))
        if realized_total is not None:
            metrics_lines.append(f"Realizowany PnL (brutto): {_format_money(realized_total)}")

        last_position = _as_float(metrics.get("last_position_value"))
        if last_position is not None:
            metrics_lines.append(f"Ostatnia wartość pozycji: {_format_money(last_position)}")

        per_symbol = metrics.get("per_symbol")
        if isinstance(per_symbol, Mapping):
            symbol_lines: list[tuple[float, str]] = []
            for symbol, payload in per_symbol.items():
                if not isinstance(payload, Mapping):
                    continue

                total_notional_sym = _as_float(payload.get("total_notional")) or 0.0
                orders_sym = _as_int(payload.get("orders")) or 0
                fees_value = _as_float(payload.get("fees"))
                net_quantity = _as_float(payload.get("net_quantity"))
                last_symbol_value = _as_float(payload.get("last_position_value"))
                realized_symbol = _as_float(payload.get("realized_pnl"))

                if not (
                    orders_sym
                    or total_notional_sym
                    or (fees_value is not None and fees_value)
                    or (net_quantity is not None and abs(net_quantity) > 1e-9)
                    or (last_symbol_value is not None and last_symbol_value > 0)
                    or (realized_symbol is not None and abs(realized_symbol) > 1e-9)
                ):
                    continue

                parts = [f"{symbol}: zlecenia {orders_sym}"]
                if total_notional_sym:
                    parts.append(f"wolumen {_format_money(total_notional_sym)}")
                if fees_value is not None:
                    parts.append(f"opłaty {_format_money(fees_value, decimals=4)}")
                if net_quantity is not None and abs(net_quantity) > 1e-6:
                    parts.append(f"netto {net_quantity:+.4f}")
                if last_symbol_value is not None and last_symbol_value > 0:
                    parts.append(f"wartość {_format_money(last_symbol_value)}")
                if realized_symbol is not None and abs(realized_symbol) > 1e-6:
                    parts.append(f"PnL {_format_money(realized_symbol)}")

                symbol_lines.append((total_notional_sym, ", ".join(parts)))

            if symbol_lines:
                symbol_lines.sort(key=lambda item: item[0], reverse=True)
                top_lines = [item[1] for item in symbol_lines[:3]]
                metrics_lines.append("Instrumenty: " + "; ".join(top_lines))

    # Opcjonalne linie o stanie ryzyka
    risk_lines: list[str] = []
    risk_state = summary.get("risk_state")
    if isinstance(risk_state, Mapping) and risk_state:
        profile_name = str(risk_state.get("profile", "unknown"))
        risk_lines.append(f"Profil ryzyka: {profile_name}")

        active_positions = _as_int(risk_state.get("active_positions")) or 0
        gross_notional = _as_float(risk_state.get("gross_notional"))
        exposure_line = f"Aktywne pozycje: {active_positions}"
        if gross_notional is not None:
            exposure_line += f" | Ekspozycja brutto: {_format_money(gross_notional)}"
        risk_lines.append(exposure_line)

        positions_raw = risk_state.get("positions")
        if isinstance(positions_raw, Mapping) and positions_raw:
            formatted: list[tuple[float, str]] = []
            for symbol, payload in positions_raw.items():
                if not isinstance(payload, Mapping):
                    continue
                entry = _normalize_position_entry(str(symbol), payload)
                if entry is not None:
                    formatted.append(entry)
            if formatted:
                formatted.sort(key=lambda item: item[0], reverse=True)
                formatted_lines = [text for _value, text in formatted[:5]]
                risk_lines.append("Pozycje: " + "; ".join(formatted_lines))

        daily_loss_pct = _as_float(risk_state.get("daily_loss_pct"))
        drawdown_pct = _as_float(risk_state.get("drawdown_pct"))
        risk_lines.append(
            "Dzienna strata: {loss} | Obsunięcie: {dd}".format(
                loss=_format_percentage(daily_loss_pct),
                dd=_format_percentage(drawdown_pct),
            )
        )
        liquidation = bool(risk_state.get("force_liquidation"))
        risk_lines.append("Force liquidation: TAK" if liquidation else "Force liquidation: NIE")

        limits = risk_state.get("limits")
        if isinstance(limits, Mapping):
            limit_parts: list[str] = []
            max_positions = _as_int(limits.get("max_positions"))
            if max_positions is not None:
                limit_parts.append(f"max pozycje {max_positions}")
            max_exposure = _as_float(limits.get("max_position_pct"))
            if max_exposure is not None:
                limit_parts.append(f"max ekspozycja {_format_percentage(max_exposure)}")
            max_leverage = _as_float(limits.get("max_leverage"))
            if max_leverage is not None:
                limit_parts.append(f"max dźwignia {max_leverage:.2f}x")
            daily_limit = _as_float(limits.get("daily_loss_limit"))
            if daily_limit is not None:
                limit_parts.append(f"dzienna strata {_format_percentage(daily_limit)}")
            drawdown_limit = _as_float(limits.get("drawdown_limit"))
            if drawdown_limit is not None:
                limit_parts.append(f"obsunięcie {_format_percentage(drawdown_limit)}")
            target_vol = _as_float(limits.get("target_volatility"))
            if target_vol is not None:
                limit_parts.append(f"target vol {_format_percentage(target_vol)}")
            stop_loss_atr = _as_float(limits.get("stop_loss_atr_multiple"))
            if stop_loss_atr is not None:
                limit_parts.append(f"stop loss ATR× {stop_loss_atr:.2f}")
            if limit_parts:
                risk_lines.append("Limity: " + ", ".join(limit_parts))

    # Dodatkowe linie o danych (manifest/cache), jeśli dołączono do summary
    data_lines: list[str] = []
    data_checks = summary.get("data_checks")
    if isinstance(data_checks, Mapping):
        manifest_info = data_checks.get("manifest")
        if isinstance(manifest_info, Mapping):
            entries = manifest_info.get("entries") or []
            if isinstance(entries, list):
                total_entries = len(entries)
                issues_count = 0
                for entry in entries:
                    issues = entry.get("issues")
                    if isinstance(issues, list) and any(issues):
                        issues_count += 1
                status_text = str(manifest_info.get("status", "n/a")).upper()
                if total_entries:
                    data_lines.append(
                        f"Manifest OHLCV: {status_text} ({total_entries} wpisów, problemy: {issues_count})"
                    )
                else:
                    data_lines.append(f"Manifest OHLCV: {status_text} (brak wpisów)")
        cache_info = data_checks.get("cache")
        if isinstance(cache_info, Mapping) and cache_info:
            fragments: list[str] = []
            for symbol, payload in sorted(cache_info.items()):
                fragment = str(symbol)
                if isinstance(payload, Mapping):
                    intervals_payload = payload.get("intervals")
                    if isinstance(intervals_payload, Mapping) and intervals_payload:
                        interval_parts: list[str] = []
                        for interval_name, interval_payload in sorted(intervals_payload.items()):
                            interval_fragment = str(interval_name)
                            if isinstance(interval_payload, Mapping):
                                coverage_int = _as_int(interval_payload.get("coverage_bars"))
                                required_int = _as_int(interval_payload.get("required_bars"))
                                row_count_int = _as_int(interval_payload.get("row_count"))
                                details: list[str] = []
                                if coverage_int is not None and required_int is not None:
                                    details.append(f"pokrycie {coverage_int}/{required_int}")
                                if row_count_int is not None:
                                    details.append(f"wiersze {row_count_int}")
                                if details:
                                    interval_fragment += " (" + ", ".join(details) + ")"
                            interval_parts.append(interval_fragment)
                        if interval_parts:
                            fragment += " [" + "; ".join(interval_parts) + "]"
                            fragments.append(fragment)
                            continue

                coverage_int = _as_int(payload.get("coverage_bars") if isinstance(payload, Mapping) else None)
                required_int = _as_int(payload.get("required_bars") if isinstance(payload, Mapping) else None)
                row_count_int = _as_int(payload.get("row_count") if isinstance(payload, Mapping) else None)
                if coverage_int is not None and required_int is not None:
                    fragment += f": pokrycie {coverage_int}/{required_int}"
                if row_count_int is not None:
                    fragment += f", wiersze {row_count_int}"
                fragments.append(fragment)
            if fragments:
                data_lines.append("Cache offline: " + "; ".join(fragments))
        precheck_info = data_checks.get("paper_precheck")
        if isinstance(precheck_info, Mapping):
            status = str(precheck_info.get("status", "unknown")).upper()
            cov_status = str(precheck_info.get("coverage_status", "unknown"))
            risk_status = str(precheck_info.get("risk_status", "unknown"))
            warn_parts: list[str] = []
            coverage_warnings = precheck_info.get("coverage_warnings")
            if isinstance(coverage_warnings, list) and coverage_warnings:
                warn_parts.append(f"coverage_warn={len(coverage_warnings)}")
            config_payload = precheck_info.get("config")
            if isinstance(config_payload, Mapping):
                config_warns = config_payload.get("warnings") or []
                if config_warns:
                    warn_parts.append(f"config_warn={len(config_warns)}")
            risk_payload = precheck_info.get("risk")
            if isinstance(risk_payload, Mapping):
                risk_warns = risk_payload.get("warnings") or []
                if risk_warns:
                    warn_parts.append(f"risk_warn={len(risk_warns)}")
            warn_suffix = f" ({', '.join(warn_parts)})" if warn_parts else ""
            data_lines.append(
                f"Paper pre-check: {status} (coverage={cov_status}, risk={risk_status})"
                + warn_suffix
            )

    # Info o magazynie raportu
    storage_lines: list[str] = []
    storage_info = summary.get("storage")
    if isinstance(storage_info, Mapping) and storage_info:
        status = str(storage_info.get("status", "unknown")).upper()
        free_mb = _as_float(storage_info.get("free_mb"))
        total_mb = _as_float(storage_info.get("total_mb"))
        threshold_mb = _as_float(storage_info.get("threshold_mb"))
        parts = [f"status={status}"]
        if free_mb is not None:
            parts.append(f"wolne {free_mb:.2f} MB")
        if total_mb is not None:
            parts.append(f"całkowite {total_mb:.2f} MB")
        if threshold_mb is not None:
            parts.append(f"próg {threshold_mb:.2f} MB")
        storage_lines.append("Magazyn raportu: " + ", ".join(parts))

    lines = [
        f"Środowisko: {environment}",
        f"Zakres dat: {start} → {end}",
        f"Liczba zleceń: {orders_count}",
        f"Liczba wpisów w ledgerze: {ledger_entries}",
    ]
    if metrics_lines:
        lines.extend(metrics_lines)
    if risk_lines:
        lines.extend(risk_lines)
    if data_lines:
        lines.extend(data_lines)
    if storage_lines:
        lines.extend(storage_lines)
    lines.append("Alerty: " + "; ".join(alert_lines))
    lines.append(f"SHA-256 summary.json: {summary_sha256}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Walidacja cache + manifest
# --------------------------------------------------------------------------------------
def _ensure_smoke_cache(
    *,
    pipeline: Any,
    symbols: Sequence[str],
    interval: str,
    start_ms: int,
    end_ms: int,
    required_bars: int,
    tick_ms: int,
) -> Mapping[str, object]:
    """Sprawdza, czy lokalny cache zawiera dane potrzebne do smoke testu."""
    required_intervals = _collect_required_intervals(pipeline, symbols=symbols)
    if not required_intervals:
        required_intervals = (interval,)

    normalized_primary = _normalize_interval_token(interval)
    tick_map: dict[str, int] = {}
    required_map: dict[str, int] = {}

    for candidate in required_intervals:
        normalized = _normalize_interval_token(candidate)
        if not normalized:
            continue
        if normalized == normalized_primary:
            tick_map[normalized] = max(1, int(tick_ms))
            required_map[normalized] = int(required_bars)
            continue
        try:
            candidate_tick_ms = _interval_to_milliseconds(candidate)
        except ValueError:
            _LOGGER.warning("Pominięto nieobsługiwany interwał manifestu: %s", candidate)
            continue
        tick_map[normalized] = candidate_tick_ms
        window_bars = max(1, int((end_ms - start_ms) / max(1, candidate_tick_ms)) + 2)
        required_map[normalized] = window_bars

    effective_intervals = [
        candidate
        for candidate in required_intervals
        if _normalize_interval_token(candidate) in tick_map
    ]
    if not effective_intervals:
        effective_intervals = [interval]
        tick_map.setdefault(normalized_primary or interval, max(1, int(tick_ms)))
        required_map.setdefault(normalized_primary or interval, int(required_bars))

    manifest_report = _verify_manifest_coverage(
        pipeline=pipeline,
        symbols=symbols,
        intervals=effective_intervals,
        end_ms=end_ms,
        required_bars_map=required_map,
    )

    data_source = getattr(pipeline, "data_source", None)
    storage = getattr(data_source, "storage", None)
    cache_reports: dict[str, dict[str, Mapping[str, object]]] = {}

    if storage is None:
        _LOGGER.warning(
            "Nie mogę zweryfikować cache – pipeline nie udostępnia storage'u. Pomijam kontrolę.",
        )
    else:
        try:
            metadata: MutableMapping[str, str] = storage.metadata()
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Nie udało się odczytać metadanych cache: %s", exc)
            metadata = {}

        for candidate in effective_intervals:
            normalized = _normalize_interval_token(candidate)
            if not normalized:
                continue

            candidate_tick_ms = tick_map.get(normalized, max(1, int(tick_ms)))
            candidate_required = required_map.get(normalized, int(required_bars))

            issues: list[tuple[str, str]] = []
            for symbol in symbols:
                key = f"{symbol}::{candidate}"
                row_count: int | None = None
                last_timestamp: int | None = None

                if metadata:
                    raw_rows = metadata.get(f"row_count::{symbol}::{candidate}")
                    if raw_rows is not None:
                        try:
                            row_count = int(raw_rows)
                        except (TypeError, ValueError):
                            _LOGGER.warning(
                                "Nieprawidłowa wartość row_count dla %s (%s): %s",
                                symbol,
                                candidate,
                                raw_rows,
                            )
                    raw_last = metadata.get(f"last_timestamp::{symbol}::{candidate}")
                    if raw_last is not None:
                        try:
                            last_timestamp = int(float(raw_last))
                        except (TypeError, ValueError):
                            _LOGGER.warning(
                                "Nieprawidłowa wartość last_timestamp dla %s (%s): %s",
                                symbol,
                                candidate,
                                raw_last,
                            )

                try:
                    payload = storage.read(key)
                except KeyError:
                    issues.append((str(symbol), "brak wpisu w cache"))
                    continue

                rows = list(payload.get("rows", []))
                if not rows:
                    issues.append((str(symbol), "puste dane w cache"))
                    continue

                if row_count is None:
                    row_count = len(rows)
                if last_timestamp is None:
                    last_timestamp = int(float(rows[-1][0]))

                first_timestamp = int(float(rows[0][0]))

                if row_count < candidate_required:
                    issues.append((str(symbol), f"za mało świec ({row_count} < {candidate_required})"))
                    continue

                if last_timestamp < end_ms:
                    issues.append((str(symbol), f"ostatnia świeca {last_timestamp} < wymaganego końca {end_ms}"))
                    continue

                if first_timestamp > start_ms:
                    issues.append((str(symbol), f"pierwsza świeca {first_timestamp} > wymaganego startu {start_ms}"))
                    continue

                coverage = ((last_timestamp - first_timestamp) // max(1, candidate_tick_ms)) + 1
                if coverage < candidate_required:
                    issues.append((str(symbol), f"pokrycie obejmuje {coverage} świec (wymagane {candidate_required})"))
                    continue

                symbol_entry = cache_reports.setdefault(str(symbol), {})
                interval_map = symbol_entry.setdefault("intervals", {})
                interval_map[str(candidate)] = {
                    "row_count": int(row_count),
                    "first_timestamp_ms": first_timestamp,
                    "last_timestamp_ms": last_timestamp,
                    "coverage_bars": int(coverage),
                    "required_bars": int(candidate_required),
                }

            if issues:
                for symbol_name, reason in issues:
                    _LOGGER.error(
                        "Cache offline dla symbolu %s (%s) nie spełnia wymagań smoke testu: %s",
                        symbol_name,
                        candidate,
                        reason,
                    )
                raise RuntimeError(
                    "Cache offline nie obejmuje wymaganego zakresu danych. Uruchom scripts/seed_paper_cache.py, "
                    "aby zbudować deterministyczny seed przed smoke testem.",
                )

    result: dict[str, object] = {
        "interval": interval,
        "intervals": [str(value) for value in effective_intervals],
        "symbols": [str(symbol) for symbol in symbols],
        "required_bars": int(required_bars),
        "tick_ms": int(max(1, tick_ms)),
        "window_ms": {"start": int(start_ms), "end": int(end_ms)},
        "required_bars_map": {key: int(value) for key, value in required_map.items()},
        "tick_ms_map": {key: int(value) for key, value in tick_map.items()},
    }
    if manifest_report:
        result["manifest"] = manifest_report
    if cache_reports:
        result["cache"] = cache_reports
    return result


def _verify_manifest_coverage(
    *,
    pipeline: Any,
    symbols: Sequence[str],
    intervals: Sequence[str],
    end_ms: int,
    required_bars_map: Mapping[str, int],
) -> Mapping[str, object] | None:
    """Waliduje metadane manifestu przed uruchomieniem smoke testu."""
    bootstrap = getattr(pipeline, "bootstrap", None)
    if bootstrap is None:
        return None

    environment_cfg = getattr(bootstrap, "environment", None)
    core_config = getattr(bootstrap, "core_config", None)
    if environment_cfg is None or core_config is None:
        return None
    if not hasattr(core_config, "instrument_universes"):
        return None

    universe_name = getattr(environment_cfg, "instrument_universe", None)
    cache_root = getattr(environment_cfg, "data_cache_path", None)
    exchange_name = getattr(environment_cfg, "exchange", None)
    if not universe_name or not cache_root or not exchange_name:
        return None

    manifest_path = Path(cache_root) / "ohlcv_manifest.sqlite"
    if not manifest_path.exists():
        _LOGGER.warning(
            "Manifest %s nie istnieje – pomijam kontrolę metadanych i sprawdzam wyłącznie surowe pliki.",
            manifest_path,
        )
        return None

    try:
        universe = core_config.instrument_universes[universe_name]
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning(
            "Nie udało się pobrać uniwersum instrumentów '%s' z konfiguracji: %s – pomijam kontrolę manifestu.",
            universe_name,
            exc,
        )
        return None

    as_of = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)
    try:
        statuses = evaluate_coverage(
            manifest_path=manifest_path,
            universe=universe,
            exchange_name=exchange_name,
            as_of=as_of,
        )
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("Nie udało się ocenić pokrycia manifestu: %s", exc)
        return None

    normalized_map: dict[str, str] = {}
    ordered_normalized: list[str] = []
    for candidate in intervals:
        normalized = _normalize_interval_token(candidate)
        if not normalized:
            continue
        if normalized not in required_bars_map:
            # pomiń interwały, których nie umiemy zmapować do wymagań
            continue
        if normalized not in normalized_map:
            normalized_map[normalized] = str(candidate)
            ordered_normalized.append(normalized)

    if not ordered_normalized:
        return None

    tracked_symbols = {str(symbol).lower() for symbol in symbols}
    status_by_key: dict[tuple[str, str], object] = {}
    for status in statuses:
        normalized = _normalize_interval_token(status.interval)
        if not normalized:
            continue
        status_by_key[(status.symbol.lower(), normalized)] = status

    issues: list[str] = []
    entries_payload: list[dict[str, object]] = []

    for symbol in symbols:
        symbol_str = str(symbol)
        symbol_key = symbol_str.lower()
        for normalized in ordered_normalized:
            display_interval = normalized_map[normalized]
            status = status_by_key.get((symbol_key, normalized))
            required_rows = required_bars_map.get(normalized, 0)
            if status is None:
                issues.append(
                    f"{symbol_str}/{display_interval}: manifest nie zawiera wpisu – uruchom scripts/seed_paper_cache.py."
                )
                continue

            entry = status.manifest_entry
            if status.issues:
                issues.extend(
                    _render_manifest_issue(status.symbol, status.interval, issue)
                    for issue in status.issues
                )

            row_count = entry.row_count
            if row_count is None:
                issues.append(
                    f"{status.symbol}/{status.interval}: manifest nie zawiera licznika świec (row_count)"
                )
            elif required_rows and row_count < required_rows:
                issues.append(
                    f"{status.symbol}/{status.interval}: manifest raportuje jedynie {row_count} świec (< {required_rows})"
                )

            last_ts = entry.last_timestamp_ms
            if last_ts is None:
                issues.append(
                    f"{status.symbol}/{status.interval}: manifest nie zawiera ostatniego stempla czasowego"
                )
            elif last_ts < end_ms:
                issues.append(
                    f"{status.symbol}/{status.interval}: ostatnia świeca w manifescie ({last_ts}) < wymaganego końca ({end_ms})"
                )

            entries_payload.append(
                {
                    "symbol": status.symbol,
                    "interval": status.interval,
                    "status": status.status,
                    "issues": list(status.issues),
                    "row_count": entry.row_count,
                    "required_rows": status.required_rows,
                    "gap_minutes": entry.gap_minutes,
                    "last_timestamp_ms": entry.last_timestamp_ms,
                    "last_timestamp_iso": entry.last_timestamp_iso,
                }
            )

    if issues:
        for detail in issues:
            _LOGGER.error("Manifest OHLCV: %s", detail)
        raise RuntimeError(
            "Manifest danych OHLCV jest niekompletny dla smoke testu. Uruchom scripts/seed_paper_cache.py lub pełny backfill, "
            "aby zaktualizować manifest."
        )

    required_rows_payload = {
        normalized_map[token]: int(required_bars_map[token]) for token in ordered_normalized
    }

    return {
        "status": "ok",
        "as_of": as_of.isoformat(),
        "intervals": [normalized_map[token] for token in ordered_normalized],
        "required_rows": required_rows_payload,
        "symbols": sorted(str(symbol) for symbol in symbols),
        "entries": entries_payload,
    }


def _render_manifest_issue(symbol: str, interval: str, issue: str) -> str:
    if issue.startswith("manifest_status:"):
        status = issue.split(":", 1)[1]
        return f"{symbol}/{interval}: status manifestu = {status}"
    if issue == "missing_row_count":
        return f"{symbol}/{interval}: manifest nie zawiera informacji o liczbie świec"
    if issue.startswith("insufficient_rows:"):
        payload = issue.split(":", 1)[1]
        return f"{symbol}/{interval}: manifest raportuje zbyt mało świec ({payload})"
    return f"{symbol}/{interval}: {issue}"


# --------------------------------------------------------------------------------------
# Adapter offline dla smoke testu
# --------------------------------------------------------------------------------------
class _OfflineExchangeAdapter(ExchangeAdapter):
    """Minimalny adapter giełdowy działający offline dla trybu paper-smoke."""

    name = "offline"

    def __init__(self, credentials: ExchangeCredentials, **_: object) -> None:
        super().__init__(credentials)

    def configure_network(self, *, ip_allowlist: tuple[str, ...] | None = None) -> None:  # noqa: D401, ARG002
        return None

    def fetch_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            balances={"USDT": 100_000.0},
            total_equity=100_000.0,
            available_margin=100_000.0,
            maintenance_margin=0.0,
        )

    def fetch_symbols(self):  # pragma: no cover
        return ()

    def fetch_ohlcv(  # noqa: D401, ARG002
        self,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        limit: int | None = None,
    ):
        return []

    def place_order(self, request):  # pragma: no cover
        raise NotImplementedError

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:  # pragma: no cover
        raise NotImplementedError

    def stream_public_data(self, *, channels):  # pragma: no cover
        raise NotImplementedError

    def stream_private_data(self, *, channels):  # pragma: no cover
        raise NotImplementedError


def _offline_adapter_factory(credentials: ExchangeCredentials, **kwargs: object) -> ExchangeAdapter:
    return _OfflineExchangeAdapter(credentials, **kwargs)


# --------------------------------------------------------------------------------------
# Pętla realtime
# --------------------------------------------------------------------------------------
def _run_loop(runner: DailyTrendRealtimeRunner, poll_seconds: float) -> int:
    interval = max(1.0, poll_seconds)
    stop = False

    def _signal_handler(_signo, _frame) -> None:  # type: ignore[override]
        nonlocal stop
        stop = True
        _LOGGER.info("Otrzymano sygnał zatrzymania – kończę pętlę realtime")

    for signame in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signame, _signal_handler)

    _LOGGER.info("Start pętli realtime (co %s s)", interval)
    while not stop:
        start = time.monotonic()
        try:
            results = runner.run_once()
            if results:
                _log_order_results(results)
        except Exception:  # noqa: BLE001
            _LOGGER.exception("Błąd podczas iteracji realtime")
        elapsed = time.monotonic() - start
        sleep_for = max(1.0, interval - elapsed)
        if stop:
            break
        time.sleep(sleep_for)
    return 0


# --------------------------------------------------------------------------------------
# Główna funkcja CLI
# --------------------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(argv_list)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), stream=sys.stdout)

    provided_flags = {item for item in argv_list if isinstance(item, str) and item.startswith("--")}
    cli_fail_flag = "--fail-on-security-warnings" in provided_flags
    fail_on_security_source = "cli" if cli_fail_flag else "default"

    env_fail_raw = os.environ.get(_ENV_FAIL_ON_SECURITY_WARNINGS)
    env_fail_value: bool | None = None
    env_fail_applied = False
    if env_fail_raw is not None:
        try:
            env_fail_value = _parse_env_bool(env_fail_raw, variable=_ENV_FAIL_ON_SECURITY_WARNINGS)
        except ValueError as exc:
            _LOGGER.error("%s", exc)
            return 2
        if cli_fail_flag:
            _LOGGER.info(
                "Pominięto %s=%s, ponieważ ustawiono flagę CLI --fail-on-security-warnings.",
                _ENV_FAIL_ON_SECURITY_WARNINGS,
                env_fail_raw,
            )
        else:
            args.fail_on_security_warnings = env_fail_value
            env_fail_applied = True
            fail_on_security_source = f"env:{_ENV_FAIL_ON_SECURITY_WARNINGS}"
            _LOGGER.info(
                "Zastosowano %s=%s – ostrzeżenia bezpieczeństwa będą traktowane jako %s.",
                _ENV_FAIL_ON_SECURITY_WARNINGS,
                env_fail_raw,
                "błąd" if env_fail_value else "ostrzeżenie",
            )

    runtime_plan_path: Path | None = None
    if args.runtime_plan_jsonl:
        runtime_plan_path = Path(str(args.runtime_plan_jsonl)).expanduser()

    env_pipeline_raw = os.environ.get(_ENV_PIPELINE_MODULES)
    env_pipeline_modules, env_pipeline_reason = _parse_modules_env_value(
        env_pipeline_raw, env_var=_ENV_PIPELINE_MODULES
    )
    env_pipeline_applied = False

    env_realtime_raw = os.environ.get(_ENV_REALTIME_MODULES)
    env_realtime_modules, env_realtime_reason = _parse_modules_env_value(
        env_realtime_raw, env_var=_ENV_REALTIME_MODULES
    )
    env_realtime_applied = False

    pipeline_modules: Sequence[str] | None = args.pipeline_modules
    realtime_modules: Sequence[str] | None = args.realtime_modules
    pipeline_origin: str | None = None
    realtime_origin: str | None = None

    if pipeline_modules is None:
        if env_pipeline_modules is not None:
            pipeline_modules = env_pipeline_modules
            pipeline_origin = f"zmienna środowiskowa {_ENV_PIPELINE_MODULES}"
            env_pipeline_applied = True
    else:
        pipeline_origin = "flagi CLI (--pipeline-module)"
        if env_pipeline_modules is not None or env_pipeline_raw is not None:
            _LOGGER.info(
                "Pominięto moduły pipeline z %s na rzecz flag CLI (--pipeline-module)",
                _ENV_PIPELINE_MODULES,
            )

    if realtime_modules is None:
        if env_realtime_modules is not None:
            realtime_modules = env_realtime_modules
            realtime_origin = f"zmienna środowiskowa {_ENV_REALTIME_MODULES}"
            env_realtime_applied = True
    else:
        realtime_origin = "flagi CLI (--realtime-module)"
        if env_realtime_modules is not None or env_realtime_raw is not None:
            _LOGGER.info(
                "Pominięto moduły realtime z %s na rzecz flag CLI (--realtime-module)",
                _ENV_REALTIME_MODULES,
            )

    if env_pipeline_raw is not None and not env_pipeline_applied and env_pipeline_reason is None:
        env_pipeline_reason = "cli_override"

    if env_realtime_raw is not None and not env_realtime_applied and env_realtime_reason is None:
        env_realtime_reason = "cli_override"

    pipeline_origin = pipeline_origin or (
        "flagi CLI (--pipeline-module)" if pipeline_modules is not None else None
    )
    realtime_origin = realtime_origin or (
        "flagi CLI (--realtime-module)" if realtime_modules is not None else None
    )

    try:
        _apply_runtime_overrides(
            pipeline_modules,
            realtime_modules,
            pipeline_origin=pipeline_origin,
            realtime_origin=realtime_origin,
        )
    except ImportError as exc:
        _LOGGER.error("Nie można załadować modułów runtime: %s", exc)
        return 2

    snapshot = get_runtime_module_candidates()
    _LOGGER.debug(
        (
            "Aktywne moduły runtime – pipeline (%s, moduł %s): %s | "
            "realtime (%s, moduł %s): %s"
        ),
        snapshot.pipeline_origin or "brak informacji",
        snapshot.pipeline_resolved_from or "nieustalony",
        ", ".join(snapshot.pipeline_modules),
        snapshot.realtime_origin or "brak informacji",
        snapshot.realtime_resolved_from or "nieustalony",
        ", ".join(snapshot.realtime_modules),
    )

    if args.print_runtime_modules:
        payload = snapshot.to_json_payload()
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0

    config_path = Path(args.config)

    if args.print_risk_profiles:
        return _print_risk_profiles(config_path, args.environment)

    try:
        secret_manager = _create_secret_manager(args)
    except SecretStorageError as exc:
        _LOGGER.error("Nie udało się zainicjalizować magazynu sekretów: %s", exc)
        return 2

    if not config_path.exists():
        _LOGGER.error("Plik konfiguracyjny %s nie istnieje", config_path)
        return 1

    validation_result = _load_validated_core_config(
        config_path,
        environment=args.environment,
        strategy=args.strategy,
        controller=args.controller,
        risk_profile=args.risk_profile,
    )
    if validation_result is None:
        return 2

    core_config_validated = validation_result.config
    validated_strategy = validation_result.strategy_name
    validated_controller = validation_result.controller_name
    validated_risk_profile = validation_result.risk_profile_name

    precheck_payload: Mapping[str, object] | None = None
    precheck_audit_metadata: Mapping[str, object] | None = None
    audit_log_path: Path | None = None
    audit_json_path: Path | None = None
    summary_output_path: Path | None = None
    operator_name: str | None = None
    if args.paper_smoke:
        audit_dir = None
        if args.paper_precheck_audit_dir is not None:
            audit_dir_arg = str(args.paper_precheck_audit_dir).strip()
            if audit_dir_arg:
                audit_dir = Path(audit_dir_arg)

        if args.paper_smoke_audit_log is not None:
            audit_log_arg = str(args.paper_smoke_audit_log).strip()
            if audit_log_arg:
                audit_log_path = Path(audit_log_arg)

        if args.paper_smoke_json_log is not None:
            audit_json_arg = str(args.paper_smoke_json_log).strip()
            if audit_json_arg:
                audit_json_path = Path(audit_json_arg)

        if args.paper_smoke_summary_json is not None:
            summary_arg = str(args.paper_smoke_summary_json).strip()
            if summary_arg:
                summary_output_path = Path(summary_arg)

        operator_name = _resolve_operator_name(args.paper_smoke_operator)

        precheck_payload, precheck_exit, precheck_audit_metadata = _run_paper_precheck_for_smoke(
            config_path=config_path,
            environment=args.environment,
            fail_on_warnings=args.paper_precheck_fail_on_warnings,
            skip=args.skip_paper_precheck,
            audit_dir=audit_dir,
        )
        if precheck_exit != 0:
            return int(precheck_exit)

    adapter_factories: Mapping[str, ExchangeAdapterFactory] | None = None
    if args.paper_smoke:
        adapter_factories = {
            "binance_spot": _offline_adapter_factory,
            "binance_futures": _offline_adapter_factory,
            "kraken_spot": _offline_adapter_factory,
            "kraken_futures": _offline_adapter_factory,
            "zonda_spot": _offline_adapter_factory,
        }

    try:
        pipeline = build_daily_trend_pipeline(
            environment_name=args.environment,
            strategy_name=validated_strategy,
            controller_name=validated_controller,
            config_path=config_path,
            secret_manager=secret_manager,
            adapter_factories=adapter_factories,
            risk_profile_name=validated_risk_profile,
        )
    except Exception as exc:  # noqa: BLE001
        _LOGGER.exception("Nie udało się zbudować pipeline'u daily trend: %s", exc)
        return 1

    # Bezpieczne logowanie (mock/test może nie mieć pól)
    strategy_name = getattr(pipeline, "strategy_name", args.strategy)
    controller_name = getattr(pipeline, "controller_name", args.controller)
    environment_cfg = getattr(pipeline, "bootstrap", None)
    environment_cfg = getattr(environment_cfg, "environment", None)
    offline_mode = bool(getattr(environment_cfg, "offline_mode", False))
    if offline_mode:
        _LOGGER.info(
            "Środowisko %s działa w trybie offline – pomijam operacje wymagające sieci.",
            args.environment,
        )
    _LOGGER.info(
        "Pipeline gotowy: środowisko=%s, strategia=%s, kontroler=%s",
        args.environment,
        strategy_name,
        controller_name,
    )

    plan_payload: Mapping[str, object] | None = None
    need_plan_snapshot = bool(
        runtime_plan_path is not None
        or args.print_runtime_plan
        or args.fail_on_security_warnings
    )
    if need_plan_snapshot:
        try:
            config_for_plan = getattr(pipeline.bootstrap, "core_config", None)
            if config_for_plan is None:
                config_for_plan = core_config_validated
            if config_for_plan is None:
                config_for_plan = load_core_config(config_path)

            plan_payload = _build_runtime_plan_payload(
                args=args,
                snapshot=snapshot,
                pipeline=pipeline,
                config=config_for_plan,
                environment_name=args.environment,
                cli_pipeline_modules=args.pipeline_modules,
                cli_realtime_modules=args.realtime_modules,
                env_pipeline_modules=env_pipeline_modules,
                env_pipeline_raw=env_pipeline_raw,
                env_pipeline_applied=env_pipeline_applied,
                env_pipeline_reason=env_pipeline_reason,
                env_realtime_modules=env_realtime_modules,
                env_realtime_raw=env_realtime_raw,
                env_realtime_applied=env_realtime_applied,
                env_realtime_reason=env_realtime_reason,
                cli_fail_on_security_flag=cli_fail_flag,
                env_fail_on_security_raw=env_fail_raw,
                env_fail_on_security_applied=env_fail_applied,
                env_fail_on_security_value=env_fail_value,
                fail_on_security_source=fail_on_security_source,
                precheck_payload=precheck_payload,
                precheck_audit_metadata=precheck_audit_metadata,
                operator_name=operator_name,
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.error("Nie udało się zbudować planu runtime: %s", exc)
            return 2

    security_warnings_detected = False
    if plan_payload is not None and args.fail_on_security_warnings:
        security_warnings_detected = _log_security_warnings(
            plan_payload,
            fail_on_warnings=True,
            logger=_LOGGER,
            context="run_daily_trend.runtime_plan",
        )

    if runtime_plan_path is not None and plan_payload is not None:
        try:
            _append_runtime_plan_jsonl(runtime_plan_path, plan_payload)
            _LOGGER.info(
                "Zapisano plan runtime do %s (profil=%s)",
                runtime_plan_path,
                plan_payload.get("risk_profile") or "brak",
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.error(
                "Nie udało się zapisać planu runtime do %s: %s",
                runtime_plan_path,
                exc,
            )
            return 2

    if args.print_runtime_plan and plan_payload is not None:
        json.dump(plan_payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        if args.fail_on_security_warnings and security_warnings_detected:
            return 3
        return 0

    if args.fail_on_security_warnings and plan_payload is not None and security_warnings_detected:
        return 3

    environment = pipeline.bootstrap.environment.environment
    if environment is Environment.LIVE and not args.allow_live:
        _LOGGER.error(
            "Środowisko %s to LIVE – dla bezpieczeństwa użyj --allow-live po wcześniejszych testach paper.",
            args.environment,
        )
        return 3

    if args.dry_run:
        _LOGGER.info("Dry-run zakończony sukcesem. Pipeline gotowy do uruchomienia.")
        return 0

    if args.paper_smoke:
        try:
            start_ms, end_ms, window_meta = _resolve_date_window(args.date_window)
        except ValueError as exc:
            _LOGGER.error("Niepoprawny zakres dat: %s", exc)
            return 1

        end_dt = datetime.fromisoformat(window_meta["end"])
        tick_seconds = float(getattr(pipeline.controller, "tick_seconds", 86400.0) or 86400.0)
        tick_ms = max(1, int(tick_seconds * 1000))
        window_duration_ms = max(0, end_ms - start_ms)
        approx_bars = max(1, int(window_duration_ms / tick_ms) + 1)
        history_bars = max(1, min(int(args.history_bars), approx_bars))
        runner_start_ms = max(0, end_ms - history_bars * tick_ms)
        sync_start = min(start_ms, runner_start_ms)

        _LOGGER.info(
            "Startuję smoke test paper trading dla %s w zakresie %s – %s.",
            args.environment,
            window_meta["start"],
            window_meta["end"],
        )

        required_bars = max(history_bars, max(1, int((end_ms - sync_start) / tick_ms) + 1))
        smoke_cache_report: Mapping[str, object] | None = None
        try:
            smoke_cache_report = _ensure_smoke_cache(
                pipeline=pipeline,
                symbols=pipeline.controller.symbols,
                interval=pipeline.controller.interval,
                start_ms=sync_start,
                end_ms=end_ms,
                required_bars=required_bars,
                tick_ms=tick_ms,
            )
        except RuntimeError as exc:
            _LOGGER.error("%s", exc)
            return 1

        pipeline.backfill_service.synchronize(
            symbols=pipeline.controller.symbols,
            interval=pipeline.controller.interval,
            start=sync_start,
            end=end_ms,
        )

        trading_controller = create_trading_controller(
            pipeline, pipeline.bootstrap.alert_router, health_check_interval=0.0,
        )

        runner = DailyTrendRealtimeRunner(
            controller=pipeline.controller,
            trading_controller=trading_controller,
            history_bars=history_bars,
            clock=lambda end=end_dt: end,
        )

        results = runner.run_once()
        if results:
            _log_order_results(results)
        else:
            _LOGGER.info("Smoke test zakończony – brak sygnałów w zadanym oknie.")

        report_dir = _prepare_smoke_report_directory(args.smoke_output)
        storage_info = _collect_storage_health(report_dir, min_free_mb=args.smoke_min_free_mb)
        alert_snapshot = pipeline.bootstrap.alert_router.health_snapshot()

        # Snapshot stanu ryzyka (opcjonalnie)
        risk_snapshot: Mapping[str, object] | None = None
        try:
            risk_engine = getattr(pipeline.bootstrap, "risk_engine", None)
            if risk_engine is not None and hasattr(risk_engine, "snapshot_state"):
                risk_snapshot = risk_engine.snapshot_state(pipeline.risk_profile_name)
        except NotImplementedError:
            _LOGGER.warning("Silnik ryzyka nie udostępnia metody snapshot_state – pomijam stan ryzyka")
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Nie udało się pobrać stanu ryzyka: %s", exc)

        combined_checks: dict[str, object] = {}
        if smoke_cache_report:
            combined_checks.update(smoke_cache_report)
        if precheck_payload is not None:
            combined_checks["paper_precheck"] = json.loads(json.dumps(precheck_payload))
        data_checks: Mapping[str, object] | None = combined_checks or None

        summary_path = _export_smoke_report(
            report_dir=report_dir,
            results=results,
            ledger=pipeline.execution_service.ledger(),
            window=window_meta,
            environment=args.environment,
            alert_snapshot=alert_snapshot,
            risk_state=risk_snapshot,
            data_checks=data_checks,
            storage_info=storage_info,
        )
        summary_hash = _hash_file(summary_path)
        try:
            stored_report_path = store_environment_report(
                summary_path,
                pipeline.bootstrap.environment,
                now=datetime.now(timezone.utc),
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.error(
                "Nie udało się zapisać kopii raportu w magazynie środowiska: %s",
                exc,
            )
            stored_report_path = None
        else:
            if stored_report_path is not None:
                _LOGGER.info(
                    "Kopia raportu smoke zapisana w %s",
                    stored_report_path,
                )
        try:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            _LOGGER.error("Nie udało się odczytać summary.json: %s", exc)
            summary_payload = {
                "environment": args.environment,
                "window": dict(window_meta),
                "orders": [],
                "ledger_entries": 0,
                "alert_snapshot": alert_snapshot,
                "risk_state": risk_snapshot or {},
            }

        summary_text = _render_smoke_summary(summary=summary_payload, summary_sha256=summary_hash)
        summary_txt_path = summary_path.with_suffix(".txt")
        summary_txt_path.write_text(summary_text + "\n", encoding="utf-8")
        readme_path = _write_smoke_readme(report_dir)
        _LOGGER.info("Raport smoke testu zapisany w %s (summary sha256=%s)", report_dir, summary_hash)
        _LOGGER.info("Podsumowanie smoke testu:%s%s", os.linesep, summary_text)

        archive_path: Path | None = None
        reporting_cfg = getattr(getattr(pipeline.bootstrap, "core_config", None), "reporting", None)
        resolved_upload_cfg = SmokeArchiveUploader.resolve_config(reporting_cfg)
        resolved_json_sync_cfg = PaperSmokeJsonSynchronizer.resolve_config(reporting_cfg)

        upload_cfg = None
        json_sync_cfg = None
        if offline_mode:
            if resolved_upload_cfg is not None:
                _LOGGER.info(
                    "Tryb offline: pomijam upload archiwum smoke testu (backend=%s)",
                    getattr(resolved_upload_cfg, "backend", "unknown"),
                )
            if resolved_json_sync_cfg is not None:
                _LOGGER.info("Tryb offline: pomijam synchronizację dziennika JSON smoke testów")
        else:
            upload_cfg = resolved_upload_cfg
            json_sync_cfg = resolved_json_sync_cfg

        archive_required = bool(args.archive_smoke or upload_cfg)

        if archive_required:
            archive_path = _archive_smoke_report(report_dir)
            if args.archive_smoke:
                _LOGGER.info("Utworzono archiwum smoke testu: %s", archive_path)
            else:
                _LOGGER.info("Archiwum smoke testu wygenerowane automatycznie na potrzeby uploadu: %s", archive_path)

        upload_result = None
        if upload_cfg and archive_path:
            try:
                uploader = SmokeArchiveUploader(upload_cfg, secret_manager=secret_manager)
                upload_result = uploader.upload(
                    archive_path,
                    environment=args.environment,
                    summary_sha256=summary_hash,
                    window=window_meta,
                )
                _LOGGER.info(
                    "Przesłano archiwum smoke testu (%s) do %s", upload_result.backend, upload_result.location
                )
            except Exception as exc:  # noqa: BLE001
                _LOGGER.error("Nie udało się przesłać archiwum smoke testu: %s", exc)

        storage_context: dict[str, str] = {}
        storage_status = None
        if isinstance(storage_info, Mapping):
            storage_status = str(storage_info.get("status", ""))
            storage_context = {"storage_status": storage_status}
            free_mb = storage_info.get("free_mb")
            if free_mb is not None:
                storage_context["storage_free_mb"] = f"{float(free_mb):.2f}"
            threshold_mb = storage_info.get("threshold_mb")
            if threshold_mb is not None:
                storage_context["storage_threshold_mb"] = f"{float(threshold_mb):.2f}"

        storage_status_lower = storage_status.lower() if storage_status else ""
        fail_low_storage = bool(args.smoke_fail_on_low_space and storage_status_lower == "low")

        precheck_status = None
        precheck_status_text = None
        precheck_coverage_status = None
        precheck_risk_status = None
        if isinstance(precheck_payload, Mapping):
            precheck_status = str(precheck_payload.get("status", "unknown"))
            precheck_coverage_status = str(
                precheck_payload.get("coverage_status", "unknown")
            )
            precheck_risk_status = str(precheck_payload.get("risk_status", "unknown"))
            precheck_status_text = (
                f"{precheck_status} (coverage={precheck_coverage_status}, "
                f"risk={precheck_risk_status})"
            )

        body = (
            "Zakończono smoke test paper trading."
            f" Zamówienia: {len(results)}, raport: {summary_path},"
            f" sha256: {summary_hash}"
        )
        if precheck_status_text:
            body += f" Pre-check: {precheck_status_text}."
        if storage_status_lower == "low":
            free_str = storage_context.get("storage_free_mb")
            thresh_str = storage_context.get("storage_threshold_mb")
            if free_str and thresh_str:
                body += f" Ostrzeżenie: wolne miejsce {free_str} MB poniżej progu {thresh_str} MB."
            elif free_str:
                body += f" Ostrzeżenie: niskie wolne miejsce ({free_str} MB)."

        severity = "warning" if storage_status_lower == "low" else "info"
        if precheck_status and precheck_status.lower() == "warning":
            severity = "warning"

        precheck_context: dict[str, str] = {}
        if isinstance(precheck_payload, Mapping):
            precheck_context = {
                "paper_precheck_status": precheck_status or "unknown",
                "paper_precheck_coverage_status": precheck_coverage_status or "unknown",
                "paper_precheck_risk_status": precheck_risk_status or "unknown",
            }
            manifest_path_value = precheck_payload.get("manifest_path")
            if manifest_path_value:
                precheck_context["paper_precheck_manifest"] = str(manifest_path_value)
            coverage_warnings = precheck_payload.get("coverage_warnings")
            if isinstance(coverage_warnings, list) and coverage_warnings:
                precheck_context["paper_precheck_coverage_warnings"] = ",".join(
                    str(w) for w in coverage_warnings
                )
            risk_payload = precheck_payload.get("risk")
            if isinstance(risk_payload, Mapping):
                risk_warnings = risk_payload.get("warnings") or []
                if risk_warnings:
                    precheck_context["paper_precheck_risk_warnings"] = ",".join(
                        str(w) for w in risk_warnings
                    )

        precheck_metadata_for_log = precheck_audit_metadata
        if precheck_metadata_for_log is None and isinstance(precheck_payload, Mapping):
            audit_record = precheck_payload.get("audit_record")
            if isinstance(audit_record, Mapping):
                precheck_metadata_for_log = audit_record

        archive_context: dict[str, str] = {}
        if upload_result:
            archive_context["archive_upload_backend"] = upload_result.backend
            archive_context["archive_upload_location"] = upload_result.location
            for key, value in upload_result.metadata.items():
                archive_context[f"archive_upload_{key}"] = str(value)

        alert_context: dict[str, str] = {
            "environment": args.environment,
            "report_dir": str(report_dir),
            "orders": str(len(results)),
            "summary_sha256": summary_hash,
            "summary_text_path": str(summary_txt_path),
            "readme_path": str(readme_path),
            **({"archive_path": str(archive_path)} if archive_path else {}),
            **archive_context,
            **storage_context,
            **precheck_context,
        }
        if offline_mode:
            alert_context["offline_mode"] = "true"

        markdown_entry_id: str | None = None
        json_record: Mapping[str, object] | None = None
        json_sync_result = None
        log_timestamp = datetime.now(timezone.utc)

        if audit_log_path is not None:
            try:
                markdown_entry_id = _append_smoke_audit_entry(
                    log_path=audit_log_path,
                    timestamp=log_timestamp,
                    operator=operator_name or _resolve_operator_name(None),
                    environment=args.environment,
                    window=window_meta,
                    summary_path=summary_path,
                    summary_sha256=summary_hash,
                    severity=severity,
                    precheck_metadata=precheck_metadata_for_log,
                    precheck_status=precheck_status,
                    precheck_coverage_status=precheck_coverage_status,
                    precheck_risk_status=precheck_risk_status,
                )
            except Exception:  # noqa: BLE001
                _LOGGER.exception(
                    "Nie udało się zaktualizować logu audytu paper trading: %s", audit_log_path
                )

        if audit_json_path is not None:
            try:
                json_record = _append_smoke_json_log_entry(
                    json_path=audit_json_path,
                    timestamp=log_timestamp,
                    operator=operator_name or _resolve_operator_name(None),
                    environment=args.environment,
                    window=window_meta,
                    summary_path=summary_path,
                    summary_sha256=summary_hash,
                    severity=severity,
                    precheck_metadata=precheck_metadata_for_log,
                    precheck_payload=precheck_payload if isinstance(precheck_payload, Mapping) else None,
                    precheck_status=precheck_status,
                    precheck_coverage_status=precheck_coverage_status,
                    precheck_risk_status=precheck_risk_status,
                    markdown_entry_id=markdown_entry_id,
                )
            except Exception:  # noqa: BLE001
                _LOGGER.exception(
                    "Nie udało się zaktualizować dziennika JSON smoke testów: %s", audit_json_path
                )

        if markdown_entry_id:
            alert_context["paper_smoke_audit_entry_id"] = markdown_entry_id
            alert_context["paper_smoke_audit_log_path"] = str(audit_log_path)

        publish_record_id: str | None = None
        if json_record:
            alert_context["paper_smoke_json_log_path"] = str(audit_json_path)
            record_id = json_record.get("record_id")
            if record_id:
                publish_record_id = str(record_id)
                alert_context["paper_smoke_json_record_id"] = publish_record_id
            precheck_meta = json_record.get("precheck_metadata")
            if isinstance(precheck_meta, Mapping):
                report_sha = precheck_meta.get("report_sha256")
                if report_sha:
                    alert_context["paper_precheck_report_sha256"] = str(report_sha)
                report_path_value = precheck_meta.get("report_path")
                if report_path_value:
                    alert_context["paper_precheck_report_path"] = str(report_path_value)

            if json_sync_cfg:
                json_sync_result = _sync_smoke_json_log(
                    json_sync_cfg=json_sync_cfg,
                    json_log_path=audit_json_path,
                    environment=args.environment,
                    record_id=str(record_id or ""),
                    timestamp=log_timestamp,
                    secret_manager=secret_manager,
                )
                if json_sync_result:
                    alert_context["paper_smoke_json_sync_backend"] = json_sync_result.backend
                    alert_context["paper_smoke_json_sync_location"] = json_sync_result.location
                    metadata = json_sync_result.metadata
                    version_id = metadata.get("version_id") if isinstance(metadata, Mapping) else None
                    if version_id:
                        alert_context["paper_smoke_json_sync_version_id"] = str(version_id)

        auto_publish_required = bool(args.paper_smoke_auto_publish_required)
        auto_publish_enabled = bool(args.paper_smoke_auto_publish or auto_publish_required)

        publish_exit_code: int | None = None
        publish_result: Mapping[str, object] | None = _normalize_publish_result(
            {"status": "skipped", "reason": "auto_publish_disabled"},
            exit_code=None,
            required=auto_publish_required,
        )
        publish_requirement_failed = False

        if offline_mode:
            if auto_publish_enabled:
                _LOGGER.info(
                    "Tryb offline: automatyczna publikacja artefaktów zostanie pominięta."
                )
            if auto_publish_required:
                _LOGGER.warning(
                    "Tryb offline: wymaganie automatycznej publikacji artefaktów zostaje pominięte."
                )
            auto_publish_enabled = False
            auto_publish_required = False
            publish_result = _normalize_publish_result(
                {"status": "skipped", "reason": "offline_mode"},
                exit_code=None,
                required=False,
            )

        if auto_publish_enabled:
            publish_exit_code, publish_payload = _auto_publish_smoke_artifacts(
                config_path=Path(args.config),
                environment=args.environment,
                report_dir=report_dir,
                json_log_path=audit_json_path,
                summary_json_path=summary_output_path,
                archive_path=archive_path,
                record_id=publish_record_id,
                skip_json_sync=json_sync_result is not None,
                skip_archive_upload=upload_result is not None,
                dry_run=args.dry_run,
            )
            publish_result = _normalize_publish_result(
                publish_payload,
                exit_code=publish_exit_code,
                required=auto_publish_required,
            )
            publish_exit_code = _coerce_exit_code(publish_result.get("exit_code"))

            if auto_publish_required and not _is_publish_result_ok(publish_result):
                publish_requirement_failed = True

            if publish_result:
                _append_publish_context(alert_context, publish_result)
            if publish_requirement_failed:
                severity = "error"
                body += " Publikacja artefaktów wymagana – wynik nie spełnia kryteriów."
            elif publish_exit_code and publish_exit_code != 0:
                severity = "error"
                body += " Publikacja artefaktów zakończona błędem."
            elif publish_result and str(publish_result.get("status")) == "ok":
                body += " Artefakty opublikowane automatycznie."

        message = AlertMessage(
            category="paper_smoke",
            title=f"Smoke test paper trading ({args.environment})",
            body=body,
            severity=severity,
            context=alert_context,
        )
        pipeline.bootstrap.alert_router.dispatch(message)

        if json_record:
            compliance_context = {
                "environment": args.environment,
                "json_log_path": str(audit_json_path),
                "json_record_id": str(json_record.get("record_id", "")),
                "summary_sha256": summary_hash,
                "operator": operator_name or _resolve_operator_name(None),
                "paper_precheck_status": precheck_status or "unknown",
                "paper_precheck_coverage_status": precheck_coverage_status or "unknown",
                "paper_precheck_risk_status": precheck_risk_status or "unknown",
            }
            if upload_result:
                compliance_context["archive_upload_backend"] = upload_result.backend
                compliance_context["archive_upload_location"] = upload_result.location
                for key, value in upload_result.metadata.items():
                    compliance_context[f"archive_upload_{key}"] = str(value)
            if json_sync_result:
                compliance_context["paper_smoke_json_sync_backend"] = json_sync_result.backend
                compliance_context["paper_smoke_json_sync_location"] = json_sync_result.location
                compliance_context.update(json_sync_result.metadata)
            if publish_result:
                _append_publish_context(compliance_context, publish_result)
            compliance_message = AlertMessage(
                category="paper_smoke_compliance",
                title=f"Compliance audit log updated ({args.environment})",
                body=(
                    "Zaktualizowano dziennik JSONL smoke testów paper tradingu. "
                    f"Rekord: {json_record.get('record_id', 'unknown')}"
                ),
                severity=(
                    "error"
                    if severity.lower() == "error"
                    else ("warning" if severity.lower() == "warning" else "info")
                ),
                context=compliance_context,
            )
            pipeline.bootstrap.alert_router.dispatch(compliance_message)

        if summary_output_path is not None:
            try:
                summary_payload = _build_smoke_summary_payload(
                    environment=args.environment,
                    timestamp=log_timestamp,
                    operator=operator_name or _resolve_operator_name(None),
                    window=window_meta,
                    report_dir=report_dir,
                    summary_path=summary_path,
                    summary_sha256=summary_hash,
                    severity=severity,
                    storage_context=storage_context or None,
                    precheck_status=precheck_status,
                    precheck_coverage_status=precheck_coverage_status,
                    precheck_risk_status=precheck_risk_status,
                    precheck_payload=precheck_payload,
                    json_log_path=audit_json_path,
                    json_record=json_record,
                    json_sync_result=json_sync_result,
                    archive_path=archive_path,
                    archive_upload_result=upload_result,
                    publish_result=publish_result,
                )
                _write_smoke_summary_json(summary_output_path, summary_payload)
                _LOGGER.info("Zapisano podsumowanie smoke testu do %s", summary_output_path)
            except Exception:  # noqa: BLE001
                _LOGGER.exception(
                    "Nie udało się zapisać podsumowania smoke testu do %s",
                    summary_output_path,
                )

        if publish_requirement_failed:
            status_text = str(publish_result.get("status", "unknown")) if publish_result else "unknown"
            _LOGGER.error(
                "Smoke test zakończony niepowodzeniem: auto-publikacja artefaktów wymagana, status=%s, exit_code=%s.",
                status_text,
                publish_exit_code,
            )
            return 6

        if fail_low_storage:
            free_str = storage_context.get("storage_free_mb", "?")
            thresh_str = storage_context.get("storage_threshold_mb", str(args.smoke_min_free_mb or "?"))
            _LOGGER.error(
                "Smoke test zakończony niepowodzeniem: wolne miejsce %s MB poniżej wymaganego progu %s MB.",
                free_str,
                thresh_str,
            )
            return 4

        return 0

    # normalny tryb realtime / run-once
    trading_controller = create_trading_controller(
        pipeline, pipeline.bootstrap.alert_router, health_check_interval=args.health_interval,
    )

    runner = DailyTrendRealtimeRunner(
        controller=pipeline.controller,
        trading_controller=trading_controller,
        history_bars=max(1, args.history_bars),
    )

    if args.run_once:
        _LOGGER.info("Uruchamiam pojedynczą iterację strategii dla środowiska %s", args.environment)
        results = runner.run_once()
        if results:
            _log_order_results(results)
        else:
            _LOGGER.info("Brak sygnałów w tej iteracji – nic nie zlecam.")
        return 0

    return _run_loop(runner, args.poll_seconds)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
