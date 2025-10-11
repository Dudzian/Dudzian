"""Uruchamia lokalny serwer MetricsService dla telemetrii powłoki Qt/QML."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.telemetry_risk_profiles import (
    get_metrics_service_overrides,
    list_risk_profile_names,
    load_risk_profiles_with_metadata,
    risk_profile_metadata,
)

# --- tryb rozszerzony: CoreConfig + audyt plików/bezpieczeństwa ---------------
from bot_core.config.loader import load_core_config
from bot_core.config.models import CoreConfig, MetricsServiceConfig
try:
    # w niektórych gałęziach może nie istnieć
    from bot_core.config.models import MetricsServiceTlsConfig  # type: ignore
except Exception:
    MetricsServiceTlsConfig = None  # type: ignore[assignment]

from bot_core.runtime.file_metadata import (
    directory_metadata as _directory_metadata,
    file_reference_metadata as _file_reference_metadata,
    log_security_warnings as _log_security_warnings,
    permissions_from_mode as _permissions_from_mode,
    security_flags_from_mode as _security_flags_from_mode,
)

# --- runtime gRPC jest opcjonalny podczas dev/CI (brak stubów/grpcio) --------
try:  # pragma: no cover - defensywne, gdy moduł runtime nie jest dostępny
    from bot_core.runtime import JsonlSink as _RuntimeJsonlSink, create_metrics_server
except Exception as exc:  # pragma: no cover - import z runtime może się nie powieść
    _RuntimeJsonlSink = None  # type: ignore[assignment]
    create_metrics_server = None  # type: ignore[assignment]
    _METRICS_RUNTIME_IMPORT_ERROR = exc
else:  # pragma: no cover - informacje diagnostyczne
    _METRICS_RUNTIME_IMPORT_ERROR = None

JsonlSink = _RuntimeJsonlSink  # alias dla zachowania istniejącego API modułu

METRICS_RUNTIME_AVAILABLE = JsonlSink is not None and create_metrics_server is not None
METRICS_RUNTIME_UNAVAILABLE_MESSAGE = (
    "Brak wsparcia gRPC dla MetricsService (zainstaluj grpcio i wygeneruj stuby bot_core.generated.*)."
)
if _METRICS_RUNTIME_IMPORT_ERROR is not None:
    METRICS_RUNTIME_UNAVAILABLE_MESSAGE = (
        f"{METRICS_RUNTIME_UNAVAILABLE_MESSAGE} Szczegóły: {_METRICS_RUNTIME_IMPORT_ERROR}"
    )

try:  # pragma: no cover - integracja alertów UI jest opcjonalna
    from bot_core.runtime.metrics_alerts import (  # type: ignore
        DEFAULT_UI_ALERTS_JSONL_PATH,
        UiTelemetryAlertSink,
    )
except Exception:  # pragma: no cover - brak modułu alertów UI
    UiTelemetryAlertSink = None  # type: ignore
    DEFAULT_UI_ALERTS_JSONL_PATH = Path("logs/ui_telemetry_alerts.jsonl")

try:  # pragma: no cover - router alertów może być niedostępny
    from bot_core.alerts import (  # type: ignore
        DefaultAlertRouter,
        FileAlertAuditLog,
        InMemoryAlertAuditLog,
    )
except Exception:  # pragma: no cover - brak infrastruktury alertowej
    DefaultAlertRouter = None  # type: ignore
    InMemoryAlertAuditLog = None  # type: ignore
    FileAlertAuditLog = None  # type: ignore

LOGGER = logging.getLogger("run_metrics_service")

UI_ALERT_MODE_CHOICES = ("enable", "jsonl", "disable")
UI_ALERT_AUDIT_BACKEND_CHOICES = ("auto", "file", "memory")
_DEFAULT_UI_CATEGORY = "ui.performance"
_DEFAULT_UI_SEVERITY_ACTIVE = "warning"
_DEFAULT_UI_SEVERITY_RECOVERED = "info"
_DEFAULT_OVERLAY_SEVERITY_CRITICAL = "critical"
_DEFAULT_OVERLAY_THRESHOLD = 2
_DEFAULT_JANK_SEVERITY_SPIKE = "warning"
_DEFAULT_JANK_SEVERITY_CRITICAL: str | None = None
_DEFAULT_JANK_CRITICAL_THRESHOLD_MS: float | None = None
_DEFAULT_UI_ALERT_AUDIT_PATTERN = "metrics-ui-alerts-%Y%m%d.jsonl"
_DEFAULT_UI_ALERT_AUDIT_RETENTION_DAYS = 90


def _configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _parse_env_bool(value: str, *, variable: str, parser: argparse.ArgumentParser) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    parser.error(
        f"Nieprawidłowa wartość '{value}' w zmiennej {variable} – użyj true/false, 1/0, yes/no."
    )
    raise AssertionError("parser.error nie przerwał działania")


def _initial_value_sources(provided_flags: Iterable[str]) -> dict[str, str]:
    provided = set(provided_flags)
    sources: dict[str, str] = {}
    if "--host" in provided:
        sources["host"] = "cli"
    if "--port" in provided:
        sources["port"] = "cli"
    if "--history-size" in provided:
        sources["history_size"] = "cli"
    if "--no-log-sink" in provided:
        sources["log_sink"] = "cli"
    if "--jsonl" in provided:
        sources["jsonl_path"] = "cli"
    if "--jsonl-fsync" in provided:
        sources["jsonl_fsync"] = "cli"
    if "--risk-profiles-file" in provided:
        sources["risk_profiles_file"] = "cli"
    if "--print-risk-profiles" in provided:
        sources["print_risk_profiles"] = "cli"
    if "--ui-alerts-jsonl" in provided:
        sources["ui_alerts_jsonl_path"] = "cli"
    if "--disable-ui-alerts" in provided:
        sources["disable_ui_alerts"] = "cli"
    if "--ui-alerts-risk-profile" in provided:
        sources["ui_alerts_risk_profile"] = "cli"
    if "--ui-alerts-reduce-mode" in provided:
        sources["ui_alerts_reduce_mode"] = "cli"
    if "--ui-alerts-overlay-mode" in provided:
        sources["ui_alerts_overlay_mode"] = "cli"
    if "--ui-alerts-reduce-category" in provided:
        sources["ui_alerts_reduce_category"] = "cli"
    if "--ui-alerts-reduce-active-severity" in provided:
        sources["ui_alerts_reduce_active_severity"] = "cli"
    if "--ui-alerts-reduce-recovered-severity" in provided:
        sources["ui_alerts_reduce_recovered_severity"] = "cli"
    if "--ui-alerts-overlay-category" in provided:
        sources["ui_alerts_overlay_category"] = "cli"
    if "--ui-alerts-overlay-exceeded-severity" in provided:
        sources["ui_alerts_overlay_exceeded_severity"] = "cli"
    if "--ui-alerts-overlay-recovered-severity" in provided:
        sources["ui_alerts_overlay_recovered_severity"] = "cli"
    if "--ui-alerts-overlay-critical-severity" in provided:
        sources["ui_alerts_overlay_critical_severity"] = "cli"
    if "--ui-alerts-overlay-critical-threshold" in provided:
        sources["ui_alerts_overlay_critical_threshold"] = "cli"
    if "--ui-alerts-jank-mode" in provided:
        sources["ui_alerts_jank_mode"] = "cli"
    if "--ui-alerts-jank-category" in provided:
        sources["ui_alerts_jank_category"] = "cli"
    if "--ui-alerts-jank-spike-severity" in provided:
        sources["ui_alerts_jank_spike_severity"] = "cli"
    if "--ui-alerts-jank-critical-severity" in provided:
        sources["ui_alerts_jank_critical_severity"] = "cli"
    if "--ui-alerts-jank-critical-over-ms" in provided:
        sources["ui_alerts_jank_critical_over_ms"] = "cli"
    if "--ui-alerts-audit-dir" in provided:
        sources["ui_alerts_audit_dir"] = "cli"
    if "--ui-alerts-audit-backend" in provided:
        sources["ui_alerts_audit_backend"] = "cli"
    if "--ui-alerts-audit-pattern" in provided:
        sources["ui_alerts_audit_pattern"] = "cli"
    if "--ui-alerts-audit-retention-days" in provided:
        sources["ui_alerts_audit_retention_days"] = "cli"
    if "--ui-alerts-audit-fsync" in provided:
        sources["ui_alerts_audit_fsync"] = "cli"
    if "--fail-on-security-warnings" in provided:
        sources["fail_on_security_warnings"] = "cli"
    if "--tls-cert" in provided:
        sources["tls_cert"] = "cli"
    if "--tls-key" in provided:
        sources["tls_key"] = "cli"
    if "--tls-client-ca" in provided:
        sources["tls_client_ca"] = "cli"
    if "--tls-require-client-cert" in provided:
        sources["tls_require_client_cert"] = "cli"
    if "--auth-token" in provided:
        sources["auth_token"] = "cli"
    return sources


def _finalize_value_sources(sources: dict[str, str]) -> dict[str, str]:
    keys = {
        "host",
        "port",
        "history_size",
        "log_sink",
        "jsonl_path",
        "jsonl_fsync",
        "risk_profiles_file",
        "fail_on_security_warnings",
        "tls_cert",
        "tls_key",
        "tls_client_ca",
        "tls_require_client_cert",
        "ui_alerts_jsonl_path",
        "ui_alerts_reduce_mode",
        "ui_alerts_overlay_mode",
        "ui_alerts_reduce_category",
        "ui_alerts_reduce_active_severity",
        "ui_alerts_reduce_recovered_severity",
        "ui_alerts_overlay_category",
        "ui_alerts_overlay_exceeded_severity",
        "ui_alerts_overlay_recovered_severity",
        "ui_alerts_overlay_critical_severity",
        "ui_alerts_overlay_critical_threshold",
        "ui_alerts_jank_mode",
        "ui_alerts_jank_category",
        "ui_alerts_jank_spike_severity",
        "ui_alerts_jank_critical_severity",
        "ui_alerts_jank_critical_over_ms",
        "ui_alerts_audit_dir",
        "ui_alerts_audit_backend",
        "ui_alerts_audit_pattern",
        "ui_alerts_audit_retention_days",
        "ui_alerts_audit_fsync",
        "disable_ui_alerts",
        "ui_alerts_risk_profile",
        "auth_token",
    }
    for key in keys:
        sources.setdefault(key, "default")
    return sources


def _compact_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Zwraca kopię mapy bez wartości None."""
    return {key: value for key, value in mapping.items() if value is not None}


def _apply_environment_overrides(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
    provided_flags: set[str],
    value_sources: dict[str, str],
) -> tuple[Mapping[str, object], set[str]]:
    """Nakłada ustawienia ze zmiennych środowiskowych na argumenty CLI."""

    entries: list[dict[str, object]] = []
    override_keys: set[str] = set()

    def record_entry(**entry: object) -> None:
        entries.append(entry)  # type: ignore[arg-type]

    def skip_due_to_cli(option: str, env_var: str, raw: str) -> None:
        record_entry(
            option=option,
            variable=env_var,
            raw_value=raw,
            applied=False,
            reason="cli_override",
        )

    def apply_value(option: str, env_var: str, raw: str, value: Any, **extra: object) -> None:
        override_keys.add(option)
        value_sources[option] = "env"
        record_entry(
            option=option,
            variable=env_var,
            raw_value=raw,
            applied=True,
            parsed_value=value,
            **extra,
        )

    env_map = [
        ("host", "RUN_METRICS_SERVICE_HOST", "--host"),
        ("port", "RUN_METRICS_SERVICE_PORT", "--port"),
        ("history_size", "RUN_METRICS_SERVICE_HISTORY_SIZE", "--history-size"),
        ("log_sink", "RUN_METRICS_SERVICE_NO_LOG_SINK", "--no-log-sink"),
        ("jsonl_path", "RUN_METRICS_SERVICE_JSONL", "--jsonl"),
        ("jsonl_fsync", "RUN_METRICS_SERVICE_JSONL_FSYNC", "--jsonl-fsync"),
        (
            "risk_profiles_file",
            "RUN_METRICS_SERVICE_RISK_PROFILES_FILE",
            "--risk-profiles-file",
        ),
        (
            "print_risk_profiles",
            "RUN_METRICS_SERVICE_PRINT_RISK_PROFILES",
            "--print-risk-profiles",
        ),
        (
            "ui_alerts_jsonl_path",
            "RUN_METRICS_SERVICE_UI_ALERTS_JSONL",
            "--ui-alerts-jsonl",
        ),
        (
            "disable_ui_alerts",
            "RUN_METRICS_SERVICE_DISABLE_UI_ALERTS",
            "--disable-ui-alerts",
        ),
        (
            "ui_alerts_risk_profile",
            "RUN_METRICS_SERVICE_UI_ALERTS_RISK_PROFILE",
            "--ui-alerts-risk-profile",
        ),
        (
            "ui_alerts_reduce_mode",
            "RUN_METRICS_SERVICE_UI_ALERTS_REDUCE_MODE",
            "--ui-alerts-reduce-mode",
        ),
        (
            "ui_alerts_overlay_mode",
            "RUN_METRICS_SERVICE_UI_ALERTS_OVERLAY_MODE",
            "--ui-alerts-overlay-mode",
        ),
        (
            "ui_alerts_reduce_category",
            "RUN_METRICS_SERVICE_UI_ALERTS_REDUCE_CATEGORY",
            "--ui-alerts-reduce-category",
        ),
        (
            "ui_alerts_reduce_active_severity",
            "RUN_METRICS_SERVICE_UI_ALERTS_REDUCE_ACTIVE_SEVERITY",
            "--ui-alerts-reduce-active-severity",
        ),
        (
            "ui_alerts_reduce_recovered_severity",
            "RUN_METRICS_SERVICE_UI_ALERTS_REDUCE_RECOVERED_SEVERITY",
            "--ui-alerts-reduce-recovered-severity",
        ),
        (
            "ui_alerts_overlay_category",
            "RUN_METRICS_SERVICE_UI_ALERTS_OVERLAY_CATEGORY",
            "--ui-alerts-overlay-category",
        ),
        (
            "ui_alerts_overlay_exceeded_severity",
            "RUN_METRICS_SERVICE_UI_ALERTS_OVERLAY_EXCEEDED_SEVERITY",
            "--ui-alerts-overlay-exceeded-severity",
        ),
        (
            "ui_alerts_overlay_recovered_severity",
            "RUN_METRICS_SERVICE_UI_ALERTS_OVERLAY_RECOVERED_SEVERITY",
            "--ui-alerts-overlay-recovered-severity",
        ),
        (
            "ui_alerts_overlay_critical_severity",
            "RUN_METRICS_SERVICE_UI_ALERTS_OVERLAY_CRITICAL_SEVERITY",
            "--ui-alerts-overlay-critical-severity",
        ),
        (
            "ui_alerts_overlay_critical_threshold",
            "RUN_METRICS_SERVICE_UI_ALERTS_OVERLAY_CRITICAL_THRESHOLD",
            "--ui-alerts-overlay-critical-threshold",
        ),
        (
            "ui_alerts_jank_mode",
            "RUN_METRICS_SERVICE_UI_ALERTS_JANK_MODE",
            "--ui-alerts-jank-mode",
        ),
        (
            "ui_alerts_jank_category",
            "RUN_METRICS_SERVICE_UI_ALERTS_JANK_CATEGORY",
            "--ui-alerts-jank-category",
        ),
        (
            "ui_alerts_jank_spike_severity",
            "RUN_METRICS_SERVICE_UI_ALERTS_JANK_SPIKE_SEVERITY",
            "--ui-alerts-jank-spike-severity",
        ),
        (
            "ui_alerts_jank_critical_severity",
            "RUN_METRICS_SERVICE_UI_ALERTS_JANK_CRITICAL_SEVERITY",
            "--ui-alerts-jank-critical-severity",
        ),
        (
            "ui_alerts_jank_critical_over_ms",
            "RUN_METRICS_SERVICE_UI_ALERTS_JANK_CRITICAL_OVER_MS",
            "--ui-alerts-jank-critical-over-ms",
        ),
        (
            "ui_alerts_audit_dir",
            "RUN_METRICS_SERVICE_UI_ALERTS_AUDIT_DIR",
            "--ui-alerts-audit-dir",
        ),
        (
            "ui_alerts_audit_backend",
            "RUN_METRICS_SERVICE_UI_ALERTS_AUDIT_BACKEND",
            "--ui-alerts-audit-backend",
        ),
        (
            "ui_alerts_audit_pattern",
            "RUN_METRICS_SERVICE_UI_ALERTS_AUDIT_PATTERN",
            "--ui-alerts-audit-pattern",
        ),
        (
            "ui_alerts_audit_retention_days",
            "RUN_METRICS_SERVICE_UI_ALERTS_AUDIT_RETENTION_DAYS",
            "--ui-alerts-audit-retention-days",
        ),
        (
            "ui_alerts_audit_fsync",
            "RUN_METRICS_SERVICE_UI_ALERTS_AUDIT_FSYNC",
            "--ui-alerts-audit-fsync",
        ),
        (
            "fail_on_security_warnings",
            "RUN_METRICS_SERVICE_FAIL_ON_SECURITY_WARNINGS",
            "--fail-on-security-warnings",
        ),
        ("tls_cert", "RUN_METRICS_SERVICE_TLS_CERT", "--tls-cert"),
        ("tls_key", "RUN_METRICS_SERVICE_TLS_KEY", "--tls-key"),
        ("tls_client_ca", "RUN_METRICS_SERVICE_TLS_CLIENT_CA", "--tls-client-ca"),
        (
            "tls_require_client_cert",
            "RUN_METRICS_SERVICE_TLS_REQUIRE_CLIENT_CERT",
            "--tls-require-client-cert",
        ),
        ("shutdown_after", "RUN_METRICS_SERVICE_SHUTDOWN_AFTER", "--shutdown-after"),
        ("log_level", "RUN_METRICS_SERVICE_LOG_LEVEL", "--log-level"),
        ("auth_token", "RUN_METRICS_SERVICE_AUTH_TOKEN", "--auth-token"),
    ]

    for option, env_var, cli_flag in env_map:
        raw = os.getenv(env_var)
        if raw is None:
            continue
        if cli_flag in provided_flags:
            skip_due_to_cli(option, env_var, raw)
            continue

        if option == "host":
            args.host = raw
            apply_value(option, env_var, raw, raw)
        elif option == "port":
            try:
                port_value = int(raw)
            except ValueError:
                parser.error(
                    f"Zmienna {env_var} musi zawierać liczbę całkowitą – otrzymano '{raw}'."
                )
            args.port = port_value
            apply_value(option, env_var, raw, port_value)
        elif option == "history_size":
            try:
                history = int(raw)
            except ValueError:
                parser.error(
                    f"Zmienna {env_var} musi zawierać liczbę całkowitą – otrzymano '{raw}'."
                )
            args.history_size = history
            apply_value(option, env_var, raw, history)
        elif option == "log_sink":
            value = _parse_env_bool(raw, variable=env_var, parser=parser)
            args.no_log_sink = bool(value)
            apply_value(option, env_var, raw, bool(value))
        elif option == "jsonl_path":
            normalized = raw.strip().lower()
            if normalized in {"", "none", "null", "off", "disable", "disabled"}:
                args.jsonl = None
                value_sources[option] = "env_disabled"
                override_keys.add(option)
                record_entry(
                    option=option,
                    variable=env_var,
                    raw_value=raw,
                    applied=True,
                    parsed_value=None,
                    note="jsonl_disabled",
                )
            else:
                path_value = Path(raw)
                args.jsonl = path_value
                apply_value(option, env_var, raw, str(path_value))
        elif option == "jsonl_fsync":
            fsync_value = _parse_env_bool(raw, variable=env_var, parser=parser)
            args.jsonl_fsync = bool(fsync_value)
            apply_value(option, env_var, raw, bool(fsync_value))
        elif option == "risk_profiles_file":
            normalized = raw.strip()
            if not normalized:
                args.risk_profiles_file = None
                apply_value(option, env_var, raw, None, note="risk_profiles_disabled")
            else:
                path_value = Path(normalized).expanduser()
                args.risk_profiles_file = path_value
                apply_value(option, env_var, raw, str(path_value))
        elif option == "print_risk_profiles":
            enabled = _parse_env_bool(raw, variable=env_var, parser=parser)
            args.print_risk_profiles = bool(enabled)
            apply_value(option, env_var, raw, bool(enabled))
        elif option == "ui_alerts_jsonl_path":
            normalized = raw.strip().lower()
            if normalized in {"", "none", "null", "off", "disable", "disabled"}:
                args.ui_alerts_jsonl = None
                value_sources[option] = "env_disabled"
                override_keys.add(option)
                record_entry(
                    option=option,
                    variable=env_var,
                    raw_value=raw,
                    applied=True,
                    parsed_value=None,
                    note="ui_alerts_disabled",
                )
            else:
                path_value = Path(raw)
                args.ui_alerts_jsonl = path_value
                apply_value(option, env_var, raw, str(path_value))
        elif option == "disable_ui_alerts":
            disabled = _parse_env_bool(raw, variable=env_var, parser=parser)
            args.disable_ui_alerts = bool(disabled)
            apply_value(option, env_var, raw, bool(disabled))
        elif option == "ui_alerts_risk_profile":
            normalized = raw.strip().lower()
            args.ui_alerts_risk_profile = normalized
            apply_value(option, env_var, raw, normalized)
        elif option == "ui_alerts_reduce_mode":
            normalized = raw.strip().lower()
            if normalized not in UI_ALERT_MODE_CHOICES:
                parser.error(
                    f"Nieprawidłowa wartość '{raw}' w {env_var} – dozwolone: {', '.join(UI_ALERT_MODE_CHOICES)}."
                )
            args.ui_alerts_reduce_mode = normalized
            apply_value(option, env_var, raw, normalized)
        elif option == "ui_alerts_overlay_mode":
            normalized = raw.strip().lower()
            if normalized not in UI_ALERT_MODE_CHOICES:
                parser.error(
                    f"Nieprawidłowa wartość '{raw}' w {env_var} – dozwolone: {', '.join(UI_ALERT_MODE_CHOICES)}."
                )
            args.ui_alerts_overlay_mode = normalized
            apply_value(option, env_var, raw, normalized)
        elif option == "ui_alerts_jank_mode":
            normalized = raw.strip().lower()
            if normalized not in UI_ALERT_MODE_CHOICES:
                parser.error(
                    f"Nieprawidłowa wartość '{raw}' w {env_var} – dozwolone: {', '.join(UI_ALERT_MODE_CHOICES)}."
                )
            args.ui_alerts_jank_mode = normalized
            apply_value(option, env_var, raw, normalized)
        elif option in {
            "ui_alerts_reduce_category",
            "ui_alerts_reduce_active_severity",
            "ui_alerts_reduce_recovered_severity",
            "ui_alerts_overlay_category",
            "ui_alerts_overlay_exceeded_severity",
            "ui_alerts_overlay_recovered_severity",
            "ui_alerts_overlay_critical_severity",
        }:
            setattr(args, option, raw)
            apply_value(option, env_var, raw, raw)
        elif option == "ui_alerts_jank_category":
            args.ui_alerts_jank_category = raw
            apply_value(option, env_var, raw, raw)
        elif option == "ui_alerts_jank_spike_severity":
            args.ui_alerts_jank_spike_severity = raw
            apply_value(option, env_var, raw, raw)
        elif option == "ui_alerts_jank_critical_severity":
            normalized = raw.strip().lower()
            if normalized in {"", "none", "null", "off", "disable", "disabled"}:
                args.ui_alerts_jank_critical_severity = None
                value_sources[option] = "env_disabled"
                override_keys.add(option)
                record_entry(
                    option=option,
                    variable=env_var,
                    raw_value=raw,
                    applied=True,
                    parsed_value=None,
                    note="jank_critical_severity_disabled",
                )
            else:
                args.ui_alerts_jank_critical_severity = raw
                apply_value(option, env_var, raw, raw)
        elif option == "ui_alerts_jank_critical_over_ms":
            try:
                threshold_ms = float(raw)
            except ValueError:
                parser.error(
                    f"Zmienna {env_var} musi zawierać liczbę zmiennoprzecinkową – otrzymano '{raw}'."
                )
            args.ui_alerts_jank_critical_over_ms = threshold_ms
            apply_value(option, env_var, raw, threshold_ms)
        elif option == "ui_alerts_audit_dir":
            normalized = raw.strip().lower()
            if normalized in {"", "none", "null", "off", "disable", "disabled"}:
                args.ui_alerts_audit_dir = None
                value_sources[option] = "env_disabled"
                override_keys.add(option)
                record_entry(
                    option=option,
                    variable=env_var,
                    raw_value=raw,
                    applied=True,
                    parsed_value=None,
                    note="ui_alerts_audit_disabled",
                )
            else:
                path_value = Path(raw)
                args.ui_alerts_audit_dir = path_value
                apply_value(option, env_var, raw, str(path_value))
        elif option == "ui_alerts_audit_backend":
            normalized = raw.strip().lower()
            if normalized in {"", "auto"}:
                args.ui_alerts_audit_backend = None
                value_sources[option] = "env_auto"
                override_keys.add(option)
                record_entry(
                    option=option,
                    variable=env_var,
                    raw_value=raw,
                    applied=True,
                    parsed_value=None,
                    note="ui_alerts_audit_backend_auto",
                )
            elif normalized in UI_ALERT_AUDIT_BACKEND_CHOICES:
                args.ui_alerts_audit_backend = normalized
                apply_value(option, env_var, raw, normalized)
            else:
                parser.error(
                    f"Nieprawidłowa wartość '{raw}' w {env_var} – dozwolone: {', '.join(UI_ALERT_AUDIT_BACKEND_CHOICES)}."
                )
        elif option == "ui_alerts_audit_pattern":
            normalized = raw.strip()
            if not normalized:
                args.ui_alerts_audit_pattern = None
                value_sources[option] = "env_disabled"
                override_keys.add(option)
                record_entry(
                    {
                        "option": option,
                        "variable": env_var,
                        "raw_value": raw,
                        "applied": True,
                        "parsed_value": None,
                        "note": "ui_alerts_audit_pattern_default",
                    }
                )
            else:
                args.ui_alerts_audit_pattern = normalized
                apply_value(option, env_var, raw, normalized)
        elif option == "ui_alerts_audit_retention_days":
            normalized = raw.strip()
            if normalized == "":
                args.ui_alerts_audit_retention_days = None
                value_sources[option] = "env_disabled"
                override_keys.add(option)
                record_entry(
                    {
                        "option": option,
                        "variable": env_var,
                        "raw_value": raw,
                        "applied": True,
                        "parsed_value": None,
                        "note": "ui_alerts_audit_retention_default",
                    }
                )
            else:
                try:
                    retention = int(raw)
                except ValueError:
                    parser.error(
                        f"Zmienna {env_var} musi zawierać liczbę całkowitą – otrzymano '{raw}'."
                    )
                args.ui_alerts_audit_retention_days = retention
                apply_value(option, env_var, raw, retention)
        elif option == "ui_alerts_audit_fsync":
            fsync_value = _parse_env_bool(raw, variable=env_var, parser=parser)
            args.ui_alerts_audit_fsync = bool(fsync_value)
            apply_value(option, env_var, raw, bool(fsync_value))
        elif option == "ui_alerts_overlay_critical_threshold":
            try:
                threshold = int(raw)
            except ValueError:
                parser.error(
                    f"Zmienna {env_var} musi zawierać liczbę całkowitą – otrzymano '{raw}'."
                )
            args.ui_alerts_overlay_critical_threshold = threshold
            apply_value(option, env_var, raw, threshold)
        elif option == "fail_on_security_warnings":
            value = _parse_env_bool(raw, variable=env_var, parser=parser)
            args.fail_on_security_warnings = bool(value)
            apply_value(option, env_var, raw, bool(value))
        elif option == "tls_cert":
            normalized = raw.strip().lower()
            if normalized in {"", "none", "null", "off", "disable", "disabled"}:
                args.tls_cert = None
                value_sources[option] = "env_disabled"
                override_keys.add(option)
                record_entry(
                    option=option,
                    variable=env_var,
                    raw_value=raw,
                    applied=True,
                    parsed_value=None,
                    note="tls_cert_disabled",
                )
            else:
                path_value = Path(raw)
                args.tls_cert = path_value
                apply_value(option, env_var, raw, str(path_value))
        elif option == "tls_key":
            normalized = raw.strip().lower()
            if normalized in {"", "none", "null", "off", "disable", "disabled"}:
                args.tls_key = None
                value_sources[option] = "env_disabled"
                override_keys.add(option)
                record_entry(
                    option=option,
                    variable=env_var,
                    raw_value=raw,
                    applied=True,
                    parsed_value=None,
                    note="tls_key_disabled",
                )
            else:
                path_value = Path(raw)
                args.tls_key = path_value
                apply_value(option, env_var, raw, str(path_value))
        elif option == "tls_client_ca":
            normalized = raw.strip().lower()
            if normalized in {"", "none", "null", "off", "disable", "disabled"}:
                args.tls_client_ca = None
                value_sources[option] = "env_disabled"
                override_keys.add(option)
                record_entry(
                    option=option,
                    variable=env_var,
                    raw_value=raw,
                    applied=True,
                    parsed_value=None,
                    note="tls_client_ca_disabled",
                )
            else:
                path_value = Path(raw)
                args.tls_client_ca = path_value
                apply_value(option, env_var, raw, str(path_value))
        elif option == "tls_require_client_cert":
            require_value = _parse_env_bool(raw, variable=env_var, parser=parser)
            args.tls_require_client_cert = bool(require_value)
            apply_value(option, env_var, raw, bool(require_value))
        elif option == "shutdown_after":
            try:
                timeout_value = float(raw)
            except ValueError:
                parser.error(
                    f"Zmienna {env_var} musi zawierać liczbę zmiennoprzecinkową – otrzymano '{raw}'."
                )
            args.shutdown_after = timeout_value
            apply_value(option, env_var, raw, timeout_value)
        elif option == "log_level":
            args.log_level = raw
            apply_value(option, env_var, raw, raw)
        elif option == "auth_token":
            args.auth_token = raw
            apply_value(option, env_var, raw, raw)

    section: Mapping[str, object] | None = None
    if entries:
        section = {"entries": entries}
    return section or {}, override_keys


def _ui_alerts_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "reduce_mode": args.ui_alerts_reduce_mode or "enable",
        "overlay_mode": args.ui_alerts_overlay_mode or "enable",
        "jank_mode": args.ui_alerts_jank_mode or "enable",
        "reduce_motion_category": args.ui_alerts_reduce_category or _DEFAULT_UI_CATEGORY,
        "reduce_motion_severity_active": (
            args.ui_alerts_reduce_active_severity or _DEFAULT_UI_SEVERITY_ACTIVE
        ),
        "reduce_motion_severity_recovered": (
            args.ui_alerts_reduce_recovered_severity or _DEFAULT_UI_SEVERITY_RECOVERED
        ),
        "overlay_category": args.ui_alerts_overlay_category or _DEFAULT_UI_CATEGORY,
        "overlay_severity_exceeded": (
            args.ui_alerts_overlay_exceeded_severity or _DEFAULT_UI_SEVERITY_ACTIVE
        ),
        "overlay_severity_recovered": (
            args.ui_alerts_overlay_recovered_severity or _DEFAULT_UI_SEVERITY_RECOVERED
        ),
        "overlay_severity_critical": (
            args.ui_alerts_overlay_critical_severity or _DEFAULT_OVERLAY_SEVERITY_CRITICAL
        ),
        "overlay_critical_threshold": (
            args.ui_alerts_overlay_critical_threshold
            if args.ui_alerts_overlay_critical_threshold is not None
            else _DEFAULT_OVERLAY_THRESHOLD
        ),
        "jank_category": args.ui_alerts_jank_category or _DEFAULT_UI_CATEGORY,
        "jank_severity_spike": args.ui_alerts_jank_spike_severity or _DEFAULT_JANK_SEVERITY_SPIKE,
        "jank_severity_critical": (
            args.ui_alerts_jank_critical_severity or _DEFAULT_JANK_SEVERITY_CRITICAL
        ),
        "jank_critical_over_ms": (
            args.ui_alerts_jank_critical_over_ms
            if args.ui_alerts_jank_critical_over_ms is not None
            else _DEFAULT_JANK_CRITICAL_THRESHOLD_MS
        ),
    }


def _load_custom_risk_profiles(
    args: argparse.Namespace, *, parser: argparse.ArgumentParser
) -> None:
    """Ładuje dodatkowe profile ryzyka z pliku, jeśli podano."""

    path_value = getattr(args, "risk_profiles_file", None)
    if not path_value:
        args._ui_alerts_risk_profiles_file_metadata = None
        return

    target = Path(path_value).expanduser()
    try:
        registered, metadata = load_risk_profiles_with_metadata(
            target, origin_label=f"metrics_service:{target}"
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))
    except Exception as exc:  # noqa: BLE001 - chcemy pełny komunikat
        parser.error(f"Nie udało się wczytać profili ryzyka z {target}: {exc}")
    else:
        if registered:
            LOGGER.info(
                "Załadowano %s profil(e) ryzyka telemetrii z %s", len(registered), target
            )

    args._ui_alerts_risk_profiles_file_metadata = dict(metadata)


def _apply_risk_profile_defaults(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
    value_sources: dict[str, str],
) -> None:
    profile_name = getattr(args, "ui_alerts_risk_profile", None)
    if not profile_name:
        return

    try:
        overrides = get_metrics_service_overrides(profile_name)
    except KeyError:
        parser.error(
            f"Profil ryzyka {profile_name!r} nie jest obsługiwany. Dostępne: {', '.join(list_risk_profile_names())}"
        )
        return  # pragma: no cover - parser.error przerywa wykonanie

    args._ui_alerts_risk_profile_metadata = risk_profile_metadata(profile_name)

    for option, value in overrides.items():
        source = value_sources.get(option)
        if source == "cli" or (isinstance(source, str) and source.startswith("env")):
            continue
        setattr(args, option, value)
        value_sources[option] = f"risk_profile:{profile_name}"

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Startuje serwer MetricsService odbierający telemetrię UI (gRPC).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Adres nasłuchu (domyślnie 127.0.0.1).")
    parser.add_argument(
        "--port",
        type=int,
        default=50062,
        help="Port gRPC (0 = wybierz losowy wolny port).",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=1024,
        help="Rozmiar pamięci historii metryk (domyślnie 1024 wpisy).",
    )
    parser.add_argument(
        "--no-log-sink",
        action="store_true",
        help="Wyłącz domyślny LoggingSink (loguje snapshoty do stdout).",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Ścieżka do pliku JSONL, do którego mają trafiać snapshoty telemetryczne.",
    )
    parser.add_argument(
        "--jsonl-fsync",
        action="store_true",
        help="Wymuś fsync po każdym wpisie JSONL (kosztem wydajności).",
    )
    parser.add_argument(
        "--ui-alerts-jsonl",
        type=Path,
        default=None,
        help="Ścieżka pliku JSONL dla alertów UI (reduce-motion/overlay).",
    )
    parser.add_argument(
        "--disable-ui-alerts",
        action="store_true",
        help="Wyłącza generowanie alertów UI po stronie serwera telemetrii.",
    )
    parser.add_argument(
        "--ui-alerts-risk-profile",
        default=None,
        help="Zastosuj predefiniowany profil ryzyka dla alertów UI.",
    )
    parser.add_argument(
        "--risk-profiles-file",
        type=Path,
        default=None,
        help=(
            "Ścieżka do pliku JSON/YAML z dodatkowymi profilami ryzyka telemetrii. "
            "Profil można następnie wybrać przez --ui-alerts-risk-profile."
        ),
    )
    parser.add_argument(
        "--print-risk-profiles",
        action="store_true",
        help="Wypisz dostępne profile ryzyka telemetrii i zakończ działanie.",
    )
    parser.add_argument(
        "--ui-alerts-reduce-mode",
        choices=UI_ALERT_MODE_CHOICES,
        default=None,
        help="Steruje wysyłką alertów reduce-motion (enable/disable).",
    )
    parser.add_argument(
        "--ui-alerts-overlay-mode",
        choices=UI_ALERT_MODE_CHOICES,
        default=None,
        help="Steruje wysyłką alertów budżetu overlayów (enable/disable).",
    )
    parser.add_argument(
        "--ui-alerts-reduce-category",
        default=None,
        help="Nadpisuje kategorię alertów reduce-motion (domyślnie ui.performance).",
    )
    parser.add_argument(
        "--ui-alerts-reduce-active-severity",
        default=None,
        help="Severity alertu przy aktywacji reduce-motion (domyślnie warning).",
    )
    parser.add_argument(
        "--ui-alerts-reduce-recovered-severity",
        default=None,
        help="Severity alertu przy powrocie z reduce-motion (domyślnie info).",
    )
    parser.add_argument(
        "--ui-alerts-overlay-category",
        default=None,
        help="Nadpisuje kategorię alertów overlay budget (domyślnie ui.performance).",
    )
    parser.add_argument(
        "--ui-alerts-overlay-exceeded-severity",
        default=None,
        help="Severity alertu przy przekroczeniu budżetu overlayów (domyślnie warning).",
    )
    parser.add_argument(
        "--ui-alerts-overlay-recovered-severity",
        default=None,
        help="Severity alertu przy powrocie budżetu overlayów (domyślnie info).",
    )
    parser.add_argument(
        "--ui-alerts-overlay-critical-severity",
        default=None,
        help="Severity alertu krytycznego (różnica >= threshold).",
    )
    parser.add_argument(
        "--ui-alerts-overlay-critical-threshold",
        type=int,
        default=None,
        help="Liczba nakładek powyżej limitu wymuszająca severity krytyczne.",
    )
    parser.add_argument(
        "--ui-alerts-jank-mode",
        choices=UI_ALERT_MODE_CHOICES,
        default=None,
        help="Steruje wysyłką alertów jank (enable/jsonl/disable).",
    )
    parser.add_argument(
        "--ui-alerts-jank-category",
        default=None,
        help="Nadpisuje kategorię alertów jank (domyślnie ui.performance).",
    )
    parser.add_argument(
        "--ui-alerts-jank-spike-severity",
        default=None,
        help="Severity alertu jank dla pojedynczego skoku (domyślnie warning).",
    )
    parser.add_argument(
        "--ui-alerts-jank-critical-severity",
        default=None,
        help="Severity alertu jank po przekroczeniu progu krytycznego (domyślnie brak).",
    )
    parser.add_argument(
        "--ui-alerts-jank-critical-over-ms",
        type=float,
        default=None,
        help="Próg przekroczenia (ms) dla eskalacji alertu jank do severity krytycznego.",
    )
    parser.add_argument(
        "--ui-alerts-audit-dir",
        type=Path,
        default=None,
        help="Katalog audytu alertów UI (JSONL rotowane dziennie).",
    )
    parser.add_argument(
        "--ui-alerts-audit-backend",
        choices=UI_ALERT_AUDIT_BACKEND_CHOICES,
        default=None,
        help="Preferowany backend audytu alertów UI (auto/file/memory).",
    )
    parser.add_argument(
        "--ui-alerts-audit-pattern",
        default=None,
        help="Wzorzec nazw plików audytu UI (strftime, domyślnie metrics-ui-alerts-%%Y%%m%%d.jsonl).",
    )
    parser.add_argument(
        "--ui-alerts-audit-retention-days",
        type=int,
        default=None,
        help="Retencja audytu alertów UI w dniach (domyślnie 90).",
    )
    parser.add_argument(
        "--ui-alerts-audit-fsync",
        action="store_true",
        help="Wymuś fsync po każdym wpisie w plikach audytu alertów UI.",
    )

    # TLS/mTLS
    parser.add_argument(
        "--tls-cert",
        type=Path,
        default=None,
        help="Ścieżka certyfikatu TLS serwera (PEM).",
    )
    parser.add_argument(
        "--tls-key",
        type=Path,
        default=None,
        help="Ścieżka klucza prywatnego TLS serwera (PEM).",
    )
    parser.add_argument(
        "--tls-client-ca",
        type=Path,
        default=None,
        help="Opcjonalny plik CA klientów do mTLS (PEM).",
    )
    parser.add_argument(
        "--tls-require-client-cert",
        action="store_true",
        help="Wymagaj certyfikatu klienta (mTLS).",
    )

    # Token (gałęzie bez TLS mogą go wymagać)
    parser.add_argument(
        "--auth-token",
        default=None,
        help="Opcjonalny token autoryzacyjny wymagany od klientów (Authorization: Bearer token).",
    )

    parser.add_argument(
        "--shutdown-after",
        type=float,
        default=None,
        help="Automatycznie zatrzymaj serwer po tylu sekundach (przydatne w CI).",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        help="Poziom logowania (debug, info, warning, error).",
    )
    parser.add_argument(
        "--print-address",
        action="store_true",
        help="Wypisz końcowy adres serwera na stdout (do użycia w skryptach CI).",
    )
    parser.add_argument(
        "--config-plan-jsonl",
        type=Path,
        default=None,
        help="Dopisuje do wskazanego pliku JSONL audyt konfiguracji serwera przed startem.",
    )
    parser.add_argument(
        "--print-config-plan",
        action="store_true",
        help="Wypisz na stdout efektywną konfigurację MetricsService i zakończ działanie.",
    )
    parser.add_argument(
        "--fail-on-security-warnings",
        action="store_true",
        help=(
            "Zakończ działanie, jeśli konfiguracja zawiera ostrzeżenia bezpieczeństwa dotyczące plików JSONL/TLS."
        ),
    )
    parser.add_argument(
        "--core-config",
        type=Path,
        default=None,
        help=(
            "Ścieżka do pliku core.yaml. Ustawienia runtime.metrics_service zostaną użyte jako"
            " domyślne wartości flag CLI."
        ),
    )
    return parser


def _build_server(
    *,
    host: str,
    port: int,
    history_size: int,
    enable_logging_sink: bool,
    jsonl_path: Path | None,
    jsonl_fsync: bool,
    auth_token: str | None,
    enable_ui_alerts: bool,
    ui_alerts_jsonl_path: Path | None,
    ui_alerts_options: Mapping[str, Any] | None = None,
    ui_alerts_config: Mapping[str, Any] | None = None,
    ui_alerts_audit_dir: Path | None = None,
    ui_alerts_audit_backend: str | None = None,
    ui_alerts_audit_pattern: str | None = None,
    ui_alerts_audit_retention_days: int | None = None,
    ui_alerts_audit_fsync: bool = False,
    extra_sinks: Iterable = (),
    tls_config=None,
):
    if not METRICS_RUNTIME_AVAILABLE:
        raise RuntimeError(METRICS_RUNTIME_UNAVAILABLE_MESSAGE)

    sinks = list(extra_sinks)
    ui_alerts_path: Path | None = None
    ui_alerts_settings: dict[str, Any] | None = None
    if enable_ui_alerts:
        requested_backend = (ui_alerts_audit_backend or "auto").lower()
        if requested_backend not in UI_ALERT_AUDIT_BACKEND_CHOICES:
            raise ValueError(f"Nieobsługiwany backend audytu UI: {requested_backend}")
        audit_backend: dict[str, object] = {"requested": requested_backend}
        file_backend_requested = ui_alerts_audit_dir is not None or requested_backend == "file"
        file_backend_error = False
        memory_forced = requested_backend == "memory"
        if requested_backend == "file" and ui_alerts_audit_dir is None:
            raise ValueError("Backend plikowy audytu UI wymaga --ui-alerts-audit-dir")
        if UiTelemetryAlertSink is None or DefaultAlertRouter is None:
            LOGGER.debug(
                "UiTelemetryAlertSink lub infrastruktura alertów jest niedostępna – alerty UI będą wyłączone."
            )
        else:
            ui_alerts_path = (
                ui_alerts_jsonl_path.expanduser()
                if ui_alerts_jsonl_path is not None
                else DEFAULT_UI_ALERTS_JSONL_PATH.expanduser()
            )
            audit_log = None
            if not memory_forced and ui_alerts_audit_dir is not None:
                if FileAlertAuditLog is None:
                    if requested_backend == "file":
                        raise RuntimeError(
                            "Wymuszono backend plikowy alertów UI, ale FileAlertAuditLog jest niedostępny."
                        )
                    LOGGER.warning(
                        "Wymuszono katalog audytu alertów UI (%s), ale FileAlertAuditLog nie jest dostępny – przełączam na backend w pamięci.",
                        ui_alerts_audit_dir,
                    )
                else:
                    resolved_dir = Path(ui_alerts_audit_dir).expanduser()
                    pattern = ui_alerts_audit_pattern or _DEFAULT_UI_ALERT_AUDIT_PATTERN
                    retention = (
                        ui_alerts_audit_retention_days
                        if ui_alerts_audit_retention_days is not None
                        else _DEFAULT_UI_ALERT_AUDIT_RETENTION_DAYS
                    )
                    try:
                        audit_log = FileAlertAuditLog(
                            directory=resolved_dir,
                            filename_pattern=pattern,
                            retention_days=retention,
                            fsync=ui_alerts_audit_fsync,
                        )
                        audit_backend.update(
                            {
                                "backend": "file",
                                "directory": str(resolved_dir),
                                "pattern": pattern,
                                "retention_days": retention,
                                "fsync": bool(ui_alerts_audit_fsync),
                            }
                        )
                    except Exception as exc:
                        file_backend_error = True
                        if requested_backend == "file":
                            raise RuntimeError(
                                "Nie udało się zainicjalizować FileAlertAuditLog przy wymuszonym backendzie file"
                            ) from exc
                        LOGGER.exception(
                            "Nie udało się zainicjalizować FileAlertAuditLog – używam pamięci jako backendu audytu."
                        )
            if audit_log is None:
                if InMemoryAlertAuditLog is None:
                    LOGGER.debug(
                        "Brak dostępnego backendu audytu alertów UI – alerty zostaną wyłączone."
                    )
                    ui_alerts_path = None
                    audit_backend.setdefault("backend", None)
                    audit_backend.setdefault("note", "no_audit_backend_available")
                else:
                    audit_log = InMemoryAlertAuditLog()
                    audit_backend.update(
                        {
                            "backend": "memory",
                            "fsync": bool(ui_alerts_audit_fsync),
                        }
                    )
                    if requested_backend == "memory" and ui_alerts_audit_dir is not None:
                        audit_backend["note"] = "directory_ignored_memory_backend"
                    elif file_backend_requested and requested_backend != "memory":
                        audit_backend["note"] = (
                            "file_backend_error" if file_backend_error else "file_backend_unavailable"
                        )
            elif "backend" not in audit_backend:
                audit_backend["backend"] = "file"
            if ui_alerts_path is not None and audit_log is not None:
                try:
                    router = DefaultAlertRouter(audit_log=audit_log)

                    class _LoggingChannel:
                        name = "metrics_ui_alerts"

                        def send(self, message) -> None:  # pragma: no cover - log only
                            LOGGER.warning(
                                "[UI ALERT] %s | %s | %s",
                                message.category,
                                message.severity,
                                message.title,
                            )

                        def health_check(self) -> dict[str, str]:  # pragma: no cover
                            return {"status": "ok"}

                    router.register(_LoggingChannel())
                except Exception:
                    LOGGER.exception("Nie udało się zainicjalizować routera alertów UI – alerty zostaną wyłączone.")
                    router = None
                if router is not None:
                    options = dict(ui_alerts_options or {})
                    reduce_mode = str(options.get("reduce_mode") or "enable").lower()
                    overlay_mode = str(options.get("overlay_mode") or "enable").lower()
                    jank_mode = str(options.get("jank_mode") or "enable").lower()
                    reduce_dispatch = reduce_mode == "enable"
                    overlay_dispatch = overlay_mode == "enable"
                    jank_dispatch = jank_mode == "enable"
                    reduce_logging = reduce_mode in {"enable", "jsonl"}
                    overlay_logging = overlay_mode in {"enable", "jsonl"}
                    jank_logging = jank_mode in {"enable", "jsonl"}
                    reduce_category = options.get("reduce_category") or _DEFAULT_UI_CATEGORY
                    reduce_active = options.get("reduce_active_severity") or _DEFAULT_UI_SEVERITY_ACTIVE
                    reduce_recovered = options.get("reduce_recovered_severity") or _DEFAULT_UI_SEVERITY_RECOVERED
                    overlay_category = options.get("overlay_category") or _DEFAULT_UI_CATEGORY
                    overlay_exceeded = options.get("overlay_exceeded_severity") or _DEFAULT_UI_SEVERITY_ACTIVE
                    overlay_recovered = options.get("overlay_recovered_severity") or _DEFAULT_UI_SEVERITY_RECOVERED
                    overlay_critical = options.get("overlay_critical_severity") or _DEFAULT_OVERLAY_SEVERITY_CRITICAL
                    overlay_threshold_opt = options.get("overlay_critical_threshold")
                    overlay_threshold = (
                        int(overlay_threshold_opt)
                        if overlay_threshold_opt is not None
                        else _DEFAULT_OVERLAY_THRESHOLD
                    )
                    jank_category = options.get("jank_category") or _DEFAULT_UI_CATEGORY
                    jank_spike = options.get("jank_spike_severity") or _DEFAULT_JANK_SEVERITY_SPIKE
                    jank_critical = options.get("jank_critical_severity")
                    if jank_critical is None and _DEFAULT_JANK_SEVERITY_CRITICAL is not None:
                        jank_critical = _DEFAULT_JANK_SEVERITY_CRITICAL
                    jank_threshold_opt = options.get("jank_critical_over_ms")
                    if jank_threshold_opt is not None:
                        try:
                            jank_threshold = float(jank_threshold_opt)
                        except (TypeError, ValueError):
                            jank_threshold = None
                    else:
                        jank_threshold = _DEFAULT_JANK_CRITICAL_THRESHOLD_MS
                    sink_kwargs = dict(
                        jsonl_path=ui_alerts_path,
                        enable_reduce_motion_alerts=reduce_dispatch,
                        enable_overlay_alerts=overlay_dispatch,
                        enable_jank_alerts=jank_dispatch,
                        log_reduce_motion_events=reduce_logging,
                        log_overlay_events=overlay_logging,
                        log_jank_events=jank_logging,
                        reduce_motion_category=reduce_category,
                        reduce_motion_severity_active=reduce_active,
                        reduce_motion_severity_recovered=reduce_recovered,
                        overlay_category=overlay_category,
                        overlay_severity_exceeded=overlay_exceeded,
                        overlay_severity_recovered=overlay_recovered,
                        jank_category=jank_category,
                        jank_severity_spike=jank_spike,
                    )
                    if overlay_critical:
                        sink_kwargs["overlay_severity_critical"] = overlay_critical
                    if overlay_threshold is not None:
                        sink_kwargs["overlay_critical_threshold"] = overlay_threshold
                    if jank_critical:
                        sink_kwargs["jank_severity_critical"] = jank_critical
                    if jank_threshold is not None:
                        sink_kwargs["jank_critical_over_ms"] = float(jank_threshold)

                    ui_alerts_settings = {
                        "path": str(ui_alerts_path),
                        "reduce_mode": reduce_mode,
                        "overlay_mode": overlay_mode,
                        "jank_mode": jank_mode,
                        "reduce_motion_alerts": reduce_dispatch,
                        "overlay_alerts": overlay_dispatch,
                        "jank_alerts": jank_dispatch,
                        "reduce_motion_logging": reduce_logging,
                        "overlay_logging": overlay_logging,
                        "jank_logging": jank_logging,
                        "reduce_motion_category": reduce_category,
                        "reduce_motion_severity_active": reduce_active,
                        "reduce_motion_severity_recovered": reduce_recovered,
                        "overlay_category": overlay_category,
                        "overlay_severity_exceeded": overlay_exceeded,
                        "overlay_severity_recovered": overlay_recovered,
                        "overlay_severity_critical": overlay_critical,
                        "overlay_critical_threshold": overlay_threshold,
                        "jank_category": jank_category,
                        "jank_severity_spike": jank_spike,
                        "jank_severity_critical": jank_critical,
                        "jank_critical_over_ms": jank_threshold,
                    }
                    if ui_alerts_config is not None:
                        ui_alerts_settings.update(dict(ui_alerts_config))
                        ui_alerts_settings.setdefault("path", str(ui_alerts_path))

                    try:
                        ui_sink = UiTelemetryAlertSink(router, **sink_kwargs)
                    except Exception:
                        LOGGER.exception(
                            "Nie udało się uruchomić UiTelemetryAlertSink – alerty UI zostaną pominięte"
                        )
                        ui_alerts_path = None
                        ui_alerts_settings = None
                    else:
                        if "backend" not in audit_backend:
                            audit_backend.setdefault("backend", "memory")
                            audit_backend.setdefault("fsync", bool(ui_alerts_audit_fsync))
                        ui_alerts_settings["audit"] = dict(audit_backend)
                        sinks.append(ui_sink)
                        LOGGER.info("Alerty UI aktywne (log JSONL: %s)", ui_alerts_path)
                else:
                    ui_alerts_path = None
                    ui_alerts_settings = None
    if jsonl_path is not None:
        if JsonlSink is None:
            raise RuntimeError(METRICS_RUNTIME_UNAVAILABLE_MESSAGE)
        sinks.append(JsonlSink(jsonl_path, fsync=jsonl_fsync))
    if create_metrics_server is None:
        raise RuntimeError(METRICS_RUNTIME_UNAVAILABLE_MESSAGE)

    # Odporność na różne sygnatury create_metrics_server
    base_kwargs = dict(
        host=host,
        port=port,
        history_size=history_size,
        enable_logging_sink=enable_logging_sink,
        sinks=sinks,
    )
    # preferuj przekazanie wszystkiego; jeśli runtime nie zna opcji – degraduj łagodnie
    attempts: list[dict[str, Any]] = []
    kw = dict(base_kwargs)
    if tls_config is not None:
        kw["tls_config"] = tls_config
    if auth_token is not None:
        kw["auth_token"] = auth_token
    if ui_alerts_path is not None:
        kw["ui_alerts_jsonl_path"] = ui_alerts_path
    if ui_alerts_settings is not None:
        kw["ui_alerts_config"] = ui_alerts_settings
    attempts.append(kw)

    # fallbacki
    if "tls_config" in kw:
        k2 = dict(base_kwargs)
        if auth_token is not None:
            k2["auth_token"] = auth_token
        if ui_alerts_path is not None:
            k2["ui_alerts_jsonl_path"] = ui_alerts_path
        if ui_alerts_settings is not None:
            k2["ui_alerts_config"] = ui_alerts_settings
        attempts.append(k2)
    if "auth_token" in kw:
        k3 = dict(base_kwargs)
        if tls_config is not None:
            k3["tls_config"] = tls_config
        if ui_alerts_path is not None:
            k3["ui_alerts_jsonl_path"] = ui_alerts_path
        if ui_alerts_settings is not None:
            k3["ui_alerts_config"] = ui_alerts_settings
        attempts.append(k3)
    attempts.append(dict(base_kwargs))

    last_exc: Exception | None = None
    for k in attempts:
        try:
            if ui_alerts_path is not None:
                k.setdefault("ui_alerts_jsonl_path", ui_alerts_path)
            return create_metrics_server(**k)  # type: ignore[misc]
        except TypeError as exc:
            last_exc = exc
            continue
    # jeśli wszystkie próby zawiodły
    raise RuntimeError(f"Nie udało się wywołać create_metrics_server z kompatybilnymi argumentami: {last_exc}")


def _collect_provided_flags(raw_args: Iterable[str]) -> set[str]:
    """Zwraca zbiór flag CLI przekazanych przez operatora."""
    provided: set[str] = set()
    for token in raw_args:
        if not token.startswith("--"):
            continue
        name = token.split("=", 1)[0]
        provided.add(name)
    return provided


def _apply_core_metrics_config(
    args: argparse.Namespace,
    metrics_config: MetricsServiceConfig,
    *,
    provided_flags: set[str],
    env_overrides: set[str],
    value_sources: dict[str, str],
) -> dict[str, str]:
    """Aktualizuje parametry CLI na podstawie CoreConfig i zwraca źródła wartości."""
    sources: dict[str, str] = {}

    def flag_provided(flag: str) -> bool:
        return flag in provided_flags

    def mark_env(option: str) -> None:
        sources[option] = value_sources.get(option, "env")

    # Najpierw oznaczamy parametry nadpisane przez zmienne środowiskowe.
    for option in env_overrides:
        mark_env(option)

    if "host" not in env_overrides:
        if flag_provided("--host"):
            sources.setdefault("host", "cli")
        else:
            args.host = metrics_config.host
            sources["host"] = "core_config"
            value_sources.setdefault("host", "core_config")

    if "port" not in env_overrides:
        if flag_provided("--port"):
            sources.setdefault("port", "cli")
        else:
            args.port = metrics_config.port
            sources["port"] = "core_config"
            value_sources.setdefault("port", "core_config")

    if "history_size" not in env_overrides:
        if flag_provided("--history-size"):
            sources.setdefault("history_size", "cli")
        else:
            args.history_size = metrics_config.history_size
            sources["history_size"] = "core_config"
            value_sources.setdefault("history_size", "core_config")

    if "log_sink" not in env_overrides:
        if flag_provided("--no-log-sink"):
            sources.setdefault("log_sink", "cli")
        else:
            args.no_log_sink = not getattr(metrics_config, "log_sink", True)
            sources["log_sink"] = "core_config"
            value_sources.setdefault("log_sink", "core_config")

    if "jsonl_path" not in env_overrides:
        if flag_provided("--jsonl"):
            sources.setdefault("jsonl_path", "cli")
        elif getattr(metrics_config, "jsonl_path", None):
            args.jsonl = Path(metrics_config.jsonl_path)  # type: ignore[arg-type]
            sources["jsonl_path"] = "core_config"
            value_sources.setdefault("jsonl_path", "core_config")
        else:
            sources.setdefault("jsonl_path", "core_config_none")
            value_sources.setdefault("jsonl_path", "core_config_none")

    if "ui_alerts_jsonl_path" not in env_overrides:
        if flag_provided("--ui-alerts-jsonl"):
            sources.setdefault("ui_alerts_jsonl_path", "cli")
        elif getattr(metrics_config, "ui_alerts_jsonl_path", None):
            args.ui_alerts_jsonl = Path(metrics_config.ui_alerts_jsonl_path)  # type: ignore[arg-type]
            sources["ui_alerts_jsonl_path"] = "core_config"
            value_sources.setdefault("ui_alerts_jsonl_path", "core_config")
        else:
            sources.setdefault("ui_alerts_jsonl_path", "core_config_none")
            value_sources.setdefault("ui_alerts_jsonl_path", "core_config_none")

    def _mode_from_config(*, dispatch_attr: str, mode_attr: str) -> tuple[str, str]:
        raw_mode = getattr(metrics_config, mode_attr, None)
        if raw_mode is not None:
            normalized = str(raw_mode).lower()
            if normalized not in UI_ALERT_MODE_CHOICES:
                normalized = (
                    "enable" if bool(getattr(metrics_config, dispatch_attr, False)) else "disable"
                )
                origin = "core_config_invalid_mode"
            else:
                origin = "core_config_mode"
            return normalized, origin
        dispatch_enabled = bool(getattr(metrics_config, dispatch_attr, False))
        normalized = "enable" if dispatch_enabled else "disable"
        origin = "core_config_enable" if dispatch_enabled else "core_config_disable"
        return normalized, origin

    if "ui_alerts_reduce_mode" not in env_overrides:
        if flag_provided("--ui-alerts-reduce-mode"):
            sources.setdefault("ui_alerts_reduce_mode", "cli")
        else:
            normalized, origin = _mode_from_config(
                dispatch_attr="reduce_motion_alerts",
                mode_attr="reduce_motion_mode",
            )
            args.ui_alerts_reduce_mode = normalized
            sources["ui_alerts_reduce_mode"] = origin
            value_sources.setdefault("ui_alerts_reduce_mode", origin)

    if "ui_alerts_overlay_mode" not in env_overrides:
        if flag_provided("--ui-alerts-overlay-mode"):
            sources.setdefault("ui_alerts_overlay_mode", "cli")
        else:
            normalized, origin = _mode_from_config(
                dispatch_attr="overlay_alerts",
                mode_attr="overlay_alert_mode",
            )
            args.ui_alerts_overlay_mode = normalized
            sources["ui_alerts_overlay_mode"] = origin
            value_sources.setdefault("ui_alerts_overlay_mode", origin)

    if "ui_alerts_jank_mode" not in env_overrides:
        if flag_provided("--ui-alerts-jank-mode"):
            sources.setdefault("ui_alerts_jank_mode", "cli")
        else:
            normalized, origin = _mode_from_config(
                dispatch_attr="jank_alerts",
                mode_attr="jank_alert_mode",
            )
            args.ui_alerts_jank_mode = normalized
            sources["ui_alerts_jank_mode"] = origin
            value_sources.setdefault("ui_alerts_jank_mode", origin)

    if "ui_alerts_audit_backend" not in env_overrides:
        if flag_provided("--ui-alerts-audit-backend"):
            sources.setdefault("ui_alerts_audit_backend", "cli")
        else:
            raw_backend = getattr(metrics_config, "ui_alerts_audit_backend", None)
            if raw_backend is not None:
                normalized = str(raw_backend).strip().lower()
                if normalized in {"", "auto"}:
                    origin = "core_config_auto"
                    args.ui_alerts_audit_backend = None
                elif normalized in UI_ALERT_AUDIT_BACKEND_CHOICES:
                    origin = "core_config"
                    args.ui_alerts_audit_backend = normalized
                else:
                    origin = "core_config_invalid_backend"
                    args.ui_alerts_audit_backend = None
                sources["ui_alerts_audit_backend"] = origin
                value_sources.setdefault("ui_alerts_audit_backend", origin)

    def _assign_if_present(option: str, attr_name: str, default_value: str | None = None) -> None:
        if option in env_overrides:
            return
        cli_flag = f"--{option.replace('_', '-')}"
        if flag_provided(cli_flag):
            sources.setdefault(option, "cli")
            return
        if getattr(metrics_config, attr_name, None) is not None:
            value = getattr(metrics_config, attr_name)
            setattr(args, option, value)
            sources[option] = "core_config"
            value_sources.setdefault(option, "core_config")
        elif default_value is not None:
            setattr(args, option, default_value)
            sources.setdefault(option, "default")
            value_sources.setdefault(option, "default")

    _assign_if_present("ui_alerts_reduce_category", "reduce_motion_category", _DEFAULT_UI_CATEGORY)
    _assign_if_present(
        "ui_alerts_reduce_active_severity",
        "reduce_motion_severity_active",
        _DEFAULT_UI_SEVERITY_ACTIVE,
    )
    _assign_if_present(
        "ui_alerts_reduce_recovered_severity",
        "reduce_motion_severity_recovered",
        _DEFAULT_UI_SEVERITY_RECOVERED,
    )
    _assign_if_present("ui_alerts_overlay_category", "overlay_alert_category", _DEFAULT_UI_CATEGORY)
    _assign_if_present(
        "ui_alerts_overlay_exceeded_severity",
        "overlay_alert_severity_exceeded",
        _DEFAULT_UI_SEVERITY_ACTIVE,
    )
    _assign_if_present(
        "ui_alerts_overlay_recovered_severity",
        "overlay_alert_severity_recovered",
        _DEFAULT_UI_SEVERITY_RECOVERED,
    )
    _assign_if_present(
        "ui_alerts_overlay_critical_severity",
        "overlay_alert_severity_critical",
        _DEFAULT_OVERLAY_SEVERITY_CRITICAL,
    )
    _assign_if_present(
        "ui_alerts_jank_category",
        "jank_alert_category",
        _DEFAULT_UI_CATEGORY,
    )
    _assign_if_present(
        "ui_alerts_jank_spike_severity",
        "jank_alert_severity_spike",
        _DEFAULT_JANK_SEVERITY_SPIKE,
    )
    _assign_if_present(
        "ui_alerts_jank_critical_severity",
        "jank_alert_severity_critical",
        _DEFAULT_JANK_SEVERITY_CRITICAL,
    )

    if "ui_alerts_overlay_critical_threshold" not in env_overrides:
        if flag_provided("--ui-alerts-overlay-critical-threshold"):
            sources.setdefault("ui_alerts_overlay_critical_threshold", "cli")
        elif getattr(metrics_config, "overlay_alert_critical_threshold", None) is not None:
            threshold_value = getattr(metrics_config, "overlay_alert_critical_threshold")
            try:
                args.ui_alerts_overlay_critical_threshold = int(threshold_value)
            except (TypeError, ValueError):
                args.ui_alerts_overlay_critical_threshold = None
            else:
                sources["ui_alerts_overlay_critical_threshold"] = "core_config"
                value_sources.setdefault("ui_alerts_overlay_critical_threshold", "core_config")
        else:
            args.ui_alerts_overlay_critical_threshold = _DEFAULT_OVERLAY_THRESHOLD
            sources.setdefault("ui_alerts_overlay_critical_threshold", "default")
            value_sources.setdefault("ui_alerts_overlay_critical_threshold", "default")

    if "ui_alerts_jank_critical_over_ms" not in env_overrides:
        if flag_provided("--ui-alerts-jank-critical-over-ms"):
            sources.setdefault("ui_alerts_jank_critical_over_ms", "cli")
        elif getattr(metrics_config, "jank_alert_critical_over_ms", None) is not None:
            raw_value = getattr(metrics_config, "jank_alert_critical_over_ms")
            try:
                args.ui_alerts_jank_critical_over_ms = float(raw_value)
            except (TypeError, ValueError):
                args.ui_alerts_jank_critical_over_ms = None
            else:
                sources["ui_alerts_jank_critical_over_ms"] = "core_config"
                value_sources.setdefault("ui_alerts_jank_critical_over_ms", "core_config")
        else:
            args.ui_alerts_jank_critical_over_ms = _DEFAULT_JANK_CRITICAL_THRESHOLD_MS
            sources.setdefault("ui_alerts_jank_critical_over_ms", "default")
            value_sources.setdefault("ui_alerts_jank_critical_over_ms", "default")

    if "jsonl_fsync" not in env_overrides:
        if flag_provided("--jsonl-fsync"):
            sources.setdefault("jsonl_fsync", "cli")
        else:
            args.jsonl_fsync = bool(getattr(metrics_config, "jsonl_fsync", False))
            sources["jsonl_fsync"] = "core_config"
            value_sources.setdefault("jsonl_fsync", "core_config")

    # auth token (jeśli występuje w configu i nie nadpisany)
    if "auth_token" not in env_overrides and not flag_provided("--auth-token"):
        token = getattr(metrics_config, "auth_token", None)
        if token:
            args.auth_token = token
            sources.setdefault("auth_token", "core_config")
            value_sources.setdefault("auth_token", "core_config")

    # TLS – użyjemy getattr, by działać bez klas TLS w starszych gałęziach
    tls_cfg = getattr(metrics_config, "tls", None)
    tls_overridden = {
        key for key in ("tls_cert", "tls_key", "tls_client_ca", "tls_require_client_cert") if key in env_overrides
    }

    enabled = bool(getattr(tls_cfg, "enabled", False)) if tls_cfg is not None else False
    if enabled:
        tls_cli = flag_provided("--tls-cert") or flag_provided("--tls-key")
        if tls_cli and not tls_overridden:
            sources.setdefault("tls_cert", "cli")
            sources.setdefault("tls_key", "cli")
            sources.setdefault(
                "tls_client_ca",
                "cli" if flag_provided("--tls-client-ca") else "core_config_disabled",
            )
            sources.setdefault(
                "tls_require_client_cert",
                "cli" if flag_provided("--tls-require-client-cert") else "core_config_disabled",
            )
        else:
            if "tls_cert" not in tls_overridden and "tls_key" not in tls_overridden:
                cert = getattr(tls_cfg, "certificate_path", None)
                key = getattr(tls_cfg, "private_key_path", None)
                if cert and key:
                    args.tls_cert = Path(cert)
                    args.tls_key = Path(key)
                    sources.setdefault("tls_cert", "core_config")
                    sources.setdefault("tls_key", "core_config")
                    value_sources.setdefault("tls_cert", "core_config")
                    value_sources.setdefault("tls_key", "core_config")
                else:
                    sources.setdefault("tls_cert", "core_config_incomplete")
                    sources.setdefault("tls_key", "core_config_incomplete")
                    value_sources.setdefault("tls_cert", "core_config_incomplete")
                    value_sources.setdefault("tls_key", "core_config_incomplete")
            if "tls_client_ca" not in tls_overridden:
                ca = getattr(tls_cfg, "client_ca_path", None)
                if ca and not flag_provided("--tls-client-ca"):
                    args.tls_client_ca = Path(ca)
                    sources.setdefault("tls_client_ca", "core_config")
                    value_sources.setdefault("tls_client_ca", "core_config")
                else:
                    sources.setdefault(
                        "tls_client_ca",
                        "cli" if flag_provided("--tls-client-ca") else "core_config_none",
                    )
                    value_sources.setdefault(
                        "tls_client_ca",
                        "cli" if flag_provided("--tls-client-ca") else "core_config_none",
                    )
            if "tls_require_client_cert" not in tls_overridden:
                if not flag_provided("--tls-require-client-cert"):
                    args.tls_require_client_cert = bool(getattr(tls_cfg, "require_client_auth", False))
                    sources.setdefault("tls_require_client_cert", "core_config")
                    value_sources.setdefault("tls_require_client_cert", "core_config")
                else:
                    sources.setdefault("tls_require_client_cert", "cli")
    else:
        if "tls_cert" not in tls_overridden:
            sources.setdefault("tls_cert", "cli" if flag_provided("--tls-cert") else "core_config_disabled")
            value_sources.setdefault(
                "tls_cert",
                "cli" if flag_provided("--tls-cert") else "core_config_disabled",
            )
        if "tls_key" not in tls_overridden:
            sources.setdefault("tls_key", "cli" if flag_provided("--tls-key") else "core_config_disabled")
            value_sources.setdefault(
                "tls_key",
                "cli" if flag_provided("--tls-key") else "core_config_disabled",
            )
        if "tls_client_ca" not in tls_overridden:
            sources.setdefault(
                "tls_client_ca",
                "cli" if flag_provided("--tls-client-ca") else "core_config_disabled",
            )
            value_sources.setdefault(
                "tls_client_ca",
                "cli" if flag_provided("--tls-client-ca") else "core_config_disabled",
            )
        if "tls_require_client_cert" not in tls_overridden:
            sources.setdefault(
                "tls_require_client_cert",
                "cli" if flag_provided("--tls-require-client-cert") else "core_config_disabled",
            )
            value_sources.setdefault(
                "tls_require_client_cert",
                "cli" if flag_provided("--tls-require-client-cert") else "core_config_disabled",
            )

    # metadane poza parametrami startowymi
    value_sources.setdefault(
        "ui_alerts_jsonl_path",
        "core_config" if getattr(metrics_config, "ui_alerts_jsonl_path", None) else "core_config_none",
    )
    sources.setdefault(
        "ui_alerts_jsonl_path",
        "core_config" if getattr(metrics_config, "ui_alerts_jsonl_path", None) else "core_config_none",
    )

    return sources


def _build_core_config_section(
    core_config: CoreConfig,
    metrics_config: MetricsServiceConfig | None,
    applied_sources: Mapping[str, str],
    *,
    requested_path: Path | None = None,
) -> Mapping[str, object]:
    """Tworzy sekcję metadanych CoreConfig do snapshotu audytowego."""
    section: dict[str, object] = {
        "path": core_config.source_path,
        "directory": core_config.source_directory,
        "metrics_service_defined": metrics_config is not None,
    }

    if requested_path is not None:
        section["cli_argument"] = str(Path(requested_path).expanduser())

    if core_config.source_path:
        section["file"] = _file_reference_metadata(
            core_config.source_path, role="core_config"
        )
    if core_config.source_directory:
        try:
            section["directory_absolute_path"] = str(
                Path(core_config.source_directory).expanduser().resolve(strict=False)
            )
        except Exception:  # noqa: BLE001
            section["directory_absolute_path"] = str(
                Path(core_config.source_directory).expanduser().absolute()
            )

    metrics_section: dict[str, Any] = {
        "applied_sources": dict(applied_sources),
    }

    if metrics_config is not None:
        metrics_section["enabled"] = bool(getattr(metrics_config, "enabled", True))
        metrics_values: dict[str, Any] = {
            "host": getattr(metrics_config, "host", None),
            "port": getattr(metrics_config, "port", None),
            "history_size": getattr(metrics_config, "history_size", None),
            "log_sink": getattr(metrics_config, "log_sink", None),
            "jsonl_path": getattr(metrics_config, "jsonl_path", None),
            "jsonl_fsync": getattr(metrics_config, "jsonl_fsync", None),
            "ui_alerts_jsonl_path": getattr(metrics_config, "ui_alerts_jsonl_path", None),
            "auth_token_set": bool(getattr(metrics_config, "auth_token", None)),
        }
        if getattr(metrics_config, "jsonl_path", None):
            metrics_values["jsonl_file"] = _file_reference_metadata(
                metrics_config.jsonl_path, role="jsonl"  # type: ignore[arg-type]
            )
        if getattr(metrics_config, "ui_alerts_jsonl_path", None):
            metrics_values["ui_alerts_file"] = _file_reference_metadata(
                metrics_config.ui_alerts_jsonl_path, role="ui_alerts_jsonl"  # type: ignore[arg-type]
            )
        tls_cfg = getattr(metrics_config, "tls", None)
        if tls_cfg is not None:
            tls_section: dict[str, Any] = {
                "enabled": bool(getattr(tls_cfg, "enabled", False)),
                "certificate_path": getattr(tls_cfg, "certificate_path", None),
                "private_key_path": getattr(tls_cfg, "private_key_path", None),
                "client_ca_path": getattr(tls_cfg, "client_ca_path", None),
                "require_client_auth": bool(getattr(tls_cfg, "require_client_auth", False)),
            }
            if getattr(tls_cfg, "certificate_path", None):
                tls_section["certificate_file"] = _file_reference_metadata(
                    tls_cfg.certificate_path, role="tls_cert"  # type: ignore[arg-type]
                )
            if getattr(tls_cfg, "private_key_path", None):
                tls_section["private_key_file"] = _file_reference_metadata(
                    tls_cfg.private_key_path, role="tls_key"  # type: ignore[arg-type]
                )
            if getattr(tls_cfg, "client_ca_path", None):
                tls_section["client_ca_file"] = _file_reference_metadata(
                    tls_cfg.client_ca_path, role="tls_client_ca"  # type: ignore[arg-type]
                )
            metrics_values["tls"] = tls_section
        metrics_section["values"] = metrics_values

    section["metrics_service"] = metrics_section
    return section


def _build_config_plan_payload(
    *,
    args: argparse.Namespace,
    server,
    core_config_section: Mapping[str, object] | None = None,
    runtime_unavailable_reason: str | None = None,
    runtime_unavailable_details: str | None = None,
    environment_overrides: Mapping[str, object] | None = None,
    parameter_sources: Mapping[str, str] | None = None,
) -> Mapping[str, object]:
    """Zwraca opis efektywnej konfiguracji MetricsService."""

    payload: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "address": getattr(server, "address", None) if server is not None else None,
        "host": args.host,
        "port": args.port,
        "history_size": getattr(server, "history_size", args.history_size)
        if server is not None
        else args.history_size,
        "logging_sink_enabled": not args.no_log_sink,
        "shutdown_after": args.shutdown_after,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "working_directory": str(Path.cwd()),
    }

    tls_configured = bool(args.tls_cert and args.tls_key)

    jsonl_section: dict[str, object] = {
        "configured": bool(args.jsonl),
        "fsync": bool(args.jsonl_fsync),
    }
    if args.jsonl:
        jsonl_path = Path(args.jsonl).expanduser()
        jsonl_section["path"] = str(jsonl_path)
        jsonl_section["file"] = _file_reference_metadata(jsonl_path, role="jsonl")
    payload["jsonl_sink"] = jsonl_section

    jsonl_arg_path = Path(args.jsonl).expanduser() if args.jsonl else None

    reduce_mode_value = (args.ui_alerts_reduce_mode or "enable").lower()
    overlay_mode_value = (args.ui_alerts_overlay_mode or "enable").lower()
    jank_mode_value = (args.ui_alerts_jank_mode or "enable").lower()
    reduce_dispatch = reduce_mode_value == "enable"
    overlay_dispatch = overlay_mode_value == "enable"
    jank_dispatch = jank_mode_value == "enable"
    reduce_logging = reduce_mode_value in {"enable", "jsonl"}
    overlay_logging = overlay_mode_value in {"enable", "jsonl"}
    jank_logging = jank_mode_value in {"enable", "jsonl"}

    ui_alerts_section: dict[str, object] = {
        "configured": bool(args.ui_alerts_jsonl),
        "disabled": bool(args.disable_ui_alerts),
        "reduce_mode": reduce_mode_value,
        "overlay_mode": overlay_mode_value,
        "jank_mode": jank_mode_value,
        "reduce_motion_dispatch": reduce_dispatch,
        "overlay_dispatch": overlay_dispatch,
        "jank_dispatch": jank_dispatch,
        "reduce_motion_logging": reduce_logging,
        "overlay_logging": overlay_logging,
        "jank_logging": jank_logging,
    }
    audit_dir_path = (
        Path(args.ui_alerts_audit_dir).expanduser() if args.ui_alerts_audit_dir else None
    )
    audit_pattern = args.ui_alerts_audit_pattern or _DEFAULT_UI_ALERT_AUDIT_PATTERN
    audit_retention = (
        args.ui_alerts_audit_retention_days
        if args.ui_alerts_audit_retention_days is not None
        else _DEFAULT_UI_ALERT_AUDIT_RETENTION_DAYS
    )
    requested_backend = (args.ui_alerts_audit_backend or "auto").lower()
    memory_forced = requested_backend == "memory"
    file_backend_supported = audit_dir_path is not None and FileAlertAuditLog is not None and not memory_forced
    audit_section: dict[str, object] = {
        "requested": requested_backend,
        "fsync": bool(args.ui_alerts_audit_fsync),
    }
    if requested_backend == "file" and audit_dir_path is None:
        audit_section["backend"] = None
        audit_section["note"] = "file_backend_requires_directory"
    elif requested_backend == "file" and FileAlertAuditLog is None:
        audit_section["backend"] = None
        audit_section["note"] = "file_backend_unavailable"
    elif requested_backend == "memory":
        audit_section["backend"] = "memory"
        if audit_dir_path is not None:
            audit_section["note"] = "directory_ignored_memory_backend"
    elif file_backend_supported:
        directory_meta = _directory_metadata(audit_dir_path)
        directory_meta["role"] = "ui_alerts_audit_dir"
        audit_section.update(
            {
                "backend": "file",
                "directory": str(audit_dir_path),
                "pattern": audit_pattern,
                "retention_days": audit_retention,
                "directory_metadata": directory_meta,
            }
        )
    else:
        audit_section.setdefault("backend", "memory")
        if audit_dir_path is not None:
            audit_section.setdefault("note", "file_backend_unavailable")
    if args.ui_alerts_jsonl:
        ui_path = Path(args.ui_alerts_jsonl).expanduser()
        ui_alerts_section["path"] = str(ui_path)
        ui_alerts_section["file"] = _file_reference_metadata(ui_path, role="ui_alerts_jsonl")
    ui_alerts_section.update(
        {
            "reduce_motion_category": args.ui_alerts_reduce_category or _DEFAULT_UI_CATEGORY,
            "reduce_motion_severity_active": args.ui_alerts_reduce_active_severity or _DEFAULT_UI_SEVERITY_ACTIVE,
            "reduce_motion_severity_recovered": args.ui_alerts_reduce_recovered_severity or _DEFAULT_UI_SEVERITY_RECOVERED,
            "overlay_category": args.ui_alerts_overlay_category or _DEFAULT_UI_CATEGORY,
            "overlay_severity_exceeded": args.ui_alerts_overlay_exceeded_severity or _DEFAULT_UI_SEVERITY_ACTIVE,
            "overlay_severity_recovered": args.ui_alerts_overlay_recovered_severity or _DEFAULT_UI_SEVERITY_RECOVERED,
            "overlay_severity_critical": args.ui_alerts_overlay_critical_severity or _DEFAULT_OVERLAY_SEVERITY_CRITICAL,
            "overlay_critical_threshold": (
                args.ui_alerts_overlay_critical_threshold
                if args.ui_alerts_overlay_critical_threshold is not None
                else _DEFAULT_OVERLAY_THRESHOLD
            ),
            "jank_category": args.ui_alerts_jank_category or _DEFAULT_UI_CATEGORY,
            "jank_severity_spike": args.ui_alerts_jank_spike_severity or _DEFAULT_JANK_SEVERITY_SPIKE,
            "jank_severity_critical": (
                args.ui_alerts_jank_critical_severity
                if args.ui_alerts_jank_critical_severity is not None
                else _DEFAULT_JANK_SEVERITY_CRITICAL
            ),
            "jank_critical_over_ms": (
                args.ui_alerts_jank_critical_over_ms
                if args.ui_alerts_jank_critical_over_ms is not None
                else _DEFAULT_JANK_CRITICAL_THRESHOLD_MS
            ),
        }
    )
    risk_profile_meta = getattr(args, "_ui_alerts_risk_profile_metadata", None)
    if risk_profile_meta:
        ui_alerts_section["risk_profile"] = dict(risk_profile_meta)
    risk_profiles_file_meta = getattr(args, "_ui_alerts_risk_profiles_file_metadata", None)
    if risk_profiles_file_meta:
        ui_alerts_section["risk_profiles_file"] = dict(risk_profiles_file_meta)
    ui_alerts_section["audit"] = audit_section
    payload["ui_alerts"] = ui_alerts_section
    payload["metrics"] = {
        "history_size": payload["history_size"],
        "logging_sink_enabled": not args.no_log_sink,
        "jsonl_sink": dict(jsonl_section),
        "ui_alerts": dict(ui_alerts_section),
    }

    ui_alerts_arg_path = Path(args.ui_alerts_jsonl).expanduser() if args.ui_alerts_jsonl else None

    if server is None:
        runtime_state: dict[str, object] = {
            "available": False,
            "reason": runtime_unavailable_reason or "server_not_started",
            "address": None,
            "history_size": args.history_size,
            "sink_count": 0,
            "sinks": [],
            "logging_sink_active": not args.no_log_sink,
            "jsonl_sink_active": bool(args.jsonl),
            "metadata_source": "unavailable",
        }
        if runtime_unavailable_details is not None:
            runtime_state["details"] = runtime_unavailable_details
        runtime_state["jsonl_sink"] = {
            "path": str(jsonl_arg_path) if jsonl_arg_path else None,
            "fsync": bool(args.jsonl_fsync),
        }
        if jsonl_arg_path is not None:
            runtime_state["jsonl_sink"]["file"] = _file_reference_metadata(
                jsonl_arg_path, role="jsonl"
            )
        runtime_state["ui_alerts_sink"] = {
            "path": str(ui_alerts_arg_path) if ui_alerts_arg_path else None,
            "disabled": bool(args.disable_ui_alerts),
        }
        runtime_state["ui_alerts_sink"]["config"] = _ui_alerts_config_from_args(args)
        if ui_alerts_arg_path is not None:
            runtime_state["ui_alerts_sink"]["file"] = _file_reference_metadata(
                ui_alerts_arg_path, role="ui_alerts_jsonl"
            )
        runtime_state["tls"] = {
            "configured": tls_configured,
            "enabled": tls_configured,
            "require_client_auth": bool(args.tls_require_client_cert),
        }
        runtime_state["additional_sinks"] = []
        runtime_state["sink_descriptions"] = []
    else:
        runtime_metadata = dict(getattr(server, "runtime_metadata", {}) or {})
        sinks_info: list[dict[str, object]] = []
        jsonl_runtime_details: list[dict[str, object]] = []
        logging_sink_active = False
        jsonl_sink_type = JsonlSink if JsonlSink is not None else ()
        for sink in getattr(server, "sinks", ()):  # type: ignore[attr-defined]
            info: dict[str, object] = {
                "class": sink.__class__.__name__,
                "module": sink.__class__.__module__,
            }
            path = getattr(sink, "_path", None)
            if path:
                info["path"] = str(Path(path))
            if hasattr(sink, "_fsync"):
                info["fsync"] = bool(getattr(sink, "_fsync", False))
            sinks_info.append(info)
            if jsonl_sink_type and isinstance(sink, jsonl_sink_type):  # type: ignore[arg-type]
                jsonl_runtime_details.append(info)
            if sink.__class__.__name__ == "LoggingSink":
                logging_sink_active = True

        runtime_state = {
            "available": True,
            "reason": None,
            "address": getattr(server, "address", None),
            "history_size": getattr(server, "history_size", args.history_size),
            "sink_count": len(sinks_info),
            "sinks": sinks_info,
            "logging_sink_active": logging_sink_active,
            "jsonl_sink_active": bool(jsonl_runtime_details),
            "metadata_source": "runtime_metadata" if runtime_metadata else "args_only",
        }

        jsonl_runtime_path = runtime_metadata.get("jsonl_sink", {}).get("path")
        if jsonl_runtime_path:
            runtime_state["jsonl_sink"] = {
                "path": jsonl_runtime_path,
                "fsync": runtime_metadata.get("jsonl_sink", {}).get(
                    "fsync", bool(args.jsonl_fsync)
                ),
                "file": _file_reference_metadata(Path(jsonl_runtime_path), role="jsonl"),
            }
        elif jsonl_arg_path is not None:
            runtime_state["jsonl_sink"] = {
                "path": str(jsonl_arg_path),
                "fsync": bool(args.jsonl_fsync),
                "file": _file_reference_metadata(jsonl_arg_path, role="jsonl"),
            }
        else:
            runtime_state["jsonl_sink"] = {"path": None, "fsync": False}

        ui_runtime_meta = runtime_metadata.get("ui_alerts_sink", {})
        if ui_runtime_meta:
            runtime_state["ui_alerts_sink"] = {
                "path": ui_runtime_meta.get("path"),
                "active": bool(ui_runtime_meta.get("active")),
            }
            if ui_runtime_meta.get("config"):
                runtime_state["ui_alerts_sink"]["config"] = ui_runtime_meta.get("config")
            if ui_runtime_meta.get("path"):
                try:
                    runtime_state["ui_alerts_sink"]["file"] = _file_reference_metadata(
                        Path(ui_runtime_meta["path"]), role="ui_alerts_jsonl"
                    )
                except Exception:  # pragma: no cover - diagnostyka
                    runtime_state["ui_alerts_sink"]["file_error"] = "failed_to_stat"
        else:
            runtime_state["ui_alerts_sink"] = {
                "path": str(ui_alerts_arg_path) if ui_alerts_arg_path else None,
                "active": not args.disable_ui_alerts and bool(ui_alerts_arg_path),
            }
            runtime_state["ui_alerts_sink"]["config"] = _ui_alerts_config_from_args(args)
        tls_metadata = runtime_metadata.get("tls", {})
        tls_enabled_value = bool(
            tls_metadata.get(
                "configured",
                getattr(server, "tls_enabled", tls_configured),
            )
        )
        runtime_state["tls"] = {
            "configured": tls_enabled_value,
            "enabled": tls_enabled_value,
            "require_client_auth": bool(
                tls_metadata.get(
                    "require_client_auth",
                    getattr(
                        server,
                        "tls_client_auth_required",
                        args.tls_require_client_cert,
                    ),
                )
            ),
        }
        default_sink_names = {"LoggingSink", "JsonlSink"}
        runtime_state["additional_sinks"] = runtime_metadata.get(
            "additional_sink_descriptions",
            [
                {"class": info["class"], "module": info["module"]}
                for info in sinks_info
                if info["class"] not in default_sink_names
            ],
        )
        runtime_state["sink_descriptions"] = runtime_metadata.get(
            "sink_descriptions",
            [{"class": info["class"], "module": info["module"]} for info in sinks_info],
        )

    config_section = runtime_state["ui_alerts_sink"].setdefault("config", {})
    config_section["audit"] = dict(audit_section)
    if risk_profile_meta := getattr(args, "_ui_alerts_risk_profile_metadata", None):
        config_section["risk_profile"] = dict(risk_profile_meta)
    if risk_profiles_file_meta := getattr(
        args, "_ui_alerts_risk_profiles_file_metadata", None
    ):
        config_section["risk_profiles_file"] = dict(risk_profiles_file_meta)
    if ui_alerts_arg_path is not None and "file" not in runtime_state["ui_alerts_sink"]:
        runtime_state["ui_alerts_sink"]["file"] = _file_reference_metadata(
            ui_alerts_arg_path, role="ui_alerts_jsonl"
        )

    payload["runtime_state"] = runtime_state

    if core_config_section is not None:
        payload["core_config"] = core_config_section
    environment_entries: list[Mapping[str, object]] = []
    if environment_overrides:
        payload["environment_overrides"] = environment_overrides
        environment_entries = list(environment_overrides.get("entries", ()))
    if parameter_sources:
        payload["parameter_sources"] = dict(parameter_sources)

    fail_parameter_source = (
        (parameter_sources or {}).get("fail_on_security_warnings", "default")
        if parameter_sources
        else "default"
    )
    fail_env_entry: Mapping[str, object] | None = None
    for entry in environment_entries:
        if entry.get("option") == "fail_on_security_warnings":
            fail_env_entry = entry
            break

    env_variable = "RUN_METRICS_SERVICE_FAIL_ON_SECURITY_WARNINGS"
    if fail_parameter_source.startswith("env") and fail_env_entry is None:
        # środowisko wskazane, ale wpis nie został zarejestrowany (np. brak planu)
        fail_env_entry = {
            "variable": env_variable,
            "applied": False,
        }

    if fail_parameter_source == "cli":
        fail_source = "cli"
    elif fail_parameter_source.startswith("env"):
        fail_source = f"env:{env_variable}"
    else:
        fail_source = fail_parameter_source

    security_section = {
        "fail_on_security_warnings": _compact_mapping(
            {
                "enabled": bool(args.fail_on_security_warnings),
                "source": fail_source,
                "parameter_source": fail_parameter_source,
                "cli_flag": "--fail-on-security-warnings",
                "environment_variable": (
                    fail_env_entry.get("variable") if fail_env_entry else None
                ),
                "environment_raw_value": (
                    fail_env_entry.get("raw_value") if fail_env_entry else None
                ),
                "environment_applied": (
                    fail_env_entry.get("applied") if fail_env_entry is not None else None
                ),
                "environment_reason": (
                    fail_env_entry.get("reason") if fail_env_entry else None
                ),
                "environment_parsed_value": (
                    fail_env_entry.get("parsed_value") if fail_env_entry else None
                ),
                "environment_note": (
                    fail_env_entry.get("note") if fail_env_entry else None
                ),
            }
        ),
        "parameter_sources": {
            "fail_on_security_warnings": fail_parameter_source,
        },
    }

    payload["security"] = security_section

    tls_section: dict[str, object] = {
        "configured": tls_configured,
        "require_client_auth": bool(args.tls_require_client_cert),
    }
    if args.tls_cert:
        tls_section["certificate"] = _file_reference_metadata(args.tls_cert, role="tls_cert")
    if args.tls_key:
        tls_section["private_key"] = _file_reference_metadata(args.tls_key, role="tls_key")
    if args.tls_client_ca:
        tls_section["client_ca"] = _file_reference_metadata(
            args.tls_client_ca, role="tls_client_ca"
        )
    payload["tls"] = tls_section

    payload["notes"] = [
        "Serwer przeznaczony wyłącznie do telemetrii UI (reduce motion / overlay budget).",
        "Zanim przejdziesz do produkcji, upewnij się, że pipeline demo→paper→live zakończył się sukcesem.",
    ]

    return payload


def _append_config_plan_jsonl(path: Path, payload: Mapping[str, object]) -> Path:
    """Dopisuje konfigurację serwera do pliku JSONL."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")
    return path


def _install_signal_handlers(stop_callback) -> None:
    def handler(signum, _frame) -> None:  # pragma: no cover - reakcja na sygnał
        LOGGER.info("Otrzymano sygnał %s – zatrzymuję serwer telemetrii", signum)
        stop_callback()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except ValueError:  # pragma: no cover - np. uruchomienie w wątku
            LOGGER.warning("Nie udało się zarejestrować handlera sygnału %s", sig)


def _print_risk_profiles(
    selected: str | None,
    *,
    profiles_file_meta: Mapping[str, Any] | None = None,
    core_metadata: Mapping[str, Any] | None = None,
) -> None:
    profiles: dict[str, Mapping[str, Any]] = {}
    for name in list_risk_profile_names():
        profiles[name] = risk_profile_metadata(name)

    payload: dict[str, Any] = {"risk_profiles": profiles}
    if selected:
        normalized = selected.strip().lower()
        payload["selected"] = normalized
        payload["selected_profile"] = profiles.get(normalized)
    if profiles_file_meta:
        payload["risk_profiles_file"] = dict(profiles_file_meta)
    if core_metadata:
        payload["core_config"] = dict(core_metadata)

    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(raw_args)

    _configure_logging(args.log_level)

    provided_flags = _collect_provided_flags(raw_args)
    value_sources = _initial_value_sources(provided_flags)
    env_section_mapping, env_override_keys = _apply_environment_overrides(
        args,
        parser=parser,
        provided_flags=provided_flags,
        value_sources=value_sources,
    )
    environment_overrides_section = env_section_mapping or None
    core_config_section: Mapping[str, object] | None = None
    if args.core_config:
        try:
            core_config = load_core_config(args.core_config)
        except Exception as exc:  # noqa: BLE001 - szczegóły błędu przekazujemy operatorowi
            parser.error(f"Nie udało się wczytać --core-config: {exc}")
        else:
            metrics_config = getattr(core_config, "metrics_service", None)
            applied_sources: dict[str, str] = {}
            if metrics_config is not None:
                applied_sources = _apply_core_metrics_config(
                    args,
                    metrics_config,
                    provided_flags=provided_flags,
                    env_overrides=env_override_keys,
                    value_sources=value_sources,
                )
                tls_cfg = getattr(metrics_config, "tls", None)
                if tls_cfg is not None and bool(getattr(tls_cfg, "enabled", False)):
                    if not getattr(tls_cfg, "certificate_path", None) or not getattr(tls_cfg, "private_key_path", None):
                        parser.error(
                            "Sekcja runtime.metrics_service.tls wymaga pól certificate_path oraz private_key_path"
                        )
                if not bool(getattr(metrics_config, "enabled", True)):
                    LOGGER.warning(
                        "runtime.metrics_service.enabled = false w %s – kontynuuję uruchamianie zgodnie z flagami.",
                        args.core_config,
                    )
            else:
                LOGGER.warning(
                    "Plik %s nie zawiera sekcji runtime.metrics_service – używam wartości z flag CLI.",
                    args.core_config,
                )

            core_config_section = _build_core_config_section(
                core_config,
                metrics_config,
                applied_sources,
                requested_path=args.core_config,
            )

    _load_custom_risk_profiles(args, parser=parser)

    _apply_risk_profile_defaults(args, parser=parser, value_sources=value_sources)

    if getattr(args, "print_risk_profiles", False):
        _print_risk_profiles(
            args.ui_alerts_risk_profile,
            profiles_file_meta=getattr(args, "_ui_alerts_risk_profiles_file_metadata", None),
            core_metadata=core_config_section,
        )
        return 0

    value_sources = _finalize_value_sources(value_sources)

    if args.history_size <= 0:
        parser.error("--history-size musi być dodatnie")
    if bool(args.tls_cert) ^ bool(args.tls_key):
        parser.error("TLS wymaga jednoczesnego podania --tls-cert oraz --tls-key")
    for option in ("tls_cert", "tls_key", "tls_client_ca"):
        path = getattr(args, option)
        if path and not Path(path).exists():
            parser.error(f"Ścieżka {option.replace('_', '-')} nie istnieje: {path}")

    backend_choice = (args.ui_alerts_audit_backend or "auto").lower()
    if backend_choice not in {"auto", "file", "memory"}:
        parser.error("--ui-alerts-audit-backend wymaga wartości auto/file/memory")
    if backend_choice == "file" and not args.ui_alerts_audit_dir:
        parser.error("--ui-alerts-audit-backend file wymaga jednoczesnego podania --ui-alerts-audit-dir")

    plan_requested = bool(
        args.config_plan_jsonl or args.print_config_plan or args.fail_on_security_warnings
    )
    if not METRICS_RUNTIME_AVAILABLE:
        config_plan: Mapping[str, object] | None = None
        security_warnings_detected = False
        if plan_requested:
            config_plan = _build_config_plan_payload(
                args=args,
                server=None,
                core_config_section=core_config_section,
                runtime_unavailable_reason="metrics_runtime_unavailable",
                runtime_unavailable_details=METRICS_RUNTIME_UNAVAILABLE_MESSAGE,
                environment_overrides=environment_overrides_section,
                parameter_sources=value_sources,
            )
            if args.fail_on_security_warnings:
                security_warnings_detected = _log_security_warnings(
                    config_plan,
                    fail_on_warnings=True,
                    logger=LOGGER,
                    context="run_metrics_service.config_plan",
                )
            if args.config_plan_jsonl:
                destination = Path(args.config_plan_jsonl).expanduser()
                _append_config_plan_jsonl(destination, config_plan)
                LOGGER.warning(
                    "Zapisano audyt konfiguracji MetricsService do %s (serwer nie został uruchomiony: brak wsparcia gRPC)",
                    destination,
                )
            if args.print_config_plan:
                json.dump(config_plan, sys.stdout, ensure_ascii=False, indent=2)
                sys.stdout.write("\n")
                if args.fail_on_security_warnings and security_warnings_detected:
                    return 3
                return 0
            if args.fail_on_security_warnings and security_warnings_detected:
                return 3

        message = METRICS_RUNTIME_UNAVAILABLE_MESSAGE
        if config_plan is not None and args.config_plan_jsonl and not args.print_config_plan:
            LOGGER.error(message)
            return 2
        parser.error(message)

    ui_alerts_options = {
        "reduce_mode": args.ui_alerts_reduce_mode,
        "overlay_mode": args.ui_alerts_overlay_mode,
        "reduce_category": args.ui_alerts_reduce_category,
        "reduce_active_severity": args.ui_alerts_reduce_active_severity,
        "reduce_recovered_severity": args.ui_alerts_reduce_recovered_severity,
        "overlay_category": args.ui_alerts_overlay_category,
        "overlay_exceeded_severity": args.ui_alerts_overlay_exceeded_severity,
        "overlay_recovered_severity": args.ui_alerts_overlay_recovered_severity,
        "overlay_critical_severity": args.ui_alerts_overlay_critical_severity,
        "overlay_critical_threshold": args.ui_alerts_overlay_critical_threshold,
        "jank_mode": args.ui_alerts_jank_mode,
        "jank_category": args.ui_alerts_jank_category,
        "jank_spike_severity": args.ui_alerts_jank_spike_severity,
        "jank_critical_severity": args.ui_alerts_jank_critical_severity,
        "jank_critical_over_ms": args.ui_alerts_jank_critical_over_ms,
    }

    ui_alerts_config_payload: dict[str, object] | None = None
    if not args.disable_ui_alerts:
        reduce_mode = (args.ui_alerts_reduce_mode or "enable").lower()
        overlay_mode = (args.ui_alerts_overlay_mode or "enable").lower()
        jank_mode = (args.ui_alerts_jank_mode or "enable").lower()
        reduce_dispatch = reduce_mode == "enable"
        overlay_dispatch = overlay_mode == "enable"
        jank_dispatch = jank_mode == "enable"
        reduce_logging = reduce_mode in {"enable", "jsonl"}
        overlay_logging = overlay_mode in {"enable", "jsonl"}
        jank_logging = jank_mode in {"enable", "jsonl"}
        reduce_category = args.ui_alerts_reduce_category or _DEFAULT_UI_CATEGORY
        reduce_active = args.ui_alerts_reduce_active_severity or _DEFAULT_UI_SEVERITY_ACTIVE
        reduce_recovered = (
            args.ui_alerts_reduce_recovered_severity or _DEFAULT_UI_SEVERITY_RECOVERED
        )
        overlay_category = args.ui_alerts_overlay_category or _DEFAULT_UI_CATEGORY
        overlay_exceeded = (
            args.ui_alerts_overlay_exceeded_severity or _DEFAULT_UI_SEVERITY_ACTIVE
        )
        overlay_recovered = (
            args.ui_alerts_overlay_recovered_severity or _DEFAULT_UI_SEVERITY_RECOVERED
        )
        overlay_critical = (
            args.ui_alerts_overlay_critical_severity or _DEFAULT_OVERLAY_SEVERITY_CRITICAL
        )
        overlay_threshold = (
            args.ui_alerts_overlay_critical_threshold
            if args.ui_alerts_overlay_critical_threshold is not None
            else _DEFAULT_OVERLAY_THRESHOLD
        )
        jank_category = args.ui_alerts_jank_category or _DEFAULT_UI_CATEGORY
        jank_spike = args.ui_alerts_jank_spike_severity or _DEFAULT_JANK_SEVERITY_SPIKE
        jank_critical = args.ui_alerts_jank_critical_severity or _DEFAULT_JANK_SEVERITY_CRITICAL
        jank_threshold = (
            args.ui_alerts_jank_critical_over_ms
            if args.ui_alerts_jank_critical_over_ms is not None
            else _DEFAULT_JANK_CRITICAL_THRESHOLD_MS
        )
        ui_path = (
            Path(args.ui_alerts_jsonl).expanduser()
            if args.ui_alerts_jsonl
            else DEFAULT_UI_ALERTS_JSONL_PATH.expanduser()
        )
        audit_directory = (
            Path(args.ui_alerts_audit_dir).expanduser() if args.ui_alerts_audit_dir else None
        )
        audit_pattern = args.ui_alerts_audit_pattern or _DEFAULT_UI_ALERT_AUDIT_PATTERN
        audit_retention = (
            args.ui_alerts_audit_retention_days
            if args.ui_alerts_audit_retention_days is not None
            else _DEFAULT_UI_ALERT_AUDIT_RETENTION_DAYS
        )
        requested_backend = (args.ui_alerts_audit_backend or "auto").lower()
        file_backend_supported = audit_directory is not None and FileAlertAuditLog is not None
        audit_config: dict[str, object] = {
            "requested": requested_backend,
            "fsync": bool(args.ui_alerts_audit_fsync),
        }
        if requested_backend == "file":
            if audit_directory is None:
                audit_config["backend"] = None
                audit_config["note"] = "file_backend_requires_directory"
            elif not file_backend_supported:
                audit_config["backend"] = None
                audit_config["note"] = "file_backend_unavailable"
            else:
                audit_config.update(
                    {
                        "backend": "file",
                        "directory": str(audit_directory),
                        "pattern": audit_pattern,
                        "retention_days": audit_retention,
                    }
                )
        elif requested_backend == "memory":
            audit_config["backend"] = "memory"
            if audit_directory is not None:
                audit_config["note"] = "directory_ignored_memory_backend"
        else:  # auto
            if file_backend_supported:
                audit_config.update(
                    {
                        "backend": "file",
                        "directory": str(audit_directory),
                        "pattern": audit_pattern,
                        "retention_days": audit_retention,
                    }
                )
            else:
                audit_config["backend"] = "memory"
                if audit_directory is not None:
                    audit_config["note"] = "file_backend_unavailable"
        ui_alerts_config_payload = {
            "path": str(ui_path),
            "reduce_mode": reduce_mode,
            "overlay_mode": overlay_mode,
            "jank_mode": jank_mode,
            "reduce_motion_alerts": reduce_dispatch,
            "overlay_alerts": overlay_dispatch,
            "jank_alerts": jank_dispatch,
            "reduce_motion_logging": reduce_logging,
            "overlay_logging": overlay_logging,
            "jank_logging": jank_logging,
            "reduce_motion_category": reduce_category,
            "reduce_motion_severity_active": reduce_active,
            "reduce_motion_severity_recovered": reduce_recovered,
            "overlay_category": overlay_category,
            "overlay_severity_exceeded": overlay_exceeded,
            "overlay_severity_recovered": overlay_recovered,
            "overlay_severity_critical": overlay_critical,
            "overlay_critical_threshold": overlay_threshold,
            "jank_category": jank_category,
            "jank_severity_spike": jank_spike,
            "jank_severity_critical": jank_critical,
            "jank_critical_over_ms": jank_threshold,
            "audit": audit_config,
        }

    server = _build_server(
        host=args.host,
        port=args.port,
        history_size=args.history_size,
        enable_logging_sink=not args.no_log_sink,
        jsonl_path=Path(args.jsonl) if args.jsonl else None,
        jsonl_fsync=args.jsonl_fsync,
        auth_token=args.auth_token,
        enable_ui_alerts=not args.disable_ui_alerts,
        ui_alerts_jsonl_path=Path(args.ui_alerts_jsonl) if args.ui_alerts_jsonl else None,
        ui_alerts_options=ui_alerts_options,
        ui_alerts_config=ui_alerts_config_payload,
        ui_alerts_audit_dir=audit_directory,
        ui_alerts_audit_backend=args.ui_alerts_audit_backend,
        ui_alerts_audit_pattern=args.ui_alerts_audit_pattern,
        ui_alerts_audit_retention_days=args.ui_alerts_audit_retention_days,
        ui_alerts_audit_fsync=args.ui_alerts_audit_fsync,
        tls_config={
            "certificate_path": Path(args.tls_cert) if args.tls_cert else None,
            "private_key_path": Path(args.tls_key) if args.tls_key else None,
            "client_ca_path": Path(args.tls_client_ca) if args.tls_client_ca else None,
            "require_client_auth": bool(args.tls_require_client_cert),
        }
        if args.tls_cert and args.tls_key
        else None,
    )

    should_stop = False

    def request_stop() -> None:
        nonlocal should_stop
        if should_stop:
            return
        should_stop = True
        server.stop(grace=1.0)

    _install_signal_handlers(request_stop)

    server.start()
    LOGGER.info("MetricsService uruchomiony na %s", server.address)
    if args.print_address:
        print(server.address)
    if args.auth_token:
        LOGGER.info(
            "Serwer wymaga nagłówka Authorization: Bearer <token> od klientów telemetrycznych"
        )

    config_plan: Mapping[str, object] | None = None
    security_warnings_detected = False
    if plan_requested:
        config_plan = _build_config_plan_payload(
            args=args,
            server=server,
            core_config_section=core_config_section,
            environment_overrides=environment_overrides_section,
            parameter_sources=value_sources,
        )
        if args.fail_on_security_warnings:
            security_warnings_detected = _log_security_warnings(
                config_plan,
                fail_on_warnings=True,
                logger=LOGGER,
                context="run_metrics_service.config_plan",
            )
        if args.config_plan_jsonl:
            destination = Path(args.config_plan_jsonl).expanduser()
            _append_config_plan_jsonl(destination, config_plan)
            LOGGER.info("Zapisano audyt konfiguracji MetricsService do %s", destination)
        if args.print_config_plan:
            json.dump(config_plan, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
            request_stop()
            server.stop(grace=0)
            if args.fail_on_security_warnings and security_warnings_detected:
                return 3
            return 0
        if args.fail_on_security_warnings and security_warnings_detected:
            request_stop()
            server.stop(grace=0)
            return 3

    try:
        if args.shutdown_after is not None:
            LOGGER.info(
                "Serwer zakończy pracę automatycznie po %.2f s (lub szybciej po sygnale)",
                args.shutdown_after,
            )
            terminated = server.wait_for_termination(timeout=args.shutdown_after)
            if not terminated and not should_stop:
                LOGGER.info("Limit czasu minął – zatrzymuję serwer telemetrii")
        else:
            LOGGER.info("Naciśnij Ctrl+C, aby zakończyć pracę serwera.")
            server.wait_for_termination()
    except KeyboardInterrupt:  # pragma: no cover - zależy od środowiska
        LOGGER.info("Przerwano przez użytkownika – zatrzymuję serwer.")
    finally:
        if not should_stop:
            server.stop(grace=1.0)
        LOGGER.info("Serwer MetricsService został zatrzymany.")
    return 0


if __name__ == "__main__":  # pragma: no cover - ścieżka uruchomienia skryptu
    sys.exit(main())
