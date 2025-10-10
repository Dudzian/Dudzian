"""Uruchamia lokalny stub tradingowy gRPC dla powłoki Qt/QML."""

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

from bot_core.testing import (
    InMemoryTradingDataset,
    TradingStubServer,
    build_default_dataset,
    load_dataset_from_yaml,
    merge_datasets,
)

from bot_core.runtime.file_metadata import (
    file_reference_metadata,
    log_security_warnings as _log_security_warnings,
)

# MetricsService (opcjonalny — gdy brak gRPC/stubów, po prostu nie uruchamiamy)
try:  # pragma: no cover - zależności opcjonalne
    from bot_core.runtime import JsonlSink, create_metrics_server
except Exception:  # pragma: no cover - brak generowanych stubów lub grpcio
    JsonlSink = None  # type: ignore
    create_metrics_server = None  # type: ignore

# Sink alertów UI (opcjonalny)
try:  # pragma: no cover
    from bot_core.runtime.metrics_alerts import (  # type: ignore
        DEFAULT_UI_ALERTS_JSONL_PATH,
        UiTelemetryAlertSink,
    )
except Exception:  # pragma: no cover
    UiTelemetryAlertSink = None  # type: ignore
    DEFAULT_UI_ALERTS_JSONL_PATH = Path("logs/ui_telemetry_alerts.jsonl")

# Router alertów (opcjonalny – różne gałęzie)
DefaultAlertRouter = None
InMemoryAlertAuditLog = None
FileAlertAuditLog = None
try:  # najczęściej tak:
    from bot_core.alerts import (  # type: ignore
        DefaultAlertRouter as _Router,
        FileAlertAuditLog as _FileAudit,
        InMemoryAlertAuditLog as _Audit,
    )
    DefaultAlertRouter = _Router
    InMemoryAlertAuditLog = _Audit
    FileAlertAuditLog = _FileAudit
except Exception:
    try:  # czasem w oddzielnym module
        from bot_core.alerts.audit import (  # type: ignore
            FileAlertAuditLog as _FileAudit2,
            InMemoryAlertAuditLog as _Audit2,
        )
        from bot_core.alerts import DefaultAlertRouter as _Router2  # type: ignore
        DefaultAlertRouter = _Router2
        InMemoryAlertAuditLog = _Audit2
        FileAlertAuditLog = _FileAudit2
    except Exception:
        pass

LOGGER = logging.getLogger("run_trading_stub_server")

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


def _collect_provided_flags(raw_args: Iterable[str]) -> set[str]:
    """Zwraca zbiór flag przekazanych operatorowi (np. do rozstrzygania env)."""
    provided: set[str] = set()
    for token in raw_args:
        if not token.startswith("--"):
            continue
        provided.add(token.split("=", 1)[0])
    return provided


def _parse_env_bool(value: str, *, variable: str, parser: argparse.ArgumentParser) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    parser.error(
        f"Nieprawidłowa wartość '{value}' w zmiennej {variable} – użyj true/false, 1/0, yes/no."
    )
    raise AssertionError("parser.error powinno zakończyć działanie")


def _parse_env_int(value: str, *, variable: str, parser: argparse.ArgumentParser) -> int:
    try:
        return int(value)
    except ValueError:
        parser.error(f"Zmienna {variable} musi być liczbą całkowitą – otrzymano '{value}'.")
    raise AssertionError("parser.error powinno zakończyć działanie")


def _parse_env_float(value: str, *, variable: str, parser: argparse.ArgumentParser) -> float:
    try:
        return float(value)
    except ValueError:
        parser.error(f"Zmienna {variable} musi być liczbą zmiennoprzecinkową – otrzymano '{value}'.")
    raise AssertionError("parser.error powinno zakończyć działanie")


def _initial_value_sources(provided_flags: Iterable[str]) -> dict[str, str]:
    provided = set(provided_flags)
    mapping = {
        "--host": "host",
        "--port": "port",
        "--dataset": "dataset_paths",
        "--no-default-dataset": "include_default_dataset",
        "--shutdown-after": "shutdown_after",
        "--max-workers": "max_workers",
        "--stream-repeat": "stream_repeat",
        "--stream-interval": "stream_interval",
        "--log-level": "log_level",
        "--print-address": "print_address",
        "--enable-metrics": "enable_metrics",
        "--metrics-host": "metrics_host",
        "--metrics-port": "metrics_port",
        "--metrics-history-size": "metrics_history_size",
        "--metrics-jsonl": "metrics_jsonl_path",
        "--metrics-jsonl-fsync": "metrics_jsonl_fsync",
        "--metrics-disable-log-sink": "metrics_disable_log_sink",
        "--metrics-tls-cert": "metrics_tls_cert",
        "--metrics-tls-key": "metrics_tls_key",
        "--metrics-tls-client-ca": "metrics_tls_client_ca",
        "--metrics-tls-require-client-cert": "metrics_tls_require_client_cert",
        "--metrics-ui-alerts-jsonl": "metrics_ui_alerts_jsonl_path",
        "--disable-metrics-ui-alerts": "disable_metrics_ui_alerts",
        "--metrics-ui-alerts-reduce-mode": "metrics_ui_alerts_reduce_mode",
        "--metrics-ui-alerts-overlay-mode": "metrics_ui_alerts_overlay_mode",
        "--metrics-ui-alerts-reduce-category": "metrics_ui_alerts_reduce_category",
        "--metrics-ui-alerts-reduce-active-severity": "metrics_ui_alerts_reduce_active_severity",
        "--metrics-ui-alerts-reduce-recovered-severity": "metrics_ui_alerts_reduce_recovered_severity",
        "--metrics-ui-alerts-overlay-category": "metrics_ui_alerts_overlay_category",
        "--metrics-ui-alerts-overlay-exceeded-severity": "metrics_ui_alerts_overlay_exceeded_severity",
        "--metrics-ui-alerts-overlay-recovered-severity": "metrics_ui_alerts_overlay_recovered_severity",
        "--metrics-ui-alerts-overlay-critical-severity": "metrics_ui_alerts_overlay_critical_severity",
        "--metrics-ui-alerts-overlay-critical-threshold": "metrics_ui_alerts_overlay_critical_threshold",
        "--metrics-ui-alerts-jank-mode": "metrics_ui_alerts_jank_mode",
        "--metrics-ui-alerts-jank-category": "metrics_ui_alerts_jank_category",
        "--metrics-ui-alerts-jank-spike-severity": "metrics_ui_alerts_jank_spike_severity",
        "--metrics-ui-alerts-jank-critical-severity": "metrics_ui_alerts_jank_critical_severity",
        "--metrics-ui-alerts-jank-critical-over-ms": "metrics_ui_alerts_jank_critical_over_ms",
        "--metrics-ui-alerts-audit-dir": "metrics_ui_alerts_audit_dir",
        "--metrics-ui-alerts-audit-backend": "metrics_ui_alerts_audit_backend",
        "--metrics-ui-alerts-audit-pattern": "metrics_ui_alerts_audit_pattern",
        "--metrics-ui-alerts-audit-retention-days": "metrics_ui_alerts_audit_retention_days",
        "--metrics-ui-alerts-audit-fsync": "metrics_ui_alerts_audit_fsync",
        "--metrics-print-address": "metrics_print_address",
        "--metrics-auth-token": "metrics_auth_token",
        "--print-runtime-plan": "print_runtime_plan",
        "--runtime-plan-jsonl": "runtime_plan_jsonl_path",
        "--fail-on-security-warnings": "fail_on_security_warnings",
    }
    return {val: "cli" for key, val in mapping.items() if key in provided}


def _finalize_value_sources(sources: Mapping[str, str]) -> dict[str, str]:
    keys = {
        "host",
        "port",
        "dataset_paths",
        "include_default_dataset",
        "shutdown_after",
        "max_workers",
        "stream_repeat",
        "stream_interval",
        "log_level",
        "print_address",
        "enable_metrics",
        "metrics_host",
        "metrics_port",
        "metrics_history_size",
        "metrics_jsonl_path",
        "metrics_jsonl_fsync",
        "metrics_disable_log_sink",
        "metrics_tls_cert",
        "metrics_tls_key",
        "metrics_tls_client_ca",
        "metrics_tls_require_client_cert",
        "metrics_ui_alerts_jsonl_path",
        "disable_metrics_ui_alerts",
        "metrics_print_address",
        "metrics_ui_alerts_reduce_mode",
        "metrics_ui_alerts_overlay_mode",
        "metrics_ui_alerts_reduce_category",
        "metrics_ui_alerts_reduce_active_severity",
        "metrics_ui_alerts_reduce_recovered_severity",
        "metrics_ui_alerts_overlay_category",
        "metrics_ui_alerts_overlay_exceeded_severity",
        "metrics_ui_alerts_overlay_recovered_severity",
        "metrics_ui_alerts_overlay_critical_severity",
        "metrics_ui_alerts_overlay_critical_threshold",
        "metrics_ui_alerts_jank_mode",
        "metrics_ui_alerts_jank_category",
        "metrics_ui_alerts_jank_spike_severity",
        "metrics_ui_alerts_jank_critical_severity",
        "metrics_ui_alerts_jank_critical_over_ms",
        "metrics_ui_alerts_audit_dir",
        "metrics_ui_alerts_audit_backend",
        "metrics_ui_alerts_audit_pattern",
        "metrics_ui_alerts_audit_retention_days",
        "metrics_ui_alerts_audit_fsync",
        "metrics_auth_token",
        "print_runtime_plan",
        "runtime_plan_jsonl_path",
        "fail_on_security_warnings",
    }
    finalized = dict(sources)
    for key in keys:
        finalized.setdefault(key, "default")
    return finalized


def _apply_environment_overrides(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
    provided_flags: set[str],
    value_sources: dict[str, str],
) -> tuple[list[Mapping[str, object]], set[str]]:
    """Nakłada ustawienia ze zmiennych środowiskowych na argumenty CLI."""
    entries: list[Mapping[str, object]] = []
    override_keys: set[str] = set()

    def record_entry(entry: dict[str, object]) -> None:
        entries.append(entry)

    def skip_due_to_cli(option: str, env_var: str, raw_value: str) -> None:
        record_entry(
            {"option": option, "variable": env_var, "raw_value": raw_value, "applied": False, "reason": "cli_override"}
        )

    def normalize_value(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [str(v) if isinstance(v, Path) else v for v in value]
        return value

    def apply_value(option: str, env_var: str, raw_value: str, parsed_value: Any) -> None:
        override_keys.add(option)
        value_sources[option] = "env"
        record_entry(
            {
                "option": option,
                "variable": env_var,
                "raw_value": raw_value,
                "applied": True,
                "parsed_value": normalize_value(parsed_value),
            }
        )

    def env_present(flag: str) -> bool:
        return flag in provided_flags

    # proste helpery
    def parse_bool(var: str) -> bool:
        return _parse_env_bool(os.environ[var], variable=var, parser=parser)

    def parse_int(var: str) -> int:
        return _parse_env_int(os.environ[var], variable=var, parser=parser)

    def parse_float(var: str) -> float:
        return _parse_env_float(os.environ[var], variable=var, parser=parser)

    # mapowania ENV
    if (raw := os.getenv("RUN_TRADING_STUB_HOST")) is not None:
        if env_present("--host"):
            skip_due_to_cli("host", "RUN_TRADING_STUB_HOST", raw)
        else:
            args.host = raw
            apply_value("host", "RUN_TRADING_STUB_HOST", raw, raw)

    if (raw := os.getenv("RUN_TRADING_STUB_PORT")) is not None:
        if env_present("--port"):
            skip_due_to_cli("port", "RUN_TRADING_STUB_PORT", raw)
        else:
            args.port = parse_int("RUN_TRADING_STUB_PORT")
            apply_value("port", "RUN_TRADING_STUB_PORT", raw, args.port)

    if (raw := os.getenv("RUN_TRADING_STUB_DATASETS")) is not None:
        if env_present("--dataset"):
            skip_due_to_cli("dataset_paths", "RUN_TRADING_STUB_DATASETS", raw)
        else:
            parts = [p.strip() for p in raw.split(os.pathsep) if p.strip()]
            paths = [Path(p).expanduser() for p in parts]
            args.dataset = paths
            apply_value("dataset_paths", "RUN_TRADING_STUB_DATASETS", raw, [str(p) for p in paths])

    if (raw := os.getenv("RUN_TRADING_STUB_NO_DEFAULT_DATASET")) is not None:
        if env_present("--no-default-dataset"):
            skip_due_to_cli("include_default_dataset", "RUN_TRADING_STUB_NO_DEFAULT_DATASET", raw)
        else:
            disabled = parse_bool("RUN_TRADING_STUB_NO_DEFAULT_DATASET")
            args.no_default_dataset = disabled
            apply_value("include_default_dataset", "RUN_TRADING_STUB_NO_DEFAULT_DATASET", raw, not disabled)

    if (raw := os.getenv("RUN_TRADING_STUB_SHUTDOWN_AFTER")) is not None:
        if env_present("--shutdown-after"):
            skip_due_to_cli("shutdown_after", "RUN_TRADING_STUB_SHUTDOWN_AFTER", raw)
        else:
            args.shutdown_after = parse_float("RUN_TRADING_STUB_SHUTDOWN_AFTER")
            apply_value("shutdown_after", "RUN_TRADING_STUB_SHUTDOWN_AFTER", raw, args.shutdown_after)

    if (raw := os.getenv("RUN_TRADING_STUB_MAX_WORKERS")) is not None:
        if env_present("--max-workers"):
            skip_due_to_cli("max_workers", "RUN_TRADING_STUB_MAX_WORKERS", raw)
        else:
            args.max_workers = parse_int("RUN_TRADING_STUB_MAX_WORKERS")
            apply_value("max_workers", "RUN_TRADING_STUB_MAX_WORKERS", raw, args.max_workers)

    if (raw := os.getenv("RUN_TRADING_STUB_STREAM_REPEAT")) is not None:
        if env_present("--stream-repeat"):
            skip_due_to_cli("stream_repeat", "RUN_TRADING_STUB_STREAM_REPEAT", raw)
        else:
            args.stream_repeat = parse_bool("RUN_TRADING_STUB_STREAM_REPEAT")
            apply_value("stream_repeat", "RUN_TRADING_STUB_STREAM_REPEAT", raw, args.stream_repeat)

    if (raw := os.getenv("RUN_TRADING_STUB_STREAM_INTERVAL")) is not None:
        if env_present("--stream-interval"):
            skip_due_to_cli("stream_interval", "RUN_TRADING_STUB_STREAM_INTERVAL", raw)
        else:
            args.stream_interval = parse_float("RUN_TRADING_STUB_STREAM_INTERVAL")
            apply_value("stream_interval", "RUN_TRADING_STUB_STREAM_INTERVAL", raw, args.stream_interval)

    if (raw := os.getenv("RUN_TRADING_STUB_LOG_LEVEL")) is not None:
        if env_present("--log-level"):
            skip_due_to_cli("log_level", "RUN_TRADING_STUB_LOG_LEVEL", raw)
        else:
            args.log_level = raw
            apply_value("log_level", "RUN_TRADING_STUB_LOG_LEVEL", raw, raw)

    if (raw := os.getenv("RUN_TRADING_STUB_PRINT_ADDRESS")) is not None:
        if env_present("--print-address"):
            skip_due_to_cli("print_address", "RUN_TRADING_STUB_PRINT_ADDRESS", raw)
        else:
            args.print_address = parse_bool("RUN_TRADING_STUB_PRINT_ADDRESS")
            apply_value("print_address", "RUN_TRADING_STUB_PRINT_ADDRESS", raw, args.print_address)

    # Metrics ENV
    if (raw := os.getenv("RUN_TRADING_STUB_ENABLE_METRICS")) is not None:
        if env_present("--enable-metrics"):
            skip_due_to_cli("enable_metrics", "RUN_TRADING_STUB_ENABLE_METRICS", raw)
        else:
            args.enable_metrics = parse_bool("RUN_TRADING_STUB_ENABLE_METRICS")
            apply_value("enable_metrics", "RUN_TRADING_STUB_ENABLE_METRICS", raw, args.enable_metrics)

    if (raw := os.getenv("RUN_TRADING_STUB_METRICS_HOST")) is not None:
        if env_present("--metrics-host"):
            skip_due_to_cli("metrics_host", "RUN_TRADING_STUB_METRICS_HOST", raw)
        else:
            args.metrics_host = raw
            apply_value("metrics_host", "RUN_TRADING_STUB_METRICS_HOST", raw, raw)

    if (raw := os.getenv("RUN_TRADING_STUB_METRICS_PORT")) is not None:
        if env_present("--metrics-port"):
            skip_due_to_cli("metrics_port", "RUN_TRADING_STUB_METRICS_PORT", raw)
        else:
            args.metrics_port = parse_int("RUN_TRADING_STUB_METRICS_PORT")
            apply_value("metrics_port", "RUN_TRADING_STUB_METRICS_PORT", raw, args.metrics_port)

    if (raw := os.getenv("RUN_TRADING_STUB_METRICS_HISTORY_SIZE")) is not None:
        if env_present("--metrics-history-size"):
            skip_due_to_cli("metrics_history_size", "RUN_TRADING_STUB_METRICS_HISTORY_SIZE", raw)
        else:
            args.metrics_history_size = parse_int("RUN_TRADING_STUB_METRICS_HISTORY_SIZE")
            apply_value("metrics_history_size", "RUN_TRADING_STUB_METRICS_HISTORY_SIZE", raw, args.metrics_history_size)

    if (raw := os.getenv("RUN_TRADING_STUB_METRICS_JSONL")) is not None:
        if env_present("--metrics-jsonl"):
            skip_due_to_cli("metrics_jsonl_path", "RUN_TRADING_STUB_METRICS_JSONL", raw)
        else:
            args.metrics_jsonl = Path(raw).expanduser()
            apply_value("metrics_jsonl_path", "RUN_TRADING_STUB_METRICS_JSONL", raw, str(args.metrics_jsonl))

    if (raw := os.getenv("RUN_TRADING_STUB_METRICS_JSONL_FSYNC")) is not None:
        if env_present("--metrics-jsonl-fsync"):
            skip_due_to_cli("metrics_jsonl_fsync", "RUN_TRADING_STUB_METRICS_JSONL_FSYNC", raw)
        else:
            args.metrics_jsonl_fsync = parse_bool("RUN_TRADING_STUB_METRICS_JSONL_FSYNC")
            apply_value("metrics_jsonl_fsync", "RUN_TRADING_STUB_METRICS_JSONL_FSYNC", raw, args.metrics_jsonl_fsync)

    if (raw := os.getenv("RUN_TRADING_STUB_METRICS_DISABLE_LOG_SINK")) is not None:
        if env_present("--metrics-disable-log-sink"):
            skip_due_to_cli("metrics_disable_log_sink", "RUN_TRADING_STUB_METRICS_DISABLE_LOG_SINK", raw)
        else:
            args.metrics_disable_log_sink = parse_bool("RUN_TRADING_STUB_METRICS_DISABLE_LOG_SINK")
            apply_value(
                "metrics_disable_log_sink",
                "RUN_TRADING_STUB_METRICS_DISABLE_LOG_SINK",
                raw,
                args.metrics_disable_log_sink,
            )

    if (raw := os.getenv("RUN_TRADING_STUB_METRICS_TLS_CERT")) is not None:
        if env_present("--metrics-tls-cert"):
            skip_due_to_cli("metrics_tls_cert", "RUN_TRADING_STUB_METRICS_TLS_CERT", raw)
        else:
            args.metrics_tls_cert = Path(raw).expanduser()
            apply_value("metrics_tls_cert", "RUN_TRADING_STUB_METRICS_TLS_CERT", raw, str(args.metrics_tls_cert))

    if (raw := os.getenv("RUN_TRADING_STUB_METRICS_TLS_KEY")) is not None:
        if env_present("--metrics-tls-key"):
            skip_due_to_cli("metrics_tls_key", "RUN_TRADING_STUB_METRICS_TLS_KEY", raw)
        else:
            args.metrics_tls_key = Path(raw).expanduser()
            apply_value("metrics_tls_key", "RUN_TRADING_STUB_METRICS_TLS_KEY", raw, str(args.metrics_tls_key))

    if (raw := os.getenv("RUN_TRADING_STUB_METRICS_TLS_CLIENT_CA")) is not None:
        if env_present("--metrics-tls-client-ca"):
            skip_due_to_cli("metrics_tls_client_ca", "RUN_TRADING_STUB_METRICS_TLS_CLIENT_CA", raw)
        else:
            args.metrics_tls_client_ca = Path(raw).expanduser()
            apply_value(
                "metrics_tls_client_ca",
                "RUN_TRADING_STUB_METRICS_TLS_CLIENT_CA",
                raw,
                str(args.metrics_tls_client_ca),
            )

    if (raw := os.getenv("RUN_TRADING_STUB_METRICS_TLS_REQUIRE_CLIENT_CERT")) is not None:
        if env_present("--metrics-tls-require-client-cert"):
            skip_due_to_cli(
                "metrics_tls_require_client_cert",
                "RUN_TRADING_STUB_METRICS_TLS_REQUIRE_CLIENT_CERT",
                raw,
            )
        else:
            args.metrics_tls_require_client_cert = parse_bool(
                "RUN_TRADING_STUB_METRICS_TLS_REQUIRE_CLIENT_CERT"
            )
            apply_value(
                "metrics_tls_require_client_cert",
                "RUN_TRADING_STUB_METRICS_TLS_REQUIRE_CLIENT_CERT",
                raw,
                args.metrics_tls_require_client_cert,
            )

    if (raw := os.getenv("RUN_TRADING_STUB_METRICS_UI_ALERTS_JSONL")) is not None:
        if env_present("--metrics-ui-alerts-jsonl"):
            skip_due_to_cli("metrics_ui_alerts_jsonl_path", "RUN_TRADING_STUB_METRICS_UI_ALERTS_JSONL", raw)
        else:
            args.metrics_ui_alerts_jsonl = Path(raw).expanduser()
            apply_value(
                "metrics_ui_alerts_jsonl_path",
                "RUN_TRADING_STUB_METRICS_UI_ALERTS_JSONL",
                raw,
                str(args.metrics_ui_alerts_jsonl),
            )

    if (raw := os.getenv("RUN_TRADING_STUB_DISABLE_METRICS_UI_ALERTS")) is not None:
        if env_present("--disable-metrics-ui-alerts"):
            skip_due_to_cli("disable_metrics_ui_alerts", "RUN_TRADING_STUB_DISABLE_METRICS_UI_ALERTS", raw)
        else:
            args.disable_metrics_ui_alerts = parse_bool("RUN_TRADING_STUB_DISABLE_METRICS_UI_ALERTS")
            apply_value(
                "disable_metrics_ui_alerts",
                "RUN_TRADING_STUB_DISABLE_METRICS_UI_ALERTS",
                raw,
                args.disable_metrics_ui_alerts,
            )

    if (raw := os.getenv("RUN_TRADING_STUB_METRICS_UI_ALERTS_REDUCE_MODE")) is not None:
        if env_present("--metrics-ui-alerts-reduce-mode"):
            skip_due_to_cli("metrics_ui_alerts_reduce_mode", "RUN_TRADING_STUB_METRICS_UI_ALERTS_REDUCE_MODE", raw)
        else:
            normalized = raw.strip().lower()
            if normalized not in UI_ALERT_MODE_CHOICES:
                raise SystemExit(
                    f"RUN_TRADING_STUB_METRICS_UI_ALERTS_REDUCE_MODE musi być jedną z wartości: {', '.join(UI_ALERT_MODE_CHOICES)}"
                )
            args.metrics_ui_alerts_reduce_mode = normalized
            apply_value(
                "metrics_ui_alerts_reduce_mode",
                "RUN_TRADING_STUB_METRICS_UI_ALERTS_REDUCE_MODE",
                raw,
                normalized,
            )

    if (raw := os.getenv("RUN_TRADING_STUB_METRICS_UI_ALERTS_OVERLAY_MODE")) is not None:
        if env_present("--metrics-ui-alerts-overlay-mode"):
            skip_due_to_cli("metrics_ui_alerts_overlay_mode", "RUN_TRADING_STUB_METRICS_UI_ALERTS_OVERLAY_MODE", raw)
        else:
            normalized = raw.strip().lower()
            if normalized not in UI_ALERT_MODE_CHOICES:
                raise SystemExit(
                    f"RUN_TRADING_STUB_METRICS_UI_ALERTS_OVERLAY_MODE musi być jedną z wartości: {', '.join(UI_ALERT_MODE_CHOICES)}"
                )
            args.metrics_ui_alerts_overlay_mode = normalized
            apply_value(
                "metrics_ui_alerts_overlay_mode",
                "RUN_TRADING_STUB_METRICS_UI_ALERTS_OVERLAY_MODE",
                raw,
                normalized,
            )

    if (raw := os.getenv("RUN_TRADING_STUB_METRICS_UI_ALERTS_JANK_MODE")) is not None:
        if env_present("--metrics-ui-alerts-jank-mode"):
            skip_due_to_cli(
                "metrics_ui_alerts_jank_mode",
                "RUN_TRADING_STUB_METRICS_UI_ALERTS_JANK_MODE",
                raw,
            )
        else:
            normalized = raw.strip().lower()
            if normalized not in UI_ALERT_MODE_CHOICES:
                raise SystemExit(
                    f"RUN_TRADING_STUB_METRICS_UI_ALERTS_JANK_MODE musi być jedną z wartości: {', '.join(UI_ALERT_MODE_CHOICES)}"
                )
            args.metrics_ui_alerts_jank_mode = normalized
            apply_value(
                "metrics_ui_alerts_jank_mode",
                "RUN_TRADING_STUB_METRICS_UI_ALERTS_JANK_MODE",
                raw,
                normalized,
            )

    for option, env_var, attr in (
        (
            "metrics_ui_alerts_reduce_category",
            "RUN_TRADING_STUB_METRICS_UI_ALERTS_REDUCE_CATEGORY",
            "metrics_ui_alerts_reduce_category",
        ),
        (
            "metrics_ui_alerts_reduce_active_severity",
            "RUN_TRADING_STUB_METRICS_UI_ALERTS_REDUCE_ACTIVE_SEVERITY",
            "metrics_ui_alerts_reduce_active_severity",
        ),
        (
            "metrics_ui_alerts_reduce_recovered_severity",
            "RUN_TRADING_STUB_METRICS_UI_ALERTS_REDUCE_RECOVERED_SEVERITY",
            "metrics_ui_alerts_reduce_recovered_severity",
        ),
        (
            "metrics_ui_alerts_overlay_category",
            "RUN_TRADING_STUB_METRICS_UI_ALERTS_OVERLAY_CATEGORY",
            "metrics_ui_alerts_overlay_category",
        ),
        (
            "metrics_ui_alerts_overlay_exceeded_severity",
            "RUN_TRADING_STUB_METRICS_UI_ALERTS_OVERLAY_EXCEEDED_SEVERITY",
            "metrics_ui_alerts_overlay_exceeded_severity",
        ),
        (
            "metrics_ui_alerts_overlay_recovered_severity",
            "RUN_TRADING_STUB_METRICS_UI_ALERTS_OVERLAY_RECOVERED_SEVERITY",
            "metrics_ui_alerts_overlay_recovered_severity",
        ),
        (
            "metrics_ui_alerts_overlay_critical_severity",
            "RUN_TRADING_STUB_METRICS_UI_ALERTS_OVERLAY_CRITICAL_SEVERITY",
            "metrics_ui_alerts_overlay_critical_severity",
        ),
        (
            "metrics_ui_alerts_jank_category",
            "RUN_TRADING_STUB_METRICS_UI_ALERTS_JANK_CATEGORY",
            "metrics_ui_alerts_jank_category",
        ),
        (
            "metrics_ui_alerts_jank_spike_severity",
            "RUN_TRADING_STUB_METRICS_UI_ALERTS_JANK_SPIKE_SEVERITY",
            "metrics_ui_alerts_jank_spike_severity",
        ),
    ):
        if (raw := os.getenv(env_var)) is not None:
            if env_present(f"--{option.replace('_', '-')}"):
                skip_due_to_cli(option, env_var, raw)
            else:
                setattr(args, attr, raw)
                apply_value(option, env_var, raw, raw)

    if (raw := os.getenv("RUN_TRADING_STUB_METRICS_UI_ALERTS_JANK_CRITICAL_SEVERITY")) is not None:
        if env_present("--metrics-ui-alerts-jank-critical-severity"):
            skip_due_to_cli(
                "metrics_ui_alerts_jank_critical_severity",
                "RUN_TRADING_STUB_METRICS_UI_ALERTS_JANK_CRITICAL_SEVERITY",
                raw,
            )
        else:
            normalized = raw.strip().lower()
            if normalized in {"", "none", "null", "off", "disable", "disabled"}:
                args.metrics_ui_alerts_jank_critical_severity = None
                value_sources["metrics_ui_alerts_jank_critical_severity"] = "env_disabled"
                override_keys.add("metrics_ui_alerts_jank_critical_severity")
                record_entry(
                    {
                        "option": "metrics_ui_alerts_jank_critical_severity",
                        "variable": "RUN_TRADING_STUB_METRICS_UI_ALERTS_JANK_CRITICAL_SEVERITY",
                        "raw_value": raw,
                        "applied": True,
                        "parsed_value": None,
                        "note": "jank_critical_severity_disabled",
                    }
                )
            else:
                args.metrics_ui_alerts_jank_critical_severity = raw
                apply_value(
                    "metrics_ui_alerts_jank_critical_severity",
                    "RUN_TRADING_STUB_METRICS_UI_ALERTS_JANK_CRITICAL_SEVERITY",
                    raw,
                    raw,
                )

    if (raw := os.getenv("RUN_TRADING_STUB_METRICS_UI_ALERTS_JANK_CRITICAL_OVER_MS")) is not None:
        if env_present("--metrics-ui-alerts-jank-critical-over-ms"):
            skip_due_to_cli(
                "metrics_ui_alerts_jank_critical_over_ms",
                "RUN_TRADING_STUB_METRICS_UI_ALERTS_JANK_CRITICAL_OVER_MS",
                raw,
            )
        else:
            threshold_ms = parse_float("RUN_TRADING_STUB_METRICS_UI_ALERTS_JANK_CRITICAL_OVER_MS")
            args.metrics_ui_alerts_jank_critical_over_ms = threshold_ms
            apply_value(
                "metrics_ui_alerts_jank_critical_over_ms",
                "RUN_TRADING_STUB_METRICS_UI_ALERTS_JANK_CRITICAL_OVER_MS",
                raw,
                threshold_ms,
            )

    if (raw := os.getenv("RUN_TRADING_STUB_METRICS_UI_ALERTS_OVERLAY_CRITICAL_THRESHOLD")) is not None:
        if env_present("--metrics-ui-alerts-overlay-critical-threshold"):
            skip_due_to_cli(
                "metrics_ui_alerts_overlay_critical_threshold",
                "RUN_TRADING_STUB_METRICS_UI_ALERTS_OVERLAY_CRITICAL_THRESHOLD",
                raw,
            )
        else:
            args.metrics_ui_alerts_overlay_critical_threshold = parse_int(
                "RUN_TRADING_STUB_METRICS_UI_ALERTS_OVERLAY_CRITICAL_THRESHOLD"
            )
            apply_value(
                "metrics_ui_alerts_overlay_critical_threshold",
                "RUN_TRADING_STUB_METRICS_UI_ALERTS_OVERLAY_CRITICAL_THRESHOLD",
                raw,
                args.metrics_ui_alerts_overlay_critical_threshold,
            )

    if (raw := os.getenv("RUN_TRADING_STUB_METRICS_UI_ALERTS_AUDIT_DIR")) is not None:
        if env_present("--metrics-ui-alerts-audit-dir"):
            skip_due_to_cli("metrics_ui_alerts_audit_dir", "RUN_TRADING_STUB_METRICS_UI_ALERTS_AUDIT_DIR", raw)
        else:
            normalized = raw.strip().lower()
            if normalized in {"", "none", "null", "off", "disable", "disabled"}:
                args.metrics_ui_alerts_audit_dir = None
                value_sources["metrics_ui_alerts_audit_dir"] = "env_disabled"
                override_keys.add("metrics_ui_alerts_audit_dir")
                record_entry(
                    {
                        "option": "metrics_ui_alerts_audit_dir",
                        "variable": "RUN_TRADING_STUB_METRICS_UI_ALERTS_AUDIT_DIR",
                        "raw_value": raw,
                        "applied": True,
                        "parsed_value": None,
                        "note": "metrics_ui_alerts_audit_disabled",
                    }
                )
            else:
                args.metrics_ui_alerts_audit_dir = Path(raw).expanduser()
                apply_value(
                    "metrics_ui_alerts_audit_dir",
                    "RUN_TRADING_STUB_METRICS_UI_ALERTS_AUDIT_DIR",
                    raw,
                    str(args.metrics_ui_alerts_audit_dir),
                )
    if (raw := os.getenv("RUN_TRADING_STUB_METRICS_UI_ALERTS_AUDIT_BACKEND")) is not None:
        if env_present("--metrics-ui-alerts-audit-backend"):
            skip_due_to_cli(
                "metrics_ui_alerts_audit_backend",
                "RUN_TRADING_STUB_METRICS_UI_ALERTS_AUDIT_BACKEND",
                raw,
            )
        else:
            normalized = raw.strip().lower()
            if normalized in {"", "auto"}:
                args.metrics_ui_alerts_audit_backend = None
                value_sources["metrics_ui_alerts_audit_backend"] = "env_auto"
                override_keys.add("metrics_ui_alerts_audit_backend")
                record_entry(
                    {
                        "option": "metrics_ui_alerts_audit_backend",
                        "variable": "RUN_TRADING_STUB_METRICS_UI_ALERTS_AUDIT_BACKEND",
                        "raw_value": raw,
                        "applied": True,
                        "parsed_value": None,
                        "note": "metrics_ui_alerts_audit_backend_auto",
                    }
                )
            elif normalized in UI_ALERT_AUDIT_BACKEND_CHOICES:
                args.metrics_ui_alerts_audit_backend = normalized
                apply_value(
                    "metrics_ui_alerts_audit_backend",
                    "RUN_TRADING_STUB_METRICS_UI_ALERTS_AUDIT_BACKEND",
                    raw,
                    normalized,
                )
            else:
                parser.error(
                    f"RUN_TRADING_STUB_METRICS_UI_ALERTS_AUDIT_BACKEND musi być jedną z wartości: {', '.join(UI_ALERT_AUDIT_BACKEND_CHOICES)}"
                )

    if (raw := os.getenv("RUN_TRADING_STUB_METRICS_UI_ALERTS_AUDIT_PATTERN")) is not None:
        if env_present("--metrics-ui-alerts-audit-pattern"):
            skip_due_to_cli(
                "metrics_ui_alerts_audit_pattern",
                "RUN_TRADING_STUB_METRICS_UI_ALERTS_AUDIT_PATTERN",
                raw,
            )
        else:
            normalized = raw.strip()
            if not normalized:
                args.metrics_ui_alerts_audit_pattern = None
                value_sources["metrics_ui_alerts_audit_pattern"] = "env_disabled"
                override_keys.add("metrics_ui_alerts_audit_pattern")
                record_entry(
                    {
                        "option": "metrics_ui_alerts_audit_pattern",
                        "variable": "RUN_TRADING_STUB_METRICS_UI_ALERTS_AUDIT_PATTERN",
                        "raw_value": raw,
                        "applied": True,
                        "parsed_value": None,
                        "note": "metrics_ui_alerts_audit_pattern_default",
                    }
                )
            else:
                args.metrics_ui_alerts_audit_pattern = normalized
                apply_value(
                    "metrics_ui_alerts_audit_pattern",
                    "RUN_TRADING_STUB_METRICS_UI_ALERTS_AUDIT_PATTERN",
                    raw,
                    normalized,
                )

    if (raw := os.getenv("RUN_TRADING_STUB_METRICS_UI_ALERTS_AUDIT_RETENTION_DAYS")) is not None:
        if env_present("--metrics-ui-alerts-audit-retention-days"):
            skip_due_to_cli(
                "metrics_ui_alerts_audit_retention_days",
                "RUN_TRADING_STUB_METRICS_UI_ALERTS_AUDIT_RETENTION_DAYS",
                raw,
            )
        else:
            normalized = raw.strip()
            if normalized == "":
                args.metrics_ui_alerts_audit_retention_days = None
                value_sources["metrics_ui_alerts_audit_retention_days"] = "env_disabled"
                override_keys.add("metrics_ui_alerts_audit_retention_days")
                record_entry(
                    {
                        "option": "metrics_ui_alerts_audit_retention_days",
                        "variable": "RUN_TRADING_STUB_METRICS_UI_ALERTS_AUDIT_RETENTION_DAYS",
                        "raw_value": raw,
                        "applied": True,
                        "parsed_value": None,
                        "note": "metrics_ui_alerts_audit_retention_default",
                    }
                )
            else:
                args.metrics_ui_alerts_audit_retention_days = parse_int(
                    "RUN_TRADING_STUB_METRICS_UI_ALERTS_AUDIT_RETENTION_DAYS"
                )
                apply_value(
                    "metrics_ui_alerts_audit_retention_days",
                    "RUN_TRADING_STUB_METRICS_UI_ALERTS_AUDIT_RETENTION_DAYS",
                    raw,
                    args.metrics_ui_alerts_audit_retention_days,
                )

    if (raw := os.getenv("RUN_TRADING_STUB_METRICS_UI_ALERTS_AUDIT_FSYNC")) is not None:
        if env_present("--metrics-ui-alerts-audit-fsync"):
            skip_due_to_cli("metrics_ui_alerts_audit_fsync", "RUN_TRADING_STUB_METRICS_UI_ALERTS_AUDIT_FSYNC", raw)
        else:
            args.metrics_ui_alerts_audit_fsync = parse_bool("RUN_TRADING_STUB_METRICS_UI_ALERTS_AUDIT_FSYNC")
            apply_value(
                "metrics_ui_alerts_audit_fsync",
                "RUN_TRADING_STUB_METRICS_UI_ALERTS_AUDIT_FSYNC",
                raw,
                args.metrics_ui_alerts_audit_fsync,
            )

    if (raw := os.getenv("RUN_TRADING_STUB_METRICS_PRINT_ADDRESS")) is not None:
        if env_present("--metrics-print-address"):
            skip_due_to_cli("metrics_print_address", "RUN_TRADING_STUB_METRICS_PRINT_ADDRESS", raw)
        else:
            args.metrics_print_address = parse_bool("RUN_TRADING_STUB_METRICS_PRINT_ADDRESS")
            apply_value("metrics_print_address", "RUN_TRADING_STUB_METRICS_PRINT_ADDRESS", raw, args.metrics_print_address)

    if (raw := os.getenv("RUN_TRADING_STUB_METRICS_AUTH_TOKEN")) is not None:
        if env_present("--metrics-auth-token"):
            skip_due_to_cli("metrics_auth_token", "RUN_TRADING_STUB_METRICS_AUTH_TOKEN", raw)
        else:
            args.metrics_auth_token = raw
            apply_value("metrics_auth_token", "RUN_TRADING_STUB_METRICS_AUTH_TOKEN", raw, raw)

    # Audyt/plan
    if (raw := os.getenv("RUN_TRADING_STUB_PRINT_RUNTIME_PLAN")) is not None:
        if env_present("--print-runtime-plan"):
            skip_due_to_cli("print_runtime_plan", "RUN_TRADING_STUB_PRINT_RUNTIME_PLAN", raw)
        else:
            args.print_runtime_plan = parse_bool("RUN_TRADING_STUB_PRINT_RUNTIME_PLAN")
            apply_value("print_runtime_plan", "RUN_TRADING_STUB_PRINT_RUNTIME_PLAN", raw, args.print_runtime_plan)

    if (raw := os.getenv("RUN_TRADING_STUB_RUNTIME_PLAN_JSONL")) is not None:
        if env_present("--runtime-plan-jsonl"):
            skip_due_to_cli("runtime_plan_jsonl_path", "RUN_TRADING_STUB_RUNTIME_PLAN_JSONL", raw)
        else:
            args.runtime_plan_jsonl = Path(raw).expanduser()
            apply_value("runtime_plan_jsonl_path", "RUN_TRADING_STUB_RUNTIME_PLAN_JSONL", raw, str(args.runtime_plan_jsonl))

    if (raw := os.getenv("RUN_TRADING_STUB_FAIL_ON_SECURITY_WARNINGS")) is not None:
        if env_present("--fail-on-security-warnings"):
            skip_due_to_cli("fail_on_security_warnings", "RUN_TRADING_STUB_FAIL_ON_SECURITY_WARNINGS", raw)
        else:
            args.fail_on_security_warnings = parse_bool("RUN_TRADING_STUB_FAIL_ON_SECURITY_WARNINGS")
            apply_value(
                "fail_on_security_warnings",
                "RUN_TRADING_STUB_FAIL_ON_SECURITY_WARNINGS",
                raw,
                args.fail_on_security_warnings,
            )

    return entries, override_keys


def _load_dataset(dataset_paths: Iterable[Path], include_default: bool) -> InMemoryTradingDataset:
    dataset = build_default_dataset() if include_default else InMemoryTradingDataset()
    for path in dataset_paths:
        overlay = load_dataset_from_yaml(path)
        merge_datasets(dataset, overlay)
        LOGGER.info("Załadowano dataset z pliku %s", path)
    return dataset


def _build_ui_alert_sink(
    args,
) -> tuple[object, Path, Mapping[str, object]] | None:
    if args.disable_metrics_ui_alerts:
        return None
    if UiTelemetryAlertSink is None or DefaultAlertRouter is None:
        LOGGER.debug("Sink alertów telemetrii UI niedostępny – pomijam.")
        return None
    path = (
        Path(args.metrics_ui_alerts_jsonl).expanduser()
        if args.metrics_ui_alerts_jsonl
        else DEFAULT_UI_ALERTS_JSONL_PATH.expanduser()
    )
    try:
        audit_log = None
        raw_backend = getattr(args, "metrics_ui_alerts_audit_backend", None)
        requested_backend = (raw_backend or "auto").lower()
        if requested_backend not in UI_ALERT_AUDIT_BACKEND_CHOICES:
            raise ValueError(f"Nieobsługiwany backend audytu UI: {requested_backend}")
        audit_backend: dict[str, object] = {"requested": requested_backend}
        audit_dir = (
            Path(args.metrics_ui_alerts_audit_dir).expanduser()
            if getattr(args, "metrics_ui_alerts_audit_dir", None)
            else None
        )
        audit_pattern = (
            getattr(args, "metrics_ui_alerts_audit_pattern", None)
            or _DEFAULT_UI_ALERT_AUDIT_PATTERN
        )
        retention_override = getattr(args, "metrics_ui_alerts_audit_retention_days", None)
        audit_retention = (
            retention_override
            if retention_override is not None
            else _DEFAULT_UI_ALERT_AUDIT_RETENTION_DAYS
        )
        file_backend_requested = audit_dir is not None or requested_backend == "file"
        memory_forced = requested_backend == "memory"
        file_backend_error = False
        if requested_backend == "file" and audit_dir is None:
            raise ValueError("Backend plikowy audytu UI wymaga podania --metrics-ui-alerts-audit-dir")
        if not memory_forced and audit_dir is not None:
            if FileAlertAuditLog is None:
                if requested_backend == "file":
                    raise RuntimeError(
                        "Wymuszono backend plikowy alertów UI, ale FileAlertAuditLog nie jest dostępny w środowisku."
                    )
                LOGGER.warning(
                    "Wymuszono katalog audytu alertów UI (%s), ale FileAlertAuditLog nie jest dostępny – przełączam na backend w pamięci.",
                    audit_dir,
                )
            else:
                try:
                    audit_log = FileAlertAuditLog(
                        directory=audit_dir,
                        filename_pattern=audit_pattern,
                        retention_days=audit_retention,
                        fsync=bool(getattr(args, "metrics_ui_alerts_audit_fsync", False)),
                    )
                except Exception as exc:
                    file_backend_error = True
                    if requested_backend == "file":
                        raise RuntimeError(
                            "Nie udało się zainicjalizować FileAlertAuditLog dla wymuszonego backendu file"
                        ) from exc
                    LOGGER.exception(
                        "Nie udało się zainicjalizować FileAlertAuditLog w stubie – użyję audytu w pamięci."
                    )
                else:
                    audit_backend.update(
                        {
                            "backend": "file",
                            "directory": str(audit_dir),
                            "pattern": audit_pattern,
                            "retention_days": audit_retention,
                            "fsync": bool(getattr(args, "metrics_ui_alerts_audit_fsync", False)),
                        }
                    )
        if audit_log is None:
            if InMemoryAlertAuditLog is None:
                LOGGER.debug("Brak backendu audytu dla alertów UI – pomijam sink.")
                audit_backend.setdefault("backend", None)
                audit_backend.setdefault("note", "no_audit_backend_available")
                return None
            audit_log = InMemoryAlertAuditLog()
            audit_backend.update(
                {
                    "backend": "memory",
                    "fsync": bool(getattr(args, "metrics_ui_alerts_audit_fsync", False)),
                }
            )
            if requested_backend == "memory" and audit_dir is not None:
                audit_backend["note"] = "directory_ignored_memory_backend"
            elif file_backend_requested and requested_backend != "memory":
                audit_backend["note"] = (
                    "file_backend_error" if file_backend_error else "file_backend_unavailable"
                )
        elif "backend" not in audit_backend:
            audit_backend["backend"] = "file"

        router = DefaultAlertRouter(audit_log=audit_log)
        reduce_mode = (args.metrics_ui_alerts_reduce_mode or "enable").lower()
        overlay_mode = (args.metrics_ui_alerts_overlay_mode or "enable").lower()
        jank_mode = (args.metrics_ui_alerts_jank_mode or "enable").lower()
        reduce_dispatch = reduce_mode == "enable"
        overlay_dispatch = overlay_mode == "enable"
        jank_dispatch = jank_mode == "enable"
        reduce_logging = reduce_mode in {"enable", "jsonl"}
        overlay_logging = overlay_mode in {"enable", "jsonl"}
        jank_logging = jank_mode in {"enable", "jsonl"}
        reduce_category = args.metrics_ui_alerts_reduce_category or _DEFAULT_UI_CATEGORY
        reduce_active = (
            args.metrics_ui_alerts_reduce_active_severity or _DEFAULT_UI_SEVERITY_ACTIVE
        )
        reduce_recovered = (
            args.metrics_ui_alerts_reduce_recovered_severity or _DEFAULT_UI_SEVERITY_RECOVERED
        )
        overlay_category = args.metrics_ui_alerts_overlay_category or _DEFAULT_UI_CATEGORY
        overlay_exceeded = (
            args.metrics_ui_alerts_overlay_exceeded_severity or _DEFAULT_UI_SEVERITY_ACTIVE
        )
        overlay_recovered = (
            args.metrics_ui_alerts_overlay_recovered_severity or _DEFAULT_UI_SEVERITY_RECOVERED
        )
        overlay_critical = (
            args.metrics_ui_alerts_overlay_critical_severity or _DEFAULT_OVERLAY_SEVERITY_CRITICAL
        )
        threshold = (
            args.metrics_ui_alerts_overlay_critical_threshold
            if args.metrics_ui_alerts_overlay_critical_threshold is not None
            else _DEFAULT_OVERLAY_THRESHOLD
        )
        jank_category = args.metrics_ui_alerts_jank_category or _DEFAULT_UI_CATEGORY
        jank_spike = (
            args.metrics_ui_alerts_jank_spike_severity or _DEFAULT_JANK_SEVERITY_SPIKE
        )
        jank_critical = (
            args.metrics_ui_alerts_jank_critical_severity or _DEFAULT_JANK_SEVERITY_CRITICAL
        )
        jank_threshold = (
            args.metrics_ui_alerts_jank_critical_over_ms
            if args.metrics_ui_alerts_jank_critical_over_ms is not None
            else _DEFAULT_JANK_CRITICAL_THRESHOLD_MS
        )
        sink_kwargs = dict(
            jsonl_path=path,
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
        if threshold is not None:
            sink_kwargs["overlay_critical_threshold"] = threshold
        if jank_critical:
            sink_kwargs["jank_severity_critical"] = jank_critical
        if jank_threshold is not None:
            sink_kwargs["jank_critical_over_ms"] = jank_threshold

        sink_settings: dict[str, object] = {
            "path": str(path),
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
            "overlay_critical_threshold": threshold,
            "jank_category": jank_category,
            "jank_severity_spike": jank_spike,
            "jank_severity_critical": jank_critical,
            "jank_critical_over_ms": jank_threshold,
        }
        sink_settings["audit"] = dict(audit_backend)

        sink = UiTelemetryAlertSink(router, **sink_kwargs)
    except Exception:
        LOGGER.exception("Nie udało się zainicjalizować UiTelemetryAlertSink – kontynuuję bez alertów UI")
        return None
    LOGGER.info("Sink alertów telemetrii UI aktywny (log: %s)", path)
    return sink, path, sink_settings


def _start_metrics_service(args) -> tuple[object | None, str | None]:
    if not args.enable_metrics:
        return None, None
    if create_metrics_server is None:
        LOGGER.warning("create_metrics_server nie jest dostępne – pomijam serwer telemetrii")
        return None, None

    sinks: list[object] = []
    if JsonlSink is not None and args.metrics_jsonl:
        sinks.append(JsonlSink(Path(args.metrics_jsonl), fsync=args.metrics_jsonl_fsync))
    ui_sink_bundle = _build_ui_alert_sink(args)
    ui_alerts_path: Path | None = None
    ui_alerts_settings: Mapping[str, object] | None = None
    if ui_sink_bundle is not None:
        ui_sink, ui_alerts_path, ui_alerts_settings = ui_sink_bundle
        sinks.append(ui_sink)

    tls_config = None
    if args.metrics_tls_cert and args.metrics_tls_key:
        tls_config = {
            "certificate_path": Path(args.metrics_tls_cert),
            "private_key_path": Path(args.metrics_tls_key),
            "client_ca_path": Path(args.metrics_tls_client_ca) if args.metrics_tls_client_ca else None,
            "require_client_auth": bool(args.metrics_tls_require_client_cert),
        }

    audit_dir_path = (
        Path(args.metrics_ui_alerts_audit_dir).expanduser()
        if getattr(args, "metrics_ui_alerts_audit_dir", None)
        else None
    )
    audit_pattern = getattr(args, "metrics_ui_alerts_audit_pattern", None)
    audit_retention = getattr(args, "metrics_ui_alerts_audit_retention_days", None)
    audit_fsync = bool(getattr(args, "metrics_ui_alerts_audit_fsync", False))

    base_kwargs = dict(
        host=args.metrics_host,
        port=args.metrics_port,
        history_size=args.metrics_history_size,
        enable_logging_sink=not args.metrics_disable_log_sink,
        sinks=sinks or None,
    )
    attempts: list[dict[str, Any]] = []

    kw = dict(base_kwargs)
    if tls_config is not None:
        kw["tls_config"] = tls_config
    if args.metrics_auth_token:
        kw["auth_token"] = args.metrics_auth_token
    if ui_alerts_path is not None:
        kw["ui_alerts_jsonl_path"] = ui_alerts_path
    if ui_alerts_settings is not None:
        kw["ui_alerts_config"] = ui_alerts_settings
    if audit_dir_path is not None:
        kw["ui_alerts_audit_dir"] = audit_dir_path
    backend_choice = getattr(args, "metrics_ui_alerts_audit_backend", None)
    if backend_choice is not None:
        kw["ui_alerts_audit_backend"] = backend_choice
    if audit_pattern is not None:
        kw["ui_alerts_audit_pattern"] = audit_pattern
    if audit_retention is not None:
        kw["ui_alerts_audit_retention_days"] = audit_retention
    kw["ui_alerts_audit_fsync"] = audit_fsync
    attempts.append(kw)

    if "tls_config" in kw:
        k2 = dict(base_kwargs)
        if args.metrics_auth_token:
            k2["auth_token"] = args.metrics_auth_token
        if ui_alerts_path is not None:
            k2["ui_alerts_jsonl_path"] = ui_alerts_path
        if ui_alerts_settings is not None:
            k2["ui_alerts_config"] = ui_alerts_settings
        if audit_dir_path is not None:
            k2["ui_alerts_audit_dir"] = audit_dir_path
        if backend_choice is not None:
            k2["ui_alerts_audit_backend"] = backend_choice
        if audit_pattern is not None:
            k2["ui_alerts_audit_pattern"] = audit_pattern
        if audit_retention is not None:
            k2["ui_alerts_audit_retention_days"] = audit_retention
        k2["ui_alerts_audit_fsync"] = audit_fsync
        attempts.append(k2)
    if "auth_token" in kw:
        k3 = dict(base_kwargs)
        if tls_config is not None:
            k3["tls_config"] = tls_config
        if ui_alerts_path is not None:
            k3["ui_alerts_jsonl_path"] = ui_alerts_path
        if ui_alerts_settings is not None:
            k3["ui_alerts_config"] = ui_alerts_settings
        if audit_dir_path is not None:
            k3["ui_alerts_audit_dir"] = audit_dir_path
        if backend_choice is not None:
            k3["ui_alerts_audit_backend"] = backend_choice
        if audit_pattern is not None:
            k3["ui_alerts_audit_pattern"] = audit_pattern
        if audit_retention is not None:
            k3["ui_alerts_audit_retention_days"] = audit_retention
        k3["ui_alerts_audit_fsync"] = audit_fsync
        attempts.append(k3)
    fallback_kwargs = dict(base_kwargs)
    if ui_alerts_settings is not None:
        fallback_kwargs["ui_alerts_config"] = ui_alerts_settings
    if ui_alerts_path is not None:
        fallback_kwargs["ui_alerts_jsonl_path"] = ui_alerts_path
    if audit_dir_path is not None:
        fallback_kwargs["ui_alerts_audit_dir"] = audit_dir_path
    if backend_choice is not None:
        fallback_kwargs["ui_alerts_audit_backend"] = backend_choice
    if audit_pattern is not None:
        fallback_kwargs["ui_alerts_audit_pattern"] = audit_pattern
    if audit_retention is not None:
        fallback_kwargs["ui_alerts_audit_retention_days"] = audit_retention
    fallback_kwargs["ui_alerts_audit_fsync"] = audit_fsync
    attempts.append(fallback_kwargs)
    attempts.append(dict(base_kwargs))

    last_exc: Exception | None = None
    for opt in attempts:
        try:
            if ui_alerts_path is not None:
                opt.setdefault("ui_alerts_jsonl_path", ui_alerts_path)
            server = create_metrics_server(**opt)  # type: ignore[misc]
            server.start()
            address = getattr(server, "address", f"{args.metrics_host}:{args.metrics_port}")
            LOGGER.info("Serwer MetricsService uruchomiony na %s", address)
            if args.metrics_print_address and address:
                print(address)
            if args.metrics_auth_token:
                LOGGER.info("Serwer telemetrii wymaga Authorization: Bearer <token>")
            return server, address
        except TypeError as exc:
            last_exc = exc
            continue
        except Exception:
            LOGGER.exception("Nie udało się utworzyć serwera MetricsService")
            return None, None

    LOGGER.error("Nie udało się wywołać create_metrics_server z kompatybilnymi argumentami: %s", last_exc)
    return None, None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Startuje stub tradingowy gRPC dla środowiska developerskiego UI.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Adres hosta (domyślnie 127.0.0.1).")
    parser.add_argument("--port", type=int, default=50051, help="Port (0 = losowy).")
    parser.add_argument(
        "--dataset", action="append", type=Path, default=None, help="Plik YAML z danymi stubu (można podać wielokrotnie)."
    )
    parser.add_argument("--no-default-dataset", action="store_true", help="Nie ładuj datasetu domyślnego.")
    parser.add_argument("--shutdown-after", type=float, default=None, help="Auto-stop po tylu sekundach.")
    parser.add_argument("--max-workers", type=int, default=8, help="Wątki w puli gRPC (domyślnie 8).")
    parser.add_argument("--stream-repeat", action="store_true", help="Pętla streamu incrementów.")
    parser.add_argument("--stream-interval", type=float, default=0.0, help="Odstęp (s) między incrementami.")
    parser.add_argument("--log-level", default="info", help="Poziom logowania (debug, info, warning, error).")
    parser.add_argument("--print-address", action="store_true", help="Wypisz adres stubu na stdout.")

    # --- MetricsService (opcjonalny towarzyszący serwer telemetrii UI) ---
    parser.add_argument("--enable-metrics", action="store_true", help="Uruchom towarzyszący MetricsService.")
    metrics_group = parser.add_argument_group("metrics", "Opcje serwera MetricsService")
    metrics_group.add_argument("--metrics-host", default="127.0.0.1", help="Host MetricsService.")
    metrics_group.add_argument("--metrics-port", type=int, default=50061, help="Port MetricsService (0 = losowy).")
    metrics_group.add_argument("--metrics-history-size", type=int, default=512, help="Rozmiar historii snapshotów.")
    metrics_group.add_argument("--metrics-jsonl", type=Path, default=None, help="Plik JSONL na snapshoty metryk.")
    metrics_group.add_argument("--metrics-jsonl-fsync", action="store_true", help="fsync po każdym wpisie JSONL.")
    metrics_group.add_argument("--metrics-disable-log-sink", action="store_true", help="Wyłącz LoggingSink.")
    metrics_group.add_argument("--metrics-tls-cert", type=Path, default=None, help="Certyfikat TLS (PEM).")
    metrics_group.add_argument("--metrics-tls-key", type=Path, default=None, help="Klucz prywatny TLS (PEM).")
    metrics_group.add_argument("--metrics-tls-client-ca", type=Path, default=None, help="CA klientów mTLS (PEM).")
    metrics_group.add_argument("--metrics-tls-require-client-cert", action="store_true", help="Wymagaj mTLS.")
    metrics_group.add_argument("--metrics-auth-token", default=None, help="Token Bearer wymagany przez telemetrię.")
    metrics_group.add_argument(
        "--metrics-ui-alerts-jsonl", type=Path, default=None, help="Plik JSONL dla alertów UI (domyślnie logs/ui_telemetry_alerts.jsonl)."
    )
    metrics_group.add_argument("--disable-metrics-ui-alerts", action="store_true", help="Wyłącz alerty telemetrii UI.")
    metrics_group.add_argument(
        "--metrics-ui-alerts-reduce-mode",
        choices=UI_ALERT_MODE_CHOICES,
        default=None,
        help="Tryb alertów reduce-motion (enable/disable).",
    )
    metrics_group.add_argument(
        "--metrics-ui-alerts-overlay-mode",
        choices=UI_ALERT_MODE_CHOICES,
        default=None,
        help="Tryb alertów budżetu overlayów (enable/disable).",
    )
    metrics_group.add_argument(
        "--metrics-ui-alerts-reduce-category",
        default=None,
        help="Kategoria alertów reduce-motion (domyślnie ui.performance).",
    )
    metrics_group.add_argument(
        "--metrics-ui-alerts-reduce-active-severity",
        default=None,
        help="Severity alertu przy aktywacji reduce-motion (domyślnie warning).",
    )
    metrics_group.add_argument(
        "--metrics-ui-alerts-reduce-recovered-severity",
        default=None,
        help="Severity alertu przy powrocie z reduce-motion (domyślnie info).",
    )
    metrics_group.add_argument(
        "--metrics-ui-alerts-overlay-category",
        default=None,
        help="Kategoria alertów overlay budget (domyślnie ui.performance).",
    )
    metrics_group.add_argument(
        "--metrics-ui-alerts-overlay-exceeded-severity",
        default=None,
        help="Severity alertu przy przekroczeniu overlay budget (domyślnie warning).",
    )
    metrics_group.add_argument(
        "--metrics-ui-alerts-overlay-recovered-severity",
        default=None,
        help="Severity alertu przy powrocie overlay budget (domyślnie info).",
    )
    metrics_group.add_argument(
        "--metrics-ui-alerts-overlay-critical-severity",
        default=None,
        help="Severity krytycznego alertu overlay (różnica >= próg).",
    )
    metrics_group.add_argument(
        "--metrics-ui-alerts-overlay-critical-threshold",
        type=int,
        default=None,
        help="Próg nadwyżki nakładek powodujący alert krytyczny.",
    )
    metrics_group.add_argument(
        "--metrics-ui-alerts-jank-mode",
        choices=UI_ALERT_MODE_CHOICES,
        default=None,
        help="Tryb alertów jank (enable/jsonl/disable).",
    )
    metrics_group.add_argument(
        "--metrics-ui-alerts-jank-category",
        default=None,
        help="Kategoria alertów jank (domyślnie ui.performance).",
    )
    metrics_group.add_argument(
        "--metrics-ui-alerts-jank-spike-severity",
        default=None,
        help="Severity alertu jank przy pojedynczym skoku (domyślnie warning).",
    )
    metrics_group.add_argument(
        "--metrics-ui-alerts-jank-critical-severity",
        default=None,
        help="Severity alertu jank po przekroczeniu progu krytycznego (domyślnie brak).",
    )
    metrics_group.add_argument(
        "--metrics-ui-alerts-jank-critical-over-ms",
        type=float,
        default=None,
        help="Próg (ms) przekroczenia limitu klatki dla alertu jank o severity krytycznym.",
    )
    metrics_group.add_argument(
        "--metrics-ui-alerts-audit-dir",
        type=Path,
        default=None,
        help="Katalog audytu alertów UI emitowanych przez stub (JSONL).",
    )
    metrics_group.add_argument(
        "--metrics-ui-alerts-audit-backend",
        choices=UI_ALERT_AUDIT_BACKEND_CHOICES,
        default=None,
        help="Preferowany backend audytu alertów UI (auto/file/memory).",
    )
    metrics_group.add_argument(
        "--metrics-ui-alerts-audit-pattern",
        default=None,
        help="Wzorzec nazw plików audytu UI (domyślnie metrics-ui-alerts-%%Y%%m%%d.jsonl).",
    )
    metrics_group.add_argument(
        "--metrics-ui-alerts-audit-retention-days",
        type=int,
        default=None,
        help="Retencja plików audytu alertów UI w dniach (domyślnie 90).",
    )
    metrics_group.add_argument(
        "--metrics-ui-alerts-audit-fsync",
        action="store_true",
        help="Wymuś fsync po każdym wpisie audytu alertów UI.",
    )
    metrics_group.add_argument("--metrics-print-address", action="store_true", help="Wypisz adres MetricsService.")

    # --- Audyt/Plan runtime ---
    audit_group = parser.add_argument_group("audit", "Opcje audytu runtime")
    audit_group.add_argument("--print-runtime-plan", action="store_true", help="Wypisz plan runtime i wyjdź.")
    audit_group.add_argument("--runtime-plan-jsonl", type=Path, default=None, help="Dopisz plan runtime do JSONL.")
    audit_group.add_argument(
        "--fail-on-security-warnings",
        action="store_true",
        help="Zakończ działanie, jeśli plan zawiera ostrzeżenia bezpieczeństwa JSONL/TLS.",
    )
    return parser


def _install_signal_handlers(stop_callback) -> None:
    def handler(signum, _frame) -> None:  # pragma: no cover
        LOGGER.info("Otrzymano sygnał %s – trwa zatrzymywanie serwera", signum)
        stop_callback()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except ValueError:
            LOGGER.warning("Nie udało się zarejestrować handlera sygnału %s", sig)


def _dataset_summary(dataset: InMemoryTradingDataset) -> Mapping[str, object]:
    """Buduje skrócone statystyki datasetu dla audytu."""
    history_entries = sum(len(series) for series in dataset.history.values())
    stream_snapshot_entries = sum(len(series) for series in dataset.stream_snapshots.values())
    stream_increment_entries = sum(len(series) for series in dataset.stream_increments.values())
    risk_state_entries = sum(len(series) for series in dataset.risk_states.values())
    return {
        "history_series": len(dataset.history),
        "history_entries": history_entries,
        "stream_snapshot_series": len(dataset.stream_snapshots),
        "stream_snapshot_entries": stream_snapshot_entries,
        "stream_increment_series": len(dataset.stream_increments),
        "stream_increment_entries": stream_increment_entries,
        "risk_state_series": len(dataset.risk_states),
        "risk_state_entries": risk_state_entries,
        "metrics_samples": len(dataset.metrics),
        "performance_guard": dict(dataset.performance_guard),
        "health_defined": dataset.health is not None,
    }


def _build_dataset_plan(dataset: InMemoryTradingDataset, dataset_paths: Iterable[Path], include_default: bool) -> Mapping[str, object]:
    sources: list[Mapping[str, object]] = []
    if include_default:
        sources.append({"type": "default", "description": "build_default_dataset"})
    for raw_path in dataset_paths:
        path = Path(str(raw_path)).expanduser()
        sources.append({"type": "file", "path": str(path), "metadata": file_reference_metadata(path, role="stub_dataset")})
    return {"include_default": include_default, "sources": sources, "summary": _dataset_summary(dataset)}


def _build_metrics_plan(args) -> Mapping[str, object]:
    metrics_available = create_metrics_server is not None
    ui_alert_deps = UiTelemetryAlertSink is not None and DefaultAlertRouter is not None and InMemoryAlertAuditLog is not None
    jsonl_info: Mapping[str, object]
    if args.metrics_jsonl:
        jsonl_path = Path(str(args.metrics_jsonl)).expanduser()
        jsonl_info = {
            "configured": True,
            "path": str(jsonl_path),
            "metadata": file_reference_metadata(jsonl_path, role="jsonl"),
            "fsync_enabled": bool(args.metrics_jsonl_fsync),
        }
    else:
        jsonl_info = {"configured": False, "fsync_enabled": bool(args.metrics_jsonl_fsync)}

    tls_configured = bool(args.metrics_tls_cert and args.metrics_tls_key)
    if tls_configured:
        cert_path = Path(str(args.metrics_tls_cert)).expanduser()
        key_path = Path(str(args.metrics_tls_key)).expanduser()
        client_ca_path = Path(str(args.metrics_tls_client_ca)).expanduser() if args.metrics_tls_client_ca else None
        tls_info: Mapping[str, object] = {
            "configured": True,
            "require_client_auth": bool(args.metrics_tls_require_client_cert),
            "certificate": file_reference_metadata(cert_path, role="tls_cert"),
            "private_key": file_reference_metadata(key_path, role="tls_key"),
        }
        if client_ca_path is not None:
            tls_info["client_ca"] = file_reference_metadata(client_ca_path, role="tls_client_ca")
    else:
        tls_info = {"configured": False, "require_client_auth": bool(args.metrics_tls_require_client_cert)}  # type: ignore[assignment]

    ui_alert_path = Path(str(args.metrics_ui_alerts_jsonl)).expanduser() if args.metrics_ui_alerts_jsonl else DEFAULT_UI_ALERTS_JSONL_PATH.expanduser()
    reduce_mode_value = (args.metrics_ui_alerts_reduce_mode or "enable").lower()
    overlay_mode_value = (args.metrics_ui_alerts_overlay_mode or "enable").lower()
    jank_mode_value = (args.metrics_ui_alerts_jank_mode or "enable").lower()
    reduce_dispatch = reduce_mode_value == "enable"
    overlay_dispatch = overlay_mode_value == "enable"
    jank_dispatch = jank_mode_value == "enable"
    reduce_logging = reduce_mode_value in {"enable", "jsonl"}
    overlay_logging = overlay_mode_value in {"enable", "jsonl"}
    jank_logging = jank_mode_value in {"enable", "jsonl"}
    sink_expected = (
        reduce_logging
        or overlay_logging
        or jank_logging
        or reduce_dispatch
        or overlay_dispatch
        or jank_dispatch
    )
    audit_dir_arg = getattr(args, "metrics_ui_alerts_audit_dir", None)
    audit_dir = (
        Path(str(audit_dir_arg)).expanduser()
        if audit_dir_arg
        else None
    )
    audit_pattern = (
        getattr(args, "metrics_ui_alerts_audit_pattern", None)
        or _DEFAULT_UI_ALERT_AUDIT_PATTERN
    )
    retention_override = getattr(args, "metrics_ui_alerts_audit_retention_days", None)
    audit_retention = (
        retention_override
        if retention_override is not None
        else _DEFAULT_UI_ALERT_AUDIT_RETENTION_DAYS
    )
    raw_backend_choice = getattr(args, "metrics_ui_alerts_audit_backend", None)
    requested_backend = (raw_backend_choice or "auto").lower()
    memory_forced = requested_backend == "memory"
    file_backend_supported = audit_dir is not None and FileAlertAuditLog is not None and not memory_forced
    audit_info: Mapping[str, object]
    if requested_backend == "file" and audit_dir is None:
        audit_info = {
            "requested": requested_backend,
            "backend": None,
            "note": "file_backend_requires_directory",
            "fsync": bool(getattr(args, "metrics_ui_alerts_audit_fsync", False)),
        }
    elif requested_backend == "file" and FileAlertAuditLog is None:
        audit_info = {
            "requested": requested_backend,
            "backend": None,
            "note": "file_backend_unavailable",
            "fsync": bool(getattr(args, "metrics_ui_alerts_audit_fsync", False)),
        }
    elif requested_backend == "memory":
        audit_info = {
            "requested": requested_backend,
            "backend": "memory",
            "fsync": bool(getattr(args, "metrics_ui_alerts_audit_fsync", False)),
            "note": "directory_ignored_memory_backend"
            if audit_dir is not None
            else None,
        }
        if audit_info["note"] is None:
            audit_info = {k: v for k, v in audit_info.items() if v is not None}
    elif file_backend_supported:
        audit_info = {
            "requested": requested_backend,
            "backend": "file",
            "directory": str(audit_dir),
            "pattern": audit_pattern,
            "retention_days": audit_retention,
            "fsync": bool(getattr(args, "metrics_ui_alerts_audit_fsync", False)),
        }
    else:
        audit_info = {
            "requested": requested_backend,
            "backend": "memory",
            "fsync": bool(getattr(args, "metrics_ui_alerts_audit_fsync", False)),
        }
        if audit_dir is not None:
            audit_info["note"] = "file_backend_unavailable"

    ui_alerts_info: Mapping[str, object] = {
        "configured": not bool(args.disable_metrics_ui_alerts),
        "available": ui_alert_deps,
        "expected_active": bool(args.enable_metrics and not args.disable_metrics_ui_alerts and metrics_available and ui_alert_deps and sink_expected),
        "path": str(ui_alert_path),
        "metadata": file_reference_metadata(ui_alert_path, role="ui_alerts_jsonl"),
        "source": "cli" if args.metrics_ui_alerts_jsonl else "default",
        "reduce_mode": reduce_mode_value,
        "overlay_mode": overlay_mode_value,
        "jank_mode": jank_mode_value,
        "reduce_motion_dispatch": reduce_dispatch,
        "overlay_dispatch": overlay_dispatch,
        "jank_dispatch": jank_dispatch,
        "reduce_motion_logging": reduce_logging,
        "overlay_logging": overlay_logging,
        "jank_logging": jank_logging,
        "reduce_motion_category": args.metrics_ui_alerts_reduce_category or _DEFAULT_UI_CATEGORY,
        "reduce_motion_severity_active": args.metrics_ui_alerts_reduce_active_severity or _DEFAULT_UI_SEVERITY_ACTIVE,
        "reduce_motion_severity_recovered": args.metrics_ui_alerts_reduce_recovered_severity or _DEFAULT_UI_SEVERITY_RECOVERED,
        "overlay_category": args.metrics_ui_alerts_overlay_category or _DEFAULT_UI_CATEGORY,
        "overlay_severity_exceeded": args.metrics_ui_alerts_overlay_exceeded_severity or _DEFAULT_UI_SEVERITY_ACTIVE,
        "overlay_severity_recovered": args.metrics_ui_alerts_overlay_recovered_severity or _DEFAULT_UI_SEVERITY_RECOVERED,
        "overlay_severity_critical": args.metrics_ui_alerts_overlay_critical_severity or _DEFAULT_OVERLAY_SEVERITY_CRITICAL,
        "overlay_critical_threshold": (
            args.metrics_ui_alerts_overlay_critical_threshold
            if args.metrics_ui_alerts_overlay_critical_threshold is not None
            else _DEFAULT_OVERLAY_THRESHOLD
        ),
        "jank_category": args.metrics_ui_alerts_jank_category or _DEFAULT_UI_CATEGORY,
        "jank_severity_spike": args.metrics_ui_alerts_jank_spike_severity or _DEFAULT_JANK_SEVERITY_SPIKE,
        "jank_severity_critical": (
            args.metrics_ui_alerts_jank_critical_severity
            if args.metrics_ui_alerts_jank_critical_severity is not None
            else _DEFAULT_JANK_SEVERITY_CRITICAL
        ),
        "jank_critical_over_ms": (
            args.metrics_ui_alerts_jank_critical_over_ms
            if args.metrics_ui_alerts_jank_critical_over_ms is not None
            else _DEFAULT_JANK_CRITICAL_THRESHOLD_MS
        ),
        "audit": audit_info,
    }

    warnings: list[str] = []
    if args.enable_metrics and not metrics_available:
        warnings.append("create_metrics_server nie jest dostępne – telemetria nie uruchomi się.")
    if args.enable_metrics and not args.disable_metrics_ui_alerts and not ui_alert_deps:
        warnings.append("UiTelemetryAlertSink lub router alertów nie są dostępne – alerty UI będą wyłączone.")

    return {
        "available": metrics_available,
        "enabled": bool(args.enable_metrics),
        "host": args.metrics_host,
        "port": args.metrics_port,
        "history_size": args.metrics_history_size,
        "log_sink_enabled": not bool(args.metrics_disable_log_sink),
        "auth_token_set": bool(args.metrics_auth_token),
        "jsonl": jsonl_info,
        "ui_alerts": ui_alerts_info,
        "tls": tls_info,
        "warnings": warnings,
    }


def _build_runtime_plan_payload(
    *,
    args,
    dataset: InMemoryTradingDataset,
    dataset_paths: Iterable[Path],
    environment_overrides: Iterable[Mapping[str, object]],
    parameter_sources: Mapping[str, str],
) -> Mapping[str, object]:
    dataset_plan = _build_dataset_plan(dataset, dataset_paths, include_default=not args.no_default_dataset)
    metrics_plan = _build_metrics_plan(args)
    plan: dict[str, object] = {
        "version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "server": {
            "host": args.host,
            "port": args.port,
            "max_workers": args.max_workers,
            "stream_repeat": bool(args.stream_repeat),
            "stream_interval": args.stream_interval,
            "shutdown_after": args.shutdown_after,
            "log_level": args.log_level,
        },
        "dataset": dataset_plan,
        "metrics": metrics_plan,
    }
    overrides_list = [dict(entry) for entry in environment_overrides]
    plan["environment"] = {"overrides": overrides_list, "parameter_sources": dict(parameter_sources)}

    fail_parameter_source = parameter_sources.get("fail_on_security_warnings", "default")
    fail_env_entry = next((entry for entry in overrides_list if entry.get("option") == "fail_on_security_warnings"), None)
    env_variable = "RUN_TRADING_STUB_FAIL_ON_SECURITY_WARNINGS"
    fail_source = "cli" if fail_parameter_source == "cli" else (f"env:{env_variable}" if fail_parameter_source.startswith("env") else fail_parameter_source)

    plan["security"] = {
        "fail_on_security_warnings": {
            k: v
            for k, v in {
                "enabled": bool(args.fail_on_security_warnings),
                "source": fail_source,
                "parameter_source": fail_parameter_source,
                "cli_flag": "--fail-on-security-warnings",
                "environment_variable": (fail_env_entry.get("variable") if fail_env_entry else None),
                "environment_raw_value": (fail_env_entry.get("raw_value") if fail_env_entry else None),
                "environment_applied": (fail_env_entry.get("applied") if fail_env_entry is not None else None),
                "environment_reason": (fail_env_entry.get("reason") if fail_env_entry else None),
                "environment_parsed_value": (fail_env_entry.get("parsed_value") if fail_env_entry else None),
                "environment_note": (fail_env_entry.get("note") if fail_env_entry else None),
            }.items()
            if v is not None
        },
        "parameter_sources": {"fail_on_security_warnings": fail_parameter_source},
    }
    return plan


def _append_runtime_plan_jsonl(path: Path, payload: Mapping[str, object]) -> Path:
    """Dopisuje wpis planu runtime do pliku JSONL."""
    path = Path(str(path)).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_args)

    provided_flags = _collect_provided_flags(raw_args)
    value_sources = _initial_value_sources(provided_flags)
    environment_overrides, _ = _apply_environment_overrides(
        args, parser=parser, provided_flags=provided_flags, value_sources=value_sources
    )
    parameter_sources = _finalize_value_sources(value_sources)

    _configure_logging(args.log_level)

    if args.stream_interval < 0:
        parser.error("--stream-interval musi być liczbą nieujemną")
    if args.enable_metrics and args.metrics_history_size <= 0:
        parser.error("--metrics-history-size musi być dodatnie przy włączonej telemetrii")
    if args.enable_metrics and args.metrics_port < 0:
        parser.error("--metrics-port musi być liczbą nieujemną")
    if args.enable_metrics:
        if bool(args.metrics_tls_cert) ^ bool(args.metrics_tls_key):
            parser.error("TLS wymaga jednoczesnego podania --metrics-tls-cert oraz --metrics-tls-key")
        for option in ("metrics_tls_cert", "metrics_tls_key", "metrics_tls_client_ca"):
            path = getattr(args, option)
            if path and not Path(path).exists():
                parser.error(f"Ścieżka {option.replace('_', '-')} nie istnieje: {path}")

    backend_choice = (getattr(args, "metrics_ui_alerts_audit_backend", None) or "auto").lower()
    if backend_choice not in UI_ALERT_AUDIT_BACKEND_CHOICES:
        parser.error("--metrics-ui-alerts-audit-backend wymaga wartości auto/file/memory")
    if backend_choice == "file" and not getattr(args, "metrics_ui_alerts_audit_dir", None):
        parser.error(
            "--metrics-ui-alerts-audit-backend file wymaga jednoczesnego podania --metrics-ui-alerts-audit-dir"
        )

    dataset_paths = args.dataset or []
    dataset = _load_dataset(dataset_paths, include_default=not args.no_default_dataset)

    # Plan runtime (opcjonalny)
    runtime_plan_payload: Mapping[str, object] | None = None
    need_runtime_plan = bool(args.print_runtime_plan or args.runtime_plan_jsonl or args.fail_on_security_warnings)
    if need_runtime_plan:
        try:
            runtime_plan_payload = _build_runtime_plan_payload(
                args=args,
                dataset=dataset,
                dataset_paths=dataset_paths,
                environment_overrides=environment_overrides,
                parameter_sources=parameter_sources,
            )
        except Exception:
            LOGGER.exception("Nie udało się zbudować planu runtime stubu tradingowego")
            return 2

    security_warnings_detected = False
    if runtime_plan_payload is not None and args.fail_on_security_warnings:
        security_warnings_detected = _log_security_warnings(
            runtime_plan_payload,
            fail_on_warnings=True,
            logger=LOGGER,
            context="run_trading_stub_server.runtime_plan",
        )

    if args.runtime_plan_jsonl and runtime_plan_payload is not None:
        try:
            destination = _append_runtime_plan_jsonl(args.runtime_plan_jsonl, runtime_plan_payload)
            LOGGER.info("Plan runtime zapisany do %s", destination)
        except Exception as exc:
            LOGGER.error("Nie udało się zapisać planu runtime do %s: %s", args.runtime_plan_jsonl, exc)
            return 2

    if args.print_runtime_plan and runtime_plan_payload is not None:
        json.dump(runtime_plan_payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        if args.fail_on_security_warnings and security_warnings_detected:
            return 3
        return 0

    if args.fail_on_security_warnings and runtime_plan_payload is not None and security_warnings_detected:
        return 3

    if dataset.performance_guard:
        guard_preview = ", ".join(f"{key}={value}" for key, value in sorted(dataset.performance_guard.items()))
        LOGGER.info("Konfiguracja performance guard: %s", guard_preview)

    metrics_server, metrics_address = _start_metrics_service(args)

    server = TradingStubServer(
        dataset=dataset,
        host=args.host,
        port=args.port,
        max_workers=args.max_workers,
        stream_repeat=args.stream_repeat,
        stream_interval=args.stream_interval,
    )

    should_stop = False
    metrics_stopped = False

    def request_stop() -> None:
        nonlocal should_stop, metrics_stopped
        should_stop = True
        server.stop(grace=1.0)
        if metrics_server is not None and not metrics_stopped:
            try:
                metrics_server.stop(grace=1.0)
            except Exception:
                LOGGER.exception("Błąd podczas zatrzymywania MetricsService")
            metrics_stopped = True

    _install_signal_handlers(request_stop)

    server.start()
    LOGGER.info("Serwer stub wystartował na %s", server.address)
    if args.print_address:
        print(server.address)
    if args.metrics_print_address and metrics_address:
        print(metrics_address)

    try:
        if args.shutdown_after is not None:
            LOGGER.info("Serwer zakończy pracę automatycznie po %.2f s (lub szybciej po sygnale)", args.shutdown_after)
            terminated = server.wait_for_termination(timeout=args.shutdown_after)
            if not terminated and not should_stop:
                LOGGER.info("Limit czasu minął – zatrzymuję serwer stub i telemetrię.")
                request_stop()
        else:
            LOGGER.info("Naciśnij Ctrl+C, aby zakończyć pracę stubu.")
            server.wait_for_termination()
    except KeyboardInterrupt:  # pragma: no cover
        LOGGER.info("Przerwano przez użytkownika – zatrzymywanie serwera.")
    finally:
        if not should_stop:
            server.stop(grace=1.0)
        if metrics_server is not None and not metrics_stopped:
            try:
                metrics_server.stop(grace=1.0)
            except Exception:
                LOGGER.exception("Błąd podczas zatrzymywania MetricsService")
        LOGGER.info("Serwer stub został zatrzymany.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
