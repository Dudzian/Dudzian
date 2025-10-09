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
try:  # najczęściej tak:
    from bot_core.alerts import DefaultAlertRouter as _Router, InMemoryAlertAuditLog as _Audit
    DefaultAlertRouter = _Router
    InMemoryAlertAuditLog = _Audit
except Exception:
    try:  # czasem w oddzielnym module
        from bot_core.alerts.audit import InMemoryAlertAuditLog as _Audit2  # type: ignore
        from bot_core.alerts import DefaultAlertRouter as _Router2  # type: ignore
        DefaultAlertRouter = _Router2
        InMemoryAlertAuditLog = _Audit2
    except Exception:
        pass

LOGGER = logging.getLogger("run_trading_stub_server")


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
        "--metrics-print-address": "metrics_print_address",
        "--metrics-auth-token": "metrics_auth_token",
        "--print-runtime-plan": "print_runtime_plan",
        "--runtime-plan-jsonl": "runtime_plan_jsonl_path",
        "--fail-on-security-warnings": "fail_on_security_warnings",
    }
    return {key: "cli" for key, val in mapping.items() if key in provided}


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


def _build_ui_alert_sink(args) -> object | None:
    if args.disable_metrics_ui_alerts:
        return None
    if UiTelemetryAlertSink is None or DefaultAlertRouter is None or InMemoryAlertAuditLog is None:
        LOGGER.debug("Sink alertów telemetrii UI niedostępny – pomijam.")
        return None
    path = Path(args.metrics_ui_alerts_jsonl).expanduser() if args.metrics_ui_alerts_jsonl else DEFAULT_UI_ALERTS_JSONL_PATH.expanduser()
    try:
        router = DefaultAlertRouter(audit_log=InMemoryAlertAuditLog())
        sink = UiTelemetryAlertSink(router, jsonl_path=path)
    except Exception:
        LOGGER.exception("Nie udało się zainicjalizować UiTelemetryAlertSink – kontynuuję bez alertów UI")
        return None
    LOGGER.info("Sink alertów telemetrii UI aktywny (log: %s)", path)
    return sink


def _start_metrics_service(args) -> tuple[object | None, str | None]:
    if not args.enable_metrics:
        return None, None
    if create_metrics_server is None:
        LOGGER.warning("create_metrics_server nie jest dostępne – pomijam serwer telemetrii")
        return None, None

    sinks: list[object] = []
    if JsonlSink is not None and args.metrics_jsonl:
        sinks.append(JsonlSink(Path(args.metrics_jsonl), fsync=args.metrics_jsonl_fsync))
    ui_sink = _build_ui_alert_sink(args)
    if ui_sink is not None:
        sinks.append(ui_sink)

    tls_config = None
    if args.metrics_tls_cert and args.metrics_tls_key:
        tls_config = {
            "certificate_path": Path(args.metrics_tls_cert),
            "private_key_path": Path(args.metrics_tls_key),
            "client_ca_path": Path(args.metrics_tls_client_ca) if args.metrics_tls_client_ca else None,
            "require_client_auth": bool(args.metrics_tls_require_client_cert),
        }

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
    attempts.append(kw)

    if "tls_config" in kw:
        attempts.append({k: v for k, v in base_kwargs.items() if True} | ({"auth_token": args.metrics_auth_token} if args.metrics_auth_token else {}))
    if "auth_token" in kw:
        attempts.append({k: v for k, v in base_kwargs.items() if True} | ({"tls_config": tls_config} if tls_config is not None else {}))
    attempts.append(dict(base_kwargs))

    last_exc: Exception | None = None
    for opt in attempts:
        try:
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
    ui_alerts_info: Mapping[str, object] = {
        "configured": not bool(args.disable_metrics_ui_alerts),
        "available": ui_alert_deps,
        "expected_active": bool(args.enable_metrics and not args.disable_metrics_ui_alerts and metrics_available and ui_alert_deps),
        "path": str(ui_alert_path),
        "metadata": file_reference_metadata(ui_alert_path, role="ui_alerts_jsonl"),
        "source": "cli" if args.metrics_ui_alerts_jsonl else "default",
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
