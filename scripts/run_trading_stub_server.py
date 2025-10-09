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

LOGGER = logging.getLogger("run_trading_stub_server")


try:  # pragma: no cover - opcjonalne zależności gRPC/alertów
    from bot_core.runtime import JsonlSink, create_metrics_server
except Exception:  # pragma: no cover - brak generowanych stubów lub grpcio
    JsonlSink = None  # type: ignore
    create_metrics_server = None  # type: ignore

try:  # pragma: no cover - sink alertów UI może nie być dostępny
    from bot_core.runtime.metrics_alerts import (  # type: ignore
        DEFAULT_UI_ALERTS_JSONL_PATH,
        UiTelemetryAlertSink,
    )
except Exception:  # pragma: no cover - brak modułu telemetrii UI
    UiTelemetryAlertSink = None  # type: ignore
    DEFAULT_UI_ALERTS_JSONL_PATH = Path("logs/ui_telemetry_alerts.jsonl")

try:  # pragma: no cover - brak pełnej infrastruktury alertów w starszych gałęziach
    from bot_core.alerts import DefaultAlertRouter
    from bot_core.alerts.audit import InMemoryAlertAuditLog
except Exception:  # pragma: no cover - brak routera alertów
    DefaultAlertRouter = None  # type: ignore
    InMemoryAlertAuditLog = None  # type: ignore


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
    except ValueError as exc:  # noqa: F841 - tylko dla komunikatu błędu
        parser.error(f"Zmienna {variable} musi być liczbą całkowitą – otrzymano '{value}'.")
    raise AssertionError("parser.error powinno zakończyć działanie")


def _parse_env_float(value: str, *, variable: str, parser: argparse.ArgumentParser) -> float:
    try:
        return float(value)
    except ValueError as exc:  # noqa: F841 - jedynie do komunikatu błędu
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
        "--print-runtime-plan": "print_runtime_plan",
        "--runtime-plan-jsonl": "runtime_plan_jsonl_path",
        "--fail-on-security-warnings": "fail_on_security_warnings",
    }

    sources: dict[str, str] = {}
    for flag, key in mapping.items():
        if flag in provided:
            sources[key] = "cli"
    return sources


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
            {
                "option": option,
                "variable": env_var,
                "raw_value": raw_value,
                "applied": False,
                "reason": "cli_override",
            }
        )

    def normalize_value(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            normalized_list: list[Any] = []
            for item in value:
                if isinstance(item, Path):
                    normalized_list.append(str(item))
                else:
                    normalized_list.append(item)
            return normalized_list
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

    raw = os.getenv("RUN_TRADING_STUB_HOST")
    if raw is not None:
        if env_present("--host"):
            skip_due_to_cli("host", "RUN_TRADING_STUB_HOST", raw)
        else:
            args.host = raw
            apply_value("host", "RUN_TRADING_STUB_HOST", raw, raw)

    raw = os.getenv("RUN_TRADING_STUB_PORT")
    if raw is not None:
        if env_present("--port"):
            skip_due_to_cli("port", "RUN_TRADING_STUB_PORT", raw)
        else:
            port_value = _parse_env_int(raw, variable="RUN_TRADING_STUB_PORT", parser=parser)
            args.port = port_value
            apply_value("port", "RUN_TRADING_STUB_PORT", raw, port_value)

    raw = os.getenv("RUN_TRADING_STUB_DATASETS")
    if raw is not None:
        if env_present("--dataset"):
            skip_due_to_cli("dataset_paths", "RUN_TRADING_STUB_DATASETS", raw)
        else:
            parts = [segment.strip() for segment in raw.split(os.pathsep) if segment.strip()]
            dataset_paths = [Path(part).expanduser() for part in parts]
            args.dataset = dataset_paths
            apply_value(
                "dataset_paths",
                "RUN_TRADING_STUB_DATASETS",
                raw,
                [str(path) for path in dataset_paths],
            )

    raw = os.getenv("RUN_TRADING_STUB_NO_DEFAULT_DATASET")
    if raw is not None:
        if env_present("--no-default-dataset"):
            skip_due_to_cli("include_default_dataset", "RUN_TRADING_STUB_NO_DEFAULT_DATASET", raw)
        else:
            disabled = _parse_env_bool(
                raw,
                variable="RUN_TRADING_STUB_NO_DEFAULT_DATASET",
                parser=parser,
            )
            args.no_default_dataset = disabled
            apply_value(
                "include_default_dataset",
                "RUN_TRADING_STUB_NO_DEFAULT_DATASET",
                raw,
                not disabled,
            )

    raw = os.getenv("RUN_TRADING_STUB_SHUTDOWN_AFTER")
    if raw is not None:
        if env_present("--shutdown-after"):
            skip_due_to_cli("shutdown_after", "RUN_TRADING_STUB_SHUTDOWN_AFTER", raw)
        else:
            value = _parse_env_float(raw, variable="RUN_TRADING_STUB_SHUTDOWN_AFTER", parser=parser)
            args.shutdown_after = value
            apply_value("shutdown_after", "RUN_TRADING_STUB_SHUTDOWN_AFTER", raw, value)

    raw = os.getenv("RUN_TRADING_STUB_MAX_WORKERS")
    if raw is not None:
        if env_present("--max-workers"):
            skip_due_to_cli("max_workers", "RUN_TRADING_STUB_MAX_WORKERS", raw)
        else:
            value = _parse_env_int(raw, variable="RUN_TRADING_STUB_MAX_WORKERS", parser=parser)
            args.max_workers = value
            apply_value("max_workers", "RUN_TRADING_STUB_MAX_WORKERS", raw, value)

    raw = os.getenv("RUN_TRADING_STUB_STREAM_REPEAT")
    if raw is not None:
        if env_present("--stream-repeat"):
            skip_due_to_cli("stream_repeat", "RUN_TRADING_STUB_STREAM_REPEAT", raw)
        else:
            value = _parse_env_bool(raw, variable="RUN_TRADING_STUB_STREAM_REPEAT", parser=parser)
            args.stream_repeat = value
            apply_value("stream_repeat", "RUN_TRADING_STUB_STREAM_REPEAT", raw, value)

    raw = os.getenv("RUN_TRADING_STUB_STREAM_INTERVAL")
    if raw is not None:
        if env_present("--stream-interval"):
            skip_due_to_cli("stream_interval", "RUN_TRADING_STUB_STREAM_INTERVAL", raw)
        else:
            value = _parse_env_float(raw, variable="RUN_TRADING_STUB_STREAM_INTERVAL", parser=parser)
            args.stream_interval = value
            apply_value("stream_interval", "RUN_TRADING_STUB_STREAM_INTERVAL", raw, value)

    raw = os.getenv("RUN_TRADING_STUB_LOG_LEVEL")
    if raw is not None:
        if env_present("--log-level"):
            skip_due_to_cli("log_level", "RUN_TRADING_STUB_LOG_LEVEL", raw)
        else:
            args.log_level = raw
            apply_value("log_level", "RUN_TRADING_STUB_LOG_LEVEL", raw, raw)

    raw = os.getenv("RUN_TRADING_STUB_PRINT_ADDRESS")
    if raw is not None:
        if env_present("--print-address"):
            skip_due_to_cli("print_address", "RUN_TRADING_STUB_PRINT_ADDRESS", raw)
        else:
            value = _parse_env_bool(raw, variable="RUN_TRADING_STUB_PRINT_ADDRESS", parser=parser)
            args.print_address = value
            apply_value("print_address", "RUN_TRADING_STUB_PRINT_ADDRESS", raw, value)

    raw = os.getenv("RUN_TRADING_STUB_ENABLE_METRICS")
    if raw is not None:
        if env_present("--enable-metrics"):
            skip_due_to_cli("enable_metrics", "RUN_TRADING_STUB_ENABLE_METRICS", raw)
        else:
            value = _parse_env_bool(raw, variable="RUN_TRADING_STUB_ENABLE_METRICS", parser=parser)
            args.enable_metrics = value
            apply_value("enable_metrics", "RUN_TRADING_STUB_ENABLE_METRICS", raw, value)

    raw = os.getenv("RUN_TRADING_STUB_METRICS_HOST")
    if raw is not None:
        if env_present("--metrics-host"):
            skip_due_to_cli("metrics_host", "RUN_TRADING_STUB_METRICS_HOST", raw)
        else:
            args.metrics_host = raw
            apply_value("metrics_host", "RUN_TRADING_STUB_METRICS_HOST", raw, raw)

    raw = os.getenv("RUN_TRADING_STUB_METRICS_PORT")
    if raw is not None:
        if env_present("--metrics-port"):
            skip_due_to_cli("metrics_port", "RUN_TRADING_STUB_METRICS_PORT", raw)
        else:
            value = _parse_env_int(raw, variable="RUN_TRADING_STUB_METRICS_PORT", parser=parser)
            args.metrics_port = value
            apply_value("metrics_port", "RUN_TRADING_STUB_METRICS_PORT", raw, value)

    raw = os.getenv("RUN_TRADING_STUB_METRICS_HISTORY_SIZE")
    if raw is not None:
        if env_present("--metrics-history-size"):
            skip_due_to_cli("metrics_history_size", "RUN_TRADING_STUB_METRICS_HISTORY_SIZE", raw)
        else:
            value = _parse_env_int(
                raw,
                variable="RUN_TRADING_STUB_METRICS_HISTORY_SIZE",
                parser=parser,
            )
            args.metrics_history_size = value
            apply_value("metrics_history_size", "RUN_TRADING_STUB_METRICS_HISTORY_SIZE", raw, value)

    raw = os.getenv("RUN_TRADING_STUB_METRICS_JSONL")
    if raw is not None:
        if env_present("--metrics-jsonl"):
            skip_due_to_cli("metrics_jsonl_path", "RUN_TRADING_STUB_METRICS_JSONL", raw)
        else:
            path = Path(raw).expanduser()
            args.metrics_jsonl = path
            apply_value("metrics_jsonl_path", "RUN_TRADING_STUB_METRICS_JSONL", raw, str(path))

    raw = os.getenv("RUN_TRADING_STUB_METRICS_JSONL_FSYNC")
    if raw is not None:
        if env_present("--metrics-jsonl-fsync"):
            skip_due_to_cli("metrics_jsonl_fsync", "RUN_TRADING_STUB_METRICS_JSONL_FSYNC", raw)
        else:
            value = _parse_env_bool(
                raw,
                variable="RUN_TRADING_STUB_METRICS_JSONL_FSYNC",
                parser=parser,
            )
            args.metrics_jsonl_fsync = value
            apply_value("metrics_jsonl_fsync", "RUN_TRADING_STUB_METRICS_JSONL_FSYNC", raw, value)

    raw = os.getenv("RUN_TRADING_STUB_METRICS_DISABLE_LOG_SINK")
    if raw is not None:
        if env_present("--metrics-disable-log-sink"):
            skip_due_to_cli(
                "metrics_disable_log_sink",
                "RUN_TRADING_STUB_METRICS_DISABLE_LOG_SINK",
                raw,
            )
        else:
            value = _parse_env_bool(
                raw,
                variable="RUN_TRADING_STUB_METRICS_DISABLE_LOG_SINK",
                parser=parser,
            )
            args.metrics_disable_log_sink = value
            apply_value("metrics_disable_log_sink", "RUN_TRADING_STUB_METRICS_DISABLE_LOG_SINK", raw, value)

    raw = os.getenv("RUN_TRADING_STUB_METRICS_TLS_CERT")
    if raw is not None:
        if env_present("--metrics-tls-cert"):
            skip_due_to_cli("metrics_tls_cert", "RUN_TRADING_STUB_METRICS_TLS_CERT", raw)
        else:
            path = Path(raw).expanduser()
            args.metrics_tls_cert = path
            apply_value("metrics_tls_cert", "RUN_TRADING_STUB_METRICS_TLS_CERT", raw, str(path))

    raw = os.getenv("RUN_TRADING_STUB_METRICS_TLS_KEY")
    if raw is not None:
        if env_present("--metrics-tls-key"):
            skip_due_to_cli("metrics_tls_key", "RUN_TRADING_STUB_METRICS_TLS_KEY", raw)
        else:
            path = Path(raw).expanduser()
            args.metrics_tls_key = path
            apply_value("metrics_tls_key", "RUN_TRADING_STUB_METRICS_TLS_KEY", raw, str(path))

    raw = os.getenv("RUN_TRADING_STUB_METRICS_TLS_CLIENT_CA")
    if raw is not None:
        if env_present("--metrics-tls-client-ca"):
            skip_due_to_cli(
                "metrics_tls_client_ca",
                "RUN_TRADING_STUB_METRICS_TLS_CLIENT_CA",
                raw,
            )
        else:
            path = Path(raw).expanduser()
            args.metrics_tls_client_ca = path
            apply_value("metrics_tls_client_ca", "RUN_TRADING_STUB_METRICS_TLS_CLIENT_CA", raw, str(path))

    raw = os.getenv("RUN_TRADING_STUB_METRICS_TLS_REQUIRE_CLIENT_CERT")
    if raw is not None:
        if env_present("--metrics-tls-require-client-cert"):
            skip_due_to_cli(
                "metrics_tls_require_client_cert",
                "RUN_TRADING_STUB_METRICS_TLS_REQUIRE_CLIENT_CERT",
                raw,
            )
        else:
            value = _parse_env_bool(
                raw,
                variable="RUN_TRADING_STUB_METRICS_TLS_REQUIRE_CLIENT_CERT",
                parser=parser,
            )
            args.metrics_tls_require_client_cert = value
            apply_value(
                "metrics_tls_require_client_cert",
                "RUN_TRADING_STUB_METRICS_TLS_REQUIRE_CLIENT_CERT",
                raw,
                value,
            )

    raw = os.getenv("RUN_TRADING_STUB_METRICS_UI_ALERTS_JSONL")
    if raw is not None:
        if env_present("--metrics-ui-alerts-jsonl"):
            skip_due_to_cli(
                "metrics_ui_alerts_jsonl_path",
                "RUN_TRADING_STUB_METRICS_UI_ALERTS_JSONL",
                raw,
            )
        else:
            path = Path(raw).expanduser()
            args.metrics_ui_alerts_jsonl = path
            apply_value(
                "metrics_ui_alerts_jsonl_path",
                "RUN_TRADING_STUB_METRICS_UI_ALERTS_JSONL",
                raw,
                str(path),
            )

    raw = os.getenv("RUN_TRADING_STUB_DISABLE_METRICS_UI_ALERTS")
    if raw is not None:
        if env_present("--disable-metrics-ui-alerts"):
            skip_due_to_cli(
                "disable_metrics_ui_alerts",
                "RUN_TRADING_STUB_DISABLE_METRICS_UI_ALERTS",
                raw,
            )
        else:
            value = _parse_env_bool(
                raw,
                variable="RUN_TRADING_STUB_DISABLE_METRICS_UI_ALERTS",
                parser=parser,
            )
            args.disable_metrics_ui_alerts = value
            apply_value(
                "disable_metrics_ui_alerts",
                "RUN_TRADING_STUB_DISABLE_METRICS_UI_ALERTS",
                raw,
                value,
            )

    raw = os.getenv("RUN_TRADING_STUB_METRICS_PRINT_ADDRESS")
    if raw is not None:
        if env_present("--metrics-print-address"):
            skip_due_to_cli(
                "metrics_print_address",
                "RUN_TRADING_STUB_METRICS_PRINT_ADDRESS",
                raw,
            )
        else:
            value = _parse_env_bool(
                raw,
                variable="RUN_TRADING_STUB_METRICS_PRINT_ADDRESS",
                parser=parser,
            )
            args.metrics_print_address = value
            apply_value(
                "metrics_print_address",
                "RUN_TRADING_STUB_METRICS_PRINT_ADDRESS",
                raw,
                value,
            )

    raw = os.getenv("RUN_TRADING_STUB_PRINT_RUNTIME_PLAN")
    if raw is not None:
        if env_present("--print-runtime-plan"):
            skip_due_to_cli("print_runtime_plan", "RUN_TRADING_STUB_PRINT_RUNTIME_PLAN", raw)
        else:
            value = _parse_env_bool(
                raw,
                variable="RUN_TRADING_STUB_PRINT_RUNTIME_PLAN",
                parser=parser,
            )
            args.print_runtime_plan = value
            apply_value("print_runtime_plan", "RUN_TRADING_STUB_PRINT_RUNTIME_PLAN", raw, value)

    raw = os.getenv("RUN_TRADING_STUB_RUNTIME_PLAN_JSONL")
    if raw is not None:
        if env_present("--runtime-plan-jsonl"):
            skip_due_to_cli("runtime_plan_jsonl_path", "RUN_TRADING_STUB_RUNTIME_PLAN_JSONL", raw)
        else:
            path = Path(raw).expanduser()
            args.runtime_plan_jsonl = path
            apply_value(
                "runtime_plan_jsonl_path",
                "RUN_TRADING_STUB_RUNTIME_PLAN_JSONL",
                raw,
                str(path),
            )

    raw = os.getenv("RUN_TRADING_STUB_FAIL_ON_SECURITY_WARNINGS")
    if raw is not None:
        if env_present("--fail-on-security-warnings"):
            skip_due_to_cli(
                "fail_on_security_warnings",
                "RUN_TRADING_STUB_FAIL_ON_SECURITY_WARNINGS",
                raw,
            )
        else:
            value = _parse_env_bool(
                raw,
                variable="RUN_TRADING_STUB_FAIL_ON_SECURITY_WARNINGS",
                parser=parser,
            )
            args.fail_on_security_warnings = value
            apply_value(
                "fail_on_security_warnings",
                "RUN_TRADING_STUB_FAIL_ON_SECURITY_WARNINGS",
                raw,
                value,
            )

    return entries, override_keys

def _load_dataset(
    dataset_paths: Iterable[Path],
    include_default: bool,
) -> InMemoryTradingDataset:
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

    path = (
        Path(args.metrics_ui_alerts_jsonl)
        if args.metrics_ui_alerts_jsonl
        else DEFAULT_UI_ALERTS_JSONL_PATH.expanduser()
    )
    try:
        router = DefaultAlertRouter(audit_log=InMemoryAlertAuditLog())
        sink = UiTelemetryAlertSink(router, jsonl_path=path)
    except Exception:  # pragma: no cover - defensywnie logujemy błędy inicjalizacji
        LOGGER.exception("Nie udało się zainicjalizować UiTelemetryAlertSink – kontynuuję bez alertów UI")
        return None
    LOGGER.info("Sink alertów telemetrii UI aktywny (log: %s)", path)
    return sink


def _start_metrics_service(args):
    if not args.enable_metrics:
        return None
    if create_metrics_server is None:
        LOGGER.warning("create_metrics_server nie jest dostępne – pomijam serwer telemetrii")
        return None

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
            "client_ca_path": Path(args.metrics_tls_client_ca)
            if args.metrics_tls_client_ca
            else None,
            "require_client_auth": bool(args.metrics_tls_require_client_cert),
        }

    try:
        server = create_metrics_server(
            host=args.metrics_host,
            port=args.metrics_port,
            history_size=args.metrics_history_size,
            enable_logging_sink=not args.metrics_disable_log_sink,
            sinks=sinks or None,
            tls_config=tls_config,
        )
    except Exception:  # pragma: no cover - logowanie błędów startu
        LOGGER.exception("Nie udało się utworzyć serwera MetricsService")
        return None

    server.start()
    LOGGER.info("Serwer MetricsService uruchomiony na %s", getattr(server, "address", "<unknown>"))
    if args.metrics_print_address and hasattr(server, "address"):
        print(server.address)
    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Startuje stub tradingowy gRPC dla środowiska developerskiego UI.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Adres hosta, na którym ma nasłuchiwać serwer (domyślnie 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Port nasłuchu (0 = wybierz losowy wolny port).",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=Path,
        default=None,
        help="Ścieżka do pliku YAML z danymi stubu (można podać wielokrotnie).",
    )
    parser.add_argument(
        "--no-default-dataset",
        action="store_true",
        help="Nie ładuj domyślnego datasetu – użyj tylko danych z plików YAML.",
    )
    parser.add_argument(
        "--shutdown-after",
        type=float,
        default=None,
        help="Automatycznie zatrzymaj serwer po tylu sekundach (przydatne w CI).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Liczba wątków w puli gRPC (domyślnie 8).",
    )
    parser.add_argument(
        "--stream-repeat",
        action="store_true",
        help="Powtarzaj w pętli strumień incrementów (symulacja ciągłego feedu).",
    )
    parser.add_argument(
        "--stream-interval",
        type=float,
        default=0.0,
        help="Odstęp w sekundach pomiędzy kolejnymi incrementami podczas pętli.",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        help="Poziom logowania (debug, info, warning, error).",
    )
    parser.add_argument(
        "--print-address",
        action="store_true",
        help="Wypisz sam adres serwera na stdout (np. do użycia w skryptach).",
    )
    parser.add_argument(
        "--enable-metrics",
        action="store_true",
        help="Uruchom pomocniczy serwer MetricsService do telemetrii UI.",
    )

    metrics_group = parser.add_argument_group("metrics", "Opcje serwera MetricsService")
    metrics_group.add_argument(
        "--metrics-host",
        default="127.0.0.1",
        help="Adres hosta dla serwera MetricsService (domyślnie 127.0.0.1).",
    )
    metrics_group.add_argument(
        "--metrics-port",
        type=int,
        default=50061,
        help="Port MetricsService (0 = wybierz losowy).",
    )
    metrics_group.add_argument(
        "--metrics-history-size",
        type=int,
        default=512,
        help="Rozmiar historii snapshotów MetricsService (domyślnie 512).",
    )
    metrics_group.add_argument(
        "--metrics-jsonl",
        type=Path,
        default=None,
        help="Plik JSONL na snapshoty metryk (opcjonalnie).",
    )
    metrics_group.add_argument(
        "--metrics-jsonl-fsync",
        action="store_true",
        help="Wymuś fsync po każdym wpisie JSONL z metrykami.",
    )
    metrics_group.add_argument(
        "--metrics-disable-log-sink",
        action="store_true",
        help="Wyłącz domyślny LoggingSink w MetricsService.",
    )
    metrics_group.add_argument(
        "--metrics-tls-cert",
        type=Path,
        default=None,
        help="Ścieżka certyfikatu TLS serwera MetricsService (PEM).",
    )
    metrics_group.add_argument(
        "--metrics-tls-key",
        type=Path,
        default=None,
        help="Ścieżka klucza prywatnego TLS serwera MetricsService (PEM).",
    )
    metrics_group.add_argument(
        "--metrics-tls-client-ca",
        type=Path,
        default=None,
        help="Opcjonalny plik CA klientów dla mTLS (PEM).",
    )
    metrics_group.add_argument(
        "--metrics-tls-require-client-cert",
        action="store_true",
        help="Wymagaj certyfikatu klienta przy połączeniach TLS (mTLS).",
    )
    metrics_group.add_argument(
        "--metrics-ui-alerts-jsonl",
        type=Path,
        default=None,
        help="Ścieżka JSONL na alerty telemetrii UI (domyślnie logs/ui_telemetry_alerts.jsonl).",
    )
    metrics_group.add_argument(
        "--disable-metrics-ui-alerts",
        action="store_true",
        help="Nie rejestruj sinka alertów telemetrii UI.",
    )
    metrics_group.add_argument(
        "--metrics-print-address",
        action="store_true",
        help="Wypisz adres uruchomionego MetricsService na stdout.",
    )

    audit_group = parser.add_argument_group("audit", "Opcje audytu runtime")
    audit_group.add_argument(
        "--print-runtime-plan",
        action="store_true",
        help="Wypisz plan konfiguracji stubu (datasety, telemetria, ścieżki JSONL/TLS) i zakończ działanie.",
    )
    audit_group.add_argument(
        "--runtime-plan-jsonl",
        type=Path,
        default=None,
        help="Ścieżka JSONL, do której należy dopisać snapshot planu runtime przed startem serwera.",
    )
    audit_group.add_argument(
        "--fail-on-security-warnings",
        action="store_true",
        help="Zakończ działanie, jeśli plan runtime zawiera ostrzeżenia bezpieczeństwa plików JSONL/TLS.",
    )
    return parser


def _install_signal_handlers(stop_callback) -> None:
    def handler(signum, _frame) -> None:  # pragma: no cover - reakcja na sygnał
        LOGGER.info("Otrzymano sygnał %s – trwa zatrzymywanie serwera", signum)
        stop_callback()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except ValueError:  # pragma: no cover - np. uruchomienie w wątku
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


def _build_dataset_plan(
    dataset: InMemoryTradingDataset,
    dataset_paths: Iterable[Path],
    include_default: bool,
) -> Mapping[str, object]:
    sources: list[Mapping[str, object]] = []
    if include_default:
        sources.append(
            {
                "type": "default",
                "description": "build_default_dataset",
            }
        )

    for raw_path in dataset_paths:
        path = Path(str(raw_path)).expanduser()
        sources.append(
            {
                "type": "file",
                "path": str(path),
                "metadata": file_reference_metadata(path, role="stub_dataset"),
            }
        )

    return {
        "include_default": include_default,
        "sources": sources,
        "summary": _dataset_summary(dataset),
    }


def _build_metrics_plan(args) -> Mapping[str, object]:
    metrics_available = create_metrics_server is not None
    ui_alert_dependencies_available = (
        UiTelemetryAlertSink is not None
        and DefaultAlertRouter is not None
        and InMemoryAlertAuditLog is not None
    )

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
    tls_info: Mapping[str, object] | None = None
    if tls_configured:
        cert_path = Path(str(args.metrics_tls_cert)).expanduser()
        key_path = Path(str(args.metrics_tls_key)).expanduser()
        client_ca_path = (
            Path(str(args.metrics_tls_client_ca)).expanduser()
            if args.metrics_tls_client_ca
            else None
        )
        tls_info = {
            "configured": True,
            "require_client_auth": bool(args.metrics_tls_require_client_cert),
            "certificate": file_reference_metadata(cert_path, role="tls_cert"),
            "private_key": file_reference_metadata(key_path, role="tls_key"),
        }
        if client_ca_path is not None:
            tls_info["client_ca"] = file_reference_metadata(client_ca_path, role="tls_client_ca")
    else:
        tls_info = {
            "configured": False,
            "require_client_auth": bool(args.metrics_tls_require_client_cert),
        }

    ui_alert_path = (
        Path(str(args.metrics_ui_alerts_jsonl)).expanduser()
        if args.metrics_ui_alerts_jsonl
        else DEFAULT_UI_ALERTS_JSONL_PATH.expanduser()
    )
    ui_alerts_info: Mapping[str, object] = {
        "configured": not bool(args.disable_metrics_ui_alerts),
        "available": ui_alert_dependencies_available,
        "expected_active": bool(
            args.enable_metrics
            and not args.disable_metrics_ui_alerts
            and metrics_available
            and ui_alert_dependencies_available
        ),
        "path": str(ui_alert_path),
        "metadata": file_reference_metadata(ui_alert_path, role="ui_alerts_jsonl"),
        "source": "cli" if args.metrics_ui_alerts_jsonl else "default",
    }

    warnings: list[str] = []
    if args.enable_metrics and not metrics_available:
        warnings.append("create_metrics_server nie jest dostępne – telemetria nie uruchomi się.")
    if (
        args.enable_metrics
        and not args.disable_metrics_ui_alerts
        and not ui_alert_dependencies_available
    ):
        warnings.append("UiTelemetryAlertSink lub router alertów nie są dostępne – alerty UI będą wyłączone.")

    return {
        "available": metrics_available,
        "enabled": bool(args.enable_metrics),
        "host": args.metrics_host,
        "port": args.metrics_port,
        "history_size": args.metrics_history_size,
        "log_sink_enabled": not bool(args.metrics_disable_log_sink),
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
    plan["environment"] = {
        "overrides": overrides_list,
        "parameter_sources": dict(parameter_sources),
    }

    fail_parameter_source = parameter_sources.get("fail_on_security_warnings", "default")
    fail_env_entry = None
    for entry in overrides_list:
        if entry.get("option") == "fail_on_security_warnings":
            fail_env_entry = entry
            break

    env_variable = "RUN_TRADING_STUB_FAIL_ON_SECURITY_WARNINGS"
    if fail_parameter_source == "cli":
        fail_source = "cli"
    elif fail_parameter_source.startswith("env"):
        fail_source = f"env:{env_variable}"
    else:
        fail_source = fail_parameter_source

    security_payload = {
        "fail_on_security_warnings": {
            key: value
            for key, value in {
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
            }.items()
            if value is not None
        },
        "parameter_sources": {
            "fail_on_security_warnings": fail_parameter_source,
        },
    }

    plan["security"] = security_payload

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
        args,
        parser=parser,
        provided_flags=provided_flags,
        value_sources=value_sources,
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

    runtime_plan_payload: Mapping[str, object] | None = None
    need_runtime_plan = bool(
        args.print_runtime_plan or args.runtime_plan_jsonl or args.fail_on_security_warnings
    )
    if need_runtime_plan:
        try:
            runtime_plan_payload = _build_runtime_plan_payload(
                args=args,
                dataset=dataset,
                dataset_paths=dataset_paths,
                environment_overrides=environment_overrides,
                parameter_sources=parameter_sources,
            )
        except Exception:  # noqa: BLE001
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
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Nie udało się zapisać planu runtime do %s: %s", args.runtime_plan_jsonl, exc)
            return 2
        else:
            LOGGER.info("Plan runtime zapisany do %s", destination)

    if args.print_runtime_plan and runtime_plan_payload is not None:
        json.dump(runtime_plan_payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        if args.fail_on_security_warnings and security_warnings_detected:
            return 3
        return 0

    if args.fail_on_security_warnings and runtime_plan_payload is not None and security_warnings_detected:
        return 3

    if dataset.performance_guard:
        guard_preview = ", ".join(
            f"{key}={value}" for key, value in sorted(dataset.performance_guard.items())
        )
        LOGGER.info("Konfiguracja performance guard: %s", guard_preview)

    server = TradingStubServer(
        dataset=dataset,
        host=args.host,
        port=args.port,
        max_workers=args.max_workers,
        stream_repeat=args.stream_repeat,
        stream_interval=args.stream_interval,
    )

    should_stop = False
    metrics_server = _start_metrics_service(args)
    metrics_stopped = False

    def request_stop() -> None:
        nonlocal should_stop
        nonlocal metrics_stopped
        should_stop = True
        server.stop(grace=1.0)
        if metrics_server is not None and not metrics_stopped:
            metrics_server.stop(grace=1.0)
            metrics_stopped = True

    _install_signal_handlers(request_stop)

    server.start()
    LOGGER.info("Serwer stub wystartował na %s", server.address)
    if args.print_address:
        print(server.address)

    try:
        if args.shutdown_after is not None:
            LOGGER.info(
                "Serwer zakończy pracę automatycznie po %.2f s (lub szybciej po sygnale)",
                args.shutdown_after,
            )
            terminated = server.wait_for_termination(timeout=args.shutdown_after)
            if not terminated and not should_stop:
                LOGGER.info("Limit czasu minął – zatrzymuję serwer stub i telemetrię.")
                request_stop()
        else:
            LOGGER.info("Naciśnij Ctrl+C, aby zakończyć pracę stubu.")
            server.wait_for_termination()
    except KeyboardInterrupt:  # pragma: no cover - zależy od środowiska testowego
        LOGGER.info("Przerwano przez użytkownika – zatrzymywanie serwera.")
    finally:
        if not should_stop:
            server.stop(grace=1.0)
        if metrics_server is not None and not metrics_stopped:
            metrics_server.stop(grace=1.0)
        LOGGER.info("Serwer stub został zatrzymany.")
    return 0


if __name__ == "__main__":  # pragma: no cover - ścieżka uruchomienia skryptu
    sys.exit(main())
