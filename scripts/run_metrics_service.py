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

LOGGER = logging.getLogger("run_metrics_service")


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
        "fail_on_security_warnings",
        "tls_cert",
        "tls_key",
        "tls_client_ca",
        "tls_require_client_cert",
        "ui_alerts_jsonl_path",
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
    extra_sinks: Iterable = (),
    tls_config=None,
):
    if not METRICS_RUNTIME_AVAILABLE:
        raise RuntimeError(METRICS_RUNTIME_UNAVAILABLE_MESSAGE)

    sinks = list(extra_sinks)
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
    attempts.append(kw)

    # fallbacki
    if "tls_config" in kw:
        k2 = dict(base_kwargs)
        if auth_token is not None:
            k2["auth_token"] = auth_token
        attempts.append(k2)
    if "auth_token" in kw:
        k3 = dict(base_kwargs)
        if tls_config is not None:
            k3["tls_config"] = tls_config
        attempts.append(k3)
    attempts.append(dict(base_kwargs))

    last_exc: Exception | None = None
    for k in attempts:
        try:
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

    payload["runtime_state"] = runtime_state

    if core_config_section:
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
            if metrics_config is not None and not bool(getattr(metrics_config, "enabled", True)):
                LOGGER.warning(
                    "runtime.metrics_service.enabled = false w %s – kontynuuję uruchamianie zgodnie z flagami.",
                    args.core_config,
                )

    value_sources = _finalize_value_sources(value_sources)

    if args.history_size <= 0:
        parser.error("--history-size musi być dodatnie")
    if bool(args.tls_cert) ^ bool(args.tls_key):
        parser.error("TLS wymaga jednoczesnego podania --tls-cert oraz --tls-key")
    for option in ("tls_cert", "tls_key", "tls_client_ca"):
        path = getattr(args, option)
        if path and not Path(path).exists():
            parser.error(f"Ścieżka {option.replace('_', '-')} nie istnieje: {path}")

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

    server = _build_server(
        host=args.host,
        port=args.port,
        history_size=args.history_size,
        enable_logging_sink=not args.no_log_sink,
        jsonl_path=Path(args.jsonl) if args.jsonl else None,
        jsonl_fsync=args.jsonl_fsync,
        auth_token=args.auth_token,
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
            if args.fail_on_security_warnings && security_warnings_detected:
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
