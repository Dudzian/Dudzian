"""Uruchamia lokalny backend bota (AutoTrader + serwer gRPC) na potrzeby UI."""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from bot_core.api.server import build_local_runtime_context, LocalRuntimeServer
from bot_core.exchanges.base import Environment as ExchangeEnvironment
from bot_core.logging.config import install_metrics_logging_handler
from bot_core.runtime.state_manager import RuntimeStateError, RuntimeStateManager
from bot_core.security.base import SecretStorageError
from core.reporting import DemoPaperReport, GuardrailReport

_LOGGER = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )


def _write_ready_payload(
    address: str,
    *,
    ready_file: Optional[str],
    emit_stdout: bool,
    metrics_url: str | None = None,
) -> None:
    payload = {"event": "ready", "address": address, "pid": os.getpid()}
    if metrics_url:
        payload["metrics_url"] = metrics_url
    serialized = json.dumps(payload, ensure_ascii=False)
    if ready_file:
        Path(ready_file).expanduser().write_text(serialized, encoding="utf-8")
    if emit_stdout:
        print(serialized, flush=True)


def _determine_entrypoint_name(config: Any, entrypoint_obj: Any, fallback: str | None) -> str:
    if fallback:
        return fallback
    entrypoints = getattr(getattr(config, "trading", None), "entrypoints", {}) or {}
    for name, candidate in entrypoints.items():
        if candidate is entrypoint_obj:
            return str(name)
    return fallback or "unknown"


def _normalize_environment(value: Any) -> ExchangeEnvironment:
    if isinstance(value, ExchangeEnvironment):
        return value
    try:
        return ExchangeEnvironment(str(value).lower())
    except Exception:
        return ExchangeEnvironment.PAPER


def _validate_runtime_context(context: Any, mode: str) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {"mode": mode}

    entrypoint_cfg = getattr(context, "entrypoint", None)
    details["environment"] = getattr(entrypoint_cfg, "environment", None)
    details["strategy"] = getattr(entrypoint_cfg, "strategy", None)
    details["risk_profile"] = getattr(entrypoint_cfg, "risk_profile", None)

    pipeline = getattr(context, "pipeline", None)
    bootstrap = getattr(pipeline, "bootstrap", None)
    runtime_environment = getattr(bootstrap, "environment", None)
    keychain_key = getattr(runtime_environment, "keychain_key", None)
    credential_purpose = getattr(runtime_environment, "credential_purpose", "trading")
    expected_env = _normalize_environment(getattr(runtime_environment, "environment", ExchangeEnvironment.PAPER))
    details["keychain_key"] = keychain_key
    details["expected_environment"] = expected_env.value

    if keychain_key:
        try:
            context.secret_manager.load_exchange_credentials(
                keychain_key,
                expected_environment=expected_env,
                purpose=str(credential_purpose or "trading"),
                required_permissions=getattr(runtime_environment, "required_permissions", None),
                forbidden_permissions=getattr(runtime_environment, "forbidden_permissions", None),
            )
            details["credentials_validated"] = True
        except SecretStorageError as exc:
            errors.append(f"Walidacja poświadczeń '{keychain_key}' nie powiodła się: {exc}")
        except Exception as exc:  # pragma: no cover - defensywne
            errors.append(f"Nie udało się zweryfikować poświadczeń '{keychain_key}': {exc}")
    else:
        warnings.append("Środowisko nie określa klucza API – pomijam walidację poświadczeń.")

    controller = getattr(pipeline, "controller", None)
    symbols = tuple(getattr(controller, "symbols", ()) or ())
    details["symbols"] = list(symbols)
    if not symbols:
        errors.append("Kontroler strategii nie udostępnia żadnych symboli handlowych.")

    return {"errors": errors, "warnings": warnings, "details": details}


def _collect_paper_exchange_metrics(context: Any) -> dict[str, Any]:
    config = getattr(context, "config", None)
    trading_cfg = getattr(config, "trading", None)
    entrypoints = getattr(trading_cfg, "entrypoints", {}) if trading_cfg is not None else {}
    if not isinstance(entrypoints, Mapping):
        return {}

    metrics_registry = getattr(context, "metrics_registry", None)
    rate_metric = None
    error_metric = None
    health_metric = None
    if metrics_registry is not None:
        try:
            rate_metric = metrics_registry.get("bot_exchange_rate_limited_total")
        except KeyError:  # pragma: no cover - rejestr bez metryki rate-limitów
            rate_metric = None
        try:
            error_metric = metrics_registry.get("bot_exchange_errors_total")
        except KeyError:  # pragma: no cover - rejestr bez metryki błędów sieciowych
            error_metric = None
        try:
            health_metric = metrics_registry.get("bot_exchange_health_status")
        except KeyError:  # pragma: no cover - opcjonalna metryka zdrowia
            health_metric = None

    summary: dict[str, Any] = {}
    severities = ("warning", "error", "critical")
    for name, entrypoint in entrypoints.items():
        environment = str(getattr(entrypoint, "environment", "") or "")
        if "paper" not in environment.lower():
            continue
        exchange = environment.split("_")[0].lower() if environment else str(name).lower()
        rate_value = 0.0
        if rate_metric is not None:
            try:
                rate_value = float(rate_metric.value(labels={"exchange": exchange}))
            except Exception:  # pragma: no cover - defensywne
                rate_value = 0.0
        error_value = 0.0
        if error_metric is not None:
            for severity in severities:
                try:
                    error_value += float(
                        error_metric.value(labels={"exchange": exchange, "severity": severity})
                    )
                except Exception:  # pragma: no cover - defensywne
                    continue
        health_value: float | None = None
        if health_metric is not None:
            try:
                health_value = float(health_metric.value(labels={"exchange": exchange}))
            except Exception:  # pragma: no cover - defensywne
                health_value = None

        summary[str(name)] = {
            "environment": environment,
            "exchange": exchange,
            "rate_limited_events": rate_value,
            "network_errors": error_value,
            "health_status": health_value,
        }

    return summary


def _write_report(report_dir: str | Path, payload: Mapping[str, Any]) -> Path:
    report_path = Path(report_dir).expanduser()
    report_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    filename = f"run_local_bot_{payload.get('mode', 'unknown')}_{timestamp}.json"
    destination = report_path / filename
    sanitized = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
    destination.write_text(json.dumps(sanitized, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/runtime.yaml", help="Ścieżka do pliku runtime.yaml")
    parser.add_argument("--entrypoint", help="Nazwa punktu wejścia z sekcji trading.entrypoints")
    parser.add_argument("--host", default="127.0.0.1", help="Adres nasłuchiwania serwera gRPC")
    parser.add_argument("--port", default=0, type=int, help="Port serwera gRPC (0 = automatyczny)")
    parser.add_argument("--log-level", default="INFO", help="Poziom logowania")
    parser.add_argument("--ready-file", help="Opcjonalny plik, do którego zostanie zapisany adres serwera")
    parser.add_argument(
        "--mode",
        choices=("demo", "paper", "live"),
        default="paper",
        help="Tryb uruchomienia runtime (demo → paper → live).",
    )
    parser.add_argument(
        "--state-dir",
        default="var/runtime",
        help="Katalog przechowujący checkpointy scenariuszy E2E.",
    )
    parser.add_argument(
        "--report-dir",
        default="logs/e2e",
        help="Katalog, w którym zostanie zapisany raport końcowy scenariusza.",
    )
    parser.add_argument(
        "--report-markdown-dir",
        default="reports/e2e",
        help="Katalog docelowy raportu Markdown generowanego na koniec scenariusza.",
    )
    parser.add_argument(
        "--no-ready-stdout",
        action="store_true",
        help="Nie wypisuj komunikatu gotowości na stdout (użyteczne przy uruchomieniach z QProcess)",
    )
    parser.add_argument(
        "--manual-confirm",
        action="store_true",
        help="Nie aktywuj automatycznie auto-tradingu (wymaga ręcznego potwierdzenia w UI)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        _LOGGER.error("Nie odnaleziono pliku konfiguracji runtime: %s", config_path)
        return 2

    state_manager = RuntimeStateManager(args.state_dir)
    report_payload: dict[str, Any] = {
        "mode": args.mode,
        "config_path": str(config_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "errors": [],
        "warnings": [],
    }

    context = None
    server: LocalRuntimeServer | None = None
    exit_code = 0
    runtime_started = False
    checkpoint = None
    decision_events: tuple[Mapping[str, Any], ...] = ()
    paper_metrics_summary: dict[str, Any] | None = None

    try:
        context = build_local_runtime_context(
            config_path=str(config_path),
            entrypoint=args.entrypoint,
        )

        entrypoint_name = _determine_entrypoint_name(context.config, context.entrypoint, args.entrypoint)
        report_payload["entrypoint"] = entrypoint_name

        validation = _validate_runtime_context(context, args.mode)
        report_payload["validation"] = validation["details"]
        report_payload["warnings"].extend(validation["warnings"])
        if validation["errors"]:
            report_payload["errors"].extend(validation["errors"])
            report_payload["status"] = "validation_failed"
            exit_code = 3

        if exit_code == 0 and args.mode in {"paper", "live"}:
            try:
                checkpoint = state_manager.require_checkpoint(
                    target_mode=args.mode,
                    entrypoint=entrypoint_name,
                )
                report_payload["checkpoint"] = checkpoint.to_dict()
            except RuntimeStateError as exc:
                message = str(exc)
                report_payload["errors"].append(message)
                report_payload["status"] = "blocked"
                exit_code = 3

        if exit_code == 0 and args.mode == "live":
            try:
                execution_settings = getattr(context.config, "execution", None)
                if execution_settings is not None:
                    setattr(execution_settings, "default_mode", "live")
            except Exception:  # pragma: no cover - defensywne
                _LOGGER.debug("Nie udało się wymusić trybu live w konfiguracji runtime", exc_info=True)

        if exit_code == 0:
            observability_cfg = getattr(context.config, "observability", None)
            enable_metrics_handler = True
            if observability_cfg is not None:
                enable_metrics_handler = getattr(observability_cfg, "enable_log_metrics", True)
            if enable_metrics_handler:
                install_metrics_logging_handler()

            context.start(auto_confirm=not args.manual_confirm)
            runtime_started = True
            server = LocalRuntimeServer(context, host=args.host, port=args.port)
            server.start()
            metrics_url = context.metrics_endpoint
            report_payload["metrics_endpoint"] = metrics_url
            _write_ready_payload(
                server.address,
                ready_file=args.ready_file,
                emit_stdout=not args.no_ready_stdout,
                metrics_url=metrics_url,
            )

            stop_event = threading.Event()

            def _handle_signal(signum, frame):  # noqa: D401, ANN001
                del signum, frame
                stop_event.set()

            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, _handle_signal)

            try:
                while not stop_event.is_set():
                    time.sleep(0.5)
            except KeyboardInterrupt:  # pragma: no cover - alternatywna obsługa sygnałów
                stop_event.set()

    except Exception as exc:  # pragma: no cover - defensywne logowanie
        if exit_code == 0:
            exit_code = 2
        report_payload["errors"].append(str(exc))
        report_payload.setdefault("status", "error")
        _LOGGER.error("Błąd podczas uruchamiania lokalnego runtime: %s", exc)
    finally:
        if context is not None:
            try:
                pipeline = getattr(context, "pipeline", None)
                bootstrap = getattr(pipeline, "bootstrap", None)
                journal = getattr(bootstrap, "decision_journal", None)
                if journal is not None and hasattr(journal, "export"):
                    exported = tuple(journal.export())
                    decision_events = exported
                    report_payload["decision_events"] = len(exported)
            except Exception:  # pragma: no cover - defensywne
                _LOGGER.debug("Nie udało się pobrać zdarzeń z decision journal", exc_info=True)

        if paper_metrics_summary is None and context is not None:
            try:
                paper_metrics_summary = _collect_paper_exchange_metrics(context)
            except Exception:  # pragma: no cover - defensywne
                _LOGGER.debug(
                    "Nie udało się zebrać metryk papierowych scenariuszy giełdowych", exc_info=True
                )
                paper_metrics_summary = {}

        if server is not None:
            try:
                server.stop(0.5)
            except Exception:  # pragma: no cover - defensywne
                _LOGGER.debug("Błąd podczas zatrzymywania serwera gRPC", exc_info=True)
        if context is not None and runtime_started:
            try:
                context.stop()
            except Exception:  # pragma: no cover - defensywne
                _LOGGER.debug("Błąd podczas zatrzymywania kontekstu runtime", exc_info=True)

        if exit_code == 0 and args.mode == "demo":
            try:
                checkpoint = state_manager.record_checkpoint(
                    entrypoint=report_payload.get("entrypoint", args.entrypoint or ""),
                    mode="demo",
                    config_path=str(config_path),
                    metadata={
                        "metrics_endpoint": report_payload.get("metrics_endpoint"),
                    },
                )
            except Exception:  # pragma: no cover - defensywne
                _LOGGER.debug("Nie udało się zapisać checkpointu po scenariuszu demo", exc_info=True)
        if checkpoint is not None:
            report_payload["checkpoint"] = checkpoint.to_dict()

        report_payload.setdefault("status", "success" if exit_code == 0 else "error")
        report_payload["finished_at"] = datetime.now(timezone.utc).isoformat()
        guardrail_payload: dict[str, Any] | None = None
        try:
            demo_report = DemoPaperReport.from_payload(report_payload, decision_events=decision_events)
            markdown_path = demo_report.write_markdown(args.report_markdown_dir)
            report_payload["report_markdown"] = str(markdown_path)
        except Exception:  # pragma: no cover - defensywne
            _LOGGER.debug("Nie udało się wygenerować raportu Markdown", exc_info=True)
        try:
            registry = getattr(context, "metrics_registry", None) if context is not None else None
            io_queue_cfg = getattr(getattr(context, "config", None), "io_queue", None) if context is not None else None
            log_directory = getattr(io_queue_cfg, "log_directory", None) if io_queue_cfg is not None else None
            if not log_directory:
                log_directory = "logs/guardrails"
            validation_details = report_payload.get("validation", {}) if isinstance(report_payload.get("validation"), Mapping) else {}
            environment_hint = (
                str(validation_details.get("expected_environment") or validation_details.get("environment") or "")
            )
            guardrail_report = GuardrailReport.from_sources(
                registry=registry,
                log_directory=log_directory,
                environment_hint=environment_hint or None,
            )
            guardrail_dir = Path("reports/guardrails")
            guardrail_path = guardrail_report.write_markdown(guardrail_dir)
            guardrail_payload = {
                "report_markdown": str(guardrail_path),
                "summary": guardrail_report.to_dict(),
                "log_directory": str(Path(log_directory).expanduser()),
            }
        except Exception:  # pragma: no cover - defensywne
            _LOGGER.debug("Nie udało się wygenerować raportu guardrail", exc_info=True)
        if guardrail_payload:
            report_payload["guardrails"] = guardrail_payload
        if paper_metrics_summary is not None:
            report_payload["paper_exchange_metrics"] = paper_metrics_summary
        try:
            report_json = _write_report(args.report_dir, report_payload)
            report_payload["report_json"] = str(report_json)
        except Exception:  # pragma: no cover - defensywne
            _LOGGER.debug("Nie udało się zapisać raportu E2E", exc_info=True)

    return exit_code


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    sys.exit(main())
