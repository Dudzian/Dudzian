"""Uruchamia lokalny serwer MetricsService dla telemetrii powłoki Qt/QML."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Iterable

try:  # pragma: no cover - opcjonalne zależności gRPC
    from bot_core.runtime import JsonlSink, create_metrics_server
except Exception as exc:  # pragma: no cover - komunikat dla developera
    raise SystemExit(
        "Brak wsparcia gRPC. Upewnij się, że wygenerowałeś stuby (scripts/generate_trading_stubs.py) "
        "i masz zainstalowane pakiety grpcio." 
    ) from exc


LOGGER = logging.getLogger("run_metrics_service")


def _configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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
    return parser


def _build_server(
    *,
    host: str,
    port: int,
    history_size: int,
    enable_logging_sink: bool,
    jsonl_path: Path | None,
    jsonl_fsync: bool,
    extra_sinks: Iterable = (),
):
    sinks = list(extra_sinks)
    if jsonl_path is not None:
        sinks.append(JsonlSink(jsonl_path, fsync=jsonl_fsync))
    return create_metrics_server(
        host=host,
        port=port,
        history_size=history_size,
        enable_logging_sink=enable_logging_sink,
        sinks=sinks,
    )


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
    parser = build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)

    if args.history_size <= 0:
        parser.error("--history-size musi być dodatnie")

    server = _build_server(
        host=args.host,
        port=args.port,
        history_size=args.history_size,
        enable_logging_sink=not args.no_log_sink,
        jsonl_path=Path(args.jsonl) if args.jsonl else None,
        jsonl_fsync=args.jsonl_fsync,
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

