"""Uruchamia lokalny stub tradingowy gRPC dla powłoki Qt/QML."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Iterable

from bot_core.testing import (
    InMemoryTradingDataset,
    TradingStubServer,
    build_default_dataset,
    load_dataset_from_yaml,
    merge_datasets,
)

LOGGER = logging.getLogger("run_trading_stub_server")


def _configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)

    if args.stream_interval < 0:
        parser.error("--stream-interval musi być liczbą nieujemną")

    dataset_paths = args.dataset or []
    dataset = _load_dataset(dataset_paths, include_default=not args.no_default_dataset)

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

    def request_stop() -> None:
        nonlocal should_stop
        should_stop = True
        server.stop(grace=1.0)

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
                LOGGER.info("Limit czasu minął – zatrzymuję serwer stub.")
        else:
            LOGGER.info("Naciśnij Ctrl+C, aby zakończyć pracę stubu.")
            server.wait_for_termination()
    except KeyboardInterrupt:  # pragma: no cover - zależy od środowiska testowego
        LOGGER.info("Przerwano przez użytkownika – zatrzymywanie serwera.")
    finally:
        if not should_stop:
            server.stop(grace=1.0)
        LOGGER.info("Serwer stub został zatrzymany.")
    return 0


if __name__ == "__main__":  # pragma: no cover - ścieżka uruchomienia skryptu
    sys.exit(main())
