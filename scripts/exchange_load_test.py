"""Skrypt obciążeniowy dla adapterów giełdowych."""
from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, Sequence

from bot_core.exchanges.manager import ExchangeManager


_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class LoadTestResult:
    """Podsumowanie wyników pojedynczego wątku."""

    requests: int
    failures: int
    latency_samples: Sequence[float]


def _build_manager(args: argparse.Namespace) -> ExchangeManager:
    manager = ExchangeManager(exchange_id=args.exchange)
    mode = args.mode.lower()
    if mode == "paper":
        manager.set_mode(paper=True)
    elif mode == "spot":
        manager.set_mode(spot=True, testnet=args.testnet)
    elif mode == "margin":
        manager.set_mode(margin=True, testnet=args.testnet)
    elif mode == "futures":
        manager.set_mode(futures=True, testnet=args.testnet)
    else:  # pragma: no cover - walidacja argumentów odbywa się wcześniej
        raise ValueError(f"Nieznany tryb: {args.mode}")

    if args.api_key or args.secret:
        manager.set_credentials(args.api_key, args.secret, passphrase=args.passphrase)

    if args.paper_variant:
        manager.set_paper_variant(args.paper_variant)

    manager.load_markets()
    return manager


def _run_worker(args: argparse.Namespace, deadline: float) -> LoadTestResult:
    manager = _build_manager(args)
    requests = 0
    failures = 0
    latencies: list[float] = []
    operation = args.operation
    symbol = args.symbol
    interval = args.interval
    limit = args.limit

    while time.monotonic() < deadline:
        start = time.monotonic()
        try:
            if operation == "ticker":
                manager.fetch_ticker(symbol)
            elif operation == "ohlcv":
                manager.fetch_ohlcv(symbol, timeframe=interval, limit=limit)
            elif operation == "order_book":
                manager.fetch_order_book(symbol, limit=limit)
            else:  # pragma: no cover - argumenty sprawdzane wcześniej
                raise ValueError(f"Nieznana operacja: {operation}")
        except Exception as exc:  # noqa: BLE001 - skrypt ma raportować błędy
            failures += 1
            _LOGGER.warning("Błąd podczas wykonywania %s (%s): %s", operation, symbol, exc)
        else:
            latency = time.monotonic() - start
            latencies.append(latency)
        finally:
            requests += 1
        if args.sleep > 0:
            time.sleep(args.sleep)
    return LoadTestResult(requests=requests, failures=failures, latency_samples=tuple(latencies))


def _aggregate(results: Iterable[LoadTestResult]) -> None:
    total_requests = 0
    total_failures = 0
    latencies: list[float] = []
    for result in results:
        total_requests += result.requests
        total_failures += result.failures
        latencies.extend(result.latency_samples)
    success = total_requests - total_failures
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    p95_latency = 0.0
    if latencies:
        sorted_latencies = sorted(latencies)
        index = int(0.95 * (len(sorted_latencies) - 1))
        p95_latency = sorted_latencies[index]
    _LOGGER.info("Wyniki testu obciążeniowego")
    _LOGGER.info("  Liczba żądań: %s", total_requests)
    _LOGGER.info("  Udane żądania: %s", success)
    _LOGGER.info("  Niepowodzenia: %s", total_failures)
    _LOGGER.info("  Średnie opóźnienie: %.3fs", avg_latency)
    _LOGGER.info("  95 percentyl opóźnienia: %.3fs", p95_latency)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test obciążeniowy adaptera giełdowego")
    parser.add_argument("symbol", help="Symbol rynkowy, np. BTC/USDT")
    parser.add_argument("--exchange", default="binance", help="Id giełdy (domyślnie binance)")
    parser.add_argument(
        "--mode",
        choices=("paper", "spot", "margin", "futures"),
        default="paper",
        help="Tryb działania managera",
    )
    parser.add_argument(
        "--operation",
        choices=("ticker", "ohlcv", "order_book"),
        default="ticker",
        help="Operacja wykonywana podczas testu",
    )
    parser.add_argument("--interval", default="1m", help="Interwał OHLCV (dla operation=ohlcv)")
    parser.add_argument("--limit", type=int, default=100, help="Limit wyników dla OHLCV/order book")
    parser.add_argument("--duration", type=float, default=60.0, help="Czas trwania testu w sekundach")
    parser.add_argument("--concurrency", type=int, default=4, help="Liczba równoległych pracowników")
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Opcjonalna przerwa między żądaniami w sekundach",
    )
    parser.add_argument("--api-key", dest="api_key", default="", help="Klucz API giełdy")
    parser.add_argument("--secret", default="", help="Sekret API giełdy")
    parser.add_argument("--passphrase", default="", help="Opcjonalne hasło API")
    parser.add_argument("--paper-variant", default="", help="Wariant symulatora papierowego")
    parser.add_argument("--testnet", action="store_true", help="Włącza tryb testnet dla spot/margin/futures")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Poziom logowania",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    deadline = time.monotonic() + args.duration
    workers = max(1, args.concurrency)
    _LOGGER.info(
        "Start testu obciążeniowego: exchange=%s mode=%s operation=%s concurrency=%s duration=%ss",
        args.exchange,
        args.mode,
        args.operation,
        workers,
        args.duration,
    )
    results: list[LoadTestResult] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="exchange-load") as executor:
        futures = [executor.submit(_run_worker, args, deadline) for _ in range(workers)]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                _LOGGER.error("Wątek testowy zakończył się błędem: %s", exc)
    _aggregate(results)


if __name__ == "__main__":  # pragma: no cover - skrypt wykonywany manualnie
    main()
