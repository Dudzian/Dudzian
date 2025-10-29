"""Uruchamia lokalny poller REST, który zapisuje metadane instrumentów do plików JSON."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
from pathlib import Path
from typing import Iterable

from bot_core.runtime.market_data_service import RestMarketDataPoller

_LOG = logging.getLogger(__name__)
_DEFAULT_OUTPUT = Path("var/cache/market_data")


def _write_snapshots(poller: RestMarketDataPoller, exchanges: Iterable[str], output_dir: Path) -> None:
    """Zapisuje ostatnie snapshoty instrumentów do pojedynczych plików JSON."""

    output_dir.mkdir(parents=True, exist_ok=True)
    aggregated: dict[str, list[dict[str, object]]] = {}
    for exchange in exchanges:
        normalized = exchange.upper()
        snapshot = poller.snapshot(exchange)
        aggregated[normalized] = snapshot
        file_path = output_dir / f"{exchange.lower()}.json"
        file_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        _LOG.debug("Zapisano snapshot %s (%d instrumentów) do %s", normalized, len(snapshot), file_path)
    aggregate_path = output_dir / "snapshots.json"
    aggregate_path.write_text(json.dumps(aggregated, ensure_ascii=False, indent=2), encoding="utf-8")
    _LOG.debug("Zapisano zagregowany snapshot do %s", aggregate_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lokalny poller REST generujący snapshoty instrumentów giełdowych",
    )
    parser.add_argument(
        "-e",
        "--exchange",
        dest="exchange",
        action="append",
        required=True,
        help="Id giełdy (można powtarzać)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=120.0,
        help="Interwał odświeżania danych w sekundach (domyślnie 120)",
    )
    parser.add_argument(
        "--flush-interval",
        type=float,
        default=None,
        help="Co ile sekund zapisywać snapshoty na dysk (domyślnie jak --interval)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Katalog docelowy dla plików JSON (domyślnie {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--profile",
        default="paper",
        help="Nazwa profilu środowiska z config/exchanges (domyślnie 'paper')",
    )
    parser.add_argument(
        "--config-dir",
        help="Opcjonalny katalog z plikami YAML profili (nadpisuje config/exchanges)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Pobierz snapshot raz i zakończ działanie",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Poziom logowania (np. DEBUG, INFO)",
    )
    return parser


def run_cli(args: argparse.Namespace, *, poller_factory=RestMarketDataPoller) -> int:
    level = getattr(logging, str(args.log_level or "INFO").upper(), logging.INFO)
    logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    exchanges = [entry.strip() for entry in args.exchange or [] if entry and entry.strip()]
    if not exchanges:
        raise SystemExit("Wymagana jest co najmniej jedna giełda (--exchange)")

    profile = (args.profile or "").strip()
    poller_kwargs = {
        "interval": float(args.interval),
        "profile": profile or None,
    }
    if args.config_dir:
        poller_kwargs["config_dir"] = args.config_dir

    poller = poller_factory(exchanges, **poller_kwargs)

    flush_interval = float(args.flush_interval or args.interval or 60.0)
    output_dir = Path(args.output).expanduser().resolve()

    _LOG.info(
        "Start pollera REST dla %s (profil=%s, interval=%.2fs, flush=%.2fs)",
        ", ".join(exchanges),
        profile or "brak",
        args.interval,
        flush_interval,
    )

    try:
        poller.refresh_now()
        _write_snapshots(poller, exchanges, output_dir)
        if args.once:
            _LOG.info("Zakończono po jednorazowym odświeżeniu")
            return 0

        stop_event = threading.Event()

        def _handle_signal(signum, _frame):  # pragma: no cover - zależne od systemu
            _LOG.info("Otrzymano sygnał %s – kończenie pracy pollera", signum)
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):  # pragma: no cover - platform specific
            signal.signal(sig, _handle_signal)

        poller.start()
        _LOG.info("Poller REST działa w tle. Naciśnij Ctrl+C aby zakończyć.")

        while not stop_event.wait(timeout=flush_interval):
            _write_snapshots(poller, exchanges, output_dir)

    except KeyboardInterrupt:  # pragma: no cover - obsługa sygnałów
        _LOG.info("Przerwano działanie pollera (KeyboardInterrupt)")
    finally:
        poller.stop()

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_cli(args)
    except Exception as exc:  # pragma: no cover - diagnostyka CLI
        _LOG.error("Błąd pollera REST: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
