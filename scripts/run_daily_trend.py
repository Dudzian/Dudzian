"""CLI do uruchamiania pipeline'u strategii Daily Trend w trybie paper/testnet."""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from bot_core.alerts import AlertMessage
from bot_core.exchanges.base import Environment, OrderResult
from bot_core.runtime.pipeline import build_daily_trend_pipeline, create_trading_controller
from bot_core.runtime.realtime import DailyTrendRealtimeRunner
from bot_core.security import SecretManager, SecretStorageError, create_default_secret_storage

_LOGGER = logging.getLogger(__name__)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Uruchamia strategię trend-following D1 w trybie paper/testnet."
    )
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do CoreConfig")
    parser.add_argument(
        "--environment",
        default="binance_paper",
        help="Nazwa środowiska z pliku konfiguracyjnego (np. binance_paper)",
    )
    parser.add_argument(
        "--strategy",
        default="core_daily_trend",
        help="Nazwa strategii z sekcji strategies w configu",
    )
    parser.add_argument(
        "--controller",
        default="daily_trend_core",
        help="Nazwa kontrolera runtime z sekcji runtime.controllers",
    )
    parser.add_argument(
        "--history-bars",
        type=int,
        default=180,
        help="Liczba świec wykorzystywanych do analizy na starcie każdej iteracji",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=900.0,
        help="Jak często sprawdzać nowe sygnały (sekundy) w trybie ciągłym",
    )
    parser.add_argument(
        "--health-interval",
        type=float,
        default=3600.0,
        help="Interwał raportów health-check (sekundy)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Poziom logowania",
    )
    parser.add_argument(
        "--secret-namespace",
        default="dudzian.trading",
        help="Namespace używany przy zapisie sekretów w systemowym keychainie",
    )
    parser.add_argument(
        "--headless-passphrase",
        default=None,
        help="Hasło do szyfrowania magazynu sekretów w środowisku headless (Linux)",
    )
    parser.add_argument(
        "--headless-storage",
        default=None,
        help="Ścieżka pliku magazynu sekretów dla trybu headless",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Uruchom pojedynczą iterację i zakończ (np. do harmonogramu cron)",
    )
    parser.add_argument(
        "--paper-smoke",
        action="store_true",
        help="Uruchom test dymny strategii paper trading (backfill + pojedyncza iteracja)",
    )
    parser.add_argument(
        "--date-window",
        default=None,
        help="Zakres dat w formacie START:END (np. 2024-01-01:2024-02-15) dla trybu --paper-smoke",
    )
    parser.add_argument(
        "--allow-live",
        action="store_true",
        help="Zezwól na uruchomienie na środowisku LIVE (domyślnie blokowane)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Zbuduj pipeline bez wykonywania iteracji (walidacja konfiguracji)",
    )
    return parser.parse_args(argv)


def _create_secret_manager(args: argparse.Namespace) -> SecretManager:
    storage = create_default_secret_storage(
        namespace=args.secret_namespace,
        headless_passphrase=args.headless_passphrase,
        headless_path=args.headless_storage,
    )
    return SecretManager(storage, namespace=args.secret_namespace)


def _log_order_results(results: Iterable[OrderResult]) -> None:
    for result in results:
        _LOGGER.info(
            "Zlecenie zrealizowane: id=%s status=%s qty=%s avg_price=%s",
            result.order_id,
            result.status,
            result.filled_quantity,
            result.avg_price,
        )


def _parse_iso_date(value: str, *, is_end: bool) -> datetime:
    text = value.strip()
    if not text:
        raise ValueError("wartość daty nie może być pusta")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:  # pragma: no cover - walidacja argumentów CLI
        raise ValueError(f"nieprawidłowy format daty: {text}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    if "T" not in text and " " not in text:
        # W przypadku zakresów dziennych interpretujemy datę końcową jako koniec dnia.
        if is_end:
            parsed = parsed + timedelta(days=1) - timedelta(milliseconds=1)
    return parsed


def _resolve_date_window(arg: str | None, *, default_days: int = 30) -> tuple[int, int, Mapping[str, str]]:
    if not arg:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=default_days)
    else:
        parts = arg.split(":", maxsplit=1)
        if len(parts) != 2:
            raise ValueError("zakres musi mieć format START:END")
        start_dt = _parse_iso_date(parts[0], is_end=False)
        end_dt = _parse_iso_date(parts[1], is_end=True)
    if start_dt > end_dt:
        raise ValueError("data początkowa jest późniejsza niż końcowa")
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    return start_ms, end_ms, {
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
    }


def _export_smoke_report(
    *,
    report_dir: Path,
    results: Sequence[OrderResult],
    ledger: Iterable[Mapping[str, object]],
    window: Mapping[str, str],
    environment: str,
    alert_snapshot: Mapping[str, Mapping[str, str]],
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    ledger_entries = list(ledger)
    ledger_path = report_dir / "ledger.jsonl"
    with ledger_path.open("w", encoding="utf-8") as handle:
        for entry in ledger_entries:
            json.dump(entry, handle, ensure_ascii=False)
            handle.write("\n")

    summary = {
        "environment": environment,
        "window": dict(window),
        "orders": [
            {
                "order_id": result.order_id,
                "status": result.status,
                "filled_quantity": result.filled_quantity,
                "avg_price": result.avg_price,
            }
            for result in results
        ],
        "ledger_entries": len(ledger_entries),
        "alert_snapshot": {channel: dict(data) for channel, data in alert_snapshot.items()},
    }

    summary_path = report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_path


def _run_loop(runner: DailyTrendRealtimeRunner, poll_seconds: float) -> int:
    interval = max(1.0, poll_seconds)
    stop = False

    def _signal_handler(_signo, _frame) -> None:  # type: ignore[override]
        nonlocal stop
        stop = True
        _LOGGER.info("Otrzymano sygnał zatrzymania – kończę pętlę realtime")

    for signame in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signame, _signal_handler)

    _LOGGER.info("Start pętli realtime (co %s s)", interval)
    while not stop:
        start = time.monotonic()
        try:
            results = runner.run_once()
            if results:
                _log_order_results(results)
        except Exception:  # noqa: BLE001
            _LOGGER.exception("Błąd podczas iteracji realtime")
        elapsed = time.monotonic() - start
        sleep_for = max(1.0, interval - elapsed)
        if stop:
            break
        time.sleep(sleep_for)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), stream=sys.stdout)

    try:
        secret_manager = _create_secret_manager(args)
    except SecretStorageError as exc:
        _LOGGER.error("Nie udało się zainicjalizować magazynu sekretów: %s", exc)
        return 2

    config_path = Path(args.config)
    if not config_path.exists():
        _LOGGER.error("Plik konfiguracyjny %s nie istnieje", config_path)
        return 1

    try:
        pipeline = build_daily_trend_pipeline(
            environment_name=args.environment,
            strategy_name=args.strategy,
            controller_name=args.controller,
            config_path=config_path,
            secret_manager=secret_manager,
        )
    except Exception as exc:  # noqa: BLE001
        _LOGGER.exception("Nie udało się zbudować pipeline'u daily trend: %s", exc)
        return 1

    environment = pipeline.bootstrap.environment.environment
    if environment is Environment.LIVE and not args.allow_live:
        _LOGGER.error(
            "Środowisko %s to LIVE – dla bezpieczeństwa użyj --allow-live po wcześniejszych testach paper.",
            args.environment,
        )
        return 3

    if args.dry_run:
        _LOGGER.info("Dry-run zakończony sukcesem. Pipeline gotowy do uruchomienia.")
        return 0

    if args.paper_smoke:
        try:
            start_ms, end_ms, window_meta = _resolve_date_window(args.date_window)
        except ValueError as exc:
            _LOGGER.error("Niepoprawny zakres dat: %s", exc)
            return 1

        _LOGGER.info(
            "Startuję smoke test paper trading dla %s w zakresie %s – %s.",
            args.environment,
            window_meta["start"],
            window_meta["end"],
        )

        pipeline.backfill_service.synchronize(
            symbols=pipeline.controller.symbols,
            interval=pipeline.controller.interval,
            start=start_ms,
            end=end_ms,
        )

        trading_controller = create_trading_controller(
            pipeline,
            pipeline.bootstrap.alert_router,
            health_check_interval=0.0,
        )

        runner = DailyTrendRealtimeRunner(
            controller=pipeline.controller,
            trading_controller=trading_controller,
            history_bars=max(1, args.history_bars),
        )

        results = runner.run_once()
        if results:
            _log_order_results(results)
        else:
            _LOGGER.info("Smoke test zakończony – brak sygnałów w zadanym oknie.")

        report_dir = Path(tempfile.mkdtemp(prefix="daily_trend_smoke_"))
        alert_snapshot = pipeline.bootstrap.alert_router.health_snapshot()
        summary_path = _export_smoke_report(
            report_dir=report_dir,
            results=results,
            ledger=pipeline.execution_service.ledger(),
            window=window_meta,
            environment=args.environment,
            alert_snapshot=alert_snapshot,
        )
        _LOGGER.info("Raport smoke testu zapisany w %s", report_dir)

        message = AlertMessage(
            category="paper_smoke",
            title=f"Smoke test paper trading ({args.environment})",
            body=(
                "Zakończono smoke test paper trading."
                f" Zamówienia: {len(results)}, raport: {summary_path}"
            ),
            severity="info",
            context={
                "environment": args.environment,
                "report_dir": str(report_dir),
                "orders": str(len(results)),
            },
        )
        pipeline.bootstrap.alert_router.dispatch(message)
        return 0

    trading_controller = create_trading_controller(
        pipeline,
        pipeline.bootstrap.alert_router,
        health_check_interval=args.health_interval,
    )

    runner = DailyTrendRealtimeRunner(
        controller=pipeline.controller,
        trading_controller=trading_controller,
        history_bars=max(1, args.history_bars),
    )

    if args.run_once:
        _LOGGER.info("Uruchamiam pojedynczą iterację strategii dla środowiska %s", args.environment)
        results = runner.run_once()
        if results:
            _log_order_results(results)
        else:
            _LOGGER.info("Brak sygnałów w tej iteracji – nic nie zlecam.")
        return 0

    return _run_loop(runner, args.poll_seconds)


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    raise SystemExit(main())
