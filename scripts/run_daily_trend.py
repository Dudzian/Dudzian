"""CLI do uruchamiania pipeline'u strategii Daily Trend w trybie paper/testnet."""
from __future__ import annotations

import argparse
import json
import hashlib
import os
import logging
import signal
import sys
import shutil
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import deque
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from typing import Any

from bot_core.alerts import AlertMessage
from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeAdapterFactory,
    ExchangeCredentials,
    OrderResult,
)
from bot_core.data.intervals import (
    interval_to_milliseconds as _interval_to_milliseconds,
    normalize_interval_token as _normalize_interval_token,
)
from bot_core.reporting.upload import SmokeArchiveUploader
from bot_core.data.ohlcv import evaluate_coverage
from bot_core.runtime.pipeline import build_daily_trend_pipeline, create_trading_controller
from bot_core.runtime.realtime import DailyTrendRealtimeRunner
from bot_core.security import SecretManager, SecretStorageError, create_default_secret_storage

_LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Argumenty CLI
# --------------------------------------------------------------------------------------
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
        default=None,
        help="Nazwa strategii z sekcji strategies (domyślnie pobierana z konfiguracji środowiska)",
    )
    parser.add_argument(
        "--controller",
        default=None,
        help="Nazwa kontrolera runtime (domyślnie pobierana z konfiguracji środowiska)",
    )
    parser.add_argument(
        "--risk-profile",
        default=None,
        help="Nazwa profilu ryzyka z sekcji risk_profiles (domyślnie używany profil przypisany do środowiska)",
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
        "--archive-smoke",
        action="store_true",
        help="Po zakończeniu smoke testu spakuj raport do archiwum ZIP z instrukcją audytu",
    )
    parser.add_argument(
        "--smoke-output",
        default=None,
        help="Opcjonalny katalog bazowy na raporty smoke testu; w środku powstanie podkatalog daily_trend_smoke_*.",
    )
    parser.add_argument(
        "--smoke-min-free-mb",
        type=float,
        default=None,
        help=(
            "Minimalna ilość wolnego miejsca (w MB) wymagana w katalogu raportu smoke; "
            "przy niższej wartości zgłosimy ostrzeżenie i oznaczymy raport."
        ),
    )
    parser.add_argument(
        "--smoke-fail-on-low-space",
        action="store_true",
        help=(
            "Traktuj ostrzeżenie o niskim wolnym miejscu jako błąd – po zapisaniu raportu "
            "zakończ proces kodem != 0."
        ),
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


# --------------------------------------------------------------------------------------
# Pomocnicze I/O, formaty i konwersje
# --------------------------------------------------------------------------------------
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
    except ValueError as exc:  # pragma: no cover
        raise ValueError(f"nieprawidłowy format daty: {text}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    if "T" not in text and " " not in text:
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
    return start_ms, end_ms, {"start": start_dt.isoformat(), "end": end_dt.isoformat()}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _as_int(value: object) -> int | None:
    float_value = _as_float(value)
    if float_value is None:
        return None
    try:
        return int(float_value)
    except (TypeError, ValueError):  # pragma: no cover
        return None


def _format_money(value: float, *, decimals: int = 2) -> str:
    formatted = f"{value:,.{decimals}f}"
    return formatted.replace(",", " ")


def _format_percentage(value: float | None, *, decimals: int = 2) -> str:
    if value is None:
        return "n/d"
    return f"{value * 100:.{decimals}f}%"


def _normalize_position_entry(symbol: str, payload: Mapping[str, object]) -> tuple[float, str] | None:
    """Buduje opis pojedynczej pozycji do raportu tekstowego."""
    notional = _as_float(payload.get("notional"))
    if notional is None or notional <= 0:
        return None
    side = str(payload.get("side", "")).strip().upper() or "?"
    description = f"{symbol}: {side} {_format_money(notional)}"
    return notional, description


# --------------------------------------------------------------------------------------
# Metryki ledger / PnL FIFO long/short
# --------------------------------------------------------------------------------------
def _compute_ledger_metrics(ledger_entries: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    counts: MutableMapping[str, int] = {"buy": 0, "sell": 0}
    other_counts: MutableMapping[str, int] = {}
    notionals: MutableMapping[str, float] = {"buy": 0.0, "sell": 0.0}
    other_notionals: MutableMapping[str, float] = {}
    total_fees = 0.0
    last_position_value: float | None = None
    per_symbol: dict[str, dict[str, float]] = {}
    pnl_trackers: dict[str, dict[str, object]] = {}
    realized_pnl_total = 0.0
    eps = 1e-9

    for entry in ledger_entries:
        if not isinstance(entry, Mapping):
            continue

        side = str(entry.get("side", "")).lower()
        quantity = _as_float(entry.get("quantity")) or 0.0
        price = _as_float(entry.get("price")) or 0.0
        notional_value = abs(quantity) * max(price, 0.0)
        abs_quantity = abs(quantity)

        if side in ("buy", "sell"):
            counts[side] += 1
            notionals[side] += notional_value
        else:
            side_key = side or "unknown"
            other_counts[side_key] = other_counts.get(side_key, 0) + 1
            other_notionals[side_key] = other_notionals.get(side_key, 0.0) + notional_value

        fee_value = _as_float(entry.get("fee"))
        if fee_value is not None:
            total_fees += fee_value

        position_value = _as_float(entry.get("position_value"))
        if position_value is not None:
            last_position_value = position_value

        symbol = entry.get("symbol")
        if symbol:
            symbol_key = str(symbol)
            stats = per_symbol.setdefault(
                symbol_key,
                {
                    "orders": 0,
                    "buy_orders": 0,
                    "sell_orders": 0,
                    "other_orders": 0,
                    "buy_quantity": 0.0,
                    "sell_quantity": 0.0,
                    "other_quantity": 0.0,
                    "buy_notional": 0.0,
                    "sell_notional": 0.0,
                    "other_notional": 0.0,
                    "total_notional": 0.0,
                    "net_quantity": 0.0,
                    "fees": 0.0,
                    "realized_pnl": 0.0,
                },
            )

            stats["orders"] += 1
            if side == "buy":
                stats["buy_orders"] += 1
                stats["buy_quantity"] += quantity
                stats["buy_notional"] += notional_value
                stats["net_quantity"] += quantity
            elif side == "sell":
                stats["sell_orders"] += 1
                stats["sell_quantity"] += quantity
                stats["sell_notional"] += notional_value
                stats["net_quantity"] -= quantity
            else:
                stats["other_orders"] += 1
                stats["other_quantity"] += quantity
                stats["other_notional"] += notional_value

            stats["total_notional"] = (
                stats["buy_notional"] + stats["sell_notional"] + stats["other_notional"]
            )

            if fee_value is not None:
                stats["fees"] += fee_value

            if position_value is not None:
                stats["last_position_value"] = position_value

            tracker = pnl_trackers.setdefault(
                symbol_key,
                {
                    "long_lots": deque(),
                    "short_lots": deque(),
                    "realized_pnl": 0.0,
                },
            )
            long_lots: deque[tuple[float, float]] = tracker["long_lots"]  # type: ignore[assignment]
            short_lots: deque[tuple[float, float]] = tracker["short_lots"]  # type: ignore[assignment]
            realized_symbol: float = tracker["realized_pnl"]  # type: ignore[assignment]

            remaining_qty = abs_quantity

            if side == "buy":
                while remaining_qty > eps and short_lots:
                    lot_qty, lot_price = short_lots[0]
                    matched = min(remaining_qty, lot_qty)
                    realized_symbol += (lot_price - price) * matched
                    lot_qty -= matched
                    remaining_qty -= matched
                    if lot_qty <= eps:
                        short_lots.popleft()
                    else:
                        short_lots[0] = (lot_qty, lot_price)
                if remaining_qty > eps:
                    long_lots.append((remaining_qty, price))
            elif side == "sell":
                while remaining_qty > eps and long_lots:
                    lot_qty, lot_price = long_lots[0]
                    matched = min(remaining_qty, lot_qty)
                    realized_symbol += (price - lot_price) * matched
                    lot_qty -= matched
                    remaining_qty -= matched
                    if lot_qty <= eps:
                        long_lots.popleft()
                    else:
                        long_lots[0] = (lot_qty, lot_price)
                if remaining_qty > eps:
                    short_lots.append((remaining_qty, price))

            tracker["realized_pnl"] = realized_symbol
            stats["realized_pnl"] = realized_symbol
            previous_realized = tracker.get("_realized_accumulator", 0.0)
            realized_pnl_total += realized_symbol - float(previous_realized)
            tracker["_realized_accumulator"] = realized_symbol

    total_notional = sum(notionals.values()) + sum(other_notionals.values())

    side_counts: MutableMapping[str, int] = {
        "buy": counts.get("buy", 0),
        "sell": counts.get("sell", 0),
    }
    for key, value in other_counts.items():
        if value:
            side_counts[key] = value

    notional_payload: MutableMapping[str, float] = {
        "buy": notionals.get("buy", 0.0),
        "sell": notionals.get("sell", 0.0),
    }
    for key, value in other_notionals.items():
        if value:
            notional_payload[key] = value
    notional_payload["total"] = total_notional

    metrics: dict[str, object] = {
        "side_counts": dict(side_counts),
        "notional": dict(notional_payload),
        "total_fees": total_fees,
    }
    metrics["realized_pnl_total"] = realized_pnl_total
    if last_position_value is not None:
        metrics["last_position_value"] = last_position_value
    if per_symbol:
        metrics["per_symbol"] = {
            symbol: {k: (float(v) if isinstance(v, float) else v) for k, v in stats.items()}
            for symbol, stats in per_symbol.items()
        }
    return metrics


# --------------------------------------------------------------------------------------
# Raport smoke
# --------------------------------------------------------------------------------------
def _export_smoke_report(
    *,
    report_dir: Path,
    results: Sequence[OrderResult],
    ledger: Iterable[Mapping[str, object]],
    window: Mapping[str, str],
    environment: str,
    alert_snapshot: Mapping[str, Mapping[str, str]],
    risk_state: Mapping[str, object] | None = None,
    data_checks: Mapping[str, object] | None = None,
    storage_info: Mapping[str, object] | None = None,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    ledger_entries = list(ledger)
    ledger_path = report_dir / "ledger.jsonl"
    with ledger_path.open("w", encoding="utf-8") as handle:
        for entry in ledger_entries:
            json.dump(entry, handle, ensure_ascii=False)
            handle.write("\n")

    metrics = _compute_ledger_metrics(ledger_entries)

    summary: dict[str, object] = {
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
        "metrics": metrics,
        "alert_snapshot": {channel: dict(data) for channel, data in alert_snapshot.items()},
    }
    if risk_state:
        summary["risk_state"] = dict(risk_state)
    if data_checks:
        summary["data_checks"] = json.loads(json.dumps(data_checks))
    if storage_info:
        summary["storage"] = json.loads(json.dumps(storage_info))

    if risk_state:
        summary["risk_state"] = dict(risk_state)

    if data_checks:
        summary["data_checks"] = json.loads(json.dumps(data_checks))

    if storage_info:
        summary["storage"] = json.loads(json.dumps(storage_info))

    if risk_state:
        summary["risk_state"] = dict(risk_state)

    if data_checks:
        summary["data_checks"] = json.loads(json.dumps(data_checks))

    if storage_info:
        summary["storage"] = json.loads(json.dumps(storage_info))

    if risk_state:
        summary["risk_state"] = dict(risk_state)

    if data_checks:
        summary["data_checks"] = json.loads(json.dumps(data_checks))

    if storage_info:
        summary["storage"] = json.loads(json.dumps(storage_info))

    if note:
        stripped = note.strip()
        if stripped:
            summary["note"] = stripped

    if risk_state:
        summary["risk_state"] = dict(risk_state)

    if data_checks:
        summary["data_checks"] = json.loads(json.dumps(data_checks))

    if storage_info:
        summary["storage"] = json.loads(json.dumps(storage_info))

    if note:
        stripped = note.strip()
        if stripped:
            summary["note"] = stripped

    summary_path = report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_path


def _write_smoke_readme(report_dir: Path) -> Path:
    readme_path = report_dir / "README.txt"
    readme_text = (
        "Daily Trend – smoke test paper trading\n"
        "======================================\n\n"
        "Ten katalog zawiera artefakty pojedynczego uruchomienia trybu --paper-smoke.\n"
        "Na potrzeby audytu:\n\n"
        "1. Zweryfikuj hash SHA-256 pliku summary.json zapisany w logu CLI oraz w alertach.\n"
        "2. Przepisz treść summary.txt do dziennika audytowego (docs/audit/paper_trading_log.md).\n"
        "3. Zabezpiecz ledger.jsonl (pełna historia decyzji) w repozytorium operacyjnym.\n"
        "4. Zarchiwizowany plik ZIP można przechowywać w sejfie audytu przez min. 24 miesiące.\n"
    )
    readme_path.write_text(readme_text + "\n", encoding="utf-8")
    return readme_path


def _archive_smoke_report(report_dir: Path) -> Path:
    archive_path_str = shutil.make_archive(str(report_dir), "zip", root_dir=report_dir)
    return Path(archive_path_str)


_MEGABYTE = 1024 * 1024


def _collect_storage_health(directory: Path, *, min_free_mb: float | None) -> Mapping[str, object]:
    """Zwraca informacje o stanie przestrzeni dyskowej dla raportu smoke."""
    info: dict[str, object] = {"directory": str(directory)}
    threshold_mb = float(min_free_mb) if min_free_mb is not None else None
    if threshold_mb is not None and threshold_mb < 0:
        threshold_mb = 0.0
    threshold_bytes = int(threshold_mb * _MEGABYTE) if threshold_mb is not None else None

    try:
        usage = shutil.disk_usage(directory)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("Nie udało się odczytać informacji o wolnym miejscu dla %s: %s", directory, exc)
        info.update({"status": "unknown", "error": str(exc)})
        if threshold_bytes is not None:
            info["threshold_bytes"] = threshold_bytes
            info["threshold_mb"] = threshold_bytes / _MEGABYTE
        return info

    free_bytes = int(usage.free)
    total_bytes = int(usage.total)
    info.update(
        {
            "status": "ok",
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
            "free_mb": free_bytes / _MEGABYTE,
            "total_mb": total_bytes / _MEGABYTE,
        }
    )

    if threshold_bytes is not None:
        info["threshold_bytes"] = threshold_bytes
        info["threshold_mb"] = threshold_bytes / _MEGABYTE
        if free_bytes < threshold_bytes:
            info["status"] = "low"
            _LOGGER.warning(
                "Wolne miejsce w katalogu raportu %s: %.2f MB (< %.2f MB)",
                directory,
                free_bytes / _MEGABYTE,
                threshold_bytes / _MEGABYTE,
            )

    return info


def _collect_required_intervals(
    pipeline: Any,
    *,
    symbols: Sequence[str],
) -> tuple[str, ...]:
    """Zwraca uporządkowaną listę interwałów wymaganych do smoke testu."""
    intervals: list[str] = []
    seen: set[str] = set()

    def _add_interval(value: str | None) -> None:
        normalized = _normalize_interval_token(value)
        if not normalized:
            return
        if normalized in seen:
            return
        seen.add(normalized)
        intervals.append(value or normalized)

    primary_interval = getattr(getattr(pipeline, "controller", None), "interval", None)
    if primary_interval:
        _add_interval(primary_interval)

    bootstrap = getattr(pipeline, "bootstrap", None)
    if bootstrap is None:
        return tuple(intervals)

    environment_cfg = getattr(bootstrap, "environment", None)
    core_config = getattr(bootstrap, "core_config", None)
    if environment_cfg is None or core_config is None:
        return tuple(intervals)

    universe_name = getattr(environment_cfg, "instrument_universe", None)
    exchange_name = getattr(environment_cfg, "exchange", None)
    if not universe_name or not exchange_name:
        return tuple(intervals)

    if not hasattr(core_config, "instrument_universes"):
        return tuple(intervals)

    try:
        universe = core_config.instrument_universes[universe_name]
    except Exception:  # noqa: BLE001
        return tuple(intervals)

    tracked_symbols = {str(symbol).lower() for symbol in symbols}
    for instrument in getattr(universe, "instruments", ()):  # type: ignore[attr-defined]
        symbol = instrument.exchange_symbols.get(exchange_name) if instrument else None
        if symbol and symbol.lower() in tracked_symbols:
            for window in getattr(instrument, "backfill_windows", ()):  # type: ignore[attr-defined]
                _add_interval(getattr(window, "interval", None))

    return tuple(intervals)


def _prepare_smoke_report_directory(target: str | None) -> Path:
    """Zwraca katalog na raport smoke testu, tworząc go jeśli potrzeba."""
    if target:
        base_dir = Path(target).expanduser()
        base_dir.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="daily_trend_smoke_", dir=str(base_dir)))
    return Path(tempfile.mkdtemp(prefix="daily_trend_smoke_"))


def _render_smoke_summary(*, summary: Mapping[str, object], summary_sha256: str) -> str:
    environment = str(summary.get("environment", "unknown"))
    window = summary.get("window", {})
    if isinstance(window, Mapping):
        start = str(window.get("start", "?"))
        end = str(window.get("end", "?"))
    else:  # pragma: no cover
        start = end = "?"

    orders = summary.get("orders", [])
    orders_count = len(orders) if isinstance(orders, Sequence) else 0
    ledger_entries = summary.get("ledger_entries", 0)
    try:
        ledger_entries = int(ledger_entries)
    except Exception:  # noqa: BLE001
        ledger_entries = 0

    alert_snapshot = summary.get("alert_snapshot", {})
    alert_lines: list[str] = []
    if isinstance(alert_snapshot, Mapping):
        for channel, data in alert_snapshot.items():
            status = "UNKNOWN"
            detail: str | None = None
            if isinstance(data, Mapping):
                raw_status = data.get("status")
                if raw_status is not None:
                    status = str(raw_status).upper()
                raw_detail = data.get("detail")
                if raw_detail:
                    detail = str(raw_detail)
            channel_name = str(channel)
            if detail:
                alert_lines.append(f"{channel_name}: {status} ({detail})")
            else:
                alert_lines.append(f"{channel_name}: {status}")
    if not alert_lines:
        alert_lines.append("brak danych o kanałach alertów")

    metrics_lines: list[str] = []
    metrics = summary.get("metrics")
    if isinstance(metrics, Mapping):
        side_counts = metrics.get("side_counts")
        if isinstance(side_counts, Mapping):
            buy_count = _as_int(side_counts.get("buy")) or 0
            sell_count = _as_int(side_counts.get("sell")) or 0
            if buy_count or sell_count:
                metrics_lines.append(f"Zlecenia BUY/SELL: {buy_count}/{sell_count}")
            other_sides = [
                f"{str(name).upper()}: {_as_int(value) or 0}"
                for name, value in side_counts.items()
                if str(name).lower() not in {"buy", "sell"}
            ]
            if other_sides:
                metrics_lines.append("Inne strony: " + ", ".join(other_sides))

        notionals = metrics.get("notional")
        if isinstance(notionals, Mapping) and notionals:
            buy_notional = _as_float(notionals.get("buy")) or 0.0
            sell_notional = _as_float(notionals.get("sell")) or 0.0
            total_notional = _as_float(notionals.get("total")) or (buy_notional + sell_notional)
            metrics_lines.append(
                "Wolumen BUY: {buy} | SELL: {sell} | Razem: {total}".format(
                    buy=_format_money(buy_notional),
                    sell=_format_money(sell_notional),
                    total=_format_money(total_notional),
                )
            )
            other_notional_lines = [
                f"{str(name).upper()}: {_format_money(_as_float(value) or 0.0)}"
                for name, value in notionals.items()
                if str(name).lower() not in {"buy", "sell", "total"}
            ]
            if other_notional_lines:
                metrics_lines.append("Wolumen inne: " + "; ".join(other_notional_lines))

        total_fees = _as_float(metrics.get("total_fees"))
        if total_fees is not None:
            metrics_lines.append(f"Łączne opłaty: {_format_money(total_fees, decimals=4)}")

        realized_total = _as_float(metrics.get("realized_pnl_total"))
        if realized_total is not None:
            metrics_lines.append(f"Realizowany PnL (brutto): {_format_money(realized_total)}")

        last_position = _as_float(metrics.get("last_position_value"))
        if last_position is not None:
            metrics_lines.append(f"Ostatnia wartość pozycji: {_format_money(last_position)}")

        per_symbol = metrics.get("per_symbol")
        if isinstance(per_symbol, Mapping):
            symbol_lines: list[tuple[float, str]] = []
            for symbol, payload in per_symbol.items():
                if not isinstance(payload, Mapping):
                    continue

                total_notional_sym = _as_float(payload.get("total_notional")) or 0.0
                orders_sym = _as_int(payload.get("orders")) or 0
                fees_value = _as_float(payload.get("fees"))
                net_quantity = _as_float(payload.get("net_quantity"))
                last_symbol_value = _as_float(payload.get("last_position_value"))
                realized_symbol = _as_float(payload.get("realized_pnl"))

                if not (
                    orders_sym
                    or total_notional_sym
                    or (fees_value is not None and fees_value)
                    or (net_quantity is not None and abs(net_quantity) > 1e-9)
                    or (last_symbol_value is not None and last_symbol_value > 0)
                    or (realized_symbol is not None and abs(realized_symbol) > 1e-9)
                ):
                    continue

                parts = [f"{symbol}: zlecenia {orders_sym}"]
                if total_notional_sym:
                    parts.append(f"wolumen {_format_money(total_notional_sym)}")
                if fees_value is not None:
                    parts.append(f"opłaty {_format_money(fees_value, decimals=4)}")
                if net_quantity is not None and abs(net_quantity) > 1e-6:
                    parts.append(f"netto {net_quantity:+.4f}")
                if last_symbol_value is not None and last_symbol_value > 0:
                    parts.append(f"wartość {_format_money(last_symbol_value)}")
                if realized_symbol is not None and abs(realized_symbol) > 1e-6:
                    parts.append(f"PnL {_format_money(realized_symbol)}")

                symbol_lines.append((total_notional_sym, ", ".join(parts)))

            if symbol_lines:
                symbol_lines.sort(key=lambda item: item[0], reverse=True)
                top_lines = [item[1] for item in symbol_lines[:3]]
                metrics_lines.append("Instrumenty: " + "; ".join(top_lines))

    # Opcjonalne linie o stanie ryzyka
    risk_lines: list[str] = []
    risk_state = summary.get("risk_state")
    if isinstance(risk_state, Mapping) and risk_state:
        profile_name = str(risk_state.get("profile", "unknown"))
        risk_lines.append(f"Profil ryzyka: {profile_name}")

        active_positions = _as_int(risk_state.get("active_positions")) or 0
        gross_notional = _as_float(risk_state.get("gross_notional"))
        exposure_line = f"Aktywne pozycje: {active_positions}"
        if gross_notional is not None:
            exposure_line += f" | Ekspozycja brutto: {_format_money(gross_notional)}"
        risk_lines.append(exposure_line)

        positions_raw = risk_state.get("positions")
        if isinstance(positions_raw, Mapping) and positions_raw:
            formatted: list[tuple[float, str]] = []
            for symbol, payload in positions_raw.items():
                if not isinstance(payload, Mapping):
                    continue
                entry = _normalize_position_entry(str(symbol), payload)
                if entry is not None:
                    formatted.append(entry)
            if formatted:
                formatted.sort(key=lambda item: item[0], reverse=True)
                formatted_lines = [text for _value, text in formatted[:5]]
                risk_lines.append("Pozycje: " + "; ".join(formatted_lines))

        daily_loss_pct = _as_float(risk_state.get("daily_loss_pct"))
        drawdown_pct = _as_float(risk_state.get("drawdown_pct"))
        risk_lines.append(
            "Dzienna strata: {loss} | Obsunięcie: {dd}".format(
                loss=_format_percentage(daily_loss_pct),
                dd=_format_percentage(drawdown_pct),
            )
        )
        liquidation = bool(risk_state.get("force_liquidation"))
        risk_lines.append("Force liquidation: TAK" if liquidation else "Force liquidation: NIE")

        limits = risk_state.get("limits")
        if isinstance(limits, Mapping):
            limit_parts: list[str] = []
            max_positions = _as_int(limits.get("max_positions"))
            if max_positions is not None:
                limit_parts.append(f"max pozycje {max_positions}")
            max_exposure = _as_float(limits.get("max_position_pct"))
            if max_exposure is not None:
                limit_parts.append(f"max ekspozycja {_format_percentage(max_exposure)}")
            max_leverage = _as_float(limits.get("max_leverage"))
            if max_leverage is not None:
                limit_parts.append(f"max dźwignia {max_leverage:.2f}x")
            daily_limit = _as_float(limits.get("daily_loss_limit"))
            if daily_limit is not None:
                limit_parts.append(f"dzienna strata {_format_percentage(daily_limit)}")
            drawdown_limit = _as_float(limits.get("drawdown_limit"))
            if drawdown_limit is not None:
                limit_parts.append(f"obsunięcie {_format_percentage(drawdown_limit)}")
            target_vol = _as_float(limits.get("target_volatility"))
            if target_vol is not None:
                limit_parts.append(f"target vol {_format_percentage(target_vol)}")
            stop_loss_atr = _as_float(limits.get("stop_loss_atr_multiple"))
            if stop_loss_atr is not None:
                limit_parts.append(f"stop loss ATR× {stop_loss_atr:.2f}")
            if limit_parts:
                risk_lines.append("Limity: " + ", ".join(limit_parts))

    # Dodatkowe linie o danych (manifest/cache), jeśli dołączono do summary
    data_lines: list[str] = []
    data_checks = summary.get("data_checks")
    if isinstance(data_checks, Mapping):
        manifest_info = data_checks.get("manifest")
        if isinstance(manifest_info, Mapping):
            entries = manifest_info.get("entries") or []
            if isinstance(entries, list):
                total_entries = len(entries)
                issues_count = 0
                for entry in entries:
                    issues = entry.get("issues")
                    if isinstance(issues, list) and any(issues):
                        issues_count += 1
                status_text = str(manifest_info.get("status", "n/a")).upper()
                if total_entries:
                    data_lines.append(
                        f"Manifest OHLCV: {status_text} ({total_entries} wpisów, problemy: {issues_count})"
                    )
                else:
                    data_lines.append(f"Manifest OHLCV: {status_text} (brak wpisów)")
        cache_info = data_checks.get("cache")
        if isinstance(cache_info, Mapping) and cache_info:
            fragments: list[str] = []
            for symbol, payload in sorted(cache_info.items()):
                fragment = str(symbol)
                if isinstance(payload, Mapping):
                    intervals_payload = payload.get("intervals")
                    if isinstance(intervals_payload, Mapping) and intervals_payload:
                        interval_parts: list[str] = []
                        for interval_name, interval_payload in sorted(intervals_payload.items()):
                            interval_fragment = str(interval_name)
                            if isinstance(interval_payload, Mapping):
                                coverage_int = _as_int(interval_payload.get("coverage_bars"))
                                required_int = _as_int(interval_payload.get("required_bars"))
                                row_count_int = _as_int(interval_payload.get("row_count"))
                                details: list[str] = []
                                if coverage_int is not None and required_int is not None:
                                    details.append(f"pokrycie {coverage_int}/{required_int}")
                                if row_count_int is not None:
                                    details.append(f"wiersze {row_count_int}")
                                if details:
                                    interval_fragment += " (" + ", ".join(details) + ")"
                            interval_parts.append(interval_fragment)
                        if interval_parts:
                            fragment += " [" + "; ".join(interval_parts) + "]"
                            fragments.append(fragment)
                            continue

                coverage_int = _as_int(payload.get("coverage_bars") if isinstance(payload, Mapping) else None)
                required_int = _as_int(payload.get("required_bars") if isinstance(payload, Mapping) else None)
                row_count_int = _as_int(payload.get("row_count") if isinstance(payload, Mapping) else None)
                if coverage_int is not None and required_int is not None:
                    fragment += f": pokrycie {coverage_int}/{required_int}"
                if row_count_int is not None:
                    fragment += f", wiersze {row_count_int}"
                fragments.append(fragment)
            if fragments:
                data_lines.append("Cache offline: " + "; ".join(fragments))

    # Info o magazynie raportu
    storage_lines: list[str] = []
    storage_info = summary.get("storage")
    if isinstance(storage_info, Mapping) and storage_info:
        status = str(storage_info.get("status", "unknown")).upper()
        free_mb = _as_float(storage_info.get("free_mb"))
        total_mb = _as_float(storage_info.get("total_mb"))
        threshold_mb = _as_float(storage_info.get("threshold_mb"))
        parts = [f"status={status}"]
        if free_mb is not None:
            parts.append(f"wolne {free_mb:.2f} MB")
        if total_mb is not None:
            parts.append(f"całkowite {total_mb:.2f} MB")
        if threshold_mb is not None:
            parts.append(f"próg {threshold_mb:.2f} MB")
        storage_lines.append("Magazyn raportu: " + ", ".join(parts))

    lines = [
        f"Środowisko: {environment}",
        f"Zakres dat: {start} → {end}",
        f"Liczba zleceń: {orders_count}",
        f"Liczba wpisów w ledgerze: {ledger_entries}",
    ]
    if metrics_lines:
        lines.extend(metrics_lines)
    if risk_lines:
        lines.extend(risk_lines)
    if data_lines:
        lines.extend(data_lines)
    if storage_lines:
        lines.extend(storage_lines)
    lines.append("Alerty: " + "; ".join(alert_lines))
    lines.append(f"SHA-256 summary.json: {summary_sha256}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Walidacja cache + manifest
# --------------------------------------------------------------------------------------
def _ensure_smoke_cache(
    *,
    pipeline: Any,
    symbols: Sequence[str],
    interval: str,
    start_ms: int,
    end_ms: int,
    required_bars: int,
    tick_ms: int,
) -> Mapping[str, object]:
    """Sprawdza, czy lokalny cache zawiera dane potrzebne do smoke testu."""
    required_intervals = _collect_required_intervals(pipeline, symbols=symbols)
    if not required_intervals:
        required_intervals = (interval,)

    normalized_primary = _normalize_interval_token(interval)
    tick_map: dict[str, int] = {}
    required_map: dict[str, int] = {}

    for candidate in required_intervals:
        normalized = _normalize_interval_token(candidate)
        if not normalized:
            continue
        if normalized == normalized_primary:
            tick_map[normalized] = max(1, int(tick_ms))
            required_map[normalized] = int(required_bars)
            continue
        try:
            candidate_tick_ms = _interval_to_milliseconds(candidate)
        except ValueError:
            _LOGGER.warning("Pominięto nieobsługiwany interwał manifestu: %s", candidate)
            continue
        tick_map[normalized] = candidate_tick_ms
        window_bars = max(1, int((end_ms - start_ms) / max(1, candidate_tick_ms)) + 2)
        required_map[normalized] = window_bars

    effective_intervals = [
        candidate
        for candidate in required_intervals
        if _normalize_interval_token(candidate) in tick_map
    ]
    if not effective_intervals:
        effective_intervals = [interval]
        tick_map.setdefault(normalized_primary or interval, max(1, int(tick_ms)))
        required_map.setdefault(normalized_primary or interval, int(required_bars))

    manifest_report = _verify_manifest_coverage(
        pipeline=pipeline,
        symbols=symbols,
        intervals=effective_intervals,
        end_ms=end_ms,
        required_bars_map=required_map,
    )

    data_source = getattr(pipeline, "data_source", None)
    storage = getattr(data_source, "storage", None)
    cache_reports: dict[str, dict[str, Mapping[str, object]]] = {}

    if storage is None:
        _LOGGER.warning(
            "Nie mogę zweryfikować cache – pipeline nie udostępnia storage'u. Pomijam kontrolę.",
        )
    else:
        try:
            metadata: MutableMapping[str, str] = storage.metadata()
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Nie udało się odczytać metadanych cache: %s", exc)
            metadata = {}

        for candidate in effective_intervals:
            normalized = _normalize_interval_token(candidate)
            if not normalized:
                continue

            candidate_tick_ms = tick_map.get(normalized, max(1, int(tick_ms)))
            candidate_required = required_map.get(normalized, int(required_bars))

            issues: list[tuple[str, str]] = []
            for symbol in symbols:
                key = f"{symbol}::{candidate}"
                row_count: int | None = None
                last_timestamp: int | None = None

                if metadata:
                    raw_rows = metadata.get(f"row_count::{symbol}::{candidate}")
                    if raw_rows is not None:
                        try:
                            row_count = int(raw_rows)
                        except (TypeError, ValueError):
                            _LOGGER.warning(
                                "Nieprawidłowa wartość row_count dla %s (%s): %s",
                                symbol,
                                candidate,
                                raw_rows,
                            )
                    raw_last = metadata.get(f"last_timestamp::{symbol}::{candidate}")
                    if raw_last is not None:
                        try:
                            last_timestamp = int(float(raw_last))
                        except (TypeError, ValueError):
                            _LOGGER.warning(
                                "Nieprawidłowa wartość last_timestamp dla %s (%s): %s",
                                symbol,
                                candidate,
                                raw_last,
                            )

                try:
                    payload = storage.read(key)
                except KeyError:
                    issues.append((str(symbol), "brak wpisu w cache"))
                    continue

                rows = list(payload.get("rows", []))
                if not rows:
                    issues.append((str(symbol), "puste dane w cache"))
                    continue

                if row_count is None:
                    row_count = len(rows)
                if last_timestamp is None:
                    last_timestamp = int(float(rows[-1][0]))

                first_timestamp = int(float(rows[0][0]))

                if row_count < candidate_required:
                    issues.append((str(symbol), f"za mało świec ({row_count} < {candidate_required})"))
                    continue

                if last_timestamp < end_ms:
                    issues.append((str(symbol), f"ostatnia świeca {last_timestamp} < wymaganego końca {end_ms}"))
                    continue

                if first_timestamp > start_ms:
                    issues.append((str(symbol), f"pierwsza świeca {first_timestamp} > wymaganego startu {start_ms}"))
                    continue

                coverage = ((last_timestamp - first_timestamp) // max(1, candidate_tick_ms)) + 1
                if coverage < candidate_required:
                    issues.append((str(symbol), f"pokrycie obejmuje {coverage} świec (wymagane {candidate_required})"))
                    continue

                symbol_entry = cache_reports.setdefault(str(symbol), {})
                interval_map = symbol_entry.setdefault("intervals", {})
                interval_map[str(candidate)] = {
                    "row_count": int(row_count),
                    "first_timestamp_ms": first_timestamp,
                    "last_timestamp_ms": last_timestamp,
                    "coverage_bars": int(coverage),
                    "required_bars": int(candidate_required),
                }

            if issues:
                for symbol_name, reason in issues:
                    _LOGGER.error(
                        "Cache offline dla symbolu %s (%s) nie spełnia wymagań smoke testu: %s",
                        symbol_name,
                        candidate,
                        reason,
                    )
                raise RuntimeError(
                    "Cache offline nie obejmuje wymaganego zakresu danych. Uruchom scripts/seed_paper_cache.py, "
                    "aby zbudować deterministyczny seed przed smoke testem.",
                )

    result: dict[str, object] = {
        "interval": interval,
        "intervals": [str(value) for value in effective_intervals],
        "symbols": [str(symbol) for symbol in symbols],
        "required_bars": int(required_bars),
        "tick_ms": int(max(1, tick_ms)),
        "window_ms": {"start": int(start_ms), "end": int(end_ms)},
        "required_bars_map": {key: int(value) for key, value in required_map.items()},
        "tick_ms_map": {key: int(value) for key, value in tick_map.items()},
    }
    if manifest_report:
        result["manifest"] = manifest_report
    if cache_reports:
        result["cache"] = cache_reports
    return result


def _verify_manifest_coverage(
    *,
    pipeline: Any,
    symbols: Sequence[str],
    intervals: Sequence[str],
    end_ms: int,
    required_bars_map: Mapping[str, int],
) -> Mapping[str, object] | None:
    """Waliduje metadane manifestu przed uruchomieniem smoke testu."""
    bootstrap = getattr(pipeline, "bootstrap", None)
    if bootstrap is None:
        return None

    environment_cfg = getattr(bootstrap, "environment", None)
    core_config = getattr(bootstrap, "core_config", None)
    if environment_cfg is None or core_config is None:
        return None
    if not hasattr(core_config, "instrument_universes"):
        return None

    universe_name = getattr(environment_cfg, "instrument_universe", None)
    cache_root = getattr(environment_cfg, "data_cache_path", None)
    exchange_name = getattr(environment_cfg, "exchange", None)
    if not universe_name or not cache_root or not exchange_name:
        return None

    manifest_path = Path(cache_root) / "ohlcv_manifest.sqlite"
    if not manifest_path.exists():
        _LOGGER.warning(
            "Manifest %s nie istnieje – pomijam kontrolę metadanych i sprawdzam wyłącznie surowe pliki.",
            manifest_path,
        )
        return None

    try:
        universe = core_config.instrument_universes[universe_name]
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning(
            "Nie udało się pobrać uniwersum instrumentów '%s' z konfiguracji: %s – pomijam kontrolę manifestu.",
            universe_name,
            exc,
        )
        return None

    as_of = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)
    try:
        statuses = evaluate_coverage(
            manifest_path=manifest_path,
            universe=universe,
            exchange_name=exchange_name,
            as_of=as_of,
        )
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("Nie udało się ocenić pokrycia manifestu: %s", exc)
        return None

    normalized_map: dict[str, str] = {}
    ordered_normalized: list[str] = []
    for candidate in intervals:
        normalized = _normalize_interval_token(candidate)
        if not normalized:
            continue
        if normalized not in required_bars_map:
            # pomiń interwały, których nie umiemy zmapować do wymagań
            continue
        if normalized not in normalized_map:
            normalized_map[normalized] = str(candidate)
            ordered_normalized.append(normalized)

    if not ordered_normalized:
        return None

    tracked_symbols = {str(symbol).lower() for symbol in symbols}
    status_by_key: dict[tuple[str, str], object] = {}
    for status in statuses:
        normalized = _normalize_interval_token(status.interval)
        if not normalized:
            continue
        status_by_key[(status.symbol.lower(), normalized)] = status

    issues: list[str] = []
    entries_payload: list[dict[str, object]] = []

    for symbol in symbols:
        symbol_str = str(symbol)
        symbol_key = symbol_str.lower()
        for normalized in ordered_normalized:
            display_interval = normalized_map[normalized]
            status = status_by_key.get((symbol_key, normalized))
            required_rows = required_bars_map.get(normalized, 0)
            if status is None:
                issues.append(
                    f"{symbol_str}/{display_interval}: manifest nie zawiera wpisu – uruchom scripts/seed_paper_cache.py."
                )
                continue

            entry = status.manifest_entry
            if status.issues:
                issues.extend(
                    _render_manifest_issue(status.symbol, status.interval, issue)
                    for issue in status.issues
                )

            row_count = entry.row_count
            if row_count is None:
                issues.append(
                    f"{status.symbol}/{status.interval}: manifest nie zawiera licznika świec (row_count)"
                )
            elif required_rows and row_count < required_rows:
                issues.append(
                    f"{status.symbol}/{status.interval}: manifest raportuje jedynie {row_count} świec (< {required_rows})"
                )

            last_ts = entry.last_timestamp_ms
            if last_ts is None:
                issues.append(
                    f"{status.symbol}/{status.interval}: manifest nie zawiera ostatniego stempla czasowego"
                )
            elif last_ts < end_ms:
                issues.append(
                    f"{status.symbol}/{status.interval}: ostatnia świeca w manifescie ({last_ts}) < wymaganego końca ({end_ms})"
                )

            entries_payload.append(
                {
                    "symbol": status.symbol,
                    "interval": status.interval,
                    "status": status.status,
                    "issues": list(status.issues),
                    "row_count": entry.row_count,
                    "required_rows": status.required_rows,
                    "gap_minutes": entry.gap_minutes,
                    "last_timestamp_ms": entry.last_timestamp_ms,
                    "last_timestamp_iso": entry.last_timestamp_iso,
                }
            )

    if issues:
        for detail in issues:
            _LOGGER.error("Manifest OHLCV: %s", detail)
        raise RuntimeError(
            "Manifest danych OHLCV jest niekompletny dla smoke testu. Uruchom scripts/seed_paper_cache.py lub pełny backfill, "
            "aby zaktualizować manifest."
        )

    required_rows_payload = {
        normalized_map[token]: int(required_bars_map[token]) for token in ordered_normalized
    }

    return {
        "status": "ok",
        "as_of": as_of.isoformat(),
        "intervals": [normalized_map[token] for token in ordered_normalized],
        "required_rows": required_rows_payload,
        "symbols": sorted(str(symbol) for symbol in symbols),
        "entries": entries_payload,
    }


def _render_manifest_issue(symbol: str, interval: str, issue: str) -> str:
    if issue.startswith("manifest_status:"):
        status = issue.split(":", 1)[1]
        return f"{symbol}/{interval}: status manifestu = {status}"
    if issue == "missing_row_count":
        return f"{symbol}/{interval}: manifest nie zawiera informacji o liczbie świec"
    if issue.startswith("insufficient_rows:"):
        payload = issue.split(":", 1)[1]
        return f"{symbol}/{interval}: manifest raportuje zbyt mało świec ({payload})"
    return f"{symbol}/{interval}: {issue}"


# --------------------------------------------------------------------------------------
# Adapter offline dla smoke testu
# --------------------------------------------------------------------------------------
class _OfflineExchangeAdapter(ExchangeAdapter):
    """Minimalny adapter giełdowy działający offline dla trybu paper-smoke."""

    name = "offline"

    def __init__(self, credentials: ExchangeCredentials, **_: object) -> None:
        super().__init__(credentials)

    def configure_network(self, *, ip_allowlist: tuple[str, ...] | None = None) -> None:  # noqa: D401, ARG002
        return None

    def fetch_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            balances={"USDT": 100_000.0},
            total_equity=100_000.0,
            available_margin=100_000.0,
            maintenance_margin=0.0,
        )

    def fetch_symbols(self):  # pragma: no cover
        return ()

    def fetch_ohlcv(  # noqa: D401, ARG002
        self,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        limit: int | None = None,
    ):
        return []

    def place_order(self, request):  # pragma: no cover
        raise NotImplementedError

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:  # pragma: no cover
        raise NotImplementedError

    def stream_public_data(self, *, channels):  # pragma: no cover
        raise NotImplementedError

    def stream_private_data(self, *, channels):  # pragma: no cover
        raise NotImplementedError


def _offline_adapter_factory(credentials: ExchangeCredentials, **kwargs: object) -> ExchangeAdapter:
    return _OfflineExchangeAdapter(credentials, **kwargs)


# --------------------------------------------------------------------------------------
# Pętla realtime
# --------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------
# Główna funkcja CLI
# --------------------------------------------------------------------------------------
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

    adapter_factories: Mapping[str, ExchangeAdapterFactory] | None = None
    if args.paper_smoke:
        adapter_factories = {
            "binance_spot": _offline_adapter_factory,
            "binance_futures": _offline_adapter_factory,
            "kraken_spot": _offline_adapter_factory,
            "kraken_futures": _offline_adapter_factory,
            "zonda_spot": _offline_adapter_factory,
        }

    try:
        pipeline = build_daily_trend_pipeline(
            environment_name=args.environment,
            strategy_name=args.strategy,
            controller_name=args.controller,
            config_path=config_path,
            secret_manager=secret_manager,
            adapter_factories=adapter_factories,
            risk_profile_name=args.risk_profile,
        )
    except Exception as exc:  # noqa: BLE001
        _LOGGER.exception("Nie udało się zbudować pipeline'u daily trend: %s", exc)
        return 1

    # Bezpieczne logowanie (mock/test może nie mieć pól)
    strategy_name = getattr(pipeline, "strategy_name", args.strategy)
    controller_name = getattr(pipeline, "controller_name", args.controller)
    _LOGGER.info(
        "Pipeline gotowy: środowisko=%s, strategia=%s, kontroler=%s",
        args.environment,
        strategy_name,
        controller_name,
    )

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

        end_dt = datetime.fromisoformat(window_meta["end"])
        tick_seconds = float(getattr(pipeline.controller, "tick_seconds", 86400.0) or 86400.0)
        tick_ms = max(1, int(tick_seconds * 1000))
        window_duration_ms = max(0, end_ms - start_ms)
        approx_bars = max(1, int(window_duration_ms / tick_ms) + 1)
        history_bars = max(1, min(int(args.history_bars), approx_bars))
        runner_start_ms = max(0, end_ms - history_bars * tick_ms)
        sync_start = min(start_ms, runner_start_ms)

        _LOGGER.info(
            "Startuję smoke test paper trading dla %s w zakresie %s – %s.",
            args.environment,
            window_meta["start"],
            window_meta["end"],
        )

        required_bars = max(history_bars, max(1, int((end_ms - sync_start) / tick_ms) + 1))
        data_checks: Mapping[str, object] | None = None
        try:
            data_checks = _ensure_smoke_cache(
                pipeline=pipeline,
                symbols=pipeline.controller.symbols,
                interval=pipeline.controller.interval,
                start_ms=sync_start,
                end_ms=end_ms,
                required_bars=required_bars,
                tick_ms=tick_ms,
            )
        except RuntimeError as exc:
            _LOGGER.error("%s", exc)
            return 1

        pipeline.backfill_service.synchronize(
            symbols=pipeline.controller.symbols,
            interval=pipeline.controller.interval,
            start=sync_start,
            end=end_ms,
        )

        trading_controller = create_trading_controller(
            pipeline, pipeline.bootstrap.alert_router, health_check_interval=0.0,
        )

        runner = DailyTrendRealtimeRunner(
            controller=pipeline.controller,
            trading_controller=trading_controller,
            history_bars=history_bars,
            clock=lambda end=end_dt: end,
        )

        results = runner.run_once()
        if results:
            _log_order_results(results)
        else:
            _LOGGER.info("Smoke test zakończony – brak sygnałów w zadanym oknie.")

        report_dir = _prepare_smoke_report_directory(args.smoke_output)
        storage_info = _collect_storage_health(report_dir, min_free_mb=args.smoke_min_free_mb)
        alert_snapshot = pipeline.bootstrap.alert_router.health_snapshot()

        # Snapshot stanu ryzyka (opcjonalnie)
        risk_snapshot: Mapping[str, object] | None = None
        try:
            risk_engine = getattr(pipeline.bootstrap, "risk_engine", None)
            if risk_engine is not None and hasattr(risk_engine, "snapshot_state"):
                risk_snapshot = risk_engine.snapshot_state(pipeline.risk_profile_name)
        except NotImplementedError:
            _LOGGER.warning("Silnik ryzyka nie udostępnia metody snapshot_state – pomijam stan ryzyka")
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Nie udało się pobrać stanu ryzyka: %s", exc)

        summary_path = _export_smoke_report(
            report_dir=report_dir,
            results=results,
            ledger=pipeline.execution_service.ledger(),
            window=window_meta,
            environment=args.environment,
            alert_snapshot=alert_snapshot,
            risk_state=risk_snapshot,
            data_checks=data_checks,
            storage_info=storage_info,
        )
        summary_hash = _hash_file(summary_path)
        try:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            _LOGGER.error("Nie udało się odczytać summary.json: %s", exc)
            summary_payload = {
                "environment": args.environment,
                "window": dict(window_meta),
                "orders": [],
                "ledger_entries": 0,
                "alert_snapshot": alert_snapshot,
                "risk_state": risk_snapshot or {},
            }

        summary_text = _render_smoke_summary(summary=summary_payload, summary_sha256=summary_hash)
        summary_txt_path = summary_path.with_suffix(".txt")
        summary_txt_path.write_text(summary_text + "\n", encoding="utf-8")
        readme_path = _write_smoke_readme(report_dir)
        _LOGGER.info("Raport smoke testu zapisany w %s (summary sha256=%s)", report_dir, summary_hash)
        _LOGGER.info("Podsumowanie smoke testu:%s%s", os.linesep, summary_text)

        archive_path: Path | None = None
        archive_required = bool(args.archive_smoke or (SmokeArchiveUploader.resolve_config(
            getattr(getattr(pipeline.bootstrap, "core_config", None), "reporting", None)
        )))
        upload_cfg = SmokeArchiveUploader.resolve_config(
            getattr(getattr(pipeline.bootstrap, "core_config", None), "reporting", None)
        )

        if archive_required:
            archive_path = _archive_smoke_report(report_dir)
            if args.archive_smoke:
                _LOGGER.info("Utworzono archiwum smoke testu: %s", archive_path)
            else:
                _LOGGER.info("Archiwum smoke testu wygenerowane automatycznie na potrzeby uploadu: %s", archive_path)

        upload_result = None
        if upload_cfg and archive_path:
            try:
                uploader = SmokeArchiveUploader(upload_cfg, secret_manager=secret_manager)
                upload_result = uploader.upload(
                    archive_path,
                    environment=args.environment,
                    summary_sha256=summary_hash,
                    window=window_meta,
                )
                _LOGGER.info("Przesłano archiwum smoke testu (%s) do %s", upload_result.backend, upload_result.location)
            except Exception as exc:  # noqa: BLE001
                _LOGGER.error("Nie udało się przesłać archiwum smoke testu: %s", exc)

        storage_context: dict[str, str] = {}
        storage_status = None
        if isinstance(storage_info, Mapping):
            storage_status = str(storage_info.get("status", ""))
            storage_context = {"storage_status": storage_status}
            free_mb = storage_info.get("free_mb")
            if free_mb is not None:
                storage_context["storage_free_mb"] = f"{float(free_mb):.2f}"
            threshold_mb = storage_info.get("threshold_mb")
            if threshold_mb is not None:
                storage_context["storage_threshold_mb"] = f"{float(threshold_mb):.2f}"

        storage_status_lower = storage_status.lower() if storage_status else ""
        fail_low_storage = bool(args.smoke_fail_on_low_space and storage_status_lower == "low")

        body = (
            "Zakończono smoke test paper trading."
            f" Zamówienia: {len(results)}, raport: {summary_path},"
            f" sha256: {summary_hash}"
        )
        if storage_status_lower == "low":
            free_str = storage_context.get("storage_free_mb")
            thresh_str = storage_context.get("storage_threshold_mb")
            if free_str and thresh_str:
                body += f" Ostrzeżenie: wolne miejsce {free_str} MB poniżej progu {thresh_str} MB."
            elif free_str:
                body += f" Ostrzeżenie: niskie wolne miejsce ({free_str} MB)."

        severity = "warning" if storage_status_lower == "low" else "info"

        message = AlertMessage(
            category="paper_smoke",
            title=f"Smoke test paper trading ({args.environment})",
            body=body,
            severity=severity,
            context={
                "environment": args.environment,
                "report_dir": str(report_dir),
                "orders": str(len(results)),
                "summary_sha256": summary_hash,
                "summary_text_path": str(summary_txt_path),
                "readme_path": str(readme_path),
                **({"archive_path": str(archive_path)} if archive_path else {}),
                **(
                    {
                        "archive_upload_backend": upload_result.backend,
                        "archive_upload_location": upload_result.location,
                    }
                    if upload_result
                    else {}
                ),
                **storage_context,
            },
        )
        pipeline.bootstrap.alert_router.dispatch(message)
        if fail_low_storage:
            free_str = storage_context.get("storage_free_mb", "?")
            thresh_str = storage_context.get("storage_threshold_mb", str(args.smoke_min_free_mb or "?"))
            _LOGGER.error(
                "Smoke test zakończony niepowodzeniem: wolne miejsce %s MB poniżej wymaganego progu %s MB.",
                free_str,
                thresh_str,
            )
            return 4
        return 0

    # normalny tryb realtime / run-once
    trading_controller = create_trading_controller(
        pipeline, pipeline.bootstrap.alert_router, health_check_interval=args.health_interval,
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
