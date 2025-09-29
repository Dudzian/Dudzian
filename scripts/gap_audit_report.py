"""Raportuje stan luk danych na podstawie logów audytowych JSONL."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

from bot_core.data.ohlcv import GapAuditRecord


@dataclass(slots=True)
class GapSummary:
    exchange: str
    symbol: str
    interval: str
    last_record: GapAuditRecord
    warning_count: int
    incident_count: int
    sms_count: int

    @property
    def severity(self) -> int:
        order = {
            "sms_escalated": 0,
            "incident": 1,
            "missing_metadata": 1,
            "invalid_metadata": 1,
            "warning": 2,
            "ok": 3,
        }
        return order.get(self.last_record.status, 4)


def load_records(
    path: str | Path,
    *,
    environment: str | None = None,
    exchange: str | None = None,
    since_hours: float | None = None,
) -> list[GapAuditRecord]:
    """Wczytuje wpisy audytowe filtrując środowisko, giełdę i zakres czasu."""

    audit_path = Path(path)
    if not audit_path.exists():
        raise SystemExit(f"Plik audytu {audit_path} nie istnieje")

    threshold: datetime | None = None
    if since_hours is not None and since_hours > 0:
        threshold = datetime.now(timezone.utc) - timedelta(hours=float(since_hours))

    records: list[GapAuditRecord] = []
    with audit_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = GapAuditRecord.from_json(line)
            except ValueError:
                continue
            if environment and record.environment != environment:
                continue
            if exchange and record.exchange != exchange:
                continue
            if threshold and record.timestamp < threshold:
                continue
            records.append(record)

    return records


def summarize_records(
    records: Sequence[GapAuditRecord],
    *,
    window_hours: float = 24.0,
) -> list[GapSummary]:
    """Buduje podsumowanie per symbol/interwał na podstawie wpisów."""

    if not records:
        return []

    now = datetime.now(timezone.utc)
    threshold = now - timedelta(hours=max(0.0, window_hours)) if window_hours > 0 else None

    summaries: dict[tuple[str, str, str], GapSummary] = {}
    for record in records:
        key = (record.exchange, record.symbol, record.interval)
        summary = summaries.get(key)
        if summary is None:
            summary = GapSummary(
                exchange=record.exchange,
                symbol=record.symbol,
                interval=record.interval,
                last_record=record,
                warning_count=0,
                incident_count=0,
                sms_count=0,
            )
            summaries[key] = summary
        else:
            if record.timestamp >= summary.last_record.timestamp:
                summary.last_record = record

        if threshold is None or record.timestamp >= threshold:
            if record.status == "warning":
                summary.warning_count += 1
            elif record.status in {"incident", "missing_metadata", "invalid_metadata"}:
                summary.incident_count += 1
            elif record.status == "sms_escalated":
                summary.sms_count += 1

    return sorted(
        summaries.values(),
        key=lambda item: (item.severity, -(item.last_record.gap_minutes or -1.0), item.symbol),
    )


def _format_value(value: object | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def format_summary_table(summaries: Sequence[GapSummary]) -> str:
    """Formatuje listę podsumowań do tabeli tekstowej."""

    headers = (
        "Exchange",
        "Symbol",
        "Interval",
        "Status",
        "Gap[min]",
        "Incident[min]",
        "Rows",
        "Last candle",
        "Warn(24h)",
        "Inc(24h)",
        "SMS(24h)",
        "Last event",
    )

    rows: list[tuple[str, ...]] = []
    for summary in summaries:
        record = summary.last_record
        rows.append(
            (
                summary.exchange,
                summary.symbol,
                summary.interval,
                record.status,
                _format_value(record.gap_minutes),
                _format_value(record.incident_minutes),
                _format_value(record.row_count),
                record.last_timestamp or "-",
                str(summary.warning_count),
                str(summary.incident_count),
                str(summary.sms_count),
                record.timestamp.isoformat(),
            )
        )

    columns = list(zip(headers, *rows)) if rows else [(header,) for header in headers]
    widths = [max(len(str(value)) for value in column) for column in columns]

    def _format_row(values: Iterable[str]) -> str:
        return "  ".join(value.ljust(width) for value, width in zip(values, widths))

    lines = [_format_row(headers)]
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append(_format_row(row))
    if not rows:
        lines.append("(Brak wpisów spełniających kryteria)")
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raport luk danych OHLCV na podstawie pliku audytu JSONL")
    parser.add_argument("audit_file", help="Ścieżka do pliku JSONL z logami audytu luk")
    parser.add_argument("--environment", help="Filtruj wpisy po nazwie środowiska")
    parser.add_argument("--exchange", help="Filtruj wpisy po nazwie giełdy")
    parser.add_argument(
        "--since-hours",
        type=float,
        default=None,
        help="Weź pod uwagę tylko wpisy młodsze niż podana liczba godzin",
    )
    parser.add_argument(
        "--window-hours",
        type=float,
        default=24.0,
        help="Okno czasowe (w godzinach) do zliczania ostrzeżeń/incydentów",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    records = load_records(
        args.audit_file,
        environment=args.environment,
        exchange=args.exchange,
        since_hours=args.since_hours,
    )

    summaries = summarize_records(records, window_hours=args.window_hours)
    table = format_summary_table(summaries)
    print(table)
    return 0


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    sys.exit(main())

