"""Generowanie dziennych raportów z symulatora paper tradingu."""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from zipfile import ZIP_DEFLATED, ZipFile

from bot_core.execution.paper import PaperTradingExecutionService
from bot_core.runtime.journal import TradingDecisionJournal


@dataclass(slots=True)
class PaperReportArtifacts:
    """Ścieżki do wygenerowanych plików w archiwum raportu."""

    archive_path: Path
    ledger_rows: int
    decision_events: int


def _ensure_timezone(value: tzinfo | None) -> tzinfo:
    return value or timezone.utc


def _normalize_timestamp(value: object) -> datetime | None:
    try:
        ts = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(ts, timezone.utc)


def _parse_iso_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc)


def _time_window(report_date: date, tz: tzinfo) -> tuple[datetime, datetime]:
    start_local = datetime.combine(report_date, dt_time.min, tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def _ledger_header() -> Sequence[str]:
    return (
        "timestamp_utc",
        "timestamp_local",
        "order_id",
        "symbol",
        "side",
        "quantity",
        "price",
        "fee",
        "fee_asset",
        "status",
        "leverage",
        "position_value",
    )


def _format_number(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "0"
    return f"{number:.10f}".rstrip("0").rstrip(".") if number else "0"


def _writerows(
    rows: Sequence[tuple[Mapping[str, object], datetime]],
    *,
    tz: tzinfo,
) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(_ledger_header())
    for payload, timestamp in rows:
        local_ts = timestamp.astimezone(tz)
        writer.writerow(
            (
                timestamp.isoformat(),
                local_ts.isoformat(),
                str(payload.get("order_id", "")),
                str(payload.get("symbol", "")),
                str(payload.get("side", "")),
                _format_number(payload.get("quantity")),
                _format_number(payload.get("price")),
                _format_number(payload.get("fee")),
                str(payload.get("fee_asset", "")),
                str(payload.get("status", "")),
                _format_number(payload.get("leverage")),
                _format_number(payload.get("position_value")),
            )
        )
    return buffer.getvalue()


def _filter_ledger(
    entries: Iterable[Mapping[str, object]],
    *,
    start: datetime,
    end: datetime,
) -> list[tuple[Mapping[str, object], datetime]]:
    filtered: list[tuple[Mapping[str, object], datetime]] = []
    for entry in entries:
        timestamp = _normalize_timestamp(entry.get("timestamp"))
        if timestamp is None or not (start <= timestamp < end):
            continue
        filtered.append((entry, timestamp))
    filtered.sort(key=lambda item: item[1])
    return filtered


def _filter_decisions(
    journal: TradingDecisionJournal,
    *,
    start: datetime,
    end: datetime,
) -> list[Mapping[str, object]]:
    selected: list[Mapping[str, object]] = []
    for record in journal.export():
        if not isinstance(record, Mapping):
            continue
        timestamp = _parse_iso_timestamp(record.get("timestamp"))
        if timestamp is None or not (start <= timestamp < end):
            continue
        selected.append(record)
    selected.sort(key=lambda item: _parse_iso_timestamp(item.get("timestamp")) or start)
    return selected


def _summary_payload(
    *,
    report_date: date,
    tz: tzinfo,
    ledger_rows: Sequence[tuple[Mapping[str, object], datetime]],
    decision_events: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    total_notional = 0.0
    total_fees = 0.0
    for payload, _ in ledger_rows:
        try:
            quantity = float(payload.get("quantity", 0.0))
        except (TypeError, ValueError):
            quantity = 0.0
        try:
            price = float(payload.get("price", 0.0))
        except (TypeError, ValueError):
            price = 0.0
        try:
            fee = float(payload.get("fee", 0.0))
        except (TypeError, ValueError):
            fee = 0.0
        total_notional += quantity * price
        total_fees += fee

    timezone_name = tz.tzname(datetime.combine(report_date, dt_time.min, tzinfo=tz)) or str(tz)

    return {
        "report_date": report_date.isoformat(),
        "timezone": timezone_name,
        "ledger_rows": len(ledger_rows),
        "decision_events": len(decision_events),
        "traded_notional": round(total_notional, 8),
        "fees_paid": round(total_fees, 8),
    }


def generate_daily_paper_report(
    *,
    execution_service: PaperTradingExecutionService,
    output_dir: str | Path,
    decision_journal: TradingDecisionJournal | None = None,
    report_date: date | None = None,
    tz: tzinfo | None = None,
    include_summary: bool = True,
    ledger_entries: Iterable[Mapping[str, object]] | None = None,
) -> PaperReportArtifacts:
    """Generuje archiwum ZIP z dziennym blotterem i (opcjonalnie) dziennikiem decyzji."""
    tzinfo_obj = _ensure_timezone(tz)
    report_day = report_date or datetime.now(tzinfo_obj).date()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    window_start, window_end = _time_window(report_day, tzinfo_obj)

    # Pozwala wstrzyknąć wpisy ledgeru (np. w testach) lub pobrać je z serwisu wykonawczego.
    ledger_source = execution_service.ledger() if ledger_entries is None else ledger_entries
    ledger_filtered = _filter_ledger(ledger_source, start=window_start, end=window_end)

    decisions_filtered: list[Mapping[str, object]] = []
    if decision_journal is not None:
        decisions_filtered = _filter_decisions(decision_journal, start=window_start, end=window_end)

    csv_payload = _writerows(ledger_filtered, tz=tzinfo_obj)

    archive_name = f"paper-report-{report_day.isoformat()}.zip"
    archive_path = output_path / archive_name

    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("ledger.csv", csv_payload)
        if decisions_filtered:
            jsonl_payload = "\n".join(
                json.dumps(entry, separators=(",", ":"), ensure_ascii=False)
                for entry in decisions_filtered
            )
            archive.writestr("decisions.jsonl", jsonl_payload + "\n")
        if include_summary:
            summary = _summary_payload(
                report_date=report_day,
                tz=tzinfo_obj,
                ledger_rows=ledger_filtered,
                decision_events=decisions_filtered,
            )
            archive.writestr(
                "summary.json",
                json.dumps(summary, ensure_ascii=False, separators=(",", ":")),
            )

    return PaperReportArtifacts(
        archive_path=archive_path,
        ledger_rows=len(ledger_filtered),
        decision_events=len(decisions_filtered),
    )


__all__ = ["generate_daily_paper_report", "PaperReportArtifacts"]
