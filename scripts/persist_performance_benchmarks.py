"""Aggregate CI performance reports into rolling Parquet/SQLite datasets.

The script merges JSON reports emitted by performance tests (UI render, backtest
throughput and per-strategy benchmarks), normalizes them into a tabular schema
and enforces a rolling window to keep artifacts small. It preserves commit SHA
and timestamps for downstream correlation in dashboards.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ReportPayload:
    source: str
    scenario: str
    dataset: str | None
    timeframe: str | None
    pair_count: int | None
    bars_per_second: float | None
    pairs_per_second: float | None
    avg_ms: float | None
    p90_ms: float | None
    git_commit: str
    timestamp_utc: str

    @property
    def as_dict(self) -> dict[str, object | None]:
        return {
            "source": self.source,
            "scenario": self.scenario,
            "dataset": self.dataset,
            "timeframe": self.timeframe,
            "pair_count": self.pair_count,
            "bars_per_second": self.bars_per_second,
            "pairs_per_second": self.pairs_per_second,
            "avg_ms": self.avg_ms,
            "p90_ms": self.p90_ms,
            "git_commit": self.git_commit,
            "timestamp_utc": self.timestamp_utc,
        }


def _load_json_reports(report_roots: Iterable[Path]) -> list[Path]:
    reports: list[Path] = []
    for root in report_roots:
        if root.exists():
            reports.extend(sorted(root.rglob("*.json")))
    return reports


def _normalize_payload(report_path: Path, payload: dict) -> ReportPayload:
    if "performance_backtests" in report_path.parts:
        source = "backtest"
        scenario = f"backtests_{payload.get('pair_count', 'unknown')}pairs"
        dataset = None
        timeframe = str(payload.get("timeframe")) if payload.get("timeframe") else None
        pair_count = int(payload.get("pair_count")) if payload.get("pair_count") is not None else None
        bars_per_second = None
        pairs_per_second = float(payload.get("pairs_per_second", 0))
        avg_ms = float(payload.get("avg_ms", 0.0)) if payload.get("avg_ms") is not None else None
        p90_ms = float(payload.get("p90_ms", 0.0)) if payload.get("p90_ms") is not None else None
    elif "benchmark_backtests" in report_path.parts:
        source = "backtest_benchmark"
        scenario = str(payload.get("scenario", payload.get("dataset", "unknown")))
        dataset = str(payload.get("dataset", "")) or None
        timeframe = str(payload.get("timeframe", "")) or None
        pair_count = None
        bars_per_second = float(payload.get("bars_per_second", 0))
        pairs_per_second = None
        avg_ms = float(payload.get("avg_ms", 0.0)) if payload.get("avg_ms") is not None else None
        p90_ms = float(payload.get("p95_ms", 0.0)) if payload.get("p95_ms") is not None else None
    else:
        source = "ui_render"
        scenario = str(payload.get("scenario", "unknown"))
        dataset = None
        timeframe = None
        pair_count = None
        bars_per_second = None
        pairs_per_second = None
        avg_ms = float(payload.get("avg_ms", 0.0)) if payload.get("avg_ms") is not None else None
        p90_ms = float(payload.get("p90_ms", 0.0)) if payload.get("p90_ms") is not None else None

    git_commit = str(payload.get("git_commit", "unknown"))
    timestamp = str(payload.get("timestamp_utc", "")) or ""

    return ReportPayload(
        source=source,
        scenario=scenario,
        dataset=dataset,
        timeframe=timeframe,
        pair_count=pair_count,
        bars_per_second=bars_per_second,
        pairs_per_second=pairs_per_second,
        avg_ms=avg_ms,
        p90_ms=p90_ms,
        git_commit=git_commit,
        timestamp_utc=timestamp,
    )


def _load_existing_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _apply_window(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    if df.empty:
        return df
    df_sorted = df.sort_values(by=["source", "scenario", "timestamp_utc", "git_commit"])
    return (
        df_sorted.groupby(["source", "scenario"], as_index=False, sort=False)
        .tail(window_size)
        .reset_index(drop=True)
    )


def _persist_sqlite(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        df.to_sql("benchmark_samples", conn, if_exists="replace", index=False)


def _persist_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-root",
        action="append",
        type=Path,
        dest="report_roots",
        help="Ścieżka do katalogu z raportami JSON (może wystąpić wielokrotnie).",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        required=True,
        help="Plik wyjściowy Parquet.",
    )
    parser.add_argument(
        "--sqlite",
        type=Path,
        required=True,
        help="Plik wyjściowy SQLite.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=120,
        help="Maksymalna liczba rekordów utrzymywanych per źródło/scenariusz.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report_roots = args.report_roots or []
    reports = _load_json_reports(report_roots)

    payloads: list[ReportPayload] = []
    for report_path in reports:
        try:
            raw = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive fallback in CI
            print(f"[warn] Pomijam raport {report_path}: {exc}")
            continue
        payloads.append(_normalize_payload(report_path, raw))

    df_new = pd.DataFrame([payload.as_dict for payload in payloads])
    df_existing = _load_existing_dataframe(args.parquet)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_windowed = _apply_window(df_combined, args.window_size)

    _persist_parquet(df_windowed, args.parquet)
    _persist_sqlite(df_windowed, args.sqlite)

    print(
        f"Zapisano {len(df_windowed)} rekordów (window={args.window_size}) "
        f"do {args.parquet} oraz {args.sqlite}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
