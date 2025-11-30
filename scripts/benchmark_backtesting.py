"""Benchmark throughput i latencji backtestów dla kluczowych strategii."""
from __future__ import annotations

import argparse
import datetime
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - ścieżka uruchomieniowa
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.data import BacktestDatasetLibrary  # noqa: E402


_BASELINES = {
    "mean_reversion": {"bars_per_second": 750.0, "p95_ms": 5.0},
    "volatility_target": {"bars_per_second": 650.0, "p95_ms": 5.0},
    "cross_exchange_arbitrage": {"bars_per_second": 600.0, "p95_ms": 6.0},
}


@dataclass(slots=True)
class BenchmarkResult:
    dataset: str
    timeframe: str
    bars: int
    duration_seconds: float
    bars_per_second: float
    avg_ms: float
    p95_ms: float
    sla_bars_per_second: float
    sla_p95_ms: float
    regression_budget: float

    @property
    def is_regressing(self) -> bool:
        throughput_floor = self.sla_bars_per_second * (1.0 - self.regression_budget)
        latency_ceiling = self.sla_p95_ms * (1.0 + self.regression_budget)
        return self.bars_per_second < throughput_floor or self.p95_ms > latency_ceiling


def _load_dataframe(library: BacktestDatasetLibrary, dataset: str) -> pd.DataFrame:
    descriptor = library.describe(dataset)
    frame = library.load_dataframe(
        dataset,
        index_column="timestamp",
        datetime_columns={"timestamp": "s"},
    )
    frame = frame.sort_index()
    frame.index = pd.to_datetime(frame.index, utc=True)
    if {"open", "high", "low", "close"}.issubset(frame.columns):
        return frame
    # Uzupełnienie braków dla prostych strategii: w razie braku kolumn OHLC
    frame = frame.rename(columns={"base_bid": "open", "quote_ask": "close"})
    for column in ("open", "high", "low", "close"):
        if column not in frame.columns:
            frame[column] = frame["open"] if "open" in frame.columns else frame.iloc[:, 0]
    return frame


def _simulate_strategy_step(strategy: str, row: Mapping[str, object]) -> float:
    start = time.perf_counter()
    # Proste kalkulacje przypominające profile CPU/GPU realnych strategii.
    price = float(row.get("close", row.get("open", 0.0)))
    volume = float(row.get("volume", 0.0))
    if strategy == "mean_reversion":
        zscore = float(row.get("z_score", 0.0))
        _ = (price * 0.5 + zscore * 2.5) / (abs(volume) + 1.0)
    elif strategy == "volatility_target":
        realized_vol = float(row.get("realized_volatility", 0.0))
        target_vol = float(row.get("target_volatility", 0.0) or 0.2)
        _ = (realized_vol / max(target_vol, 1e-6)) * price
    else:
        spread = float(row.get("quote_ask", price)) - float(row.get("quote_bid", price))
        available = float(row.get("available_volume", volume))
        _ = spread * available
    return (time.perf_counter() - start) * 1000.0


def _p95(latencies_ms: Iterable[float]) -> float:
    samples = sorted(latencies_ms)
    if not samples:
        return 0.0
    index = max(0, int(len(samples) * 0.95) - 1)
    return samples[index]


def _run_benchmark(
    *,
    dataset: str,
    timeframe: str,
    frame: pd.DataFrame,
    regression_budget: float,
) -> BenchmarkResult:
    latencies: list[float] = []
    start = time.perf_counter()
    for _, row in frame.iterrows():
        latencies.append(_simulate_strategy_step(dataset, row))
    duration = time.perf_counter() - start
    bars = len(frame)
    throughput = bars / duration if duration > 0 else float("inf")
    baseline = _BASELINES.get(dataset, {"bars_per_second": 500.0, "p95_ms": 6.0})
    return BenchmarkResult(
        dataset=dataset,
        timeframe=timeframe,
        bars=bars,
        duration_seconds=duration,
        bars_per_second=throughput,
        avg_ms=statistics.mean(latencies) if latencies else 0.0,
        p95_ms=_p95(latencies),
        sla_bars_per_second=float(baseline["bars_per_second"]),
        sla_p95_ms=float(baseline["p95_ms"]),
        regression_budget=regression_budget,
    )


def _log_report(result: BenchmarkResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        .strip()
    )
    payload = {
        "scenario": result.dataset,
        "dataset": result.dataset,
        "timeframe": result.timeframe,
        "bars": result.bars,
        "duration_seconds": result.duration_seconds,
        "bars_per_second": result.bars_per_second,
        "avg_ms": result.avg_ms,
        "p95_ms": result.p95_ms,
        "sla_bars_per_second": result.sla_bars_per_second,
        "sla_p95_ms": result.sla_p95_ms,
        "regression_budget": result.regression_budget,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "git_commit": commit,
        "regression": result.is_regressing,
    }
    (output_dir / f"backtest_benchmark_{result.dataset}.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/backtests/normalized/manifest.yaml"),
        help="Ścieżka do manifestu znormalizowanych danych backtestowych.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/ci/benchmark_backtests"),
        help="Katalog na raporty JSON z benchmarków.",
    )
    parser.add_argument(
        "--regression-budget",
        type=float,
        default=0.10,
        help="Dopuszczalna regresja (procentowo) względem SLA bazowych.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    library = BacktestDatasetLibrary(args.manifest)
    results: list[BenchmarkResult] = []
    failing: list[BenchmarkResult] = []

    for dataset in library.list_dataset_names():
        frame = _load_dataframe(library, dataset)
        descriptor = library.describe(dataset)
        result = _run_benchmark(
            dataset=dataset,
            timeframe=descriptor.interval,
            frame=frame,
            regression_budget=args.regression_budget,
        )
        _log_report(result, args.output_dir)
        results.append(result)
        if result.is_regressing:
            failing.append(result)

    summary = {
        "benchmarks": [result.dataset for result in results],
        "regressions": [result.dataset for result in failing],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))

    if failing:
        details = ", ".join(result.dataset for result in failing)
        raise SystemExit(
            f"Regresja wydajności backtestów przekracza budżet {args.regression_budget:.0%}: {details}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
