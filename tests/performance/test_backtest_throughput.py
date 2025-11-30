from __future__ import annotations

import datetime
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable
import pandas as pd
import pytest

from bot_core.backtest.engine import BacktestEngine
from bot_core.backtest.simulation import MatchingConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "reports/ci/performance_backtests"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

_SLA_THROUGHPUT = {
    (2, "1m"): 1.5,
    (4, "5m"): 1.0,
    (6, "1h"): 0.6,
}


class _ThroughputStrategy:
    def __init__(self) -> None:
        self._counter = 0

    async def prepare(self, *_args, **_kwargs) -> None:  # pragma: no cover - setup helper
        return None

    async def handle_market_data(self, *_args, **_kwargs) -> SimpleNamespace:
        self._counter += 1
        if self._counter % 25 == 0:
            return SimpleNamespace(action="BUY", size=0.1, stop_loss=None, take_profit=None)
        return SimpleNamespace(action="HOLD", size=0.0, stop_loss=None, take_profit=None)

    async def notify_fill(self, *_args, **_kwargs) -> None:  # pragma: no cover - not exercised
        return None

    async def shutdown(self) -> None:  # pragma: no cover - teardown helper
        return None


def _context_builder(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(**payload)


def _timeframe_to_freq(timeframe: str) -> str:
    mapping = {"1m": "1min", "5m": "5min", "1h": "1h"}
    return mapping.get(timeframe, "1min")


def _build_dataframe(*, timeframe: str, rows: int = 720) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq=_timeframe_to_freq(timeframe))
    base = pd.Series(range(rows), index=index, dtype=float)
    data = pd.DataFrame(
        {
            "open": 100.0 + base * 0.01,
            "high": 100.1 + base * 0.01,
            "low": 99.9 + base * 0.01,
            "close": 100.0 + base * 0.02,
            "volume": 10_000 + base,
        }
    )
    return data


def _run_backtest_pair(symbol: str, timeframe: str) -> dict:
    engine = BacktestEngine(
        strategy_factory=_ThroughputStrategy,
        context_builder=_context_builder,
        data=_build_dataframe(timeframe=timeframe),
        symbol=symbol,
        timeframe=timeframe,
        initial_balance=100_000.0,
        matching=MatchingConfig(),
    )
    report = engine.run()
    return {
        "trades": len(report.trades),
        "equity_points": len(report.equity_curve),
        "final_balance": report.final_balance,
    }


def _log_report(pair_count: int, timeframe: str, duration_s: float, throughput: float, runs: Iterable[dict]) -> None:
    commit = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        .strip()
    )
    payload = {
        "pair_count": pair_count,
        "timeframe": timeframe,
        "duration_seconds": duration_s,
        "pairs_per_second": throughput,
        "runs": list(runs),
        "sla_pairs_per_second": _SLA_THROUGHPUT[(pair_count, timeframe)],
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "git_commit": commit,
    }
    output_path = REPORT_DIR / f"backtests_{pair_count}pairs_{timeframe}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@pytest.mark.performance
@pytest.mark.parametrize("pair_count,timeframe", list(_SLA_THROUGHPUT))
def test_backtest_throughput(pair_count: int, timeframe: str) -> None:
    start = time.perf_counter()
    runs = [_run_backtest_pair(f"PAIR{idx}/USDT", timeframe) for idx in range(pair_count)]
    duration_s = time.perf_counter() - start
    throughput = pair_count / duration_s if duration_s > 0 else float("inf")

    _log_report(pair_count, timeframe, duration_s, throughput, runs)

    sla = _SLA_THROUGHPUT[(pair_count, timeframe)]
    assert throughput >= sla, (
        f"Backtest throughput below SLA for {pair_count} pairs @ {timeframe}: "
        f"{throughput:.2f} pairs/s < {sla} pairs/s"
    )
