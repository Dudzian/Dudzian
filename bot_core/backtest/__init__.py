from __future__ import annotations

from .engine import (
    BacktestEngine,
    BacktestError,
    BacktestReport,
    BacktestTrade,
    HistoricalDataProvider,
    PerformanceMetrics,
)
from .ma import Bar as MABar, Trade as MATrade, param_grid_fast_slow, simulate_trades_ma
from .metrics import MetricsResult, compute_metrics, to_dict
from .reporting import export_report, render_html_report
from .simulation import BacktestFill, MatchingConfig, MatchingEngine
from .trade_loader import load_trades
from .trend_following import (
    DEFAULT_FEE,
    MIN_SL_PCT,
    BacktestConfig,
    EntryParams,
    ExchangeLike,
    ExitParams,
    StrategyParams,
    TradeFill,
    TradeRecord,
    TrendBacktestEngine,
)

__all__ = [
    "BacktestEngine",
    "BacktestError",
    "BacktestFill",
    "BacktestReport",
    "BacktestTrade",
    "BacktestConfig",
    "DEFAULT_FEE",
    "EntryParams",
    "ExchangeLike",
    "HistoricalDataProvider",
    "load_trades",
    "MABar",
    "MATrade",
    "MetricsResult",
    "MIN_SL_PCT",
    "MatchingConfig",
    "MatchingEngine",
    "PerformanceMetrics",
    "StrategyParams",
    "TradeFill",
    "TradeRecord",
    "TrendBacktestEngine",
    "compute_metrics",
    "export_report",
    "param_grid_fast_slow",
    "render_html_report",
    "simulate_trades_ma",
    "to_dict",
]
