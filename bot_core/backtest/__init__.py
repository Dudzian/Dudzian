from __future__ import annotations

import importlib

from bot_core.optional import missing_module_proxy
from .ma import Bar as MABar, Trade as MATrade, param_grid_fast_slow, simulate_trades_ma
from .metrics import MetricsResult, compute_metrics, to_dict
from .reporting import export_report, render_html_report
from .simulation import MatchingConfig, MatchingEngine, SimulationScenario
from .models import BacktestFill, BacktestReport, BacktestTrade, PerformanceMetrics
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

_PANDAS_EXPORTS: dict[str, tuple[str, str]] = {
    "BacktestEngine": ("bot_core.backtest.engine", "BacktestEngine"),
    "BacktestError": ("bot_core.backtest.engine", "BacktestError"),
    "DataProviderProtocol": ("bot_core.backtest.engine", "DataProviderProtocol"),
    "HistoricalDataProvider": ("bot_core.backtest.engine", "HistoricalDataProvider"),
    "OHLCVBar": ("bot_core.backtest.providers", "OHLCVBar"),
    "TradeTick": ("bot_core.backtest.providers", "TradeTick"),
    "PandasHistoryProvider": ("bot_core.backtest.providers", "PandasHistoryProvider"),
    "ListHistoryProvider": ("bot_core.backtest.providers", "ListHistoryProvider"),
    "load_trades": ("bot_core.backtest.trade_loader", "load_trades"),
}


def __getattr__(name: str):
    export = _PANDAS_EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module, symbol = export
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != "pandas":
            raise
        value = missing_module_proxy(
            f"{name} wymaga opcjonalnej zależności 'pandas'.",
            cause=exc,
        )
        globals()[name] = value
        return value
    value = getattr(mod, symbol)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_PANDAS_EXPORTS.keys()))


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
    "DataProviderProtocol",
    "HistoricalDataProvider",
    "OHLCVBar",
    "TradeTick",
    "PandasHistoryProvider",
    "ListHistoryProvider",
    "load_trades",
    "MABar",
    "MATrade",
    "MetricsResult",
    "MIN_SL_PCT",
    "MatchingConfig",
    "MatchingEngine",
    "SimulationScenario",
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
