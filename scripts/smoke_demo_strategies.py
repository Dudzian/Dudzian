from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

from bot_core.config.loader import load_core_config
from bot_core.data.backtest_library import BacktestDatasetLibrary
from bot_core.runtime.journal import InMemoryTradingDecisionJournal
from bot_core.runtime.multi_strategy_scheduler import MultiStrategyScheduler, StrategyDataFeed
from bot_core.runtime.pipeline import InMemoryStrategySignalSink, _collect_strategy_definitions
from bot_core.security.guards import LicenseCapabilityError
from bot_core.strategies.base import MarketSnapshot, StrategyEngine
from bot_core.strategies.catalog import DEFAULT_STRATEGY_CATALOG
from bot_core.strategies.cross_exchange_arbitrage import (
    CrossExchangeArbitrageSettings,
    CrossExchangeArbitrageStrategy,
)
from bot_core.strategies.day_trading import DayTradingSettings, DayTradingStrategy
from bot_core.strategies.mean_reversion import MeanReversionSettings, MeanReversionStrategy
from bot_core.strategies.options import OptionsIncomeSettings, OptionsIncomeStrategy
from bot_core.strategies.scalping import ScalpingSettings, ScalpingStrategy
from bot_core.strategies.statistical_arbitrage import (
    StatisticalArbitrageSettings,
    StatisticalArbitrageStrategy,
)
from bot_core.strategies.volatility_target import (
    VolatilityTargetSettings,
    VolatilityTargetStrategy,
)


@dataclass(slots=True)
class SmokeResult:
    """Rezultat demonstracyjnego uruchomienia strategii."""

    cycles: int
    telemetry: Mapping[str, Mapping[str, float]]
    emitted_signals: Mapping[str, int]


LOGGER = logging.getLogger(__name__)


class _ReplayFeed(StrategyDataFeed):
    """Prosty feed odtwarzający dane z biblioteki backtestowej."""

    def __init__(self, snapshots: Sequence[MarketSnapshot]):
        self._history = tuple(snapshots)
        self._cursor = 0

    def load_history(self, strategy_name: str, bars: int) -> Sequence[MarketSnapshot]:
        if bars <= 0:
            return ()
        return self._history[: min(len(self._history), bars)]

    def fetch_latest(self, strategy_name: str) -> Sequence[MarketSnapshot]:
        if self._cursor >= len(self._history):
            return ()
        snapshot = self._history[self._cursor]
        self._cursor += 1
        return (snapshot,)


def _instantiate_strategies(core_config) -> Mapping[str, StrategyEngine]:
    catalog = DEFAULT_STRATEGY_CATALOG
    definitions = _collect_strategy_definitions(core_config)
    registry: dict[str, StrategyEngine] = {}
    for name, cfg in getattr(core_config, "mean_reversion_strategies", {}).items():
        registry[name] = MeanReversionStrategy(
            MeanReversionSettings(
                lookback=cfg.lookback,
                entry_zscore=cfg.entry_zscore,
                exit_zscore=cfg.exit_zscore,
                max_holding_period=cfg.max_holding_period,
                volatility_cap=cfg.volatility_cap,
                min_volume_usd=cfg.min_volume_usd,
            )
        )
    for name, cfg in getattr(core_config, "volatility_target_strategies", {}).items():
        registry[name] = VolatilityTargetStrategy(
            VolatilityTargetSettings(
                target_volatility=cfg.target_volatility,
                lookback=cfg.lookback,
                rebalance_threshold=cfg.rebalance_threshold,
                min_allocation=cfg.min_allocation,
                max_allocation=cfg.max_allocation,
                floor_volatility=cfg.floor_volatility,
            )
        )
    for name, cfg in getattr(core_config, "cross_exchange_arbitrage_strategies", {}).items():
        registry[name] = CrossExchangeArbitrageStrategy(
            CrossExchangeArbitrageSettings(
                primary_exchange=cfg.primary_exchange,
                secondary_exchange=cfg.secondary_exchange,
                spread_entry=cfg.spread_entry,
                spread_exit=cfg.spread_exit,
                max_notional=cfg.max_notional,
                max_open_seconds=cfg.max_open_seconds,
            )
        )
    for name, cfg in getattr(core_config, "scalping_strategies", {}).items():
        registry[name] = ScalpingStrategy(
            ScalpingSettings(
                min_price_change=cfg.min_price_change,
                take_profit=cfg.take_profit,
                stop_loss=cfg.stop_loss,
                max_hold_bars=cfg.max_hold_bars,
            )
        )
    for name, cfg in getattr(core_config, "options_income_strategies", {}).items():
        registry[name] = OptionsIncomeStrategy(
            OptionsIncomeSettings(
                min_iv=cfg.min_iv,
                max_delta=cfg.max_delta,
                min_days_to_expiry=cfg.min_days_to_expiry,
                roll_threshold_iv=cfg.roll_threshold_iv,
            )
        )
    for name, cfg in getattr(core_config, "statistical_arbitrage_strategies", {}).items():
        registry[name] = StatisticalArbitrageStrategy(
            StatisticalArbitrageSettings(
                lookback=cfg.lookback,
                spread_entry_z=cfg.spread_entry_z,
                spread_exit_z=cfg.spread_exit_z,
                max_notional=cfg.max_notional,
            )
        )
    for name, cfg in getattr(core_config, "day_trading_strategies", {}).items():
        registry[name] = DayTradingStrategy(
            DayTradingSettings(
                momentum_window=cfg.momentum_window,
                volatility_window=cfg.volatility_window,
                entry_threshold=cfg.entry_threshold,
                exit_threshold=cfg.exit_threshold,
                take_profit_atr=cfg.take_profit_atr,
                stop_loss_atr=cfg.stop_loss_atr,
                max_holding_bars=cfg.max_holding_bars,
                atr_floor=cfg.atr_floor,
                bias_strength=cfg.bias_strength,
            )
        )
    return registry


def _resolve_dataset_name(strategy_name: str) -> str:
    lowered = strategy_name.lower()
    if "mean_reversion" in lowered:
        return "mean_reversion"
    if "volatility_target" in lowered:
        return "volatility_target"
    if "cross_exchange" in lowered:
        return "cross_exchange_arbitrage"
    if "scalping" in lowered:
        return "mean_reversion"
    if "day_trading" in lowered or "intraday" in lowered:
        return "mean_reversion"
    if "options" in lowered:
        return "volatility_target"
    if "statistical" in lowered or "pairs" in lowered:
        return "mean_reversion"
    raise KeyError(f"Brak zmapowanego datasetu dla strategii {strategy_name}")


def _row_to_snapshot(dataset: str, row: Mapping[str, object]) -> MarketSnapshot:
    if dataset == "mean_reversion":
        return MarketSnapshot(
            symbol=str(row["instrument"]),
            timestamp=int(row["timestamp"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
    if dataset == "volatility_target":
        price = float(row["close"])
        return MarketSnapshot(
            symbol=str(row["instrument"]),
            timestamp=int(row["timestamp"]),
            open=price,
            high=price,
            low=price,
            close=price,
            volume=float(row["volume"]),
        )
    if dataset == "cross_exchange_arbitrage":
        price = float(row["base_ask"])
        return MarketSnapshot(
            symbol=str(row["instrument"]),
            timestamp=int(row["timestamp"]),
            open=price,
            high=price,
            low=float(row["base_bid"]),
            close=price,
            volume=float(row["available_volume"]),
            indicators={
                "primary_bid": float(row["base_bid"]),
                "primary_ask": float(row["base_ask"]),
                "secondary_bid": float(row["quote_bid"]),
                "secondary_ask": float(row["quote_ask"]),
                "secondary_timestamp": int(row["timestamp"]),
            },
        )
    raise KeyError(f"Nieobsługiwany dataset {dataset}")


def _build_feed(library: BacktestDatasetLibrary, dataset_name: str) -> _ReplayFeed:
    rows = library.load_typed_rows(dataset_name)
    snapshots = [_row_to_snapshot(dataset_name, row) for row in rows]
    return _ReplayFeed(tuple(snapshots))


async def _run_cycles(scheduler: MultiStrategyScheduler, cycles: int) -> None:
    for _ in range(cycles):
        await scheduler.run_once()


def run_demo(
    *,
    config_path: Path,
    manifest_path: Path,
    environment: str,
    scheduler_name: str | None,
    cycles: int,
) -> SmokeResult:
    core_config = load_core_config(config_path)
    schedulers = getattr(core_config, "multi_strategy_schedulers", {})
    if not schedulers:
        raise RuntimeError("Konfiguracja nie zawiera schedulerów multi-strategy")
    resolved_name = scheduler_name or next(iter(schedulers))
    scheduler_cfg = schedulers.get(resolved_name)
    if scheduler_cfg is None:
        raise KeyError(f"Scheduler {resolved_name} nie istnieje w konfiguracji")

    strategies = _instantiate_strategies(core_config)
    library = BacktestDatasetLibrary(manifest_path)
    telemetry_payloads: MutableMapping[str, Mapping[str, float]] = {}

    def telemetry_emitter(name: str, payload: Mapping[str, float]) -> None:
        telemetry_payloads[name] = dict(payload)

    journal = InMemoryTradingDecisionJournal()

    scheduler = MultiStrategyScheduler(
        environment=environment,
        portfolio="demo-portfolio",
        telemetry_emitter=telemetry_emitter,
        decision_journal=journal,
    )
    sink = InMemoryStrategySignalSink()

    for schedule in scheduler_cfg.schedules:
        strategy = strategies.get(schedule.strategy)
        if strategy is None:
            raise KeyError(f"Brak strategii {schedule.strategy} w konfiguracji")
        dataset = _resolve_dataset_name(schedule.strategy)
        feed = _build_feed(library, dataset)
        scheduler.register_schedule(
            name=schedule.name,
            strategy_name=schedule.strategy,
            strategy=strategy,
            feed=feed,
            sink=sink,
            cadence_seconds=schedule.cadence_seconds,
            max_drift_seconds=schedule.max_drift_seconds,
            warmup_bars=schedule.warmup_bars,
            risk_profile=schedule.risk_profile,
            max_signals=schedule.max_signals,
        )

    asyncio.run(_run_cycles(scheduler, cycles))

    emitted: MutableMapping[str, int] = {schedule.name: 0 for schedule in scheduler_cfg.schedules}
    for schedule_name, signals in sink.export():
        emitted[schedule_name] = emitted.get(schedule_name, 0) + len(signals)

    return SmokeResult(cycles=cycles, telemetry=dict(telemetry_payloads), emitted_signals=dict(emitted))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke test multi-strategy w trybie demo")
    parser.add_argument("--config", default="config/core.yaml", help="Ścieżka do core.yaml")
    parser.add_argument(
        "--manifest",
        default="data/backtests/normalized/manifest.yaml",
        help="Manifest znormalizowanych danych backtestowych",
    )
    parser.add_argument("--environment", default="demo", help="Nazwa środowiska runtime")
    parser.add_argument("--scheduler", default=None, help="Nazwa scheduler-a (opcjonalnie)")
    parser.add_argument("--cycles", type=int, default=3, help="Liczba cykli run_once do wykonania")
    args = parser.parse_args(argv)

    result = run_demo(
        config_path=Path(args.config).expanduser().resolve(),
        manifest_path=Path(args.manifest).expanduser().resolve(),
        environment=args.environment,
        scheduler_name=args.scheduler,
        cycles=max(1, args.cycles),
    )
    print(json.dumps({
        "cycles": result.cycles,
        "telemetry": result.telemetry,
        "emitted_signals": result.emitted_signals,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - skrypt CLI
    raise SystemExit(main())
