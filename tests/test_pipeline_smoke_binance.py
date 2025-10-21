"""Budowanie gotowych pipeline'ów strategii trend-following na podstawie konfiguracji."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping

from bot_core.alerts import DefaultAlertRouter
from bot_core.config.models import (
    ControllerRuntimeConfig,
    CoreConfig,
    DailyTrendMomentumStrategyConfig,
    EnvironmentConfig,
    InstrumentUniverseConfig,
)
from bot_core.data.base import OHLCVRequest
from bot_core.data.ohlcv import (
    CachedOHLCVSource,
    DualCacheStorage,
    OHLCVBackfillService,
    ParquetCacheStorage,
    PublicAPIDataSource,
    SQLiteCacheStorage,
)
from bot_core.execution.base import ExecutionContext, ExecutionService
from bot_core.execution.paper import MarketMetadata, PaperTradingExecutionService
from bot_core.exchanges.base import AccountSnapshot, Environment, ExchangeAdapterFactory
from bot_core.runtime.bootstrap import BootstrapContext, bootstrap_environment
from bot_core.runtime.pipeline import (
    _DEFAULT_LEDGER_SUBDIR,
    _build_account_loader,
    _build_execution_service,
    _build_markets,
    _normalize_paper_settings,
    _optional_float,
    _resolve_runtime,
    _resolve_strategy,
    _resolve_universe,
)
from bot_core.runtime.controller import DailyTrendController
from bot_core.security import SecretManager
from bot_core.strategies.daily_trend import DailyTrendMomentumSettings, DailyTrendMomentumStrategy

# Opcjonalny kontroler handlu – może nie istnieć w starszych gałęziach.
try:
    from bot_core.runtime.controller import TradingController  # type: ignore
except Exception:  # pragma: no cover
    TradingController = None  # type: ignore


@dataclass(slots=True)
class DailyTrendPipeline:
    """Opakowanie na komponenty gotowego pipeline'u strategii dziennej."""

    bootstrap: BootstrapContext
    controller: DailyTrendController
    backfill_service: OHLCVBackfillService
    data_source: CachedOHLCVSource
    execution_service: ExecutionService
    strategy: DailyTrendMomentumStrategy
    strategy_name: str
    controller_name: str
    # Utrzymujemy nazwę profilu ryzyka jako wygodny skrót (używane w testach i helperach).
    risk_profile_name: str


def build_daily_trend_pipeline(
    *,
    environment_name: str,
    strategy_name: str | None,
    controller_name: str | None,
    config_path: str | Path,
    secret_manager: SecretManager,
    adapter_factories: Mapping[str, ExchangeAdapterFactory] | None = None,
    risk_profile_name: str | None = None,
) -> DailyTrendPipeline:
    """Tworzy kompletny pipeline strategii trend-following D1 dla środowiska paper/testnet."""
    bootstrap_ctx = bootstrap_environment(
        environment_name,
        config_path=config_path,
        secret_manager=secret_manager,
        adapter_factories=adapter_factories,
        risk_profile_name=risk_profile_name,
    )
    core_config = bootstrap_ctx.core_config
    environment = bootstrap_ctx.environment
    effective_risk_profile = bootstrap_ctx.risk_profile_name

    resolved_strategy_name = strategy_name or getattr(environment, "default_strategy", None)
    if not resolved_strategy_name:
        raise ValueError(
            "Środowisko '{environment}' nie ma zdefiniowanej domyślnej strategii, a parametr strategy_name nie został podany."
            .format(environment=environment_name)
        )

    resolved_controller_name = controller_name or getattr(environment, "default_controller", None)
    if not resolved_controller_name:
        raise ValueError(
            "Środowisko '{environment}' nie ma zdefiniowanego domyślnego kontrolera runtime, a parametr controller_name nie został podany."
            .format(environment=environment_name)
        )

    strategy_cfg = _resolve_strategy(core_config, resolved_strategy_name)
    runtime_cfg = _resolve_runtime(core_config, resolved_controller_name)
    universe = _resolve_universe(core_config, environment)

    paper_settings = _normalize_paper_settings(environment)
    allowed_quotes = paper_settings["allowed_quotes"]

    markets = _build_markets(universe, environment.exchange, allowed_quotes, paper_settings)
    if not markets:
        raise ValueError(
            "Brak instrumentów spełniających kryteria paper tradingu – skonfiguruj quote_assets/valuation_asset."
        )

    cache_root = Path(environment.data_cache_path)
    parquet_storage = ParquetCacheStorage(
        cache_root / "ohlcv_parquet",
        namespace=environment.exchange,
    )
    manifest_storage = SQLiteCacheStorage(
        cache_root / "ohlcv_manifest.sqlite",
        store_rows=False,
    )
    storage = DualCacheStorage(primary=parquet_storage, manifest=manifest_storage)

    public_source = PublicAPIDataSource(exchange_adapter=bootstrap_ctx.adapter)
    cached_source = CachedOHLCVSource(storage=storage, upstream=public_source)
    backfill_service = OHLCVBackfillService(cached_source)

    execution_service = _build_execution_service(markets, paper_settings)

    strategy = DailyTrendMomentumStrategy(
        DailyTrendMomentumSettings(
            fast_ma=strategy_cfg.fast_ma,
            slow_ma=strategy_cfg.slow_ma,
            breakout_lookback=strategy_cfg.breakout_lookback,
            momentum_window=strategy_cfg.momentum_window,
            atr_window=strategy_cfg.atr_window,
            atr_multiplier=strategy_cfg.atr_multiplier,
            min_trend_strength=strategy_cfg.min_trend_strength,
            min_momentum=strategy_cfg.min_momentum,
        )
    )

    execution_metadata: MutableMapping[str, str] = {}
    if paper_settings["default_leverage"] > 1.0:
        execution_metadata["leverage"] = f"{paper_settings['default_leverage']:.2f}"

    execution_context = ExecutionContext(
        portfolio_id=paper_settings["portfolio_id"],
        risk_profile=effective_risk_profile,
        environment=environment.environment.value,
        metadata=execution_metadata,
    )

    account_loader = _build_account_loader(
        execution_service=execution_service,
        data_source=cached_source,
        markets=markets,
        interval=runtime_cfg.interval,
        valuation_asset=paper_settings["valuation_asset"],
        cash_assets=allowed_quotes,
    )

    controller = DailyTrendController(
        core_config=core_config,
        environment_name=environment_name,
        controller_name=resolved_controller_name,
        symbols=tuple(markets.keys()),
        backfill_service=backfill_service,
        data_source=cached_source,
        strategy=strategy,
        risk_engine=bootstrap_ctx.risk_engine,
        execution_service=execution_service,
        account_loader=account_loader,
        execution_context=execution_context,
        position_size=paper_settings["position_size"],
    )

    return DailyTrendPipeline(
        bootstrap=bootstrap_ctx,
        controller=controller,
        backfill_service=backfill_service,
        data_source=cached_source,
        execution_service=execution_service,
        strategy=strategy,
        strategy_name=resolved_strategy_name,
        controller_name=resolved_controller_name,
        risk_profile_name=effective_risk_profile,
    )


def create_trading_controller(
    pipeline: DailyTrendPipeline,
    alert_router: DefaultAlertRouter,
    *,
    health_check_interval: float | int | timedelta = 3600,
    order_metadata_defaults: Mapping[str, object] | None = None,
) -> "TradingController":
    """Buduje TradingController spięty z komponentami pipeline'u."""
    if TradingController is None:
        raise RuntimeError("TradingController nie jest dostępny w tej gałęzi.")

    controller = pipeline.controller
    execution_context = controller.execution_context
    environment_cfg = pipeline.bootstrap.environment

    defaults = dict(order_metadata_defaults or {})
    defaults.setdefault("order_type", "market")

    return TradingController(
        risk_engine=pipeline.bootstrap.risk_engine,
        execution_service=pipeline.execution_service,
        alert_router=alert_router,
        account_snapshot_provider=controller.account_loader,
        portfolio_id=execution_context.portfolio_id,
        environment=environment_cfg.environment.value,
        risk_profile=pipeline.risk_profile_name,
        order_metadata_defaults=defaults,
        health_check_interval=health_check_interval,
        execution_metadata=execution_context.metadata,
        decision_journal=pipeline.bootstrap.decision_journal,
    )


# --------------------------------------------------------------------------------------
# Funkcje pomocnicze
# --------------------------------------------------------------------------------------

# Funkcje pomocnicze są importowane z bot_core.runtime.pipeline, aby uniknąć
# duplikowania implementacji w testach.



__all__ = ["DailyTrendPipeline", "build_daily_trend_pipeline", "create_trading_controller"]
