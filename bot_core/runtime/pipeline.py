"""Budowanie gotowych pipeline'ów strategii trend-following na podstawie konfiguracji."""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Mapping, MutableMapping, Sequence

from bot_core.alerts import DefaultAlertRouter
from bot_core.config.models import (
    ControllerRuntimeConfig,
    CoreConfig,
    DailyTrendMomentumStrategyConfig,
    EnvironmentConfig,
    InstrumentUniverseConfig,
    MeanReversionStrategyConfig,
    VolatilityTargetingStrategyConfig,
    CrossExchangeArbitrageStrategyConfig,
    StrategyScheduleConfig,
    MultiStrategySchedulerConfig,
)
from bot_core.data import CachedOHLCVSource, create_cached_ohlcv_source, resolve_cache_namespace
from bot_core.data.base import OHLCVRequest
from bot_core.data.ohlcv import OHLCVBackfillService
from bot_core.execution.base import ExecutionContext, ExecutionService
from bot_core.execution.paper import MarketMetadata, PaperTradingExecutionService
from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeAdapterFactory,
)
from bot_core.market_intel import MarketIntelAggregator, MarketIntelQuery, MarketIntelSnapshot
from bot_core.portfolio import PortfolioDecisionLog, PortfolioGovernor
from bot_core.runtime.bootstrap import BootstrapContext, bootstrap_environment
from bot_core.runtime.multi_strategy_scheduler import (
    MultiStrategyScheduler,
    StrategyDataFeed,
    StrategySignalSink,
)
from bot_core.runtime.portfolio_coordinator import PortfolioRuntimeCoordinator
from bot_core.runtime.portfolio_inputs import (
    build_slo_status_provider,
    build_stress_override_provider,
)
from bot_core.runtime.tco_reporting import RuntimeTCOReporter
from bot_core.runtime.controller import DailyTrendController
from bot_core.security import SecretManager
from bot_core.strategies.daily_trend import DailyTrendMomentumSettings, DailyTrendMomentumStrategy
from bot_core.strategies.mean_reversion import MeanReversionSettings, MeanReversionStrategy
from bot_core.strategies.volatility_target import VolatilityTargetSettings, VolatilityTargetStrategy
from bot_core.strategies.cross_exchange_arbitrage import (
    CrossExchangeArbitrageSettings,
    CrossExchangeArbitrageStrategy,
)
from bot_core.strategies.base import StrategyEngine, StrategySignal, MarketSnapshot

_DEFAULT_LEDGER_SUBDIR = Path("audit/ledger")
_LOGGER = logging.getLogger(__name__)


def _minutes_to_timedelta(value: float | int | None, default_minutes: float) -> timedelta | None:
    minutes = default_minutes if value in (None, "") else float(value)
    if minutes <= 0:
        return None
    return timedelta(minutes=minutes)


def _create_cached_source(adapter: ExchangeAdapter, environment: EnvironmentConfig) -> CachedOHLCVSource:
    """Buduje źródło OHLCV korzystające z lokalnego cache i snapshotów REST."""

    cache_root = Path(environment.data_cache_path)
    data_source_cfg = getattr(environment, "data_source", None)
    enable_snapshots = True
    namespace = resolve_cache_namespace(environment)
    offline_mode = bool(getattr(environment, "offline_mode", False))
    allow_network_upstream = not offline_mode
    if data_source_cfg is not None:
        enable_snapshots = bool(getattr(data_source_cfg, "enable_snapshots", True))
    if offline_mode:
        enable_snapshots = False

    return create_cached_ohlcv_source(
        adapter,
        cache_directory=cache_root / "ohlcv_parquet",
        manifest_path=cache_root / "ohlcv_manifest.sqlite",
        enable_snapshots=enable_snapshots,
        allow_network_upstream=allow_network_upstream,
        namespace=namespace,
    )


# Opcjonalny kontroler handlu – może nie istnieć w starszych gałęziach.
try:
    from bot_core.runtime.controller import TradingController  # type: ignore
except Exception:  # pragma: no cover
    TradingController = None  # type: ignore

try:
    from bot_core.portfolio import PortfolioGovernor  # type: ignore
except Exception:  # pragma: no cover - PortfolioGovernor może być niedostępny
    PortfolioGovernor = None  # type: ignore


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
    risk_profile_name: str
    tco_reporter: RuntimeTCOReporter | None = None


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

    cached_source = _create_cached_source(bootstrap_ctx.adapter, environment)
    storage = cached_source.storage
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
        strategy_name=resolved_strategy_name,
        exchange_name=environment.exchange,
        tco_reporter=bootstrap_ctx.tco_reporter,
        tco_metadata={
            "pipeline": "daily_trend",
            "environment": environment_name,
        },
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
        tco_reporter=bootstrap_ctx.tco_reporter,
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
        strategy_name=pipeline.strategy_name,
        exchange_name=environment_cfg.exchange,
        tco_reporter=pipeline.tco_reporter,
        tco_metadata={
            "pipeline": "daily_trend",
            "environment": environment_cfg.name,
            "controller": pipeline.controller_name,
        },
    )


# --------------------------------------------------------------------------------------
# Funkcje pomocnicze
# --------------------------------------------------------------------------------------


def _resolve_strategy(core_config: CoreConfig, strategy_name: str) -> DailyTrendMomentumStrategyConfig:
    try:
        return core_config.strategies[strategy_name]
    except KeyError as exc:  # pragma: no cover - kontrola konfiguracji
        raise KeyError(f"Brak strategii '{strategy_name}' w konfiguracji core") from exc


def _resolve_runtime(core_config: CoreConfig, controller_name: str) -> ControllerRuntimeConfig:
    try:
        return core_config.runtime_controllers[controller_name]
    except KeyError as exc:  # pragma: no cover - kontrola konfiguracji
        raise KeyError(f"Brak konfiguracji runtime dla kontrolera '{controller_name}'") from exc


def _resolve_universe(core_config: CoreConfig, environment: EnvironmentConfig) -> InstrumentUniverseConfig:
    if not environment.instrument_universe:
        raise ValueError(
            f"Środowisko {environment.name} nie ma przypisanego instrument_universe w config/core.yaml"
        )
    try:
        return core_config.instrument_universes[environment.instrument_universe]
    except KeyError as exc:
        raise KeyError(
            f"Konfiguracja nie zawiera uniwersum '{environment.instrument_universe}' wymaganego przez środowisko {environment.name}."
        ) from exc


def _normalize_paper_settings(environment: EnvironmentConfig) -> MutableMapping[str, object]:
    if environment.environment not in {Environment.PAPER, Environment.TESTNET}:
        raise ValueError("Pipeline paper trading jest dostępny wyłącznie dla środowisk paper/testnet.")

    # adapter_settings może nie istnieć w danej gałęzi modeli – użyj bezpiecznego getattr
    raw_adapter = getattr(environment, "adapter_settings", {}) or {}
    raw_settings = raw_adapter.get("paper_trading", {}) or {}

    base_path = Path(environment.data_cache_path)

    valuation_asset = str(raw_settings.get("valuation_asset", "USDT")).upper()
    position_size = max(0.0, float(raw_settings.get("position_size", 0.1)))
    default_leverage = max(1.0, float(raw_settings.get("default_leverage", 1.0)))
    allowed_quotes = {
        *(str(asset).upper() for asset in raw_settings.get("quote_assets", (valuation_asset,))),
    }

    initial_balances = {
        str(asset).upper(): float(amount)
        for asset, amount in (raw_settings.get("initial_balances", {}) or {}).items()
    }
    for quote in allowed_quotes:
        initial_balances.setdefault(quote, 100_000.0)

    default_market = raw_settings.get("default_market", {}) or {}
    market_overrides = {
        str(symbol): {str(k): v for k, v in entry.items()}
        for symbol, entry in (raw_settings.get("markets", {}) or {}).items()
    }

    ledger_directory_setting = raw_settings.get("ledger_directory")
    ledger_directory: Path | None
    if ledger_directory_setting is None:
        ledger_directory = base_path / _DEFAULT_LEDGER_SUBDIR
    else:
        text = str(ledger_directory_setting).strip()
        if not text:
            ledger_directory = None
        else:
            candidate = Path(text)
            ledger_directory = candidate if candidate.is_absolute() else base_path / candidate

    ledger_filename_pattern = str(raw_settings.get("ledger_filename_pattern", "ledger-%Y%m%d.jsonl"))
    ledger_retention_days_raw = raw_settings.get("ledger_retention_days", 730)
    if ledger_retention_days_raw is None:
        ledger_retention_days = None
    else:
        text_value = str(ledger_retention_days_raw).strip()
        ledger_retention_days = None if not text_value else int(float(text_value))
    ledger_fsync = bool(raw_settings.get("ledger_fsync", False))

    return {
        "valuation_asset": valuation_asset,
        "position_size": position_size,
        "allowed_quotes": allowed_quotes,
        "default_leverage": default_leverage,
        "initial_balances": initial_balances,
        "default_market": default_market,
        "market_overrides": market_overrides,
        "portfolio_id": str(raw_settings.get("portfolio_id", environment.name)),
        "maker_fee": float(raw_settings.get("maker_fee", 0.0004)),
        "taker_fee": float(raw_settings.get("taker_fee", 0.0006)),
        "slippage_bps": float(raw_settings.get("slippage_bps", 5.0)),
        "ledger_directory": ledger_directory,
        "ledger_filename_pattern": ledger_filename_pattern,
        "ledger_retention_days": ledger_retention_days,
        "ledger_fsync": ledger_fsync,
    }


def _build_markets(
    universe: InstrumentUniverseConfig,
    exchange_name: str,
    allowed_quotes: set[str],
    paper_settings: Mapping[str, object],
) -> Mapping[str, MarketMetadata]:
    markets: dict[str, MarketMetadata] = {}
    default_market: Mapping[str, object] = paper_settings["default_market"]  # type: ignore[assignment]
    overrides: Mapping[str, Mapping[str, object]] = paper_settings["market_overrides"]  # type: ignore[assignment]

    for instrument in universe.instruments:
        symbol = instrument.exchange_symbols.get(exchange_name)
        if not symbol:
            continue
        quote = instrument.quote_asset.upper()
        if quote not in allowed_quotes:
            continue

        per_symbol = overrides.get(symbol, {})
        market = MarketMetadata(
            base_asset=instrument.base_asset.upper(),
            quote_asset=quote,
            min_quantity=float(per_symbol.get("min_quantity", default_market.get("min_quantity", 0.0))),
            min_notional=float(per_symbol.get("min_notional", default_market.get("min_notional", 0.0))),
            step_size=_optional_float(per_symbol.get("step_size", default_market.get("step_size"))),
            tick_size=_optional_float(per_symbol.get("tick_size", default_market.get("tick_size"))),
        )
        markets[symbol] = market

    return markets


def _build_execution_service(
    markets: Mapping[str, MarketMetadata],
    paper_settings: Mapping[str, object],
) -> PaperTradingExecutionService:
    return PaperTradingExecutionService(
        markets,
        initial_balances=paper_settings["initial_balances"],  # type: ignore[arg-type]
        maker_fee=float(paper_settings["maker_fee"]),
        taker_fee=float(paper_settings["taker_fee"]),
        slippage_bps=float(paper_settings["slippage_bps"]),
        ledger_directory=paper_settings["ledger_directory"],
        ledger_filename_pattern=str(paper_settings["ledger_filename_pattern"]),
        ledger_retention_days=paper_settings["ledger_retention_days"],  # type: ignore[arg-type]
        ledger_fsync=bool(paper_settings["ledger_fsync"]),
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _build_account_loader(
    *,
    execution_service: PaperTradingExecutionService,
    data_source: CachedOHLCVSource,
    markets: Mapping[str, MarketMetadata],
    interval: str,
    valuation_asset: str,
    cash_assets: set[str],
) -> Callable[[], AccountSnapshot]:
    storage = data_source.storage
    price_cache: MutableMapping[str, tuple[float, float]] = {}

    def latest_price(symbol: str) -> float:
        cache_key = data_source._cache_key(symbol, interval)  # pylint: disable=protected-access
        latest_cached: float | None = None
        try:
            latest_cached = storage.latest_timestamp(cache_key)
        except AttributeError:
            try:
                rows = storage.read(cache_key)["rows"]
            except KeyError:
                rows = []
            if rows:
                latest_cached = float(rows[-1][0])
        cached_entry = price_cache.get(symbol)
        if cached_entry and latest_cached is not None and abs(cached_entry[0] - latest_cached) < 1e-6:
            return cached_entry[1]

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        request = OHLCVRequest(symbol=symbol, interval=interval, start=0, end=now_ms)
        response = data_source.fetch_ohlcv(request)
        if not response.rows:
            raise RuntimeError(
                f"Brak danych OHLCV dla symbolu {symbol} – wykonaj backfill (scripts/backfill.py) przed startem strategii"
            )
        last_row = response.rows[-1]
        close_price = float(last_row[4])
        timestamp = float(last_row[0])
        price_cache[symbol] = (timestamp, close_price)
        return close_price

    valuation_target = valuation_asset.upper()
    cash_like_assets = {valuation_target}
    cash_like_assets.update(asset.upper() for asset in cash_assets)

    def loader() -> AccountSnapshot:
        raw_balances = execution_service.balances()
        balances: MutableMapping[str, float] = {
            str(asset).upper(): float(amount) for asset, amount in raw_balances.items()
        }
        short_positions = execution_service.short_positions()

        pair_prices: dict[tuple[str, str], float] = {}
        for symbol, market in markets.items():
            price = latest_price(symbol)
            pair_prices[(market.base_asset.upper(), market.quote_asset.upper())] = price

        adjacency: defaultdict[str, list[tuple[str, float]]] = defaultdict(list)
        for (base, quote), price in pair_prices.items():
            if price <= 0:
                continue
            adjacency[base].append((quote, price))
            adjacency[quote].append((base, 1.0 / price))
        adjacency.setdefault(valuation_target, [])

        def _conversion_rate(source: str, target: str) -> float:
            src = source.upper()
            dst = target.upper()
            if src == dst:
                return 1.0
            if src not in adjacency:
                raise RuntimeError(
                    f"Brak ścieżki konwersji dla aktywa {src} – rozszerz quote_assets lub dodaj parę referencyjną."
                )
            visited = {src}
            queue = deque([(src, 1.0)])
            while queue:
                asset, rate = queue.popleft()
                for neighbor, weight in adjacency.get(asset, ()):
                    if neighbor in visited:
                        continue
                    new_rate = rate * weight
                    if neighbor == dst:
                        return new_rate
                    visited.add(neighbor)
                    queue.append((neighbor, new_rate))
            raise RuntimeError(
                f"Nie udało się przeliczyć {source} na {target}. Dodaj brakujące instrumenty triangulacyjne."
            )

        def convert_amount(asset: str, amount: float) -> float:
            if abs(amount) < 1e-18:
                return 0.0
            if asset.upper() == valuation_target:
                return amount
            rate = _conversion_rate(asset, valuation_target)
            return amount * rate

        total_equity = 0.0
        for asset, amount in balances.items():
            total_equity += convert_amount(asset, amount)

        for symbol, details in short_positions.items():
            quantity = float(details.get("quantity", 0.0))
            if quantity <= 0:
                continue
            margin = float(details.get("margin", 0.0))
            market = markets.get(symbol)
            if market is None:
                continue
            current_price = pair_prices.get((market.base_asset.upper(), market.quote_asset.upper()))
            if current_price is None:
                current_price = latest_price(symbol)
                pair_prices[(market.base_asset.upper(), market.quote_asset.upper())] = current_price
                if current_price > 0:
                    adjacency[market.base_asset.upper()].append((market.quote_asset.upper(), current_price))
                    adjacency[market.quote_asset.upper()].append((market.base_asset.upper(), 1.0 / current_price))
            total_equity += convert_amount(market.quote_asset, margin)
            liability = current_price * quantity
            total_equity -= convert_amount(market.quote_asset, liability)

        available_margin = 0.0
        for asset, amount in balances.items():
            if amount <= 0:
                continue
            if asset in cash_like_assets:
                available_margin += convert_amount(asset, amount)

        return AccountSnapshot(
            balances=dict(balances),
            total_equity=total_equity,
            available_margin=available_margin,
            maintenance_margin=0.0,
        )

    return loader


@dataclass(slots=True)
class MultiStrategyRuntime:
    """Zestaw komponentów do uruchomienia scheduler-a multi-strategy."""

    bootstrap: BootstrapContext
    scheduler: MultiStrategyScheduler
    data_feed: StrategyDataFeed
    signal_sink: StrategySignalSink
    strategies: Mapping[str, StrategyEngine]
    schedules: tuple[StrategyScheduleConfig, ...]
    portfolio_coordinator: PortfolioRuntimeCoordinator | None = None
    portfolio_governor: PortfolioGovernor | None = None
    tco_reporter: RuntimeTCOReporter | None = None


class OHLCVStrategyFeed(StrategyDataFeed):
    """Strategiczny feed korzystający z lokalnego cache OHLCV."""

    def __init__(
        self,
        data_source: CachedOHLCVSource,
        *,
        symbols_map: Mapping[str, Sequence[str]],
        interval_map: Mapping[str, str],
        default_interval: str = "1h",
    ) -> None:
        self._data_source = data_source
        self._symbols_map = {key: tuple(values) for key, values in symbols_map.items()}
        self._interval_map = dict(interval_map)
        self._default_interval = default_interval

    def load_history(self, strategy_name: str, bars: int) -> Sequence[MarketSnapshot]:
        interval = self._interval_map.get(strategy_name, self._default_interval)
        symbols = self._symbols_map.get(strategy_name, ())
        if not symbols or bars <= 0:
            return ()
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        snapshots: list[MarketSnapshot] = []
        for symbol in symbols:
            request = OHLCVRequest(symbol=symbol, interval=interval, start=0, end=now_ms, limit=bars)
            response = self._data_source.fetch_ohlcv(request)
            snapshots.extend(_response_to_snapshots(symbol, response))
        snapshots.sort(key=lambda snap: (snap.symbol, snap.timestamp))
        return tuple(snapshots)

    def fetch_latest(self, strategy_name: str) -> Sequence[MarketSnapshot]:
        interval = self._interval_map.get(strategy_name, self._default_interval)
        symbols = self._symbols_map.get(strategy_name, ())
        if not symbols:
            return ()
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        snapshots: list[MarketSnapshot] = []
        for symbol in symbols:
            request = OHLCVRequest(symbol=symbol, interval=interval, start=0, end=now_ms, limit=1)
            response = self._data_source.fetch_ohlcv(request)
            converted = _response_to_snapshots(symbol, response)
            if converted:
                snapshots.append(converted[-1])
        snapshots.sort(key=lambda snap: (snap.symbol, snap.timestamp))
        return tuple(snapshots)


class InMemoryStrategySignalSink(StrategySignalSink):
    """Lekki sink zapisujący sygnały dla celów audytu/regresji."""

    def __init__(self) -> None:
        self._records: list[tuple[str, Sequence[StrategySignal]]] = []

    def submit(
        self,
        *,
        strategy_name: str,
        schedule_name: str,
        risk_profile: str,
        timestamp: datetime,
        signals: Sequence[StrategySignal],
    ) -> None:
        self._records.append((schedule_name, tuple(signals)))

    def export(self) -> Sequence[tuple[str, Sequence[StrategySignal]]]:
        return tuple(self._records)


def _response_to_snapshots(symbol: str, response: object) -> list[MarketSnapshot]:
    if not response or not getattr(response, "rows", None):
        return []
    columns = [str(column).lower() for column in getattr(response, "columns", [])]
    index = {column: idx for idx, column in enumerate(columns)}
    required = {"open", "high", "low", "close"}
    if not required.issubset(index):
        return []
    time_idx = index.get("open_time")
    if time_idx is None:
        time_idx = index.get("timestamp")
    if time_idx is None:
        return []
    volume_idx = index.get("volume")
    snapshots: list[MarketSnapshot] = []
    for row in getattr(response, "rows", []):
        if len(row) <= index["close"]:
            continue
        snapshots.append(
            MarketSnapshot(
                symbol=symbol,
                timestamp=int(float(row[time_idx])),
                open=float(row[index["open"]]),
                high=float(row[index["high"]]),
                low=float(row[index["low"]]),
                close=float(row[index["close"]]),
                volume=float(row[volume_idx]) if volume_idx is not None else 0.0,
            )
        )
    snapshots.sort(key=lambda item: item.timestamp)
    return snapshots


def build_multi_strategy_runtime(
    *,
    environment_name: str,
    scheduler_name: str | None,
    config_path: str | Path,
    secret_manager: SecretManager,
    adapter_factories: Mapping[str, ExchangeAdapterFactory] | None = None,
    telemetry_emitter: Callable[[str, Mapping[str, float]], None] | None = None,
) -> MultiStrategyRuntime:
    bootstrap_ctx = bootstrap_environment(
        environment_name,
        config_path=config_path,
        secret_manager=secret_manager,
        adapter_factories=adapter_factories,
    )
    core_config = bootstrap_ctx.core_config
    environment = bootstrap_ctx.environment
    scheduler_configs = getattr(core_config, "multi_strategy_schedulers", {})
    if not scheduler_configs:
        raise ValueError("Brak zdefiniowanych schedulerów multi-strategy w konfiguracji")
    resolved_scheduler_name = scheduler_name or next(iter(scheduler_configs))
    scheduler_cfg = scheduler_configs.get(resolved_scheduler_name)
    if scheduler_cfg is None:
        raise KeyError(f"Nie znaleziono scheduler-a {resolved_scheduler_name}")

    paper_settings = _normalize_paper_settings(environment)
    allowed_quotes = paper_settings["allowed_quotes"]
    universe = _resolve_universe(core_config, environment)
    markets = _build_markets(universe, environment.exchange, allowed_quotes, paper_settings)
    if not markets:
        raise ValueError("Brak instrumentów dla scheduler-a multi-strategy – sprawdź instrument_universe")

    cached_source = _create_cached_source(bootstrap_ctx.adapter, environment)
    storage = cached_source.storage
    market_intel = MarketIntelAggregator(storage)

    strategies = _instantiate_strategies(core_config)
    interval_map: dict[str, str] = {}
    symbols_map: dict[str, Sequence[str]] = {}
    all_symbols = tuple(markets.keys())
    for schedule in scheduler_cfg.schedules:
        interval = schedule.interval or "1h"
        interval_map[schedule.strategy] = interval
        symbols_map.setdefault(schedule.strategy, all_symbols)

    data_feed = OHLCVStrategyFeed(cached_source, symbols_map=symbols_map, interval_map=interval_map)
    signal_sink = InMemoryStrategySignalSink()
    portfolio_governor = getattr(bootstrap_ctx, "portfolio_governor", None)
    scheduler = MultiStrategyScheduler(
        environment=environment_name,
        portfolio=str(paper_settings["portfolio_id"]),
        telemetry_emitter=telemetry_emitter,
        decision_journal=bootstrap_ctx.decision_journal,
        portfolio_governor=portfolio_governor,
    )

    portfolio_coordinator: PortfolioRuntimeCoordinator | None = None
    governor_name = getattr(scheduler_cfg, "portfolio_governor", None)
    if governor_name:
        governor_cfg = core_config.portfolio_governors.get(governor_name)
        if governor_cfg is None:
            raise KeyError(
                f"Scheduler {resolved_scheduler_name} wskazuje PortfolioGovernora '{governor_name}', którego nie ma w konfiguracji"
            )

        decision_log = (
            bootstrap_ctx.portfolio_decision_log
            if getattr(bootstrap_ctx, "portfolio_decision_log", None) is not None
            else PortfolioDecisionLog()
        )
        governor = PortfolioGovernor(governor_cfg, decision_log=decision_log)

        asset_symbols = [asset.symbol for asset in governor_cfg.assets]
        interval = governor_cfg.market_intel_interval or "1h"
        lookback = int(getattr(governor_cfg, "market_intel_lookback_bars", 168) or 168)
        inputs_cfg = getattr(scheduler_cfg, "portfolio_inputs", None)
        data_cache_root = Path(environment.data_cache_path).expanduser()
        fallback_candidates = (
            data_cache_root,
            data_cache_root.parent,
        )
        fallback_directories = tuple(dict.fromkeys(candidate for candidate in fallback_candidates if str(candidate)))

        slo_provider = None
        stress_provider = None
        if inputs_cfg is not None:
            slo_age = _minutes_to_timedelta(
                getattr(inputs_cfg, "slo_max_age_minutes", None),
                default_minutes=120.0,
            )
            stress_age = _minutes_to_timedelta(
                getattr(inputs_cfg, "stress_max_age_minutes", None),
                default_minutes=240.0,
            )
            slo_path = getattr(inputs_cfg, "slo_report_path", None)
            if slo_path:
                slo_provider = build_slo_status_provider(
                    slo_path,
                    fallback_directories=fallback_directories,
                    max_age=slo_age,
                )
            stress_path = getattr(inputs_cfg, "stress_lab_report_path", None)
            if stress_path:
                stress_provider = build_stress_override_provider(
                    stress_path,
                    fallback_directories=fallback_directories,
                    max_age=stress_age,
                )

        def _market_data_provider() -> Mapping[str, MarketIntelSnapshot]:
            if not asset_symbols:
                return {}
            queries = [
                MarketIntelQuery(symbol=symbol, interval=interval, lookback_bars=lookback)
                for symbol in asset_symbols
            ]
            try:
                return market_intel.build_many(queries)
            except Exception:  # pragma: no cover - diagnostyka danych
                _LOGGER.exception("PortfolioGovernor: błąd budowania metryk Market Intel")
                snapshots: dict[str, MarketIntelSnapshot] = {}
                for query in queries:
                    try:
                        snapshots[query.symbol] = market_intel.build_snapshot(query)
                    except Exception:
                        _LOGGER.debug(
                            "Brak metryk Market Intel dla %s", query.symbol, exc_info=True
                        )
                return snapshots

        def _allocation_provider() -> tuple[float, Mapping[str, float]]:
            latest: dict[str, float] = {symbol: 0.0 for symbol in asset_symbols}
            for _, signals in signal_sink.export():
                for signal in signals:
                    if signal.symbol not in latest:
                        continue
                    weight = signal.metadata.get("current_allocation")
                    if weight is None:
                        continue
                    try:
                        latest[signal.symbol] = float(weight)
                    except (TypeError, ValueError):  # pragma: no cover - diagnostyka metadanych
                        _LOGGER.debug(
                            "Niepoprawna wartość current_allocation=%s dla %s",
                            weight,
                            signal.symbol,
                            exc_info=True,
                        )
                        continue
            return 1.0, latest

        def _metadata_provider() -> Mapping[str, object]:
            return {
                "environment": environment_name,
                "scheduler": resolved_scheduler_name,
                "governor": governor_name,
            }

        portfolio_coordinator = PortfolioRuntimeCoordinator(
            governor,
            allocation_provider=_allocation_provider,
            market_data_provider=_market_data_provider,
            stress_override_provider=stress_provider,
            slo_status_provider=slo_provider,
            metadata_provider=_metadata_provider,
        )
        scheduler.attach_portfolio_coordinator(portfolio_coordinator)

    for schedule in scheduler_cfg.schedules:
        strategy = strategies.get(schedule.strategy)
        if strategy is None:
            raise KeyError(f"Strategia {schedule.strategy} nie została zarejestrowana w konfiguracji")
        scheduler.register_schedule(
            name=schedule.name,
            strategy_name=schedule.strategy,
            strategy=strategy,
            feed=data_feed,
            sink=signal_sink,
            cadence_seconds=schedule.cadence_seconds,
            max_drift_seconds=schedule.max_drift_seconds,
            warmup_bars=schedule.warmup_bars,
            risk_profile=schedule.risk_profile,
            max_signals=schedule.max_signals,
        )

    return MultiStrategyRuntime(
        bootstrap=bootstrap_ctx,
        scheduler=scheduler,
        data_feed=data_feed,
        signal_sink=signal_sink,
        strategies=strategies,
        schedules=tuple(scheduler_cfg.schedules),
        portfolio_coordinator=portfolio_coordinator,
        portfolio_governor=portfolio_governor,
        tco_reporter=bootstrap_ctx.tco_reporter,
    )


def _instantiate_strategies(core_config: CoreConfig) -> dict[str, StrategyEngine]:
    registry: dict[str, StrategyEngine] = {}
    for name, cfg in getattr(core_config, "strategies", {}).items():
        registry[name] = DailyTrendMomentumStrategy(
            DailyTrendMomentumSettings(
                fast_ma=cfg.fast_ma,
                slow_ma=cfg.slow_ma,
                breakout_lookback=cfg.breakout_lookback,
                momentum_window=cfg.momentum_window,
                atr_window=cfg.atr_window,
                atr_multiplier=cfg.atr_multiplier,
                min_trend_strength=cfg.min_trend_strength,
                min_momentum=cfg.min_momentum,
            )
        )
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
    return registry


__all__ = [
    "DailyTrendPipeline",
    "build_daily_trend_pipeline",
    "create_trading_controller",
    "MultiStrategyRuntime",
    "build_multi_strategy_runtime",
]
