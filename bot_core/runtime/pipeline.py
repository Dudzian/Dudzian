"""Budowanie gotowych pipeline'ów strategii trend-following na podstawie konfiguracji."""
from __future__ import annotations

import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

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
from bot_core.exchanges.streaming import LocalLongPollStream, StreamBatch
from bot_core.market_intel import MarketIntelAggregator, MarketIntelQuery, MarketIntelSnapshot
from bot_core.portfolio import PortfolioDecisionLog, PortfolioGovernor
from bot_core.runtime.bootstrap import BootstrapContext, bootstrap_environment
from bot_core.runtime.multi_strategy_scheduler import (
    MultiStrategyScheduler,
    StrategyDataFeed,
    StrategySignalSink,
)
from bot_core.runtime.journal import TradingDecisionEvent, TradingDecisionJournal
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

try:  # pragma: no cover - moduł decision może być opcjonalny
    from bot_core.decision import DecisionCandidate, DecisionEvaluation
except Exception:  # pragma: no cover
    DecisionCandidate = None  # type: ignore
    DecisionEvaluation = Any  # type: ignore

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
# Konsumpcja streamów long-pollowych
# --------------------------------------------------------------------------------------


def consume_stream(
    stream: Iterable[StreamBatch],
    *,
    handle_batch: Callable[[StreamBatch], None],
    heartbeat_interval: float = 15.0,
    idle_timeout: float | None = 60.0,
    on_heartbeat: Callable[[float], None] | None = None,
    stop_condition: Callable[[], bool] | None = None,
    clock: Callable[[], float] | None = None,
) -> None:
    """Przetwarza paczki danych ze strumienia REST/gRPC.

    Funkcja jest synchroniczna – zakłada, że `stream` jest iteratorem zwracającym
    kolejne obiekty :class:`StreamBatch`. Obsługuje podstawowe heartbeaty oraz
    raportowanie braku danych poprzez wyjątek :class:`TimeoutError`.
    """

    iterator = iter(stream)
    closers: list[Callable[[], None]] = []
    for candidate in (getattr(iterator, "close", None), getattr(stream, "close", None)):
        if callable(candidate):
            closers.append(candidate)
    if closers:
        # deduplikacja referencji, np. gdy iterator i stream to ten sam obiekt
        unique_closers: dict[tuple[object | None, object], Callable[[], None]] = {}
        for closer in closers:
            key = (getattr(closer, "__self__", None), getattr(closer, "__func__", closer))
            unique_closers[key] = closer
        closers = list(unique_closers.values())
    time_source = clock or time.monotonic
    last_event_at = time_source()
    last_heartbeat_at = last_event_at
    heartbeat_interval = max(0.0, float(heartbeat_interval))
    timeout_value = None if idle_timeout is None else max(0.0, float(idle_timeout))

    try:
        while True:
            if stop_condition and stop_condition():
                break

            try:
                batch = next(iterator)
            except StopIteration:
                break

            observed_at = float(getattr(batch, "received_at", time_source()))
            if batch.events:
                handle_batch(batch)
                last_event_at = observed_at
                last_heartbeat_at = observed_at
            else:
                if batch.heartbeat:
                    if on_heartbeat:
                        on_heartbeat(observed_at)
                    last_heartbeat_at = observed_at
                elif on_heartbeat and (observed_at - last_heartbeat_at) >= heartbeat_interval:
                    on_heartbeat(observed_at)
                    last_heartbeat_at = observed_at

            if timeout_value is not None and (observed_at - last_event_at) >= timeout_value:
                raise TimeoutError(
                    f"Brak nowych danych w streamie '{batch.channel}' przez {timeout_value:.1f} s"
                )
    finally:
        for closer in closers:
            try:
                closer()
            except Exception:  # pragma: no cover - logowanie zamiast przerywania finalizacji
                _LOGGER.warning("Nie udało się zamknąć streamu long-pollowego", exc_info=True)


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
    stream_feed: "StreamingStrategyFeed | None" = None
    decision_sink: "DecisionAwareSignalSink | None" = None

    def shutdown(self) -> None:
        """Zatrzymuje komponenty dodatkowe (np. stream feed)."""

        if self.stream_feed is not None:
            self.stream_feed.stop()


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


class StreamingStrategyFeed(StrategyDataFeed):
    """Łączy `LocalLongPollStream` z interfejsem StrategyDataFeed."""

    def __init__(
        self,
        *,
        history_feed: StrategyDataFeed,
        stream_factory: Callable[[], Iterable[StreamBatch]],
        symbols_map: Mapping[str, Sequence[str]],
        buffer_size: int = 256,
        heartbeat_interval: float = 15.0,
        idle_timeout: float | None = 60.0,
        restart_delay: float = 5.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._history_feed = history_feed
        self._stream_factory = stream_factory
        self._symbols_map = {key: tuple(values) for key, values in symbols_map.items()}
        self._buffer_size = max(1, int(buffer_size))
        self._heartbeat_interval = max(0.0, float(heartbeat_interval))
        self._idle_timeout = idle_timeout if idle_timeout is None else max(0.0, float(idle_timeout))
        self._restart_delay = max(0.5, float(restart_delay))
        self._logger = logger or logging.getLogger(__name__)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._buffers: dict[str, deque[MarketSnapshot]] = {}
        self._known_symbols = {
            symbol
            for values in self._symbols_map.values()
            for symbol in values
        }
        for symbol in self._known_symbols:
            self._buffers[symbol] = deque(maxlen=self._buffer_size)
        self._last_event_at: float | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="strategy-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None

    def ingest_batch(self, batch: StreamBatch) -> None:
        """Przetwarza paczkę danych dostarczoną ze streamu."""

        if not batch.events:
            return
        snapshots: list[MarketSnapshot] = []
        for event in batch.events:
            try:
                snapshot = self._event_to_snapshot(event)
            except Exception:
                self._logger.debug("Nie udało się sparsować zdarzenia streamu: %s", event, exc_info=True)
                continue
            if snapshot is None:
                continue
            if self._known_symbols and snapshot.symbol not in self._known_symbols:
                continue
            snapshots.append(snapshot)
        if not snapshots:
            return
        with self._lock:
            for snapshot in snapshots:
                buffer = self._buffers.setdefault(snapshot.symbol, deque(maxlen=self._buffer_size))
                buffer.append(snapshot)
            self._last_event_at = time.monotonic()

    def load_history(self, strategy_name: str, bars: int) -> Sequence[MarketSnapshot]:
        return self._history_feed.load_history(strategy_name, bars)

    def fetch_latest(self, strategy_name: str) -> Sequence[MarketSnapshot]:
        symbols = self._symbols_map.get(strategy_name, ())
        if not symbols:
            return ()
        collected: list[MarketSnapshot] = []
        with self._lock:
            for symbol in symbols:
                queue = self._buffers.get(symbol)
                if not queue:
                    continue
                while queue:
                    collected.append(queue.popleft())
        collected.sort(key=lambda item: (item.symbol, item.timestamp))
        return tuple(collected)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                stream = self._stream_factory()
                consume_stream(
                    stream,
                    handle_batch=self.ingest_batch,
                    heartbeat_interval=self._heartbeat_interval,
                    idle_timeout=self._idle_timeout,
                    on_heartbeat=self._handle_heartbeat,
                    stop_condition=self._stop_event.is_set,
                )
            except TimeoutError:
                self._logger.warning("Brak nowych danych w streamie strategii przez dłuższy czas")
            except Exception:  # pragma: no cover - logowanie dla diagnostyki
                self._logger.exception("Błąd podczas przetwarzania streamu strategii")
            if self._stop_event.is_set():
                break
            time.sleep(self._restart_delay)

    def _handle_heartbeat(self, timestamp: float) -> None:
        if self._last_event_at is None:
            return
        drift = timestamp - self._last_event_at
        if drift > max(self._heartbeat_interval, 1.0):
            self._logger.debug("Opóźnienie streamu strategii %.2f s", drift)

    @staticmethod
    def _event_to_snapshot(event: Mapping[str, Any]) -> MarketSnapshot | None:
        symbol_raw = event.get("symbol") or event.get("pair") or event.get("instrument")
        if not symbol_raw:
            return None
        symbol = str(symbol_raw)
        timestamp_raw = (
            event.get("timestamp")
            or event.get("time")
            or event.get("ts")
            or time.time()
        )
        try:
            timestamp_value = float(timestamp_raw)
        except (TypeError, ValueError):
            timestamp_value = time.time()
        if timestamp_value > 1e15:
            timestamp_ms = int(timestamp_value)
        elif timestamp_value > 1e12:
            timestamp_ms = int(timestamp_value)
        else:
            timestamp_ms = int(timestamp_value * 1000.0)

        last_price = StreamingStrategyFeed._float(event.get("last_price") or event.get("price") or event.get("close"))
        if last_price is None:
            return None
        open_price = StreamingStrategyFeed._float(event.get("open_price") or event.get("open")) or last_price
        high_price = StreamingStrategyFeed._float(event.get("high_24h") or event.get("high")) or max(open_price, last_price)
        low_price = StreamingStrategyFeed._float(event.get("low_24h") or event.get("low")) or min(open_price, last_price)
        volume = StreamingStrategyFeed._float(
            event.get("volume_24h_base")
            or event.get("volume")
            or event.get("base_volume")
        ) or 0.0

        indicators: dict[str, float] = {}
        for key in (
            "best_bid",
            "best_ask",
            "price_change_percent",
            "volume_24h_quote",
        ):
            value = StreamingStrategyFeed._float(event.get(key))
            if value is not None:
                indicators[key] = value

        return MarketSnapshot(
            symbol=symbol,
            timestamp=timestamp_ms,
            open=open_price,
            high=high_price,
            low=low_price,
            close=last_price,
            volume=volume,
            indicators=indicators,
        )

    @staticmethod
    def _float(value: object | None) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def _build_streaming_feed(
    *,
    bootstrap: BootstrapContext,
    environment: EnvironmentConfig,
    base_feed: StrategyDataFeed,
    symbols_map: Mapping[str, Sequence[str]],
) -> StreamingStrategyFeed | None:
    stream_cfg = getattr(environment, "stream", None)
    adapter_settings = getattr(environment, "adapter_settings", None)
    if stream_cfg is None or not isinstance(adapter_settings, Mapping):
        return None
    stream_settings_raw = adapter_settings.get("stream")
    if not isinstance(stream_settings_raw, Mapping):
        return None
    stream_settings = dict(stream_settings_raw)
    host = getattr(stream_cfg, "host", "127.0.0.1")
    port = getattr(stream_cfg, "port", 8765)
    base_url = str(stream_settings.get("base_url") or f"http://{host}:{port}")
    default_path = f"/stream/{environment.exchange}/public"
    path = str(
        stream_settings.get("public_path")
        or stream_settings.get("path")
        or default_path
    )
    raw_channels = stream_settings.get("public_channels") or stream_settings.get("channels")
    channels: list[str]
    if raw_channels is None:
        channels = ["ticker"]
    elif isinstance(raw_channels, str):
        channels = [token.strip() for token in raw_channels.split(",") if token.strip()]
    else:
        channels = [str(token).strip() for token in raw_channels if str(token).strip()]
    if not channels:
        channels = ["ticker"]
    all_symbols = tuple(
        dict.fromkeys(symbol for values in symbols_map.values() for symbol in values)
    )

    def _build_params() -> dict[str, object]:
        params: dict[str, object] = {}
        base_params = stream_settings.get("params")
        if isinstance(base_params, Mapping):
            params.update(base_params)
        public_params = stream_settings.get("public_params")
        if isinstance(public_params, Mapping):
            params.update(public_params)
        if "symbols" not in params and "symbol" not in params and all_symbols:
            params["symbols"] = ",".join(all_symbols)
        return params

    def _build_headers() -> Mapping[str, str] | None:
        headers = stream_settings.get("headers")
        if not isinstance(headers, Mapping):
            return None
        return {str(key): str(value) for key, value in headers.items()}

    channel_param = stream_settings.get("public_channel_param")
    if channel_param is None:
        channel_param = stream_settings.get("channel_param")
    channel_param = str(channel_param).strip() if channel_param not in (None, "") else None
    cursor_param = stream_settings.get("public_cursor_param")
    if cursor_param is None:
        cursor_param = stream_settings.get("cursor_param")
    cursor_param = str(cursor_param).strip() if cursor_param not in (None, "") else None
    initial_cursor = stream_settings.get("public_initial_cursor")
    if initial_cursor is None:
        initial_cursor = stream_settings.get("initial_cursor")

    serializer = stream_settings.get("public_channel_serializer")
    if serializer is None:
        serializer = stream_settings.get("channel_serializer")
    channel_serializer = None
    if callable(serializer):
        channel_serializer = serializer
    else:
        separator = stream_settings.get("public_channel_separator")
        if separator is None:
            separator = stream_settings.get("channel_separator")
        if isinstance(separator, str) and separator:
            channel_serializer = lambda values, sep=separator: sep.join(values)  # noqa: E731

    def _build_body_params() -> Mapping[str, object] | None:
        body: dict[str, object] = {}
        base_body = stream_settings.get("body_params")
        if isinstance(base_body, Mapping):
            body.update(base_body)
        scope_body = stream_settings.get("public_body_params")
        if isinstance(scope_body, Mapping):
            body.update(scope_body)
        return body or None

    body_encoder = stream_settings.get("public_body_encoder")
    if body_encoder is None:
        body_encoder = stream_settings.get("body_encoder")

    jitter_raw = stream_settings.get("jitter")
    if isinstance(jitter_raw, Sequence):
        try:
            jitter = tuple(float(item) for item in jitter_raw)
        except (TypeError, ValueError):
            jitter = (0.05, 0.30)
        if len(jitter) != 2:
            jitter = (0.05, 0.30)
    else:
        jitter = (0.05, 0.30)

    poll_interval = float(stream_settings.get("poll_interval", 0.5))
    timeout = float(stream_settings.get("timeout", 10.0))
    max_retries = int(stream_settings.get("max_retries", 3))
    backoff_base = float(stream_settings.get("backoff_base", 0.25))
    backoff_cap = float(stream_settings.get("backoff_cap", 2.0))
    http_method = stream_settings.get("public_method") or stream_settings.get("method", "GET")
    params_in_body = bool(stream_settings.get("public_params_in_body", stream_settings.get("params_in_body", False)))
    channels_in_body = bool(stream_settings.get("public_channels_in_body", stream_settings.get("channels_in_body", False)))
    cursor_in_body = bool(stream_settings.get("public_cursor_in_body", stream_settings.get("cursor_in_body", False)))

    def _factory() -> LocalLongPollStream:
        return LocalLongPollStream(
            base_url=base_url,
            path=path,
            channels=channels,
            adapter=environment.exchange,
            scope="public",
            environment=environment.environment.value,
            params=_build_params(),
            headers=_build_headers(),
            poll_interval=poll_interval,
            timeout=timeout,
            max_retries=max_retries,
            backoff_base=backoff_base,
            backoff_cap=backoff_cap,
            jitter=jitter,
            channel_param=channel_param,
            cursor_param=cursor_param,
            initial_cursor=initial_cursor,
            channel_serializer=channel_serializer,
            http_method=str(http_method or "GET"),
            params_in_body=params_in_body,
            channels_in_body=channels_in_body,
            cursor_in_body=cursor_in_body,
            body_params=_build_body_params(),
            body_encoder=body_encoder,
        )

    heartbeat_interval = float(stream_settings.get("heartbeat_interval", 15.0))
    idle_timeout_raw = stream_settings.get("idle_timeout")
    idle_timeout = None if idle_timeout_raw in (None, "") else float(idle_timeout_raw)
    restart_delay = float(stream_settings.get("restart_delay", 5.0))
    buffer_size = int(stream_settings.get("buffer_size", 256))

    return StreamingStrategyFeed(
        history_feed=base_feed,
        stream_factory=_factory,
        symbols_map=symbols_map,
        buffer_size=buffer_size,
        heartbeat_interval=heartbeat_interval,
        idle_timeout=idle_timeout,
        restart_delay=restart_delay,
        logger=_LOGGER,
    )


def _estimate_default_notional(paper_settings: Mapping[str, object]) -> float:
    balances = paper_settings.get("initial_balances") or {}
    valuation_asset = str(paper_settings.get("valuation_asset", "USDT"))
    balance = 0.0
    if isinstance(balances, Mapping):
        try:
            balance = float(balances.get(valuation_asset, 0.0))
        except (TypeError, ValueError):
            balance = 0.0
        if balance <= 0:
            for value in balances.values():
                try:
                    balance = float(value)
                except (TypeError, ValueError):
                    continue
                if balance > 0:
                    break
    position_size = 0.0
    try:
        position_size = float(paper_settings.get("position_size", 0.1))
    except (TypeError, ValueError):
        position_size = 0.1
    default_notional = balance * max(position_size, 0.0)
    if default_notional <= 0:
        default_notional = 1_000.0
    return default_notional


def _build_decision_sink(
    *,
    bootstrap: BootstrapContext,
    base_sink: InMemoryStrategySignalSink,
    default_notional: float,
    environment_name: str,
    portfolio_id: str | None,
) -> DecisionAwareSignalSink | None:
    orchestrator = getattr(bootstrap, "decision_orchestrator", None)
    if orchestrator is None or DecisionCandidate is None:
        return None
    risk_engine = getattr(bootstrap, "risk_engine", None)
    if risk_engine is None:
        return None
    decision_config = getattr(bootstrap, "decision_engine_config", None)
    min_probability = 0.55
    if decision_config is not None:
        try:
            min_probability = float(getattr(decision_config, "min_probability", min_probability))
        except (TypeError, ValueError):  # pragma: no cover - konfiguracja może zawierać tekst
            min_probability = 0.55
    exchange_name = getattr(bootstrap.environment, "exchange", "")
    journal = getattr(bootstrap, "decision_journal", None)
    return DecisionAwareSignalSink(
        base_sink=base_sink,
        orchestrator=orchestrator,
        risk_engine=risk_engine,
        default_notional=default_notional,
        environment=environment_name,
        exchange=str(exchange_name or ""),
        min_probability=min_probability,
        portfolio=portfolio_id,
        journal=journal,
    )
class DecisionAwareSignalSink(StrategySignalSink):
    """Filtruje sygnały strategii przez DecisionOrchestratora."""

    def __init__(
        self,
        *,
        base_sink: InMemoryStrategySignalSink,
        orchestrator: Any,
        risk_engine: Any,
        default_notional: float,
        environment: str,
        exchange: str,
        min_probability: float = 0.55,
        portfolio: str | None = None,
        journal: TradingDecisionJournal | None = None,
    ) -> None:
        self._base_sink = base_sink
        self._orchestrator = orchestrator
        self._risk_engine = risk_engine
        self._default_notional = max(0.0, float(default_notional)) or 1_000.0
        self._environment = environment
        self._exchange = exchange
        self._min_probability = max(0.0, min(1.0, float(min_probability)))
        self._logger = logging.getLogger(__name__)
        self._evaluations: list[DecisionEvaluation] = []
        self._portfolio = str(portfolio) if portfolio is not None else ""
        self._journal = journal

    def submit(
        self,
        *,
        strategy_name: str,
        schedule_name: str,
        risk_profile: str,
        timestamp: datetime,
        signals: Sequence[StrategySignal],
    ) -> None:
        if DecisionCandidate is None or self._orchestrator is None:
            self._base_sink.submit(
                strategy_name=strategy_name,
                schedule_name=schedule_name,
                risk_profile=risk_profile,
                timestamp=timestamp,
                signals=signals,
            )
            return

        accepted: list[StrategySignal] = []
        risk_snapshot = self._build_risk_snapshot(risk_profile)
        for signal in signals:
            candidate = self._build_candidate(strategy_name, risk_profile, signal)
            if candidate is None:
                continue
            try:
                evaluation = self._orchestrator.evaluate_candidate(candidate, risk_snapshot)
            except Exception:  # pragma: no cover - diagnostyka orchestratora
                self._logger.exception("DecisionOrchestrator odrzucił kandydata przez wyjątek")
                continue
            self._evaluations.append(evaluation)
            self._record_evaluation(
                evaluation=evaluation,
                candidate=candidate,
                signal=signal,
                strategy_name=strategy_name,
                schedule_name=schedule_name,
                risk_profile=risk_profile,
                timestamp=timestamp,
            )
            if evaluation.accepted:
                accepted.append(signal)
            else:
                self._logger.debug(
                    "DecisionOrchestrator odrzucił sygnał %s/%s: %s",
                    strategy_name,
                    signal.symbol,
                    ", ".join(evaluation.reasons) or "brak powodów",
                )

        if not accepted:
            return

        self._base_sink.submit(
            strategy_name=strategy_name,
            schedule_name=schedule_name,
            risk_profile=risk_profile,
            timestamp=timestamp,
            signals=tuple(accepted),
        )

    def export(self) -> Sequence[tuple[str, Sequence[StrategySignal]]]:
        return self._base_sink.export()

    def evaluations(self) -> Sequence[DecisionEvaluation]:
        return tuple(self._evaluations)

    def _record_evaluation(
        self,
        *,
        evaluation: DecisionEvaluation,
        candidate: DecisionCandidate,
        signal: StrategySignal,
        strategy_name: str,
        schedule_name: str,
        risk_profile: str,
        timestamp: datetime,
    ) -> None:
        journal = self._journal
        if journal is None or TradingDecisionEvent is None:
            return

        metadata: dict[str, str] = {}
        candidate_metadata = getattr(candidate, "metadata", {})
        if isinstance(candidate_metadata, Mapping):
            for key, value in candidate_metadata.items():
                metadata.setdefault(str(key), str(value))

        def _store_float(name: str, value: float | None) -> None:
            if value is None:
                return
            try:
                metadata.setdefault(name, f"{float(value):.6f}")
            except (TypeError, ValueError):  # pragma: no cover - defensywnie
                return

        _store_float("expected_probability", getattr(candidate, "expected_probability", None))
        _store_float("expected_return_bps", getattr(candidate, "expected_return_bps", None))
        _store_float("notional", getattr(candidate, "notional", None))
        _store_float("cost_bps", getattr(evaluation, "cost_bps", None))
        _store_float("net_edge_bps", getattr(evaluation, "net_edge_bps", None))

        reasons = getattr(evaluation, "reasons", ())
        if reasons:
            metadata.setdefault("decision_reasons", ";".join(str(reason) for reason in reasons))
        risk_flags = getattr(evaluation, "risk_flags", ())
        if risk_flags:
            metadata.setdefault("risk_flags", ";".join(str(flag) for flag in risk_flags))
        stress_failures = getattr(evaluation, "stress_failures", ())
        if stress_failures:
            metadata.setdefault(
                "stress_failures",
                ";".join(str(failure) for failure in stress_failures),
            )

        metadata.setdefault("decision_status", "accepted" if evaluation.accepted else "rejected")
        metadata.setdefault("source", "decision_orchestrator")

        portfolio = self._portfolio or metadata.get("portfolio_id") or metadata.get("portfolio")
        if portfolio is None:
            portfolio = self._environment
        latency_ms = getattr(evaluation, "latency_ms", None)

        event = TradingDecisionEvent(
            event_type="decision_evaluation",
            timestamp=timestamp,
            environment=self._environment,
            portfolio=str(portfolio),
            risk_profile=risk_profile,
            symbol=getattr(candidate, "symbol", signal.symbol),
            side=str(signal.side),
            schedule=schedule_name,
            strategy=strategy_name,
            status="accepted" if evaluation.accepted else "rejected",
            confidence=float(signal.confidence),
            latency_ms=latency_ms if isinstance(latency_ms, (int, float)) else None,
            telemetry_namespace=f"{self._environment}.decision.{schedule_name}",
            metadata=metadata,
        )

        try:
            journal.record(event)
        except Exception:  # pragma: no cover - dziennik nie powinien blokować handlu
            self._logger.debug("Nie udało się zapisać decision_evaluation", exc_info=True)

    def _build_candidate(
        self,
        strategy_name: str,
        risk_profile: str,
        signal: StrategySignal,
    ) -> DecisionCandidate | None:
        if DecisionCandidate is None:
            return None
        raw_metadata = getattr(signal, "metadata", None)
        if isinstance(raw_metadata, Mapping):
            metadata = dict(raw_metadata)
        elif raw_metadata is None:
            metadata = {}
        else:
            try:
                metadata = dict(raw_metadata)  # type: ignore[arg-type]
            except Exception:  # pragma: no cover - defensywnie obsługujemy nietypowe typy
                metadata = {}
        probability = self._extract_probability(signal)
        if probability < self._min_probability:
            return None
        expected_return = self._extract_expected_return(signal, metadata)
        cost_override = self._extract_cost_override(metadata)
        latency_ms = self._extract_latency(metadata)
        notional = self._extract_notional(metadata)
        action = "exit" if str(signal.side).upper() in {"SELL", "EXIT", "FLAT"} else "enter"
        metadata.setdefault("environment", self._environment)
        metadata.setdefault("exchange", self._exchange)
        metadata.setdefault("schedule", strategy_name)
        return DecisionCandidate(
            strategy=strategy_name,
            action=action,
            risk_profile=risk_profile,
            symbol=signal.symbol,
            notional=notional,
            expected_return_bps=expected_return,
            expected_probability=probability,
            cost_bps_override=cost_override,
            latency_ms=latency_ms,
            metadata=metadata,
        )

    def _build_risk_snapshot(self, risk_profile: str) -> Mapping[str, object]:
        loader = getattr(self._risk_engine, "snapshot_state", None)
        if callable(loader):
            try:
                snapshot = loader(risk_profile)
                if snapshot is not None:
                    return snapshot
            except Exception:  # pragma: no cover - diagnostyka risk engine
                self._logger.debug("Nie udało się pobrać snapshotu ryzyka", exc_info=True)
        return {}

    def _extract_probability(self, signal: StrategySignal) -> float:
        metadata_prob = None
        metadata = signal.metadata or {}
        if isinstance(metadata, Mapping):
            candidate = metadata.get("expected_probability") or metadata.get("probability")
            if candidate is None and isinstance(metadata.get("ai_manager"), Mapping):
                candidate = metadata["ai_manager"].get("success_probability")
            metadata_prob = candidate
        prob = None
        if metadata_prob is not None:
            try:
                prob = float(metadata_prob)
            except (TypeError, ValueError):
                prob = None
        if prob is None:
            prob = float(signal.confidence)
        return max(self._min_probability, min(0.995, prob))

    def _extract_expected_return(self, signal: StrategySignal, metadata: Mapping[str, Any]) -> float:
        candidate = metadata.get("expected_return_bps")
        if candidate is None and isinstance(metadata.get("ai_manager"), Mapping):
            candidate = metadata["ai_manager"].get("expected_return_bps")
        if candidate is None:
            base = max(0.0, float(signal.confidence) - 0.5)
            candidate = 5.0 + base * 20.0
        try:
            return float(candidate)
        except (TypeError, ValueError):
            return 5.0

    def _extract_cost_override(self, metadata: Mapping[str, Any]) -> float | None:
        candidate = metadata.get("cost_bps") or metadata.get("slippage_bps")
        if candidate is None:
            return None
        try:
            return float(candidate)
        except (TypeError, ValueError):
            return None

    def _extract_latency(self, metadata: Mapping[str, Any]) -> float | None:
        latency = metadata.get("latency_ms") or metadata.get("latency")
        if latency is None and isinstance(metadata.get("decision_engine"), Mapping):
            latency = metadata["decision_engine"].get("latency_ms")
        if latency is None:
            return None
        try:
            return float(latency)
        except (TypeError, ValueError):
            return None

    def _extract_notional(self, metadata: Mapping[str, Any]) -> float:
        candidate = metadata.get("notional") or metadata.get("target_notional")
        if candidate is None:
            return self._default_notional
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            return self._default_notional
        if value <= 0:
            return self._default_notional
        return value


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

    base_feed = OHLCVStrategyFeed(cached_source, symbols_map=symbols_map, interval_map=interval_map)
    stream_feed = _build_streaming_feed(
        bootstrap=bootstrap_ctx,
        environment=environment,
        base_feed=base_feed,
        symbols_map=symbols_map,
    )
    if stream_feed is not None:
        stream_feed.start()
        data_feed: StrategyDataFeed = stream_feed
    else:
        data_feed = base_feed
    base_sink = InMemoryStrategySignalSink()
    decision_sink = _build_decision_sink(
        bootstrap=bootstrap_ctx,
        base_sink=base_sink,
        default_notional=_estimate_default_notional(paper_settings),
        environment_name=environment_name,
        portfolio_id=str(paper_settings.get("portfolio_id", "")),
    )
    signal_sink: StrategySignalSink = decision_sink or base_sink
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
        stream_feed=stream_feed,
        decision_sink=decision_sink,
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
    "consume_stream",
    "OHLCVStrategyFeed",
    "InMemoryStrategySignalSink",
    "StreamingStrategyFeed",
    "DecisionAwareSignalSink",
]
