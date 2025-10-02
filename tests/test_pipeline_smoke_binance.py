"""Budowanie gotowych pipeline'ów strategii trend-following na podstawie konfiguracji."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Mapping, MutableMapping, Any

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
from bot_core.runtime.controller import DailyTrendController
from bot_core.security import SecretManager
from bot_core.strategies.daily_trend import DailyTrendMomentumSettings, DailyTrendMomentumStrategy

_DEFAULT_LEDGER_SUBDIR = Path("audit/ledger")

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


__all__ = ["DailyTrendPipeline", "build_daily_trend_pipeline", "create_trading_controller"]
