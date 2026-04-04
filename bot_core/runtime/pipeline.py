"""Budowanie gotowych pipeline'ów strategii trend-following na podstawie konfiguracji."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import threading
import time
import weakref
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
)

from bot_core.runtime.observability import AlertSink
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
    MultiStrategySuspensionConfig,
    SignalLimitOverrideConfig,
    RuntimeAppConfig,
    RuntimeExecutionSettings,
    RuntimeOptimizationSettings,
    StrategyOptimizationTaskConfig,
)
from bot_core.ai.repository import FilesystemModelRepository, ModelRepository
from bot_core.ai.opportunity_shadow_adapter import OpportunityRuntimeShadowAdapter
from bot_core.ai.trading_engine import TradingOpportunityAI
from bot_core.data import CachedOHLCVSource
from bot_core.data.base import OHLCVRequest, OHLCVResponse
from bot_core.data.ohlcv import OHLCVBackfillService
from bot_core.execution.base import ExecutionContext, ExecutionService, PriceResolver
from bot_core.execution.paper import MarketMetadata, PaperTradingExecutionService
from bot_core.execution import resolve_execution_mode
from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeAdapterFactory,
)
from bot_core.exchanges.streaming import LocalLongPollStream, StreamBatch
from bot_core.observability.metrics import MetricsRegistry
from bot_core.market_intel import MarketIntelAggregator, MarketIntelQuery, MarketIntelSnapshot
from bot_core.optimization import (
    OptimizationScheduler,
    OptimizationTaskQueue,
    StrategyOptimizer,
)
from bot_core.portfolio import (
    MultiPortfolioScheduler,
    PortfolioDecision,
    PortfolioDecisionLog,
    PortfolioGovernor,
    StrategyHealthMonitor,
    StrategyPortfolioGovernor,
    load_market_intel_report,
)
from bot_core.runtime.bootstrap import BootstrapContext, bootstrap_environment
from bot_core.risk import StressOverrideRecommendation
from bot_core.security.guards import (
    LicenseCapabilityError,
    get_capability_guard,
)
from bot_core.runtime.capital_policies import (
    BlendedCapitalAllocation,
    CapitalAllocationPolicy,
    DrawdownAdaptiveAllocation,
    EqualWeightAllocation,
    FixedWeightAllocation,
    MetricWeightRule,
    MetricWeightedAllocation,
    RiskParityAllocation,
    RiskProfileBudgetAllocation,
    SignalStrengthAllocation,
    SmoothedCapitalAllocationPolicy,
    TagQuotaAllocation,
    VolatilityTargetAllocation,
)
from bot_core.runtime.multi_strategy_scheduler import (
    MultiStrategyScheduler,
    StrategyDataFeed,
    StrategySignalSink,
)
from bot_core.runtime.scheduler import AsyncIOTaskQueue
from bot_core.runtime.journal import TradingDecisionEvent, TradingDecisionJournal
from bot_core.runtime.portfolio_coordinator import PortfolioRuntimeCoordinator
from bot_core.runtime.portfolio_inputs import (
    build_slo_status_provider,
    build_stress_override_provider,
    load_stress_overrides,
)
from bot_core.runtime.tco_reporting import RuntimeTCOReporter
from bot_core.runtime.controller import DailyTrendController
from bot_core.runtime.pipeline_config_loader import PipelineConfigLoader
from bot_core.runtime.strategy_bootstrapper import StrategyBootstrapper
from bot_core.runtime.execution_bootstrapper import ExecutionBootstrapper
from bot_core.runtime.risk_bootstrapper import RiskBootstrapper
from bot_core.runtime.data_source_bootstrapper import DataSourceBootstrapper
from bot_core.security import SecretManager
from bot_core.strategies.base import StrategyEngine, StrategySignal, MarketSnapshot
from bot_core.strategies.catalog import (
    DEFAULT_STRATEGY_CATALOG,
    StrategyCatalog,
    StrategyDefinition,
)


try:  # pragma: no cover - strategia może być opcjonalna w starszych gałęziach
    from bot_core.strategies.daily_trend import (  # type: ignore
        DailyTrendMomentumSettings,
        DailyTrendMomentumStrategy,
    )
except Exception:  # pragma: no cover - fallback gdy moduł nie istnieje
    DailyTrendMomentumSettings = None  # type: ignore
    DailyTrendMomentumStrategy = None  # type: ignore

try:  # pragma: no cover - moduł decision może być opcjonalny
    from bot_core.decision import (
        DecisionCandidate,
        DecisionContext,
        DecisionEvaluation,
        summarize_evaluation_payloads,
    )
except Exception:  # pragma: no cover
    DecisionCandidate = None  # type: ignore
    DecisionContext = Any  # type: ignore
    DecisionEvaluation = Any  # type: ignore
    summarize_evaluation_payloads = None  # type: ignore

_DEFAULT_LEDGER_SUBDIR = Path("audit/ledger")
_LOGGER = logging.getLogger(__name__)
_TEST_MODE_ENV = "DUDZIAN_TEST_MODE"
_PIPELINE_THREAD_NAME = "PipelineStream"
_PIPELINE_CONFIG_LOADER = PipelineConfigLoader()
_STRATEGY_BOOTSTRAPPER = StrategyBootstrapper()
_EXECUTION_BOOTSTRAPPER = ExecutionBootstrapper()
_RISK_BOOTSTRAPPER = RiskBootstrapper()
_DATA_SOURCE_BOOTSTRAPPER = DataSourceBootstrapper()


def _is_test_mode_enabled() -> bool:
    return os.getenv(_TEST_MODE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _to_path(value: object | None) -> Path | None:
    if value in (None, "", False):
        return None
    try:
        path = Path(str(value)).expanduser()
    except Exception:
        return None
    return path


def _unique_paths(paths: Iterable[Path | None]) -> tuple[Path, ...]:
    seen: set[str] = set()
    result: list[Path] = []
    for candidate in paths:
        if candidate is None:
            continue
        normalized = candidate.expanduser()
        key = str(normalized)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return tuple(result)


def _resolve_latest_report(
    directories: Sequence[Path | None], *, prefix: str, suffix: str = ".json"
) -> Path | None:
    latest_path: Path | None = None
    latest_mtime = float("-inf")
    for base in directories:
        if base is None:
            continue
        directory = base.expanduser()
        if not directory.exists() or not directory.is_dir():
            continue
        try:
            candidates = list(directory.glob(f"{prefix}*{suffix}"))
        except Exception:  # pragma: no cover - glob errors should not break runtime
            _LOGGER.debug("Nie udało się przeszukać katalogu %s", directory, exc_info=True)
            continue
        for candidate in candidates:
            try:
                mtime = candidate.stat().st_mtime
            except OSError:
                continue
            if mtime > latest_mtime:
                latest_path = candidate
                latest_mtime = mtime
    return latest_path


def _load_market_intel_snapshots_from_reports(
    governor_name: str,
    config: object | None,
    directories: Sequence[Path],
) -> Mapping[str, MarketIntelSnapshot]:
    search_dirs: list[Path | None] = list(directories)
    output_dir = _to_path(getattr(config, "output_directory", None)) if config else None
    if output_dir is not None:
        search_dirs.insert(0, output_dir)
    normalized_dirs = _unique_paths(search_dirs)
    if not normalized_dirs:
        return {}

    name_slug = governor_name.strip().lower().replace(" ", "_") or "portfolio"
    prefixes = (
        f"market_intel_{name_slug}_",
        f"marketintel_{name_slug}_",
        "market_intel_",
    )

    for prefix in prefixes:
        report_path = _resolve_latest_report(normalized_dirs, prefix=prefix)
        if report_path is None:
            continue
        try:
            snapshots, _ = load_market_intel_report(report_path)
        except Exception:  # pragma: no cover - diagnostyka raportów
            _LOGGER.exception(
                "PortfolioGovernor: błąd wczytania raportu Market Intel %s", report_path
            )
            continue
        if snapshots:
            _LOGGER.debug(
                "PortfolioGovernor: użyto fallbackowego raportu Market Intel %s",
                report_path,
            )
            return snapshots
    return {}


def _resolve_latest_stress_report(governor_name: str, directories: Sequence[Path]) -> Path | None:
    name_slug = governor_name.strip().lower().replace(" ", "_") or "portfolio"
    prefixes = (
        f"stress_lab_{name_slug}_",
        "stress_lab_",
    )
    normalized_dirs = _unique_paths(directories)
    for prefix in prefixes:
        report_path = _resolve_latest_report(normalized_dirs, prefix=prefix)
        if report_path is not None:
            return report_path
    return None


def _load_stress_overrides_from_reports(
    governor_name: str,
    directories: Sequence[Path],
    *,
    max_age: timedelta | None = None,
) -> tuple[StressOverrideRecommendation, ...]:
    report_path = _resolve_latest_stress_report(governor_name, directories)
    if report_path is None:
        return ()
    try:
        overrides = load_stress_overrides(report_path, max_age=max_age)
    except Exception:  # pragma: no cover - diagnostyka raportów
        _LOGGER.exception("PortfolioGovernor: błąd wczytania raportu Stress Lab %s", report_path)
        return ()
    if overrides:
        _LOGGER.debug("PortfolioGovernor: użyto fallbackowego raportu Stress Lab %s", report_path)
    return overrides


def _ensure_local_market_data_availability(
    environment: EnvironmentConfig,
    data_source: CachedOHLCVSource,
    markets: Mapping[str, MarketMetadata],
    interval: str,
    *,
    backfill_service: OHLCVBackfillService | None = None,
    adapter: ExchangeAdapter | None = None,
) -> None:
    """Zapewnia minimalny zestaw danych OHLCV wymaganych do startu runtime."""
    _DATA_SOURCE_BOOTSTRAPPER.ensure_local_market_data_availability(
        environment=environment,
        data_source=data_source,
        markets=markets,
        interval=interval,
        backfill_service=backfill_service,
        adapter=adapter,
    )


def _minutes_to_timedelta(value: float | int | None, default_minutes: float) -> timedelta | None:
    minutes = default_minutes if value in (None, "") else float(value)
    if minutes <= 0:
        return None
    return timedelta(minutes=minutes)


def _safe_float(value: Any, *, default: float) -> float:
    if value in (None, ""):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _resolve_environment_name_for_mode(
    core_config: CoreConfig,
    *,
    aliases: Sequence[str],
    environment_type: Environment | None,
    prefer_offline: bool | None = None,
) -> str | None:
    lowered_aliases = tuple(alias.lower() for alias in aliases)

    def _matches(name: str) -> bool:
        lowered = name.lower()
        return any(alias in lowered for alias in lowered_aliases)

    for name in core_config.environments.keys():
        if _matches(name):
            return name

    if prefer_offline is not None:
        for name, env in core_config.environments.items():
            if bool(getattr(env, "offline_mode", False)) is prefer_offline:
                return name

    if environment_type is not None:
        for name, env in core_config.environments.items():
            env_kind = getattr(env, "environment", None)
            if env_kind is environment_type:
                return name
            if isinstance(env_kind, str) and env_kind.lower() == environment_type.value:
                return name

    return next(iter(core_config.environments), None)


def _create_cached_source(
    adapter: ExchangeAdapter, environment: EnvironmentConfig
) -> CachedOHLCVSource:
    """Buduje źródło OHLCV korzystające z lokalnego cache i snapshotów REST."""

    return _DATA_SOURCE_BOOTSTRAPPER.create_cached_source(adapter=adapter, environment=environment)


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
    core_config: CoreConfig | None = None,
    runtime_config: RuntimeAppConfig | None = None,
) -> DailyTrendPipeline:
    """Tworzy kompletny pipeline strategii trend-following D1 dla środowiska paper/testnet."""
    bootstrap_ctx = _RISK_BOOTSTRAPPER.bootstrap_context(
        environment_name=environment_name,
        config_path=config_path,
        secret_manager=secret_manager,
        adapter_factories=adapter_factories,
        risk_profile_name=risk_profile_name,
        core_config=core_config,
        bootstrap_fn=bootstrap_environment,
    )
    core_config = bootstrap_ctx.core_config
    environment = bootstrap_ctx.environment
    effective_risk_profile = bootstrap_ctx.risk_profile_name
    guard = getattr(bootstrap_ctx, "capability_guard", None) or get_capability_guard()

    resolved_strategy_name = strategy_name or getattr(environment, "default_strategy", None)
    if not resolved_strategy_name:
        raise ValueError(
            "Środowisko '{environment}' nie ma zdefiniowanej domyślnej strategii, a parametr strategy_name nie został podany.".format(
                environment=environment_name
            )
        )

    if guard is not None:
        try:
            guard.require_strategy(
                "trend_d1",
                message=(
                    f"Strategia Trend D1 '{resolved_strategy_name}' wymaga aktywnej licencji Trend D1."
                ),
            )
            guard.reserve_slot("bot")
            slot_kind = (
                "live_controller"
                if environment.environment is Environment.LIVE
                else "paper_controller"
            )
            guard.reserve_slot(slot_kind)
        except LicenseCapabilityError as exc:
            raise RuntimeError(str(exc)) from exc

    resolved_controller_name = controller_name or getattr(environment, "default_controller", None)
    if not resolved_controller_name:
        raise ValueError(
            "Środowisko '{environment}' nie ma zdefiniowanego domyślnego kontrolera runtime, a parametr controller_name nie został podany.".format(
                environment=environment_name
            )
        )

    strategy_cfg = _resolve_strategy(core_config, resolved_strategy_name)
    runtime_cfg = _resolve_runtime(core_config, resolved_controller_name)
    universe = _resolve_universe(core_config, environment)

    execution_settings: RuntimeExecutionSettings | None = (
        getattr(runtime_config, "execution", None) if runtime_config is not None else None
    )
    execution_mode = resolve_execution_mode(execution_settings, environment)

    if execution_mode == "paper":
        paper_settings = _normalize_paper_settings(environment)
    else:
        paper_settings = _derive_live_settings(environment, universe)
    allowed_quotes = paper_settings["allowed_quotes"]

    markets = _build_markets(universe, environment.exchange, allowed_quotes, paper_settings)
    if not markets:
        raise ValueError(
            "Brak instrumentów spełniających kryteria paper tradingu – skonfiguruj quote_assets/valuation_asset."
        )

    cached_source = _create_cached_source(bootstrap_ctx.adapter, environment)
    backfill_service = OHLCVBackfillService(cached_source)
    _ensure_local_market_data_availability(
        environment,
        cached_source,
        markets,
        runtime_cfg.interval,
        backfill_service=backfill_service,
        adapter=bootstrap_ctx.adapter,
    )
    storage = cached_source.storage

    price_resolver = _build_price_resolver(cached_source, runtime_cfg.interval)

    execution_service = _select_execution_service(
        bootstrap_ctx=bootstrap_ctx,
        markets=markets,
        paper_settings=paper_settings,
        runtime_settings=execution_settings,
        execution_mode=execution_mode,
        price_resolver=price_resolver,
    )

    if DailyTrendMomentumStrategy is None or DailyTrendMomentumSettings is None:
        raise RuntimeError("Moduł daily_trend_momentum nie jest dostępny w tej wersji instalacji.")

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
    execution_metadata["mode"] = execution_mode

    execution_context = ExecutionContext(
        portfolio_id=paper_settings["portfolio_id"],
        risk_profile=effective_risk_profile,
        environment=environment.environment.value,
        metadata=execution_metadata,
        price_resolver=price_resolver,
    )

    if isinstance(execution_service, PaperTradingExecutionService):
        account_loader = _build_account_loader(
            execution_service=execution_service,
            data_source=cached_source,
            markets=markets,
            interval=runtime_cfg.interval,
            valuation_asset=paper_settings["valuation_asset"],
            cash_assets=allowed_quotes,
        )
    else:
        adapter = getattr(bootstrap_ctx, "adapter", None)
        if not isinstance(adapter, ExchangeAdapter):
            raise RuntimeError(
                "Tryb live wymaga aktywnego adaptera giełdowego udostępnionego przez bootstrap"
            )
        account_loader = _build_live_account_loader(adapter)

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
    alert_router: AlertSink,
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


async def consume_stream_async(
    stream: AsyncIterable[StreamBatch] | LocalLongPollStream,
    *,
    handle_batch: Callable[[StreamBatch], Awaitable[None] | None],
    heartbeat_interval: float = 15.0,
    idle_timeout: float | None = 60.0,
    on_heartbeat: Callable[[float], Awaitable[None] | None] | None = None,
    stop_condition: Callable[[], Awaitable[bool] | bool] | None = None,
    clock: Callable[[], float] | None = None,
) -> None:
    """Asynchroniczna wersja :func:`consume_stream`.

    Funkcja oczekuje asynchronicznego iteratora (np. :class:`LocalLongPollStream`
    używanego w trybie async) i zapewnia te same gwarancje dotyczące heartbeatów,
    limitów bezczynności oraz finalizacji strumienia.
    """

    if isinstance(stream, AsyncIterator):
        iterator: AsyncIterator[StreamBatch] = stream
    else:
        aiter = getattr(stream, "__aiter__", None)
        if callable(aiter):
            iterator = aiter()
        else:
            raise TypeError("consume_stream_async wymaga asynchronicznego strumienia")

    closers: list[Callable[[], object]] = []
    for target in (iterator, stream):
        if target is None:
            continue
        closer = getattr(target, "aclose", None)
        if not callable(closer):
            closer = getattr(target, "close", None)
        if callable(closer):
            closers.append(closer)
    if closers:
        unique: dict[tuple[object | None, object], Callable[[], object]] = {}
        for closer in closers:
            key = (getattr(closer, "__self__", None), getattr(closer, "__func__", closer))
            unique[key] = closer
        closers = list(unique.values())

    time_source = clock or time.monotonic
    last_event_at = time_source()
    last_heartbeat_at = last_event_at
    heartbeat_interval = max(0.0, float(heartbeat_interval))
    timeout_value = None if idle_timeout is None else max(0.0, float(idle_timeout))

    async def _call_maybe_async(func: Callable[..., object] | None, *args: object) -> object | None:
        if func is None:
            return None
        result = func(*args)
        if inspect.isawaitable(result):
            return await result  # type: ignore[return-value]
        return result

    try:
        while True:
            if stop_condition is not None:
                should_stop = await _call_maybe_async(stop_condition)
                if bool(should_stop):
                    break

            try:
                batch = await iterator.__anext__()
            except StopAsyncIteration:
                break

            observed_at = float(getattr(batch, "received_at", time_source()))
            if batch.events:
                await _call_maybe_async(handle_batch, batch)
                last_event_at = observed_at
                last_heartbeat_at = observed_at
            else:
                if batch.heartbeat:
                    await _call_maybe_async(on_heartbeat, observed_at)
                    last_heartbeat_at = observed_at
                elif on_heartbeat and (observed_at - last_heartbeat_at) >= heartbeat_interval:
                    await _call_maybe_async(on_heartbeat, observed_at)
                    last_heartbeat_at = observed_at

            if timeout_value is not None and (observed_at - last_event_at) >= timeout_value:
                raise TimeoutError(
                    f"Brak nowych danych w streamie '{batch.channel}' przez {timeout_value:.1f} s"
                )
    finally:
        for closer in closers:
            try:
                result = closer()
                if inspect.isawaitable(result):
                    await result  # type: ignore[func-returns-value]
            except Exception:  # pragma: no cover - diagnostyka zamykania strumienia
                _LOGGER.warning("Nie udało się zamknąć streamu long-pollowego", exc_info=True)


# --------------------------------------------------------------------------------------
# Funkcje pomocnicze
# --------------------------------------------------------------------------------------


def _resolve_strategy(
    core_config: CoreConfig, strategy_name: str
) -> DailyTrendMomentumStrategyConfig:
    try:
        return core_config.strategies[strategy_name]
    except KeyError as exc:  # pragma: no cover - kontrola konfiguracji
        raise KeyError(f"Brak strategii '{strategy_name}' w konfiguracji core") from exc


def _resolve_runtime(core_config: CoreConfig, controller_name: str) -> ControllerRuntimeConfig:
    try:
        return core_config.runtime_controllers[controller_name]
    except KeyError as exc:  # pragma: no cover - kontrola konfiguracji
        raise KeyError(f"Brak konfiguracji runtime dla kontrolera '{controller_name}'") from exc


def _resolve_universe(
    core_config: CoreConfig, environment: EnvironmentConfig
) -> InstrumentUniverseConfig:
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
        raise ValueError(
            "Pipeline paper trading jest dostępny wyłącznie dla środowisk paper/testnet."
        )

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

    ledger_filename_pattern = str(
        raw_settings.get("ledger_filename_pattern", "ledger-%Y%m%d.jsonl")
    )
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


def _derive_live_settings(
    environment: EnvironmentConfig,
    universe: InstrumentUniverseConfig,
) -> MutableMapping[str, object]:
    """Buduje minimalny zestaw parametrów dla trybu live."""

    adapter_settings = getattr(environment, "adapter_settings", {}) or {}
    live_settings = adapter_settings.get("live_trading", {}) or {}

    allowed_quotes: set[str] = {
        instrument.quote_asset.upper()
        for instrument in getattr(universe, "instruments", ())
        if getattr(instrument, "quote_asset", None)
    }
    if not allowed_quotes:
        fallback = str(live_settings.get("valuation_asset", "USDT"))
        allowed_quotes = {fallback.upper() or "USDT"}

    valuation_asset = str(live_settings.get("valuation_asset", next(iter(allowed_quotes)))).upper()
    position_size = max(0.0, float(live_settings.get("position_size", 0.0)))
    default_leverage = max(1.0, float(live_settings.get("default_leverage", 1.0)))
    initial_balances = {
        str(asset).upper(): float(amount)
        for asset, amount in (live_settings.get("initial_balances", {}) or {}).items()
    }

    return {
        "valuation_asset": valuation_asset,
        "position_size": position_size,
        "allowed_quotes": allowed_quotes,
        "default_leverage": default_leverage,
        "initial_balances": initial_balances,
        "default_market": {
            "min_quantity": 0.0,
            "min_notional": 0.0,
            "step_size": None,
            "tick_size": None,
        },
        "market_overrides": {},
        "portfolio_id": str(live_settings.get("portfolio_id", environment.name)),
        "maker_fee": float(live_settings.get("maker_fee", 0.0)),
        "taker_fee": float(live_settings.get("taker_fee", 0.0)),
        "slippage_bps": float(live_settings.get("slippage_bps", 0.0)),
        "ledger_directory": None,
        "ledger_filename_pattern": "ledger-%Y%m%d.jsonl",
        "ledger_retention_days": None,
        "ledger_fsync": False,
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
            min_quantity=float(
                per_symbol.get("min_quantity", default_market.get("min_quantity", 0.0))
            ),
            min_notional=float(
                per_symbol.get("min_notional", default_market.get("min_notional", 0.0))
            ),
            step_size=_optional_float(per_symbol.get("step_size", default_market.get("step_size"))),
            tick_size=_optional_float(per_symbol.get("tick_size", default_market.get("tick_size"))),
        )
        markets[symbol] = market

    return markets


def _build_execution_service(
    markets: Mapping[str, MarketMetadata],
    paper_settings: Mapping[str, object],
    *,
    price_resolver: PriceResolver | None = None,
) -> PaperTradingExecutionService:
    return _EXECUTION_BOOTSTRAPPER.build_paper_execution_service(
        markets,
        paper_settings,
        price_resolver=price_resolver,
    )


def _select_execution_service(
    *,
    bootstrap_ctx: BootstrapContext,
    markets: Mapping[str, MarketMetadata],
    paper_settings: Mapping[str, object],
    runtime_settings: RuntimeExecutionSettings | None,
    execution_mode: str,
    price_resolver: PriceResolver | None = None,
) -> ExecutionService:
    """Zwraca usługę egzekucyjną preferując instancję z bootstrapu."""

    return _EXECUTION_BOOTSTRAPPER.bootstrap_execution_service(
        bootstrap_ctx=bootstrap_ctx,
        markets=markets,
        paper_settings=paper_settings,
        runtime_settings=runtime_settings,
        execution_mode=execution_mode,
        price_resolver=price_resolver,
    )


def _build_price_resolver(data_source: CachedOHLCVSource, interval: str) -> PriceResolver:
    storage = data_source.storage

    def resolver(symbol: str) -> float | None:
        cache_key = data_source._cache_key(symbol, interval)  # pylint: disable=protected-access
        try:
            payload = storage.read(cache_key)
        except (AttributeError, KeyError):
            return None
        if not isinstance(payload, Mapping):
            return None
        rows = payload.get("rows", [])  # type: ignore[assignment]
        if not rows:
            return None
        last_row = rows[-1]
        if not last_row:
            return None
        try:
            price = float(last_row[4])
        except (IndexError, TypeError, ValueError):
            return None
        return price if price > 0 else None

    return resolver


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
        if (
            cached_entry
            and latest_cached is not None
            and abs(cached_entry[0] - latest_cached) < 1e-6
        ):
            return cached_entry[1]

        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        request = OHLCVRequest(symbol=symbol, interval=interval, start=0, end=now_ms)
        response = data_source.fetch_ohlcv(request)
        if not response.rows:
            market = markets.get(symbol)
            mid_price = getattr(market, "mid_price", None) if market is not None else None
            if mid_price is None:
                return 1.0
            return float(mid_price)
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
                    adjacency[market.base_asset.upper()].append(
                        (market.quote_asset.upper(), current_price)
                    )
                    adjacency[market.quote_asset.upper()].append(
                        (market.base_asset.upper(), 1.0 / current_price)
                    )
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


def _build_live_account_loader(adapter: ExchangeAdapter) -> Callable[[], AccountSnapshot]:
    def loader() -> AccountSnapshot:
        snapshot = adapter.fetch_account_snapshot()
        if not isinstance(snapshot, AccountSnapshot):
            raise TypeError("Adapter giełdowy zwrócił nieprawidłowy snapshot konta")
        return snapshot

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
    capital_policy: CapitalAllocationPolicy
    portfolio_coordinator: PortfolioRuntimeCoordinator | None = None
    portfolio_governor: PortfolioGovernor | None = None
    tco_reporter: RuntimeTCOReporter | None = None
    stream_feed: "StreamingStrategyFeed | None" = None
    stream_feed_task: "asyncio.Task[None] | None" = None
    decision_sink: "DecisionAwareSignalSink | None" = None
    optimization_scheduler: OptimizationScheduler | None = None
    optimization_queue: OptimizationTaskQueue | None = None

    def shutdown(self) -> None:
        """Zatrzymuje komponenty dodatkowe (np. stream feed)."""

        task = self.stream_feed_task
        if task is not None and not task.done():
            task.cancel()
        if self.stream_feed is not None:
            self.stream_feed.stop()
        self.stream_feed_task = None
        if self.optimization_scheduler is not None:
            self.optimization_scheduler.stop()
        if self.optimization_queue is not None:
            self.optimization_queue.shutdown()

    def start_stream(self) -> None:
        """Uruchamia strumień strategii w trybie synchronicznym."""

        if self.stream_feed is None:
            return
        self.stream_feed.start()

    def start_stream_async(
        self,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> asyncio.Task[None] | None:
        """Uruchamia strumień strategii na aktywnej pętli asyncio."""

        if self.stream_feed is None:
            return None

        task = self.stream_feed.start_async(loop=loop)
        self.stream_feed_task = task
        return task

    async def shutdown_async(self) -> None:
        """Asynchronicznie zatrzymuje komponenty runtime."""

        if self.stream_feed is not None:
            await self.stream_feed.stop_async()
        self.stream_feed_task = None
        if self.optimization_scheduler is not None:
            self.optimization_scheduler.stop()
        if self.optimization_queue is not None:
            self.optimization_queue.shutdown()

    def diagnostics_snapshot(
        self,
        *,
        include_metrics: bool = True,
        only_active: bool = False,
        tag: str | None = None,
        strategy: str | None = None,
    ) -> Mapping[str, object]:
        """Zwraca rozszerzoną migawkę administracyjną runtime."""

        snapshot = self.scheduler.administrative_snapshot(
            include_metrics=include_metrics,
            only_active=only_active,
            tag=tag,
            strategy=strategy,
        )
        strategies_summary = {}
        for name, engine in self.strategies.items():
            engine_cls = type(engine)
            strategies_summary[name] = {
                "engine_class": f"{engine_cls.__module__}.{engine_cls.__name__}",
                "has_metadata": hasattr(engine, "metadata"),
            }
        snapshot["runtime"] = {
            "environment": snapshot.get("environment"),
            "schedule_names": [cfg.name for cfg in self.schedules],
            "strategy_count": len(self.strategies),
            "has_portfolio_coordinator": self.portfolio_coordinator is not None,
        }
        snapshot["strategies"] = strategies_summary
        return snapshot

    def trigger_optimization(self, task_name: str | None = None):
        if self.optimization_scheduler is None:
            return ()
        return self.optimization_scheduler.trigger(task_name)


class OHLCVStrategyFeed(StrategyDataFeed):
    """Strategiczny feed korzystający z lokalnego cache OHLCV."""

    def __init__(
        self,
        data_source: CachedOHLCVSource,
        *,
        symbols_map: Mapping[str, Sequence[str]],
        interval_map: Mapping[str, str],
        default_interval: str = "1h",
        max_workers: int | None = None,
    ) -> None:
        self._data_source = data_source
        self._symbols_map = {key: tuple(values) for key, values in symbols_map.items()}
        self._interval_map = dict(interval_map)
        self._default_interval = default_interval
        self._max_workers = max_workers if max_workers and max_workers > 0 else 4

    def load_history(self, strategy_name: str, bars: int) -> Sequence[MarketSnapshot]:
        interval = self._interval_map.get(strategy_name, self._default_interval)
        symbols = self._symbols_map.get(strategy_name, ())
        if not symbols or bars <= 0:
            return ()
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        snapshots: list[MarketSnapshot] = []
        tasks: Mapping[str, OHLCVRequest] = {
            symbol: OHLCVRequest(
                symbol=symbol,
                interval=interval,
                start=0,
                end=now_ms,
                limit=bars,
            )
            for symbol in symbols
        }
        if len(tasks) == 1 or self._max_workers == 1:
            for symbol, request in tasks.items():
                response = self._data_source.fetch_ohlcv(request)
                snapshots.extend(_response_to_snapshots(symbol, response))
        else:
            workers = min(self._max_workers, len(tasks))
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="ohlcv-feed"
            ) as executor:
                futures = {
                    executor.submit(self._data_source.fetch_ohlcv, request): symbol
                    for symbol, request in tasks.items()
                }
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        response = future.result()
                    except Exception as exc:  # pragma: no cover - logowanie diagnostyczne
                        _LOGGER.warning(
                            "Nie udało się pobrać historii OHLCV (%s/%s): %s",
                            strategy_name,
                            symbol,
                            exc,
                        )
                        continue
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
        tasks: Mapping[str, OHLCVRequest] = {
            symbol: OHLCVRequest(
                symbol=symbol,
                interval=interval,
                start=0,
                end=now_ms,
                limit=1,
            )
            for symbol in symbols
        }
        if len(tasks) == 1 or self._max_workers == 1:
            for symbol, request in tasks.items():
                response = self._data_source.fetch_ohlcv(request)
                converted = _response_to_snapshots(symbol, response)
                if converted:
                    snapshots.append(converted[-1])
        else:
            workers = min(self._max_workers, len(tasks))
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="ohlcv-feed"
            ) as executor:
                futures = {
                    executor.submit(self._data_source.fetch_ohlcv, request): symbol
                    for symbol, request in tasks.items()
                }
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        response = future.result()
                    except Exception as exc:  # pragma: no cover - diagnostyka środowiska
                        _LOGGER.warning(
                            "Nie udało się pobrać ostatniej świecy OHLCV (%s/%s): %s",
                            strategy_name,
                            symbol,
                            exc,
                        )
                        continue
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

    _active_lock = threading.Lock()
    _active_instances: "weakref.WeakSet[StreamingStrategyFeed]" = weakref.WeakSet()

    def __init__(
        self,
        *,
        history_feed: StrategyDataFeed,
        stream_factory: Callable[[], Iterable[StreamBatch] | AsyncIterable[StreamBatch]],
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
        self._async_task: asyncio.Task[None] | None = None
        self._disabled = False
        self._buffers: dict[str, deque[MarketSnapshot]] = {}
        self._known_symbols = {symbol for values in self._symbols_map.values() for symbol in values}
        for symbol in self._known_symbols:
            self._buffers[symbol] = deque(maxlen=self._buffer_size)
        self._last_event_at: float | None = None

    def start(self) -> None:
        if _is_test_mode_enabled():
            self._disabled = True
            self._stop_event.set()
            return
        if self._disabled:
            return
        if self._async_task and not self._async_task.done():
            raise RuntimeError(
                "StreamingStrategyFeed jest już uruchomiony w trybie asynchronicznym."
            )
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name=_PIPELINE_THREAD_NAME, daemon=True
        )
        self._thread.start()
        self._register_instance()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(0.0, float(timeout)))
            if self._thread.is_alive():
                raise TimeoutError(
                    "StreamingStrategyFeed.close_all_active timeout: "
                    f"worker thread still alive after {timeout:.1f}s"
                )
        self._thread = None
        task = self._async_task
        if task and not task.done():
            task.cancel()
        self._unregister_instance()

    def ingest_batch(self, batch: StreamBatch) -> None:
        """Przetwarza paczkę danych dostarczoną ze streamu."""

        if not batch.events:
            return
        snapshots: list[MarketSnapshot] = []
        for event in batch.events:
            try:
                snapshot = self._event_to_snapshot(event)
            except Exception:
                self._logger.debug(
                    "Nie udało się sparsować zdarzenia streamu: %s", event, exc_info=True
                )
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

    def start_async(
        self,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> asyncio.Task[None]:
        """Uruchamia przetwarzanie streamu w pętli asynchronicznej."""
        if self._disabled:
            if not _is_test_mode_enabled():
                if self._async_task is None:
                    if loop is None:
                        try:
                            loop = asyncio.get_running_loop()
                        except RuntimeError:  # pragma: no cover - wywołanie spoza pętli zdarzeń
                            loop = asyncio.get_event_loop()
                    self._async_task = loop.create_task(asyncio.sleep(0))
                return self._async_task
            task = self._async_task
            if task is not None and not task.done():
                task.cancel()
            self._async_task = None
            self._disabled = False

        if self._thread and self._thread.is_alive():
            raise RuntimeError(
                "StreamingStrategyFeed jest już uruchomiony w trybie synchronicznym."
            )
        if self._async_task and not self._async_task.done():
            return self._async_task

        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:  # pragma: no cover - wywołanie spoza pętli zdarzeń
                loop = asyncio.get_event_loop()

        self._stop_event.clear()
        self._async_task = loop.create_task(self._run_loop_async())
        self._register_instance()
        return self._async_task

    async def stop_async(self) -> None:
        """Zatrzymuje asynchroniczną pętlę konsumpcji streamu."""

        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self.stop()

        task = self._async_task
        if task is None:
            return
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:  # pragma: no cover - oczekiwany scenariusz anulowania
            pass
        self._unregister_instance()

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
            except StopIteration:
                break
            except TimeoutError:
                self._logger.warning("Brak nowych danych w streamie strategii przez dłuższy czas")
            except Exception:  # pragma: no cover - logowanie dla diagnostyki
                self._logger.exception("Błąd podczas przetwarzania streamu strategii")
            if self._stop_event.is_set():
                break
            time.sleep(self._restart_delay)

    async def _run_loop_async(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    stream = self._stream_factory()
                    is_async_stream = callable(getattr(stream, "__aiter__", None))
                    if not is_async_stream:
                        raise TypeError(
                            "StreamingStrategyFeed.start_async wymaga, aby stream_factory zwracał AsyncIterable[StreamBatch], got: "
                            f"{type(stream)!r}"
                        )
                    await consume_stream_async(
                        stream,
                        handle_batch=self.ingest_batch,
                        heartbeat_interval=self._heartbeat_interval,
                        idle_timeout=self._idle_timeout,
                        on_heartbeat=self._handle_heartbeat,
                        stop_condition=self._stop_event.is_set,
                    )
                except (StopIteration, StopAsyncIteration):
                    break
                except asyncio.CancelledError:
                    raise
                except TypeError:
                    raise
                except TimeoutError:
                    self._logger.warning(
                        "Brak nowych danych w streamie strategii przez dłuższy czas"
                    )
                except Exception:  # pragma: no cover - logowanie dla diagnostyki
                    self._logger.exception("Błąd podczas przetwarzania streamu strategii")

                if self._stop_event.is_set():
                    break
                if self._restart_delay > 0:
                    await asyncio.sleep(self._restart_delay)
        except asyncio.CancelledError:
            raise
        finally:
            self._async_task = None
            self._unregister_instance()

    def _handle_heartbeat(self, timestamp: float) -> None:
        if self._last_event_at is None:
            return
        drift = timestamp - self._last_event_at
        if drift > max(self._heartbeat_interval, 1.0):
            self._logger.debug("Opóźnienie streamu strategii %.2f s", drift)

    def _register_instance(self) -> None:
        with self._active_lock:
            self._active_instances.add(self)

    def _unregister_instance(self) -> None:
        with self._active_lock:
            try:
                self._active_instances.remove(self)
            except KeyError:
                pass

    @classmethod
    def close_all_active(cls, *, timeout: float = 5.0) -> None:
        with cls._active_lock:
            active = list(cls._active_instances)
        for pipeline in active:
            try:
                pipeline.stop(timeout=timeout)
            except TimeoutError:
                raise
            except Exception:  # pragma: no cover - defensywnie w teardownie
                _LOGGER.debug("Nie udało się zamknąć StreamingStrategyFeed", exc_info=True)

    @staticmethod
    def _event_to_snapshot(event: Mapping[str, Any]) -> MarketSnapshot | None:
        symbol_raw = event.get("symbol") or event.get("pair") or event.get("instrument")
        if not symbol_raw:
            return None
        symbol = str(symbol_raw)
        timestamp_raw = (
            event.get("timestamp") or event.get("time") or event.get("ts") or time.time()
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

        last_price = StreamingStrategyFeed._float(
            event.get("last_price") or event.get("price") or event.get("close")
        )
        if last_price is None:
            return None
        open_price = (
            StreamingStrategyFeed._float(event.get("open_price") or event.get("open")) or last_price
        )
        high_price = StreamingStrategyFeed._float(
            event.get("high_24h") or event.get("high")
        ) or max(open_price, last_price)
        low_price = StreamingStrategyFeed._float(event.get("low_24h") or event.get("low")) or min(
            open_price, last_price
        )
        volume = (
            StreamingStrategyFeed._float(
                event.get("volume_24h_base") or event.get("volume") or event.get("base_volume")
            )
            or 0.0
        )

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


def _resolve_adapter_metrics_registry(
    adapter: ExchangeAdapter | object | None,
) -> MetricsRegistry | None:
    """Wyszukuje rejestr metryk powiązany z adapterem giełdowym."""

    return _DATA_SOURCE_BOOTSTRAPPER.resolve_adapter_metrics_registry(adapter)


def _build_streaming_feed(
    *,
    stream_config: object | None = None,
    stream_settings: Mapping[str, object] | None = None,
    adapter_metrics: MetricsRegistry | None = None,
    base_feed: StrategyDataFeed | None = None,
    symbols_map: Mapping[str, Sequence[str]] | None = None,
    exchange: str | None = None,
    environment_name: str | None = None,
    environment: EnvironmentConfig | None = None,
    bootstrap: bool = False,
    **_: Any,
) -> StreamingStrategyFeed | None:
    if environment is not None:
        if stream_config is None:
            stream_config = getattr(environment, "stream", None)
        if stream_settings is None:
            adapter_settings = getattr(environment, "adapter_settings", None)
            if isinstance(adapter_settings, Mapping):
                stream_settings = adapter_settings.get("stream")
        if exchange is None:
            exchange = getattr(environment, "exchange", None)
        if environment_name is None:
            environment_name = getattr(getattr(environment, "environment", None), "value", None)

    if adapter_metrics is None and bootstrap is not None:
        adapter_metrics = _resolve_adapter_metrics_registry(getattr(bootstrap, "adapter", None))

    if (
        stream_config is None
        or not isinstance(stream_settings, Mapping)
        or base_feed is None
        or symbols_map is None
        or exchange is None
    ):
        return None
    stream_settings = dict(stream_settings)
    host = getattr(stream_config, "host", "127.0.0.1")
    port = getattr(stream_config, "port", 8765)
    base_url = str(stream_settings.get("base_url") or f"http://{host}:{port}")
    default_path = f"/stream/{exchange}/public"
    path = str(stream_settings.get("public_path") or stream_settings.get("path") or default_path)
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
    params_in_body = bool(
        stream_settings.get("public_params_in_body", stream_settings.get("params_in_body", False))
    )
    channels_in_body = bool(
        stream_settings.get(
            "public_channels_in_body", stream_settings.get("channels_in_body", False)
        )
    )
    cursor_in_body = bool(
        stream_settings.get("public_cursor_in_body", stream_settings.get("cursor_in_body", False))
    )

    buffer_size_raw = stream_settings.get("buffer_size", 256)
    try:
        feed_buffer_size = int(buffer_size_raw)
    except (TypeError, ValueError):
        feed_buffer_size = 256
    if feed_buffer_size < 1:
        feed_buffer_size = 1

    stream_buffer_raw = stream_settings.get("public_buffer_size")
    if stream_buffer_raw is None:
        stream_buffer_size = feed_buffer_size
    else:
        try:
            stream_buffer_size = int(stream_buffer_raw)
        except (TypeError, ValueError):
            stream_buffer_size = feed_buffer_size
        if stream_buffer_size < 1:
            stream_buffer_size = 1

    def _factory() -> LocalLongPollStream:
        return LocalLongPollStream(
            base_url=base_url,
            path=path,
            channels=channels,
            adapter=exchange,
            scope="public",
            environment=environment_name or "",
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
            buffer_size=stream_buffer_size,
            metrics_registry=adapter_metrics,
        ).start()

    heartbeat_interval = float(stream_settings.get("heartbeat_interval", 15.0))
    idle_timeout_raw = stream_settings.get("idle_timeout")
    idle_timeout = None if idle_timeout_raw in (None, "") else float(idle_timeout_raw)
    restart_delay = float(stream_settings.get("restart_delay", 5.0))

    feed = StreamingStrategyFeed(
        history_feed=base_feed,
        stream_factory=_factory,
        symbols_map=symbols_map,
        buffer_size=feed_buffer_size,
        heartbeat_interval=heartbeat_interval,
        idle_timeout=idle_timeout,
        restart_delay=restart_delay,
        logger=_LOGGER,
    )

    if bootstrap:
        starter = getattr(feed, "start", None)
        if callable(starter):
            starter()

    return feed


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


def _collect_fixed_weight_entries(source: Any, *, prefix: str | None = None) -> Mapping[str, float]:
    result: dict[str, float] = {}
    if isinstance(source, Mapping):
        for key, value in source.items():
            lowered = str(key).lower()
            if lowered in {
                "strategies",
                "strategy",
                "schedules",
                "schedule",
                "profiles",
                "profile",
            }:
                result.update(_collect_fixed_weight_entries(value, prefix=prefix))
                continue
            if isinstance(value, Mapping):
                next_prefix = f"{prefix}:{key}" if prefix else str(key)
                result.update(_collect_fixed_weight_entries(value, prefix=next_prefix))
                continue
            composite = f"{prefix}:{key}" if prefix else str(key)
            try:
                result[composite] = float(value)
            except (TypeError, ValueError):
                continue
    elif isinstance(source, Sequence):
        for entry in source:
            if not isinstance(entry, Mapping):
                continue
            name_value = entry.get("schedule") or entry.get("strategy") or entry.get("name")
            weight_value = entry.get("weight")
            profile_value = entry.get("profile") or entry.get("risk_profile")
            if name_value is None or weight_value is None:
                continue
            composite = str(name_value)
            if profile_value not in (None, ""):
                composite = f"{composite}:{profile_value}"
            try:
                result[composite] = float(weight_value)
            except (TypeError, ValueError):
                continue
    return result


def _collect_profile_weight_entries(source: Any) -> Mapping[str, float]:
    result: dict[str, float] = {}
    if isinstance(source, Mapping):
        for profile, weight in source.items():
            if profile in (None, ""):
                continue
            try:
                result[str(profile).lower()] = float(weight)
            except (TypeError, ValueError):
                continue
        return result
    if isinstance(source, Sequence):
        for entry in source:
            if not isinstance(entry, Mapping):
                continue
            profile = entry.get("profile") or entry.get("risk_profile") or entry.get("name")
            weight = entry.get("weight")
            if profile in (None, "") or weight in (None, ""):
                continue
            try:
                result[str(profile).lower()] = float(weight)
            except (TypeError, ValueError):
                continue
    return result


def _collect_tag_weight_entries(source: Any) -> Mapping[str, float]:
    result: dict[str, float] = {}
    if isinstance(source, Mapping):
        for tag, weight in source.items():
            if tag in (None, "") or weight in (None, ""):
                continue
            try:
                result[str(tag)] = float(weight)
            except (TypeError, ValueError):
                continue
        return result
    if isinstance(source, Sequence):
        for entry in source:
            if isinstance(entry, Mapping):
                tag = entry.get("tag") or entry.get("name") or entry.get("label")
                weight = entry.get("weight")
                if tag in (None, "") or weight in (None, ""):
                    continue
                try:
                    result[str(tag)] = float(weight)
                except (TypeError, ValueError):
                    continue
    return result


def _apply_initial_suspensions(
    scheduler: MultiStrategyScheduler,
    suspensions: Sequence[MultiStrategySuspensionConfig] | None,
) -> None:
    if not suspensions:
        return
    for entry in suspensions:
        kind = (getattr(entry, "kind", "schedule") or "schedule").lower()
        target = getattr(entry, "target", None)
        if not target:
            continue
        reason = getattr(entry, "reason", None)
        until = getattr(entry, "until", None)
        duration = getattr(entry, "duration_seconds", None)
        if kind == "tag":
            scheduler.suspend_tag(
                target,
                reason=reason,
                until=until,
                duration_seconds=duration,
            )
        else:
            scheduler.suspend_schedule(
                target,
                reason=reason,
                until=until,
                duration_seconds=duration,
            )


def _build_quality_lookup(
    model_repository: ModelRepository | None,
) -> Callable[[str], Mapping[str, Any] | None]:
    if model_repository is None:
        return lambda _: None

    def loader(version: str) -> Mapping[str, Any] | None:
        try:
            return model_repository.get_quality_report(version)
        except Exception:
            return None

    return loader


def _build_optimization_evaluator(
    runtime: "MultiStrategyRuntime",
    task_cfg: StrategyOptimizationTaskConfig,
    optimization_cfg: RuntimeOptimizationSettings,
    quality_lookup: Callable[[str], Mapping[str, Any] | None],
) -> Callable[[StrategyEngine, Mapping[str, Any]], tuple[float, Mapping[str, Any]]]:
    history_bars = (
        getattr(task_cfg.evaluation, "history_bars", None) or optimization_cfg.default_history_bars
    )
    warmup_bars = getattr(task_cfg.evaluation, "warmup_bars", 0)
    warmup_bars = max(0, min(history_bars - 1, int(warmup_bars)))
    data_feed = getattr(runtime, "data_feed", None)
    baseline_quality = quality_lookup(task_cfg.strategy)

    def evaluator(engine: StrategyEngine, _: Mapping[str, Any]) -> tuple[float, Mapping[str, Any]]:
        if data_feed is None or not hasattr(data_feed, "load_history"):
            return 0.0, {"warning": "no-data-feed"}
        try:
            history = data_feed.load_history(task_cfg.strategy, history_bars)
        except Exception as exc:  # pragma: no cover - defensywne
            return 0.0, {"error": str(exc), "stage": "load_history"}
        if not history:
            return 0.0, {"warning": "empty-history", "bars": 0}
        if warmup_bars:
            try:
                engine.warm_up(tuple(history[:warmup_bars]))
            except Exception as exc:  # pragma: no cover - defensywne
                return 0.0, {"error": str(exc), "stage": "warm_up"}
        score_sum = 0.0
        signal_count = 0
        try:
            for snapshot in history[warmup_bars:]:
                signals = engine.on_data(snapshot)
                for signal in signals:
                    score_sum += abs(float(getattr(signal, "confidence", 0.0)))
                    signal_count += 1
        except Exception as exc:  # pragma: no cover - defensywne
            return 0.0, {"error": str(exc), "stage": "on_data"}
        average = score_sum / signal_count if signal_count else 0.0
        metadata = {
            "signals": signal_count,
            "bars": len(history),
            "avg_confidence": average,
        }
        if baseline_quality is not None:
            metadata["baseline_quality"] = baseline_quality
        return average, metadata

    return evaluator


def _configure_optimization_scheduler(
    runtime: "MultiStrategyRuntime",
    *,
    core_config: CoreConfig,
    optimization_cfg: RuntimeOptimizationSettings | None,
    model_repository: ModelRepository | None = None,
    catalog: StrategyCatalog | None = None,
) -> OptimizationScheduler | None:
    if optimization_cfg is None or not optimization_cfg.enabled:
        return None

    definitions = _collect_strategy_definitions(core_config)
    optimizer = StrategyOptimizer(catalog or DEFAULT_STRATEGY_CATALOG)
    optimization_queue = OptimizationTaskQueue(
        max_workers=getattr(optimization_cfg, "max_concurrent_jobs", 1)
    )
    quality_lookup = _build_quality_lookup(model_repository)
    scheduler = OptimizationScheduler(
        optimizer,
        report_directory=getattr(optimization_cfg, "report_directory", None),
        task_queue=optimization_queue,
    )

    for task_cfg in optimization_cfg.tasks:
        definition = definitions.get(task_cfg.strategy)
        if definition is None:
            _LOGGER.warning(
                "Pominięto zadanie optymalizacji '%s' – brak definicji strategii '%s'",
                task_cfg.name,
                task_cfg.strategy,
            )
            continue
        evaluator = _build_optimization_evaluator(
            runtime, task_cfg, optimization_cfg, quality_lookup
        )
        scheduler.add_task(
            config=task_cfg,
            definition=definition,
            evaluator=evaluator,
            default_algorithm=optimization_cfg.default_algorithm,
        )

    if not scheduler.has_tasks():
        optimization_queue.shutdown()
        return None

    scheduler.start()
    runtime.optimization_scheduler = scheduler
    runtime.optimization_queue = optimization_queue
    return scheduler


def _apply_initial_signal_limits(
    scheduler: MultiStrategyScheduler,
    limits: Mapping[str, Mapping[str, SignalLimitOverrideConfig]] | None,
) -> None:
    if not limits:
        return
    for strategy, profiles in limits.items():
        for profile, override in profiles.items():
            scheduler.configure_signal_limit(strategy, profile, override)


def _collect_metric_weight_specs(source: Any) -> tuple[MetricWeightRule, ...]:
    rules: list[MetricWeightRule] = []

    def _parse_single(metric_name: str, definition: Any) -> None:
        if metric_name in (None, ""):
            return
        name = str(metric_name)
        weight = 1.0
        default = 0.0
        clamp_min: float | None = None
        clamp_max: float | None = None
        absolute = False
        scale = 1.0

        if isinstance(definition, Mapping):
            weight_source = (
                definition.get("weight")
                or definition.get("score")
                or definition.get("coefficient")
                or definition.get("multiplier")
            )
            if weight_source not in (None, ""):
                weight = _safe_float(weight_source, default=1.0)
            default_source = (
                definition.get("default") or definition.get("missing") or definition.get("fallback")
            )
            if default_source not in (None, ""):
                default = _safe_float(default_source, default=0.0)
            min_source = (
                definition.get("min") or definition.get("clamp_min") or definition.get("floor")
            )
            if min_source not in (None, ""):
                clamp_min = _safe_float(min_source, default=0.0)
            max_source = (
                definition.get("max") or definition.get("clamp_max") or definition.get("cap")
            )
            if max_source not in (None, ""):
                clamp_max = _safe_float(max_source, default=0.0)
            absolute = _as_bool(
                definition.get("absolute")
                or definition.get("abs")
                or definition.get("use_abs")
                or False,
                default=False,
            )
            scale_source = definition.get("scale") or definition.get("factor")
            if scale_source not in (None, ""):
                scale = _safe_float(scale_source, default=1.0)
        elif isinstance(definition, Sequence):
            try:
                weight = _safe_float(definition[0], default=1.0)  # type: ignore[index]
            except (IndexError, TypeError):
                weight = 1.0
            try:
                default = _safe_float(definition[1], default=0.0)  # type: ignore[index]
            except (IndexError, TypeError, ValueError):
                default = 0.0
        else:
            try:
                weight = float(definition)
            except (TypeError, ValueError):
                weight = 1.0

        if not math.isfinite(weight) or weight == 0.0:
            return
        try:
            rule = MetricWeightRule(
                metric=name,
                weight=float(weight),
                default=float(default),
                clamp_min=float(clamp_min) if clamp_min is not None else None,
                clamp_max=float(clamp_max) if clamp_max is not None else None,
                absolute=bool(absolute),
                scale=float(scale),
            )
        except (TypeError, ValueError):
            return
        rules.append(rule)

    if isinstance(source, Mapping):
        for metric_name, definition in source.items():
            _parse_single(metric_name, definition)
    elif isinstance(source, Sequence):
        for entry in source:
            if isinstance(entry, Mapping):
                metric_name = entry.get("metric") or entry.get("name")
                if metric_name in (None, ""):
                    continue
                definition = dict(entry)
                definition.setdefault("weight", entry.get("weight") or entry.get("score"))
                _parse_single(metric_name, definition)
    return tuple(rules)


def _policy_factory_from_spec(
    spec: Mapping[str, Any] | str | None,
) -> Callable[[], CapitalAllocationPolicy]:
    def _default() -> CapitalAllocationPolicy:
        return RiskParityAllocation()

    if spec in (None, ""):
        return _default
    if isinstance(spec, str) or isinstance(spec, Mapping):
        cached_spec = spec

        def _factory() -> CapitalAllocationPolicy:
            policy, _ = _resolve_capital_policy(cached_spec, _allow_profile=False)
            return policy

        return _factory
    if callable(spec):  # pragma: no cover - ścieżka programistyczna
        return spec  # type: ignore[return-value]
    return _default


def _assign_policy_label(
    policy: CapitalAllocationPolicy, label: str | None
) -> CapitalAllocationPolicy:
    if label in (None, ""):
        return policy
    try:
        setattr(policy, "name", str(label))
    except Exception:  # pragma: no cover - defensywnie
        pass
    return policy


def _resolve_capital_policy(
    spec: Mapping[str, Any] | str | None,
    *,
    _allow_profile: bool = True,
) -> tuple[CapitalAllocationPolicy, float | None]:
    if spec is None:
        return RiskParityAllocation(), None
    if isinstance(spec, str):
        normalized = spec.strip().lower().replace("-", "_")
        if normalized in {"equal", "equal_weight", "uniform"}:
            return EqualWeightAllocation(), None
        if normalized in {"volatility_target", "vol_target", "target_volatility"}:
            return VolatilityTargetAllocation(), None
        if normalized in {"signal", "signal_strength", "signals"}:
            return SignalStrengthAllocation(), None
        if normalized in {"fixed", "fixed_weight", "manual"}:
            return FixedWeightAllocation({}, label="fixed_weight"), None
        if normalized in {"risk_profile", "profile_budget", "risk_budget"} and _allow_profile:
            return RiskProfileBudgetAllocation({}, label="risk_profile_budget"), None
        if normalized not in {"risk", "risk_parity"}:
            _LOGGER.warning(
                "Nieznana polityka alokacji kapitału '%s' – używam risk_parity",
                spec,
            )
        return RiskParityAllocation(), None

    if not isinstance(spec, Mapping):
        _LOGGER.warning(
            "Nieprawidłowa definicja polityki alokacji kapitału (%s) – używam risk_parity",
            type(spec).__name__,
        )
        return RiskParityAllocation(), None

    name_value = spec.get("name") or spec.get("policy") or spec.get("type")
    name = str(name_value or "risk_parity").strip().lower().replace("-", "_")
    label_value = spec.get("label") or spec.get("alias")
    label = str(label_value) if isinstance(label_value, str) and label_value else None
    rebalance_entry = (
        spec.get("rebalance_seconds")
        or spec.get("rebalance_interval")
        or spec.get("rebalance")
        or spec.get("interval_seconds")
    )
    rebalance_seconds: float | None = None
    if rebalance_entry not in (None, ""):
        try:
            rebalance_seconds = float(rebalance_entry)
        except (TypeError, ValueError):
            _LOGGER.debug(
                "Nie udało się sparsować rebalance_seconds=%s dla polityki kapitału",
                rebalance_entry,
                exc_info=True,
            )
            rebalance_seconds = None

    if name in {"equal", "equal_weight", "uniform"}:
        policy = _assign_policy_label(EqualWeightAllocation(), label)
        return policy, rebalance_seconds
    if name in {"volatility_target", "vol_target", "target_volatility"}:
        policy = _assign_policy_label(VolatilityTargetAllocation(), label)
        return policy, rebalance_seconds
    if name in {"signal", "signal_strength", "signals"}:
        policy = _assign_policy_label(SignalStrengthAllocation(), label)
        return policy, rebalance_seconds
    if name in {"metric", "metric_weighted", "telemetry_weighted", "metric_score"}:
        metrics_source = (
            spec.get("metrics") or spec.get("rules") or spec.get("weights") or spec.get("signals")
        )
        metric_rules = _collect_metric_weight_specs(metrics_source)
        if not metric_rules:
            _LOGGER.warning(
                "Polityka metric_weighted nie zawiera żadnych metryk – używam fallbacku",
            )
        default_entry = spec.get("default_score") or spec.get("bias") or spec.get("base_score")
        default_score = _safe_float(default_entry, default=0.0)
        shift_entry = spec.get("shift_epsilon") or spec.get("shift") or spec.get("epsilon")
        shift_epsilon = _safe_float(shift_entry, default=1e-6)
        fallback_spec = (
            spec.get("fallback") or spec.get("fallback_policy") or spec.get("default_policy")
        )
        fallback_policy: CapitalAllocationPolicy | None = None
        if isinstance(fallback_spec, str) and fallback_spec.strip().lower() in {
            "none",
            "disabled",
            "disable",
        }:
            fallback_policy = None
        elif fallback_spec not in (None, ""):
            fallback_policy, _ = _resolve_capital_policy(
                fallback_spec, _allow_profile=_allow_profile
            )
        policy = MetricWeightedAllocation(
            metric_rules,
            label=label or name,
            default_score=default_score,
            fallback_policy=fallback_policy,
            shift_epsilon=shift_epsilon,
        )
        return policy, rebalance_seconds
    if name in {"tag_quota", "tag_budget", "tag_weight", "tag_split"}:
        tags_source = (
            spec.get("tags") or spec.get("tag_weights") or spec.get("weights") or spec.get("groups")
        )
        tag_weights = _collect_tag_weight_entries(tags_source)
        if not tag_weights:
            _LOGGER.warning(
                "Polityka tag_quota nie zawiera żadnych tagów – używam fallbacku",
            )
        default_weight_entry = (
            spec.get("default_weight")
            or spec.get("unassigned_weight")
            or spec.get("fallback_weight")
        )
        default_weight = _safe_float(default_weight_entry, default=0.0)
        if default_weight <= 0.0:
            default_weight = None
        fallback_spec = spec.get("fallback") or spec.get("fallback_policy")
        fallback_policy: CapitalAllocationPolicy | None = None
        if fallback_spec not in (None, ""):
            fallback_policy, _ = _resolve_capital_policy(
                fallback_spec,
                _allow_profile=_allow_profile,
            )
        inner_spec = spec.get("within_tag") or spec.get("within") or spec.get("inner_policy")
        inner_factory: Callable[[], CapitalAllocationPolicy] | None = None
        if inner_spec not in (None, ""):
            inner_factory = _policy_factory_from_spec(inner_spec)
        prefer_primary_entry = spec.get("primary_only")
        if prefer_primary_entry in (None, ""):
            prefer_primary_entry = spec.get("prefer_primary")
        prefer_primary = _as_bool(prefer_primary_entry, default=True)
        policy = TagQuotaAllocation(
            tag_weights,
            label=label or name,
            fallback_policy=fallback_policy,
            inner_policy_factory=inner_factory,
            default_weight=default_weight,
            prefer_primary=prefer_primary,
        )
        return policy, rebalance_seconds
    if name in {"drawdown", "drawdown_guard", "drawdown_adaptive", "drawdown_aware"}:
        warning_entry = (
            spec.get("warning_pct")
            or spec.get("warning")
            or spec.get("soft_limit_pct")
            or spec.get("soft_limit")
        )
        panic_entry = (
            spec.get("panic_pct")
            or spec.get("panic")
            or spec.get("hard_limit_pct")
            or spec.get("limit_pct")
        )
        pressure_entry = spec.get("pressure_weight") or spec.get("pressure")
        min_weight_entry = spec.get("min_weight") or spec.get("floor_weight")
        max_weight_entry = spec.get("max_weight") or spec.get("cap_weight")

        warning_pct = _safe_float(warning_entry, default=10.0)
        panic_pct = _safe_float(panic_entry, default=20.0)
        pressure_weight = _safe_float(pressure_entry, default=0.7)
        min_weight = _safe_float(min_weight_entry, default=0.05)
        max_weight = _safe_float(max_weight_entry, default=1.0)

        policy = DrawdownAdaptiveAllocation(
            warning_drawdown_pct=warning_pct,
            panic_drawdown_pct=panic_pct,
            pressure_weight=pressure_weight,
            min_weight=min_weight,
            max_weight=max_weight,
        )
        effective_label = label or name
        return _assign_policy_label(policy, effective_label), rebalance_seconds
    if name in {"smoothed", "smoothed_allocation", "ewma", "ema"}:
        base_entry = (
            spec.get("base")
            or spec.get("inner")
            or spec.get("delegate")
            or spec.get("base_policy")
            or spec.get("inner_policy")
        )
        smoothing_entry = spec.get("alpha") or spec.get("smoothing_factor") or spec.get("smoothing")
        min_delta_entry = spec.get("min_delta") or spec.get("threshold") or spec.get("min_step")
        floor_entry = spec.get("floor_weight") or spec.get("min_weight") or spec.get("floor")

        base_spec = base_entry or "risk_parity"
        base_policy, _ = _resolve_capital_policy(base_spec, _allow_profile=_allow_profile)
        smoothing_factor = _safe_float(smoothing_entry, default=0.35)
        min_delta = _safe_float(min_delta_entry, default=0.0)
        floor_weight = _safe_float(floor_entry, default=0.0)

        policy = SmoothedCapitalAllocationPolicy(
            base_policy,
            smoothing_factor=smoothing_factor,
            min_delta=min_delta,
            floor_weight=floor_weight,
        )
        effective_label = label or name
        return _assign_policy_label(policy, effective_label), rebalance_seconds
    if name in {"blended", "composite", "weighted_mix", "mix"}:
        components_source = (
            spec.get("components")
            or spec.get("policies")
            or spec.get("allocators")
            or spec.get("mix")
        )
        components: list[tuple[CapitalAllocationPolicy, float, str | None]] = []
        if isinstance(components_source, Mapping):
            for comp_label, entry in components_source.items():
                component_spec = entry
                weight_value = 1.0
                if isinstance(entry, Mapping):
                    weight_source = (
                        entry.get("weight") or entry.get("share") or entry.get("coefficient")
                    )
                    weight_value = _safe_float(weight_source, default=1.0)
                    component_spec = (
                        entry.get("policy")
                        or entry.get("allocator")
                        or entry.get("spec")
                        or entry.get("definition")
                        or entry.get("name")
                        or entry
                    )
                if component_spec in (None, "") or weight_value <= 0:
                    continue
                component_policy, _ = _resolve_capital_policy(component_spec, _allow_profile=False)
                components.append((component_policy, weight_value, str(comp_label)))
        elif isinstance(components_source, Sequence):
            for entry in components_source:
                component_spec: Any = entry
                weight_value = 1.0
                label_value: str | None = None
                if isinstance(entry, Mapping):
                    weight_source = (
                        entry.get("weight") or entry.get("share") or entry.get("coefficient")
                    )
                    weight_value = _safe_float(weight_source, default=1.0)
                    label_entry = entry.get("label") or entry.get("alias")
                    if isinstance(label_entry, str) and label_entry:
                        label_value = label_entry
                    component_spec = (
                        entry.get("policy")
                        or entry.get("allocator")
                        or entry.get("spec")
                        or entry.get("definition")
                        or entry.get("name")
                        or entry
                    )
                if component_spec in (None, "") or weight_value <= 0:
                    continue
                component_policy, _ = _resolve_capital_policy(component_spec, _allow_profile=False)
                components.append((component_policy, weight_value, label_value))
        else:
            _LOGGER.debug(
                "Polityka blended wymaga listy komponentów – otrzymano %s",
                type(components_source).__name__,
            )
        normalize_flag = (
            spec.get("normalize_components")
            or spec.get("normalize")
            or spec.get("normalize_weights")
        )
        normalize_components = _as_bool(normalize_flag, default=True)
        fallback_spec = (
            spec.get("fallback") or spec.get("fallback_policy") or spec.get("default_policy")
        )
        fallback_policy: CapitalAllocationPolicy | None
        if isinstance(fallback_spec, str) and fallback_spec.strip().lower() in {
            "none",
            "disabled",
            "disable",
        }:
            fallback_policy = None
        elif fallback_spec in (None, ""):
            fallback_policy = RiskParityAllocation()
        else:
            fallback_policy, _ = _resolve_capital_policy(
                fallback_spec, _allow_profile=_allow_profile
            )
        if not components:
            fallback_label = getattr(fallback_policy, "name", "uniform")
            _LOGGER.warning(
                "Polityka blended nie zawiera komponentów – używam fallbacku %s",
                fallback_label,
            )
        effective_label = label or name
        policy = BlendedCapitalAllocation(
            tuple(components),
            label=effective_label,
            normalize_components=normalize_components,
            fallback_policy=fallback_policy,
        )
        return policy, rebalance_seconds
    if name in {"fixed", "fixed_weight", "manual"}:
        weights_source = spec.get("weights") or spec.get("allocations")
        weights = _collect_fixed_weight_entries(weights_source)
        if not weights:
            _LOGGER.warning(
                "Polityka fixed_weight nie zawiera żadnych wag – używam risk_parity",
            )
            fallback = _assign_policy_label(RiskParityAllocation(), label)
            return fallback, rebalance_seconds
        return FixedWeightAllocation(weights, label=label), rebalance_seconds
    if name in {"risk_profile", "profile_budget", "risk_budget"}:
        if not _allow_profile:
            _LOGGER.warning(
                "Zagnieżdżona polityka risk_profile jest niedozwolona – używam risk_parity",
            )
            fallback = _assign_policy_label(RiskParityAllocation(), label)
            return fallback, rebalance_seconds
        profile_source = (
            spec.get("profiles")
            or spec.get("profile_weights")
            or spec.get("weights")
            or spec.get("allocations")
        )
        profile_weights = _collect_profile_weight_entries(profile_source)
        floor_value = (
            spec.get("profile_floor")
            or spec.get("floor")
            or spec.get("min_weight")
            or spec.get("minimum")
        )
        floor: float = 0.0
        if floor_value not in (None, ""):
            try:
                floor = max(0.0, float(floor_value))
            except (TypeError, ValueError):
                _LOGGER.debug(
                    "Nie udało się sparsować floor=%s dla risk_profile",
                    floor_value,
                    exc_info=True,
                )
                floor = 0.0
        inner_spec = (
            spec.get("within_profile")
            or spec.get("inner_policy")
            or spec.get("profile_policy")
            or spec.get("strategy_policy")
        )
        inner_factory = _policy_factory_from_spec(inner_spec)
        policy = RiskProfileBudgetAllocation(
            profile_weights,
            label=label,
            profile_floor=floor,
            inner_policy_factory=inner_factory,
        )
        return policy, rebalance_seconds
    if name not in {"risk", "risk_parity"}:
        _LOGGER.warning(
            "Nieznana polityka alokacji kapitału '%s' – używam risk_parity",
            name,
        )
    policy = _assign_policy_label(RiskParityAllocation(), label)
    return policy, rebalance_seconds


def resolve_capital_policy_spec(
    spec: Mapping[str, Any] | str | None,
    *,
    allow_profile: bool = True,
) -> tuple[CapitalAllocationPolicy, float | None]:
    """Publiczny helper zwracający politykę kapitału oraz zalecany interwał."""

    return _resolve_capital_policy(spec, _allow_profile=allow_profile)


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
    history_limit = 256
    if decision_config is not None:
        try:
            history_limit = int(getattr(decision_config, "evaluation_history_limit", history_limit))
        except (TypeError, ValueError):  # pragma: no cover - konfiguracja może być uszkodzona
            history_limit = 256

    opportunity_shadow_adapter = OpportunityRuntimeShadowAdapter(
        journal=journal,
        engine=TradingOpportunityAI(repository=FilesystemModelRepository(Path("data/ai"))),
    )

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
        evaluation_history_limit=history_limit,
        opportunity_shadow_adapter=opportunity_shadow_adapter,
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
        evaluation_history_limit: int = 256,
        opportunity_shadow_adapter: OpportunityRuntimeShadowAdapter | None = None,
    ) -> None:
        self._base_sink = base_sink
        self._orchestrator = orchestrator
        self._risk_engine = risk_engine
        self._default_notional = max(0.0, float(default_notional)) or 1_000.0
        self._environment = environment
        self._exchange = exchange
        self._min_probability = max(0.0, min(1.0, float(min_probability)))
        self._logger = logging.getLogger(__name__)
        self._evaluations: deque[DecisionEvaluation] = deque(
            maxlen=max(1, int(evaluation_history_limit))
        )
        self._portfolio = str(portfolio) if portfolio is not None else ""
        self._journal = journal
        self._opportunity_shadow_adapter = opportunity_shadow_adapter

    @staticmethod
    def _coerce_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_thresholds(
        snapshot: Mapping[str, object] | None,
    ) -> Mapping[str, float | None] | None:
        if not snapshot or not isinstance(snapshot, Mapping):
            return None
        normalized: dict[str, float | None] = {}
        for key, value in snapshot.items():
            key_str = str(key)
            coerced = DecisionAwareSignalSink._coerce_float(value)
            normalized[key_str] = coerced
        return normalized

    @staticmethod
    def _candidate_to_mapping(candidate: object) -> Mapping[str, object]:
        if candidate is None:
            return {}
        if hasattr(candidate, "to_mapping"):
            try:
                payload = candidate.to_mapping()  # type: ignore[assignment]
                if isinstance(payload, Mapping):
                    return dict(payload)
            except Exception:  # pragma: no cover - defensywne
                pass
        payload_dict: dict[str, object] = {}
        for attribute in (
            "strategy",
            "action",
            "risk_profile",
            "symbol",
            "notional",
            "expected_return_bps",
            "expected_probability",
            "cost_bps_override",
            "latency_ms",
        ):
            if hasattr(candidate, attribute):
                value = getattr(candidate, attribute)
                if value is not None:
                    payload_dict[attribute] = value
        metadata = getattr(candidate, "metadata", None)
        if isinstance(metadata, Mapping):
            payload_dict["metadata"] = dict(metadata)
        return payload_dict

    def _serialize_evaluation_payload(
        self,
        evaluation: object,
        *,
        include_candidate: bool,
    ) -> Mapping[str, object]:
        payload: dict[str, object]
        if hasattr(evaluation, "to_mapping"):
            try:
                mapping = evaluation.to_mapping()  # type: ignore[assignment]
            except Exception:  # pragma: no cover - defensywne
                mapping = None
            if isinstance(mapping, Mapping):
                payload = dict(mapping)
            else:
                payload = {}
        else:
            payload = {
                "accepted": bool(getattr(evaluation, "accepted", False)),
                "reasons": list(getattr(evaluation, "reasons", ())),
                "risk_flags": list(getattr(evaluation, "risk_flags", ())),
                "stress_failures": list(getattr(evaluation, "stress_failures", ())),
                "cost_bps": getattr(evaluation, "cost_bps", None),
                "net_edge_bps": getattr(evaluation, "net_edge_bps", None),
                "model_expected_return_bps": getattr(evaluation, "model_expected_return_bps", None),
                "model_success_probability": getattr(evaluation, "model_success_probability", None),
                "model_name": getattr(evaluation, "model_name", None),
            }

        if not include_candidate:
            payload.pop("candidate", None)
        else:
            candidate_payload: Mapping[str, object] | None = None
            candidate_obj = getattr(evaluation, "candidate", None)
            if isinstance(payload.get("candidate"), Mapping):
                candidate_payload = dict(payload["candidate"])
            elif candidate_obj is not None:
                candidate_payload = self._candidate_to_mapping(candidate_obj)
            if candidate_payload:
                payload["candidate"] = candidate_payload

        thresholds = getattr(evaluation, "thresholds_snapshot", None)
        normalized_thresholds = self._normalize_thresholds(thresholds)
        if normalized_thresholds:
            payload.setdefault("thresholds", dict(normalized_thresholds))

        latency = getattr(evaluation, "latency_ms", None)
        if latency is not None:
            try:
                payload["latency_ms"] = float(latency)
            except (TypeError, ValueError):  # pragma: no cover - defensywne
                pass

        evaluated_at = getattr(evaluation, "evaluated_at", None)
        if evaluated_at is not None:
            payload["evaluated_at"] = evaluated_at

        selection = getattr(evaluation, "model_selection", None)
        if selection is not None and "model_selection" not in payload:
            if hasattr(selection, "to_mapping"):
                try:
                    selection_payload = selection.to_mapping()  # type: ignore[assignment]
                except Exception:  # pragma: no cover - defensywne
                    selection_payload = None
            elif isinstance(selection, Mapping):
                selection_payload = dict(selection)
            else:
                selection_payload = None
            if selection_payload:
                payload["model_selection"] = selection_payload

        return payload

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
            candidate, rejection_info = self._build_candidate(
                strategy_name,
                risk_profile,
                signal,
            )
            if candidate is None:
                self._record_filtered_signal(
                    signal=signal,
                    strategy_name=strategy_name,
                    schedule_name=schedule_name,
                    risk_profile=risk_profile,
                    timestamp=timestamp,
                    rejection_info=rejection_info,
                )
                continue
            try:
                evaluation = self._orchestrator.evaluate_candidate(
                    candidate,
                    DecisionContext(
                        risk_snapshot=risk_snapshot,
                        runtime={
                            "timestamp": timestamp,
                            "environment": self._environment,
                            "schedule_name": schedule_name,
                            "strategy_name": strategy_name,
                        },
                    ),
                )
            except Exception as exc:  # pragma: no cover - diagnostyka orchestratora
                self._logger.exception("DecisionOrchestrator odrzucił kandydata przez wyjątek")
                self._record_filtered_signal(
                    signal=signal,
                    strategy_name=strategy_name,
                    schedule_name=schedule_name,
                    risk_profile=risk_profile,
                    timestamp=timestamp,
                    rejection_info={
                        "reason": "orchestrator_exception",
                        "error": str(exc),
                    },
                )
                continue
            if evaluation is None or not hasattr(evaluation, "accepted"):
                self._logger.warning(
                    "DecisionOrchestrator zwrócił niepoprawną ewaluację (%s)",
                    type(evaluation).__name__ if evaluation is not None else "None",
                )
                self._record_filtered_signal(
                    signal=signal,
                    strategy_name=strategy_name,
                    schedule_name=schedule_name,
                    risk_profile=risk_profile,
                    timestamp=timestamp,
                    rejection_info={
                        "reason": "invalid_evaluation",
                        "evaluation_type": type(evaluation).__name__
                        if evaluation is not None
                        else "None",
                    },
                )
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

    def evaluation_history(
        self,
        *,
        limit: int | None = None,
        include_candidates: bool = False,
    ) -> Sequence[Mapping[str, object]]:
        if not self._evaluations:
            return ()
        records = list(self._evaluations)
        if limit is not None:
            try:
                limit_int = int(limit)
            except (TypeError, ValueError):  # pragma: no cover - defensywne
                limit_int = 0
            if limit_int <= 0:
                return ()
            records = records[-limit_int:]
        history: list[Mapping[str, object]] = []
        for evaluation in records:
            history.append(
                self._serialize_evaluation_payload(evaluation, include_candidate=include_candidates)
            )
        return tuple(history)

    def evaluation_summary_v2(self) -> Mapping[str, object]:
        if summarize_evaluation_payloads is None:
            raise RuntimeError(
                "Decision summary aggregation module is unavailable – Stage6 "
                "runtime wymaga bot_core.decision.summarize_evaluation_payloads"
            )

        evaluations = tuple(self._evaluations)
        history_limit = self._evaluations.maxlen or len(evaluations)
        payloads = [
            self._serialize_evaluation_payload(evaluation, include_candidate=True)
            for evaluation in evaluations
        ]
        summary = summarize_evaluation_payloads(
            payloads,
            history_limit=history_limit,
        )
        return summary.model_dump(exclude_none=True)

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

        _store_float(
            "model_success_probability",
            getattr(evaluation, "model_success_probability", None),
        )
        _store_float(
            "model_expected_return_bps",
            getattr(evaluation, "model_expected_return_bps", None),
        )

        model_name = getattr(evaluation, "model_name", None)
        if model_name:
            metadata.setdefault("model_name", str(model_name))

        thresholds_snapshot = getattr(evaluation, "thresholds_snapshot", None)
        normalized_thresholds = self._normalize_thresholds(thresholds_snapshot)
        if normalized_thresholds:
            try:
                metadata.setdefault(
                    "decision_thresholds",
                    json.dumps(normalized_thresholds, sort_keys=True, ensure_ascii=False),
                )
            except (TypeError, ValueError):  # pragma: no cover - defensywne
                pass

        model_selection = getattr(evaluation, "model_selection", None)
        selection_mapping: Mapping[str, object] | None = None
        if model_selection is not None:
            if hasattr(model_selection, "to_mapping"):
                try:
                    selection_mapping = model_selection.to_mapping()  # type: ignore[assignment]
                except Exception:  # pragma: no cover - defensywne
                    selection_mapping = None
            elif isinstance(model_selection, Mapping):
                selection_mapping = model_selection
        if selection_mapping:
            try:
                metadata.setdefault(
                    "model_selection",
                    json.dumps(selection_mapping, sort_keys=True, ensure_ascii=False),
                )
            except (TypeError, ValueError):  # pragma: no cover - defensywne
                pass

        portfolio = self._portfolio or metadata.get("portfolio_id") or metadata.get("portfolio")
        if portfolio is None:
            portfolio = self._environment
        latency_ms = getattr(evaluation, "latency_ms", None)

        confidence_value: float | None
        try:
            confidence_value = float(signal.confidence)
        except (TypeError, ValueError):
            confidence_value = None

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
            confidence=confidence_value,
            latency_ms=latency_ms if isinstance(latency_ms, (int, float)) else None,
            telemetry_namespace=f"{self._environment}.decision.{schedule_name}",
            metadata=metadata,
        )

        try:
            journal.record(event)
        except Exception:  # pragma: no cover - dziennik nie powinien blokować handlu
            self._logger.debug("Nie udało się zapisać decision_evaluation", exc_info=True)
        self._emit_opportunity_shadow_proposal(
            evaluation=evaluation,
            candidate=candidate,
            signal=signal,
            strategy_name=strategy_name,
            schedule_name=schedule_name,
            risk_profile=risk_profile,
            timestamp=timestamp,
            portfolio=str(portfolio),
        )

    def _emit_opportunity_shadow_proposal(
        self,
        *,
        evaluation: DecisionEvaluation,
        candidate: DecisionCandidate,
        signal: StrategySignal,
        strategy_name: str,
        schedule_name: str,
        risk_profile: str,
        timestamp: datetime,
        portfolio: str,
    ) -> None:
        adapter = self._opportunity_shadow_adapter
        if adapter is None:
            return
        try:
            adapter.emit_shadow_proposal(
                evaluation=evaluation,
                candidate=candidate,
                signal=signal,
                strategy_name=strategy_name,
                schedule_name=schedule_name,
                risk_profile=risk_profile,
                timestamp=timestamp,
                environment=self._environment,
                portfolio=portfolio,
            )
        except Exception:  # pragma: no cover - adapter nie może blokować execution path
            self._logger.debug("Nie udało się wyemitować opportunity shadow proposal", exc_info=True)

    def _build_candidate(
        self,
        strategy_name: str,
        risk_profile: str,
        signal: StrategySignal,
    ) -> tuple[DecisionCandidate | None, Mapping[str, Any] | None]:
        if DecisionCandidate is None:
            return None, None
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
            return None, {
                "reason": "probability_below_threshold",
                "probability": probability,
                "min_probability": self._min_probability,
            }
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
        ), None

    def _record_filtered_signal(
        self,
        *,
        signal: StrategySignal,
        strategy_name: str,
        schedule_name: str,
        risk_profile: str,
        timestamp: datetime,
        rejection_info: Mapping[str, Any] | None,
    ) -> None:
        journal = self._journal
        if journal is None or TradingDecisionEvent is None:
            return

        metadata: dict[str, str] = {
            "decision_status": "filtered",
            "source": "decision_orchestrator",
        }
        reason: str | None = None
        probability: float | None = None
        min_probability: float | None = None
        error_detail: str | None = None
        evaluation_type: str | None = None
        if rejection_info:
            reason = str(rejection_info.get("reason") or "") or None
            raw_probability = rejection_info.get("probability")
            raw_min_probability = rejection_info.get("min_probability")
            error_detail = str(rejection_info.get("error") or "") or None
            evaluation_type = str(rejection_info.get("evaluation_type") or "") or None
            try:
                probability = float(raw_probability) if raw_probability is not None else None
            except (TypeError, ValueError):  # pragma: no cover - defensywnie
                probability = None
            try:
                min_probability = (
                    float(raw_min_probability) if raw_min_probability is not None else None
                )
            except (TypeError, ValueError):  # pragma: no cover - defensywnie
                min_probability = None

        if reason:
            metadata.setdefault("decision_reason", reason)
        if min_probability is not None:
            metadata.setdefault("min_probability", f"{min_probability:.6f}")
        if probability is not None:
            metadata.setdefault("expected_probability", f"{probability:.6f}")
        if error_detail:
            metadata.setdefault("decision_error", error_detail)
        if evaluation_type:
            metadata.setdefault("evaluation_type", evaluation_type)

        metadata.setdefault("environment", self._environment)
        metadata.setdefault("exchange", self._exchange)
        metadata.setdefault("schedule", strategy_name)

        confidence_value = probability
        if confidence_value is None:
            try:
                confidence_value = float(signal.confidence)  # type: ignore[arg-type]
            except (TypeError, ValueError):  # pragma: no cover - defensywnie
                confidence_value = None

        event = TradingDecisionEvent(
            event_type="decision_evaluation",
            timestamp=timestamp,
            environment=self._environment,
            portfolio=self._portfolio or self._environment,
            risk_profile=risk_profile,
            symbol=signal.symbol,
            side=str(signal.side),
            schedule=schedule_name,
            strategy=strategy_name,
            status="filtered",
            confidence=confidence_value,
            telemetry_namespace=f"{self._environment}.decision.{schedule_name}",
            metadata=metadata,
        )

        try:
            journal.record(event)
        except Exception:  # pragma: no cover - dziennik nie powinien blokować handlu
            self._logger.debug(
                "Nie udało się zapisać decision_evaluation (filtered)",
                exc_info=True,
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

        probability: float | None = None
        if metadata_prob is not None:
            try:
                probability = float(metadata_prob)
            except (TypeError, ValueError):
                probability = None

        if probability is None:
            try:
                probability = float(signal.confidence)
            except (TypeError, ValueError):
                probability = None

        if probability is None:
            return 0.0

        return max(0.0, min(0.995, probability))

    def _extract_expected_return(
        self, signal: StrategySignal, metadata: Mapping[str, Any]
    ) -> float:
        candidate = metadata.get("expected_return_bps")
        if candidate is None and isinstance(metadata.get("ai_manager"), Mapping):
            candidate = metadata["ai_manager"].get("expected_return_bps")
        if candidate is None:
            confidence: float | None
            try:
                confidence = float(signal.confidence)
            except (TypeError, ValueError):
                confidence = None
            if confidence is None:
                return 5.0
            base = max(0.0, confidence - 0.5)
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


def _bootstrap_portfolio_coordinator(
    *,
    scheduler: MultiStrategyScheduler,
    scheduler_cfg: MultiStrategySchedulerConfig,
    bootstrap_ctx: BootstrapContext,
    core_config: CoreConfig,
    signal_sink: StrategySignalSink,
    market_intel: MarketIntelAggregator,
    environment: EnvironmentConfig,
    environment_name: str,
    resolved_scheduler_name: str,
) -> PortfolioRuntimeCoordinator | None:
    governor_name = getattr(scheduler_cfg, "portfolio_governor", None)
    if not governor_name:
        return None

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
    fallback_candidates: tuple[Path | None, ...] = (data_cache_root, data_cache_root.parent)
    market_intel_cfg = getattr(core_config, "market_intel", None)
    stress_lab_cfg = getattr(core_config, "stress_lab", None)
    stage6_dirs: list[Path | None] = []
    if market_intel_cfg is not None:
        stage6_dirs.append(_to_path(getattr(market_intel_cfg, "output_directory", None)))
    if stress_lab_cfg is not None:
        stage6_dirs.append(_to_path(getattr(stress_lab_cfg, "report_directory", None)))
    fallback_directories = _unique_paths((*fallback_candidates, *stage6_dirs))

    slo_provider = None
    stress_provider = None
    stress_age: timedelta | None = _minutes_to_timedelta(None, default_minutes=240.0)
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
    if stress_provider is None:
        overrides = _load_stress_overrides_from_reports(
            governor_name,
            fallback_directories,
            max_age=stress_age,
        )
        if overrides:
            cached_overrides = tuple(overrides)

            def _fallback_stress_provider() -> Sequence[StressOverrideRecommendation]:
                return cached_overrides

            stress_provider = _fallback_stress_provider

    def _market_data_provider() -> Mapping[str, MarketIntelSnapshot]:
        if not asset_symbols:
            return {}
        queries = [
            MarketIntelQuery(symbol=symbol, interval=interval, lookback_bars=lookback)
            for symbol in asset_symbols
        ]
        snapshots: dict[str, MarketIntelSnapshot] = {}
        fallback_snapshots: Mapping[str, MarketIntelSnapshot] | None = None
        try:
            snapshots.update(market_intel.build_many(queries))
        except Exception:  # pragma: no cover - diagnostyka danych
            _LOGGER.exception("PortfolioGovernor: błąd budowania metryk Market Intel")
        missing_symbols = [query.symbol for query in queries if query.symbol not in snapshots]
        if missing_symbols and market_intel_cfg is not None:
            fallback_snapshots = _load_market_intel_snapshots_from_reports(
                governor_name,
                market_intel_cfg,
                fallback_directories,
            )
            for symbol in missing_symbols:
                if fallback_snapshots and symbol in fallback_snapshots:
                    snapshots[symbol] = fallback_snapshots[symbol]
        if len(snapshots) < len(queries):
            for query in queries:
                if query.symbol in snapshots:
                    continue
                try:
                    snapshots[query.symbol] = market_intel.build_snapshot(query)
                    continue
                except Exception:
                    _LOGGER.debug("Brak metryk Market Intel dla %s", query.symbol, exc_info=True)
                if fallback_snapshots is None and market_intel_cfg is not None:
                    fallback_snapshots = _load_market_intel_snapshots_from_reports(
                        governor_name,
                        market_intel_cfg,
                        fallback_directories,
                    )
                if fallback_snapshots and query.symbol in fallback_snapshots:
                    snapshots[query.symbol] = fallback_snapshots[query.symbol]
        if snapshots:
            return snapshots
        if market_intel_cfg is not None:
            fallback_snapshots = _load_market_intel_snapshots_from_reports(
                governor_name,
                market_intel_cfg,
                fallback_directories,
            )
            if fallback_snapshots:
                return fallback_snapshots
        return {}

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

    dynamic_policy_cfg = getattr(scheduler_cfg, "dynamic_capital_policy", None)
    if dynamic_policy_cfg and isinstance(governor, StrategyPortfolioGovernor):
        label = None
        if isinstance(dynamic_policy_cfg, Mapping):
            label = dynamic_policy_cfg.get("label")
        elif hasattr(dynamic_policy_cfg, "label"):
            label = getattr(dynamic_policy_cfg, "label")
        label_text = str(label or "governor_dynamic")

        def _sync_capital_policy(_decision: PortfolioDecision) -> None:
            if hasattr(governor, "current_weights_snapshot"):
                weights = governor.current_weights_snapshot()
            else:
                attr = getattr(governor, "current_weights", None)
                if callable(attr):
                    weights = dict(attr())
                elif isinstance(attr, Mapping):
                    weights = dict(attr)
                else:
                    weights = {}
            if not weights:
                return
            policy = FixedWeightAllocation(dict(weights), label=label_text)
            snapshot_builder = getattr(governor, "dynamic_policy_snapshot", None)
            if callable(snapshot_builder):
                try:
                    policy.metadata = snapshot_builder()
                except Exception:  # pragma: no cover - diagnostyka metadanych
                    _LOGGER.debug(
                        "PortfolioGovernor: nie udało się zbudować migawki dynamicznej polityki",
                        exc_info=True,
                    )
            scheduler.set_capital_policy(policy)

        portfolio_coordinator.set_capital_policy_listener(_sync_capital_policy)

    if bootstrap_ctx.tco_reporter is not None:
        reporter = bootstrap_ctx.tco_reporter
        consumer = getattr(governor, "update_costs_from_report", None)
        if callable(consumer):
            portfolio_coordinator.set_tco_report_hooks(
                provider=lambda: reporter.build_report(),
                consumer=consumer,
            )
    return portfolio_coordinator


def build_multi_strategy_runtime(
    *,
    environment_name: str,
    scheduler_name: str | None,
    config_path: str | Path,
    secret_manager: SecretManager,
    adapter_factories: Mapping[str, ExchangeAdapterFactory] | None = None,
    telemetry_emitter: Callable[[str, Mapping[str, float]], None] | None = None,
    runtime_config: RuntimeAppConfig | None = None,
    start_stream: bool = True,
    stream_async: bool = False,
    async_loop: asyncio.AbstractEventLoop | None = None,
) -> MultiStrategyRuntime:
    bootstrap_ctx = _RISK_BOOTSTRAPPER.bootstrap_context(
        environment_name=environment_name,
        config_path=config_path,
        secret_manager=secret_manager,
        adapter_factories=adapter_factories,
        bootstrap_fn=bootstrap_environment,
    )
    guard = getattr(bootstrap_ctx, "capability_guard", None) or get_capability_guard()
    if guard is not None:
        try:
            guard.require_runtime(
                "multi_strategy_scheduler",
                message="Scheduler multi-strategy wymaga aktywnego runtime Multi-Strategy.",
            )
            guard.require_module(
                "walk_forward",
                message="Moduł Walk Forward jest wymagany do uruchomienia scheduler-a multi-strategy.",
            )
            guard.reserve_slot("bot")
            slot_kind = (
                "live_controller"
                if bootstrap_ctx.environment.environment is Environment.LIVE
                else "paper_controller"
            )
            guard.reserve_slot(slot_kind)
        except LicenseCapabilityError as exc:
            raise RuntimeError(str(exc)) from exc
    core_config = bootstrap_ctx.core_config
    environment = bootstrap_ctx.environment
    resolved_scheduler = _PIPELINE_CONFIG_LOADER.resolve_multi_strategy_scheduler(
        core_config=core_config,
        scheduler_name=scheduler_name,
    )
    resolved_scheduler_name = resolved_scheduler.scheduler_name
    scheduler_cfg = resolved_scheduler.scheduler_config

    paper_settings = _normalize_paper_settings(environment)
    allowed_quotes = paper_settings["allowed_quotes"]
    universe = _resolve_universe(core_config, environment)
    markets = _build_markets(universe, environment.exchange, allowed_quotes, paper_settings)
    if not markets:
        raise ValueError(
            "Brak instrumentów dla scheduler-a multi-strategy – sprawdź instrument_universe"
        )

    cached_source = _create_cached_source(bootstrap_ctx.adapter, environment)
    storage = cached_source.storage
    market_intel = MarketIntelAggregator(storage)

    strategy_bootstrap = _STRATEGY_BOOTSTRAPPER.bootstrap(core_config)
    strategies = dict(strategy_bootstrap.strategies)
    interval_map: dict[str, str] = {}
    symbols_map: dict[str, Sequence[str]] = {}
    all_symbols = tuple(markets.keys())
    for schedule in scheduler_cfg.schedules:
        interval = schedule.interval or "1h"
        interval_map[schedule.strategy] = interval
        symbols_map.setdefault(schedule.strategy, all_symbols)

    base_feed = OHLCVStrategyFeed(cached_source, symbols_map=symbols_map, interval_map=interval_map)
    adapter_settings = getattr(environment, "adapter_settings", None)
    adapter_metrics = _resolve_adapter_metrics_registry(getattr(bootstrap_ctx, "adapter", None))
    stream_feed = _DATA_SOURCE_BOOTSTRAPPER.build_streaming_feed(
        stream_factory=_build_streaming_feed,
        stream_config=getattr(environment, "stream", None),
        stream_settings=(
            adapter_settings.get("stream") if isinstance(adapter_settings, Mapping) else None
        ),
        adapter_metrics=adapter_metrics,
        base_feed=base_feed,
        symbols_map=symbols_map,
        exchange=environment.exchange,
        environment_name=getattr(environment.environment, "value", None),
    )
    stream_task: asyncio.Task[None] | None = None
    if stream_feed is not None:
        if start_stream:
            if stream_async:
                try:
                    stream_task = stream_feed.start_async(loop=async_loop)
                except RuntimeError as exc:
                    raise RuntimeError(
                        "Asynchroniczne uruchomienie streamu wymaga aktywnej pętli asyncio."
                    ) from exc
            else:
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
    capital_policy_spec = getattr(scheduler_cfg, "capital_policy", None)
    capital_policy, policy_interval = _resolve_capital_policy(capital_policy_spec)
    allocation_interval_raw = getattr(
        scheduler_cfg,
        "allocation_rebalance_seconds",
        None,
    )

    allocation_interval: float | None = None
    if allocation_interval_raw not in (None, ""):
        try:
            allocation_interval = float(allocation_interval_raw)
        except (TypeError, ValueError):
            _LOGGER.debug(
                "Nie udało się sparsować allocation_rebalance_seconds=%s",
                allocation_interval_raw,
                exc_info=True,
            )
            allocation_interval = None
    if allocation_interval is None and policy_interval is not None:
        allocation_interval = policy_interval

    io_dispatcher, io_guardrails = _RISK_BOOTSTRAPPER.bootstrap_io_guardrails(
        runtime_config=runtime_config,
        bootstrap_ctx=bootstrap_ctx,
        environment_name=environment_name,
    )

    scheduler = MultiStrategyScheduler(
        environment=environment_name,
        portfolio=str(paper_settings["portfolio_id"]),
        telemetry_emitter=telemetry_emitter,
        decision_journal=bootstrap_ctx.decision_journal,
        portfolio_governor=portfolio_governor,
        capital_policy=capital_policy,
        allocation_rebalance_seconds=allocation_interval,
        io_dispatcher=io_dispatcher,
    )

    signal_limits = getattr(scheduler_cfg, "signal_limits", None)
    _RISK_BOOTSTRAPPER.bind_scheduler_limits(scheduler, signal_limits=signal_limits)

    portfolio_coordinator = _bootstrap_portfolio_coordinator(
        scheduler=scheduler,
        scheduler_cfg=scheduler_cfg,
        bootstrap_ctx=bootstrap_ctx,
        core_config=core_config,
        signal_sink=signal_sink,
        market_intel=market_intel,
        environment=environment,
        environment_name=environment_name,
        resolved_scheduler_name=resolved_scheduler_name,
    )

    _STRATEGY_BOOTSTRAPPER.validate_schedule_strategies(
        schedules=scheduler_cfg.schedules,
        strategies=strategies,
    )

    for schedule in scheduler_cfg.schedules:
        strategy = strategies[schedule.strategy]
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

    _apply_initial_signal_limits(
        scheduler,
        getattr(scheduler_cfg, "initial_signal_limits", None),
    )

    _apply_initial_suspensions(
        scheduler,
        getattr(scheduler_cfg, "initial_suspensions", None),
    )

    runtime_instance = MultiStrategyRuntime(
        bootstrap=bootstrap_ctx,
        scheduler=scheduler,
        data_feed=data_feed,
        signal_sink=signal_sink,
        strategies=strategies,
        schedules=tuple(scheduler_cfg.schedules),
        capital_policy=capital_policy,
        portfolio_coordinator=portfolio_coordinator,
        portfolio_governor=portfolio_governor,
        tco_reporter=bootstrap_ctx.tco_reporter,
        stream_feed=stream_feed,
        stream_feed_task=stream_task,
        decision_sink=decision_sink,
    )

    optimization_cfg = getattr(runtime_config, "optimization", None) if runtime_config else None
    optimization_repo: ModelRepository | None = None
    try:
        optimization_repo = FilesystemModelRepository(Path("ai_models"))
    except Exception:
        _LOGGER.debug("Nie udało się zainicjalizować FilesystemModelRepository", exc_info=True)

    optimization_scheduler = _configure_optimization_scheduler(
        runtime_instance,
        core_config=core_config,
        optimization_cfg=optimization_cfg,
        model_repository=optimization_repo,
        catalog=DEFAULT_STRATEGY_CATALOG,
    )
    if optimization_scheduler is not None and hasattr(scheduler, "add_portfolio_decision_listener"):

        def _trigger_after_portfolio(decision: PortfolioDecision) -> None:
            if not getattr(decision, "rebalance_required", False):
                return
            try:
                optimization_scheduler.trigger()
            except Exception:  # pragma: no cover - diagnostyka schedulerów optymalizacji
                _LOGGER.exception(
                    "PortfolioGovernor: błąd wyzwolenia optymalizacji po decyzji portfelowej"
                )

        scheduler.add_portfolio_decision_listener(_trigger_after_portfolio)

    return runtime_instance


def _collect_strategy_definitions(core_config: CoreConfig) -> dict[str, StrategyDefinition]:
    return _STRATEGY_BOOTSTRAPPER.collect_definitions(core_config)


def _instantiate_strategies(
    core_config: CoreConfig, *, catalog: StrategyCatalog | None = None
) -> dict[str, StrategyEngine]:
    definitions = _STRATEGY_BOOTSTRAPPER.collect_definitions(core_config)
    bootstrapper = (
        _STRATEGY_BOOTSTRAPPER if catalog is None else StrategyBootstrapper(catalog=catalog)
    )
    return bootstrapper.instantiate(definitions)


def describe_strategy_definitions(
    core_config: CoreConfig,
    *,
    catalog: StrategyCatalog | None = None,
) -> Sequence[Mapping[str, object]]:
    """Buduje opis strategii skonfigurowanych w pliku core."""

    catalog = catalog or DEFAULT_STRATEGY_CATALOG
    definitions = _collect_strategy_definitions(core_config)
    described = catalog.describe_definitions(definitions, include_metadata=True)
    return list(described)


def describe_multi_strategy_configuration(
    *,
    config_path: str | Path,
    scheduler_name: str | None = None,
    catalog: StrategyCatalog | None = None,
    include_strategy_definitions: bool = True,
    only_scheduler_definitions: bool = False,
) -> Mapping[str, object]:
    """Opisuje konfigurację scheduler-a multi-strategy bez uruchamiania runtime."""

    resolved_catalog = catalog or DEFAULT_STRATEGY_CATALOG
    core_config = _PIPELINE_CONFIG_LOADER.load_core_config(config_path)
    scheduler_configs = getattr(core_config, "multi_strategy_schedulers", {})
    if not scheduler_configs:
        raise ValueError("Konfiguracja nie zawiera sekcji multi_strategy_schedulers.")

    resolved_name = scheduler_name or next(iter(scheduler_configs))
    scheduler_cfg = scheduler_configs.get(resolved_name)
    if scheduler_cfg is None:
        raise KeyError(f"Nie znaleziono scheduler-a '{resolved_name}'.")

    definitions = _collect_strategy_definitions(core_config)
    guard = get_capability_guard()
    schedules: list[dict[str, object]] = []
    blocked_schedules: list[str] = []
    blocked_strategies: list[str] = []
    blocked_strategy_capabilities: dict[str, str] = {}
    blocked_schedule_capabilities: dict[str, str] = {}
    strategy_capabilities: dict[str, str] = {}

    for schedule in scheduler_cfg.schedules:
        entry: dict[str, object] = {
            "name": schedule.name,
            "strategy": schedule.strategy,
            "risk_profile": schedule.risk_profile,
            "cadence_seconds": int(schedule.cadence_seconds),
            "max_drift_seconds": int(schedule.max_drift_seconds),
            "warmup_bars": int(schedule.warmup_bars),
            "max_signals": int(schedule.max_signals),
        }
        if schedule.interval:
            entry["interval"] = schedule.interval
        definition = definitions.get(schedule.strategy)
        capability_id: str | None = None
        if definition is not None:
            entry["engine"] = definition.engine
            if definition.risk_profile:
                entry["definition_risk_profile"] = definition.risk_profile
            raw_capability = definition.metadata.get("capability")
            if raw_capability not in (None, ""):
                capability_id = str(raw_capability).strip() or None
            try:
                spec = resolved_catalog.get(definition.engine)
                if spec.capability:
                    capability_id = spec.capability
                tags = tuple(dict.fromkeys((*spec.default_tags, *definition.tags)))
                entry["tags"] = list(tags)
                if spec.capability:
                    entry["capability"] = spec.capability
                entry["license_tier"] = spec.license_tier
                entry["risk_classes"] = list(
                    dict.fromkeys((*spec.risk_classes, *definition.risk_classes))
                )
                entry["required_data"] = list(
                    dict.fromkeys((*spec.required_data, *definition.required_data))
                )
            except KeyError:
                if definition.tags:
                    entry["tags"] = list(dict.fromkeys(definition.tags))
                if definition.license_tier:
                    entry["license_tier"] = definition.license_tier
                if definition.risk_classes:
                    entry["risk_classes"] = list(dict.fromkeys(definition.risk_classes))
                if definition.required_data:
                    entry["required_data"] = list(dict.fromkeys(definition.required_data))
                if capability_id is None:
                    extra_capability = definition.metadata.get("capability")
                    if extra_capability not in (None, ""):
                        capability_id = str(extra_capability).strip() or None
        normalized_strategy = (schedule.strategy or "").strip()
        if capability_id and normalized_strategy:
            strategy_capabilities.setdefault(normalized_strategy, capability_id)

        if guard is not None and capability_id:
            try:
                if not guard.capabilities.is_strategy_enabled(capability_id):
                    if schedule.name not in blocked_schedules:
                        blocked_schedules.append(schedule.name)
                    strategy_name = schedule.strategy
                    if strategy_name not in blocked_strategies:
                        blocked_strategies.append(strategy_name)
                    if capability_id and strategy_name:
                        blocked_strategy_capabilities.setdefault(strategy_name, capability_id)
                    if schedule.name and capability_id:
                        blocked_schedule_capabilities.setdefault(schedule.name, capability_id)
                    continue
            except AttributeError:
                pass
        schedules.append(entry)

    schedules.sort(key=lambda item: item["name"])
    allowed_strategies = {str(entry["strategy"]) for entry in schedules if entry.get("strategy")}

    policy_spec = getattr(scheduler_cfg, "capital_policy", None)
    policy, policy_interval = _resolve_capital_policy(policy_spec, _allow_profile=False)
    policy_summary: dict[str, object] = {
        "name": getattr(policy, "name", policy.__class__.__name__),
    }
    if policy_interval is not None:
        policy_summary["policy_interval_seconds"] = float(policy_interval)
    cfg_interval = getattr(scheduler_cfg, "allocation_rebalance_seconds", None)
    if cfg_interval not in (None, ""):
        try:
            policy_summary["configured_rebalance_seconds"] = float(cfg_interval)
        except (TypeError, ValueError):
            policy_summary["configured_rebalance_seconds"] = cfg_interval

    def _serialize_limit_tree(
        tree: Mapping[str, Mapping[str, object]],
        *,
        blocked: dict[str, set[str]] | None = None,
        blocked_capabilities: dict[str, str] | None = None,
    ) -> Mapping[str, Mapping[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for strategy_name, profiles in (tree or {}).items():
            strategy_key = str(strategy_name)
            if allowed_strategies and strategy_key not in allowed_strategies:
                if blocked is not None:
                    blocked_profiles = blocked.setdefault(strategy_key, set())
                    if isinstance(profiles, Mapping):
                        for profile_name in profiles.keys():
                            blocked_profiles.add(str(profile_name))
                    else:
                        blocked_profiles.add("*")
                if blocked_capabilities is not None:
                    capability_id = blocked_strategy_capabilities.get(
                        strategy_key
                    ) or strategy_capabilities.get(strategy_key)
                    if capability_id:
                        blocked_capabilities.setdefault(strategy_key, capability_id)
                continue
            if not isinstance(profiles, Mapping):
                continue
            profile_entry: dict[str, object] = {}
            for profile_name, raw_limit in profiles.items():
                if isinstance(raw_limit, SignalLimitOverrideConfig):
                    payload: dict[str, object] = {"limit": int(raw_limit.limit)}
                    if raw_limit.reason:
                        payload["reason"] = raw_limit.reason
                    if raw_limit.until:
                        payload["until"] = raw_limit.until.isoformat()
                    if raw_limit.duration_seconds is not None:
                        payload["duration_seconds"] = float(raw_limit.duration_seconds)
                    profile_entry[profile_name] = payload
                else:
                    try:
                        profile_entry[profile_name] = {"limit": int(raw_limit)}
                    except (TypeError, ValueError):
                        continue
            if profile_entry:
                result[strategy_key] = profile_entry
        return result

    suspensions_payload: list[dict[str, object]] = []
    blocked_suspensions: list[dict[str, object]] = []
    blocked_suspension_capabilities: dict[str, str] = {}
    for suspension in getattr(scheduler_cfg, "initial_suspensions", ()):
        payload: dict[str, object] = {
            "kind": suspension.kind,
            "target": suspension.target,
        }
        kind = (suspension.kind or "schedule").lower()
        target = str(suspension.target)
        if suspension.reason:
            payload["reason"] = suspension.reason
        if suspension.until:
            payload["until"] = suspension.until.isoformat()
        if suspension.duration_seconds is not None:
            payload["duration_seconds"] = float(suspension.duration_seconds)
        if kind != "tag" and allowed_strategies and target not in allowed_strategies:
            capability_id: str | None = None
            if kind == "schedule":
                capability_id = (
                    blocked_schedule_capabilities.get(target)
                    or strategy_capabilities.get(target)
                    or blocked_strategy_capabilities.get(target)
                )
            else:
                capability_id = blocked_strategy_capabilities.get(
                    target
                ) or strategy_capabilities.get(target)
            if capability_id:
                payload["capability"] = capability_id
                key = f"{kind}:{target}".strip(":")
                if key:
                    blocked_suspension_capabilities.setdefault(key, capability_id)
            blocked_suspensions.append(dict(payload))
            continue
        suspensions_payload.append(payload)

    blocked_initial_limits: dict[str, set[str]] = {}
    blocked_static_limits: dict[str, set[str]] = {}
    blocked_initial_limit_capabilities: dict[str, str] = {}
    blocked_static_limit_capabilities: dict[str, str] = {}

    initial_limits = _serialize_limit_tree(
        getattr(scheduler_cfg, "initial_signal_limits", {}),
        blocked=blocked_initial_limits,
        blocked_capabilities=blocked_initial_limit_capabilities,
    )
    static_limits = _serialize_limit_tree(
        getattr(scheduler_cfg, "signal_limits", {}),
        blocked=blocked_static_limits,
        blocked_capabilities=blocked_static_limit_capabilities,
    )

    summary: dict[str, object] = {
        "config_path": str(Path(config_path).expanduser()),
        "scheduler": resolved_name,
        "capital_policy": policy_summary,
        "schedules": schedules,
        "initial_suspensions": suspensions_payload,
        "initial_signal_limits": initial_limits,
    }
    if blocked_schedules:
        summary["blocked_schedules"] = blocked_schedules
    if blocked_strategies:
        summary["blocked_strategies"] = blocked_strategies
    if blocked_strategy_capabilities:
        summary["blocked_capabilities"] = {
            name: blocked_strategy_capabilities[name]
            for name in blocked_strategies
            if name in blocked_strategy_capabilities
        }
    if blocked_schedule_capabilities:
        summary["blocked_schedule_capabilities"] = {
            name: blocked_schedule_capabilities[name]
            for name in blocked_schedules
            if name in blocked_schedule_capabilities
        }
    if blocked_suspensions:
        summary["blocked_suspensions"] = blocked_suspensions
    if blocked_suspension_capabilities:
        summary["blocked_suspension_capabilities"] = {
            name: blocked_suspension_capabilities[name]
            for name in sorted(blocked_suspension_capabilities)
        }
    if static_limits:
        summary["signal_limits"] = static_limits
    merged_initial_capabilities: dict[str, str] = dict(blocked_initial_limit_capabilities)
    if blocked_initial_limits:
        summary["blocked_initial_signal_limits"] = {
            name: sorted(profiles) for name, profiles in blocked_initial_limits.items()
        }
        for name in blocked_initial_limits:
            capability_id = (
                merged_initial_capabilities.get(name)
                or blocked_strategy_capabilities.get(name)
                or strategy_capabilities.get(name)
            )
            if capability_id:
                merged_initial_capabilities[name] = capability_id
    if merged_initial_capabilities:
        summary["blocked_initial_signal_limit_capabilities"] = {
            name: merged_initial_capabilities[name] for name in sorted(merged_initial_capabilities)
        }

    merged_static_capabilities: dict[str, str] = dict(blocked_static_limit_capabilities)
    if blocked_static_limits:
        summary["blocked_signal_limits"] = {
            name: sorted(profiles) for name, profiles in blocked_static_limits.items()
        }
        for name in blocked_static_limits:
            capability_id = (
                merged_static_capabilities.get(name)
                or blocked_strategy_capabilities.get(name)
                or strategy_capabilities.get(name)
            )
            if capability_id:
                merged_static_capabilities[name] = capability_id
    if merged_static_capabilities:
        summary["blocked_signal_limit_capabilities"] = {
            name: merged_static_capabilities[name] for name in sorted(merged_static_capabilities)
        }
    if getattr(scheduler_cfg, "portfolio_governor", None):
        summary["portfolio_governor"] = scheduler_cfg.portfolio_governor
    if include_strategy_definitions:
        if only_scheduler_definitions:
            used_names = {entry["strategy"] for entry in schedules}
            definitions_map = {
                name: definition
                for name, definition in _collect_strategy_definitions(core_config).items()
                if name in used_names
            }
            summary["strategies"] = resolved_catalog.describe_definitions(
                definitions_map,
                include_metadata=True,
            )
        else:
            summary["strategies"] = describe_strategy_definitions(
                core_config,
                catalog=resolved_catalog,
            )
    return summary


def _build_mode_runtime(
    mode: str,
    *,
    config_path: str | Path,
    secret_manager: SecretManager,
    scheduler_name: str | None,
    telemetry_emitter: Callable[[str, Mapping[str, float]], None] | None,
    adapter_factories: Mapping[str, ExchangeAdapterFactory] | None,
    environment_aliases: Sequence[str],
    environment_type: Environment | None,
    prefer_offline: bool | None,
    environment_name: str | None,
) -> MultiStrategyRuntime:
    core_config = _PIPELINE_CONFIG_LOADER.load_core_config(config_path)
    resolved_env = environment_name or _resolve_environment_name_for_mode(
        core_config,
        aliases=environment_aliases,
        environment_type=environment_type,
        prefer_offline=prefer_offline,
    )
    if resolved_env is None:
        raise ValueError(
            f"Nie udało się odnaleźć środowiska {mode} w konfiguracji. Dodaj sekcję environments."
        )
    return build_multi_strategy_runtime(
        environment_name=resolved_env,
        scheduler_name=scheduler_name,
        config_path=config_path,
        secret_manager=secret_manager,
        adapter_factories=adapter_factories,
        telemetry_emitter=telemetry_emitter,
    )


def build_demo_multi_strategy_runtime(
    *,
    config_path: str | Path,
    secret_manager: SecretManager,
    scheduler_name: str | None = None,
    telemetry_emitter: Callable[[str, Mapping[str, float]], None] | None = None,
    adapter_factories: Mapping[str, ExchangeAdapterFactory] | None = None,
    environment_name: str | None = None,
) -> MultiStrategyRuntime:
    return _build_mode_runtime(
        "demo",
        config_path=config_path,
        secret_manager=secret_manager,
        scheduler_name=scheduler_name,
        telemetry_emitter=telemetry_emitter,
        adapter_factories=adapter_factories,
        environment_aliases=("demo", "offline", "test", "sandbox"),
        environment_type=Environment.TESTNET,
        prefer_offline=True,
        environment_name=environment_name,
    )


def build_paper_multi_strategy_runtime(
    *,
    config_path: str | Path,
    secret_manager: SecretManager,
    scheduler_name: str | None = None,
    telemetry_emitter: Callable[[str, Mapping[str, float]], None] | None = None,
    adapter_factories: Mapping[str, ExchangeAdapterFactory] | None = None,
    environment_name: str | None = None,
) -> MultiStrategyRuntime:
    return _build_mode_runtime(
        "paper",
        config_path=config_path,
        secret_manager=secret_manager,
        scheduler_name=scheduler_name,
        telemetry_emitter=telemetry_emitter,
        adapter_factories=adapter_factories,
        environment_aliases=("paper", "stage6"),
        environment_type=Environment.PAPER,
        prefer_offline=False,
        environment_name=environment_name,
    )


def build_live_multi_strategy_runtime(
    *,
    config_path: str | Path,
    secret_manager: SecretManager,
    scheduler_name: str | None = None,
    telemetry_emitter: Callable[[str, Mapping[str, float]], None] | None = None,
    adapter_factories: Mapping[str, ExchangeAdapterFactory] | None = None,
    environment_name: str | None = None,
) -> MultiStrategyRuntime:
    return _build_mode_runtime(
        "live",
        config_path=config_path,
        secret_manager=secret_manager,
        scheduler_name=scheduler_name,
        telemetry_emitter=telemetry_emitter,
        adapter_factories=adapter_factories,
        environment_aliases=("live", "prod", "production"),
        environment_type=Environment.LIVE,
        prefer_offline=False,
        environment_name=environment_name,
    )


def build_multi_portfolio_scheduler_from_config(
    *,
    core_config: CoreConfig,
    catalog: StrategyCatalog | None = None,
    audit_logger: Callable[[Mapping[str, object]], None] | None = None,
    health_monitor: StrategyHealthMonitor | None = None,
    clock: Callable[[], datetime] | None = None,
) -> MultiPortfolioScheduler:
    definitions = getattr(core_config, "multi_portfolio", None)
    if definitions is None:
        raise ValueError("Konfiguracja nie zawiera sekcji multi_portfolio")
    catalog = catalog or DEFAULT_STRATEGY_CATALOG
    scheduler = MultiPortfolioScheduler(
        catalog,
        audit_logger=audit_logger,
        health_monitor=health_monitor,
        clock=clock,
    )
    entries = _PIPELINE_CONFIG_LOADER.resolve_multi_portfolio_entries(definitions)
    for entry in entries:
        binding = _PIPELINE_CONFIG_LOADER.build_portfolio_binding(entry)
        scheduler.register_portfolio(binding)
    return scheduler


def describe_multi_portfolio_state(
    scheduler: MultiPortfolioScheduler,
) -> Sequence[Mapping[str, object]]:
    return [
        scheduler.portfolio_state(portfolio_id)
        for portfolio_id in scheduler.registered_portfolios()
    ]


Pipeline = StreamingStrategyFeed


__all__ = [
    "DailyTrendPipeline",
    "build_daily_trend_pipeline",
    "create_trading_controller",
    "MultiStrategyRuntime",
    "build_multi_strategy_runtime",
    "build_demo_multi_strategy_runtime",
    "build_paper_multi_strategy_runtime",
    "build_live_multi_strategy_runtime",
    "build_multi_portfolio_scheduler_from_config",
    "describe_multi_portfolio_state",
    "consume_stream",
    "consume_stream_async",
    "OHLCVStrategyFeed",
    "InMemoryStrategySignalSink",
    "StreamingStrategyFeed",
    "Pipeline",
    "DecisionAwareSignalSink",
    "describe_strategy_definitions",
    "describe_multi_strategy_configuration",
]
