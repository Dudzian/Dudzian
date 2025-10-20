# auto_trader.py
# Walk-forward + auto-reoptimization + optional auto-trade loop, integrowany z emiterami zdarzeń.
from __future__ import annotations

import threading
import time
import statistics
import asyncio
import math
from typing import (
    Iterable,
    Mapping,
    Optional,
    Sequence,
    List,
    Dict,
    Any,
    Callable,
    Tuple,
    TYPE_CHECKING,
)
from datetime import datetime, timezone
import inspect
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
try:
    from collections import deque
except Exception:
    # minimal fallback
    class deque(list):
        def __init__(self, maxlen=None): super().__init__(); self.maxlen=maxlen
        def append(self, x):
            super().append(x)
            if self.maxlen and len(self) > self.maxlen:
                del self[0]
        def popleft(self): return super().pop(0)

from importlib import import_module

from bot_core.alerts import AlertSeverity, emit_alert as _core_emit_alert
from bot_core.auto_trader.app import EmitterLike, RiskDecision as _CoreRiskDecision
from bot_core.risk.events import RiskDecisionLog
from bot_core.risk.repository import FileRiskRepository
try:  # pragma: no cover - środowiska testowe mogą nie zawierać kompletu modeli
    from bot_core.config.models import RiskProfileConfig  # type: ignore
except Exception:  # pragma: no cover - fallback gdy modele konfiguracji są niedostępne
    RiskProfileConfig = None  # type: ignore
from KryptoLowca.logging_utils import get_logger, LOGS_DIR
from KryptoLowca.config_manager import StrategyConfig
from KryptoLowca.telemetry.prometheus_exporter import metrics as prometheus_metrics
from KryptoLowca.core.services import ExecutionService, RiskService, SignalService, exception_guard
from KryptoLowca.core.services.data_provider import ExchangeDataProvider
try:  # pragma: no cover - zależność opcjonalna w środowisku CI
    from bot_core.market_intel import MarketIntelAggregator, MarketIntelQuery  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - fallback gdy moduł nie występuje
    MarketIntelAggregator = None  # type: ignore[assignment]
    MarketIntelQuery = None  # type: ignore[assignment]
try:  # pragma: no cover - zależności runtime mogą być niekompletne
    from bot_core.runtime import PaperTradingAdapter  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - fallback gdy adapter nie jest eksportowany
    try:
        from bot_core.runtime.paper_trading import PaperTradingAdapter  # type: ignore
    except Exception:
        PaperTradingAdapter = None  # type: ignore

try:  # pragma: no cover - resolve_core_config_path nie zawsze dostępny w __all__
    from bot_core.runtime import resolve_core_config_path  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - fallback do modułu paths
    try:
        from bot_core.runtime.paths import resolve_core_config_path  # type: ignore
    except Exception:  # pragma: no cover - ostateczny fallback
        def resolve_core_config_path(*_: Any, **__: Any) -> Path | None:
            return None
from bot_core.runtime.metadata import (
    RiskManagerSettings,
    load_risk_manager_settings,
    load_runtime_entrypoint_metadata,
)
from KryptoLowca.strategies.base import DataProvider, StrategyMetadata, StrategySignal

try:  # pragma: no cover - zależności bot_core mogą nie być dostępne w każdym środowisku
    from bot_core.decision.ai_connector import AIManagerDecisionConnector  # type: ignore
    from bot_core.execution.base import ExecutionContext as CoreExecutionContext  # type: ignore
    from bot_core.execution.base import ExecutionService as CoreExecutionService  # type: ignore
    from bot_core.exchanges.base import AccountSnapshot, OrderRequest  # type: ignore
    from bot_core.risk.base import RiskProfile  # type: ignore
    from bot_core.risk.engine import ThresholdRiskEngine  # type: ignore
    from bot_core.risk.factory import build_risk_profile_from_config  # type: ignore
    from bot_core.risk.profiles.manual import ManualProfile  # type: ignore
except Exception:  # pragma: no cover - fallback gdy bot_core nie jest kompletny
    AIManagerDecisionConnector = None  # type: ignore
    CoreExecutionContext = None  # type: ignore
    CoreExecutionService = None  # type: ignore
    AccountSnapshot = None  # type: ignore
    OrderRequest = None  # type: ignore
    ThresholdRiskEngine = None  # type: ignore
    RiskProfile = None  # type: ignore
    build_risk_profile_from_config = None  # type: ignore

try:  # pragma: no cover - RiskDecisionLog może nie być dostępny w trybie offline
    from bot_core.risk.events import RiskDecisionLog  # type: ignore
except Exception:  # pragma: no cover - fallback defensywny
    RiskDecisionLog = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from KryptoLowca.data.market_data import MarketDataProvider, MarketDataRequest

logger = get_logger(__name__)


def _emit_alert(*args: Any, **kwargs: Any) -> None:
    """Deleguje alerty do funkcji eksportowanej przez pakiet."""

    module = import_module(__package__ or "KryptoLowca.auto_trader")
    handler: Callable[..., None] = getattr(module, "emit_alert", _core_emit_alert)
    handler(*args, **kwargs)

RiskDecision = _CoreRiskDecision


class _NullExchangeAdapter:
    """Minimalny adapter wykorzystywany, gdy nie podano właściwego wykonawcy."""

    def __init__(self, emitter: EmitterLike | None) -> None:
        self._emitter = emitter

    async def submit_order(self, *, symbol: str, side: str, size: float, **kwargs: Any) -> Mapping[str, Any]:
        message = "Execution adapter not configured; skipping order"
        if self._emitter is not None:
            try:
                self._emitter.log(message, level="WARNING", component="AutoTrader")
            except Exception:  # pragma: no cover - defensywne logowanie
                logger.warning(message)
        logger.warning("%s (symbol=%s side=%s size=%s)", message, symbol, side, size)
        return {
            "status": "skipped",
            "symbol": symbol,
            "side": side,
            "size": size,
        }


class AutoTrader:
    """
    - Listens to trade_closed events to compute rolling PF & Expectancy
    - Monitors ATR (if 'bar' events are emitted)
    - Triggers reoptimization when thresholds break
    - Optional walk-forward scheduler (time-based)
    - Optional auto-trade loop (asks AI for decision and executes via GUI bridge)
    - Runtime-reconfigurable via ControlPanel (configure()/set_enable_auto_trade()).
    """
    BACKTEST_GUARD_MAX_AGE_S = 30 * 24 * 3600
    def __init__(
        self,
        emitter: EmitterLike,
        gui,
        symbol_getter: Callable[[], str],
        pf_min: float = 1.3,
        expectancy_min: float = 0.0,
        metrics_window: int = 30,
        atr_ratio_threshold: float = 0.5,   # +50% vs baseline
        atr_baseline_len: int = 100,
        reopt_cooldown_s: int = 1800,       # 30 min cooldown
        walkforward_interval_s: Optional[int] = 3600,  # every 1h
        walkforward_min_closed_trades: int = 10,
        enable_auto_trade: bool = True,
        auto_trade_interval_s: int = 30,
        market_data_provider: Optional["MarketDataProvider"] = None,
        *,
        signal_service: Optional[SignalService] = None,
        risk_service: Optional[RiskService] = None,
        execution_service: Optional[ExecutionService] = None,
        data_provider: Optional[DataProvider] = None,
        bootstrap_context: Any | None = None,
        core_risk_engine: Any | None = None,
        core_execution_service: Any | None = None,
        ai_connector: Any | None = None,
        market_intel: Any | None = None,
    ) -> None:
        self.emitter = emitter
        self.gui = gui
        self.symbol_getter = symbol_getter
        self._db_manager = getattr(gui, "db", None)

        self.pf_min = pf_min
        self.expectancy_min = expectancy_min
        self.metrics_window = metrics_window

        self.atr_ratio_threshold = atr_ratio_threshold
        self.atr_baseline_len = atr_baseline_len

        self.reopt_cooldown_s = reopt_cooldown_s
        self.last_reopt_ts = 0.0

        self.walkforward_interval_s = walkforward_interval_s
        self.walkforward_min_closed_trades = walkforward_min_closed_trades

        self.enable_auto_trade = bool(enable_auto_trade)
        self._auto_trade_user_confirmed = False
        self.auto_trade_interval_s = auto_trade_interval_s

        self._closed_pnls: deque = deque(maxlen=max(10, metrics_window))
        self._atr_values: deque = deque(maxlen=max(50, atr_baseline_len*2))
        self._atr_baseline: Optional[float] = None

        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._lock = threading.RLock()
        self._strategy_config: StrategyConfig = StrategyConfig.presets()["SAFE"].validate()
        self._strategy_override = False
        self._strategy_config_error_notified = False
        self._reduce_only_until: Dict[str, float] = {}
        self._risk_lock_until: float = 0.0
        self._last_risk_audit: Optional[Dict[str, Any]] = None
        self._market_data_provider = market_data_provider
        self._signal_service = signal_service or SignalService()
        self._provided_risk_service = risk_service
        self._emergency_execution_mode = False
        self._emergency_execution_adapter: _NullExchangeAdapter | None = None
        self._compliance_live_allowed = False

        if execution_service is None:
            self._execution_service = self._activate_emergency_execution_mode()
        else:
            self._execution_service = execution_service
        if self._emergency_execution_mode:
            self._live_execution_adapter = None
        else:
            self._live_execution_adapter = getattr(self._execution_service, "_adapter", None)
        self._core_config_path: Path | None = None
        self._risk_profile_name: Optional[str] = None
        self._risk_profile_config: Optional["RiskProfileConfig"] = None
        self._risk_manager_settings: RiskManagerSettings | None = None
        self._risk_watch_interval = 5.0
        self._risk_watch_stop = threading.Event()
        self._risk_watch_thread: threading.Thread | None = None
        self._risk_config_mtime: float | None = None

        try:
            self._core_config_path = resolve_core_config_path()
        except Exception:  # pragma: no cover - środowiska bez pełnego runtime
            logger.debug("Nie udało się ustalić ścieżki konfiguracji core", exc_info=True)
            self._core_config_path = None

        runtime_metadata = load_runtime_entrypoint_metadata(
            "auto_trader",
            config_path=self._core_config_path,
            logger=logger,
        )
        self._runtime_metadata = runtime_metadata.to_dict() if runtime_metadata else {}
        compliance_flag = bool(self._runtime_metadata.get("compliance_live_allowed"))
        self._compliance_live_allowed = compliance_flag and not self._emergency_execution_mode
        if runtime_metadata:
            self._risk_profile_name = getattr(runtime_metadata, "risk_profile", None)
            logger.info("Runtime entrypoint auto_trader: %s", self._runtime_metadata)

        (
            resolved_name,
            profile_config,
            risk_manager_settings,
        ) = load_risk_manager_settings(
            "auto_trader",
            profile_name=self._risk_profile_name,
            config_path=self._core_config_path,
            logger=logger,
        )
        if resolved_name:
            self._risk_profile_name = resolved_name
        self._risk_profile_config = profile_config
        if risk_manager_settings is None:
            risk_manager_settings = RiskManagerSettings(
                max_risk_per_trade=0.02,
                max_daily_loss_pct=0.10,
                max_portfolio_risk=0.10,
                max_positions=5,
                emergency_stop_drawdown=0.15,
            )
        self._risk_manager_settings = risk_manager_settings
        self._risk_repository_dir = self._resolve_risk_repository_dir()
        self._risk_repository = FileRiskRepository(self._risk_repository_dir)
        self._risk_decision_log = RiskDecisionLog(
            max_entries=1_000,
            jsonl_path=self._resolve_risk_decision_log_path(),
        )
        self._uses_custom_risk_service = self._provided_risk_service is not None
        if self._provided_risk_service is None:
            service_kwargs = self._risk_manager_settings.risk_service_kwargs()
            self._risk_service = RiskService(**service_kwargs)
        else:
            self._risk_service = self._provided_risk_service
            service_kwargs = self._risk_manager_settings.risk_service_kwargs()
            for attr, value in service_kwargs.items():
                if hasattr(self._risk_service, attr):
                    try:
                        setattr(self._risk_service, attr, value)
                    except Exception:  # pragma: no cover - defensywne
                        logger.debug("Nie udało się zaktualizować %s w RiskService", attr, exc_info=True)
        self._provided_risk_service = None
        self._risk_manager_adapter = self._build_risk_manager_adapter(
            self._risk_manager_settings,
            profile_name=self._risk_profile_name,
        )
        if self._risk_profile_config is not None:
            applied = self._apply_runtime_risk_budget(self._strategy_config, force=True)
            if applied is not self._strategy_config:
                self._strategy_config = applied
            logger.info(
                "Zastosowano profil ryzyka %s: max_notional=%.4f trade_risk=%.4f max_leverage=%.2f",
                self._risk_profile_name,
                self._strategy_config.max_position_notional_pct,
                self._strategy_config.trade_risk_pct,
                self._strategy_config.max_leverage,
            )
        self._risk_config_mtime = self._get_risk_config_mtime()
        self._data_provider: Optional[DataProvider] = data_provider or self._build_data_provider()
        self._service_mode_enabled = self._data_provider is not None
        self._cooldowns: Dict[str, float] = {}
        self._service_tasks: Dict[Tuple[str, str], asyncio.Task[Any]] = {}
        self._service_loop: Optional[asyncio.AbstractEventLoop] = None
        self._paper_adapter: Optional[PaperTradingAdapter] = None
        self._paper_enabled = False
        self._exchange_config: Optional[Dict[str, Any]] = None
        self._refresh_execution_mode()

        self._bootstrap_context = bootstrap_context
        self._core_risk_engine = None
        self._core_execution_service = None
        self._core_execution_environment = "paper"
        self._core_portfolio_id = "autotrader"
        self._core_risk_profile: RiskProfile | None = None  # type: ignore[assignment]
        self._core_risk_profile_name: Optional[str] = None
        self._core_ai_connector: Optional[AIManagerDecisionConnector] = None
        self._core_ai_notional_by_symbol: Dict[str, float] = {}
        self._core_ai_default_notional: float | None = None
        self._core_account_equity: float = 1_000_000.0
        self._market_intel: Any | None = market_intel or getattr(gui, "market_intel", None)
        self._market_intel_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._market_intel_cache_ts: Dict[Tuple[str, str], float] = {}
        self._market_intel_cache_ttl = 60.0

        if bootstrap_context is not None:
            self._core_risk_engine = core_risk_engine or getattr(
                bootstrap_context, "risk_engine", None
            )
            ctx_profile_obj = getattr(bootstrap_context, "risk_profile", None)
            if RiskProfile is None or isinstance(ctx_profile_obj, RiskProfile):
                self._core_risk_profile = ctx_profile_obj
            ctx_profile_name = getattr(bootstrap_context, "risk_profile_name", None)
            if ctx_profile_name:
                self._core_risk_profile_name = str(ctx_profile_name)
            if self._core_risk_profile is not None:
                derived_name = getattr(self._core_risk_profile, "name", None)
                if derived_name:
                    self._core_risk_profile_name = str(derived_name)
            if (
                self._risk_profile_config is None
                and getattr(bootstrap_context, "risk_profile_config", None) is not None
            ):
                self._risk_profile_config = getattr(
                    bootstrap_context, "risk_profile_config", None
                )
            env = getattr(bootstrap_context, "environment", None)
            if env is not None:
                self._core_portfolio_id = getattr(env, "name", self._core_portfolio_id)
                env_value = getattr(getattr(env, "environment", None), "value", None)
                if env_value:
                    self._core_execution_environment = str(env_value)
            if self._core_risk_profile_name and not self._risk_profile_name:
                self._risk_profile_name = self._core_risk_profile_name
            candidate_execution_service = core_execution_service
            if (
                candidate_execution_service is None
                and CoreExecutionService is not None
                and isinstance(execution_service, CoreExecutionService)
            ):
                candidate_execution_service = execution_service
            if (
                candidate_execution_service is None
                and CoreExecutionService is not None
            ):
                context_service = getattr(bootstrap_context, "execution_service", None)
                if isinstance(context_service, CoreExecutionService):
                    candidate_execution_service = context_service
            self._core_execution_service = candidate_execution_service

            ai_manager_ctx = getattr(bootstrap_context, "ai_manager", None)
            env_ai_cfg = getattr(env, "ai", None) if env is not None else None
            default_strategy = None
            default_action = "enter"
            default_notional = None
            threshold_bps = getattr(bootstrap_context, "ai_threshold_bps", None)
            if env_ai_cfg is not None:
                default_strategy = getattr(env_ai_cfg, "default_strategy", None)
                default_action = getattr(env_ai_cfg, "default_action", default_action)
                default_notional = getattr(env_ai_cfg, "default_notional", None)
            if ai_manager_ctx is not None and AIManagerDecisionConnector is not None and self._core_risk_profile:
                try:
                    connector = ai_connector or AIManagerDecisionConnector(
                        ai_manager=ai_manager_ctx,
                        strategy=str(
                            default_strategy
                            or getattr(env, "default_strategy", "auto_ai_signal")
                        ),
                        risk_profile=self._core_risk_profile,
                        default_notional=float(default_notional or 1_000.0),
                        action=str(default_action or "enter"),
                        threshold_bps=threshold_bps,
                    )
                except Exception:  # pragma: no cover - diagnostyka inicjalizacji
                    logger.exception("Failed to initialise AIManagerDecisionConnector")
                    connector = None
                self._core_ai_connector = connector
                if connector is not None:
                    self._core_ai_default_notional = connector.default_notional
            if getattr(bootstrap_context, "ai_model_bindings", None):
                for binding in getattr(bootstrap_context, "ai_model_bindings", ()):  # type: ignore[attr-defined]
                    notional_value = getattr(binding, "notional", None)
                    symbol_value = getattr(binding, "symbol", None)
                    if notional_value and symbol_value:
                        normalized = self._normalize_symbol(symbol_value)
                        self._core_ai_notional_by_symbol[normalized] = float(notional_value)

        if self._core_risk_engine is not None:
            with self._lock:
                self._ensure_core_risk_profile_registered_locked()

        if self._core_ai_default_notional is None and self._core_ai_connector is not None:
            self._core_ai_default_notional = self._core_ai_connector.default_notional

        with self._lock:
            self._ensure_core_risk_profile_registered_locked()

        self._started = False
        self._auto_trade_thread_active = False

        # Subscribe to events
        emitter.on("trade_closed", self._on_trade_closed, tag="autotrader")
        emitter.on("bar", self._on_bar, tag="autotrader")

    # -- Public API --
    def _activate_emergency_execution_mode(self) -> ExecutionService:
        self._emergency_execution_mode = True
        adapter = _NullExchangeAdapter(self.emitter)
        self._emergency_execution_adapter = adapter
        self._compliance_live_allowed = False
        warning = (
            "ExecutionService not provided – AutoTrader switched to emergency paper-only mode. "
            "Live trading flag disabled."
        )
        try:
            self.emitter.log(warning, level="WARNING", component="AutoTrader")
        except Exception:
            logger.warning(warning)
        else:
            logger.warning(warning)
        return ExecutionService(adapter)

    def _get_emergency_adapter(self) -> _NullExchangeAdapter:
        if self._emergency_execution_adapter is None:
            self._emergency_execution_adapter = _NullExchangeAdapter(self.emitter)
        return self._emergency_execution_adapter

    def _build_data_provider(self) -> Optional[DataProvider]:
        ex_mgr = getattr(self.gui, "ex_mgr", None)
        if ex_mgr is None:
            return None
        try:
            return ExchangeDataProvider(ex_mgr)
        except Exception:  # pragma: no cover - defensywne
            logger.exception("Failed to initialise ExchangeDataProvider")
            return None

    def _apply_runtime_risk_budget(
        self,
        cfg: StrategyConfig,
        *,
        force: bool = False,
    ) -> StrategyConfig:
        if self._risk_profile_config is None:
            return cfg
        if not force and self._strategy_override:
            return cfg
        try:
            return cfg.apply_risk_profile(self._risk_profile_config)
        except Exception:
            logger.debug(
                "Nie udało się zastosować profilu ryzyka %s do konfiguracji strategii",
                self._risk_profile_name,
                exc_info=True,
            )
            return cfg

    def _settings_to_adapter_config(self, settings: RiskManagerSettings) -> Dict[str, Any]:
        payload: Dict[str, Any] = dict(settings.to_dict())
        payload.setdefault("max_risk_per_trade", float(settings.max_risk_per_trade))
        payload.setdefault("max_portfolio_risk", float(settings.max_portfolio_risk))
        payload.setdefault("max_positions", int(settings.max_positions))
        payload.setdefault("max_daily_loss_pct", float(settings.max_daily_loss_pct))
        payload.setdefault("max_drawdown_pct", float(settings.emergency_stop_drawdown))
        payload.setdefault("hard_drawdown_pct", float(settings.emergency_stop_drawdown))
        payload.setdefault("emergency_stop_drawdown", float(settings.emergency_stop_drawdown))
        if settings.confidence_level is not None:
            payload.setdefault("confidence_level", float(settings.confidence_level))
        if settings.target_volatility is not None:
            payload.setdefault("target_volatility", float(settings.target_volatility))
        if settings.profile_name:
            payload.setdefault("risk_profile_name", settings.profile_name)
        return payload

    def _resolve_risk_repository_dir(self) -> Path:
        environment = (
            str(self._runtime_metadata.get("environment"))
            if self._runtime_metadata.get("environment")
            else self._core_execution_environment
        ) or "paper"
        slug = "".join(ch if ch.isalnum() else "_" for ch in environment.lower()) or "paper"
        path = Path("var") / "data" / slug / "risk_state"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve_risk_decision_log_path(self) -> Path:
        environment = (
            str(self._runtime_metadata.get("environment"))
            if self._runtime_metadata.get("environment")
            else self._core_execution_environment
        ) or "paper"
        slug = "".join(ch if ch.isalnum() else "_" for ch in environment.lower()) or "paper"
        return LOGS_DIR / f"auto_trader_risk_decisions_{slug}.jsonl"

    def _build_risk_manager_adapter(
        self,
        settings: RiskManagerSettings,
        *,
        profile_name: str | None,
    ) -> ThresholdRiskManager | None:
        try:
            config = self._settings_to_adapter_config(settings)
            if profile_name:
                config.setdefault("risk_profile_name", profile_name)
            mode = self._resolve_mode()
            return ThresholdRiskManager(
                config=config,
                db_manager=self._db_manager,
                mode=mode,
                profile_name=profile_name,
                decision_log=self._risk_decision_log,
                repository=self._risk_repository,
            )
        except Exception:
            logger.exception("Nie udało się zainicjalizować adaptera ThresholdRiskEngine")
            return None

    def update_risk_manager_settings(
        self,
        settings: RiskManagerSettings,
        *,
        profile_name: str | None = None,
        profile_config: Any | None = None,
    ) -> None:
        if not isinstance(settings, RiskManagerSettings):
            raise TypeError("Oczekiwano instancji RiskManagerSettings")

        with self._lock:
            self._risk_manager_settings = settings
            if profile_name:
                normalized_profile = str(profile_name)
                self._risk_profile_name = normalized_profile
                self._core_risk_profile_name = normalized_profile
                if self._core_risk_profile is not None:
                    current_name = getattr(self._core_risk_profile, "name", None)
                    if current_name and current_name != normalized_profile:
                        self._core_risk_profile = None
            if profile_config is not None:
                self._risk_profile_config = profile_config

            service_kwargs = settings.risk_service_kwargs()
            for attr, value in service_kwargs.items():
                if hasattr(self._risk_service, attr):
                    try:
                        setattr(self._risk_service, attr, value)
                    except Exception:
                        logger.debug(
                            "Nie udało się zaktualizować atrybutu %s w RiskService",
                            attr,
                            exc_info=True,
                        )

            if self._risk_profile_config is not None:
                updated = self._apply_runtime_risk_budget(self._strategy_config, force=True)
                if updated is not self._strategy_config:
                    self._strategy_config = updated
                    logger.info(
                        "Zaktualizowano konfigurację strategii na podstawie profilu %s",
                        self._risk_profile_name,
                    )
            if self._core_risk_engine is not None:
                self._ensure_core_risk_profile_registered_locked()

            self._risk_manager_adapter = self._build_risk_manager_adapter(
                self._risk_manager_settings,
                profile_name=self._risk_profile_name,
            )
            self._ensure_core_risk_profile_registered_locked()

        message = f"Risk profile active: {self._risk_profile_name or 'default'}"
        try:
            self.emitter.log(message, component="AutoTrader")
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.info(message)

    def update_account_equity(self, equity: float) -> None:
        """Synchronizuje kapitał konta wykorzystywany przez silnik ryzyka."""

        try:
            value = float(equity)
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.debug("Nie udało się zaktualizować kapitału konta", exc_info=True)
            return

        if value < 0:
            value = 0.0

        with self._lock:
            self._core_account_equity = value
            try:
                self._runtime_metadata["account_equity"] = value
            except Exception:
                logger.debug("Runtime metadata nie pozwoliło na zapis equity", exc_info=True)

    def _ensure_core_risk_profile_registered_locked(self) -> None:
        """Gwarantuje, że aktywny profil został zarejestrowany w silniku core."""

        if ThresholdRiskEngine is None or self._core_risk_engine is None:
            return

        risk_engine = self._core_risk_engine
        profile_name = self._core_risk_profile or self._risk_profile_name or "default"

        config_obj = self._risk_profile_config
        settings = self._risk_manager_settings

        profile = None
        if build_risk_profile_from_config is not None and config_obj is not None:
            candidate = config_obj
            if RiskProfileConfig is not None and not isinstance(candidate, RiskProfileConfig):
                if isinstance(candidate, Mapping):
                    try:
                        candidate = RiskProfileConfig(**dict(candidate))
                    except Exception:
                        candidate = SimpleNamespace(**dict(candidate))
            elif isinstance(candidate, Mapping):
                candidate = SimpleNamespace(**dict(candidate))
            try:
                profile = build_risk_profile_from_config(candidate)  # type: ignore[arg-type]
            except Exception:
                logger.debug(
                    "Nie udało się zbudować profilu ryzyka z konfiguracji bootstrapu",
                    exc_info=True,
                )

        if profile is None and ManualProfile is not None:
            def _read_float(name: str, default: float) -> float:
                sources: tuple[Any, ...] = (
                    config_obj,
                    settings,
                )
                for source in sources:
                    if source is None:
                        continue
                    value: Any
                    if isinstance(source, Mapping):
                        if name in source and source[name] is not None:
                            value = source[name]
                        else:
                            continue
                    elif hasattr(source, name):
                        value = getattr(source, name)
                    else:
                        continue
                    try:
                        return float(value)
                    except Exception:
                        continue
                return default

            def _read_int(name: str, default: int) -> int:
                sources: tuple[Any, ...] = (
                    config_obj,
                    settings,
                )
                for source in sources:
                    if source is None:
                        continue
                    value: Any
                    if isinstance(source, Mapping):
                        if name in source and source[name] is not None:
                            value = source[name]
                        else:
                            continue
                    elif hasattr(source, name):
                        value = getattr(source, name)
                    else:
                        continue
                    try:
                        return int(value)
                    except Exception:
                        continue
                return default

            max_positions = _read_int("max_open_positions", int(getattr(settings, "max_positions", 5) or 5))
            max_leverage = _read_float("max_leverage", 1.0)
            drawdown = _read_float(
                "hard_drawdown_pct",
                float(getattr(settings, "emergency_stop_drawdown", 0.2) or 0.2),
            )
            daily_loss = _read_float(
                "max_daily_loss_pct",
                float(getattr(settings, "max_daily_loss_pct", 0.05) or 0.05),
            )
            position_pct = _read_float(
                "max_position_pct",
                float(getattr(settings, "max_risk_per_trade", 0.05) or 0.05),
            )
            target_volatility = _read_float(
                "target_volatility",
                float(getattr(settings, "target_volatility", 0.0) or 0.0),
            )
            stop_loss_atr = _read_float("stop_loss_atr_multiple", 1.0)

            try:
                profile = ManualProfile(
                    name=str(profile_name),
                    max_positions=max_positions,
                    max_leverage=max_leverage,
                    drawdown_limit=drawdown,
                    daily_loss_limit=daily_loss,
                    max_position_pct=position_pct,
                    target_volatility=target_volatility,
                    stop_loss_atr_multiple=stop_loss_atr,
                )
            except Exception:
                logger.debug("Nie udało się zbudować profilu ManualProfile", exc_info=True)
                profile = None

        if profile is None:
            return

        try:
            if getattr(profile, "name", None) != profile_name:
                try:
                    setattr(profile, "name", str(profile_name))
                except Exception:
                    logger.debug("Nie udało się ustawić nazwy profilu ryzyka", exc_info=True)
            risk_engine.register_profile(profile)
            self._core_risk_profile = getattr(profile, "name", str(profile_name))
        except Exception:
            logger.exception("Rejestracja profilu ryzyka w ThresholdRiskEngine nie powiodła się")

    def reload_risk_manager_settings(
        self,
        *,
        profile_name: str | None = None,
        config_path: Path | None = None,
    ) -> tuple[str | None, RiskManagerSettings, Any | None]:
        """Ponownie wczytuje ustawienia profilu ryzyka z runtime metadata."""

        candidate = profile_name or self._risk_profile_name
        cfg_path = config_path or self._core_config_path
        try:
            resolved_name, profile_cfg, settings = load_risk_manager_settings(
                "auto_trader",
                profile_name=candidate,
                config_path=cfg_path,
                logger=logger,
            )
        except Exception:
            logger.exception("Nie udało się przeładować profilu ryzyka AutoTradera")
            if self._risk_manager_settings is None:
                raise
            return self._risk_profile_name, self._risk_manager_settings, self._risk_profile_config

        final_name = resolved_name or candidate
        self.update_risk_manager_settings(
            settings,
            profile_name=final_name,
            profile_config=profile_cfg,
        )
        self._risk_config_mtime = self._get_risk_config_mtime()
        return final_name, settings, profile_cfg

    def _ensure_core_risk_profile_registered_locked(self) -> bool:
        """Gwarantuje, że rdzeniowy silnik ryzyka ma zarejestrowany aktywny profil."""

        risk_engine = self._core_risk_engine
        if risk_engine is None:
            return False
        register = getattr(risk_engine, "register_profile", None)
        if not callable(register):
            return False

        profile_obj: Any | None = self._core_risk_profile
        profile_name = self._core_risk_profile_name or self._risk_profile_name
        if profile_obj is None:
            config = self._risk_profile_config
            if config is not None and build_risk_profile_from_config is not None:
                try:
                    profile_obj = build_risk_profile_from_config(config)  # type: ignore[arg-type]
                except TypeError:
                    if isinstance(config, Mapping):
                        try:
                            from bot_core.config.models import RiskProfileConfig  # type: ignore

                            profile_obj = build_risk_profile_from_config(  # type: ignore[arg-type]
                                RiskProfileConfig(**dict(config))
                            )
                        except Exception:
                            logger.debug(
                                "Nie udało się zbudować profilu ryzyka z mapy konfiguracji",
                                exc_info=True,
                            )
                    else:
                        logger.debug(
                            "Nieobsługiwany typ konfiguracji profilu ryzyka: %s",
                            type(config),
                        )
                except Exception:
                    logger.debug(
                        "Nie udało się zbudować profilu ryzyka z konfiguracji",
                        exc_info=True,
                    )

        if profile_obj is None:
            return False

        derived_name = getattr(profile_obj, "name", None)
        if derived_name:
            profile_name = str(derived_name)
        if profile_name:
            profile_name = str(profile_name)

        try:
            existing = tuple(risk_engine.profile_names())  # type: ignore[attr-defined]
        except Exception:
            existing = ()

        if profile_name and profile_name in existing:
            self._core_risk_profile = profile_obj if RiskProfile is None or isinstance(profile_obj, RiskProfile) else None
            self._core_risk_profile_name = profile_name
            if not self._risk_profile_name:
                self._risk_profile_name = profile_name
            return True

        try:
            register(profile_obj)
        except Exception:
            logger.debug(
                "Nie udało się zarejestrować profilu ryzyka %s w silniku core",
                profile_name or "<unknown>",
                exc_info=True,
            )
            return False

        if RiskProfile is None or isinstance(profile_obj, RiskProfile):
            self._core_risk_profile = profile_obj
        else:
            self._core_risk_profile = None
        if profile_name:
            self._core_risk_profile_name = profile_name
            self._risk_profile_name = profile_name
        return True

    def _get_risk_config_mtime(self) -> float | None:
        if not self._core_config_path:
            return None
        try:
            return Path(self._core_config_path).stat().st_mtime
        except FileNotFoundError:
            return None
        except Exception:  # pragma: no cover - defensywne logowanie
            logger.debug("Nie udało się pobrać mtime konfiguracji core", exc_info=True)
            return None

    def _start_risk_watcher(self) -> None:
        if self._core_config_path is None:
            return
        if self._risk_watch_thread and self._risk_watch_thread.is_alive():
            return

        self._risk_watch_stop.clear()

        def _loop() -> None:
            while not self._risk_watch_stop.wait(self._risk_watch_interval):
                try:
                    self._check_risk_config_change()
                except Exception:  # pragma: no cover - defensywne
                    logger.exception("Watcher profilu ryzyka AutoTradera zgłosił wyjątek")

        thread = threading.Thread(target=_loop, name="autotrader-risk-watch", daemon=True)
        thread.start()
        self._risk_watch_thread = thread

    def _stop_risk_watcher(self) -> None:
        self._risk_watch_stop.set()
        thread = self._risk_watch_thread
        if thread and thread.is_alive():
            thread.join(timeout=1.5)
        self._risk_watch_thread = None
        self._risk_watch_stop = threading.Event()

    def _check_risk_config_change(self) -> bool:
        new_mtime = self._get_risk_config_mtime()
        if new_mtime is None:
            self._risk_config_mtime = None
            return False
        if self._risk_config_mtime is None:
            self._risk_config_mtime = new_mtime
            return False
        if new_mtime <= self._risk_config_mtime:
            return False
        self._risk_config_mtime = new_mtime
        try:
            self.reload_risk_manager_settings()
        except Exception:  # pragma: no cover - diagnostyka runtime
            logger.exception("Automatyczne przeładowanie profilu ryzyka nie powiodło się")
            return False
        return True

    def start(self) -> None:
        self._stop.clear()
        self._started = True
        self._threads = []
        # Walk-forward loop only if configured
        if self.walkforward_interval_s:
            t = threading.Thread(target=self._walkforward_loop, daemon=True)
            t.start()
            self._threads.append(t)
        if self.enable_auto_trade and self._auto_trade_user_confirmed:
            self._start_auto_trade_thread()
        elif self.enable_auto_trade:
            message = (
                "Auto-trade awaiting explicit activation. Użyj panelu sterowania, "
                "aby włączyć handel automatyczny."
            )
            try:
                self.emitter.log(message, level="WARNING", component="AutoTrader")
            except Exception:
                logger.warning(message)
        self._start_risk_watcher()
        self.emitter.log("AutoTrader started.", component="AutoTrader")
        logger.info("AutoTrader worker threads started")

    def _start_auto_trade_thread(self) -> None:
        if self._auto_trade_thread_active:
            return
        t2 = threading.Thread(target=self._auto_trade_loop, daemon=True)
        t2.start()
        self._threads.append(t2)
        self._auto_trade_thread_active = True

    def stop(self) -> None:
        self._stop.set()
        self.emitter.off("trade_closed", tag="autotrader")
        self.emitter.off("bar", tag="autotrader")
        self._stop_risk_watcher()
        self.emitter.log("AutoTrader stopped.", component="AutoTrader")
        logger.info("AutoTrader stop requested")
        for t in list(self._threads):
            try:
                if t.is_alive():
                    t.join(timeout=2.0)
            except Exception:
                logger.exception("Error while joining AutoTrader thread")
        self._threads.clear()
        self._auto_trade_thread_active = False
        self._auto_trade_user_confirmed = False
        self._started = False

    def set_enable_auto_trade(self, flag: bool) -> None:
        with self._lock:
            self.enable_auto_trade = bool(flag)
            if not self.enable_auto_trade:
                self._auto_trade_user_confirmed = False
            if (
                self.enable_auto_trade
                and self._auto_trade_user_confirmed
                and self._started
                and not self._auto_trade_thread_active
            ):
                self._start_auto_trade_thread()
        self.emitter.log(f"Auto-Trade {'ENABLED' if flag else 'DISABLED'}.", component="AutoTrader")

    def confirm_auto_trade(self, confirmed: bool = True) -> None:
        """Ustawia ręczne potwierdzenie niezbędne do startu auto-trade."""

        with self._lock:
            self._auto_trade_user_confirmed = bool(confirmed)
            status = "CONFIRMED" if self._auto_trade_user_confirmed else "REVOKED"
            should_start = (
                self.enable_auto_trade
                and self._auto_trade_user_confirmed
                and self._started
                and not self._auto_trade_thread_active
            )
        self.emitter.log(
            f"Auto-Trade confirmation {status}.",
            component="AutoTrader",
        )
        if should_start:
            self._start_auto_trade_thread()

    def configure(self, **kwargs: Any) -> None:
        """Runtime reconfiguration from the ControlPanel."""
        with self._lock:
            for key, val in kwargs.items():
                if key == "strategy":
                    self._update_strategy_config(val)
                    continue
                if key == "exchange":
                    if isinstance(val, Mapping):
                        exchange_payload = dict(val)
                        if "adapter" in exchange_payload:
                            self._live_execution_adapter = exchange_payload["adapter"]
                        self._exchange_config = exchange_payload
                    else:
                        self._exchange_config = None
                    if not self._paper_enabled:
                        adapter = self._live_execution_adapter or self._get_emergency_adapter()
                        self._execution_service.set_adapter(adapter)
                    continue
                if key == "enable_auto_trade":
                    self.set_enable_auto_trade(bool(val))
                    continue
                if key == "confirm_auto_trade":
                    self.confirm_auto_trade(bool(val))
                    continue
                if not hasattr(self, key):
                    continue
                setattr(self, key, val)
        self._refresh_execution_mode()
        self.emitter.log(f"AutoTrader reconfigured: {kwargs}", component="AutoTrader")

    # -- Event handlers --
    def _on_trade_closed(
        self,
        symbol: str,
        side: str,
        entry: float,
        exit: float,
        pnl: float,
        ts: float,
        meta: Dict[str, Any] | None = None,
        **_,
    ) -> None:
        self._closed_pnls.append(pnl)
        try:
            prometheus_metrics.record_trade_close(symbol, float(pnl))
        except Exception:
            logger.debug("Prometheus record_trade_close skipped", exc_info=True)
        pf, exp, win_rate = self._compute_metrics()
        self.emitter.emit(
            "metrics_updated",
            pf=pf,
            expectancy=exp,
            win_rate=win_rate,
            window=len(self._closed_pnls),
            ts=time.time(),
        )
        self._persist_performance_metrics(symbol, pf, exp, win_rate)

        # Check thresholds
        trigger_reason = None
        details: Dict[str, Any] = {}
        if pf is not None and pf < self.pf_min:
            trigger_reason = "pf_drop"
            details["pf"] = pf
        if exp is not None and exp < self.expectancy_min:
            trigger_reason = (trigger_reason + "+expectancy_drop") if trigger_reason else "expectancy_drop"
            details["expectancy"] = exp
        if trigger_reason:
            self._maybe_reoptimize(trigger_reason, details)

    def _on_bar(self, symbol: str, o: float, h: float, l: float, c: float, ts: float, **_) -> None:
        # TR approximation (if no prev close given we use current bar-only proxy)
        tr = max(h - l, abs(h - c), abs(l - c))
        self._atr_values.append(tr)
        # Compute ATR using simple moving average of TRs
        if len(self._atr_values) >= max(14, self.atr_baseline_len):
            atr = sum(list(self._atr_values)[-14:]) / 14.0
            if self._atr_baseline is None and len(self._atr_values) >= self.atr_baseline_len:
                self._atr_baseline = sum(list(self._atr_values)[:self.atr_baseline_len]) / float(self.atr_baseline_len)
            baseline = self._atr_baseline or atr
            ratio = (atr - baseline) / baseline if baseline > 0 else 0.0
            self.emitter.emit("atr_updated", atr=atr, baseline=baseline, ratio=ratio, ts=ts)
            if self._atr_baseline and ratio >= self.atr_ratio_threshold:
                self._maybe_reoptimize("atr_spike", {"atr": atr, "baseline": baseline, "ratio": ratio})

    # -- Helpers --
    def _compute_metrics(self) -> tuple[Optional[float], Optional[float], Optional[float]]:
        if not self._closed_pnls:
            return (None, None, None)
        pnls = list(self._closed_pnls)
        wins = [p for p in pnls if p > 0]
        losses = [-p for p in pnls if p < 0]
        gross_profit = sum(wins)
        gross_loss = sum(losses)
        pf = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else None)
        # Expectancy per trade (avg pnl)
        expectancy = statistics.mean(pnls) if pnls else None
        win_rate = (len(wins) / len(pnls)) if pnls else None
        return (pf, expectancy, win_rate)

    def _resolve_db(self):
        db = self._db_manager
        if db is None:
            candidate = getattr(self.gui, "db", None)
            if candidate is not None:
                db = candidate
                self._db_manager = candidate
        if db is None or not hasattr(db, "sync"):
            return None
        return db

    def _resolve_mode(self) -> str:
        network_var = getattr(self.gui, "network_var", None)
        if network_var is not None and hasattr(network_var, "get"):
            try:
                network = str(network_var.get()).strip().lower()
            except Exception:
                network = ""
            if network in {"testnet", "paper", "demo"}:
                return "paper"
        return "live"

    def _log_metric(
        self,
        metric: str,
        value: Optional[float],
        *,
        symbol: str,
        window: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if value is None:
            return
        db = self._resolve_db()
        if db is None:
            return
        context: Dict[str, Any] = {"symbol": symbol, "window": window, "source": "AutoTrader"}
        if extra:
            context.update(extra)
        payload = {
            "metric": metric,
            "value": float(value),
            "window": window,
            "symbol": symbol,
            "mode": self._resolve_mode(),
            "context": context,
        }
        try:
            db.sync.log_performance_metric(payload)
        except Exception:
            logger.exception("Nie udało się zapisać metryki %s", metric)

    def _persist_performance_metrics(
        self,
        symbol: str,
        pf: Optional[float],
        expectancy: Optional[float],
        win_rate: Optional[float],
    ) -> None:
        window = len(self._closed_pnls)
        extra = {
            "profit_factor": pf,
            "expectancy": expectancy,
            "win_rate": win_rate,
        }
        self._log_metric("auto_trader_expectancy", expectancy, symbol=symbol, window=window, extra=extra)
        self._log_metric("auto_trader_profit_factor", pf, symbol=symbol, window=window, extra=extra)
        self._log_metric("auto_trader_win_rate", win_rate, symbol=symbol, window=window, extra=extra)

    def _maybe_reoptimize(self, reason: str, details: Dict[str, Any]) -> None:
        now = time.time()
        if now - self.last_reopt_ts < self.reopt_cooldown_s:
            self.emitter.log(f"Reopt skipped (cooldown): {reason} {details}", level="DEBUG", component="AutoTrader")
            return
        self.last_reopt_ts = now
        self.emitter.emit("reopt_triggered", reason=reason, details=details, ts=now)
        # Try to call AI retrain if available
        ai = getattr(self.gui, "ai_mgr", None)
        if ai is not None and hasattr(ai, "train"):
            try:
                # non-blocking retrain in thread
                threading.Thread(target=self._call_train_safe, args=(ai,), daemon=True).start()
            except Exception as e:
                self.emitter.log(f"AI retrain failed to start: {e!r}", level="ERROR", component="AutoTrader")
                logger.exception("AI retrain thread failed to start")
        else:
            self.emitter.log("AI manager not available; reopt event emitted only.", level="WARNING", component="AutoTrader")

    def _call_train_safe(self, ai) -> None:
        try:
            self.emitter.log("AI retrain started...", component="AutoTrader")
            ai.train()
            self.emitter.log("AI retrain finished.", component="AutoTrader")
        except Exception as e:
            self.emitter.log(f"AI retrain error: {e!r}", level="ERROR", component="AutoTrader")
            logger.exception("AI retrain raised an exception")

    # -- Loops --
    def _walkforward_loop(self) -> None:
        last_ts = 0.0
        while not self._stop.is_set():
            try:
                now = time.time()
                if last_ts == 0.0:
                    last_ts = now
                wf = self.walkforward_interval_s or 0
                if wf > 0 and (now - last_ts) >= wf:
                    # Optional guard: only if we have enough closed trades since last step
                    if len(self._closed_pnls) >= self.walkforward_min_closed_trades:
                        self._maybe_reoptimize("walk_forward_tick", {"closed_trades": len(self._closed_pnls)})
                    last_ts = now
            except Exception as e:
                self.emitter.log(f"Walk-forward loop error: {e!r}", level="ERROR", component="AutoTrader")
                logger.exception("Walk-forward loop error")
            self._stop.wait(1.0)

    def _run_service_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._service_loop = loop
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._service_loop_main())
        except Exception:  # pragma: no cover - defensywny log
            logger.exception("Service-based auto trade loop crashed")
        finally:
            try:
                pending = [task for task in asyncio.all_tasks(loop=loop) if not task.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                logger.exception("Failed to shutdown auto trade service loop")
            finally:
                asyncio.set_event_loop(None)
                loop.close()
                self._service_loop = None

    async def _service_loop_main(self) -> None:
        try:
            while not self._stop.is_set():
                schedule = self._resolve_schedule_entries()
                await self._ensure_service_schedule(schedule)
                await asyncio.sleep(0.5)
        finally:
            await self._cancel_service_tasks()

    async def _ensure_service_schedule(self, schedule: List[Tuple[str, str]]) -> None:
        desired = {(sym, tf) for sym, tf in schedule if sym}
        for key in list(self._service_tasks):
            if key not in desired:
                task = self._service_tasks.pop(key)
                task.cancel()
                with exception_guard("AutoTrader.scheduler"):
                    await asyncio.gather(task, return_exceptions=True)
        for entry in schedule:
            if entry not in self._service_tasks:
                symbol, timeframe = entry
                if not symbol:
                    continue
                self._service_tasks[entry] = self._create_service_task(symbol, timeframe)

    def _create_service_task(self, symbol: str, timeframe: str) -> asyncio.Task[Any]:
        task = asyncio.create_task(self._symbol_service_loop(symbol, timeframe))
        entry = (symbol, timeframe)

        def _handle_completion(completed: asyncio.Task[Any]) -> None:
            self._service_tasks.pop(entry, None)
            if completed.cancelled():
                return
            try:
                exc = completed.exception()
            except asyncio.CancelledError:
                return
            except Exception:  # pragma: no cover - defensywne logowanie callbacku
                logger.exception(
                    "Failed to inspect service task exception for %s@%s",
                    symbol,
                    timeframe,
                )
                return
            if exc is None:
                return

            message = (
                f"Service task for {symbol}@{timeframe} crashed: {exc!r}"
            )
            try:
                self.emitter.log(message, level="ERROR", component="AutoTrader")
            except Exception:  # pragma: no cover - nie przerywamy callbacku
                logger.exception("Failed to forward service task crash to emitter")

            logger.error(
                "Service task for %s@%s crashed",
                symbol,
                timeframe,
                exc_info=exc,
            )

            try:
                _emit_alert(
                    message,
                    severity=AlertSeverity.ERROR,
                    source="autotrader",
                    context={
                        "symbol": symbol,
                        "timeframe": timeframe,
                    },
                    exception=exc if isinstance(exc, BaseException) else None,
                )
            except Exception:  # pragma: no cover - nie przerywamy callbacku
                logger.exception("Failed to emit alert for service task crash")

            try:
                self._register_cooldown(symbol, "service_task_crash")
            except Exception:  # pragma: no cover - nie przerywamy callbacku
                logger.exception("Failed to register cooldown after service task crash")

        task.add_done_callback(_handle_completion)
        return task

    async def _cancel_service_tasks(self) -> None:
        if not self._service_tasks:
            return
        tasks = list(self._service_tasks.values())
        for task in tasks:
            task.cancel()
        with exception_guard("AutoTrader.scheduler"):
            await asyncio.gather(*tasks, return_exceptions=True)
        self._service_tasks.clear()

    async def _symbol_service_loop(self, symbol: str, timeframe: str) -> None:
        interval = max(0.1, float(self.auto_trade_interval_s))
        while not self._stop.is_set():
            if not self.enable_auto_trade or not self._auto_trade_user_confirmed:
                await asyncio.sleep(interval)
                continue
            if self._is_symbol_on_cooldown(symbol):
                await asyncio.sleep(interval)
                continue
            await self._trade_once(symbol, timeframe)
            await asyncio.sleep(interval)

    def _resolve_schedule_entries(self) -> List[Tuple[str, str]]:
        timeframe = self._resolve_timeframe()
        entries: List[Tuple[str, str]] = []
        try:
            raw = self.symbol_getter()
        except Exception:
            logger.exception("Symbol getter failed")
            return entries

        if isinstance(raw, str):
            symbol = raw.strip()
            if symbol:
                entries.append((symbol, timeframe))
            return entries
        if isinstance(raw, Mapping):
            for sym, tf in raw.items():
                symbol = str(sym).strip()
                tf_value = str(tf).strip() or timeframe
                if symbol:
                    entries.append((symbol, tf_value))
            return entries
        if isinstance(raw, Iterable):
            for item in raw:
                if isinstance(item, tuple) and len(item) >= 2:
                    symbol = str(item[0]).strip()
                    tf_value = str(item[1]).strip() or timeframe
                else:
                    symbol = str(item).strip()
                    tf_value = timeframe
                if symbol:
                    entries.append((symbol, tf_value))
            return entries
        if raw:
            symbol = str(raw).strip()
            if symbol:
                entries.append((symbol, timeframe))
        return entries

    def _resolve_timeframe(self) -> str:
        timeframe = "1m"
        tf_var = getattr(self.gui, "timeframe_var", None)
        if tf_var is not None and hasattr(tf_var, "get"):
            try:
                value = tf_var.get()
            except Exception:
                value = None
            if value:
                timeframe = str(value)
        return timeframe

    def _is_symbol_on_cooldown(self, symbol: str) -> bool:
        with self._lock:
            until = self._cooldowns.get(symbol, 0.0)
            if not until:
                return False
            now = time.time()
            if now >= until:
                self._cooldowns.pop(symbol, None)
                return False
            return True

    def _register_cooldown(self, symbol: str, reason: str, duration: Optional[float] = None) -> None:
        cfg = self._get_strategy_config()
        cooldown = float(duration) if duration is not None else max(float(cfg.violation_cooldown_s), float(self.auto_trade_interval_s))
        until = time.time() + cooldown
        with self._lock:
            self._cooldowns[symbol] = until
        self.emitter.log(
            f"Cooldown applied for {symbol}: {reason} (until {until:.0f})",
            level="WARNING",
            component="AutoTrader",
        )

    def _build_risk_portfolio_payload(
        self,
        symbol: str,
        snapshot: Mapping[str, Any],
        *,
        price: float,
    ) -> Dict[str, Any]:
        capital = float(
            snapshot.get("value")
            or snapshot.get("portfolio_value")
            or snapshot.get("equity")
            or snapshot.get("capital")
            or 0.0
        )
        positions_payload: Dict[str, Dict[str, Any]] = {}
        raw_positions = snapshot.get("positions") or snapshot.get("open_positions")
        if isinstance(raw_positions, Mapping):
            for key, value in raw_positions.items():
                if not isinstance(value, Mapping):
                    continue
                qty_raw = value.get("qty") or value.get("quantity") or value.get("size")
                try:
                    qty_value = float(qty_raw)
                except Exception:
                    continue
                if qty_value == 0:
                    continue
                entry_price_raw = value.get("entry") or value.get("entry_price") or value.get("price") or price
                try:
                    entry_price = float(entry_price_raw)
                except Exception:
                    entry_price = price
                side_raw = value.get("side") or value.get("direction")
                side = (
                    str(side_raw).lower()
                    if isinstance(side_raw, str)
                    else ("buy" if qty_value >= 0 else "sell")
                )
                positions_payload[str(key).upper()] = {
                    "symbol": str(key),
                    "qty": abs(qty_value),
                    "side": side,
                    "entry_price": entry_price,
                }
        else:
            qty_raw = snapshot.get("position") or snapshot.get("qty")
            try:
                qty_value = float(qty_raw) if qty_raw is not None else 0.0
            except Exception:
                qty_value = 0.0
            if qty_value:
                entry_price_raw = (
                    snapshot.get("entry")
                    or snapshot.get("entry_price")
                    or snapshot.get("price")
                    or price
                )
                try:
                    entry_price = float(entry_price_raw)
                except Exception:
                    entry_price = price
                side = "buy" if qty_value >= 0 else "sell"
                positions_payload[symbol.upper()] = {
                    "symbol": symbol,
                    "qty": abs(qty_value),
                    "side": side,
                    "entry_price": entry_price,
                }

        return {
            "capital": capital,
            "equity": capital,
            "positions": positions_payload,
        }

    def _build_risk_signal_payload(
        self,
        signal: StrategySignal,
        *,
        portfolio_value: float,
    ) -> Dict[str, Any]:
        payload = dict(getattr(signal, "payload", {}) or {})
        payload.setdefault("symbol", signal.symbol)
        try:
            confidence = float(signal.confidence)
        except Exception:
            confidence = 0.0
        payload.setdefault("confidence", confidence)
        payload.setdefault("strength", confidence)
        direction = str(signal.action or "").upper()
        if direction == "SELL":
            payload.setdefault("direction", "SHORT")
        else:
            payload.setdefault("direction", "LONG")
        if signal.size is not None and portfolio_value > 0:
            try:
                fraction_hint = float(signal.size) / portfolio_value
            except Exception:
                fraction_hint = None
            if fraction_hint is not None and fraction_hint > 0:
                payload.setdefault("target_fraction", max(0.0, fraction_hint))
        return payload

    def _assess_with_legacy_risk_service(
        self,
        signal: StrategySignal,
        context: StrategyContext,
        market_state: Mapping[str, Any],
        portfolio_value: float,
        cfg: StrategyConfig,
    ) -> Tuple[RiskDecision, float | None, float | None, float | None]:
        assessment = self._risk_service.assess(signal, context, market_state)
        base_value = portfolio_value if portfolio_value > 0 else 1.0
        try:
            size_value = float(assessment.size) if assessment.size is not None else None
        except Exception:
            size_value = None
        fraction = max(0.0, (size_value or 0.0) / base_value)
        details: Dict[str, Any] = {
            "market_state": dict(market_state),
            "source": "legacy_risk_service",
        }
        decision = RiskDecision(
            should_trade=bool(assessment.allow),
            fraction=fraction,
            state="ok" if assessment.allow else "reject",
            reason=assessment.reason,
            details=details,
            stop_loss_pct=assessment.stop_loss,
            take_profit_pct=assessment.take_profit,
            mode=cfg.mode,
        )
        return decision, size_value, assessment.stop_loss, assessment.take_profit

    def _assess_with_threshold_engine(
        self,
        symbol: str,
        signal: StrategySignal,
        context: StrategyContext,
        market_payload: Mapping[str, Any],
        portfolio_snapshot: Mapping[str, Any],
        market_state: Mapping[str, Any],
        cfg: StrategyConfig,
    ) -> Tuple[RiskDecision, float | None, None, None]:
        manager = getattr(self, "_risk_manager_adapter", None)
        portfolio_value = float(market_state.get("portfolio_value") or context.portfolio_value or 0.0)
        if manager is None:
            return self._assess_with_legacy_risk_service(
                signal,
                context,
                market_state,
                portfolio_value,
                cfg,
            )

        raw_price = (
            market_payload.get("price")
            or market_payload.get("close")
            or market_state.get("price")
        )
        try:
            price = float(raw_price)
        except Exception:
            price = 0.0

        if price <= 0 or portfolio_value <= 0:
            details = {
                "market_state": dict(market_state),
                "source": "threshold_risk_engine",
                "error": "missing_price_or_capital",
            }
            reason = "Brak danych o kapitale lub cenie"
            decision = RiskDecision(
                should_trade=False,
                fraction=0.0,
                state="reject",
                reason=reason,
                details=details,
                stop_loss_pct=signal.stop_loss,
                take_profit_pct=signal.take_profit,
                mode=cfg.mode,
            )
            return decision, None, None, None

        portfolio_payload = self._build_risk_portfolio_payload(symbol, portfolio_snapshot, price=price)
        signal_payload = self._build_risk_signal_payload(signal, portfolio_value=portfolio_value)
        try:
            fraction, details = manager.calculate_position_size(
                symbol,
                signal_payload,
                market_payload,
                portfolio_payload,
                return_details=True,
            )
        except Exception:
            logger.exception("Risk manager sizing failed")
            details = {
                "market_state": dict(market_state),
                "source": "threshold_risk_engine",
                "error": "sizing_exception",
            }
            decision = RiskDecision(
                should_trade=False,
                fraction=0.0,
                state="reject",
                reason="Risk manager error",
                details=details,
                stop_loss_pct=signal.stop_loss,
                take_profit_pct=signal.take_profit,
                mode=cfg.mode,
            )
            return decision, None, None, None

        recommended_fraction = max(0.0, float(details.get("recommended_size", fraction) or 0.0))
        allowed = bool(details.get("allowed", recommended_fraction > 0))
        reason = details.get("reason")
        details_payload: Dict[str, Any] = {
            "market_state": dict(market_state),
            "source": "threshold_risk_engine",
        }
        for key, value in details.items():
            if key == "recommended_size":
                continue
            details_payload[key] = value

        quantity = recommended_fraction * portfolio_value / price if price > 0 else 0.0
        if quantity <= 0:
            allowed = False
            if not reason:
                reason = "Brak dostępnego budżetu ryzyka"

        state = "ok" if allowed else "reject"
        if allowed and details_payload.get("adjustments"):
            state = "adjusted"

        decision = RiskDecision(
            should_trade=allowed,
            fraction=recommended_fraction if allowed else 0.0,
            state=state,
            reason=reason,
            details=details_payload,
            stop_loss_pct=signal.stop_loss,
            take_profit_pct=signal.take_profit,
            mode=cfg.mode,
        )
        return decision, (quantity if allowed else None), None, None

    async def _trade_once(self, symbol: str, timeframe: str) -> None:
        if self._data_provider is None:
            return
        with exception_guard("AutoTrader.trade"):
            cfg = self._get_strategy_config()
            strategy_name = cfg.preset or "SAFE"
            metadata = self._resolve_strategy_metadata(strategy_name)
            market_payload = await self._build_market_payload(symbol, timeframe)
            if not market_payload:
                return
            self._execution_service.update_market_data(symbol, timeframe, market_payload)
            price = float(market_payload.get("price") or 0.0)
            portfolio_snapshot = self._resolve_portfolio_snapshot(symbol, price)
            portfolio_value = float(
                portfolio_snapshot.get("value")
                or portfolio_snapshot.get("portfolio_value")
                or portfolio_snapshot.get("equity")
                or 0.0
            )
            position = float(portfolio_snapshot.get("position") or portfolio_snapshot.get("qty") or 0.0)
            compliance_allowed = self._compliance_live_allowed and bool(
                cfg.compliance_confirmed
                and cfg.api_keys_configured
                and cfg.acknowledged_risk_disclaimer
            )
            context = self._signal_service.build_context(
                symbol=symbol,
                timeframe=timeframe,
                portfolio_value=portfolio_value,
                position=position,
                metadata=metadata,
                mode=cfg.mode,
                compliance_live_allowed=compliance_allowed,
            )
            signal = await self._signal_service.run_strategy(
                strategy_name,
                context,
                market_payload,
                self._data_provider,
            )
            if signal is None:
                return
            market_state = self._build_market_state(portfolio_snapshot, market_payload)
            if self._uses_custom_risk_service:
                decision, size_override, sl_override, tp_override = self._assess_with_legacy_risk_service(
                    signal,
                    context,
                    market_state,
                    portfolio_value,
                    cfg,
                )
            else:
                (
                    decision,
                    size_override,
                    sl_override,
                    tp_override,
                ) = self._assess_with_threshold_engine(
                    symbol,
                    signal,
                    context,
                    market_payload,
                    portfolio_snapshot,
                    market_state,
                    cfg,
                )

            self._emit_risk_audit(
                symbol,
                signal.action or "HOLD",
                decision,
                float(market_state.get("price") or price or 0.0),
            )
            if not decision.should_trade:
                self._register_cooldown(symbol, decision.reason or "risk_rejected")
                return

            if size_override is not None:
                signal.size = size_override

            try:
                size_numeric = float(signal.size) if signal.size is not None else 0.0
            except Exception:
                size_numeric = 0.0
            if size_numeric <= 0:
                self._register_cooldown(symbol, decision.reason or "risk_zero_size")
                return

            signal.size = size_numeric

            if sl_override is not None and signal.stop_loss is None:
                signal.stop_loss = sl_override
            if tp_override is not None and signal.take_profit is None:
                signal.take_profit = tp_override
            signal.payload.setdefault("market_state", market_state)
            signal.payload.setdefault("price", market_state.get("price"))

            context.require_demo_mode()
            result = await self._execution_service.execute(signal, context)
            if result is not None:
                self._sync_paper_gui_state(
                    symbol,
                    signal.action or "",
                    float(market_state.get("price") or price or 0.0),
                    signal.size,
                )
                try:
                    prometheus_metrics.record_order(symbol, signal.action, float(signal.size or 0.0))
                except Exception:
                    logger.debug("Prometheus record_order skipped", exc_info=True)
                self.emitter.emit("auto_trade_tick", symbol=symbol, ts=time.time())
                self.emitter.log(
                    f"Auto-trade executed: {symbol} {signal.action}",
                    component="AutoTrader",
                )
            with self._lock:
                self._cooldowns.pop(symbol, None)

    def _sync_paper_gui_state(
        self,
        symbol: str,
        action: str,
        price: float,
        size: float | None,
    ) -> None:
        if not self._paper_enabled:
            return
        gui = getattr(self, "gui", None)
        if gui is None:
            return

        action_norm = str(action or "").upper()
        if action_norm not in {"BUY", "SELL"}:
            return

        side = "buy" if action_norm == "BUY" else "sell"
        try:
            price_value = float(price)
        except Exception:
            price_value = 0.0

        bridge = getattr(gui, "_bridge_execute_trade", None)
        if callable(bridge):
            try:
                bridge(symbol, side, price_value)
            except Exception:  # pragma: no cover - defensywne logowanie
                logger.debug("Paper GUI bridge sync failed", exc_info=True)

        snapshot: Mapping[str, Any] = {}
        adapter = self._paper_adapter
        if adapter is not None:
            try:
                raw_snapshot = adapter.portfolio_snapshot(symbol)
            except Exception:  # pragma: no cover - defensywne logowanie
                logger.debug("Paper portfolio snapshot failed", exc_info=True)
                raw_snapshot = None
            if isinstance(raw_snapshot, Mapping):
                snapshot = dict(raw_snapshot)

        balance = snapshot.get("value") if snapshot else None
        if balance is not None:
            try:
                balance_value = float(balance)
            except Exception:
                balance_value = None
            if balance_value is not None:
                try:
                    gui.paper_balance = balance_value
                except Exception:
                    pass
                self.update_account_equity(balance_value)
                balance_var = getattr(gui, "paper_balance_var", None)
                if balance_var is not None and hasattr(balance_var, "set"):
                    try:
                        balance_var.set(f"{balance_value:,.2f}")
                    except Exception:
                        logger.debug("Failed to update paper_balance_var", exc_info=True)

        positions_attr = getattr(gui, "_open_positions", None)
        if isinstance(positions_attr, dict):
            symbol_str = "" if symbol is None else str(symbol)
            symbol_key = symbol_str.upper() or symbol_str
            if action_norm == "SELL":
                positions_attr.pop(symbol_key, None)
            else:
                qty_candidate: Any = snapshot.get("position") if snapshot else None
                if qty_candidate is None:
                    qty_candidate = size
                try:
                    qty_value = float(qty_candidate) if qty_candidate is not None else 0.0
                except Exception:
                    qty_value = 0.0
                if qty_value <= 0 and size is not None:
                    try:
                        qty_value = float(size)
                    except Exception:
                        qty_value = 0.0
                entry_candidate = snapshot.get("price") if snapshot else None
                if entry_candidate is None:
                    entry_candidate = price_value
                try:
                    entry_value = float(entry_candidate)
                except Exception:
                    entry_value = price_value
                if qty_value > 0:
                    positions_attr[symbol_key] = {
                        "side": side,
                        "qty": qty_value,
                        "entry": entry_value,
                    }

    async def _build_market_payload(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        if self._data_provider is None:
            return None
        with exception_guard("AutoTrader.market_data"):
            ohlcv = await self._data_provider.get_ohlcv(symbol, timeframe, limit=256)
            ticker = await self._data_provider.get_ticker(symbol)
        payload: Dict[str, Any] = {
            "ohlcv": ohlcv or {},
            "ticker": ticker or {},
        }
        payload["price"] = self._extract_price_from_payload(payload)
        intel_snapshot = self._collect_market_intel(symbol, timeframe)
        if intel_snapshot:
            payload["market_intel"] = intel_snapshot
            fallback_price = self._resolve_price_from_intel(intel_snapshot)
            current_price = payload.get("price")
            try:
                current_price_value = float(current_price)
            except Exception:
                current_price_value = 0.0
            if fallback_price is not None and (current_price_value <= 0 or current_price is None):
                payload["price"] = fallback_price
        return payload

    def _collect_market_intel(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        aggregator = self._market_intel or getattr(self.gui, "market_intel", None)
        if aggregator is None:
            return None

        if self._market_intel is None:
            self._market_intel = aggregator

        cache_key = (symbol, timeframe)
        now = time.monotonic()
        cached = self._market_intel_cache.get(cache_key)
        if cached is not None:
            ts = self._market_intel_cache_ts.get(cache_key, 0.0)
            if now - ts <= self._market_intel_cache_ttl:
                return dict(cached)

        payload: Optional[Dict[str, Any]] = None
        mode = getattr(aggregator, "_mode", None)

        if mode == "cache" and MarketIntelQuery is not None and hasattr(aggregator, "build_snapshot"):
            try:
                query_symbol = self._normalise_market_intel_symbol(symbol, mode="cache")
                interval = self._normalise_market_intel_interval(timeframe)
                query = MarketIntelQuery(  # type: ignore[call-arg]
                    symbol=query_symbol,
                    interval=interval,
                    lookback_bars=96,
                )
                snapshot = aggregator.build_snapshot(query)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - zależne od implementacji agregatora
                logger.debug("Market intel snapshot failed for %s/%s", symbol, timeframe, exc_info=True)
            else:
                payload = self._snapshot_to_mapping(snapshot)
        else:
            payload = self._collect_sqlite_market_intel(aggregator, symbol)

        if payload:
            self._market_intel_cache[cache_key] = payload
            self._market_intel_cache_ts[cache_key] = now
            return dict(payload)
        return None

    def _collect_sqlite_market_intel(self, aggregator: Any, symbol: str) -> Optional[Dict[str, Any]]:
        if not hasattr(aggregator, "build"):
            return None
        try:
            baselines = aggregator.build()  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - zależne od implementacji agregatora
            logger.debug("Market intel baseline build failed for %s", symbol, exc_info=True)
            return None
        symbol_aliases = self._market_intel_symbol_aliases(symbol)
        for baseline in baselines or ():
            candidate = str(getattr(baseline, "symbol", "")).upper()
            if candidate not in symbol_aliases:
                continue
            if hasattr(baseline, "to_mapping"):
                mapping = baseline.to_mapping()  # type: ignore[attr-defined]
                return {k: v for k, v in mapping.items() if v is not None}
            attrs: Dict[str, Any] = {}
            for key in dir(baseline):
                if key.startswith("_"):
                    continue
                value = getattr(baseline, key)
                if callable(value):
                    continue
                attrs[key] = value
            if attrs:
                return {k: v for k, v in attrs.items() if v is not None}
        return None

    @staticmethod
    def _snapshot_to_mapping(snapshot: Any) -> Dict[str, Any]:
        if snapshot is None:
            return {}
        if hasattr(snapshot, "to_dict"):
            mapping = snapshot.to_dict()  # type: ignore[attr-defined]
            return {k: v for k, v in mapping.items() if v is not None}
        attrs: Dict[str, Any] = {}
        for key in dir(snapshot):
            if key.startswith("_"):
                continue
            value = getattr(snapshot, key)
            if callable(value):
                continue
            attrs[key] = value
        return {k: v for k, v in attrs.items() if v is not None}

    @staticmethod
    def _normalise_market_intel_interval(timeframe: str) -> str:
        if not timeframe:
            return "1h"
        return str(timeframe).strip()

    @staticmethod
    def _normalise_market_intel_symbol(symbol: str, *, mode: str) -> str:
        base = str(symbol or "").upper().strip()
        if mode == "cache":
            return base.replace("/", "_").replace("-", "_")
        return base.replace("/", "").replace("-", "")

    def _market_intel_symbol_aliases(self, symbol: str) -> set[str]:
        base = str(symbol or "").upper().strip()
        aliases = {
            base,
            base.replace("/", "_").replace("-", "_"),
            base.replace("/", "").replace("-", ""),
        }
        return {alias for alias in aliases if alias}

    @staticmethod
    def _resolve_price_from_intel(payload: Mapping[str, Any]) -> Optional[float]:
        for key in ("price", "mid_price", "close", "last"):
            value = payload.get(key)
            try:
                if value is not None and float(value) > 0:
                    return float(value)
            except Exception:
                continue
        metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
        if isinstance(metadata, Mapping):
            for key in ("mid_price", "price"):
                value = metadata.get(key)
                try:
                    if value is not None and float(value) > 0:
                        return float(value)
                except Exception:
                    continue
        return None

    def _build_market_state(
        self,
        portfolio_snapshot: Mapping[str, Any],
        market_payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        price = float(market_payload.get("price") or 0.0)
        daily_loss = float(
            portfolio_snapshot.get("daily_loss_pct")
            or portfolio_snapshot.get("daily_loss")
            or 0.0
        )
        portfolio_value = float(
            portfolio_snapshot.get("value")
            or portfolio_snapshot.get("portfolio_value")
            or portfolio_snapshot.get("equity")
            or 0.0
        )
        position_qty = float(
            portfolio_snapshot.get("position")
            or portfolio_snapshot.get("qty")
            or 0.0
        )
        notional = abs(position_qty * price)
        position_fraction = notional / portfolio_value if portfolio_value > 0 else 0.0
        drawdown = portfolio_snapshot.get("drawdown_pct") or portfolio_snapshot.get("max_drawdown_pct")
        if drawdown is None:
            drawdown = portfolio_snapshot.get("drawdown")
        try:
            drawdown_value = float(drawdown) if drawdown is not None else 0.0
        except Exception:
            drawdown_value = 0.0

        open_positions_raw = (
            portfolio_snapshot.get("open_positions")
            or portfolio_snapshot.get("positions")
            or portfolio_snapshot.get("active_positions")
        )
        open_positions = self._infer_open_positions(open_positions_raw, position_qty)

        portfolio_exposure = portfolio_snapshot.get("portfolio_exposure_pct")
        try:
            exposure_value = float(portfolio_exposure)
        except Exception:
            exposure_value = position_fraction

        return {
            "price": price,
            "daily_loss_pct": daily_loss,
            "portfolio_value": portfolio_value,
            "position_notional_pct": position_fraction,
            "portfolio_exposure_pct": max(0.0, exposure_value),
            "open_positions": open_positions,
            "drawdown_pct": drawdown_value,
        }

    @staticmethod
    def _infer_open_positions(raw: Any, current_position: float) -> int:
        if raw is None:
            return 1 if current_position else 0
        if isinstance(raw, (int, float)):
            try:
                value = int(raw)
            except Exception:
                value = 0
            return max(0, value)
        if isinstance(raw, Mapping):
            count = 0
            for entry in raw.values():
                if isinstance(entry, Mapping):
                    qty = entry.get("qty") or entry.get("quantity")
                    try:
                        if float(qty):
                            count += 1
                    except Exception:
                        continue
                elif entry:
                    count += 1
            if count:
                return count
            return 1 if current_position else 0
        if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)):
            count = 0
            for entry in raw:
                if isinstance(entry, Mapping):
                    qty = entry.get("qty") or entry.get("quantity")
                    try:
                        if float(qty):
                            count += 1
                    except Exception:
                        continue
                elif entry:
                    count += 1
            return count
        return 1 if current_position else 0

    def _extract_price_from_payload(self, market_payload: Mapping[str, Any]) -> float:
        ticker = market_payload.get("ticker") if isinstance(market_payload, Mapping) else None
        if isinstance(ticker, Mapping):
            for key in ("last", "close", "bid", "ask", "price"):
                value = ticker.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
        ohlcv = market_payload.get("ohlcv") if isinstance(market_payload, Mapping) else None
        if isinstance(ohlcv, Mapping):
            close = ohlcv.get("close")
            if isinstance(close, (int, float)):
                return float(close)
        if isinstance(ohlcv, pd.DataFrame) and not ohlcv.empty:
            try:
                return float(ohlcv["close"].iloc[-1])
            except Exception:
                pass
        if isinstance(ohlcv, Iterable):
            try:
                last = list(ohlcv)[-1]
                if isinstance(last, Mapping):
                    value = last.get("close")
                    if isinstance(value, (int, float)):
                        return float(value)
                elif isinstance(last, (list, tuple)) and last:
                    candidate = last[4] if len(last) > 4 else last[-2]
                    if isinstance(candidate, (int, float)):
                        return float(candidate)
            except Exception:
                pass
        return 0.0

    def _resolve_strategy_metadata(self, strategy_name: str) -> StrategyMetadata:
        try:
            strategy_cls = self._signal_service._registry.get(strategy_name)
        except KeyError:
            return StrategyMetadata(name=strategy_name or "Unknown", description="AutoTrader context")
        metadata = getattr(strategy_cls, "metadata", None)
        if isinstance(metadata, StrategyMetadata):
            return metadata
        return StrategyMetadata(name=strategy_cls.__name__, description="AutoTrader context")

    def _resolve_portfolio_snapshot(self, symbol: str, price: float) -> Dict[str, Any]:
        adapter_snapshot = self._execution_service.portfolio_snapshot(symbol)
        if adapter_snapshot:
            return {
                "value": float(adapter_snapshot.get("value", 0.0)),
                "position": float(adapter_snapshot.get("position", 0.0)),
                "daily_loss_pct": 0.0,
                "price": float(adapter_snapshot.get("price", price)),
            }
        snapshot_fn = getattr(self.gui, "get_portfolio_snapshot", None)
        if callable(snapshot_fn):
            try:
                snapshot = snapshot_fn(symbol=symbol)
                if isinstance(snapshot, Mapping):
                    return dict(snapshot)
            except Exception:
                logger.exception("Portfolio snapshot retrieval failed")
        balance = float(getattr(self.gui, "paper_balance", 0.0) or 0.0)
        positions = getattr(self.gui, "_open_positions", {})
        qty = 0.0
        if isinstance(positions, Mapping):
            entry = positions.get(symbol)
            if isinstance(entry, Mapping):
                qty = float(entry.get("qty") or entry.get("quantity") or 0.0)
                entry_price = float(entry.get("entry") or price or 0.0)
            else:
                entry_price = price
        else:
            entry_price = price
        notional = qty * float(price or entry_price or 0.0)
        return {
            "value": balance + notional,
            "position": qty,
            "daily_loss_pct": 0.0,
        }

    def _refresh_execution_mode(self) -> None:
        cfg = self._get_strategy_config()
        exchange_cfg = self._exchange_config or {}
        testnet_active = bool(exchange_cfg.get("testnet", True))
        mode = str(getattr(cfg, "mode", "")).lower()
        has_live_adapter = self._live_execution_adapter is not None
        if mode in {"demo", "paper"} or (not has_live_adapter and testnet_active):
            self._enable_paper_trading()
        else:
            self._disable_paper_trading()

    def _resolve_paper_initial_balance(self) -> float:
        def _coerce_positive(value: Any) -> float | None:
            if value is None:
                return None
            try:
                numeric = float(value)
            except Exception:
                return None
            if numeric > 0:
                return numeric
            return None

        gui_balance = _coerce_positive(getattr(self.gui, "paper_balance", None))
        if gui_balance is not None:
            return gui_balance

        strategy_cfg: StrategyConfig | None
        strategy_balance: float | None = None
        try:
            strategy_cfg = self._get_strategy_config()
        except Exception:
            strategy_cfg = getattr(self, "_strategy_config", None)
        if strategy_cfg is not None:
            for attr in ("paper_balance", "paper_capital", "initial_balance", "starting_balance"):
                candidate = _coerce_positive(getattr(strategy_cfg, attr, None))
                if candidate is not None:
                    strategy_balance = candidate
                    break
            if strategy_balance is None:
                max_usd = _coerce_positive(getattr(strategy_cfg, "max_position_usd", None))
                notional_pct = _coerce_positive(getattr(strategy_cfg, "max_position_notional_pct", None))
                if max_usd is not None and notional_pct:
                    try:
                        derived = max_usd / notional_pct
                    except ZeroDivisionError:
                        derived = 0.0
                    if derived > 0:
                        strategy_balance = derived
        if strategy_balance is not None:
            return strategy_balance

        profile_balance: float | None = None
        profile_cfg = self._risk_profile_config
        if profile_cfg is not None:
            profile_mapping: Mapping[str, Any] | None = profile_cfg if isinstance(profile_cfg, Mapping) else None
            for attr in ("paper_balance", "paper_capital", "initial_balance", "starting_balance", "notional_capital"):
                value = getattr(profile_cfg, attr, None)
                if value is None and profile_mapping is not None:
                    value = profile_mapping.get(attr)
                candidate = _coerce_positive(value)
                if candidate is not None:
                    profile_balance = candidate
                    break
            if profile_balance is None:
                pct_value = _coerce_positive(
                    profile_mapping.get("max_position_pct") if profile_mapping is not None else getattr(profile_cfg, "max_position_pct", None)
                )
                if pct_value:
                    base_usd = _coerce_positive(getattr(strategy_cfg, "max_position_usd", None) if strategy_cfg is not None else None)
                    if base_usd is not None:
                        try:
                            derived = base_usd / pct_value
                        except ZeroDivisionError:
                            derived = 0.0
                        if derived > 0:
                            profile_balance = derived
        if profile_balance is not None:
            return profile_balance

        risk_settings = getattr(self, "_risk_manager_settings", None)
        if risk_settings is not None:
            per_trade_pct = _coerce_positive(getattr(risk_settings, "max_risk_per_trade", None))
            strategy_usd = _coerce_positive(getattr(strategy_cfg, "max_position_usd", None) if strategy_cfg is not None else None)
            if per_trade_pct and strategy_usd is not None:
                try:
                    derived = strategy_usd / per_trade_pct
                except ZeroDivisionError:
                    derived = 0.0
                if derived > 0:
                    return derived

        return 10_000.0

    def _enable_paper_trading(self) -> None:
        if self._paper_enabled:
            return
        initial_balance = self._resolve_paper_initial_balance()
        self._paper_adapter = PaperTradingAdapter(initial_balance=initial_balance)
        self._execution_service.set_adapter(self._paper_adapter)
        self._paper_enabled = True
        self.emitter.log("Paper trading engine enabled", component="AutoTrader")

    def _disable_paper_trading(self) -> None:
        if not self._paper_enabled:
            return
        target_adapter = self._live_execution_adapter or self._get_emergency_adapter()
        self._execution_service.set_adapter(target_adapter)
        self._paper_adapter = None
        self._paper_enabled = False
        self.emitter.log("Paper trading engine disabled", component="AutoTrader")

    def _auto_trade_loop(self) -> None:
        if self._service_mode_enabled:
            self._run_service_loop()
            return
        core_mode = (
            self._core_risk_engine is not None
            and self._core_execution_service is not None
            and self._core_ai_connector is not None
            and OrderRequest is not None
            and AccountSnapshot is not None
            and CoreExecutionContext is not None
        )
        while not self._stop.is_set():
            try:
                if not self.enable_auto_trade or not self._auto_trade_user_confirmed:
                    self._stop.wait(self.auto_trade_interval_s)
                    continue
                symbol = self.symbol_getter()
                if not symbol:
                    self._stop.wait(self.auto_trade_interval_s)
                    continue
                timeframe = "1m"
                tf_var = getattr(self.gui, "timeframe_var", None)
                if tf_var is not None and hasattr(tf_var, "get"):
                    try:
                        timeframe = tf_var.get() or timeframe
                    except Exception:
                        pass

                ex = getattr(self.gui, "ex_mgr", None)
                ai = getattr(self.gui, "ai_mgr", None)

                if core_mode:
                    handled = self._handle_core_auto_trade(symbol, timeframe)
                    self._stop.wait(self.auto_trade_interval_s)
                    if handled:
                        continue

                if hasattr(self.gui, "is_demo_mode_active"):
                    try:
                        if not self.gui.is_demo_mode_active():
                            guard_fn = getattr(self.gui, "is_live_trading_allowed", None)
                            if callable(guard_fn) and not guard_fn():
                                msg = "Auto-trade blocked: live trading requires explicit confirmation."
                                self.emitter.log(msg, level="WARNING", component="AutoTrader")
                                logger.warning("%s Skipping auto trade for %s", msg, symbol)
                                self._stop.wait(self.auto_trade_interval_s)
                                continue
                    except Exception:
                        logger.exception("Failed to evaluate live trading guard")

                df: Optional[pd.DataFrame] = None
                last_price: Optional[float] = None
                last_pred: Optional[float] = None

                if ai is not None and hasattr(ai, "predict_series"):
                    try:
                        last_pred, df, last_price = self._obtain_prediction(ai, symbol, timeframe, ex)
                    except Exception as e:
                        self.emitter.log(
                            f"predict_series failed: {e!r}", level="ERROR", component="AutoTrader"
                        )
                        logger.exception("predict_series failed during auto trade loop")

                if last_price is None:
                    last_price = self._resolve_market_price(symbol, ex, df)

                side: Optional[str] = None
                if last_pred is not None and ai is not None:
                    threshold_bps = float(getattr(ai, "ai_threshold_bps", 5.0))
                    threshold = threshold_bps / 10_000.0
                    if last_pred >= threshold:
                        side = "BUY"
                    elif last_pred <= -threshold:
                        side = "SELL"
                elif last_pred is not None:
                    # brak ai_threshold -> użyj domyślnego progu
                    threshold = 5.0 / 10_000.0
                    if last_pred >= threshold:
                        side = "BUY"
                    elif last_pred <= -threshold:
                        side = "SELL"

                if side is None:
                    self.emitter.log(
                        f"Auto-trade skipped for {symbol}: no valid model signal.",
                        level="WARNING",
                        component="AutoTrader",
                    )
                    logger.info("Skipping auto trade for %s due to missing model signal", symbol)
                    self._stop.wait(self.auto_trade_interval_s)
                    continue

                if last_price is None:
                    self.emitter.log(
                        f"Auto-trade skipped for {symbol}: missing market price.",
                        level="WARNING",
                        component="AutoTrader",
                    )
                    logger.warning("Skipping auto trade for %s due to missing market price", symbol)
                    self._stop.wait(self.auto_trade_interval_s)
                    continue

                signal_payload = self._build_signal_payload(symbol, side, last_pred)
                decision = self._evaluate_risk(symbol, side, float(last_price), signal_payload, df)
                self._emit_risk_audit(symbol, side, decision, float(last_price))
                if not decision.should_trade:
                    self._stop.wait(self.auto_trade_interval_s)
                    continue

                if hasattr(self.gui, "_bridge_execute_trade"):
                    try:
                        setattr(self.gui, "_autotrade_risk_context", decision.to_dict())
                    except Exception:
                        pass
                    self.gui._bridge_execute_trade(symbol, side.lower(), float(last_price))
                    try:
                        prometheus_metrics.record_order(symbol, side, decision.fraction)
                    except Exception:
                        logger.debug("Prometheus record_order skipped", exc_info=True)
                    self.emitter.emit("auto_trade_tick", symbol=symbol, ts=time.time())
                    self.emitter.log(f"Auto-trade executed: {symbol} {side}", component="AutoTrader")
                    logger.info("Auto trade executed for %s (%s)", symbol, side)
                else:
                    self.emitter.log(
                        "_bridge_execute_trade missing on GUI",
                        level="ERROR",
                        component="AutoTrader",
                    )
                    logger.error("GUI bridge missing for auto trade execution")
            except Exception as e:
                self.emitter.log(f"Auto trade tick error: {e!r}", level="ERROR", component="AutoTrader")
                logger.exception("Unhandled exception inside auto trade loop")
            self._stop.wait(self.auto_trade_interval_s)

    def _handle_core_auto_trade(self, symbol: str, timeframe: str) -> bool:
        connector = self._core_ai_connector
        risk_engine = self._core_risk_engine
        execution_service = self._core_execution_service
        if (
            connector is None
            or risk_engine is None
            or execution_service is None
            or OrderRequest is None
            or AccountSnapshot is None
            or CoreExecutionContext is None
        ):
            return False

        ai_manager = getattr(connector, "ai_manager", None)
        if ai_manager is None:
            return False

        ex_mgr = getattr(self.gui, "ex_mgr", None)
        try:
            last_pred, df, last_price = self._obtain_prediction(
                ai_manager, symbol, timeframe, ex_mgr
            )
        except Exception as exc:  # pragma: no cover - diagnostyka awarii AI
            self.emitter.log(
                f"AI prediction failed: {exc!r}", level="ERROR", component="AutoTrader"
            )
            logger.exception("AI prediction failed during core auto trade loop")
            return True

        if df is None or df.empty:
            self.emitter.log(
                f"Auto-trade skipped for {symbol}: brak danych rynkowych.",
                level="WARNING",
                component="AutoTrader",
            )
            return True

        if last_pred is None:
            self.emitter.log(
                f"Auto-trade skipped for {symbol}: brak sygnału AI.",
                level="WARNING",
                component="AutoTrader",
            )
            return True

        price = float(last_price if last_price is not None else df["close"].iloc[-1])
        normalized_symbol = self._normalize_symbol(symbol)
        candidate = connector.candidate_from_signal(
            symbol=normalized_symbol,
            signal=float(last_pred),
            timestamp=df.index[-1] if not df.empty else None,
            notional=self._resolve_core_notional(normalized_symbol),
        )
        if candidate is None:
            self.emitter.log(
                f"Auto-trade skipped for {symbol}: sygnał poniżej progu.",
                level="WARNING",
                component="AutoTrader",
            )
            return True

        side = "BUY" if float(last_pred) >= 0 else "SELL"
        quantity = candidate.notional / max(price, 1e-9)
        if quantity <= 0:
            self.emitter.log(
                f"Auto-trade skipped for {symbol}: nieprawidłowa wielkość zlecenia.",
                level="WARNING",
                component="AutoTrader",
            )
            return True

        metadata: Dict[str, object] = {"decision_candidate": candidate.to_mapping()}
        order_request = OrderRequest(
            symbol=normalized_symbol,
            side=side.lower(),
            quantity=quantity,
            order_type="market",
            price=price,
            metadata=metadata,
        )

        try:
            account_snapshot = self._core_account_snapshot()
        except Exception:  # pragma: no cover - brak klas bot_core
            return False

        profile_name = (
            self._core_risk_profile_name
            or (getattr(self._core_risk_profile, "name", None) if self._core_risk_profile is not None else None)
            or candidate.risk_profile
        )

        try:
            risk_result = risk_engine.apply_pre_trade_checks(
                order_request,
                account=account_snapshot,
                profile_name=profile_name,
            )
        except Exception as exc:  # pragma: no cover - diagnostyka silnika ryzyka
            self.emitter.log(
                f"Core risk checks failed: {exc!r}",
                level="ERROR",
                component="AutoTrader",
            )
            logger.exception("Core risk engine error during auto trade")
            return True

        portfolio_ctx = self._core_portfolio_context(account_snapshot)
        decision = self._build_core_risk_decision(
            symbol=normalized_symbol,
            side=side,
            price=price,
            candidate=candidate,
            order_request=order_request,
            risk_result=risk_result,
            account_snapshot=account_snapshot,
            portfolio_ctx=portfolio_ctx,
            profile_name=profile_name,
        )
        decision = self._finalize_risk_decision(
            normalized_symbol,
            side,
            price,
            decision,
            portfolio_ctx,
        )
        self._emit_risk_audit(normalized_symbol, side, decision, price)
        if not decision.should_trade:
            reason_text = (
                decision.details.get("risk_reason")
                if isinstance(decision.details, Mapping)
                else None
            ) or risk_result.reason or decision.reason
            self.emitter.log(
                f"Core auto-trade denied for {symbol}: {reason_text}",
                level="WARNING",
                component="AutoTrader",
            )
            return True

        try:
            context = self._build_core_execution_context(metadata)
            result = execution_service.execute(order_request, context)
        except Exception as exc:  # pragma: no cover - diagnostyka egzekucji
            self.emitter.log(
                f"Core execution failed: {exc!r}",
                level="ERROR",
                component="AutoTrader",
            )
            logger.exception("Core execution service failed")
            return True

        self._post_core_fill(normalized_symbol, side, order_request, result)
        try:
            prometheus_metrics.record_order(normalized_symbol, side, quantity)
        except Exception:
            logger.debug("Prometheus record_order skipped", exc_info=True)
        self.emitter.emit("auto_trade_tick", symbol=normalized_symbol, ts=time.time())
        self.emitter.log(
            f"Auto-trade executed (core): {normalized_symbol} {side}",
            component="AutoTrader",
        )
        logger.info("Core auto trade executed for %s (%s)", normalized_symbol, side)
        return True

    def _normalize_symbol(self, symbol: str) -> str:
        return str(symbol).replace("/", "").upper()

    def _resolve_core_notional(self, symbol: str) -> float:
        normalized = self._normalize_symbol(symbol)
        if normalized in self._core_ai_notional_by_symbol:
            return float(self._core_ai_notional_by_symbol[normalized])
        if self._core_ai_default_notional is not None:
            return float(self._core_ai_default_notional)
        if self._core_ai_connector is not None:
            return float(self._core_ai_connector.default_notional)
        return 0.0

    def _core_account_snapshot(self) -> Any:
        if AccountSnapshot is None:
            raise RuntimeError("AccountSnapshot class unavailable")
        balances = {"USDT": float(self._core_account_equity)}
        return AccountSnapshot(
            balances=balances,
            total_equity=float(self._core_account_equity),
            available_margin=float(self._core_account_equity),
            maintenance_margin=float(self._core_account_equity) * 0.1,
        )

    def _core_portfolio_context(self, account_snapshot: Any) -> Dict[str, Any]:
        context: Dict[str, Any] = {"positions": {}, "total_notional": 0.0}
        equity = 0.0
        available_margin = 0.0
        maintenance_margin = 0.0
        try:
            equity = float(getattr(account_snapshot, "total_equity", 0.0) or 0.0)
        except Exception:
            equity = 0.0
        try:
            available_margin = float(
                getattr(account_snapshot, "available_margin", 0.0) or 0.0
            )
        except Exception:
            available_margin = 0.0
        try:
            maintenance_margin = float(
                getattr(account_snapshot, "maintenance_margin", 0.0) or 0.0
            )
        except Exception:
            maintenance_margin = 0.0

        context["equity"] = equity
        context["cash"] = available_margin
        context["total_notional"] = max(0.0, equity)
        context["account_snapshot"] = {
            "total_equity": equity,
            "available_margin": available_margin,
            "maintenance_margin": maintenance_margin,
        }

        balances = getattr(account_snapshot, "balances", None)
        if isinstance(balances, Mapping):
            safe_balances: Dict[str, float] = {}
            for key, value in balances.items():
                try:
                    safe_balances[str(key)] = float(value)
                except Exception:
                    continue
            context["balances"] = safe_balances

        return context

    def _build_core_execution_context(
        self, metadata: Mapping[str, object]
    ) -> Any:
        if CoreExecutionContext is None:
            raise RuntimeError("ExecutionContext class unavailable")
        meta: Dict[str, object] = {"source": "AutoTrader"}
        meta.update(dict(metadata))
        profile_name: Any = self._core_risk_profile_name
        if profile_name is None and self._core_risk_profile is not None:
            profile_name = getattr(self._core_risk_profile, "name", None)
        if (profile_name is None or profile_name == "") and self._core_ai_connector is not None:
            connector_profile = getattr(self._core_ai_connector, "risk_profile", None)
            profile_name = getattr(connector_profile, "name", None) or connector_profile
        return CoreExecutionContext(
            portfolio_id=self._core_portfolio_id,
            risk_profile=str(profile_name or "default"),
            environment=str(self._core_execution_environment),
            metadata=meta,
        )

    def _build_core_risk_decision(
        self,
        *,
        symbol: str,
        side: str,
        price: float,
        candidate: Any,
        order_request: Any,
        risk_result: Any,
        account_snapshot: Any,
        portfolio_ctx: Mapping[str, Any],
        profile_name: str,
    ) -> RiskDecision:
        risk_allowed = bool(getattr(risk_result, "allowed", False))
        reason_text = getattr(risk_result, "reason", None)
        adjustments_raw = getattr(risk_result, "adjustments", None)
        metadata_raw = getattr(risk_result, "metadata", None)

        try:
            account_value = float(getattr(account_snapshot, "total_equity", 0.0) or 0.0)
        except Exception:
            try:
                account_value = float(portfolio_ctx.get("equity", 0.0) or 0.0)
            except Exception:
                account_value = 0.0

        try:
            candidate_notional = float(getattr(candidate, "notional", 0.0) or 0.0)
        except Exception:
            candidate_notional = 0.0

        base_fraction = 0.0
        if account_value > 0 and candidate_notional > 0:
            base_fraction = candidate_notional / account_value
        base_fraction = max(0.0, min(1.0, base_fraction))
        recommended_fraction = base_fraction

        adjustments: Dict[str, Any] | None = None
        if isinstance(adjustments_raw, Mapping):
            adjustments = {}
            for key, value in adjustments_raw.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    adjustments[str(key)] = float(value)
                else:
                    adjustments[str(key)] = value

            max_quantity = adjustments.get("max_quantity")
            if (
                isinstance(max_quantity, (int, float))
                and account_value > 0
                and price > 0
            ):
                try:
                    recommended_notional = float(max_quantity) * float(price)
                except Exception:
                    recommended_notional = 0.0
                if recommended_notional > 0:
                    recommended_fraction = max(
                        0.0,
                        min(1.0, recommended_notional / account_value),
                    )

        metadata_details: Dict[str, Any] | None = None
        if isinstance(metadata_raw, Mapping):
            metadata_details = {str(key): value for key, value in metadata_raw.items()}

        risk_engine_details: Dict[str, Any] = {"allowed": risk_allowed}
        if reason_text:
            risk_engine_details["reason"] = reason_text
        if adjustments:
            risk_engine_details["adjustments"] = adjustments
        if metadata_details:
            risk_engine_details["metadata"] = metadata_details

        risk_state: Mapping[str, Any] | None = None
        engine = self._core_risk_engine
        if engine is not None and hasattr(engine, "snapshot_state"):
            try:
                snapshot = engine.snapshot_state(profile_name)
            except Exception:
                snapshot = None
            if isinstance(snapshot, Mapping):
                risk_state = {str(k): v for k, v in snapshot.items()}
        if risk_state:
            risk_engine_details["state"] = risk_state

        limit_events: List[Dict[str, Any]] = []
        if not risk_allowed:
            event: Dict[str, Any] = {"type": "risk_engine_denied"}
            if reason_text:
                event["reason"] = reason_text
            limit_events.append(event)

        if adjustments:
            limit_events.append({"type": "risk_engine_adjustment", "adjustments": adjustments})

        if adjustments and recommended_fraction + 1e-9 < base_fraction:
            limit_events.append(
                {
                    "type": "risk_engine_clamp",
                    "baseline": base_fraction,
                    "value": recommended_fraction,
                }
            )

        requested_fraction = base_fraction
        effective_fraction = recommended_fraction if risk_allowed else 0.0

        order_details: Dict[str, Any] = {
            "symbol": getattr(order_request, "symbol", symbol),
            "side": getattr(order_request, "side", side),
        }
        try:
            order_details["quantity"] = float(getattr(order_request, "quantity", 0.0) or 0.0)
        except Exception:
            order_details["quantity"] = 0.0
        order_price = getattr(order_request, "price", None)
        try:
            if order_price is not None:
                order_details["price"] = float(order_price)
        except Exception:
            pass
        order_details["order_type"] = getattr(order_request, "order_type", None)
        metadata_section = getattr(order_request, "metadata", None)
        if isinstance(metadata_section, Mapping) and metadata_section:
            order_details["metadata"] = {str(k): v for k, v in metadata_section.items()}

        details: Dict[str, Any] = {
            "risk_engine_allowed": risk_allowed,
            "risk_engine": risk_engine_details,
            "risk_reason": reason_text,
            "account_value": account_value,
            "candidate_notional": candidate_notional,
            "recommended_size": recommended_fraction,
            "requested_fraction": requested_fraction,
            "order_request": order_details,
        }

        if hasattr(candidate, "to_mapping"):
            try:
                details["decision_candidate"] = candidate.to_mapping()
            except Exception:
                details["decision_candidate"] = dict(getattr(candidate, "__dict__", {}))
        elif hasattr(candidate, "__dict__"):
            details["decision_candidate"] = dict(getattr(candidate, "__dict__", {}))

        if limit_events:
            details["limit_events"] = limit_events

        decision = RiskDecision(
            should_trade=risk_allowed,
            fraction=effective_fraction,
            state="ok" if risk_allowed else "lock",
            reason="risk_ok" if risk_allowed else "risk_engine_denied",
            details=details,
            stop_loss_pct=None,
            take_profit_pct=None,
            mode=str(self._core_execution_environment),
        )

        return decision

    def _post_core_fill(
        self,
        symbol: str,
        side: str,
        request: Any,
        result: Any,
    ) -> None:
        if self._core_risk_engine is None:
            return
        raw_response_source = getattr(result, "raw_response", None)
        request_metadata_source = getattr(request, "metadata", None)

        def _coerce_float(value: Any) -> float | None:
            if value is None:
                return None
            try:
                result = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(result):
                return None
            return result

        def _locate_float(
            *values: Any,
            search_keys: Iterable[str] | None = None,
            containers: Sequence[Any] = (),
            nested_search_keys: Iterable[str] | None = None,
        ) -> float | None:
            key_sequences: tuple[tuple[str, ...], tuple[str, ...]] | None = None

            def _maybe_extract(container: Any) -> float | None:
                nonlocal key_sequences
                if container is None:
                    return None
                if key_sequences is None:
                    primary_keys = tuple(search_keys or ())
                    secondary_keys = tuple(nested_search_keys or ())
                    key_sequences = (primary_keys, secondary_keys)
                else:
                    primary_keys, secondary_keys = key_sequences

                def _extract_from(container_obj: Any, keys: tuple[str, ...]) -> float | None:
                    if not keys:
                        return None
                    for key in keys:
                        extracted = _extract_nested(container_obj, {key})
                        coerced = _coerce_float(extracted)
                        if coerced is not None:
                            return coerced
                        if (
                            extracted is not None
                            and secondary_keys
                            and _should_descend(extracted)
                        ):
                            nested = _extract_nested(extracted, secondary_keys)
                            coerced_nested = _coerce_float(nested)
                            if coerced_nested is not None:
                                return coerced_nested
                    return None

                primary_result = _extract_from(container, primary_keys)
                if primary_result is not None:
                    return primary_result
                if secondary_keys:
                    secondary_result = _extract_from(container, secondary_keys)
                    if secondary_result is not None:
                        return secondary_result
                return None

            for candidate in values:
                coerced = _coerce_float(candidate)
                if coerced is not None:
                    return coerced
                if (search_keys or nested_search_keys) and _should_descend(candidate):
                    nested_value = _maybe_extract(candidate)
                    if nested_value is not None:
                        return nested_value
            if search_keys or nested_search_keys:
                for container in containers:
                    nested_value = _maybe_extract(container)
                    if nested_value is not None:
                        return nested_value
            return None

        def _first_valid_float(
            *values: Any,
            default: float = 0.0,
            search_keys: Iterable[str] | None = None,
            containers: Sequence[Any] = (),
        ) -> float:
            located = _locate_float(
                *values, search_keys=search_keys, containers=containers
            )
            if located is not None:
                return located
            return float(default)

        def _as_mapping(candidate: Any) -> Mapping[str, Any] | None:
            if isinstance(candidate, Mapping):
                return candidate
            if hasattr(candidate, "_asdict"):
                try:
                    mapping = candidate._asdict()  # type: ignore[attr-defined]
                except Exception:
                    mapping = None
                else:
                    if isinstance(mapping, Mapping):
                        return mapping
            if hasattr(candidate, "__dict__"):
                try:
                    mapping = vars(candidate)
                except TypeError:
                    return None
                if isinstance(mapping, Mapping):
                    return mapping
            return None

        def _should_descend(value: Any) -> bool:
            if value is None:
                return False
            if isinstance(value, (str, bytes, bytearray)):
                return False
            if isinstance(value, Mapping):
                return True
            if isinstance(value, Sequence):
                return True
            if isinstance(value, set):
                return True
            return _as_mapping(value) is not None

        def _extract_nested(container: Any, keys: Iterable[str]) -> Any | None:
            key_set = set(keys)
            stack: list[Any] = [container]
            seen: set[int] = set()

            while stack:
                current = stack.pop()
                marker = id(current)
                if marker in seen:
                    continue
                seen.add(marker)

                mapping_view = _as_mapping(current)
                if mapping_view is not None:
                    for key, value in mapping_view.items():
                        if key in key_set and value is not None:
                            return value
                        if _should_descend(value):
                            stack.append(value)
                    continue

                if isinstance(current, (Sequence, set)) and not isinstance(
                    current, (str, bytes, bytearray)
                ):
                    if (
                        isinstance(current, Sequence)
                        and len(current) == 2
                        and isinstance(current[0], str)
                        and current[0] in key_set
                        and current[1] is not None
                    ):
                        return current[1]
                    iterable = current if isinstance(current, Sequence) else list(current)
                    for item in iterable:
                        if isinstance(item, Sequence) and not isinstance(
                            item, (str, bytes, bytearray)
                        ):
                            if (
                                len(item) == 2
                                and isinstance(item[0], str)
                                and item[0] in key_set
                                and item[1] is not None
                            ):
                                return item[1]
                        if _should_descend(item):
                            stack.append(item)

            return None

        def _normalize_non_empty(value: Any) -> Any | None:
            if value is None:
                return None
            try:
                if pd.isna(value):
                    return None
            except Exception:
                pass
            if isinstance(value, str):
                candidate = value.strip()
                return candidate or None
            return value

        def _normalize_liquidity(value: Any) -> str | None:
            if value is None:
                return None
            if isinstance(value, bool):
                return "maker" if value else "taker"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric = float(value)
                if math.isfinite(numeric):
                    if math.isclose(numeric, 1.0, rel_tol=1e-9, abs_tol=1e-9):
                        return "maker"
                    if math.isclose(numeric, 0.0, rel_tol=1e-9, abs_tol=1e-9):
                        return "taker"
            if isinstance(value, str):
                candidate = value.strip().lower()
                if not candidate:
                    return None
                mapping = {
                    "maker": "maker",
                    "m": "maker",
                    "maker_flag": "maker",
                    "makerrole": "maker",
                    "add": "maker",
                    "passive": "maker",
                    "taker": "taker",
                    "t": "taker",
                    "taker_flag": "taker",
                    "takerrole": "taker",
                    "remove": "taker",
                    "active": "taker",
                }
                normalized = mapping.get(candidate)
                if normalized is not None:
                    return normalized
                if candidate in {"true", "yes"}:
                    return "maker"
                if candidate in {"false", "no"}:
                    return "taker"
            return None

        def _first_non_empty(
            *values: Any,
            search_keys: Iterable[str] | None = None,
            containers: Sequence[Any] = (),
            transform: Callable[[Any], Any] | None = None,
            nested_search_keys: Iterable[str] | None = None,
        ) -> Any | None:
            nested_keys = tuple(nested_search_keys or ())

            def _resolve(candidate: Any) -> Any | None:
                normalized = _normalize_non_empty(candidate)
                if normalized is None:
                    return None
                resolved = normalized
                if nested_keys and _should_descend(resolved):
                    for key in nested_keys:
                        extracted = _extract_nested(resolved, {key})
                        nested_normalized = _normalize_non_empty(extracted)
                        if nested_normalized is not None:
                            resolved = nested_normalized
                            break
                if _should_descend(resolved):
                    return None
                if transform is not None:
                    try:
                        return transform(resolved)
                    except Exception:
                        return None
                return resolved

            for candidate in values:
                resolved = _resolve(candidate)
                if resolved is not None:
                    return resolved
            if search_keys or nested_keys:
                combined_keys = tuple(search_keys or ())

                def _probe(container: Any) -> Any | None:
                    if container is None:
                        return None
                    if combined_keys:
                        for key in combined_keys:
                            extracted = _extract_nested(container, {key})
                            resolved = _resolve(extracted)
                            if resolved is not None:
                                return resolved
                    if nested_keys:
                        for key in nested_keys:
                            extracted = _extract_nested(container, {key})
                            resolved = _resolve(extracted)
                            if resolved is not None:
                                return resolved
                    return None

                for container in containers:
                    resolved = _probe(container)
                    if resolved is not None:
                        return resolved
            return None

        def _clone_for_log(
            value: Any,
            *,
            max_depth: int = 6,
            _seen: set[int] | None = None,
        ) -> Any:
            if max_depth <= 0:
                return repr(value)
            if value is None or isinstance(value, (str, bytes, bytearray, int, float, bool)):
                return value

            if _seen is None:
                _seen = set()
            marker = id(value)
            if marker in _seen:
                return "<cycle>"
            _seen.add(marker)

            mapping_view = _as_mapping(value)
            if mapping_view is not None:
                cloned: Dict[str, Any] = {}
                for key, child in mapping_view.items():
                    cloned[str(key)] = _clone_for_log(child, max_depth=max_depth - 1, _seen=_seen)
                return cloned

            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return [
                    _clone_for_log(item, max_depth=max_depth - 1, _seen=_seen)
                    for item in value
                ]

            if isinstance(value, set):
                try:
                    ordered = sorted(value, key=repr)
                except Exception:
                    ordered = list(value)
                return [
                    _clone_for_log(item, max_depth=max_depth - 1, _seen=_seen)
                    for item in ordered
                ]

            return value

        raw_response = _clone_for_log(raw_response_source)
        cloned_request_metadata = _clone_for_log(request_metadata_source)
        request_metadata = cloned_request_metadata if cloned_request_metadata is not None else {}
        raw_search_container = raw_response_source if raw_response_source is not None else raw_response
        request_search_container = (
            request_metadata_source if request_metadata_source is not None else request_metadata
        )

        def _normalize_timestamp(value: Any) -> datetime | None:
            if value is None:
                return None
            try:
                if pd.isna(value):
                    return None
            except Exception:
                pass
            if isinstance(value, datetime):
                if value.tzinfo is None:
                    return value.replace(tzinfo=timezone.utc)
                return value.astimezone(timezone.utc)
            if isinstance(value, pd.Timestamp):
                if pd.isna(value):
                    return None
                ts = value.tz_convert("UTC") if value.tzinfo else value.tz_localize("UTC")
                normalized = ts.to_pydatetime()
                if normalized is None:
                    return None
                return normalized
            if isinstance(value, (int, float)):
                numeric = float(value)
                if not math.isfinite(numeric):
                    return None
                abs_value = abs(numeric)
                if abs_value >= 1e18:  # nanoseconds
                    numeric /= 1_000_000_000.0
                elif abs_value >= 1e15:  # microseconds
                    numeric /= 1_000_000.0
                elif abs_value >= 1e12:  # milliseconds
                    numeric /= 1_000.0
                return datetime.fromtimestamp(numeric, tz=timezone.utc)
            if isinstance(value, str):
                candidate = value.strip()
                if not candidate:
                    return None
                lowered = candidate.lower()
                if lowered in {"nan", "nat"}:
                    return None
                if candidate.isdigit():
                    try:
                        numeric = float(candidate)
                    except ValueError:
                        numeric = None
                    else:
                        return _normalize_timestamp(numeric)
                try:
                    parsed = pd.to_datetime(candidate, utc=True)
                except Exception:
                    return None
                if isinstance(parsed, pd.Timestamp):
                    if pd.isna(parsed):
                        return None
                    return parsed.to_pydatetime()
                return None
            return None

        avg_price_val = _first_valid_float(
            getattr(result, "avg_price", None),
            getattr(result, "price", None),
            getattr(request, "avg_price", None),
            getattr(request, "price", None),
            default=0.0,
            search_keys=(
                "avg_price",
                "average_price",
                "price",
                "avgPrice",
                "averagePrice",
                "executionPrice",
                "fill_price",
                "executedPrice",
            ),
            containers=(raw_search_container, request_search_container),
        )
        filled_qty_val = _first_valid_float(
            getattr(result, "filled_quantity", None),
            getattr(result, "quantity", None),
            getattr(request, "filled_quantity", None),
            getattr(request, "quantity", None),
            default=0.0,
            search_keys=(
                "filled_quantity",
                "quantity",
                "qty",
                "filledQty",
                "executedQty",
                "size",
                "amount",
                "baseQty",
                "base_quantity",
            ),
            containers=(raw_search_container, request_search_container),
        )
        default_notional = abs(avg_price_val) * abs(filled_qty_val)
        notional_value = _first_valid_float(
            getattr(result, "notional", None),
            getattr(result, "filled_notional", None),
            getattr(request, "notional", None),
            getattr(request, "filled_notional", None),
            default=default_notional,
            search_keys=(
                "notional",
                "filled_notional",
                "quoteQty",
                "quote_quantity",
                "cummulativeQuoteQty",
                "quote_volume",
                "quoteAmount",
                "fillNotional",
                "cost",
            ),
            containers=(raw_search_container, request_search_container),
        )

        pnl_value = _locate_float(
            getattr(result, "pnl", None),
            search_keys=(
                "pnl",
                "realized_pnl",
                "realizedPnl",
                "profit",
                "profit_loss",
                "realizedProfit",
            ),
            containers=(raw_search_container, request_search_container),
        )
        pnl_value = float(pnl_value or 0.0)

        fee_value = _locate_float(
            getattr(result, "fee", None),
            getattr(result, "commission", None),
            getattr(request, "fee", None),
            getattr(request, "commission", None),
            search_keys=(
                "fee",
                "fees",
                "commission",
                "fee_amount",
                "feeAmount",
                "feeValue",
                "fee_cost",
                "commissionAmount",
                "commission_amount",
                "tradingFee",
                "fill_fee",
            ),
            containers=(raw_search_container, request_search_container),
            nested_search_keys=(
                "amount",
                "value",
                "cost",
                "feeAmount",
                "feeValue",
                "fee_cost",
                "commissionAmount",
                "commission_amount",
                "commissionValue",
            ),
        )

        if (
            fee_value is not None
            and not math.isclose(default_notional, 0.0)
            and math.isclose(abs(notional_value), abs(fee_value), rel_tol=1e-9, abs_tol=1e-9)
        ):
            notional_value = default_notional

        notional = abs(notional_value)

        fee_currency = _first_non_empty(
            getattr(result, "fee_currency", None),
            getattr(result, "commission_currency", None),
            getattr(result, "fee_asset", None),
            getattr(request, "fee_currency", None),
            getattr(request, "commission_currency", None),
            getattr(request, "fee_asset", None),
            search_keys=(
                "fee_currency",
                "feeCurrency",
                "fee_asset",
                "feeAsset",
                "commissionCurrency",
                "commissionAsset",
                "feeAssetCode",
            ),
            containers=(raw_search_container, request_search_container),
            transform=str,
            nested_search_keys=(
                "currency",
                "asset",
                "code",
                "symbol",
                "commissionCurrency",
                "commissionAsset",
                "feeCurrency",
                "feeAsset",
            ),
        )

        fee_rate_value = _locate_float(
            getattr(result, "fee_rate", None),
            getattr(result, "commission_rate", None),
            getattr(result, "maker_commission_rate", None),
            getattr(result, "taker_commission_rate", None),
            getattr(request, "fee_rate", None),
            getattr(request, "commission_rate", None),
            getattr(request, "maker_commission_rate", None),
            getattr(request, "taker_commission_rate", None),
            search_keys=(
                "fee_rate",
                "commission_rate",
                "maker_commission_rate",
                "taker_commission_rate",
                "makerCommissionRate",
                "takerCommissionRate",
                "commissionRate",
                "feeRate",
                "fee_rate_pct",
                "commissionRatePct",
            ),
            containers=(raw_search_container, request_search_container),
            nested_search_keys=(
                "rate",
                "feeRate",
                "commissionRate",
                "makerCommissionRate",
                "takerCommissionRate",
                "commission_rate",
                "fee_rate",
            ),
        )

        liquidity_role = _first_non_empty(
            getattr(result, "liquidity", None),
            getattr(result, "liquidity_type", None),
            getattr(result, "execution_role", None),
            getattr(result, "executionRole", None),
            getattr(result, "trade_liquidity", None),
            getattr(result, "is_maker", None),
            getattr(request, "liquidity", None),
            getattr(request, "liquidity_type", None),
            getattr(request, "execution_role", None),
            getattr(request, "executionRole", None),
            getattr(request, "trade_liquidity", None),
            getattr(request, "is_maker", None),
            search_keys=(
                "liquidity",
                "liquidity_type",
                "liquidityType",
                "execution_role",
                "executionRole",
                "execution_type",
                "executionType",
                "trade_liquidity",
                "tradeLiquidity",
                "liquidity_role",
                "liquidityRole",
                "makerFlag",
                "makerRole",
                "is_maker",
                "isMaker",
                "maker",
            ),
            containers=(raw_search_container, request_search_container),
            nested_search_keys=(
                "liquidity",
                "liquidity_type",
                "liquidityType",
                "execution_role",
                "executionRole",
                "execution_type",
                "executionType",
                "trade_liquidity",
                "tradeLiquidity",
                "makerFlag",
                "maker",
                "is_maker",
                "isMaker",
                "role",
                "type",
            ),
            transform=_normalize_liquidity,
        )

        order_id = _first_non_empty(
            getattr(result, "order_id", None),
            getattr(result, "orderId", None),
            getattr(request, "order_id", None),
            getattr(request, "orderId", None),
            search_keys=(
                "order_id",
                "orderId",
                "id",
                "origClientOrderId",
                "orig_order_id",
            ),
            containers=(raw_search_container, request_search_container),
            transform=str,
        )

        client_order_id = _first_non_empty(
            getattr(result, "client_order_id", None),
            getattr(result, "clientOrderId", None),
            getattr(request, "client_order_id", None),
            getattr(request, "clientOrderId", None),
            search_keys=(
                "client_order_id",
                "clientOrderId",
                "client_id",
                "clientId",
                "origClientOrderId",
            ),
            containers=(raw_search_container, request_search_container),
            transform=str,
        )

        trade_id = _first_non_empty(
            getattr(result, "trade_id", None),
            getattr(result, "tradeId", None),
            search_keys=(
                "trade_id",
                "tradeId",
                "deal_id",
                "dealId",
            ),
            containers=(raw_search_container, request_search_container),
            transform=str,
        )

        fill_id = _first_non_empty(
            getattr(result, "fill_id", None),
            getattr(result, "fillId", None),
            search_keys=(
                "fill_id",
                "fillId",
                "execution_id",
                "execId",
                "executionId",
            ),
            containers=(raw_search_container, request_search_container),
            transform=str,
        )

        timestamp_candidate = _normalize_timestamp(getattr(result, "timestamp", None))
        if timestamp_candidate is None:
            timestamp_candidate = _normalize_timestamp(getattr(result, "ts", None))
        if timestamp_candidate is None and raw_search_container is not None:
            timestamp_candidate = _normalize_timestamp(
                _extract_nested(
                    raw_search_container,
                    {
                        "timestamp",
                        "ts",
                        "time",
                        "transactTime",
                        "updateTime",
                        "filled_at",
                        "created_at",
                        "event_time",
                    },
                )
            )
        if timestamp_candidate is None and request_search_container is not None:
            timestamp_candidate = _normalize_timestamp(
                _extract_nested(
                    request_search_container,
                    {
                        "timestamp",
                        "ts",
                        "time",
                        "decision_time",
                        "generated_at",
                        "created_at",
                    },
                )
            )

        explicit_position_value = _locate_float(
            getattr(result, "position_value", None),
            getattr(result, "positionValue", None),
            getattr(request, "position_value", None),
            getattr(request, "positionValue", None),
            search_keys={
                "position_value",
                "positionValue",
                "position_notional",
                "positionNotional",
                "position_value_usd",
                "positionValueUsd",
                "exposure",
                "positionExposure",
            },
            containers=(raw_search_container, request_search_container),
        )

        explicit_position_delta = _locate_float(
            getattr(result, "position_delta", None),
            getattr(result, "positionDelta", None),
            getattr(request, "position_delta", None),
            getattr(request, "positionDelta", None),
            search_keys={
                "position_delta",
                "positionDelta",
                "position_change",
                "positionChange",
                "delta_notional",
                "deltaNotional",
                "positionDeltaNotional",
            },
            containers=(raw_search_container, request_search_container),
        )

        side_lower = str(side or "").lower()
        default_position_value = notional if side_lower == "buy" else 0.0
        position_value = (
            explicit_position_value
            if explicit_position_value is not None
            else default_position_value
        )
        position_delta = (
            explicit_position_delta
            if explicit_position_delta is not None
            else (notional if side_lower == "buy" else -notional)
        )

        profile_name = self._core_risk_profile
        candidate_metadata = None
        if isinstance(request_metadata, Mapping):
            candidate_metadata = request_metadata.get("decision_candidate")
        if candidate_metadata is None and request_search_container is not None:
            candidate_metadata = _extract_nested(
                request_search_container,
                {"decision_candidate"},
            )
        if not profile_name and isinstance(candidate_metadata, Mapping):
            candidate_profile = candidate_metadata.get("risk_profile")
            if candidate_profile:
                profile_name = str(candidate_profile)
        if not profile_name and self._core_ai_connector is not None:
            profile_name = getattr(self._core_ai_connector, "risk_profile", None)
        profile_name = profile_name or "default"

        try:
            self._core_risk_engine.on_fill(
                profile_name=profile_name,
                symbol=symbol,
                side=side_lower,
                position_value=position_value,
                pnl=pnl_value,
                timestamp=timestamp_candidate,
            )
        except Exception:
            logger.debug("Failed to update risk engine post fill", exc_info=True)

        decision_log = getattr(self._core_risk_engine, "_decision_log", None)
        if decision_log is None:
            decision_log = getattr(self._core_risk_engine, "decision_log", None)
        record_log = None
        if RiskDecisionLog is not None and isinstance(decision_log, RiskDecisionLog):
            record_log = decision_log
        elif decision_log is not None and hasattr(decision_log, "record"):
            record_log = decision_log
        if record_log is not None:
            metadata_payload: Dict[str, object] = {"source": "auto_trader_core_fill"}
            if request_metadata_source is not None or request_metadata:
                metadata_payload["request"] = request_metadata
            if raw_response_source is not None or raw_response:
                metadata_payload["fill"] = raw_response
            metrics: Dict[str, object] = {
                "avg_price": avg_price_val,
                "filled_quantity": filled_qty_val,
                "notional": notional,
                "position_value": position_value,
                "pnl": pnl_value,
            }
            metrics["position_delta"] = position_delta
            if fee_value is not None:
                metrics["fee"] = fee_value
            if fee_currency is not None:
                metrics["fee_currency"] = fee_currency
            if fee_rate_value is not None:
                metrics["fee_rate"] = fee_rate_value
            if timestamp_candidate is not None:
                metadata_payload["fill_timestamp"] = timestamp_candidate.isoformat()
            if liquidity_role is not None:
                metadata_payload["liquidity"] = liquidity_role
            identifiers: Dict[str, str] = {}
            if order_id is not None:
                identifiers["order_id"] = order_id
            if client_order_id is not None:
                identifiers["client_order_id"] = client_order_id
            if trade_id is not None:
                identifiers["trade_id"] = trade_id
            if fill_id is not None:
                identifiers["fill_id"] = fill_id
            if identifiers:
                metadata_payload["identifiers"] = identifiers
            metadata_payload["metrics"] = metrics
            try:
                record_log.record(
                    profile=profile_name,
                    symbol=symbol,
                    side=side_lower,
                    quantity=filled_qty_val,
                    price=avg_price_val,
                    notional=notional,
                    allowed=True,
                    reason="fill",
                    metadata=metadata_payload,
                )
            except Exception:
                logger.debug("Failed to append fill to risk decision log", exc_info=True)
    # --- Prediction helpers ---
    def _resolve_prediction_result(self, result: Any, *, context: str) -> Any:
        if not inspect.isawaitable(result):
            return result

        async def _await_result(awaitable: Any) -> Any:
            return await awaitable

        try:
            if self._service_loop is not None:
                try:
                    if self._service_loop.is_running():
                        future = asyncio.run_coroutine_threadsafe(
                            _await_result(result), self._service_loop
                        )
                        return future.result()
                except Exception:
                    # Jeśli pętla nie działa lub zgłosi błąd – fallback do lokalnego wykonania
                    pass

            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None

            if running_loop is not None:
                container: Dict[str, Any] = {}
                errors: List[BaseException] = []

                def _execute_in_thread() -> None:
                    try:
                        container["value"] = asyncio.run(_await_result(result))
                    except BaseException as thread_exc:  # pragma: no cover - propagacja błędu
                        errors.append(thread_exc)

                thread = threading.Thread(target=_execute_in_thread, daemon=True)
                thread.start()
                thread.join(timeout=30.0)
                if thread.is_alive():
                    raise TimeoutError("Timed out waiting for coroutine result")
                if errors:
                    raise errors[0]
                return container.get("value")

            return asyncio.run(_await_result(result))
        except Exception as exc:
            message = f"{context} coroutine failed: {exc!r}"
            try:
                self.emitter.log(message, level="ERROR", component="AutoTrader")
            except Exception:
                logger.exception("Emitter failed while logging coroutine error")
            logger.error("Coroutine execution failed during %s", context, exc_info=exc)
            return None

    def _obtain_prediction(
        self,
        ai: Any,
        symbol: str,
        timeframe: str,
        ex: Any,
    ) -> Tuple[Optional[float], Optional[pd.DataFrame], Optional[float]]:
        predict_fn = getattr(ai, "predict_series", None)
        if not callable(predict_fn):
            return (None, None, None)

        df: Optional[pd.DataFrame] = None
        last_price: Optional[float] = None
        last_pred: Optional[float] = None

        try:
            sig = inspect.signature(predict_fn)
            params = sig.parameters
        except (TypeError, ValueError):
            sig = None
            params = {}

        # 1) Spróbuj wywołania na podstawie symbolu/bars – zgodność z prostym API
        if "symbol" in params and last_pred is None:
            kwargs: Dict[str, Any] = {"symbol": symbol}
            if "timeframe" in params:
                kwargs["timeframe"] = timeframe
            if "bars" in params:
                kwargs["bars"] = 256
            elif "limit" in params:
                kwargs["limit"] = 256
            try:
                preds = predict_fn(**kwargs)
                preds = self._resolve_prediction_result(
                    preds, context=f"predict_series[{symbol}]"
                )
                last_pred = self._extract_last_pred(preds)
            except Exception:
                last_pred = None

        # 2) Klasyczny wariant – przekazanie DataFrame z OHLCV
        if last_pred is None:
            df = self._ensure_dataframe(symbol, timeframe, ex)
            if df is not None and not df.empty:
                last_price = float(df["close"].iloc[-1])
                call_attempts = []
                if sig is None or "feature_cols" in params:
                    call_attempts.append({"feature_cols": ["open", "high", "low", "close", "volume"]})
                call_attempts.append({})  # bez dodatkowych argumentów
                for extra in call_attempts:
                    try:
                        preds = predict_fn(df, **extra)
                        preds = self._resolve_prediction_result(
                            preds, context=f"predict_series[{symbol}]"
                        )
                        last_pred = self._extract_last_pred(preds)
                    except TypeError:
                        # jeśli feature_cols niepasuje – spróbuj kolejnego wariantu
                        continue
                    except Exception:
                        continue
                    if last_pred is not None:
                        break

        if last_price is None:
            last_price = self._resolve_market_price(symbol, ex, df)

        return (last_pred, df, last_price)

    def _ensure_dataframe(
        self, symbol: str, timeframe: str, ex: Any
    ) -> Optional[pd.DataFrame]:
        if self._market_data_provider is not None:
            try:
                from KryptoLowca.data.market_data import MarketDataRequest

                request = MarketDataRequest(symbol=symbol, timeframe=timeframe, limit=256)
                df = self._market_data_provider.get_historical(request)
                return df
            except Exception as exc:
                self.emitter.log(
                    f"MarketDataProvider failed: {exc!r}",
                    level="ERROR",
                    component="AutoTrader",
                )
                logger.exception("MarketDataProvider.get_historical failed")
        if ex is None or not hasattr(ex, "fetch_ohlcv"):
            return None
        try:
            raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=256) or []
        except Exception as exc:
            self.emitter.log(
                f"fetch_ohlcv failed: {exc!r}", level="ERROR", component="AutoTrader"
            )
            return None
        if not raw:
            return None
        try:
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        except Exception:
            df = pd.DataFrame(raw)
            expected = ["timestamp", "open", "high", "low", "close", "volume"]
            for idx, col in enumerate(df.columns):
                if idx < len(expected):
                    df.rename(columns={col: expected[idx]}, inplace=True)
        return df

    def _resolve_market_price(
        self, symbol: str, ex: Any, df: Optional[pd.DataFrame]
    ) -> Optional[float]:
        if self._market_data_provider is not None:
            price = self._market_data_provider.get_latest_price(symbol)
            if price is not None:
                return float(price)
        if df is not None and not df.empty and "close" in df.columns:
            try:
                return float(df["close"].iloc[-1])
            except Exception:
                pass
        if ex is None:
            return None
        if hasattr(ex, "fetch_ticker"):
            try:
                ticker = ex.fetch_ticker(symbol) or {}
                for key in ("last", "close", "bid", "ask"):
                    val = ticker.get(key)
                    if val is not None:
                        return float(val)
            except Exception as exc:
                self.emitter.log(
                    f"fetch_ticker failed: {exc!r}", level="ERROR", component="AutoTrader"
                )
        return None

    @staticmethod
    def _extract_last_pred(preds: Any) -> Optional[float]:
        if preds is None:
            return None
        series: Optional[pd.Series]
        if isinstance(preds, pd.Series):
            series = preds
        elif isinstance(preds, pd.DataFrame):
            if preds.empty:
                return None
            series = preds.iloc[:, -1]
        else:
            try:
                series = pd.Series(list(preds))
            except Exception:
                return None
        if series is None:
            return None
        series = series.dropna()
        if series.empty:
            return None
        try:
            return float(series.iloc[-1])
        except Exception:
            return None

    # --- Strategy & risk helpers -------------------------------------------------
    def _update_strategy_config(self, value: Any) -> None:
        try:
            if isinstance(value, StrategyConfig):
                cfg = value.validate()
            elif isinstance(value, str):
                cfg = StrategyConfig.from_preset(value)
            elif isinstance(value, dict):
                cfg = StrategyConfig(**value).validate()
            else:
                raise TypeError("Nieobsługiwany format konfiguracji strategii")
            if cfg.mode == "live":
                backtest_ts = getattr(cfg, "backtest_passed_at", None)
                freshness_window = getattr(cfg, "BACKTEST_VALIDITY_WINDOW_S", 0.0)
                reason: Optional[str] = None
                if not backtest_ts:
                    reason = "brak potwierdzonego backtestu"
                else:
                    age = time.time() - float(backtest_ts)
                    if freshness_window and age > float(freshness_window):
                        hours = max(1, int(freshness_window // 3600))
                        reason = (
                            "wynik backtestu jest przeterminowany (starszy niż "
                            f"{hours}h)"
                        )
                if reason:
                    message = (
                        "Odrzucono przełączenie strategii w tryb LIVE – "
                        f"{reason}. Uruchom backtest i ponów próbę."
                    )
                    self.emitter.log(message, level="WARNING", component="AutoTrader")
                    logger.warning("%s", message)
                    return
        except Exception as exc:  # pragma: no cover - logujemy i utrzymujemy stare ustawienia
            self.emitter.log(
                f"Nieprawidłowa konfiguracja strategii: {exc!r}",
                level="ERROR",
                component="AutoTrader",
            )
            logger.exception("Strategy config update failed")
            return
        if cfg.mode == "live":
            passed_at = cfg.backtest_passed_at or 0.0
            now_ts = time.time()
            if passed_at <= 0:
                message = (
                    "Odrzucono przełączenie strategii w tryb LIVE: brak potwierdzonego backtestu."
                )
                self.emitter.log(message, level="WARNING", component="AutoTrader")
                logger.warning(message)
                return
            if now_ts - passed_at > self.BACKTEST_GUARD_MAX_AGE_S:
                message = (
                    "Odrzucono przełączenie strategii w tryb LIVE: wynik backtestu jest przestarzały."
                )
                self.emitter.log(message, level="WARNING", component="AutoTrader")
                logger.warning(message)
                return
        with self._lock:
            self._strategy_config = cfg
            self._strategy_override = True
        self.emitter.log(
            f"Strategia zaktualizowana: {cfg.preset} mode={cfg.mode} max_notional={cfg.max_position_notional_pct}",
            level="INFO",
            component="AutoTrader",
        )

    def _get_strategy_config(self) -> StrategyConfig:
        cfg = self._strategy_config
        if self._strategy_override:
            return cfg
        cfg_manager = getattr(self.gui, "cfg", None)
        loader = getattr(cfg_manager, "load_strategy_config", None) if cfg_manager else None
        if callable(loader):
            try:
                loaded = loader()
                if isinstance(loaded, StrategyConfig):
                    cfg = loaded.validate()
                elif isinstance(loaded, dict):
                    cfg = StrategyConfig(**loaded).validate()
                self._strategy_config_error_notified = False
            except Exception as exc:  # pragma: no cover - unikamy zalewania logów
                if not self._strategy_config_error_notified:
                    self.emitter.log(
                        f"Nie udało się wczytać konfiguracji strategii: {exc!r}",
                        level="WARNING",
                        component="AutoTrader",
                    )
                    logger.warning("Failed to refresh strategy config", exc_info=True)
                    self._strategy_config_error_notified = True
        cfg_applied = self._apply_runtime_risk_budget(cfg)
        self._strategy_config = cfg_applied
        return cfg_applied

    @staticmethod
    def _build_signal_payload(symbol: str, side: str, prediction: Optional[float]) -> Dict[str, Any]:
        try:
            pred_value = float(prediction) if prediction is not None else 0.0
        except Exception:
            pred_value = 0.0
        direction = "LONG" if str(side).upper() == "BUY" else "SHORT"
        strength = abs(pred_value)
        confidence = min(1.0, max(0.0, strength * 10.0))
        return {
            "symbol": symbol,
            "direction": direction,
            "prediction": pred_value,
            "strength": strength,
            "confidence": confidence,
        }

    def _evaluate_risk(
        self,
        symbol: str,
        side: str,
        price: float,
        signal_payload: Dict[str, Any],
        market_df: Optional[pd.DataFrame],
    ) -> RiskDecision:
        with self._lock:
            strategy_cfg = self._get_strategy_config()
        now = time.time()
        side_u = side.upper()

        ro_until = self._reduce_only_until.get(symbol, 0.0)
        if ro_until and now < ro_until and side_u == "BUY":
            details = {"until": ro_until, "now": now, "policy": "reduce_only"}
            return self._finalize_risk_decision(
                symbol,
                side,
                price,
                RiskDecision(
                    should_trade=False,
                    fraction=0.0,
                    state="lock",
                    reason="reduce_only_active",
                    details=details,
                    stop_loss_pct=strategy_cfg.default_sl,
                    take_profit_pct=strategy_cfg.default_tp,
                    mode=strategy_cfg.mode,
                ),
                None,
            )

        if self._risk_lock_until and now < self._risk_lock_until and side_u == "BUY":
            details = {"until": self._risk_lock_until, "now": now, "policy": "cooldown"}
            return self._finalize_risk_decision(
                symbol,
                side,
                price,
                RiskDecision(
                    should_trade=False,
                    fraction=0.0,
                    state="lock",
                    reason="cooldown_active",
                    details=details,
                    stop_loss_pct=strategy_cfg.default_sl,
                    take_profit_pct=strategy_cfg.default_tp,
                    mode=strategy_cfg.mode,
                ),
                None,
            )

        env_mode = self._resolve_mode()
        if strategy_cfg.mode == "demo" and env_mode != "paper":
            details = {"configured_mode": strategy_cfg.mode, "env_mode": env_mode}
            return self._finalize_risk_decision(
                symbol,
                side,
                price,
                RiskDecision(
                    should_trade=False,
                    fraction=0.0,
                    state="lock",
                    reason="demo_mode_enforced",
                    details=details,
                    stop_loss_pct=strategy_cfg.default_sl,
                    take_profit_pct=strategy_cfg.default_tp,
                    mode=strategy_cfg.mode,
                ),
                None,
            )

        compliance_state = {
            "compliance_confirmed": bool(strategy_cfg.compliance_confirmed),
            "api_keys_configured": bool(strategy_cfg.api_keys_configured),
            "acknowledged_risk_disclaimer": bool(
                strategy_cfg.acknowledged_risk_disclaimer
            ),
        }
        if strategy_cfg.mode == "live":
            missing_checks = [name for name, ok in compliance_state.items() if not ok]
            if missing_checks:
                summary = ", ".join(missing_checks)
                log_message = (
                    "Live trading blocked: missing compliance confirmations -> "
                    f"{summary}"
                )
                self.emitter.log(log_message, level="WARNING", component="AutoTrader")
                return self._finalize_risk_decision(
                    symbol,
                    side,
                    price,
                    RiskDecision(
                        should_trade=False,
                        fraction=0.0,
                        state="lock",
                        reason="live_compliance_missing",
                        details={
                            "missing_checks": missing_checks,
                            "compliance_state": compliance_state,
                        },
                        stop_loss_pct=strategy_cfg.default_sl,
                        take_profit_pct=strategy_cfg.default_tp,
                        mode=strategy_cfg.mode,
                    ),
                    None,
                )

        portfolio_ctx = self._build_portfolio_context(symbol, price)
        risk_mgr = getattr(self.gui, "risk_mgr", None)
        try:
            base_fraction = float(strategy_cfg.trade_risk_pct)
        except Exception:
            base_fraction = 0.0
        fraction = base_fraction
        details: Dict[str, Any] = {}
        risk_engine_snapshot: Optional[Dict[str, Any]] = None

        try:
            positions_ctx = portfolio_ctx.get("positions") or {}
            open_positions = sum(
                1 for entry in positions_ctx.values() if (entry or {}).get("size")
            )
            prometheus_metrics.set_open_positions(count=open_positions, mode=strategy_cfg.mode)
        except Exception:
            logger.debug("Nie udało się ustawić metryki open_positions", exc_info=True)

        stop_loss_pct = strategy_cfg.default_sl
        take_profit_pct = strategy_cfg.default_tp

        market_payload: Any
        if isinstance(market_df, pd.DataFrame):
            market_payload = market_df
        else:
            market_payload = {"price": price}

        if risk_mgr is not None and hasattr(risk_mgr, "calculate_position_size"):
            try:
                try:
                    result = risk_mgr.calculate_position_size(
                        symbol=symbol,
                        signal=signal_payload,
                        market_data=market_payload,
                        portfolio=portfolio_ctx,
                        return_details=True,
                    )
                except TypeError as exc:
                    logger.warning(
                        "Risk manager %s signature rejected keyword invocation (%s); retrying positional",
                        type(risk_mgr).__name__,
                        exc,
                    )
                    result = risk_mgr.calculate_position_size(
                        symbol,
                        signal_payload,
                        market_payload,
                        portfolio_ctx,
                    )

                fraction_val, details_val, sl_override, tp_override = self._normalize_risk_result(result)
                if fraction_val is not None:
                    fraction = fraction_val
                if details_val:
                    details = details_val
                if sl_override is not None:
                    stop_loss_pct = sl_override
                if tp_override is not None:
                    take_profit_pct = tp_override
            except Exception as exc:
                self.emitter.log(
                    f"Risk sizing error: {exc!r}", level="ERROR", component="AutoTrader"
                )
                logger.exception("Risk manager calculate_position_size failed")
                fraction = 0.0
                details = {"error": str(exc)}
            else:
                if hasattr(risk_mgr, "latest_guard_state"):
                    try:
                        snapshot_candidate = risk_mgr.latest_guard_state()
                    except Exception:
                        logger.debug(
                            "Risk manager latest_guard_state failed", exc_info=True
                        )
                    else:
                        if isinstance(snapshot_candidate, Mapping):
                            risk_engine_snapshot = {
                                str(key): value for key, value in snapshot_candidate.items()
                            }
        else:
            details["risk_mgr"] = "missing"

        adjustments_raw = details.get("adjustments") if isinstance(details, dict) else None
        adjustments: Optional[Dict[str, Any]] = None
        if isinstance(adjustments_raw, Mapping):
            adjustments = {str(key): value for key, value in adjustments_raw.items()}

        metadata_raw = details.get("metadata") if isinstance(details, dict) else None
        metadata_details: Optional[Dict[str, Any]] = None
        if isinstance(metadata_raw, Mapping):
            metadata_details = {str(key): value for key, value in metadata_raw.items()}

        try:
            fraction = float(fraction)
        except Exception:
            fraction = 0.0
        fraction = max(0.0, min(1.0, fraction))

        recommended_fraction = fraction
        if isinstance(details, dict):
            candidate_fraction = details.get("recommended_size")
            if candidate_fraction is None:
                candidate_fraction = details.get("fraction") or details.get("size")
            if candidate_fraction is not None:
                try:
                    recommended_fraction = float(candidate_fraction)
                except Exception:
                    recommended_fraction = fraction
        recommended_fraction = max(0.0, min(1.0, recommended_fraction))
        details["recommended_size"] = recommended_fraction
        fraction = recommended_fraction

        risk_allowed: Optional[bool] = None
        risk_reason_text: Optional[str] = None
        if isinstance(details, dict):
            raw_allowed = details.get("allowed")
            if raw_allowed is not None:
                risk_allowed = bool(raw_allowed)
            raw_reason = details.get("reason")
            if raw_reason is not None:
                risk_reason_text = str(raw_reason)
                details["risk_reason"] = risk_reason_text

        risk_engine_details: Dict[str, Any] = {}
        if risk_allowed is not None:
            risk_engine_details["allowed"] = risk_allowed
            details["risk_engine_allowed"] = risk_allowed
        if risk_reason_text:
            risk_engine_details["reason"] = risk_reason_text
        if adjustments:
            risk_engine_details["adjustments"] = {
                key: (
                    float(value)
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                    else value
                )
                for key, value in adjustments.items()
            }
        if metadata_details:
            risk_engine_details["metadata"] = metadata_details
        if risk_engine_snapshot:
            risk_engine_details["state"] = risk_engine_snapshot
        if risk_engine_details:
            details["risk_engine"] = risk_engine_details

        try:
            if stop_loss_pct is not None:
                stop_loss_pct = float(stop_loss_pct)
        except Exception:
            stop_loss_pct = strategy_cfg.default_sl

        try:
            if take_profit_pct is not None:
                take_profit_pct = float(take_profit_pct)
        except Exception:
            take_profit_pct = strategy_cfg.default_tp

        state = "ok"
        limit_events: List[Dict[str, Any]] = []

        if risk_allowed is False:
            limit_event = {"type": "risk_engine_denied"}
            if risk_reason_text:
                limit_event["reason"] = risk_reason_text
            limit_events.append(limit_event)
            details.setdefault("limit_events", []).extend(limit_events)
            if strategy_cfg.reduce_only_after_violation:
                self._trigger_reduce_only(symbol, "risk_engine_denied", strategy_cfg)
            return self._finalize_risk_decision(
                symbol,
                side,
                price,
                RiskDecision(
                    should_trade=False,
                    fraction=fraction,
                    state="lock",
                    reason="risk_engine_denied",
                    details=details,
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct,
                    mode=strategy_cfg.mode,
                ),
                portfolio_ctx,
            )

        if adjustments:
            limit_events.append(
                {
                    "type": "risk_engine_adjustment",
                    "adjustments": {
                        key: (
                            float(value)
                            if isinstance(value, (int, float)) and not isinstance(value, bool)
                            else value
                        )
                        for key, value in adjustments.items()
                    },
                }
            )
            if state != "lock":
                state = "warn"

        if base_fraction > 0 and fraction < base_fraction - 1e-9:
            limit_events.append(
                {
                    "type": "risk_engine_clamp",
                    "baseline": base_fraction,
                    "value": fraction,
                }
            )
            if state != "lock":
                state = "warn"

        # limit pozycyjny względem notional
        max_pct = float(strategy_cfg.max_position_notional_pct)
        if max_pct > 0.0 and fraction > max_pct:
            limit_events.append({
                "type": "max_position_notional_pct",
                "value": fraction,
                "threshold": max_pct,
            })
            fraction = max_pct
            state = "warn"

        account_value = self._resolve_account_value(portfolio_ctx)
        positions = portfolio_ctx.get("positions") or {}
        position_ctx = positions.get(symbol, {})
        symbol_notional = float(position_ctx.get("notional", 0.0) or 0.0)
        total_notional = float(portfolio_ctx.get("total_notional", 0.0) or 0.0)

        if account_value <= 0.0:
            if strategy_cfg.reduce_only_after_violation:
                self._trigger_reduce_only(symbol, "no_account_value", strategy_cfg)
            details.update({"account_value": account_value, "portfolio_ctx": portfolio_ctx})
            return self._finalize_risk_decision(
                symbol,
                side,
                price,
                RiskDecision(
                    should_trade=False,
                    fraction=0.0,
                    state="lock",
                    reason="account_value_non_positive",
                    details=details,
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct,
                    mode=strategy_cfg.mode,
                ),
                portfolio_ctx,
            )

        projected_notional = total_notional
        if side_u == "BUY":
            projected_notional += fraction * account_value
        else:
            projected_notional = max(total_notional - symbol_notional, 0.0)

        leverage_after = projected_notional / max(account_value, 1e-9)
        if side_u == "BUY" and leverage_after > strategy_cfg.max_leverage + 1e-6:
            limit_events.append({
                "type": "max_leverage",
                "value": leverage_after,
                "threshold": strategy_cfg.max_leverage,
            })
            if strategy_cfg.reduce_only_after_violation:
                self._trigger_reduce_only(symbol, "max_leverage", strategy_cfg)
            return self._finalize_risk_decision(
                symbol,
                side,
                price,
                RiskDecision(
                    should_trade=False,
                    fraction=0.0,
                    state="lock",
                    reason="max_leverage_exceeded",
                    details={
                        "leverage_after": leverage_after,
                        "max_leverage": strategy_cfg.max_leverage,
                        "limit_events": limit_events,
                    },
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct,
                    mode=strategy_cfg.mode,
                ),
                portfolio_ctx,
            )

        if fraction <= 0.0:
            if strategy_cfg.reduce_only_after_violation:
                self._trigger_reduce_only(symbol, "fraction_non_positive", strategy_cfg)
            details["limit_events"] = limit_events
            return self._finalize_risk_decision(
                symbol,
                side,
                price,
                RiskDecision(
                    should_trade=False,
                    fraction=0.0,
                    state="lock",
                    reason="risk_fraction_zero",
                    details=details,
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct,
                    mode=strategy_cfg.mode,
                ),
                portfolio_ctx,
            )

        if side_u == "SELL" and ro_until and now < ro_until:
            # pozwól zamknąć pozycję i wyczyść reduce-only
            self._reduce_only_until.pop(symbol, None)
            details["reduce_only_cleared"] = True

        if limit_events:
            details.setdefault("limit_events", []).extend(limit_events)
            if state != "lock":
                state = "warn"

        decision_details = {
            **details,
            "account_value": account_value,
            "projected_notional": projected_notional,
            "current_notional": total_notional,
            "symbol_notional": symbol_notional,
        }

        decision = RiskDecision(
            should_trade=True,
            fraction=fraction,
            state=state,
            reason="risk_ok" if state == "ok" else "risk_clamped",
            details=decision_details,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            mode=strategy_cfg.mode,
        )
        self._apply_violation_cooldown(symbol, side_u, strategy_cfg, decision)
        return self._finalize_risk_decision(
            symbol,
            side,
            price,
            decision,
            portfolio_ctx,
        )

    def _finalize_risk_decision(
        self,
        symbol: str,
        side: str,
        price: float,
        decision: RiskDecision,
        portfolio_ctx: Mapping[str, Any] | None,
    ) -> RiskDecision:
        try:
            self._record_risk_decision(
                symbol=symbol,
                side=side,
                price=price,
                decision=decision,
                portfolio_ctx=portfolio_ctx,
            )
        except Exception:
            logger.debug("Risk decision logging failed", exc_info=True)
        return decision

    def _record_risk_decision(
        self,
        *,
        symbol: str,
        side: str,
        price: float,
        decision: RiskDecision,
        portfolio_ctx: Mapping[str, Any] | None,
    ) -> None:
        log = getattr(self, "_risk_decision_log", None)
        if log is None:
            return

        profile = (
            self._risk_profile_name
            or getattr(self._risk_manager_settings, "profile_name", None)
            or self._core_risk_profile
            or "default"
        )

        try:
            fraction = float(decision.fraction)
        except Exception:
            fraction = 0.0

        allowed = bool(decision.should_trade and decision.state != "lock")

        details: Mapping[str, Any] | None = None
        if isinstance(decision.details, Mapping):
            details = decision.details

        account_value: float | None = None
        if details is not None:
            candidate = details.get("account_value") or details.get("portfolio_value")
            try:
                if candidate is not None:
                    account_value = float(candidate)
            except (TypeError, ValueError):
                account_value = None

        if account_value is None and portfolio_ctx is not None:
            try:
                account_value = float(self._resolve_account_value(portfolio_ctx))
            except Exception:
                account_value = None

        notional: float | None = None
        quantity = 0.0
        if account_value is not None and account_value > 0:
            notional = max(0.0, account_value * fraction)
            if price > 0:
                try:
                    quantity = notional / price
                except Exception:
                    quantity = 0.0

        adjustments: Dict[str, Any] | None = None
        if details is not None:
            risk_info = details.get("risk_engine")
            if isinstance(risk_info, Mapping):
                candidate = risk_info.get("adjustments")
                if isinstance(candidate, Mapping):
                    try:
                        adjustments = {
                            str(key): (
                                float(value)
                                if isinstance(value, (int, float)) and not isinstance(value, bool)
                                else value
                            )
                            for key, value in candidate.items()
                        }
                    except Exception:
                        adjustments = None

        metadata: Dict[str, Any] = {
            "mode": decision.mode,
            "state": decision.state,
            "reason": decision.reason,
            "stop_loss_pct": decision.stop_loss_pct,
            "take_profit_pct": decision.take_profit_pct,
            "source": "auto_trader",
        }
        if details is not None:
            metadata["details"] = dict(details)
            if "limit_events" in details:
                metadata["limit_events"] = details.get("limit_events")

        log.record(
            profile=str(profile or "default"),
            symbol=self._normalize_symbol(symbol),
            side=str(side).lower(),
            quantity=max(0.0, float(quantity)),
            price=price if price > 0 else None,
            notional=notional,
            allowed=allowed,
            reason=decision.reason,
            adjustments=adjustments,
            metadata=metadata,
        )

    def _emit_risk_audit(self, symbol: str, side: str, decision: RiskDecision, price: float) -> None:
        payload = {
            "symbol": symbol,
            "side": side,
            "state": decision.state,
            "reason": decision.reason,
            "fraction": float(decision.fraction),
            "price": float(price),
            "mode": decision.mode,
            "details": decision.details,
            "stop_loss_pct": decision.stop_loss_pct,
            "take_profit_pct": decision.take_profit_pct,
            "ts": time.time(),
            "schema_version": 1,
        }
        self._last_risk_audit = payload
        try:
            prometheus_metrics.observe_risk(symbol, decision.state, decision.fraction, decision.mode)
        except Exception:
            logger.debug("Prometheus observe_risk skipped", exc_info=True)
        try:
            self.emitter.emit("risk_guard_event", **payload)
        except Exception:  # pragma: no cover - audyt nie może zatrzymać bota
            logger.exception("Failed to emit risk_guard_event")

        db_manager = self._resolve_db()
        if db_manager is not None:
            limit_events: Optional[List[str]] = None
            if isinstance(decision.details, dict):
                candidate = decision.details.get("limit_events")
                if isinstance(candidate, (list, tuple)):
                    limit_events = [str(item) for item in candidate]
            db_payload = {
                "symbol": symbol,
                "state": decision.state,
                "fraction": float(decision.fraction),
                "side": side,
                "reason": decision.reason,
                "price": float(price),
                "mode": decision.mode,
                "limit_events": limit_events,
                "details": decision.details,
                "stop_loss_pct": decision.stop_loss_pct,
                "take_profit_pct": decision.take_profit_pct,
                "should_trade": decision.should_trade,
            }
            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop is not None:
                    async_method = getattr(db_manager, "log_risk_audit", None)
                    if callable(async_method):
                        result = async_method(db_payload)
                        if inspect.isawaitable(result):
                            task = loop.create_task(result)

                            def _handle_task(t: asyncio.Task[Any]) -> None:
                                try:
                                    t.result()
                                except Exception:  # pragma: no cover - logowanie awarii w tle
                                    logger.exception("Async risk audit log failed")

                            task.add_done_callback(_handle_task)
                        else:
                            logger.debug("Async log_risk_audit returned non-awaitable result")
                    else:
                        logger.debug("No async log_risk_audit available on db manager")
                else:
                    sync = getattr(db_manager, "sync", None)
                    log_method = None
                    if sync is not None:
                        log_method = getattr(sync, "log_risk_audit", None)
                    if log_method is None:
                        log_method = getattr(db_manager, "log_risk_audit", None)
                    if callable(log_method):
                        result = log_method(db_payload)
                        if inspect.isawaitable(result):
                            asyncio.run(result)
                    else:
                        logger.debug("No log_risk_audit method available on db manager")
            except Exception:  # pragma: no cover - logowanie awarii
                logger.exception("Failed to persist risk audit log")

        msg = (
            f"Risk state={decision.state} reason={decision.reason} symbol={symbol} side={side} fraction={decision.fraction:.4f}"
        )
        level = "INFO"
        if decision.state == "warn":
            level = "WARNING"
        elif decision.state == "lock":
            level = "WARNING" if decision.should_trade else "ERROR"
        self.emitter.log(msg, level=level, component="AutoTrader")
        if decision.state != "ok":
            severity = AlertSeverity.WARNING if decision.state == "warn" else AlertSeverity.ERROR
            _emit_alert(
                f"Risk guard {decision.state} ({decision.reason}) dla {symbol}",
                severity=severity,
                source="risk_guard",
                context={
                    "symbol": symbol,
                    "side": side,
                    "fraction": float(decision.fraction),
                    "state": decision.state,
                    "reason": decision.reason,
                    "limit_events": decision.details.get("limit_events"),
                    "cooldown_until": decision.details.get("cooldown_until"),
                },
            )

    @staticmethod
    def _normalize_risk_result(
        result: Any,
    ) -> Tuple[Optional[float], Dict[str, Any], Optional[float], Optional[float]]:
        fraction: Optional[float] = None
        details: Dict[str, Any] = {}
        stop_loss_override: Optional[float] = None
        take_profit_override: Optional[float] = None

        if hasattr(result, "recommended_size"):
            try:
                recommended = float(getattr(result, "recommended_size", 0.0))
            except Exception:
                recommended = 0.0
            fraction = recommended
            details = {
                "recommended_size": recommended,
                "max_allowed_size": float(
                    getattr(result, "max_allowed_size", recommended) or recommended
                ),
                "kelly_size": float(getattr(result, "kelly_size", recommended) or recommended),
                "risk_adjusted_size": float(
                    getattr(result, "risk_adjusted_size", recommended) or recommended
                ),
            }
            confidence = getattr(result, "confidence_level", None)
            if confidence is not None:
                try:
                    details["confidence_level"] = float(confidence)
                except Exception:
                    details["confidence_level"] = confidence
            reasoning = getattr(result, "reasoning", None)
            if reasoning is not None:
                details["reasoning"] = reasoning
            for attr_name in ("stop_loss_pct", "stop_loss"):
                sl_value = getattr(result, attr_name, None)
                if sl_value is not None:
                    try:
                        stop_loss_override = float(sl_value)
                        break
                    except Exception:
                        continue
            for attr_name in ("take_profit_pct", "take_profit"):
                tp_value = getattr(result, attr_name, None)
                if tp_value is not None:
                    try:
                        take_profit_override = float(tp_value)
                        break
                    except Exception:
                        continue
        elif isinstance(result, tuple) and len(result) == 2:
            fraction, details_val = result
            if isinstance(details_val, dict):
                details = dict(details_val)
            else:
                details = {"details": details_val}
        elif isinstance(result, dict):
            details = dict(result)
            raw_fraction = details.get(
                "recommended_size",
                details.get("fraction", details.get("size")),
            )
            if raw_fraction is not None:
                try:
                    fraction = float(raw_fraction)
                except Exception:
                    fraction = None
        else:
            fraction = result if result is not None else None

        try:
            if fraction is not None:
                fraction = float(fraction)
        except Exception:
            fraction = None

        if fraction is not None:
            details.setdefault("recommended_size", fraction)

        return fraction, details, stop_loss_override, take_profit_override

    def _resolve_account_value(self, portfolio_ctx: Dict[str, Any]) -> float:
        account_value = portfolio_ctx.get("equity")
        try:
            if account_value is not None:
                return float(account_value)
        except Exception:
            pass
        cash = portfolio_ctx.get("cash")
        try:
            if cash is not None:
                return float(cash)
        except Exception:
            pass
        candidates = ["paper_balance", "account_balance", "equity", "cash"]
        for attr in candidates:
            if hasattr(self.gui, attr):
                try:
                    return float(getattr(self.gui, attr))
                except Exception:
                    continue
        return 0.0

    def _build_portfolio_context(self, symbol: str, ref_price: float) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "positions": {},
            "total_notional": 0.0,
        }
        raw_positions = getattr(self.gui, "_open_positions", None)
        if not isinstance(raw_positions, dict):
            raw_positions = getattr(self.gui, "open_positions", None)
        if isinstance(raw_positions, dict):
            for sym, pos in raw_positions.items():
                try:
                    qty = float(pos.get("qty", 0.0) or 0.0)
                    entry = float(pos.get("entry") or pos.get("price") or ref_price or 0.0)
                except Exception:
                    qty = 0.0
                    entry = ref_price or 0.0
                notional = abs(qty * entry)
                context["positions"][sym] = {
                    "qty": qty,
                    "entry": entry,
                    "side": str(pos.get("side", "")).upper(),
                    "notional": notional,
                }
                context["total_notional"] += notional

        cash_candidates = [
            ("paper_balance", getattr(self.gui, "paper_balance", None)),
            ("account_balance", getattr(self.gui, "account_balance", None)),
        ]
        cash_value = 0.0
        for name, val in cash_candidates:
            try:
                if val is not None:
                    cash_value = float(val)
                    break
            except Exception:
                continue
        context["cash"] = cash_value
        context["equity"] = max(cash_value, cash_value + context["total_notional"])
        return context

    def _trigger_reduce_only(self, symbol: str, reason: str, cfg: StrategyConfig) -> None:
        cooldown = max(float(cfg.violation_cooldown_s), 1.0)
        until = time.time() + cooldown
        self._reduce_only_until[symbol] = until
        self._risk_lock_until = max(self._risk_lock_until, until)
        self.emitter.log(
            f"Reduce-only aktywne dla {symbol} przez {cooldown:.0f}s (powód: {reason})",
            level="WARNING",
            component="AutoTrader",
        )
        _emit_alert(
            f"Reduce-only dla {symbol} przez {cooldown:.0f}s",
            severity=AlertSeverity.ERROR,
            source="risk_guard",
            context={
                "symbol": symbol,
                "reason": reason,
                "cooldown_seconds": cooldown,
                "cooldown_until": until,
            },
        )

    def _apply_violation_cooldown(
        self,
        symbol: str,
        side: str,
        cfg: StrategyConfig,
        decision: RiskDecision,
    ) -> None:
        if side != "BUY":
            return
        if decision.state not in {"warn", "lock"}:
            return

        cooldown = max(float(cfg.violation_cooldown_s), 1.0)
        now = time.time()
        previous_lock = self._risk_lock_until
        proposed_until = now + cooldown
        existing_until = float(decision.details.get("cooldown_until", 0.0) or 0.0)
        if existing_until and existing_until >= proposed_until:
            decision.details.setdefault("cooldown_seconds", cooldown)
            decision.details["cooldown_until"] = existing_until
            self._risk_lock_until = max(self._risk_lock_until, existing_until)
            return

        until = proposed_until
        decision.details.setdefault("cooldown_seconds", cooldown)
        decision.details["cooldown_until"] = until
        self._risk_lock_until = max(self._risk_lock_until, until)

        if previous_lock >= self._risk_lock_until:
            return

        self.emitter.log(
            f"Aktywowano cooldown po naruszeniu limitów ({decision.state}) do {until:.0f}",
            level="WARNING",
            component="AutoTrader",
        )
        _emit_alert(
            f"Cooldown ryzyka dla {symbol}",
            severity=AlertSeverity.WARNING if decision.state == "warn" else AlertSeverity.ERROR,
            source="risk_guard",
            context={
                "symbol": symbol,
                "state": decision.state,
                "reason": decision.reason,
                "cooldown_seconds": cooldown,
                "cooldown_until": until,
            },
        )
