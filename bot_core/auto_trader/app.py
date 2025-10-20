"""Lightweight auto-trading controller used by tests and runtime scaffolding.

This module re-implements the bare minimum of the legacy ``AutoTrader``
behaviour in a dependency-free manner so that it can operate without the
monolithic application package.  The original implementation pulled a large
amount of infrastructure (event emitters, Prometheus exporters, runtime
services).  For unit tests we only need predictable threading semantics and
state transitions.

The implementation below focuses on deterministic start/stop logic, manual
activation flow and logging hooks.  It still exposes a small ``RiskDecision``
structure for compatibility with code that serialises decisions.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Protocol

import pandas as pd

from bot_core.ai.regime import (
    MarketRegime,
    MarketRegimeAssessment,
    MarketRegimeClassifier,
    RegimeSummary,
    RiskLevel,
)
from bot_core.ai.config_loader import load_risk_thresholds


LOGGER = logging.getLogger(__name__)


class EmitterLike(Protocol):
    """Minimal protocol expected from GUI/event emitter integrations."""

    def on(self, event: str, callback: Callable[..., Any], *, tag: str | None = None) -> None:
        ...  # pragma: no cover - optional interface used by runtime only

    def off(self, event: str, *, tag: str | None = None) -> None:
        ...  # pragma: no cover - optional interface used by runtime only

    def emit(self, event: str, **payload: Any) -> None:
        ...  # pragma: no cover - optional interface used by runtime only

    def log(self, message: str, *args: Any, **kwargs: Any) -> None:
        ...


@dataclass(slots=True)
class RiskDecision:
    """Serializable snapshot describing the outcome of a risk engine check."""

    should_trade: bool
    fraction: float
    state: str
    reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    mode: str = "demo"
    cooldown_active: bool = False
    cooldown_remaining_s: Optional[float] = None
    cooldown_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "should_trade": self.should_trade,
            "fraction": float(self.fraction),
            "state": self.state,
            "reason": self.reason,
            "details": dict(self.details),
            "mode": self.mode,
        }
        if self.stop_loss_pct is not None:
            payload["stop_loss_pct"] = float(self.stop_loss_pct)
        if self.take_profit_pct is not None:
            payload["take_profit_pct"] = float(self.take_profit_pct)
        payload["cooldown_active"] = self.cooldown_active
        if self.cooldown_remaining_s is not None:
            payload["cooldown_remaining_s"] = float(self.cooldown_remaining_s)
        if self.cooldown_reason is not None:
            payload["cooldown_reason"] = self.cooldown_reason
        return payload


@dataclass(slots=True)
class GuardrailTrigger:
    """Structured details about a guardrail that forced a HOLD signal."""

    name: str
    label: str
    comparator: str
    threshold: float
    value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "label": self.label,
            "comparator": self.comparator,
            "threshold": float(self.threshold),
        }
        if self.value is not None:
            payload["value"] = float(self.value)
        return payload


class AutoTrader:
    """Small cooperative wrapper around an auto-trading loop.

    The class is intentionally tiny – it exists so that unit tests can exercise
    manual confirmation logic without pulling in the whole legacy runtime.  It
    exposes the same public attributes that the tests rely on (``enable_auto_trade``
    and ``_auto_trade_user_confirmed``) and uses an overridable ``_auto_trade_loop``
    method executed inside a worker thread when the user confirms auto-trading.
    """

    def __init__(
        self,
        emitter: EmitterLike,
        gui: Any,
        symbol_getter: Callable[[], str],
        pf_min: float = 1.3,
        expectancy_min: float = 0.0,
        metrics_window: int = 30,
        atr_ratio_threshold: float = 0.5,
        atr_baseline_len: int = 100,
        reopt_cooldown_s: int = 1800,
        walkforward_interval_s: Optional[int] = 3600,
        walkforward_min_closed_trades: int = 10,
        enable_auto_trade: bool = True,
        auto_trade_interval_s: float = 30.0,
        market_data_provider: Optional[Any] = None,
        *,
        signal_service: Optional[Any] = None,
        risk_service: Optional[Any] = None,
        execution_service: Optional[Any] = None,
        data_provider: Optional[Any] = None,
        bootstrap_context: Any | None = None,
        core_risk_engine: Any | None = None,
        core_execution_service: Any | None = None,
        ai_connector: Any | None = None,
        thresholds_loader: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self.emitter = emitter
        self.gui = gui
        self.symbol_getter = symbol_getter
        self.market_data_provider = market_data_provider

        self.enable_auto_trade = bool(enable_auto_trade)
        self.auto_trade_interval_s = float(auto_trade_interval_s)

        self.signal_service = signal_service
        self.risk_service = risk_service
        self.execution_service = execution_service
        self.data_provider = data_provider
        self.bootstrap_context = bootstrap_context
        self.core_risk_engine = core_risk_engine
        self.core_execution_service = core_execution_service
        self.ai_connector = ai_connector
        self.ai_manager: Any | None = getattr(gui, "ai_mgr", None)
        self._thresholds_loader: Callable[[], Mapping[str, Any]] = (
            thresholds_loader or load_risk_thresholds
        )
        self._thresholds: Mapping[str, Any] = {}
        self.reload_thresholds()

        self.current_strategy: str = "neutral"
        self.current_leverage: float = 1.0
        self.current_stop_loss_pct: float = 0.02
        self.current_take_profit_pct: float = 0.04
        self._last_signal: str | None = None
        self._last_regime: MarketRegimeAssessment | None = None
        self._last_risk_decision: RiskDecision | None = None
        self._cooldown_until: float = 0.0
        self._cooldown_reason: str | None = None
        self._last_guardrail_reasons: list[str] = []
        self._last_guardrail_triggers: list[GuardrailTrigger] = []

        self._stop = threading.Event()
        self._auto_trade_stop = threading.Event()
        self._auto_trade_thread: threading.Thread | None = None
        self._auto_trade_thread_active = False
        self._auto_trade_user_confirmed = False
        self._started = False
        self._lock = threading.RLock()

    def reload_thresholds(self) -> None:
        """Reload cached risk thresholds from the configured loader."""

        self._thresholds = self._thresholds_loader()

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def _log(self, message: str, *, level: int = logging.INFO, **kwargs: Any) -> None:
        if hasattr(self.emitter, "log"):
            try:
                self.emitter.log(message, level=logging.getLevelName(level), component="AutoTrader", **kwargs)
                return
            except Exception:  # pragma: no cover - defensive logging
                LOGGER.log(level, "Emitter logging failed", exc_info=True)
        LOGGER.log(level, message)

    def _run_auto_trade_thread(self) -> None:
        try:
            while not self._auto_trade_stop.is_set() and not self._stop.is_set():
                self._auto_trade_thread_active = True
                try:
                    self._auto_trade_loop()
                finally:
                    self._auto_trade_thread_active = False
                if self._auto_trade_stop.wait(self.auto_trade_interval_s):
                    break
        except Exception:  # pragma: no cover - keep thread resilient
            LOGGER.exception("Auto-trade loop crashed")
        finally:
            self._auto_trade_thread_active = False
            self._auto_trade_stop.set()

    def _start_auto_trade_thread_locked(self) -> None:
        if self._auto_trade_thread is not None and self._auto_trade_thread.is_alive():
            return
        self._auto_trade_stop.clear()
        self._auto_trade_thread = threading.Thread(
            target=self._run_auto_trade_thread,
            name="AutoTraderThread",
            daemon=True,
        )
        self._auto_trade_thread.start()

    def _cancel_auto_trade_thread_locked(self) -> None:
        self._auto_trade_stop.set()
        thread = self._auto_trade_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._auto_trade_thread = None
        self._auto_trade_thread_active = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._stop.clear()
            self._started = True
            if self.enable_auto_trade and not self._auto_trade_user_confirmed:
                self._log("Auto-trade awaiting explicit activation")
            if self.enable_auto_trade and self._auto_trade_user_confirmed:
                self._start_auto_trade_thread_locked()

    def confirm_auto_trade(self, flag: bool) -> None:
        with self._lock:
            self._auto_trade_user_confirmed = bool(flag)
            if not self._started or not self.enable_auto_trade:
                return
            if self._auto_trade_user_confirmed:
                self._start_auto_trade_thread_locked()
            else:
                self._cancel_auto_trade_thread_locked()

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            self._started = False
            self._stop.set()
            self._cancel_auto_trade_thread_locked()
            self._log("AutoTrader stopped.")

    # ------------------------------------------------------------------
    # Market intelligence helpers -------------------------------------
    # ------------------------------------------------------------------
    def _resolve_ai_manager(self) -> Any | None:
        if self.ai_manager is not None:
            return self.ai_manager
        candidate = getattr(self.gui, "ai_mgr", None)
        if candidate is not None:
            self.ai_manager = candidate
            return candidate
        if self.ai_connector is not None:
            return self.ai_connector
        return None

    def _fetch_market_data(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        provider = self.market_data_provider or self.data_provider
        if provider is None:
            return None

        def _coerce(result: Any) -> pd.DataFrame | None:
            if result is None:
                return None
            if isinstance(result, pd.DataFrame):
                return result
            try:
                df = pd.DataFrame(result)
            except Exception:
                return None
            return df

        if hasattr(provider, "get_historical"):
            getter = getattr(provider, "get_historical")
            try:
                return _coerce(getter(symbol=symbol, timeframe=timeframe, limit=256))
            except TypeError:
                try:
                    return _coerce(getter(symbol, timeframe, 256))
                except TypeError:
                    return _coerce(getter(symbol, timeframe))
        if callable(provider):
            try:
                return _coerce(provider(symbol=symbol, timeframe=timeframe))
            except TypeError:
                try:
                    return _coerce(provider(symbol, timeframe))
                except TypeError:
                    try:
                        return _coerce(provider(symbol))
                    except TypeError:
                        try:
                            return _coerce(provider())
                        except TypeError:
                            return None
        return None

    def _map_regime_to_signal(
        self,
        assessment: MarketRegimeAssessment,
        last_return: float,
        *,
        summary: RegimeSummary | None = None,
    ) -> str:
        cfg = self._thresholds["auto_trader"]["map_regime_to_signal"]
        if assessment.confidence < float(cfg.get("assessment_confidence", 0.2)):
            return "hold"
        if summary is not None and summary.confidence < float(cfg.get("summary_confidence", 0.45)):
            return "hold"
        if summary is not None and summary.stability < float(cfg.get("summary_stability", 0.4)):
            return "hold"
        if summary is not None and summary.risk_level in {RiskLevel.ELEVATED, RiskLevel.CRITICAL}:
            return "hold"
        if summary is not None and summary.risk_trend > float(cfg.get("risk_trend", 0.15)):
            return "hold"
        if summary is not None and summary.risk_volatility > float(cfg.get("risk_volatility", 0.18)):
            return "hold"
        if summary is not None and summary.regime_persistence < float(cfg.get("regime_persistence", 0.25)):
            return "hold"
        if summary is not None and summary.transition_rate > float(cfg.get("transition_rate", 0.55)):
            return "hold"
        if summary is not None and summary.confidence_trend < float(cfg.get("confidence_trend", -0.15)):
            return "hold"
        if summary is not None and summary.confidence_volatility >= float(cfg.get("confidence_volatility", 0.15)):
            return "hold"
        if (
            summary is not None
            and summary.regime_streak <= int(cfg.get("regime_streak", 1))
            and summary.stability < float(cfg.get("stability_for_short_streak", 0.7))
        ):
            return "hold"
        if summary is not None and summary.resilience_score <= float(cfg.get("resilience_score", 0.3)):
            return "hold"
        if summary is not None and summary.stress_balance <= float(cfg.get("stress_balance", 0.35)):
            return "hold"
        if summary is not None and summary.regime_entropy >= float(cfg.get("regime_entropy", 0.75)):
            return "hold"
        if summary is not None and summary.instability_score > float(cfg.get("instability_score", 0.65)):
            return "hold"
        if summary is not None and summary.confidence_decay > float(cfg.get("confidence_decay", 0.2)):
            return "hold"
        if summary is not None and summary.drawdown_pressure >= float(cfg.get("drawdown_pressure", 0.6)):
            return "hold"
        if summary is not None and summary.liquidity_pressure >= float(cfg.get("liquidity_pressure", 0.65)):
            return "hold"
        if summary is not None and summary.volatility_ratio >= float(cfg.get("volatility_ratio", 1.55)):
            return "hold"
        if summary is not None and summary.degradation_score >= float(cfg.get("degradation_score", 0.55)):
            return "hold"
        if summary is not None and summary.stability_projection <= float(cfg.get("stability_projection", 0.4)):
            return "hold"
        if summary is not None and summary.volume_trend_volatility >= float(cfg.get("volume_trend_volatility", 0.18)):
            return "hold"
        if summary is not None and summary.liquidity_gap >= float(cfg.get("liquidity_gap", 0.6)):
            return "hold"
        if summary is not None and summary.stress_projection >= float(cfg.get("stress_projection", 0.6)):
            return "hold"
        if summary is not None and summary.confidence_resilience <= float(cfg.get("confidence_resilience", 0.4)):
            return "hold"
        if summary is not None and summary.distribution_pressure >= float(cfg.get("distribution_pressure", 0.55)):
            return "hold"
        if (
            summary is not None
            and abs(summary.skewness_bias) >= float(cfg.get("skewness_bias", 1.2))
            and summary.risk_score >= float(cfg.get("risk_score", 0.45))
        ):
            return "hold"
        if (
            summary is not None
            and summary.kurtosis_excess >= float(cfg.get("kurtosis_excess", 1.5))
            and summary.risk_score >= float(cfg.get("risk_score", 0.45))
        ):
            return "hold"
        if (
            summary is not None
            and abs(summary.volume_imbalance) >= float(cfg.get("volume_imbalance", 0.5))
            and summary.liquidity_pressure >= float(cfg.get("liquidity_pressure_support", 0.45))
        ):
            return "hold"
        if summary is not None and summary.volatility_trend > float(cfg.get("volatility_trend", 0.02)):
            return "hold"
        if summary is not None and summary.drawdown_trend > float(cfg.get("drawdown_trend", 0.08)):
            return "hold"
        if assessment.risk_score >= float(cfg.get("risk_score", 0.75)):
            return "hold"
        if assessment.regime is MarketRegime.TREND:
            return "buy" if last_return >= 0 else "sell"
        if assessment.regime is MarketRegime.MEAN_REVERSION:
            return "sell" if last_return > 0 else "buy"
        threshold = float(cfg.get("return_threshold", 0.001))
        if last_return > threshold:
            return "buy"
        if last_return < -threshold:
            return "sell"
        return "hold"

    def _adjust_strategy_parameters(
        self,
        assessment: MarketRegimeAssessment,
        *,
        aggregated_risk: float | None = None,
        summary: RegimeSummary | None = None,
    ) -> None:
        risk = float(aggregated_risk) if aggregated_risk is not None else assessment.risk_score
        cfg = self._thresholds["auto_trader"]["adjust_strategy_parameters"]
        def _t(name: str, default: float) -> float:
            value = cfg.get(name, default)
            return float(value if isinstance(value, (int, float)) else default)
        if risk >= float(cfg.get("high_risk", 0.75)):
            self.current_strategy = "capital_preservation"
            self.current_leverage = 0.0
            self.current_stop_loss_pct = 0.01
            self.current_take_profit_pct = 0.02
        elif assessment.regime is MarketRegime.TREND:
            self.current_strategy = "trend_following"
            self.current_leverage = 2.0 if risk < float(cfg.get("trend_low_risk", 0.4)) else 1.5
            self.current_stop_loss_pct = 0.03 if risk < float(cfg.get("trend_low_risk", 0.4)) else 0.04
            self.current_take_profit_pct = 0.06 if risk < float(cfg.get("trend_low_risk", 0.4)) else 0.04
        elif assessment.regime is MarketRegime.MEAN_REVERSION:
            self.current_strategy = "mean_reversion"
            self.current_leverage = 1.0 if risk < float(cfg.get("mean_reversion_low_risk", 0.4)) else 0.7
            self.current_stop_loss_pct = 0.015 if risk < float(cfg.get("mean_reversion_low_risk", 0.4)) else 0.02
            self.current_take_profit_pct = 0.03 if risk < float(cfg.get("mean_reversion_low_risk", 0.4)) else 0.025
        else:
            self.current_strategy = "intraday_breakout"
            self.current_leverage = 0.8 if risk < float(cfg.get("intraday_low_risk", 0.5)) else 0.5
            self.current_stop_loss_pct = 0.02 if risk < float(cfg.get("intraday_low_risk", 0.5)) else 0.03
            self.current_take_profit_pct = 0.025 if risk < float(cfg.get("intraday_low_risk", 0.5)) else 0.02

        if summary is not None:
            if summary.risk_level is RiskLevel.CRITICAL:
                self.current_strategy = "capital_preservation"
                self.current_leverage = 0.0
                self.current_stop_loss_pct = 0.01
                self.current_take_profit_pct = 0.02
            elif summary.risk_level is RiskLevel.ELEVATED:
                self.current_leverage = min(
                    self.current_leverage,
                    0.8 if assessment.regime is MarketRegime.TREND else 0.5,
                )
                self.current_stop_loss_pct = max(
                    self.current_stop_loss_pct * float(cfg.get("risk_level_elevated_stop_loss", 0.85)),
                    0.01,
                )
                self.current_take_profit_pct = max(
                    self.current_take_profit_pct * float(cfg.get("risk_level_elevated_take_profit", 0.9)),
                    self.current_stop_loss_pct * 1.4,
                )
            elif summary.risk_level is RiskLevel.CALM and risk < float(cfg.get("risk_level_calm", 0.5)):
                self.current_leverage = min(
                    self.current_leverage * float(cfg.get("risk_level_calm_leverage", 1.2)),
                    2.5,
                )
                self.current_take_profit_pct = min(
                    self.current_take_profit_pct * float(cfg.get("risk_level_calm_take_profit", 1.1)),
                    0.08,
                )

            if summary.resilience_score <= float(cfg.get("resilience_low", 0.35)):
                self.current_leverage = min(self.current_leverage, 0.45)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.88, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.92, self.current_stop_loss_pct * 1.35)
            elif (
                summary.resilience_score >= float(cfg.get("resilience_high", 0.65))
                and summary.stress_balance >= float(cfg.get("stress_balance_high", 0.6))
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and risk < float(cfg.get("risk_level_calm_upper", 0.6))
            ):
                self.current_leverage = min(self.current_leverage * 1.05, 3.2)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.05, 0.09)

            if summary.regime_entropy >= float(cfg.get("entropy_high", 0.75)):
                self.current_leverage = min(self.current_leverage, 0.45)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.9, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.92, self.current_stop_loss_pct * 1.35)
            elif (
                summary.regime_entropy <= float(cfg.get("entropy_low", 0.45))
                and summary.resilience_score >= float(cfg.get("resilience_high", 0.65))
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and risk < float(cfg.get("risk_level_calm_upper", 0.6))
            ):
                self.current_leverage = min(self.current_leverage * 1.03, 3.25)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.03, 0.09)

            if summary.risk_volatility >= float(cfg.get("risk_volatility_high", 0.2)):
                self.current_leverage = min(self.current_leverage, 0.6)
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.9, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.95, self.current_stop_loss_pct * 1.3)

            if summary.regime_persistence <= float(cfg.get("regime_persistence_low", 0.3)):
                self.current_leverage = min(self.current_leverage, 0.4)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                else:
                    if not self.current_strategy.endswith("_probing"):
                        self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.85, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.4)
            elif (
                summary.regime_persistence >= float(cfg.get("regime_persistence_high", 0.65))
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and summary.risk_volatility < float(cfg.get("risk_volatility_high", 0.2))
                and summary.instability_score <= float(cfg.get("instability_ceiling", 0.4))
            ):
                self.current_leverage = min(self.current_leverage * 1.05, 2.75)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.05, 0.085)

            if summary.confidence_volatility >= float(cfg.get("confidence_volatility_high", 0.15)):
                self.current_leverage = min(self.current_leverage, 0.5)
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.9, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.92, self.current_stop_loss_pct * 1.35)

            if summary.confidence_trend < float(cfg.get("confidence_trend_low", -0.1)):
                self.current_leverage = min(self.current_leverage, 0.6)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.85, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.4)

            if summary.regime_streak <= int(cfg.get("regime_streak_low", 1)):
                self.current_leverage = min(self.current_leverage, 0.4)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.85, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.4)

            if (
                summary.confidence_trend > float(cfg.get("confidence_trend_high", 0.12))
                and summary.confidence_volatility <= float(cfg.get("confidence_volatility_low", 0.08))
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and summary.regime_persistence >= float(cfg.get("regime_persistence_high", 0.65))
                and summary.instability_score <= float(cfg.get("instability_ceiling", 0.4))
            ):
                self.current_leverage = min(self.current_leverage * 1.1, 3.0)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.08, 0.09)

            if summary.transition_rate >= float(cfg.get("transition_rate_high", 0.5)):
                self.current_leverage = min(self.current_leverage, 0.45)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.85, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.4)

            if summary.instability_score >= _t("instability_critical", 0.75):
                self.current_strategy = "capital_preservation"
                self.current_leverage = 0.0
                self.current_stop_loss_pct = 0.01
                self.current_take_profit_pct = 0.02
            elif summary.instability_score >= _t("instability_elevated", 0.6):
                self.current_leverage = min(self.current_leverage, 0.5)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.85, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.92, self.current_stop_loss_pct * 1.4)
            elif summary.instability_score <= _t("instability_low", 0.25) and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}:
                self.current_leverage = min(self.current_leverage * 1.08, 3.2)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.07, 0.095)

            if summary.drawdown_pressure >= _t("drawdown_critical", 0.75):
                self.current_strategy = "capital_preservation"
                self.current_leverage = 0.0
                self.current_stop_loss_pct = 0.01
                self.current_take_profit_pct = 0.02
            elif summary.drawdown_pressure >= _t("drawdown_elevated", 0.55):
                self.current_leverage = min(self.current_leverage, 0.45)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.85, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.4)
            elif (
                summary.drawdown_pressure <= _t("drawdown_low", 0.3)
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
            ):
                self.current_leverage = min(self.current_leverage * 1.06, 3.1)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.05, 0.09)

            if summary.liquidity_pressure >= _t("liquidity_pressure_high", 0.7):
                self.current_leverage = min(self.current_leverage, 0.35)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.88, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.35)
            elif (
                summary.liquidity_pressure <= _t("liquidity_pressure_low", 0.35)
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and risk < _t("low_risk_enhancement_cap", 0.5)
            ):
                self.current_leverage = min(self.current_leverage * 1.04, 3.0)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.04, 0.085)

            if summary.confidence_decay >= _t("confidence_decay_high", 0.25):
                self.current_leverage = min(self.current_leverage, 0.45)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.85, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.4)

            if summary.degradation_score >= _t("degradation_critical", 0.6):
                self.current_strategy = "capital_preservation"
                self.current_leverage = 0.0
                self.current_stop_loss_pct = 0.01
                self.current_take_profit_pct = 0.02
            elif summary.degradation_score >= _t("degradation_elevated", 0.45):
                self.current_leverage = min(self.current_leverage, 0.35)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.84, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.35)
            elif (
                summary.degradation_score <= _t("degradation_low", 0.25)
                and summary.stability_projection >= _t("stability_projection_high", 0.65)
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and risk < _t("low_risk_enhancement_cap", 0.5)
            ):
                self.current_leverage = min(self.current_leverage * 1.03, 3.1)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.04, 0.09)

            if summary.stability_projection <= _t("stability_projection_low", 0.4):
                self.current_leverage = min(self.current_leverage, 0.4)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.84, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.35)
            elif (
                summary.stability_projection >= _t("stability_projection_high", 0.65)
                and summary.degradation_score <= _t("degradation_positive_cap", 0.3)
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and risk < _t("low_risk_enhancement_cap", 0.5)
            ):
                self.current_leverage = min(self.current_leverage * 1.02, 3.05)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.02, 0.09)

            if summary.volume_trend_volatility >= _t("volume_trend_volatility_high", 0.2):
                self.current_leverage = min(self.current_leverage, 0.35)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.83, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.35)
            elif (
                summary.volume_trend_volatility <= _t("volume_trend_volatility_low", 0.1)
                and summary.degradation_score <= _t("degradation_positive_cap", 0.3)
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
            ):
                self.current_leverage = min(self.current_leverage * 1.01, 3.0)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.01, 0.085)

            if summary.volatility_trend > _t("volatility_trend_high", 0.02):
                self.current_leverage = min(self.current_leverage, 0.4)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.85, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.35)
            elif (
                summary.volatility_trend <= _t("volatility_trend_relief", 0.0)
                and summary.degradation_score <= _t("degradation_positive_cap", 0.3)
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
            ):
                self.current_leverage = min(self.current_leverage * 1.01, 3.1)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.01, 0.088)

            if summary.drawdown_trend > _t("drawdown_trend_high", 0.08):
                self.current_leverage = min(self.current_leverage, 0.35)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.82, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.88, self.current_stop_loss_pct * 1.35)
            elif (
                summary.drawdown_trend <= _t("drawdown_trend_relief", 0.0)
                and summary.degradation_score <= _t("degradation_positive_cap", 0.3)
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
            ):
                self.current_leverage = min(self.current_leverage * 1.02, 3.05)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.02, 0.09)

            if summary.stress_index >= _t("stress_index_critical", 0.8):
                self.current_strategy = "capital_preservation"
                self.current_leverage = 0.0
                self.current_stop_loss_pct = 0.01
                self.current_take_profit_pct = 0.02
            elif summary.stress_index >= _t("stress_index_elevated", 0.6):
                self.current_leverage = min(self.current_leverage, 0.35)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.82, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.88, self.current_stop_loss_pct * 1.35)
            elif (
                summary.stress_index <= _t("stress_index_low", 0.28)
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and risk < _t("stress_relief_risk_cap", 0.55)
            ):
                self.current_leverage = min(self.current_leverage * 1.04, 3.2)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.04, 0.09)

            if summary.stress_momentum >= _t("stress_momentum_high", 0.65):
                self.current_leverage = min(self.current_leverage, 0.3)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.82, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.88, self.current_stop_loss_pct * 1.35)
            elif (
                summary.stress_momentum <= _t("stress_momentum_low", 0.35)
                and summary.stress_index <= _t("stress_index_relief", 0.35)
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and risk < _t("low_risk_enhancement_cap", 0.5)
            ):
                self.current_leverage = min(self.current_leverage * 1.02, 3.05)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.02, 0.088)

            if summary.tail_risk_index >= _t("tail_risk_high", 0.55):
                self.current_leverage = min(self.current_leverage, 0.35)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.84, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.35)
            elif (
                summary.tail_risk_index <= _t("tail_risk_low", 0.2)
                and summary.stress_index <= _t("stress_index_tail_relief", 0.3)
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and risk < _t("low_risk_enhancement_cap", 0.5)
            ):
                self.current_leverage = min(self.current_leverage * 1.03, 3.1)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.03, 0.085)

            if summary.shock_frequency >= _t("shock_frequency_high", 0.55):
                self.current_leverage = min(self.current_leverage, 0.3)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.82, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.88, self.current_stop_loss_pct * 1.35)
            elif (
                summary.shock_frequency <= _t("shock_frequency_low", 0.25)
                and summary.regime_persistence >= _t("regime_persistence_positive", 0.6)
                and summary.stress_index <= _t("stress_index_relief", 0.35)
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
            ):
                self.current_leverage = min(self.current_leverage * 1.02, 3.0)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.02, 0.085)

            if (
                summary.distribution_pressure >= _t("distribution_pressure_adjust_high", 0.65)
                or abs(summary.skewness_bias) >= _t("skewness_bias_adjust_high", 1.3)
                or summary.kurtosis_excess >= _t("kurtosis_adjust_high", 1.8)
                or (
                    abs(summary.volume_imbalance)
                    >= _t("volume_imbalance_adjust_high", 0.55)
                    and summary.liquidity_pressure
                    >= float(cfg.get("liquidity_pressure_support", 0.45))
                )
            ):
                self.current_leverage = min(self.current_leverage, 0.3)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.82, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.88, self.current_stop_loss_pct * 1.35)
            elif (
                summary.distribution_pressure <= _t("distribution_pressure_adjust_low", 0.3)
                and abs(summary.skewness_bias) <= _t("skewness_bias_adjust_low", 0.8)
                and summary.kurtosis_excess <= _t("kurtosis_adjust_low", 1.0)
                and abs(summary.volume_imbalance) <= _t("volume_imbalance_adjust_low", 0.35)
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and risk < _t("low_risk_enhancement_cap", 0.5)
            ):
                self.current_leverage = min(self.current_leverage * 1.02, 3.05)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.02, 0.09)

            if summary.liquidity_gap >= _t("liquidity_gap_high", 0.65):
                self.current_leverage = min(self.current_leverage, 0.35)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.84, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.35)
            elif (
                summary.liquidity_gap <= _t("liquidity_gap_low", 0.35)
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and summary.resilience_score >= _t("resilience_mid", 0.55)
                and risk < _t("low_risk_enhancement_cap", 0.5)
            ):
                self.current_leverage = min(self.current_leverage * 1.02, 3.05)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.02, 0.09)

            if summary.liquidity_trend >= _t("liquidity_trend_high", 0.6):
                self.current_leverage = min(self.current_leverage, 0.35)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.84, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.35)
            elif (
                summary.liquidity_trend <= _t("liquidity_trend_low", 0.35)
                and summary.liquidity_gap <= _t("liquidity_gap_relief", 0.4)
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and risk < _t("low_risk_enhancement_cap", 0.5)
            ):
                self.current_leverage = min(self.current_leverage * 1.02, 3.05)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.02, 0.09)

            if summary.stress_projection >= _t("stress_projection_critical", 0.65):
                self.current_strategy = "capital_preservation"
                self.current_leverage = 0.0
                self.current_stop_loss_pct = 0.01
                self.current_take_profit_pct = 0.02
            elif summary.stress_projection >= _t("stress_projection_elevated", 0.55):
                self.current_leverage = min(self.current_leverage, 0.35)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.84, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.35)
            elif (
                summary.stress_projection <= _t("stress_projection_low", 0.35)
                and summary.stress_index <= _t("stress_index_relief", 0.35)
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and risk < _t("moderate_risk_enhancement_cap", 0.45)
            ):
                self.current_leverage = min(self.current_leverage * 1.02, 3.0)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.02, 0.085)

            if summary.confidence_resilience <= float(cfg.get("confidence_resilience", 0.4)):
                self.current_leverage = min(self.current_leverage, 0.45)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.85, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.35)
            elif (
                summary.confidence_resilience >= _t("confidence_resilience_high", 0.65)
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and risk < _t("moderate_risk_enhancement_cap", 0.45)
                and summary.resilience_score >= _t("resilience_mid", 0.55)
            ):
                self.current_leverage = min(self.current_leverage * 1.03, 3.1)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.03, 0.09)

            if summary.confidence_fragility >= _t("confidence_fragility_high", 0.55):
                self.current_leverage = min(self.current_leverage, 0.35)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.84, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.35)
            elif (
                summary.confidence_fragility <= _t("confidence_fragility_low", 0.35)
                and summary.confidence_resilience >= _t("confidence_resilience_mid", 0.6)
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and risk < _t("moderate_risk_enhancement_cap", 0.45)
            ):
                self.current_leverage = min(self.current_leverage * 1.02, 3.0)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.02, 0.085)

            if summary.volatility_of_volatility >= _t("volatility_of_volatility_high", 0.03):
                self.current_leverage = min(self.current_leverage, 0.4)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.85, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.35)
            elif (
                summary.volatility_of_volatility <= _t("volatility_of_volatility_low", 0.015)
                and summary.stress_index <= _t("stress_index_relief", 0.35)
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
            ):
                self.current_leverage = min(self.current_leverage * 1.02, 3.05)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.02, 0.085)

        if summary is not None and risk < _t("summary_risk_cap", 0.75):
            if summary.risk_trend > _t("summary_risk_trend_high", 0.05):
                self.current_leverage = max(0.0, self.current_leverage - 0.3)
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.8, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.5)
            if summary.stability < _t("summary_stability_floor", 0.4):
                self.current_leverage = min(self.current_leverage, 0.5)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                else:
                    self.current_strategy = f"{self.current_strategy}_probing"

        self.current_leverage = float(max(self.current_leverage, 0.0))
        self.current_stop_loss_pct = float(max(self.current_stop_loss_pct, 0.005))
        self.current_take_profit_pct = float(max(self.current_take_profit_pct, self.current_stop_loss_pct * 1.2))

    def _apply_signal_guardrails(
        self,
        signal: str,
        effective_risk: float,
        summary: RegimeSummary | None,
    ) -> str:
        reasons: list[str] = []
        triggers: list[GuardrailTrigger] = []

        def _finalise(result: str) -> str:
            self._last_guardrail_reasons = reasons
            self._last_guardrail_triggers = triggers
            return result

        if signal == "hold":
            return _finalise(signal)

        guardrails = self._thresholds["auto_trader"].get("signal_guardrails", {})

        def _label(name: str) -> str:
            return name.replace("_", " ")

        def _add_reason(name: str, comparator: str, threshold: float, value: float | None = None) -> None:
            label = _label(name)
            message = f"{label} {comparator} {threshold:.3f}"
            if value is not None:
                message = f"{message} (value={value:.3f})"
            reasons.append(message)
            triggers.append(
                GuardrailTrigger(
                    name=name,
                    label=label,
                    comparator=comparator,
                    threshold=float(threshold),
                    value=float(value) if value is not None else None,
                )
            )

        def _coerce_threshold(name: str, default: float) -> float | None:
            raw = guardrails.get(name, default)
            if raw is None:
                return None
            try:
                return float(raw)
            except (TypeError, ValueError):
                self._log(
                    f"Ignoring invalid guardrail threshold {name!r}: {raw!r}",
                    level=logging.DEBUG,
                )
                return float(default)

        risk_cap = _coerce_threshold("effective_risk_cap", 0.75)
        if risk_cap is not None and effective_risk >= risk_cap:
            _add_reason("effective_risk", ">=", risk_cap, effective_risk)
            return _finalise("hold")
        if summary is None:
            return _finalise(signal)

        def _metric(name: str) -> float | None:
            value = getattr(summary, name, None)
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        for name, default in (
            ("stress_index", 0.65),
            ("tail_risk_index", 0.6),
            ("shock_frequency", 0.6),
            ("stress_momentum", 0.65),
            ("regime_entropy", 0.8),
            ("confidence_fragility", 0.65),
            ("degradation_score", 0.6),
            ("volume_trend_volatility", 0.2),
            ("liquidity_gap", 0.6),
            ("stress_projection", 0.6),
            ("liquidity_trend", 0.6),
        ):
            threshold = _coerce_threshold(name, default)
            if threshold is None:
                continue
            value = _metric(name)
            if value is not None and value >= threshold:
                _add_reason(name, ">=", threshold, value)
                return _finalise("hold")

        for name, default in (("volatility_trend", 0.025), ("drawdown_trend", 0.1)):
            threshold = _coerce_threshold(name, default)
            if threshold is None:
                continue
            value = _metric(name)
            if value is not None and value > threshold:
                _add_reason(name, ">", threshold, value)
                return _finalise("hold")

        for name, default in (
            ("resilience_score", 0.3),
            ("stress_balance", 0.35),
            ("stability_projection", 0.35),
            ("confidence_resilience", 0.4),
        ):
            threshold = _coerce_threshold(name, default)
            if threshold is None:
                continue
            value = _metric(name)
            if value is not None and value <= threshold:
                _add_reason(name, "<=", threshold, value)
                return _finalise("hold")

        return _finalise(signal)

    def _update_cooldown(
        self,
        *,
        summary: RegimeSummary | None,
        effective_risk: float,
    ) -> tuple[bool, float]:
        now = time.monotonic()
        cooldown_cfg = self._thresholds["auto_trader"]["cooldown"]
        if summary is not None:
            severity_weights = cooldown_cfg.get("severity_weights", {})
            stability_gap = float(cooldown_cfg.get("stability_projection_gap", 0.45))
            confidence_gap = float(cooldown_cfg.get("confidence_resilience_gap", 0.6))
            severity = max(
                summary.cooldown_score
                * float(severity_weights.get("cooldown_score", 1.0)),
                summary.severe_event_rate
                * float(severity_weights.get("severe_event_rate", 0.8)),
                summary.stress_index * float(severity_weights.get("stress_index", 0.7)),
                summary.stress_projection
                * float(severity_weights.get("stress_projection", 0.75)),
                summary.stress_momentum
                * float(severity_weights.get("stress_momentum", 0.7)),
                summary.degradation_score
                * float(severity_weights.get("degradation_score", 0.75)),
                max(0.0, stability_gap - summary.stability_projection)
                * float(severity_weights.get("stability_projection_gap", 0.6)),
                summary.liquidity_gap
                * float(severity_weights.get("liquidity_gap", 0.65)),
                summary.liquidity_trend
                * float(severity_weights.get("liquidity_trend", 0.6)),
                max(0.0, confidence_gap - summary.confidence_resilience)
                * float(severity_weights.get("confidence_resilience_gap", 0.6)),
                summary.confidence_fragility
                * float(severity_weights.get("confidence_fragility", 0.65)),
            )
            distribution_weights = cooldown_cfg.get("distribution_weights", {})
            normalisers = cooldown_cfg.get("distribution_normalisers", {})
            distribution_flags = max(
                summary.distribution_pressure
                * float(distribution_weights.get("distribution_pressure", 1.0)),
                min(
                    1.0,
                    abs(summary.skewness_bias)
                    / float(normalisers.get("skewness_bias", 1.6)),
                ),
                min(
                    1.0,
                    max(0.0, summary.kurtosis_excess)
                    / float(normalisers.get("kurtosis_excess", 3.0)),
                ),
                min(
                    1.0,
                    abs(summary.volume_imbalance)
                    / float(normalisers.get("volume_imbalance", 0.55)),
                ),
            )
            severity = max(
                severity,
                distribution_flags
                * float(severity_weights.get("distribution_flags_weight", 0.75)),
                max(0.0, float(cooldown_cfg.get("resilience_gap", 0.55)) - summary.resilience_score)
                * float(severity_weights.get("resilience_gap_weight", 0.65)),
                max(0.0, float(cooldown_cfg.get("stress_balance_gap", 0.5)) - summary.stress_balance)
                * float(severity_weights.get("stress_balance_gap_weight", 0.6)),
                summary.stress_projection * float(severity_weights.get("stress_projection", 0.7)),
                summary.liquidity_gap * float(severity_weights.get("liquidity_gap", 0.65)),
                summary.stress_momentum * float(severity_weights.get("stress_momentum", 0.7)),
                summary.liquidity_trend * float(severity_weights.get("liquidity_trend", 0.6)),
                summary.confidence_fragility * float(severity_weights.get("confidence_fragility", 0.65)),
                max(0.0, summary.regime_entropy - float(cooldown_cfg.get("entropy_gap", 0.65)))
                * float(severity_weights.get("entropy_excess_weight", 0.6)),
            )
            critical_cfg = cooldown_cfg.get("critical", {})
            if (
                effective_risk >= float(critical_cfg.get("risk", 0.85))
                or summary.risk_level is RiskLevel.CRITICAL
                or severity >= float(critical_cfg.get("severity", 0.75))
                or summary.degradation_score >= float(critical_cfg.get("degradation", 0.7))
                or summary.stability_projection <= float(critical_cfg.get("stability_projection", 0.25))
                or summary.distribution_pressure >= float(critical_cfg.get("distribution_pressure", 0.8))
                or distribution_flags >= float(critical_cfg.get("distribution_flags", 0.85))
                or summary.resilience_score <= float(critical_cfg.get("resilience_score", 0.25))
                or summary.stress_balance <= float(critical_cfg.get("stress_balance", 0.25))
                or summary.regime_entropy >= float(critical_cfg.get("regime_entropy", 0.85))
                or summary.stress_projection >= float(critical_cfg.get("stress_projection", 0.75))
                or summary.liquidity_gap >= float(critical_cfg.get("liquidity_gap", 0.75))
                or summary.stress_momentum >= float(critical_cfg.get("stress_momentum", 0.75))
                or summary.liquidity_trend >= float(critical_cfg.get("liquidity_trend", 0.7))
                or summary.confidence_resilience <= float(critical_cfg.get("confidence_resilience", 0.25))
                or summary.confidence_fragility >= float(critical_cfg.get("confidence_fragility", 0.7))
            ):
                duration = max(
                    self.auto_trade_interval_s * float(critical_cfg.get("duration_multiplier", 5.0)),
                    float(critical_cfg.get("duration_min", 300.0)),
                )
                self._cooldown_until = max(self._cooldown_until, now + duration)
                self._cooldown_reason = "critical_risk"
            elif (
                severity >= float(cooldown_cfg.get("elevated", {}).get("severity", 0.55))
                or (
                    summary.risk_level is RiskLevel.ELEVATED
                    and summary.stress_index >= float(cooldown_cfg.get("elevated", {}).get("stress_index", 0.6))
                )
                or summary.degradation_score >= float(cooldown_cfg.get("elevated", {}).get("degradation", 0.55))
                or summary.stability_projection <= float(cooldown_cfg.get("elevated", {}).get("stability_projection", 0.35))
                or summary.distribution_pressure >= float(cooldown_cfg.get("elevated", {}).get("distribution_pressure", 0.6))
                or distribution_flags >= float(cooldown_cfg.get("elevated", {}).get("distribution_flags", 0.65))
                or summary.resilience_score <= float(cooldown_cfg.get("elevated", {}).get("resilience_score", 0.4))
                or summary.stress_balance <= float(cooldown_cfg.get("elevated", {}).get("stress_balance", 0.4))
                or summary.regime_entropy >= float(cooldown_cfg.get("elevated", {}).get("regime_entropy", 0.75))
                or summary.stress_projection >= float(cooldown_cfg.get("elevated", {}).get("stress_projection", 0.6))
                or summary.liquidity_gap >= float(cooldown_cfg.get("elevated", {}).get("liquidity_gap", 0.6))
                or summary.stress_momentum >= float(cooldown_cfg.get("elevated", {}).get("stress_momentum", 0.6))
                or summary.liquidity_trend >= float(cooldown_cfg.get("elevated", {}).get("liquidity_trend", 0.6))
                or summary.confidence_resilience <= float(cooldown_cfg.get("elevated", {}).get("confidence_resilience", 0.35))
                or summary.confidence_fragility >= float(cooldown_cfg.get("elevated", {}).get("confidence_fragility", 0.6))
            ):
                elevated_cfg = cooldown_cfg.get("elevated", {})
                duration = max(
                    self.auto_trade_interval_s * float(elevated_cfg.get("duration_multiplier", 3.0)),
                    float(elevated_cfg.get("duration_min", 180.0)),
                )
                self._cooldown_until = max(self._cooldown_until, now + duration)
                self._cooldown_reason = "elevated_risk"
            elif (
                severity >= float(cooldown_cfg.get("instability", {}).get("severity", 0.45))
                and summary.risk_level in {RiskLevel.ELEVATED, RiskLevel.WATCH}
                or summary.degradation_score >= float(cooldown_cfg.get("instability", {}).get("degradation", 0.45))
                or summary.distribution_pressure >= float(cooldown_cfg.get("instability", {}).get("distribution_pressure", 0.5))
                or distribution_flags >= float(cooldown_cfg.get("instability", {}).get("distribution_flags", 0.55))
                or summary.resilience_score <= float(cooldown_cfg.get("instability", {}).get("resilience_score", 0.45))
                or summary.stress_balance <= float(cooldown_cfg.get("instability", {}).get("stress_balance", 0.4))
                or summary.regime_entropy >= float(cooldown_cfg.get("instability", {}).get("regime_entropy", 0.65))
                or summary.stress_projection >= float(cooldown_cfg.get("instability", {}).get("stress_projection", 0.5))
                or summary.liquidity_gap >= float(cooldown_cfg.get("instability", {}).get("liquidity_gap", 0.5))
                or summary.stress_momentum >= float(cooldown_cfg.get("instability", {}).get("stress_momentum", 0.5))
                or summary.liquidity_trend >= float(cooldown_cfg.get("instability", {}).get("liquidity_trend", 0.5))
                or summary.confidence_resilience <= float(cooldown_cfg.get("instability", {}).get("confidence_resilience", 0.4))
                or summary.confidence_fragility >= float(cooldown_cfg.get("instability", {}).get("confidence_fragility", 0.5))
            ):
                instability_cfg = cooldown_cfg.get("instability", {})
                duration = max(
                    self.auto_trade_interval_s * float(instability_cfg.get("duration_multiplier", 2.0)),
                    float(instability_cfg.get("duration_min", 120.0)),
                )
                self._cooldown_until = max(self._cooldown_until, now + duration)
                self._cooldown_reason = "instability_spike"
            elif (
                summary.cooldown_score <= float(cooldown_cfg.get("release", {}).get("cooldown_score", 0.35))
                and summary.recovery_potential >= float(cooldown_cfg.get("release", {}).get("recovery_potential", 0.6))
                and effective_risk <= float(cooldown_cfg.get("release", {}).get("risk", 0.55))
                and summary.degradation_score <= float(cooldown_cfg.get("release", {}).get("degradation_score", 0.35))
                and summary.stability_projection >= float(cooldown_cfg.get("release", {}).get("stability_projection", 0.45))
                and summary.distribution_pressure <= float(cooldown_cfg.get("release", {}).get("distribution_pressure", 0.4))
                and abs(summary.skewness_bias) <= float(cooldown_cfg.get("release", {}).get("skewness_bias", 0.9))
                and summary.kurtosis_excess <= float(cooldown_cfg.get("release", {}).get("kurtosis_excess", 1.2))
                and abs(summary.volume_imbalance) <= float(cooldown_cfg.get("release", {}).get("volume_imbalance", 0.4))
                and summary.resilience_score >= float(cooldown_cfg.get("release", {}).get("resilience_score", 0.55))
                and summary.stress_balance >= float(cooldown_cfg.get("release", {}).get("stress_balance", 0.5))
                and summary.regime_entropy <= float(cooldown_cfg.get("release", {}).get("regime_entropy", 0.55))
                and summary.liquidity_gap <= float(cooldown_cfg.get("release", {}).get("liquidity_gap", 0.4))
                and summary.stress_projection <= float(cooldown_cfg.get("release", {}).get("stress_projection", 0.4))
                and summary.stress_momentum <= float(cooldown_cfg.get("release", {}).get("stress_momentum", 0.4))
                and summary.liquidity_trend <= float(cooldown_cfg.get("release", {}).get("liquidity_trend", 0.4))
                and summary.confidence_resilience >= float(cooldown_cfg.get("release", {}).get("confidence_resilience", 0.55))
                and summary.confidence_fragility <= float(cooldown_cfg.get("release", {}).get("confidence_fragility", 0.4))
            ):
                self._cooldown_until = 0.0
                self._cooldown_reason = None
        elif effective_risk >= float(cooldown_cfg.get("high_risk_fallback", {}).get("risk", 0.9)):
            fallback_cfg = cooldown_cfg.get("high_risk_fallback", {})
            duration = max(
                self.auto_trade_interval_s * float(fallback_cfg.get("duration_multiplier", 2.0)),
                float(fallback_cfg.get("duration_min", 150.0)),
            )
            self._cooldown_until = max(self._cooldown_until, now + duration)
            self._cooldown_reason = "high_risk"

        remaining = max(0.0, self._cooldown_until - now)
        active = remaining > 0.0
        if active and summary is not None:
            release_active_cfg = cooldown_cfg.get("release_active", {})
            if (
                summary.recovery_potential >= float(release_active_cfg.get("recovery_potential", 0.7))
                and summary.cooldown_score <= float(release_active_cfg.get("cooldown_score", 0.4))
                and summary.severe_event_rate <= float(release_active_cfg.get("severe_event_rate", 0.4))
                and effective_risk <= float(release_active_cfg.get("risk", 0.55))
                and summary.degradation_score <= float(release_active_cfg.get("degradation_score", 0.35))
                and summary.stability_projection >= float(release_active_cfg.get("stability_projection", 0.5))
                and summary.distribution_pressure <= float(release_active_cfg.get("distribution_pressure", 0.4))
                and abs(summary.skewness_bias) <= float(release_active_cfg.get("skewness_bias", 0.9))
                and summary.kurtosis_excess <= float(release_active_cfg.get("kurtosis_excess", 1.2))
                and abs(summary.volume_imbalance) <= float(release_active_cfg.get("volume_imbalance", 0.4))
                and summary.resilience_score >= float(release_active_cfg.get("resilience_score", 0.6))
                and summary.stress_balance >= float(release_active_cfg.get("stress_balance", 0.55))
                and summary.regime_entropy <= float(release_active_cfg.get("regime_entropy", 0.5))
                and summary.liquidity_gap <= float(release_active_cfg.get("liquidity_gap", 0.4))
                and summary.stress_projection <= float(release_active_cfg.get("stress_projection", 0.35))
                and summary.confidence_resilience >= float(release_active_cfg.get("confidence_resilience", 0.6))
            ):
                self._cooldown_until = 0.0
                self._cooldown_reason = None
                remaining = 0.0
                active = False
        if not active:
            self._cooldown_until = 0.0
            self._cooldown_reason = None
        return active, remaining

    def _build_risk_decision(
        self,
        symbol: str,
        signal: str,
        assessment: MarketRegimeAssessment,
        *,
        effective_risk: float,
        summary: RegimeSummary | None = None,
        cooldown_active: bool = False,
        cooldown_remaining: float = 0.0,
        cooldown_reason: str | None = None,
        guardrail_reasons: list[str] | None = None,
        guardrail_triggers: list[GuardrailTrigger] | None = None,
    ) -> RiskDecision:
        should_trade = signal in {"buy", "sell"} and self.current_leverage > 0 and not cooldown_active
        fraction = self.current_leverage if should_trade else 0.0
        if cooldown_active:
            state = "halted"
        else:
            state = "risk_off" if effective_risk >= 0.75 else "ready"
        reason = f"Regime {assessment.regime.value}"
        details = {
            "symbol": symbol,
            "signal": signal,
            "confidence": assessment.confidence,
            "risk_score": assessment.risk_score,
            "effective_risk": effective_risk,
            "strategy": self.current_strategy,
        }
        details["cooldown_active"] = cooldown_active
        details["cooldown_remaining_s"] = cooldown_remaining
        details["cooldown_reason"] = cooldown_reason
        details["guardrail_reasons"] = list(guardrail_reasons or [])
        details["guardrail_triggers"] = [trigger.to_dict() for trigger in guardrail_triggers or []]
        if summary is not None:
            details["summary"] = summary.to_dict()
        mode = "demo"
        if hasattr(self.gui, "is_demo_mode_active"):
            try:
                mode = "demo" if self.gui.is_demo_mode_active() else "live"
            except Exception:
                mode = "demo"
        return RiskDecision(
            should_trade=should_trade,
            fraction=fraction,
            state=state,
            reason=reason,
            details=details,
            stop_loss_pct=self.current_stop_loss_pct,
            take_profit_pct=self.current_take_profit_pct,
            mode=mode,
            cooldown_active=cooldown_active,
            cooldown_remaining_s=cooldown_remaining if cooldown_active else None,
            cooldown_reason=cooldown_reason if cooldown_active else None,
        )

    # ------------------------------------------------------------------
    # Extension hook ----------------------------------------------------
    # ------------------------------------------------------------------
    def _auto_trade_loop(self) -> None:
        try:
            symbol = self.symbol_getter()
        except Exception as exc:  # pragma: no cover - defensive guard
            self._log(f"Failed to resolve trading symbol: {exc!r}", level=logging.ERROR)
            self._auto_trade_stop.wait(self.auto_trade_interval_s)
            return

        timeframe = "1h"
        timeframe_var = getattr(self.gui, "timeframe_var", None)
        if timeframe_var is not None and hasattr(timeframe_var, "get"):
            try:
                timeframe = str(timeframe_var.get())
            except Exception:
                timeframe = "1h"

        ai_manager = self._resolve_ai_manager()
        if not symbol or ai_manager is None:
            self._log("Auto-trade prerequisites missing AI manager or symbol", level=logging.DEBUG)
            self._auto_trade_stop.wait(self.auto_trade_interval_s)
            return

        market_data = self._fetch_market_data(symbol, timeframe)
        if market_data is None or market_data.empty:
            self._log(
                f"No market data available for {symbol} on {timeframe}",
                level=logging.WARNING,
            )
            self._auto_trade_stop.wait(self.auto_trade_interval_s)
            return

        assessment: MarketRegimeAssessment
        try:
            if hasattr(ai_manager, "assess_market_regime"):
                try:
                    assessment = ai_manager.assess_market_regime(symbol, market_data)
                except TypeError:
                    assessment = ai_manager.assess_market_regime(market_data, symbol=symbol)
            else:
                classifier = MarketRegimeClassifier()
                assessment = classifier.assess(market_data, symbol=symbol)
        except Exception as exc:
            self._log(f"AI manager regime assessment failed: {exc!r}", level=logging.ERROR)
            self._auto_trade_stop.wait(self.auto_trade_interval_s)
            return

        summary: RegimeSummary | None = None
        if hasattr(ai_manager, "get_regime_summary"):
            try:
                summary = ai_manager.get_regime_summary(symbol)
            except Exception:
                summary = None

        returns = market_data.get("close")
        last_return = 0.0
        if isinstance(returns, pd.Series):
            changes = returns.pct_change().dropna()
            if not changes.empty:
                last_return = float(changes.iloc[-1])

        effective_risk = assessment.risk_score
        if summary is not None:
            effective_risk = max(
                effective_risk,
                float(summary.risk_score + min(summary.risk_volatility * 0.8, 0.25)),
            )
            if summary.regime_persistence < 0.25:
                effective_risk = max(effective_risk, 0.7)
            if summary.instability_score >= 0.6:
                effective_risk = max(
                    effective_risk,
                    min(1.0, summary.instability_score * 0.9 + summary.risk_score * 0.4),
                )
            if summary.transition_rate >= 0.5:
                effective_risk = max(
                    effective_risk,
                    min(1.0, assessment.risk_score + summary.transition_rate * 0.4),
                )
            if summary.confidence_decay > 0:
                effective_risk = max(
                    effective_risk,
                    min(1.0, assessment.risk_score + summary.confidence_decay * 0.5),
                )
            if summary.drawdown_pressure >= 0.5:
                effective_risk = max(
                    effective_risk,
                    min(1.0, assessment.risk_score + summary.drawdown_pressure * 0.5),
                )
            if summary.liquidity_pressure >= 0.5:
                effective_risk = max(
                    effective_risk,
                    min(1.0, assessment.risk_score + summary.liquidity_pressure * 0.4),
                )
            if summary.volatility_ratio >= 1.35:
                effective_risk = max(
                    effective_risk,
                    min(
                        1.0,
                        assessment.risk_score
                        + min(summary.volatility_ratio - 1.0, 1.0) * 0.3,
                    ),
                )
            if summary.stress_index >= 0.6:
                effective_risk = max(
                    effective_risk,
                    min(1.0, assessment.risk_score + summary.stress_index * 0.45),
                )
            elif summary.stress_index >= 0.4:
                effective_risk = max(
                    effective_risk,
                    min(1.0, assessment.risk_score + summary.stress_index * 0.35),
                )
            if summary.tail_risk_index >= 0.5:
                effective_risk = max(
                    effective_risk,
                    min(1.0, assessment.risk_score + summary.tail_risk_index * 0.4),
                )
            if summary.shock_frequency >= 0.55:
                effective_risk = max(
                    effective_risk,
                    min(1.0, assessment.risk_score + summary.shock_frequency * 0.35),
                )
            if summary.volatility_of_volatility >= 0.03:
                effective_risk = max(
                    effective_risk,
                    min(
                        1.0,
                        assessment.risk_score
                        + min(summary.volatility_of_volatility / 0.04, 1.0) * 0.3,
                    ),
                )
            if summary.degradation_score >= 0.5:
                effective_risk = max(
                    effective_risk,
                    min(1.0, assessment.risk_score + summary.degradation_score * 0.45),
                )
            if summary.stability_projection <= 0.45:
                effective_risk = max(
                    effective_risk,
                    min(
                        1.0,
                        assessment.risk_score
                        + max(0.0, 0.45 - summary.stability_projection) * 0.6,
                    ),
                )
            if summary.volume_trend_volatility >= 0.18:
                effective_risk = max(
                    effective_risk,
                    min(
                        1.0,
                        assessment.risk_score
                        + min(summary.volume_trend_volatility / 0.25, 1.0) * 0.3,
                    ),
                )
            if summary.volatility_trend > 0.015:
                effective_risk = max(
                    effective_risk,
                    min(
                        1.0,
                        assessment.risk_score
                        + min(summary.volatility_trend / 0.03, 1.0) * 0.25,
                    ),
                )
            if summary.drawdown_trend > 0.05:
                effective_risk = max(
                    effective_risk,
                    min(
                        1.0,
                        assessment.risk_score
                        + min(summary.drawdown_trend / 0.2, 1.0) * 0.35,
                    ),
                )
            if summary.distribution_pressure >= 0.55:
                effective_risk = max(
                    effective_risk,
                    min(
                        1.0,
                        assessment.risk_score
                        + min(summary.distribution_pressure, 1.0) * 0.4,
                    ),
                )
            if summary.liquidity_gap >= 0.55:
                effective_risk = max(
                    effective_risk,
                    min(
                        1.0,
                        assessment.risk_score
                        + min(summary.liquidity_gap, 1.0) * 0.35,
                    ),
                )
            if summary.stress_projection >= 0.5:
                effective_risk = max(
                    effective_risk,
                    min(
                        1.0,
                        assessment.risk_score
                        + min(summary.stress_projection, 1.0) * 0.4,
                    ),
                )
            if summary.stress_momentum >= 0.55:
                effective_risk = max(
                    effective_risk,
                    min(
                        1.0,
                        assessment.risk_score
                        + min(summary.stress_momentum, 1.0) * 0.38,
                    ),
                )
            if summary.liquidity_trend >= 0.6:
                effective_risk = max(
                    effective_risk,
                    min(
                        1.0,
                        assessment.risk_score
                        + min(summary.liquidity_trend, 1.0) * 0.35,
                    ),
                )
            if summary.confidence_fragility >= 0.5:
                effective_risk = max(
                    effective_risk,
                    min(
                        1.0,
                        assessment.risk_score
                        + min(summary.confidence_fragility, 1.0) * 0.35,
                    ),
                )
            if summary.resilience_score <= 0.35:
                effective_risk = max(
                    effective_risk,
                    min(
                        1.0,
                        assessment.risk_score
                        + min((0.35 - summary.resilience_score) * 0.7, 0.25)
                        + min(summary.cooldown_score * 0.25, 0.15),
                    ),
                )
            if summary.confidence_resilience <= 0.45:
                effective_risk = max(
                    effective_risk,
                    min(
                        1.0,
                        assessment.risk_score
                        + min((0.45 - summary.confidence_resilience) * 0.6, 0.2),
                    ),
                )
            if summary.stress_balance <= 0.4:
                effective_risk = max(
                    effective_risk,
                    min(
                        1.0,
                        assessment.risk_score
                        + min((0.4 - summary.stress_balance) * 0.65, 0.22),
                    ),
                )
            if summary.regime_entropy >= 0.7:
                effective_risk = max(
                    effective_risk,
                    min(
                        1.0,
                        assessment.risk_score
                        + min(summary.regime_entropy * 0.35, 0.25),
                    ),
                )
            elif (
                summary.resilience_score >= 0.7
                and summary.stress_balance >= 0.6
                and summary.regime_entropy <= 0.5
                and summary.liquidity_gap <= 0.45
                and summary.stress_projection <= 0.45
                and summary.confidence_resilience >= 0.6
                and summary.stress_momentum <= 0.45
                and summary.liquidity_trend <= 0.45
                and summary.confidence_fragility <= 0.45
            ):
                relief = min(max(0.0, summary.resilience_score - 0.7) * 0.25, 0.12)
                entropy_relief = min(max(0.0, 0.5 - summary.regime_entropy) * 0.18, 0.1)
                liquidity_relief = min(max(0.0, 0.45 - summary.liquidity_gap) * 0.18, 0.1)
                confidence_relief = min(
                    max(0.0, summary.confidence_resilience - 0.6) * 0.2,
                    0.1,
                )
                projection_relief = min(max(0.0, 0.45 - summary.stress_projection) * 0.2, 0.1)
                momentum_relief = min(max(0.0, 0.45 - summary.stress_momentum) * 0.2, 0.1)
                liquidity_trend_relief = min(max(0.0, 0.45 - summary.liquidity_trend) * 0.2, 0.1)
                fragility_relief = min(max(0.0, 0.45 - summary.confidence_fragility) * 0.2, 0.1)
                reduction = max(
                    0.0,
                    relief
                    + max(0.0, entropy_relief)
                    + liquidity_relief
                    + confidence_relief
                    + projection_relief,
                )
                reduction += momentum_relief + liquidity_trend_relief + fragility_relief
                effective_risk = max(assessment.risk_score * 0.85, effective_risk - reduction)
            if abs(summary.skewness_bias) >= 1.2:
                effective_risk = max(
                    effective_risk,
                    min(
                        1.0,
                        assessment.risk_score
                        + min(abs(summary.skewness_bias) / 1.8, 1.0) * 0.25,
                    ),
                )
            if summary.kurtosis_excess >= 1.5:
                effective_risk = max(
                    effective_risk,
                    min(
                        1.0,
                        assessment.risk_score
                        + min(summary.kurtosis_excess / 3.5, 1.0) * 0.25,
                    ),
                )
            if (
                abs(summary.volume_imbalance) >= 0.5
                and summary.liquidity_pressure >= 0.45
            ):
                effective_risk = max(
                    effective_risk,
                    min(
                        1.0,
                        assessment.risk_score
                        + min(abs(summary.volume_imbalance) / 0.7, 1.0) * 0.25,
                    ),
                )
            confidence_penalty = 0.0
            if summary.confidence_trend < -0.05:
                confidence_penalty += min(abs(summary.confidence_trend) * 0.6, 0.15)
            if summary.confidence_volatility >= 0.1:
                confidence_penalty += min(summary.confidence_volatility * 0.5, 0.15)
            if summary.regime_streak <= 1:
                confidence_penalty += 0.08
            if summary.confidence_decay > 0:
                confidence_penalty += min(summary.confidence_decay * 0.6, 0.18)
            if summary.drawdown_pressure >= 0.6:
                confidence_penalty += min(summary.drawdown_pressure * 0.4, 0.18)
            if summary.liquidity_pressure >= 0.6:
                confidence_penalty += min(summary.liquidity_pressure * 0.35, 0.15)
            if summary.stress_index >= 0.5:
                confidence_penalty += min(summary.stress_index * 0.4, 0.2)
            if summary.shock_frequency >= 0.45:
                confidence_penalty += min(summary.shock_frequency * 0.35, 0.18)
            if summary.tail_risk_index >= 0.45:
                confidence_penalty += min(summary.tail_risk_index * 0.35, 0.18)
            if summary.degradation_score >= 0.4:
                confidence_penalty += min(summary.degradation_score * 0.4, 0.2)
            if summary.stability_projection <= 0.45:
                confidence_penalty += min((0.45 - summary.stability_projection) * 0.4, 0.18)
            if summary.volatility_trend > 0.015:
                confidence_penalty += min(summary.volatility_trend / 0.03 * 0.2, 0.12)
            if summary.drawdown_trend > 0.05:
                confidence_penalty += min(summary.drawdown_trend / 0.2 * 0.25, 0.15)
            if summary.volume_trend_volatility >= 0.18:
                confidence_penalty += min(
                    summary.volume_trend_volatility / 0.25 * 0.2,
                    0.12,
                )
            if summary.distribution_pressure >= 0.5:
                confidence_penalty += min(summary.distribution_pressure * 0.4, 0.2)
            if summary.liquidity_gap >= 0.5:
                confidence_penalty += min(summary.liquidity_gap * 0.35, 0.18)
            if summary.stress_projection >= 0.5:
                confidence_penalty += min(summary.stress_projection * 0.35, 0.18)
            if summary.stress_momentum >= 0.5:
                confidence_penalty += min(summary.stress_momentum * 0.35, 0.18)
            if summary.liquidity_trend >= 0.5:
                confidence_penalty += min(summary.liquidity_trend * 0.3, 0.16)
            if summary.confidence_fragility >= 0.45:
                confidence_penalty += min(summary.confidence_fragility * 0.35, 0.2)
            if summary.confidence_resilience <= 0.5:
                confidence_penalty += min(max(0.0, 0.5 - summary.confidence_resilience) * 0.5, 0.2)
            if summary.regime_entropy >= 0.65:
                confidence_penalty += min(summary.regime_entropy * 0.3, 0.15)
            if summary.resilience_score < 0.45:
                confidence_penalty += min(max(0.0, 0.45 - summary.resilience_score) * 0.5, 0.2)
            if summary.stress_balance < 0.5:
                confidence_penalty += min(max(0.0, 0.5 - summary.stress_balance) * 0.45, 0.18)
            if abs(summary.skewness_bias) >= 1.1:
                confidence_penalty += min(abs(summary.skewness_bias) / 2.0, 0.18)
            if summary.kurtosis_excess >= 1.4:
                confidence_penalty += min(summary.kurtosis_excess / 3.2, 0.18)
            if (
                abs(summary.volume_imbalance) >= 0.45
                and summary.liquidity_pressure >= 0.45
            ):
                confidence_penalty += min(abs(summary.volume_imbalance) / 0.7, 0.15)
            if confidence_penalty:
                effective_risk = max(effective_risk, min(1.0, assessment.risk_score + confidence_penalty))

        cooldown_active, cooldown_remaining = self._update_cooldown(
            summary=summary,
            effective_risk=effective_risk,
        )
        self._adjust_strategy_parameters(assessment, aggregated_risk=effective_risk, summary=summary)
        signal = self._map_regime_to_signal(assessment, last_return, summary=summary)
        signal = self._apply_signal_guardrails(signal, effective_risk, summary)
        guardrail_reasons = list(self._last_guardrail_reasons)
        guardrail_triggers = [trigger.to_dict() for trigger in self._last_guardrail_triggers]
        if guardrail_reasons and signal == "hold":
            self._log(
                "Signal overridden by guardrails",
                level=logging.INFO,
                reasons=guardrail_reasons,
                triggers=guardrail_triggers,
            )
        if cooldown_active:
            signal = "hold"
        decision = self._build_risk_decision(
            symbol,
            signal,
            assessment,
            effective_risk=effective_risk,
            summary=summary,
            cooldown_active=cooldown_active,
            cooldown_remaining=cooldown_remaining,
            cooldown_reason=self._cooldown_reason,
            guardrail_reasons=guardrail_reasons,
            guardrail_triggers=self._last_guardrail_triggers,
        )

        self._last_signal = signal
        self._last_regime = assessment
        self._last_risk_decision = decision

        self._log(
            f"Auto-trade decision[{symbol}]: regime={assessment.regime.value} signal={signal} risk={assessment.risk_score:.2f} effective_risk={effective_risk:.2f}",
            level=logging.INFO,
        )
        if hasattr(self.emitter, "emit"):
            payload = {
                "symbol": symbol,
                "signal": signal,
                "regime": assessment.regime.value,
                "confidence": assessment.confidence,
                "risk_score": assessment.risk_score,
                "effective_risk": effective_risk,
                "strategy": self.current_strategy,
                "cooldown_active": cooldown_active,
                "cooldown_remaining_s": cooldown_remaining,
                "cooldown_reason": self._cooldown_reason,
                "decision": decision.to_dict(),
                "guardrail_reasons": guardrail_reasons,
                "guardrail_triggers": guardrail_triggers,
            }
            if summary is not None:
                payload["summary"] = summary.to_dict()
            try:  # pragma: no cover - optional integration
                self.emitter.emit("auto_trade_signal", **payload)
            except Exception:
                self._log("Emitter failed to broadcast auto_trade_signal", level=logging.DEBUG)

        self._auto_trade_stop.wait(self.auto_trade_interval_s)

    # Compatibility helpers -------------------------------------------
    def set_enable_auto_trade(self, flag: bool) -> None:
        self.enable_auto_trade = bool(flag)
        if not flag:
            self.confirm_auto_trade(False)

    def is_running(self) -> bool:
        return self._started and not self._stop.is_set()


__all__ = ["AutoTrader", "RiskDecision", "EmitterLike", "GuardrailTrigger"]
