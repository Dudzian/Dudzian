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

import copy
import enum
import logging
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol, cast

import pandas as pd

from bot_core.ai.regime import (
    MarketRegime,
    MarketRegimeAssessment,
    MarketRegimeClassifier,
    RegimeSummary,
    RiskLevel,
)


LOGGER = logging.getLogger(__name__)


_NO_FILTER = object()
_UNKNOWN_SERVICE = "<unknown>"


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
        risk_evaluations_limit: int | None = 256,
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

        self.current_strategy: str = "neutral"
        self.current_leverage: float = 1.0
        self.current_stop_loss_pct: float = 0.02
        self.current_take_profit_pct: float = 0.04
        self._last_signal: str | None = None
        self._last_regime: MarketRegimeAssessment | None = None
        self._last_risk_decision: RiskDecision | None = None
        self._cooldown_until: float = 0.0
        self._cooldown_reason: str | None = None

        self._stop = threading.Event()
        self._auto_trade_stop = threading.Event()
        self._auto_trade_thread: threading.Thread | None = None
        self._auto_trade_thread_active = False
        self._auto_trade_user_confirmed = False
        self._started = False
        self._lock = threading.RLock()
        self._risk_evaluations: list[dict[str, Any]] = []
        self._risk_evaluations_limit: int | None = None
        self.configure_risk_evaluation_history(risk_evaluations_limit)

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

    @staticmethod
    def _map_regime_to_signal(
        assessment: MarketRegimeAssessment,
        last_return: float,
        *,
        summary: RegimeSummary | None = None,
    ) -> str:
        if assessment.confidence < 0.2:
            return "hold"
        if summary is not None and summary.confidence < 0.45:
            return "hold"
        if summary is not None and summary.stability < 0.4:
            return "hold"
        if summary is not None and summary.risk_level in {RiskLevel.ELEVATED, RiskLevel.CRITICAL}:
            return "hold"
        if summary is not None and summary.risk_trend > 0.15:
            return "hold"
        if summary is not None and summary.risk_volatility > 0.18:
            return "hold"
        if summary is not None and summary.regime_persistence < 0.25:
            return "hold"
        if summary is not None and summary.transition_rate > 0.55:
            return "hold"
        if summary is not None and summary.confidence_trend < -0.15:
            return "hold"
        if summary is not None and summary.confidence_volatility >= 0.15:
            return "hold"
        if summary is not None and summary.regime_streak <= 1 and summary.stability < 0.7:
            return "hold"
        if summary is not None and summary.resilience_score <= 0.3:
            return "hold"
        if summary is not None and summary.stress_balance <= 0.35:
            return "hold"
        if summary is not None and summary.regime_entropy >= 0.75:
            return "hold"
        if summary is not None and summary.instability_score > 0.65:
            return "hold"
        if summary is not None and summary.confidence_decay > 0.2:
            return "hold"
        if summary is not None and summary.drawdown_pressure >= 0.6:
            return "hold"
        if summary is not None and summary.liquidity_pressure >= 0.65:
            return "hold"
        if summary is not None and summary.volatility_ratio >= 1.55:
            return "hold"
        if summary is not None and summary.degradation_score >= 0.55:
            return "hold"
        if summary is not None and summary.stability_projection <= 0.4:
            return "hold"
        if summary is not None and summary.volume_trend_volatility >= 0.18:
            return "hold"
        if summary is not None and summary.liquidity_gap >= 0.6:
            return "hold"
        if summary is not None and summary.stress_projection >= 0.6:
            return "hold"
        if summary is not None and summary.confidence_resilience <= 0.4:
            return "hold"
        if summary is not None and summary.distribution_pressure >= 0.55:
            return "hold"
        if summary is not None and abs(summary.skewness_bias) >= 1.2 and summary.risk_score >= 0.45:
            return "hold"
        if summary is not None and summary.kurtosis_excess >= 1.5 and summary.risk_score >= 0.45:
            return "hold"
        if (
            summary is not None
            and abs(summary.volume_imbalance) >= 0.5
            and summary.liquidity_pressure >= 0.45
        ):
            return "hold"
        if summary is not None and summary.volatility_trend > 0.02:
            return "hold"
        if summary is not None and summary.drawdown_trend > 0.08:
            return "hold"
        if assessment.risk_score >= 0.75:
            return "hold"
        if assessment.regime is MarketRegime.TREND:
            return "buy" if last_return >= 0 else "sell"
        if assessment.regime is MarketRegime.MEAN_REVERSION:
            return "sell" if last_return > 0 else "buy"
        threshold = 0.001
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
        if risk >= 0.75:
            self.current_strategy = "capital_preservation"
            self.current_leverage = 0.0
            self.current_stop_loss_pct = 0.01
            self.current_take_profit_pct = 0.02
        elif assessment.regime is MarketRegime.TREND:
            self.current_strategy = "trend_following"
            self.current_leverage = 2.0 if risk < 0.4 else 1.5
            self.current_stop_loss_pct = 0.03 if risk < 0.4 else 0.04
            self.current_take_profit_pct = 0.06 if risk < 0.4 else 0.04
        elif assessment.regime is MarketRegime.MEAN_REVERSION:
            self.current_strategy = "mean_reversion"
            self.current_leverage = 1.0 if risk < 0.4 else 0.7
            self.current_stop_loss_pct = 0.015 if risk < 0.4 else 0.02
            self.current_take_profit_pct = 0.03 if risk < 0.4 else 0.025
        else:
            self.current_strategy = "intraday_breakout"
            self.current_leverage = 0.8 if risk < 0.5 else 0.5
            self.current_stop_loss_pct = 0.02 if risk < 0.5 else 0.03
            self.current_take_profit_pct = 0.025 if risk < 0.5 else 0.02

        if summary is not None:
            if summary.risk_level is RiskLevel.CRITICAL:
                self.current_strategy = "capital_preservation"
                self.current_leverage = 0.0
                self.current_stop_loss_pct = 0.01
                self.current_take_profit_pct = 0.02
            elif summary.risk_level is RiskLevel.ELEVATED:
                self.current_leverage = min(self.current_leverage, 0.8 if assessment.regime is MarketRegime.TREND else 0.5)
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.85, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.4)
            elif summary.risk_level is RiskLevel.CALM and risk < 0.5:
                self.current_leverage = min(self.current_leverage * 1.2, 2.5)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.1, 0.08)

            if summary.resilience_score <= 0.35:
                self.current_leverage = min(self.current_leverage, 0.45)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.88, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.92, self.current_stop_loss_pct * 1.35)
            elif (
                summary.resilience_score >= 0.65
                and summary.stress_balance >= 0.6
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and risk < 0.6
            ):
                self.current_leverage = min(self.current_leverage * 1.05, 3.2)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.05, 0.09)

            if summary.regime_entropy >= 0.75:
                self.current_leverage = min(self.current_leverage, 0.45)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.9, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.92, self.current_stop_loss_pct * 1.35)
            elif (
                summary.regime_entropy <= 0.45
                and summary.resilience_score >= 0.6
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and risk < 0.6
            ):
                self.current_leverage = min(self.current_leverage * 1.03, 3.25)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.03, 0.09)

            if summary.risk_volatility >= 0.2:
                self.current_leverage = min(self.current_leverage, 0.6)
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.9, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.95, self.current_stop_loss_pct * 1.3)

            if summary.regime_persistence <= 0.3:
                self.current_leverage = min(self.current_leverage, 0.4)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                else:
                    if not self.current_strategy.endswith("_probing"):
                        self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.85, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.4)
            elif (
                summary.regime_persistence >= 0.65
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and summary.risk_volatility < 0.15
                and summary.instability_score <= 0.4
            ):
                self.current_leverage = min(self.current_leverage * 1.05, 2.75)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.05, 0.085)

            if summary.confidence_volatility >= 0.15:
                self.current_leverage = min(self.current_leverage, 0.5)
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.9, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.92, self.current_stop_loss_pct * 1.35)

            if summary.confidence_trend < -0.1:
                self.current_leverage = min(self.current_leverage, 0.6)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.85, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.4)

            if summary.regime_streak <= 1:
                self.current_leverage = min(self.current_leverage, 0.4)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.85, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.4)

            if (
                summary.confidence_trend > 0.12
                and summary.confidence_volatility <= 0.08
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and summary.regime_persistence >= 0.6
                and summary.instability_score <= 0.35
            ):
                self.current_leverage = min(self.current_leverage * 1.1, 3.0)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.08, 0.09)

            if summary.transition_rate >= 0.5:
                self.current_leverage = min(self.current_leverage, 0.45)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.85, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.4)

            if summary.instability_score >= 0.75:
                self.current_strategy = "capital_preservation"
                self.current_leverage = 0.0
                self.current_stop_loss_pct = 0.01
                self.current_take_profit_pct = 0.02
            elif summary.instability_score >= 0.6:
                self.current_leverage = min(self.current_leverage, 0.5)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.85, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.92, self.current_stop_loss_pct * 1.4)
            elif summary.instability_score <= 0.25 and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}:
                self.current_leverage = min(self.current_leverage * 1.08, 3.2)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.07, 0.095)

            if summary.drawdown_pressure >= 0.75:
                self.current_strategy = "capital_preservation"
                self.current_leverage = 0.0
                self.current_stop_loss_pct = 0.01
                self.current_take_profit_pct = 0.02
            elif summary.drawdown_pressure >= 0.55:
                self.current_leverage = min(self.current_leverage, 0.45)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.85, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.4)
            elif summary.drawdown_pressure <= 0.3 and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}:
                self.current_leverage = min(self.current_leverage * 1.06, 3.1)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.05, 0.09)

            if summary.liquidity_pressure >= 0.7:
                self.current_leverage = min(self.current_leverage, 0.35)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.88, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.35)
            elif (
                summary.liquidity_pressure <= 0.35
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and risk < 0.5
            ):
                self.current_leverage = min(self.current_leverage * 1.04, 3.0)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.04, 0.085)

            if summary.confidence_decay >= 0.25:
                self.current_leverage = min(self.current_leverage, 0.45)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.85, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.4)

            if summary.degradation_score >= 0.6:
                self.current_strategy = "capital_preservation"
                self.current_leverage = 0.0
                self.current_stop_loss_pct = 0.01
                self.current_take_profit_pct = 0.02
            elif summary.degradation_score >= 0.45:
                self.current_leverage = min(self.current_leverage, 0.35)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.84, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.35)
            elif (
                summary.degradation_score <= 0.25
                and summary.stability_projection >= 0.6
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and risk < 0.5
            ):
                self.current_leverage = min(self.current_leverage * 1.03, 3.1)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.04, 0.09)

            if summary.stability_projection <= 0.4:
                self.current_leverage = min(self.current_leverage, 0.4)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.84, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.35)
            elif (
                summary.stability_projection >= 0.65
                and summary.degradation_score <= 0.3
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and risk < 0.5
            ):
                self.current_leverage = min(self.current_leverage * 1.02, 3.05)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.02, 0.09)

            if summary.volume_trend_volatility >= 0.2:
                self.current_leverage = min(self.current_leverage, 0.35)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.83, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.35)
            elif (
                summary.volume_trend_volatility <= 0.1
                and summary.degradation_score <= 0.3
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
            ):
                self.current_leverage = min(self.current_leverage * 1.01, 3.0)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.01, 0.085)

            if summary.volatility_trend > 0.02:
                self.current_leverage = min(self.current_leverage, 0.4)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.85, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.35)
            elif (
                summary.volatility_trend <= 0.0
                and summary.degradation_score <= 0.3
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
            ):
                self.current_leverage = min(self.current_leverage * 1.01, 3.1)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.01, 0.088)

            if summary.drawdown_trend > 0.08:
                self.current_leverage = min(self.current_leverage, 0.35)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.82, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.88, self.current_stop_loss_pct * 1.35)
            elif (
                summary.drawdown_trend <= 0.0
                and summary.degradation_score <= 0.3
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
            ):
                self.current_leverage = min(self.current_leverage * 1.02, 3.05)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.02, 0.09)

            if summary.stress_index >= 0.8:
                self.current_strategy = "capital_preservation"
                self.current_leverage = 0.0
                self.current_stop_loss_pct = 0.01
                self.current_take_profit_pct = 0.02
            elif summary.stress_index >= 0.6:
                self.current_leverage = min(self.current_leverage, 0.35)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.82, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.88, self.current_stop_loss_pct * 1.35)
            elif (
                summary.stress_index <= 0.28
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and risk < 0.55
            ):
                self.current_leverage = min(self.current_leverage * 1.04, 3.2)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.04, 0.09)

            if summary.stress_momentum >= 0.65:
                self.current_leverage = min(self.current_leverage, 0.3)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.82, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.88, self.current_stop_loss_pct * 1.35)
            elif (
                summary.stress_momentum <= 0.35
                and summary.stress_index <= 0.35
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and risk < 0.5
            ):
                self.current_leverage = min(self.current_leverage * 1.02, 3.05)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.02, 0.088)

            if summary.tail_risk_index >= 0.55:
                self.current_leverage = min(self.current_leverage, 0.35)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.84, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.35)
            elif (
                summary.tail_risk_index <= 0.2
                and summary.stress_index <= 0.3
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and risk < 0.5
            ):
                self.current_leverage = min(self.current_leverage * 1.03, 3.1)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.03, 0.085)

            if summary.shock_frequency >= 0.55:
                self.current_leverage = min(self.current_leverage, 0.3)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.82, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.88, self.current_stop_loss_pct * 1.35)
            elif (
                summary.shock_frequency <= 0.25
                and summary.regime_persistence >= 0.6
                and summary.stress_index <= 0.35
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
            ):
                self.current_leverage = min(self.current_leverage * 1.02, 3.0)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.02, 0.085)

            if (
                summary.distribution_pressure >= 0.65
                or abs(summary.skewness_bias) >= 1.3
                or summary.kurtosis_excess >= 1.8
                or (
                    abs(summary.volume_imbalance) >= 0.55
                    and summary.liquidity_pressure >= 0.45
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
                summary.distribution_pressure <= 0.3
                and abs(summary.skewness_bias) <= 0.8
                and summary.kurtosis_excess <= 1.0
                and abs(summary.volume_imbalance) <= 0.35
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and risk < 0.5
            ):
                self.current_leverage = min(self.current_leverage * 1.02, 3.05)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.02, 0.09)

            if summary.liquidity_gap >= 0.65:
                self.current_leverage = min(self.current_leverage, 0.35)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.84, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.35)
            elif (
                summary.liquidity_gap <= 0.35
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and summary.resilience_score >= 0.55
                and risk < 0.5
            ):
                self.current_leverage = min(self.current_leverage * 1.02, 3.05)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.02, 0.09)

            if summary.liquidity_trend >= 0.6:
                self.current_leverage = min(self.current_leverage, 0.35)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.84, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.35)
            elif (
                summary.liquidity_trend <= 0.35
                and summary.liquidity_gap <= 0.4
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and risk < 0.5
            ):
                self.current_leverage = min(self.current_leverage * 1.02, 3.05)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.02, 0.09)

            if summary.stress_projection >= 0.65:
                self.current_strategy = "capital_preservation"
                self.current_leverage = 0.0
                self.current_stop_loss_pct = 0.01
                self.current_take_profit_pct = 0.02
            elif summary.stress_projection >= 0.55:
                self.current_leverage = min(self.current_leverage, 0.35)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.84, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.35)
            elif (
                summary.stress_projection <= 0.35
                and summary.stress_index <= 0.35
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and risk < 0.45
            ):
                self.current_leverage = min(self.current_leverage * 1.02, 3.0)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.02, 0.085)

            if summary.confidence_resilience <= 0.4:
                self.current_leverage = min(self.current_leverage, 0.45)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.85, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.35)
            elif (
                summary.confidence_resilience >= 0.65
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and risk < 0.45
                and summary.resilience_score >= 0.6
            ):
                self.current_leverage = min(self.current_leverage * 1.03, 3.1)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.03, 0.09)

            if summary.confidence_fragility >= 0.55:
                self.current_leverage = min(self.current_leverage, 0.35)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.84, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.35)
            elif (
                summary.confidence_fragility <= 0.35
                and summary.confidence_resilience >= 0.6
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
                and risk < 0.45
            ):
                self.current_leverage = min(self.current_leverage * 1.02, 3.0)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.02, 0.085)

            if summary.volatility_of_volatility >= 0.03:
                self.current_leverage = min(self.current_leverage, 0.4)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                elif not self.current_strategy.endswith("_probing"):
                    self.current_strategy = f"{self.current_strategy}_probing"
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.85, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.35)
            elif (
                summary.volatility_of_volatility <= 0.015
                and summary.stress_index <= 0.35
                and summary.risk_level in {RiskLevel.CALM, RiskLevel.BALANCED}
            ):
                self.current_leverage = min(self.current_leverage * 1.02, 3.05)
                self.current_take_profit_pct = min(self.current_take_profit_pct * 1.02, 0.085)

        if summary is not None and risk < 0.75:
            if summary.risk_trend > 0.05:
                self.current_leverage = max(0.0, self.current_leverage - 0.3)
                self.current_stop_loss_pct = max(self.current_stop_loss_pct * 0.8, 0.01)
                self.current_take_profit_pct = max(self.current_take_profit_pct * 0.9, self.current_stop_loss_pct * 1.5)
            if summary.stability < 0.4:
                self.current_leverage = min(self.current_leverage, 0.5)
                if self.current_leverage == 0:
                    self.current_strategy = "capital_preservation"
                else:
                    self.current_strategy = f"{self.current_strategy}_probing"

        self.current_leverage = float(max(self.current_leverage, 0.0))
        self.current_stop_loss_pct = float(max(self.current_stop_loss_pct, 0.005))
        self.current_take_profit_pct = float(max(self.current_take_profit_pct, self.current_stop_loss_pct * 1.2))

    def _update_cooldown(
        self,
        *,
        summary: RegimeSummary | None,
        effective_risk: float,
    ) -> tuple[bool, float]:
        now = time.monotonic()
        if summary is not None:
            severity = max(
                summary.cooldown_score,
                summary.severe_event_rate * 0.8,
                summary.stress_index * 0.7,
                summary.stress_projection * 0.75,
                summary.stress_momentum * 0.7,
                summary.degradation_score * 0.75,
                max(0.0, 0.45 - summary.stability_projection) * 0.6,
                summary.liquidity_gap * 0.65,
                summary.liquidity_trend * 0.6,
                max(0.0, 0.45 - summary.confidence_resilience) * 0.6,
                summary.confidence_fragility * 0.65,
            )
            distribution_flags = max(
                summary.distribution_pressure,
                min(1.0, abs(summary.skewness_bias) / 1.6),
                min(1.0, max(0.0, summary.kurtosis_excess) / 3.0),
                min(1.0, abs(summary.volume_imbalance) / 0.55),
            )
            severity = max(
                severity,
                distribution_flags * 0.75,
                max(0.0, 0.55 - summary.resilience_score) * 0.65,
                max(0.0, 0.5 - summary.stress_balance) * 0.6,
                summary.stress_projection * 0.7,
                summary.liquidity_gap * 0.6,
                summary.stress_momentum * 0.65,
                summary.liquidity_trend * 0.6,
                summary.confidence_fragility * 0.65,
                max(0.0, summary.regime_entropy - 0.65) * 0.6,
            )
            if (
                effective_risk >= 0.85
                or summary.risk_level is RiskLevel.CRITICAL
                or severity >= 0.75
                or summary.degradation_score >= 0.7
                or summary.stability_projection <= 0.25
                or summary.distribution_pressure >= 0.8
                or distribution_flags >= 0.85
                or summary.resilience_score <= 0.25
                or summary.stress_balance <= 0.25
                or summary.regime_entropy >= 0.85
                or summary.stress_projection >= 0.75
                or summary.liquidity_gap >= 0.75
                or summary.stress_momentum >= 0.75
                or summary.liquidity_trend >= 0.7
                or summary.confidence_resilience <= 0.25
                or summary.confidence_fragility >= 0.7
            ):
                duration = max(self.auto_trade_interval_s * 5.0, 300.0)
                self._cooldown_until = max(self._cooldown_until, now + duration)
                self._cooldown_reason = "critical_risk"
            elif (
                severity >= 0.55
                or (
                    summary.risk_level is RiskLevel.ELEVATED
                    and summary.stress_index >= 0.6
                )
                or summary.degradation_score >= 0.55
                or summary.stability_projection <= 0.35
                or summary.distribution_pressure >= 0.6
                or distribution_flags >= 0.65
                or summary.resilience_score <= 0.4
                or summary.stress_balance <= 0.4
                or summary.regime_entropy >= 0.75
                or summary.stress_projection >= 0.6
                or summary.liquidity_gap >= 0.6
                or summary.stress_momentum >= 0.6
                or summary.liquidity_trend >= 0.6
                or summary.confidence_resilience <= 0.35
                or summary.confidence_fragility >= 0.6
            ):
                duration = max(self.auto_trade_interval_s * 3.0, 180.0)
                self._cooldown_until = max(self._cooldown_until, now + duration)
                self._cooldown_reason = "elevated_risk"
            elif (
                severity >= 0.45
                and summary.risk_level in {RiskLevel.ELEVATED, RiskLevel.WATCH}
                or summary.degradation_score >= 0.45
                or summary.distribution_pressure >= 0.5
                or distribution_flags >= 0.55
                or summary.resilience_score <= 0.45
                or summary.stress_balance <= 0.4
                or summary.regime_entropy >= 0.65
                or summary.stress_projection >= 0.5
                or summary.liquidity_gap >= 0.5
                or summary.stress_momentum >= 0.5
                or summary.liquidity_trend >= 0.5
                or summary.confidence_resilience <= 0.4
                or summary.confidence_fragility >= 0.5
            ):
                duration = max(self.auto_trade_interval_s * 2.0, 120.0)
                self._cooldown_until = max(self._cooldown_until, now + duration)
                self._cooldown_reason = "instability_spike"
            elif (
                summary.cooldown_score <= 0.35
                and summary.recovery_potential >= 0.6
                and effective_risk <= 0.55
                and summary.degradation_score <= 0.35
                and summary.stability_projection >= 0.45
                and summary.distribution_pressure <= 0.4
                and abs(summary.skewness_bias) <= 0.9
                and summary.kurtosis_excess <= 1.2
                and abs(summary.volume_imbalance) <= 0.4
                and summary.resilience_score >= 0.55
                and summary.stress_balance >= 0.5
                and summary.regime_entropy <= 0.55
                and summary.liquidity_gap <= 0.4
                and summary.stress_projection <= 0.4
                and summary.stress_momentum <= 0.4
                and summary.liquidity_trend <= 0.4
                and summary.confidence_resilience >= 0.55
                and summary.confidence_fragility <= 0.4
            ):
                self._cooldown_until = 0.0
                self._cooldown_reason = None
        elif effective_risk >= 0.9:
            duration = max(self.auto_trade_interval_s * 2.0, 150.0)
            self._cooldown_until = max(self._cooldown_until, now + duration)
            self._cooldown_reason = "high_risk"

        remaining = max(0.0, self._cooldown_until - now)
        active = remaining > 0.0
        if active and summary is not None:
            if (
                summary.recovery_potential >= 0.7
                and summary.cooldown_score <= 0.4
                and summary.severe_event_rate <= 0.4
                and effective_risk <= 0.55
                and summary.degradation_score <= 0.35
                and summary.stability_projection >= 0.5
                and summary.distribution_pressure <= 0.4
                and abs(summary.skewness_bias) <= 0.9
                and summary.kurtosis_excess <= 1.2
                and abs(summary.volume_imbalance) <= 0.4
                and summary.resilience_score >= 0.6
                and summary.stress_balance >= 0.55
                and summary.regime_entropy <= 0.5
                and summary.liquidity_gap <= 0.4
                and summary.stress_projection <= 0.35
                and summary.confidence_resilience >= 0.6
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
        risk_service = getattr(self, "risk_service", None)
        if risk_service is None:
            risk_service = getattr(self, "core_risk_engine", None)

        execution_service = getattr(self, "execution_service", None)
        if execution_service is None:
            execution_service = getattr(self, "core_execution_service", None)

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
        if effective_risk >= 0.75:
            signal = "hold"
        elif summary is not None and (
            summary.stress_index >= 0.65
            or summary.tail_risk_index >= 0.6
            or summary.shock_frequency >= 0.6
            or summary.stress_momentum >= 0.65
        ):
            signal = "hold"
        elif summary is not None and (
            summary.resilience_score <= 0.3
            or summary.stress_balance <= 0.35
            or summary.regime_entropy >= 0.8
            or summary.confidence_fragility >= 0.65
        ):
            signal = "hold"
        elif summary is not None and (
            summary.degradation_score >= 0.6
            or summary.stability_projection <= 0.35
            or summary.volume_trend_volatility >= 0.2
            or summary.volatility_trend > 0.025
            or summary.drawdown_trend > 0.1
            or summary.liquidity_gap >= 0.6
            or summary.stress_projection >= 0.6
            or summary.confidence_resilience <= 0.4
            or summary.liquidity_trend >= 0.6
        ):
            signal = "hold"
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
            }
            if summary is not None:
                payload["summary"] = summary.to_dict()
            try:  # pragma: no cover - optional integration
                self.emitter.emit("auto_trade_signal", **payload)
            except Exception:
                self._log("Emitter failed to broadcast auto_trade_signal", level=logging.DEBUG)

        normalized_approval: bool | None = None
        recorded_approval: bool | None = None
        risk_response: Any = None
        risk_error: Exception | None = None
        risk_invoked = False
        if risk_service is not None:
            evaluate_fn = getattr(risk_service, "evaluate_decision", None)
            if not callable(evaluate_fn):
                evaluate_fn = getattr(risk_service, "evaluate", None)
            if not callable(evaluate_fn) and callable(risk_service):
                evaluate_fn = cast(Callable[[RiskDecision], Any], risk_service)
            if callable(evaluate_fn):
                risk_invoked = True
                try:
                    risk_response = evaluate_fn(decision)
                    self._store_risk_response_metadata(decision, risk_response)
                except Exception as exc:  # pragma: no cover - defensive guard
                    self._log(
                        f"Risk service evaluation failed: {exc!r}",
                        level=logging.ERROR,
                    )
                    normalized_approval = False
                    recorded_approval = False
                    risk_error = exc
                else:
                    recorded_approval = self._coerce_risk_approval(risk_response)
                    if recorded_approval is None:
                        self._log(
                            "Risk service returned an unsupported approval response; treating as rejected",
                            level=logging.DEBUG,
                        )
                        normalized_approval = False
                    else:
                        normalized_approval = recorded_approval
        if risk_invoked:
            self._record_risk_evaluation(
                decision,
                approved=recorded_approval,
                normalized=normalized_approval,
                response=risk_response,
                service=risk_service,
                error=risk_error,
            )

        if normalized_approval:
            with self._lock:
                cooldown_active = decision.cooldown_active
                should_trade = decision.should_trade
                service = execution_service

            if cooldown_active:
                self._log(
                    "Risk evaluation approved trade but cooldown is active; skipping execution",
                    level=logging.DEBUG,
                )
            elif not should_trade:
                self._log(
                    "Risk evaluation approved trade but decision is not actionable; skipping execution",
                    level=logging.DEBUG,
                )
            elif service is not None:
                execute_fn = getattr(service, "execute_decision", None)
                if not callable(execute_fn):
                    execute_fn = getattr(service, "execute", None)
                if not callable(execute_fn) and callable(service):
                    execute_fn = cast(Callable[[RiskDecision], Any], service)
                if callable(execute_fn):
                    try:
                        execute_fn(decision)
                    except Exception as exc:  # pragma: no cover - defensive guard
                        self._log(
                            f"Execution service failed to execute trade: {exc!r}",
                            level=logging.ERROR,
                        )
            else:
                self._log(
                    "Risk evaluation approved trade but execution service is not configured",
                    level=logging.DEBUG,
                )

        self._auto_trade_stop.wait(self.auto_trade_interval_s)

    @staticmethod
    def _coerce_risk_approval(response: Any) -> bool | None:
        if response is None:
            return False
        if isinstance(response, bool):
            return response
        if isinstance(response, (int, float)):
            return response > 0
        if isinstance(response, enum.Enum):
            enum_result = AutoTrader._coerce_risk_approval(response.value)
            if enum_result is not None:
                return enum_result
            return AutoTrader._coerce_risk_approval(response.name)
        if isinstance(response, str):
            lowered = response.strip().lower()
            if lowered in {
                "true",
                "t",
                "yes",
                "y",
                "approved",
                "approve",
                "allow",
                "allowed",
                "ok",
                "go",
                "proceed",
            }:
                return True
            if lowered in {
                "false",
                "f",
                "no",
                "n",
                "deny",
                "denied",
                "block",
                "blocked",
                "stop",
            }:
                return False
            try:
                numeric = float(lowered)
            except ValueError:
                return None
            return numeric > 0
        if isinstance(response, (list, tuple)):
            for item in response:
                coerced = AutoTrader._coerce_risk_approval(item)
                if coerced is not None:
                    return coerced
            return None
        if isinstance(response, dict):
            for key in (
                "approved",
                "approve",
                "allow",
                "allowed",
                "ok",
                "permitted",
                "should_trade",
                "should_execute",
            ):
                if key in response:
                    return AutoTrader._coerce_risk_approval(response[key])
            return None
        for key in (
            "approved",
            "approve",
            "allow",
            "allowed",
            "ok",
            "permitted",
            "should_trade",
            "should_execute",
        ):
            if hasattr(response, key):
                return AutoTrader._coerce_risk_approval(getattr(response, key))
        return None

    @staticmethod
    def _truncate_repr(value: Any, *, limit: int = 160) -> str:
        text = repr(value)
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    @staticmethod
    def _summarize_risk_response(response: Any) -> dict[str, Any]:
        summary: dict[str, Any] = {"type": type(response).__name__}
        if isinstance(response, (bool, int, float)):
            summary["value"] = response
        elif isinstance(response, str):
            trimmed = response.strip()
            summary["value"] = trimmed if len(trimmed) <= 120 else trimmed[:117] + "..."
        elif isinstance(response, dict):
            summary["keys"] = sorted(map(str, response.keys()))[:8]
        elif isinstance(response, (list, tuple, set)):
            preview = list(response)[:3]
            summary["size"] = len(response)
            if preview:
                summary["preview"] = [AutoTrader._truncate_repr(item, limit=60) for item in preview]
        else:
            summary["repr"] = AutoTrader._truncate_repr(response)
        return summary

    @staticmethod
    def _store_risk_response_metadata(decision: RiskDecision, response: Any) -> None:
        summary = AutoTrader._summarize_risk_response(response)
        bucket = decision.details.setdefault("risk_service", {})
        bucket["response"] = summary

    def _record_risk_evaluation(
        self,
        decision: RiskDecision,
        *,
        approved: bool | None,
        normalized: bool | None,
        response: Any,
        service: Any,
        error: Exception | None,
    ) -> None:
        normalized_value = normalized if normalized is not None else approved
        entry: dict[str, Any] = {
            "timestamp": time.time(),
            "approved": approved,
            "normalized": normalized_value,
            "decision": decision.to_dict(),
        }
        if service is not None:
            entry["service"] = type(service).__name__
        if error is not None:
            entry["error"] = repr(error)
        else:
            entry["response"] = self._summarize_risk_response(response)
        with self._lock:
            self._risk_evaluations.append(entry)
            limit = self._risk_evaluations_limit
            if limit is not None and limit >= 0:
                if limit == 0:
                    self._risk_evaluations.clear()
                else:
                    overflow = len(self._risk_evaluations) - limit
                    if overflow > 0:
                        del self._risk_evaluations[:overflow]

    # Compatibility helpers -------------------------------------------
    def set_enable_auto_trade(self, flag: bool) -> None:
        self.enable_auto_trade = bool(flag)
        if not flag:
            self.confirm_auto_trade(False)

    def is_running(self) -> bool:
        return self._started and not self._stop.is_set()

    @staticmethod
    def _prepare_bool_filter(value: object) -> set[bool | None] | None:
        if value is _NO_FILTER:
            return None
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
            return {cast(bool | None, item) for item in value}
        return {cast(bool | None, value)}

    @staticmethod
    def _prepare_service_filter(value: object) -> set[str] | None:
        if value is _NO_FILTER:
            return None
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
            return {
                _UNKNOWN_SERVICE if item is None else str(item)
                for item in value
            }
        if value is None:
            return {_UNKNOWN_SERVICE}
        return {str(value)}

    @staticmethod
    def _normalize_time_bound(value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            return None
        return float(timestamp.value) / 1_000_000_000

    def _apply_risk_evaluation_filters(
        self,
        records: Iterable[dict[str, Any]],
        *,
        include_errors: bool,
        approved_filter: set[bool | None] | None,
        normalized_filter: set[bool | None] | None,
        service_filter: set[str] | None,
        since_ts: float | None,
        until_ts: float | None,
    ) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        for entry in records:
            if not include_errors and "error" in entry:
                continue
            if approved_filter is not None and entry.get("approved") not in approved_filter:
                continue
            if normalized_filter is not None and entry.get("normalized") not in normalized_filter:
                continue
            service_key = entry.get("service") or _UNKNOWN_SERVICE
            if service_filter is not None and service_key not in service_filter:
                continue
            timestamp = entry.get("timestamp")
            if since_ts is not None and (timestamp is None or timestamp < since_ts):
                continue
            if until_ts is not None and (timestamp is None or timestamp > until_ts):
                continue
            filtered.append(entry)
        return filtered

    def get_risk_evaluations(
        self,
        *,
        approved: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        normalized: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        include_errors: bool = True,
        limit: int | None = None,
        reverse: bool = False,
        service: str | None | Iterable[str | None] | object = _NO_FILTER,
        since: Any = None,
        until: Any = None,
    ) -> list[dict[str, Any]]:
        approved_filter = self._prepare_bool_filter(approved)
        normalized_filter = self._prepare_bool_filter(normalized)
        service_filter = self._prepare_service_filter(service)
        since_ts = self._normalize_time_bound(since)
        until_ts = self._normalize_time_bound(until)

        normalized_limit: int | None
        if limit is None:
            normalized_limit = None
        else:
            try:
                normalized_limit = int(limit)
            except (TypeError, ValueError):  # pragma: no cover - defensive guard
                normalized_limit = None
            else:
                if normalized_limit < 0:
                    normalized_limit = 0

        with self._lock:
            records = list(self._risk_evaluations)

        filtered_records = self._apply_risk_evaluation_filters(
            records,
            include_errors=include_errors,
            approved_filter=approved_filter,
            normalized_filter=normalized_filter,
            service_filter=service_filter,
            since_ts=since_ts,
            until_ts=until_ts,
        )

        if reverse:
            iterator = reversed(filtered_records)
        else:
            iterator = iter(filtered_records)

        results: list[dict[str, Any]] = []
        for entry in iterator:
            results.append(copy.deepcopy(entry))
            if normalized_limit is not None and len(results) >= normalized_limit:
                break
        return results

    def clear_risk_evaluations(self) -> None:
        with self._lock:
            self._risk_evaluations.clear()

    def summarize_risk_evaluations(
        self,
        *,
        approved: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        normalized: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        include_errors: bool = True,
        service: str | None | Iterable[str | None] | object = _NO_FILTER,
        since: Any = None,
        until: Any = None,
    ) -> dict[str, Any]:
        approved_filter = self._prepare_bool_filter(approved)
        normalized_filter = self._prepare_bool_filter(normalized)
        service_filter = self._prepare_service_filter(service)
        since_ts = self._normalize_time_bound(since)
        until_ts = self._normalize_time_bound(until)
        with self._lock:
            records = list(self._risk_evaluations)

        filtered_records = self._apply_risk_evaluation_filters(
            records,
            include_errors=include_errors,
            approved_filter=approved_filter,
            normalized_filter=normalized_filter,
            service_filter=service_filter,
            since_ts=since_ts,
            until_ts=until_ts,
        )

        total = len(filtered_records)
        summary: dict[str, Any] = {
            "total": total,
            "approved": 0,
            "rejected": 0,
            "unknown": 0,
            "errors": 0,
            "raw_true": 0,
            "raw_false": 0,
            "raw_none": 0,
            "services": {},
        }
        if total == 0:
            summary["approval_rate"] = 0.0
            summary["error_rate"] = 0.0
            return summary

        summary["first_timestamp"] = filtered_records[0]["timestamp"]
        summary["last_timestamp"] = filtered_records[-1]["timestamp"]

        services_summary: dict[str, dict[str, Any]] = {}

        for entry in filtered_records:
            normalized_value = entry.get("normalized")
            if normalized_value is True:
                summary["approved"] += 1
            elif normalized_value is False:
                summary["rejected"] += 1
            else:
                summary["unknown"] += 1

            raw_value = entry.get("approved")
            if raw_value is True:
                summary["raw_true"] += 1
            elif raw_value is False:
                summary["raw_false"] += 1
            else:
                summary["raw_none"] += 1

            has_error = "error" in entry
            if has_error:
                summary["errors"] += 1

            service_key = entry.get("service") or _UNKNOWN_SERVICE
            bucket = services_summary.setdefault(
                service_key,
                {
                    "total": 0,
                    "approved": 0,
                    "rejected": 0,
                    "unknown": 0,
                    "errors": 0,
                    "raw_true": 0,
                    "raw_false": 0,
                    "raw_none": 0,
                },
            )
            bucket["total"] += 1

            if normalized_value is True:
                bucket["approved"] += 1
            elif normalized_value is False:
                bucket["rejected"] += 1
            else:
                bucket["unknown"] += 1

            if raw_value is True:
                bucket["raw_true"] += 1
            elif raw_value is False:
                bucket["raw_false"] += 1
            else:
                bucket["raw_none"] += 1

            if has_error:
                bucket["errors"] += 1

        summary["services"] = services_summary
        summary["approval_rate"] = summary["approved"] / total
        summary["error_rate"] = summary["errors"] / total

        for bucket in services_summary.values():
            total_bucket = bucket["total"]
            if total_bucket:
                bucket["approval_rate"] = bucket["approved"] / total_bucket
                bucket["error_rate"] = bucket["errors"] / total_bucket
            else:  # pragma: no cover - defensive guard
                bucket["approval_rate"] = 0.0
                bucket["error_rate"] = 0.0

        return summary

    def risk_evaluations_to_dataframe(
        self,
        *,
        approved: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        normalized: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        include_errors: bool = True,
        service: str | None | Iterable[str | None] | object = _NO_FILTER,
        since: Any = None,
        until: Any = None,
        flatten_decision: bool = False,
        decision_prefix: str = "decision_",
        decision_fields: Iterable[Any] | Any | None = None,
        drop_decision_column: bool = False,
        fill_value: Any = pd.NA,
    ) -> pd.DataFrame:
        """Return risk evaluations as a pandas DataFrame with optional filters."""

        approved_filter = self._prepare_bool_filter(approved)
        normalized_filter = self._prepare_bool_filter(normalized)
        service_filter = self._prepare_service_filter(service)
        since_ts = self._normalize_time_bound(since)
        until_ts = self._normalize_time_bound(until)

        normalized_decision_fields: list[Any] | None
        if decision_fields is None:
            normalized_decision_fields = None
        else:
            if isinstance(decision_fields, Iterable) and not isinstance(
                decision_fields,
                (str, bytes, bytearray),
            ):
                candidates = decision_fields
            else:
                candidates = [decision_fields]

            normalized_decision_fields = []
            for candidate in candidates:
                if candidate is None:
                    continue
                if any(existing == candidate for existing in normalized_decision_fields):
                    continue
                normalized_decision_fields.append(candidate)
            if not normalized_decision_fields:
                normalized_decision_fields = []

        with self._lock:
            records = list(self._risk_evaluations)

        filtered_records = self._apply_risk_evaluation_filters(
            records,
            include_errors=include_errors,
            approved_filter=approved_filter,
            normalized_filter=normalized_filter,
            service_filter=service_filter,
            since_ts=since_ts,
            until_ts=until_ts,
        )

        base_columns = [
            "timestamp",
            "approved",
            "normalized",
            "decision",
            "service",
            "response",
            "error",
        ]

        if not filtered_records:
            empty_columns = list(base_columns)
            if drop_decision_column:
                empty_columns = [
                    column for column in empty_columns if column != "decision"
                ]
            if flatten_decision and normalized_decision_fields:
                prefix = str(decision_prefix)
                empty_columns.extend(
                    f"{prefix}{field}" for field in normalized_decision_fields
                )
            return pd.DataFrame(columns=empty_columns)

        rows = [copy.deepcopy(entry) for entry in filtered_records]
        df = pd.DataFrame.from_records(rows)
        for column in base_columns:
            if column not in df.columns:
                df[column] = pd.NA

        flattened_columns: list[str] = []
        if flatten_decision and "decision" in df.columns:
            prefix = str(decision_prefix)
            decision_series = df["decision"]
            if normalized_decision_fields is not None:
                ordered_keys = list(normalized_decision_fields)
            else:
                ordered_keys: list[Any] = []
                for payload in decision_series:
                    if isinstance(payload, dict):
                        for key in payload.keys():
                            if not any(existing == key for existing in ordered_keys):
                                ordered_keys.append(key)
            for key in ordered_keys:
                column_name = f"{prefix}{key}"
                df[column_name] = [
                    copy.deepcopy(payload[key])
                    if isinstance(payload, dict) and key in payload
                    else copy.deepcopy(fill_value)
                    for payload in decision_series
                ]
                flattened_columns.append(column_name)

        if drop_decision_column and "decision" in df.columns:
            df = df.drop(columns=["decision"])

        remaining_columns = [
            column
            for column in df.columns
            if column not in base_columns and column not in flattened_columns
        ]
        ordered_base_columns = [
            column
            for column in base_columns
            if column in df.columns and (column != "decision" or not drop_decision_column)
        ]
        ordered_columns = ordered_base_columns + flattened_columns + remaining_columns
        return df[ordered_columns]

    def configure_risk_evaluation_history(self, limit: int | None) -> None:
        normalised: int | None
        if limit is None:
            normalised = None
        else:
            try:
                normalised = int(limit)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                normalised = None
            else:
                if normalised < 0:
                    normalised = 0
        with self._lock:
            self._risk_evaluations_limit = normalised
            if normalised is not None:
                if normalised == 0:
                    self._risk_evaluations.clear()
                else:
                    overflow = len(self._risk_evaluations) - normalised
                    if overflow > 0:
                        del self._risk_evaluations[:overflow]


__all__ = ["AutoTrader", "RiskDecision", "EmitterLike"]
