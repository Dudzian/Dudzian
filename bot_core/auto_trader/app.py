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
from datetime import datetime, timezone, tzinfo
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, cast

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


_NO_FILTER = object()
_UNKNOWN_SERVICE = "<unknown>"
_CONTROLLER_HISTORY_DEFAULT_LIMIT = 32


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
        risk_evaluations_limit: int | None = 256,
        risk_evaluations_ttl_s: float | None = None,
        controller_runner: Any | None = None,
        controller_runner_factory: Callable[[], Any] | None = None,
        controller_cycle_history_limit: int | None = 32,
        controller_cycle_history_ttl_s: float | None = None,
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

        self._controller_runner: Any | None = controller_runner
        self._controller_runner_factory: Callable[[], Any] | None = controller_runner_factory

        self.current_strategy: str = "neutral"
        self.current_leverage: float = 1.0
        self.current_stop_loss_pct: float = 0.02
        self.current_take_profit_pct: float = 0.04
        self._last_signal: str | None = None
        self._last_regime: MarketRegimeAssessment | None = None
        self._last_risk_decision: RiskDecision | None = None
        self._controller_cycle_signals: tuple[Any, ...] | None = None
        self._controller_cycle_results: tuple[Any, ...] | None = None
        self._controller_cycle_started_at: float | None = None
        self._controller_cycle_finished_at: float | None = None
        self._controller_cycle_last_duration: float | None = None
        self._controller_cycle_sequence: int = 0
        self._controller_cycle_last_orders: int = 0
        self._controller_cycle_history: list[dict[str, Any]] = []
        self._controller_cycle_history_limit = self._normalise_cycle_history_limit(
            controller_cycle_history_limit
        )
        self._controller_cycle_history_ttl_s = self._normalise_cycle_history_ttl(
            controller_cycle_history_ttl_s
        )
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
        self._risk_evaluations: list[dict[str, Any]] = []
        self._risk_evaluations_limit: int | None = None
        self._risk_evaluations_ttl_s: float | None = self._normalise_cycle_history_ttl(
            risk_evaluations_ttl_s
        )
        self.configure_risk_evaluation_history(risk_evaluations_limit)

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

    def configure_controller_runner(
        self,
        runner: Any | None = None,
        *,
        factory: Callable[[], Any] | None = None,
    ) -> None:
        """Configure an optional realtime runner bridging controller signals to TradingController."""

        with self._lock:
            self._controller_runner = runner
            self._controller_runner_factory = factory

        if runner is not None:
            self._log("Controller runner attached", level=logging.INFO)
        elif factory is not None:
            self._log("Controller runner factory configured", level=logging.INFO)
        else:
            self._log("Controller runner disabled", level=logging.DEBUG)

    def _resolve_controller_runner(self) -> Any | None:
        with self._lock:
            runner = self._controller_runner
            factory = self._controller_runner_factory if runner is None else None

        if runner is not None:
            return runner
        if factory is None:
            return None

        try:
            candidate = factory()
        except Exception as exc:  # pragma: no cover - defensive guard
            self._log(
                f"Controller runner factory failed: {exc!r}",
                level=logging.ERROR,
            )
            return None

        if candidate is None:
            self._log("Controller runner factory returned None", level=logging.DEBUG)
            return None

        with self._lock:
            self._controller_runner = candidate

        self._log("Controller runner instantiated", level=logging.INFO)
        return candidate

    def _execute_controller_runner_cycle(self, runner: Any) -> None:
        run_once = getattr(runner, "run_once", None)
        if not callable(run_once):
            self._log(
                "Configured controller runner does not expose run_once(); disabling bridge",
                level=logging.ERROR,
            )
            with self._lock:
                if runner is self._controller_runner:
                    self._controller_runner = None
            return

        invocation_started = time.time()

        try:
            results = run_once()
        except Exception as exc:  # pragma: no cover - defensive guard
            self._log(
                f"Controller runner cycle failed: {exc!r}",
                level=logging.ERROR,
            )
            return

        def _normalise_sequence(payload: Any) -> tuple[Any, ...]:
            if payload is None:
                return ()
            if isinstance(payload, tuple):
                return payload
            if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes)):
                try:
                    return tuple(payload)
                except TypeError:
                    payload = list(payload)
                    return tuple(payload)
            return (payload,)

        cycle_signals = getattr(runner, "last_cycle_signals", None)
        raw_cycle_results = getattr(runner, "last_cycle_results", None)
        if raw_cycle_results is None:
            raw_cycle_results = results

        stored_signals = _normalise_sequence(cycle_signals)
        stored_results = _normalise_sequence(raw_cycle_results)

        orders_count = len(stored_results)
        last_signal_label: str | None = None
        if stored_signals:
            try:
                signal_payload = getattr(stored_signals[-1], "signal", stored_signals[-1])
                side = getattr(signal_payload, "side", None)
                if isinstance(side, str):
                    last_signal_label = side.lower()
            except Exception:  # pragma: no cover - optional metadata only
                last_signal_label = None

        started_at = getattr(runner, "last_cycle_started_at", None)
        started_timestamp: float | None = None
        if started_at is not None:
            if hasattr(started_at, "timestamp"):
                try:
                    started_timestamp = float(started_at.timestamp())  # type: ignore[call-arg]
                except Exception:  # pragma: no cover - defensive guard
                    started_timestamp = None
            else:
                try:
                    started_timestamp = float(started_at)  # type: ignore[arg-type]
                except (TypeError, ValueError):  # pragma: no cover - defensive guard
                    started_timestamp = None
        elif stored_signals or stored_results:
            started_timestamp = float(invocation_started)

        finished_timestamp = float(time.time())
        duration_seconds: float | None = None
        if started_timestamp is not None:
            duration_seconds = max(0.0, finished_timestamp - started_timestamp)
        elif stored_signals or stored_results:
            duration_seconds = max(0.0, finished_timestamp - invocation_started)
        telemetry_payload = {
            "signals": stored_signals,
            "results": stored_results,
            "started_at": started_timestamp,
            "finished_at": finished_timestamp,
            "duration_s": duration_seconds,
            "orders": orders_count,
        }

        self._log(
            "AutoTrader controller runner executed cycle",
            level=logging.INFO,
            orders=orders_count,
            last_signal=last_signal_label,
            signals=len(stored_signals),
        )

        sequence = 0
        trimmed_by_limit = 0
        trimmed_by_ttl = 0
        limit_snapshot = self._controller_cycle_history_limit
        ttl_snapshot = self._controller_cycle_history_ttl_s
        history_size = 0
        with self._lock:
            self._controller_cycle_signals = stored_signals
            self._controller_cycle_results = stored_results
            self._controller_cycle_started_at = started_timestamp
            self._controller_cycle_finished_at = finished_timestamp
            self._controller_cycle_last_duration = duration_seconds
            self._controller_cycle_last_orders = orders_count
            self._controller_cycle_sequence += 1
            sequence = self._controller_cycle_sequence

            history_entry = {
                "sequence": sequence,
                "signals": stored_signals,
                "results": stored_results,
                "started_at": started_timestamp,
                "finished_at": finished_timestamp,
                "duration_s": duration_seconds,
                "orders": orders_count,
            }
            self._controller_cycle_history.append(history_entry)
            trimmed_by_limit, trimmed_by_ttl = self._prune_controller_cycle_history_locked(
                reference_time=finished_timestamp
            )
            limit_snapshot = self._controller_cycle_history_limit
            ttl_snapshot = self._controller_cycle_history_ttl_s
            history_size = len(self._controller_cycle_history)

        telemetry_payload["sequence"] = sequence

        if trimmed_by_limit or trimmed_by_ttl:
            self._log(
                "Przycięto historię cykli kontrolera po nowym cyklu",
                level=logging.DEBUG,
                limit=None if limit_snapshot <= 0 else limit_snapshot,
                ttl=ttl_snapshot,
                trimmed_by_limit=trimmed_by_limit,
                trimmed_by_ttl=trimmed_by_ttl,
                history=history_size,
            )

        if last_signal_label:
            self._last_signal = last_signal_label
        self._last_risk_decision = None

        emitter_emit = getattr(self.emitter, "emit", None)
        if callable(emitter_emit):
            try:
                emitter_emit("auto_trader.controller_cycle", **telemetry_payload)
            except Exception:  # pragma: no cover - defensive logging
                self._log(
                    "Emitter failed to publish controller cycle telemetry",
                    level=logging.DEBUG,
                )

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
        runner = self._resolve_controller_runner()
        if runner is not None:
            self._execute_controller_runner_cycle(runner)
            self._auto_trade_stop.wait(self.auto_trade_interval_s)
            return

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

    def _log_risk_history_trimmed(
        self,
        *,
        context: str,
        trimmed: int,
        ttl: float | None,
        history: int,
    ) -> None:
        if trimmed:
            self._log(
                "Przycięto historię ocen ryzyka na podstawie TTL",
                level=logging.DEBUG,
                context=context,
                trimmed=trimmed,
                ttl=ttl,
                history=history,
            )

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
        trimmed_by_ttl = 0
        ttl_snapshot: float | None = None
        history_size = 0
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
            trimmed_by_ttl = self._prune_risk_evaluations_locked(
                reference_time=entry["timestamp"]
            )
            ttl_snapshot = self._risk_evaluations_ttl_s
            history_size = len(self._risk_evaluations)
        self._log_risk_history_trimmed(
            context="record",
            trimmed=trimmed_by_ttl,
            ttl=ttl_snapshot,
            history=history_size,
        )

    def _prune_risk_evaluations_locked(
        self,
        *,
        reference_time: float | None = None,
    ) -> int:
        trimmed = 0
        ttl = self._risk_evaluations_ttl_s
        if ttl is None or ttl <= 0.0:
            return 0
        history = self._risk_evaluations
        if not history:
            return 0

        try:
            cutoff_reference = (
                float(reference_time)
                if reference_time is not None
                else float(time.time())
            )
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            cutoff_reference = float(time.time())

        cutoff = cutoff_reference - ttl
        if cutoff <= float("-inf"):
            return 0

        retained: list[dict[str, Any]] = []
        for entry in history:
            timestamp = entry.get("timestamp")
            if timestamp is None or timestamp >= cutoff:
                retained.append(entry)
            else:
                trimmed += 1
        if trimmed:
            history[:] = retained
        return trimmed

    # Compatibility helpers -------------------------------------------
    def set_enable_auto_trade(self, flag: bool) -> None:
        self.enable_auto_trade = bool(flag)
        if not flag:
            self.confirm_auto_trade(False)

    def is_running(self) -> bool:
        return self._started and not self._stop.is_set()

    @staticmethod
    def _normalise_cycle_history_limit(limit: int | None) -> int:
        if limit is None:
            return -1
        try:
            normalized = int(limit)
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            return _CONTROLLER_HISTORY_DEFAULT_LIMIT
        if normalized <= 0:
            return -1
        return normalized

    @staticmethod
    def _normalise_cycle_history_ttl(ttl: float | None) -> float | None:
        if ttl is None:
            return None
        try:
            normalized = float(ttl)
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            return None
        if not normalized or normalized <= 0.0:
            return None
        return float(normalized)

    @staticmethod
    def _normalize_history_export_limit(limit: object) -> int | None:
        if limit is None:
            return None
        try:
            normalized = int(limit)
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            return None
        if normalized <= 0:
            return 0
        return normalized

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

        trimmed_by_ttl = 0
        ttl_snapshot: float | None = None
        history_size = 0
        with self._lock:
            trimmed_by_ttl = self._prune_risk_evaluations_locked()
            records = list(self._risk_evaluations)
            ttl_snapshot = self._risk_evaluations_ttl_s
            history_size = len(self._risk_evaluations)
        self._log_risk_history_trimmed(
            context="get",
            trimmed=trimmed_by_ttl,
            ttl=ttl_snapshot,
            history=history_size,
        )

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

    def _prune_controller_cycle_history_locked(
        self,
        *,
        reference_time: float | None = None,
    ) -> tuple[int, int]:
        trimmed_by_limit = 0
        trimmed_by_ttl = 0
        history = self._controller_cycle_history

        limit = self._controller_cycle_history_limit
        if limit > 0 and len(history) > limit:
            trimmed_by_limit = len(history) - limit
            if trimmed_by_limit > 0:
                del history[:trimmed_by_limit]

        ttl = self._controller_cycle_history_ttl_s
        if ttl is not None and ttl > 0.0 and history:
            try:
                cutoff_reference = (
                    float(reference_time)
                    if reference_time is not None
                    else float(time.time())
                )
            except (TypeError, ValueError):  # pragma: no cover - defensive guard
                cutoff_reference = float(time.time())

            cutoff = cutoff_reference - ttl
            if cutoff > float("-inf"):
                retained: list[dict[str, Any]] = []
                for entry in history:
                    timestamp = entry.get("finished_at")
                    if timestamp is None:
                        timestamp = entry.get("started_at")
                    if timestamp is None or timestamp >= cutoff:
                        retained.append(entry)
                    else:
                        trimmed_by_ttl += 1
                if trimmed_by_ttl:
                    history[:] = retained

        return trimmed_by_limit, trimmed_by_ttl

    def get_last_controller_cycle(self) -> dict[str, Any] | None:
        """Zwraca zrzut ostatniego cyklu runnera realtime.

        Słownik zawiera surowe obiekty sygnałów i wyników zwrócone przez runnera
        oraz znacznik czasu rozpoczęcia cyklu (w sekundach unix epoch), jeśli był
        dostępny.  Zwracana jest kopia danych, dzięki czemu wywołujący nie może
        zmodyfikować wewnętrznego stanu AutoTradera.
        """

        duration = None
        orders = 0
        with self._lock:
            if (
                self._controller_cycle_signals is None
                and self._controller_cycle_results is None
                and self._controller_cycle_started_at is None
                and self._controller_cycle_finished_at is None
            ):
                return None

            signals = tuple(self._controller_cycle_signals or ())
            results = tuple(self._controller_cycle_results or ())
            started_at = self._controller_cycle_started_at
            finished_at = self._controller_cycle_finished_at
            sequence = self._controller_cycle_sequence
            duration = self._controller_cycle_last_duration
            orders = self._controller_cycle_last_orders

        if (
            not signals
            and not results
            and started_at is None
            and finished_at is None
        ):
            return None

        return {
            "signals": signals,
            "results": results,
            "started_at": started_at,
            "finished_at": finished_at,
            "sequence": sequence,
            "duration_s": duration,
            "orders": orders,
        }

    def get_controller_cycle_history(
        self,
        *,
        limit: int | None = None,
        reverse: bool = False,
    ) -> list[dict[str, Any]]:
        """Zwraca historię cykli bridge'a realtime.

        Parametr ``limit`` ogranicza liczbę rekordów (domyślnie wykorzystuje
        wewnętrzny limit AutoTradera), a ``reverse`` pozwala uzyskać dane w
        kolejności malejącej po sekwencji.
        """

        if limit is not None:
            try:
                normalized_limit = int(limit)
            except (TypeError, ValueError):  # pragma: no cover - defensive guard
                normalized_limit = None
            else:
                if normalized_limit < 0:
                    normalized_limit = 0
                if normalized_limit == 0:
                    return []
        else:
            normalized_limit = None

        with self._lock:
            history = list(self._controller_cycle_history)

        if not history:
            return []

        iterator: Iterable[dict[str, Any]]
        if reverse:
            iterator = reversed(history)
        else:
            iterator = iter(history)

        results: list[dict[str, Any]] = []
        for entry in iterator:
            copied = {
                "sequence": entry.get("sequence"),
                "signals": tuple(entry.get("signals", ())),
                "results": tuple(entry.get("results", ())),
                "started_at": entry.get("started_at"),
                "finished_at": entry.get("finished_at"),
                "duration_s": entry.get("duration_s"),
                "orders": entry.get("orders"),
            }
            results.append(copied)
            if normalized_limit is not None and len(results) >= normalized_limit:
                break
        return results

    def set_controller_cycle_history_limit(self, limit: int | None) -> int:
        """Aktualizuje limit przechowywania historii cykli kontrolera.

        Zwracana wartość to znormalizowany limit – ``-1`` oznacza brak
        ograniczenia (historia rośnie do rozmiaru pamięci).  Podanie
        ``None`` lub wartości nie-dodatniej dezaktywuje przycinanie historii.
        """

        normalized = self._normalise_cycle_history_limit(limit)
        trimmed_by_limit = 0
        trimmed_by_ttl = 0
        ttl_snapshot: float | None = None
        history_size = 0
        with self._lock:
            self._controller_cycle_history_limit = normalized
            trimmed_by_limit, trimmed_by_ttl = self._prune_controller_cycle_history_locked()
            ttl_snapshot = self._controller_cycle_history_ttl_s
            history_size = len(self._controller_cycle_history)
        self._log(
            "Zmieniono limit historii cykli kontrolera",
            level=logging.DEBUG,
            limit=None if normalized <= 0 else normalized,
            ttl=ttl_snapshot,
            trimmed_by_limit=trimmed_by_limit,
            trimmed_by_ttl=trimmed_by_ttl,
            history=history_size,
        )
        return normalized

    def get_controller_cycle_history_ttl(self) -> float | None:
        """Zwraca obowiązujący TTL (w sekundach) dla historii cykli kontrolera."""

        with self._lock:
            ttl = self._controller_cycle_history_ttl_s
        return ttl

    def set_controller_cycle_history_ttl(self, ttl: float | None) -> float | None:
        """Aktualizuje czas życia rekordów historii cykli kontrolera."""

        normalized = self._normalise_cycle_history_ttl(ttl)
        trimmed_by_limit = 0
        trimmed_by_ttl = 0
        limit_snapshot = 0
        history_size = 0
        with self._lock:
            self._controller_cycle_history_ttl_s = normalized
            trimmed_by_limit, trimmed_by_ttl = self._prune_controller_cycle_history_locked()
            limit_snapshot = self._controller_cycle_history_limit
            history_size = len(self._controller_cycle_history)
        self._log(
            "Zmieniono TTL historii cykli kontrolera",
            level=logging.DEBUG,
            ttl=normalized,
            limit=None if limit_snapshot <= 0 else limit_snapshot,
            trimmed_by_limit=trimmed_by_limit,
            trimmed_by_ttl=trimmed_by_ttl,
            history=history_size,
        )
        return normalized

    def clear_controller_cycle_history(self) -> None:
        """Usuwa wszystkie zapisane cykle kontrolera."""

        cleared = 0
        with self._lock:
            if self._controller_cycle_history:
                cleared = len(self._controller_cycle_history)
                self._controller_cycle_history.clear()
        if cleared:
            self._log(
                "Wyczyszczono historię cykli kontrolera",
                level=logging.DEBUG,
                cleared=cleared,
            )

    def summarize_controller_cycle_history(
        self,
        *,
        since: object = None,
        until: object = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Buduje zbiorczy raport z historii cykli kontrolera.

        Parametry ``since`` i ``until`` pozwalają ograniczyć analizę do
        zadanego przedziału czasowego (akceptują ``datetime``, ``Timestamp``
        Pandas oraz float/int jako sekundę epoki).  Opcjonalny ``limit``
        ogranicza liczbę najnowszych rekordów uwzględnionych w raporcie –
        ``0`` zwraca pusty raport.
        """

        normalized_limit: int | None
        if limit is None:
            normalized_limit = None
        else:
            try:
                normalized_limit = int(limit)
            except (TypeError, ValueError):  # pragma: no cover - defensive guard
                normalized_limit = None
            else:
                if normalized_limit <= 0:
                    normalized_limit = 0

        since_ts = self._normalize_time_bound(since)
        until_ts = self._normalize_time_bound(until)

        with self._lock:
            history_snapshot = list(self._controller_cycle_history)
            limit_cfg = self._controller_cycle_history_limit
            ttl_cfg = self._controller_cycle_history_ttl_s

        effective_history: list[dict[str, Any]] = []
        for entry in history_snapshot:
            timestamp = entry.get("finished_at")
            if timestamp is None:
                timestamp = entry.get("started_at")
            if since_ts is not None and (timestamp is None or timestamp < since_ts):
                continue
            if until_ts is not None and (timestamp is None or timestamp > until_ts):
                continue
            effective_history.append(entry)

        if normalized_limit == 0:
            effective_history = []
        elif normalized_limit is not None and len(effective_history) > normalized_limit:
            effective_history = effective_history[-normalized_limit:]

        total = len(effective_history)
        summary: dict[str, Any] = {
            "total": total,
            "filters": {
                "since": since_ts,
                "until": until_ts,
                "limit": normalized_limit,
            },
            "config": {
                "limit": None if limit_cfg <= 0 else limit_cfg,
                "ttl": ttl_cfg,
            },
        }

        if total == 0:
            summary.update(
                {
                    "orders": {
                        "total": 0,
                        "average": 0.0,
                        "min": 0,
                        "max": 0,
                    },
                    "signals": {
                        "total": 0,
                        "average": 0.0,
                        "min": 0,
                        "max": 0,
                        "by_side": {},
                    },
                    "results": {
                        "total": 0,
                        "average": 0.0,
                        "min": 0,
                        "max": 0,
                        "status_counts": {},
                    },
                    "duration": {
                        "total": 0.0,
                        "average": 0.0,
                        "min": None,
                        "max": None,
                    },
                    "first_sequence": None,
                    "last_sequence": None,
                    "first_timestamp": None,
                    "last_timestamp": None,
                }
            )
            return summary

        orders_per_cycle: list[int] = []
        signals_per_cycle: list[int] = []
        results_per_cycle: list[int] = []
        durations: list[float] = []
        signal_sides: Counter[str] = Counter()
        result_statuses: Counter[str] = Counter()
        first_sequence: int | None = None
        last_sequence: int | None = None
        first_timestamp: float | None = None
        last_timestamp: float | None = None

        for entry in effective_history:
            sequence = entry.get("sequence")
            if sequence is not None:
                try:
                    sequence_int = int(sequence)
                except (TypeError, ValueError):  # pragma: no cover - defensive guard
                    sequence_int = None
                else:
                    if first_sequence is None:
                        first_sequence = sequence_int
                    last_sequence = sequence_int

            timestamp = entry.get("finished_at")
            if timestamp is None:
                timestamp = entry.get("started_at")
            if timestamp is not None:
                try:
                    timestamp_float = float(timestamp)
                except (TypeError, ValueError):  # pragma: no cover - defensive guard
                    timestamp_float = None
                else:
                    if first_timestamp is None:
                        first_timestamp = timestamp_float
                    last_timestamp = timestamp_float

            orders_value = entry.get("orders")
            if isinstance(orders_value, (int, float)):
                orders_count = max(0, int(orders_value))
            else:
                orders_count = len(entry.get("results", ()) or ())
            orders_per_cycle.append(orders_count)

            signals_sequence = entry.get("signals") or ()
            results_sequence = entry.get("results") or ()

            signals_count = len(signals_sequence)
            results_count = len(results_sequence)
            signals_per_cycle.append(signals_count)
            results_per_cycle.append(results_count)

            duration_value = entry.get("duration_s")
            if duration_value is not None:
                try:
                    durations.append(max(0.0, float(duration_value)))
                except (TypeError, ValueError):  # pragma: no cover - defensive guard
                    pass

            for raw_signal in signals_sequence:
                side = None
                payload = getattr(raw_signal, "signal", raw_signal)
                if isinstance(payload, Mapping):
                    side = payload.get("side")
                if side is None:
                    side = getattr(payload, "side", None)
                if side is None and isinstance(raw_signal, Mapping):
                    side = raw_signal.get("side")
                if side is None:
                    side = getattr(raw_signal, "side", None)
                if side is None:
                    continue
                side_str = str(side).lower()
                signal_sides[side_str] += 1

            for raw_result in results_sequence:
                status = getattr(raw_result, "status", None)
                if status is None and isinstance(raw_result, Mapping):
                    status = raw_result.get("status")
                if status is None:
                    continue
                result_statuses[str(status).lower()] += 1

        def _aggregate_numbers(values: list[int]) -> dict[str, Any]:
            if not values:
                return {"total": 0, "average": 0.0, "min": 0, "max": 0}
            total_value = sum(values)
            return {
                "total": total_value,
                "average": total_value / len(values),
                "min": min(values),
                "max": max(values),
            }

        duration_metrics: dict[str, Any]
        if durations:
            total_duration = sum(durations)
            duration_metrics = {
                "total": total_duration,
                "average": total_duration / len(durations),
                "min": min(durations),
                "max": max(durations),
            }
        else:
            duration_metrics = {
                "total": 0.0,
                "average": 0.0,
                "min": None,
                "max": None,
            }

        summary.update(
            {
                "orders": _aggregate_numbers(orders_per_cycle),
                "signals": {
                    **_aggregate_numbers(signals_per_cycle),
                    "by_side": dict(signal_sides),
                },
                "results": {
                    **_aggregate_numbers(results_per_cycle),
                    "status_counts": dict(result_statuses),
                },
                "duration": duration_metrics,
                "first_sequence": first_sequence,
                "last_sequence": last_sequence,
                "first_timestamp": first_timestamp,
                "last_timestamp": last_timestamp,
            }
        )
        return summary

    def _filtered_controller_cycle_history(
        self,
        *,
        since_ts: float | None,
        until_ts: float | None,
        reverse: bool,
    ) -> list[tuple[dict[str, Any], float | None, float | None]]:
        with self._lock:
            history_snapshot = list(self._controller_cycle_history)

        if not history_snapshot:
            return []

        filtered: list[tuple[dict[str, Any], float | None, float | None]] = []
        for entry in history_snapshot:
            started_raw = entry.get("started_at")
            finished_raw = entry.get("finished_at")
            started_ts = self._normalize_time_bound(started_raw)
            finished_ts = self._normalize_time_bound(finished_raw)
            pivot_ts = finished_ts if finished_ts is not None else started_ts
            if since_ts is not None and (pivot_ts is None or pivot_ts < since_ts):
                continue
            if until_ts is not None and (pivot_ts is None or pivot_ts > until_ts):
                continue
            filtered.append((entry, started_ts, finished_ts))

        if reverse:
            filtered.reverse()

        return filtered

    def controller_cycle_history_to_records(
        self,
        *,
        since: object = None,
        until: object = None,
        limit: int | None = None,
        reverse: bool = False,
        include_sequences: bool = True,
        include_counts: bool = True,
        coerce_timestamps: bool = False,
        tz: tzinfo | None = timezone.utc,
    ) -> list[dict[str, Any]]:
        """Zwraca listę rekordów historii cykli kontrolera."""

        normalized_limit = self._normalize_history_export_limit(limit)
        if normalized_limit == 0:
            return []

        since_ts = self._normalize_time_bound(since)
        until_ts = self._normalize_time_bound(until)

        filtered = self._filtered_controller_cycle_history(
            since_ts=since_ts,
            until_ts=until_ts,
            reverse=reverse,
        )

        if not filtered:
            return []

        def _convert_timestamp(value_ts: float | None, raw: object) -> object:
            if not coerce_timestamps:
                return raw
            if value_ts is None:
                return None
            if tz is not None:
                return datetime.fromtimestamp(value_ts, tz=tz)
            return datetime.fromtimestamp(value_ts, tz=timezone.utc).replace(tzinfo=None)

        records: list[dict[str, Any]] = []
        for entry, started_ts, finished_ts in filtered:
            signals = tuple(entry.get("signals", ()) or ())
            results = tuple(entry.get("results", ()) or ())
            orders_value = entry.get("orders")
            if isinstance(orders_value, (int, float)):
                orders_count = max(0, int(orders_value))
            else:
                orders_count = len(results)

            started_raw = entry.get("started_at")
            finished_raw = entry.get("finished_at")

            record: dict[str, Any] = {
                "sequence": entry.get("sequence"),
                "duration_s": entry.get("duration_s"),
                "orders": orders_count,
                "started_at": _convert_timestamp(started_ts, started_raw),
                "finished_at": _convert_timestamp(finished_ts, finished_raw),
            }

            if include_counts:
                record["signals_count"] = len(signals)
                record["results_count"] = len(results)

            if include_sequences:
                record["signals"] = signals
                record["results"] = results

            records.append(record)
            if normalized_limit is not None and len(records) >= normalized_limit:
                break

        return records

    def controller_cycle_history_to_dataframe(
        self,
        *,
        since: object = None,
        until: object = None,
        limit: int | None = None,
        reverse: bool = False,
        include_sequences: bool = True,
        include_counts: bool = True,
        coerce_timestamps: bool = True,
    ) -> pd.DataFrame:
        """Buduje ``DataFrame`` z historią cykli kontrolera.

        Parametry ``since`` i ``until`` filtrują rekordy według czasu zakończenia
        (z zapasem czasu rozpoczęcia jeśli ``finished_at`` jest niedostępne).
        ``limit`` oraz ``reverse`` odwzorowują zachowanie ``get_controller_cycle_history``.
        ``include_sequences`` pozwala kontrolować obecność surowych sekwencji sygnałów
        i wyników, natomiast ``include_counts`` dodaje kolumny z ich licznością.
        Włączenie ``coerce_timestamps`` zamienia znaczniki czasu na ``Timestamp`` UTC,
        co ułatwia dalszą analizę w Pandas.
        """

        normalized_limit = self._normalize_history_export_limit(limit)
        if normalized_limit == 0:
            columns = [
                "sequence",
                "started_at",
                "finished_at",
                "duration_s",
                "orders",
            ]
            if include_counts:
                columns.extend(["signals_count", "results_count"])
            if include_sequences:
                columns.extend(["signals", "results"])
            return pd.DataFrame(columns=columns)

        since_ts = self._normalize_time_bound(since)
        until_ts = self._normalize_time_bound(until)

        filtered = self._filtered_controller_cycle_history(
            since_ts=since_ts,
            until_ts=until_ts,
            reverse=reverse,
        )

        if not filtered:
            columns = [
                "sequence",
                "started_at",
                "finished_at",
                "duration_s",
                "orders",
            ]
            if include_counts:
                columns.extend(["signals_count", "results_count"])
            if include_sequences:
                columns.extend(["signals", "results"])
            return pd.DataFrame(columns=columns)

        rows: list[dict[str, Any]] = []
        for entry, started_ts, finished_ts in filtered:
            signals = tuple(entry.get("signals", ()) or ())
            results = tuple(entry.get("results", ()) or ())
            orders_value = entry.get("orders")
            if isinstance(orders_value, (int, float)):
                orders_count = max(0, int(orders_value))
            else:
                orders_count = len(results)

            started_raw = entry.get("started_at")
            finished_raw = entry.get("finished_at")

            row: dict[str, Any] = {
                "sequence": entry.get("sequence"),
                "duration_s": entry.get("duration_s"),
                "orders": orders_count,
            }

            if coerce_timestamps:
                row["started_at"] = (
                    pd.to_datetime(started_ts, unit="s", utc=True)
                    if started_ts is not None
                    else pd.NaT
                )
                row["finished_at"] = (
                    pd.to_datetime(finished_ts, unit="s", utc=True)
                    if finished_ts is not None
                    else pd.NaT
                )
            else:
                row["started_at"] = started_raw
                row["finished_at"] = finished_raw

            if include_counts:
                row["signals_count"] = len(signals)
                row["results_count"] = len(results)

            if include_sequences:
                row["signals"] = signals
                row["results"] = results

            rows.append(row)
            if normalized_limit is not None and len(rows) >= normalized_limit:
                break

        df = pd.DataFrame.from_records(rows)

        expected_columns = [
            "sequence",
            "started_at",
            "finished_at",
            "duration_s",
            "orders",
        ]
        if include_counts:
            expected_columns.extend(["signals_count", "results_count"])
        if include_sequences:
            expected_columns.extend(["signals", "results"])

        for column in expected_columns:
            if column not in df.columns:
                df[column] = pd.NA

        return df[expected_columns]

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
        trimmed_by_ttl = 0
        ttl_snapshot: float | None = None
        history_size = 0
        with self._lock:
            trimmed_by_ttl = self._prune_risk_evaluations_locked()
            records = list(self._risk_evaluations)
            ttl_snapshot = self._risk_evaluations_ttl_s
            history_size = len(self._risk_evaluations)
        self._log_risk_history_trimmed(
            context="summarize",
            trimmed=trimmed_by_ttl,
            ttl=ttl_snapshot,
            history=history_size,
        )

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

        trimmed_by_ttl = 0
        ttl_snapshot: float | None = None
        history_size = 0
        with self._lock:
            trimmed_by_ttl = self._prune_risk_evaluations_locked()
            records = list(self._risk_evaluations)
            ttl_snapshot = self._risk_evaluations_ttl_s
            history_size = len(self._risk_evaluations)
        self._log_risk_history_trimmed(
            context="dataframe",
            trimmed=trimmed_by_ttl,
            ttl=ttl_snapshot,
            history=history_size,
        )

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
        trimmed_by_ttl = 0
        ttl_snapshot: float | None = None
        history_size = 0
        with self._lock:
            self._risk_evaluations_limit = normalised
            if normalised is not None:
                if normalised == 0:
                    self._risk_evaluations.clear()
                else:
                    overflow = len(self._risk_evaluations) - normalised
                    if overflow > 0:
                        del self._risk_evaluations[:overflow]
            trimmed_by_ttl = self._prune_risk_evaluations_locked()
            ttl_snapshot = self._risk_evaluations_ttl_s
            history_size = len(self._risk_evaluations)
        self._log_risk_history_trimmed(
            context="configure",
            trimmed=trimmed_by_ttl,
            ttl=ttl_snapshot,
            history=history_size,
        )

    def get_risk_evaluations_ttl(self) -> float | None:
        """Zwraca obowiązujący TTL (w sekundach) dla historii ocen ryzyka."""

        with self._lock:
            ttl = self._risk_evaluations_ttl_s
        return ttl

    def set_risk_evaluations_ttl(self, ttl: float | None) -> float | None:
        """Aktualizuje czas życia historii ocen ryzyka."""

        normalized = self._normalise_cycle_history_ttl(ttl)
        trimmed_by_ttl = 0
        history_size = 0
        limit_snapshot: int | None = None
        with self._lock:
            self._risk_evaluations_ttl_s = normalized
            trimmed_by_ttl = self._prune_risk_evaluations_locked()
            history_size = len(self._risk_evaluations)
            limit_snapshot = self._risk_evaluations_limit
        self._log(
            "Zmieniono TTL historii ocen ryzyka",
            level=logging.DEBUG,
            ttl=normalized,
            limit=limit_snapshot,
            trimmed=trimmed_by_ttl,
            history=history_size,
        )
        return normalized


__all__ = ["AutoTrader", "RiskDecision", "EmitterLike", "GuardrailTrigger"]
