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

import asyncio
import copy
import enum
import json
import logging
import math
import threading
import time
import uuid
from bisect import bisect_right
from datetime import datetime, timedelta, timezone, tzinfo
from collections import Counter, OrderedDict
from pathlib import Path
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence, cast

import pandas as pd

from bot_core.alerts.base import AlertMessage, AlertRouter
from bot_core.auto_trader.audit import DecisionAuditLog
from bot_core.auto_trader.schedule import (
    ScheduleOverride,
    ScheduleState,
    TradingSchedule,
)
from bot_core.ai.regime import (
    MarketRegime,
    MarketRegimeAssessment,
    MarketRegimeClassifier,
    RegimeSummary,
    RiskLevel,
)
from bot_core.ai.config_loader import load_risk_thresholds
from bot_core.config.models import DecisionEngineConfig, DecisionOrchestratorThresholds
from bot_core.decision import DecisionCandidate, DecisionEvaluation, DecisionOrchestrator
from bot_core.execution import (
    ExecutionContext,
    ExecutionService,
    MarketMetadata,
    PaperTradingExecutionService,
)
from bot_core.exchanges.base import OrderRequest
from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.observability import MetricsRegistry, get_global_metrics_registry
from bot_core.trading.strategies import StrategyCatalog
from bot_core.trading.strategy_aliasing import (
    StrategyAliasResolver,
    canonical_alias_map,
    normalise_suffixes,
)
from bot_core.runtime.journal import TradingDecisionJournal, log_decision_event


LOGGER = logging.getLogger(__name__)


_NO_FILTER = object()
_UNKNOWN_SERVICE = "<unknown>"
_MISSING_GUARDRAIL_LABEL = "<no-label>"
_MISSING_GUARDRAIL_COMPARATOR = "<no-comparator>"
_MISSING_GUARDRAIL_UNIT = "<no-unit>"
_MISSING_DECISION_STATE = "<no-state>"
_MISSING_DECISION_REASON = "<no-reason>"
_MISSING_DECISION_MODE = "<no-mode>"
_MISSING_DECISION_ID = "<no-decision-id>"
_APPROVAL_APPROVED = "approved"
_APPROVAL_DENIED = "denied"
_APPROVAL_UNKNOWN = "<unknown-approval>"
_NORMALIZED_NORMALIZED = "normalized"
_NORMALIZED_RAW = "raw"
_NORMALIZED_UNKNOWN = "<unknown-normalization>"
_CONTROLLER_HISTORY_DEFAULT_LIMIT = 32
_SCHEDULE_SYMBOL = "<schedule>"


class GuardrailTimelineRecords(list):
    """Lista kubełków timeline'u guardrail wraz z metadanymi podsumowania."""

    def __init__(self, records: Iterable[dict[str, Any]], summary: Mapping[str, Any]):
        super().__init__(records)
        self.summary: dict[str, Any] = dict(summary)


def _extract_guardrail_timeline_metadata(summary: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in (
        "bucket_s",
        "total",
        "evaluations",
        "guardrail_rate",
        "approval_states",
        "normalization_states",
        "first_timestamp",
        "last_timestamp",
        "filters",
        "services",
        "guardrail_reasons",
        "guardrail_triggers",
        "guardrail_trigger_labels",
        "guardrail_trigger_comparators",
        "guardrail_trigger_units",
        "guardrail_trigger_thresholds",
        "guardrail_trigger_values",
        "decision_states",
        "decision_reasons",
        "decision_modes",
        "missing_timestamp",
    ):
        value = summary.get(key)
        if value is not None:
            metadata[key] = copy.deepcopy(value)
    return metadata


class _ServiceDecisionTotals(dict[str, Any]):
    """Rozszerzone statystyki usług z tolerancyjnym porównaniem słowników."""

    def __eq__(self, other: object) -> bool:  # pragma: no cover - try to reuse dict impl when possible
        if isinstance(other, Mapping):
            for key, value in other.items():
                if self.get(key) != value:
                    return False
            return True
        return super().__eq__(other)


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


def _serialize_schedule_window(window: ScheduleWindow) -> dict[str, Any]:
    """Return a serialisable snapshot of a schedule window."""

    payload: dict[str, Any] = {
        "start": window.start.isoformat(),
        "end": window.end.isoformat(),
        "mode": window.mode,
        "allow_trading": bool(window.allow_trading),
        "days": sorted(int(day) for day in window.days),
        "duration_s": int(window.duration.total_seconds()),
    }
    if window.label is not None:
        payload["label"] = window.label
    return payload


def _serialize_schedule_override(override: ScheduleOverride) -> dict[str, Any]:
    """Return a serialisable snapshot of a schedule override."""

    payload: dict[str, Any] = {
        "start": override.start.astimezone(timezone.utc).isoformat(),
        "end": override.end.astimezone(timezone.utc).isoformat(),
        "mode": override.mode,
        "allow_trading": bool(override.allow_trading),
        "duration_s": int(override.duration.total_seconds()),
    }
    if override.label is not None:
        payload["label"] = override.label
    return payload


def _serialize_schedule_state(state: ScheduleState) -> dict[str, Any]:
    """Return a serialisable snapshot of a schedule state."""

    payload: dict[str, Any] = {
        "mode": state.mode,
        "is_open": bool(state.is_open),
    }
    if state.window is not None:
        payload["window"] = _serialize_schedule_window(state.window)
    if state.next_transition is not None:
        payload["next_transition"] = state.next_transition.astimezone(timezone.utc).isoformat()
    if state.override is not None:
        payload["override"] = _serialize_schedule_override(state.override)
    if state.next_override is not None:
        payload["next_override"] = _serialize_schedule_override(state.next_override)
    remaining = state.time_until_transition
    if remaining is not None:
        payload["time_until_transition_s"] = remaining
    next_override_delay = state.time_until_next_override
    if next_override_delay is not None:
        payload["time_until_next_override_s"] = next_override_delay
    payload["override_active"] = state.override_active
    return payload


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
    unit: Optional[str] = None
    value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "label": self.label,
            "comparator": self.comparator,
            "threshold": float(self.threshold),
        }
        if self.unit is not None:
            payload["unit"] = self.unit
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

    _STRATEGY_SUFFIXES: tuple[str, ...] = ("_probing",)
    _STRATEGY_ALIAS_MAP: Mapping[str, str] = {
        "intraday_breakout": "day_trading",
    }
    _ALIAS_RESOLVER: StrategyAliasResolver | None = None

    @classmethod
    def _alias_resolver(cls) -> StrategyAliasResolver:
        resolver = cls._ALIAS_RESOLVER
        if (
            resolver is None
            or resolver.base_alias_map is not cls._STRATEGY_ALIAS_MAP
            or resolver.base_suffixes != cls._STRATEGY_SUFFIXES
        ):
            resolver = StrategyAliasResolver(
                cls._STRATEGY_ALIAS_MAP,
                cls._STRATEGY_SUFFIXES,
            )
            cls._ALIAS_RESOLVER = resolver
        return resolver

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
        trusted_auto_confirm: bool = False,
        work_schedule: TradingSchedule | None = None,
        decision_audit_log: DecisionAuditLog | None = None,
        decision_journal: TradingDecisionJournal | None = None,
        decision_journal_context: Mapping[str, object] | None = None,
        portfolio_manager: Any | None = None,
        strategy_catalog: StrategyCatalog | None = None,
        strategy_alias_map: Mapping[str, str] | None = None,
        strategy_alias_suffixes: Iterable[str] | None = None,
    ) -> None:
        self.emitter = emitter
        self.gui = gui
        self.symbol_getter = symbol_getter
        self.market_data_provider = market_data_provider

        alias_override = canonical_alias_map(strategy_alias_map)
        suffix_override = (
            normalise_suffixes(strategy_alias_suffixes)
            if strategy_alias_suffixes is not None
            else None
        )
        base_resolver = type(self)._alias_resolver()
        override_resolver = base_resolver.extend(
            alias_map=alias_override,
            suffixes=suffix_override,
        )
        self._alias_resolver_override: StrategyAliasResolver | None = (
            None if override_resolver is base_resolver else override_resolver
        )
        self._strategy_alias_map_override: Mapping[str, str] | None = (
            alias_override or None
        )
        self._strategy_alias_suffix_override: tuple[str, ...] | None = (
            tuple(suffix_override) if suffix_override is not None else None
        )

        self.enable_auto_trade = bool(enable_auto_trade)
        self.auto_trade_interval_s = float(auto_trade_interval_s)

        self.signal_service = signal_service
        self._ai_feature_column_names: tuple[str, ...] | None = None
        self._ai_feature_column_source: str = "default"
        self._ai_feature_column_snapshot: tuple[str, ...] = ()
        self._ai_feature_columns: Callable[[pd.DataFrame], list[str]] = (
            self._default_ai_feature_columns
        )
        self.risk_service = risk_service
        self.execution_service = execution_service
        self.data_provider = data_provider
        self.bootstrap_context = bootstrap_context
        self.portfolio_manager = (
            portfolio_manager
            or getattr(gui, "portfolio_manager", None)
            or getattr(gui, "portfolio_mgr", None)
        )
        if self.portfolio_manager is None and bootstrap_context is not None:
            self.portfolio_manager = getattr(bootstrap_context, "portfolio_manager", None)

        bootstrap_orchestrator = (
            getattr(bootstrap_context, "decision_orchestrator", None)
            if bootstrap_context is not None
            else None
        )
        bootstrap_risk_engine = (
            getattr(bootstrap_context, "risk_engine", None)
            if bootstrap_context is not None
            else None
        )
        self._decision_engine_config = (
            getattr(bootstrap_context, "decision_engine_config", None)
            if bootstrap_context is not None
            else None
        )
        self._risk_profile_name = str(
            getattr(bootstrap_context, "risk_profile_name", "paper") or "paper"
        )
        self._portfolio_id = str(
            getattr(bootstrap_context, "portfolio_id", "autotrader") or "autotrader"
        )
        self._environment_name = self._detect_environment_name(bootstrap_context)
        self._strategy_catalog = strategy_catalog or StrategyCatalog.default()
        if core_risk_engine is not None:
            self.core_risk_engine = core_risk_engine
        else:
            self.core_risk_engine = bootstrap_risk_engine or self._build_default_risk_engine()
        self._decision_risk_engine = bootstrap_risk_engine or self.core_risk_engine

        if self._decision_engine_config is None:
            self._decision_engine_config = self._build_default_decision_engine_config()

        self.configure_ai_feature_columns_from_signal_service()

        if bootstrap_orchestrator is not None:
            self.decision_orchestrator = bootstrap_orchestrator
        else:
            self.decision_orchestrator = self._build_decision_orchestrator()
        self._attach_decision_orchestrator()

        self.core_execution_service = core_execution_service
        if self.core_execution_service is None and bootstrap_context is not None:
            self.core_execution_service = getattr(bootstrap_context, "execution_service", None)
        self._default_execution_service: ExecutionService | None = None
        self._default_execution_symbol: str | None = None
        self._execution_context: ExecutionContext | None = None
        self.ai_connector = ai_connector
        self.ai_manager: Any | None = getattr(gui, "ai_mgr", None)
        if self.ai_manager is None and bootstrap_context is not None:
            self.ai_manager = getattr(bootstrap_context, "ai_manager", None)
        self._thresholds_loader: Callable[[], Mapping[str, Any]] = (
            thresholds_loader or load_risk_thresholds
        )
        self._thresholds: Mapping[str, Any] = {}
        self.reload_thresholds()

        if decision_audit_log is not None:
            log_instance = decision_audit_log
        else:
            context_log = (
                getattr(bootstrap_context, "decision_audit_log", None)
                if bootstrap_context is not None
                else None
            )
            log_instance = context_log if context_log is not None else DecisionAuditLog()
        self._decision_audit_log = log_instance
        journal_instance: TradingDecisionJournal | None = decision_journal
        if journal_instance is None and bootstrap_context is not None:
            journal_instance = getattr(bootstrap_context, "decision_journal", None)
        if journal_instance is None:
            journal_instance = getattr(gui, "decision_journal", None)
        if journal_instance is not None and not hasattr(journal_instance, "record"):
            journal_instance = None
        self._decision_journal: TradingDecisionJournal | None = journal_instance

        context_source: Mapping[str, object] | None = decision_journal_context
        if context_source is None and bootstrap_context is not None:
            context_source = getattr(bootstrap_context, "decision_journal_context", None)
        if context_source is None:
            context_source = getattr(gui, "decision_journal_context", None)
        context_map: dict[str, str] = {}
        if isinstance(context_source, Mapping):
            for key, value in context_source.items():
                if value is None:
                    continue
                try:
                    token = str(value).strip()
                except Exception:
                    continue
                if token:
                    context_map[str(key)] = token
        for key, value in (
            ("environment", self._environment_name),
            ("portfolio", self._portfolio_id),
            ("risk_profile", self._risk_profile_name),
        ):
            if key not in context_map and value is not None:
                context_map[key] = str(value)
        self._decision_journal_context: dict[str, str] = context_map
        self._initial_mode = self._detect_initial_mode()
        self._work_schedule = work_schedule or self._build_default_work_schedule()
        self._schedule_state: ScheduleState | None = None
        self._schedule_mode: str = self._initial_mode
        self._auto_restart_backoff_s = 1.0
        self._auto_restart_backoff_max_s = 60.0
        self._restart_attempts = 0
        self._last_schedule_snapshot: tuple[str, bool] | None = None
        self._execution_metadata: dict[str, str] = {}

        self.alert_router: AlertRouter | None = getattr(bootstrap_context, "alert_router", None)
        self._metrics: MetricsRegistry = get_global_metrics_registry()
        self._base_metric_labels: Mapping[str, str] = {
            "environment": self._environment_name,
            "portfolio": self._portfolio_id,
            "risk_profile": self._risk_profile_name,
        }
        self._metric_cycle_total = self._metrics.counter(
            "auto_trader_cycles_total",
            "Liczba wykonanych cykli AutoTradera.",
        )
        self._metric_strategy_switch_total = self._metrics.counter(
            "auto_trader_strategy_switch_total",
            "Liczba przełączeń strategii przez AutoTradera.",
        )
        self._metric_guardrail_blocks_total = self._metrics.counter(
            "auto_trader_guardrail_blocks_total",
            "Liczba blokad transakcji przez guardrail.",
        )
        self._metric_recalibration_total = self._metrics.counter(
            "auto_trader_recalibrations_triggered_total",
            "Liczba zleconych rekalkibracji strategii.",
        )
        self._metric_schedule_closed_seconds = self._metrics.histogram(
            "auto_trader_schedule_block_duration_seconds",
            "Czas oczekiwania na otwarcie harmonogramu handlu.",
            (10, 60, 300, 900, 1800, 3600),
        )
        self._metric_schedule_open_gauge = self._metrics.gauge(
            "auto_trader_schedule_open",
            "Stan harmonogramu handlu (1 oznacza otwarty).",
        )
        self._metric_strategy_state_gauge = self._metrics.gauge(
            "auto_trader_strategy_active",
            "Aktywna strategia handlowa AutoTradera.",
        )
        self._last_strategy_metric: str | None = None
        self._schedule_last_alert_state: bool | None = None

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
        self._last_ai_context: Mapping[str, Any] | None = None
        self._cooldown_until: float = 0.0
        self._cooldown_reason: str | None = None
        self._last_guardrail_reasons: list[str] = []
        self._last_guardrail_triggers: list[GuardrailTrigger] = []
        self._ai_degraded = False
        self._active_decision_id: str | None = None

        self._stop = threading.Event()
        self._auto_trade_stop = threading.Event()
        self._auto_trade_thread: threading.Thread | None = None
        self._auto_trade_thread_active = False
        self._trusted_auto_confirm = bool(trusted_auto_confirm)
        self._auto_trade_user_confirmed = self._trusted_auto_confirm
        self._started = False
        self._lock = threading.RLock()

        initial_state = self.get_schedule_state()
        self._schedule_last_alert_state = initial_state.is_open
        self._metric_schedule_open_gauge.set(
            1.0 if initial_state.is_open else 0.0,
            labels=self._base_metric_labels,
        )
        self._last_schedule_snapshot = (initial_state.mode, initial_state.is_open)
        self._update_strategy_metrics(self.current_strategy)
        self._risk_evaluations: list[dict[str, Any]] = []
        self._risk_evaluations_limit: int | None = None
        self._risk_evaluations_ttl_s: float | None = self._normalise_cycle_history_ttl(
            risk_evaluations_ttl_s
        )
        self._risk_evaluation_listeners: set[Callable[[Mapping[str, Any]], None]] = set()
        self.configure_risk_evaluation_history(risk_evaluations_limit)

    def reload_thresholds(self) -> None:
        """Reload cached risk thresholds from the configured loader."""

        self._thresholds = self._thresholds_loader()

    def _compute_ai_signal_context(
        self,
        ai_manager: Any | None,
        symbol: str,
        market_data: pd.DataFrame,
    ) -> Mapping[str, object] | None:
        """Return AI signal metadata used by guardrails.

        Older versions of :class:`AutoTrader` did not expose this helper which
        made tests crash when they tried to interrogate the signal context.  To
        remain backward compatible we provide a small delegating stub that falls
        back to an empty mapping if the specialised implementation is missing.
        """

        compute = getattr(self, "_compute_ai_signal_context_impl", None)
        if compute is None:
            return {}
        return compute(ai_manager, symbol, market_data)

    def _compute_ai_signal_context_impl(
        self,
        ai_manager: Any | None,
        symbol: str,
        market_data: pd.DataFrame,
    ) -> Mapping[str, object]:
        if ai_manager is None:
            return {}

        threshold_raw = getattr(ai_manager, "ai_threshold_bps", None)
        try:
            threshold_bps = float(threshold_raw) if threshold_raw is not None else 0.0
        except (TypeError, ValueError):
            threshold_bps = 0.0

        predictions: Any | None = None
        predict_series = getattr(ai_manager, "predict_series", None)
        if callable(predict_series):
            try:
                predictions = predict_series(symbol, market_data)
            except TypeError:
                predictions = predict_series(market_data, symbol=symbol)
            except Exception as exc:
                self._log(
                    "AI manager prediction failed", level=logging.DEBUG, symbol=symbol, error=repr(exc)
                )
                predictions = None

        if asyncio.iscoroutine(predictions) or isinstance(predictions, asyncio.Future):
            try:
                predictions = asyncio.run(predictions)
            except RuntimeError:
                loop = asyncio.get_event_loop()
                predictions = loop.run_until_complete(predictions)
        elif hasattr(predictions, "__await__"):
            try:
                predictions = asyncio.run(predictions)  # type: ignore[arg-type]
            except RuntimeError:
                loop = asyncio.get_event_loop()
                predictions = loop.run_until_complete(predictions)  # type: ignore[arg-type]

        if not isinstance(predictions, pd.Series):
            try:
                predictions = pd.Series(predictions) if predictions is not None else None
            except Exception:
                predictions = None

        prediction_value: float | None = None
        evaluated_at: str | float | None = None
        if isinstance(predictions, pd.Series) and not predictions.empty:
            raw_value = predictions.iloc[-1]
            try:
                prediction_value = float(raw_value)
            except (TypeError, ValueError):
                prediction_value = None

            index_value = predictions.index[-1]
            if hasattr(index_value, "isoformat"):
                evaluated_at = index_value.isoformat()  # type: ignore[call-arg]
            elif isinstance(index_value, (int, float)):
                evaluated_at = float(index_value)

        context: dict[str, object] = {"threshold_bps": threshold_bps}
        if evaluated_at is not None:
            context["evaluated_at"] = evaluated_at

        if prediction_value is None or not math.isfinite(prediction_value):
            context["direction"] = "hold"
            return context

        prediction_bps = prediction_value * 10_000.0
        context["prediction"] = prediction_value
        context["prediction_bps"] = prediction_bps

        probability_raw = getattr(ai_manager, "prediction_probability", None)
        if probability_raw is None:
            probability_fn = getattr(ai_manager, "predict_probability", None)
            if callable(probability_fn):
                try:
                    probability_raw = probability_fn(symbol=symbol, market_data=market_data)
                except TypeError:
                    try:
                        probability_raw = probability_fn(symbol, market_data)
                    except Exception:
                        probability_raw = None
                except Exception:
                    probability_raw = None

        try:
            if probability_raw is not None:
                probability = float(probability_raw)
            else:
                probability = None
        except (TypeError, ValueError):
            probability = None
        else:
            if probability is not None and math.isfinite(probability):
                probability = max(0.0, min(1.0, probability))
                context["probability"] = probability

        threshold_abs = abs(threshold_bps)
        direction = "hold"
        if prediction_bps > 0 and prediction_bps >= threshold_abs:
            direction = "buy"
        elif prediction_bps < 0 and -prediction_bps >= threshold_abs:
            direction = "sell"
        context["direction"] = direction
        return context

    def _default_ai_feature_columns(self, market_data: pd.DataFrame) -> list[str]:
        """Return ordered feature columns available in ``market_data``.

        Guardrail logic and historical integrations expect this helper to be
        tolerant – the return value must only contain columns that exist in the
        provided frame.  We prefer the canonical OHLCV ordering but fall back
        to whatever string columns the frame exposes to avoid crashes when the
        upstream service provides bespoke inputs.
        """

        self._ai_feature_column_source = "default"
        preferred = ("open", "high", "low", "close", "volume")
        available = [column for column in preferred if column in market_data.columns]
        if available:
            self._ai_feature_column_snapshot = tuple(available)
            return available
        fallback = self._deduplicate_feature_columns(
            column for column in market_data.columns if isinstance(column, str)
        )
        self._ai_feature_column_snapshot = tuple(fallback)
        return fallback

    @staticmethod
    def _normalize_feature_column_sequence(columns: Iterable[str]) -> list[str]:
        """Return a deduplicated list of non-empty column names."""

        seen: set[str] = set()
        normalised: list[str] = []
        for column in columns:
            candidate = column.strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            normalised.append(candidate)
        return normalised

    @staticmethod
    def _deduplicate_feature_columns(columns: Iterable[str]) -> list[str]:
        """Return original column names deduplicated by their stripped value."""

        seen: set[str] = set()
        deduplicated: list[str] = []
        for column in columns:
            candidate = column.strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            deduplicated.append(column)
        return deduplicated

    def configure_ai_feature_columns(
        self,
        columns: Iterable[str] | None,
        *,
        allow_fallback: bool = True,
    ) -> None:
        """Configure the callable providing AI feature columns.

        ``columns`` can originate from runtime services (``signal_service``)
        or configuration objects.  The values are normalised to strings and the
        resulting callable filters them against the actual market data to avoid
        stale configuration errors.  When ``allow_fallback`` is ``True`` and no
        configured column is available we fall back to the default OHLCV-based
        heuristic for backward compatibility with existing guardrails.
        """

        normalised = tuple(
            self._normalize_feature_column_sequence(
                column for column in (columns or ()) if isinstance(column, str)
            )
        )

        if not normalised:
            self._ai_feature_column_names = None
            self._ai_feature_column_source = "default"
            self._ai_feature_column_snapshot = ()
            self._ai_feature_columns = self._default_ai_feature_columns
            return

        self._ai_feature_column_names = normalised
        self._ai_feature_column_source = "configured"
        self._ai_feature_column_snapshot = ()

        def _configured_feature_columns(market_data: pd.DataFrame) -> list[str]:
            column_map: dict[str, str] = {}
            for raw in market_data.columns:
                if not isinstance(raw, str):
                    continue
                key = raw.strip()
                if key and key not in column_map:
                    column_map[key] = raw
            resolved: list[str] = []
            for column in normalised:
                actual = column_map.get(column, column)
                if actual in market_data.columns:
                    resolved.append(actual)
            selected = self._deduplicate_feature_columns(resolved)
            if selected or not allow_fallback:
                self._ai_feature_column_source = "configured"
                self._ai_feature_column_snapshot = tuple(selected)
                return selected
            fallback = self._default_ai_feature_columns(market_data)
            self._ai_feature_column_source = "fallback"
            self._ai_feature_column_snapshot = tuple(fallback)
            return fallback

        self._ai_feature_columns = _configured_feature_columns

    def configure_ai_feature_columns_from_signal_service(self) -> None:
        """Initialise AI feature columns using runtime metadata.

        The resolver honours overrides present on ``_decision_engine_config``
        (to maintain compatibility with existing hooks) and falls back to
        probing ``signal_service`` for hints.  Errors are logged at DEBUG level
        to avoid polluting normal operation – guardrails rely on the
        configuration being resilient in degraded environments.
        """

        columns: Iterable[str] | None = None

        config = getattr(self, "_decision_engine_config", None)
        if config is not None:
            columns = getattr(config, "ai_feature_columns", None)

        if not columns:
            service = getattr(self, "signal_service", None)
            if service is not None:
                resolver = getattr(service, "get_ai_feature_columns", None)
                if callable(resolver):
                    try:
                        columns = resolver()
                    except TypeError:
                        try:
                            columns = resolver(self)
                        except Exception:
                            columns = None
                            self._log(
                                "Signal service feature column resolver failed", level=logging.DEBUG
                            )
                    except Exception:
                        columns = None
                        self._log(
                            "Signal service feature column resolver failed", level=logging.DEBUG
                        )
                if not columns:
                    candidate = getattr(service, "ai_feature_columns", None)
                    if candidate is None:
                        candidate = getattr(service, "feature_columns", None)
                    if candidate is None:
                        candidate = getattr(service, "FEATURE_COLUMNS", None)
                    columns = candidate

        if columns:
            self.configure_ai_feature_columns(columns)

    def _feature_column_metadata(
        self, selected: Iterable[str] | None = None
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        configured = self._ai_feature_column_names
        if configured:
            metadata["configured_feature_columns"] = [str(column) for column in configured]
        if selected is not None:
            columns = self._deduplicate_feature_columns(
                str(column)
                for column in selected
                if isinstance(column, str) and str(column).strip()
            )
        else:
            columns = self._deduplicate_feature_columns(self._ai_feature_column_snapshot)
        if columns:
            metadata["feature_columns"] = columns
        if metadata:
            metadata["feature_columns_source"] = self._ai_feature_column_source
        return metadata

    def _augment_metadata_with_feature_columns(
        self, metadata: Mapping[str, object] | None = None
    ) -> Mapping[str, object] | None:
        """Merge provided metadata with feature column descriptors."""

        combined: dict[str, object] = {}
        feature_metadata = self._feature_column_metadata()
        if feature_metadata:
            combined.update(feature_metadata)
        if metadata:
            if isinstance(metadata, Mapping):
                for key, value in metadata.items():
                    combined[str(key)] = value
            else:  # pragma: no cover - defensive fallback for legacy payloads
                combined["metadata"] = metadata
        return combined or None

    def _normalise_cycle_history_limit(self, limit: int | None) -> int:
        if limit is None:
            return -1
        try:
            value = int(limit)
        except (TypeError, ValueError):
            return -1
        if value <= 0:
            return -1
        return value

    def _normalise_cycle_history_ttl(self, ttl: float | None) -> float | None:
        if ttl is None:
            return None
        try:
            value = float(ttl)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or value <= 0.0:
            return None
        return value

    # ------------------------------------------------------------------
    # Normalisers
    # ------------------------------------------------------------------
    def _normalise_cycle_history_limit(self, limit: int | float | None) -> int:
        """Coerce history limit values into an internal representation.

        ``None`` and non-positive values disable trimming and are represented as
        ``-1``.  Invalid inputs fall back to ``-1`` as well so callers do not
        need to handle conversion errors.
        """

        if limit is None:
            return -1
        try:
            # ``int(True)`` evaluates to ``1`` which is acceptable, so there is no
            # need for a dedicated bool branch here.
            normalized = int(limit)
        except (TypeError, ValueError):
            return -1
        if normalized <= 0:
            return -1
        return normalized

    def _normalise_cycle_history_ttl(self, ttl: float | int | None) -> float | None:
        """Return a positive TTL in seconds or ``None`` when disabled."""

        if ttl is None:
            return None
        try:
            normalized = float(ttl)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(normalized) or normalized <= 0.0:
            return None
        return normalized

    # ------------------------------------------------------------------
    # Normalisers
    # ------------------------------------------------------------------
    def _normalise_cycle_history_limit(self, limit: int | float | None) -> int:
        """Coerce history limit values into an internal representation.

        ``None`` and non-positive values disable trimming and are represented as
        ``-1``.  Invalid inputs fall back to ``-1`` as well so callers do not
        need to handle conversion errors.
        """

        if limit is None:
            return -1
        try:
            # ``int(True)`` evaluates to ``1`` which is acceptable, so there is no
            # need for a dedicated bool branch here.
            normalized = int(limit)
        except (TypeError, ValueError):
            return -1
        if normalized <= 0:
            return -1
        return normalized

    def _normalise_cycle_history_ttl(self, ttl: float | int | None) -> float | None:
        """Return a positive TTL in seconds or ``None`` when disabled."""

        if ttl is None:
            return None
        try:
            normalized = float(ttl)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(normalized) or normalized <= 0.0:
            return None
        return normalized

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
        while not self._auto_trade_stop.is_set() and not self._stop.is_set():
            self._auto_trade_thread_active = True
            try:
                self._auto_trade_loop()
            except Exception as exc:  # pragma: no cover - keep thread resilient
                self._restart_attempts += 1
                delay = min(self._auto_restart_backoff_s, self._auto_restart_backoff_max_s)
                LOGGER.exception("Auto-trade loop crashed; restarting in %.2fs", delay)
                self._record_decision_audit_stage(
                    "auto_trade_crash",
                    symbol=_SCHEDULE_SYMBOL,
                    payload={"attempt": self._restart_attempts, "delay": delay, "error": str(exc)},
                )
                self._auto_restart_backoff_s = min(self._auto_restart_backoff_s * 2.0, self._auto_restart_backoff_max_s)
                self._auto_trade_thread_active = False
                if self._auto_trade_stop.wait(delay) or self._stop.is_set():
                    break
                continue
            else:
                self._restart_attempts = 0
                self._auto_restart_backoff_s = 1.0
            finally:
                self._auto_trade_thread_active = False
            if self._auto_trade_stop.wait(self.auto_trade_interval_s):
                break
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

    def _build_default_risk_engine(self) -> Any | None:
        try:
            return ThresholdRiskEngine()
        except Exception:  # pragma: no cover - środowisko testowe może nie mieć zależności
            LOGGER.debug("AutoTrader could not create default ThresholdRiskEngine", exc_info=True)
            return None

    def _build_default_decision_engine_config(self) -> DecisionEngineConfig:
        thresholds = DecisionOrchestratorThresholds(
            max_cost_bps=200.0,
            min_net_edge_bps=-1000.0,
            max_daily_loss_pct=1.0,
            max_drawdown_pct=1.0,
            max_position_ratio=10.0,
            max_open_positions=50,
            max_latency_ms=2000.0,
        )
        return DecisionEngineConfig(
            orchestrator=thresholds,
            min_probability=0.0,
            require_cost_data=False,
            penalty_cost_bps=0.0,
        )



    def _metric_label_payload(self, **extra: Any) -> Mapping[str, str]:
        payload: Dict[str, str] = {str(key): str(value) for key, value in self._base_metric_labels.items()}
        for key, value in extra.items():
            payload[str(key)] = str(value)
        return payload

    @staticmethod
    def _unique_list(values: Iterable[object]) -> list[str]:
        seen: dict[str, None] = {}
        result: list[str] = []
        for raw in values:
            text = str(raw).strip()
            if not text or text in seen:
                continue
            seen[text] = None
            result.append(text)
        return result

    @staticmethod
    def _normalise_strategy_metadata_value(value: object) -> object:
        if isinstance(value, Mapping):
            return {
                str(key): AutoTrader._normalise_strategy_metadata_value(val)
                for key, val in value.items()
            }
        if isinstance(value, tuple):
            return [str(item) for item in value]
        if isinstance(value, list):
            return [str(item) for item in value]
        return copy.deepcopy(value)

    def _alias_resolver_instance(self) -> StrategyAliasResolver:
        """Zwraca resolver aliasów z uwzględnieniem nadpisanych map i sufiksów."""

        override = self._alias_resolver_override
        if override is not None:
            return override
        return type(self)._alias_resolver()

    def configure_strategy_aliases(
        self,
        alias_map: Mapping[str, str | Sequence[str]] | None = None,
        *,
        suffixes: Iterable[str] | None = None,
    ) -> None:
        """Aktualizuje lokalne mapowanie aliasów i sufiksów strategii."""

        normalized_map = canonical_alias_map(alias_map)
        normalized_suffixes = (
            normalise_suffixes(suffixes) if suffixes is not None else None
        )

        base_resolver = type(self)._alias_resolver()
        new_resolver = base_resolver.extend(
            alias_map=normalized_map,
            suffixes=normalized_suffixes,
        )

        self._alias_resolver_override = (
            None if new_resolver is base_resolver else new_resolver
        )
        self._strategy_alias_map_override = normalized_map or None
        self._strategy_alias_suffix_override = (
            tuple(normalized_suffixes)
            if normalized_suffixes is not None
            else None
        )

    def _strategy_metadata_candidates(self, name: str | None) -> tuple[str, ...]:
        base = str(name or "").strip()
        if not base:
            return ()
        resolver = self._alias_resolver_instance()
        return resolver.candidates(base)

    def _strategy_metadata_summary(
        self, metadata: Mapping[str, object]
    ) -> dict[str, list[str]]:
        summary: dict[str, list[str]] = {}

        def _extract(key: str) -> list[str]:
            value = metadata.get(key)
            if value is None:
                return []
            if isinstance(value, (list, tuple)):
                return self._unique_list(value)
            if isinstance(value, set):
                return self._unique_list(sorted(value))
            text = str(value).strip()
            return [text] if text else []

        mapping = {
            "license_tier": "license_tiers",
            "risk_classes": "risk_classes",
            "required_data": "required_data",
            "capability": "capabilities",
            "tags": "tags",
        }
        for source, target in mapping.items():
            values = _extract(source)
            if values:
                summary[target] = values
        return summary

    def _strategy_metadata_bundle(
        self, name: str | None
    ) -> tuple[dict[str, object], dict[str, list[str]]]:
        metadata: dict[str, object] = {}
        summary: dict[str, list[str]] = {}
        matched_catalog: str | None = None
        for candidate in self._strategy_metadata_candidates(name):
            payload = self._strategy_catalog.metadata_for(candidate)
            if payload:
                matched_catalog = candidate
                metadata = {
                    str(key): self._normalise_strategy_metadata_value(value)
                    for key, value in payload.items()
                    if key != "name"
                }
                break
        strategy_name = str(name or "").strip()
        if metadata:
            metadata["name"] = strategy_name or (matched_catalog or "")
            if matched_catalog and matched_catalog != strategy_name:
                metadata.setdefault("catalog_name", matched_catalog)
            summary = self._strategy_metadata_summary(metadata)
        elif strategy_name:
            metadata["name"] = strategy_name
        return metadata, summary

    def _update_strategy_metrics(self, strategy: str) -> None:
        strategy_label = str(strategy)
        labels = self._metric_label_payload(strategy=strategy_label)
        self._metric_strategy_state_gauge.set(1.0, labels=labels)
        if self._last_strategy_metric and self._last_strategy_metric != strategy_label:
            previous_labels = self._metric_label_payload(strategy=self._last_strategy_metric)
            self._metric_strategy_state_gauge.set(0.0, labels=previous_labels)
            self._metric_strategy_switch_total.inc(labels=self._base_metric_labels)
        if self._last_strategy_metric != strategy_label:
            self._last_strategy_metric = strategy_label

    def _emit_alert(
        self,
        category: str,
        title: str,
        body: str,
        *,
        severity: str = "info",
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        router = getattr(self, "alert_router", None)
        if router is None:
            return False
        context_payload = self._metric_label_payload()
        if context:
            for key, value in context.items():
                context_payload[str(key)] = str(value)
        message = AlertMessage(
            category=str(category),
            title=str(title),
            body=str(body),
            severity=str(severity),
            context=context_payload,
        )
        try:
            router.dispatch(message)
        except Exception:  # pragma: no cover - defensive logging
            self._log(
                "Failed to dispatch alert message",
                level=logging.DEBUG,
                category=category,
                title=title,
            )
            return False
        return True

    def _process_orchestrator_recalibrations(self) -> None:
        orchestrator = self._resolve_decision_orchestrator()
        if orchestrator is None:
            return
        due_recalibrations = getattr(orchestrator, "due_recalibrations", None)
        mark_recalibrated = getattr(orchestrator, "mark_recalibrated", None)
        if not callable(due_recalibrations):
            return
        try:
            schedules = due_recalibrations()
        except Exception:  # pragma: no cover - defensive logging
            self._log(
                "DecisionOrchestrator.due_recalibrations failed",
                level=logging.DEBUG,
            )
            return
        if not schedules:
            return
        for schedule in schedules:
            strategy = str(getattr(schedule, "strategy", "<unknown>"))
            interval = getattr(schedule, "interval", None)
            next_run = getattr(schedule, "next_run", None)
            if callable(self._metric_recalibration_total.inc):
                self._metric_recalibration_total.inc(
                    labels=self._metric_label_payload(strategy=strategy)
                )
            context = {
                "strategy": strategy,
                "next_run": getattr(next_run, "isoformat", lambda: str(next_run))(),
            }
            if isinstance(interval, timedelta):
                context["interval_s"] = f"{interval.total_seconds():.0f}"
            self._emit_alert(
                "auto_trader.recalibration",
                "Wymagana rekalkibracja strategii",
                f"Strategia {strategy} wymaga ponownej kalibracji.",
                severity="warning",
                context=context,
            )
            if callable(mark_recalibrated):
                try:
                    mark_recalibrated(strategy)
                except Exception:  # pragma: no cover - defensive logging
                    self._log(
                        "Failed to acknowledge strategy recalibration",
                        level=logging.DEBUG,
                        strategy=strategy,
                    )

    @staticmethod
    def _normalise_cycle_history_limit(limit: int | None) -> int:
        """Normalise history limits provided by public setters."""

        if limit is None:
            return _CONTROLLER_HISTORY_DEFAULT_LIMIT
        try:
            value = int(limit)
        except (TypeError, ValueError):
            return _CONTROLLER_HISTORY_DEFAULT_LIMIT
        if value <= 0:
            return -1
        return value

    @staticmethod
    def _normalise_cycle_history_ttl(ttl: float | None) -> float | None:
        if ttl is None:
            return None
        try:
            value = float(ttl)
        except (TypeError, ValueError):
            return None
        return max(0.0, value)



    def _ensure_work_schedule(self) -> TradingSchedule:
        schedule = getattr(self, "_work_schedule", None)
        if schedule is None:
            schedule = self._build_default_work_schedule()
            self._work_schedule = schedule
        return schedule

    def _describe_schedule(self, schedule: TradingSchedule) -> ScheduleState:
        """Return a :class:`ScheduleState` snapshot for the provided schedule."""

        describe = getattr(schedule, "describe", None)
        reference = datetime.now(timezone.utc)

        if callable(describe):
            try:
                state = describe(reference)
            except TypeError:
                # ``TradingSchedule.describe`` accepts an optional reference
                # argument.  Older or simplified implementations may expose a
                # no-argument variant, so fall back to calling it without
                # parameters.  If both fail we handle it below.
                state = describe()
            except Exception:  # pragma: no cover - defensive fallback
                LOGGER.exception("Failed to describe trading schedule")
            else:
                if isinstance(state, ScheduleState):
                    return state

        # Fallback used when the schedule does not expose ``describe`` or
        # returns an unexpected payload.  We synthesise a minimal
        # ``ScheduleState`` snapshot so callers can continue operating.
        mode = getattr(schedule, "default_mode", "live")
        allow_trading = getattr(schedule, "allow_trading", True)
        return ScheduleState(
            mode=mode,
            is_open=bool(allow_trading),
            window=None,
            next_transition=None,
            reference_time=reference,
        )



    def describe_work_schedule(self) -> dict[str, Any]:
        schedule = self._ensure_work_schedule()
        state = self._describe_schedule(schedule)
        self._schedule_state = state
        self._schedule_mode = state.mode
        description = schedule.to_payload()
        description["state"] = _serialize_schedule_state(state)
        return description



    def set_schedule_overrides(
        self,
        overrides: Sequence[ScheduleOverride] | Sequence[Mapping[str, object]],
        *,
        reason: str | None = None,
    ) -> ScheduleState:
        schedule = self._ensure_work_schedule()
        schedule = schedule.with_overrides(overrides)
        return self.set_work_schedule(schedule, reason=reason or "overrides_update")

    @staticmethod
    def _overrides_overlap(first: ScheduleOverride, second: ScheduleOverride) -> bool:
        return not (first.end <= second.start or first.start >= second.end)

    def add_schedule_override(
        self,
        override: ScheduleOverride | Mapping[str, object],
        *,
        allow_overlap: bool = False,
        replace_overlaps: bool = False,
        reason: str | None = None,
    ) -> ScheduleState:
        if allow_overlap and replace_overlaps:
            raise ValueError("allow_overlap cannot be combined with replace_overlaps")

        schedule = self._ensure_work_schedule()
        if isinstance(override, Mapping):
            override_obj = ScheduleOverride.from_mapping(
                override, default_timezone=schedule.timezone
            )
        elif isinstance(override, ScheduleOverride):
            override_obj = override
        else:
            raise TypeError(
                "Override must be ScheduleOverride instance or mapping payload"
            )

        overrides = list(schedule.overrides)
        overlapping_indexes: list[int] = []
        for idx, existing in enumerate(overrides):
            if self._overrides_overlap(existing, override_obj):
                overlapping_indexes.append(idx)

        if overlapping_indexes:
            if replace_overlaps:
                for idx in reversed(overlapping_indexes):
                    overrides.pop(idx)
            elif not allow_overlap:
                raise ValueError("Schedule override overlaps existing override")

        overrides.append(override_obj)
        overrides.sort(key=lambda item: item.start)
        return self.set_schedule_overrides(
            overrides,
            reason=reason or ("override_replaced" if overlapping_indexes else "override_added"),
        )

    def remove_schedule_override(
        self,
        *,
        label: str | None = None,
        start: datetime | None = None,
        reason: str | None = None,
    ) -> ScheduleState:
        if label is None and start is None:
            raise ValueError("label or start must be provided to remove override")

        schedule = self._ensure_work_schedule()
        timezone_obj = schedule.timezone
        normalized_start: datetime | None = None
        if start is not None:
            normalized_start = start
            if normalized_start.tzinfo is None:
                normalized_start = normalized_start.replace(tzinfo=timezone_obj)
            else:
                normalized_start = normalized_start.astimezone(timezone_obj)

        remaining: list[ScheduleOverride] = []
        removed = False
        for override in schedule.overrides:
            matches = True
            if label is not None and override.label != label:
                matches = False
            if matches and normalized_start is not None and override.start != normalized_start:
                matches = False
            if matches:
                removed = True
                continue
            remaining.append(override)

        if not removed:
            raise LookupError("No schedule override matched the provided criteria")

        return self.set_schedule_overrides(
            remaining,
            reason=reason or "override_removed",
        )







    def _handle_schedule_transition(self, state: ScheduleState, *, reason: str) -> None:
        gauge_value = 1.0 if state.is_open else 0.0
        self._metric_schedule_open_gauge.set(gauge_value, labels=self._base_metric_labels)
        if not state.is_open:
            delay = state.time_until_transition or self.auto_trade_interval_s
            if reason in {"transition", "blocked"}:
                self._metric_schedule_closed_seconds.observe(
                    float(delay),
                    labels=self._base_metric_labels,
                )
            if reason in {"transition", "blocked"} and self._schedule_last_alert_state is not False:
                context = {"mode": state.mode}
                if state.next_transition is not None:
                    context["next_transition"] = state.next_transition.astimezone(timezone.utc).isoformat()
                self._emit_alert(
                    "auto_trader.schedule",
                    "Harmonogram handlu zamknięty",
                    "AutoTrader oczekuje na ponowne otwarcie okna handlu.",
                    severity="warning",
                    context=context,
                )
            self._schedule_last_alert_state = False
        else:
            if reason == "transition" and self._schedule_last_alert_state is False:
                context = {"mode": state.mode}
                self._emit_alert(
                    "auto_trader.schedule",
                    "Harmonogram handlu wznowiony",
                    "Okno handlu zostało ponownie otwarte.",
                    severity="info",
                    context=context,
                )
            self._schedule_last_alert_state = True











    def _build_decision_orchestrator(self) -> DecisionOrchestrator:
        try:
            return DecisionOrchestrator(self._decision_engine_config)
        except Exception as exc:  # pragma: no cover - diagnostic aid
            raise RuntimeError("AutoTrader requires a functional DecisionOrchestrator") from exc

    def _attach_decision_orchestrator(self) -> None:
        orchestrator = getattr(self, "decision_orchestrator", None)
        if orchestrator is None:
            raise RuntimeError("DecisionOrchestrator could not be initialised")
        engine = getattr(self.core_risk_engine, "attach_decision_orchestrator", None)
        if callable(engine):
            try:
                engine(orchestrator)
            except Exception:  # pragma: no cover - defensive logging
                LOGGER.exception(
                    "AutoTrader could not attach DecisionOrchestrator to risk engine",
                )

    def _detect_initial_mode(self) -> str:
        if hasattr(self.gui, "is_demo_mode_active"):
            try:
                return "demo" if self.gui.is_demo_mode_active() else "live"
            except Exception:  # pragma: no cover - GUI may raise
                LOGGER.debug("GUI demo mode detection failed", exc_info=True)
        return "demo"

    def _build_default_work_schedule(self) -> TradingSchedule:
        tz_name = None
        context = getattr(self, "bootstrap_context", None)
        if context is not None:
            tz_name = getattr(context, "timezone", None)
        if isinstance(tz_name, str) and tz_name.strip():
            return TradingSchedule.always_on(mode=self._initial_mode, timezone_name=str(tz_name))
        return TradingSchedule.always_on(mode=self._initial_mode)

    @staticmethod
    def _coerce_schedule_overrides(
        overrides: ScheduleOverride
        | Mapping[str, object]
        | Sequence[object]
    ) -> list[ScheduleOverride]:
        def _convert(item: object) -> list[ScheduleOverride]:
            if isinstance(item, ScheduleOverride):
                return [item]
            if isinstance(item, Mapping):
                return [ScheduleOverride.from_mapping(item)]
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                collected: list[ScheduleOverride] = []
                for entry in item:
                    collected.extend(_convert(entry))
                return collected
            raise TypeError("override definitions must be ScheduleOverride or mappings")

        converted = _convert(overrides)
        if not converted:
            raise ValueError("at least one schedule override must be provided")
        return converted

    @staticmethod
    def _coerce_override_labels(labels: str | Sequence[object] | None) -> set[str] | None:
        if labels is None:
            return None
        if isinstance(labels, str):
            label = labels.strip()
            return {label} if label else set()
        if isinstance(labels, Sequence) and not isinstance(labels, (bytes, bytearray)):
            result: set[str] = set()
            for item in labels:
                if item is None:
                    continue
                token = str(item).strip()
                if token:
                    result.add(token)
            return result
        raise TypeError("labels must be a string or a sequence of strings")

    def get_work_schedule(self) -> TradingSchedule:
        """Return the current trading schedule, initialising defaults if missing."""

        with self._lock:
            schedule = getattr(self, "_work_schedule", None)
            if schedule is None:
                schedule = self._build_default_work_schedule()
                self._work_schedule = schedule
        return schedule

    def set_work_schedule(
        self,
        schedule: TradingSchedule | Mapping[str, object] | Sequence[object] | None,
        *,
        reason: str | None = None,
    ) -> ScheduleState:
        """Replace the active work schedule and publish its state."""

        with self._decision_audit_scope() as decision_id:
            if schedule is not None and not isinstance(schedule, TradingSchedule):
                if isinstance(schedule, Mapping) or (
                    isinstance(schedule, Sequence)
                    and not isinstance(schedule, (str, bytes, bytearray))
                ):
                    schedule = TradingSchedule.from_payload(schedule)
                else:
                    raise TypeError(
                        "schedule must be a TradingSchedule or serialisable payload"
                    )
            if schedule is None:
                schedule = self._build_default_work_schedule()
                with self._lock:
                    self._work_schedule = schedule
                update_reason = reason or "reset"
            else:
                update_reason = reason or "update"

            state = self._describe_schedule(schedule)
            with self._lock:
                self._work_schedule = schedule
                self._schedule_state = state
                self._schedule_mode = state.mode
                self._last_schedule_snapshot = (state.mode, state.is_open)

            status = "open" if state.is_open else "closed"
            self._log(
                f"Trading schedule updated to mode={state.mode} ({status})",
                level=logging.INFO,
                reason=update_reason,
            )

            payload = state.to_mapping()
            payload["reason"] = update_reason
            self._record_decision_audit_stage(
                "schedule_configured",
                symbol=_SCHEDULE_SYMBOL,
                payload=payload,
                decision_id=decision_id,
            )
            self._log_decision_event(
                "schedule_configured",
                status=status,
                metadata=payload,
            )
            self._emit_schedule_state_event(state, reason=update_reason)
            return state





    def schedule_strategy_recalibration(
        self,
        strategy: str,
        *,
        interval_s: float,
        first_run: datetime | None = None,
    ) -> None:
        orchestrator = self._resolve_decision_orchestrator()
        scheduler = getattr(orchestrator, "schedule_strategy_recalibration", None) if orchestrator else None
        if not callable(scheduler):
            raise RuntimeError("DecisionOrchestrator does not support strategy scheduling")
        interval = timedelta(seconds=float(max(interval_s, 0.0)))
        schedule = scheduler(strategy, interval, first_run=first_run)
        next_run = getattr(schedule, "next_run", None)
        self._log(
            "Strategy recalibration scheduled",
            level=logging.INFO,
            strategy=strategy,
            interval_s=interval.total_seconds(),
            next_run=getattr(next_run, "isoformat", lambda: str(next_run))(),
        )



























    def apply_schedule_override(
        self,
        strategy: str,
        *,
        interval_s: float,
        first_run: datetime | None = None,
    ) -> None:
        orchestrator = self._resolve_decision_orchestrator()
        scheduler = getattr(orchestrator, "schedule_strategy_recalibration", None) if orchestrator else None
        if not callable(scheduler):
            raise RuntimeError("DecisionOrchestrator does not support strategy scheduling")
        interval = timedelta(seconds=float(max(interval_s, 0.0)))
        schedule = scheduler(strategy, interval, first_run=first_run)
        next_run = getattr(schedule, "next_run", None)
        self._log(
            "Strategy recalibration scheduled",
            level=logging.INFO,
            strategy=strategy,
            interval_s=interval.total_seconds(),
            next_run=getattr(next_run, "isoformat", lambda: str(next_run))(),
        )































    def schedule_strategy_recalibration(
        self,
        strategy: str,
        *,
        interval_s: float,
        first_run: datetime | None = None,
    ) -> None:
        orchestrator = self._resolve_decision_orchestrator()
        scheduler = getattr(orchestrator, "schedule_strategy_recalibration", None) if orchestrator else None
        if not callable(scheduler):
            raise RuntimeError("DecisionOrchestrator does not support strategy scheduling")
        interval = timedelta(seconds=float(max(interval_s, 0.0)))
        schedule = scheduler(strategy, interval, first_run=first_run)
        next_run = getattr(schedule, "next_run", None)
        self._log(
            "Strategy recalibration scheduled",
            level=logging.INFO,
            strategy=strategy,
            interval_s=interval.total_seconds(),
            next_run=getattr(next_run, "isoformat", lambda: str(next_run))(),
        )



























    def apply_schedule_override(
        self,
        overrides: ScheduleOverride | Mapping[str, object] | Sequence[object],
        *,
        reason: str | None = None,
        replace: bool = False,
    ) -> ScheduleState:
        with self._decision_audit_scope() as decision_id:
            schedule = getattr(self, "_work_schedule", None)
            if schedule is None:
                schedule = self._build_default_work_schedule()
            overrides_list = self._coerce_schedule_overrides(overrides)
            existing = schedule.overrides
            combined: tuple[ScheduleOverride, ...]
            if replace:
                combined = tuple(overrides_list)
            else:
                combined = existing + tuple(overrides_list)
            updated_schedule = schedule.with_overrides(combined)
            update_reason = reason or ("override_replace" if replace else "override")
            state = self.set_work_schedule(updated_schedule, reason=update_reason)
            payload = state.to_mapping()
            payload["reason"] = update_reason
            payload["overrides_applied"] = [
                item.to_mapping(include_duration=True, timezone_hint=timezone.utc)
                for item in overrides_list
            ]
            payload["override_replace"] = bool(replace)
            self._record_decision_audit_stage(
                "schedule_override_applied",
                symbol=_SCHEDULE_SYMBOL,
                payload=payload,
                decision_id=decision_id,
            )
            self._log_decision_event(
                "schedule_override_applied",
                status="open" if state.is_open else "closed",
                metadata=payload,
            )
            return state

    def list_schedule_overrides(self) -> tuple[ScheduleOverride, ...]:
        """Return a snapshot of overrides currently applied to the schedule."""

        schedule = self.get_work_schedule()
        return schedule.overrides

    def clear_schedule_overrides(
        self,
        *,
        labels: str | Sequence[object] | None = None,
        reason: str | None = None,
    ) -> ScheduleState:
        with self._decision_audit_scope() as decision_id:
            schedule = getattr(self, "_work_schedule", None)
            if schedule is None:
                schedule = self._build_default_work_schedule()
            existing = schedule.overrides
            if not existing:
                return self.get_schedule_state()

            label_filter = self._coerce_override_labels(labels)
            if label_filter:
                filtered = tuple(
                    override
                    for override in existing
                    if override.label is None or override.label not in label_filter
                )
            else:
                filtered = ()

            if filtered == existing:
                return self.get_schedule_state()

            updated_schedule = schedule.with_overrides(filtered)
            update_reason = reason or "override_clear"
            state = self.set_work_schedule(updated_schedule, reason=update_reason)
            payload = state.to_mapping()
            payload["reason"] = update_reason
            payload["remaining_overrides"] = [
                item.to_mapping(include_duration=True, timezone_hint=timezone.utc)
                for item in filtered
            ]
            if label_filter is not None:
                payload["cleared_labels"] = sorted(label_filter)
            self._record_decision_audit_stage(
                "schedule_override_cleared",
                symbol=_SCHEDULE_SYMBOL,
                payload=payload,
                decision_id=decision_id,
            )
            return state

    def get_schedule_state(self) -> ScheduleState:
        """Return the latest schedule state, recalculating it if needed."""

        schedule = self.get_work_schedule()
        state = self._describe_schedule(schedule)
        with self._lock:
            self._schedule_state = state
            self._schedule_mode = state.mode
        return state

    def is_schedule_open(self) -> bool:
        """Return ``True`` when the work schedule allows trading."""

        return self.get_schedule_state().is_open

    @staticmethod
    def _detect_environment_name(bootstrap_context: Any | None) -> str:
        if bootstrap_context is None:
            return "paper"
        candidate = getattr(bootstrap_context, "environment", None)
        if isinstance(candidate, str):
            return candidate
        value = getattr(candidate, "value", None)
        if isinstance(value, str):
            return value
        name = getattr(candidate, "name", None)
        if isinstance(name, str):
            return name
        alt = getattr(bootstrap_context, "environment_name", None)
        if isinstance(alt, str):
            return alt
        return "paper"

    def _resolve_execution_service(self, symbol: str) -> Any:
        service = self.execution_service or self.core_execution_service
        if service is not None:
            return service
        if (
            self._default_execution_service is None
            or (self._default_execution_symbol is not None and self._default_execution_symbol != symbol)
        ):
            try:
                self._default_execution_service = self._build_default_execution_service(symbol)
            except Exception as exc:  # pragma: no cover - defensive logging
                self._default_execution_service = None
                LOGGER.error(
                    "Failed to initialise default execution service for %s: %s",
                    symbol,
                    exc,
                )
            else:
                self._default_execution_symbol = symbol
        return self._default_execution_service

    def _build_default_execution_service(self, symbol: str) -> ExecutionService:
        base_asset, quote_asset = self._split_symbol(symbol)
        metadata = MarketMetadata(base_asset=base_asset, quote_asset=quote_asset)
        balances = {quote_asset: 100_000.0}
        return PaperTradingExecutionService({symbol: metadata}, initial_balances=balances)

    @staticmethod
    def _split_symbol(symbol: str) -> tuple[str, str]:
        normalized = symbol.strip().upper()
        common_quotes = ("USDT", "USDC", "USD", "EUR", "BTC", "ETH", "BNB", "BUSD")
        for quote in common_quotes:
            if normalized.endswith(quote) and len(normalized) > len(quote):
                return normalized[: -len(quote)], quote
        if len(normalized) > 3:
            return normalized[:-3], normalized[-3:]
        return normalized or "ASSET", "USDT"

    def _resolve_execution_context(self) -> ExecutionContext:
        if self._execution_context is None:
            metadata = dict(self._execution_metadata)
            self._execution_context = ExecutionContext(
                portfolio_id=self._portfolio_id,
                risk_profile=self._risk_profile_name,
                environment=self._environment_name,
                metadata=metadata,
            )
        return self._execution_context

    def _enforce_work_schedule(self) -> bool:
        schedule = getattr(self, "_work_schedule", None)
        if schedule is None:
            return True
        state = self._describe_schedule(schedule)
        self._schedule_state = state
        self._schedule_mode = state.mode
        snapshot = (state.mode, state.is_open)
        if snapshot != self._last_schedule_snapshot:
            status = "open" if state.is_open else "closed"
            self._log(
                f"Trading schedule switched to mode={state.mode} ({status})",
                level=logging.INFO,
            )
            payload = state.to_mapping()
            payload["reason"] = "transition"
            self._record_decision_audit_stage(
                "schedule_transition",
                symbol=_SCHEDULE_SYMBOL,
                payload=payload,
            )
            self._log_decision_event(
                "schedule_transition",
                status="open" if state.is_open else "closed",
                metadata=payload,
            )
            self._emit_schedule_state_event(state, reason="transition")
            self._last_schedule_snapshot = snapshot
        if not state.is_open:
            delay = state.time_until_transition or self.auto_trade_interval_s
            payload = state.to_mapping()
            payload["reason"] = "blocked"
            self._record_decision_audit_stage(
                "schedule_blocked",
                symbol=_SCHEDULE_SYMBOL,
                payload=payload,
            )
            self._log_decision_event(
                "schedule_blocked",
                status="closed",
                metadata=payload,
            )
            self._auto_trade_stop.wait(delay)
            return False
        return True

    @staticmethod
    def _generate_decision_id() -> str:
        return uuid.uuid4().hex

    @staticmethod
    def _normalize_decision_id(value: Any | None) -> str | None:
        if value is None:
            return None
        token = str(value).strip()
        return token or None

    @contextmanager
    def _decision_audit_scope(self, *, decision_id: str | None = None):
        existing = self._active_decision_id
        if existing is not None:
            yield existing
            return
        normalized = self._normalize_decision_id(decision_id) or self._generate_decision_id()
        self._active_decision_id = normalized
        try:
            yield normalized
        finally:
            if self._active_decision_id == normalized:
                self._active_decision_id = None

    def _record_decision_audit_stage(
        self,
        stage: str,
        *,
        symbol: str,
        payload: Mapping[str, object] | None = None,
        risk_snapshot: Mapping[str, object] | None = None,
        portfolio_snapshot: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
        decision_id: str | None = None,
    ) -> None:
        log = getattr(self, "_decision_audit_log", None)
        if log is None:
            return
        try:
            normalized_decision_id = (
                self._normalize_decision_id(decision_id)
                or self._active_decision_id
                or self._generate_decision_id()
            )
            payload_dict = dict(payload or {})
            if normalized_decision_id is not None:
                payload_dict.setdefault("decision_id", normalized_decision_id)
            augmented_metadata = self._augment_metadata_with_feature_columns(metadata)
            record = log.record(
                stage,
                symbol,
                mode=self._schedule_mode,
                payload=payload_dict,
                risk_snapshot=risk_snapshot,
                portfolio_snapshot=portfolio_snapshot,
                metadata=augmented_metadata,
            )
            self._emit_decision_audit_event(record)
        except Exception:  # pragma: no cover - audit log failures should not break trading
            LOGGER.debug("Decision audit logging failed", exc_info=True)

    def _log_decision_event(
        self,
        event: str,
        *,
        symbol: str | None = None,
        status: str | None = None,
        side: str | None = None,
        quantity: float | None = None,
        price: float | None = None,
        metadata: Mapping[str, object] | None = None,
        confidence: float | None = None,
        latency_ms: float | None = None,
    ) -> None:
        journal = getattr(self, "_decision_journal", None)
        if journal is None:
            return
        try:
            context = getattr(self, "_decision_journal_context", None)
            environment = self._environment_name
            portfolio = self._portfolio_id
            risk_profile = self._risk_profile_name
            merged_meta: dict[str, object] = {}
            if isinstance(context, Mapping):
                env_value = context.get("environment")
                if env_value is not None:
                    environment = str(env_value)
                portfolio_value = context.get("portfolio")
                if portfolio_value is not None:
                    portfolio = str(portfolio_value)
                risk_value = context.get("risk_profile")
                if risk_value is not None:
                    risk_profile = str(risk_value)
                for key, value in context.items():
                    if key in {"environment", "portfolio", "risk_profile"}:
                        continue
                    merged_meta[str(key)] = value
            for key, value in self._execution_metadata.items():
                merged_meta.setdefault(str(key), value)
            if metadata:
                for key, value in metadata.items():
                    merged_meta[str(key)] = value
            log_decision_event(
                journal,
                event=event,
                environment=str(environment),
                portfolio=str(portfolio),
                risk_profile=str(risk_profile),
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                status=status,
                schedule=self._schedule_mode,
                strategy=self.current_strategy if self.current_strategy else None,
                metadata=merged_meta,
                confidence=confidence,
                latency_ms=latency_ms,
            )
        except Exception:  # pragma: no cover - journaling issues must not break trading
            LOGGER.debug("Decision journal logging failed", exc_info=True)

    def _emit_decision_audit_event(self, record: DecisionAuditRecord) -> None:
        emitter_emit = getattr(self.emitter, "emit", None)
        if not callable(emitter_emit):
            return
        payload = record.to_mapping()
        try:
            emitter_emit("auto_trader.decision_audit", **payload)
        except Exception:  # pragma: no cover - emission should not break trading
            LOGGER.debug("Decision audit emission failed", exc_info=True)

    def _emit_schedule_state_event(self, state: ScheduleState, *, reason: str | None = None) -> None:
        emitter_emit = getattr(self.emitter, "emit", None)
        if not callable(emitter_emit):
            return
        payload = state.to_mapping()
        if reason is not None:
            payload["reason"] = reason
        try:
            emitter_emit("auto_trader.schedule_state", **payload)
        except Exception:  # pragma: no cover - emission should not break trading
            LOGGER.debug("Schedule state emission failed", exc_info=True)

    @staticmethod
    def _safe_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_decision_confidence(details: Mapping[str, Any] | None) -> float | None:
        if not isinstance(details, Mapping):
            return None
        candidates: list[Mapping[str, Any]] = []
        ai_block = details.get("ai")
        if isinstance(ai_block, Mapping):
            candidates.append(ai_block)
        engine_block = details.get("decision_engine")
        if isinstance(engine_block, Mapping):
            nested_ai = engine_block.get("ai")
            if isinstance(nested_ai, Mapping):
                candidates.append(nested_ai)
        for block in candidates:
            for key in ("probability", "success_probability", "confidence"):
                candidate = AutoTrader._safe_float(block.get(key))
                if candidate is not None:
                    return candidate
        return None

    def _capture_risk_snapshot(self) -> Mapping[str, object] | None:
        service = self.risk_service or getattr(self, "core_risk_engine", None)
        if service is None:
            return None
        snapshot_fn = getattr(service, "snapshot_state", None)
        if not callable(snapshot_fn):
            return None
        try:
            return snapshot_fn(self._risk_profile_name)
        except TypeError:
            try:
                return snapshot_fn(profile_name=self._risk_profile_name)
            except TypeError:
                try:
                    return snapshot_fn(profile=self._risk_profile_name)
                except Exception:
                    LOGGER.debug("Risk snapshot capture failed", exc_info=True)
        except Exception:
            LOGGER.debug("Risk snapshot capture failed", exc_info=True)
        return None

    def _capture_portfolio_snapshot(self) -> Mapping[str, object] | None:
        manager = getattr(self, "portfolio_manager", None)
        if manager is None:
            return None
        candidates = (
            "snapshot",
            "snapshot_state",
            "get_snapshot",
            "get_state",
            "portfolio_state",
            "summary",
            "get_summary",
            "to_dict",
        )
        for attr in candidates:
            getter = getattr(manager, attr, None)
            if not callable(getter):
                continue
            try:
                result = getter()
            except TypeError:
                try:
                    result = getter(self._risk_profile_name)
                except Exception:
                    continue
            except Exception:
                continue
            if result is None:
                continue
            if isinstance(result, Mapping):
                return dict(result)
            if hasattr(result, "_asdict"):
                try:
                    return dict(result._asdict())  # type: ignore[call-arg]
                except Exception:
                    continue
            if hasattr(result, "__dict__"):
                return dict(vars(result))
            try:
                return dict(result)  # type: ignore[arg-type]
            except Exception:
                continue
        return None



















    def _build_order_request(self, symbol: str, decision: RiskDecision) -> OrderRequest:
        signal = str(decision.details.get("signal", "hold")).lower()
        side = "buy" if signal not in {"buy", "sell"} else signal
        quantity = float(decision.fraction or 0.0)
        if quantity <= 0:
            quantity = 1.0 if decision.should_trade else 0.0
        metadata: dict[str, str] = {}
        for key, value in decision.details.items():
            if isinstance(value, (str, int, float, bool)):
                metadata[str(key)] = str(value)
        metadata["mode"] = decision.mode
        return OrderRequest(
            symbol=symbol,
            side=side,
            quantity=abs(quantity),
            order_type="market",
            metadata=metadata,
        )
        return candidate


    def _dispatch_execution(self, service: Any, decision: RiskDecision, symbol: str) -> None:
        try:
            if isinstance(service, ExecutionService):
                request = self._build_order_request(symbol, decision)
                if request.quantity <= 0:
                    self._record_decision_audit_stage(
                        "execution_skipped",
                        symbol=symbol,
                        payload={"reason": "zero_quantity"},
                        portfolio_snapshot=self._capture_portfolio_snapshot(),
                    )
                    return
                context = self._resolve_execution_context()
                service.execute(request, context)
                payload = {
                    "order": {
                        "symbol": request.symbol,
                        "side": request.side,
                        "quantity": request.quantity,
                        "order_type": request.order_type,
                    }
                }
                self._record_decision_audit_stage(
                    "execution_submitted",
                    symbol=symbol,
                    payload=payload,
                    portfolio_snapshot=self._capture_portfolio_snapshot(),
                )
                return

            execute_fn = getattr(service, "execute_decision", None)
            if callable(execute_fn):
                execute_fn(decision)
                self._record_decision_audit_stage(
                    "execution_submitted",
                    symbol=symbol,
                    payload={"adapter": "execute_decision", "decision": decision.to_dict()},
                    portfolio_snapshot=self._capture_portfolio_snapshot(),
                )
                return

            execute_fn = getattr(service, "execute", None)
            if callable(execute_fn):
                payload: Mapping[str, object]
                calls_attr = getattr(service, "calls", None)
                methods_attr = getattr(service, "methods", None)
                previous_calls = len(calls_attr) if isinstance(calls_attr, list) else None
                previous_methods = len(methods_attr) if isinstance(methods_attr, list) else None
                try:
                    execute_fn(decision)
                    payload = {"adapter": "execute", "decision": decision.to_dict()}
                except TypeError:
                    request = self._build_order_request(symbol, decision)
                    if request.quantity <= 0:
                        self._record_decision_audit_stage(
                            "execution_skipped",
                            symbol=symbol,
                            payload={"reason": "zero_quantity"},
                            portfolio_snapshot=self._capture_portfolio_snapshot(),
                        )
                        return
                    context = self._resolve_execution_context()
                    execute_fn(request, context)  # type: ignore[arg-type]
                    payload = {
                        "adapter": "execute",
                        "order": {
                            "symbol": request.symbol,
                            "side": request.side,
                            "quantity": request.quantity,
                            "order_type": request.order_type,
                        },
                    }
                else:
                    self._trim_execution_records(calls_attr, previous_calls)
                    self._trim_execution_records(methods_attr, previous_methods)
                self._record_decision_audit_stage(
                    "execution_submitted",
                    symbol=symbol,
                    payload=payload,
                    portfolio_snapshot=self._capture_portfolio_snapshot(),
                )
                return

            if callable(service):
                service(decision)
                self._record_decision_audit_stage(
                    "execution_submitted",
                    symbol=symbol,
                    payload={"adapter": "callable", "decision": decision.to_dict()},
                    portfolio_snapshot=self._capture_portfolio_snapshot(),
                )
                return

            raise TypeError("Configured execution service is not callable")
        except Exception:
            self._record_decision_audit_stage(
                "execution_failed",
                symbol=symbol,
                payload={"error": "execution_exception"},
                portfolio_snapshot=self._capture_portfolio_snapshot(),
            )
            raise

    @staticmethod
    def _trim_execution_records(container: Any, previous_len: int | None) -> None:
        if not isinstance(container, list) or previous_len is None:
            return
        if len(container) <= previous_len + 1:
            return
        del container[previous_len + 1 :]

    @staticmethod
    def _normalize_decision_fields(
        decision_fields: Iterable[Any] | Any | None,
    ) -> list[Any] | None:
        if decision_fields is None:
            return None
        if isinstance(decision_fields, Iterable) and not isinstance(
            decision_fields,
            (str, bytes, bytearray),
        ):
            candidates = decision_fields
        else:
            candidates = [decision_fields]

        normalized: list[Any] = []
        for candidate in candidates:
            if candidate is None:
                continue
            if any(existing == candidate for existing in normalized):
                continue
            normalized.append(candidate)
        if not normalized:
            return []
        return normalized

    @staticmethod
    def _iter_filter_values(value: Any) -> Iterable[Any]:
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
            return value
        return (value,)

    @staticmethod
    def _coerce_optional_bool(value: Any) -> bool | None | object:
        if value is _NO_FILTER:
            return _NO_FILTER
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return None
            if math.isnan(numeric) or math.isinf(numeric):
                return None
            return bool(numeric)
        if isinstance(value, str):
            token = value.strip().lower()
            if not token or token in {"none", "null"}:
                return None
            if token in {"1", "true", "t", "yes", "y", "on"}:
                return True
            if token in {"0", "false", "f", "no", "n", "off"}:
                return False
        return None

    def _prepare_bool_filter(
        self, value: bool | None | Iterable[bool | None] | object
    ) -> set[bool | None] | None:
        if value is _NO_FILTER:
            return None
        normalized: set[bool | None] = set()
        for candidate in self._iter_filter_values(value):
            token = self._coerce_optional_bool(candidate)
            if token is _NO_FILTER:
                continue
            if token is None:
                normalized.add(None)
            elif isinstance(token, bool):
                normalized.add(token)
        return normalized or None

    def _prepare_service_filter(
        self, value: str | None | Iterable[str | None] | object
    ) -> set[str] | None:
        if value is _NO_FILTER:
            return None
        normalized: set[str] = set()
        for candidate in self._iter_filter_values(value):
            if candidate is _NO_FILTER:
                continue
            if candidate is None:
                normalized.add(_UNKNOWN_SERVICE)
                continue
            token = str(candidate).strip()
            if token:
                normalized.add(token)
        return normalized or None

    def _prepare_string_filter(
        self, value: str | Iterable[str] | object
    ) -> set[str] | None:
        if value is _NO_FILTER:
            return None
        normalized: set[str] = set()
        for candidate in self._iter_filter_values(value):
            if candidate is _NO_FILTER or candidate is None:
                continue
            token = str(candidate).strip()
            if token:
                normalized.add(token)
        return normalized or None

    def _prepare_guardrail_filter(
        self,
        value: str | Iterable[str | None] | object,
        *,
        missing_token: str,
    ) -> set[str] | None:
        if value is _NO_FILTER:
            return None
        normalized: set[str] = set()
        for candidate in self._iter_filter_values(value):
            if candidate is _NO_FILTER:
                continue
            if candidate is None:
                normalized.add(missing_token)
                continue
            token = str(candidate).strip()
            normalized.add(token or missing_token)
        return normalized or None

    def _prepare_guardrail_numeric_filter(
        self,
        value: float | None | Iterable[float | None] | object,
    ) -> tuple[set[float], bool] | None:
        if value is _NO_FILTER:
            return None
        normalized: set[float] = set()
        include_null = False
        for candidate in self._iter_filter_values(value):
            if candidate is _NO_FILTER:
                continue
            if candidate is None:
                include_null = True
                continue
            number = self._coerce_float(candidate)
            if number is None:
                continue
            normalized.add(number)
        if not normalized and not include_null:
            return None
        return normalized, include_null

    def _resolve_risk_evaluation_filters(
        self,
        *,
        approved: bool | None | Iterable[bool | None] | object,
        normalized: bool | None | Iterable[bool | None] | object,
        service: str | None | Iterable[str | None] | object,
        decision_state: str | Iterable[str | None] | object,
        decision_reason: str | Iterable[str | None] | object,
        decision_mode: str | Iterable[str | None] | object,
        decision_id: str | Iterable[str | None] | object,
        since: Any,
        until: Any,
        decision_fields: Iterable[Any] | Any | None,
    ) -> tuple[
        set[bool | None] | None,
        set[bool | None] | None,
        set[str] | None,
        set[str] | None,
        set[str] | None,
        set[str] | None,
        set[str] | None,
        list[Any] | None,
        float | None,
        float | None,
    ]:
        approved_filter = self._prepare_bool_filter(approved)
        normalized_filter = self._prepare_bool_filter(normalized)
        service_filter = self._prepare_service_filter(service)
        decision_state_filter = self._prepare_decision_filter(
            decision_state,
            missing_token=_MISSING_DECISION_STATE,
        )
        decision_reason_filter = self._prepare_decision_filter(
            decision_reason,
            missing_token=_MISSING_DECISION_REASON,
        )
        decision_mode_filter = self._prepare_decision_filter(
            decision_mode,
            missing_token=_MISSING_DECISION_MODE,
        )
        decision_id_filter = self._prepare_decision_filter(
            decision_id,
            missing_token=_MISSING_DECISION_ID,
        )
        since_ts = self._normalize_time_bound(since)
        until_ts = self._normalize_time_bound(until)
        normalized_decision_fields = self._normalize_decision_fields(decision_fields)
        return (
            approved_filter,
            normalized_filter,
            service_filter,
            decision_state_filter,
            decision_reason_filter,
            decision_mode_filter,
            decision_id_filter,
            normalized_decision_fields,
            since_ts,
            until_ts,
        )

    def _collect_filtered_risk_evaluations(
        self,
        *,
        include_errors: bool,
        approved_filter: set[bool | None] | None,
        normalized_filter: set[bool | None] | None,
        service_filter: set[str] | None,
        since_ts: float | None,
        until_ts: float | None,
        state_filter: set[str] | None,
        reason_filter: set[str] | None,
        mode_filter: set[str] | None,
        decision_id_filter: set[str] | None,
    ) -> tuple[list[dict[str, Any]], int, float | None, int]:
        with self._lock:
            trimmed_by_ttl = self._prune_risk_evaluations_locked()
            records = list(self._risk_evaluations)
            ttl_snapshot = self._risk_evaluations_ttl_s
            history_size = len(records)

        filtered: list[dict[str, Any]] = []
        for entry in records:
            if approved_filter is not None and entry.get("approved") not in approved_filter:
                continue

            normalized_value = entry.get("normalized")
            if normalized_value is True:
                normalized_token: bool | None = True
            elif normalized_value is False:
                normalized_token = False
            elif normalized_value is None:
                normalized_token = None
            else:
                normalized_token = None
            if normalized_filter is not None and normalized_token not in normalized_filter:
                continue

            service_raw = entry.get("service")
            service_key = service_raw if service_raw is not None else _UNKNOWN_SERVICE
            if service_filter is not None and service_key not in service_filter:
                continue

            timestamp_raw = entry.get("timestamp")
            timestamp_value = self._normalize_time_bound(timestamp_raw)
            if timestamp_value is None:
                try:
                    timestamp_value = float(timestamp_raw)
                except (TypeError, ValueError):
                    timestamp_value = None
            if since_ts is not None and (timestamp_value is None or timestamp_value < since_ts):
                continue
            if until_ts is not None and (timestamp_value is None or timestamp_value > until_ts):
                continue

            decision_payload = entry.get("decision") or {}
            state_value = decision_payload.get("state")
            state_key = str(state_value) if state_value is not None else _MISSING_DECISION_STATE
            if state_filter is not None and state_key not in state_filter:
                continue

            reason_value = decision_payload.get("reason")
            reason_key = str(reason_value) if reason_value is not None else _MISSING_DECISION_REASON
            if reason_filter is not None and reason_key not in reason_filter:
                continue

            mode_value = decision_payload.get("mode")
            mode_key = str(mode_value) if mode_value is not None else _MISSING_DECISION_MODE
            if mode_filter is not None and mode_key not in mode_filter:
                continue

            decision_id_token = self._normalize_decision_id(entry.get("decision_id"))
            decision_id_key = decision_id_token or _MISSING_DECISION_ID
            if decision_id_filter is not None and decision_id_key not in decision_id_filter:
                continue

            if not include_errors and "error" in entry:
                continue

            filtered.append(copy.deepcopy(entry))

        return filtered, trimmed_by_ttl, ttl_snapshot, history_size

    def _build_risk_evaluation_records(
        self,
        entries: Iterable[Mapping[str, Any]],
        *,
        normalized_decision_fields: list[Any] | None,
        flatten_decision: bool,
        decision_prefix: str,
        drop_decision_column: bool,
        fill_value: Any,
        coerce_timestamps: bool,
        tz: tzinfo | None,
    ) -> list[dict[str, Any]]:
        prefix = str(decision_prefix)
        records: list[dict[str, Any]] = []
        for entry in entries:
            record = copy.deepcopy(dict(entry))
            record["timestamp"] = self._normalize_timestamp_for_export(
                record.get("timestamp"),
                coerce=False,
                tz=tz,
            )

            decision_payload = record.get("decision")
            if flatten_decision:
                if normalized_decision_fields is not None:
                    fields = list(normalized_decision_fields)
                elif isinstance(decision_payload, Mapping):
                    fields = list(decision_payload.keys())
                else:
                    fields = []
                for field in fields:
                    column_name = f"{prefix}{field}" if prefix else str(field)
                    if isinstance(decision_payload, Mapping) and field in decision_payload:
                        value = copy.deepcopy(decision_payload[field])
                    else:
                        value = copy.deepcopy(fill_value)
                    if value is pd.NA:
                        value = None
                    record[column_name] = value
                if drop_decision_column:
                    record.pop("decision", None)
            elif drop_decision_column:
                record.pop("decision", None)

            records.append(record)

        if coerce_timestamps:
            for record in records:
                value = record.get("timestamp")
                if isinstance(value, datetime):
                    record["timestamp"] = value.isoformat()
                elif isinstance(value, pd.Timestamp):
                    record["timestamp"] = value.to_pydatetime().isoformat()
                elif value is not None:
                    record["timestamp"] = str(value)

        return records

    @staticmethod
    def _jsonify_risk_evaluation_records(
        records: Iterable[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        def _convert(value: Any) -> Any:
            if value is pd.NA:
                return None
            if isinstance(value, (str, int, bool)) or value is None:
                return value
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    return str(value)
                return value
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, pd.Timestamp):
                return value.to_pydatetime().isoformat()
            if isinstance(value, Mapping):
                return {str(key): _convert(val) for key, val in value.items()}
            if isinstance(value, (list, tuple, set, frozenset)):
                return [_convert(item) for item in value]
            return str(value)

        return [{str(key): _convert(val) for key, val in record.items()} for record in records]

    def _prepare_decision_filter(
        self,
        value: str | Iterable[str | None] | object,
        *,
        missing_token: str,
    ) -> set[str] | None:
        if value is _NO_FILTER:
            return None
        normalized: set[str] = set()
        for candidate in self._iter_filter_values(value):
            if candidate is _NO_FILTER:
                continue
            if candidate is None:
                normalized.add(missing_token)
                continue
            token = str(candidate).strip()
            normalized.add(token or missing_token)
        return normalized or None

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if value is None or value is _NO_FILTER:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(numeric):  # pragma: no cover - defensive guard
            return None
        return numeric

    @staticmethod
    def _ensure_datetime(value: Any, tz: tzinfo | None) -> datetime | None:
        tzinfo = tz or timezone.utc
        if value is None or value is _NO_FILTER:
            return None
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, pd.Timestamp):
            dt = value.to_pydatetime()
        elif isinstance(value, (int, float)):
            numeric = float(value)
            if math.isnan(numeric) or math.isinf(numeric):
                return None
            dt = datetime.fromtimestamp(numeric, tzinfo)
        elif isinstance(value, str):
            token = value.strip()
            if not token:
                return None
            try:
                numeric = float(token)
            except ValueError:
                try:
                    dt = datetime.fromisoformat(token)
                except ValueError:
                    return None
            else:
                if math.isnan(numeric) or math.isinf(numeric):
                    return None
                dt = datetime.fromtimestamp(numeric, tzinfo)
        else:
            timestamp_method = getattr(value, "timestamp", None)
            if callable(timestamp_method):
                try:
                    numeric = float(timestamp_method())
                except (TypeError, ValueError):
                    return None
                if math.isnan(numeric) or math.isinf(numeric):
                    return None
                dt = datetime.fromtimestamp(numeric, tzinfo)
            else:
                return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tzinfo)
        else:
            dt = dt.astimezone(tzinfo)
        return dt

    def _normalize_time_bound(self, value: Any) -> float | None:
        dt = self._ensure_datetime(value, timezone.utc)
        if dt is None:
            return None
        return dt.timestamp()

    def _normalize_timestamp_for_export(
        self,
        value: Any,
        *,
        coerce: bool,
        tz: tzinfo | None,
    ) -> Any:
        target_tz = tz if tz is not None else timezone.utc
        dt = self._ensure_datetime(value, target_tz)
        if dt is None:
            return None
        if coerce:
            if tz is None:
                return pd.Timestamp(dt.replace(tzinfo=None))
            return pd.Timestamp(dt)
        return dt.timestamp()

    def _normalize_approval_flag(self, value: Any) -> str:
        token = self._coerce_optional_bool(value)
        if token is True:
            return _APPROVAL_APPROVED
        if token is False:
            return _APPROVAL_DENIED
        return _APPROVAL_UNKNOWN

    @staticmethod
    def _normalize_normalization_flag(value: Any) -> str:
        if value is True:
            return _NORMALIZED_NORMALIZED
        if value is False:
            return _NORMALIZED_RAW
        return _NORMALIZED_UNKNOWN

    @staticmethod
    def _init_guardrail_numeric_stats() -> dict[str, Any]:
        return {
            "count": 0,
            "missing": 0,
            "sum": 0.0,
            "min": None,
            "max": None,
        }

    @staticmethod
    def _ingest_guardrail_numeric_value(
        stats: dict[str, Any] | None,
        value: Any,
    ) -> None:
        if stats is None:
            return
        numeric = AutoTrader._coerce_float(value)
        if numeric is None:
            stats["missing"] = stats.get("missing", 0) + 1
            return
        stats["count"] = stats.get("count", 0) + 1
        stats["sum"] = stats.get("sum", 0.0) + numeric
        current_min = stats.get("min")
        stats["min"] = numeric if current_min is None else min(current_min, numeric)
        current_max = stats.get("max")
        stats["max"] = numeric if current_max is None else max(current_max, numeric)

    @staticmethod
    def _finalize_guardrail_numeric_stats(
        stats: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not stats:
            return {}
        count = int(stats.get("count", 0) or 0)
        missing = int(stats.get("missing", 0) or 0)
        if count == 0 and missing == 0:
            return {}
        total_sum = float(stats.get("sum", 0.0))
        result: dict[str, Any] = {
            "count": count,
            "missing": missing,
            "sum": total_sum,
            "min": stats.get("min"),
            "max": stats.get("max"),
        }
        if count:
            result["average"] = total_sum / count
        else:
            result["average"] = None if missing else 0.0
        return result

    @staticmethod
    def _create_decision_bucket() -> dict[str, Any]:
        return {
            "total": 0,
            "approved": 0,
            "rejected": 0,
            "unknown": 0,
            "errors": 0,
            "raw_true": 0,
            "raw_false": 0,
            "raw_none": 0,
            "approval_rate": 0.0,
            "error_rate": 0.0,
            "services": Counter(),
        }

    @staticmethod
    def _update_decision_bucket(
        bucket: dict[str, Any],
        *,
        normalized_value: bool | None,
        raw_value: bool | None,
        has_error: bool,
        service_key: str,
    ) -> None:
        bucket["total"] = bucket.get("total", 0) + 1

        if normalized_value is True:
            bucket["approved"] = bucket.get("approved", 0) + 1
        elif normalized_value is False:
            bucket["rejected"] = bucket.get("rejected", 0) + 1
        else:
            bucket["unknown"] = bucket.get("unknown", 0) + 1

        if raw_value is True:
            bucket["raw_true"] = bucket.get("raw_true", 0) + 1
        elif raw_value is False:
            bucket["raw_false"] = bucket.get("raw_false", 0) + 1
        else:
            bucket["raw_none"] = bucket.get("raw_none", 0) + 1

        if has_error:
            bucket["errors"] = bucket.get("errors", 0) + 1

        services = bucket.setdefault("services", Counter())
        if isinstance(services, Counter):
            services[service_key] += 1
        else:  # pragma: no cover - defensive guard for unexpected payloads
            bucket["services"] = Counter({service_key: 1})

    @staticmethod
    def _finalize_decision_bucket(bucket: dict[str, Any]) -> None:
        total = int(bucket.get("total", 0) or 0)
        approved = int(bucket.get("approved", 0) or 0)
        errors = int(bucket.get("errors", 0) or 0)

        bucket["approval_rate"] = approved / total if total else 0.0
        bucket["error_rate"] = errors / total if total else 0.0

        services = bucket.get("services")
        if isinstance(services, Counter):
            bucket["services"] = {
                name: int(count)
                for name, count in sorted(
                    services.items(), key=lambda item: (-item[1], item[0])
                )
            }
        elif isinstance(services, Mapping):
            bucket["services"] = dict(services)
        else:
            bucket["services"] = {}

    @staticmethod
    def _finalize_dimension_counter(counter: Counter[str] | None) -> dict[str, int]:
        if not counter:
            return {}
        return {
            key: int(count)
            for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        }

    @staticmethod
    def _sort_decision_dimension(
        payload: Mapping[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        sorted_payload: dict[str, dict[str, Any]] = {}
        for key, bucket in sorted(
            payload.items(), key=lambda item: (-item[1].get("total", 0), item[0])
        ):
            normalized_bucket = dict(bucket)
            services = normalized_bucket.get("services")
            if isinstance(services, Counter):
                normalized_bucket["services"] = {
                    name: int(count)
                    for name, count in sorted(
                        services.items(), key=lambda item: (-item[1], item[0])
                    )
                }
            elif isinstance(services, Mapping):
                normalized_bucket["services"] = dict(services)
            sorted_payload[key] = normalized_bucket
        return sorted_payload

    @staticmethod
    def _normalize_decision_dimension_value(
        value: Any,
        *,
        missing_token: str,
    ) -> str:
        if value is None:
            return missing_token
        token = str(value).strip()
        return token or missing_token

    @staticmethod
    def _serialize_filter_snapshot(values: Iterable[Any] | None) -> list[Any] | None:
        if values is None:
            return None
        return [value for value in sorted(values, key=lambda item: (str(item).lower(), str(item)))]

    @staticmethod
    def _serialize_numeric_filter_snapshot(
        payload: tuple[set[float], bool] | None,
    ) -> Mapping[str, Any] | None:
        if payload is None:
            return None
        values, include_missing = payload
        return {
            "values": [float(item) for item in sorted(values)],
            "include_missing": bool(include_missing),
        }

    def _snapshot_decision_timeline_filters(
        self,
        *,
        approved_filter: set[bool | None] | None,
        normalized_filter: set[bool | None] | None,
        include_errors: bool,
        service_filter: set[str] | None,
        decision_state_filter: set[str] | None,
        decision_reason_filter: set[str] | None,
        decision_mode_filter: set[str] | None,
        decision_id_filter: set[str] | None,
        since_ts: float | None,
        until_ts: float | None,
        include_services: bool,
        include_decision_dimensions: bool,
        fill_gaps: bool,
        coerce_timestamps: bool,
        tz_value: tzinfo | None,
    ) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "approved": self._serialize_filter_snapshot(approved_filter),
            "normalized": self._serialize_filter_snapshot(normalized_filter),
            "include_errors": include_errors,
            "service": self._serialize_filter_snapshot(service_filter),
            "decision_state": self._serialize_filter_snapshot(decision_state_filter),
            "decision_reason": self._serialize_filter_snapshot(decision_reason_filter),
            "decision_mode": self._serialize_filter_snapshot(decision_mode_filter),
            "decision_id": self._serialize_filter_snapshot(decision_id_filter),
            "since": since_ts,
            "until": until_ts,
            "include_services": include_services,
            "include_decision_dimensions": include_decision_dimensions,
            "fill_gaps": fill_gaps,
            "coerce_timestamps": coerce_timestamps,
            "tz": tz_value.tzname(None) if tz_value is not None else None,
        }
        return snapshot

    def _snapshot_guardrail_timeline_filters(
        self,
        *,
        approved_filter: set[bool | None] | None,
        normalized_filter: set[bool | None] | None,
        include_errors: bool,
        service_filter: set[str] | None,
        decision_state_filter: set[str] | None,
        decision_reason_filter: set[str] | None,
        decision_mode_filter: set[str] | None,
        decision_id_filter: set[str] | None,
        reason_filter: set[str] | None,
        trigger_filter: set[str] | None,
        trigger_label_filter: set[str] | None,
        trigger_comparator_filter: set[str] | None,
        trigger_unit_filter: set[str] | None,
        trigger_threshold_filter: tuple[set[float], bool] | None,
        trigger_threshold_min: float | None,
        trigger_threshold_max: float | None,
        trigger_value_filter: tuple[set[float], bool] | None,
        trigger_value_min: float | None,
        trigger_value_max: float | None,
        since_ts: float | None,
        until_ts: float | None,
        include_services: bool,
        include_guardrail_dimensions: bool,
        include_decision_dimensions: bool,
        fill_gaps: bool,
        coerce_timestamps: bool,
        tz_value: tzinfo | None,
    ) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "approved": self._serialize_filter_snapshot(approved_filter),
            "normalized": self._serialize_filter_snapshot(normalized_filter),
            "include_errors": include_errors,
            "service": self._serialize_filter_snapshot(service_filter),
            "decision_state": self._serialize_filter_snapshot(decision_state_filter),
            "decision_reason": self._serialize_filter_snapshot(decision_reason_filter),
            "decision_mode": self._serialize_filter_snapshot(decision_mode_filter),
            "decision_id": self._serialize_filter_snapshot(decision_id_filter),
            "reason": self._serialize_filter_snapshot(reason_filter),
            "trigger": self._serialize_filter_snapshot(trigger_filter),
            "trigger_label": self._serialize_filter_snapshot(trigger_label_filter),
            "trigger_comparator": self._serialize_filter_snapshot(
                trigger_comparator_filter
            ),
            "trigger_unit": self._serialize_filter_snapshot(trigger_unit_filter),
            "trigger_threshold": self._serialize_numeric_filter_snapshot(
                trigger_threshold_filter
            ),
            "trigger_threshold_min": trigger_threshold_min,
            "trigger_threshold_max": trigger_threshold_max,
            "trigger_value": self._serialize_numeric_filter_snapshot(trigger_value_filter),
            "trigger_value_min": trigger_value_min,
            "trigger_value_max": trigger_value_max,
            "since": since_ts,
            "until": until_ts,
            "include_services": include_services,
            "include_guardrail_dimensions": include_guardrail_dimensions,
            "include_decision_dimensions": include_decision_dimensions,
            "fill_gaps": fill_gaps,
            "coerce_timestamps": coerce_timestamps,
            "tz": tz_value.tzname(None) if tz_value is not None else None,
        }
        return snapshot

    def _build_risk_evaluation_records(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        normalized_decision_fields: Sequence[Any] | None,
        flatten_decision: bool,
        decision_prefix: str,
        drop_decision_column: bool,
        fill_value: Any,
        coerce_timestamps: bool,
        tz: tzinfo | None,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for entry in records:
            record = copy.deepcopy(dict(entry))
            record["timestamp"] = self._normalize_timestamp_for_export(
                record.get("timestamp"),
                coerce=coerce_timestamps,
                tz=tz,
            )

            if "service" not in record and entry.get("service") is None:
                record.pop("service", None)

            record.setdefault("response", None)
            record.setdefault("error", None)

            decision_payload = record.get("decision")
            if flatten_decision:
                if not isinstance(decision_payload, Mapping):
                    decision_mapping: Mapping[str, Any] = {}
                else:
                    decision_mapping = decision_payload

                if normalized_decision_fields:
                    fields = list(normalized_decision_fields)
                else:
                    fields = list(decision_mapping.keys())

                for field in fields:
                    column_name = f"{decision_prefix}{field}"
                    if isinstance(decision_mapping, Mapping) and field in decision_mapping:
                        value = copy.deepcopy(decision_mapping[field])
                    else:
                        value = fill_value
                    record[column_name] = value

                if drop_decision_column:
                    record.pop("decision", None)
            output.append(record)
        return output

    def _jsonify_risk_evaluation_records(
        self, records: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        json_ready: list[dict[str, Any]] = []
        for entry in records:
            payload = copy.deepcopy(dict(entry))
            timestamp_value = payload.get("timestamp")
            dt = self._ensure_datetime(timestamp_value, timezone.utc)
            payload["timestamp"] = dt.isoformat() if dt is not None else None
            json_ready.append(payload)
        return json_ready

        probability = self._ai_probability_from_prediction(value)
        evaluated_at_raw = predictions.index[-1]
        evaluated_at: str | float | None
        if hasattr(evaluated_at_raw, "isoformat"):
            evaluated_at = evaluated_at_raw.isoformat()
        elif isinstance(evaluated_at_raw, (int, float)):
            evaluated_at = float(evaluated_at_raw)
        else:
            evaluated_at = None

        snapshot: Dict[str, object] = {
            "prediction": value,
            "prediction_bps": prediction_bps,
            "threshold_bps": threshold,
            "direction": direction,
            "probability": probability,
        }
        if evaluated_at is not None:
            snapshot["evaluated_at"] = evaluated_at

        self._log(
            "AI prediction snapshot",
            level=logging.DEBUG,
            symbol=symbol,
            prediction_bps=prediction_bps,
            direction=direction,
            threshold_bps=threshold,
        )
        return snapshot

    def _snapshot_guardrail_timeline_filters(
        self,
        *,
        approved_filter: set[bool | None] | None,
        normalized_filter: set[bool | None] | None,
        include_errors: bool,
        service_filter: set[str] | None,
        decision_state_filter: set[str] | None,
        decision_reason_filter: set[str] | None,
        decision_mode_filter: set[str] | None,
        decision_id_filter: set[str] | None,
        reason_filter: set[str] | None,
        trigger_filter: set[str] | None,
        trigger_label_filter: set[str] | None,
        trigger_comparator_filter: set[str] | None,
        trigger_unit_filter: set[str] | None,
        trigger_threshold_filter: tuple[set[float], bool] | None,
        trigger_threshold_min: float | None,
        trigger_threshold_max: float | None,
        trigger_value_filter: tuple[set[float], bool] | None,
        trigger_value_min: float | None,
        trigger_value_max: float | None,
        since_ts: float | None,
        until_ts: float | None,
        include_services: bool,
        include_guardrail_dimensions: bool,
        include_decision_dimensions: bool,
        fill_gaps: bool,
        coerce_timestamps: bool,
        tz_value: tzinfo | None,
    ) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "approved": self._serialize_filter_snapshot(approved_filter),
            "normalized": self._serialize_filter_snapshot(normalized_filter),
            "include_errors": include_errors,
            "service": self._serialize_filter_snapshot(service_filter),
            "decision_state": self._serialize_filter_snapshot(decision_state_filter),
            "decision_reason": self._serialize_filter_snapshot(decision_reason_filter),
            "decision_mode": self._serialize_filter_snapshot(decision_mode_filter),
            "decision_id": self._serialize_filter_snapshot(decision_id_filter),
            "reason": self._serialize_filter_snapshot(reason_filter),
            "trigger": self._serialize_filter_snapshot(trigger_filter),
            "trigger_label": self._serialize_filter_snapshot(trigger_label_filter),
            "trigger_comparator": self._serialize_filter_snapshot(
                trigger_comparator_filter
            ),
            "trigger_unit": self._serialize_filter_snapshot(trigger_unit_filter),
            "trigger_threshold": self._serialize_numeric_filter_snapshot(
                trigger_threshold_filter
            ),
            "trigger_threshold_min": trigger_threshold_min,
            "trigger_threshold_max": trigger_threshold_max,
            "trigger_value": self._serialize_numeric_filter_snapshot(trigger_value_filter),
            "trigger_value_min": trigger_value_min,
            "trigger_value_max": trigger_value_max,
            "since": since_ts,
            "until": until_ts,
            "include_services": include_services,
            "include_guardrail_dimensions": include_guardrail_dimensions,
            "include_decision_dimensions": include_decision_dimensions,
            "fill_gaps": fill_gaps,
            "coerce_timestamps": coerce_timestamps,
            "tz": tz_value.tzname(None) if tz_value is not None else None,
        }
        return snapshot

    def _build_risk_evaluation_records(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        normalized_decision_fields: Sequence[Any] | None,
        flatten_decision: bool,
        decision_prefix: str,
        drop_decision_column: bool,
        fill_value: Any,
        coerce_timestamps: bool,
        tz: tzinfo | None,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for entry in records:
            record = copy.deepcopy(dict(entry))
            record["timestamp"] = self._normalize_timestamp_for_export(
                record.get("timestamp"),
                coerce=coerce_timestamps,
                tz=tz,
            )

            if "service" not in record and entry.get("service") is None:
                record.pop("service", None)

            record.setdefault("response", None)
            record.setdefault("error", None)

            decision_payload = record.get("decision")
            if flatten_decision:
                if not isinstance(decision_payload, Mapping):
                    decision_mapping: Mapping[str, Any] = {}
                else:
                    decision_mapping = decision_payload

                if normalized_decision_fields:
                    fields = list(normalized_decision_fields)
                else:
                    fields = list(decision_mapping.keys())

                for field in fields:
                    column_name = f"{decision_prefix}{field}"
                    if isinstance(decision_mapping, Mapping) and field in decision_mapping:
                        value = copy.deepcopy(decision_mapping[field])
                    else:
                        value = fill_value
                    record[column_name] = value

                if drop_decision_column:
                    record.pop("decision", None)
            output.append(record)
        return output

    def _jsonify_risk_evaluation_records(
        self, records: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        json_ready: list[dict[str, Any]] = []
        for entry in records:
            payload = copy.deepcopy(dict(entry))
            timestamp_value = payload.get("timestamp")
            dt = self._ensure_datetime(timestamp_value, timezone.utc)
            payload["timestamp"] = dt.isoformat() if dt is not None else None
            json_ready.append(payload)
        return json_ready

        probability = self._ai_probability_from_prediction(value)
        evaluated_at_raw = predictions.index[-1]
        evaluated_at: str | float | None
        if hasattr(evaluated_at_raw, "isoformat"):
            evaluated_at = evaluated_at_raw.isoformat()
        elif isinstance(evaluated_at_raw, (int, float)):
            evaluated_at = float(evaluated_at_raw)
        else:
            evaluated_at = None

        snapshot: Dict[str, object] = {
            "prediction": value,
            "prediction_bps": prediction_bps,
            "threshold_bps": threshold,
            "direction": direction,
            "probability": probability,
        }
        if evaluated_at is not None:
            snapshot["evaluated_at"] = evaluated_at

        self._log(
            "AI prediction snapshot",
            level=logging.DEBUG,
            symbol=symbol,
            prediction_bps=prediction_bps,
            direction=direction,
            threshold_bps=threshold,
        )
        return snapshot

    def _normalize_ai_context(
        self,
        ai_context: Mapping[str, object],
        *,
        default_return_bps: float,
        default_probability: float,
    ) -> tuple[float, float, Dict[str, Any]]:
        normalized_return = float(default_return_bps)
        normalized_probability = max(0.0, min(1.0, float(default_probability)))
        payload: Dict[str, Any] = {}

        prediction_raw = ai_context.get("prediction")
        if prediction_raw is not None:
            try:
                payload["prediction"] = float(prediction_raw)
            except (TypeError, ValueError):
                pass

        prediction_bps_raw = ai_context.get("prediction_bps")
        if prediction_bps_raw is not None:
            try:
                normalized_return = float(prediction_bps_raw)
            except (TypeError, ValueError):
                pass
        payload["prediction_bps"] = normalized_return

        threshold_raw = ai_context.get("threshold_bps")
        try:
            payload["threshold_bps"] = float(threshold_raw) if threshold_raw is not None else 0.0
        except (TypeError, ValueError):
            payload["threshold_bps"] = 0.0

        payload["direction"] = ai_context.get("direction")

        probability_raw = ai_context.get("probability")
        if probability_raw is not None:
            try:
                ai_probability = max(0.0, min(1.0, float(probability_raw)))
            except (TypeError, ValueError):
                ai_probability = None
            if ai_probability is not None:
                payload["probability"] = ai_probability
                normalized_probability = max(normalized_probability, ai_probability)

        if "evaluated_at" in ai_context:
            payload["evaluated_at"] = ai_context["evaluated_at"]

        return normalized_return, normalized_probability, payload

    def _resolve_decision_orchestrator(self) -> Any | None:
        orchestrator = self.decision_orchestrator
        if orchestrator is not None:
            return orchestrator
        if self.bootstrap_context is not None:
            return getattr(self.bootstrap_context, "decision_orchestrator", None)
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

    def _decision_risk_profile_name(self) -> str:
        if self.bootstrap_context is not None:
            profile = getattr(self.bootstrap_context, "risk_profile_name", None)
            if profile:
                return str(profile)
        return "default"

    def _estimate_candidate_notional(self, symbol: str) -> float:
        del symbol  # symbol not used yet
        leverage = abs(getattr(self, "current_leverage", 1.0))
        return max(1000.0, leverage * 1000.0)

    def _build_decision_candidate(
        self,
        *,
        symbol: str,
        signal: str,
        market_data: pd.DataFrame,
        assessment: MarketRegimeAssessment,
        last_return: float,
        ai_context: Mapping[str, object] | None = None,
        ai_manager: Any | None = None,
    ) -> Any | None:
        if DecisionCandidate is None:
            return None
        if market_data.empty:
            return None

        feature_cols = self._ai_feature_columns(market_data)
        try:
            row = market_data.iloc[-1]
        except Exception:
            return None
        features: Dict[str, float] = {}
        decision_section: Dict[str, Any]
        selected_columns = [
            column for column in feature_cols if column in getattr(row, "index", ())
        ]
        if selected_columns:
            for column in selected_columns:
                value = row.get(column)
                try:
                    features[str(column)] = float(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
        else:
            for key, value in row.items():
                try:
                    features[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        features["assessment_confidence"] = float(assessment.confidence)
        features["assessment_risk"] = float(assessment.risk_score)
        features["signal_direction"] = 1.0 if signal == "buy" else -1.0
        timestamp = getattr(row, "name", None)
        metadata: Dict[str, Any] = {
            "auto_trader": {
                "signal": signal,
                "strategy": self.current_strategy,
            },
            "decision_engine": {
                "features": features,
                "generated_at": timestamp,
            },
        }
        decision_section = metadata["decision_engine"]
        decision_section.update(self._feature_column_metadata(selected_columns))
        expected_return = float(last_return * 10_000.0)
        if signal == "sell":
            expected_return = -abs(expected_return)
        elif signal == "buy":
            expected_return = abs(expected_return)
        expected_probability = max(0.0, min(1.0, float(assessment.confidence)))
        candidate_notional = self._estimate_candidate_notional(symbol)
        if ai_manager is not None and hasattr(ai_manager, "build_decision_engine_payload"):
            try:
                engine_payload = ai_manager.build_decision_engine_payload(
                    strategy=self.current_strategy,
                    action="enter" if signal == "buy" else "exit",
                    risk_profile=self._decision_risk_profile_name(),
                    symbol=symbol,
                    notional=candidate_notional,
                    features=features,
                )
            except Exception:  # pragma: no cover - diagnostyka integracji
                engine_payload = None
                LOGGER.debug("AI manager decision payload generation failed", exc_info=True)
            else:
                if isinstance(engine_payload, Mapping):
                    decision_section.update(engine_payload)
                    if ai_context is None and isinstance(engine_payload.get("ai"), Mapping):
                        ai_context = engine_payload.get("ai")
        if ai_context:
            expected_return, expected_probability, ai_payload = self._normalize_ai_context(
                ai_context,
                default_return_bps=expected_return,
                default_probability=expected_probability,
            )
            decision_section["ai"] = ai_payload
        candidate = DecisionCandidate(
            strategy=self.current_strategy,
            action="enter" if signal == "buy" else "exit",
            risk_profile=self._decision_risk_profile_name(),
            symbol=symbol,
            notional=candidate_notional,
            expected_return_bps=expected_return,
            expected_probability=expected_probability,
            metadata=metadata,
        )
        return candidate

    def _build_risk_snapshot(self, profile: str) -> Mapping[str, object]:
        engine = self._decision_risk_engine or self.core_risk_engine
        if engine is not None and hasattr(engine, "snapshot_state"):
            try:
                snapshot = engine.snapshot_state(profile)
            except Exception:
                snapshot = None
            if snapshot:
                return snapshot
        return {
            "profile": profile,
            "start_of_day_equity": 100_000.0,
            "last_equity": 100_000.0,
            "peak_equity": 100_000.0,
            "daily_realized_pnl": 0.0,
            "positions": {},
        }

    def _decision_threshold_snapshot(self) -> Mapping[str, object]:
        config = self._decision_engine_config
        thresholds: Mapping[str, object] | None = None
        if config is not None:
            orchestrator_cfg = getattr(config, "orchestrator", None)
            if orchestrator_cfg is not None:
                thresholds = {
                    "max_cost_bps": getattr(orchestrator_cfg, "max_cost_bps", None),
                    "min_net_edge_bps": getattr(orchestrator_cfg, "min_net_edge_bps", None),
                    "min_probability": getattr(config, "min_probability", None),
                }
        return thresholds or {}

    def _serialize_decision_evaluation(
        self,
        evaluation: Any,
        *,
        thresholds: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        payload: Dict[str, Any] = {
            "accepted": bool(getattr(evaluation, "accepted", False)),
            "reasons": list(getattr(evaluation, "reasons", ())),
            "model": getattr(evaluation, "model_name", None),
            "net_edge_bps": getattr(evaluation, "net_edge_bps", None),
            "cost_bps": getattr(evaluation, "cost_bps", None),
            "model_expected_return_bps": getattr(
                evaluation, "model_expected_return_bps", None
            ),
            "model_success_probability": getattr(
                evaluation, "model_success_probability", None
            ),
        }
        candidate = getattr(evaluation, "candidate", None)
        if candidate is not None:
            candidate_metadata = getattr(candidate, "metadata", None)
            if isinstance(candidate_metadata, Mapping):
                decision_meta = candidate_metadata.get("decision_engine")
                if isinstance(decision_meta, Mapping):
                    features = decision_meta.get("features")
                    if isinstance(features, Mapping):
                        payload.setdefault("features", dict(features))
                    for key in (
                        "feature_columns",
                        "feature_columns_source",
                        "configured_feature_columns",
                    ):
                        if key in decision_meta:
                            payload.setdefault(key, copy.deepcopy(decision_meta[key]))
                    if "ai" in decision_meta and "ai" not in payload:
                        ai_payload = decision_meta["ai"]
                        if isinstance(ai_payload, Mapping):
                            payload["ai"] = copy.deepcopy(ai_payload)
        selection = getattr(evaluation, "model_selection", None)
        if selection is not None and hasattr(selection, "to_mapping"):
            try:
                payload["model_selection"] = selection.to_mapping()
            except Exception:
                payload["model_selection"] = {
                    "selected": getattr(selection, "selected", None),
                }
        snapshot = thresholds or getattr(evaluation, "thresholds_snapshot", None)
        if snapshot:
            payload["thresholds"] = dict(snapshot)
        return payload

    def _evaluate_decision_candidate(
        self,
        *,
        symbol: str,
        signal: str,
        market_data: pd.DataFrame,
        assessment: MarketRegimeAssessment,
        last_return: float,
        ai_context: Mapping[str, object] | None = None,
        ai_manager: Any | None = None,
    ) -> Any | None:
        orchestrator = self._resolve_decision_orchestrator()
        if orchestrator is None or DecisionCandidate is None:
            return None
        candidate = self._build_decision_candidate(
            symbol=symbol,
            signal=signal,
            market_data=market_data,
            assessment=assessment,
            last_return=last_return,
            ai_context=ai_context,
            ai_manager=ai_manager,
        )
        if candidate is None:
            return None
        snapshot = self._build_risk_snapshot(candidate.risk_profile)
        try:
            evaluation = orchestrator.evaluate_candidate(candidate, snapshot)
        except Exception as exc:  # pragma: no cover - defensywne logowanie
            self._log(
                f"DecisionOrchestrator evaluation failed: {exc!r}",
                level=logging.ERROR,
            )
            return None
        return evaluation

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

    def _apply_orchestrator_strategy_selection(
        self, assessment: MarketRegimeAssessment
    ) -> None:
        orchestrator = self._resolve_decision_orchestrator()
        if orchestrator is None:
            return
        selector = getattr(orchestrator, "select_strategy", None)
        if not callable(selector):
            return
        try:
            selected = selector(assessment.regime)
        except Exception:  # pragma: no cover - defensive logging
            self._log(
                "DecisionOrchestrator.select_strategy failed",
                level=logging.DEBUG,
            )
            return
        if not selected:
            return
        selected_name = str(selected)
        if selected_name != self.current_strategy:
            self._log(
                "Strategy overridden by DecisionOrchestrator",
                level=logging.INFO,
                previous=self.current_strategy,
                selected=selected_name,
                regime=assessment.regime.value,
            )
        self.current_strategy = selected_name
        self._update_strategy_metrics(self.current_strategy)

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

    def _handle_guardrail_trigger(
        self,
        symbol: str,
        reasons: Sequence[str],
        triggers: Sequence[GuardrailTrigger],
    ) -> None:
        if not reasons:
            return
        unique_guardrails: set[str] = set()
        for trigger in triggers:
            guardrail_name = str(getattr(trigger, "name", None) or getattr(trigger, "label", "unknown"))
            unique_guardrails.add(guardrail_name)
        if not unique_guardrails:
            unique_guardrails.add("unknown")
        for name in unique_guardrails:
            self._metric_guardrail_blocks_total.inc(
                labels=self._metric_label_payload(guardrail=name)
            )
        alert_context = {
            "symbol": symbol,
            "strategy": self.current_strategy,
            "reasons": "; ".join(str(reason) for reason in reasons),
        }
        self._emit_alert(
            "auto_trader.guardrail",
            "Guardrail zablokował transakcję",
            "\n".join(str(reason) for reason in reasons),
            severity="warning",
            context=alert_context,
        )
        if self.current_strategy != "capital_preservation":
            previous = self.current_strategy
            self.current_strategy = "capital_preservation"
            self._log(
                "Guardrail forced strategy fallback",
                level=logging.INFO,
                previous=previous,
            )
            self._update_strategy_metrics(self.current_strategy)

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
        decision_engine: Any | None = None,
        ai_context: Mapping[str, object] | None = None,
    ) -> RiskDecision:
        should_trade = signal in {"buy", "sell"} and self.current_leverage > 0 and not cooldown_active
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
        decision_payload: Dict[str, Any] | None = None
        if decision_engine is not None:
            decision_payload = self._serialize_decision_evaluation(
                decision_engine,
                thresholds=self._decision_threshold_snapshot(),
            )
        feature_metadata = self._feature_column_metadata()
        if feature_metadata:
            if decision_payload is None:
                decision_payload = {}
            for key, value in feature_metadata.items():
                decision_payload.setdefault(key, copy.deepcopy(value))
        if ai_context:
            base_return_raw = ai_context.get("prediction_bps", 0.0)
            try:
                base_return = float(base_return_raw)
            except (TypeError, ValueError):
                base_return = 0.0
            base_probability_raw = ai_context.get("probability", 0.0)
            try:
                base_probability = float(base_probability_raw)
            except (TypeError, ValueError):
                base_probability = 0.0
            _, _, ai_payload = self._normalize_ai_context(
                ai_context,
                default_return_bps=base_return,
                default_probability=base_probability,
            )
            if decision_payload is None:
                decision_payload = {}
            decision_payload["ai"] = ai_payload
        if decision_payload is not None:
            details["decision_engine"] = decision_payload
        mode = self._schedule_mode
        gui_mode: str | None = None
        if hasattr(self.gui, "is_demo_mode_active"):
            try:
                gui_mode = "demo" if self.gui.is_demo_mode_active() else "live"
            except Exception:
                gui_mode = None
        if mode not in {"demo", "live"}:
            mode = gui_mode or "demo"
        elif gui_mode is not None and gui_mode != mode:
            self._log(
                "Schedule mode overrides GUI mode",
                level=logging.DEBUG,
                schedule_mode=mode,
                gui_mode=gui_mode,
            )
        if getattr(self, "_ai_degraded", False):
            details["ai_degraded"] = True
            if mode == "live":
                should_trade = False
                state = "halted"
                reason = "AI backend degraded"
        fraction = self.current_leverage if should_trade else 0.0
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
        if not self._enforce_work_schedule():
            return
        self._metric_cycle_total.inc(labels=self._base_metric_labels)
        self._process_orchestrator_recalibrations()
        runner = self._resolve_controller_runner()
        if runner is not None:
            self._execute_controller_runner_cycle(runner)
            self._auto_trade_stop.wait(self.auto_trade_interval_s)
            return

        risk_service = getattr(self, "risk_service", None)
        if risk_service is None:
            risk_service = getattr(self, "core_risk_engine", None)

        try:
            symbol = self.symbol_getter()
        except Exception as exc:  # pragma: no cover - defensive guard
            self._log(f"Failed to resolve trading symbol: {exc!r}", level=logging.ERROR)
            self._log_decision_event(
                "cycle_failed",
                status="symbol_error",
                metadata={"error": repr(exc)},
            )
            self._auto_trade_stop.wait(self.auto_trade_interval_s)
            return

        execution_service = self._resolve_execution_service(symbol)

        timeframe = "1h"
        timeframe_var = getattr(self.gui, "timeframe_var", None)
        if timeframe_var is not None and hasattr(timeframe_var, "get"):
            try:
                timeframe = str(timeframe_var.get())
            except Exception:
                timeframe = "1h"

        self._log_decision_event(
            "cycle_started",
            symbol=str(symbol) if symbol else None,
            status="pending",
            metadata={"timeframe": timeframe},
        )

        ai_manager = self._resolve_ai_manager()
        self._ai_degraded = bool(getattr(ai_manager, "is_degraded", False)) if ai_manager else False
        if self._ai_degraded:
            self._log(
                "AI manager running in degraded mode – live trades require explicit confirmation.",
                level=logging.WARNING,
            )
        if not symbol or ai_manager is None:
            self._log("Auto-trade prerequisites missing AI manager or symbol", level=logging.DEBUG)
            self._log_decision_event(
                "cycle_skipped",
                symbol=str(symbol) if symbol else None,
                status="missing_prerequisites",
                metadata={
                    "has_symbol": bool(symbol),
                    "has_ai_manager": ai_manager is not None,
                },
            )
            self._auto_trade_stop.wait(self.auto_trade_interval_s)
            return

        market_data = self._fetch_market_data(symbol, timeframe)
        if market_data is None or market_data.empty:
            self._log(
                f"No market data available for {symbol} on {timeframe}",
                level=logging.WARNING,
            )
            self._log_decision_event(
                "cycle_skipped",
                symbol=str(symbol) if symbol else None,
                status="no_market_data",
                metadata={"timeframe": timeframe},
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

        ai_context = self._compute_ai_signal_context(ai_manager, symbol, market_data)
        self._last_ai_context = ai_context
        ai_direction = None
        ai_prediction_bps = 0.0
        ai_threshold_bps = 0.0
        if ai_context:
            ai_direction = ai_context.get("direction")
            try:
                ai_prediction_bps = float(ai_context.get("prediction_bps", 0.0))
            except (TypeError, ValueError):
                ai_prediction_bps = 0.0
            try:
                ai_threshold_bps = float(ai_context.get("threshold_bps", 0.0))
            except (TypeError, ValueError):
                ai_threshold_bps = 0.0

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
        self._apply_orchestrator_strategy_selection(assessment)
        signal = self._map_regime_to_signal(assessment, last_return, summary=summary)
        signal = self._apply_signal_guardrails(signal, effective_risk, summary)
        pre_ai_signal = signal
        ai_force_hold = False
        if ai_context is None:
            if pre_ai_signal in {"buy", "sell"}:
                self._log(
                    "AI predictions unavailable – forcing HOLD",
                    level=logging.WARNING,
                    symbol=symbol,
                )
            signal = "hold"
            ai_force_hold = True
        else:
            if ai_direction == "hold":
                if pre_ai_signal in {"buy", "sell"}:
                    self._log(
                        "AI prediction below quality threshold; forcing HOLD",
                        level=logging.INFO,
                        symbol=symbol,
                        prediction_bps=ai_prediction_bps,
                        threshold_bps=ai_threshold_bps,
                    )
                signal = "hold"
                ai_force_hold = True
            elif pre_ai_signal in {"buy", "sell"} and ai_direction not in (None, pre_ai_signal):
                self._log(
                    "AI prediction disagrees with regime signal; forcing HOLD",
                    level=logging.INFO,
                    symbol=symbol,
                    regime_signal=pre_ai_signal,
                    ai_direction=ai_direction,
                    prediction_bps=ai_prediction_bps,
                )
                signal = "hold"
                ai_force_hold = True

        decision_evaluation: Any | None = None
        evaluation_payload: Mapping[str, object] | None = None
        if signal in {"buy", "sell"}:
            decision_evaluation = self._evaluate_decision_candidate(
                symbol=symbol,
                signal=signal,
                market_data=market_data,
                assessment=assessment,
                last_return=last_return,
                ai_context=ai_context,
                ai_manager=ai_manager,
            )
            if decision_evaluation is not None:
                thresholds_snapshot = self._decision_threshold_snapshot()
                evaluation_payload = self._serialize_decision_evaluation(
                    decision_evaluation,
                    thresholds=thresholds_snapshot,
                )
                if not getattr(decision_evaluation, "accepted", True):
                    self._log(
                        "DecisionOrchestrator rejected signal",  # type: ignore[arg-type]
                        level=logging.INFO,
                        symbol=symbol,
                        signal=signal,
                        reasons=list(getattr(decision_evaluation, "reasons", ())),
                        thresholds=thresholds_snapshot,
                    )
                    signal = "hold"
                    ai_force_hold = True
        if evaluation_payload is not None:
            accepted = bool(evaluation_payload.get("accepted"))
            evaluation_status = "accepted" if accepted else "rejected"
            evaluation_confidence = self._safe_float(
                evaluation_payload.get("model_success_probability")
            )
            self._log_decision_event(
                "decision_evaluated",
                symbol=str(symbol) if symbol else None,
                status=evaluation_status,
                metadata=evaluation_payload,
                confidence=evaluation_confidence,
            )
            self._record_decision_audit_stage(
                "decision_evaluated",
                symbol=symbol,
                payload=evaluation_payload,
            )

        guardrail_reasons = list(self._last_guardrail_reasons)
        guardrail_objects = list(self._last_guardrail_triggers)
        guardrail_triggers = [trigger.to_dict() for trigger in guardrail_objects]
        if guardrail_reasons and signal == "hold" and not ai_force_hold:
            self._log(
                "Signal overridden by guardrails",
                level=logging.INFO,
                reasons=guardrail_reasons,
                triggers=guardrail_triggers,
            )
            self._log_decision_event(
                "decision_guardrail",
                symbol=str(symbol) if symbol else None,
                status="blocked",
                metadata={
                    "reasons": list(guardrail_reasons),
                    "triggers": guardrail_triggers,
                },
            )
            self._handle_guardrail_trigger(symbol, guardrail_reasons, guardrail_objects)
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
            decision_engine=decision_evaluation,
            ai_context=ai_context,
        )

        decision_metadata: dict[str, object] = {
            "state": decision.state,
            "reason": decision.reason,
            "mode": decision.mode,
            "details": copy.deepcopy(decision.details),
            "cooldown_active": decision.cooldown_active,
            "cooldown_reason": decision.cooldown_reason,
        }
        decision_status = "trade" if decision.should_trade else "hold"
        decision_side = None
        if isinstance(decision.details, Mapping):
            decision_side = decision.details.get("signal")
        if decision_side is None and decision.should_trade:
            decision_side = signal
        confidence = self._extract_decision_confidence(decision.details)
        self._log_decision_event(
            "decision_composed",
            symbol=str(symbol) if symbol else None,
            status=decision_status,
            side=str(decision_side) if decision_side is not None else None,
            quantity=decision.fraction,
            metadata=decision_metadata,
            confidence=confidence,
        )
        self._record_decision_audit_stage(
            "decision_composed",
            symbol=symbol,
            payload=decision.to_dict(),
            portfolio_snapshot=self._capture_portfolio_snapshot(),
        )
        self._last_signal = signal
        self._last_regime = assessment
        self._last_risk_decision = decision

        risk_service = self._resolve_risk_service()
        risk_invoked = risk_service is not None
        risk_response: Any | None = None
        recorded_approval: bool | None = None
        normalized_approval: bool | None = None
        risk_error: BaseException | None = None

        if risk_service is not None:
            try:
                risk_response = self._invoke_risk_service(risk_service, decision)
            except Exception as exc:  # pragma: no cover - defensive guard
                risk_error = exc
                self._log(
                    "Risk service evaluation failed; treating as rejected",
                    level=logging.ERROR,
                    symbol=symbol,
                    error=repr(exc),
                )
                recorded_approval = False
                normalized_approval = False
            else:
                recorded_approval, normalized_approval = self._normalize_risk_approval(
                    decision,
                    risk_service,
                    risk_response,
                )

        if normalized_approval is None:
            normalized_approval = decision.should_trade if not risk_invoked else False
        if recorded_approval is None and risk_invoked:
            recorded_approval = normalized_approval

        if risk_invoked:
            self._record_risk_evaluation(
                decision,
                approved=recorded_approval,
                normalized=normalized_approval,
                response=risk_response,
                service=risk_service,
                error=risk_error,
            )
            audit_payload: dict[str, Any] = {
                "approved": recorded_approval,
                "normalized": normalized_approval,
            }
            if risk_service is not None:
                audit_payload["service"] = type(risk_service).__name__
            if risk_response is not None:
                audit_payload["response"] = self._summarize_risk_response(risk_response)
            if risk_error is not None:
                audit_payload["error"] = str(risk_error)
            self._log_decision_event(
                "risk_evaluated",
                symbol=str(symbol) if symbol else None,
                status="approved" if normalized_approval else "rejected",
                metadata=audit_payload,
            )
            self._record_decision_audit_stage(
                "risk_evaluated",
                symbol=symbol,
                payload=audit_payload,
                risk_snapshot=self._capture_risk_snapshot(),
            )
        else:
            self._record_decision_audit_stage(
                "risk_skipped",
                symbol=symbol,
                payload={"reason": "no_service"},
            )
            self._log_decision_event(
                "risk_skipped",
                symbol=str(symbol) if symbol else None,
                status="skipped",
                metadata={"reason": "no_service"},
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
                self._log_decision_event(
                    "execution_skipped",
                    symbol=str(symbol) if symbol else None,
                    status="cooldown",
                    metadata={"reason": "cooldown"},
                )
                self._record_decision_audit_stage(
                    "execution_skipped",
                    symbol=symbol,
                    payload={"reason": "cooldown"},
                    risk_snapshot=self._capture_risk_snapshot(),
                )
            elif not should_trade:
                self._log(
                    "Risk evaluation approved trade but decision is not actionable; skipping execution",
                    level=logging.DEBUG,
                )
                self._log_decision_event(
                    "execution_skipped",
                    symbol=str(symbol) if symbol else None,
                    status="not_actionable",
                    metadata={"reason": "not_actionable"},
                )
                self._record_decision_audit_stage(
                    "execution_skipped",
                    symbol=symbol,
                    payload={"reason": "not_actionable"},
                    risk_snapshot=self._capture_risk_snapshot(),
                )
            elif service is not None:
                try:
                    self._dispatch_execution(service, decision, symbol)
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
                self._log_decision_event(
                    "execution_skipped",
                    symbol=str(symbol) if symbol else None,
                    status="no_service",
                    metadata={"reason": "no_service"},
                )
                self._record_decision_audit_stage(
                    "execution_skipped",
                    symbol=symbol,
                    payload={"reason": "no_service"},
                    risk_snapshot=self._capture_risk_snapshot(),
                )
        elif decision.should_trade:
            self._log_decision_event(
                "execution_skipped",
                symbol=str(symbol) if symbol else None,
                status="risk_rejected",
                metadata={"reason": "risk_rejected"},
            )

    def run_cycle_once(self) -> None:
        """Execute a single auto-trading cycle synchronously.

        The helper is primarily used by lightweight schedulers and tests.  It
        reuses the same internal routine as the background worker, ensuring the
        full pipeline – data fetch, AIManager evaluation and
        DecisionOrchestrator integration – is exercised deterministically.
        """

        self._auto_trade_loop()

    def _resolve_risk_service(self) -> Any | None:
        risk_service = getattr(self, "risk_service", None)
        if risk_service is None:
            risk_service = getattr(self, "core_risk_engine", None)
        return risk_service

    def _invoke_risk_service(self, service: Any, decision: "RiskDecision") -> Any:
        if hasattr(service, "evaluate_decision"):
            return service.evaluate_decision(decision)
        if callable(service):  # pragma: no branch - simple delegation
            return service(decision)
        raise TypeError("Configured risk service is not callable")

    def _apply_risk_evaluation_limit_locked(
        self, limit: int | None
    ) -> int:
        history = self._risk_evaluations
        if limit is None or limit < 0 or not history:
            return 0
        if limit == 0:
            trimmed = len(history)
            history.clear()
            return trimmed
        overflow = len(history) - limit
        if overflow > 0:
            del history[:overflow]
            return overflow
        return 0

    def _prune_risk_evaluations_locked(
        self, *, reference_time: float | None = None
    ) -> int:
        history = self._risk_evaluations
        ttl = self._risk_evaluations_ttl_s
        if ttl is None or ttl <= 0 or not history:
            return 0
        cutoff = (reference_time if reference_time is not None else time.time()) - ttl
        trimmed = 0
        retained: list[dict[str, Any]] = []
        for entry in history:
            try:
                timestamp_value = float(entry.get("timestamp", cutoff + ttl))
            except (TypeError, ValueError):
                timestamp_value = cutoff + ttl
            if timestamp_value >= cutoff:
                retained.append(entry)
            else:
                trimmed += 1
        if trimmed:
            history[:] = retained
        return trimmed

    def _log_risk_history_trimmed(
        self,
        *,
        context: str,
        trimmed: int,
        ttl: float | None,
        history: int,
    ) -> None:
        if trimmed <= 0:
            return
        self._log(
            "Risk evaluation history trimmed",
            level=logging.DEBUG,
            context=context,
            trimmed=trimmed,
            ttl=ttl,
            remaining=history,
        )

    def _store_risk_evaluation_entry(
        self,
        entry: dict[str, Any],
        *,
        reference_time: float | None = None,
    ) -> tuple[int, int, int | None, float | None, int]:
        timestamp = entry.get("timestamp")
        try:
            timestamp_value = float(timestamp) if timestamp is not None else float(time.time())
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            timestamp_value = float(time.time())

        with self._lock:
            history = self._risk_evaluations
            if history:
                positions = [float(item.get("timestamp", 0.0)) for item in history]
                index = bisect_right(positions, timestamp_value)
            else:
                index = 0
            history.insert(index, entry)
            limit_snapshot = self._risk_evaluations_limit
            trimmed_by_limit = self._apply_risk_evaluation_limit_locked(limit_snapshot)
            trimmed_by_ttl = self._prune_risk_evaluations_locked(
                reference_time=reference_time if reference_time is not None else timestamp_value
            )
            ttl_snapshot = self._risk_evaluations_ttl_s
            history_size = len(history)

        return trimmed_by_limit, trimmed_by_ttl, limit_snapshot, ttl_snapshot, history_size

    def _build_risk_evaluation_event_payload(
        self,
        entry: Mapping[str, Any],
        *,
        trimmed_by_limit: int,
        trimmed_by_ttl: int,
        history_size: int,
        limit_snapshot: int | None,
        ttl_snapshot: float | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = copy.deepcopy(dict(entry))
        payload.setdefault("normalized", payload.get("approved"))
        payload["history_trimmed_by_limit"] = trimmed_by_limit
        payload["history_trimmed_by_ttl"] = trimmed_by_ttl
        payload["history_size"] = history_size
        payload["history_limit"] = limit_snapshot
        payload["history_ttl"] = ttl_snapshot
        return payload

    def _emit_risk_evaluation_event(self, payload: Mapping[str, Any]) -> None:
        emitter_emit = getattr(self.emitter, "emit", None)
        if not callable(emitter_emit):
            return
        try:
            emitter_emit("auto_trader.risk_evaluation", **dict(payload))
        except Exception:  # pragma: no cover - emission should not break trading
            LOGGER.debug("Risk evaluation emission failed", exc_info=True)

    def _notify_risk_evaluation_listeners(self, payload: Mapping[str, Any]) -> None:
        with self._lock:
            listeners = tuple(self._risk_evaluation_listeners)
        if not listeners:
            return
        for listener in listeners:
            try:
                listener(copy.deepcopy(dict(payload)))
            except Exception:  # pragma: no cover - listeners should not break trading
                LOGGER.debug("Risk evaluation listener failed", exc_info=True)

    def add_risk_evaluation_listener(
        self, listener: Callable[[Mapping[str, Any]], None]
    ) -> None:
        """Rejestruje obserwatora nowych wpisów historii ocen ryzyka."""

        if not callable(listener):
            raise TypeError("listener musi być wywoływalny")
        with self._lock:
            self._risk_evaluation_listeners.add(listener)

    def remove_risk_evaluation_listener(
        self, listener: Callable[[Mapping[str, Any]], None]
    ) -> None:
        """Usuwa wcześniej zarejestrowanego obserwatora ocen ryzyka."""

        with self._lock:
            self._risk_evaluation_listeners.discard(listener)

    def _record_risk_evaluation(
        self,
        decision: "RiskDecision",
        *,
        approved: bool | None,
        normalized: bool | None,
        response: Any,
        service: Any,
        error: Exception | None,
    ) -> None:
        normalized_value = normalized if normalized is not None else approved
        active_decision_id = (
            self._normalize_decision_id(self._active_decision_id)
            or self._generate_decision_id()
        )
        entry: dict[str, Any] = {
            "timestamp": time.time(),
            "approved": approved,
            "normalized": normalized_value,
            "decision": decision.to_dict(),
        }
        entry["decision_id"] = active_decision_id
        feature_metadata = self._feature_column_metadata()
        if feature_metadata:
            metadata = copy.deepcopy(dict(entry.get("metadata", {})))
            for key, value in feature_metadata.items():
                metadata.setdefault(key, copy.deepcopy(value))
            if metadata:
                entry["metadata"] = metadata
        if service is not None:
            entry["service"] = type(service).__name__
        if error is not None:
            entry["error"] = repr(error)
        else:
            response_summary = self._summarize_risk_response(response)
            if response_summary is not None:
                entry["response"] = response_summary
        (
            trimmed_by_limit,
            trimmed_by_ttl,
            limit_snapshot,
            ttl_snapshot,
            history_size,
        ) = self._store_risk_evaluation_entry(
            entry,
            reference_time=entry["timestamp"],
        )
        self._log_risk_history_trimmed(
            context="get",
            trimmed=trimmed_by_ttl,
            ttl=ttl_snapshot,
            history=history_size,
        )
        payload = self._build_risk_evaluation_event_payload(
            entry,
            trimmed_by_limit=trimmed_by_limit,
            trimmed_by_ttl=trimmed_by_ttl,
            history_size=history_size,
            limit_snapshot=limit_snapshot,
            ttl_snapshot=ttl_snapshot,
        )
        self._emit_risk_evaluation_event(payload)
        self._notify_risk_evaluation_listeners(payload)

    def _resolve_risk_service(self) -> Any | None:
        risk_service = getattr(self, "risk_service", None)
        if risk_service is None:
            risk_service = getattr(self, "core_risk_engine", None)
        return risk_service

    def _invoke_risk_service(self, service: Any, decision: "RiskDecision") -> Any:
        if hasattr(service, "evaluate_decision"):
            return service.evaluate_decision(decision)
        if callable(service):  # pragma: no branch - simple delegation
            return service(decision)
        raise TypeError("Configured risk service is not callable")

    def _normalize_risk_approval(
        self,
        decision: "RiskDecision",
        service: Any,
        response: Any,
    ) -> tuple[bool | None, bool | None]:
        approval: bool | None = None
        normalized: bool | None = None

        risk_details = decision.details.setdefault("risk_service", {})
        risk_details["service"] = type(service).__name__
        summary = self._summarize_risk_response(response)
        if summary is not None:
            risk_details["response"] = summary

        candidate = response
        additional_context: Mapping[str, Any] | None = None
        if isinstance(response, tuple) and response:
            candidate = response[0]
            if len(response) > 1 and isinstance(response[1], Mapping):
                additional_context = dict(response[1])
        elif isinstance(response, list) and response:
            candidate = response[0]
        if additional_context:
            risk_details["context"] = additional_context

        approval = self._coerce_risk_approval(candidate)
        if approval is None:
            if isinstance(response, Mapping):
                for key in ("approved", "allow", "should_trade", "trade"):
                    if key in response:
                        approval = self._coerce_risk_approval(response[key])
                        if approval is not None:
                            break
            elif hasattr(response, "__dict__"):
                for key in ("approved", "allow", "should_trade", "trade"):
                    if hasattr(response, key):
                        approval = self._coerce_risk_approval(getattr(response, key))
                        if approval is not None:
                            break

        if approval is not None:
            normalized = approval

        return approval, normalized

    def _coerce_risk_approval(self, candidate: Any) -> bool | None:
        if isinstance(candidate, bool):
            return candidate
        if isinstance(candidate, enum.Enum):
            return self._coerce_risk_approval(candidate.value)
        if isinstance(candidate, (int, float)):
            if candidate >= 1:
                return True
            if candidate <= 0:
                return False
            return None
        if isinstance(candidate, str):
            value = candidate.strip().lower()
            if value in {"true", "yes", "y", "allow", "allowed", "approve", "approved", "accept", "accepted", "ok"}:
                return True
            if value in {"false", "no", "n", "deny", "denied", "block", "blocked", "reject", "rejected"}:
                return False
            return None
        return None

    def _summarize_risk_response(self, response: Any) -> dict[str, Any] | None:
        if response is None:
            return None

        summary: dict[str, Any] = {"type": type(response).__name__}
        if isinstance(response, str):
            value = response.strip()
            if len(value) > 120:
                value = value[:117] + "..."
            summary["value"] = value
        elif isinstance(response, (bool, int, float)):
            summary["value"] = response
        elif isinstance(response, Mapping):
            summary["size"] = len(response)
            summary["keys"] = sorted(str(key) for key in response.keys())
        elif isinstance(response, (list, tuple, set, frozenset)):
            sequence = list(response)
            summary["size"] = len(sequence)
            if sequence:
                preview = sequence[:5]
                summary["preview"] = [repr(item) for item in preview]
        else:
            summary["repr"] = repr(response)
        return summary

        iterator: Iterable[dict[str, Any]]
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



    def get_grouped_decision_audit_entries(
        self,
        *,
        limit: int | None = None,
        reverse: bool = False,
        stage: str | Sequence[object] | None = None,
        symbol: str | Sequence[object] | None = None,
        mode: str | Sequence[object] | None = None,
        decision_id: str | Sequence[object] | None = None,
        since: Any = None,
        until: Any = None,
        has_risk_snapshot: bool | None = None,
        has_portfolio_snapshot: bool | None = None,
        timezone_hint: timezone | tzinfo | None = timezone.utc,
        include_unidentified: bool = False,
    ) -> Mapping[str | None, Sequence[Mapping[str, object]]]:
        """Return audit entries grouped by decision identifier."""

        log = getattr(self, "_decision_audit_log", None)
        if log is None:
            return {}
        return log.group_by_decision(
            limit=limit,
            reverse=reverse,
            stage=stage,
            symbol=symbol,
            mode=mode,
            decision_id=decision_id,
            since=since,
            until=until,
            has_risk_snapshot=has_risk_snapshot,
            has_portfolio_snapshot=has_portfolio_snapshot,
            timezone_hint=timezone_hint,
            include_unidentified=include_unidentified,
        )

    def get_decision_audit_trace(
        self,
        decision_id: Any,
        *,
        stage: str | Sequence[object] | None = None,
        symbol: str | Sequence[object] | None = None,
        mode: str | Sequence[object] | None = None,
        since: Any = None,
        until: Any = None,
        has_risk_snapshot: bool | None = None,
        has_portfolio_snapshot: bool | None = None,
        timezone_hint: timezone | tzinfo | None = timezone.utc,
        include_payload: bool = True,
        include_snapshots: bool = True,
        include_metadata: bool = True,
    ) -> Sequence[Mapping[str, object]]:
        """Return ordered audit entries for a specific decision identifier."""

        log = getattr(self, "_decision_audit_log", None)
        if log is None:
            return ()
        return log.trace_decision(
            decision_id,
            stage=stage,
            symbol=symbol,
            mode=mode,
            since=since,
            until=until,
            has_risk_snapshot=has_risk_snapshot,
            has_portfolio_snapshot=has_portfolio_snapshot,
            timezone_hint=timezone_hint,
            include_payload=include_payload,
            include_snapshots=include_snapshots,
            include_metadata=include_metadata,
        )

    def add_decision_audit_listener(
        self, listener: Callable[[DecisionAuditRecord], None]
    ) -> bool:
        log = getattr(self, "_decision_audit_log", None)
        if log is None:
            return False
        log.add_listener(listener)
        return True

    def remove_decision_audit_listener(
        self, listener: Callable[[DecisionAuditRecord], None]
    ) -> bool:
        log = getattr(self, "_decision_audit_log", None)
        if log is None:
            return False
        return log.remove_listener(listener)

    def get_decision_audit_summary(
        self,
        *,
        limit: int | None = None,
        reverse: bool = False,
        stage: str | Sequence[object] | None = None,
        symbol: str | Sequence[object] | None = None,
        mode: str | Sequence[object] | None = None,
        decision_id: str | Sequence[object] | None = None,
        since: Any = None,
        until: Any = None,
        has_risk_snapshot: bool | None = None,
        has_portfolio_snapshot: bool | None = None,
    ) -> Mapping[str, object]:
        log = getattr(self, "_decision_audit_log", None)
        if log is None:
            return {
                "count": 0,
                "stages": {},
                "symbols": {},
                "modes": {},
                "decision_ids": {},
                "unique_decision_ids": 0,
                "with_risk_snapshot": 0,
                "with_portfolio_snapshot": 0,
            }
        return log.summarize(
            limit=limit,
            reverse=reverse,
            stage=stage,
            symbol=symbol,
            mode=mode,
            decision_id=decision_id,
            since=since,
            until=until,
            has_risk_snapshot=has_risk_snapshot,
            has_portfolio_snapshot=has_portfolio_snapshot,
        )

    def get_decision_audit_dataframe(
        self,
        *,
        limit: int | None = 20,
        reverse: bool = False,
        stage: str | Sequence[object] | None = None,
        symbol: str | Sequence[object] | None = None,
        mode: str | Sequence[object] | None = None,
        decision_id: str | Sequence[object] | None = None,
        since: Any = None,
        until: Any = None,
        has_risk_snapshot: bool | None = None,
        has_portfolio_snapshot: bool | None = None,
        timezone_hint: timezone | tzinfo | None = timezone.utc,
    ) -> Any:
        """Return a ``pandas.DataFrame`` representation of the audit log."""

        log = getattr(self, "_decision_audit_log", None)
        if log is None:
            try:
                import pandas as pd
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
                raise RuntimeError(
                    "pandas is required to export the decision audit log as a DataFrame",
                ) from exc

            empty_frame = pd.DataFrame(
                {
                    "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
                    "stage": pd.Series(dtype="object"),
                    "symbol": pd.Series(dtype="object"),
                    "mode": pd.Series(dtype="object"),
                    "decision_id": pd.Series(dtype="object"),
                    "payload": pd.Series(dtype="object"),
                    "risk_snapshot": pd.Series(dtype="object"),
                    "portfolio_snapshot": pd.Series(dtype="object"),
                    "metadata": pd.Series(dtype="object"),
                }
            )
            empty_frame.attrs["audit_filters"] = {
                "limit": limit,
                "reverse": reverse,
                "stage": stage,
                "symbol": symbol,
                "mode": mode,
                "decision_id": decision_id,
                "since": since,
                "until": until,
                "has_risk_snapshot": has_risk_snapshot,
                "has_portfolio_snapshot": has_portfolio_snapshot,
                "timezone_hint": timezone_hint,
            }
            return empty_frame

        return log.to_dataframe(
            limit=limit,
            reverse=reverse,
            stage=stage,
            symbol=symbol,
            mode=mode,
            decision_id=decision_id,
            since=since,
            until=until,
            has_risk_snapshot=has_risk_snapshot,
            has_portfolio_snapshot=has_portfolio_snapshot,
            timezone_hint=timezone_hint,
        )


    def trim_decision_audit_log(
        self,
        *,
        before: Any | None = None,
        max_age_s: float | int | None = None,
    ) -> int:
        log = getattr(self, "_decision_audit_log", None)
        if log is None:
            return 0
        return log.trim(before=before, max_age_s=max_age_s)

    def export_decision_audit_log(
        self,
        *,
        limit: int | None = None,
        reverse: bool = False,
        stage: str | Iterable[object] | None = None,
        symbol: str | Iterable[object] | None = None,
        mode: str | Iterable[object] | None = None,
        decision_id: str | Iterable[object] | None = None,
        since: Any = None,
        until: Any = None,
        has_risk_snapshot: bool | None = None,
        has_portfolio_snapshot: bool | None = None,
        timezone_hint: timezone | tzinfo | None = timezone.utc,
    ) -> Mapping[str, object]:
        log = getattr(self, "_decision_audit_log", None)
        if log is None:
            def _normalize_filter(
                value: str | Iterable[object] | None,
            ) -> tuple[str, ...] | None:
                normalized = DecisionAuditLog._normalize_token_filter(value)
                if normalized is None:
                    return None
                return tuple(sorted(normalized))

            return {
                "version": 1,
                "entries": [],
                "retention": {
                    "max_entries": 0,
                    "max_age_s": None,
                },
                "filters": {
                    "limit": limit,
                    "reverse": reverse,
                    "stage": _normalize_filter(stage),
                    "symbol": _normalize_filter(symbol),
                    "mode": _normalize_filter(mode),
                    "decision_id": _normalize_filter(decision_id),
                    "since": since,
                    "until": until,
                    "has_risk_snapshot": has_risk_snapshot,
                    "has_portfolio_snapshot": has_portfolio_snapshot,
                    "timezone_hint": timezone_hint.tzname(None)
                    if isinstance(timezone_hint, (timezone, tzinfo))
                    else timezone_hint,
                },
            }
        return log.export(
            limit=limit,
            reverse=reverse,
            stage=stage,
            symbol=symbol,
            mode=mode,
            decision_id=decision_id,
            since=since,
            until=until,
            has_risk_snapshot=has_risk_snapshot,
            has_portfolio_snapshot=has_portfolio_snapshot,
            timezone_hint=timezone_hint,
        )

    def load_decision_audit_log(
        self,
        payload: Mapping[str, object],
        *,
        merge: bool = False,
        notify_listeners: bool = False,
    ) -> int:
        log = getattr(self, "_decision_audit_log", None)
        if log is None:
            return 0
        return log.load(payload, merge=merge, notify_listeners=notify_listeners)









    def _collect_guardrail_events(
        self,
        *,
        include_errors: bool,
        approved_filter: set[bool | None] | None,
        normalized_filter: set[bool | None] | None,
        service_filter: set[str] | None,
        decision_state_filter: set[str] | None,
        decision_reason_filter: set[str] | None,
        decision_mode_filter: set[str] | None,
        decision_id_filter: set[str] | None = None,
        since_ts: float | None,
        until_ts: float | None,
        reason_filter: set[str] | None,
        trigger_filter: set[str] | None,
        trigger_label_filter: set[str] | None,
        trigger_comparator_filter: set[str] | None,
        trigger_unit_filter: set[str] | None,
        trigger_threshold_filter: tuple[set[float], bool] | None,
        trigger_threshold_min: float | None,
        trigger_threshold_max: float | None,
        trigger_value_filter: tuple[set[float], bool] | None,
        trigger_value_min: float | None,
        trigger_value_max: float | None,
    ) -> tuple[
        list[tuple[dict[str, Any], tuple[str, ...], tuple[dict[str, Any], ...]]],
        int,
        float | None,
        int,
        list[dict[str, Any]],
    ]:
        (
            filtered_records,
            trimmed_by_ttl,
            ttl_snapshot,
            history_size,
        ) = self._collect_filtered_risk_evaluations(
            include_errors=include_errors,
            approved_filter=approved_filter,
            normalized_filter=normalized_filter,
            service_filter=service_filter,
            since_ts=since_ts,
            until_ts=until_ts,
            state_filter=decision_state_filter,
            reason_filter=decision_reason_filter,
            mode_filter=decision_mode_filter,
            decision_id_filter=decision_id_filter,
        )

        reason_filters_active = reason_filter is not None
        trigger_filters_active = any(
            filter_value is not None
            for filter_value in (
                trigger_filter,
                trigger_label_filter,
                trigger_comparator_filter,
                trigger_unit_filter,
                trigger_threshold_filter,
                trigger_value_filter,
            )
        ) or any(
            bound is not None
            for bound in (
                trigger_threshold_min,
                trigger_threshold_max,
                trigger_value_min,
                trigger_value_max,
            )
        )

        guardrail_records: list[
            tuple[dict[str, Any], tuple[str, ...], tuple[dict[str, Any], ...]]
        ] = []

        for entry in filtered_records:
            reasons, triggers, trigger_tokens = self._extract_guardrail_dimensions(entry)
            if not reasons and not triggers:
                continue

            if reason_filters_active and not any(
                reason in reason_filter  # type: ignore[arg-type]
                for reason in reasons
            ):
                continue

            if trigger_filters_active and not self._guardrail_trigger_matches(
                trigger_tokens,
                trigger_filter=trigger_filter,
                trigger_label_filter=trigger_label_filter,
                trigger_comparator_filter=trigger_comparator_filter,
                trigger_unit_filter=trigger_unit_filter,
                trigger_threshold_filter=trigger_threshold_filter,
                trigger_threshold_min=trigger_threshold_min,
                trigger_threshold_max=trigger_threshold_max,
                trigger_value_filter=trigger_value_filter,
                trigger_value_min=trigger_value_min,
                trigger_value_max=trigger_value_max,
            ):
                continue

            guardrail_records.append((entry, reasons, triggers))

        return (
            guardrail_records,
            trimmed_by_ttl,
            ttl_snapshot,
            history_size,
            filtered_records,
        )

    def _extract_guardrail_dimensions(
        self, entry: Mapping[str, Any]
    ) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...], list[dict[str, Any]]]:
        decision_payload = entry.get("decision") if isinstance(entry, Mapping) else None
        details: Mapping[str, Any] | None = None
        if isinstance(decision_payload, Mapping):
            details = decision_payload.get("details")
        if not isinstance(details, Mapping):
            details = {}

        raw_reasons = details.get("guardrail_reasons") or ()
        reasons: list[str] = []
        for reason in raw_reasons:
            if reason is None:
                continue
            token = str(reason).strip()
            if token:
                reasons.append(token)

        raw_triggers = details.get("guardrail_triggers") or ()
        export_triggers: list[dict[str, Any]] = []
        trigger_tokens: list[dict[str, Any]] = []

        for raw_trigger in raw_triggers:
            if hasattr(raw_trigger, "to_dict") and callable(raw_trigger.to_dict):
                trigger_payload = raw_trigger.to_dict()
            elif is_dataclass(raw_trigger):
                trigger_payload = asdict(raw_trigger)
            elif isinstance(raw_trigger, Mapping):
                trigger_payload = dict(raw_trigger)
            else:
                trigger_payload = {"name": raw_trigger}

            name_raw = trigger_payload.get("name")
            label_raw = trigger_payload.get("label")
            comparator_raw = trigger_payload.get("comparator")
            unit_raw = trigger_payload.get("unit")
            threshold_raw = trigger_payload.get("threshold")
            value_raw = trigger_payload.get("value")

            export_entry: dict[str, Any] = {}
            if name_raw is not None:
                export_entry["name"] = str(name_raw)
            export_entry["label"] = label_raw if label_raw is not None else None
            export_entry["comparator"] = (
                comparator_raw if comparator_raw is not None else None
            )
            if threshold_raw is not None:
                export_entry["threshold"] = threshold_raw
            if unit_raw is not None:
                export_entry["unit"] = unit_raw
            if value_raw is not None:
                export_entry["value"] = value_raw
            export_triggers.append(export_entry)

            normalized_threshold = self._coerce_float(threshold_raw)
            normalized_value = self._coerce_float(value_raw)
            trigger_tokens.append(
                {
                    "name": (
                        str(name_raw).strip()
                        if name_raw is not None and str(name_raw).strip()
                        else _UNKNOWN_SERVICE
                    ),
                    "label": (
                        str(label_raw).strip()
                        if label_raw is not None and str(label_raw).strip()
                        else _MISSING_GUARDRAIL_LABEL
                    ),
                    "comparator": (
                        str(comparator_raw).strip()
                        if comparator_raw is not None and str(comparator_raw).strip()
                        else _MISSING_GUARDRAIL_COMPARATOR
                    ),
                    "unit": (
                        str(unit_raw).strip()
                        if unit_raw is not None and str(unit_raw).strip()
                        else _MISSING_GUARDRAIL_UNIT
                    ),
                    "threshold": normalized_threshold,
                    "value": normalized_value,
                }
            )

        return tuple(reasons), tuple(export_triggers), trigger_tokens

    def _guardrail_trigger_matches(
        self,
        trigger_tokens: Sequence[Mapping[str, Any]],
        *,
        trigger_filter: set[str] | None,
        trigger_label_filter: set[str] | None,
        trigger_comparator_filter: set[str] | None,
        trigger_unit_filter: set[str] | None,
        trigger_threshold_filter: tuple[set[float], bool] | None,
        trigger_threshold_min: float | None,
        trigger_threshold_max: float | None,
        trigger_value_filter: tuple[set[float], bool] | None,
        trigger_value_min: float | None,
        trigger_value_max: float | None,
    ) -> bool:
        if not trigger_tokens:
            return False

        for trigger in trigger_tokens:
            name_token = str(trigger.get("name", _UNKNOWN_SERVICE))
            label_token = str(trigger.get("label", _MISSING_GUARDRAIL_LABEL))
            comparator_token = str(
                trigger.get("comparator", _MISSING_GUARDRAIL_COMPARATOR)
            )
            unit_token = str(trigger.get("unit", _MISSING_GUARDRAIL_UNIT))
            threshold_value = trigger.get("threshold")
            value_value = trigger.get("value")

            if trigger_filter is not None and name_token not in trigger_filter:
                continue
            if (
                trigger_label_filter is not None
                and label_token not in trigger_label_filter
            ):
                continue
            if (
                trigger_comparator_filter is not None
                and comparator_token not in trigger_comparator_filter
            ):
                continue
            if trigger_unit_filter is not None and unit_token not in trigger_unit_filter:
                continue

            if trigger_threshold_filter is not None:
                value_set, include_missing = trigger_threshold_filter
                if threshold_value is None:
                    if not include_missing:
                        continue
                elif threshold_value not in value_set:
                    continue

            if (
                trigger_threshold_min is not None
                and (
                    threshold_value is None
                    or threshold_value < trigger_threshold_min
                )
            ):
                continue
            if (
                trigger_threshold_max is not None
                and (
                    threshold_value is None
                    or threshold_value > trigger_threshold_max
                )
            ):
                continue

            if trigger_value_filter is not None:
                value_set, include_missing = trigger_value_filter
                if value_value is None:
                    if not include_missing:
                        continue
                elif value_value not in value_set:
                    continue

            if (
                trigger_value_min is not None
                and (value_value is None or value_value < trigger_value_min)
            ):
                continue
            if (
                trigger_value_max is not None
                and (value_value is None or value_value > trigger_value_max)
            ):
                continue

            return True

        return False

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
        state_filter: set[str] | None,
        reason_filter: set[str] | None,
        mode_filter: set[str] | None,
        decision_id_filter: set[str] | None = None,
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

            timestamp_value = self._normalize_time_bound(entry.get("timestamp"))
            if since_ts is not None and (timestamp_value is None or timestamp_value < since_ts):
                continue
            if until_ts is not None and (timestamp_value is None or timestamp_value > until_ts):
                continue

            decision_id_value = entry.get("decision_id")
            decision_id_token = (
                str(decision_id_value)
                if decision_id_value is not None
                else _MISSING_DECISION_ID
            )
            if decision_id_filter is not None and decision_id_token not in decision_id_filter:
                continue

            decision_payload = entry.get("decision")
            decision_state_token = _MISSING_DECISION_STATE
            decision_reason_token = _MISSING_DECISION_REASON
            decision_mode_token = _MISSING_DECISION_MODE
            if isinstance(decision_payload, Mapping):
                state_value = decision_payload.get("state")
                if state_value is not None:
                    decision_state_token = str(state_value)
                reason_value = decision_payload.get("reason")
                if reason_value is not None:
                    decision_reason_token = str(reason_value)
                mode_value = decision_payload.get("mode")
                if mode_value is not None:
                    decision_mode_token = str(mode_value)

            if state_filter is not None and decision_state_token not in state_filter:
                continue
            if reason_filter is not None and decision_reason_token not in reason_filter:
                continue
            if mode_filter is not None and decision_mode_token not in mode_filter:
                continue

            filtered.append(copy.deepcopy(dict(entry)))

        return filtered



    def _resolve_risk_evaluation_filters(
        self,
        *,
        approved: bool | None | Iterable[bool | None] | object,
        normalized: bool | None | Iterable[bool | None] | object,
        service: str | None | Iterable[str | None] | object,
        decision_state: str | Iterable[str | None] | object,
        decision_reason: str | Iterable[str | None] | object,
        decision_mode: str | Iterable[str | None] | object,
        decision_id: str | Iterable[str | None] | object,
        since: Any,
        until: Any,
        decision_fields: Iterable[Any] | Any | None,
    ) -> tuple[
        set[bool | None] | None,
        set[bool | None] | None,
        set[str] | None,
        set[str] | None,
        set[str] | None,
        set[str] | None,
        set[str] | None,
        list[Any] | None,
        float | None,
        float | None,
    ]:
        approved_filter = self._prepare_bool_filter(approved)
        normalized_filter = self._prepare_bool_filter(normalized)
        service_filter = self._prepare_service_filter(service)
        decision_state_filter = self._prepare_decision_filter(
            decision_state,
            missing_token=_MISSING_DECISION_STATE,
        )
        decision_reason_filter = self._prepare_decision_filter(
            decision_reason,
            missing_token=_MISSING_DECISION_REASON,
        )
        decision_mode_filter = self._prepare_decision_filter(
            decision_mode,
            missing_token=_MISSING_DECISION_MODE,
        )
        decision_id_filter = self._prepare_decision_filter(
            decision_id,
            missing_token=_MISSING_DECISION_ID,
        )
        normalized_decision_fields = self._normalize_decision_fields(decision_fields)
        since_ts = self._normalize_time_bound(since)
        until_ts = self._normalize_time_bound(until)
        return (
            approved_filter,
            normalized_filter,
            service_filter,
            decision_state_filter,
            decision_reason_filter,
            decision_mode_filter,
            decision_id_filter,
            normalized_decision_fields,
            since_ts,
            until_ts,
        )

    def _collect_filtered_risk_evaluations(
        self,
        *,
        include_errors: bool,
        approved_filter: set[bool | None] | None,
        normalized_filter: set[bool | None] | None,
        service_filter: set[str] | None,
        since_ts: float | None,
        until_ts: float | None,
        state_filter: set[str] | None,
        reason_filter: set[str] | None,
        mode_filter: set[str] | None,
        decision_id_filter: set[str] | None = None,
    ) -> tuple[list[dict[str, Any]], int, float | None, int]:
        with self._lock:
            trimmed_by_ttl = self._prune_risk_evaluations_locked()
            history_snapshot = list(self._risk_evaluations)
            ttl_snapshot = self._risk_evaluations_ttl_s
            history_size = len(self._risk_evaluations)

        filtered_records = self._apply_risk_evaluation_filters(
            history_snapshot,
            include_errors=include_errors,
            approved_filter=approved_filter,
            normalized_filter=normalized_filter,
            service_filter=service_filter,
            since_ts=since_ts,
            until_ts=until_ts,
            state_filter=state_filter,
            reason_filter=reason_filter,
            mode_filter=mode_filter,
            decision_id_filter=decision_id_filter,
        )
        return filtered_records, trimmed_by_ttl, ttl_snapshot, history_size

    def _normalize_history_export_limit(
        self, limit: Any
    ) -> int | None:
        if limit is None or limit is _NO_FILTER:
            return None
        if isinstance(limit, bool):
            raise TypeError("limit must be an integer or None")
        try:
            normalized = int(limit)
        except (TypeError, ValueError) as exc:
            raise TypeError("limit must be an integer or None") from exc
        if normalized < 0:
            return None
        return normalized

    def _resolve_guardrail_event_filters(
        self,
        *,
        approved: bool | None | Iterable[bool | None] | object,
        normalized: bool | None | Iterable[bool | None] | object,
        service: str | None | Iterable[str | None] | object,
        decision_state: str | Iterable[str | None] | object,
        decision_reason: str | Iterable[str | None] | object,
        decision_mode: str | Iterable[str | None] | object,
        decision_id: str | Iterable[str | None] | object,
        reason: str | Iterable[str] | object,
        trigger: str | Iterable[str] | object,
        trigger_label: str | Iterable[str | None] | object,
        trigger_comparator: str | Iterable[str | None] | object,
        trigger_unit: str | Iterable[str | None] | object,
        trigger_threshold: float | None | Iterable[float | None] | object,
        trigger_threshold_min: Any,
        trigger_threshold_max: Any,
        trigger_value: float | None | Iterable[float | None] | object,
        trigger_value_min: Any,
        trigger_value_max: Any,
        since: Any,
        until: Any,
    ) -> tuple[
        set[bool | None] | None,
        set[bool | None] | None,
        set[str] | None,
        set[str] | None,
        set[str] | None,
        set[str] | None,
        set[str] | None,
        set[str] | None,
        set[str] | None,
        set[str] | None,
        set[str] | None,
        set[str] | None,
        tuple[set[float], bool] | None,
        float | None,
        float | None,
        tuple[set[float], bool] | None,
        float | None,
        float | None,
        float | None,
        float | None,
    ]:
        approved_filter = self._prepare_bool_filter(approved)
        normalized_filter = self._prepare_bool_filter(normalized)
        service_filter = self._prepare_service_filter(service)
        decision_state_filter = self._prepare_decision_filter(
            decision_state,
            missing_token=_MISSING_DECISION_STATE,
        )
        decision_reason_filter = self._prepare_decision_filter(
            decision_reason,
            missing_token=_MISSING_DECISION_REASON,
        )
        decision_mode_filter = self._prepare_decision_filter(
            decision_mode,
            missing_token=_MISSING_DECISION_MODE,
        )
        decision_id_filter = self._prepare_decision_filter(
            decision_id,
            missing_token=_MISSING_DECISION_ID,
        )
        reason_filter = self._prepare_string_filter(reason)
        trigger_filter = self._prepare_string_filter(trigger)
        trigger_label_filter = self._prepare_guardrail_filter(
            trigger_label,
            missing_token=_MISSING_GUARDRAIL_LABEL,
        )
        trigger_comparator_filter = self._prepare_guardrail_filter(
            trigger_comparator,
            missing_token=_MISSING_GUARDRAIL_COMPARATOR,
        )
        trigger_unit_filter = self._prepare_guardrail_filter(
            trigger_unit,
            missing_token=_MISSING_GUARDRAIL_UNIT,
        )
        trigger_threshold_filter = self._prepare_guardrail_numeric_filter(
            trigger_threshold
        )
        trigger_value_filter = self._prepare_guardrail_numeric_filter(trigger_value)
        trigger_threshold_min_value = (
            self._coerce_float(trigger_threshold_min)
            if trigger_threshold_min is not None
            else None
        )
        trigger_threshold_max_value = (
            self._coerce_float(trigger_threshold_max)
            if trigger_threshold_max is not None
            else None
        )
        trigger_value_min_value = (
            self._coerce_float(trigger_value_min)
            if trigger_value_min is not None
            else None
        )
        trigger_value_max_value = (
            self._coerce_float(trigger_value_max)
            if trigger_value_max is not None
            else None
        )
        since_ts = self._normalize_time_bound(since)
        until_ts = self._normalize_time_bound(until)
        return (
            approved_filter,
            normalized_filter,
            service_filter,
            decision_state_filter,
            decision_reason_filter,
            decision_mode_filter,
            decision_id_filter,
            reason_filter,
            trigger_filter,
            trigger_label_filter,
            trigger_comparator_filter,
            trigger_unit_filter,
            trigger_threshold_filter,
            trigger_threshold_min_value,
            trigger_threshold_max_value,
            trigger_value_filter,
            trigger_value_min_value,
            trigger_value_max_value,
            since_ts,
            until_ts,
        )

    def get_risk_evaluations(
        self,
        *,
        approved: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        normalized: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        include_errors: bool = True,
        service: str | None | Iterable[str | None] | object = _NO_FILTER,
        decision_state: str | Iterable[str | None] | object = _NO_FILTER,
        decision_reason: str | Iterable[str | None] | object = _NO_FILTER,
        decision_mode: str | Iterable[str | None] | object = _NO_FILTER,
        decision_id: str | Iterable[str | None] | object = _NO_FILTER,
        reason: str | Iterable[str] | object = _NO_FILTER,
        trigger: str | Iterable[str] | object = _NO_FILTER,
        trigger_label: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_comparator: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_unit: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_threshold: float | None | Iterable[float | None] | object = _NO_FILTER,
        trigger_threshold_min: Any = None,
        trigger_threshold_max: Any = None,
        trigger_value: float | None | Iterable[float | None] | object = _NO_FILTER,
        trigger_value_min: Any = None,
        trigger_value_max: Any = None,
        since: Any = None,
        until: Any = None,
        limit: Any = None,
        reverse: bool = False,
    ) -> list[dict[str, Any]]:
        (
            approved_filter,
            normalized_filter,
            service_filter,
            decision_state_filter,
            decision_reason_filter,
            decision_mode_filter,
            decision_id_filter,
            _,
            since_ts,
            until_ts,
        ) = self._resolve_risk_evaluation_filters(
            approved=approved,
            normalized=normalized,
            service=service,
            decision_state=decision_state,
            decision_reason=decision_reason,
            decision_mode=decision_mode,
            decision_id=decision_id,
            since=since,
            until=until,
            decision_fields=None,
        )

        (
            filtered_records,
            trimmed_by_ttl,
            ttl_snapshot,
            history_size,
        ) = self._collect_filtered_risk_evaluations(
            include_errors=include_errors,
            approved_filter=approved_filter,
            normalized_filter=normalized_filter,
            service_filter=service_filter,
            since_ts=since_ts,
            until_ts=until_ts,
            state_filter=decision_state_filter,
            reason_filter=decision_reason_filter,
            mode_filter=decision_mode_filter,
            decision_id_filter=decision_id_filter,
        )

        self._log_risk_history_trimmed(
            context="get",
            trimmed=trimmed_by_ttl,
            ttl=ttl_snapshot,
            history=history_size,
        )

        records = list(reversed(filtered_records)) if reverse else list(filtered_records)
        normalized_limit = self._normalize_history_export_limit(limit)
        if normalized_limit is not None:
            if normalized_limit == 0:
                records = []
            elif reverse:
                records = records[:normalized_limit]
            else:
                records = records[-normalized_limit:]
        return records

    def clear_risk_evaluations(self) -> None:
        with self._lock:
            self._risk_evaluations.clear()

    def get_risk_evaluations(
        self,
        *,
        approved: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        normalized: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        include_errors: bool = True,
        service: str | None | Iterable[str | None] | object = _NO_FILTER,
        decision_state: str | Iterable[str | None] | object = _NO_FILTER,
        decision_reason: str | Iterable[str | None] | object = _NO_FILTER,
        decision_mode: str | Iterable[str | None] | object = _NO_FILTER,
        decision_id: str | Iterable[str | None] | object = _NO_FILTER,
        since: Any = None,
        until: Any = None,
        limit: int | None = None,
        reverse: bool = False,
    ) -> list[dict[str, Any]]:
        (
            approved_filter,
            normalized_filter,
            service_filter,
            decision_state_filter,
            decision_reason_filter,
            decision_mode_filter,
            decision_id_filter,
            _,
            since_ts,
            until_ts,
        ) = self._resolve_risk_evaluation_filters(
            approved=approved,
            normalized=normalized,
            service=service,
            decision_state=decision_state,
            decision_reason=decision_reason,
            decision_mode=decision_mode,
            decision_id=decision_id,
            since=since,
            until=until,
            decision_fields=None,
        )

        (
            filtered_records,
            trimmed_by_ttl,
            ttl_snapshot,
            history_size,
        ) = self._collect_filtered_risk_evaluations(
            include_errors=include_errors,
            approved_filter=approved_filter,
            normalized_filter=normalized_filter,
            service_filter=service_filter,
            since_ts=since_ts,
            until_ts=until_ts,
            state_filter=decision_state_filter,
            reason_filter=decision_reason_filter,
            mode_filter=decision_mode_filter,
            decision_id_filter=decision_id_filter,
        )

        self._log_risk_history_trimmed(
            context="get",
            trimmed=trimmed_by_ttl,
            ttl=ttl_snapshot,
            history=history_size,
        )

        ordered_records = list(reversed(filtered_records)) if reverse else list(filtered_records)

        if limit is not None:
            try:
                normalized_limit = int(limit)
            except (TypeError, ValueError):
                normalized_limit = None
            else:
                if normalized_limit <= 0:
                    ordered_records = []
                else:
                    ordered_records = ordered_records[:normalized_limit]

        return [copy.deepcopy(record) for record in ordered_records]


    def get_decision_audit_entries(
        self,
        limit: int = 20,
        **filters: Any,
    ) -> Sequence[Mapping[str, object]]:
        log = getattr(self, "_decision_audit_log", None)
        if log is None:
            return ()
        query: dict[str, Any] = dict(filters)
        query.setdefault("limit", limit)
        try:
            return log.query_dicts(**query)
        except AttributeError:
            return log.to_dicts(limit)

    def clear_decision_audit_log(self) -> None:
        log = getattr(self, "_decision_audit_log", None)
        if log is not None:
            log.clear()



    def summarize_risk_evaluations(
        self,
        *,
        approved: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        normalized: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        include_errors: bool = True,
        service: str | None | Iterable[str | None] | object = _NO_FILTER,
        decision_state: str | Iterable[str | None] | object = _NO_FILTER,
        decision_reason: str | Iterable[str | None] | object = _NO_FILTER,
        decision_mode: str | Iterable[str | None] | object = _NO_FILTER,
        decision_id: str | Iterable[str | None] | object = _NO_FILTER,
        reason: str | Iterable[str] | object = _NO_FILTER,
        trigger: str | Iterable[str] | object = _NO_FILTER,
        trigger_label: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_comparator: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_unit: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_threshold: float | None | Iterable[float | None] | object = _NO_FILTER,
        trigger_threshold_min: Any = None,
        trigger_threshold_max: Any = None,
        trigger_value: float | None | Iterable[float | None] | object = _NO_FILTER,
        trigger_value_min: Any = None,
        trigger_value_max: Any = None,
        since: Any = None,
        until: Any = None,
    ) -> dict[str, Any]:
        approved_filter = self._prepare_bool_filter(approved)
        normalized_filter = self._prepare_bool_filter(normalized)
        service_filter = self._prepare_service_filter(service)
        decision_state_filter = self._prepare_decision_filter(
            decision_state,
            missing_token=_MISSING_DECISION_STATE,
        )
        decision_reason_filter = self._prepare_decision_filter(
            decision_reason,
            missing_token=_MISSING_DECISION_REASON,
        )
        decision_mode_filter = self._prepare_decision_filter(
            decision_mode,
            missing_token=_MISSING_DECISION_MODE,
        )
        decision_id_filter = self._prepare_decision_filter(
            decision_id,
            missing_token=_MISSING_DECISION_ID,
        )
        since_ts = self._normalize_time_bound(since)
        until_ts = self._normalize_time_bound(until)
        (
            filtered_records,
            trimmed_by_ttl,
            ttl_snapshot,
            history_size,
        ) = self._collect_filtered_risk_evaluations(
            include_errors=include_errors,
            approved_filter=approved_filter,
            normalized_filter=normalized_filter,
            service_filter=service_filter,
            since_ts=since_ts,
            until_ts=until_ts,
            state_filter=decision_state_filter,
            reason_filter=decision_reason_filter,
            mode_filter=decision_mode_filter,
            decision_id_filter=decision_id_filter,
        )
        self._log_risk_history_trimmed(
            context="summarize",
            trimmed=trimmed_by_ttl,
            ttl=ttl_snapshot,
            history=history_size,
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

    def summarize_risk_guardrails(
        self,
        *,
        approved: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        normalized: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        include_errors: bool = True,
        service: str | None | Iterable[str | None] | object = _NO_FILTER,
        decision_state: str | Iterable[str | None] | object = _NO_FILTER,
        decision_reason: str | Iterable[str | None] | object = _NO_FILTER,
        decision_mode: str | Iterable[str | None] | object = _NO_FILTER,
        decision_id: str | Iterable[str | None] | object = _NO_FILTER,
        reason: str | Iterable[str] | object = _NO_FILTER,
        trigger: str | Iterable[str] | object = _NO_FILTER,
        trigger_label: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_comparator: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_unit: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_threshold: float | None | Iterable[float | None] | object = _NO_FILTER,
        trigger_threshold_min: Any = None,
        trigger_threshold_max: Any = None,
        trigger_value: float | None | Iterable[float | None] | object = _NO_FILTER,
        trigger_value_min: Any = None,
        trigger_value_max: Any = None,
        since: Any = None,
        until: Any = None,
    ) -> dict[str, Any]:
        (
            approved_filter,
            normalized_filter,
            service_filter,
            decision_state_filter,
            decision_reason_filter,
            decision_mode_filter,
            decision_id_filter,
            reason_filter,
            trigger_filter,
            trigger_label_filter,
            trigger_comparator_filter,
            trigger_unit_filter,
            trigger_threshold_filter,
            trigger_threshold_min_value,
            trigger_threshold_max_value,
            trigger_value_filter,
            trigger_value_min_value,
            trigger_value_max_value,
            since_ts,
            until_ts,
        ) = self._resolve_guardrail_event_filters(
            approved=approved,
            normalized=normalized,
            service=service,
            decision_state=decision_state,
            decision_reason=decision_reason,
            decision_mode=decision_mode,
            decision_id=decision_id,
            reason=reason,
            trigger=trigger,
            trigger_label=trigger_label,
            trigger_comparator=trigger_comparator,
            trigger_unit=trigger_unit,
            trigger_threshold=trigger_threshold,
            trigger_threshold_min=trigger_threshold_min,
            trigger_threshold_max=trigger_threshold_max,
            trigger_value=trigger_value,
            trigger_value_min=trigger_value_min,
            trigger_value_max=trigger_value_max,
            since=since,
            until=until,
        )

        (
            guardrail_records,
            trimmed_by_ttl,
            ttl_snapshot,
            history_size,
            filtered_records,
        ) = self._collect_guardrail_events(
            include_errors=include_errors,
            approved_filter=approved_filter,
            normalized_filter=normalized_filter,
            service_filter=service_filter,
            decision_state_filter=decision_state_filter,
            decision_reason_filter=decision_reason_filter,
            decision_mode_filter=decision_mode_filter,
            decision_id_filter=decision_id_filter,
            since_ts=since_ts,
            until_ts=until_ts,
            reason_filter=reason_filter,
            trigger_filter=trigger_filter,
            trigger_label_filter=trigger_label_filter,
            trigger_comparator_filter=trigger_comparator_filter,
            trigger_unit_filter=trigger_unit_filter,
            trigger_threshold_filter=trigger_threshold_filter,
            trigger_threshold_min=trigger_threshold_min_value,
            trigger_threshold_max=trigger_threshold_max_value,
            trigger_value_filter=trigger_value_filter,
            trigger_value_min=trigger_value_min_value,
            trigger_value_max=trigger_value_max_value,
        )
        self._log_risk_history_trimmed(
            context="guardrails",
            trimmed=trimmed_by_ttl,
            ttl=ttl_snapshot,
            history=history_size,
        )

        base_total = len(filtered_records)
        apply_guardrail_filters = any(
            filter_set is not None
            for filter_set in (
                reason_filter,
                trigger_filter,
                trigger_label_filter,
                trigger_comparator_filter,
                trigger_unit_filter,
                trigger_threshold_filter,
                trigger_value_filter,
            )
        ) or any(
            bound is not None
            for bound in (
                trigger_threshold_min_value,
                trigger_threshold_max_value,
                trigger_value_min_value,
                trigger_value_max_value,
            )
        )
        total = len(guardrail_records) if apply_guardrail_filters else base_total
        summary: dict[str, Any] = {
            "total": total,
            "guardrail_events": 0,
            "reasons": {},
            "triggers": {},
            "services": {},
        }
        if total == 0:
            return summary

        reason_counts: Counter[str] = Counter()
        service_buckets: dict[str, dict[str, Any]] = {}
        trigger_buckets: dict[str, dict[str, Any]] = {}

        if apply_guardrail_filters:
            service_source = [entry for entry, _, _ in guardrail_records]
        else:
            service_source = filtered_records

        for entry in service_source:
            service_key = entry.get("service") or _UNKNOWN_SERVICE
            service_name = str(service_key)
            service_buckets.setdefault(
                service_name,
                {
                    "total": 0,
                    "guardrail_events": 0,
                    "reasons": Counter(),
                    "triggers": Counter(),
                },
            )
            service_buckets[service_name]["total"] += 1

        for entry, reasons, triggers in guardrail_records:
            service_key = entry.get("service") or _UNKNOWN_SERVICE
            service_name = str(service_key)
            service_bucket = service_buckets.setdefault(
                service_name,
                {
                    "total": 0,
                    "guardrail_events": 0,
                    "reasons": Counter(),
                    "triggers": Counter(),
                },
            )

            for reason_value in reasons:
                reason_counts[reason_value] += 1
                service_bucket["reasons"][reason_value] += 1

            for trigger_entry in triggers:
                trigger_name_raw = trigger_entry.get("name")
                trigger_name = (
                    str(trigger_name_raw)
                    if trigger_name_raw is not None
                    else "<unknown>"
                )
                trigger_bucket = trigger_buckets.setdefault(
                    trigger_name,
                    {
                        "count": 0,
                        "label": None,
                        "comparator": None,
                        "unit": None,
                        "threshold": None,
                        "value_min": None,
                        "value_max": None,
                        "value_last": None,
                        "_value_sum": 0.0,
                        "_value_count": 0,
                        "services": Counter(),
                    },
                )
                trigger_bucket["count"] += 1
                trigger_label = trigger_entry.get("label")
                if trigger_label and trigger_bucket["label"] is None:
                    trigger_bucket["label"] = str(trigger_label)
                comparator = trigger_entry.get("comparator")
                if comparator and trigger_bucket["comparator"] is None:
                    trigger_bucket["comparator"] = str(comparator)

                trigger_unit = trigger_entry.get("unit")
                if trigger_unit and trigger_bucket["unit"] is None:
                    trigger_bucket["unit"] = str(trigger_unit)

                threshold_value = trigger_entry.get("threshold")
                threshold_float = AutoTrader._coerce_float(threshold_value)
                if trigger_bucket["threshold"] is None:
                    trigger_bucket["threshold"] = (
                        threshold_float
                        if threshold_float is not None
                        else threshold_value
                    )

                value_raw = trigger_entry.get("value")
                value_float = AutoTrader._coerce_float(value_raw)
                trigger_bucket["value_last"] = copy.deepcopy(value_raw)
                if value_float is not None:
                    current_min = trigger_bucket["value_min"]
                    if current_min is None or value_float < current_min:
                        trigger_bucket["value_min"] = value_float
                    current_max = trigger_bucket["value_max"]
                    if current_max is None or value_float > current_max:
                        trigger_bucket["value_max"] = value_float
                    trigger_bucket["_value_sum"] += value_float
                    trigger_bucket["_value_count"] += 1

                trigger_bucket["services"][service_name] += 1
                service_bucket["triggers"][trigger_name] += 1

            summary["guardrail_events"] += 1
            service_bucket["guardrail_events"] += 1

        summary["reasons"] = dict(reason_counts)

        normalized_triggers: dict[str, Any] = {}
        for trigger_name, trigger_bucket in trigger_buckets.items():
            value_count = trigger_bucket.pop("_value_count", 0)
            value_sum = trigger_bucket.pop("_value_sum", 0.0)
            trigger_bucket["services"] = dict(trigger_bucket["services"])
            if value_count:
                trigger_bucket["value_avg"] = value_sum / value_count
            else:
                trigger_bucket["value_avg"] = None
            normalized_triggers[trigger_name] = trigger_bucket
        summary["triggers"] = normalized_triggers

        summary["services"] = {
            service_name: {
                "total": bucket["total"],
                "guardrail_events": bucket["guardrail_events"],
                "reasons": dict(bucket["reasons"]),
                "triggers": dict(bucket["triggers"]),
            }
            for service_name, bucket in service_buckets.items()
        }

        return summary

    def summarize_risk_decision_dimensions(
        self,
        *,
        approved: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        normalized: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        include_errors: bool = True,
        service: str | None | Iterable[str | None] | object = _NO_FILTER,
        decision_state: str | Iterable[str | None] | object = _NO_FILTER,
        decision_reason: str | Iterable[str | None] | object = _NO_FILTER,
        decision_mode: str | Iterable[str | None] | object = _NO_FILTER,
        decision_id: str | Iterable[str | None] | object = _NO_FILTER,
        since: Any = None,
        until: Any = None,
    ) -> dict[str, Any]:
        """Agreguje historię decyzji ryzyka po stanie, powodzie i trybie."""

        approved_filter = self._prepare_bool_filter(approved)
        normalized_filter = self._prepare_bool_filter(normalized)
        service_filter = self._prepare_service_filter(service)
        decision_state_filter = self._prepare_decision_filter(
            decision_state,
            missing_token=_MISSING_DECISION_STATE,
        )
        decision_reason_filter = self._prepare_decision_filter(
            decision_reason,
            missing_token=_MISSING_DECISION_REASON,
        )
        decision_mode_filter = self._prepare_decision_filter(
            decision_mode,
            missing_token=_MISSING_DECISION_MODE,
        )
        decision_id_filter = self._prepare_decision_filter(
            decision_id,
            missing_token=_MISSING_DECISION_ID,
        )
        since_ts = self._normalize_time_bound(since)
        until_ts = self._normalize_time_bound(until)

        (
            filtered_records,
            trimmed_by_ttl,
            ttl_snapshot,
            history_size,
        ) = self._collect_filtered_risk_evaluations(
            include_errors=include_errors,
            approved_filter=approved_filter,
            normalized_filter=normalized_filter,
            service_filter=service_filter,
            since_ts=since_ts,
            until_ts=until_ts,
            state_filter=decision_state_filter,
            reason_filter=decision_reason_filter,
            mode_filter=decision_mode_filter,
            decision_id_filter=decision_id_filter,
        )
        self._log_risk_history_trimmed(
            context="decision-dimensions",
            trimmed=trimmed_by_ttl,
            ttl=ttl_snapshot,
            history=history_size,
        )

        total = len(filtered_records)
        summary: dict[str, Any] = {
            "total": total,
            "states": {},
            "reasons": {},
            "modes": {},
            "combinations": [],
            "services": {},
            "first_timestamp": None,
            "last_timestamp": None,
        }
        if total == 0:
            return summary

        states_summary: dict[str, dict[str, Any]] = {}
        reasons_summary: dict[str, dict[str, Any]] = {}
        modes_summary: dict[str, dict[str, Any]] = {}
        combinations_summary: dict[tuple[str, str, str], dict[str, Any]] = {}
        services_counter: Counter[str] = Counter()

        for entry in filtered_records:
            decision_payload = entry.get("decision") or {}
            state_raw = decision_payload.get("state")
            reason_raw = decision_payload.get("reason")
            mode_raw = decision_payload.get("mode")

            state_key = str(state_raw) if state_raw is not None else _MISSING_DECISION_STATE
            reason_key = (
                _MISSING_DECISION_REASON
                if reason_raw is None
                else str(reason_raw)
            )
            mode_key = (
                _MISSING_DECISION_MODE
                if mode_raw is None
                else str(mode_raw)
            )

            normalized_value = entry.get("normalized")
            if normalized_value is True:
                normalized_state: bool | None = True
            elif normalized_value is False:
                normalized_state = False
            else:
                normalized_state = None

            raw_value = entry.get("approved")
            if raw_value is True:
                raw_state: bool | None = True
            elif raw_value is False:
                raw_state = False
            else:
                raw_state = None

            has_error = "error" in entry
            service_key = entry.get("service") or _UNKNOWN_SERVICE
            service_name = str(service_key)
            services_counter[service_name] += 1

            state_bucket = states_summary.setdefault(
                state_key, self._create_decision_bucket()
            )
            self._update_decision_bucket(
                state_bucket,
                normalized_value=normalized_state,
                raw_value=raw_state,
                has_error=has_error,
                service_key=service_name,
            )

            reason_bucket = reasons_summary.setdefault(
                reason_key, self._create_decision_bucket()
            )
            self._update_decision_bucket(
                reason_bucket,
                normalized_value=normalized_state,
                raw_value=raw_state,
                has_error=has_error,
                service_key=service_name,
            )

            mode_bucket = modes_summary.setdefault(
                mode_key, self._create_decision_bucket()
            )
            self._update_decision_bucket(
                mode_bucket,
                normalized_value=normalized_state,
                raw_value=raw_state,
                has_error=has_error,
                service_key=service_name,
            )

            combination_bucket = combinations_summary.setdefault(
                (state_key, reason_key, mode_key),
                self._create_decision_bucket(),
            )
            self._update_decision_bucket(
                combination_bucket,
                normalized_value=normalized_state,
                raw_value=raw_state,
                has_error=has_error,
                service_key=service_name,
            )

        for bucket in states_summary.values():
            self._finalize_decision_bucket(bucket)
        for bucket in reasons_summary.values():
            self._finalize_decision_bucket(bucket)
        for bucket in modes_summary.values():
            self._finalize_decision_bucket(bucket)
        for bucket in combinations_summary.values():
            self._finalize_decision_bucket(bucket)

        combinations_list: list[dict[str, Any]] = []
        for (state_key, reason_key, mode_key), bucket in combinations_summary.items():
            payload = {**bucket}
            payload["state"] = state_key
            payload["reason"] = reason_key
            payload["mode"] = mode_key
            combinations_list.append(payload)

        combinations_list.sort(
            key=lambda item: (
                -item.get("total", 0),
                item.get("state", ""),
                item.get("reason", ""),
                item.get("mode", ""),
            )
        )

        summary["states"] = self._sort_decision_dimension(states_summary)
        summary["reasons"] = self._sort_decision_dimension(reasons_summary)
        summary["modes"] = self._sort_decision_dimension(modes_summary)
        summary["combinations"] = combinations_list
        summary["services"] = {
            name: count
            for name, count in sorted(
                services_counter.items(),
                key=lambda item: (-item[1], item[0]),
            )
        }
        summary["first_timestamp"] = filtered_records[0].get("timestamp")
        summary["last_timestamp"] = filtered_records[-1].get("timestamp")

        return summary



    def _build_risk_decision_timeline(
        self,
        *,
        context: str,
        bucket_value: float,
        include_errors: bool,
        include_services: bool,
        include_decision_dimensions: bool,
        fill_gaps: bool,
        coerce_timestamps: bool,
        tz: tzinfo | None,
        approved_filter: set[bool | None] | None,
        normalized_filter: set[bool | None] | None,
        service_filter: set[str] | None,
        decision_state_filter: set[str] | None,
        decision_reason_filter: set[str] | None,
        decision_mode_filter: set[str] | None,
        decision_id_filter: set[str] | None,
        since_ts: float | None,
        until_ts: float | None,
    ) -> dict[str, Any]:
        (
            filtered_records,
            trimmed_by_ttl,
            ttl_snapshot,
            history_size,
        ) = self._collect_filtered_risk_evaluations(
            include_errors=include_errors,
            approved_filter=approved_filter,
            normalized_filter=normalized_filter,
            service_filter=service_filter,
            since_ts=since_ts,
            until_ts=until_ts,
            state_filter=decision_state_filter,
            reason_filter=decision_reason_filter,
            mode_filter=decision_mode_filter,
            decision_id_filter=decision_id_filter,
        )
        self._log_risk_history_trimmed(
            context=context,
            trimmed=trimmed_by_ttl,
            ttl=ttl_snapshot,
            history=history_size,
        )

        summary: dict[str, Any] = {
            "bucket_s": bucket_value,
            "total": len(filtered_records),
            "buckets": [],
            "first_timestamp": None,
            "last_timestamp": None,
        }
        summary["filters"] = self._snapshot_decision_timeline_filters(
            approved_filter=approved_filter,
            normalized_filter=normalized_filter,
            include_errors=include_errors,
            service_filter=service_filter,
            decision_state_filter=decision_state_filter,
            decision_reason_filter=decision_reason_filter,
            decision_mode_filter=decision_mode_filter,
            decision_id_filter=decision_id_filter,
            since_ts=since_ts,
            until_ts=until_ts,
            include_services=include_services,
            include_decision_dimensions=include_decision_dimensions,
            fill_gaps=fill_gaps,
            coerce_timestamps=coerce_timestamps,
            tz_value=tz,
        )
        summary_totals = {
            "approved": 0,
            "rejected": 0,
            "unknown": 0,
            "errors": 0,
            "raw_true": 0,
            "raw_false": 0,
            "raw_none": 0,
        }
        total_services: dict[str, dict[str, int]] | None = None
        if include_services:
            summary["services"] = {}
            total_services = {}
        summary.update(summary_totals)
        summary["approval_rate"] = 0.0
        summary["error_rate"] = 0.0
        if not filtered_records:
            return summary

        def build_bucket() -> dict[str, Any]:
            bucket = self._create_decision_bucket()
            if include_decision_dimensions:
                bucket["states"] = Counter()
                bucket["reasons"] = Counter()
                bucket["modes"] = Counter()
            return bucket

        bucket_map: dict[int, dict[str, Any]] = {}
        missing_bucket: dict[str, Any] | None = None
        first_ts: float | None = None
        last_ts: float | None = None

        for entry in filtered_records:
            timestamp_value = self._normalize_time_bound(entry.get("timestamp"))
            if timestamp_value is not None and (
                not math.isfinite(timestamp_value) or math.isnan(timestamp_value)
            ):
                timestamp_value = None

            if timestamp_value is None:
                if missing_bucket is None:
                    missing_bucket = build_bucket()
                bucket_payload = missing_bucket
            else:
                if first_ts is None or timestamp_value < first_ts:
                    first_ts = timestamp_value
                if last_ts is None or timestamp_value > last_ts:
                    last_ts = timestamp_value

                bucket_index = int(math.floor(timestamp_value / bucket_value))
                bucket_payload = bucket_map.get(bucket_index)
                if bucket_payload is None:
                    bucket_payload = build_bucket()
                    bucket_map[bucket_index] = bucket_payload

            normalized_value = entry.get("normalized")
            raw_value = entry.get("approved")
            has_error = "error" in entry
            service_key = entry.get("service") or _UNKNOWN_SERVICE

            if normalized_value is True:
                summary_totals["approved"] += 1
            elif normalized_value is False:
                summary_totals["rejected"] += 1
            else:
                summary_totals["unknown"] += 1

            if raw_value is True:
                summary_totals["raw_true"] += 1
            elif raw_value is False:
                summary_totals["raw_false"] += 1
            else:
                summary_totals["raw_none"] += 1

            if has_error:
                summary_totals["errors"] += 1

            self._update_decision_bucket(
                bucket_payload,
                normalized_value=normalized_value,
                raw_value=raw_value,
                has_error=has_error,
                service_key=str(service_key),
            )

            if total_services is not None:
                service_totals = total_services.setdefault(
                    str(service_key),
                    {
                        "evaluations": 0,
                        "approved": 0,
                        "rejected": 0,
                        "unknown": 0,
                        "errors": 0,
                        "raw_true": 0,
                        "raw_false": 0,
                        "raw_none": 0,
                    },
                )
                service_totals["evaluations"] += 1
                if normalized_value is True:
                    service_totals["approved"] += 1
                elif normalized_value is False:
                    service_totals["rejected"] += 1
                else:
                    service_totals["unknown"] += 1

                if raw_value is True:
                    service_totals["raw_true"] += 1
                elif raw_value is False:
                    service_totals["raw_false"] += 1
                else:
                    service_totals["raw_none"] += 1

                if has_error:
                    service_totals["errors"] += 1

            if include_decision_dimensions:
                decision_payload = entry.get("decision")
                decision_map = (
                    decision_payload
                    if isinstance(decision_payload, Mapping)
                    else {}
                )
                state_value = self._normalize_decision_dimension_value(
                    decision_map.get("state"),
                    missing_token=_MISSING_DECISION_STATE,
                )
                reason_value = self._normalize_decision_dimension_value(
                    decision_map.get("reason"),
                    missing_token=_MISSING_DECISION_REASON,
                )
                mode_value = self._normalize_decision_dimension_value(
                    decision_map.get("mode"),
                    missing_token=_MISSING_DECISION_MODE,
                )

                states_counter = bucket_payload.setdefault("states", Counter())
                reasons_counter = bucket_payload.setdefault("reasons", Counter())
                modes_counter = bucket_payload.setdefault("modes", Counter())

                states_counter[state_value] += 1
                reasons_counter[reason_value] += 1
                modes_counter[mode_value] += 1

        if fill_gaps and bucket_map:
            bucket_indices = sorted(bucket_map.keys())
            for bucket_index in range(bucket_indices[0], bucket_indices[-1] + 1):
                bucket_map.setdefault(bucket_index, build_bucket())

        buckets_output: list[dict[str, Any]] = []
        for bucket_index in sorted(bucket_map.keys()):
            bucket_payload = bucket_map[bucket_index]
            self._finalize_decision_bucket(bucket_payload)

            start_ts = bucket_index * bucket_value
            end_ts = start_ts + bucket_value

            bucket_summary: dict[str, Any] = {
                "index": bucket_index,
                "start": self._normalize_timestamp_for_export(
                    start_ts,
                    coerce=coerce_timestamps,
                    tz=tz,
                ),
                "end": self._normalize_timestamp_for_export(
                    end_ts,
                    coerce=coerce_timestamps,
                    tz=tz,
                ),
                "total": bucket_payload.get("total", 0),
                "approved": bucket_payload.get("approved", 0),
                "rejected": bucket_payload.get("rejected", 0),
                "unknown": bucket_payload.get("unknown", 0),
                "errors": bucket_payload.get("errors", 0),
                "raw_true": bucket_payload.get("raw_true", 0),
                "raw_false": bucket_payload.get("raw_false", 0),
                "raw_none": bucket_payload.get("raw_none", 0),
                "approval_rate": bucket_payload.get("approval_rate", 0.0),
                "error_rate": bucket_payload.get("error_rate", 0.0),
            }

            if include_services:
                bucket_summary["services"] = dict(bucket_payload.get("services", {}))
            if include_decision_dimensions:
                bucket_summary["states"] = self._finalize_dimension_counter(
                    bucket_payload.get("states")
                )
                bucket_summary["reasons"] = self._finalize_dimension_counter(
                    bucket_payload.get("reasons")
                )
                bucket_summary["modes"] = self._finalize_dimension_counter(
                    bucket_payload.get("modes")
                )

            buckets_output.append(bucket_summary)

        summary["buckets"] = buckets_output
        summary["first_timestamp"] = first_ts
        summary["last_timestamp"] = last_ts
        summary.update(summary_totals)
        total_count = summary.get("total", 0) or 0
        summary["approval_rate"] = (
            summary_totals["approved"] / total_count if total_count else 0.0
        )
        summary["error_rate"] = (
            summary_totals["errors"] / total_count if total_count else 0.0
        )

        if total_services is not None:
            summary["services"] = {
                service_name: _ServiceDecisionTotals(
                    {
                        "evaluations": totals.get("evaluations", 0),
                        "approved": totals.get("approved", 0),
                        "rejected": totals.get("rejected", 0),
                        "unknown": totals.get("unknown", 0),
                        "errors": totals.get("errors", 0),
                        "raw_true": totals.get("raw_true", 0),
                        "raw_false": totals.get("raw_false", 0),
                        "raw_none": totals.get("raw_none", 0),
                    }
                )
                for service_name, totals in sorted(
                    total_services.items(),
                    key=lambda item: (-item[1].get("evaluations", 0), item[0]),
                )
            }

        if missing_bucket is not None and missing_bucket.get("total", 0):
            self._finalize_decision_bucket(missing_bucket)
            missing_summary: dict[str, Any] = {
                "index": None,
                "total": missing_bucket.get("total", 0),
                "approved": missing_bucket.get("approved", 0),
                "rejected": missing_bucket.get("rejected", 0),
                "unknown": missing_bucket.get("unknown", 0),
                "errors": missing_bucket.get("errors", 0),
                "raw_true": missing_bucket.get("raw_true", 0),
                "raw_false": missing_bucket.get("raw_false", 0),
                "raw_none": missing_bucket.get("raw_none", 0),
                "approval_rate": missing_bucket.get("approval_rate", 0.0),
                "error_rate": missing_bucket.get("error_rate", 0.0),
            }
            if include_services:
                missing_summary["services"] = dict(missing_bucket.get("services", {}))
            if include_decision_dimensions:
                missing_summary["states"] = self._finalize_dimension_counter(
                    missing_bucket.get("states")
                )
                missing_summary["reasons"] = self._finalize_dimension_counter(
                    missing_bucket.get("reasons")
                )
                missing_summary["modes"] = self._finalize_dimension_counter(
                    missing_bucket.get("modes")
                )
            summary["missing_timestamp"] = missing_summary

        if not include_services:
            for bucket in summary["buckets"]:
                bucket.pop("services", None)
            missing_entry = summary.get("missing_timestamp")
            if isinstance(missing_entry, dict):
                missing_entry.pop("services", None)

        return summary

    def summarize_risk_decision_timeline(
        self,
        *,
        bucket_s: float,
        approved: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        normalized: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        include_errors: bool = True,
        service: str | None | Iterable[str | None] | object = _NO_FILTER,
        decision_state: str | Iterable[str | None] | object = _NO_FILTER,
        decision_reason: str | Iterable[str | None] | object = _NO_FILTER,
        decision_mode: str | Iterable[str | None] | object = _NO_FILTER,
        decision_id: str | Iterable[str | None] | object = _NO_FILTER,
        since: Any = None,
        until: Any = None,
        include_services: bool = True,
        include_decision_dimensions: bool = False,
        fill_gaps: bool = False,
        coerce_timestamps: bool = False,
        tz: tzinfo | None = timezone.utc,
    ) -> dict[str, Any]:
        """Agreguje decyzje ryzyka w kubełkach czasowych."""

        try:
            bucket_value = float(bucket_s)
        except (TypeError, ValueError) as exc:  # pragma: no cover - walidacja wejścia
            raise ValueError("bucket_s must be a positive number") from exc
        if not math.isfinite(bucket_value) or bucket_value <= 0.0:
            raise ValueError("bucket_s must be a positive number")

        approved_filter = self._prepare_bool_filter(approved)
        normalized_filter = self._prepare_bool_filter(normalized)
        service_filter = self._prepare_service_filter(service)
        decision_state_filter = self._prepare_decision_filter(
            decision_state,
            missing_token=_MISSING_DECISION_STATE,
        )
        decision_reason_filter = self._prepare_decision_filter(
            decision_reason,
            missing_token=_MISSING_DECISION_REASON,
        )
        decision_mode_filter = self._prepare_decision_filter(
            decision_mode,
            missing_token=_MISSING_DECISION_MODE,
        )
        decision_id_filter = self._prepare_decision_filter(
            decision_id,
            missing_token=_MISSING_DECISION_ID,
        )

        since_ts = self._normalize_time_bound(since)
        until_ts = self._normalize_time_bound(until)

        return self._build_risk_decision_timeline(
            context="decision-timeline",
            bucket_value=bucket_value,
            include_errors=include_errors,
            include_services=include_services,
            include_decision_dimensions=include_decision_dimensions,
            fill_gaps=fill_gaps,
            coerce_timestamps=coerce_timestamps,
            tz=tz,
            approved_filter=approved_filter,
            normalized_filter=normalized_filter,
            service_filter=service_filter,
            decision_state_filter=decision_state_filter,
            decision_reason_filter=decision_reason_filter,
            decision_mode_filter=decision_mode_filter,
            decision_id_filter=decision_id_filter,
            since_ts=since_ts,
            until_ts=until_ts,
        )


    def risk_decision_timeline_to_records(
        self,
        *,
        bucket_s: float,
        approved: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        normalized: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        include_errors: bool = True,
        service: str | None | Iterable[str | None] | object = _NO_FILTER,
        decision_state: str | Iterable[str | None] | object = _NO_FILTER,
        decision_reason: str | Iterable[str | None] | object = _NO_FILTER,
        decision_mode: str | Iterable[str | None] | object = _NO_FILTER,
        decision_id: str | Iterable[str | None] | object = _NO_FILTER,
        since: Any = None,
        until: Any = None,
        include_services: bool = True,
        include_decision_dimensions: bool = False,
        fill_gaps: bool = False,
        include_missing_bucket: bool = False,
        coerce_timestamps: bool = False,
        tz: tzinfo | None = timezone.utc,
    ) -> list[dict[str, Any]]:
        """Eksportuje kubełki timeline'u decyzji w formie listy rekordów."""

        try:
            bucket_value = float(bucket_s)
        except (TypeError, ValueError) as exc:  # pragma: no cover - walidacja wejścia
            raise ValueError("bucket_s must be a positive number") from exc
        if not math.isfinite(bucket_value) or bucket_value <= 0.0:
            raise ValueError("bucket_s must be a positive number")

        approved_filter = self._prepare_bool_filter(approved)
        normalized_filter = self._prepare_bool_filter(normalized)
        service_filter = self._prepare_service_filter(service)
        decision_state_filter = self._prepare_decision_filter(
            decision_state,
            missing_token=_MISSING_DECISION_STATE,
        )
        decision_reason_filter = self._prepare_decision_filter(
            decision_reason,
            missing_token=_MISSING_DECISION_REASON,
        )
        decision_mode_filter = self._prepare_decision_filter(
            decision_mode,
            missing_token=_MISSING_DECISION_MODE,
        )
        decision_id_filter = self._prepare_decision_filter(
            decision_id,
            missing_token=_MISSING_DECISION_ID,
        )

        since_ts = self._normalize_time_bound(since)
        until_ts = self._normalize_time_bound(until)

        summary = self._build_risk_decision_timeline(
            context="decision-timeline-records",
            bucket_value=bucket_value,
            include_errors=include_errors,
            include_services=include_services,
            include_decision_dimensions=include_decision_dimensions,
            fill_gaps=fill_gaps,
            coerce_timestamps=coerce_timestamps,
            tz=tz,
            approved_filter=approved_filter,
            normalized_filter=normalized_filter,
            service_filter=service_filter,
            decision_state_filter=decision_state_filter,
            decision_reason_filter=decision_reason_filter,
            decision_mode_filter=decision_mode_filter,
            decision_id_filter=decision_id_filter,
            since_ts=since_ts,
            until_ts=until_ts,
        )

        records: list[dict[str, Any]] = []
        for bucket in summary.get("buckets", []):
            bucket_record = copy.deepcopy(bucket)
            bucket_record.setdefault("bucket_type", "bucket")
            records.append(bucket_record)

        if include_missing_bucket:
            missing_entry = summary.get("missing_timestamp")
            if isinstance(missing_entry, Mapping) and missing_entry.get("total", 0):
                missing_record = copy.deepcopy(missing_entry)
                missing_record.setdefault("bucket_type", "missing")
                missing_record.setdefault("index", None)
                missing_record["start"] = None
                missing_record["end"] = None
                records.append(missing_record)

        metadata = _extract_guardrail_timeline_metadata(summary)

        if include_services and isinstance(summary.get("services"), Mapping):
            summary_record = {
                "bucket_type": "summary",
                "index": None,
                "start": self._normalize_timestamp_for_export(
                    summary.get("first_timestamp"),
                    coerce=coerce_timestamps,
                    tz=tz,
                ),
                "end": self._normalize_timestamp_for_export(
                    summary.get("last_timestamp"),
                    coerce=coerce_timestamps,
                    tz=tz,
                ),
                "total": summary.get("total", 0),
                "approved": summary.get("approved", 0),
                "rejected": summary.get("rejected", 0),
                "unknown": summary.get("unknown", 0),
                "errors": summary.get("errors", 0),
                "raw_true": summary.get("raw_true", 0),
                "raw_false": summary.get("raw_false", 0),
                "raw_none": summary.get("raw_none", 0),
                "guardrail_rate": summary.get("guardrail_rate", 0.0),
                "approval_rate": summary.get("approval_rate", 0.0),
                "error_rate": summary.get("error_rate", 0.0),
                "services": copy.deepcopy(summary.get("services")),
            }
            if "guardrail_trigger_thresholds" in summary:
                summary_record["guardrail_trigger_thresholds"] = copy.deepcopy(
                    summary.get("guardrail_trigger_thresholds")
                )
            if "guardrail_trigger_values" in summary:
                summary_record["guardrail_trigger_values"] = copy.deepcopy(
                    summary.get("guardrail_trigger_values")
                )
            if include_decision_dimensions:
                summary_record.setdefault("states", {})
                summary_record.setdefault("reasons", {})
                summary_record.setdefault("modes", {})
            records.append(summary_record)

        return GuardrailTimelineRecords(records, metadata)

    def risk_decision_timeline_to_dataframe(
        self,
        *,
        bucket_s: float,
        approved: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        normalized: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        include_errors: bool = True,
        service: str | None | Iterable[str | None] | object = _NO_FILTER,
        decision_state: str | Iterable[str | None] | object = _NO_FILTER,
        decision_reason: str | Iterable[str | None] | object = _NO_FILTER,
        decision_mode: str | Iterable[str | None] | object = _NO_FILTER,
        decision_id: str | Iterable[str | None] | object = _NO_FILTER,
        since: Any = None,
        until: Any = None,
        include_services: bool = False,
        include_decision_dimensions: bool = False,
        fill_gaps: bool = False,
        include_missing_bucket: bool = False,
        coerce_timestamps: bool = True,
        tz: tzinfo | None = timezone.utc,
    ) -> pd.DataFrame:
        """Buduje ``DataFrame`` z kubełkami timeline'u decyzji ryzyka."""

        records = self.risk_decision_timeline_to_records(
            bucket_s=bucket_s,
            approved=approved,
            normalized=normalized,
            include_errors=include_errors,
            service=service,
            decision_state=decision_state,
            decision_reason=decision_reason,
            decision_mode=decision_mode,
            decision_id=decision_id,
            since=since,
            until=until,
            include_services=include_services,
            include_decision_dimensions=include_decision_dimensions,
            fill_gaps=fill_gaps,
            include_missing_bucket=include_missing_bucket,
            coerce_timestamps=coerce_timestamps,
            tz=tz,
        )

        base_columns = [
            "bucket_type",
            "index",
            "start",
            "end",
            "total",
            "approved",
            "rejected",
            "unknown",
            "errors",
            "raw_true",
            "raw_false",
            "raw_none",
            "approval_rate",
            "error_rate",
        ]
        if include_services:
            base_columns.append("services")
        if include_decision_dimensions:
            base_columns.extend(["states", "reasons", "modes"])

        if not records:
            return pd.DataFrame(columns=base_columns)

        rows: list[dict[str, Any]] = []
        for record in records:
            row = {column: record.get(column) for column in base_columns}
            rows.append(row)

        df = pd.DataFrame.from_records(rows, columns=base_columns)
        return df

    def _build_guardrail_event_record(
        self,
        entry: Mapping[str, Any],
        reasons: Sequence[str],
        triggers: Sequence[Mapping[str, Any]],
        *,
        include_decision: bool,
        include_service: bool,
        include_response: bool,
        include_error: bool,
        include_guardrail_dimensions: bool,
        coerce_timestamps: bool,
        tz: tzinfo | None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "timestamp": self._normalize_timestamp_for_export(
                entry.get("timestamp"),
                coerce=coerce_timestamps,
                tz=tz,
            ),
            "approved": entry.get("approved"),
            "normalized": entry.get("normalized"),
            "decision_id": self._normalize_decision_id(entry.get("decision_id")),
        }

        if include_service:
            record["service"] = entry.get("service") or _UNKNOWN_SERVICE
        if include_response:
            record["response"] = copy.deepcopy(entry.get("response"))
        if include_error:
            record["error"] = copy.deepcopy(entry.get("error"))
        if include_guardrail_dimensions:
            record["guardrail_reasons"] = tuple(reasons)
            record["guardrail_triggers"] = tuple(
                copy.deepcopy(trigger) for trigger in triggers
            )
            record["guardrail_reason_count"] = len(reasons)
            record["guardrail_trigger_count"] = len(triggers)
        if include_decision:
            record["decision"] = copy.deepcopy(entry.get("decision"))

        return record

    def guardrail_events_to_records(
        self,
        *,
        approved: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        normalized: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        include_errors: bool = True,
        service: str | None | Iterable[str | None] | object = _NO_FILTER,
        decision_state: str | Iterable[str | None] | object = _NO_FILTER,
        decision_reason: str | Iterable[str | None] | object = _NO_FILTER,
        decision_mode: str | Iterable[str | None] | object = _NO_FILTER,
        decision_id: str | Iterable[str | None] | object = _NO_FILTER,
        reason: str | Iterable[str] | object = _NO_FILTER,
        trigger: str | Iterable[str] | object = _NO_FILTER,
        trigger_label: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_comparator: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_unit: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_threshold: float | None | Iterable[float | None] | object = _NO_FILTER,
        trigger_threshold_min: Any = None,
        trigger_threshold_max: Any = None,
        trigger_value: float | None | Iterable[float | None] | object = _NO_FILTER,
        trigger_value_min: Any = None,
        trigger_value_max: Any = None,
        since: Any = None,
        until: Any = None,
        limit: int | None = None,
        reverse: bool = False,
        include_decision: bool = True,
        include_service: bool = True,
        include_response: bool = True,
        include_error: bool = True,
        include_guardrail_dimensions: bool = True,
        coerce_timestamps: bool = False,
        tz: tzinfo | None = timezone.utc,
    ) -> list[dict[str, Any]]:
        """Eksportuje zdarzenia guardrail z historii ocen ryzyka."""

        normalized_limit = self._normalize_history_export_limit(limit)

        (
            approved_filter,
            normalized_filter,
            service_filter,
            decision_state_filter,
            decision_reason_filter,
            decision_mode_filter,
            decision_id_filter,
            reason_filter,
            trigger_filter,
            trigger_label_filter,
            trigger_comparator_filter,
            trigger_unit_filter,
            trigger_threshold_filter,
            trigger_threshold_min_value,
            trigger_threshold_max_value,
            trigger_value_filter,
            trigger_value_min_value,
            trigger_value_max_value,
            since_ts,
            until_ts,
        ) = self._resolve_guardrail_event_filters(
            approved=approved,
            normalized=normalized,
            service=service,
            decision_state=decision_state,
            decision_reason=decision_reason,
            decision_mode=decision_mode,
            decision_id=decision_id,
            reason=reason,
            trigger=trigger,
            trigger_label=trigger_label,
            trigger_comparator=trigger_comparator,
            trigger_unit=trigger_unit,
            trigger_threshold=trigger_threshold,
            trigger_threshold_min=trigger_threshold_min,
            trigger_threshold_max=trigger_threshold_max,
            trigger_value=trigger_value,
            trigger_value_min=trigger_value_min,
            trigger_value_max=trigger_value_max,
            since=since,
            until=until,
        )

        (
            guardrail_records,
            trimmed_by_ttl,
            ttl_snapshot,
            history_size,
            _filtered_records,
        ) = self._collect_guardrail_events(
            include_errors=include_errors,
            approved_filter=approved_filter,
            normalized_filter=normalized_filter,
            service_filter=service_filter,
            decision_state_filter=decision_state_filter,
            decision_reason_filter=decision_reason_filter,
            decision_mode_filter=decision_mode_filter,
            decision_id_filter=decision_id_filter,
            since_ts=since_ts,
            until_ts=until_ts,
            reason_filter=reason_filter,
            trigger_filter=trigger_filter,
            trigger_label_filter=trigger_label_filter,
            trigger_comparator_filter=trigger_comparator_filter,
            trigger_unit_filter=trigger_unit_filter,
            trigger_threshold_filter=trigger_threshold_filter,
            trigger_threshold_min=trigger_threshold_min_value,
            trigger_threshold_max=trigger_threshold_max_value,
            trigger_value_filter=trigger_value_filter,
            trigger_value_min=trigger_value_min_value,
            trigger_value_max=trigger_value_max_value,
        )

        self._log_risk_history_trimmed(
            context="guardrail-records",
            trimmed=trimmed_by_ttl,
            ttl=ttl_snapshot,
            history=history_size,
        )

        records_sequence = list(guardrail_records)
        if reverse:
            records_sequence.reverse()

        if normalized_limit is not None:
            if normalized_limit == 0:
                records_sequence = []
            elif reverse:
                records_sequence = records_sequence[:normalized_limit]
            elif len(records_sequence) > normalized_limit:
                records_sequence = records_sequence[-normalized_limit:]

        output: list[dict[str, Any]] = []
        for entry, reasons, triggers in records_sequence:
            record = self._build_guardrail_event_record(
                entry,
                reasons,
                triggers,
                include_decision=include_decision,
                include_service=include_service,
                include_response=include_response,
                include_error=include_error,
                include_guardrail_dimensions=include_guardrail_dimensions,
                coerce_timestamps=coerce_timestamps,
                tz=tz,
            )
            output.append(record)

        return output

    def guardrail_events_to_dataframe(
        self,
        *,
        approved: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        normalized: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        include_errors: bool = True,
        service: str | None | Iterable[str | None] | object = _NO_FILTER,
        decision_state: str | Iterable[str | None] | object = _NO_FILTER,
        decision_reason: str | Iterable[str | None] | object = _NO_FILTER,
        decision_mode: str | Iterable[str | None] | object = _NO_FILTER,
        decision_id: str | Iterable[str | None] | object = _NO_FILTER,
        reason: str | Iterable[str] | object = _NO_FILTER,
        trigger: str | Iterable[str] | object = _NO_FILTER,
        trigger_label: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_comparator: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_unit: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_threshold: float | None | Iterable[float | None] | object = _NO_FILTER,
        trigger_threshold_min: Any = None,
        trigger_threshold_max: Any = None,
        trigger_value: float | None | Iterable[float | None] | object = _NO_FILTER,
        trigger_value_min: Any = None,
        trigger_value_max: Any = None,
        since: Any = None,
        until: Any = None,
        limit: int | None = None,
        reverse: bool = False,
        include_decision: bool = False,
        include_service: bool = True,
        include_response: bool = True,
        include_error: bool = True,
        include_guardrail_dimensions: bool = True,
        coerce_timestamps: bool = True,
        tz: tzinfo | None = timezone.utc,
    ) -> pd.DataFrame:
        records = self.guardrail_events_to_records(
            approved=approved,
            normalized=normalized,
            include_errors=include_errors,
            service=service,
            decision_state=decision_state,
            decision_reason=decision_reason,
            decision_mode=decision_mode,
            decision_id=decision_id,
            reason=reason,
            trigger=trigger,
            trigger_label=trigger_label,
            trigger_comparator=trigger_comparator,
            trigger_unit=trigger_unit,
            trigger_threshold=trigger_threshold,
            trigger_threshold_min=trigger_threshold_min,
            trigger_threshold_max=trigger_threshold_max,
            trigger_value=trigger_value,
            trigger_value_min=trigger_value_min,
            trigger_value_max=trigger_value_max,
            since=since,
            until=until,
            limit=limit,
            reverse=reverse,
            include_decision=include_decision,
            include_service=include_service,
            include_response=include_response,
            include_error=include_error,
            include_guardrail_dimensions=include_guardrail_dimensions,
            coerce_timestamps=coerce_timestamps,
            tz=tz,
        )
        if not records:
            return pd.DataFrame()
        return pd.DataFrame.from_records(records)


    def export_guardrail_events(
        self,
        *,
        approved: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        normalized: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        include_errors: bool = True,
        service: str | None | Iterable[str | None] | object = _NO_FILTER,
        decision_state: str | Iterable[str | None] | object = _NO_FILTER,
        decision_reason: str | Iterable[str | None] | object = _NO_FILTER,
        decision_mode: str | Iterable[str | None] | object = _NO_FILTER,
        decision_id: str | Iterable[str | None] | object = _NO_FILTER,
        reason: str | Iterable[str] | object = _NO_FILTER,
        trigger: str | Iterable[str] | object = _NO_FILTER,
        trigger_label: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_comparator: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_unit: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_threshold: float | None | Iterable[float | None] | object = _NO_FILTER,
        trigger_threshold_min: Any = None,
        trigger_threshold_max: Any = None,
        trigger_value: float | None | Iterable[float | None] | object = _NO_FILTER,
        trigger_value_min: Any = None,
        trigger_value_max: Any = None,
        since: Any = None,
        until: Any = None,
        limit: int | None = None,
        reverse: bool = False,
        include_decision: bool = True,
        include_service: bool = True,
        include_response: bool = True,
        include_error: bool = True,
        include_guardrail_dimensions: bool = True,
        coerce_timestamps: bool = False,
        tz: tzinfo | None = timezone.utc,
    ) -> Mapping[str, Any]:
        normalized_limit = self._normalize_history_export_limit(limit)

        (
            approved_filter,
            normalized_filter,
            service_filter,
            decision_state_filter,
            decision_reason_filter,
            decision_mode_filter,
            decision_id_filter,
            reason_filter,
            trigger_filter,
            trigger_label_filter,
            trigger_comparator_filter,
            trigger_unit_filter,
            trigger_threshold_filter,
            trigger_threshold_min_value,
            trigger_threshold_max_value,
            trigger_value_filter,
            trigger_value_min_value,
            trigger_value_max_value,
            since_ts,
            until_ts,
        ) = self._resolve_guardrail_event_filters(
            approved=approved,
            normalized=normalized,
            service=service,
            decision_state=decision_state,
            decision_reason=decision_reason,
            decision_mode=decision_mode,
            decision_id=decision_id,
            reason=reason,
            trigger=trigger,
            trigger_label=trigger_label,
            trigger_comparator=trigger_comparator,
            trigger_unit=trigger_unit,
            trigger_threshold=trigger_threshold,
            trigger_threshold_min=trigger_threshold_min,
            trigger_threshold_max=trigger_threshold_max,
            trigger_value=trigger_value,
            trigger_value_min=trigger_value_min,
            trigger_value_max=trigger_value_max,
            since=since,
            until=until,
        )

        (
            guardrail_records,
            trimmed_by_ttl,
            ttl_snapshot,
            history_size,
            _filtered_records,
        ) = self._collect_guardrail_events(
            include_errors=include_errors,
            approved_filter=approved_filter,
            normalized_filter=normalized_filter,
            service_filter=service_filter,
            decision_state_filter=decision_state_filter,
            decision_reason_filter=decision_reason_filter,
            decision_mode_filter=decision_mode_filter,
            decision_id_filter=decision_id_filter,
            since_ts=since_ts,
            until_ts=until_ts,
            reason_filter=reason_filter,
            trigger_filter=trigger_filter,
            trigger_label_filter=trigger_label_filter,
            trigger_comparator_filter=trigger_comparator_filter,
            trigger_unit_filter=trigger_unit_filter,
            trigger_threshold_filter=trigger_threshold_filter,
            trigger_threshold_min=trigger_threshold_min_value,
            trigger_threshold_max=trigger_threshold_max_value,
            trigger_value_filter=trigger_value_filter,
            trigger_value_min=trigger_value_min_value,
            trigger_value_max=trigger_value_max_value,
        )

        self._log_risk_history_trimmed(
            context="guardrail-export",
            trimmed=trimmed_by_ttl,
            ttl=ttl_snapshot,
            history=history_size,
        )

        records: list[dict[str, Any]] = []
        if guardrail_records:
            iterable = guardrail_records
            if reverse:
                iterable = list(reversed(guardrail_records))
            for entry, reasons, triggers in iterable:
                record = self._build_guardrail_event_record(
                    entry,
                    reasons,
                    triggers,
                    include_decision=include_decision,
                    include_service=include_service,
                    include_response=include_response,
                    include_error=include_error,
                    include_guardrail_dimensions=include_guardrail_dimensions,
                    coerce_timestamps=coerce_timestamps,
                    tz=tz,
                )
                if coerce_timestamps:
                    timestamp_value = record.get("timestamp")
                    if hasattr(timestamp_value, "isoformat"):
                        record["timestamp"] = timestamp_value.isoformat()
                records.append(record)
                if normalized_limit is not None and len(records) >= normalized_limit:
                    break

        with self._lock:
            limit_snapshot = self._risk_evaluations_limit

        def _serialize_filter(values: Iterable[object] | None) -> list[str] | None:
            if values is None:
                return None
            return sorted(str(item) for item in values)

        def _serialize_numeric_filter(
            payload: tuple[set[float], bool] | None,
        ) -> Mapping[str, Any] | None:
            if payload is None:
                return None
            value_set, include_missing_flag = payload
            return {
                "values": sorted(float(item) for item in value_set),
                "include_missing": bool(include_missing_flag),
            }

        def _serialize_bound(value: float | None) -> str | None:
            if value is None:
                return None
            try:
                numeric = float(value)
            except (TypeError, ValueError):  # pragma: no cover - defensive guard
                return None
            return datetime.fromtimestamp(numeric, tz=timezone.utc).isoformat()

        filters_payload: dict[str, Any] = {
            "approved": _serialize_filter(approved_filter),
            "normalized": _serialize_filter(normalized_filter),
            "include_errors": bool(include_errors),
            "service": _serialize_filter(service_filter),
            "decision_state": _serialize_filter(decision_state_filter),
            "decision_reason": _serialize_filter(decision_reason_filter),
            "decision_mode": _serialize_filter(decision_mode_filter),
            "decision_id": _serialize_filter(decision_id_filter),
            "reason": _serialize_filter(reason_filter),
            "trigger": _serialize_filter(trigger_filter),
            "trigger_label": _serialize_filter(trigger_label_filter),
            "trigger_comparator": _serialize_filter(trigger_comparator_filter),
            "trigger_unit": _serialize_filter(trigger_unit_filter),
            "trigger_threshold": _serialize_numeric_filter(trigger_threshold_filter),
            "trigger_threshold_min": (
                float(trigger_threshold_min_value)
                if trigger_threshold_min_value is not None
                else None
            ),
            "trigger_threshold_max": (
                float(trigger_threshold_max_value)
                if trigger_threshold_max_value is not None
                else None
            ),
            "trigger_value": _serialize_numeric_filter(trigger_value_filter),
            "trigger_value_min": (
                float(trigger_value_min_value)
                if trigger_value_min_value is not None
                else None
            ),
            "trigger_value_max": (
                float(trigger_value_max_value)
                if trigger_value_max_value is not None
                else None
            ),
            "since": _serialize_bound(since_ts),
            "until": _serialize_bound(until_ts),
            "limit": normalized_limit,
            "reverse": bool(reverse),
            "include_decision": bool(include_decision),
            "include_service": bool(include_service),
            "include_response": bool(include_response),
            "include_error": bool(include_error),
            "include_guardrail_dimensions": bool(include_guardrail_dimensions),
            "coerce_timestamps": bool(coerce_timestamps),
            "timezone": tz.tzname(None) if isinstance(tz, tzinfo) else tz,
        }

        payload: dict[str, Any] = {
            "version": 1,
            "entries": records,
            "filters": filters_payload,
            "retention": {
                "limit": limit_snapshot,
                "ttl_s": ttl_snapshot,
            },
            "trimmed_by_ttl": trimmed_by_ttl,
            "history_size": history_size,
        }
        return payload

    def dump_guardrail_events(
        self,
        destination: str | Path,
        *,
        approved: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        normalized: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        include_errors: bool = True,
        service: str | None | Iterable[str | None] | object = _NO_FILTER,
        decision_state: str | Iterable[str | None] | object = _NO_FILTER,
        decision_reason: str | Iterable[str | None] | object = _NO_FILTER,
        decision_mode: str | Iterable[str | None] | object = _NO_FILTER,
        decision_id: str | Iterable[str | None] | object = _NO_FILTER,
        reason: str | Iterable[str] | object = _NO_FILTER,
        trigger: str | Iterable[str] | object = _NO_FILTER,
        trigger_label: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_comparator: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_unit: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_threshold: float | None | Iterable[float | None] | object = _NO_FILTER,
        trigger_threshold_min: Any = None,
        trigger_threshold_max: Any = None,
        trigger_value: float | None | Iterable[float | None] | object = _NO_FILTER,
        trigger_value_min: Any = None,
        trigger_value_max: Any = None,
        since: Any = None,
        until: Any = None,
        limit: int | None = None,
        reverse: bool = False,
        include_decision: bool = True,
        include_service: bool = True,
        include_response: bool = True,
        include_error: bool = True,
        include_guardrail_dimensions: bool = True,
        coerce_timestamps: bool = False,
        tz: tzinfo | None = timezone.utc,
        ensure_ascii: bool = False,
    ) -> None:
        payload = self.export_guardrail_events(
            approved=approved,
            normalized=normalized,
            include_errors=include_errors,
            service=service,
            decision_state=decision_state,
            decision_reason=decision_reason,
            decision_mode=decision_mode,
            decision_id=decision_id,
            reason=reason,
            trigger=trigger,
            trigger_label=trigger_label,
            trigger_comparator=trigger_comparator,
            trigger_unit=trigger_unit,
            trigger_threshold=trigger_threshold,
            trigger_threshold_min=trigger_threshold_min,
            trigger_threshold_max=trigger_threshold_max,
            trigger_value=trigger_value,
            trigger_value_min=trigger_value_min,
            trigger_value_max=trigger_value_max,
            since=since,
            until=until,
            limit=limit,
            reverse=reverse,
            include_decision=include_decision,
            include_service=include_service,
            include_response=include_response,
            include_error=include_error,
            include_guardrail_dimensions=include_guardrail_dimensions,
            coerce_timestamps=coerce_timestamps,
            tz=tz,
        )

        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=ensure_ascii),
            encoding="utf-8",
        )

    def load_guardrail_events(
        self,
        payload: Mapping[str, Any],
        *,
        merge: bool = False,
        notify_listeners: bool = False,
    ) -> int:
        if not isinstance(payload, Mapping):
            raise TypeError("payload musi być słownikiem zgodnym z export_guardrail_events()")

        entries_payload = payload.get("entries", [])
        if entries_payload is None:
            entries_payload = []
        if not isinstance(entries_payload, Iterable):
            raise TypeError("entries muszą być iterowalne i zawierać słowniki")

        normalized_entries: list[dict[str, Any]] = []
        for entry in entries_payload:
            if not isinstance(entry, Mapping):
                raise TypeError("każdy entry musi być słownikiem")
            normalized_entry = dict(entry)
            normalized_entry["timestamp"] = self._normalize_time_bound(
                normalized_entry.get("timestamp")
            )
            normalized_entry["decision_id"] = self._normalize_decision_id(
                normalized_entry.get("decision_id")
            )
            normalized_entries.append(normalized_entry)

        normalized_payload = dict(payload)
        normalized_payload["entries"] = normalized_entries
        return self.load_risk_evaluations(
            normalized_payload,
            merge=merge,
            notify_listeners=notify_listeners,
        )

    def import_guardrail_events(
        self,
        source: str | Path,
        *,
        merge: bool = False,
        notify_listeners: bool = False,
    ) -> int:
        if json is None:  # pragma: no cover - środowiska bez json
            raise RuntimeError("moduł json jest wymagany do importu historii guardrail")

        path = Path(source)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("plik musi zawierać obiekt JSON zgodny z export_guardrail_events()")
        return self.load_guardrail_events(
            payload,
            merge=merge,
            notify_listeners=notify_listeners,
        )

    def get_guardrail_event_trace(
        self,
        decision_id: Any,
        *,
        approved: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        normalized: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        include_errors: bool = True,
        service: str | None | Iterable[str | None] | object = _NO_FILTER,
        decision_state: str | Iterable[str | None] | object = _NO_FILTER,
        decision_reason: str | Iterable[str | None] | object = _NO_FILTER,
        decision_mode: str | Iterable[str | None] | object = _NO_FILTER,
        decision_id_filter: str | Iterable[str | None] | object = _NO_FILTER,
        reason: str | Iterable[str] | object = _NO_FILTER,
        trigger: str | Iterable[str] | object = _NO_FILTER,
        trigger_label: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_comparator: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_unit: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_threshold: float | None | Iterable[float | None] | object = _NO_FILTER,
        trigger_threshold_min: Any = None,
        trigger_threshold_max: Any = None,
        trigger_value: float | None | Iterable[float | None] | object = _NO_FILTER,
        trigger_value_min: Any = None,
        trigger_value_max: Any = None,
        since: Any = None,
        until: Any = None,
        include_decision: bool = True,
        include_service: bool = True,
        include_response: bool = True,
        include_error: bool = True,
        include_guardrail_dimensions: bool = True,
        coerce_timestamps: bool = True,
        tz: tzinfo | None = timezone.utc,
    ) -> Sequence[Mapping[str, Any]]:
        normalized_id = self._normalize_decision_id(decision_id)
        if normalized_id is None:
            return ()

        (
            approved_filter,
            normalized_filter,
            service_filter,
            decision_state_filter,
            decision_reason_filter,
            decision_mode_filter,
            decision_id_filter,
            reason_filter,
            trigger_filter,
            trigger_label_filter,
            trigger_comparator_filter,
            trigger_unit_filter,
            trigger_threshold_filter,
            trigger_threshold_min_value,
            trigger_threshold_max_value,
            trigger_value_filter,
            trigger_value_min_value,
            trigger_value_max_value,
            since_ts,
            until_ts,
        ) = self._resolve_guardrail_event_filters(
            approved=approved,
            normalized=normalized,
            service=service,
            decision_state=decision_state,
            decision_reason=decision_reason,
            decision_mode=decision_mode,
            decision_id=decision_id_filter,
            reason=reason,
            trigger=trigger,
            trigger_label=trigger_label,
            trigger_comparator=trigger_comparator,
            trigger_unit=trigger_unit,
            trigger_threshold=trigger_threshold,
            trigger_threshold_min=trigger_threshold_min,
            trigger_threshold_max=trigger_threshold_max,
            trigger_value=trigger_value,
            trigger_value_min=trigger_value_min,
            trigger_value_max=trigger_value_max,
            since=since,
            until=until,
        )

        (
            guardrail_records,
            trimmed_by_ttl,
            ttl_snapshot,
            history_size,
            _filtered_records,
        ) = self._collect_guardrail_events(
            include_errors=include_errors,
            approved_filter=approved_filter,
            normalized_filter=normalized_filter,
            service_filter=service_filter,
            decision_state_filter=decision_state_filter,
            decision_reason_filter=decision_reason_filter,
            decision_mode_filter=decision_mode_filter,
            decision_id_filter=decision_id_filter,
            since_ts=since_ts,
            until_ts=until_ts,
            reason_filter=reason_filter,
            trigger_filter=trigger_filter,
            trigger_label_filter=trigger_label_filter,
            trigger_comparator_filter=trigger_comparator_filter,
            trigger_unit_filter=trigger_unit_filter,
            trigger_threshold_filter=trigger_threshold_filter,
            trigger_threshold_min=trigger_threshold_min_value,
            trigger_threshold_max=trigger_threshold_max_value,
            trigger_value_filter=trigger_value_filter,
            trigger_value_min=trigger_value_min_value,
            trigger_value_max=trigger_value_max_value,
        )

        guardrail_records = [
            (entry, reasons, triggers)
            for entry, reasons, triggers in guardrail_records
            if self._normalize_decision_id(entry.get("decision_id")) == normalized_id
        ]

        self._log_risk_history_trimmed(
            context="guardrail-trace",
            trimmed=trimmed_by_ttl,
            ttl=ttl_snapshot,
            history=history_size,
        )

        if not guardrail_records:
            return ()

        first_timestamp = self._normalize_time_bound(guardrail_records[0][0].get("timestamp"))
        if first_timestamp is None:
            first_timestamp = 0.0
        previous_timestamp = first_timestamp

        timeline: list[Mapping[str, Any]] = []
        for index, (entry, reasons, triggers) in enumerate(guardrail_records):
            record = self._build_guardrail_event_record(
                entry,
                reasons,
                triggers,
                include_decision=include_decision,
                include_service=include_service,
                include_response=include_response,
                include_error=include_error,
                include_guardrail_dimensions=include_guardrail_dimensions,
                coerce_timestamps=coerce_timestamps,
                tz=tz,
            )
            if record.get("decision_id") is None:
                record["decision_id"] = normalized_id

            timestamp_value = self._normalize_time_bound(entry.get("timestamp"))
            if timestamp_value is None:
                timestamp_value = previous_timestamp if index else first_timestamp

            record["timestamp"] = self._normalize_timestamp_for_export(
                entry.get("timestamp"),
                coerce=coerce_timestamps,
                tz=tz,
            )
            record["step_index"] = index
            record["elapsed_since_first_s"] = float(max(0.0, timestamp_value - first_timestamp))
            record["elapsed_since_previous_s"] = float(
                max(0.0, timestamp_value - previous_timestamp if index else 0.0)
            )

            timeline.append(record)
            previous_timestamp = timestamp_value

        return tuple(timeline)

    def get_grouped_guardrail_events(
        self,
        *,
        approved: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        normalized: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        include_errors: bool = True,
        service: str | None | Iterable[str | None] | object = _NO_FILTER,
        decision_state: str | Iterable[str | None] | object = _NO_FILTER,
        decision_reason: str | Iterable[str | None] | object = _NO_FILTER,
        decision_mode: str | Iterable[str | None] | object = _NO_FILTER,
        decision_id: str | Iterable[str | None] | object = _NO_FILTER,
        reason: str | Iterable[str] | object = _NO_FILTER,
        trigger: str | Iterable[str] | object = _NO_FILTER,
        trigger_label: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_comparator: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_unit: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_threshold: float | None | Iterable[float | None] | object = _NO_FILTER,
        trigger_threshold_min: Any = None,
        trigger_threshold_max: Any = None,
        trigger_value: float | None | Iterable[float | None] | object = _NO_FILTER,
        trigger_value_min: Any = None,
        trigger_value_max: Any = None,
        since: Any = None,
        until: Any = None,
        include_unidentified: bool = False,
        include_decision: bool = True,
        include_service: bool = True,
        include_response: bool = True,
        include_error: bool = True,
        include_guardrail_dimensions: bool = True,
        coerce_timestamps: bool = True,
        tz: tzinfo | None = timezone.utc,
    ) -> Mapping[str | None, Sequence[Mapping[str, Any]]]:
        (
            approved_filter,
            normalized_filter,
            service_filter,
            decision_state_filter,
            decision_reason_filter,
            decision_mode_filter,
            decision_id_filter,
            reason_filter,
            trigger_filter,
            trigger_label_filter,
            trigger_comparator_filter,
            trigger_unit_filter,
            trigger_threshold_filter,
            trigger_threshold_min_value,
            trigger_threshold_max_value,
            trigger_value_filter,
            trigger_value_min_value,
            trigger_value_max_value,
            since_ts,
            until_ts,
        ) = self._resolve_guardrail_event_filters(
            approved=approved,
            normalized=normalized,
            service=service,
            decision_state=decision_state,
            decision_reason=decision_reason,
            decision_mode=decision_mode,
            decision_id=decision_id,
            reason=reason,
            trigger=trigger,
            trigger_label=trigger_label,
            trigger_comparator=trigger_comparator,
            trigger_unit=trigger_unit,
            trigger_threshold=trigger_threshold,
            trigger_threshold_min=trigger_threshold_min,
            trigger_threshold_max=trigger_threshold_max,
            trigger_value=trigger_value,
            trigger_value_min=trigger_value_min,
            trigger_value_max=trigger_value_max,
            since=since,
            until=until,
        )

        (
            guardrail_records,
            trimmed_by_ttl,
            ttl_snapshot,
            history_size,
            _filtered_records,
        ) = self._collect_guardrail_events(
            include_errors=include_errors,
            approved_filter=approved_filter,
            normalized_filter=normalized_filter,
            service_filter=service_filter,
            decision_state_filter=decision_state_filter,
            decision_reason_filter=decision_reason_filter,
            decision_mode_filter=decision_mode_filter,
            decision_id_filter=decision_id_filter,
            since_ts=since_ts,
            until_ts=until_ts,
            reason_filter=reason_filter,
            trigger_filter=trigger_filter,
            trigger_label_filter=trigger_label_filter,
            trigger_comparator_filter=trigger_comparator_filter,
            trigger_unit_filter=trigger_unit_filter,
            trigger_threshold_filter=trigger_threshold_filter,
            trigger_threshold_min=trigger_threshold_min_value,
            trigger_threshold_max=trigger_threshold_max_value,
            trigger_value_filter=trigger_value_filter,
            trigger_value_min=trigger_value_min_value,
            trigger_value_max=trigger_value_max_value,
        )

        self._log_risk_history_trimmed(
            context="guardrail-group",
            trimmed=trimmed_by_ttl,
            ttl=ttl_snapshot,
            history=history_size,
        )

        grouped: OrderedDict[str | None, list[dict[str, Any]]] = OrderedDict()
        for entry, reasons, triggers in guardrail_records:
            normalized_decision_id = self._normalize_decision_id(entry.get("decision_id"))
            if normalized_decision_id is None and not include_unidentified:
                continue

            key = normalized_decision_id
            if key not in grouped:
                grouped[key] = []

            record = self._build_guardrail_event_record(
                entry,
                reasons,
                triggers,
                include_decision=include_decision,
                include_service=include_service,
                include_response=include_response,
                include_error=include_error,
                include_guardrail_dimensions=include_guardrail_dimensions,
                coerce_timestamps=coerce_timestamps,
                tz=tz,
            )
            record["timestamp"] = self._normalize_timestamp_for_export(
                entry.get("timestamp"),
                coerce=coerce_timestamps,
                tz=tz,
            )
            record["decision_id"] = normalized_decision_id
            grouped[key].append(record)

        return {key: tuple(values) for key, values in grouped.items()}

    def _build_guardrail_timeline(
        self,
        *,
        context: str,
        bucket_value: float,
        include_errors: bool,
        include_services: bool,
        include_guardrail_dimensions: bool,
        include_decision_dimensions: bool,
        fill_gaps: bool,
        coerce_timestamps: bool,
        tz: tzinfo | None,
        approved_filter: Any,
        normalized_filter: Any,
        service_filter: Any,
        decision_state_filter: Any,
        decision_reason_filter: Any,
        decision_mode_filter: Any,
        decision_id_filter: Any,
        reason_filter: Any,
        trigger_filter: Any,
        trigger_label_filter: Any,
        trigger_comparator_filter: Any,
        trigger_unit_filter: Any,
        trigger_threshold_filter: Any,
        trigger_threshold_min: float | None,
        trigger_threshold_max: float | None,
        trigger_value_filter: Any,
        trigger_value_min: float | None,
        trigger_value_max: float | None,
        since_ts: float | None,
        until_ts: float | None,
    ) -> dict[str, Any]:
        (
            guardrail_records,
            trimmed_by_ttl,
            ttl_snapshot,
            history_size,
            filtered_records,
        ) = self._collect_guardrail_events(
            include_errors=include_errors,
            approved_filter=approved_filter,
            normalized_filter=normalized_filter,
            service_filter=service_filter,
            decision_state_filter=decision_state_filter,
            decision_reason_filter=decision_reason_filter,
            decision_mode_filter=decision_mode_filter,
            since_ts=since_ts,
            until_ts=until_ts,
            reason_filter=reason_filter,
            trigger_filter=trigger_filter,
            trigger_label_filter=trigger_label_filter,
            trigger_comparator_filter=trigger_comparator_filter,
            trigger_unit_filter=trigger_unit_filter,
            trigger_threshold_filter=trigger_threshold_filter,
            trigger_threshold_min=trigger_threshold_min,
            trigger_threshold_max=trigger_threshold_max,
            trigger_value_filter=trigger_value_filter,
            trigger_value_min=trigger_value_min,
            trigger_value_max=trigger_value_max,
        )
        self._log_risk_history_trimmed(
            context=context,
            trimmed=trimmed_by_ttl,
            ttl=ttl_snapshot,
            history=history_size,
        )

        summary: dict[str, Any] = {
            "bucket_s": bucket_value,
            "total": len(guardrail_records),
            "evaluations": len(filtered_records),
            "buckets": [],
            "first_timestamp": None,
            "last_timestamp": None,
        }
        summary["approval_states"] = {}
        summary["normalization_states"] = {}
        summary["filters"] = self._snapshot_guardrail_timeline_filters(
            approved_filter=approved_filter,
            normalized_filter=normalized_filter,
            include_errors=include_errors,
            service_filter=service_filter,
            decision_state_filter=decision_state_filter,
            decision_reason_filter=decision_reason_filter,
            decision_mode_filter=decision_mode_filter,
            decision_id_filter=decision_id_filter,
            reason_filter=reason_filter,
            trigger_filter=trigger_filter,
            trigger_label_filter=trigger_label_filter,
            trigger_comparator_filter=trigger_comparator_filter,
            trigger_unit_filter=trigger_unit_filter,
            trigger_threshold_filter=trigger_threshold_filter,
            trigger_threshold_min=trigger_threshold_min,
            trigger_threshold_max=trigger_threshold_max,
            trigger_value_filter=trigger_value_filter,
            trigger_value_min=trigger_value_min,
            trigger_value_max=trigger_value_max,
            since_ts=since_ts,
            until_ts=until_ts,
            include_services=include_services,
            include_guardrail_dimensions=include_guardrail_dimensions,
            include_decision_dimensions=include_decision_dimensions,
            fill_gaps=fill_gaps,
            coerce_timestamps=coerce_timestamps,
            tz_value=tz,
        )
        total_approval_states: Counter[str] = Counter()
        total_normalization_states: Counter[str] = Counter()
        total_services: dict[str, dict[str, int]] | None = None
        if include_services:
            summary["services"] = {}
            total_services = {}
        total_guardrail_reasons: Counter[str] | None = None
        total_guardrail_triggers: Counter[str] | None = None
        total_guardrail_trigger_labels: Counter[str] | None = None
        total_guardrail_trigger_comparators: Counter[str] | None = None
        total_guardrail_trigger_units: Counter[str] | None = None
        if include_guardrail_dimensions:
            summary["guardrail_trigger_thresholds"] = {}
            summary["guardrail_trigger_values"] = {}
            total_guardrail_reasons = Counter()
            total_guardrail_triggers = Counter()
            total_guardrail_trigger_labels = Counter()
            total_guardrail_trigger_comparators = Counter()
            total_guardrail_trigger_units = Counter()
        total_decision_states: Counter[str] | None = None
        total_decision_reasons: Counter[str] | None = None
        total_decision_modes: Counter[str] | None = None
        if include_decision_dimensions:
            total_decision_states = Counter()
            total_decision_reasons = Counter()
            total_decision_modes = Counter()

        if not filtered_records and not guardrail_records:
            return summary

        total_threshold_stats = (
            self._init_guardrail_numeric_stats()
            if include_guardrail_dimensions
            else None
        )
        total_value_stats = (
            self._init_guardrail_numeric_stats()
            if include_guardrail_dimensions
            else None
        )

        def build_bucket() -> dict[str, Any]:
            bucket: dict[str, Any] = {
                "guardrail_events": 0,
                "evaluations": 0,
            }
            bucket["approval_states"] = Counter()
            bucket["normalization_states"] = Counter()
            if include_services:
                bucket["services"] = {}
            if include_guardrail_dimensions:
                bucket["guardrail_reasons"] = Counter()
                bucket["guardrail_triggers"] = Counter()
                bucket["guardrail_trigger_labels"] = Counter()
                bucket["guardrail_trigger_comparators"] = Counter()
                bucket["guardrail_trigger_units"] = Counter()
                bucket["guardrail_trigger_thresholds"] = (
                    self._init_guardrail_numeric_stats()
                )
                bucket["guardrail_trigger_values"] = (
                    self._init_guardrail_numeric_stats()
                )
            if include_decision_dimensions:
                bucket["decision_states"] = Counter()
                bucket["decision_reasons"] = Counter()
                bucket["decision_modes"] = Counter()
            return bucket

        bucket_map: dict[int, dict[str, Any]] = {}
        missing_bucket: dict[str, Any] | None = None
        first_ts: float | None = None
        last_ts: float | None = None

        def resolve_bucket(timestamp_value: float | None) -> dict[str, Any]:
            nonlocal first_ts, last_ts, missing_bucket
            if timestamp_value is not None:
                if not math.isfinite(timestamp_value) or math.isnan(timestamp_value):
                    timestamp_value = None

            if timestamp_value is None:
                if missing_bucket is None:
                    missing_bucket = build_bucket()
                return missing_bucket

            if first_ts is None or timestamp_value < first_ts:
                first_ts = timestamp_value
            if last_ts is None or timestamp_value > last_ts:
                last_ts = timestamp_value

            bucket_index = int(math.floor(timestamp_value / bucket_value))
            bucket_payload = bucket_map.get(bucket_index)
            if bucket_payload is None:
                bucket_payload = build_bucket()
                bucket_map[bucket_index] = bucket_payload
            return bucket_payload

        for entry in filtered_records:
            timestamp_value = self._normalize_time_bound(entry.get("timestamp"))
            bucket_payload = resolve_bucket(timestamp_value)
            bucket_payload["evaluations"] += 1
            approval_state = self._normalize_approval_flag(entry.get("approved"))
            bucket_payload.setdefault("approval_states", Counter())[approval_state] += 1
            total_approval_states[approval_state] += 1
            normalization_state = self._normalize_normalization_flag(
                entry.get("normalized")
            )
            bucket_payload.setdefault("normalization_states", Counter())[normalization_state] += 1
            total_normalization_states[normalization_state] += 1

            if include_services:
                service_key = str(entry.get("service") or _UNKNOWN_SERVICE)
                services_bucket = bucket_payload.setdefault("services", {})
                service_payload = services_bucket.get(service_key)
                if service_payload is None:
                    service_payload = {"evaluations": 0, "guardrail_events": 0}
                    services_bucket[service_key] = service_payload
                service_payload["evaluations"] += 1
                if total_services is not None:
                    service_totals = total_services.setdefault(
                        service_key,
                        {"evaluations": 0, "guardrail_events": 0},
                    )
                    service_totals["evaluations"] += 1

        for entry, reasons, triggers in guardrail_records:
            timestamp_value = self._normalize_time_bound(entry.get("timestamp"))
            bucket_payload = resolve_bucket(timestamp_value)
            bucket_payload["guardrail_events"] += 1

            if include_services:
                service_key = str(entry.get("service") or _UNKNOWN_SERVICE)
                services_bucket = bucket_payload.setdefault("services", {})
                service_payload = services_bucket.get(service_key)
                if service_payload is None:
                    service_payload = {"evaluations": 0, "guardrail_events": 0}
                    services_bucket[service_key] = service_payload
                service_payload["guardrail_events"] += 1
                if total_services is not None:
                    service_totals = total_services.setdefault(
                        service_key,
                        {"evaluations": 0, "guardrail_events": 0},
                    )
                    service_totals["guardrail_events"] += 1

            if include_guardrail_dimensions:
                reasons_counter = bucket_payload.setdefault(
                    "guardrail_reasons", Counter()
                )
                for reason_value in reasons:
                    normalized_reason = str(reason_value)
                    reasons_counter[normalized_reason] += 1
                    if total_guardrail_reasons is not None:
                        total_guardrail_reasons[normalized_reason] += 1

                triggers_counter = bucket_payload.setdefault(
                    "guardrail_triggers", Counter()
                )
                for trigger_entry in triggers:
                    trigger_name_raw = trigger_entry.get("name")
                    trigger_name = (
                        str(trigger_name_raw)
                        if trigger_name_raw is not None
                        else "<unknown>"
                    )
                    triggers_counter[trigger_name] += 1
                    if total_guardrail_triggers is not None:
                        total_guardrail_triggers[trigger_name] += 1

                labels_counter = bucket_payload.setdefault(
                    "guardrail_trigger_labels", Counter()
                )
                comparators_counter = bucket_payload.setdefault(
                    "guardrail_trigger_comparators", Counter()
                )
                units_counter = bucket_payload.setdefault(
                    "guardrail_trigger_units", Counter()
                )
                threshold_stats = bucket_payload.get(
                    "guardrail_trigger_thresholds"
                )
                value_stats = bucket_payload.get("guardrail_trigger_values")
                for trigger_entry in triggers:
                    label_raw = trigger_entry.get("label")
                    label_value = (
                        _MISSING_GUARDRAIL_LABEL
                        if label_raw is None
                        else str(label_raw)
                    )
                    labels_counter[label_value] += 1
                    if total_guardrail_trigger_labels is not None:
                        total_guardrail_trigger_labels[label_value] += 1

                    comparator_raw = trigger_entry.get("comparator")
                    comparator_value = (
                        _MISSING_GUARDRAIL_COMPARATOR
                        if comparator_raw is None
                        else str(comparator_raw)
                    )
                    comparators_counter[comparator_value] += 1
                    if total_guardrail_trigger_comparators is not None:
                        total_guardrail_trigger_comparators[
                            comparator_value
                        ] += 1

                    unit_raw = trigger_entry.get("unit")
                    unit_value = (
                        _MISSING_GUARDRAIL_UNIT
                        if unit_raw is None
                        else str(unit_raw)
                    )
                    units_counter[unit_value] += 1
                    if total_guardrail_trigger_units is not None:
                        total_guardrail_trigger_units[unit_value] += 1

                    if threshold_stats is not None:
                        self._ingest_guardrail_numeric_value(
                            threshold_stats,
                            trigger_entry.get("threshold"),
                        )
                    if value_stats is not None:
                        self._ingest_guardrail_numeric_value(
                            value_stats,
                            trigger_entry.get("value"),
                        )
                    if total_threshold_stats is not None:
                        self._ingest_guardrail_numeric_value(
                            total_threshold_stats,
                            trigger_entry.get("threshold"),
                        )
                    if total_value_stats is not None:
                        self._ingest_guardrail_numeric_value(
                            total_value_stats,
                            trigger_entry.get("value"),
                        )

            if include_decision_dimensions:
                decision_payload = entry.get("decision")
                decision_map = (
                    decision_payload
                    if isinstance(decision_payload, Mapping)
                    else {}
                )
                state_value = self._normalize_decision_dimension_value(
                    decision_map.get("state"),
                    missing_token=_MISSING_DECISION_STATE,
                )
                reason_value = self._normalize_decision_dimension_value(
                    decision_map.get("reason"),
                    missing_token=_MISSING_DECISION_REASON,
                )
                mode_value = self._normalize_decision_dimension_value(
                    decision_map.get("mode"),
                    missing_token=_MISSING_DECISION_MODE,
                )

                states_counter = bucket_payload.setdefault(
                    "decision_states", Counter()
                )
                reasons_counter = bucket_payload.setdefault(
                    "decision_reasons", Counter()
                )
                modes_counter = bucket_payload.setdefault(
                    "decision_modes", Counter()
                )

                states_counter[state_value] += 1
                if total_decision_states is not None:
                    total_decision_states[state_value] += 1
                reasons_counter[reason_value] += 1
                if total_decision_reasons is not None:
                    total_decision_reasons[reason_value] += 1
                modes_counter[mode_value] += 1
                if total_decision_modes is not None:
                    total_decision_modes[mode_value] += 1

        if fill_gaps and bucket_map:
            bucket_indices = sorted(bucket_map.keys())
            for bucket_index in range(bucket_indices[0], bucket_indices[-1] + 1):
                bucket_map.setdefault(bucket_index, build_bucket())

        buckets_output: list[dict[str, Any]] = []
        for bucket_index in sorted(bucket_map.keys()):
            bucket_payload = bucket_map[bucket_index]
            start_ts = bucket_index * bucket_value
            end_ts = start_ts + bucket_value

            bucket_summary: dict[str, Any] = {
                "index": bucket_index,
                "start": self._normalize_timestamp_for_export(
                    start_ts,
                    coerce=coerce_timestamps,
                    tz=tz,
                ),
                "end": self._normalize_timestamp_for_export(
                    end_ts,
                    coerce=coerce_timestamps,
                    tz=tz,
                ),
                "guardrail_events": bucket_payload.get("guardrail_events", 0),
                "evaluations": bucket_payload.get("evaluations", 0),
            }
            guardrail_events = bucket_summary["guardrail_events"]
            evaluations = bucket_summary["evaluations"]
            bucket_summary["guardrail_rate"] = (
                guardrail_events / evaluations if evaluations else 0.0
            )
            bucket_summary["approval_states"] = self._finalize_dimension_counter(
                bucket_payload.get("approval_states")
            )
            bucket_summary["normalization_states"] = self._finalize_dimension_counter(
                bucket_payload.get("normalization_states")
            )

            if include_services:
                services_payload = bucket_payload.get("services", {})
                bucket_summary["services"] = {
                    service_name: {
                        "evaluations": data.get("evaluations", 0),
                        "guardrail_events": data.get("guardrail_events", 0),
                    }
                    for service_name, data in sorted(services_payload.items())
                }

            if include_guardrail_dimensions:
                bucket_summary["guardrail_reasons"] = self._finalize_dimension_counter(
                    bucket_payload.get("guardrail_reasons")
                )
                bucket_summary["guardrail_triggers"] = self._finalize_dimension_counter(
                    bucket_payload.get("guardrail_triggers")
                )
                bucket_summary[
                    "guardrail_trigger_labels"
                ] = self._finalize_dimension_counter(
                    bucket_payload.get("guardrail_trigger_labels")
                )
                bucket_summary[
                    "guardrail_trigger_comparators"
                ] = self._finalize_dimension_counter(
                    bucket_payload.get("guardrail_trigger_comparators")
                )
                bucket_summary[
                    "guardrail_trigger_units"
                ] = self._finalize_dimension_counter(
                    bucket_payload.get("guardrail_trigger_units")
                )
                bucket_summary["guardrail_trigger_thresholds"] = (
                    self._finalize_guardrail_numeric_stats(
                        bucket_payload.get("guardrail_trigger_thresholds")
                    )
                )
                bucket_summary["guardrail_trigger_values"] = (
                    self._finalize_guardrail_numeric_stats(
                        bucket_payload.get("guardrail_trigger_values")
                    )
                )

            if include_decision_dimensions:
                bucket_summary["decision_states"] = self._finalize_dimension_counter(
                    bucket_payload.get("decision_states")
                )
                bucket_summary["decision_reasons"] = self._finalize_dimension_counter(
                    bucket_payload.get("decision_reasons")
                )
                bucket_summary["decision_modes"] = self._finalize_dimension_counter(
                    bucket_payload.get("decision_modes")
                )

            buckets_output.append(bucket_summary)

        summary["buckets"] = buckets_output
        summary["first_timestamp"] = first_ts
        summary["last_timestamp"] = last_ts
        total_guardrail_events = summary.get("total", 0) or 0
        total_evaluations = summary.get("evaluations", 0) or 0
        summary["guardrail_rate"] = (
            total_guardrail_events / total_evaluations
            if total_evaluations
            else 0.0
        )
        summary["approval_states"] = self._finalize_dimension_counter(
            total_approval_states
        )
        summary["normalization_states"] = self._finalize_dimension_counter(
            total_normalization_states
        )
        if total_services is not None:
            summary["services"] = {
                service_name: {
                    "evaluations": totals.get("evaluations", 0),
                    "guardrail_events": totals.get("guardrail_events", 0),
                }
                for service_name, totals in sorted(total_services.items())
            }
        if include_guardrail_dimensions:
            summary["guardrail_trigger_thresholds"] = (
                self._finalize_guardrail_numeric_stats(total_threshold_stats)
            )
            summary["guardrail_trigger_values"] = (
                self._finalize_guardrail_numeric_stats(total_value_stats)
            )
            summary["guardrail_reasons"] = self._finalize_dimension_counter(
                total_guardrail_reasons
            )
            summary["guardrail_triggers"] = self._finalize_dimension_counter(
                total_guardrail_triggers
            )
            summary["guardrail_trigger_labels"] = self._finalize_dimension_counter(
                total_guardrail_trigger_labels
            )
            summary["guardrail_trigger_comparators"] = (
                self._finalize_dimension_counter(total_guardrail_trigger_comparators)
            )
            summary["guardrail_trigger_units"] = self._finalize_dimension_counter(
                total_guardrail_trigger_units
            )

        if include_decision_dimensions:
            summary["decision_states"] = self._finalize_dimension_counter(
                total_decision_states
            )
            summary["decision_reasons"] = self._finalize_dimension_counter(
                total_decision_reasons
            )
            summary["decision_modes"] = self._finalize_dimension_counter(
                total_decision_modes
            )

        if missing_bucket is not None and (
            missing_bucket.get("guardrail_events", 0)
            or missing_bucket.get("evaluations", 0)
        ):
            missing_summary: dict[str, Any] = {
                "index": None,
                "start": None,
                "end": None,
                "guardrail_events": missing_bucket.get("guardrail_events", 0),
                "evaluations": missing_bucket.get("evaluations", 0),
            }
            missing_guardrail_events = missing_summary["guardrail_events"]
            missing_evaluations = missing_summary["evaluations"]
            missing_summary["guardrail_rate"] = (
                missing_guardrail_events / missing_evaluations
                if missing_evaluations
                else 0.0
            )
            missing_summary["approval_states"] = self._finalize_dimension_counter(
                missing_bucket.get("approval_states")
            )
            missing_summary["normalization_states"] = self._finalize_dimension_counter(
                missing_bucket.get("normalization_states")
            )
            if include_services:
                services_payload = missing_bucket.get("services", {})
                missing_summary["services"] = {
                    service_name: {
                        "evaluations": data.get("evaluations", 0),
                        "guardrail_events": data.get("guardrail_events", 0),
                    }
                    for service_name, data in sorted(services_payload.items())
                }
            if include_guardrail_dimensions:
                missing_summary["guardrail_reasons"] = self._finalize_dimension_counter(
                    missing_bucket.get("guardrail_reasons")
                )
                missing_summary["guardrail_triggers"] = self._finalize_dimension_counter(
                    missing_bucket.get("guardrail_triggers")
                )
                missing_summary[
                    "guardrail_trigger_labels"
                ] = self._finalize_dimension_counter(
                    missing_bucket.get("guardrail_trigger_labels")
                )
                missing_summary[
                    "guardrail_trigger_comparators"
                ] = self._finalize_dimension_counter(
                    missing_bucket.get("guardrail_trigger_comparators")
                )
                missing_summary[
                    "guardrail_trigger_units"
                ] = self._finalize_dimension_counter(
                    missing_bucket.get("guardrail_trigger_units")
                )
                missing_summary["guardrail_trigger_thresholds"] = (
                    self._finalize_guardrail_numeric_stats(
                        missing_bucket.get("guardrail_trigger_thresholds")
                    )
                )
                missing_summary["guardrail_trigger_values"] = (
                    self._finalize_guardrail_numeric_stats(
                        missing_bucket.get("guardrail_trigger_values")
                    )
                )
            if include_decision_dimensions:
                missing_summary["decision_states"] = self._finalize_dimension_counter(
                    missing_bucket.get("decision_states")
                )
                missing_summary["decision_reasons"] = self._finalize_dimension_counter(
                    missing_bucket.get("decision_reasons")
                )
                missing_summary["decision_modes"] = self._finalize_dimension_counter(
                    missing_bucket.get("decision_modes")
                )
            summary["missing_timestamp"] = missing_summary

        return summary

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

    def summarize_guardrail_timeline(
        self,
        *,
        bucket_s: float,
        approved: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        normalized: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        include_errors: bool = True,
        service: str | None | Iterable[str | None] | object = _NO_FILTER,
        decision_state: str | Iterable[str | None] | object = _NO_FILTER,
        decision_reason: str | Iterable[str | None] | object = _NO_FILTER,
        decision_mode: str | Iterable[str | None] | object = _NO_FILTER,
        decision_id: str | Iterable[str | None] | object = _NO_FILTER,
        reason: str | Iterable[str] | object = _NO_FILTER,
        trigger: str | Iterable[str] | object = _NO_FILTER,
        trigger_label: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_comparator: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_unit: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_threshold: float | None | Iterable[float | None] | object = _NO_FILTER,
        trigger_threshold_min: Any = None,
        trigger_threshold_max: Any = None,
        trigger_value: float | None | Iterable[float | None] | object = _NO_FILTER,
        trigger_value_min: Any = None,
        trigger_value_max: Any = None,
        since: Any = None,
        until: Any = None,
        include_services: bool = True,
        include_guardrail_dimensions: bool = True,
        include_decision_dimensions: bool = False,
        fill_gaps: bool = False,
        coerce_timestamps: bool = False,
        tz: tzinfo | None = timezone.utc,
    ) -> dict[str, Any]:
        try:
            bucket_value = float(bucket_s)
        except (TypeError, ValueError) as exc:  # pragma: no cover - walidacja wejścia
            raise ValueError("bucket_s must be a positive number") from exc
        if not math.isfinite(bucket_value) or bucket_value <= 0.0:
            raise ValueError("bucket_s must be a positive number")

        approved_filter = self._prepare_bool_filter(approved)
        normalized_filter = self._prepare_bool_filter(normalized)
        service_filter = self._prepare_service_filter(service)
        reason_filter = self._prepare_string_filter(reason)
        trigger_filter = self._prepare_string_filter(trigger)
        trigger_label_filter = self._prepare_guardrail_filter(
            trigger_label,
            missing_token=_MISSING_GUARDRAIL_LABEL,
        )
        trigger_comparator_filter = self._prepare_guardrail_filter(
            trigger_comparator,
            missing_token=_MISSING_GUARDRAIL_COMPARATOR,
        )
        trigger_unit_filter = self._prepare_guardrail_filter(
            trigger_unit,
            missing_token=_MISSING_GUARDRAIL_UNIT,
        )
        trigger_threshold_filter = self._prepare_guardrail_numeric_filter(
            trigger_threshold
        )
        trigger_value_filter = self._prepare_guardrail_numeric_filter(trigger_value)
        decision_state_filter = self._prepare_decision_filter(
            decision_state,
            missing_token=_MISSING_DECISION_STATE,
        )
        decision_reason_filter = self._prepare_decision_filter(
            decision_reason,
            missing_token=_MISSING_DECISION_REASON,
        )
        decision_mode_filter = self._prepare_decision_filter(
            decision_mode,
            missing_token=_MISSING_DECISION_MODE,
        )
        decision_id_filter = self._prepare_decision_filter(
            decision_id,
            missing_token=_MISSING_DECISION_ID,
        )

        trigger_threshold_min_value = (
            self._coerce_float(trigger_threshold_min)
            if trigger_threshold_min is not None
            else None
        )
        trigger_threshold_max_value = (
            self._coerce_float(trigger_threshold_max)
            if trigger_threshold_max is not None
            else None
        )
        trigger_value_min_value = (
            self._coerce_float(trigger_value_min)
            if trigger_value_min is not None
            else None
        )
        trigger_value_max_value = (
            self._coerce_float(trigger_value_max)
            if trigger_value_max is not None
            else None
        )

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
            context="guardrail-timeline",
            trimmed=trimmed_by_ttl,
            ttl=ttl_snapshot,
            history=history_size,
        )

        return self._build_guardrail_timeline(
            context="guardrail-timeline",
            bucket_value=bucket_value,
            include_errors=include_errors,
            include_services=include_services,
            include_guardrail_dimensions=include_guardrail_dimensions,
            include_decision_dimensions=include_decision_dimensions,
            fill_gaps=fill_gaps,
            coerce_timestamps=coerce_timestamps,
            tz=tz,
            approved_filter=approved_filter,
            normalized_filter=normalized_filter,
            service_filter=service_filter,
            decision_state_filter=decision_state_filter,
            decision_reason_filter=decision_reason_filter,
            decision_mode_filter=decision_mode_filter,
            decision_id_filter=decision_id_filter,
            reason_filter=reason_filter,
            trigger_filter=trigger_filter,
            trigger_label_filter=trigger_label_filter,
            trigger_comparator_filter=trigger_comparator_filter,
            trigger_unit_filter=trigger_unit_filter,
            trigger_threshold_filter=trigger_threshold_filter,
            trigger_threshold_min=trigger_threshold_min_value,
            trigger_threshold_max=trigger_threshold_max_value,
            trigger_value_filter=trigger_value_filter,
            trigger_value_min=trigger_value_min_value,
            trigger_value_max=trigger_value_max_value,
            since_ts=since_ts,
            until_ts=until_ts,
        )

    def guardrail_timeline_to_records(
        self,
        *,
        bucket_s: float,
        approved: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        normalized: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        include_errors: bool = True,
        service: str | None | Iterable[str | None] | object = _NO_FILTER,
        decision_state: str | Iterable[str | None] | object = _NO_FILTER,
        decision_reason: str | Iterable[str | None] | object = _NO_FILTER,
        decision_mode: str | Iterable[str | None] | object = _NO_FILTER,
        decision_id: str | Iterable[str | None] | object = _NO_FILTER,
        reason: str | Iterable[str] | object = _NO_FILTER,
        trigger: str | Iterable[str] | object = _NO_FILTER,
        trigger_label: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_comparator: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_unit: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_threshold: float | None | Iterable[float | None] | object = _NO_FILTER,
        trigger_threshold_min: Any = None,
        trigger_threshold_max: Any = None,
        trigger_value: float | None | Iterable[float | None] | object = _NO_FILTER,
        trigger_value_min: Any = None,
        trigger_value_max: Any = None,
        since: Any = None,
        until: Any = None,
        include_services: bool = True,
        include_guardrail_dimensions: bool = True,
        include_decision_dimensions: bool = False,
        fill_gaps: bool = False,
        include_missing_bucket: bool = False,
        coerce_timestamps: bool = False,
        tz: tzinfo | None = timezone.utc,
        include_summary_metadata: bool = False,
    ) -> list[dict[str, Any]]:
        summary = self.summarize_guardrail_timeline(
            bucket_s=bucket_s,
            approved=approved,
            normalized=normalized,
            include_errors=include_errors,
            service=service,
            decision_state=decision_state,
            decision_reason=decision_reason,
            decision_mode=decision_mode,
            decision_id=decision_id,
            reason=reason,
            trigger=trigger,
            trigger_label=trigger_label,
            trigger_comparator=trigger_comparator,
            trigger_unit=trigger_unit,
            trigger_threshold=trigger_threshold,
            trigger_threshold_min=trigger_threshold_min,
            trigger_threshold_max=trigger_threshold_max,
            trigger_value=trigger_value,
            trigger_value_min=trigger_value_min,
            trigger_value_max=trigger_value_max,
            since=since,
            until=until,
            include_services=include_services,
            include_guardrail_dimensions=include_guardrail_dimensions,
            include_decision_dimensions=include_decision_dimensions,
            fill_gaps=fill_gaps,
            coerce_timestamps=coerce_timestamps,
            tz=tz,
        )

        records: list[dict[str, Any]] = []
        for bucket in summary.get("buckets", []):
            bucket_record = copy.deepcopy(bucket)
            bucket_record.setdefault("bucket_type", "bucket")
            if not include_services:
                bucket_record.pop("services", None)
            if not include_guardrail_dimensions:
                for key in (
                    "guardrail_reasons",
                    "guardrail_triggers",
                    "guardrail_trigger_labels",
                    "guardrail_trigger_comparators",
                    "guardrail_trigger_units",
                    "guardrail_trigger_thresholds",
                    "guardrail_trigger_values",
                ):
                    bucket_record.pop(key, None)
            if not include_decision_dimensions:
                for key in ("decision_states", "decision_reasons", "decision_modes"):
                    bucket_record.pop(key, None)
            records.append(bucket_record)

        if include_missing_bucket:
            missing_entry = summary.get("missing_timestamp")
            if isinstance(missing_entry, Mapping) and (
                missing_entry.get("guardrail_events")
                or missing_entry.get("evaluations")
            ):
                missing_record = copy.deepcopy(missing_entry)
                missing_record.setdefault("bucket_type", "missing")
                missing_record.setdefault("index", None)
                missing_record["start"] = None
                missing_record["end"] = None
                if not include_services:
                    missing_record.pop("services", None)
                if not include_guardrail_dimensions:
                    for key in (
                        "guardrail_reasons",
                        "guardrail_triggers",
                        "guardrail_trigger_labels",
                        "guardrail_trigger_comparators",
                        "guardrail_trigger_units",
                        "guardrail_trigger_thresholds",
                        "guardrail_trigger_values",
                    ):
                        missing_record.pop(key, None)
                if not include_decision_dimensions:
                    for key in ("decision_states", "decision_reasons", "decision_modes"):
                        missing_record.pop(key, None)
                records.append(missing_record)

        if include_summary_metadata:
            summary_record = {
                key: copy.deepcopy(value)
                for key, value in summary.items()
                if key != "buckets"
            }
            summary_record.setdefault("bucket_type", "summary")
            summary_record.setdefault("index", None)
            summary_record.setdefault("start", None)
            summary_record.setdefault("end", None)
            summary_record.setdefault("guardrail_events", summary.get("total", 0))
            if not include_services:
                summary_record.pop("services", None)
            if not include_guardrail_dimensions:
                for key in (
                    "guardrail_reasons",
                    "guardrail_triggers",
                    "guardrail_trigger_labels",
                    "guardrail_trigger_comparators",
                    "guardrail_trigger_units",
                    "guardrail_trigger_thresholds",
                    "guardrail_trigger_values",
                ):
                    summary_record.pop(key, None)
            if not include_decision_dimensions:
                for key in ("decision_states", "decision_reasons", "decision_modes"):
                    summary_record.pop(key, None)
            if coerce_timestamps:
                for timestamp_key in ("first_timestamp", "last_timestamp"):
                    if timestamp_key in summary_record:
                        summary_record[timestamp_key] = (
                            self._normalize_timestamp_for_export(
                                summary_record[timestamp_key],
                                coerce=True,
                                tz=tz,
                            )
                        )
            records.append(summary_record)

        metadata = _extract_guardrail_timeline_metadata(summary)
        return GuardrailTimelineRecords(records, metadata)

    def guardrail_timeline_to_dataframe(
        self,
        *,
        bucket_s: float,
        approved: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        normalized: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        include_errors: bool = True,
        service: str | None | Iterable[str | None] | object = _NO_FILTER,
        decision_state: str | Iterable[str | None] | object = _NO_FILTER,
        decision_reason: str | Iterable[str | None] | object = _NO_FILTER,
        decision_mode: str | Iterable[str | None] | object = _NO_FILTER,
        decision_id: str | Iterable[str | None] | object = _NO_FILTER,
        reason: str | Iterable[str] | object = _NO_FILTER,
        trigger: str | Iterable[str] | object = _NO_FILTER,
        trigger_label: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_comparator: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_unit: str | Iterable[str | None] | object = _NO_FILTER,
        trigger_threshold: float | None | Iterable[float | None] | object = _NO_FILTER,
        trigger_threshold_min: Any = None,
        trigger_threshold_max: Any = None,
        trigger_value: float | None | Iterable[float | None] | object = _NO_FILTER,
        trigger_value_min: Any = None,
        trigger_value_max: Any = None,
        since: Any = None,
        until: Any = None,
        include_services: bool = False,
        include_guardrail_dimensions: bool = False,
        include_decision_dimensions: bool = False,
        fill_gaps: bool = False,
        include_missing_bucket: bool = False,
        coerce_timestamps: bool = True,
        tz: tzinfo | None = timezone.utc,
        include_summary_metadata: bool = False,
    ) -> pd.DataFrame:
        records = self.guardrail_timeline_to_records(
            bucket_s=bucket_s,
            approved=approved,
            normalized=normalized,
            include_errors=include_errors,
            service=service,
            decision_state=decision_state,
            decision_reason=decision_reason,
            decision_mode=decision_mode,
            decision_id=decision_id,
            reason=reason,
            trigger=trigger,
            trigger_label=trigger_label,
            trigger_comparator=trigger_comparator,
            trigger_unit=trigger_unit,
            trigger_threshold=trigger_threshold,
            trigger_threshold_min=trigger_threshold_min,
            trigger_threshold_max=trigger_threshold_max,
            trigger_value=trigger_value,
            trigger_value_min=trigger_value_min,
            trigger_value_max=trigger_value_max,
            since=since,
            until=until,
            include_services=True,
            include_guardrail_dimensions=True,
            include_decision_dimensions=True,
            fill_gaps=fill_gaps,
            include_missing_bucket=include_missing_bucket,
            coerce_timestamps=coerce_timestamps,
            tz=tz,
            include_summary_metadata=include_summary_metadata,
        )

        summary_metadata = getattr(records, "summary", None)

        if not records:
            base_columns = [
                "index",
                "start",
                "end",
                "guardrail_events",
                "evaluations",
                "guardrail_rate",
                "bucket_type",
            ]
            optional_columns: list[tuple[bool, str]] = [
                (include_services, "services"),
                (include_guardrail_dimensions, "guardrail_reasons"),
                (include_guardrail_dimensions, "guardrail_triggers"),
                (include_guardrail_dimensions, "guardrail_trigger_labels"),
                (include_guardrail_dimensions, "guardrail_trigger_comparators"),
                (include_guardrail_dimensions, "guardrail_trigger_units"),
                (include_guardrail_dimensions, "guardrail_trigger_thresholds"),
                (include_guardrail_dimensions, "guardrail_trigger_values"),
                (include_decision_dimensions, "decision_states"),
                (include_decision_dimensions, "decision_reasons"),
                (include_decision_dimensions, "decision_modes"),
            ]
            for include_flag, column_name in optional_columns:
                if include_flag:
                    base_columns.append(column_name)
            df = pd.DataFrame(columns=base_columns)
        else:
            drop_columns: set[str] = set()
            if not include_services:
                drop_columns.add("services")
            if not include_guardrail_dimensions:
                drop_columns.update(
                    {
                        "guardrail_reasons",
                        "guardrail_triggers",
                        "guardrail_trigger_labels",
                        "guardrail_trigger_comparators",
                        "guardrail_trigger_units",
                        "guardrail_trigger_thresholds",
                        "guardrail_trigger_values",
                    }
                )
            if not include_decision_dimensions:
                drop_columns.update(
                    {"decision_states", "decision_reasons", "decision_modes"}
                )

            if drop_columns:
                sanitized_records = [
                    {
                        key: copy.deepcopy(value)
                        for key, value in record.items()
                        if key not in drop_columns
                    }
                    for record in records
                ]
            else:
                sanitized_records = [copy.deepcopy(record) for record in records]

            df = pd.DataFrame(sanitized_records)

        if summary_metadata is not None and (
            include_services
            and include_guardrail_dimensions
            and include_decision_dimensions
        ):
            df.attrs["guardrail_summary"] = copy.deepcopy(summary_metadata)
        else:
            df.attrs["guardrail_summary"] = self.summarize_guardrail_timeline(
                bucket_s=bucket_s,
                approved=approved,
                normalized=normalized,
                include_errors=include_errors,
                service=service,
                decision_state=decision_state,
                decision_reason=decision_reason,
                decision_mode=decision_mode,
                decision_id=decision_id,
                reason=reason,
                trigger=trigger,
                trigger_label=trigger_label,
                trigger_comparator=trigger_comparator,
                trigger_unit=trigger_unit,
                trigger_threshold=trigger_threshold,
                trigger_threshold_min=trigger_threshold_min,
                trigger_threshold_max=trigger_threshold_max,
                trigger_value=trigger_value,
                trigger_value_min=trigger_value_min,
                trigger_value_max=trigger_value_max,
                since=since,
                until=until,
                include_services=True,
                include_guardrail_dimensions=True,
                include_decision_dimensions=True,
                fill_gaps=fill_gaps,
                coerce_timestamps=coerce_timestamps,
                tz=tz,
            )

        return df

    def build_lifecycle_snapshot(
        self,
        *,
        bucket_s: float = 3600.0,
        tz: tzinfo | None = timezone.utc,
    ) -> dict[str, Any]:
        tzinfo = tz or timezone.utc
        now_dt = datetime.now(tzinfo)
        try:
            symbol = self.symbol_getter()
        except Exception:  # pragma: no cover - defensywne logowanie
            LOGGER.debug("Symbol getter failed during lifecycle snapshot", exc_info=True)
            symbol = "<unknown>"

        schedule_state = self.get_schedule_state()
        schedule_payload = _serialize_schedule_state(schedule_state)

        try:
            guardrail_summary = self.summarize_guardrail_timeline(
                bucket_s=bucket_s,
                include_errors=True,
                include_services=True,
                include_guardrail_dimensions=True,
                include_decision_dimensions=True,
                coerce_timestamps=True,
                tz=tzinfo,
            )
        except Exception:  # pragma: no cover - defensywne logowanie
            LOGGER.debug("Guardrail timeline summary failed", exc_info=True)
            guardrail_summary = self._fallback_guardrail_summary()

        try:
            decision_summary = self.summarize_risk_decision_timeline(
                bucket_s=bucket_s,
                include_errors=True,
                include_services=True,
                include_decision_dimensions=True,
                coerce_timestamps=True,
                tz=tzinfo,
            )
        except Exception:  # pragma: no cover - defensywne logowanie
            LOGGER.debug("Decision timeline summary failed", exc_info=True)
            decision_summary = self._fallback_decision_summary()

        current_strategy = self.current_strategy
        current_leverage = self.current_leverage
        current_stop_loss = self.current_stop_loss_pct
        current_take_profit = self.current_take_profit_pct

        with self._lock:
            last_guardrail_reasons = list(self._last_guardrail_reasons)
            last_guardrail_triggers = [trigger.to_dict() for trigger in self._last_guardrail_triggers]
            last_decision = (
                self._last_risk_decision.to_dict()
                if self._last_risk_decision is not None
                else None
            )
            last_regime = (
                self._last_regime.to_dict()
                if self._last_regime is not None
                else None
            )
            last_signal = self._last_signal
            ai_context = (
                copy.deepcopy(self._last_ai_context)
                if self._last_ai_context is not None
                else None
            )
            ai_degraded = bool(self._ai_degraded)
            cooldown_until = self._cooldown_until
            cooldown_reason = self._cooldown_reason
            controller_history = [copy.deepcopy(entry) for entry in self._controller_cycle_history]
            controller_summary = {
                "sequence": self._controller_cycle_sequence,
                "last_duration_s": self._controller_cycle_last_duration,
                "last_orders": self._controller_cycle_last_orders,
                "history": controller_history,
            }
            risk_history_size = len(self._risk_evaluations)
            auto_trade_state = {
                "active": bool(self._auto_trade_thread_active),
                "user_confirmed": bool(self._auto_trade_user_confirmed),
                "trusted_auto_confirm": bool(self._trusted_auto_confirm),
                "started": bool(self._started),
            }
            schedule_last_alert = self._schedule_last_alert_state
            current_strategy = self.current_strategy
            current_leverage = self.current_leverage
            current_stop_loss = self.current_stop_loss_pct
            current_take_profit = self.current_take_profit_pct

        now_ts = time.time()
        if cooldown_until and cooldown_until > now_ts:
            cooldown_remaining = cooldown_until - now_ts
        else:
            cooldown_remaining = 0.0
        cooldown_dt = self._ensure_datetime(cooldown_until, tzinfo)
        cooldown_until_iso = cooldown_dt.isoformat() if cooldown_dt is not None else None

        metrics_labels = self._base_metric_labels
        cycles_total = self._metric_cycle_total.value(labels=metrics_labels)
        strategy_switch_total = self._metric_strategy_switch_total.value(labels=metrics_labels)
        guardrail_blocks_total = 0.0
        guardrail_values = getattr(self._metric_guardrail_blocks_total, "_values", {})
        for label_tuple, value in getattr(guardrail_values, "items", lambda: [])():
            labels = dict(label_tuple)
            if all(labels.get(key) == val for key, val in metrics_labels.items()):
                guardrail_blocks_total += float(value)
        recalibration_total = self._metric_recalibration_total.value(labels=metrics_labels)
        schedule_open_value = self._metric_schedule_open_gauge.value(labels=metrics_labels)
        strategy_gauge_value = self._metric_strategy_state_gauge.value(labels=metrics_labels)
        histogram_state = self._metric_schedule_closed_seconds.snapshot(labels=metrics_labels)
        histogram_buckets: dict[str, int] = {}
        for boundary, count in sorted(histogram_state.counts.items(), key=lambda item: item[0]):
            if math.isinf(boundary):
                key = "+inf"
            else:
                key = str(boundary)
            histogram_buckets[key] = int(count)

        metrics_snapshot = {
            "cycles_total": cycles_total,
            "strategy_switch_total": strategy_switch_total,
            "guardrail_blocks_total": guardrail_blocks_total,
            "recalibration_total": recalibration_total,
            "schedule_open": schedule_open_value,
            "strategy_active": strategy_gauge_value,
            "schedule_block_histogram": {
                "sum": histogram_state.sum,
                "count": histogram_state.count,
                "buckets": histogram_buckets,
            },
        }

        recalibrations: list[dict[str, Any]] = []
        orchestrator = getattr(self, "decision_orchestrator", None)
        due_recalibrations = getattr(orchestrator, "due_recalibrations", None)
        if callable(due_recalibrations):
            try:
                for item in due_recalibrations():
                    strategy = getattr(item, "strategy", None) or getattr(item, "name", None)
                    next_run = getattr(item, "next_run", None)
                    payload: dict[str, Any] = {"strategy": strategy}
                    if next_run is not None:
                        payload["next_run"] = self._normalize_timestamp_for_export(
                            next_run,
                            coerce=False,
                            tz=tzinfo,
                        )
                    recalibrations.append(payload)
            except Exception:  # pragma: no cover - defensywne logowanie
                self._log(
                    "DecisionOrchestrator.due_recalibrations snapshot failed",
                    level=logging.DEBUG,
                )

        strategy_metadata, strategy_summary = self._strategy_metadata_bundle(current_strategy)
        strategy_section: dict[str, Any] = {
            "current": current_strategy,
            "leverage": current_leverage,
            "stop_loss_pct": current_stop_loss,
            "take_profit_pct": current_take_profit,
            "last_signal": last_signal,
            "last_regime": last_regime,
        }
        if strategy_metadata:
            strategy_section["metadata"] = strategy_metadata
        if strategy_summary:
            strategy_section["metadata_summary"] = strategy_summary

        snapshot = {
            "timestamp": now_dt.isoformat(),
            "symbol": symbol,
            "environment": self._environment_name,
            "portfolio": self._portfolio_id,
            "risk_profile": self._risk_profile_name,
            "schedule": schedule_payload,
            "strategy": strategy_section,
            "guardrails": {
                "summary": guardrail_summary,
                "last_reasons": last_guardrail_reasons,
                "last_triggers": last_guardrail_triggers,
            },
            "risk_decisions": {
                "summary": decision_summary,
                "last_decision": last_decision,
                "history_size": risk_history_size,
                "thresholds": self._decision_threshold_snapshot(),
            },
            "ai": {
                "degraded": ai_degraded,
                "context": ai_context,
                "threshold_bps": getattr(
                    getattr(self, "ai_manager", None),
                    "ai_threshold_bps",
                    None,
                ),
            },
            "controller": {
                **controller_summary,
                "auto_trade": auto_trade_state,
                "schedule_last_alert": schedule_last_alert,
            },
            "cooldown": {
                "active": cooldown_remaining > 0.0,
                "remaining_s": cooldown_remaining if cooldown_remaining > 0.0 else 0.0,
                "reason": cooldown_reason,
                "until": cooldown_until_iso,
            },
            "metrics": metrics_snapshot,
            "recalibrations": recalibrations,
        }

        feature_metadata = self._feature_column_metadata()
        if feature_metadata:
            risk_decision_section = snapshot["risk_decisions"]
            for key, value in feature_metadata.items():
                risk_decision_section[key] = copy.deepcopy(value)

        return snapshot

    def risk_evaluations_to_dataframe(
        self,
        *,
        approved: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        normalized: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        include_errors: bool = True,
        service: str | None | Iterable[str | None] | object = _NO_FILTER,
        decision_state: str | Iterable[str | None] | object = _NO_FILTER,
        decision_reason: str | Iterable[str | None] | object = _NO_FILTER,
        decision_mode: str | Iterable[str | None] | object = _NO_FILTER,
        decision_id: str | Iterable[str | None] | object = _NO_FILTER,
        since: Any = None,
        until: Any = None,
        flatten_decision: bool = False,
        decision_prefix: str = "decision_",
        decision_fields: Iterable[Any] | Any | None = None,
        drop_decision_column: bool = False,
        fill_value: Any = pd.NA,
        coerce_timestamps: bool = False,
        tz: tzinfo | None = timezone.utc,
    ) -> pd.DataFrame:
        """Return risk evaluations as a pandas DataFrame with optional filters."""

        (
            approved_filter,
            normalized_filter,
            service_filter,
            decision_state_filter,
            decision_reason_filter,
            decision_mode_filter,
            decision_id_filter,
            normalized_decision_fields,
            since_ts,
            until_ts,
        ) = self._resolve_risk_evaluation_filters(
            approved=approved,
            normalized=normalized,
            service=service,
            decision_state=decision_state,
            decision_reason=decision_reason,
            decision_mode=decision_mode,
            decision_id=decision_id,
            since=since,
            until=until,
            decision_fields=decision_fields,
        )

        (
            filtered_records,
            trimmed_by_ttl,
            ttl_snapshot,
            history_size,
        ) = self._collect_filtered_risk_evaluations(
            include_errors=include_errors,
            approved_filter=approved_filter,
            normalized_filter=normalized_filter,
            service_filter=service_filter,
            since_ts=since_ts,
            until_ts=until_ts,
            state_filter=decision_state_filter,
            reason_filter=decision_reason_filter,
            mode_filter=decision_mode_filter,
            decision_id_filter=decision_id_filter,
        )

        self._log_risk_history_trimmed(
            context="dataframe",
            trimmed=trimmed_by_ttl,
            ttl=ttl_snapshot,
            history=history_size,
        )

        base_columns = [
            "timestamp",
            "approved",
            "normalized",
            "decision_id",
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

        if coerce_timestamps and "timestamp" in df.columns:
            df["timestamp"] = [
                self._normalize_timestamp_for_export(value, coerce=True, tz=tz)
                for value in df["timestamp"].tolist()
            ]

        flattened_columns: list[str] = []
        if flatten_decision and "decision" in df.columns:
            prefix = str(decision_prefix)
            decision_series = df["decision"]
            ordered_keys: list[Any]
            if normalized_decision_fields is not None:
                ordered_keys = list(normalized_decision_fields)
            else:
                ordered_keys = []
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

    def risk_evaluations_to_records(
        self,
        *,
        approved: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        normalized: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        include_errors: bool = True,
        service: str | None | Iterable[str | None] | object = _NO_FILTER,
        decision_state: str | Iterable[str | None] | object = _NO_FILTER,
        decision_reason: str | Iterable[str | None] | object = _NO_FILTER,
        decision_mode: str | Iterable[str | None] | object = _NO_FILTER,
        decision_id: str | Iterable[str | None] | object = _NO_FILTER,
        since: Any = None,
        until: Any = None,
        flatten_decision: bool = False,
        decision_prefix: str = "decision_",
        decision_fields: Iterable[Any] | Any | None = None,
        drop_decision_column: bool = False,
        fill_value: Any = pd.NA,
        coerce_timestamps: bool = False,
        tz: tzinfo | None = timezone.utc,
    ) -> list[dict[str, Any]]:
        """Eksportuje historię ocen ryzyka jako listę słowników."""

        (
            approved_filter,
            normalized_filter,
            service_filter,
            decision_state_filter,
            decision_reason_filter,
            decision_mode_filter,
            decision_id_filter,
            normalized_decision_fields,
            since_ts,
            until_ts,
        ) = self._resolve_risk_evaluation_filters(
            approved=approved,
            normalized=normalized,
            service=service,
            decision_state=decision_state,
            decision_reason=decision_reason,
            decision_mode=decision_mode,
            decision_id=decision_id,
            since=since,
            until=until,
            decision_fields=decision_fields,
        )

        (
            filtered_records,
            trimmed_by_ttl,
            ttl_snapshot,
            history_size,
        ) = self._collect_filtered_risk_evaluations(
            include_errors=include_errors,
            approved_filter=approved_filter,
            normalized_filter=normalized_filter,
            service_filter=service_filter,
            since_ts=since_ts,
            until_ts=until_ts,
            state_filter=decision_state_filter,
            reason_filter=decision_reason_filter,
            mode_filter=decision_mode_filter,
            decision_id_filter=decision_id_filter,
        )
        self._log_risk_history_trimmed(
            context="records",
            trimmed=trimmed_by_ttl,
            ttl=ttl_snapshot,
            history=history_size,
        )

        return self._build_risk_evaluation_records(
            filtered_records,
            normalized_decision_fields=normalized_decision_fields,
            flatten_decision=flatten_decision,
            decision_prefix=decision_prefix,
            drop_decision_column=drop_decision_column,
            fill_value=fill_value,
            coerce_timestamps=coerce_timestamps,
            tz=tz,
        )

    def get_grouped_risk_evaluations(
        self,
        *,
        approved: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        normalized: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        include_errors: bool = True,
        service: str | None | Iterable[str | None] | object = _NO_FILTER,
        decision_state: str | Iterable[str | None] | object = _NO_FILTER,
        decision_reason: str | Iterable[str | None] | object = _NO_FILTER,
        decision_mode: str | Iterable[str | None] | object = _NO_FILTER,
        decision_id: str | Iterable[str | None] | object = _NO_FILTER,
        since: Any = None,
        until: Any = None,
        include_unidentified: bool = False,
        coerce_timestamps: bool = True,
        tz: tzinfo | None = timezone.utc,
    ) -> Mapping[str | None, Sequence[Mapping[str, Any]]]:
        """Zwraca wpisy historii ryzyka pogrupowane wg identyfikatora decyzji."""

        (
            approved_filter,
            normalized_filter,
            service_filter,
            decision_state_filter,
            decision_reason_filter,
            decision_mode_filter,
            decision_id_filter,
            _,
            since_ts,
            until_ts,
        ) = self._resolve_risk_evaluation_filters(
            approved=approved,
            normalized=normalized,
            service=service,
            decision_state=decision_state,
            decision_reason=decision_reason,
            decision_mode=decision_mode,
            decision_id=decision_id,
            since=since,
            until=until,
            decision_fields=None,
        )

        (
            filtered_records,
            trimmed_by_ttl,
            ttl_snapshot,
            history_size,
        ) = self._collect_filtered_risk_evaluations(
            include_errors=include_errors,
            approved_filter=approved_filter,
            normalized_filter=normalized_filter,
            service_filter=service_filter,
            since_ts=since_ts,
            until_ts=until_ts,
            state_filter=decision_state_filter,
            reason_filter=decision_reason_filter,
            mode_filter=decision_mode_filter,
            decision_id_filter=decision_id_filter,
        )

        self._log_risk_history_trimmed(
            context="group",
            trimmed=trimmed_by_ttl,
            ttl=ttl_snapshot,
            history=history_size,
        )

        grouped: OrderedDict[str | None, list[dict[str, Any]]] = OrderedDict()
        for entry in filtered_records:
            normalized_decision_id = self._normalize_decision_id(entry.get("decision_id"))
            if normalized_decision_id is None and not include_unidentified:
                continue

            key = normalized_decision_id
            if key not in grouped:
                grouped[key] = []

            record = copy.deepcopy(entry)
            record["decision_id"] = normalized_decision_id
            record["timestamp"] = self._normalize_timestamp_for_export(
                record.get("timestamp"),
                coerce=coerce_timestamps,
                tz=tz,
            )
            grouped[key].append(record)

        return {key: tuple(values) for key, values in grouped.items()}

    def get_risk_evaluation_trace(
        self,
        decision_id: Any,
        *,
        approved: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        normalized: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        include_errors: bool = True,
        service: str | None | Iterable[str | None] | object = _NO_FILTER,
        decision_state: str | Iterable[str | None] | object = _NO_FILTER,
        decision_reason: str | Iterable[str | None] | object = _NO_FILTER,
        decision_mode: str | Iterable[str | None] | object = _NO_FILTER,
        since: Any = None,
        until: Any = None,
        include_decision: bool = True,
        include_service: bool = True,
        include_response: bool = True,
        include_error: bool = True,
        coerce_timestamps: bool = True,
        tz: tzinfo | None = timezone.utc,
    ) -> Sequence[Mapping[str, Any]]:
        """Buduje uporządkowaną oś czasu ocen ryzyka dla podanej decyzji."""

        normalized_id = self._normalize_decision_id(decision_id)
        if normalized_id is None:
            return ()

        (
            approved_filter,
            normalized_filter,
            service_filter,
            decision_state_filter,
            decision_reason_filter,
            decision_mode_filter,
            decision_id_filter,
            _,
            since_ts,
            until_ts,
        ) = self._resolve_risk_evaluation_filters(
            approved=approved,
            normalized=normalized,
            service=service,
            decision_state=decision_state,
            decision_reason=decision_reason,
            decision_mode=decision_mode,
            decision_id=[normalized_id],
            since=since,
            until=until,
            decision_fields=None,
        )

        (
            filtered_records,
            trimmed_by_ttl,
            ttl_snapshot,
            history_size,
        ) = self._collect_filtered_risk_evaluations(
            include_errors=include_errors,
            approved_filter=approved_filter,
            normalized_filter=normalized_filter,
            service_filter=service_filter,
            since_ts=since_ts,
            until_ts=until_ts,
            state_filter=decision_state_filter,
            reason_filter=decision_reason_filter,
            mode_filter=decision_mode_filter,
            decision_id_filter=decision_id_filter,
        )

        self._log_risk_history_trimmed(
            context="trace",
            trimmed=trimmed_by_ttl,
            ttl=ttl_snapshot,
            history=history_size,
        )

        if not filtered_records:
            return ()

        first_timestamp = self._normalize_time_bound(filtered_records[0].get("timestamp"))
        if first_timestamp is None:
            first_timestamp = 0.0
        previous_timestamp = first_timestamp

        timeline: list[Mapping[str, Any]] = []
        for index, entry in enumerate(filtered_records):
            record = copy.deepcopy(entry)
            record["decision_id"] = self._normalize_decision_id(record.get("decision_id")) or normalized_id
            record["timestamp"] = self._normalize_timestamp_for_export(
                record.get("timestamp"),
                coerce=coerce_timestamps,
                tz=tz,
            )

            if not include_decision:
                record.pop("decision", None)
            if not include_service:
                record.pop("service", None)
            if not include_response:
                record.pop("response", None)
            if not include_error:
                record.pop("error", None)

            timestamp_value = self._normalize_time_bound(entry.get("timestamp"))
            if timestamp_value is None:
                timestamp_value = previous_timestamp if index else first_timestamp

            elapsed_since_first = max(0.0, timestamp_value - first_timestamp)
            elapsed_since_previous = max(
                0.0,
                timestamp_value - previous_timestamp if index else 0.0,
            )

            record["step_index"] = index
            record["elapsed_since_first_s"] = float(elapsed_since_first)
            record["elapsed_since_previous_s"] = float(elapsed_since_previous)

            timeline.append(record)
            previous_timestamp = timestamp_value

        return tuple(timeline)



    def export_risk_evaluations(
        self,
        *,
        approved: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        normalized: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        include_errors: bool = True,
        service: str | None | Iterable[str | None] | object = _NO_FILTER,
        decision_state: str | Iterable[str | None] | object = _NO_FILTER,
        decision_reason: str | Iterable[str | None] | object = _NO_FILTER,
        decision_mode: str | Iterable[str | None] | object = _NO_FILTER,
        decision_id: str | Iterable[str | None] | object = _NO_FILTER,
        since: Any = None,
        until: Any = None,
        flatten_decision: bool = False,
        decision_prefix: str = "decision_",
        decision_fields: Iterable[Any] | Any | None = None,
        drop_decision_column: bool = False,
        fill_value: Any = pd.NA,
        coerce_timestamps: bool = False,
        tz: tzinfo | None = timezone.utc,
    ) -> Mapping[str, Any]:
        (
            approved_filter,
            normalized_filter,
            service_filter,
            decision_state_filter,
            decision_reason_filter,
            decision_mode_filter,
            decision_id_filter,
            normalized_decision_fields,
            since_ts,
            until_ts,
        ) = self._resolve_risk_evaluation_filters(
            approved=approved,
            normalized=normalized,
            service=service,
            decision_state=decision_state,
            decision_reason=decision_reason,
            decision_mode=decision_mode,
            decision_id=decision_id,
            since=since,
            until=until,
            decision_fields=decision_fields,
        )

        (
            filtered_records,
            trimmed_by_ttl,
            ttl_snapshot,
            history_size,
        ) = self._collect_filtered_risk_evaluations(
            include_errors=include_errors,
            approved_filter=approved_filter,
            normalized_filter=normalized_filter,
            service_filter=service_filter,
            since_ts=since_ts,
            until_ts=until_ts,
            state_filter=decision_state_filter,
            reason_filter=decision_reason_filter,
            mode_filter=decision_mode_filter,
            decision_id_filter=decision_id_filter,
        )

        self._log_risk_history_trimmed(
            context="export",
            trimmed=trimmed_by_ttl,
            ttl=ttl_snapshot,
            history=history_size,
        )

        records = self._build_risk_evaluation_records(
            filtered_records,
            normalized_decision_fields=normalized_decision_fields,
            flatten_decision=flatten_decision,
            decision_prefix=decision_prefix,
            drop_decision_column=drop_decision_column,
            fill_value=fill_value,
            coerce_timestamps=coerce_timestamps,
            tz=tz,
        )
        json_ready = self._jsonify_risk_evaluation_records(records)

        with self._lock:
            limit_snapshot = self._risk_evaluations_limit

        def _serialize_filter(values: Iterable[object] | None) -> list[str] | None:
            if values is None:
                return None
            return sorted(str(item) for item in values)

        filters_payload: dict[str, Any] = {
            "approved": _serialize_filter(approved_filter),
            "normalized": _serialize_filter(normalized_filter),
            "include_errors": bool(include_errors),
            "service": _serialize_filter(service_filter),
            "decision_state": _serialize_filter(decision_state_filter),
            "decision_reason": _serialize_filter(decision_reason_filter),
            "decision_mode": _serialize_filter(decision_mode_filter),
            "decision_id": _serialize_filter(decision_id_filter),
            "since": since_ts.isoformat() if since_ts is not None else None,
            "until": until_ts.isoformat() if until_ts is not None else None,
            "flatten_decision": bool(flatten_decision),
            "decision_prefix": str(decision_prefix),
            "decision_fields": (
                list(normalized_decision_fields)
                if normalized_decision_fields is not None
                else None
            ),
            "drop_decision_column": bool(drop_decision_column),
            "fill_value_repr": repr(fill_value),
            "coerce_timestamps": bool(coerce_timestamps),
            "timezone": tz.tzname(None) if isinstance(tz, tzinfo) else tz,
        }

        payload: dict[str, Any] = {
            "version": 1,
            "entries": json_ready,
            "filters": filters_payload,
            "retention": {
                "limit": limit_snapshot,
                "ttl_s": ttl_snapshot,
            },
            "trimmed_by_ttl": trimmed_by_ttl,
            "history_size": history_size,
        }
        return payload

    def dump_risk_evaluations(
        self,
        destination: str | Path,
        *,
        approved: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        normalized: bool | None | Iterable[bool | None] | object = _NO_FILTER,
        include_errors: bool = True,
        service: str | None | Iterable[str | None] | object = _NO_FILTER,
        decision_state: str | Iterable[str | None] | object = _NO_FILTER,
        decision_reason: str | Iterable[str | None] | object = _NO_FILTER,
        decision_mode: str | Iterable[str | None] | object = _NO_FILTER,
        decision_id: str | Iterable[str | None] | object = _NO_FILTER,
        since: Any = None,
        until: Any = None,
        flatten_decision: bool = False,
        decision_prefix: str = "decision_",
        decision_fields: Iterable[Any] | Any | None = None,
        drop_decision_column: bool = False,
        fill_value: Any = pd.NA,
        coerce_timestamps: bool = False,
        tz: tzinfo | None = timezone.utc,
        ensure_ascii: bool = False,
    ) -> None:
        payload = self.export_risk_evaluations(
            approved=approved,
            normalized=normalized,
            include_errors=include_errors,
            service=service,
            decision_state=decision_state,
            decision_reason=decision_reason,
            decision_mode=decision_mode,
            decision_id=decision_id,
            since=since,
            until=until,
            flatten_decision=flatten_decision,
            decision_prefix=decision_prefix,
            decision_fields=decision_fields,
            drop_decision_column=drop_decision_column,
            fill_value=fill_value,
            coerce_timestamps=coerce_timestamps,
            tz=tz,
        )

        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=ensure_ascii),
            encoding="utf-8",
        )

    def load_risk_evaluations(
        self,
        payload: Mapping[str, Any],
        *,
        merge: bool = False,
        notify_listeners: bool = False,
    ) -> int:
        """Ładuje historię ocen ryzyka z mapy zwróconej przez ``export_risk_evaluations``."""

        if not isinstance(payload, Mapping):
            raise TypeError("payload musi być słownikiem zgodnym z export_risk_evaluations()")

        entries_payload = payload.get("entries", [])
        if entries_payload is None:
            entries_payload = []
        if not isinstance(entries_payload, Iterable):
            raise TypeError("entries muszą być iterowalne i zawierać słowniki")

        filters_payload = payload.get("filters", {}) or {}
        if not isinstance(filters_payload, Mapping):
            raise TypeError("filters muszą być słownikiem")

        flatten_decision = bool(filters_payload.get("flatten_decision", False))
        drop_decision_column = bool(filters_payload.get("drop_decision_column", False))
        decision_prefix = str(filters_payload.get("decision_prefix", "decision_"))

        raw_decision_fields = filters_payload.get("decision_fields")
        if raw_decision_fields is None:
            decision_fields: list[Any] = []
        elif isinstance(raw_decision_fields, (str, bytes)):
            decision_fields = [raw_decision_fields]
        elif isinstance(raw_decision_fields, Iterable):
            decision_fields = [field for field in raw_decision_fields]
        else:
            decision_fields = [raw_decision_fields]

        decision_fields = [field if isinstance(field, str) else str(field) for field in decision_fields]

        records: list[dict[str, Any]] = []
        for entry in entries_payload:
            if not isinstance(entry, Mapping):
                raise TypeError("każdy wpis musi być słownikiem")

            timestamp_raw = entry.get("timestamp")
            timestamp_value = self._normalize_time_bound(timestamp_raw)
            if timestamp_value is None:
                raise ValueError("wpis historii ocen ryzyka wymaga znacznika czasu")

            approved_value = copy.deepcopy(entry.get("approved"))
            normalized_value = copy.deepcopy(entry.get("normalized"))
            if normalized_value is None:
                normalized_value = approved_value

            base_decision = entry.get("decision")
            if isinstance(base_decision, Mapping):
                decision_payload = {
                    str(key): copy.deepcopy(value) for key, value in base_decision.items()
                }
            else:
                decision_payload = {}

            if flatten_decision:
                candidate_fields = set(decision_fields)
                for key in entry.keys():
                    if isinstance(key, str) and key.startswith(decision_prefix):
                        candidate_fields.add(key[len(decision_prefix) :])
                for field in candidate_fields:
                    column_name = f"{decision_prefix}{field}"
                    if column_name in entry:
                        decision_payload.setdefault(
                            field if isinstance(field, str) else str(field),
                            copy.deepcopy(entry[column_name]),
                        )
            elif drop_decision_column and not decision_payload:
                decision_payload = {}

            record: dict[str, Any] = {
                "timestamp": float(timestamp_value),
                "approved": approved_value,
                "normalized": normalized_value,
                "decision": decision_payload,
            }

            for key in ("service", "response", "error"):
                if key in entry:
                    record[key] = copy.deepcopy(entry[key])

            decision_id_value = entry.get("decision_id")
            normalized_decision_id = self._normalize_decision_id(decision_id_value)
            if normalized_decision_id is not None or decision_id_value is not None:
                record["decision_id"] = (
                    normalized_decision_id if normalized_decision_id is not None else copy.deepcopy(decision_id_value)
                )

            metadata_payload = entry.get("metadata")
            if isinstance(metadata_payload, Mapping):
                record["metadata"] = {
                    str(key): copy.deepcopy(value) for key, value in metadata_payload.items()
                }
            elif metadata_payload is not None:
                record["metadata"] = copy.deepcopy(metadata_payload)

            records.append(record)

        records.sort(key=lambda item: float(item.get("timestamp", 0.0)))

        retention_payload = payload.get("retention", {}) or {}
        if not isinstance(retention_payload, Mapping):
            raise TypeError("retention musi być słownikiem")

        trimmed_by_limit_total = 0
        trimmed_by_ttl_total = 0
        limit_snapshot: int | None = None
        ttl_snapshot: float | None = None
        history_size = 0

        with self._lock:
            if not merge:
                self._risk_evaluations.clear()

            if "limit" in retention_payload:
                limit_raw = retention_payload.get("limit")
                if limit_raw is None:
                    normalized_limit: int | None = None
                else:
                    try:
                        normalized_limit = int(limit_raw)
                    except (TypeError, ValueError):
                        normalized_limit = None
                    else:
                        if normalized_limit < 0:
                            normalized_limit = 0
                self._risk_evaluations_limit = normalized_limit

            if "ttl_s" in retention_payload:
                ttl_raw = retention_payload.get("ttl_s")
                if ttl_raw is None:
                    self._risk_evaluations_ttl_s = None
                else:
                    self._risk_evaluations_ttl_s = self._normalise_cycle_history_ttl(ttl_raw)

            limit_snapshot = self._risk_evaluations_limit
            ttl_snapshot = self._risk_evaluations_ttl_s
            trimmed_by_limit_total += self._apply_risk_evaluation_limit_locked(limit_snapshot)
            trimmed_by_ttl_total += self._prune_risk_evaluations_locked()
            history_size = len(self._risk_evaluations)

        records.sort(key=lambda item: float(item.get("timestamp", 0.0)))

        event_payloads: list[dict[str, Any]] = []
        for record in records:
            (
                trimmed_by_limit,
                trimmed_by_ttl,
                limit_snapshot,
                ttl_snapshot,
                history_size,
            ) = self._store_risk_evaluation_entry(
                record,
                reference_time=record.get("timestamp"),
            )
            trimmed_by_limit_total += trimmed_by_limit
            trimmed_by_ttl_total += trimmed_by_ttl
            event_payloads.append(
                self._build_risk_evaluation_event_payload(
                    record,
                    trimmed_by_limit=trimmed_by_limit,
                    trimmed_by_ttl=trimmed_by_ttl,
                    history_size=history_size,
                    limit_snapshot=limit_snapshot,
                    ttl_snapshot=ttl_snapshot,
                )
            )

        self._log_risk_history_trimmed(
            context="load",
            trimmed=trimmed_by_ttl_total,
            ttl=ttl_snapshot,
            history=history_size,
        )
        self._log(
            "Załadowano historię ocen ryzyka",
            level=logging.DEBUG,
            loaded=len(records),
            merge=bool(merge),
            trimmed_limit=trimmed_by_limit_total,
            trimmed_ttl=trimmed_by_ttl_total,
            history=history_size,
            limit=limit_snapshot,
            ttl=ttl_snapshot,
        )

        for payload in event_payloads:
            self._emit_risk_evaluation_event(payload)
            if notify_listeners:
                self._notify_risk_evaluation_listeners(payload)

        return len(records)

    def import_risk_evaluations(
        self,
        source: str | Path,
        *,
        merge: bool = False,
        notify_listeners: bool = False,
    ) -> int:
        """Importuje historię ocen ryzyka z pliku JSON."""

        if json is None:  # pragma: no cover - środowiska bez json
            raise RuntimeError("moduł json jest wymagany do importu historii ocen ryzyka")

        path = Path(source)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("plik musi zawierać obiekt JSON zgodny z export_risk_evaluations()")
        return self.load_risk_evaluations(
            payload,
            merge=merge,
            notify_listeners=notify_listeners,
        )

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
        trimmed_by_limit = 0
        trimmed_by_ttl = 0
        ttl_snapshot: float | None = None
        history_size = 0
        with self._lock:
            self._risk_evaluations_limit = normalised
            trimmed_by_limit = self._apply_risk_evaluation_limit_locked(normalised)
            trimmed_by_ttl = self._prune_risk_evaluations_locked()
            ttl_snapshot = self._risk_evaluations_ttl_s
            history_size = len(self._risk_evaluations)
        self._log_risk_history_trimmed(
            context="configure",
            trimmed=trimmed_by_ttl,
            ttl=ttl_snapshot,
            history=history_size,
        )
        self._log(
            "Skonfigurowano limit historii ocen ryzyka",
            level=logging.DEBUG,
            limit=normalised,
            trimmed_limit=trimmed_by_limit,
            trimmed_ttl=trimmed_by_ttl,
            history=history_size,
            ttl=ttl_snapshot,
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


@dataclass(slots=True)
class AutoTraderDecisionScheduler:
    """Asynchronous scheduler driving :class:`AutoTrader` decision cycles."""

    trader: "AutoTrader"
    interval_s: float = 30.0
    loop: asyncio.AbstractEventLoop | None = None
    _task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _stop_event: asyncio.Event | None = field(init=False, default=None, repr=False)
    _thread: threading.Thread | None = field(init=False, default=None, repr=False)
    _thread_stop: threading.Event | None = field(init=False, default=None, repr=False)

    async def start(self) -> None:
        """Start the scheduler inside an asyncio event loop."""

        if self._task is not None and not self._task.done():
            return
        loop = self.loop or asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        self._task = loop.create_task(self._run_async())

    async def stop(self) -> None:
        """Request shutdown of the asynchronous scheduler and await completion."""

        task = self._task
        stop_event = self._stop_event
        if task is None or stop_event is None:
            return
        stop_event.set()
        try:
            await task
        finally:
            self._task = None
            self._stop_event = None

    def start_in_background(self) -> None:
        """Spawn a lightweight background thread executing decision cycles."""

        if self._thread is not None and self._thread.is_alive():
            return

        stop_event = threading.Event()
        self._thread_stop = stop_event

        def _worker() -> None:
            while not stop_event.is_set():
                try:
                    self.trader.run_cycle_once()
                except Exception:  # pragma: no cover - defensive guard
                    LOGGER.exception("AutoTraderDecisionScheduler cycle failed")
                if stop_event.wait(max(0.0, float(self.interval_s))):
                    break

        thread = threading.Thread(
            target=_worker,
            name="AutoTraderDecisionScheduler",
            daemon=True,
        )
        self._thread = thread
        thread.start()

    def stop_background(self) -> None:
        """Stop the background thread variant of the scheduler."""

        stop_event = self._thread_stop
        if stop_event is not None:
            stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(1.0, float(self.interval_s)))
        self._thread = None
        self._thread_stop = None

    async def _run_async(self) -> None:
        assert self._stop_event is not None
        stop_event = self._stop_event
        interval = max(0.0, float(self.interval_s))
        while not stop_event.is_set():
            try:
                await asyncio.to_thread(self.trader.run_cycle_once)
            except Exception:  # pragma: no cover - defensive guard
                LOGGER.exception("AutoTraderDecisionScheduler async cycle failed")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue


__all__ = [
    "AutoTrader",
    "AutoTraderDecisionScheduler",
    "RiskDecision",
    "EmitterLike",
    "GuardrailTrigger",
]
