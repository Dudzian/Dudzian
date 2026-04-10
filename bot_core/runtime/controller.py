"""Kontrolery spinające warstwy: dane/strategia/ryzyko/egzekucja oraz alerty."""

from __future__ import annotations

import json
import logging
import math
import uuid
from collections.abc import Sequence as SequenceABC
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Mapping,
    Protocol,
    MutableMapping,
    Sequence,
    Mapping as TypingMapping,
)

try:  # pragma: no cover - w niektórych gałęziach warstwa AI może nie być zainstalowana
    from bot_core.ai import (
        DecisionModelInference,
        FeatureDataset,
        FeatureEngineer,
        flatten_explainability,
        parse_explainability_payload,
    )
except Exception:  # pragma: no cover
    DecisionModelInference = Any  # type: ignore
    FeatureDataset = Any  # type: ignore
    FeatureEngineer = Any  # type: ignore
    flatten_explainability = lambda report, prefix="ai_explainability": {}  # type: ignore
    parse_explainability_payload = lambda payload: None  # type: ignore
from bot_core.ai.health import ModelHealthMonitor, ModelHealthStatus
from bot_core.ai.opportunity_lifecycle import (
    OpportunityAutonomyDecision,
    OpportunityAutonomyMode,
    OpportunityExecutionPermission,
    OpportunityLifecycleService,
    OpportunityPerformanceSnapshotConfig,
    evaluate_autonomy_performance_guard,
    evaluate_opportunity_execution_permission,
)
from bot_core.ai.trading_opportunity_shadow import (
    OpportunityOutcomeLabel,
    OpportunityShadowRepository,
)

# --- elastyczne importy (różne gałęzie mogą mieć różne ścieżki modułów) -----

# Alerts
from bot_core.alerts import AlertMessage  # dostępne w obu gałęziach

if TYPE_CHECKING:
    from bot_core.runtime.observability import AlertSink
else:

    class AlertSink(Protocol):
        """Minimalny interfejs sinka alertów wymagany przez TradingController."""

        def dispatch(self, message: AlertMessage) -> None: ...

        def health_snapshot(self) -> Mapping[str, Mapping[str, object]]: ...


# Execution
try:
    from bot_core.execution import ExecutionContext, ExecutionService  # re-eksport
except Exception:  # pragma: no cover
    from bot_core.execution.base import ExecutionContext, ExecutionService  # fallback

# Exchanges commons
from bot_core.exchanges.base import AccountSnapshot, OrderRequest, OrderResult
from bot_core.runtime.journal import TradingDecisionEvent, TradingDecisionJournal

try:
    from bot_core.runtime.tco_reporting import RuntimeTCOReporter
except Exception:  # pragma: no cover - fallback dla środowisk bez zależności TCO

    class RuntimeTCOReporter(Protocol):  # type: ignore[misc]
        def record_execution(self, **kwargs: object) -> None: ...


# Risk
try:
    from bot_core.risk import RiskEngine, RiskCheckResult  # re-eksport
except Exception:  # pragma: no cover
    from bot_core.risk.base import RiskEngine, RiskCheckResult  # fallback

# Strategy types (interfejsy)
try:
    from bot_core.strategies import (
        StrategySignal,
        MarketSnapshot,
    )  # gałąź z interfejsami w pakiecie
except Exception:  # pragma: no cover
    from bot_core.strategies.base import StrategySignal, MarketSnapshot  # alternatywna ścieżka

# Decision Engine (opcjonalnie)
try:  # pragma: no cover
    from bot_core.decision import (
        DecisionCandidate,
        DecisionContext,
        DecisionEvaluation,
    )  # type: ignore
    from bot_core.decision.evaluators import DecisionEvaluator
    from bot_core.decision.orchestrator import DecisionOrchestrator
    from bot_core.decision.providers import DecisionProvider
except Exception:  # pragma: no cover
    DecisionCandidate = None  # type: ignore
    DecisionContext = Any  # type: ignore
    DecisionEvaluation = Any  # type: ignore
    DecisionOrchestrator = None  # type: ignore

    class DecisionEvaluator(Protocol):  # type: ignore[misc]
        def evaluate_candidate(self, candidate: Any, context: Any) -> Any: ...

    class DecisionProvider(Protocol):  # type: ignore[misc]
        def ensure_snapshot(self, profile: str, snapshot: Mapping[str, object] | Any) -> Any: ...


# Dane OHLCV – w zależności od gałęzi
try:
    # wariant modułowy
    from bot_core.data.base import OHLCVRequest  # type: ignore
    from bot_core.data.ohlcv.backfill import OHLCVBackfillService  # type: ignore
    from bot_core.data.ohlcv.cache import CachedOHLCVSource  # type: ignore
except Exception:  # pragma: no cover
    # wariant monolityczny
    from bot_core.data.ohlcv import (  # type: ignore
        OHLCVBackfillService,
        CachedOHLCVSource,
    )

# Konfiguracja runtime kontrolerów – tylko jeśli istnieje w danej gałęzi
if TYPE_CHECKING:
    from bot_core.config.models import CoreConfig, ControllerRuntimeConfig
else:  # pragma: no cover - w czasie runtime korzystamy z typu ogólnego
    CoreConfig = Any  # type: ignore
    ControllerRuntimeConfig = Any  # type: ignore

# Observability (metryki są opcjonalne)
try:  # pragma: no cover - fallback dla gałęzi bez modułu metrics
    from bot_core.observability.metrics import (  # type: ignore
        MetricsRegistry,
        get_global_metrics_registry,
    )
except Exception:  # pragma: no cover

    class _NoopCounter:
        def inc(self, *_args, **_kwargs) -> None:
            return None

    class _NoopGauge:
        def set(self, *_args, **_kwargs) -> None:
            return None

    class MetricsRegistry:  # type: ignore[override]
        def counter(self, *_args, **_kwargs) -> _NoopCounter:
            return _NoopCounter()

        def gauge(self, *_args, **_kwargs) -> _NoopGauge:
            return _NoopGauge()

    def get_global_metrics_registry() -> MetricsRegistry:  # type: ignore[override]
        return MetricsRegistry()


_LOGGER = logging.getLogger(__name__)

_NEUTRAL_SIDES = {
    "HOLD",
    "NEUTRAL",
    "FLAT",
    "NONE",
    "REBALANCE",
    "REBALANCE_DELTA",
    "REBALANCE_DELTA_RATIO",
}
_NEUTRAL_INTENTS = {"neutral", "rebalance", "rebalance_delta", "hedge"}
_SIDE_ALIASES = {
    "LONG": "BUY",
    "ENTER_LONG": "BUY",
    "OPEN_LONG": "BUY",
    "BUY_TO_OPEN": "BUY",
    "COVER": "BUY",
    "CLOSE_SHORT": "BUY",
    "EXIT_SHORT": "BUY",
    "BUY_TO_CLOSE": "BUY",
    "SHORT": "SELL",
    "ENTER_SHORT": "SELL",
    "OPEN_SHORT": "SELL",
    "SELL_SHORT": "SELL",
    "SELL_TO_OPEN": "SELL",
    "CLOSE_LONG": "SELL",
    "EXIT_LONG": "SELL",
    "SELL_TO_CLOSE": "SELL",
}
_FILLED_EXECUTION_STATUSES = frozenset({"filled", "executed", "complete", "completed"})
_PARTIAL_EXECUTION_STATUSES = frozenset({"partial", "partially_filled", "partially-filled"})
_BUY_SIDES = frozenset({"BUY", "LONG"})
_SELL_SIDES = frozenset({"SELL", "SHORT"})
_OPPORTUNITY_AUTONOMY_PROVENANCE_KEYS = (
    "autonomy_requested_mode",
    "autonomy_upstream_effective_mode",
    "autonomy_local_guard_effective_mode",
    "autonomy_final_mode",
    "autonomous_execution_allowed",
    "assisted_override_used",
    "blocking_reason",
    "autonomy_decisive_stage",
    "autonomy_decisive_reason",
    "autonomy_primary_reason",
    "upstream_autonomy_decision_source",
    "upstream_autonomy_inference_model",
    "upstream_autonomy_inference_model_version",
)
_LIVE_AUTONOMY_ADMISSION_BLOCKER_REASONS = frozenset(
    {
        "promotion_not_ready_for_live_autonomous",
        "activation_not_ready_for_live_autonomous",
        "insufficient_final_outcomes_for_live_autonomous",
        "too_many_non_blocking_degradations_for_live_autonomous",
    }
)


def _normalize_trade_side(value: object | None) -> str | None:
    if value is None:
        return None
    candidate = str(value).strip().upper()
    if not candidate:
        return None
    if candidate in _NEUTRAL_SIDES:
        return None
    if candidate in {"BUY", "SELL"}:
        return candidate
    return _SIDE_ALIASES.get(candidate)


def _normalize_execution_status(value: object | None) -> str:
    candidate = str(value or "").strip().lower()
    return candidate or "unknown"


def _is_live_autonomy_admission_blocker_reason(reason: object | None) -> bool:
    candidate = str(reason or "").strip().lower()
    return candidate in _LIVE_AUTONOMY_ADMISSION_BLOCKER_REASONS


@dataclass(slots=True)
class _OpportunityOpenOutcomeTracker:
    correlation_key: str
    symbol: str
    side: str
    entry_price: float
    decision_timestamp: datetime
    entry_quantity: float = 0.0
    closed_quantity: float = 0.0
    model_version: str | None = None
    decision_source: str | None = None
    autonomy_requested_mode: str | None = None
    autonomy_upstream_effective_mode: str | None = None
    autonomy_local_guard_effective_mode: str | None = None
    autonomy_final_mode: str | None = None
    autonomous_execution_allowed: str | None = None
    assisted_override_used: str | None = None
    blocking_reason: str | None = None
    autonomy_decisive_stage: str | None = None
    autonomy_decisive_reason: str | None = None
    autonomy_primary_reason: str | None = None
    upstream_autonomy_decision_source: str | None = None
    upstream_autonomy_inference_model: str | None = None
    upstream_autonomy_inference_model_version: str | None = None
    environment_scope: str | None = None
    portfolio_scope: str | None = None
    restored_from_repository: bool = False


def _extract_adjusted_quantity(
    original_quantity: float,
    adjustments: Mapping[str, float] | None,
) -> float | None:
    """Zwraca dopuszczalną wielkość zlecenia zasugerowaną przez silnik ryzyka."""
    if not adjustments:
        return None

    raw_value = adjustments.get("quantity") or adjustments.get("max_quantity")
    if raw_value is None:
        return None

    try:
        candidate = float(raw_value)
    except (TypeError, ValueError):  # pragma: no cover - defensywny fallback
        return None

    candidate = max(0.0, min(candidate, original_quantity))
    if candidate <= 0.0:
        return None
    if math.isclose(candidate, original_quantity, rel_tol=1e-9, abs_tol=1e-12):
        return None
    return candidate


def _clamp_request_quantity(
    request: OrderRequest,
    account: AccountSnapshot,
    profile: object | None,
    *,
    include_trade_risk: bool,
) -> OrderRequest:
    """Dopasowuje quantity do limitów profilu (np. max_position_pct, risk per trade)."""
    if profile is None:
        return request

    price = request.price
    equity = getattr(account, "total_equity", 0.0)
    if price is None or price <= 0 or equity is None or float(equity) <= 0.0:
        return request

    side = request.side.lower()
    quantity_limits: list[float] = [request.quantity]

    try:
        max_position_pct = float(profile.max_position_exposure())
    except Exception:  # pragma: no cover - defensywne na inne implementacje profili
        max_position_pct = 0.0

    if max_position_pct > 0:
        quantity_limits.append(max_position_pct * float(equity) / price)

    if include_trade_risk:
        stop_price = request.stop_price
        if stop_price is None and request.metadata:
            try:
                stop_price = float(request.metadata.get("stop_price"))  # type: ignore[arg-type]
            except Exception:
                stop_price = None

        if stop_price is not None:
            stop_distance = price - stop_price if side == "buy" else stop_price - price
        else:
            stop_distance = None

        if stop_distance is not None and stop_distance > 0:
            try:
                min_risk, max_risk = profile.trade_risk_pct_range()
                max_trade_risk_pct = max(float(min_risk), float(max_risk), 0.0)
            except Exception:  # pragma: no cover - defensywne na inne implementacje profili
                max_trade_risk_pct = 0.0
            if max_trade_risk_pct > 0:
                quantity_limits.append((max_trade_risk_pct * float(equity)) / stop_distance)

    adjusted_qty = max(0.0, min(quantity_limits))
    if math.isclose(adjusted_qty, request.quantity, rel_tol=1e-12, abs_tol=1e-12):
        return request

    metadata = dict(request.metadata or {})
    metadata["quantity"] = float(adjusted_qty)
    return OrderRequest(
        symbol=request.symbol,
        side=request.side,
        quantity=adjusted_qty,
        order_type=request.order_type,
        price=request.price,
        time_in_force=request.time_in_force,
        client_order_id=request.client_order_id,
        stop_price=request.stop_price,
        atr=request.atr,
        metadata=metadata,
    )


# =============================================================================
# TradingController – przetwarza sygnały (BUY/SELL), pilnuje ryzyka i wysyła alerty
# =============================================================================


def _as_timedelta(value: timedelta | float | int) -> timedelta:
    if isinstance(value, timedelta):
        return value
    seconds = float(value)
    if seconds < 0:
        raise ValueError("Czas health-check nie może być ujemny")
    return timedelta(seconds=seconds)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_modes(values: Sequence[object] | None, *, default: Sequence[str]) -> tuple[str, ...]:
    candidates = values if values is not None else default
    normalized: list[str] = []
    for value in candidates:
        if value is None:
            continue
        candidate = str(value).strip().lower()
        if not candidate or candidate in normalized:
            continue
        normalized.append(candidate)
    return tuple(normalized)


def _normalize_optional_request_string(value: object | None) -> str | None:
    if value is None:
        return None
    candidate = str(value).strip()
    return candidate or None


def _as_bool(value: object | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "t", "yes", "y", "on"}


def _normalize_optional_request_float(value: object | None, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        value = candidate
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} w metadanych musi być liczbą zmiennoprzecinkową") from exc


def _validate_optional_positive_int(value: object | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} musi być dodatnią liczbą całkowitą (> 0)")
    candidate = value
    if candidate <= 0:
        raise ValueError(f"{field_name} musi być dodatnią liczbą całkowitą (> 0)")
    return candidate


@dataclass(slots=True)
class ControllerSignal:
    """Zbiera sygnał strategii wraz ze snapshotem rynku."""

    snapshot: MarketSnapshot
    signal: StrategySignal


@dataclass(slots=True, kw_only=True)
class TradingController:
    """Podstawowy kontroler zarządzający przepływem sygnałów i alertów.

    - wysyła alert o nadejściu sygnału,
    - wykonuje pre-trade checks w RiskEngine,
    - zleca egzekucję (ExecutionService),
    - wysyła alert o odrzuceniu oraz alert trybu awaryjnego (liquidation),
    - okresowo publikuje health-check kanałów alertowych.
    """

    risk_engine: RiskEngine
    execution_service: ExecutionService
    alert_router: "AlertSink"
    account_snapshot_provider: Callable[[], AccountSnapshot]
    portfolio_id: str
    environment: str
    risk_profile: str
    order_metadata_defaults: Mapping[str, object] | None = None
    clock: Callable[[], datetime] = _now
    health_check_interval: timedelta | float | int = timedelta(hours=1)
    execution_metadata: Mapping[str, str] | None = None
    metrics_registry: MetricsRegistry | None = None
    decision_journal: TradingDecisionJournal | None = None
    strategy_name: str | None = None
    exchange_name: str | None = None
    tco_reporter: RuntimeTCOReporter | None = None
    tco_metadata: Mapping[str, object] | None = None
    decision_orchestrator: DecisionEvaluator | DecisionProvider | None = None
    decision_min_probability: float | None = None
    decision_default_notional: float = 1_000.0
    ai_health_monitor: ModelHealthMonitor | None = None
    ai_signal_modes: Sequence[str] | None = None
    rules_signal_modes: Sequence[str] | None = None
    signal_mode_priorities: Mapping[str, int] | None = None
    opportunity_shadow_repository: OpportunityShadowRepository | None = None
    performance_guard_recent_final_window_size: int | None = None
    performance_guard_max_scan_labels: int | None = None

    _clock: Callable[[], datetime] = field(init=False, repr=False)
    _health_interval: timedelta = field(init=False, repr=False)
    _execution_context: ExecutionContext = field(init=False, repr=False)
    _order_defaults: dict[str, str] = field(init=False, repr=False)
    _last_health_report: datetime = field(init=False, repr=False)
    _liquidation_alerted: bool = field(init=False, repr=False)
    _liquidation_active: bool = field(init=False, repr=False)
    _metrics: MetricsRegistry = field(init=False, repr=False)
    _metric_labels: Mapping[str, str] = field(init=False, repr=False)
    _metric_signals_total: Any = field(init=False, repr=False)
    _metric_orders_total: Any = field(init=False, repr=False)
    _metric_health_reports: Any = field(init=False, repr=False)
    _metric_liquidation_state: Any = field(init=False, repr=False)
    _metric_reversal_skipped_total: Any = field(init=False, repr=False)
    _metric_reversal_denied_by_risk_total: Any = field(init=False, repr=False)
    _decision_journal: TradingDecisionJournal | None = field(init=False, repr=False)
    _strategy_name: str | None = field(init=False, repr=False, default=None)
    _exchange_name: str | None = field(init=False, repr=False, default=None)
    _tco_reporter: RuntimeTCOReporter | None = field(init=False, repr=False, default=None)
    _tco_metadata: Mapping[str, object] = field(init=False, repr=False, default_factory=dict)
    _decision_orchestrator: DecisionEvaluator | DecisionProvider | None = field(
        init=False, repr=False, default=None
    )
    _decision_min_probability: float = field(init=False, repr=False, default=0.0)
    _decision_default_notional: float = field(init=False, repr=False, default=1_000.0)
    _signal_mode_priorities: Mapping[str, int] = field(init=False, repr=False, default_factory=dict)
    _default_signal_priority: int = field(init=False, repr=False, default=0)
    _ai_signal_modes: tuple[str, ...] = field(init=False, repr=False, default_factory=tuple)
    _rules_signal_modes: tuple[str, ...] = field(init=False, repr=False, default_factory=tuple)
    _ai_health_monitor: ModelHealthMonitor | None = field(init=False, repr=False, default=None)
    _ai_failover_active: bool = field(init=False, repr=False, default=False)
    _ai_failover_reason: str | None = field(init=False, repr=False, default=None)
    _ai_health_status: ModelHealthStatus | None = field(init=False, repr=False, default=None)
    _opportunity_shadow_repository: OpportunityShadowRepository | None = field(
        init=False,
        repr=False,
        default=None,
    )
    _opportunity_open_outcomes: MutableMapping[str, _OpportunityOpenOutcomeTracker] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        self.performance_guard_recent_final_window_size = _validate_optional_positive_int(
            self.performance_guard_recent_final_window_size,
            field_name="performance_guard_recent_final_window_size",
        )
        self.performance_guard_max_scan_labels = _validate_optional_positive_int(
            self.performance_guard_max_scan_labels,
            field_name="performance_guard_max_scan_labels",
        )
        self._clock = self.clock
        self._health_interval = _as_timedelta(self.health_check_interval)
        metadata: MutableMapping[str, str] = {}
        if self.execution_metadata:
            metadata.update({str(k): str(v) for k, v in self.execution_metadata.items()})
        self._execution_context = ExecutionContext(
            portfolio_id=self.portfolio_id,
            risk_profile=self.risk_profile,
            environment=self.environment,
            metadata=metadata,
        )
        self._order_defaults = dict(self.order_metadata_defaults or {})
        self._last_health_report = self._clock()
        self._liquidation_alerted = False
        self._liquidation_active = False
        self._metrics = self.metrics_registry or get_global_metrics_registry()
        self._decision_journal = self.decision_journal
        self._metric_labels = {
            "environment": self.environment,
            "portfolio": self.portfolio_id,
            "risk_profile": self.risk_profile,
        }
        self._strategy_name = self.strategy_name
        if self._strategy_name:
            self._metric_labels["strategy"] = self._strategy_name
        self._exchange_name = self.exchange_name
        self._tco_reporter = self.tco_reporter
        self._tco_metadata = dict(self.tco_metadata or {})
        self._metric_signals_total = self._metrics.counter(
            "trading_signals_total",
            "Liczba sygnałów przetworzonych w TradingController (status=received/accepted/rejected/adjusted/neutral/skipped).",
        )
        self._metric_orders_total = self._metrics.counter(
            "trading_orders_total",
            "Liczba zleceń obsłużonych przez TradingController (result=submitted/executed/failed).",
        )
        self._metric_health_reports = self._metrics.counter(
            "trading_health_reports_total",
            "Liczba wysłanych raportów health-check przez TradingController.",
        )
        self._metric_liquidation_state = self._metrics.gauge(
            "trading_liquidation_state",
            "Stan trybu awaryjnego profilu ryzyka (1=liquidation, 0=normal).",
        )
        self._metric_reversal_skipped_total = self._metrics.counter(
            "trading_reversal_skipped_total",
            "Liczba pominiętych prób reversal (powód=disabled/untrusted_position/not_required).",
        )
        self._metric_reversal_denied_by_risk_total = self._metrics.counter(
            "trading_reversal_denied_by_risk_total",
            "Liczba prób close dla reversal odrzuconych przez risk engine.",
        )
        self._metric_liquidation_state.set(0.0, labels=self._metric_labels)
        self._decision_orchestrator = self.decision_orchestrator
        default_notional = max(100.0, float(self.decision_default_notional or 0.0))
        self._decision_default_notional = default_notional
        configured_min_probability: float | None = None
        orchestrator_config = None
        if self._decision_orchestrator is not None:
            orchestrator_config = getattr(self._decision_orchestrator, "_config", None)
        if orchestrator_config is not None:
            configured_min_probability = float(getattr(orchestrator_config, "min_probability", 0.0))
        if self.decision_min_probability is not None:
            candidate = float(self.decision_min_probability)
        elif configured_min_probability is not None:
            candidate = configured_min_probability
        else:
            candidate = 0.0
        self._decision_min_probability = max(0.0, min(0.995, candidate))
        raw_priorities = dict(self.signal_mode_priorities or {})
        normalized_priorities: dict[str, int] = {}
        for key, value in raw_priorities.items():
            if key is None:
                continue
            try:
                normalized_priorities[str(key).strip().lower()] = int(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
        if not normalized_priorities:
            normalized_priorities = {
                "ai": 100,
                "ml": 100,
                "ensemble": 100,
                "rules": 60,
                "rule": 60,
                "heuristic": 60,
                "fallback": 40,
                "default": 50,
            }
        else:
            normalized_priorities.setdefault("default", 0)
        self._signal_mode_priorities = normalized_priorities
        self._default_signal_priority = normalized_priorities.get("default", 0)
        self._ai_signal_modes = _normalize_modes(
            self.ai_signal_modes,
            default=("ai", "ml", "ensemble"),
        )
        self._rules_signal_modes = _normalize_modes(
            self.rules_signal_modes,
            default=("rules", "rule", "heuristic", "fallback"),
        )
        self._ai_health_monitor = self.ai_health_monitor
        self._ai_failover_active = False
        self._ai_failover_reason = None
        self._ai_health_status = None
        self._opportunity_shadow_repository = self.opportunity_shadow_repository
        self._opportunity_open_outcomes = {}
        self._restore_opportunity_open_outcomes()

    def _restore_opportunity_open_outcomes(self) -> None:
        repository = self._opportunity_shadow_repository
        if repository is None:
            return
        try:
            restored = repository.load_open_outcomes()
        except Exception:  # pragma: no cover - diagnostics only
            _LOGGER.debug("Nie udało się odtworzyć open Opportunity outcomes", exc_info=True)
            return
        for row in restored:
            restored_provenance = row.provenance if isinstance(row.provenance, Mapping) else {}
            restored_model_version_raw = restored_provenance.get("model_version")
            restored_decision_source_raw = restored_provenance.get("decision_source")
            restored_model_version = (
                str(restored_model_version_raw).strip()
                if restored_model_version_raw is not None
                else ""
            )
            restored_decision_source = (
                str(restored_decision_source_raw).strip()
                if restored_decision_source_raw is not None
                else ""
            )
            restored_environment_raw = restored_provenance.get("environment")
            restored_portfolio_raw = restored_provenance.get("portfolio")
            restored_environment = (
                str(restored_environment_raw).strip()
                if restored_environment_raw is not None
                else ""
            )
            restored_portfolio = (
                str(restored_portfolio_raw).strip() if restored_portfolio_raw is not None else ""
            )
            autonomy_chain = self._extract_opportunity_autonomy_provenance_chain(
                restored_provenance
            )
            self._opportunity_open_outcomes[row.correlation_key] = _OpportunityOpenOutcomeTracker(
                correlation_key=row.correlation_key,
                symbol=row.symbol,
                side=row.side,
                entry_price=row.entry_price,
                decision_timestamp=row.decision_timestamp,
                entry_quantity=float(row.entry_quantity),
                closed_quantity=float(row.closed_quantity),
                model_version=restored_model_version or None,
                decision_source=restored_decision_source or None,
                autonomy_requested_mode=autonomy_chain.get("autonomy_requested_mode"),
                autonomy_upstream_effective_mode=autonomy_chain.get(
                    "autonomy_upstream_effective_mode"
                ),
                autonomy_local_guard_effective_mode=autonomy_chain.get(
                    "autonomy_local_guard_effective_mode"
                ),
                autonomy_final_mode=autonomy_chain.get("autonomy_final_mode"),
                autonomous_execution_allowed=autonomy_chain.get("autonomous_execution_allowed"),
                assisted_override_used=autonomy_chain.get("assisted_override_used"),
                blocking_reason=autonomy_chain.get("blocking_reason"),
                autonomy_decisive_stage=autonomy_chain.get("autonomy_decisive_stage"),
                autonomy_decisive_reason=autonomy_chain.get("autonomy_decisive_reason"),
                autonomy_primary_reason=autonomy_chain.get("autonomy_primary_reason"),
                upstream_autonomy_decision_source=autonomy_chain.get(
                    "upstream_autonomy_decision_source"
                ),
                upstream_autonomy_inference_model=autonomy_chain.get(
                    "upstream_autonomy_inference_model"
                ),
                upstream_autonomy_inference_model_version=autonomy_chain.get(
                    "upstream_autonomy_inference_model_version"
                ),
                environment_scope=restored_environment or None,
                portfolio_scope=restored_portfolio or None,
                restored_from_repository=True,
            )

    def _persist_open_outcome_tracker(self, tracker: _OpportunityOpenOutcomeTracker) -> None:
        repository = self._opportunity_shadow_repository
        if repository is None:
            return
        try:
            repository.upsert_open_outcome(
                repository.OpenOutcomeState(
                    correlation_key=tracker.correlation_key,
                    symbol=tracker.symbol,
                    side=tracker.side,
                    entry_price=tracker.entry_price,
                    decision_timestamp=tracker.decision_timestamp,
                    entry_quantity=tracker.entry_quantity,
                    closed_quantity=tracker.closed_quantity,
                    provenance={
                        "source": "trading_controller_open_execution",
                        **self._extract_opportunity_autonomy_provenance_chain_from_tracker(tracker),
                        **(
                            {"environment": tracker.environment_scope}
                            if tracker.environment_scope is not None
                            else (
                                {}
                                if tracker.restored_from_repository
                                else {"environment": self.environment}
                            )
                        ),
                        **(
                            {"portfolio": tracker.portfolio_scope}
                            if tracker.portfolio_scope is not None
                            else (
                                {}
                                if tracker.restored_from_repository
                                else {"portfolio": self.portfolio_id}
                            )
                        ),
                        **(
                            {"scope_continuity": "missing_from_restored_open_outcome"}
                            if tracker.restored_from_repository
                            and (
                                tracker.environment_scope is None or tracker.portfolio_scope is None
                            )
                            else {}
                        ),
                        **(
                            {"model_version": tracker.model_version}
                            if tracker.model_version is not None
                            else {}
                        ),
                        **(
                            {"decision_source": tracker.decision_source}
                            if tracker.decision_source is not None
                            else {}
                        ),
                    },
                )
            )
        except Exception:  # pragma: no cover - diagnostics only
            _LOGGER.debug("Nie udało się utrwalić open Opportunity outcome", exc_info=True)

    def _extract_opportunity_autonomy_provenance_chain(
        self, metadata: Mapping[str, object] | None
    ) -> dict[str, str]:
        if not isinstance(metadata, Mapping):
            return {}
        chain: dict[str, str] = {}
        for key in _OPPORTUNITY_AUTONOMY_PROVENANCE_KEYS:
            raw_value = metadata.get(key)
            if raw_value is None:
                continue
            if isinstance(raw_value, bool):
                candidate = "true" if raw_value else "false"
            else:
                candidate = str(raw_value).strip()
            if candidate:
                chain[key] = candidate
        return chain

    def _extract_opportunity_autonomy_provenance_chain_from_tracker(
        self, tracker: _OpportunityOpenOutcomeTracker | None
    ) -> dict[str, str]:
        if tracker is None:
            return {}
        return {
            key: value
            for key, value in {
                "autonomy_requested_mode": tracker.autonomy_requested_mode,
                "autonomy_upstream_effective_mode": tracker.autonomy_upstream_effective_mode,
                "autonomy_local_guard_effective_mode": tracker.autonomy_local_guard_effective_mode,
                "autonomy_final_mode": tracker.autonomy_final_mode,
                "autonomous_execution_allowed": tracker.autonomous_execution_allowed,
                "assisted_override_used": tracker.assisted_override_used,
                "blocking_reason": tracker.blocking_reason,
                "autonomy_decisive_stage": tracker.autonomy_decisive_stage,
                "autonomy_decisive_reason": tracker.autonomy_decisive_reason,
                "autonomy_primary_reason": tracker.autonomy_primary_reason,
                "upstream_autonomy_decision_source": tracker.upstream_autonomy_decision_source,
                "upstream_autonomy_inference_model": tracker.upstream_autonomy_inference_model,
                "upstream_autonomy_inference_model_version": (
                    tracker.upstream_autonomy_inference_model_version
                ),
            }.items()
            if value is not None
        }

    def _discard_open_outcome_tracker(self, correlation_key: str) -> None:
        self._opportunity_open_outcomes.pop(correlation_key, None)
        repository = self._opportunity_shadow_repository
        if repository is None:
            return
        try:
            repository.remove_open_outcome(correlation_key)
        except Exception:  # pragma: no cover - diagnostics only
            _LOGGER.debug("Nie udało się usunąć open Opportunity outcome", exc_info=True)

    def _resolve_open_outcome_tracker(
        self,
        *,
        symbol: str,
        current_side: str,
        correlation_key: str,
    ) -> tuple[_OpportunityOpenOutcomeTracker | None, str]:
        if correlation_key:
            tracked = self._opportunity_open_outcomes.get(correlation_key)
            if tracked is None:
                return None, "missing"
            if str(tracked.symbol) != str(symbol):
                return None, "symbol_mismatch"
            if not self._is_closing_side(tracked.side, current_side):
                return None, "side_mismatch"
            return tracked, "resolved_by_correlation_key"

        candidates = [
            row
            for row in self._opportunity_open_outcomes.values()
            if str(row.symbol) == str(symbol) and self._is_closing_side(row.side, current_side)
        ]
        if not candidates:
            return None, "missing"
        if len(candidates) > 1:
            return None, "ambiguous"
        return candidates[0], "resolved_by_symbol_singleton"

    def _record_decision_event(
        self,
        event_type: str,
        *,
        signal: StrategySignal | None = None,
        request: OrderRequest | None = None,
        status: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if self._decision_journal is None:
            return

        meta: dict[str, object] = {}
        if metadata:
            meta.update({str(k): v for k, v in metadata.items()})
        if signal is not None:
            meta.setdefault("signal_confidence", f"{signal.confidence:.6f}")
            enriched_signal_metadata = self._clone_metadata(signal.metadata)
            self._inject_explainability_metadata(enriched_signal_metadata)
            for key, value in enriched_signal_metadata.items():
                meta.setdefault(f"signal_{key}", value)
        if request is not None:
            meta.setdefault("order_type", request.order_type)
            if request.time_in_force:
                meta.setdefault("time_in_force", request.time_in_force)
            if request.client_order_id:
                meta.setdefault("client_order_id", request.client_order_id)
            enriched_request_metadata = self._clone_metadata(getattr(request, "metadata", None))
            self._inject_explainability_metadata(enriched_request_metadata)
            for key, value in enriched_request_metadata.items():
                meta.setdefault(f"order_{key}", value)

        symbol = request.symbol if request else (signal.symbol if signal else None)
        side = None
        if signal is not None:
            side = signal.side.upper()
        elif request is not None:
            side = request.side
        quantity = request.quantity if request else None
        price = request.price if request else None

        event = TradingDecisionEvent(
            event_type=event_type,
            timestamp=self._clock(),
            environment=self.environment,
            portfolio=self.portfolio_id,
            risk_profile=self.risk_profile,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            status=status,
            metadata=meta,
        )
        try:
            self._decision_journal.record(event)
        except Exception:  # pragma: no cover - błąd w dzienniku nie powinien zatrzymać handlu
            _LOGGER.exception("Nie udało się zapisać zdarzenia audytu decyzji: %s", event_type)

    def _resolve_signal_mode(self, signal: StrategySignal) -> str:
        """Wyznacza tryb sygnału na podstawie atrybutów i metadanych."""

        metadata = getattr(signal, "metadata", None)
        candidate: str | None = None
        if isinstance(metadata, Mapping):
            metadata = dict(metadata)
        else:
            metadata = {}

        for key in ("mode", "signal_mode", "source", "origin", "strategy_type"):
            value = getattr(signal, key, None)
            if isinstance(value, str) and value.strip():
                candidate = value.strip().lower()
                break
            meta_value = metadata.get(key)
            if isinstance(meta_value, str) and meta_value.strip():
                candidate = meta_value.strip().lower()
                break

        if candidate:
            if candidate in self._signal_mode_priorities:
                return candidate
            if candidate in self._ai_signal_modes:
                return candidate
            if candidate in self._rules_signal_modes:
                return candidate

        if metadata:
            indicator_keys = {"probability", "model_name", "prediction", "threshold"}
            if indicator_keys & set(metadata):
                return self._ai_signal_modes[0] if self._ai_signal_modes else "ai"
            ai_manager_payload = metadata.get("ai_manager")
            if isinstance(ai_manager_payload, Mapping):
                return self._ai_signal_modes[0] if self._ai_signal_modes else "ai"

        return candidate or "default"

    def _update_ai_failover_state(self) -> bool:
        monitor = self._ai_health_monitor
        if monitor is None:
            self._ai_health_status = None
            if self._ai_failover_active:
                self._ai_failover_active = False
                self._ai_failover_reason = None
                self._record_decision_event(
                    "ai_failover",
                    status="cleared",
                    metadata={"reason": "ai_health_monitor_unavailable"},
                )
            return False

        try:
            status = monitor.snapshot()
        except Exception as exc:  # pragma: no cover - defensywnie wobec monitorów zewnętrznych
            _LOGGER.exception("AI health monitor snapshot failed")
            self._ai_health_status = None
            reason = "ai_health_snapshot_error"
            if not self._ai_failover_active:
                self._ai_failover_active = True
                self._ai_failover_reason = reason
                self._record_decision_event(
                    "ai_failover",
                    status="activated",
                    metadata={"reason": reason, "error": str(exc)},
                )
            elif not self._ai_failover_reason:
                self._ai_failover_reason = reason
            return True
        self._ai_health_status = status
        degraded = bool(status.degraded or status.backend_degraded or status.quality_failures > 0)
        reason = status.reason or ("backend_degraded" if status.backend_degraded else None)

        if degraded:
            if not self._ai_failover_active:
                self._ai_failover_active = True
                self._ai_failover_reason = reason
                metadata = {"reason": reason or "ai_backend_degraded"}
                if status.details:
                    metadata["details"] = ",".join(status.details)
                self._record_decision_event(
                    "ai_failover",
                    status="activated",
                    metadata=metadata,
                )
        elif self._ai_failover_active:
            self._ai_failover_active = False
            self._ai_failover_reason = None
            metadata = {"reason": reason or (status.reason or "")}
            self._record_decision_event(
                "ai_failover",
                status="cleared",
                metadata=metadata,
            )

        return self._ai_failover_active

    def _record_tco_execution(
        self,
        *,
        signal: StrategySignal,
        request: OrderRequest,
        result: OrderResult,
        order_id: str,
        avg_price: float,
        filled_qty: float,
    ) -> None:
        reporter = self._tco_reporter
        if reporter is None:
            return

        strategy_name = self._strategy_name or str(
            signal.metadata.get(
                "strategy", request.metadata.get("strategy") if request.metadata else request.symbol
            )
        )
        reference_price = request.price
        raw_response = result.raw_response if isinstance(result.raw_response, Mapping) else {}
        commission = 0.0
        fee_asset = None
        if isinstance(raw_response, Mapping):
            try:
                commission = float(raw_response.get("fee", 0.0))
            except (TypeError, ValueError):
                commission = 0.0
            fee_asset = raw_response.get("fee_asset")

        metadata: dict[str, object] = {}
        metadata.update(self._tco_metadata)
        metadata.setdefault("order_id", order_id)
        metadata.setdefault("controller", self.__class__.__name__)
        metadata.setdefault("status", result.status or "filled")
        metadata.setdefault("signal_confidence", f"{signal.confidence:.6f}")
        if fee_asset:
            metadata.setdefault("fee_asset", fee_asset)
        for key, value in signal.metadata.items():
            metadata.setdefault(f"signal_{key}", value)
        if request.metadata:
            for key, value in request.metadata.items():
                metadata.setdefault(f"order_{key}", value)

        reporter.record_execution(
            strategy=strategy_name,
            risk_profile=self.risk_profile,
            instrument=request.symbol,
            exchange=self._exchange_name or self.environment,
            side=request.side,
            quantity=filled_qty,
            executed_price=avg_price,
            reference_price=reference_price,
            commission=commission,
            metadata=metadata,
        )

    # ----------------------------------------------- API -----------------------------------------------
    def process_signals(self, signals: Sequence[StrategySignal]) -> list[OrderResult]:
        """Przetwarza listę sygnałów strategii i zarządza alertami."""
        results: list[OrderResult] = []
        ai_blocked = self._update_ai_failover_state()

        prioritized: list[tuple[int, int, StrategySignal, str]] = []
        for index, signal in enumerate(signals):
            metric_labels = dict(self._metric_labels)
            metric_labels["symbol"] = signal.symbol
            self._metric_signals_total.inc(labels={**metric_labels, "status": "received"})
            mode = self._resolve_signal_mode(signal)
            self._record_decision_event(
                "signal_received",
                signal=signal,
                status="received",
                metadata={
                    "intent": str(getattr(signal, "intent", "single")),
                    "leg_count": str(len(getattr(signal, "legs", ()) or ())),
                },
            )
            if ai_blocked and mode in self._ai_signal_modes:
                self._record_decision_event(
                    "signal_skipped",
                    signal=signal,
                    status="skipped",
                    metadata={"reason": "ai_failover_active", "mode": mode},
                )
                continue
            priority = self._signal_mode_priorities.get(mode, self._default_signal_priority)
            prioritized.append((priority, index, signal, mode))

        prioritized.sort(key=lambda item: (-item[0], item[1]))
        for _priority, _index, signal, _mode in prioritized:
            expanded_signals = self._expand_signal(signal)
            if not expanded_signals:
                continue
            for expanded_signal in expanded_signals:
                per_leg_labels = dict(self._metric_labels)
                per_leg_labels["symbol"] = expanded_signal.symbol
                try:
                    result = self._handle_signal(expanded_signal)
                except Exception:  # noqa: BLE001
                    _LOGGER.exception(
                        "Błąd podczas przetwarzania rozszerzonego sygnału %s",
                        expanded_signal,
                    )
                    raise
                if result is not None:
                    results.append(result)
                    normalized_status = _normalize_execution_status(result.status)
                    metric_result = (
                        "executed"
                        if normalized_status in _FILLED_EXECUTION_STATUSES
                        else "not_filled"
                    )
                    self._metric_orders_total.inc(
                        labels={
                            **per_leg_labels,
                            "result": metric_result,
                            "side": expanded_signal.side.upper(),
                        },
                    )

        self.maybe_report_health()
        return results

    def maybe_report_health(self, *, force: bool = False) -> None:
        """Publikuje raport health-check, gdy minął interwał lub wymusimy wysyłkę."""
        if self._health_interval.total_seconds() == 0 and not force:
            return

        now = self._clock()
        if not force and now - self._last_health_report < self._health_interval:
            return

        snapshot = self.alert_router.health_snapshot()
        body_lines = []
        for channel, data in snapshot.items():
            status = data.get("status", "unknown")
            rest = {k: v for k, v in data.items() if k != "status"}
            details = ", ".join(f"{k}={v}" for k, v in rest.items())
            if details:
                body_lines.append(f"{channel}: {status} ({details})")
            else:
                body_lines.append(f"{channel}: {status}")
        body = "\n".join(body_lines) if body_lines else "Brak kanałów alertowych do zraportowania."

        message = AlertMessage(
            category="health",
            title="Raport health-check kanałów alertowych",
            body=body,
            severity="info",
            context={
                "environment": self.environment,
                "portfolio": self.portfolio_id,
                "risk_profile": self.risk_profile,
                "channel_count": str(len(snapshot)),
                "generated_at": now.isoformat(),
            },
        )
        _LOGGER.info("Publikuję raport health-check (%s kanałów)", len(snapshot))
        try:
            self.alert_router.dispatch(message)
        except Exception:  # pragma: no cover - diagnostyka kanałów alertowych
            _LOGGER.exception("Nie udało się wysłać raportu health-check")
            return
        self._last_health_report = now
        self._metric_health_reports.inc(labels=self._metric_labels)

    def _expand_signal(self, signal: StrategySignal) -> Sequence[StrategySignal]:
        intent_raw = getattr(signal, "intent", "") or ""
        intent = str(intent_raw).strip().lower()
        legs_attr = getattr(signal, "legs", ()) or ()
        legs: Sequence[object] = tuple(legs_attr) if isinstance(legs_attr, SequenceABC) else tuple()
        base_metadata = self._clone_metadata(getattr(signal, "metadata", None))
        parent_quantity: float | None = None
        if signal.quantity is not None:
            try:
                parent_quantity = float(signal.quantity)
            except (TypeError, ValueError) as exc:
                raise ValueError("StrategySignal.quantity musi być liczbą dodatnią") from exc
            if parent_quantity <= 0:
                raise ValueError("StrategySignal.quantity musi być dodatnia")
            base_metadata.setdefault("quantity", parent_quantity)
        if not intent:
            if legs:
                intent = "multi_leg"
            else:
                side_candidate = _normalize_trade_side(signal.side)
                if side_candidate is None and str(signal.side).upper() in _NEUTRAL_SIDES:
                    intent = "neutral"
                else:
                    intent = "single"

        if intent in _NEUTRAL_INTENTS or str(signal.side).upper() in _NEUTRAL_SIDES:
            metric_labels = dict(self._metric_labels)
            metric_labels["symbol"] = signal.symbol
            self._metric_signals_total.inc(labels={**metric_labels, "status": "neutral"})
            self._record_decision_event(
                "signal_neutral",
                signal=signal,
                status="neutral",
                metadata={"intent": intent},
            )
            _LOGGER.debug("Pomijam neutralny sygnał %s intent=%s", signal.symbol, intent)
            return []

        if legs:
            expanded: list[StrategySignal] = []
            leg_count = len(legs)
            parent_client_order_id_raw = base_metadata.get("client_order_id")
            parent_client_order_id = (
                str(parent_client_order_id_raw).strip()
                if parent_client_order_id_raw is not None
                else ""
            )
            seen_client_order_ids: set[str] = set()
            for index, leg in enumerate(legs):
                leg_side = getattr(leg, "side", None)
                normalized_side = _normalize_trade_side(leg_side)
                if normalized_side is None:
                    _LOGGER.debug(
                        "Pomijam nogę sygnału %s o nierozpoznanym kierunku %s",
                        signal.symbol,
                        leg_side,
                    )
                    continue
                leg_quantity = getattr(leg, "quantity", None)
                if leg_quantity is not None:
                    try:
                        leg_quantity = float(leg_quantity)
                    except (TypeError, ValueError):
                        _LOGGER.debug(
                            "Pomijam nogę sygnału %s z niepoprawną wielkością: %s",
                            signal.symbol,
                            leg_quantity,
                        )
                        continue
                    if leg_quantity <= 0:
                        _LOGGER.debug(
                            "Pomijam nogę sygnału %s z nie-dodatnią wielkością: %s",
                            signal.symbol,
                            leg_quantity,
                        )
                        continue
                metadata = self._clone_metadata(getattr(leg, "metadata", None))
                combined_metadata: dict[str, object] = {**base_metadata}
                combined_metadata.update(metadata)
                combined_metadata["signal_intent"] = intent or "multi_leg"
                combined_metadata["leg_index"] = index
                combined_metadata["leg_count"] = leg_count

                client_order_id_raw = combined_metadata.get("client_order_id")
                client_order_id = (
                    str(client_order_id_raw).strip() if client_order_id_raw is not None else ""
                )
                has_duplicate = bool(client_order_id and client_order_id in seen_client_order_ids)
                should_force_per_leg = bool(
                    leg_count > 1
                    and parent_client_order_id
                    and (
                        not client_order_id
                        or client_order_id == parent_client_order_id
                        or has_duplicate
                    )
                )
                if should_force_per_leg:
                    per_leg_client_order_id = f"{parent_client_order_id}-L{index + 1}"
                    while per_leg_client_order_id in seen_client_order_ids:
                        per_leg_client_order_id = f"{per_leg_client_order_id}-dup"
                    combined_metadata["parent_client_order_id"] = parent_client_order_id
                    combined_metadata["client_order_id"] = per_leg_client_order_id
                    client_order_id = per_leg_client_order_id
                elif has_duplicate and client_order_id:
                    per_leg_client_order_id = f"{client_order_id}-L{index + 1}"
                    while per_leg_client_order_id in seen_client_order_ids:
                        per_leg_client_order_id = f"{per_leg_client_order_id}-dup"
                    combined_metadata["client_order_id"] = per_leg_client_order_id
                    client_order_id = per_leg_client_order_id

                if client_order_id:
                    seen_client_order_ids.add(client_order_id)

                if leg_quantity is not None:
                    combined_metadata["quantity"] = leg_quantity
                exchange = getattr(leg, "exchange", None)
                if exchange:
                    combined_metadata["exchange"] = exchange
                leg_confidence = getattr(leg, "confidence", None)
                expanded.append(
                    StrategySignal(
                        symbol=getattr(leg, "symbol", "") or signal.symbol,
                        side=normalized_side,
                        confidence=leg_confidence
                        if leg_confidence is not None
                        else signal.confidence,
                        quantity=leg_quantity if leg_quantity is not None else parent_quantity,
                        intent="single",
                        metadata=combined_metadata,
                    )
                )
            if not expanded:
                metric_labels = dict(self._metric_labels)
                metric_labels["symbol"] = signal.symbol
                self._metric_signals_total.inc(labels={**metric_labels, "status": "skipped"})
                self._record_decision_event(
                    "signal_skipped",
                    signal=signal,
                    status="skipped",
                    metadata={"reason": "no_valid_legs", "intent": intent or "multi_leg"},
                )
            return expanded

        normalized_side = _normalize_trade_side(signal.side)
        if normalized_side is None:
            if intent not in _NEUTRAL_INTENTS:
                metric_labels = dict(self._metric_labels)
                metric_labels["symbol"] = signal.symbol
                self._metric_signals_total.inc(labels={**metric_labels, "status": "skipped"})
                self._record_decision_event(
                    "signal_skipped",
                    signal=signal,
                    status="skipped",
                    metadata={"reason": "unsupported_side", "intent": intent or "single"},
                )
            _LOGGER.debug(
                "Pomijam sygnał %s o nierozpoznanym kierunku %s",
                signal.symbol,
                signal.side,
            )
            return []

        combined_metadata = {**base_metadata}
        combined_metadata.setdefault("signal_intent", intent or "single")
        return [
            StrategySignal(
                symbol=signal.symbol,
                side=normalized_side,
                confidence=signal.confidence,
                quantity=parent_quantity,
                intent="single",
                metadata=combined_metadata,
            )
        ]

    # ------------------------------------------- internals ----------------------------------------------
    def _handle_signal(self, signal: StrategySignal) -> OrderResult | None:
        self._emit_signal_alert(signal)
        decision_metadata: Mapping[str, object] | None = None
        metric_labels = dict(self._metric_labels)
        metric_labels["symbol"] = signal.symbol
        if self._decision_engine_enabled():
            candidate, rejection_info = self._build_decision_candidate(signal)
            if candidate is None:
                self._handle_decision_filtered(signal, rejection_info)
                self._metric_signals_total.inc(labels={**metric_labels, "status": "rejected"})
                return None
            evaluation = self._evaluate_decision_candidate(candidate)
            if evaluation is None:
                _LOGGER.warning(
                    "DecisionOrchestrator nie zwrócił poprawnej ewaluacji dla %s %s",
                    signal.side.upper(),
                    signal.symbol,
                )
                self._metric_signals_total.inc(labels={**metric_labels, "status": "rejected"})
                return None
            decision_metadata = self._serialize_decision_evaluation(evaluation)
            self._record_decision_evaluation_event(signal, evaluation)
            if not getattr(evaluation, "accepted", False):
                self._handle_decision_rejection(signal, evaluation)
                self._metric_signals_total.inc(labels={**metric_labels, "status": "rejected"})
                return None

        request = self._build_order_request(signal, extra_metadata=decision_metadata)
        if self._is_opportunity_autonomy_enforced(signal, request):
            permission, diagnostics = self._evaluate_opportunity_execution_permission(
                signal=signal,
                request=request,
            )
            metadata: dict[str, object] = {
                "environment": self.environment,
                **dict(diagnostics),
            }
            metadata.update(
                self._extract_upstream_autonomy_governance_metadata(
                    signal=signal,
                    request=request,
                )
            )
            if permission is not None:
                metadata.update(permission.to_dict())
                metadata["execution_permission"] = (
                    "allowed" if permission.autonomous_execution_allowed else "blocked"
                )
                metadata["autonomy_primary_reason"] = permission.primary_reason
                if permission.denial_reason:
                    metadata["blocking_reason"] = permission.denial_reason
            self._attach_opportunity_autonomy_downgrade_chain_metadata(metadata)
            blocked = permission is None or not permission.autonomous_execution_allowed
            self._record_decision_event(
                "opportunity_autonomy_enforcement",
                signal=signal,
                request=request,
                status="blocked" if blocked else "allowed",
                metadata=metadata,
            )
            if blocked:
                self._metric_signals_total.inc(labels={**metric_labels, "status": "rejected"})
                return None
            autonomy_chain = self._extract_opportunity_autonomy_provenance_chain(metadata)
            if autonomy_chain:
                request = replace(
                    request,
                    metadata={
                        **dict(request.metadata or {}),
                        **autonomy_chain,
                    },
                )
            correlation_key = str(
                (request.metadata or {}).get("opportunity_shadow_record_key") or ""
            ).strip()
            existing_open_tracker = (
                self._opportunity_open_outcomes.get(correlation_key) if correlation_key else None
            )
            mode_raw = (request.metadata or {}).get("opportunity_autonomy_mode")
            duplicate_open_guard_enabled = str(mode_raw or "").strip().lower() in {
                "paper_autonomous",
                "live_autonomous",
            }
            if (
                duplicate_open_guard_enabled
                and
                existing_open_tracker is not None
                and str(existing_open_tracker.symbol) == str(request.symbol)
                and not self._is_closing_side(str(existing_open_tracker.side), str(request.side))
            ):
                self._metric_signals_total.inc(labels={**metric_labels, "status": "skipped"})
                self._record_decision_event(
                    "signal_skipped",
                    signal=signal,
                    request=request,
                    status="skipped",
                    metadata={
                        "reason": "duplicate_autonomous_open_reentry_suppressed",
                        "proxy_correlation_key": correlation_key,
                        "existing_open_side": str(existing_open_tracker.side),
                    },
                )
                return None
        account = self.account_snapshot_provider()
        risk_result = self.risk_engine.apply_pre_trade_checks(
            request,
            account=account,
            profile_name=self.risk_profile,
        )

        adjusted_request = request
        if not risk_result.allowed:
            adjusted, effective_risk_result = self._maybe_adjust_request(
                signal,
                request,
                risk_result,
                account,
            )
            if adjusted is None:
                self._emit_order_rejected_alert(signal, request, effective_risk_result)
                self._handle_liquidation_state(effective_risk_result)
                self._metric_signals_total.inc(labels={**metric_labels, "status": "rejected"})
                adjustments = effective_risk_result.adjustments or {}
                metadata = {
                    "reason": effective_risk_result.reason or "",
                    "available_margin": f"{account.available_margin:.8f}",
                    "total_equity": f"{account.total_equity:.8f}",
                    "maintenance_margin": f"{account.maintenance_margin:.8f}",
                }
                metadata.update({f"adjust_{k}": v for k, v in adjustments.items()})
                self._record_decision_event(
                    "risk_rejected",
                    signal=signal,
                    request=request,
                    status="rejected",
                    metadata=metadata,
                )
                return None
            adjusted_request = adjusted
            self._record_decision_event(
                "risk_adjusted",
                signal=signal,
                request=adjusted_request,
                status="adjusted",
                metadata={
                    "original_quantity": f"{request.quantity:.8f}",
                    "adjusted_quantity": f"{adjusted_request.quantity:.8f}",
                    "reason": risk_result.reason or "",
                },
            )
            risk_result = effective_risk_result
            self._metric_signals_total.inc(labels={**metric_labels, "status": "adjusted"})

        adjusted_request = self._ensure_client_order_id(adjusted_request)

        self._metric_signals_total.inc(labels={**metric_labels, "status": "accepted"})
        self._record_decision_event(
            "risk_check_passed",
            signal=signal,
            request=adjusted_request,
            status="allowed",
            metadata={
                "available_margin": f"{account.available_margin:.8f}",
                "total_equity": f"{account.total_equity:.8f}",
                "maintenance_margin": f"{account.maintenance_margin:.8f}",
            },
        )
        self._metric_orders_total.inc(
            labels={**metric_labels, "result": "submitted", "side": adjusted_request.side}
        )
        self._record_decision_event(
            "order_submitted",
            signal=signal,
            request=adjusted_request,
            status="submitted",
        )

        try:
            should_execute_open_leg = self._maybe_reverse_position(
                signal,
                adjusted_request,
                metric_labels,
            )
        except Exception as exc:  # noqa: BLE001
            self._emit_execution_error_alert(signal, adjusted_request, exc)
            self._metric_orders_total.inc(
                labels={**metric_labels, "result": "failed", "side": adjusted_request.side},
            )
            self._record_decision_event(
                "order_reverse_failed",
                signal=signal,
                request=adjusted_request,
                status="failed",
                metadata={"error": str(exc)},
            )
            raise
        if not should_execute_open_leg:
            self._handle_liquidation_state(risk_result)
            return None
        try:
            result = self.execution_service.execute(adjusted_request, self._execution_context)
        except Exception as exc:  # noqa: BLE001
            self._emit_execution_error_alert(signal, adjusted_request, exc)
            self._handle_liquidation_state(risk_result)
            self._metric_orders_total.inc(
                labels={**metric_labels, "result": "failed", "side": adjusted_request.side},
            )
            self._record_decision_event(
                "order_failed",
                signal=signal,
                request=adjusted_request,
                status="failed",
                metadata={"error": str(exc)},
            )
            raise

        normalized_status = _normalize_execution_status(result.status)
        is_partial = normalized_status in _PARTIAL_EXECUTION_STATUSES
        is_filled = normalized_status in _FILLED_EXECUTION_STATUSES
        if is_filled or is_partial:
            self._emit_order_filled_alert(signal, adjusted_request, result, partial=is_partial)
        else:
            self._emit_order_not_filled_alert(
                signal,
                adjusted_request,
                result,
                normalized_status=normalized_status,
            )
        order_id = result.order_id or ""
        execution_avg_price = result.avg_price
        execution_filled_qty = result.filled_quantity
        if is_filled:
            if execution_avg_price is None:
                execution_avg_price = adjusted_request.price
            if execution_filled_qty is None:
                execution_filled_qty = adjusted_request.quantity
        metadata: dict[str, object] = {
            "order_id": order_id,
            "filled_quantity": (
                "null" if execution_filled_qty is None else f"{execution_filled_qty:.8f}"
            ),
            "avg_price": "null" if execution_avg_price is None else f"{execution_avg_price:.8f}",
            "status": normalized_status,
        }
        if isinstance(result.raw_response, TypingMapping):
            fee = result.raw_response.get("fee")
            fee_asset = result.raw_response.get("fee_asset")
            if fee is not None:
                metadata["fee"] = fee
            if fee_asset:
                metadata["fee_asset"] = fee_asset
        if is_filled:
            self._record_decision_event(
                "order_executed",
                signal=signal,
                request=adjusted_request,
                status=normalized_status,
                metadata=metadata,
            )
            self._record_tco_execution(
                signal=signal,
                request=adjusted_request,
                result=result,
                order_id=order_id,
                avg_price=(
                    execution_avg_price
                    if execution_avg_price is not None
                    else (adjusted_request.price or 0.0)
                ),
                filled_qty=(
                    execution_filled_qty
                    if execution_filled_qty is not None
                    else adjusted_request.quantity
                ),
            )
        elif is_partial:
            self._record_decision_event(
                "order_partially_executed",
                signal=signal,
                request=adjusted_request,
                status=normalized_status,
                metadata=metadata,
            )
            if execution_avg_price is not None and execution_filled_qty is not None:
                self._record_tco_execution(
                    signal=signal,
                    request=adjusted_request,
                    result=result,
                    order_id=order_id,
                    avg_price=execution_avg_price,
                    filled_qty=execution_filled_qty,
                )
        else:
            self._record_decision_event(
                "order_execution_result",
                signal=signal,
                request=adjusted_request,
                status=normalized_status,
                metadata=metadata,
            )
        self._try_attach_opportunity_outcome_label(
            signal=signal,
            request=adjusted_request,
            result=result,
            normalized_status=normalized_status,
            metadata=metadata,
        )
        self._handle_liquidation_state(risk_result)
        return result

    def _is_opportunity_autonomy_enforced(
        self,
        signal: StrategySignal,
        request: OrderRequest,
    ) -> bool:
        signal_metadata = signal.metadata if isinstance(signal.metadata, Mapping) else {}
        request_metadata = request.metadata if isinstance(request.metadata, Mapping) else {}
        keys = (
            "opportunity_autonomy_mode",
            "opportunity_autonomy_decision",
        )
        return any(key in signal_metadata or key in request_metadata for key in keys)

    def _select_opportunity_autonomy_payload(
        self,
        signal: StrategySignal,
        request: OrderRequest,
    ) -> tuple[Mapping[str, object], str] | None:
        signal_metadata = signal.metadata if isinstance(signal.metadata, Mapping) else {}
        request_metadata = request.metadata if isinstance(request.metadata, Mapping) else {}
        request_decision_payload_raw = request_metadata.get("opportunity_autonomy_decision")
        signal_decision_payload_raw = signal_metadata.get("opportunity_autonomy_decision")

        def _as_non_empty_string(value: object | None) -> str | None:
            if value is None:
                return None
            candidate = str(value).strip()
            return candidate or None

        def _normalize_reason_sequence(value: object | None) -> tuple[str, ...] | None:
            if not isinstance(value, SequenceABC) or isinstance(value, (str, bytes, bytearray)):
                return None
            normalized: list[str] = []
            for item in value:
                candidate = _as_non_empty_string(item)
                if candidate is not None:
                    normalized.append(candidate)
            return tuple(normalized)

        def _normalize_evidence_summary(value: object | None) -> Mapping[str, object] | None:
            if not isinstance(value, Mapping):
                return None
            normalized: dict[str, object] = {}
            for key, entry in sorted(value.items(), key=lambda item: str(item[0])):
                normalized[str(key)] = entry
            return normalized

        def _has_useful_governance_envelope(payload: Mapping[str, object]) -> bool:
            requested_mode = _as_non_empty_string(payload.get("requested_mode"))
            effective_mode = _as_non_empty_string(payload.get("effective_mode"))
            downgraded_raw = payload.get("downgraded")
            primary_reason = _as_non_empty_string(payload.get("primary_reason"))
            downgrade_source = _as_non_empty_string(payload.get("downgrade_source"))
            downgrade_step_count_raw = payload.get("downgrade_step_count")
            decision_source = _as_non_empty_string(payload.get("decision_source"))
            inference_model = _as_non_empty_string(payload.get("inference_model"))
            inference_model_version = _as_non_empty_string(payload.get("inference_model_version"))
            blocking_reasons = _normalize_reason_sequence(payload.get("blocking_reasons"))
            warnings = _normalize_reason_sequence(payload.get("warnings"))
            evidence_summary = _normalize_evidence_summary(payload.get("evidence_summary"))
            return any(
                (
                    requested_mode is not None,
                    effective_mode is not None,
                    isinstance(downgraded_raw, bool),
                    primary_reason is not None,
                    downgrade_source is not None,
                    (
                        isinstance(downgrade_step_count_raw, int)
                        and not isinstance(downgrade_step_count_raw, bool)
                    ),
                    decision_source is not None,
                    inference_model is not None,
                    inference_model_version is not None,
                    blocking_reasons is not None and len(blocking_reasons) > 0,
                    warnings is not None and len(warnings) > 0,
                    evidence_summary is not None and len(evidence_summary) > 0,
                )
            )

        request_payload = (
            request_decision_payload_raw
            if isinstance(request_decision_payload_raw, Mapping)
            else None
        )
        signal_payload = (
            signal_decision_payload_raw
            if isinstance(signal_decision_payload_raw, Mapping)
            else None
        )
        if request_payload is not None and _has_useful_governance_envelope(request_payload):
            return request_payload, "request"
        if signal_payload is not None and _has_useful_governance_envelope(signal_payload):
            return signal_payload, "signal"
        return None

    def _extract_upstream_autonomy_governance_metadata(
        self,
        *,
        signal: StrategySignal,
        request: OrderRequest,
    ) -> Mapping[str, object]:
        selected_payload = self._select_opportunity_autonomy_payload(signal, request)
        if selected_payload is None:
            return {}
        payload, payload_source = selected_payload

        def _as_non_empty_string(value: object | None) -> str | None:
            if value is None:
                return None
            candidate = str(value).strip()
            return candidate or None

        def _normalize_reason_sequence(value: object | None) -> tuple[str, ...] | None:
            if not isinstance(value, SequenceABC) or isinstance(value, (str, bytes, bytearray)):
                return None
            normalized: list[str] = []
            for item in value:
                candidate = _as_non_empty_string(item)
                if candidate is not None:
                    normalized.append(candidate)
            return tuple(normalized)

        def _normalize_evidence_summary(value: object | None) -> Mapping[str, object] | None:
            if not isinstance(value, Mapping):
                return None
            normalized: dict[str, object] = {}
            for key, entry in sorted(value.items(), key=lambda item: str(item[0])):
                normalized[str(key)] = entry
            return normalized

        metadata: dict[str, object] = {}
        metadata["upstream_autonomy_payload_source"] = payload_source
        requested_mode = _as_non_empty_string(payload.get("requested_mode"))
        if requested_mode is not None:
            metadata["upstream_autonomy_requested_mode"] = requested_mode

        effective_mode = _as_non_empty_string(payload.get("effective_mode"))
        if effective_mode is not None:
            metadata["upstream_autonomy_effective_mode"] = effective_mode

        downgraded_raw = payload.get("downgraded")
        if isinstance(downgraded_raw, bool):
            metadata["upstream_autonomy_downgraded"] = downgraded_raw

        primary_reason = _as_non_empty_string(payload.get("primary_reason"))
        if primary_reason is not None:
            metadata["upstream_autonomy_primary_reason"] = primary_reason

        downgrade_source = _as_non_empty_string(payload.get("downgrade_source"))
        if downgrade_source is not None:
            metadata["upstream_autonomy_downgrade_source"] = downgrade_source

        downgrade_step_count_raw = payload.get("downgrade_step_count")
        if isinstance(downgrade_step_count_raw, int) and not isinstance(
            downgrade_step_count_raw, bool
        ):
            metadata["upstream_autonomy_downgrade_step_count"] = downgrade_step_count_raw

        decision_source = _as_non_empty_string(payload.get("decision_source"))
        inference_model = _as_non_empty_string(payload.get("inference_model"))
        inference_model_version = _as_non_empty_string(payload.get("inference_model_version"))
        if decision_source == "policy":
            inference_model = None
            inference_model_version = None
        if decision_source in {"model", "hybrid"} and (
            inference_model is None or inference_model_version is None
        ):
            decision_source = None
            inference_model = None
            inference_model_version = None

        if decision_source is not None:
            metadata["upstream_autonomy_decision_source"] = decision_source
        if inference_model is not None:
            metadata["upstream_autonomy_inference_model"] = inference_model
        if inference_model_version is not None:
            metadata["upstream_autonomy_inference_model_version"] = inference_model_version

        blocking_reasons = _normalize_reason_sequence(payload.get("blocking_reasons"))
        if blocking_reasons is not None:
            metadata["upstream_autonomy_blocking_reasons"] = blocking_reasons

        warnings = _normalize_reason_sequence(payload.get("warnings"))
        if warnings is not None:
            metadata["upstream_autonomy_warnings"] = warnings

        evidence_summary = _normalize_evidence_summary(payload.get("evidence_summary"))
        if evidence_summary is not None:
            metadata["upstream_autonomy_evidence_summary"] = evidence_summary
        return metadata

    def _attach_opportunity_autonomy_downgrade_chain_metadata(
        self, metadata: MutableMapping[str, object]
    ) -> None:
        def _as_non_empty_string(value: object | None) -> str | None:
            if value is None:
                return None
            candidate = str(value).strip()
            return candidate or None

        requested_mode = _as_non_empty_string(
            metadata.get("upstream_autonomy_requested_mode")
            or metadata.get("autonomy_requested_mode")
        )
        upstream_effective_mode = _as_non_empty_string(
            metadata.get("upstream_autonomy_effective_mode")
            or metadata.get("upstream_autonomy_requested_mode")
            or metadata.get("autonomy_requested_mode")
        )
        local_guard_effective_mode = _as_non_empty_string(
            metadata.get("performance_guard_effective_mode") or upstream_effective_mode
        )
        final_mode = _as_non_empty_string(metadata.get("autonomy_mode"))

        if requested_mode is not None:
            metadata["autonomy_requested_mode"] = requested_mode
        if upstream_effective_mode is not None:
            metadata["autonomy_upstream_effective_mode"] = upstream_effective_mode
        if local_guard_effective_mode is not None:
            metadata["autonomy_local_guard_effective_mode"] = local_guard_effective_mode
        if final_mode is not None:
            metadata["autonomy_final_mode"] = final_mode

        blocking_reason = _as_non_empty_string(metadata.get("blocking_reason"))
        primary_reason = _as_non_empty_string(metadata.get("autonomy_primary_reason"))
        upstream_primary_reason = _as_non_empty_string(
            metadata.get("upstream_autonomy_primary_reason")
        )
        guard_primary_reason = _as_non_empty_string(
            metadata.get("performance_guard_primary_reason")
        )
        readiness_clamp_reason = next(
            (
                reason
                for reason in (upstream_primary_reason, primary_reason, blocking_reason)
                if _is_live_autonomy_admission_blocker_reason(reason)
            ),
            None,
        )

        if blocking_reason in {
            "performance_guard_snapshot_source_unavailable",
            "performance_guard_snapshot_load_failed",
            "autonomy_permission_evaluation_failed_after_local_guard",
            "autonomy_permission_evaluation_failed",
        }:
            metadata["autonomy_decisive_stage"] = "fail_closed"
            metadata["autonomy_decisive_reason"] = blocking_reason
            return
        if _as_bool(metadata.get("performance_guard_block_enforced")):
            metadata["autonomy_decisive_stage"] = "local_guard"
            metadata["autonomy_decisive_reason"] = (
                guard_primary_reason or blocking_reason or "performance_guard_local_kill_switch"
            )
            return
        if (
            readiness_clamp_reason is not None
            and requested_mode == "live_autonomous"
            and final_mode == "live_assisted"
        ):
            metadata["autonomy_decisive_stage"] = "readiness_clamp"
            metadata["autonomy_decisive_reason"] = readiness_clamp_reason
            return
        if (
            local_guard_effective_mode is not None
            and upstream_effective_mode is not None
            and local_guard_effective_mode != upstream_effective_mode
        ):
            metadata["autonomy_decisive_stage"] = "local_guard"
            metadata["autonomy_decisive_reason"] = (
                guard_primary_reason or primary_reason or "local_guard_effective_mode_changed"
            )
            return
        if (
            final_mode is not None
            and local_guard_effective_mode is not None
            and final_mode != local_guard_effective_mode
        ):
            metadata["autonomy_decisive_stage"] = "permission_engine"
            metadata["autonomy_decisive_reason"] = (
                blocking_reason or primary_reason or "permission_engine_adjustment"
            )
            return
        if (
            requested_mode is not None
            and upstream_effective_mode is not None
            and requested_mode != upstream_effective_mode
        ):
            metadata["autonomy_decisive_stage"] = "upstream_governance"
            metadata["autonomy_decisive_reason"] = (
                upstream_primary_reason or primary_reason or "upstream_effective_mode_changed"
            )
            return
        metadata["autonomy_decisive_stage"] = "none"
        metadata["autonomy_decisive_reason"] = primary_reason or "autonomy_chain_no_downgrade"

    def _extract_opportunity_autonomy_decision(
        self,
        signal: StrategySignal,
        request: OrderRequest,
        *,
        include_performance_guard_payload: bool = True,
    ) -> OpportunityAutonomyDecision:
        signal_metadata = signal.metadata if isinstance(signal.metadata, Mapping) else {}
        request_metadata = request.metadata if isinstance(request.metadata, Mapping) else {}
        request_decision_payload_raw = request_metadata.get("opportunity_autonomy_decision")
        signal_decision_payload_raw = signal_metadata.get("opportunity_autonomy_decision")
        request_decision_payload = (
            request_decision_payload_raw
            if isinstance(request_decision_payload_raw, Mapping)
            else None
        )
        signal_decision_payload = (
            signal_decision_payload_raw
            if isinstance(signal_decision_payload_raw, Mapping)
            else None
        )
        decision_payload = (
            request_decision_payload
            if request_decision_payload is not None
            else signal_decision_payload
        )
        performance_guard_payload = None
        if include_performance_guard_payload and decision_payload is not None:
            performance_guard_raw = decision_payload.get("performance_guard")
            if isinstance(performance_guard_raw, Mapping):
                performance_guard_payload = performance_guard_raw
        mode_raw = request_metadata.get("opportunity_autonomy_mode")
        if mode_raw is None:
            mode_raw = signal_metadata.get("opportunity_autonomy_mode")
        mode_candidates: list[OpportunityAutonomyMode] = []
        if mode_raw is not None:
            mode_candidates.append(OpportunityAutonomyMode(str(mode_raw).strip().lower()))

        def _as_non_empty_string(value: object | None) -> str | None:
            if value is None:
                return None
            candidate = str(value).strip()
            return candidate or None

        def _normalize_reason_sequence(value: object | None) -> tuple[str, ...]:
            if not isinstance(value, SequenceABC) or isinstance(value, (str, bytes, bytearray)):
                return ()
            normalized: list[str] = []
            for item in value:
                candidate = _as_non_empty_string(item)
                if candidate is not None:
                    normalized.append(candidate)
            return tuple(normalized)

        def _canonical_reason(value: object | None) -> str:
            return str(value or "").strip().lower()

        def _normalize_evidence_summary(value: object | None) -> Mapping[str, object]:
            if not isinstance(value, Mapping):
                return {}
            normalized: dict[str, object] = {}
            for key, entry in sorted(value.items(), key=lambda item: str(item[0])):
                normalized[str(key)] = entry
            return normalized

        payload_blocking_reasons: tuple[str, ...] = ()
        payload_reasons: tuple[str, ...] | None = None
        payload_warnings: tuple[str, ...] = ()
        payload_evidence_summary: Mapping[str, object] = {}
        governance_live_blocking_reason: str | None = None
        if decision_payload is not None:
            effective_mode_raw = decision_payload.get("effective_mode")
            if effective_mode_raw is not None:
                mode_candidates.append(
                    OpportunityAutonomyMode(str(effective_mode_raw).strip().lower())
                )
            payload_blocking_reasons = _normalize_reason_sequence(
                decision_payload.get("blocking_reasons")
            )
            if "reasons" in decision_payload:
                payload_reasons = _normalize_reason_sequence(decision_payload.get("reasons"))
            payload_warnings = _normalize_reason_sequence(decision_payload.get("warnings"))
            payload_evidence_summary = _normalize_evidence_summary(
                decision_payload.get("evidence_summary")
            )
            for reason in payload_blocking_reasons:
                if _is_live_autonomy_admission_blocker_reason(reason):
                    governance_live_blocking_reason = reason.lower()
                    break
        if governance_live_blocking_reason is not None and any(
            candidate == OpportunityAutonomyMode.LIVE_AUTONOMOUS for candidate in mode_candidates
        ):
            mode_candidates.append(OpportunityAutonomyMode.LIVE_ASSISTED)
        if include_performance_guard_payload and performance_guard_payload is not None:
            performance_effective_mode_raw = performance_guard_payload.get("effective_mode")
            if performance_effective_mode_raw is not None:
                mode_candidates.append(
                    OpportunityAutonomyMode(str(performance_effective_mode_raw).strip().lower())
                )
        if mode_raw is None and decision_payload is not None:
            mode_raw = decision_payload.get("mode")
        if mode_raw is not None and not mode_candidates:
            mode_candidates.append(OpportunityAutonomyMode(str(mode_raw).strip().lower()))
        if mode_raw is None and decision_payload is not None and not mode_candidates:
            payload_mode_raw = decision_payload.get("mode")
            if payload_mode_raw is not None:
                mode_candidates.append(
                    OpportunityAutonomyMode(str(payload_mode_raw).strip().lower())
                )
        if not mode_candidates:
            raise ValueError("missing_autonomy_mode")
        mode = min(
            mode_candidates,
            key=lambda item: {
                OpportunityAutonomyMode.DENIED: 0,
                OpportunityAutonomyMode.SHADOW_ONLY: 1,
                OpportunityAutonomyMode.PAPER_AUTONOMOUS: 2,
                OpportunityAutonomyMode.LIVE_ASSISTED: 3,
                OpportunityAutonomyMode.LIVE_AUTONOMOUS: 4,
            }[item],
        )
        primary_reason_raw = None
        if decision_payload is not None:
            payload_primary_reason = decision_payload.get("primary_reason")
            if payload_primary_reason is not None:
                primary_reason_raw = payload_primary_reason
        if governance_live_blocking_reason is not None:
            primary_reason_raw = governance_live_blocking_reason
        if (
            governance_live_blocking_reason is None
            and include_performance_guard_payload
            and performance_guard_payload is not None
        ):
            guard_reason = performance_guard_payload.get("primary_reason")
            if guard_reason is not None:
                primary_reason_raw = guard_reason
        if primary_reason_raw is None:
            primary_reason_raw = request_metadata.get("opportunity_autonomy_primary_reason")
        if primary_reason_raw is None:
            primary_reason_raw = signal_metadata.get("opportunity_autonomy_primary_reason")
        primary_reason = str(primary_reason_raw or "").strip() or "unspecified_primary_reason"
        if payload_reasons:
            canonical_primary_reason = _canonical_reason(primary_reason)
            normalized_reasons: list[str] = []
            matched_primary_reason = False
            for item in payload_reasons:
                if _canonical_reason(item) == canonical_primary_reason:
                    if not matched_primary_reason:
                        normalized_reasons.append(primary_reason)
                        matched_primary_reason = True
                    continue
                normalized_reasons.append(item)
            if not matched_primary_reason:
                reasons = (primary_reason, *payload_reasons)
            else:
                reasons = tuple(normalized_reasons)
        else:
            reasons = (primary_reason,)
        return OpportunityAutonomyDecision(
            mode=mode,
            primary_reason=primary_reason,
            reasons=reasons,
            blocking_reasons=payload_blocking_reasons,
            warnings=payload_warnings,
            evidence_summary=payload_evidence_summary,
        )

    def _evaluate_opportunity_execution_permission(
        self,
        *,
        signal: StrategySignal,
        request: OrderRequest,
    ) -> tuple[OpportunityExecutionPermission | None, Mapping[str, object]]:
        signal_metadata = signal.metadata if isinstance(signal.metadata, Mapping) else {}
        request_metadata = request.metadata if isinstance(request.metadata, Mapping) else {}
        diagnostics: dict[str, object] = {}
        requested_mode_raw = request_metadata.get("opportunity_autonomy_mode")
        if requested_mode_raw is None:
            requested_mode_raw = signal_metadata.get("opportunity_autonomy_mode")
        if requested_mode_raw is not None:
            diagnostics["autonomy_requested_mode"] = str(requested_mode_raw).strip().lower()
        assisted_flag_raw = request_metadata.get("autonomy_assisted_approval")
        if assisted_flag_raw is None:
            assisted_flag_raw = signal_metadata.get("autonomy_assisted_approval")
        assisted_approval = _as_bool(assisted_flag_raw)
        diagnostics["autonomy_assisted_approval_supplied"] = assisted_approval
        request_decision_payload_raw = request_metadata.get("opportunity_autonomy_decision")
        signal_decision_payload_raw = signal_metadata.get("opportunity_autonomy_decision")
        request_decision_payload = (
            request_decision_payload_raw
            if isinstance(request_decision_payload_raw, Mapping)
            else None
        )
        signal_decision_payload = (
            signal_decision_payload_raw
            if isinstance(signal_decision_payload_raw, Mapping)
            else None
        )
        decision_payload = (
            request_decision_payload
            if request_decision_payload is not None
            else signal_decision_payload
        )
        scope_model_version: str | None = None
        scope_decision_source: str | None = None
        for payload in (request_decision_payload, signal_decision_payload):
            if payload is None:
                continue
            if scope_model_version is None:
                payload_model_raw = payload.get("model_version")
                if payload_model_raw is not None:
                    candidate = str(payload_model_raw).strip()
                    if candidate:
                        scope_model_version = candidate
            if scope_decision_source is None:
                payload_source_raw = payload.get("decision_source")
                if payload_source_raw is not None:
                    candidate = str(payload_source_raw).strip()
                    if candidate:
                        scope_decision_source = candidate
        for metadata in (request_metadata, signal_metadata):
            if scope_model_version is None:
                model_raw = metadata.get("opportunity_model_version")
                if model_raw is None:
                    model_raw = metadata.get("model_version")
                if model_raw is not None:
                    candidate = str(model_raw).strip()
                    if candidate:
                        scope_model_version = candidate
            if scope_decision_source is None:
                source_raw = metadata.get("opportunity_decision_source")
                if source_raw is None:
                    source_raw = metadata.get("decision_source")
                if source_raw is not None:
                    candidate = str(source_raw).strip()
                    if candidate:
                        scope_decision_source = candidate
        if scope_model_version is not None:
            diagnostics["performance_guard_scope_model_version"] = scope_model_version
        if scope_decision_source is not None:
            diagnostics["performance_guard_scope_decision_source"] = scope_decision_source
        local_guard_applied = False
        if decision_payload is not None:
            performance_guard_payload = decision_payload.get("performance_guard")
            if isinstance(performance_guard_payload, Mapping):
                diagnostics["performance_guard_applied"] = _as_bool(
                    performance_guard_payload.get("performance_guard_applied")
                )
                diagnostics["performance_guard_hard_breach"] = _as_bool(
                    performance_guard_payload.get("hard_breach")
                )
                diagnostics["performance_guard_blocked"] = _as_bool(
                    performance_guard_payload.get("blocked")
                )
                guard_primary_reason = performance_guard_payload.get("primary_reason")
                if guard_primary_reason is not None:
                    diagnostics["performance_guard_primary_reason"] = str(guard_primary_reason)
                guard_effective_mode = performance_guard_payload.get("effective_mode")
                if guard_effective_mode is not None:
                    diagnostics["performance_guard_effective_mode"] = str(guard_effective_mode)
        try:
            decision = self._extract_opportunity_autonomy_decision(
                signal,
                request,
                include_performance_guard_payload=False,
            )
            diagnostics["autonomy_reasons"] = decision.reasons
            performance_guard_recent_final_window_size = (
                self.performance_guard_recent_final_window_size
                if self.performance_guard_recent_final_window_size is not None
                else 20
            )
            performance_guard_max_scan_labels = (
                self.performance_guard_max_scan_labels
                if self.performance_guard_max_scan_labels is not None
                else 256
            )
            diagnostics["performance_guard_recent_final_window_size"] = (
                performance_guard_recent_final_window_size
            )
            diagnostics["performance_guard_max_scan_labels"] = performance_guard_max_scan_labels
            repository = self._opportunity_shadow_repository
            if repository is None:
                diagnostics["performance_guard_source"] = "missing_repository_fail_closed"
                try:
                    fallback_decision = self._extract_opportunity_autonomy_decision(
                        signal,
                        request,
                        include_performance_guard_payload=True,
                    )
                    fallback_permission = evaluate_opportunity_execution_permission(
                        decision=fallback_decision,
                        environment=self.environment,
                        assisted_approval=assisted_approval,
                    )
                    diagnostics["autonomy_reasons"] = fallback_decision.reasons
                    if _as_bool(diagnostics.get("performance_guard_blocked")):
                        guard_primary_reason = diagnostics.get("performance_guard_primary_reason")
                        fallback_permission = replace(
                            fallback_permission,
                            autonomous_execution_allowed=False,
                            primary_reason=(
                                str(guard_primary_reason).strip()
                                if guard_primary_reason is not None
                                else "performance_guard_local_kill_switch"
                            ),
                            denial_reason="performance_guard_local_kill_switch",
                        )
                        diagnostics["performance_guard_block_enforced"] = True
                    diagnostics["fallback_autonomy_mode"] = fallback_permission.autonomy_mode.value
                    diagnostics["fallback_autonomy_primary_reason"] = (
                        fallback_permission.primary_reason
                    )
                except Exception as fallback_exc:  # noqa: BLE001
                    diagnostics["fallback_permission_error"] = str(fallback_exc)
                return None, {
                    "execution_permission": "blocked",
                    "autonomy_mode": "unavailable",
                    "autonomous_execution_allowed": False,
                    "autonomy_primary_reason": "performance_guard_snapshot_source_unavailable",
                    "blocking_reason": "performance_guard_snapshot_source_unavailable",
                    **diagnostics,
                }
            try:
                lifecycle = OpportunityLifecycleService()
                snapshot, scope_diagnostics = (
                    lifecycle.load_recent_performance_snapshot_with_scope_diagnostics(
                        shadow_repository=repository,
                        snapshot_config=OpportunityPerformanceSnapshotConfig(
                            recent_final_window_size=performance_guard_recent_final_window_size,
                            max_scan_labels=performance_guard_max_scan_labels,
                            scope_environment=self.environment,
                            scope_portfolio=self.portfolio_id,
                            scope_model_version=scope_model_version,
                            scope_decision_source=scope_decision_source,
                            require_scope_provenance=True,
                            require_lineage_provenance=(
                                scope_model_version is not None or scope_decision_source is not None
                            ),
                        ),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                diagnostics["performance_guard_source"] = "local_snapshot_source_of_truth_failed"
                try:
                    fallback_decision = self._extract_opportunity_autonomy_decision(
                        signal,
                        request,
                        include_performance_guard_payload=True,
                    )
                    fallback_permission = evaluate_opportunity_execution_permission(
                        decision=fallback_decision,
                        environment=self.environment,
                        assisted_approval=assisted_approval,
                    )
                    diagnostics["autonomy_reasons"] = fallback_decision.reasons
                    if _as_bool(diagnostics.get("performance_guard_blocked")):
                        guard_primary_reason = diagnostics.get("performance_guard_primary_reason")
                        fallback_permission = replace(
                            fallback_permission,
                            autonomous_execution_allowed=False,
                            primary_reason=(
                                str(guard_primary_reason).strip()
                                if guard_primary_reason is not None
                                else "performance_guard_local_kill_switch"
                            ),
                            denial_reason="performance_guard_local_kill_switch",
                        )
                        diagnostics["performance_guard_block_enforced"] = True
                    diagnostics["fallback_autonomy_mode"] = fallback_permission.autonomy_mode.value
                    diagnostics["fallback_autonomy_primary_reason"] = (
                        fallback_permission.primary_reason
                    )
                except Exception as fallback_exc:  # noqa: BLE001
                    diagnostics["fallback_permission_error"] = str(fallback_exc)
                return None, {
                    "execution_permission": "blocked",
                    "autonomy_mode": "unavailable",
                    "autonomous_execution_allowed": False,
                    "autonomy_primary_reason": "performance_guard_snapshot_load_failed",
                    "blocking_reason": "performance_guard_snapshot_load_failed",
                    "autonomy_permission_error": str(exc),
                    **diagnostics,
                }
            guard_decision = evaluate_autonomy_performance_guard(
                requested_mode=decision.mode,
                input_effective_mode=decision.mode,
                snapshot=snapshot,
            )
            diagnostics["performance_guard_source"] = "local_snapshot_source_of_truth"
            diagnostics["performance_guard_snapshot_window"] = snapshot.recent_window_label
            diagnostics["performance_guard_scope_environment"] = (
                scope_diagnostics.scope_environment or self.environment
            )
            diagnostics["performance_guard_scope_portfolio"] = (
                scope_diagnostics.scope_portfolio or self.portfolio_id
            )
            if scope_model_version is not None:
                diagnostics["performance_guard_scope_model_version"] = scope_model_version
            if scope_decision_source is not None:
                diagnostics["performance_guard_scope_decision_source"] = scope_decision_source
            diagnostics["performance_guard_scoped_label_count"] = (
                scope_diagnostics.scoped_label_count
            )
            diagnostics["performance_guard_excluded_label_count"] = (
                scope_diagnostics.excluded_label_count
            )
            diagnostics["performance_guard_missing_scope_provenance_count"] = (
                scope_diagnostics.missing_scope_provenance_count
            )
            diagnostics["performance_guard_missing_lineage_provenance_count"] = (
                scope_diagnostics.missing_lineage_provenance_count
            )
            diagnostics["performance_guard_applied"] = guard_decision.performance_guard_applied
            diagnostics["performance_guard_hard_breach"] = guard_decision.hard_breach
            diagnostics["performance_guard_blocked"] = guard_decision.blocked
            diagnostics["performance_guard_primary_reason"] = guard_decision.primary_reason
            diagnostics["performance_guard_effective_mode"] = guard_decision.effective_mode.value
            diagnostics["performance_guard_requested_mode"] = guard_decision.requested_mode.value
            diagnostics["performance_guard_input_effective_mode"] = (
                guard_decision.input_effective_mode.value
            )
            local_guard_applied = True
            guard_changed_effective_mode = guard_decision.effective_mode is not decision.mode
            guard_forces_block = guard_decision.blocked
            if guard_changed_effective_mode or guard_forces_block:
                decision = replace(
                    decision,
                    mode=guard_decision.effective_mode,
                    primary_reason=guard_decision.primary_reason,
                    reasons=guard_decision.reasons,
                    warnings=guard_decision.warnings,
                    evidence_summary=guard_decision.evidence_summary,
                )
            else:
                decision = replace(
                    decision,
                    mode=guard_decision.effective_mode,
                )
            diagnostics["autonomy_reasons"] = decision.reasons
            permission = evaluate_opportunity_execution_permission(
                decision=decision,
                environment=self.environment,
                assisted_approval=assisted_approval,
            )
            if _as_bool(diagnostics.get("performance_guard_blocked")):
                guard_primary_reason = diagnostics.get("performance_guard_primary_reason")
                permission = replace(
                    permission,
                    autonomous_execution_allowed=False,
                    primary_reason=(
                        str(guard_primary_reason).strip()
                        if guard_primary_reason is not None
                        else "performance_guard_local_kill_switch"
                    ),
                    denial_reason="performance_guard_local_kill_switch",
                )
                diagnostics["performance_guard_block_enforced"] = True
            return permission, diagnostics
        except Exception as exc:  # noqa: BLE001
            if local_guard_applied:
                return None, {
                    "execution_permission": "blocked",
                    "autonomy_mode": "unavailable",
                    "autonomous_execution_allowed": False,
                    "autonomy_primary_reason": "autonomy_permission_evaluation_failed_after_local_guard",
                    "blocking_reason": "autonomy_permission_evaluation_failed_after_local_guard",
                    "autonomy_permission_error": str(exc),
                    **diagnostics,
                }
            if _as_bool(diagnostics.get("performance_guard_blocked")):
                return None, {
                    "execution_permission": "blocked",
                    "autonomy_mode": "unavailable",
                    "autonomous_execution_allowed": False,
                    "autonomy_primary_reason": str(
                        diagnostics.get("performance_guard_primary_reason")
                        or "performance_guard_local_kill_switch"
                    ),
                    "blocking_reason": "performance_guard_local_kill_switch",
                    "autonomy_permission_error": str(exc),
                    "performance_guard_block_enforced": True,
                    **diagnostics,
                }
            return None, {
                "execution_permission": "blocked",
                "autonomy_mode": "unavailable",
                "autonomous_execution_allowed": False,
                "autonomy_primary_reason": "autonomy_permission_evaluation_failed",
                "blocking_reason": "autonomy_permission_evaluation_failed",
                "autonomy_permission_error": str(exc),
                **diagnostics,
            }

    def _try_attach_opportunity_outcome_label(
        self,
        *,
        signal: StrategySignal,
        request: OrderRequest,
        result: OrderResult,
        normalized_status: str,
        metadata: Mapping[str, object],
    ) -> None:
        repository = self._opportunity_shadow_repository
        if repository is None:
            return
        attach_repo_read_failure_stage: str | None = None
        try:
            existing_labels_by_key = {
                str(row.correlation_key): row for row in repository.load_outcome_labels()
            }
        except Exception:  # pragma: no cover - diagnostics only
            existing_labels_by_key = {}
            attach_repo_read_failure_stage = "outcome_labels_load_failed_before_resolution"
        try:
            shadow_by_key = {str(row.record_key): row for row in repository.load_shadow_records()}
            known_shadow_keys = set(shadow_by_key)
        except Exception:  # pragma: no cover - diagnostics only
            shadow_by_key = {}
            known_shadow_keys = set()
            if attach_repo_read_failure_stage is None:
                attach_repo_read_failure_stage = "shadow_records_load_failed_before_resolution"
        signal_metadata = signal.metadata if isinstance(signal.metadata, Mapping) else {}
        request_metadata = request.metadata if isinstance(request.metadata, Mapping) else {}
        autonomy_chain = self._extract_opportunity_autonomy_provenance_chain(request_metadata)

        def _resolve_runtime_lineage(
            correlation_key_hint: str | None,
            tracker_hint: _OpportunityOpenOutcomeTracker | None = None,
            *,
            prefer_tracker_model_version: bool = False,
        ) -> tuple[str | None, str | None]:
            lineage_model_version: str | None = None
            lineage_decision_source: str | None = None
            tracker_contract_decision_source_locked = False
            if tracker_hint is not None and tracker_hint.decision_source is not None:
                tracker_candidate = str(tracker_hint.decision_source).strip()
                if tracker_candidate:
                    tracker_autonomy_contract_present = any(
                        str(value).strip()
                        for value in (
                            tracker_hint.upstream_autonomy_decision_source,
                            tracker_hint.upstream_autonomy_inference_model,
                            tracker_hint.upstream_autonomy_inference_model_version,
                        )
                        if value is not None
                    )
                    if tracker_autonomy_contract_present:
                        lineage_decision_source = tracker_candidate
                        tracker_contract_decision_source_locked = True
            for payload_raw in (
                request_metadata.get("opportunity_autonomy_decision"),
                signal_metadata.get("opportunity_autonomy_decision"),
            ):
                if not isinstance(payload_raw, Mapping):
                    continue
                if lineage_model_version is None:
                    payload_model_raw = payload_raw.get("model_version")
                    if payload_model_raw is not None:
                        candidate = str(payload_model_raw).strip()
                        if candidate:
                            lineage_model_version = candidate
                if lineage_decision_source is None and not tracker_contract_decision_source_locked:
                    payload_source_raw = payload_raw.get("decision_source")
                    if payload_source_raw is not None:
                        candidate = str(payload_source_raw).strip()
                        if candidate:
                            lineage_decision_source = candidate
            if lineage_decision_source is None and tracker_hint is not None:
                tracker_source_raw = tracker_hint.decision_source
                if tracker_source_raw is not None:
                    tracker_source_candidate = str(tracker_source_raw).strip()
                    if tracker_source_candidate:
                        lineage_decision_source = tracker_source_candidate
            for metadata_source in (request_metadata, signal_metadata):
                if lineage_model_version is None:
                    metadata_model_raw = metadata_source.get("opportunity_model_version")
                    if metadata_model_raw is None:
                        metadata_model_raw = metadata_source.get("model_version")
                    if metadata_model_raw is not None:
                        candidate = str(metadata_model_raw).strip()
                        if candidate:
                            lineage_model_version = candidate
                if lineage_decision_source is None:
                    metadata_source_raw = metadata_source.get("opportunity_decision_source")
                    if metadata_source_raw is None:
                        metadata_source_raw = metadata_source.get("decision_source")
                    if metadata_source_raw is not None:
                        candidate = str(metadata_source_raw).strip()
                        if candidate:
                            lineage_decision_source = candidate
            if (
                prefer_tracker_model_version
                and lineage_model_version is None
                and tracker_hint is not None
            ):
                tracker_model_raw = tracker_hint.model_version
                if tracker_model_raw is not None:
                    tracker_model_candidate = str(tracker_model_raw).strip()
                    if tracker_model_candidate:
                        lineage_model_version = tracker_model_candidate
            correlation_key_candidate = str(correlation_key_hint or "").strip()
            shadow_record = shadow_by_key.get(correlation_key_candidate)
            if shadow_record is not None:
                if lineage_model_version is None:
                    candidate = str(getattr(shadow_record, "model_version", "")).strip()
                    if candidate:
                        lineage_model_version = candidate
                if lineage_decision_source is None:
                    candidate = str(getattr(shadow_record, "decision_source", "")).strip()
                    if candidate:
                        lineage_decision_source = candidate
            return lineage_model_version, lineage_decision_source

        def _resolve_runtime_scope(
            tracker_hint: _OpportunityOpenOutcomeTracker | None,
        ) -> tuple[str, str, str]:
            scope_environment = str(self.environment).strip()
            scope_portfolio = str(self.portfolio_id).strip()
            scope_resolution = "runtime_controller"
            if tracker_hint is not None:
                tracker_environment = (
                    str(tracker_hint.environment_scope).strip()
                    if tracker_hint.environment_scope is not None
                    else ""
                )
                tracker_portfolio = (
                    str(tracker_hint.portfolio_scope).strip()
                    if tracker_hint.portfolio_scope is not None
                    else ""
                )
                if tracker_hint.restored_from_repository:
                    scope_environment = tracker_environment
                    scope_portfolio = tracker_portfolio
                    scope_resolution = (
                        "restored_tracker"
                        if tracker_environment and tracker_portfolio
                        else "restored_tracker_scope_missing"
                    )
                else:
                    if tracker_environment:
                        scope_environment = tracker_environment
                    if tracker_portfolio:
                        scope_portfolio = tracker_portfolio
                    if tracker_environment and tracker_portfolio:
                        scope_resolution = "restored_tracker"
            return scope_environment, scope_portfolio, scope_resolution

        def _is_restored_tracker_scope_gap(
            tracker_hint: _OpportunityOpenOutcomeTracker | None,
        ) -> bool:
            # Auditable precedence exception: when a tracker was restored from legacy/open outcomes
            # but lacks full scope continuity, close/proxy lineage must stay conservative and
            # preserve restored tracker provenance instead of promoting shadow/opportunity lineage.
            return bool(
                tracker_hint is not None
                and tracker_hint.restored_from_repository
                and (
                    tracker_hint.environment_scope is None
                    or tracker_hint.portfolio_scope is None
                )
            )

        correlation_key = str(request_metadata.get("opportunity_shadow_record_key") or "").strip()
        side = str(request.side or "").upper()
        is_filled_or_partial = (
            normalized_status in _FILLED_EXECUTION_STATUSES
            or normalized_status in _PARTIAL_EXECUTION_STATUSES
        )
        avg_price_raw = result.avg_price if result.avg_price is not None else request.price
        avg_price: float | None = None
        try:
            if avg_price_raw is not None:
                avg_price = float(avg_price_raw)
        except (TypeError, ValueError):
            avg_price = None
        raw_decision_timestamp = request_metadata.get("opportunity_decision_timestamp")
        has_decision_timestamp_hint = isinstance(raw_decision_timestamp, str) and bool(
            raw_decision_timestamp.strip()
        )
        timestamp = self._clock()
        if has_decision_timestamp_hint:
            try:
                timestamp = datetime.fromisoformat(raw_decision_timestamp.replace("Z", "+00:00"))
            except ValueError:
                pass
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        timestamp_utc = timestamp.astimezone(timezone.utc)
        proxy_label: OpportunityOutcomeLabel | None = None

        final_label: OpportunityOutcomeLabel | None = None
        partial_label: OpportunityOutcomeLabel | None = None
        final_tracker: _OpportunityOpenOutcomeTracker | None = None
        final_resolution = ""
        unresolved_close_with_correlation_key = False
        open_intent_candidate = False
        replay_open_candidate = False
        if (
            attach_repo_read_failure_stage is not None
            and correlation_key
            and is_filled_or_partial
            and avg_price is not None
            and side in _BUY_SIDES | _SELL_SIDES
        ):
            self._record_decision_event(
                "opportunity_outcome_attach",
                signal=signal,
                request=request,
                status="attach_error",
                metadata={
                    "proxy_correlation_key": correlation_key,
                    "execution_status": normalized_status,
                    "attach_error_stage": attach_repo_read_failure_stage,
                },
            )
            return
        if is_filled_or_partial and avg_price is not None:
            tracked, resolution = self._resolve_open_outcome_tracker(
                symbol=str(request.symbol),
                current_side=side,
                correlation_key=correlation_key,
            )
            final_resolution = resolution
            shadow_record = shadow_by_key.get(correlation_key) if correlation_key else None
            expected_open_side = ""
            if shadow_record is not None:
                proposed_direction = (
                    str(getattr(shadow_record, "proposed_direction", "")).strip().lower()
                )
                expected_open_side = (
                    "BUY"
                    if proposed_direction in {"long", "buy"}
                    else ("SELL" if proposed_direction in {"short", "sell"} else "")
                )
            if correlation_key and resolution == "missing":
                if shadow_record is not None:
                    existing_quality = ""
                    existing_label = existing_labels_by_key.get(correlation_key)
                    if existing_label is not None:
                        existing_quality = str(existing_label.label_quality)
                    open_intent_candidate = (
                        expected_open_side == side
                        and OpportunityShadowRepository._quality_rank(existing_quality)
                        < OpportunityShadowRepository._quality_rank("partial_exit_unconfirmed")
                    )
            if correlation_key and resolution == "side_mismatch" and has_decision_timestamp_hint:
                existing_quality = ""
                existing_label = existing_labels_by_key.get(correlation_key)
                if existing_label is not None:
                    existing_quality = str(existing_label.label_quality)
                existing_provenance = (
                    existing_label.provenance if existing_label is not None else {}
                )
                existing_avg_price = ""
                existing_filled_quantity = ""
                if isinstance(existing_provenance, Mapping):
                    existing_avg_price = str(existing_provenance.get("avg_price", ""))
                    existing_filled_quantity = str(existing_provenance.get("filled_quantity", ""))
                incoming_avg_price = str(metadata.get("avg_price", ""))
                incoming_filled_quantity = str(metadata.get("filled_quantity", ""))
                replay_open_candidate = (
                    expected_open_side == side
                    and OpportunityShadowRepository._quality_rank(existing_quality)
                    <= OpportunityShadowRepository._quality_rank("execution_proxy_pending_exit")
                    and existing_avg_price == incoming_avg_price
                    and existing_filled_quantity == incoming_filled_quantity
                )
            if tracked is not None:
                realized_return_bps = self._realized_return_bps(
                    entry_side=tracked.side,
                    entry_price=tracked.entry_price,
                    exit_price=avg_price,
                )
                horizon_minutes = max(
                    0,
                    int((timestamp_utc - tracked.decision_timestamp).total_seconds() / 60),
                )
                close_quantity = self._safe_float(
                    metadata.get(
                        "filled_quantity",
                        result.filled_quantity
                        if result.filled_quantity is not None
                        else request.quantity,
                    )
                )
                cumulative_closed_quantity = max(
                    0.0, tracked.closed_quantity + max(close_quantity, 0.0)
                )
                tracked.closed_quantity = cumulative_closed_quantity
                self._persist_open_outcome_tracker(tracked)
                has_quantity_proof = (
                    tracked.entry_quantity > 0.0 and cumulative_closed_quantity > 0.0
                )
                is_confirmed_final_close = (
                    normalized_status in _FILLED_EXECUTION_STATUSES
                    and has_quantity_proof
                    and cumulative_closed_quantity + 1e-9 >= tracked.entry_quantity
                )
                if is_confirmed_final_close:
                    existing_quality = ""
                    existing_label = existing_labels_by_key.get(tracked.correlation_key)
                    if existing_label is not None:
                        existing_quality = str(existing_label.label_quality)
                    request_payload_has_explicit_model_version = False
                    request_payload_raw = request_metadata.get("opportunity_autonomy_decision")
                    if isinstance(request_payload_raw, Mapping):
                        payload_model_raw = request_payload_raw.get("model_version")
                        if payload_model_raw is not None and str(payload_model_raw).strip():
                            request_payload_has_explicit_model_version = True
                    signal_payload_has_explicit_model_version = False
                    signal_payload_raw = signal_metadata.get("opportunity_autonomy_decision")
                    if isinstance(signal_payload_raw, Mapping):
                        payload_model_raw = signal_payload_raw.get("model_version")
                        if payload_model_raw is not None and str(payload_model_raw).strip():
                            signal_payload_has_explicit_model_version = True
                    preserve_tracker_model_version = (
                        (not tracked.restored_from_repository)
                        or (tracked.closed_quantity > 0.0)
                        or (
                            OpportunityShadowRepository._quality_rank(existing_quality)
                            >= OpportunityShadowRepository._quality_rank("partial_exit_unconfirmed")
                        )
                    )
                    if (
                        request_payload_has_explicit_model_version
                        or signal_payload_has_explicit_model_version
                    ):
                        preserve_tracker_model_version = False
                    model_version, decision_source = _resolve_runtime_lineage(
                        tracked.correlation_key,
                        tracker_hint=tracked,
                        prefer_tracker_model_version=preserve_tracker_model_version,
                    )
                    if preserve_tracker_model_version:
                        model_version = tracked.model_version or model_version
                    scope_environment, scope_portfolio, scope_resolution = _resolve_runtime_scope(
                        tracked
                    )
                    tracked.model_version = model_version
                    tracked.decision_source = decision_source
                    tracked.environment_scope = scope_environment or None
                    tracked.portfolio_scope = scope_portfolio or None
                    tracked.autonomy_requested_mode = (
                        tracked.autonomy_requested_mode
                        or autonomy_chain.get("autonomy_requested_mode")
                    )
                    tracked.autonomy_upstream_effective_mode = (
                        tracked.autonomy_upstream_effective_mode
                        or autonomy_chain.get("autonomy_upstream_effective_mode")
                    )
                    tracked.autonomy_local_guard_effective_mode = (
                        tracked.autonomy_local_guard_effective_mode
                        or autonomy_chain.get("autonomy_local_guard_effective_mode")
                    )
                    tracked.autonomy_final_mode = tracked.autonomy_final_mode or autonomy_chain.get(
                        "autonomy_final_mode"
                    )
                    tracked.autonomous_execution_allowed = (
                        tracked.autonomous_execution_allowed
                        or autonomy_chain.get("autonomous_execution_allowed")
                    )
                    tracked.assisted_override_used = (
                        tracked.assisted_override_used
                        or autonomy_chain.get("assisted_override_used")
                    )
                    tracked.blocking_reason = tracked.blocking_reason or autonomy_chain.get(
                        "blocking_reason"
                    )
                    tracked.autonomy_decisive_stage = (
                        tracked.autonomy_decisive_stage
                        or autonomy_chain.get("autonomy_decisive_stage")
                    )
                    tracked.autonomy_decisive_reason = (
                        tracked.autonomy_decisive_reason
                        or autonomy_chain.get("autonomy_decisive_reason")
                    )
                    tracked.autonomy_primary_reason = (
                        tracked.autonomy_primary_reason
                        or autonomy_chain.get("autonomy_primary_reason")
                    )
                    tracked.upstream_autonomy_decision_source = (
                        tracked.upstream_autonomy_decision_source
                        or autonomy_chain.get("upstream_autonomy_decision_source")
                    )
                    tracked.upstream_autonomy_inference_model = (
                        tracked.upstream_autonomy_inference_model
                        or autonomy_chain.get("upstream_autonomy_inference_model")
                    )
                    tracked.upstream_autonomy_inference_model_version = (
                        tracked.upstream_autonomy_inference_model_version
                        or autonomy_chain.get("upstream_autonomy_inference_model_version")
                    )
                    final_label = OpportunityOutcomeLabel(
                        symbol=request.symbol,
                        decision_timestamp=tracked.decision_timestamp,
                        correlation_key=tracked.correlation_key,
                        horizon_minutes=horizon_minutes,
                        realized_return_bps=realized_return_bps,
                        max_favorable_excursion_bps=0.0,
                        max_adverse_excursion_bps=0.0,
                        provenance={
                            "source": "trading_controller_exit_result",
                            **self._extract_opportunity_autonomy_provenance_chain_from_tracker(
                                tracked
                            ),
                            **({"environment": scope_environment} if scope_environment else {}),
                            **({"portfolio": scope_portfolio} if scope_portfolio else {}),
                            **(
                                {"scope_continuity": scope_resolution}
                                if scope_resolution != "runtime_controller"
                                else {}
                            ),
                            "entry_price": f"{tracked.entry_price:.8f}",
                            "exit_price": f"{avg_price:.8f}",
                            "order_id": result.order_id or "",
                            "execution_status": normalized_status,
                            "excursion_metrics_available": "false",
                            "max_favorable_excursion_bps_semantics": "not_computed_in_runtime_controller_flow",
                            "max_adverse_excursion_bps_semantics": "not_computed_in_runtime_controller_flow",
                            "return_metric_scope": "entry_exit_realized_only",
                            "close_correlation_resolution": resolution,
                            "entry_quantity": f"{tracked.entry_quantity:.8f}",
                            "closed_quantity_cumulative": f"{cumulative_closed_quantity:.8f}",
                            "close_confirmation": "quantity_and_filled_status",
                            **(
                                {"model_version": model_version}
                                if model_version is not None
                                else {}
                            ),
                            **(
                                {"decision_source": decision_source}
                                if decision_source is not None
                                else {}
                            ),
                        },
                        label_quality="final",
                    )
                    final_tracker = tracked
                else:
                    existing_quality = ""
                    existing_label = existing_labels_by_key.get(tracked.correlation_key)
                    if existing_label is not None:
                        existing_quality = str(existing_label.label_quality)
                    if OpportunityShadowRepository._quality_rank(
                        existing_quality
                    ) < OpportunityShadowRepository._quality_rank("partial_exit_unconfirmed"):
                        preserve_tracker_model_version = (
                            not tracked.restored_from_repository
                            or _is_restored_tracker_scope_gap(tracked)
                        )
                        request_payload_raw = request_metadata.get("opportunity_autonomy_decision")
                        if isinstance(request_payload_raw, Mapping):
                            payload_model_raw = request_payload_raw.get("model_version")
                            if payload_model_raw is not None and str(payload_model_raw).strip():
                                preserve_tracker_model_version = False
                        signal_payload_raw = signal_metadata.get("opportunity_autonomy_decision")
                        if isinstance(signal_payload_raw, Mapping):
                            payload_model_raw = signal_payload_raw.get("model_version")
                            if payload_model_raw is not None and str(payload_model_raw).strip():
                                preserve_tracker_model_version = False
                        model_version, decision_source = _resolve_runtime_lineage(
                            tracked.correlation_key,
                            tracker_hint=tracked,
                            prefer_tracker_model_version=preserve_tracker_model_version,
                        )
                        if preserve_tracker_model_version:
                            model_version = tracked.model_version or model_version
                        scope_environment, scope_portfolio, scope_resolution = (
                            _resolve_runtime_scope(tracked)
                        )
                        tracked.model_version = model_version
                        tracked.decision_source = decision_source
                        tracked.environment_scope = scope_environment or None
                        tracked.portfolio_scope = scope_portfolio or None
                        tracked.autonomy_requested_mode = (
                            tracked.autonomy_requested_mode
                            or autonomy_chain.get("autonomy_requested_mode")
                        )
                        tracked.autonomy_upstream_effective_mode = (
                            tracked.autonomy_upstream_effective_mode
                            or autonomy_chain.get("autonomy_upstream_effective_mode")
                        )
                        tracked.autonomy_local_guard_effective_mode = (
                            tracked.autonomy_local_guard_effective_mode
                            or autonomy_chain.get("autonomy_local_guard_effective_mode")
                        )
                        tracked.autonomy_final_mode = (
                            tracked.autonomy_final_mode or autonomy_chain.get("autonomy_final_mode")
                        )
                        tracked.autonomous_execution_allowed = (
                            tracked.autonomous_execution_allowed
                            or autonomy_chain.get("autonomous_execution_allowed")
                        )
                        tracked.assisted_override_used = (
                            tracked.assisted_override_used
                            or autonomy_chain.get("assisted_override_used")
                        )
                        tracked.blocking_reason = tracked.blocking_reason or autonomy_chain.get(
                            "blocking_reason"
                        )
                        tracked.autonomy_decisive_stage = (
                            tracked.autonomy_decisive_stage
                            or autonomy_chain.get("autonomy_decisive_stage")
                        )
                        tracked.autonomy_decisive_reason = (
                            tracked.autonomy_decisive_reason
                            or autonomy_chain.get("autonomy_decisive_reason")
                        )
                        tracked.autonomy_primary_reason = (
                            tracked.autonomy_primary_reason
                            or autonomy_chain.get("autonomy_primary_reason")
                        )
                        tracked.upstream_autonomy_decision_source = (
                            tracked.upstream_autonomy_decision_source
                            or autonomy_chain.get("upstream_autonomy_decision_source")
                        )
                        tracked.upstream_autonomy_inference_model = (
                            tracked.upstream_autonomy_inference_model
                            or autonomy_chain.get("upstream_autonomy_inference_model")
                        )
                        tracked.upstream_autonomy_inference_model_version = (
                            tracked.upstream_autonomy_inference_model_version
                            or autonomy_chain.get("upstream_autonomy_inference_model_version")
                        )
                        self._persist_open_outcome_tracker(tracked)
                        partial_label = OpportunityOutcomeLabel(
                            symbol=request.symbol,
                            decision_timestamp=tracked.decision_timestamp,
                            correlation_key=tracked.correlation_key,
                            horizon_minutes=horizon_minutes,
                            realized_return_bps=realized_return_bps,
                            max_favorable_excursion_bps=0.0,
                            max_adverse_excursion_bps=0.0,
                            provenance={
                                "source": "trading_controller_partial_exit_result",
                                **self._extract_opportunity_autonomy_provenance_chain_from_tracker(
                                    tracked
                                ),
                                **({"environment": scope_environment} if scope_environment else {}),
                                **({"portfolio": scope_portfolio} if scope_portfolio else {}),
                                **(
                                    {"scope_continuity": scope_resolution}
                                    if scope_resolution != "runtime_controller"
                                    else {}
                                ),
                                "entry_price": f"{tracked.entry_price:.8f}",
                                "exit_price": f"{avg_price:.8f}",
                                "order_id": result.order_id or "",
                                "execution_status": normalized_status,
                                "close_correlation_resolution": resolution,
                                "entry_quantity": f"{tracked.entry_quantity:.8f}",
                                "closed_quantity_cumulative": f"{cumulative_closed_quantity:.8f}",
                                "close_confirmation": "insufficient_evidence_for_final_close",
                                **(
                                    {"model_version": model_version}
                                    if model_version is not None
                                    else {}
                                ),
                                **(
                                    {"decision_source": decision_source}
                                    if decision_source is not None
                                    else {}
                                ),
                            },
                            label_quality="partial_exit_unconfirmed",
                        )
            elif correlation_key and open_intent_candidate and side in _BUY_SIDES | _SELL_SIDES:
                tracker = _OpportunityOpenOutcomeTracker(
                    correlation_key=correlation_key,
                    symbol=str(request.symbol),
                    side=side,
                    entry_price=avg_price,
                    decision_timestamp=timestamp_utc,
                    entry_quantity=max(
                        0.0,
                        self._safe_float(
                            metadata.get(
                                "filled_quantity",
                                result.filled_quantity
                                if result.filled_quantity is not None
                                else request.quantity,
                            )
                        ),
                    ),
                    model_version=None,
                    decision_source=None,
                    autonomy_requested_mode=autonomy_chain.get("autonomy_requested_mode"),
                    autonomy_upstream_effective_mode=autonomy_chain.get(
                        "autonomy_upstream_effective_mode"
                    ),
                    autonomy_local_guard_effective_mode=autonomy_chain.get(
                        "autonomy_local_guard_effective_mode"
                    ),
                    autonomy_final_mode=autonomy_chain.get("autonomy_final_mode"),
                    autonomous_execution_allowed=autonomy_chain.get("autonomous_execution_allowed"),
                    assisted_override_used=autonomy_chain.get("assisted_override_used"),
                    blocking_reason=autonomy_chain.get("blocking_reason"),
                    autonomy_decisive_stage=autonomy_chain.get("autonomy_decisive_stage"),
                    autonomy_decisive_reason=autonomy_chain.get("autonomy_decisive_reason"),
                    autonomy_primary_reason=autonomy_chain.get("autonomy_primary_reason"),
                    upstream_autonomy_decision_source=autonomy_chain.get(
                        "upstream_autonomy_decision_source"
                    ),
                    upstream_autonomy_inference_model=autonomy_chain.get(
                        "upstream_autonomy_inference_model"
                    ),
                    upstream_autonomy_inference_model_version=autonomy_chain.get(
                        "upstream_autonomy_inference_model_version"
                    ),
                    environment_scope=str(self.environment).strip() or None,
                    portfolio_scope=str(self.portfolio_id).strip() or None,
                    restored_from_repository=False,
                )
                tracker_model_version, tracker_decision_source = _resolve_runtime_lineage(
                    correlation_key,
                    tracker_hint=tracker,
                )
                tracker.model_version = tracker_model_version
                tracker.decision_source = tracker_decision_source
                self._opportunity_open_outcomes[correlation_key] = tracker
                self._persist_open_outcome_tracker(tracker)
            elif (
                correlation_key
                and correlation_key in known_shadow_keys
                and resolution in {"missing", "ambiguous", "side_mismatch", "symbol_mismatch"}
                and not open_intent_candidate
                and not replay_open_candidate
            ):
                unresolved_close_with_correlation_key = True
        if unresolved_close_with_correlation_key:
            self._record_decision_event(
                "opportunity_outcome_attach",
                signal=signal,
                request=request,
                status="close_correlation_unresolved",
                metadata={
                    "execution_status": normalized_status,
                    "close_correlation_resolution": final_resolution,
                    "symbol": request.symbol,
                    "proxy_correlation_key": correlation_key,
                },
            )
            return
        if correlation_key and final_label is None and partial_label is None:
            proxy_tracker = self._opportunity_open_outcomes.get(correlation_key)
            model_version, decision_source = _resolve_runtime_lineage(
                correlation_key,
                tracker_hint=proxy_tracker,
                prefer_tracker_model_version=_is_restored_tracker_scope_gap(proxy_tracker),
            )
            scope_environment, scope_portfolio, scope_resolution = _resolve_runtime_scope(
                proxy_tracker
            )
            proxy_label = OpportunityOutcomeLabel(
                symbol=request.symbol,
                decision_timestamp=timestamp_utc,
                correlation_key=correlation_key,
                horizon_minutes=0,
                realized_return_bps=0.0,
                max_favorable_excursion_bps=0.0,
                max_adverse_excursion_bps=0.0,
                provenance={
                    "source": "trading_controller_execution_result",
                    **(
                        self._extract_opportunity_autonomy_provenance_chain_from_tracker(
                            proxy_tracker
                        )
                        if proxy_tracker is not None
                        else autonomy_chain
                    ),
                    **({"environment": scope_environment} if scope_environment else {}),
                    **({"portfolio": scope_portfolio} if scope_portfolio else {}),
                    **(
                        {"scope_continuity": scope_resolution}
                        if scope_resolution != "runtime_controller"
                        else {}
                    ),
                    "order_id": result.order_id or "",
                    "execution_status": normalized_status,
                    "filled_quantity": str(metadata.get("filled_quantity", "")),
                    "avg_price": str(metadata.get("avg_price", "")),
                    **({"model_version": model_version} if model_version is not None else {}),
                    **({"decision_source": decision_source} if decision_source is not None else {}),
                },
                label_quality="execution_proxy_pending_exit",
            )

        labels_to_attach: list[OpportunityOutcomeLabel] = []
        if proxy_label is not None:
            labels_to_attach.append(proxy_label)
        if partial_label is not None:
            labels_to_attach.append(partial_label)
        if final_label is not None:
            labels_to_attach.append(final_label)
        if (
            final_label is None
            and proxy_label is None
            and side in _BUY_SIDES | _SELL_SIDES
            and any(
                row.symbol == str(request.symbol)
                for row in self._opportunity_open_outcomes.values()
            )
            and final_resolution in {"ambiguous", "missing", "side_mismatch", "symbol_mismatch"}
        ):
            self._record_decision_event(
                "opportunity_outcome_attach",
                signal=signal,
                request=request,
                status="close_correlation_unresolved",
                metadata={
                    "execution_status": normalized_status,
                    "close_correlation_resolution": final_resolution,
                    "symbol": request.symbol,
                },
            )
        if not labels_to_attach:
            return
        try:
            attach_result = repository.attach_outcome_labels_idempotent(labels_to_attach)
            attach_metadata: dict[str, object] = {
                "execution_status": normalized_status,
                "proxy_correlation_key": correlation_key,
                "partial_correlation_key": partial_label.correlation_key
                if partial_label is not None
                else "",
                "final_correlation_key": final_label.correlation_key
                if final_label is not None
                else "",
                "close_correlation_resolution": final_resolution,
            }
            if attach_result.upgraded_correlation_keys:
                attach_status = "final_upgraded" if final_label is not None else "quality_upgraded"
                attach_metadata["upgraded"] = ";".join(attach_result.upgraded_correlation_keys)
                if final_tracker is not None:
                    self._discard_open_outcome_tracker(final_tracker.correlation_key)
            elif attach_result.attached_correlation_keys:
                if final_label is not None:
                    attach_status = "final_attached"
                elif partial_label is not None:
                    attach_status = "partial_attached"
                elif proxy_label is not None:
                    attach_status = "proxy_attached"
                else:
                    attach_status = "attached"
                if final_tracker is not None and final_label is not None:
                    self._discard_open_outcome_tracker(final_label.correlation_key)
            elif attach_result.duplicate_noop_correlation_keys:
                attach_status = "duplicate_noop"
                if final_tracker is not None:
                    self._discard_open_outcome_tracker(final_tracker.correlation_key)
            elif attach_result.conflicting_correlation_keys:
                attach_status = "conflict_rejected"
                attach_metadata["conflicts"] = ";".join(attach_result.conflicting_correlation_keys)
            elif attach_result.missing_correlation_keys:
                attach_status = "missing_shadow_record"
                attach_metadata["missing"] = ";".join(attach_result.missing_correlation_keys)
            else:
                attach_status = "skipped"
            self._record_decision_event(
                "opportunity_outcome_attach",
                signal=signal,
                request=request,
                status=attach_status,
                metadata=attach_metadata,
            )
        except Exception:  # pragma: no cover - diagnostic only, no runtime interruption
            _LOGGER.debug("Nie udało się dopiąć Opportunity outcome label", exc_info=True)
            self._record_decision_event(
                "opportunity_outcome_attach",
                signal=signal,
                request=request,
                status="attach_error",
                metadata={
                    "proxy_correlation_key": correlation_key,
                    "execution_status": normalized_status,
                    "close_correlation_resolution": final_resolution,
                },
            )

    @staticmethod
    def _is_closing_side(open_side: str, current_side: str) -> bool:
        return (open_side in _BUY_SIDES and current_side in _SELL_SIDES) or (
            open_side in _SELL_SIDES and current_side in _BUY_SIDES
        )

    @staticmethod
    def _realized_return_bps(*, entry_side: str, entry_price: float, exit_price: float) -> float:
        if entry_price <= 0:
            return 0.0
        if entry_side in _BUY_SIDES:
            return ((exit_price - entry_price) / entry_price) * 10_000.0
        if entry_side in _SELL_SIDES:
            return ((entry_price - exit_price) / entry_price) * 10_000.0
        return 0.0

    @staticmethod
    def _safe_float(value: object, *, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _maybe_reverse_position(
        self,
        signal: StrategySignal,
        request: OrderRequest,
        metric_labels: Mapping[str, str],
    ) -> bool:
        metadata = dict(request.metadata or {})
        reverse_flag = metadata.pop("reverse_position", False)
        if isinstance(reverse_flag, str):
            reverse_flag = reverse_flag.strip().lower() in {"true", "1", "yes", "on"}
        elif isinstance(reverse_flag, (int, float)):
            reverse_flag = bool(reverse_flag)
        else:
            reverse_flag = bool(reverse_flag)
        if not reverse_flag:
            self._metric_reversal_skipped_total.inc(labels={**metric_labels, "reason": "disabled"})
            return True

        qty_raw = metadata.pop("current_position_qty", None)
        side_raw = metadata.pop("current_position_side", None)
        untrusted_reason = ""
        try:
            position_qty = float(qty_raw)
        except (TypeError, ValueError):
            untrusted_reason = "invalid_qty"
            position_qty = 0.0
        if not untrusted_reason and (not math.isfinite(position_qty) or position_qty <= 0):
            untrusted_reason = "non_positive_qty"
        if not untrusted_reason and not side_raw:
            untrusted_reason = "missing_side"

        current_side = str(side_raw).upper() if side_raw else ""
        desired_side = request.side.upper()
        if not untrusted_reason and current_side not in {"LONG", "SHORT"}:
            untrusted_reason = "invalid_side"
        if not untrusted_reason and (
            (current_side == "LONG" and desired_side == "BUY")
            or (current_side == "SHORT" and desired_side == "SELL")
        ):
            self._metric_reversal_skipped_total.inc(
                labels={**metric_labels, "reason": "not_required"}
            )
            return True

        if untrusted_reason:
            self._metric_reversal_skipped_total.inc(
                labels={**metric_labels, "reason": "untrusted_position"}
            )
            self._record_decision_event(
                "reversal_skipped_untrusted_position",
                signal=signal,
                request=request,
                status="skipped",
                metadata={
                    "reason": untrusted_reason,
                    "current_position_qty": str(qty_raw),
                    "current_position_side": str(side_raw),
                },
            )
            _LOGGER.warning(
                "Pomijam reversal dla %s: niewiarygodne źródło pozycji (reason=%s, qty=%r, side=%r)",
                request.symbol,
                untrusted_reason,
                qty_raw,
                side_raw,
            )
            return True

        close_side = "SELL" if current_side == "LONG" else "BUY"
        close_metadata = {
            **metadata,
            "action": "close",
            "reverse_target": desired_side,
            "reducing_only": True,
            "is_reducing": True,
        }
        close_request = OrderRequest(
            symbol=request.symbol,
            side=close_side,
            quantity=position_qty,
            order_type="MARKET",
            price=None,
            time_in_force=None,
            client_order_id=None,
            stop_price=None,
            atr=None,
            metadata=close_metadata,
        )
        close_request = self._ensure_client_order_id(close_request)

        close_account = self.account_snapshot_provider()
        close_risk = self.risk_engine.apply_pre_trade_checks(
            close_request,
            account=close_account,
            profile_name=self.risk_profile,
        )
        self._record_decision_event(
            "reversal_close_risk_check",
            signal=signal,
            request=close_request,
            status="allowed" if close_risk.allowed else "rejected",
            metadata={
                "reason": close_risk.reason or "",
                "available_margin": f"{close_account.available_margin:.8f}",
                "total_equity": f"{close_account.total_equity:.8f}",
                "maintenance_margin": f"{close_account.maintenance_margin:.8f}",
            },
        )
        if not close_risk.allowed:
            self._metric_reversal_denied_by_risk_total.inc(
                labels={**metric_labels, "side": close_side}
            )
            self._record_decision_event(
                "reversal_denied_by_risk",
                signal=signal,
                request=close_request,
                status="rejected",
                metadata={"reason": close_risk.reason or ""},
            )
            return True

        self._metric_orders_total.inc(
            labels={**metric_labels, "result": "submitted", "side": close_side},
        )
        self._record_decision_event(
            "order_close_for_reversal",
            signal=signal,
            request=close_request,
            status="submitted",
            metadata={"position_side": current_side, "position_qty": f"{position_qty:.8f}"},
        )
        try:
            result = self.execution_service.execute(close_request, self._execution_context)
        except Exception as exc:  # noqa: BLE001
            self._metric_orders_total.inc(
                labels={**metric_labels, "result": "failed", "side": close_side},
            )
            self._record_decision_event(
                "order_close_for_reversal",
                signal=signal,
                request=close_request,
                status="failed",
                metadata={"error": str(exc)},
            )
            raise

        normalized_close_status = _normalize_execution_status(result.status)
        close_is_filled = normalized_close_status in _FILLED_EXECUTION_STATUSES
        close_metric_result = "executed" if close_is_filled else "not_filled"
        self._metric_orders_total.inc(
            labels={**metric_labels, "result": close_metric_result, "side": close_side},
        )
        self._record_decision_event(
            "order_close_for_reversal",
            signal=signal,
            request=close_request,
            status=normalized_close_status,
            metadata={"close_order_id": result.order_id or ""},
        )
        return close_is_filled

    def _maybe_adjust_request(
        self,
        signal: StrategySignal,
        request: OrderRequest,
        risk_result: RiskCheckResult,
        account: AccountSnapshot,
    ) -> tuple[OrderRequest | None, RiskCheckResult]:
        quantity = _extract_adjusted_quantity(request.quantity, risk_result.adjustments)
        if quantity is None:
            return None, risk_result

        adjusted_request = OrderRequest(
            symbol=request.symbol,
            side=request.side,
            quantity=quantity,
            order_type=request.order_type,
            price=request.price,
            time_in_force=request.time_in_force,
            client_order_id=request.client_order_id,
            stop_price=request.stop_price,
            atr=request.atr,
            metadata={**dict(request.metadata or {}), "quantity": float(quantity)},
        )
        new_result = self.risk_engine.apply_pre_trade_checks(
            adjusted_request,
            account=account,
            profile_name=self.risk_profile,
        )
        if not new_result.allowed:
            return None, new_result

        _LOGGER.info(
            "Dostosowuję sygnał %s %s: qty %.8f -> %.8f po rekomendacji risk engine.",
            signal.side.upper(),
            signal.symbol,
            request.quantity,
            quantity,
        )
        return adjusted_request, new_result

    def _build_order_request(
        self, signal: StrategySignal, *, extra_metadata: Mapping[str, object] | None = None
    ) -> OrderRequest:
        # Metadane z sygnału + domyślne z kontrolera
        metadata_source = self._clone_metadata(self._order_defaults)
        metadata_source.update(self._clone_metadata(signal.metadata))
        if extra_metadata:
            extra_payload = self._clone_metadata(extra_metadata)
            for protected_key in ("client_order_id", "order_type", "time_in_force", "exchange"):
                if protected_key in metadata_source and protected_key in extra_payload:
                    extra_payload.pop(protected_key, None)
            metadata_source.update(extra_payload)

        self._inject_explainability_metadata(metadata_source)

        # Wymagane parametry
        try:
            quantity = float(metadata_source.get("quantity", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Wielkość zlecenia (quantity) musi być liczbą zmiennoprzecinkową"
            ) from exc
        if quantity <= 0:
            raise ValueError("Wielkość zlecenia musi być dodatnia")

        price = _normalize_optional_request_float(metadata_source.get("price"), field_name="price")
        if price is None:
            metadata_source.pop("price", None)
        else:
            metadata_source["price"] = price

        order_type = str(metadata_source.get("order_type") or "market").upper()
        time_in_force_raw = metadata_source.get("time_in_force")
        client_order_id_raw = metadata_source.get("client_order_id")
        time_in_force = _normalize_optional_request_string(time_in_force_raw)
        client_order_id = _normalize_optional_request_string(client_order_id_raw)
        if time_in_force is None:
            metadata_source.pop("time_in_force", None)
        else:
            metadata_source["time_in_force"] = time_in_force
        if client_order_id is None:
            metadata_source.pop("client_order_id", None)
        else:
            metadata_source["client_order_id"] = client_order_id

        # Opcjonalne rozszerzenia
        stop_price = _normalize_optional_request_float(
            metadata_source.get("stop_price"),
            field_name="stop_price",
        )
        atr = _normalize_optional_request_float(
            metadata_source.get("atr"),
            field_name="atr",
        )
        if stop_price is None:
            metadata_source.pop("stop_price", None)
        else:
            metadata_source["stop_price"] = stop_price
        if atr is None:
            metadata_source.pop("atr", None)
        else:
            metadata_source["atr"] = atr

        return OrderRequest(
            symbol=signal.symbol,
            side=signal.side.upper(),
            quantity=quantity,
            order_type=order_type,
            price=price,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
            stop_price=stop_price,
            atr=atr,
            metadata=metadata_source,
        )

    def _ensure_client_order_id(self, request: OrderRequest) -> OrderRequest:
        client_order_id = request.client_order_id
        if isinstance(client_order_id, str) and client_order_id.strip():
            return request

        generated_id = f"tc-{uuid.uuid4().hex}"
        metadata = self._clone_metadata(request.metadata)
        metadata["client_order_id"] = generated_id
        metadata["generated_client_order_id"] = True
        return OrderRequest(
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            order_type=request.order_type,
            price=request.price,
            time_in_force=request.time_in_force,
            client_order_id=generated_id,
            stop_price=request.stop_price,
            atr=request.atr,
            metadata=metadata,
        )

    def _decision_engine_enabled(self) -> bool:
        return DecisionCandidate is not None and self._decision_orchestrator is not None

    def _clone_metadata(self, metadata: Mapping[str, object] | None) -> dict[str, object]:
        def _clone_value(value: object) -> object:
            if isinstance(value, Mapping):
                return {str(key): _clone_value(item) for key, item in value.items()}
            if isinstance(value, list):
                return [_clone_value(item) for item in value]
            if isinstance(value, tuple):
                return tuple(_clone_value(item) for item in value)
            return value

        if isinstance(metadata, Mapping):
            return {str(k): _clone_value(v) for k, v in metadata.items()}
        if metadata is None:
            return {}
        try:
            candidate = dict(metadata)  # type: ignore[arg-type]
        except Exception:
            return {}
        return {str(k): _clone_value(v) for k, v in candidate.items()}

    def _collect_explainability_payloads(self, metadata: Mapping[str, object]) -> list[object]:
        candidates: list[object] = []
        seen_strings: set[str] = set()
        for key in ("explainability", "explainability_json"):
            payload = metadata.get(key)
            if payload is not None:
                if isinstance(payload, str):
                    if payload in seen_strings:
                        continue
                    seen_strings.add(payload)
                candidates.append(payload)
        for section in ("decision_engine", "ai_manager", "ai"):
            section_payload = metadata.get(section)
            if not isinstance(section_payload, Mapping):
                continue
            for key in ("explainability", "explainability_json"):
                payload = section_payload.get(key)
                if payload is not None:
                    if isinstance(payload, str):
                        if payload in seen_strings:
                            continue
                        seen_strings.add(payload)
                    candidates.append(payload)
        return candidates

    def _inject_explainability_metadata(self, metadata: MutableMapping[str, object] | None) -> None:
        if not metadata:
            return
        for report in self._collect_explainability_payloads(metadata):
            try:
                flattened = flatten_explainability(report)
            except Exception:  # pragma: no cover - zależy od struktury raportu
                continue
            for key, value in flattened.items():
                metadata.setdefault(str(key), value)
            if "ai_explainability_json" in metadata:
                return
            try:
                payload_json = None
                if isinstance(report, str):
                    payload_json = report
                elif isinstance(report, Mapping):
                    payload_json = json.dumps(
                        dict(report),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                if payload_json is not None:
                    metadata.setdefault("ai_explainability_json", payload_json)
                    return
            except Exception:  # pragma: no cover - defensywnie
                continue

    def _normalize_signal_metadata(self, signal: StrategySignal) -> Mapping[str, object]:
        raw_metadata = getattr(signal, "metadata", None)
        if isinstance(raw_metadata, Mapping):
            metadata = dict(raw_metadata)
        elif raw_metadata is None:
            metadata = {}
        else:
            try:
                metadata = dict(raw_metadata)  # type: ignore[arg-type]
            except Exception:  # pragma: no cover - defensywne
                metadata = {}
        metadata.setdefault("environment", self.environment)
        metadata.setdefault("portfolio", self.portfolio_id)
        metadata.setdefault("controller", self._strategy_name or self.__class__.__name__)
        self._inject_explainability_metadata(metadata)
        return metadata

    def _build_decision_candidate(
        self, signal: StrategySignal
    ) -> tuple[DecisionCandidate | None, Mapping[str, object] | None]:
        if not self._decision_engine_enabled():
            return None, None
        metadata = self._normalize_signal_metadata(signal)
        probability = self._decision_extract_probability(signal, metadata)
        if probability < self._decision_min_probability:
            return None, {
                "reason": "probability_below_threshold",
                "probability": probability,
                "min_probability": self._decision_min_probability,
            }
        expected_return = self._decision_extract_expected_return(signal, metadata)
        cost_override = self._decision_extract_cost_override(metadata)
        latency_ms = self._decision_extract_latency(metadata)
        notional = self._decision_extract_notional(metadata)
        action = "exit" if str(signal.side).upper() in {"SELL", "EXIT", "FLAT"} else "enter"
        candidate = DecisionCandidate(
            strategy=self._strategy_name or metadata.get("strategy", ""),
            action=action,
            risk_profile=self.risk_profile,
            symbol=signal.symbol,
            notional=notional,
            expected_return_bps=expected_return,
            expected_probability=probability,
            cost_bps_override=cost_override,
            latency_ms=latency_ms,
            metadata=metadata,
        )
        return candidate, None

    def _decision_extract_probability(
        self, signal: StrategySignal, metadata: Mapping[str, object]
    ) -> float:
        candidate = metadata.get("expected_probability") or metadata.get("probability")
        if candidate is None and isinstance(metadata.get("ai_manager"), Mapping):
            candidate = metadata["ai_manager"].get("success_probability")
        probability: float | None = None
        if candidate is not None:
            try:
                probability = float(candidate)
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

    def _decision_extract_expected_return(
        self, signal: StrategySignal, metadata: Mapping[str, object]
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

    def _decision_extract_cost_override(self, metadata: Mapping[str, object]) -> float | None:
        candidate = metadata.get("cost_bps") or metadata.get("slippage_bps")
        if candidate is None:
            return None
        try:
            return float(candidate)
        except (TypeError, ValueError):
            return None

    def _decision_extract_latency(self, metadata: Mapping[str, object]) -> float | None:
        latency = metadata.get("latency_ms") or metadata.get("latency")
        if latency is None and isinstance(metadata.get("decision_engine"), Mapping):
            latency = metadata["decision_engine"].get("latency_ms")
        if latency is None:
            return None
        try:
            return float(latency)
        except (TypeError, ValueError):
            return None

    def _decision_extract_notional(self, metadata: Mapping[str, object]) -> float:
        candidate = metadata.get("notional") or metadata.get("target_notional")
        if candidate is None:
            return self._decision_default_notional
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            return self._decision_default_notional
        if value <= 0:
            return self._decision_default_notional
        return value

    def _decision_risk_snapshot(self, profile: str) -> Mapping[str, object]:
        loader = getattr(self.risk_engine, "snapshot_state", None)
        if callable(loader):
            try:
                snapshot = loader(profile)
                if snapshot is not None:
                    return snapshot
            except Exception:  # pragma: no cover - diagnostyka risk engine
                _LOGGER.debug("DecisionOrchestrator: snapshot_state failed", exc_info=True)
        return {}

    def _evaluate_decision_candidate(
        self, candidate: DecisionCandidate
    ) -> DecisionEvaluation | None:
        orchestrator = self._decision_orchestrator
        if orchestrator is None:
            return None
        snapshot = self._decision_risk_snapshot(candidate.risk_profile)
        try:
            evaluation = orchestrator.evaluate_candidate(
                candidate,
                DecisionContext(
                    risk_snapshot=snapshot,
                    runtime={
                        "portfolio": self.portfolio_id,
                        "environment": self.environment,
                    },
                ),
            )
        except Exception:  # pragma: no cover - diagnostyka orchestratora
            _LOGGER.exception("DecisionOrchestrator: błąd ewaluacji")
            return None
        if evaluation is None or not hasattr(evaluation, "accepted"):
            _LOGGER.warning(
                "DecisionOrchestrator: niepoprawny wynik ewaluacji (%s)",
                type(evaluation).__name__ if evaluation is not None else "None",
            )
            return None
        return evaluation

    def _serialize_decision_evaluation(
        self, evaluation: DecisionEvaluation
    ) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "decision_engine": {
                "accepted": bool(getattr(evaluation, "accepted", False)),
                "reasons": list(getattr(evaluation, "reasons", ())),
                "model": getattr(evaluation, "model_name", None),
                "net_edge_bps": getattr(evaluation, "net_edge_bps", None),
                "cost_bps": getattr(evaluation, "cost_bps", None),
                "model_expected_return_bps": getattr(evaluation, "model_expected_return_bps", None),
                "model_success_probability": getattr(evaluation, "model_success_probability", None),
            }
        }
        selection = getattr(evaluation, "model_selection", None)
        if selection is not None:
            if hasattr(selection, "to_mapping"):
                try:
                    payload["decision_engine"]["model_selection"] = selection.to_mapping()
                except Exception:  # pragma: no cover - defensywnie
                    payload["decision_engine"]["model_selection"] = {
                        "selected": getattr(selection, "selected", None)
                    }
            elif is_dataclass(selection):
                payload["decision_engine"]["model_selection"] = asdict(selection)
        return payload

    def _decision_threshold_snapshot(self) -> Mapping[str, object]:
        snapshot: dict[str, object] = {
            "min_probability": self._decision_min_probability,
        }
        orchestrator = self._decision_orchestrator
        config = getattr(orchestrator, "_config", None) if orchestrator is not None else None
        thresholds = getattr(config, "orchestrator", None) if config is not None else None
        if thresholds is not None:
            if is_dataclass(thresholds):
                snapshot.update({"orchestrator": asdict(thresholds)})
            else:
                try:
                    snapshot.update({"orchestrator": dict(thresholds)})
                except Exception:  # pragma: no cover - defensywnie
                    snapshot["orchestrator"] = thresholds
        return snapshot

    def _handle_decision_filtered(
        self, signal: StrategySignal, rejection_info: Mapping[str, object] | None
    ) -> None:
        thresholds = self._decision_threshold_snapshot()
        reason = None
        probability = None
        if rejection_info:
            reason = rejection_info.get("reason")
            probability = rejection_info.get("probability")
        _LOGGER.info(
            "DecisionOrchestrator odrzucił sygnał %s %s (pre-check): %s",
            signal.side.upper(),
            signal.symbol,
            reason or "brak szczegółów",
            extra={"thresholds": thresholds, "probability": probability},
        )
        metadata: dict[str, object] = {
            "decision_status": "filtered",
            "decision_reason": reason or "probability_below_threshold",
            "min_probability": thresholds.get("min_probability", 0.0),
        }
        if probability is not None:
            metadata["expected_probability"] = probability
        metadata["thresholds"] = json.dumps(thresholds)
        self._record_decision_event(
            "decision_evaluation",
            signal=signal,
            status="filtered",
            metadata=metadata,
        )

    def _handle_decision_rejection(
        self, signal: StrategySignal, evaluation: DecisionEvaluation
    ) -> None:
        reasons = list(getattr(evaluation, "reasons", ()))
        thresholds = self._decision_threshold_snapshot()
        _LOGGER.info(
            "DecisionOrchestrator odrzucił sygnał %s %s: %s",
            signal.side.upper(),
            signal.symbol,
            ", ".join(reasons) or "brak powodów",
            extra={"thresholds": thresholds},
        )
        metadata: dict[str, object] = {
            "decision_status": "rejected",
            "decision_reason": ", ".join(reasons) or "unknown",
            "net_edge_bps": getattr(evaluation, "net_edge_bps", None) or "",
            "cost_bps": getattr(evaluation, "cost_bps", None) or "",
            "thresholds": json.dumps(thresholds),
        }
        model_name = getattr(evaluation, "model_name", None)
        if model_name:
            metadata["decision_model"] = model_name
        self._record_decision_event(
            "decision_evaluation",
            signal=signal,
            status="rejected",
            metadata=metadata,
        )

    def _record_decision_evaluation_event(
        self, signal: StrategySignal, evaluation: DecisionEvaluation
    ) -> None:
        if not getattr(evaluation, "accepted", False):
            return
        metadata = {
            "decision_status": "accepted",
            "net_edge_bps": getattr(evaluation, "net_edge_bps", None) or "",
            "cost_bps": getattr(evaluation, "cost_bps", None) or "",
            "model_expected_return_bps": getattr(evaluation, "model_expected_return_bps", None)
            or "",
            "model_success_probability": getattr(evaluation, "model_success_probability", None)
            or "",
        }
        model_name = getattr(evaluation, "model_name", None)
        if model_name:
            metadata["decision_model"] = model_name
        self._record_decision_event(
            "decision_evaluation",
            signal=signal,
            status="accepted",
            metadata=metadata,
        )

    def _emit_signal_alert(self, signal: StrategySignal) -> None:
        context: dict[str, str] = {
            "symbol": signal.symbol,
            "side": signal.side.upper(),
            "confidence": f"{signal.confidence:.2f}",
            "environment": self.environment,
            "risk_profile": self.risk_profile,
        }
        for key, value in signal.metadata.items():
            context[f"meta_{key}"] = str(value)

        message = AlertMessage(
            category="strategy",
            title=f"Sygnał {signal.side.upper()} dla {signal.symbol}",
            body="Strategia wygenerowała sygnał wymagający decyzji egzekucyjnej.",
            severity="info",
            context=context,
        )
        _LOGGER.debug("Publikuję alert sygnału %s %s", signal.side.upper(), signal.symbol)
        self.alert_router.dispatch(message)

    def _emit_order_rejected_alert(
        self,
        signal: StrategySignal,
        request: OrderRequest,
        risk_result: RiskCheckResult,
    ) -> None:
        adjustments = risk_result.adjustments or {}
        context: dict[str, str] = {
            "symbol": request.symbol,
            "side": request.side,
            "order_type": request.order_type,
            "quantity": f"{request.quantity:.8f}",
            "environment": self.environment,
            "risk_profile": self.risk_profile,
        }
        context.update({f"adjust_{k}": str(v) for k, v in adjustments.items()})

        message = AlertMessage(
            category="risk",
            title=f"Zlecenie odrzucone przez risk engine ({request.symbol})",
            body=risk_result.reason or "Brak szczegółów",
            severity="warning",
            context=context,
        )
        _LOGGER.warning(
            "Risk engine odrzucił zlecenie %s %s: %s",
            request.side,
            request.symbol,
            risk_result.reason,
        )
        self.alert_router.dispatch(message)

    def _handle_liquidation_state(self, risk_result: RiskCheckResult) -> None:
        in_liquidation = self.risk_engine.should_liquidate(profile_name=self.risk_profile)
        self._metric_liquidation_state.set(
            1.0 if in_liquidation else 0.0, labels=self._metric_labels
        )
        if in_liquidation != self._liquidation_active:
            transition_status = "entered" if in_liquidation else "exited"
            transition_metadata: dict[str, object] = {"in_liquidation": in_liquidation}
            if risk_result.reason:
                transition_metadata["reason"] = risk_result.reason
            self._record_decision_event(
                "liquidation_state_changed",
                status=transition_status,
                metadata=transition_metadata,
            )
            self._liquidation_active = in_liquidation
        if not in_liquidation:
            if self._liquidation_alerted:
                _LOGGER.info("Profil %s wyszedł z trybu awaryjnego", self.risk_profile)
            self._liquidation_alerted = False
            return

        if self._liquidation_alerted:
            return

        reason = (
            risk_result.reason
            or "Profil ryzyka w trybie awaryjnym – przekroczono limit straty lub obsunięcia."
        )
        context = {
            "risk_profile": self.risk_profile,
            "environment": self.environment,
            "portfolio": self.portfolio_id,
        }
        message = AlertMessage(
            category="risk",
            title="Profil w trybie awaryjnym",
            body=reason,
            severity="critical",
            context=context,
        )
        _LOGGER.error("Profil %s przekroczył limity – wysyłam alert krytyczny", self.risk_profile)
        self.alert_router.dispatch(message)
        self._liquidation_alerted = True

    def _emit_order_filled_alert(
        self,
        signal: StrategySignal,
        request: OrderRequest,
        result: OrderResult,
        *,
        partial: bool = False,
    ) -> None:
        order_id = result.order_id or ""
        if partial:
            avg_price = result.avg_price
            filled_qty = result.filled_quantity
        else:
            avg_price = result.avg_price if result.avg_price is not None else (request.price or 0.0)
            filled_qty = (
                result.filled_quantity if result.filled_quantity is not None else request.quantity
            )

        context = {
            "symbol": request.symbol,
            "side": request.side,
            "order_id": order_id,
            "client_order_id": request.client_order_id or "",
            "avg_price": "unknown" if avg_price is None else f"{avg_price:.8f}",
            "filled_quantity": "unknown" if filled_qty is None else f"{filled_qty:.8f}",
            "status": result.status or "unknown",
            "environment": self.environment,
            "risk_profile": self.risk_profile,
        }

        for key, value in signal.metadata.items():
            context.setdefault(f"meta_{key}", str(value))

        message = AlertMessage(
            category="execution",
            title=(
                f"Zlecenie {request.side} {request.symbol} częściowo zrealizowane"
                if partial
                else f"Zlecenie {request.side} {request.symbol} zrealizowane"
            ),
            body=(
                "Zlecenie zostało częściowo wykonane w symulatorze/na giełdzie."
                if partial
                else "Zlecenie zostało wykonane w symulatorze/na giełdzie."
            ),
            severity="info",
            context=context,
        )
        _LOGGER.info(
            "Zlecenie %s %s zostało zrealizowane (order_id=%s, qty=%s, price=%s)",
            request.side,
            request.symbol,
            order_id,
            filled_qty,
            avg_price,
        )
        self.alert_router.dispatch(message)

    def _emit_order_not_filled_alert(
        self,
        signal: StrategySignal,
        request: OrderRequest,
        result: OrderResult,
        *,
        normalized_status: str,
    ) -> None:
        context = {
            "symbol": request.symbol,
            "side": request.side,
            "order_id": result.order_id or "",
            "client_order_id": request.client_order_id or "",
            "status": normalized_status,
            "environment": self.environment,
            "risk_profile": self.risk_profile,
        }
        for key, value in signal.metadata.items():
            context.setdefault(f"meta_{key}", str(value))
        message = AlertMessage(
            category="execution",
            title=f"Zlecenie {request.side} {request.symbol} nie zostało wykonane",
            body=(f"Egzekucja zwróciła wynik bez pełnego wykonania (status={normalized_status})."),
            severity="warning",
            context=context,
        )
        _LOGGER.warning(
            "Zlecenie %s %s zwróciło status bez wykonania: %s",
            request.side,
            request.symbol,
            normalized_status,
        )
        self.alert_router.dispatch(message)

    def _emit_execution_error_alert(
        self,
        signal: StrategySignal,
        request: OrderRequest,
        error: Exception,
    ) -> None:
        context = {
            "symbol": request.symbol,
            "side": request.side,
            "client_order_id": request.client_order_id or "",
            "environment": self.environment,
            "risk_profile": self.risk_profile,
            "error_type": type(error).__name__,
        }
        for key, value in signal.metadata.items():
            context.setdefault(f"meta_{key}", str(value))

        message = AlertMessage(
            category="execution",
            title=f"Błąd egzekucji zlecenia {request.side} {request.symbol}",
            body=str(error) or "Nieznany błąd egzekucji.",
            severity="critical",
            context=context,
        )
        _LOGGER.exception("Błąd egzekucji zlecenia %s %s: %s", request.side, request.symbol, error)
        self.alert_router.dispatch(message)


# =============================================================================
# DailyTrendController – cykl: backfill -> strategia -> ryzyko -> egzekucja
# =============================================================================


@dataclass(slots=True)
class DailyTrendController:
    """Prosty kontroler realizujący cykl: backfill -> strategia -> ryzyko -> egzekucja.

    Ten kontroler jest niezależny od warstwy alertów – nadaje się do wsadowego
    uruchamiania testów walk-forward lub automatycznych backfilli + egzekucji.
    """

    core_config: CoreConfig
    environment_name: str
    controller_name: str
    symbols: Sequence[str]
    backfill_service: OHLCVBackfillService
    data_source: CachedOHLCVSource
    strategy: (
        Any  # StrategyEngine kompatybilny: posiada on_data(snapshot)->Sequence[StrategySignal]
    )
    risk_engine: RiskEngine
    execution_service: ExecutionService
    account_loader: Callable[[], AccountSnapshot]
    execution_context: ExecutionContext
    position_size: float = 1.0
    strategy_name: str | None = None
    exchange_name: str | None = None
    tco_reporter: RuntimeTCOReporter | None = None
    tco_metadata: Mapping[str, object] | None = None

    _environment: Any = field(init=False, repr=False)
    _runtime: ControllerRuntimeConfig = field(init=False, repr=False)
    _risk_profile: str = field(init=False, repr=False)
    _positions: dict[str, float] = field(init=False, repr=False, default_factory=dict)
    _tco_metadata: Mapping[str, object] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("Wymagany jest przynajmniej jeden symbol do obsługi.")
        try:
            self._environment = self.core_config.environments[self.environment_name]
        except KeyError as exc:
            raise KeyError(
                f"Brak konfiguracji środowiska '{self.environment_name}' w CoreConfig"
            ) from exc
        try:
            self._runtime = self.core_config.runtime_controllers[self.controller_name]
        except Exception as exc:
            raise KeyError(
                f"Brak sekcji runtime dla kontrolera '{self.controller_name}' w CoreConfig"
            ) from exc
        self._risk_profile = self._environment.risk_profile
        if not self.exchange_name:
            self.exchange_name = getattr(self._environment, "exchange", None)
        self._tco_metadata = dict(self.tco_metadata or {})

    @property
    def tick_seconds(self) -> float:
        return float(self._runtime.tick_seconds)

    @property
    def interval(self) -> str:
        return str(self._runtime.interval)

    def collect_signals(self, *, start: int, end: int) -> list[ControllerSignal]:
        """Zwraca sygnały strategii wzbogacone o parametry egzekucyjne."""
        if start > end:
            raise ValueError("Parametr start nie może być większy niż end")

        self.backfill_service.synchronize(
            symbols=self.symbols,
            interval=self.interval,
            start=start,
            end=end,
        )

        collected: list[ControllerSignal] = []
        for symbol in self.symbols:
            try:
                response = self.data_source.fetch_ohlcv(  # type: ignore[attr-defined]
                    OHLCVRequest(symbol=symbol, interval=self.interval, start=start, end=end)
                )
                columns: Sequence[str] = response.columns  # type: ignore[attr-defined]
                rows: Sequence[Sequence[float]] = response.rows  # type: ignore[attr-defined]
            except Exception:
                rows = self.data_source.fetch_ohlcv(symbol, self.interval, start=start, end=end)  # type: ignore[misc]
                columns = ("open_time", "open", "high", "low", "close", "volume")

            snapshots = self._to_snapshots(symbol, columns, rows)
            for snapshot in snapshots:
                strategy_signals: Sequence[StrategySignal] = self.strategy.on_data(snapshot)
                for raw_signal in strategy_signals:
                    enriched = self._enrich_signal(raw_signal, snapshot)
                    if enriched is None:
                        continue
                    collected.append(ControllerSignal(snapshot=snapshot, signal=enriched))
        return collected

    def run_cycle(self, *, start: int, end: int) -> list[OrderResult]:
        """Przeprowadza pojedynczy cykl przetwarzania danych i składania zleceń."""
        collected = self.collect_signals(start=start, end=end)
        executed: list[OrderResult] = []
        for controller_signal in collected:
            executed.extend(
                self._handle_signals(controller_signal.snapshot, (controller_signal.signal,))
            )
        return executed

    # ----------------------------------------- helpers -----------------------------------------
    def _handle_signals(
        self,
        snapshot: MarketSnapshot,
        signals: Sequence[StrategySignal],
    ) -> list[OrderResult]:
        results: list[OrderResult] = []
        profile = getattr(self.risk_engine, "get_profile", lambda name: None)(self._risk_profile)
        for signal in signals:
            base_request = self._build_order_request(snapshot, signal)
            account_snapshot = self.account_loader()
            base_request = _clamp_request_quantity(
                base_request,
                account_snapshot,
                profile,
                include_trade_risk=False,
            )
            risk_result = self.risk_engine.apply_pre_trade_checks(
                base_request,
                account=account_snapshot,
                profile_name=self._risk_profile,
            )
            request = base_request
            if not risk_result.allowed:
                adjusted_qty = _extract_adjusted_quantity(
                    base_request.quantity, risk_result.adjustments
                )
                if adjusted_qty is not None:
                    adjusted_metadata = dict(base_request.metadata or {})
                    adjusted_metadata["quantity"] = float(adjusted_qty)
                    adjusted_request = OrderRequest(
                        symbol=base_request.symbol,
                        side=base_request.side,
                        quantity=adjusted_qty,
                        order_type=base_request.order_type,
                        price=base_request.price,
                        time_in_force=base_request.time_in_force,
                        client_order_id=base_request.client_order_id,
                        stop_price=base_request.stop_price,
                        atr=base_request.atr,
                        metadata=adjusted_metadata,
                    )
                    adjusted_request = _clamp_request_quantity(
                        adjusted_request,
                        account_snapshot,
                        profile,
                        include_trade_risk=True,
                    )
                    second_result = self.risk_engine.apply_pre_trade_checks(
                        adjusted_request,
                        account=account_snapshot,
                        profile_name=self._risk_profile,
                    )
                    if second_result.allowed:
                        _LOGGER.info(
                            "Kontroler %s: dostosowuję qty %s z %.8f do %.8f po rekomendacji ryzyka.",
                            self.controller_name,
                            snapshot.symbol,
                            base_request.quantity,
                            adjusted_qty,
                        )
                        request = adjusted_request
                        risk_result = second_result
                    else:
                        _LOGGER.info(
                            "Kontroler %s: silnik ryzyka nadal blokuje zlecenie %s mimo korekty (powód: %s)",
                            self.controller_name,
                            snapshot.symbol,
                            second_result.reason,
                        )
                        continue
                else:
                    _LOGGER.info(
                        "Kontroler %s: sygnał %s dla %s odrzucony przez silnik ryzyka (%s)",
                        self.controller_name,
                        signal.side,
                        snapshot.symbol,
                        risk_result.reason,
                    )
                    continue
            result = self.execution_service.execute(request, self.execution_context)
            self._post_fill(signal.side, snapshot.symbol, request, result)
            results.append(result)
        return results

    def _build_order_request(
        self, snapshot: MarketSnapshot, signal: StrategySignal
    ) -> OrderRequest:
        side = signal.side.lower()
        metadata = dict(signal.metadata)
        quantity = float(metadata.get("quantity", self.position_size))
        price = float(metadata.get("price", snapshot.close))
        order_type = str(metadata.get("order_type", "market"))
        time_in_force = metadata.get("time_in_force")
        client_order_id = metadata.get("client_order_id")
        stop_price_raw = metadata.get("stop_price")
        atr_raw = metadata.get("atr")

        tif_str = str(time_in_force) if time_in_force is not None else None
        client_id_str = str(client_order_id) if client_order_id is not None else None
        stop_price = float(stop_price_raw) if stop_price_raw is not None else None
        atr = float(atr_raw) if atr_raw is not None else None

        # Zapewnij spójność metadanych (analityka/telemetria)
        metadata["quantity"] = quantity
        metadata["price"] = price
        if stop_price is not None:
            metadata["stop_price"] = stop_price
        if atr is not None:
            metadata["atr"] = atr

        return OrderRequest(
            symbol=snapshot.symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            time_in_force=tif_str,
            client_order_id=client_id_str,
            stop_price=stop_price,
            atr=atr,
            metadata=metadata,
        )

    def _to_snapshots(
        self,
        symbol: str,
        columns: Sequence[str],
        rows: Sequence[Sequence[float]],
    ) -> list[MarketSnapshot]:
        if not rows:
            return []

        index: TypingMapping[str, int] = {column.lower(): idx for idx, column in enumerate(columns)}
        # open_time / timestamp
        try:
            open_time_idx = index["open_time"]
        except KeyError as exc:
            if "timestamp" in index:
                open_time_idx = index["timestamp"]
            else:
                raise ValueError("Brak kolumny open_time w danych OHLCV") from exc
        for key in ("open", "high", "low", "close"):
            if key not in index:
                raise ValueError(f"Brak kolumny '{key}' w danych OHLCV")
        open_idx = index["open"]
        high_idx = index["high"]
        low_idx = index["low"]
        close_idx = index["close"]
        volume_idx = index.get("volume")

        snapshots = [
            MarketSnapshot(
                symbol=symbol,
                timestamp=int(float(row[open_time_idx])),
                open=float(row[open_idx]),
                high=float(row[high_idx]),
                low=float(row[low_idx]),
                close=float(row[close_idx]),
                volume=float(row[volume_idx]) if volume_idx is not None else 0.0,
            )
            for row in rows
            if len(row) > close_idx
        ]
        snapshots.sort(key=lambda item: item.timestamp)
        return snapshots

    def _enrich_signal(
        self, signal: StrategySignal, snapshot: MarketSnapshot
    ) -> StrategySignal | None:
        side = signal.side.upper()
        if side not in {"BUY", "SELL"}:
            return None

        metadata: dict[str, object] = dict(signal.metadata)
        metadata.setdefault("quantity", float(self.position_size))
        metadata.setdefault("price", float(snapshot.close))
        metadata.setdefault("order_type", "market")

        return StrategySignal(
            symbol=snapshot.symbol,
            side=side,
            confidence=signal.confidence,
            metadata=metadata,
        )

    def _post_fill(
        self,
        side: str,
        symbol: str,
        request: OrderRequest,
        result: OrderResult,
    ) -> None:
        avg_price = result.avg_price or request.price or 0.0
        notional = avg_price * request.quantity
        side_lower = side.lower()
        pnl = 0.0
        if side_lower == "buy":
            self._positions[symbol] = avg_price
            position_value = notional
        else:
            entry_price = self._positions.pop(symbol, avg_price)
            pnl = (avg_price - entry_price) * request.quantity
            position_value = 0.0
        self.risk_engine.on_fill(
            profile_name=self._risk_profile,
            symbol=symbol,
            side=side_lower,
            position_value=position_value,
            pnl=pnl,
        )
        _LOGGER.info(
            "Kontroler %s: wykonano %s %s qty=%s avg_price=%s pnl=%s",
            self.controller_name,
            side_lower,
            symbol,
            request.quantity,
            avg_price,
            pnl,
        )

        reporter = self.tco_reporter
        if reporter is not None:
            raw_response = result.raw_response if isinstance(result.raw_response, Mapping) else {}
            commission = 0.0
            fee_asset = None
            if isinstance(raw_response, Mapping):
                try:
                    commission = float(raw_response.get("fee", 0.0))
                except (TypeError, ValueError):
                    commission = 0.0
                fee_asset = raw_response.get("fee_asset")
            metadata = dict(self._tco_metadata)
            metadata.setdefault("controller", self.controller_name)
            metadata.setdefault("environment", self.environment_name)
            if fee_asset:
                metadata.setdefault("fee_asset", fee_asset)
            if request.metadata:
                for key, value in request.metadata.items():
                    metadata.setdefault(f"order_{key}", value)
            if isinstance(raw_response, Mapping):
                for key, value in raw_response.items():
                    if key in {"fee", "fee_asset"}:
                        continue
                    metadata.setdefault(f"fill_{key}", value)
            filled_qty = result.filled_quantity or request.quantity
            reporter.record_execution(
                strategy=self.strategy_name or self.controller_name,
                risk_profile=self._risk_profile,
                instrument=symbol,
                exchange=self.exchange_name
                or getattr(self._environment, "exchange", self.environment_name),
                side=side_lower,
                quantity=filled_qty,
                executed_price=avg_price,
                reference_price=request.price,
                commission=commission,
                metadata=metadata,
            )


@dataclass(slots=True)
class AIDecisionLoop:
    """Prosta pętla on-line generująca kandydatów i oceniająca je w DecisionOrchestratorze."""

    feature_engineer: FeatureEngineer
    orchestrator: Any
    inference: DecisionModelInference
    risk_snapshot_provider: Callable[[str], Mapping[str, object] | Any]
    symbols: Sequence[str]
    interval: str
    strategy: str
    risk_profile: str
    default_notional: float = 1_000.0
    action: str = "enter"

    def run_online_loop(self, *, start: int, end: int) -> Sequence[DecisionEvaluation]:
        if DecisionCandidate is None:
            raise RuntimeError("Pakiet decision nie jest dostępny w tej gałęzi")
        if DecisionOrchestrator is not None and not isinstance(
            self.orchestrator, DecisionOrchestrator
        ):
            _LOGGER.debug(
                "AIDecisionLoop: używam niestandardowej implementacji orchestratora %s",
                type(self.orchestrator),
            )
        if not getattr(self.inference, "is_ready", False):
            raise RuntimeError("Model inference nie został przygotowany (brak wag)")

        dataset: FeatureDataset = self.feature_engineer.build_dataset(
            symbols=self.symbols,
            interval=self.interval,
            start=start,
            end=end,
        )
        evaluations: list[DecisionEvaluation] = []
        snapshots_cache: dict[str, Mapping[str, object] | Any] = {}
        for vector in dataset.vectors:
            metadata = {
                "model_features": dict(vector.features),
                "generated_at": vector.timestamp,
            }
            candidate = DecisionCandidate(
                strategy=self.strategy,
                action=self.action,
                risk_profile=self.risk_profile,
                symbol=vector.symbol,
                notional=self.default_notional,
                expected_return_bps=vector.target_bps,
                expected_probability=0.5,
                metadata=metadata,
            )
            snapshot = snapshots_cache.get(self.risk_profile)
            if snapshot is None:
                snapshot = self.risk_snapshot_provider(self.risk_profile)
                snapshots_cache[self.risk_profile] = snapshot
            evaluation = self.orchestrator.evaluate_candidate(
                candidate,
                DecisionContext(
                    risk_snapshot=snapshot, runtime={"vector_timestamp": vector.timestamp}
                ),
            )
            evaluations.append(evaluation)
        return evaluations


@dataclass(slots=True)
class RuntimeInProcessStub:
    """Prosty stub danych runtime wykorzystywany w testach UI.

    Umożliwia generowanie syntetycznych świec OHLCV oraz migawki ryzyka
    odpowiadającej temu, co wbudowany transport in-process udostępnia
    aplikacji desktopowej. Dzięki temu testy mogą deterministycznie
    zasilać warstwę prezentacji danymi bez uruchamiania pełnego
    backendu."""

    base_price: float = 25_000.0
    volatility_bps: float = 35.0
    candle_count: int = 120
    interval_seconds: int = 60

    def build_candles(self) -> list[dict[str, float | int]]:
        """Zwraca listę słowników reprezentujących świece OHLCV."""

        candles: list[dict[str, float | int]] = []
        timestamp = datetime.now(timezone.utc) - timedelta(
            seconds=self.interval_seconds * self.candle_count
        )
        price = float(self.base_price)
        for sequence in range(self.candle_count):
            timestamp += timedelta(seconds=self.interval_seconds)
            drift = math.sin(sequence / 8.0) * (self.volatility_bps / 10_000.0) * price
            open_price = price
            close_price = max(1.0, price + drift)
            high = max(open_price, close_price) * 1.002
            low = min(open_price, close_price) * 0.998
            volume = 1.0 + (sequence % 5) * 0.25
            candles.append(
                {
                    "timestamp": timestamp,
                    "open": float(open_price),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close_price),
                    "volume": float(volume),
                    "sequence": sequence,
                }
            )
            price = close_price
        return candles

    def build_risk_snapshot(self) -> Mapping[str, object]:
        """Zwraca słownik z danymi ryzyka kompatybilny z RiskSnapshotData."""

        generated_at = datetime.now(timezone.utc)
        return {
            "profileLabel": "BALANCED",
            "profileEnum": 1,
            "portfolioValue": self.base_price * 1.0,
            "currentDrawdown": 0.015,
            "maxDailyLoss": 0.05,
            "usedLeverage": 1.5,
            "generatedAt": generated_at,
            "exposures": [
                {
                    "code": "max_notional",
                    "maxValue": 5_000.0,
                    "currentValue": 1_200.0,
                    "thresholdValue": 4_500.0,
                },
                {
                    "code": "max_positions",
                    "maxValue": 10.0,
                    "currentValue": 3.0,
                    "thresholdValue": 8.0,
                },
            ],
            "hasData": True,
        }

    def build_health_payload(self) -> Mapping[str, object]:
        """Zwraca dane health-checka spójne z klientem in-process."""

        started_at = datetime.now(timezone.utc) - timedelta(hours=1)
        return {
            "version": "in-process",
            "gitCommit": "local",
            "startedAt": started_at,
        }


__all__ = [
    "TradingController",
    "DailyTrendController",
    "ControllerSignal",
    "AIDecisionLoop",
    "RuntimeInProcessStub",
]
