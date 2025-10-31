"""Kontrolery spinające warstwy: dane/strategia/ryzyko/egzekucja oraz alerty."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence as SequenceABC
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Mapping,
    MutableMapping,
    Sequence,
    Mapping as TypingMapping,
)

try:  # pragma: no cover - w niektórych gałęziach warstwa AI może nie być zainstalowana
    from bot_core.ai import DecisionModelInference, FeatureDataset, FeatureEngineer
except Exception:  # pragma: no cover
    DecisionModelInference = Any  # type: ignore
    FeatureDataset = Any  # type: ignore
    FeatureEngineer = Any  # type: ignore

# --- elastyczne importy (różne gałęzie mogą mieć różne ścieżki modułów) -----

# Alerts
from bot_core.alerts import AlertMessage, DefaultAlertRouter  # dostępne w obu gałęziach

# Execution
try:
    from bot_core.execution import ExecutionContext, ExecutionService  # re-eksport
except Exception:  # pragma: no cover
    from bot_core.execution.base import ExecutionContext, ExecutionService  # fallback

# Exchanges commons
from bot_core.exchanges.base import AccountSnapshot, OrderRequest, OrderResult
from bot_core.runtime.journal import TradingDecisionEvent, TradingDecisionJournal
from bot_core.runtime.tco_reporting import RuntimeTCOReporter

# Risk
try:
    from bot_core.risk import RiskEngine, RiskCheckResult  # re-eksport
except Exception:  # pragma: no cover
    from bot_core.risk.base import RiskEngine, RiskCheckResult  # fallback

# Strategy types (interfejsy)
try:
    from bot_core.strategies import StrategySignal, MarketSnapshot  # gałąź z interfejsami w pakiecie
except Exception:  # pragma: no cover
    from bot_core.strategies.base import StrategySignal, MarketSnapshot  # alternatywna ścieżka

# Decision Engine (opcjonalnie)
try:  # pragma: no cover
    from bot_core.decision import DecisionCandidate, DecisionEvaluation, DecisionOrchestrator  # type: ignore
except Exception:  # pragma: no cover
    DecisionCandidate = None  # type: ignore
    DecisionEvaluation = Any  # type: ignore
    DecisionOrchestrator = None  # type: ignore

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
    alert_router: DefaultAlertRouter
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
    decision_orchestrator: Any | None = None
    decision_min_probability: float | None = None
    decision_default_notional: float = 1_000.0

    _clock: Callable[[], datetime] = field(init=False, repr=False)
    _health_interval: timedelta = field(init=False, repr=False)
    _execution_context: ExecutionContext = field(init=False, repr=False)
    _order_defaults: dict[str, str] = field(init=False, repr=False)
    _last_health_report: datetime = field(init=False, repr=False)
    _liquidation_alerted: bool = field(init=False, repr=False)
    _metrics: MetricsRegistry = field(init=False, repr=False)
    _metric_labels: Mapping[str, str] = field(init=False, repr=False)
    _metric_signals_total: Any = field(init=False, repr=False)
    _metric_orders_total: Any = field(init=False, repr=False)
    _metric_health_reports: Any = field(init=False, repr=False)
    _metric_liquidation_state: Any = field(init=False, repr=False)
    _decision_journal: TradingDecisionJournal | None = field(init=False, repr=False)
    _strategy_name: str | None = field(init=False, repr=False, default=None)
    _exchange_name: str | None = field(init=False, repr=False, default=None)
    _tco_reporter: RuntimeTCOReporter | None = field(init=False, repr=False, default=None)
    _tco_metadata: Mapping[str, object] = field(init=False, repr=False, default_factory=dict)
    _decision_orchestrator: Any | None = field(init=False, repr=False, default=None)
    _decision_min_probability: float = field(init=False, repr=False, default=0.0)
    _decision_default_notional: float = field(init=False, repr=False, default=1_000.0)

    def __post_init__(self) -> None:
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
            "Liczba sygnałów przetworzonych w TradingController (status=received/accepted/rejected).",
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
        self._metric_liquidation_state.set(0.0, labels=self._metric_labels)
        self._decision_orchestrator = self.decision_orchestrator
        default_notional = max(100.0, float(self.decision_default_notional or 0.0))
        self._decision_default_notional = default_notional
        configured_min_probability: float | None = None
        orchestrator_config = None
        if self._decision_orchestrator is not None:
            orchestrator_config = getattr(self._decision_orchestrator, "_config", None)
        if orchestrator_config is not None:
            configured_min_probability = float(
                getattr(orchestrator_config, "min_probability", 0.0)
            )
        if self.decision_min_probability is not None:
            candidate = float(self.decision_min_probability)
        elif configured_min_probability is not None:
            candidate = configured_min_probability
        else:
            candidate = 0.0
        self._decision_min_probability = max(0.0, min(0.995, candidate))

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

        meta: dict[str, str] = {}
        if metadata:
            meta.update({str(k): str(v) for k, v in metadata.items()})
        if signal is not None:
            meta.setdefault("signal_confidence", f"{signal.confidence:.6f}")
            for key, value in signal.metadata.items():
                meta.setdefault(f"signal_{key}", str(value))
        if request is not None:
            meta.setdefault("order_type", request.order_type)
            if request.time_in_force:
                meta.setdefault("time_in_force", request.time_in_force)
            if request.client_order_id:
                meta.setdefault("client_order_id", request.client_order_id)

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
            signal.metadata.get("strategy", request.metadata.get("strategy") if request.metadata else request.symbol)
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
        for signal in signals:
            metric_labels = dict(self._metric_labels)
            metric_labels["symbol"] = signal.symbol
            self._metric_signals_total.inc(labels={**metric_labels, "status": "received"})
            self._record_decision_event(
                "signal_received",
                signal=signal,
                status="received",
                metadata={
                    "intent": str(getattr(signal, "intent", "single")),
                    "leg_count": str(len(getattr(signal, "legs", ()) or ())),
                },
            )
            expanded_signals = self._expand_signal(signal)
            if not expanded_signals:
                continue
            for expanded_signal in expanded_signals:
                per_leg_labels = dict(metric_labels)
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
                    self._metric_orders_total.inc(
                        labels={
                            **per_leg_labels,
                            "result": "executed",
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
        self.alert_router.dispatch(message)
        self._last_health_report = now
        self._metric_health_reports.inc(labels=self._metric_labels)

    def _expand_signal(self, signal: StrategySignal) -> Sequence[StrategySignal]:
        intent_raw = getattr(signal, "intent", "") or ""
        intent = str(intent_raw).strip().lower()
        legs_attr = getattr(signal, "legs", ()) or ()
        legs: Sequence[object] = (
            tuple(legs_attr) if isinstance(legs_attr, SequenceABC) else tuple()
        )
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
                combined_metadata.setdefault("signal_intent", intent or "multi_leg")
                combined_metadata.setdefault("leg_index", index)
                combined_metadata.setdefault("leg_count", leg_count)
                if leg_quantity is not None:
                    combined_metadata.setdefault("quantity", leg_quantity)
                exchange = getattr(leg, "exchange", None)
                if exchange:
                    combined_metadata.setdefault("exchange", exchange)
                leg_confidence = getattr(leg, "confidence", None)
                expanded.append(
                    StrategySignal(
                        symbol=getattr(leg, "symbol", "") or signal.symbol,
                        side=normalized_side,
                        confidence=leg_confidence if leg_confidence is not None else signal.confidence,
                        quantity=leg_quantity if leg_quantity is not None else parent_quantity,
                        intent="single",
                        metadata=combined_metadata,
                    )
                )
            if not expanded:
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
            if evaluation is not None:
                decision_metadata = self._serialize_decision_evaluation(evaluation)
                self._record_decision_evaluation_event(signal, evaluation)
                if not getattr(evaluation, "accepted", False):
                    self._handle_decision_rejection(signal, evaluation)
                    self._metric_signals_total.inc(
                        labels={**metric_labels, "status": "rejected"}
                    )
                    return None

        request = self._build_order_request(signal, extra_metadata=decision_metadata)
        account = self.account_snapshot_provider()
        risk_result = self.risk_engine.apply_pre_trade_checks(
            request,
            account=account,
            profile_name=self.risk_profile,
        )

        adjusted_request = request
        rejection_reason = risk_result.reason
        if not risk_result.allowed:
            adjusted = self._maybe_adjust_request(signal, request, risk_result, account)
            if adjusted is None:
                self._emit_order_rejected_alert(signal, request, risk_result)
                self._handle_liquidation_state(risk_result)
                self._metric_signals_total.inc(labels={**metric_labels, "status": "rejected"})
                adjustments = risk_result.adjustments or {}
                metadata = {
                    "reason": rejection_reason or "",
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
            adjusted_request, new_result = adjusted
            self._record_decision_event(
                "risk_adjusted",
                signal=signal,
                request=adjusted_request,
                status="adjusted",
                metadata={
                    "original_quantity": f"{request.quantity:.8f}",
                    "adjusted_quantity": f"{adjusted_request.quantity:.8f}",
                    "reason": rejection_reason or "",
                },
            )
            risk_result = new_result
            self._metric_signals_total.inc(labels={**metric_labels, "status": "adjusted"})

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
            self._maybe_reverse_position(signal, adjusted_request, metric_labels)
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

        self._emit_order_filled_alert(signal, adjusted_request, result)
        order_id = result.order_id or ""
        avg_price = result.avg_price or adjusted_request.price or 0.0
        filled_qty = result.filled_quantity or adjusted_request.quantity
        metadata: dict[str, object] = {
            "order_id": order_id,
            "filled_quantity": f"{filled_qty:.8f}",
            "avg_price": f"{avg_price:.8f}",
            "status": result.status or "filled",
        }
        if isinstance(result.raw_response, TypingMapping):
            fee = result.raw_response.get("fee")
            fee_asset = result.raw_response.get("fee_asset")
            if fee is not None:
                metadata["fee"] = fee
            if fee_asset:
                metadata["fee_asset"] = fee_asset
        self._record_decision_event(
            "order_executed",
            signal=signal,
            request=adjusted_request,
            status=result.status or "filled",
            metadata=metadata,
        )
        self._record_tco_execution(
            signal=signal,
            request=adjusted_request,
            result=result,
            order_id=order_id,
            avg_price=avg_price,
            filled_qty=filled_qty,
        )
        self._handle_liquidation_state(risk_result)
        return result

    def _maybe_reverse_position(
        self,
        signal: StrategySignal,
        request: OrderRequest,
        metric_labels: Mapping[str, str],
    ) -> None:
        metadata = dict(request.metadata or {})
        reverse_flag = metadata.pop("reverse_position", True)
        if isinstance(reverse_flag, str):
            reverse_flag = reverse_flag.strip().lower() not in {"false", "0", "no"}
        if not reverse_flag:
            return

        qty_raw = metadata.pop("current_position_qty", None)
        side_raw = metadata.pop("current_position_side", None)
        try:
            position_qty = float(qty_raw)
        except (TypeError, ValueError):
            return
        if position_qty <= 0:
            return
        if not side_raw:
            return
        current_side = str(side_raw).upper()
        desired_side = request.side.upper()
        if current_side not in {"LONG", "SHORT"}:
            return
        if (current_side == "LONG" and desired_side == "BUY") or (
            current_side == "SHORT" and desired_side == "SELL"
        ):
            return

        close_side = "SELL" if current_side == "LONG" else "BUY"
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
            metadata={**metadata, "action": "close", "reverse_target": desired_side},
        )

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

        self._metric_orders_total.inc(
            labels={**metric_labels, "result": "executed", "side": close_side},
        )
        self._record_decision_event(
            "order_close_for_reversal",
            signal=signal,
            request=close_request,
            status=result.status or "filled",
            metadata={"close_order_id": result.order_id or ""},
        )

    def _maybe_adjust_request(
        self,
        signal: StrategySignal,
        request: OrderRequest,
        risk_result: RiskCheckResult,
        account: AccountSnapshot,
    ) -> tuple[OrderRequest, RiskCheckResult] | None:
        quantity = _extract_adjusted_quantity(request.quantity, risk_result.adjustments)
        if quantity is None:
            return None

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
            metadata=request.metadata,
        )
        new_result = self.risk_engine.apply_pre_trade_checks(
            adjusted_request,
            account=account,
            profile_name=self.risk_profile,
        )
        if not new_result.allowed:
            return None

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
        metadata_source: dict[str, object] = dict(self._order_defaults)
        for k, v in signal.metadata.items():
            metadata_source[str(k)] = v
        if extra_metadata:
            for k, v in extra_metadata.items():
                metadata_source[str(k)] = v

        # Wymagane parametry
        try:
            quantity = float(metadata_source.get("quantity", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("Wielkość zlecenia (quantity) musi być liczbą zmiennoprzecinkową") from exc
        if quantity <= 0:
            raise ValueError("Wielkość zlecenia musi być dodatnia")

        price_value = metadata_source.get("price")
        price = float(price_value) if price_value is not None else None

        order_type = str(metadata_source.get("order_type") or "market").upper()
        time_in_force_raw = metadata_source.get("time_in_force")
        client_order_id_raw = metadata_source.get("client_order_id")

        # Opcjonalne rozszerzenia
        stop_price_raw = metadata_source.get("stop_price")
        atr_raw = metadata_source.get("atr")
        stop_price = None
        atr = None
        if stop_price_raw is not None:
            try:
                stop_price = float(stop_price_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError("stop_price w metadanych musi być liczbą zmiennoprzecinkową") from exc
        if atr_raw is not None:
            try:
                atr = float(atr_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError("atr w metadanych musi być liczbą zmiennoprzecinkową") from exc

        return OrderRequest(
            symbol=signal.symbol,
            side=signal.side.upper(),
            quantity=quantity,
            order_type=order_type,
            price=price,
            time_in_force=str(time_in_force_raw) if time_in_force_raw is not None else None,
            client_order_id=str(client_order_id_raw) if client_order_id_raw is not None else None,
            stop_price=stop_price,
            atr=atr,
            metadata=metadata_source,
        )

    def _decision_engine_enabled(self) -> bool:
        return DecisionCandidate is not None and self._decision_orchestrator is not None

    def _clone_metadata(self, metadata: Mapping[str, object] | None) -> dict[str, object]:
        if isinstance(metadata, Mapping):
            return {str(k): v for k, v in metadata.items()}
        if metadata is None:
            return {}
        try:
            candidate = dict(metadata)  # type: ignore[arg-type]
        except Exception:
            return {}
        return {str(k): v for k, v in candidate.items()}

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
            return orchestrator.evaluate_candidate(candidate, snapshot)
        except Exception:  # pragma: no cover - diagnostyka orchestratora
            _LOGGER.exception("DecisionOrchestrator: błąd ewaluacji")
            return None

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
                "model_expected_return_bps": getattr(
                    evaluation, "model_expected_return_bps", None
                ),
                "model_success_probability": getattr(
                    evaluation, "model_success_probability", None
                ),
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
            "model_expected_return_bps": getattr(
                evaluation, "model_expected_return_bps", None
            )
            or "",
            "model_success_probability": getattr(
                evaluation, "model_success_probability", None
            )
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
        self._metric_liquidation_state.set(1.0 if in_liquidation else 0.0, labels=self._metric_labels)
        if not in_liquidation:
            if self._liquidation_alerted:
                _LOGGER.info("Profil %s wyszedł z trybu awaryjnego", self.risk_profile)
            self._liquidation_alerted = False
            return

        if self._liquidation_alerted:
            return

        reason = risk_result.reason or "Profil ryzyka w trybie awaryjnym – przekroczono limit straty lub obsunięcia."
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
    ) -> None:
        order_id = result.order_id or ""
        avg_price = result.avg_price or request.price or 0.0
        filled_qty = result.filled_quantity or request.quantity

        context = {
            "symbol": request.symbol,
            "side": request.side,
            "order_id": order_id,
            "avg_price": f"{avg_price:.8f}",
            "filled_quantity": f"{filled_qty:.8f}",
            "status": result.status or "unknown",
            "environment": self.environment,
            "risk_profile": self.risk_profile,
        }

        for key, value in signal.metadata.items():
            context.setdefault(f"meta_{key}", str(value))

        message = AlertMessage(
            category="execution",
            title=f"Zlecenie {request.side} {request.symbol} zrealizowane",
            body="Zlecenie zostało wykonane w symulatorze/na giełdzie.",
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

    def _emit_execution_error_alert(
        self,
        signal: StrategySignal,
        request: OrderRequest,
        error: Exception,
    ) -> None:
        context = {
            "symbol": request.symbol,
            "side": request.side,
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
        _LOGGER.exception(
            "Błąd egzekucji zlecenia %s %s: %s", request.side, request.symbol, error
        )
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
    strategy: Any  # StrategyEngine kompatybilny: posiada on_data(snapshot)->Sequence[StrategySignal]
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
            raise KeyError(f"Brak konfiguracji środowiska '{self.environment_name}' w CoreConfig") from exc
        try:
            self._runtime = self.core_config.runtime_controllers[self.controller_name]
        except Exception as exc:
            raise KeyError(f"Brak sekcji runtime dla kontrolera '{self.controller_name}' w CoreConfig") from exc
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
        for signal in signals:
            base_request = self._build_order_request(snapshot, signal)
            account_snapshot = self.account_loader()
            risk_result = self.risk_engine.apply_pre_trade_checks(
                base_request,
                account=account_snapshot,
                profile_name=self._risk_profile,
            )
            request = base_request
            if not risk_result.allowed:
                adjusted_qty = _extract_adjusted_quantity(base_request.quantity, risk_result.adjustments)
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

    def _build_order_request(self, snapshot: MarketSnapshot, signal: StrategySignal) -> OrderRequest:
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
                exchange=self.exchange_name or getattr(self._environment, "exchange", self.environment_name),
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
        if DecisionOrchestrator is not None and not isinstance(self.orchestrator, DecisionOrchestrator):
            _LOGGER.debug("AIDecisionLoop: używam niestandardowej implementacji orchestratora %s", type(self.orchestrator))
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
            evaluation = self.orchestrator.evaluate_candidate(candidate, snapshot)
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
        timestamp = datetime.now(timezone.utc) - timedelta(seconds=self.interval_seconds * self.candle_count)
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
