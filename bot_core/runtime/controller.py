"""Kontrolery spinające warstwy: dane/strategia/ryzyko/egzekucja oraz alerty."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import (
    Any,
    Callable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Mapping as TypingMapping,
)

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
try:
    from bot_core.config.models import CoreConfig, ControllerRuntimeConfig  # type: ignore
except Exception:  # pragma: no cover
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


def _extract_adjusted_quantity(
    original_quantity: float,
    adjustments: Mapping[str, float] | None,
) -> float | None:
    """Zwraca dopuszczalną wielkość zlecenia zasugerowaną przez silnik ryzyka."""

    if not adjustments:
        return None

    raw_value = adjustments.get("quantity")
    if raw_value is None:
        raw_value = adjustments.get("max_quantity")
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

    # ----------------------------------------------- API -----------------------------------------------
    def process_signals(self, signals: Sequence[StrategySignal]) -> list[OrderResult]:
        """Przetwarza listę sygnałów strategii i zarządza alertami."""
        results: list[OrderResult] = []
        for signal in signals:
            if signal.side.upper() not in {"BUY", "SELL"}:
                _LOGGER.debug("Pomijam sygnał %s o kierunku %s", signal.symbol, signal.side)
                continue
            metric_labels = dict(self._metric_labels)
            metric_labels["symbol"] = signal.symbol
            self._metric_signals_total.inc(labels={**metric_labels, "status": "received"})
            self._record_decision_event(
                "signal_received",
                signal=signal,
                status="received",
            )
            try:
                result = self._handle_signal(signal)
            except Exception:  # noqa: BLE001
                _LOGGER.exception("Błąd podczas przetwarzania sygnału %s", signal)
                raise
            if result is not None:
                results.append(result)
                self._metric_orders_total.inc(
                    labels={**metric_labels, "result": "executed", "side": signal.side.upper()},
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

    # ------------------------------------------- internals ----------------------------------------------
    def _handle_signal(self, signal: StrategySignal) -> OrderResult | None:
        self._emit_signal_alert(signal)
        request = self._build_order_request(signal)
        account = self.account_snapshot_provider()
        risk_result = self.risk_engine.apply_pre_trade_checks(
            request,
            account=account,
            profile_name=self.risk_profile,
        )
        metric_labels = dict(self._metric_labels)
        metric_labels["symbol"] = signal.symbol

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
        self._handle_liquidation_state(risk_result)
        return result

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

    def _build_order_request(self, signal: StrategySignal) -> OrderRequest:
        metadata_source: dict[str, object] = dict(self._order_defaults)
        # w StrategySignal.metadata spodziewamy się m.in. quantity/price/order_type/time_in_force/client_order_id
        for k, v in signal.metadata.items():
            metadata_source[str(k)] = v

        try:
            quantity = float(metadata_source["quantity"])
        except KeyError as exc:
            raise ValueError("Sygnał nie zawiera wielkości zlecenia (quantity)") from exc
        except ValueError as exc:
            raise ValueError("Wielkość zlecenia musi być liczbą zmiennoprzecinkową") from exc

        if quantity <= 0:
            raise ValueError("Wielkość zlecenia musi być dodatnia")

        price_value = metadata_source.get("price")
        price = float(price_value) if price_value is not None else None

        order_type = str(metadata_source.get("order_type") or "market").upper()
        time_in_force_raw = metadata_source.get("time_in_force")
        client_order_id_raw = metadata_source.get("client_order_id")

        order_metadata: dict[str, object] = dict(metadata_source)
        order_metadata["quantity"] = quantity
        if price is not None:
            order_metadata["price"] = price
        if "stop_price" in order_metadata:
            try:
                order_metadata["stop_price"] = float(order_metadata["stop_price"])
            except (TypeError, ValueError):
                raise ValueError("stop_price w metadanych musi być liczbą zmiennoprzecinkową")
        if "atr" in order_metadata:
            try:
                order_metadata["atr"] = float(order_metadata["atr"])
            except (TypeError, ValueError):
                raise ValueError("atr w metadanych musi być liczbą zmiennoprzecinkową")

        return OrderRequest(
            symbol=signal.symbol,
            side=signal.side.upper(),
            quantity=quantity,
            order_type=order_type,
            price=price,
            time_in_force=str(time_in_force_raw) if time_in_force_raw is not None else None,
            client_order_id=str(client_order_id_raw) if client_order_id_raw is not None else None,
            metadata=order_metadata,
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

    _environment: Any = field(init=False, repr=False)
    _runtime: ControllerRuntimeConfig = field(init=False, repr=False)
    _risk_profile: str = field(init=False, repr=False)
    _positions: dict[str, float] = field(init=False, repr=False, default_factory=dict)

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
        metadata_source: dict[str, object] = dict(signal.metadata)
        metadata_source.setdefault("quantity", float(self.position_size))
        metadata_source.setdefault("price", float(snapshot.close))
        metadata_source.setdefault("order_type", "market")

        quantity_raw = metadata_source.get("quantity")
        price_raw = metadata_source.get("price")
        time_in_force_raw = metadata_source.get("time_in_force")
        client_order_id_raw = metadata_source.get("client_order_id")

        quantity = float(quantity_raw) if quantity_raw is not None else float(self.position_size)
        price = float(price_raw) if price_raw is not None else float(snapshot.close)
        order_type = str(metadata_source.get("order_type") or "market")

        tif_str = str(time_in_force_raw) if time_in_force_raw is not None else None
        client_id_str = str(client_order_id_raw) if client_order_id_raw is not None else None

        order_metadata: dict[str, object] = dict(metadata_source)
        order_metadata["quantity"] = quantity
        order_metadata["price"] = price

        if "stop_price" in order_metadata:
            try:
                order_metadata["stop_price"] = float(order_metadata["stop_price"])
            except (TypeError, ValueError):
                order_metadata.pop("stop_price", None)

        if "atr" in order_metadata:
            try:
                order_metadata["atr"] = float(order_metadata["atr"])
            except (TypeError, ValueError):
                order_metadata.pop("atr", None)

        return OrderRequest(
            symbol=snapshot.symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            time_in_force=tif_str,
            client_order_id=client_id_str,
            metadata=order_metadata,
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


__all__ = ["TradingController", "DailyTrendController", "ControllerSignal"]
