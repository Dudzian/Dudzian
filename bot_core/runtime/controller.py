"""Kontroler spinający strategię, silnik ryzyka i moduł alertów."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Mapping, MutableMapping, Sequence

from bot_core.alerts import AlertMessage, DefaultAlertRouter
from bot_core.execution import ExecutionContext, ExecutionService
from bot_core.exchanges.base import AccountSnapshot, OrderRequest, OrderResult
from bot_core.risk import RiskCheckResult, RiskEngine
from bot_core.strategies import StrategySignal

_LOGGER = logging.getLogger(__name__)


def _as_timedelta(value: timedelta | float | int) -> timedelta:
    if isinstance(value, timedelta):
        return value
    seconds = float(value)
    if seconds < 0:
        raise ValueError("Czas health-check nie może być ujemny")
    return timedelta(seconds=seconds)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True, kw_only=True)
class TradingController:
    """Podstawowy kontroler zarządzający przepływem sygnałów i alertów."""

    risk_engine: RiskEngine
    execution_service: ExecutionService
    alert_router: DefaultAlertRouter
    account_snapshot_provider: Callable[[], AccountSnapshot]
    portfolio_id: str
    environment: str
    risk_profile: str
    order_metadata_defaults: Mapping[str, str] | None = None
    clock: Callable[[], datetime] = _now
    health_check_interval: timedelta | float | int = timedelta(hours=1)
    execution_metadata: Mapping[str, str] | None = None
    _clock: Callable[[], datetime] = field(init=False, repr=False)
    _health_interval: timedelta = field(init=False, repr=False)
    _execution_context: ExecutionContext = field(init=False, repr=False)
    _order_defaults: dict[str, str] = field(init=False, repr=False)
    _last_health_report: datetime = field(init=False, repr=False)
    _liquidation_alerted: bool = field(init=False, repr=False)

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
        self._order_defaults = {
            str(k): str(v)
            for k, v in (self.order_metadata_defaults or {}).items()
        }
        self._last_health_report = self._clock()
        self._liquidation_alerted = False

    # ------------------------------------------------------------------ API --
    def process_signals(self, signals: Sequence[StrategySignal]) -> list[OrderResult]:
        """Przetwarza listę sygnałów strategii i zarządza alertami."""

        results: list[OrderResult] = []
        for signal in signals:
            if signal.side.upper() not in {"BUY", "SELL"}:
                _LOGGER.debug("Pomijam sygnał %s o kierunku %s", signal.symbol, signal.side)
                continue
            try:
                result = self._handle_signal(signal)
            except Exception:  # noqa: BLE001
                _LOGGER.exception("Błąd podczas przetwarzania sygnału %s", signal)
                raise
            if result is not None:
                results.append(result)

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

    # ------------------------------------------------------------ internals --
    def _handle_signal(self, signal: StrategySignal) -> OrderResult | None:
        self._emit_signal_alert(signal)
        request = self._build_order_request(signal)
        account = self.account_snapshot_provider()
        risk_result = self.risk_engine.apply_pre_trade_checks(
            request,
            account=account,
            profile_name=self.risk_profile,
        )

        if not risk_result.allowed:
            self._emit_order_rejected_alert(signal, request, risk_result)
            self._handle_liquidation_state(risk_result)
            return None

        result = self.execution_service.execute(request, self._execution_context)
        self._handle_liquidation_state(risk_result)
        return result

    def _build_order_request(self, signal: StrategySignal) -> OrderRequest:
        metadata = dict(self._order_defaults)
        metadata.update({str(k): str(v) for k, v in signal.metadata.items()})

        try:
            quantity = float(metadata["quantity"])
        except KeyError as exc:  # pragma: no cover - kontrakt kontrolera
            raise ValueError("Sygnał nie zawiera wielkości zlecenia (quantity)") from exc
        except ValueError as exc:
            raise ValueError("Wielkość zlecenia musi być liczbą zmiennoprzecinkową") from exc

        if quantity <= 0:
            raise ValueError("Wielkość zlecenia musi być dodatnia")

        price_value = metadata.get("price")
        price = float(price_value) if price_value is not None else None

        order_type = metadata.get("order_type", "market").upper()
        time_in_force = metadata.get("time_in_force")
        client_order_id = metadata.get("client_order_id")

        request = OrderRequest(
            symbol=signal.symbol,
            side=signal.side.upper(),
            quantity=quantity,
            order_type=order_type,
            price=price,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
        )
        return request

    def _emit_signal_alert(self, signal: StrategySignal) -> None:
        context = {
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
        context = {
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
        if not self.risk_engine.should_liquidate(profile_name=self.risk_profile):
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


__all__ = ["TradingController"]

