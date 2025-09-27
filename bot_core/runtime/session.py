"""Orkiestrator łączący strategię, ryzyko, egzekucję i alerty."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from bot_core.alerts.base import AlertMessage
from bot_core.alerts.router import DefaultAlertRouter
from bot_core.execution.base import ExecutionContext, ExecutionService
from bot_core.exchanges.base import AccountSnapshot, Environment, ExchangeAdapter, OrderRequest, OrderResult
from bot_core.risk.base import RiskEngine, RiskProfile
from bot_core.strategies.base import MarketSnapshot, StrategyEngine, StrategySignal

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class InstrumentConfig:
    """Minimalna wiedza o instrumencie potrzebna do zarządzania pozycją."""

    symbol: str
    base_asset: str
    quote_asset: str
    min_quantity: float = 0.0
    min_notional: float = 0.0
    step_size: float | None = None


@dataclass(slots=True)
class PlannedOrder:
    """Rezultat procesu wyznaczania wielkości pozycji."""

    quantity: float
    notional: float
    stop_price: float | None


class PositionSizer:
    """Odpowiada za wyznaczenie wielkości pozycji zgodnie z profilem ryzyka."""

    def __init__(self, profile: RiskProfile) -> None:
        self._profile = profile

    def size(
        self,
        *,
        signal: StrategySignal,
        snapshot: MarketSnapshot,
        account: AccountSnapshot,
        instrument: InstrumentConfig,
    ) -> tuple[PlannedOrder | None, str | None]:
        price = snapshot.close
        if price <= 0:
            return None, "Brak poprawnej ceny rynkowej – pomijam sygnał."

        max_position_pct = max(0.0, self._profile.max_position_exposure())
        if max_position_pct <= 0:
            return None, "Profil ryzyka nie zezwala na otwieranie pozycji."

        confidence = max(0.0, min(signal.confidence, 1.0))
        min_confidence = 0.1
        effective_confidence = max(min_confidence, confidence) if confidence > 0 else 0.0
        if effective_confidence <= 0:
            return None, "Sygnał o zerowej pewności został odrzucony."

        max_notional = max_position_pct * account.total_equity
        if max_notional <= 0:
            return None, "Kapitał dostępny na koncie jest zbyt niski do otwarcia pozycji."

        target_notional = max_notional * effective_confidence
        quantity = target_notional / price

        stop_price = signal.metadata.get("stop_price") if signal.metadata else None
        if isinstance(stop_price, (int, float)) and stop_price > 0:
            risk_per_unit = abs(price - float(stop_price))
            daily_limit_pct = max(0.0, self._profile.daily_loss_limit())
            if risk_per_unit > 0 and daily_limit_pct > 0:
                # Zachowujemy połowę bufora dziennej straty na inne pozycje.
                risk_budget = daily_limit_pct * account.total_equity * 0.5
                if risk_budget > 0:
                    max_qty_from_stop = risk_budget / risk_per_unit
                    quantity = min(quantity, max_qty_from_stop)
        else:
            stop_price = None

        side = signal.side.lower()
        balances = dict(account.balances)

        if side == "buy":
            available_quote = balances.get(instrument.quote_asset, account.available_margin)
            if available_quote is None:
                available_quote = account.available_margin
            max_by_cash = (available_quote / price) if price > 0 else 0.0
            quantity = min(quantity, max_by_cash)
            if quantity <= 0:
                return None, "Brak środków w walucie kwotowanej do otwarcia pozycji."
        elif side == "sell":
            available_base = balances.get(instrument.base_asset, 0.0)
            quantity = min(quantity, available_base)
            if quantity <= 0:
                return None, "Brak dostępnej ilości aktywa bazowego do sprzedaży."
        else:
            return None, "Obsługiwane są wyłącznie sygnały kupna lub sprzedaży."

        if instrument.min_quantity and quantity < instrument.min_quantity:
            return None, "Wielkość pozycji jest mniejsza niż minimalna ilość kontraktowa."

        if instrument.min_notional and quantity * price < instrument.min_notional:
            return None, "Notional pozycji jest niższy od minimalnego limitu giełdy."

        if instrument.step_size:
            if instrument.step_size <= 0:
                return None, "Niepoprawny krok wielkości pozycji."
            steps = math.floor(quantity / instrument.step_size)
            quantity = steps * instrument.step_size

        quantity = max(0.0, quantity)
        if quantity <= 0:
            return None, "Po zaokrągleniu wielkość pozycji spadła do zera."

        notional = quantity * price
        return PlannedOrder(quantity=quantity, notional=notional, stop_price=stop_price), None


class TradingSession:
    """Wiąże strategię, silnik ryzyka oraz moduł egzekucji w jednolity cykl."""

    def __init__(
        self,
        *,
        strategy: StrategyEngine,
        strategy_name: str,
        adapter: ExchangeAdapter,
        risk_engine: RiskEngine,
        risk_profile: RiskProfile,
        execution: ExecutionService,
        alert_router: DefaultAlertRouter,
        instruments: Mapping[str, InstrumentConfig],
        environment: Environment,
        portfolio_id: str,
        context_metadata: Mapping[str, str] | None = None,
        position_sizer: PositionSizer | None = None,
    ) -> None:
        self._strategy = strategy
        self._strategy_name = strategy_name
        self._adapter = adapter
        self._risk_engine = risk_engine
        self._risk_profile = risk_profile
        self._execution = execution
        self._alert_router = alert_router
        self._instruments = dict(instruments)
        self._environment = environment
        self._portfolio_id = portfolio_id
        self._position_sizer = position_sizer or PositionSizer(risk_profile)
        self._context_metadata = dict(context_metadata or {})
        self._liquidation_alerted = False
        self._execution_context = ExecutionContext(
            portfolio_id=portfolio_id,
            risk_profile=risk_profile.name,
            environment=environment.value,
            metadata={"strategy": strategy_name, **self._context_metadata},
        )

    def process_snapshot(self, snapshot: MarketSnapshot) -> Sequence[OrderResult]:
        """Przetwarza pojedynczy pakiet danych i realizuje sygnały strategii."""

        results: list[OrderResult] = []

        signals = list(self._strategy.on_data(snapshot))
        if not signals:
            return results

        if self._risk_engine.should_liquidate(profile_name=self._risk_profile.name):
            if not self._liquidation_alerted:
                self._dispatch_alert(
                    category="risk",
                    severity="critical",
                    title="Profil w trybie awaryjnym",
                    body=(
                        "Silnik ryzyka wymaga natychmiastowej redukcji ekspozycji."
                        " Pomijam nowe sygnały, dopóki profil pozostaje w trybie force-liquidation."
                    ),
                    context={"profile": self._risk_profile.name},
                )
                self._liquidation_alerted = True
            return results

        self._liquidation_alerted = False

        account = self._adapter.fetch_account_snapshot()

        for signal in signals:
            instrument = self._instruments.get(signal.symbol)
            if instrument is None:
                _LOGGER.warning("Brak metadanych instrumentu dla %s", signal.symbol)
                self._dispatch_alert(
                    category="trade",
                    severity="warning",
                    title="Pominięto sygnał",
                    body=f"Brak metadanych instrumentu {signal.symbol} uniemożliwia przygotowanie zlecenia.",
                    context={"symbol": signal.symbol, "strategy": self._strategy_name},
                )
                continue

            plan, rejection_reason = self._position_sizer.size(
                signal=signal, snapshot=snapshot, account=account, instrument=instrument
            )

            if plan is None:
                if rejection_reason:
                    self._dispatch_alert(
                        category="trade",
                        severity="warning",
                        title="Sygnał odrzucony przez sizer",
                        body=rejection_reason,
                        context={"symbol": signal.symbol, "strategy": self._strategy_name},
                    )
                continue

            order_request = OrderRequest(
                symbol=signal.symbol,
                side=signal.side,
                quantity=plan.quantity,
                order_type="market",
                price=snapshot.close,
            )

            risk_result = self._risk_engine.apply_pre_trade_checks(
                order_request,
                account=account,
                profile_name=self._risk_profile.name,
            )

            if not risk_result.allowed:
                self._dispatch_alert(
                    category="risk",
                    severity="warning",
                    title="Kontrola ryzyka odrzuciła zlecenie",
                    body=risk_result.reason or "Brak szczegółów od silnika ryzyka.",
                    context={
                        "symbol": signal.symbol,
                        "strategy": self._strategy_name,
                        "profile": self._risk_profile.name,
                    },
                )
                continue

            result = self._execution.execute(order_request, self._execution_context)
            fill_price = result.avg_price if result.avg_price is not None else snapshot.close
            position_value = result.filled_quantity * fill_price
            try:
                self._risk_engine.on_fill(
                    profile_name=self._risk_profile.name,
                    symbol=signal.symbol,
                    side=signal.side,
                    position_value=position_value,
                    pnl=0.0,
                )
            except Exception:  # pragma: no cover - silnik ryzyka nie powinien rzucać wyjątku
                _LOGGER.exception("Błąd aktualizacji stanu ryzyka po realizacji zlecenia")

            self._dispatch_alert(
                category="trade",
                severity="info",
                title=f"Zrealizowano {signal.side.upper()} {signal.symbol}",
                body=(
                    f"Wykonano zlecenie {signal.side} na {signal.symbol} w ilości {plan.quantity:.8f}"
                    f" po cenie {fill_price:.2f}."
                ),
                context={
                    "symbol": signal.symbol,
                    "strategy": self._strategy_name,
                    "profile": self._risk_profile.name,
                    "notional": f"{plan.notional:.2f}",
                },
            )

            results.append(result)

        return results

    def _dispatch_alert(
        self,
        *,
        category: str,
        severity: str,
        title: str,
        body: str,
        context: Mapping[str, str],
    ) -> None:
        message = AlertMessage(
            category=category,
            title=title,
            body=body,
            severity=severity,
            context=dict(context),
        )
        try:
            self._alert_router.dispatch(message)
        except Exception:  # pragma: no cover - powiadomienia nie powinny przerywać pętli
            _LOGGER.exception("Nie udało się wysłać alertu runtime")


__all__ = ["InstrumentConfig", "PlannedOrder", "PositionSizer", "TradingSession"]

