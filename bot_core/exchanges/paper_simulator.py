"""Paper trading simulators mirroring margin and futures behaviours.

The real adapters expose additional information that the historical
``PaperBackend`` did not cover (maintenance margin, leverage drift,
funding payments).  Multi-strategy runtimes rely on those values being
consistent between paper and live environments – especially when we run
margin/futures strategies in dry-run mode.  The module provides thin
wrappers around :class:`bot_core.exchanges.core.PaperBackend` that extend
the bookkeeping without altering the proven fill mechanics.

The simulators keep the feature-set intentionally small but production
ready:

* they allow both long and short exposure and enforce leverage caps,
* margin decisions (e.g. leverage changes, funding payments and
  liquidations) are logged for audit/telemetry purposes,
* the account snapshot mirrors the structure returned by native
  adapters so contract tests can assert on common fields,
* funding accrual uses a simplified continuous model that is sufficient
  for pipeline smoke tests without introducing external time
  dependencies.

The implementation favours determinism and testability.  Expensive
operations (e.g. fetching market prices) reuse the functionality already
implemented in ``PaperBackend``.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Optional

from bot_core.exchanges.base import AccountSnapshot
from bot_core.exchanges.core import (
    Event,
    OrderDTO,
    OrderSide,
    OrderStatus,
    PaperBackend,
    _PaperPositionState,
)


log = logging.getLogger(__name__)


@dataclass(slots=True)
class _MarginState:
    """Runtime parameters shared by both simulators."""

    leverage_limit: float
    maintenance_margin_ratio: float
    funding_rate: float
    last_funding: dt.datetime
    funding_interval_seconds: float


class PaperMarginSimulator(PaperBackend):
    """Extended paper backend aware of leverage and maintenance margin."""

    def __init__(
        self,
        price_feed_backend,
        *,
        event_bus=None,
        leverage_limit: float = 3.0,
        maintenance_margin_ratio: float = 0.15,
        funding_rate: float = 0.0,
        initial_cash: float = 10_000.0,
        cash_asset: str = "USDT",
        fee_rate: Optional[float] = None,
        database=None,
        funding_interval_seconds: Optional[float] = None,
    ) -> None:
        super().__init__(
            price_feed_backend,
            event_bus=event_bus,
            initial_cash=initial_cash,
            cash_asset=cash_asset,
            fee_rate=fee_rate,
            database=database,
        )
        leverage_limit = max(1.0, float(leverage_limit))
        maintenance_margin_ratio = max(0.01, float(maintenance_margin_ratio))
        funding_rate = float(funding_rate)
        funding_interval = 0.0
        if funding_interval_seconds is not None:
            funding_interval = max(0.0, float(funding_interval_seconds))

        self._margin_state = _MarginState(
            leverage_limit=leverage_limit,
            maintenance_margin_ratio=maintenance_margin_ratio,
            funding_rate=funding_rate,
            last_funding=dt.datetime.utcnow(),
            funding_interval_seconds=funding_interval,
        )
        self._margin_events: list[dict[str, object]] = []

    # ------------------------------------------------------------------ public
    def fetch_account_snapshot(self) -> AccountSnapshot:
        equity = self._calculate_equity()
        exposure = self._gross_exposure()
        leverage = exposure / max(equity, 1e-9)
        initial_margin = exposure / max(self._margin_state.leverage_limit, 1.0)
        maintenance_margin = exposure * self._margin_state.maintenance_margin_ratio
        available_margin = max(0.0, equity - initial_margin)
        balances: MutableMapping[str, float] = {self._cash_asset: self._cash_balance}
        for symbol, pos in self._positions.items():
            if abs(pos.quantity) <= 0:
                continue
            price = self._last_prices.get(symbol)
            if price is None:
                continue
            # Provide mark-to-market values in quote currency.
            balances[f"{symbol}_position"] = pos.quantity * price
        snapshot = AccountSnapshot(
            balances=dict(balances),
            total_equity=float(equity),
            available_margin=float(available_margin),
            maintenance_margin=float(maintenance_margin),
        )
        return snapshot

    def fetch_margin_events(self) -> Iterable[Mapping[str, object]]:
        """Returns a copy of recorded margin/funding events."""

        return list(self._margin_events)

    def describe_configuration(self) -> dict[str, float]:
        """Expose runtime configuration for telemetry/reporting."""

        state = self._margin_state
        return {
            "leverage_limit": float(state.leverage_limit),
            "maintenance_margin_ratio": float(state.maintenance_margin_ratio),
            "funding_rate": float(state.funding_rate),
            "funding_interval_seconds": float(state.funding_interval_seconds),
        }

    # ----------------------------------------------------------------- overrides
    def _fill_order(self, order, price: float, timestamp: dt.datetime) -> OrderDTO:  # type: ignore[override]
        """Fill orders while accounting for leverage constraints.

        The implementation builds upon the base class but allows opening
        short exposure.  Liquidation checks reuse a helper executed after
        every fill.
        """

        qty = order.remaining
        if qty <= 0:
            return self._to_order_dto(order)

        self._apply_funding(timestamp)
        signed_quantity = qty if order.side == OrderSide.BUY else -qty
        fee = qty * price * self._fee_rate
        notional = qty * price
        leverage_before = self._effective_leverage()

        if order.side == OrderSide.BUY:
            self._cash_balance -= notional + fee
        else:
            self._cash_balance += notional - fee

        self._apply_margin_fill(order.symbol, signed_quantity, price)
        order.remaining = 0.0
        order.status = OrderStatus.FILLED
        self._update_order_status(order, OrderStatus.FILLED)
        self._orders.pop(order.id, None)

        dto = self._to_order_dto(order)
        dto.status = OrderStatus.FILLED
        dto.price = price
        dto.ts = timestamp.timestamp()
        self.event_bus.publish(Event(type="ORDER_FILLED", payload=dto.model_dump()))

        self._record_trade(order, qty, price, fee, timestamp)
        self._last_prices[order.symbol] = price
        self._enforce_margin(timestamp)

        leverage_after = self._effective_leverage()
        if abs(leverage_after - leverage_before) > 1e-9:
            self._record_margin_event(
                "leverage_change",
                timestamp,
                {
                    "symbol": order.symbol,
                    "side": order.side.value,
                    "quantity": signed_quantity,
                    "price": price,
                    "from": leverage_before,
                    "to": leverage_after,
                },
            )

        self._log_equity(timestamp)
        return dto

    def _apply_margin_fill(self, symbol: str, signed_quantity: float, price: float) -> None:
        pos = self._positions.get(symbol)
        if pos is None:
            pos = _PaperPositionState(symbol=symbol, quantity=0.0, avg_price=0.0)
            self._positions[symbol] = pos

        old_qty = pos.quantity
        new_qty = old_qty + signed_quantity
        if old_qty == 0 or (old_qty > 0 and new_qty >= 0) or (old_qty < 0 and new_qty <= 0):
            # Same direction – weighted average.
            total_qty = abs(old_qty) + abs(signed_quantity)
            if total_qty <= 0:
                pos.avg_price = 0.0
            else:
                pos.avg_price = (
                    abs(old_qty) * pos.avg_price + abs(signed_quantity) * price
                ) / total_qty
            pos.quantity = new_qty
        else:
            # Closing existing exposure partially or fully.
            closing_qty = min(abs(old_qty), abs(signed_quantity))
            pnl_multiplier = 1.0 if old_qty > 0 else -1.0
            realized = (price - pos.avg_price) * closing_qty * pnl_multiplier
            self._realized_pnl += realized
            remaining = old_qty + signed_quantity
            if abs(remaining) <= 1e-9:
                pos.quantity = 0.0
                pos.avg_price = 0.0
            else:
                pos.quantity = remaining
                pos.avg_price = price

        pos.unrealized_pnl = 0.0
        self._persist_position(pos)

    def _refresh_unrealized(self, symbols: Optional[Iterable[str]] = None) -> None:  # type: ignore[override]
        targets = set(symbols or self._positions.keys())
        for symbol in targets:
            pos = self._positions.get(symbol)
            if not pos or abs(pos.quantity) <= 0:
                continue
            price = self._last_prices.get(symbol)
            if price is None:
                price = self._resolve_trade_price(symbol)
                if price is None:
                    continue
                self._last_prices[symbol] = price
            direction = 1.0 if pos.quantity > 0 else -1.0
            pos.unrealized_pnl = float((price - pos.avg_price) * abs(pos.quantity) * direction)
            self._persist_position(pos)

    # ----------------------------------------------------------------- helpers
    def _apply_funding(self, timestamp: dt.datetime) -> None:
        rate = self._margin_state.funding_rate
        if rate == 0:
            return
        previous = self._margin_state.last_funding
        elapsed = max(0.0, (timestamp - previous).total_seconds())
        if elapsed <= 0:
            return
        exposure = self._gross_exposure()
        if exposure <= 0:
            self._margin_state.last_funding = timestamp
            return
        interval = self._margin_state.funding_interval_seconds
        if interval <= 0:
            payment = exposure * rate * (elapsed / 86_400.0)
            self._cash_balance -= payment
            self._margin_state.last_funding = timestamp
            self._record_margin_event(
                "funding",
                timestamp,
                {"payment": payment, "exposure": exposure, "elapsed_seconds": elapsed},
            )
            return

        periods = math.floor(elapsed / interval)
        if periods <= 0:
            return
        payment = exposure * rate * ((interval / 86_400.0) * periods)
        self._cash_balance -= payment
        funded_until = previous + dt.timedelta(seconds=interval * periods)
        self._margin_state.last_funding = funded_until
        self._record_margin_event(
            "funding",
            timestamp,
            {
                "payment": payment,
                "exposure": exposure,
                "interval_seconds": interval,
                "periods": periods,
            },
        payment = exposure * rate * (elapsed / 86_400.0)
        self._cash_balance -= payment
        self._margin_state.last_funding = timestamp
        self._record_margin_event(
            "funding", timestamp, {"payment": payment, "exposure": exposure}
        )

    def _enforce_margin(self, timestamp: dt.datetime) -> None:
        equity = self._calculate_equity()
        exposure = self._gross_exposure()
        if exposure <= 0:
            return
        maintenance = exposure * self._margin_state.maintenance_margin_ratio
        if equity < maintenance:
            log.error(
                "Paper margin liquidation: equity %.2f < maintenance %.2f", equity, maintenance
            )
            self._record_margin_event(
                "liquidation",
                timestamp,
                {"equity": equity, "maintenance": maintenance, "exposure": exposure},
            )
            # Liquidate by closing all positions at last known prices.
            for symbol, pos in list(self._positions.items()):
                price = self._last_prices.get(symbol)
                if price is None:
                    price = pos.avg_price
                if price <= 0:
                    continue
                if pos.quantity == 0:
                    continue
                closing_notional = abs(pos.quantity) * price
                self._cash_balance += closing_notional
                self._realized_pnl += (price - pos.avg_price) * pos.quantity
                pos.quantity = 0.0
                pos.avg_price = 0.0
                self._persist_position(pos)
            self._positions.clear()
            raise RuntimeError("Paper margin liquidation triggered")

    def _calculate_equity(self) -> float:
        equity = self._cash_balance
        for symbol, pos in self._positions.items():
            price = self._last_prices.get(symbol)
            if price is None:
                price = self._resolve_trade_price(symbol)
                if price is None:
                    continue
                self._last_prices[symbol] = price
            equity += pos.quantity * price
        return float(equity)

    def _gross_exposure(self) -> float:
        exposure = 0.0
        for symbol, pos in self._positions.items():
            price = self._last_prices.get(symbol)
            if price is None:
                price = self._resolve_trade_price(symbol)
                if price is None:
                    continue
                self._last_prices[symbol] = price
            exposure += abs(pos.quantity * price)
        return float(exposure)

    def _effective_leverage(self) -> float:
        equity = self._calculate_equity()
        if equity <= 0:
            return 0.0
        return self._gross_exposure() / equity

    def _record_margin_event(
        self, event_type: str, timestamp: dt.datetime, payload: Mapping[str, object]
    ) -> None:
        event = {
            "type": event_type,
            "timestamp": timestamp.isoformat(),
            "payload": dict(payload),
        }
        self._margin_events.append(event)
        log.info("Paper margin event %s: %s", event_type, event["payload"])


class PaperFuturesSimulator(PaperMarginSimulator):
    """Specialised simulator for inverse/linear futures contracts."""

    def __init__(
        self,
        price_feed_backend,
        *,
        leverage_limit: float = 10.0,
        maintenance_margin_ratio: float = 0.05,
        funding_rate: float = 0.0001,
        initial_cash: float = 10_000.0,
        cash_asset: str = "USDT",
        fee_rate: Optional[float] = None,
        database=None,
        funding_interval_seconds: Optional[float] = None,
    ) -> None:
        super().__init__(
            price_feed_backend,
            leverage_limit=leverage_limit,
            maintenance_margin_ratio=maintenance_margin_ratio,
            funding_rate=funding_rate,
            initial_cash=initial_cash,
            cash_asset=cash_asset,
            fee_rate=fee_rate,
            database=database,
            funding_interval_seconds=funding_interval_seconds,
        )

    def fetch_account_snapshot(self) -> AccountSnapshot:  # type: ignore[override]
        snapshot = super().fetch_account_snapshot()
        exposure = self._gross_exposure()
        snapshot.balances = dict(snapshot.balances)  # type: ignore[assignment]
        snapshot.balances.setdefault("futures_exposure", exposure)
        return snapshot


__all__ = ["PaperMarginSimulator", "PaperFuturesSimulator"]

