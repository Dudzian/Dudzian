"""Podstawowe typy i backendy wymiany dla natywnego rdzenia."""

from __future__ import annotations

import datetime as dt
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from bot_core.database.manager import DatabaseManager

log = logging.getLogger(__name__)


# =========================
#         ENUMY
# =========================


class Mode(str, Enum):
    PAPER = "paper"
    SPOT = "spot"
    MARGIN = "margin"
    FUTURES = "futures"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    OPEN = "OPEN"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"


# =========================
#         DTO
# =========================


class MarketRules(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    symbol: str
    price_step: float = 0.0
    amount_step: float = 0.0
    min_notional: float = 0.0
    min_amount: float = 0.0
    max_amount: float | None = None
    min_price: float | None = None
    max_price: float | None = None

    def quantize_amount(self, amount: float) -> float:
        if amount <= 0:
            return 0.0
        step = self.amount_step or 0.0
        if step > 0:
            return math.floor(amount / step) * step
        return float(f"{amount:.8f}")

    def quantize_price(self, price: float) -> float:
        if price <= 0:
            return 0.0
        step = self.price_step or 0.0
        if step > 0:
            return math.floor(price / step) * step
        return float(f"{price:.8f}")


class OrderDTO(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    id: Optional[int] = None
    client_order_id: Optional[str] = None
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.OPEN
    ts: float = Field(default_factory=lambda: time.time())
    mode: Mode = Mode.PAPER
    extra: Dict[str, Any] = Field(default_factory=dict)


class TradeDTO(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    id: Optional[int] = None
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fee: float = 0.0
    order_id: Optional[int] = None
    ts: float = Field(default_factory=lambda: time.time())
    mode: Mode = Mode.PAPER
    extra: Dict[str, Any] = Field(default_factory=dict)


class PositionDTO(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    symbol: str
    side: str  # "LONG" | "SHORT"
    quantity: float
    avg_price: float
    unrealized_pnl: float = 0.0
    mode: Mode = Mode.PAPER


class SignalDTO(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    symbol: str
    direction: str  # "LONG" | "SHORT"
    confidence: float = 1.0
    extra: Dict[str, Any] = Field(default_factory=dict)


# =========================
#       Event Bus
# =========================


class Event(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    type: str
    payload: Dict[str, Any] = Field(default_factory=dict)


class EventBus:
    def __init__(self) -> None:
        self._subs: Dict[str, List[Callable[[Event], None]]] = {}

    def subscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
        self._subs.setdefault(event_type, []).append(callback)

    def publish(self, event: Event) -> None:
        for cb in self._subs.get(event.type, []):
            try:
                cb(event)
            except Exception as exc:
                log.error("EventBus subscriber error: %s", exc)


# =========================
#    Backend – interfejs
# =========================


class BaseBackend:
    """Interfejs dla backendów: PAPER / SPOT / FUTURES."""

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        self.event_bus = event_bus or EventBus()

    def load_markets(self) -> Dict[str, MarketRules]:
        raise NotImplementedError

    def get_market_rules(self, symbol: str) -> Optional[MarketRules]:
        raise NotImplementedError

    def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 500
    ) -> Optional[List[List[float]]]:
        raise NotImplementedError

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> OrderDTO:
        raise NotImplementedError

    def cancel_order(self, order_id: Any, symbol: str) -> bool:
        raise NotImplementedError

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[OrderDTO]:
        raise NotImplementedError

    def fetch_positions(self, symbol: Optional[str] = None) -> List[PositionDTO]:
        raise NotImplementedError

    def quantize_amount(self, symbol: str, amount: float) -> float:
        mr = self.get_market_rules(symbol)
        return mr.quantize_amount(amount) if mr else float(f"{amount:.8f}")

    def quantize_price(self, symbol: str, price: float) -> float:
        mr = self.get_market_rules(symbol)
        return mr.quantize_price(price) if mr else float(f"{price:.8f}")

    def min_notional(self, symbol: str) -> float:
        mr = self.get_market_rules(symbol)
        return float(mr.min_notional) if mr else 0.0


# =========================
#   PAPER backend (light)
# =========================


@dataclass(slots=True)
class _PaperOrderState:
    id: int
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float]
    remaining: float
    status: OrderStatus
    client_order_id: Optional[str]
    created_at: dt.datetime
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _PaperPositionState:
    symbol: str
    quantity: float
    avg_price: float
    side: str = "LONG"
    unrealized_pnl: float = 0.0


class PaperBackend(BaseBackend):
    FEE_RATE = 0.001

    def __init__(
        self,
        price_feed_backend: BaseBackend,
        event_bus: Optional[EventBus] = None,
        *,
        initial_cash: float = 10_000.0,
        cash_asset: str = "USDT",
        fee_rate: Optional[float] = None,
        database: Optional[DatabaseManager] = None,
    ) -> None:
        super().__init__(event_bus=event_bus)
        self._price_feed = price_feed_backend
        self._rules: Dict[str, MarketRules] = {}
        self._cash_asset = cash_asset.upper()
        self._cash_balance = max(0.0, float(initial_cash))
        base_fee = self.FEE_RATE if fee_rate is None else float(fee_rate)
        self._fee_rate = max(0.0, base_fee)
        self._managed_db = database is None
        self._db = database or DatabaseManager("sqlite+aiosqlite:///trading.db")
        if hasattr(self._db, "sync") and hasattr(self._db.sync, "init_db"):
            try:
                self._db.sync.init_db()
            except Exception:  # pragma: no cover - defensywne logowanie
                log.debug("PaperBackend init_db failed", exc_info=True)
        self._orders: Dict[int, _PaperOrderState] = {}
        self._positions: Dict[str, _PaperPositionState] = {}
        self._last_prices: Dict[str, float] = {}
        self._realized_pnl: float = 0.0

    # ------------------------------------------------------------------ market
    def load_markets(self) -> Dict[str, MarketRules]:
        if hasattr(self._price_feed, "load_markets"):
            feed_rules = self._price_feed.load_markets()
        else:
            feed_rules = {}
        self._rules = dict(feed_rules)
        return self._rules

    def get_market_rules(self, symbol: str) -> Optional[MarketRules]:
        return self._rules.get(symbol)

    def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self._price_feed.fetch_ticker(symbol)

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 500
    ) -> Optional[List[List[float]]]:
        return self._price_feed.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    # ------------------------------------------------------------------- orders
    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> OrderDTO:
        qty = self.quantize_amount(symbol, float(quantity))
        if qty <= 0:
            raise ValueError("Ilość po kwantyzacji = 0.")

        now = dt.datetime.utcnow()
        reference_price: Optional[float]
        limit_price: Optional[float] = None
        if type == OrderType.LIMIT:
            if price is None or price <= 0:
                raise ValueError("Cena wymagana dla LIMIT.")
            limit_price = self.quantize_price(symbol, float(price))
            reference_price = limit_price
        else:
            reference_price = self._resolve_trade_price(symbol)

        if reference_price is None:
            raise RuntimeError(f"Brak ceny dla {symbol}. Użyj 'Load Markets'.")

        mn = self.min_notional(symbol)
        notional = qty * reference_price
        if mn and notional < mn:
            raise ValueError(
                f"Notional {notional:.8f} < minNotional {mn:.8f} dla {symbol}"
            )

        order_id = self._record_order(
            symbol,
            side,
            type,
            qty,
            limit_price if type == OrderType.LIMIT else None,
            client_order_id,
        )

        order = _PaperOrderState(
            id=order_id,
            symbol=symbol,
            side=side,
            type=type,
            quantity=qty,
            price=limit_price,
            remaining=qty,
            status=OrderStatus.OPEN,
            client_order_id=client_order_id,
            created_at=now,
        )

        if type == OrderType.MARKET:
            dto = self._fill_order(order, reference_price, now)
            return dto

        self._orders[order_id] = order
        self._update_order_status(order, OrderStatus.OPEN)
        return self._to_order_dto(order)

    def cancel_order(self, order_id: Any, symbol: str) -> bool:
        state = self._orders.get(int(order_id))
        if state is None or state.symbol != symbol:
            return False
        if state.status == OrderStatus.FILLED:
            return False
        state.status = OrderStatus.CANCELED
        state.remaining = 0.0
        self._update_order_status(state, OrderStatus.CANCELED)
        self._orders.pop(state.id, None)
        return True

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[OrderDTO]:
        return [
            self._to_order_dto(order)
            for order in self._orders.values()
            if order.status not in {OrderStatus.CANCELED, OrderStatus.FILLED}
            and (symbol is None or order.symbol == symbol)
        ]

    # ---------------------------------------------------------------- positions
    def fetch_positions(self, symbol: Optional[str] = None) -> List[PositionDTO]:
        self._refresh_unrealized()
        return [
            PositionDTO(
                symbol=pos.symbol,
                side=pos.side,
                quantity=pos.quantity,
                avg_price=pos.avg_price,
                unrealized_pnl=pos.unrealized_pnl,
                mode=Mode.PAPER,
            )
            for pos in self._positions.values()
            if symbol is None or pos.symbol == symbol
        ]

    def fetch_balance(self) -> Dict[str, Any]:
        balances: Dict[str, float] = {self._cash_asset: float(self._cash_balance)}
        for pos in self._positions.values():
            base = pos.symbol.split("/")[0].upper()
            balances[base] = balances.get(base, 0.0) + float(pos.quantity)

        totals = {asset: float(amount) for asset, amount in balances.items()}
        out: Dict[str, Any] = {
            "free": dict(totals),
            "total": dict(totals),
        }
        out.update(totals)
        return out

    # ------------------------------------------------------------------ updates
    def set_fee_rate(self, fee_rate: float) -> None:
        self._fee_rate = max(0.0, float(fee_rate))

    def get_fee_rate(self) -> float:
        return self._fee_rate

    def process_tick(
        self,
        symbol: str,
        price: float,
        *,
        timestamp: Optional[dt.datetime] = None,
    ) -> None:
        ts = timestamp or dt.datetime.utcnow()
        px = float(price)
        self._last_prices[symbol] = px
        to_fill: List[_PaperOrderState] = []
        for order in list(self._orders.values()):
            if order.symbol != symbol:
                continue
            if order.status != OrderStatus.OPEN:
                continue
            if order.type != OrderType.LIMIT or order.price is None:
                continue
            if order.side == OrderSide.BUY and px <= order.price:
                to_fill.append(order)
            elif order.side == OrderSide.SELL and px >= order.price:
                to_fill.append(order)

        for order in to_fill:
            self._fill_order(order, px, ts)

        self._refresh_unrealized(symbols=[symbol])
        self._log_equity(ts)

    # ----------------------------------------------------------------- helpers
    def _record_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float],
        client_order_id: Optional[str],
    ) -> int:
        payload = {
            "symbol": symbol,
            "side": side.value,
            "type": order_type.value,
            "quantity": float(quantity),
            "price": float(price) if price is not None else None,
            "mode": Mode.PAPER.value,
            "client_order_id": client_order_id,
            "status": OrderStatus.OPEN.value,
        }
        recorder = getattr(self._db.sync, "record_order", None)
        if not callable(recorder):
            raise RuntimeError("DatabaseManager nie udostępnia record_order")
        order_id = int(recorder(payload))
        return order_id

    def _update_order_status(self, order: _PaperOrderState, status: OrderStatus) -> None:
        updater = getattr(self._db.sync, "update_order_status", None)
        if callable(updater):
            try:
                updater(order_id=order.id, status=status.value)
            except Exception:  # pragma: no cover
                log.debug("update_order_status failed", exc_info=True)

    def _fill_order(
        self,
        order: _PaperOrderState,
        price: float,
        timestamp: dt.datetime,
    ) -> OrderDTO:
        qty = order.remaining
        if qty <= 0:
            return self._to_order_dto(order)

        fee = qty * price * self._fee_rate
        if order.side == OrderSide.BUY:
            required = qty * price + fee
            if required > self._cash_balance + 1e-8:
                raise ValueError("Niewystarczający balans papierowy do zakupu.")
            self._cash_balance = max(0.0, self._cash_balance - required)
            self._apply_fill(order.symbol, qty, price, is_buy=True)
        else:
            self._apply_fill(order.symbol, qty, price, is_buy=False)
            self._cash_balance += max(0.0, qty * price - fee)

        self._record_trade(order, qty, price, fee, timestamp)

        order.remaining = 0.0
        order.status = OrderStatus.FILLED
        self._update_order_status(order, OrderStatus.FILLED)
        self._orders.pop(order.id, None)

        dto = self._to_order_dto(order)
        dto.status = OrderStatus.FILLED
        dto.price = price
        dto.ts = timestamp.timestamp()
        self.event_bus.publish(Event(type="ORDER_FILLED", payload=dto.model_dump()))
        self._log_equity(timestamp)
        return dto

    def _record_trade(
        self,
        order: _PaperOrderState,
        quantity: float,
        price: float,
        fee: float,
        timestamp: dt.datetime,
    ) -> None:
        recorder = getattr(self._db.sync, "record_trade", None)
        if not callable(recorder):
            return
        payload = {
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": float(quantity),
            "price": float(price),
            "fee": float(fee),
            "order_id": order.id,
            "mode": Mode.PAPER.value,
            "ts": timestamp.isoformat(),
        }
        try:
            recorder(payload)
        except Exception:  # pragma: no cover
            log.debug("record_trade failed", exc_info=True)

    def _apply_fill(self, symbol: str, quantity: float, price: float, *, is_buy: bool) -> None:
        pos = self._positions.get(symbol)
        if pos is None:
            pos = _PaperPositionState(symbol=symbol, quantity=0.0, avg_price=0.0)
            self._positions[symbol] = pos

        if is_buy:
            new_qty = pos.quantity + quantity
            if new_qty <= 0:
                new_qty = 0.0
            avg = (
                (pos.quantity * pos.avg_price + quantity * price) / new_qty
                if new_qty > 0
                else 0.0
            )
            pos.quantity = float(f"{new_qty:.8f}")
            pos.avg_price = float(f"{avg:.8f}")
        else:
            if quantity > pos.quantity + 1e-8:
                raise ValueError("Brak pozycji do sprzedaży w trybie paper.")
            realized = (price - pos.avg_price) * quantity
            self._realized_pnl += realized
            pos.quantity = float(f"{max(0.0, pos.quantity - quantity):.8f}")
            if pos.quantity <= 1e-8:
                pos.quantity = 0.0
                pos.avg_price = 0.0

        pos.unrealized_pnl = 0.0
        self._persist_position(pos)

    def _persist_position(self, position: _PaperPositionState) -> None:
        writer = getattr(self._db.sync, "upsert_position", None)
        if not callable(writer):
            return
        payload = {
            "symbol": position.symbol,
            "side": position.side,
            "quantity": float(position.quantity),
            "avg_price": float(position.avg_price),
            "unrealized_pnl": float(position.unrealized_pnl),
            "mode": Mode.PAPER.value,
        }
        try:
            writer(payload)
        except Exception:  # pragma: no cover
            log.debug("upsert_position failed", exc_info=True)

    def _refresh_unrealized(self, symbols: Optional[Iterable[str]] = None) -> None:
        targets = set(symbols or self._positions.keys())
        for symbol in targets:
            pos = self._positions.get(symbol)
            if not pos or pos.quantity <= 0:
                continue
            price = self._last_prices.get(symbol)
            if price is None:
                price = self._resolve_trade_price(symbol)
                if price is None:
                    continue
                self._last_prices[symbol] = price
            pos.unrealized_pnl = float((price - pos.avg_price) * pos.quantity)
            self._persist_position(pos)

    def _log_equity(self, timestamp: dt.datetime) -> None:
        logger = getattr(self._db.sync, "log_equity", None)
        if not callable(logger):
            return
        equity = self._cash_balance
        for pos in self._positions.values():
            price = self._last_prices.get(pos.symbol)
            if price is None:
                continue
            equity += pos.quantity * price
        payload = {
            "mode": Mode.PAPER.value,
            "equity": float(equity),
            "balance": float(self._cash_balance),
            "pnl": float(self._realized_pnl),
            "ts": timestamp.isoformat(),
        }
        try:
            logger(payload)
        except Exception:  # pragma: no cover
            log.debug("log_equity failed", exc_info=True)

    def _resolve_trade_price(self, symbol: str) -> Optional[float]:
        cached = self._last_prices.get(symbol)
        if cached is not None:
            return cached
        ticker = self.fetch_ticker(symbol) or {}
        for key in ("last", "close", "bid", "ask"):
            value = ticker.get(key)
            if value:
                try:
                    px = float(value)
                except Exception:
                    continue
                self._last_prices[symbol] = px
                return px
        return None

    @staticmethod
    def _to_order_dto(order: _PaperOrderState) -> OrderDTO:
        return OrderDTO(
            id=order.id,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            type=order.type,
            quantity=order.quantity,
            price=order.price,
            status=order.status,
            mode=Mode.PAPER,
            ts=order.created_at.timestamp(),
        )

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        if self._managed_db:
            try:
                closer = getattr(self._db, "close", None)
                if callable(closer):
                    closer()
            except Exception:
                pass


__all__ = [
    "BaseBackend",
    "Event",
    "EventBus",
    "MarketRules",
    "Mode",
    "OrderDTO",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "PaperBackend",
    "PositionDTO",
    "SignalDTO",
    "TradeDTO",
]

