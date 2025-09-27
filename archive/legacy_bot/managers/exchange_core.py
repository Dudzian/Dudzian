# managers/exchange_core.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import time
import logging
from enum import Enum
from typing import Any, Dict, Optional, List, Callable, Tuple

from pydantic import BaseModel, Field, ConfigDict, field_validator

log = logging.getLogger(__name__)

# =========================
#         ENUMY
# =========================

class Mode(str, Enum):
    PAPER = "paper"
    SPOT = "spot"
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
        # fallback – 8 miejsc po przecinku
        return float(f"{amount:.8f}")

    def quantize_price(self, price: float) -> float:
        if price <= 0:
            return 0.0
        step = self.price_step or 0.0
        if step > 0:
            return math.floor(price / step) * step
        # fallback – 8 miejsc po przecinku
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
    side: str  # "LONG" | "SHORT" (dla spot używamy "LONG")
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
            except Exception as e:
                log.error("EventBus subscriber error: %s", e)


# =========================
#    Backend – interfejs
# =========================

class BaseBackend:
    """Interfejs dla backendów: PAPER / SPOT / FUTURES."""

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        self.event_bus = event_bus or EventBus()

    # --- rynki ---
    def load_markets(self) -> Dict[str, MarketRules]:  # zwraca mapę symbol->rules
        raise NotImplementedError

    def get_market_rules(self, symbol: str) -> Optional[MarketRules]:
        raise NotImplementedError

    # --- dane ---
    def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> Optional[List[List[float]]]:
        raise NotImplementedError

    # --- zlecenia ---
    def create_order(self, symbol: str, side: OrderSide, type: OrderType,
                     quantity: float, price: Optional[float] = None,
                     client_order_id: Optional[str] = None) -> OrderDTO:
        raise NotImplementedError

    def cancel_order(self, order_id: Any, symbol: str) -> bool:
        raise NotImplementedError

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[OrderDTO]:
        raise NotImplementedError

    # --- pozycje ---
    def fetch_positions(self, symbol: Optional[str] = None) -> List[PositionDTO]:
        raise NotImplementedError

    # --- pomocnicze ---
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

# Minimalny „realistyczny” PAPER: MARKET z natychmiastowym fill’em do DB.
# Limit/SL/TP/trailing – zostawiamy na panelu (jak dotąd).

from KryptoLowca.managers.database_manager import DatabaseManager

class PaperBackend(BaseBackend):
    FEE_RATE = 0.001  # 0.1%

    def __init__(self, price_feed_backend: "BaseBackend", event_bus: Optional[EventBus] = None) -> None:
        """
        price_feed_backend – skąd brać ceny OHLCV/ticker (np. CCXT public z ExchangeManagera).
        """
        super().__init__(event_bus=event_bus)
        self._db = DatabaseManager("sqlite+aiosqlite:///trading.db")
        self._db.sync.init_db()
        self._price_feed = price_feed_backend
        self._rules: Dict[str, MarketRules] = {}

    # --- rynki i dane ---

    def load_markets(self) -> Dict[str, MarketRules]:
        # papier korzysta z reguł z feedu, jeśli są dostępne
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

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> Optional[List[List[float]]]:
        return self._price_feed.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    # --- zlecenia ---

    def create_order(self, symbol: str, side: OrderSide, type: OrderType,
                     quantity: float, price: Optional[float] = None,
                     client_order_id: Optional[str] = None) -> OrderDTO:
        if type != OrderType.MARKET:
            # w Fazie 0 – LIMIT pozostawiamy panelowi; tutaj tylko MARKET
            raise ValueError("PaperBackend w Fazie 0 obsługuje tylko MARKET; LIMIT użyj w panelu Paper.")

        # kwantyzacja
        qty = self.quantize_amount(symbol, float(quantity))
        if qty <= 0:
            raise ValueError("Ilość po kwantyzacji = 0.")

        t = self.fetch_ticker(symbol) or {}
        last = t.get("last") or t.get("close") or t.get("bid") or t.get("ask")
        if not last:
            raise RuntimeError(f"Brak ceny MARKET dla {symbol}. Użyj 'Load Markets' w GUI.")
        px = float(last)

        notional = qty * px
        mn = self.min_notional(symbol)
        if mn and notional < mn:
            raise ValueError(f"Notional {notional:.8f} < minNotional {mn:.8f} dla {symbol}")

        # zapis zlecenia i transakcji do DB
        oid = self._db.sync.record_order({
            "symbol": symbol, "side": side.value, "type": type.value,
            "quantity": qty, "price": None, "mode": Mode.PAPER.value,
            "client_order_id": client_order_id
        })
        self._db.sync.update_order_status(order_id=oid, status=OrderStatus.FILLED.value)

        fee = qty * px * self.FEE_RATE
        self._db.sync.record_trade({
            "symbol": symbol, "side": side.value, "quantity": qty,
            "price": px, "fee": fee, "order_id": oid, "mode": Mode.PAPER.value
        })

        # aktualizacja pozycji (prosty netting LONG-only dla spot)
        # W tej Fazie 0: BUY -> LONG+, SELL -> LONG- (dla futures zajmiemy się w Fazie 3)
        pos_side = "LONG"
        pos = None
        for p in self._db.sync.get_open_positions(mode=Mode.PAPER.value):
            if p.get("symbol") == symbol and p.get("side") == pos_side:
                pos = p
                break
        if side == OrderSide.BUY:
            if pos:
                old_qty = float(pos["quantity"]); old_avg = float(pos["avg_price"])
                new_qty = old_qty + qty
                new_avg = (old_qty * old_avg + qty * px) / new_qty
            else:
                new_qty, new_avg = qty, px
        else:
            if pos:
                old_qty = float(pos["quantity"])
                new_qty = max(0.0, old_qty - qty)
                new_avg = pos["avg_price"] if new_qty > 0 else 0.0
            else:
                new_qty, new_avg = 0.0, 0.0
        self._db.sync.upsert_position({
            "symbol": symbol, "side": pos_side, "quantity": float(f"{new_qty:.8f}"),
            "avg_price": float(f"{new_avg:.8f}"), "unrealized_pnl": 0.0, "mode": Mode.PAPER.value
        })

        odto = OrderDTO(
            id=oid, client_order_id=client_order_id, symbol=symbol,
            side=side, type=type, quantity=qty, price=None,
            status=OrderStatus.FILLED, mode=Mode.PAPER
        )
        self.event_bus.publish(Event(type="ORDER_FILLED", payload=odto.model_dump()))
        return odto

    def cancel_order(self, order_id: Any, symbol: str) -> bool:
        # nic nie mamy otwartego w Fazie 0 (market-only), zwracamy False by sygnalizować, że nie dotyczy
        return False

    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[OrderDTO]:
        return []

    def fetch_positions(self, symbol: Optional[str] = None) -> List[PositionDTO]:
        out: List[PositionDTO] = []
        for p in self._db.sync.get_open_positions(mode=Mode.PAPER.value):
            if symbol and p.get("symbol") != symbol:
                continue
            out.append(PositionDTO(
                symbol=p.get("symbol"), side=p.get("side"),
                quantity=p.get("quantity"), avg_price=p.get("avg_price"),
                unrealized_pnl=p.get("unrealized_pnl", 0.0), mode=Mode.PAPER
            ))
        return out
