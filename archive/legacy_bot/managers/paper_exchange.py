# managers/paper_exchange.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import uuid
import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from KryptoLowca.managers.database_manager import DatabaseManager

BUY = "BUY"
SELL = "SELL"

MARKET = "MARKET"
LIMIT = "LIMIT"
STOP = "STOP"
STOP_LIMIT = "STOP_LIMIT"


@dataclass
class OrderState:
    id: int
    client_order_id: Optional[str]
    side: str
    type: str
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    remaining: float
    status: str
    created_at: dt.datetime
    extra: Dict[str, object] = field(default_factory=dict)


@dataclass
class PositionState:
    side: str        # LONG/SHORT/FLAT
    quantity: float  # w jednostkach base (np. BTC)
    avg_price: float
    unrealized_pnl: float = 0.0


class PaperExchange:
    """
    Prosty, solidny paper-trading dla jednego symbolu (np. BTC/USDT).
    Obsługuje MARKET / LIMIT / STOP / STOP_LIMIT + fee, slippage, partial fills.
    Zapisuje do DB przez DatabaseManager (mode='paper').
    """
    def __init__(
        self,
        db: DatabaseManager,
        *,
        symbol: str = "BTC/USDT",
        starting_balance: float = 10_000.0,
        fee_rate: float = 0.001,        # 0.1%
        slippage_bps: int = 5,          # 0.05%
        max_partial_fill_qty: float = 0.05,
        min_qty: float = 1e-6,
    ) -> None:
        self.db = db
        self.symbol = symbol
        self.mode = "paper"

        self.balance_quote: float = float(starting_balance)
        self.realized_pnl: float = 0.0

        self.last_price: Optional[float] = None
        self.last_ts: Optional[dt.datetime] = None

        self.fee_rate = float(fee_rate)
        self.slippage_bps = int(slippage_bps)
        self.max_partial_fill_qty = float(max_partial_fill_qty)
        self.min_qty = float(min_qty)

        self._orders: Dict[int, OrderState] = {}
        self._position = PositionState(side="FLAT", quantity=0.0, avg_price=0.0, unrealized_pnl=0.0)

    # ---------- PUBLIC API ----------
    def create_order(
        self,
        *,
        side: str,
        type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        ts: Optional[dt.datetime] = None,
        extra: Optional[Dict[str, object]] = None,
    ) -> int:
        assert quantity > 0, "quantity > 0"
        if type in (LIMIT, STOP_LIMIT):
            assert price is not None and price > 0, "price wymagany dla LIMIT/STOP_LIMIT"
        if type in (STOP, STOP_LIMIT):
            assert stop_price is not None and stop_price > 0, "stop_price wymagany dla STOP/STOP_LIMIT"

        if not client_order_id:
            client_order_id = f"paper-{uuid.uuid4().hex[:12]}"

        o_id = self.db.sync.record_order({
            "symbol": self.symbol,
            "side": side.upper(),
            "type": type.upper(),
            "quantity": float(quantity),
            "price": float(price) if price is not None else None,
            "status": "NEW",
            "client_order_id": client_order_id,
            "mode": self.mode,
            "extra": extra or {},
        })

        now = ts or dt.datetime.utcnow()
        st = OrderState(
            id=o_id,
            client_order_id=client_order_id,
            side=side.upper(),
            type=type.upper(),
            quantity=float(quantity),
            price=float(price) if price is not None else None,
            stop_price=float(stop_price) if stop_price is not None else None,
            remaining=float(quantity),
            status="NEW",
            created_at=now,
            extra=extra or {},
        )
        self._orders[o_id] = st

        # MARKET próbujemy od razu
        if st.type == MARKET:
            self._open(st)
            self._try_fill_market(st, now)
        else:
            self._open(st)

        return o_id

    def cancel_order(self, *, order_id: Optional[int] = None, client_order_id: Optional[str] = None) -> bool:
        st = self._find_order(order_id, client_order_id)
        if st is None or st.status in ("FILLED", "CANCELED"):
            return False
        st.status = "CANCELED"
        self.db.sync.update_order_status(order_id=st.id, status="CANCELED")
        return True

    def get_open_orders(self) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for st in self._orders.values():
            if st.status not in ("FILLED", "CANCELED"):
                out.append({
                    "id": st.id,
                    "client_order_id": st.client_order_id,
                    "side": st.side,
                    "type": st.type,
                    "quantity": st.quantity,
                    "remaining": st.remaining,
                    "price": st.price,
                    "stop_price": st.stop_price,
                    "status": st.status,
                    "created_at": st.created_at.isoformat(),
                })
        return out

    def get_position(self) -> Dict[str, object]:
        self._mark_unrealized()
        return {
            "symbol": self.symbol,
            "side": self._position.side,
            "quantity": self._position.quantity,
            "avg_price": self._position.avg_price,
            "unrealized_pnl": self._position.unrealized_pnl,
            "balance_quote": self.balance_quote,
            "realized_pnl": self.realized_pnl,
        }

    def process_tick(self, price: float, ts: Optional[dt.datetime] = None) -> None:
        assert price > 0
        now = ts or dt.datetime.utcnow()
        self.last_price = float(price)
        self.last_ts = now

        # Triggery STOP/STOP_LIMIT
        for st in self._orders.values():
            if st.status in ("FILLED", "CANCELED"):
                continue
            if st.type in (STOP, STOP_LIMIT) and st.status in ("NEW", "OPEN", "PARTIALLY_FILLED"):
                if self._stop_triggered(st, price):
                    if st.type == STOP:
                        st.type = MARKET
                    elif st.type == STOP_LIMIT:
                        st.type = LIMIT
                    if st.status == "NEW":
                        self._open(st)

        # MARKET
        for st in list(self._orders.values()):
            if st.status in ("FILLED", "CANCELED"):
                continue
            if st.type == MARKET:
                self._try_fill_market(st, now)

        # LIMIT
        for st in list(self._orders.values()):
            if st.status in ("FILLED", "CANCELED"):
                continue
            if st.type == LIMIT:
                self._try_fill_limit(st, float(price), now)

        self._mark_unrealized()
        self._log_equity(now)

    # ---------- PRIVATE ----------
    def _find_order(self, order_id: Optional[int], client_order_id: Optional[str]) -> Optional[OrderState]:
        if order_id is not None:
            return self._orders.get(order_id)
        if client_order_id is not None:
            for st in self._orders.values():
                if st.client_order_id == client_order_id:
                    return st
        return None

    def _open(self, st: OrderState) -> None:
        if st.status == "NEW":
            st.status = "OPEN"
            self.db.sync.update_order_status(order_id=st.id, status="OPEN")

    def _stop_triggered(self, st: OrderState, price: float) -> bool:
        if st.stop_price is None:
            return False
        if st.side == BUY:
            return price >= st.stop_price
        else:
            return price <= st.stop_price

    def _apply_slippage(self, price: float, side: str) -> float:
        if self.slippage_bps <= 0:
            return price
        delta = price * (self.slippage_bps / 10_000.0)
        return price + delta if side == BUY else max(0.0, price - delta)

    def _try_fill_market(self, st: OrderState, now: dt.datetime) -> None:
        if self.last_price is None or st.remaining <= 0:
            return
        requested_qty = st.remaining
        exec_price = self._apply_slippage(self.last_price, st.side)
        self._fill(st, requested_qty, exec_price, now)

    def _try_fill_limit(self, st: OrderState, mkt_price: float, now: dt.datetime) -> None:
        if st.price is None or st.remaining <= 0:
            return
        crossed = (st.side == BUY and mkt_price <= st.price) or (st.side == SELL and mkt_price >= st.price)
        if not crossed:
            return
        requested_qty = min(st.remaining, self.max_partial_fill_qty) if self.max_partial_fill_qty > 0 else st.remaining
        if requested_qty < self.min_qty:
            requested_qty = st.remaining
        exec_price = float(st.price)
        self._fill(st, requested_qty, exec_price, now)

    def _fill(self, st: OrderState, requested_qty: float, price: float, now: dt.datetime) -> None:
        """
        Jednolity fill BUY/SELL z poprawnym cashflow i PnL.
        - requested_qty: ile chcemy zrealizować w TEJ iteracji (zanim zaczniemy rozbijać na close/open).
        Rejestrujemy do DB faktyczny exec_qty_this_fill (>0), łączne fee i aktualizujemy status tylko jeśli >0.
        """
        if requested_qty <= 0:
            return

        remaining_to_execute = requested_qty
        exec_qty_this_fill = 0.0
        total_fee_this_fill = 0.0
        realized = 0.0

        if st.side == BUY:
            # 1) zamknij SHORT (jeśli istnieje)
            if self._position.side == "SHORT" and remaining_to_execute > 0:
                close_qty = min(remaining_to_execute, self._position.quantity)
                if close_qty > 0:
                    # realized PnL: short zysk przy spadku ceny
                    realized += (self._position.avg_price - price) * close_qty
                    # cashflow: płacimy price*close_qty + fee
                    gross_close = close_qty * price
                    fee_close = gross_close * self.fee_rate
                    self.balance_quote -= (gross_close + fee_close)
                    total_fee_this_fill += fee_close

                    self._position.quantity -= close_qty
                    remaining_to_execute -= close_qty
                    exec_qty_this_fill += close_qty

                    if self._position.quantity <= 1e-12:
                        self._position = PositionState(side="FLAT", quantity=0.0, avg_price=0.0, unrealized_pnl=0.0)

            # 2) pozostała część BUY otwiera/rozszerza LONG
            if remaining_to_execute > 0:
                # limit środków
                affordable_qty = remaining_to_execute
                total_cost = affordable_qty * price * (1 + self.fee_rate)
                if total_cost > self.balance_quote + 1e-9:
                    # przelicz maksymalną ilość, na którą nas stać:
                    affordable_qty = (self.balance_quote / (price * (1 + self.fee_rate))) if price > 0 else 0.0

                if affordable_qty >= self.min_qty:
                    gross_open = affordable_qty * price
                    fee_open = gross_open * self.fee_rate
                    total_fee_this_fill += fee_open

                    if self._position.side == "LONG":
                        new_qty = self._position.quantity + affordable_qty
                        new_gross = self._position.avg_price * self._position.quantity + gross_open
                        self._position.avg_price = new_gross / new_qty
                        self._position.quantity = new_qty
                    else:
                        self._position.side = "LONG"
                        self._position.quantity = affordable_qty
                        self._position.avg_price = price

                    self.balance_quote -= (gross_open + fee_open)
                    exec_qty_this_fill += affordable_qty
                    remaining_to_execute -= affordable_qty

        else:  # SELL
            # 1) zamknij LONG (jeśli istnieje)
            if self._position.side == "LONG" and remaining_to_execute > 0:
                close_qty = min(remaining_to_execute, self._position.quantity)
                if close_qty > 0:
                    realized += (price - self._position.avg_price) * close_qty
                    gross_close = close_qty * price
                    fee_close = gross_close * self.fee_rate
                    self.balance_quote += (gross_close - fee_close)
                    total_fee_this_fill += fee_close

                    self._position.quantity -= close_qty
                    remaining_to_execute -= close_qty
                    exec_qty_this_fill += close_qty

                    if self._position.quantity <= 1e-12:
                        self._position = PositionState(side="FLAT", quantity=0.0, avg_price=0.0, unrealized_pnl=0.0)

            # 2) pozostała część SELL otwiera/rozszerza SHORT
            if remaining_to_execute > 0:
                gross_open = remaining_to_execute * price
                fee_open = gross_open * self.fee_rate
                total_fee_this_fill += fee_open

                if self._position.side == "SHORT":
                    new_qty = self._position.quantity + remaining_to_execute
                    new_gross = self._position.avg_price * self._position.quantity + gross_open
                    self._position.avg_price = new_gross / new_qty
                    self._position.quantity = new_qty
                else:
                    self._position.side = "SHORT"
                    self._position.quantity = remaining_to_execute
                    self._position.avg_price = price

                self.balance_quote += (gross_open - fee_open)
                exec_qty_this_fill += remaining_to_execute
                remaining_to_execute = 0.0

        # Jeśli w tej iteracji NIE było realnego wypełnienia – nic nie zapisujemy/nie zmieniamy statusu
        if exec_qty_this_fill <= 0.0:
            return

        # Zapis trade do DB (wolumen faktycznie zrealizowany oraz łączna opłata)
        self.db.sync.record_trade({
            "symbol": self.symbol,
            "side": st.side,
            "quantity": exec_qty_this_fill,
            "price": price,
            "fee": total_fee_this_fill,
            "order_id": st.id,
            "mode": self.mode
        })

        # Aktualizacja ordera po realnym wypełnieniu
        st.remaining -= exec_qty_this_fill
        if st.remaining <= 1e-12:
            st.remaining = 0.0
            st.status = "FILLED"
            self.db.sync.update_order_status(order_id=st.id, status="FILLED", price=price)
        else:
            st.status = "PARTIALLY_FILLED"
            self.db.sync.update_order_status(order_id=st.id, status="PARTIALLY_FILLED", price=price)

        # Zapis/aktualizacja pozycji w DB
        self._save_position()

        # Realized PnL
        if abs(realized) > 0:
            self.realized_pnl += realized

    def _mark_unrealized(self) -> None:
        if self.last_price is None:
            self._position.unrealized_pnl = 0.0
            return
        if self._position.side == "LONG":
            self._position.unrealized_pnl = (self.last_price - self._position.avg_price) * self._position.quantity
        elif self._position.side == "SHORT":
            self._position.unrealized_pnl = (self._position.avg_price - self.last_price) * self._position.quantity
        else:
            self._position.unrealized_pnl = 0.0

    def _save_position(self) -> None:
        if self._position.side == "FLAT" or self._position.quantity <= 1e-12:
            self.db.sync.close_position(self.symbol)
        else:
            self.db.sync.upsert_position({
                "symbol": self.symbol,
                "side": self._position.side,
                "quantity": self._position.quantity,
                "avg_price": self._position.avg_price,
                "unrealized_pnl": self._position.unrealized_pnl,
                "mode": self.mode
            })

    def _log_equity(self, now: dt.datetime) -> None:
        equity = self.balance_quote
        if self._position.side == "LONG" and self.last_price is not None:
            equity += self._position.quantity * self.last_price
        elif self._position.side == "SHORT" and self.last_price is not None:
            equity += (self._position.avg_price - self.last_price) * self._position.quantity
        pnl = self.realized_pnl + self._position.unrealized_pnl
        self.db.sync.log_equity({"equity": equity, "balance": self.balance_quote, "pnl": pnl, "mode": self.mode})
