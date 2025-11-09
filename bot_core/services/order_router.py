# services/order_router.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Optional, Union, Any, Dict, Iterable

from bot_core.events import EventBus, EmitterAdapter, EventType, Event

log = logging.getLogger("services.order_router")


@dataclass
class PaperBrokerConfig:
    symbol: str = "BTCUSDT"
    initial_cash: float = 10_000.0
    fee_bps: float = 2.0
    slippage_bps: float = 1.0
    allow_short: bool = True
    mark_to_market_on_tick: bool = True
    pnl_publish_interval_sec: float = 3.0


class PaperBroker:
    """
    Papierowy broker z obsługą pauzy (AUTOTRADE_STATUS action=trading_pause/resume).
    """
    def __init__(self, bus: EventBus, cfg: PaperBrokerConfig) -> None:
        self.bus = bus
        self.cfg = cfg

        self.cash: float = float(cfg.initial_cash)
        self.position_qty: float = 0.0
        self.avg_price: float = 0.0
        self.realized_pnl: float = 0.0
        self.last_price: Optional[float] = None
        self._last_pnl_publish = 0.0
        self._oid_seq = 0
        self._paused = False

        self.bus.subscribe(EventType.MARKET_TICK, self._on_tick)
        self.bus.subscribe(EventType.ORDER_REQUEST, self._on_order_req)
        self.bus.subscribe(EventType.AUTOTRADE_STATUS, self._on_status)

        self._publish_position()
        self._publish_pnl(force=True)

    # --- helpers ---------------------------------------------------------------------------------

    def _iter_events(self, arg: Union[Event, Iterable[Event]]) -> Iterable[Event]:
        if arg is None:
            return []
        if isinstance(arg, Event):
            return [arg]
        try:
            return list(arg)
        except Exception:
            return []

    def _gen_oid(self) -> str:
        self._oid_seq += 1
        return f"PB-{int(time.time()*1000)}-{self._oid_seq}"

    def _slip(self, side: str, px: float) -> float:
        s = float(self.cfg.slippage_bps) / 10_000.0
        return px * (1.0 + s) if side.upper().startswith("B") else px * (1.0 - s)

    def _fee(self, notional: float) -> float:
        return abs(notional) * (float(self.cfg.fee_bps) / 10_000.0)

    def _side_sign(self, side: Any) -> int:
        if isinstance(side, (int, float)):
            return 1 if side > 0 else -1
        s = str(side).upper()
        if any(k in s for k in ["BUY", "LONG"]):
            return 1
        return -1

    def _publish_position(self) -> None:
        self.bus.publish(EventType.POSITION_UPDATE, {
            "symbol": self.cfg.symbol,
            "qty": self.position_qty,
            "avg_price": self.avg_price,
            "ts": time.time()
        })

    def _publish_pnl(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_pnl_publish) < float(self.cfg.pnl_publish_interval_sec):
            return
        self._last_pnl_publish = now
        upnl = 0.0
        if self.last_price is not None and self.position_qty != 0.0:
            upnl = (self.last_price - self.avg_price) * self.position_qty
        equity = self.cash + (self.last_price or 0.0) * self.position_qty
        self.bus.publish(EventType.PNL_UPDATE, {
            "symbol": self.cfg.symbol,
            "cash": self.cash,
            "position_qty": self.position_qty,
            "avg_price": self.avg_price,
            "last_price": self.last_price,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": upnl,
            "equity": equity,
            "ts": now
        })

    # --- handlers --------------------------------------------------------------------------------

    def _on_status(self, evs):  # AUTOTRADE_STATUS (pause/resume)
        for ev in self._iter_events(evs):
            p = ev.payload or {}
            if p.get("symbol") and p["symbol"] != self.cfg.symbol:
                continue
            action = str(p.get("action", "")).lower()
            if action == "trading_pause":
                self._paused = True
                log.info("PaperBroker: trading PAUSED by %s", p.get("component"))
            elif action == "trading_resume":
                self._paused = False
                log.info("PaperBroker: trading RESUMED by %s", p.get("component"))

    def _on_tick(self, evs):
        for ev in self._iter_events(evs):
            p = ev.payload or {}
            if p.get("symbol") and p["symbol"] != self.cfg.symbol:
                continue
            px = p.get("price")
            if px is None:
                continue
            self.last_price = float(px)
        if self.cfg.mark_to_market_on_tick:
            self._publish_pnl(False)

    def _on_order_req(self, evs):
        for ev in self._iter_events(evs):
            try:
                if self._paused:
                    # zasygnalizuj do GUI/logów że odrzucamy
                    self.bus.publish(EventType.AUTOTRADE_STATUS, {
                        "component": "PaperBroker",
                        "action": "order_rejected_paused",
                        "symbol": self.cfg.symbol,
                        "ts": time.time()
                    })
                    continue

                p = (ev.payload or {}).copy()
                symbol = p.get("symbol", self.cfg.symbol)
                if symbol != self.cfg.symbol:
                    continue

                side_in = p.get("side", "SELL")
                sign = self._side_sign(side_in)
                qty = float(p.get("qty", 0.0))
                if qty <= 0.0:
                    continue

                px = p.get("price", self.last_price)
                if px is None:
                    continue

                fill = self._slip("BUY" if sign > 0 else "SELL", float(px))
                notional = qty * fill
                fee = self._fee(notional)

                # short policy
                new_pos = self.position_qty + sign * qty
                if not self.cfg.allow_short and new_pos < -1e-12:
                    self.bus.publish(EventType.AUTOTRADE_STATUS, {
                        "component": "PaperBroker",
                        "action": "order_rejected_short_not_allowed",
                        "symbol": self.cfg.symbol, "ts": time.time()
                    })
                    continue

                # PnL i stan
                realized = 0.0
                if self.position_qty == 0.0 or (self.position_qty * sign > 0):
                    total = abs(self.position_qty) + qty
                    if total > 0:
                        self.avg_price = ((abs(self.position_qty) * self.avg_price) + (qty * fill)) / total
                else:
                    closing_qty = min(abs(self.position_qty), qty)
                    pnl_part = (fill - self.avg_price) * (closing_qty * (1 if self.position_qty > 0 else -1))
                    realized += pnl_part
                    if abs(self.position_qty) < qty:
                        self.avg_price = fill

                if sign > 0:
                    self.cash -= notional + fee
                else:
                    self.cash += notional - fee

                self.position_qty = new_pos
                self.realized_pnl += realized

                oid = p.get("client_order_id") or f"PB-{int(time.time()*1000)}"
                ts = time.time()
                status = {
                    "order_id": oid, "status": "filled", "symbol": symbol,
                    "side": "BUY" if sign > 0 else "SELL", "qty": qty, "price": fill, "ts": ts
                }
                self.bus.publish(EventType.ORDER_STATUS, status)
                trade = dict(status)
                trade["realized_pnl"] = realized - fee
                self.bus.publish(EventType.TRADE_EXECUTED, trade)
                self._publish_position()
                self._publish_pnl(True)
            except Exception as e:
                log.exception("PaperBroker ORDER_REQUEST error: %s", e)

    # --- public ----------------------------------------------------------------------------------

    def place_order(self, side: Union[str, int], qty: float, price: Optional[float] = None, **kwargs):
        self._on_order_req(Event(type=EventType.ORDER_REQUEST, payload={
            "symbol": self.cfg.symbol,
            "side": side,
            "qty": float(qty),
            "price": float(price) if price is not None else self.last_price,
            **kwargs
        }))
