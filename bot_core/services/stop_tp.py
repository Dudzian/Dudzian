# services/stop_tp.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Optional, Iterable, Union

from bot_core.events import Event, EventType, EventBus

log = logging.getLogger("services.stop_tp")


@dataclass
class StopTPConfig:
    symbol: str = "BTCUSDT"
    default_sl_atr_mult: float = 2.0
    default_tp_atr_mult: float = 3.0
    cooldown_after_exit_sec: float = 5.0


class StopTPService:
    """
    SL/TP na bazie ATR i aktywnej pozycji.
    Oczekuje:
      - MarketData -> AUTOTRADE_STATUS{component='MarketData', action='atr_update', atr=...}
      - PositionSizer/Strategy -> AUTOTRADE_STATUS{action='strategy_update', params{sl_atr_mult,tp_atr_mult}}
      - PaperBroker -> POSITION_UPDATE, MARKET_TICK
    Przy naruszeniu progów wysyła ORDER_REQUEST do wyjścia z pozycji.
    """
    def __init__(self, bus: EventBus, cfg: StopTPConfig) -> None:
        self.bus = bus
        self.cfg = cfg

        self._atr: Optional[float] = None
        self._sl_mult: float = cfg.default_sl_atr_mult
        self._tp_mult: float = cfg.default_tp_atr_mult

        self._pos_qty: float = 0.0
        self._avg_price: float = 0.0
        self._last_exit_ts: float = 0.0
        self._paused: bool = False

        self.bus.subscribe(EventType.AUTOTRADE_STATUS, self._on_autotrade_status)
        self.bus.subscribe(EventType.POSITION_UPDATE, self._on_pos)
        self.bus.subscribe(EventType.MARKET_TICK, self._on_tick)

    # --- utils -----------------------------------------------------------------------------------

    def _iter(self, x):
        if x is None:
            return []
        if isinstance(x, Event):
            return [x]
        try:
            return list(x)
        except Exception:
            return []

    # --- handlers --------------------------------------------------------------------------------

    def _on_autotrade_status(self, evs):
        for ev in self._iter(evs):
            p = ev.payload or {}
            if p.get("symbol") and p["symbol"] != self.cfg.symbol:
                continue
            comp = str(p.get("component", ""))
            action = str(p.get("action", "")).lower()
            if comp == "MarketData" and action == "atr_update":
                atr = p.get("atr")
                if atr:
                    self._atr = float(atr)
            elif action in ("strategy_update", "apply_preset"):
                params = p.get("params") or p.get("preset") or {}
                if "sl_atr_mult" in params:
                    self._sl_mult = float(params["sl_atr_mult"])
                if "tp_atr_mult" in params:
                    self._tp_mult = float(params["tp_atr_mult"])
            elif action == "trading_pause":
                self._paused = True
            elif action == "trading_resume":
                self._paused = False

    def _on_pos(self, evs):
        for ev in self._iter(evs):
            p = ev.payload or {}
            if p.get("symbol") and p["symbol"] != self.cfg.symbol:
                continue
            self._pos_qty = float(p.get("qty", 0.0))
            self._avg_price = float(p.get("avg_price", self._avg_price))

    def _on_tick(self, evs):
        if self._paused or self._atr is None or self._atr <= 0.0:
            return
        now = time.time()
        if (now - self._last_exit_ts) < float(self.cfg.cooldown_after_exit_sec):
            return
        for ev in self._iter(evs):
            p = ev.payload or {}
            if p.get("symbol") and p["symbol"] != self.cfg.symbol:
                continue
            px = p.get("price")
            if px is None:
                continue
            px = float(px)

            if self._pos_qty > 0:  # long
                sl = self._avg_price - self._sl_mult * self._atr
                tp = self._avg_price + self._tp_mult * self._atr
                if px <= sl or px >= tp:
                    self._exit_position(now, px)
                    break
            elif self._pos_qty < 0:  # short
                sl = self._avg_price + self._sl_mult * self._atr
                tp = self._avg_price - self._tp_mult * self._atr
                if px >= sl or px <= tp:
                    self._exit_position(now, px)
                    break

    # --- core ------------------------------------------------------------------------------------

    def _exit_position(self, now: float, px: float):
        side = "SELL" if self._pos_qty > 0 else "BUY"
        qty = abs(self._pos_qty)
        if qty <= 0:
            return
        self.bus.publish(EventType.ORDER_REQUEST, {
            "symbol": self.cfg.symbol,
            "side": side,
            "qty": qty,
            "price": px,
            "client_order_id": f"EXIT-{int(now*1000)}"
        })
        self._last_exit_ts = now
        log.info("StopTP: exit %s qty=%.6f at %.2f (ATR=%.2f, SLx=%.2f, TPx=%.2f)",
                 side, qty, px, self._atr or -1, self._sl_mult, self._tp_mult)
