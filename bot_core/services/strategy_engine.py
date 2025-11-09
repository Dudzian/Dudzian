# services/strategy_engine.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Optional, Iterable, Union

from bot_core.events import Event, EventType, EventBus

log = logging.getLogger("services.strategy_engine")


@dataclass
class StrategyConfig:
    symbol: str = "BTCUSDT"
    enabled: bool = True
    qty: float = 0.01
    max_abs_position: float = 0.05
    fast_len: int = 20
    slow_len: int = 60
    order_cooldown_sec: float = 10.0


class StrategyEngine:
    """
    MA-cross + dynamiczny update parametrów (AUTOTRADE_STATUS action=strategy_update).
    Reaguje też na trading_pause/resume (enabled w locie).
    """
    def __init__(self, bus: EventBus, cfg: StrategyConfig) -> None:
        self.bus = bus
        self.cfg = cfg
        self._last_price: Optional[float] = None
        self._fast: Optional[float] = None
        self._slow: Optional[float] = None
        self._state: Optional[int] = None
        self._last_order_ts: float = 0.0
        self._pos_qty: float = 0.0

        self.bus.subscribe(EventType.MARKET_TICK, self._on_tick)
        self.bus.subscribe(EventType.POSITION_UPDATE, self._on_pos)
        self.bus.subscribe(EventType.AUTOTRADE_STATUS, self._on_status)

    def _iter(self, x: Union[Event, Iterable[Event], None]):
        if x is None:
            return []
        if isinstance(x, Event):
            return [x]
        try:
            return list(x)
        except Exception:
            return []

    def _ewm(self, prev: Optional[float], value: float, length: int) -> float:
        alpha = 2.0 / (max(1, length) + 1.0)
        return value if prev is None else (alpha * value + (1.0 - alpha) * prev)

    def _on_pos(self, evs):
        for ev in self._iter(evs):
            p = ev.payload or {}
            if p.get("symbol") and p["symbol"] != self.cfg.symbol:
                continue
            self._pos_qty = float(p.get("qty", 0.0))

    def _on_status(self, evs):  # AUTOTRADE_STATUS
        for ev in self._iter(evs):
            p = ev.payload or {}
            if p.get("symbol") and p["symbol"] != self.cfg.symbol:
                continue
            action = str(p.get("action", "")).lower()
            comp = str(p.get("component", ""))
            if action == "trading_pause":
                self.cfg.enabled = False
            elif action == "trading_resume":
                self.cfg.enabled = True
            elif action in ("apply_preset", "strategy_update") and comp in ("WFO", "Strategy"):
                params = p.get("preset") or p.get("params") or {}
                # bezpieczna aktualizacja
                self.cfg.fast_len = int(params.get("fast_len", self.cfg.fast_len))
                self.cfg.slow_len = int(params.get("slow_len", self.cfg.slow_len))
                self.cfg.qty = float(params.get("qty", self.cfg.qty))
                self.cfg.order_cooldown_sec = float(params.get("order_cooldown_sec", self.cfg.order_cooldown_sec))
                # reset stanu średnich żeby szybciej „dogonić” zmianę
                self._fast = None
                self._slow = None
                log.info("StrategyEngine: zaktualizowano parametry: %s", self.cfg)

    def _on_tick(self, evs):
        if not self.cfg.enabled:
            return
        now = time.time()
        for ev in self._iter(evs):
            p = ev.payload or {}
            if p.get("symbol") and p["symbol"] != self.cfg.symbol:
                continue
            px = p.get("price")
            if px is None:
                continue
            px = float(px)
            self._last_price = px
            self._fast = self._ewm(self._fast, px, self.cfg.fast_len)
            self._slow = self._ewm(self._slow, px, self.cfg.slow_len)
            if self._fast is None or self._slow is None:
                continue

            bias = 1 if self._fast > self._slow else -1
            if self._state is None:
                self._state = bias
                continue

            if bias != self._state and (now - self._last_order_ts) >= self.cfg.order_cooldown_sec:
                target = self.cfg.qty * bias
                if abs(self._pos_qty + target) <= self.cfg.max_abs_position:
                    self.bus.publish(EventType.SIGNAL, {
                        "symbol": self.cfg.symbol, "side": "BUY" if bias > 0 else "SELL", "ts": now
                    })
                    self.bus.publish(EventType.ORDER_REQUEST, {
                        "symbol": self.cfg.symbol, "side": "BUY" if bias > 0 else "SELL",
                        "qty": abs(self.cfg.qty), "price": self._last_price
                    })
                    self._last_order_ts = now
                    self._state = bias
