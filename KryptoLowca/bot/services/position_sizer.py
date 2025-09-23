# services/position_sizer.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Optional, Iterable, Union

try:
    from event_emitter_adapter import Event, EventType, EventBus
except Exception as e:
    raise ImportError(f"position_sizer: brak event_emitter_adapter ({e})")

log = logging.getLogger("services.position_sizer")


@dataclass
class PositionSizerConfig:
    symbol: str = "BTCUSDT"
    risk_per_trade_pct: float = 0.5        # % equity ryzykowany na transakcję (np. 0.5%)
    min_qty: float = 0.005
    max_qty: float = 0.1
    sl_atr_mult: float = 2.0               # stop-loss w * ATR
    tp_atr_mult: float = 3.0               # take-profit w * ATR
    atr_tf: str = "60s"                    # oczekiwana ramka ATR z MarketData
    publish_every_atr: bool = True         # aktualizuj strategię na każdą zmianę ATR


class PositionSizer:
    """
    Dynamiczne pozycjonowanie na bazie ATR i equity:
      qty = (equity * risk_pct) / (SL_ATR_mult * ATR)
    Publikuje AUTOTRADE_STATUS(action="strategy_update") z nowym qty oraz param. SL/TP,
    żeby StrategyEngine mógł z tego korzystać; a StopTPService ustawił progi wyjścia.
    """
    def __init__(self, bus: EventBus, cfg: PositionSizerConfig) -> None:
        self.bus = bus
        self.cfg = cfg
        self._equity: Optional[float] = None
        self._atr: Optional[float] = None
        self._paused: bool = False

        self.bus.subscribe(EventType.PNL_UPDATE, self._on_pnl)
        self.bus.subscribe(EventType.AUTOTRADE_STATUS, self._on_autotrade_status)

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

    def _on_pnl(self, evs):
        for ev in self._iter(evs):
            p = ev.payload or {}
            if p.get("symbol") and p["symbol"] != self.cfg.symbol:
                continue
            eq = p.get("equity")
            if eq is None:
                continue
            self._equity = float(eq)
        self._maybe_publish_update()

    def _on_autotrade_status(self, evs):
        for ev in self._iter(evs):
            p = ev.payload or {}
            comp = str(p.get("component", ""))
            action = str(p.get("action", "")).lower()
            if p.get("symbol") and p["symbol"] != self.cfg.symbol:
                continue

            if comp == "MarketData" and action == "atr_update":
                tf = str(p.get("tf", ""))
                if tf == self.cfg.atr_tf:
                    atr = p.get("atr")
                    if atr:
                        self._atr = float(atr)
                        if self.cfg.publish_every_atr:
                            self._maybe_publish_update()

            elif action == "trading_pause":
                self._paused = True
            elif action == "trading_resume":
                self._paused = False

    # --- core ------------------------------------------------------------------------------------

    def _maybe_publish_update(self):
        if self._paused:
            return
        if self._equity is None or self._atr is None or self._atr <= 0.0:
            return
        risk_cash = max(0.0, float(self.cfg.risk_per_trade_pct) / 100.0) * float(self._equity)
        denom = float(self.cfg.sl_atr_mult) * float(self._atr)
        if denom <= 0:
            return
        qty = risk_cash / denom
        qty = max(float(self.cfg.min_qty), min(float(self.cfg.max_qty), qty))

        self.bus.publish(EventType.AUTOTRADE_STATUS, {
            "component": "PositionSizer",
            "action": "strategy_update",
            "symbol": self.cfg.symbol,
            "params": {
                "qty": qty,
                "sl_atr_mult": float(self.cfg.sl_atr_mult),
                "tp_atr_mult": float(self.cfg.tp_atr_mult),
            },
            "equity": self._equity,
            "atr": self._atr,
            "ts": time.time()
        })
        log.info("PositionSizer: qty=%.6f (equity=%.2f, atr=%.2f)", qty, self._equity, self._atr)
