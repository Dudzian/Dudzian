# services/risk_manager.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Optional, Iterable, Union, Deque
from collections import deque

try:
    from event_emitter_adapter import Event, EventType, EventBus
except Exception as e:
    raise ImportError(f"risk_manager: brak event_emitter_adapter ({e})")

log = logging.getLogger("services.risk_manager")


@dataclass
class RiskConfig:
    symbol: str = "BTCUSDT"
    atr_lookback: int = 100
    atr_ewm_alpha: float = 2.0 / (100 + 1.0)
    spike_threshold_pct: float = 50.0     # spike jeśli ATR > baseline*(1+X%)
    min_bars_for_baseline: int = 150
    publish_every_n: int = 10             # co ile barów publikować ATR_UPDATE


class RiskManager:
    """
    Lekki monitor ryzyka:
    - liczy „ATR-proxy” jako EWMA z |ret|,
    - utrzymuje baseline (EWMA długie),
    - wykrywa spike i publikuje ATR_SPIKE,
    - regularnie publikuje ATR_UPDATE.
    """
    def __init__(self, bus: EventBus, cfg: RiskConfig) -> None:
        self.bus = bus
        self.cfg = cfg

        self._last_price: Optional[float] = None
        self._atr_ewm: Optional[float] = None
        self._baseline_ewm: Optional[float] = None
        self._count: int = 0

        self.bus.subscribe(EventType.MARKET_TICK, self._on_tick)

    def _iter_events(self, arg: Union[Event, Iterable[Event]]) -> Iterable[Event]:
        if arg is None:
            return []
        if isinstance(arg, Event):
            return [arg]
        try:
            return list(arg)
        except Exception:
            return []

    def _ewm(self, prev: Optional[float], value: float, alpha: float) -> float:
        return (value if prev is None else (alpha * value + (1.0 - alpha) * prev))

    def _on_tick(self, ev_or_list: Union[Event, Iterable[Event]]) -> None:
        published = False
        for ev in self._iter_events(ev_or_list):
            p = ev.payload or {}
            if p.get("symbol") and p["symbol"] != self.cfg.symbol:
                continue
            px = p.get("price")
            if px is None:
                continue
            px = float(px)
            if self._last_price is not None and px > 0:
                ret = abs(px / self._last_price - 1.0)
                # krótka EWMA (ATR-proxy)
                self._atr_ewm = self._ewm(self._atr_ewm, ret, self.cfg.atr_ewm_alpha)
                # długa baseline (2x dłuższe okno heurystycznie)
                baseline_alpha = self.cfg.atr_ewm_alpha / 2.0
                self._baseline_ewm = self._ewm(self._baseline_ewm, ret, baseline_alpha)
                self._count += 1

                if self._atr_ewm is not None and self._baseline_ewm is not None:
                    if self._count >= self.cfg.min_bars_for_baseline:
                        thr = self._baseline_ewm * (1.0 + self.cfg.spike_threshold_pct / 100.0)
                        if self._atr_ewm > thr:
                            self.bus.publish(EventType.ATR_SPIKE, {
                                "symbol": self.cfg.symbol,
                                "atr": self._atr_ewm,
                                "baseline": self._baseline_ewm,
                                "threshold": thr,
                                "ts": time.time()
                            })
                    if self._count % max(1, self.cfg.publish_every_n) == 0:
                        self.bus.publish(EventType.ATR_UPDATE, {
                            "symbol": self.cfg.symbol,
                            "atr": self._atr_ewm,
                            "baseline": self._baseline_ewm,
                            "ts": time.time()
                        })
                        published = True
            self._last_price = px
        if not published and self._atr_ewm is not None:
            # delikatnie ograniczamy flood logów – publikuj tylko co N, więc nic tutaj
            pass
