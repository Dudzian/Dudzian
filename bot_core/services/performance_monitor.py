# services/performance_monitor.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from collections import deque
from typing import Deque, Iterable, Union

from bot_core.events import Event, EventType, EventBus

log = logging.getLogger("services.performance_monitor")


@dataclass
class PerfMonitorConfig:
    symbol: str = "BTCUSDT"
    window_trades: int = 100
    min_trades_to_eval: int = 20
    pf_min: float = 1.1         # Profit Factor próg
    exp_min: float = 0.0        # Expectancy próg (na transakcję)
    consecutive_breaches: int = 3   # ile razy z rzędu poniżej progów -> trigger
    publish_every_n_trades: int = 5  # co ile transakcji publikować status


class PerformanceMonitor:
    """
    Zbiera transakcje i liczy metryki okna: Profit Factor, Expectancy.
    Gdy przez N kolejnych ewaluacji metryki są poniżej progów -> emituje WFO_TRIGGER.
    """
    def __init__(self, bus: EventBus, cfg: PerfMonitorConfig) -> None:
        self.bus = bus
        self.cfg = cfg
        self.trades: Deque[float] = deque(maxlen=cfg.window_trades)
        self._breach_streak = 0
        self._since_publish = 0

        self.bus.subscribe(EventType.TRADE_EXECUTED, self._on_trades)

    def _iter_events(self, arg: Union[Event, Iterable[Event]]) -> Iterable[Event]:
        if arg is None:
            return []
        if isinstance(arg, Event):
            return [arg]
        try:
            return list(arg)
        except Exception:
            return []

    @staticmethod
    def _profit_factor(pl: Iterable[float]) -> float:
        gp = sum(x for x in pl if x > 0)
        gl = sum(-x for x in pl if x < 0)
        if gl <= 1e-12:
            return float("inf") if gp > 0 else 1.0
        return gp / gl

    @staticmethod
    def _expectancy(pl: Iterable[float]) -> float:
        pl = list(pl)
        if not pl:
            return 0.0
        return sum(pl) / len(pl)

    def _on_trades(self, ev_or_list: Union[Event, Iterable[Event]]) -> None:
        for ev in self._iter_events(ev_or_list):
            p = ev.payload or {}
            if p.get("symbol") and p["symbol"] != self.cfg.symbol:
                continue
            # preferujemy realized_pnl; jeśli brak, spróbuj wyliczyć z side/qty/price/avg itp. – ale to już robi broker
            pnl = float(p.get("realized_pnl", 0.0))
            self.trades.append(pnl)
            self._since_publish += 1

            if len(self.trades) >= self.cfg.min_trades_to_eval:
                pf = self._profit_factor(self.trades)
                exp = self._expectancy(self.trades)

                breach = (pf < self.cfg.pf_min) or (exp < self.cfg.exp_min)
                self._breach_streak = self._breach_streak + 1 if breach else 0

                if self._since_publish >= self.cfg.publish_every_n_trades:
                    self.bus.publish(EventType.AUTOTRADE_STATUS, {
                        "component": "PerformanceMonitor",
                        "symbol": self.cfg.symbol,
                        "pf": pf,
                        "expectancy": exp,
                        "breach_streak": self._breach_streak,
                        "ts": time.time()
                    })
                    self._since_publish = 0

                if self._breach_streak >= self.cfg.consecutive_breaches:
                    self.bus.publish(EventType.WFO_TRIGGER, {
                        "reason": "performance_drop",
                        "symbol": self.cfg.symbol,
                        "pf": pf,
                        "expectancy": exp,
                        "ts": time.time()
                    })
                    # po triggerze zerujemy streak (i czekamy na WFO/cooldown)
                    self._breach_streak = 0
