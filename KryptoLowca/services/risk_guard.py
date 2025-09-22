# services/risk_guard.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Iterable, Union, Optional

try:
    from event_emitter_adapter import Event, EventType, EventBus
except Exception as e:
    raise ImportError(f"risk_guard: brak event_emitter_adapter ({e})")

log = logging.getLogger("services.risk_guard")


@dataclass
class RiskGuardConfig:
    symbol: str = "BTCUSDT"
    max_daily_loss_pct: float = 5.0        # pauza jeśli equity spadnie o X% od startu dnia
    max_drawdown_pct: float = 20.0         # pauza jeśli DD od szczytu >= X%
    auto_resume_cooldown_sec: float = 300  # po ilu sekundach spróbować wznowić
    publish_every_n: int = 5               # co ile PNL_UPDATE publikować status


class RiskGuard:
    """
    Strażnik ryzyka oparty o PNL_UPDATE:
    - zlicza equity od startu sesji i od szczytu,
    - jeśli spadek przekroczy progi -> publikuje AUTOTRADE_STATUS (action=trading_pause),
    - po cooldownie próbuje wznowić (action=trading_resume).
    """
    def __init__(self, bus: EventBus, cfg: RiskGuardConfig) -> None:
        self.bus = bus
        self.cfg = cfg

        self._session_start_equity: Optional[float] = None
        self._equity_peak: Optional[float] = None
        self._paused: bool = False
        self._last_pause_ts: float = 0.0
        self._since_pub: int = 0

        self.bus.subscribe(EventType.PNL_UPDATE, self._on_pnl)
        self.bus.subscribe(EventType.AUTOTRADE_STATUS, self._on_status)

    def _iter_events(self, x: Union[Event, Iterable[Event], None]):
        if x is None:
            return []
        if isinstance(x, Event):
            return [x]
        try:
            return list(x)
        except Exception:
            return []

    def _on_status(self, evs):
        # zewnętrzne pauzy/wznowienia też respektujemy
        for ev in self._iter_events(evs):
            p = ev.payload or {}
            if p.get("symbol") and p["symbol"] != self.cfg.symbol:
                continue
            action = str(p.get("action", "")).lower()
            if action == "trading_pause":
                self._paused = True
                self._last_pause_ts = time.time()
            elif action == "trading_resume":
                self._paused = False

    def _on_pnl(self, evs):
        now = time.time()
        for ev in self._iter_events(evs):
            p = ev.payload or {}
            if p.get("symbol") and p["symbol"] != self.cfg.symbol:
                continue
            equity = p.get("equity")
            if equity is None:
                continue
            equity = float(equity)
            self._since_pub += 1

            if self._session_start_equity is None:
                self._session_start_equity = equity
            if self._equity_peak is None or equity > self._equity_peak:
                self._equity_peak = equity

            # metryki
            daily_dd = 0.0
            if self._session_start_equity and self._session_start_equity > 0:
                daily_dd = (self._session_start_equity - equity) / self._session_start_equity * 100.0
            peak_dd = 0.0
            if self._equity_peak and self._equity_peak > 0:
                peak_dd = (self._equity_peak - equity) / self._equity_peak * 100.0

            # publikacja statusów co N update'ów
            if self._since_pub >= max(1, self.cfg.publish_every_n):
                self.bus.publish(EventType.AUTOTRADE_STATUS, {
                    "component": "RiskGuard",
                    "action": "status",
                    "symbol": self.cfg.symbol,
                    "daily_dd_pct": daily_dd,
                    "peak_dd_pct": peak_dd,
                    "paused": self._paused,
                    "ts": now
                })
                self._since_pub = 0

            # logika pauzy
            if (daily_dd >= self.cfg.max_daily_loss_pct) or (peak_dd >= self.cfg.max_drawdown_pct):
                if not self._paused:
                    self._paused = True
                    self._last_pause_ts = now
                    self.bus.publish(EventType.AUTOTRADE_STATUS, {
                        "component": "RiskGuard",
                        "action": "trading_pause",
                        "symbol": self.cfg.symbol,
                        "reason": "risk_breach",
                        "daily_dd_pct": daily_dd,
                        "peak_dd_pct": peak_dd,
                        "ts": now
                    })
            else:
                # jeśli jesteśmy po pauzie i minął cooldown -> wznowienie
                if self._paused and (now - self._last_pause_ts) >= max(0.0, float(self.cfg.auto_resume_cooldown_sec)):
                    self._paused = False
                    self.bus.publish(EventType.AUTOTRADE_STATUS, {
                        "component": "RiskGuard",
                        "action": "trading_resume",
                        "symbol": self.cfg.symbol,
                        "reason": "cooldown_elapsed",
                        "ts": now
                    })
