# services/indicators.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from dataclasses import dataclass
from collections import deque
from typing import Optional, Deque, Any

try:
    from event_emitter_adapter import EventBus, EventType
except Exception as ex:  # pragma: no cover
    EventBus = object  # type: ignore
    EventType = object  # type: ignore


def _ev(name: str):
    """Bezpieczne pobranie typu eventu z EventType, albo użycie nazwy string."""
    try:
        return getattr(EventType, name)
    except Exception:
        return name


def _bus_emit(bus: Any, event_type: Any, payload: dict) -> None:
    """
    Warstwa kompatybilności: wspiera bus.emit / bus.publish / bus.emit_event / bus.post.
    """
    for meth in ("emit", "publish", "emit_event", "post"):
        fn = getattr(bus, meth, None)
        if callable(fn):
            try:
                fn(event_type, payload)
                return
            except TypeError:
                # inna sygnatura – spróbuj dalej
                pass
    # ostatnia deska ratunku – zignoruj, ale nie psuj przepływu
    # (w praktyce nie powinniśmy tu trafić)
    return


@dataclass
class ATRConfig:
    symbol: str
    window: int = 14               # ATR z TR ~ |delta price|
    ema: bool = True               # EMA zamiast SMA
    spike_threshold_pct: float = 50.0  # % wzrostu vs baseline, żeby wysłać alert
    baseline_ema_alpha: float = 0.1    # jak szybko baseline goni ATR


class ATRMonitor:
    """
    Liczy ATR z ticków (przybliżenie: TR = abs(delta price)).
    Wysyła:
      - ATR_UPDATE: {"symbol", "atr", "last_price", "ts"}
      - ATR_SPIKE:  {"symbol", "atr", "baseline", "growth_pct", "ts"}  (jeśli growth_pct >= spike_threshold_pct)
    """
    def __init__(self, bus: EventBus, cfg: ATRConfig):
        self.bus = bus
        self.cfg = cfg
        self._last_price: Optional[float] = None
        self._atr: Optional[float] = None
        self._baseline: Optional[float] = None
        self._tr_hist: Deque[float] = deque(maxlen=max(3, cfg.window))

        # Subskrypcja ticków
        self.bus.subscribe(_ev("MARKET_TICK"), self._on_tick)

    @property
    def value(self) -> Optional[float]:
        return self._atr

    @property
    def baseline(self) -> Optional[float]:
        return self._baseline

    def _ema(self, prev: Optional[float], value: float, alpha: float) -> float:
        if prev is None:
            return value
        return prev + alpha * (value - prev)

    def _sma(self) -> Optional[float]:
        if not self._tr_hist:
            return None
        return sum(self._tr_hist) / len(self._tr_hist)

    def _on_tick(self, evt) -> None:
        if evt is None or getattr(evt, "payload", None) is None:
            return
        pld = evt.payload
        sym = pld.get("symbol")
        if sym != self.cfg.symbol:
            return
        price = float(pld.get("price", 0.0))
        ts = pld.get("ts", time.time())

        if self._last_price is not None:
            tr = abs(price - self._last_price)
            self._tr_hist.append(tr)

            if self.cfg.ema:
                # ATR EMA: atr <- atr + alpha*(tr - atr)
                alpha = 2.0 / (self.cfg.window + 1.0)
                self._atr = tr if self._atr is None else (self._atr + alpha * (tr - self._atr))
            else:
                self._atr = self._sma()

            # baseline jako wolniejsza EMA ATR
            if self._atr is not None:
                self._baseline = self._ema(self._baseline, self._atr, self.cfg.baseline_ema_alpha)

                _bus_emit(self.bus, _ev("ATR_UPDATE"), {
                    "symbol": self.cfg.symbol,
                    "atr": self._atr,
                    "last_price": price,
                    "ts": ts,
                })

                if self._baseline and self._baseline > 0:
                    growth = (self._atr - self._baseline) / self._baseline * 100.0
                    if growth >= self.cfg.spike_threshold_pct:
                        _bus_emit(self.bus, _ev("ATR_SPIKE"), {
                            "symbol": self.cfg.symbol,
                            "atr": self._atr,
                            "baseline": self._baseline,
                            "growth_pct": growth,
                            "ts": ts,
                        })

        self._last_price = price


class RollingPF:
    """Lekki licznik Profit Factor na podstawie listy wyników trade'ów."""
    def __init__(self, maxlen: int = 200):
        self.returns: Deque[float] = deque(maxlen=maxlen)

    def update(self, trade_pnl: float) -> None:
        self.returns.append(float(trade_pnl))

    @property
    def pf(self) -> Optional[float]:
        if not self.returns:
            return None
        gains = sum(x for x in self.returns if x > 0)
        losses = abs(sum(x for x in self.returns if x < 0))
        if losses == 0:
            return None if gains == 0 else float("inf")
        return gains / losses

    @property
    def expectancy(self) -> Optional[float]:
        if not self.returns:
            return None
        return sum(self.returns) / len(self.returns)
