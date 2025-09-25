# services/marketdata.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import math
import logging
from dataclasses import dataclass
from typing import Optional, Iterable, Union, Deque, Dict, Any
from collections import deque

try:
    from KryptoLowca.event_emitter_adapter import Event, EventType, EventBus
except Exception as e:
    raise ImportError(f"marketdata: brak event_emitter_adapter ({e})")

log = logging.getLogger("services.marketdata")


@dataclass
class MarketDataConfig:
    symbol: str = "BTCUSDT"
    timeframe_sec: int = 60              # 1m bary z ticków
    atr_len: int = 14                    # ATR na barach
    publish_intermediate_bars: bool = False  # czy emitować aktualizacje intra-bar (nie tylko close)


class MarketDataService:
    """
    Agreguje MARKET_TICK -> bary OHLC i liczy ATR (Wilder).
    Publikuje przez AUTOTRADE_STATUS:
      - component="MarketData", action="bar_update"  (na zamknięciu bara)
      - component="MarketData", action="atr_update"  (po przeliczeniu ATR)
    Nic nie tworzy nowych EventType (kompatybilność).
    """
    def __init__(self, bus: EventBus, cfg: MarketDataConfig) -> None:
        self.bus = bus
        self.cfg = cfg

        # state bara
        self._bar_start_ts: Optional[int] = None
        self._o = self._h = self._l = self._c = None
        self._ticks_in_bar: int = 0

        # ATR state
        self._prev_close: Optional[float] = None
        self._atr: Optional[float] = None
        self._tr_hist: Deque[float] = deque(maxlen=max(2, cfg.atr_len))

        self.bus.subscribe(EventType.MARKET_TICK, self._on_tick)
        log.info("MarketDataService: %s tf=%ss, ATR len=%d", cfg.symbol, cfg.timeframe_sec, cfg.atr_len)

    # --- utils -----------------------------------------------------------------------------------

    def _iter(self, x: Union[Event, Iterable[Event], None]):
        if x is None:
            return []
        if isinstance(x, Event):
            return [x]
        try:
            return list(x)
        except Exception:
            return []

    def _bucket(self, ts: float) -> int:
        tf = max(1, int(self.cfg.timeframe_sec))
        return int(ts // tf) * tf

    def _emit_bar(self, ts_bucket_end: int, finalized: bool = True):
        if self._o is None:
            return
        bar = {
            "t_open": ts_bucket_end - int(self.cfg.timeframe_sec),
            "t_close": ts_bucket_end,
            "open": float(self._o),
            "high": float(self._h),
            "low": float(self._l),
            "close": float(self._c),
            "ticks": int(self._ticks_in_bar),
            "final": bool(finalized),
        }
        self.bus.publish(EventType.AUTOTRADE_STATUS, {
            "component": "MarketData",
            "action": "bar_update",
            "symbol": self.cfg.symbol,
            "tf": f"{int(self.cfg.timeframe_sec)}s",
            "bar": bar,
            "ts": time.time()
        })

    def _push_tr_and_update_atr(self, high: float, low: float, close: float):
        # True Range
        if self._prev_close is None:
            tr = float(high - low)
        else:
            tr = max(
                float(high - low),
                abs(float(high - self._prev_close)),
                abs(float(low - self._prev_close))
            )
        self._tr_hist.append(tr)

        if self._atr is None:
            if len(self._tr_hist) >= self.cfg.atr_len:
                self._atr = sum(self._tr_hist) / float(self.cfg.atr_len)
        else:
            n = float(self.cfg.atr_len)
            self._atr = ((self._atr * (n - 1.0)) + tr) / n

        self._prev_close = close

        if self._atr is not None:
            self.bus.publish(EventType.AUTOTRADE_STATUS, {
                "component": "MarketData",
                "action": "atr_update",
                "symbol": self.cfg.symbol,
                "tf": f"{int(self.cfg.timeframe_sec)}s",
                "atr": float(self._atr),
                "len": int(self.cfg.atr_len),
                "ts": time.time()
            })

    # --- handlers --------------------------------------------------------------------------------

    def _on_tick(self, evs):
        for ev in self._iter(evs):
            p = ev.payload or {}
            if p.get("symbol") and p["symbol"] != self.cfg.symbol:
                continue
            px = p.get("price")
            ts = p.get("ts") or time.time()
            if px is None:
                continue
            px = float(px)
            bucket = self._bucket(ts)

            # nowy bar
            if (self._bar_start_ts is None) or (bucket > self._bar_start_ts):
                # zamknij stary bar (emit)
                if self._bar_start_ts is not None:
                    self._emit_bar(self._bar_start_ts + int(self.cfg.timeframe_sec), finalized=True)
                    # ATR z zamkniętego bara
                    self._push_tr_and_update_atr(self._h, self._l, self._c)

                # start nowego
                self._bar_start_ts = bucket
                self._o = self._h = self._l = self._c = px
                self._ticks_in_bar = 1
                continue

            # aktualizacja trwającego bara
            self._c = px
            if self._h is None or px > self._h:
                self._h = px
            if self._l is None or px < self._l:
                self._l = px
            self._ticks_in_bar += 1

            if self.cfg.publish_intermediate_bars:
                self._emit_bar(self._bar_start_ts + int(self.cfg.timeframe_sec), finalized=False)
