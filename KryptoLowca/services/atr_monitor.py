# services/atr_monitor.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from KryptoLowca.event_emitter_adapter import EventBus, EmitterAdapter, Event, EventType


@dataclass
class ATRMonitorConfig:
    atr_period: int = 14
    growth_threshold_pct: float = 40.0   # % wzrostu ATR vs baseline
    min_bars_after_baseline: int = 50    # minimum barów po resecie baseline ATR
    symbol: str = "BTCUSDT"


class ATRMonitor:
    """
    - Subskrybuje MARKET_TICK (payload: {'symbol','bar':{'high','low','close',...}})
    - Liczy ATR, zarządza baseline (reset po WFO applied/completed).
    - Gdy ATR rośnie > threshold, publikuje RISK_ALERT (kind='atr_spike').
    - Przechowuje bufor barów do odczytu przez WFO (data_provider).
    """
    def __init__(self, adapter: EmitterAdapter, cfg: Optional[ATRMonitorConfig] = None) -> None:
        self.adapter = adapter
        self.bus: EventBus = adapter.bus
        self.cfg = cfg or ATRMonitorConfig()
        self._bars: Deque[Dict[str, float]] = deque(maxlen=20000)
        self._atr_values: Deque[float] = deque(maxlen=20000)
        self._baseline_atr: Optional[float] = None
        self._bars_since_baseline: int = 0
        self._last_close: Optional[float] = None

        # Subskrypcje
        self.bus.subscribe(EventType.MARKET_TICK, self._on_ticks_batch)
        self.bus.subscribe(EventType.WFO_STATUS, self._on_wfo_status_batch)

    # --------- Public helpers ---------

    def get_bars(self) -> List[Dict[str, float]]:
        return list(self._bars)

    def get_last_price(self) -> Optional[float]:
        return self._last_close

    def get_atr(self) -> Optional[float]:
        if not self._atr_values:
            return None
        return self._atr_values[-1]

    # --------- Internals ---------

    def _on_wfo_status_batch(self, events: List[Event]) -> None:
        # Reset baseline po applied/completed (możesz zmienić wg preferencji)
        for ev in events:
            st = ev.payload.get("status") or ev.payload.get("state") or ev.payload.get("kind")
            if st in ("applied", "completed"):
                cur_atr = self.get_atr()
                if cur_atr and cur_atr == cur_atr:
                    self._baseline_atr = cur_atr
                    self._bars_since_baseline = 0
                    self.adapter.push_log(f"ATR baseline reset to {cur_atr:.6f} (WFO {st}).", level="INFO")

    def _on_ticks_batch(self, events: List[Event]) -> None:
        for ev in events:
            p = ev.payload
            if not p:
                continue
            sym = p.get("symbol") or self.cfg.symbol
            if sym != self.cfg.symbol:
                continue
            bar = p.get("bar") or {}
            high = float(bar.get("high", bar.get("close", 0.0)))
            low = float(bar.get("low", bar.get("close", 0.0)))
            close = float(bar.get("close", 0.0))
            ts = bar.get("ts")

            # Update bars buffer
            self._bars.append({"ts": ts, "high": high, "low": low, "close": close})
            self._last_close = close

            # Update ATR
            self._update_atr(high=high, low=low, close=close)

            # Risk alert?
            self._maybe_alert()

    def _update_atr(self, high: float, low: float, close: float) -> None:
        prev_close = self._last_close if self._last_close is not None else close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        if not self._atr_values:
            atr = tr
        else:
            # klasyczne Wilder EMA
            n = self.cfg.atr_period
            prev_atr = self._atr_values[-1]
            atr = (prev_atr * (n - 1) + tr) / n
        self._atr_values.append(atr)
        if self._baseline_atr is None and len(self._atr_values) >= self.cfg.atr_period:
            self._baseline_atr = atr
            self._bars_since_baseline = 0
        else:
            self._bars_since_baseline += 1

    def _maybe_alert(self) -> None:
        if self._baseline_atr is None:
            return
        if self._bars_since_baseline < self.cfg.min_bars_after_baseline:
            return
        cur = self._atr_values[-1] if self._atr_values else None
        if not cur:
            return
        growth = 0.0 if self._baseline_atr == 0 else (cur / self._baseline_atr - 1.0) * 100.0
        if growth >= self.cfg.growth_threshold_pct:
            self.adapter.publish(
                EventType.RISK_ALERT,
                {
                    "symbol": self.cfg.symbol,
                    "kind": "atr_spike",
                    "atr_current": cur,
                    "atr_baseline": self._baseline_atr,
                    "growth_pct": growth,
                    "bars_since_baseline": self._bars_since_baseline,
                },
            )
            # po alercie nie resetujemy baseline — decyzja po Twojej stronie (WFO może go zresetować po applied)
