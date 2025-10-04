# trading/auto_trade_engine.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from KryptoLowca.event_emitter_adapter import EventBus, EmitterAdapter, Event, EventType
from KryptoLowca.backtest.strategy_ma import simulate_trades_ma


@dataclass
class AutoTradeConfig:
    symbol: str = "BTCUSDT"
    qty: float = 0.01
    emit_signals: bool = True
    use_close_only: bool = True
    default_params: Optional[Dict[str, int]] = None
    risk_freeze_seconds: int = 300  # po RISK_ALERT zatrzymaj wejścia na X sekund

    def __post_init__(self):
        if self.default_params is None:
            self.default_params = {"fast": 10, "slow": 50}


class AutoTradeEngine:
    def __init__(self, adapter: EmitterAdapter, broker_submit_market, cfg: Optional[AutoTradeConfig] = None) -> None:
        # Trzymamy adapter jako Any, bo stub EmitterAdapter może nie mieć wszystkich metod (np. push_autotrade_status)
        self.adapter: Any = adapter
        self.bus: EventBus = adapter.bus
        self.cfg = cfg or AutoTradeConfig()
        self._closes: List[float] = []
        self._params = dict(self.cfg.default_params or {})
        self._last_signal: Optional[int] = None  # +1 long, -1 short, 0 flat
        self._enabled: bool = True
        self._risk_frozen_until: float = 0.0

        self._submit_market = broker_submit_market

        # Handlery muszą akceptować Event | list[Event]
        self.bus.subscribe(EventType.MARKET_TICK, self._on_ticks_batch)
        self.bus.subscribe(EventType.WFO_STATUS, self._on_wfo_status_batch)
        self.bus.subscribe(EventType.RISK_ALERT, self._on_risk_alert_batch)

    # ----- Control API -----

    def enable(self) -> None:
        self._enabled = True
        self._risk_frozen_until = 0.0
        # w runtime masz tę metodę; typingowo adapter jest Any, więc mypy nie będzie protestował
        self.adapter.push_autotrade_status("enabled", detail={"symbol": self.cfg.symbol})

    def disable(self, reason: str = "") -> None:
        self._enabled = False
        self.adapter.push_autotrade_status(
            "disabled",
            detail={"symbol": self.cfg.symbol, "reason": reason},
            level="WARN",
        )

    def apply_params(self, params: Dict[str, int]) -> None:
        self._params = dict(params)
        self.adapter.push_autotrade_status("params_applied", detail={"symbol": self.cfg.symbol, "params": self._params})

    # ----- Helpers -----

    @staticmethod
    def _as_list(events: Union[Event, List[Event]]) -> List[Event]:
        return events if isinstance(events, list) else [events]

    # ----- Event Handlers -----

    def _on_wfo_status_batch(self, events: Union[Event, List[Event]]) -> None:
        for ev in self._as_list(events):
            p = ev.payload or {}
            st = p.get("status") or p.get("state") or p.get("kind")
            if st == "applied":
                params = p.get("params") or p.get("detail", {}).get("params")
                if params:
                    self.apply_params(params)
                # po WFO zdejmij risk-freeze i włącz autotrade
                self.enable()

    def _on_risk_alert_batch(self, events: Union[Event, List[Event]]) -> None:
        now = time.time()
        for ev in self._as_list(events):
            p = ev.payload or {}
            if p.get("symbol") != self.cfg.symbol:
                continue
            # zamrażamy nowe wejścia na określony czas
            self._risk_frozen_until = max(self._risk_frozen_until, now + self.cfg.risk_freeze_seconds)
            self.adapter.push_autotrade_status(
                "risk_freeze",
                detail={"symbol": self.cfg.symbol, "until": self._risk_frozen_until, "reason": p.get("kind", "risk_alert")},
                level="WARN",
            )

    def _on_ticks_batch(self, events: Union[Event, List[Event]]) -> None:
        for ev in self._as_list(events):
            p = ev.payload or {}
            if p.get("symbol") != self.cfg.symbol:
                continue
            bar = p.get("bar") or {}
            px = float(bar.get("close", 0.0))
            self._closes.append(px)
            self._maybe_trade()

    # ----- Core -----

    def _maybe_trade(self) -> None:
        closes = self._closes
        if len(closes) < max(self._params.get("fast", 10), self._params.get("slow", 50)) + 2:
            return

        # status autotrade
        if not self._enabled:
            return
        if time.time() < self._risk_frozen_until:
            # nadal zamrożone — tylko emituj status co jakiś czas?
            return

        # szybki odczyt sygnału z MA cross
        fast = int(self._params.get("fast", 10))
        slow = int(self._params.get("slow", 50))
        sig = self._last_cross_signal(closes, fast, slow)
        if sig is None:
            return

        if self.cfg.emit_signals:
            self.adapter.publish(
                EventType.SIGNAL,
                {"symbol": self.cfg.symbol, "direction": sig, "params": dict(self._params)},
            )

        if self._last_signal is None:
            self._last_signal = 0

        # Zmiana kierunku/pozycji
        if sig > 0 and self._last_signal <= 0:
            self._submit_market("buy", self.cfg.qty)
            self._last_signal = +1
            self.adapter.push_autotrade_status("entry_long", detail={"symbol": self.cfg.symbol, "qty": self.cfg.qty})
        elif sig < 0 and self._last_signal >= 0:
            self._submit_market("sell", self.cfg.qty)
            self._last_signal = -1
            self.adapter.push_autotrade_status("entry_short", detail={"symbol": self.cfg.symbol, "qty": self.cfg.qty})

    @staticmethod
    def _sma_tail(xs: List[float], n: int) -> Optional[float]:
        if len(xs) < n:
            return None
        s = sum(xs[-n:])
        return s / n

    def _last_cross_signal(self, xs: List[float], fast: int, slow: int) -> Optional[int]:
        if fast >= slow or len(xs) < slow + 2:
            return None

        f_prev = self._sma_tail(xs[:-1], fast)
        s_prev = self._sma_tail(xs[:-1], slow)
        f_now = self._sma_tail(xs, fast)
        s_now = self._sma_tail(xs, slow)

        if None in (f_prev, s_prev, f_now, s_now):
            return None

        # Pomóż mypy zawęzić typy do float:
        assert f_prev is not None and s_prev is not None and f_now is not None and s_now is not None

        cross_up = (f_now > s_now) and (f_prev <= s_prev)
        cross_dn = (f_now < s_now) and (f_prev >= s_prev)

        if cross_up:
            return +1
        if cross_dn:
            return -1
        return 0
