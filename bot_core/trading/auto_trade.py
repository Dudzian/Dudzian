"""Minimalny silnik autotradingu oparty na prostym przecięciu średnich."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from bot_core.backtest.ma import simulate_trades_ma  # noqa: F401 - zachowaj kompatybilność API
from bot_core.events import DebounceRule, Event, EventBus, EventType, EmitterAdapter


@dataclass
class AutoTradeConfig:
    symbol: str = "BTCUSDT"
    qty: float = 0.01
    emit_signals: bool = True
    use_close_only: bool = True
    default_params: Dict[str, int] | None = None
    risk_freeze_seconds: int = 300

    def __post_init__(self) -> None:
        if self.default_params is None:
            self.default_params = {"fast": 10, "slow": 50}


class AutoTradeEngine:
    """Prosty kontroler autotradingu reagujący na ticki z EventBusa."""

    def __init__(
        self,
        adapter: EmitterAdapter,
        broker_submit_market,
        cfg: Optional[AutoTradeConfig] = None,
    ) -> None:
        self.adapter = adapter
        self.bus: EventBus = adapter.bus
        self.cfg = cfg or AutoTradeConfig()
        self._closes: List[float] = []
        self._params = dict(self.cfg.default_params)
        self._last_signal: Optional[int] = None
        self._enabled: bool = True
        self._risk_frozen_until: float = 0.0
        self._submit_market = broker_submit_market

        batch_rule = DebounceRule(window=0.1, max_batch=1)
        self.bus.subscribe(EventType.MARKET_TICK, self._on_ticks_batch, rule=batch_rule)
        self.bus.subscribe(EventType.WFO_STATUS, self._on_wfo_status_batch, rule=batch_rule)
        self.bus.subscribe(EventType.RISK_ALERT, self._on_risk_alert_batch, rule=batch_rule)

    def enable(self) -> None:
        self._enabled = True
        self._risk_frozen_until = 0.0
        self.adapter.push_autotrade_status("enabled", detail={"symbol": self.cfg.symbol})  # type: ignore[attr-defined]

    def disable(self, reason: str = "") -> None:
        self._enabled = False
        self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
            "disabled",
            detail={"symbol": self.cfg.symbol, "reason": reason},
            level="WARN",
        )

    def apply_params(self, params: Dict[str, int]) -> None:
        self._params = dict(params)
        self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
            "params_applied",
            detail={"symbol": self.cfg.symbol, "params": self._params},
        )

    def _on_wfo_status_batch(self, events: List[Event]) -> None:
        for ev in events:
            st = ev.payload.get("status") if ev.payload else None
            if st is None and ev.payload:
                st = ev.payload.get("state") or ev.payload.get("kind")
            if st == "applied":
                payload = ev.payload or {}
                params = payload.get("params") or payload.get("detail", {}).get("params")
                if params:
                    self.apply_params(params)
                self.enable()

    def _on_risk_alert_batch(self, events: List[Event]) -> None:
        now = time.time()
        for ev in events:
            payload = ev.payload or {}
            if payload.get("symbol") != self.cfg.symbol:
                continue
            self._risk_frozen_until = max(self._risk_frozen_until, now + self.cfg.risk_freeze_seconds)
            self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                "risk_freeze",
                detail={
                    "symbol": self.cfg.symbol,
                    "until": self._risk_frozen_until,
                    "reason": payload.get("kind", "risk_alert"),
                },
                level="WARN",
            )

    def _on_ticks_batch(self, events: List[Event]) -> None:
        for ev in events:
            payload = ev.payload or {}
            if payload.get("symbol") != self.cfg.symbol:
                continue
            bar = payload.get("bar") or {}
            px = float(bar.get("close", payload.get("price", 0.0)))
            self._closes.append(px)
            self._maybe_trade()

    def _maybe_trade(self) -> None:
        closes = self._closes
        if len(closes) < max(self._params.get("fast", 10), self._params.get("slow", 50)) + 2:
            return
        if not self._enabled:
            return
        if time.time() < self._risk_frozen_until:
            return
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
        if sig > 0 and self._last_signal <= 0:
            self._submit_market("buy", self.cfg.qty)
            self._last_signal = +1
            self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                "entry_long",
                detail={"symbol": self.cfg.symbol, "qty": self.cfg.qty},
            )
        elif sig < 0 and self._last_signal >= 0:
            self._submit_market("sell", self.cfg.qty)
            self._last_signal = -1
            self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                "entry_short",
                detail={"symbol": self.cfg.symbol, "qty": self.cfg.qty},
            )

    @staticmethod
    def _sma_tail(xs: List[float], n: int) -> Optional[float]:
        if len(xs) < n:
            return None
        return sum(xs[-n:]) / n

    def _last_cross_signal(self, xs: List[float], fast: int, slow: int) -> Optional[int]:
        if fast >= slow or len(xs) < slow + 2:
            return None
        f_prev = self._sma_tail(xs[:-1], fast)
        s_prev = self._sma_tail(xs[:-1], slow)
        f_now = self._sma_tail(xs, fast)
        s_now = self._sma_tail(xs, slow)
        if None in (f_prev, s_prev, f_now, s_now):
            return None
        cross_up = f_now > s_now and f_prev <= s_prev
        cross_dn = f_now < s_now and f_prev >= s_prev
        if cross_up:
            return +1
        if cross_dn:
            return -1
        return 0


__all__ = ["AutoTradeConfig", "AutoTradeEngine"]
