# auto_trader.py
# Walk-forward + auto-reoptimization + optional auto-trade loop, integrated via EventEmitter.
from __future__ import annotations

import threading
import time
import statistics
from typing import Optional, List, Dict, Any, Callable
try:
    from collections import deque
except Exception:
    # minimal fallback
    class deque(list):
        def __init__(self, maxlen=None): super().__init__(); self.maxlen=maxlen
        def append(self, x):
            super().append(x)
            if self.maxlen and len(self) > self.maxlen:
                del self[0]
        def popleft(self): return super().pop(0)

from event_emitter_adapter import EventEmitter

class AutoTrader:
    """
    - Listens to trade_closed events to compute rolling PF & Expectancy
    - Monitors ATR (if 'bar' events are emitted)
    - Triggers reoptimization when thresholds break
    - Optional walk-forward scheduler (time-based)
    - Optional auto-trade loop (asks AI for decision and executes via GUI bridge)
    - Runtime-reconfigurable via ControlPanel (configure()/set_enable_auto_trade()).
    """
    def __init__(
        self,
        emitter: EventEmitter,
        gui,
        symbol_getter: Callable[[], str],
        pf_min: float = 1.3,
        expectancy_min: float = 0.0,
        metrics_window: int = 30,
        atr_ratio_threshold: float = 0.5,   # +50% vs baseline
        atr_baseline_len: int = 100,
        reopt_cooldown_s: int = 1800,       # 30 min cooldown
        walkforward_interval_s: Optional[int] = 3600,  # every 1h
        walkforward_min_closed_trades: int = 10,
        enable_auto_trade: bool = True,
        auto_trade_interval_s: int = 30
    ) -> None:
        self.emitter = emitter
        self.gui = gui
        self.symbol_getter = symbol_getter

        self.pf_min = pf_min
        self.expectancy_min = expectancy_min
        self.metrics_window = metrics_window

        self.atr_ratio_threshold = atr_ratio_threshold
        self.atr_baseline_len = atr_baseline_len

        self.reopt_cooldown_s = reopt_cooldown_s
        self.last_reopt_ts = 0.0

        self.walkforward_interval_s = walkforward_interval_s
        self.walkforward_min_closed_trades = walkforward_min_closed_trades

        self.enable_auto_trade = enable_auto_trade
        self.auto_trade_interval_s = auto_trade_interval_s

        self._closed_pnls: deque = deque(maxlen=max(10, metrics_window))
        self._atr_values: deque = deque(maxlen=max(50, atr_baseline_len*2))
        self._atr_baseline: Optional[float] = None

        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._lock = threading.RLock()

        # Subscribe to events
        emitter.on("trade_closed", self._on_trade_closed, tag="autotrader")
        emitter.on("bar", self._on_bar, tag="autotrader")

    # -- Public API --
    def start(self) -> None:
        self._stop.clear()
        # Walk-forward loop only if configured
        if self.walkforward_interval_s:
            t = threading.Thread(target=self._walkforward_loop, daemon=True)
            t.start()
            self._threads.append(t)
        # Auto-trade loop ALWAYS started; respects enable_auto_trade flag at runtime
        t2 = threading.Thread(target=self._auto_trade_loop, daemon=True)
        t2.start()
        self._threads.append(t2)
        self.emitter.log("AutoTrader started.", component="AutoTrader")

    def stop(self) -> None:
        self._stop.set()
        self.emitter.off("trade_closed", tag="autotrader")
        self.emitter.off("bar", tag="autotrader")
        self.emitter.log("AutoTrader stopped.", component="AutoTrader")

    def set_enable_auto_trade(self, flag: bool) -> None:
        with self._lock:
            self.enable_auto_trade = bool(flag)
        self.emitter.log(f"Auto-Trade {'ENABLED' if flag else 'DISABLED'}.", component="AutoTrader")

    def configure(self, **kwargs: Any) -> None:
        """Runtime reconfiguration from the ControlPanel."""
        with self._lock:
            for key, val in kwargs.items():
                if not hasattr(self, key):
                    continue
                setattr(self, key, val)
        self.emitter.log(f"AutoTrader reconfigured: {kwargs}", component="AutoTrader")

    # -- Event handlers --
    def _on_trade_closed(self, symbol: str, side: str, entry: float, exit: float, pnl: float, ts: float, meta: Dict[str, Any] | None = None, **_) -> None:
        self._closed_pnls.append(pnl)
        pf, exp = self._compute_metrics()
        self.emitter.emit("metrics_updated", pf=pf, expectancy=exp, window=len(self._closed_pnls), ts=time.time())
        # Check thresholds
        trigger_reason = None
        details = {}
        if pf is not None and pf < self.pf_min:
            trigger_reason = "pf_drop"
            details["pf"] = pf
        if exp is not None and exp < self.expectancy_min:
            trigger_reason = (trigger_reason + "+expectancy_drop") if trigger_reason else "expectancy_drop"
            details["expectancy"] = exp
        if trigger_reason:
            self._maybe_reoptimize(trigger_reason, details)

    def _on_bar(self, symbol: str, o: float, h: float, l: float, c: float, ts: float, **_) -> None:
        # TR approximation (if no prev close given we use current bar-only proxy)
        tr = max(h - l, abs(h - c), abs(l - c))
        self._atr_values.append(tr)
        # Compute ATR using simple moving average of TRs
        if len(self._atr_values) >= max(14, self.atr_baseline_len):
            atr = sum(list(self._atr_values)[-14:]) / 14.0
            if self._atr_baseline is None and len(self._atr_values) >= self.atr_baseline_len:
                self._atr_baseline = sum(list(self._atr_values)[:self.atr_baseline_len]) / float(self.atr_baseline_len)
            baseline = self._atr_baseline or atr
            ratio = (atr - baseline) / baseline if baseline > 0 else 0.0
            self.emitter.emit("atr_updated", atr=atr, baseline=baseline, ratio=ratio, ts=ts)
            if self._atr_baseline and ratio >= self.atr_ratio_threshold:
                self._maybe_reoptimize("atr_spike", {"atr": atr, "baseline": baseline, "ratio": ratio})

    # -- Helpers --
    def _compute_metrics(self) -> tuple[Optional[float], Optional[float]]:
        if not self._closed_pnls:
            return (None, None)
        pnls = list(self._closed_pnls)
        wins = [p for p in pnls if p > 0]
        losses = [-p for p in pnls if p < 0]
        gross_profit = sum(wins)
        gross_loss = sum(losses)
        pf = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else None)
        # Expectancy per trade (avg pnl)
        expectancy = statistics.mean(pnls) if pnls else None
        return (pf, expectancy)

    def _maybe_reoptimize(self, reason: str, details: Dict[str, Any]) -> None:
        now = time.time()
        if now - self.last_reopt_ts < self.reopt_cooldown_s:
            self.emitter.log(f"Reopt skipped (cooldown): {reason} {details}", level="DEBUG", component="AutoTrader")
            return
        self.last_reopt_ts = now
        self.emitter.emit("reopt_triggered", reason=reason, details=details, ts=now)
        # Try to call AI retrain if available
        ai = getattr(self.gui, "ai_mgr", None)
        if ai is not None and hasattr(ai, "train"):
            try:
                # non-blocking retrain in thread
                threading.Thread(target=self._call_train_safe, args=(ai,), daemon=True).start()
            except Exception as e:
                self.emitter.log(f"AI retrain failed to start: {e!r}", level="ERROR", component="AutoTrader")
        else:
            self.emitter.log("AI manager not available; reopt event emitted only.", level="WARNING", component="AutoTrader")

    def _call_train_safe(self, ai) -> None:
        try:
            self.emitter.log("AI retrain started...", component="AutoTrader")
            ai.train()
            self.emitter.log("AI retrain finished.", component="AutoTrader")
        except Exception as e:
            self.emitter.log(f"AI retrain error: {e!r}", level="ERROR", component="AutoTrader")

    # -- Loops --
    def _walkforward_loop(self) -> None:
        last_ts = 0.0
        while not self._stop.is_set():
            try:
                now = time.time()
                if last_ts == 0.0:
                    last_ts = now
                wf = self.walkforward_interval_s or 0
                if wf > 0 and (now - last_ts) >= wf:
                    # Optional guard: only if we have enough closed trades since last step
                    if len(self._closed_pnls) >= self.walkforward_min_closed_trades:
                        self._maybe_reoptimize("walk_forward_tick", {"closed_trades": len(self._closed_pnls)})
                    last_ts = now
            except Exception as e:
                self.emitter.log(f"Walk-forward loop error: {e!r}", level="ERROR", component="AutoTrader")
            self._stop.wait(1.0)

    def _auto_trade_loop(self) -> None:
        while not self._stop.is_set():
            try:
                if not self.enable_auto_trade:
                    self._stop.wait(self.auto_trade_interval_s)
                    continue
                symbol = self.symbol_getter()
                if not symbol:
                    self._stop.wait(self.auto_trade_interval_s)
                    continue
                # Ask AI for direction if available
                side = None
                ai = getattr(self.gui, "ai_mgr", None)
                if ai is not None and hasattr(ai, "predict_series"):
                    try:
                        series = ai.predict_series(symbol=symbol, bars=128)
                        if hasattr(series, "__getitem__"):
                            last = float(series[-1])
                            side = "BUY" if last >= 0 else "SELL"
                    except Exception as e:
                        self.emitter.log(f"predict_series failed: {e!r}", level="ERROR", component="AutoTrader")
                if side is None:
                    side = "BUY" if int(time.time()/max(1,self.auto_trade_interval_s)) % 2 == 0 else "SELL"
                # Market price via ex_mgr if available
                mkt_price = float('nan')
                ex = getattr(self.gui, "ex_mgr", None)
                if ex is not None:
                    try:
                        if hasattr(ex, "fetch_ticker"):
                            tick = ex.fetch_ticker(symbol)
                            if isinstance(tick, dict) and "last" in tick:
                                mkt_price = float(tick["last"])
                    except Exception as e:
                        self.emitter.log(f"fetch_ticker failed: {e!r}", level="ERROR", component="AutoTrader")
                # Execute via GUI bridge
                if hasattr(self.gui, "_bridge_execute_trade"):
                    self.gui._bridge_execute_trade(symbol, side, mkt_price)
                    self.emitter.emit("auto_trade_tick", symbol=symbol, ts=time.time())
                else:
                    self.emitter.log("_bridge_execute_trade missing on GUI", level="ERROR", component="AutoTrader")
            except Exception as e:
                self.emitter.log(f"Auto trade tick error: {e!r}", level="ERROR", component="AutoTrader")
            self._stop.wait(self.auto_trade_interval_s)
