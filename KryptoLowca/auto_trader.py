# auto_trader.py
# Walk-forward + auto-reoptimization + optional auto-trade loop, integrated via EventEmitter.
from __future__ import annotations

import threading
import time
import statistics
from typing import Optional, List, Dict, Any, Callable, Tuple

import inspect

import pandas as pd
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

from KryptoLowca.event_emitter_adapter import EventEmitter

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
        self._threads = []
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
        for t in list(self._threads):
            try:
                if t.is_alive():
                    t.join(timeout=2.0)
            except Exception:
                pass
        self._threads.clear()

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
                timeframe = "1m"
                tf_var = getattr(self.gui, "timeframe_var", None)
                if tf_var is not None and hasattr(tf_var, "get"):
                    try:
                        timeframe = tf_var.get() or timeframe
                    except Exception:
                        pass

                ex = getattr(self.gui, "ex_mgr", None)
                ai = getattr(self.gui, "ai_mgr", None)

                df: Optional[pd.DataFrame] = None
                last_price: Optional[float] = None
                last_pred: Optional[float] = None

                if ai is not None and hasattr(ai, "predict_series"):
                    try:
                        last_pred, df, last_price = self._obtain_prediction(ai, symbol, timeframe, ex)
                    except Exception as e:
                        self.emitter.log(
                            f"predict_series failed: {e!r}", level="ERROR", component="AutoTrader"
                        )

                if last_price is None:
                    last_price = self._resolve_market_price(symbol, ex, df)

                side: Optional[str] = None
                if last_pred is not None and ai is not None:
                    threshold_bps = float(getattr(ai, "ai_threshold_bps", 5.0))
                    threshold = threshold_bps / 10_000.0
                    if last_pred >= threshold:
                        side = "BUY"
                    elif last_pred <= -threshold:
                        side = "SELL"
                elif last_pred is not None:
                    # brak ai_threshold -> użyj domyślnego progu
                    threshold = 5.0 / 10_000.0
                    if last_pred >= threshold:
                        side = "BUY"
                    elif last_pred <= -threshold:
                        side = "SELL"

                if side is None and last_price is not None:
                    # fallback – prosty przełącznik BUY/SELL aby utrzymać aktywność w trybie demo
                    side = "BUY" if int(time.time() / max(1, self.auto_trade_interval_s)) % 2 == 0 else "SELL"

                if side is None or last_price is None:
                    self._stop.wait(self.auto_trade_interval_s)
                    continue

                if hasattr(self.gui, "_bridge_execute_trade"):
                    self.gui._bridge_execute_trade(symbol, side.lower(), float(last_price))
                    self.emitter.emit("auto_trade_tick", symbol=symbol, ts=time.time())
                else:
                    self.emitter.log(
                        "_bridge_execute_trade missing on GUI",
                        level="ERROR",
                        component="AutoTrader",
                    )
            except Exception as e:
                self.emitter.log(f"Auto trade tick error: {e!r}", level="ERROR", component="AutoTrader")
            self._stop.wait(self.auto_trade_interval_s)

    # --- Prediction helpers ---
    def _obtain_prediction(
        self,
        ai: Any,
        symbol: str,
        timeframe: str,
        ex: Any,
    ) -> Tuple[Optional[float], Optional[pd.DataFrame], Optional[float]]:
        predict_fn = getattr(ai, "predict_series", None)
        if not callable(predict_fn):
            return (None, None, None)

        df: Optional[pd.DataFrame] = None
        last_price: Optional[float] = None
        last_pred: Optional[float] = None

        try:
            sig = inspect.signature(predict_fn)
            params = sig.parameters
        except (TypeError, ValueError):
            sig = None
            params = {}

        # 1) Spróbuj wywołania na podstawie symbolu/bars – zgodność z prostym API
        if "symbol" in params and last_pred is None:
            kwargs: Dict[str, Any] = {"symbol": symbol}
            if "timeframe" in params:
                kwargs["timeframe"] = timeframe
            if "bars" in params:
                kwargs["bars"] = 256
            elif "limit" in params:
                kwargs["limit"] = 256
            try:
                preds = predict_fn(**kwargs)
                last_pred = self._extract_last_pred(preds)
            except Exception:
                last_pred = None

        # 2) Klasyczny wariant – przekazanie DataFrame z OHLCV
        if last_pred is None:
            df = self._ensure_dataframe(symbol, timeframe, ex)
            if df is not None and not df.empty:
                last_price = float(df["close"].iloc[-1])
                call_attempts = []
                if sig is None or "feature_cols" in params:
                    call_attempts.append({"feature_cols": ["open", "high", "low", "close", "volume"]})
                call_attempts.append({})  # bez dodatkowych argumentów
                for extra in call_attempts:
                    try:
                        preds = predict_fn(df, **extra)
                        last_pred = self._extract_last_pred(preds)
                    except TypeError:
                        # jeśli feature_cols niepasuje – spróbuj kolejnego wariantu
                        continue
                    except Exception:
                        continue
                    if last_pred is not None:
                        break

        if last_price is None:
            last_price = self._resolve_market_price(symbol, ex, df)

        return (last_pred, df, last_price)

    def _ensure_dataframe(
        self, symbol: str, timeframe: str, ex: Any
    ) -> Optional[pd.DataFrame]:
        if ex is None or not hasattr(ex, "fetch_ohlcv"):
            return None
        try:
            raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=256) or []
        except Exception as exc:
            self.emitter.log(
                f"fetch_ohlcv failed: {exc!r}", level="ERROR", component="AutoTrader"
            )
            return None
        if not raw:
            return None
        try:
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        except Exception:
            df = pd.DataFrame(raw)
            expected = ["timestamp", "open", "high", "low", "close", "volume"]
            for idx, col in enumerate(df.columns):
                if idx < len(expected):
                    df.rename(columns={col: expected[idx]}, inplace=True)
        return df

    def _resolve_market_price(
        self, symbol: str, ex: Any, df: Optional[pd.DataFrame]
    ) -> Optional[float]:
        if df is not None and not df.empty and "close" in df.columns:
            try:
                return float(df["close"].iloc[-1])
            except Exception:
                pass
        if ex is None:
            return None
        if hasattr(ex, "fetch_ticker"):
            try:
                ticker = ex.fetch_ticker(symbol) or {}
                for key in ("last", "close", "bid", "ask"):
                    val = ticker.get(key)
                    if val is not None:
                        return float(val)
            except Exception as exc:
                self.emitter.log(
                    f"fetch_ticker failed: {exc!r}", level="ERROR", component="AutoTrader"
                )
        return None

    @staticmethod
    def _extract_last_pred(preds: Any) -> Optional[float]:
        if preds is None:
            return None
        series: Optional[pd.Series]
        if isinstance(preds, pd.Series):
            series = preds
        elif isinstance(preds, pd.DataFrame):
            if preds.empty:
                return None
            series = preds.iloc[:, -1]
        else:
            try:
                series = pd.Series(list(preds))
            except Exception:
                return None
        if series is None:
            return None
        series = series.dropna()
        if series.empty:
            return None
        try:
            return float(series.iloc[-1])
        except Exception:
            return None
