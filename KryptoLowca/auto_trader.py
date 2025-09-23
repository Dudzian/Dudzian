# auto_trader.py
# Walk-forward + auto-reoptimization + optional auto-trade loop, integrated via EventEmitter.
from __future__ import annotations

import threading
import time
import statistics
from typing import Optional, List, Dict, Any, Callable, TYPE_CHECKING

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

from event_emitter_adapter import EventEmitter

if TYPE_CHECKING:  # pragma: no cover - tylko na potrzeby typów
    from services.execution_service import ExecutionService
    from managers.exchange_manager import ExchangeManager
    from managers.ai_manager import AIManager
    from managers.risk_manager_adapter import RiskManager
    from managers.exchange_core import OrderDTO

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
        auto_trade_interval_s: int = 30,
        *,
        execution_service: "ExecutionService | None" = None,
        exchange_manager: "ExchangeManager | None" = None,
        ai_manager: "AIManager | None" = None,
        risk_manager: "RiskManager | None" = None,
        portfolio_getter: Optional[Callable[[], Dict[str, Any]]] = None,
        cash_asset: str = "USDT",
        fallback_fraction: float = 0.1,
        slippage_bps: float = 5.0,
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

        self.execution_service: "ExecutionService | None" = execution_service
        self.exchange_manager: "ExchangeManager | None" = exchange_manager
        self.ai_manager: "AIManager | None" = ai_manager
        self.risk_manager: "RiskManager | None" = risk_manager
        self.portfolio_getter = portfolio_getter
        self.cash_asset = str(cash_asset or "USDT").upper()
        try:
            self.fallback_fraction = max(0.0, float(fallback_fraction))
        except Exception:
            self.fallback_fraction = 0.1
        try:
            self.slippage_bps = max(0.0, float(slippage_bps))
        except Exception:
            self.slippage_bps = 5.0

        self._last_symbol: Optional[str] = None

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
                if key == "cash_asset":
                    self.cash_asset = str(val or self.cash_asset).upper()
                    continue
                if key == "fallback_fraction":
                    try:
                        self.fallback_fraction = max(0.0, float(val))
                    except Exception:
                        pass
                    continue
                if key == "slippage_bps":
                    try:
                        self.slippage_bps = max(0.0, float(val))
                    except Exception:
                        pass
                    continue
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
                executed = self._auto_trade_once()
                if executed:
                    self.emitter.emit("auto_trade_tick", symbol=self._last_symbol or "", ts=time.time())
            except Exception as e:
                self.emitter.log(f"Auto trade tick error: {e!r}", level="ERROR", component="AutoTrader")
            self._stop.wait(self.auto_trade_interval_s)

    def _auto_trade_once(self) -> bool:
        if not self.enable_auto_trade:
            return False

        symbol = self.symbol_getter()
        if not symbol:
            return False

        self._last_symbol = symbol

        ex = self._resolve_exchange_manager()
        if ex is None or not hasattr(ex, "fetch_ohlcv"):
            return False

        timeframe = "1m"
        tf_var = getattr(self.gui, "timeframe_var", None)
        if tf_var is not None and hasattr(tf_var, "get"):
            try:
                timeframe = tf_var.get() or timeframe
            except Exception:
                pass

        try:
            raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=256) or []
        except Exception as e:
            self.emitter.log(f"fetch_ohlcv failed: {e!r}", level="ERROR", component="AutoTrader")
            return False

        if not raw:
            return False

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        ai = self._resolve_ai_manager()
        if ai is None or not hasattr(ai, "predict_series"):
            return False

        try:
            preds = ai.predict_series(df, feature_cols=["open", "high", "low", "close", "volume"])
        except Exception as e:
            self.emitter.log(f"predict_series failed: {e!r}", level="ERROR", component="AutoTrader")
            preds = None

        if preds is None or len(preds.dropna()) == 0:
            return False

        series = preds.dropna()
        last_pred = float(series.iloc[-1])
        threshold_bps = float(getattr(ai, "ai_threshold_bps", 5.0))
        threshold = threshold_bps / 10_000.0

        if last_pred >= threshold:
            side = "BUY"
        elif last_pred <= -threshold:
            side = "SELL"
        else:
            return False

        order = self._execute_trade(symbol, side, df)
        return order is not None

    # -- Dependency helpers --
    def _resolve_execution_service(self) -> "ExecutionService | None":
        if self.execution_service is not None:
            return self.execution_service
        if self.gui is not None:
            svc = getattr(self.gui, "exec_service", None)
            if svc is not None:
                self.execution_service = svc
        return self.execution_service

    def _resolve_exchange_manager(self):
        if self.exchange_manager is not None:
            return self.exchange_manager
        svc = self._resolve_execution_service()
        if svc is not None and hasattr(svc, "exchange_manager"):
            self.exchange_manager = getattr(svc, "exchange_manager")
            return self.exchange_manager
        if self.gui is not None:
            mgr = getattr(self.gui, "ex_mgr", None)
            if mgr is not None:
                self.exchange_manager = mgr
        return self.exchange_manager

    def _resolve_ai_manager(self):
        if self.ai_manager is not None:
            return self.ai_manager
        if self.gui is not None:
            mgr = getattr(self.gui, "ai_mgr", None)
            if mgr is not None:
                self.ai_manager = mgr
        return self.ai_manager

    def _resolve_risk_manager(self):
        if self.risk_manager is not None:
            return self.risk_manager
        if self.gui is not None:
            mgr = getattr(self.gui, "risk_mgr", None)
            if mgr is not None:
                self.risk_manager = mgr
        return self.risk_manager

    # -- Execution helpers --
    def _execute_trade(self, symbol: str, side: str, market_df: pd.DataFrame) -> "OrderDTO | None":
        exec_service = self._resolve_execution_service()
        if exec_service is None:
            self.emitter.log("ExecutionService unavailable – trade skipped.", level="ERROR", component="AutoTrader")
            return None

        side_up = side.upper()
        price_hint = float(market_df["close"].iloc[-1]) if not market_df.empty else 0.0

        try:
            quoted_price, slip = exec_service.quote_market(
                symbol,
                "buy" if side_up == "BUY" else "sell",
                amount=None,
                fallback_bps=self.slippage_bps,
            )
        except Exception as e:
            self.emitter.log(f"quote_market failed: {e!r}", level="ERROR", component="AutoTrader")
            quoted_price, slip = (price_hint or 0.0, self.slippage_bps)

        exec_price = float(quoted_price or price_hint or 0.0)
        if exec_price <= 0:
            self.emitter.log("Brak ceny do egzekucji – trade pominięty.", level="WARNING", component="AutoTrader")
            return None

        try:
            balance = exec_service.fetch_balance() or {}
        except Exception as e:
            self.emitter.log(f"fetch_balance failed: {e!r}", level="ERROR", component="AutoTrader")
            balance = {}

        available_cash = self._extract_cash(balance)

        try:
            positions = exec_service.list_positions(symbol)
        except Exception as e:
            self.emitter.log(f"list_positions failed: {e!r}", level="ERROR", component="AutoTrader")
            positions = []

        long_qty = self._position_quantity(positions, target_side="LONG")

        if side_up == "SELL" and long_qty <= 0:
            self.emitter.log("Brak pozycji LONG do zamknięcia – SELL pominięty.", level="WARNING", component="AutoTrader")
            return None

        if side_up == "BUY" and available_cash <= 0:
            self.emitter.log("Brak wolnej gotówki – BUY pominięty.", level="WARNING", component="AutoTrader")
            return None

        fraction = self._compute_fraction(symbol, side_up, market_df, available_cash, positions)

        if side_up == "BUY":
            notional = max(0.0, available_cash * max(fraction, 0.0))
            qty = exec_service.calculate_quantity(symbol, notional, exec_price)
        else:
            qty = exec_service.quantize_amount(symbol, long_qty)
            notional = qty * exec_price

        if qty <= 0:
            self.emitter.log("Quantity <= 0 – trade skipped.", level="WARNING", component="AutoTrader")
            return None

        order = exec_service.execute_market(symbol, side_up, qty)
        price_out = float(getattr(order, "price", None) or exec_price)
        mode = getattr(exec_service.mode, "value", str(exec_service.mode))

        self.emitter.log(
            f"AutoTrade {side_up} {symbol} qty={qty:.6f} price≈{price_out:.4f} (mode={mode})",
            component="AutoTrader",
        )

        payload = {
            "symbol": symbol,
            "side": side_up,
            "qty": qty,
            "price": price_out,
            "mode": mode,
            "slip_bps": slip,
            "notional": notional,
        }
        try:
            if hasattr(order, "model_dump"):
                payload["order"] = order.model_dump()
        except Exception:
            pass
        self.emitter.emit("auto_trade_exec", **payload)
        return order

    def _compute_fraction(
        self,
        symbol: str,
        side: str,
        market_df: pd.DataFrame,
        available_cash: float,
        positions: List[Any],
    ) -> float:
        if side == "SELL":
            return 1.0

        risk_mgr = self._resolve_risk_manager()
        if risk_mgr is None:
            return self.fallback_fraction

        signal_payload = {
            "symbol": symbol,
            "direction": "LONG" if side == "BUY" else "SHORT",
            "strength": 1.0,
            "confidence": 1.0,
            "prediction": 1.0 if side == "BUY" else -1.0,
        }

        portfolio_ctx: Dict[str, Any] = {
            "cash": available_cash,
            "positions": self._positions_to_dicts(positions),
        }

        if self.portfolio_getter is not None:
            try:
                extra = self.portfolio_getter() or {}
                if isinstance(extra, dict):
                    for key, value in extra.items():
                        portfolio_ctx.setdefault(key, value)
            except Exception as exc:
                self.emitter.log(f"portfolio_getter error: {exc!r}", level="ERROR", component="AutoTrader")

        try:
            fraction = float(
                risk_mgr.calculate_position_size(
                    symbol=symbol,
                    signal=signal_payload,
                    market_data=market_df,
                    portfolio=portfolio_ctx,
                )
            )
        except Exception as exc:
            self.emitter.log(f"Risk sizing error: {exc!r}", level="ERROR", component="AutoTrader")
            return self.fallback_fraction

        if not (fraction > 0):
            return self.fallback_fraction
        return min(1.0, fraction)

    def _extract_cash(self, balance: Dict[str, Any]) -> float:
        if not isinstance(balance, dict):
            return 0.0
        asset = self.cash_asset
        if asset in balance and isinstance(balance[asset], (int, float)):
            return float(balance[asset])
        for key in ("free", "total", "balance"):
            section = balance.get(key)
            if isinstance(section, dict) and asset in section:
                amount = section.get(asset)
                if isinstance(amount, (int, float)):
                    return float(amount)
        return 0.0

    @staticmethod
    def _positions_to_dicts(positions: List[Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for pos in positions or []:
            if isinstance(pos, dict):
                out.append(dict(pos))
                continue
            data: Dict[str, Any] = {}
            for attr in ("symbol", "side", "quantity", "avg_price"):
                data[attr] = getattr(pos, attr, None)
            try:
                data["quantity"] = float(data.get("quantity", 0.0) or 0.0)
            except Exception:
                data["quantity"] = 0.0
            try:
                data["avg_price"] = float(data.get("avg_price", 0.0) or 0.0)
            except Exception:
                data["avg_price"] = 0.0
            out.append(data)
        return out

    @staticmethod
    def _position_quantity(positions: List[Any], target_side: str = "LONG") -> float:
        total = 0.0
        target = target_side.upper()
        for pos in positions or []:
            if isinstance(pos, dict):
                side = str(pos.get("side", "")).upper()
                qty_val = pos.get("quantity", 0.0)
            else:
                side = str(getattr(pos, "side", "")).upper()
                qty_val = getattr(pos, "quantity", 0.0)
            try:
                qty = float(qty_val or 0.0)
            except Exception:
                qty = 0.0
            if side == target:
                total += max(0.0, qty)
        return total
