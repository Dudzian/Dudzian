# auto_trader.py
# Walk-forward + auto-reoptimization + optional auto-trade loop, integrated via EventEmitter.
from __future__ import annotations

import threading
import time
import statistics
import asyncio
from typing import Iterable, Mapping, Optional, List, Dict, Any, Callable, Tuple, TYPE_CHECKING
import inspect
from dataclasses import dataclass, field

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

from KryptoLowca.alerts import AlertSeverity, emit_alert
from KryptoLowca.event_emitter_adapter import EventEmitter
from KryptoLowca.logging_utils import get_logger
from KryptoLowca.config_manager import StrategyConfig
from KryptoLowca.telemetry.prometheus_exporter import metrics as prometheus_metrics
from KryptoLowca.core.services import ExecutionService, RiskService, SignalService, exception_guard
from KryptoLowca.core.services.data_provider import ExchangeDataProvider
from KryptoLowca.core.services.paper_adapter import PaperTradingAdapter
from KryptoLowca.strategies.base import DataProvider, StrategyMetadata, StrategySignal

if TYPE_CHECKING:  # pragma: no cover
    from KryptoLowca.data.market_data import MarketDataProvider, MarketDataRequest

logger = get_logger(__name__)


@dataclass(slots=True)
class RiskDecision:
    should_trade: bool
    fraction: float
    state: str
    reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    mode: str = "demo"

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "should_trade": self.should_trade,
            "fraction": float(self.fraction),
            "state": self.state,
            "reason": self.reason,
            "details": dict(self.details),
            "mode": self.mode,
        }
        if self.stop_loss_pct is not None:
            payload["stop_loss_pct"] = float(self.stop_loss_pct)
        if self.take_profit_pct is not None:
            payload["take_profit_pct"] = float(self.take_profit_pct)
        return payload


class _NullExchangeAdapter:
    """Minimalny adapter wykorzystywany, gdy nie podano właściwego wykonawcy."""

    def __init__(self, emitter: EventEmitter | None) -> None:
        self._emitter = emitter

    async def submit_order(self, *, symbol: str, side: str, size: float, **kwargs: Any) -> Mapping[str, Any]:
        message = "Execution adapter not configured; skipping order"
        if self._emitter is not None:
            try:
                self._emitter.log(message, level="WARNING", component="AutoTrader")
            except Exception:  # pragma: no cover - defensywne logowanie
                logger.warning(message)
        logger.warning("%s (symbol=%s side=%s size=%s)", message, symbol, side, size)
        return {
            "status": "skipped",
            "symbol": symbol,
            "side": side,
            "size": size,
        }


class AutoTrader:
    """
    - Listens to trade_closed events to compute rolling PF & Expectancy
    - Monitors ATR (if 'bar' events are emitted)
    - Triggers reoptimization when thresholds break
    - Optional walk-forward scheduler (time-based)
    - Optional auto-trade loop (asks AI for decision and executes via GUI bridge)
    - Runtime-reconfigurable via ControlPanel (configure()/set_enable_auto_trade()).
    """
    BACKTEST_GUARD_MAX_AGE_S = 30 * 24 * 3600
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
        market_data_provider: Optional["MarketDataProvider"] = None,
        *,
        signal_service: Optional[SignalService] = None,
        risk_service: Optional[RiskService] = None,
        execution_service: Optional[ExecutionService] = None,
        data_provider: Optional[DataProvider] = None,
    ) -> None:
        self.emitter = emitter
        self.gui = gui
        self.symbol_getter = symbol_getter
        self._db_manager = getattr(gui, "db", None)

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
        self._strategy_config: StrategyConfig = StrategyConfig.presets()["SAFE"].validate()
        self._strategy_override = False
        self._strategy_config_error_notified = False
        self._reduce_only_until: Dict[str, float] = {}
        self._risk_lock_until: float = 0.0
        self._last_risk_audit: Optional[Dict[str, Any]] = None
        self._market_data_provider = market_data_provider
        self._signal_service = signal_service or SignalService()
        self._risk_service = risk_service or RiskService()
        self._execution_service = execution_service or ExecutionService(_NullExchangeAdapter(self.emitter))
        self._live_execution_adapter = getattr(self._execution_service, "_adapter", None)
        self._data_provider: Optional[DataProvider] = data_provider or self._build_data_provider()
        self._service_mode_enabled = self._data_provider is not None
        self._cooldowns: Dict[str, float] = {}
        self._service_tasks: Dict[Tuple[str, str], asyncio.Task[Any]] = {}
        self._service_loop: Optional[asyncio.AbstractEventLoop] = None
        self._paper_adapter: Optional[PaperTradingAdapter] = None
        self._paper_enabled = False
        self._exchange_config: Optional[Dict[str, Any]] = None
        self._refresh_execution_mode()

        # Subscribe to events
        emitter.on("trade_closed", self._on_trade_closed, tag="autotrader")
        emitter.on("bar", self._on_bar, tag="autotrader")

    # -- Public API --
    def _build_data_provider(self) -> Optional[DataProvider]:
        ex_mgr = getattr(self.gui, "ex_mgr", None)
        if ex_mgr is None:
            return None
        try:
            return ExchangeDataProvider(ex_mgr)
        except Exception:  # pragma: no cover - defensywne
            logger.exception("Failed to initialise ExchangeDataProvider")
            return None

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
        logger.info("AutoTrader worker threads started")

    def stop(self) -> None:
        self._stop.set()
        self.emitter.off("trade_closed", tag="autotrader")
        self.emitter.off("bar", tag="autotrader")
        self.emitter.log("AutoTrader stopped.", component="AutoTrader")
        logger.info("AutoTrader stop requested")
        for t in list(self._threads):
            try:
                if t.is_alive():
                    t.join(timeout=2.0)
            except Exception:
                logger.exception("Error while joining AutoTrader thread")
        self._threads.clear()

    def set_enable_auto_trade(self, flag: bool) -> None:
        with self._lock:
            self.enable_auto_trade = bool(flag)
        self.emitter.log(f"Auto-Trade {'ENABLED' if flag else 'DISABLED'}.", component="AutoTrader")

    def configure(self, **kwargs: Any) -> None:
        """Runtime reconfiguration from the ControlPanel."""
        with self._lock:
            for key, val in kwargs.items():
                if key == "strategy":
                    self._update_strategy_config(val)
                    continue
                if key == "exchange":
                    if isinstance(val, Mapping):
                        self._exchange_config = dict(val)
                    else:
                        self._exchange_config = None
                    continue
                if not hasattr(self, key):
                    continue
                setattr(self, key, val)
        self._refresh_execution_mode()
        self.emitter.log(f"AutoTrader reconfigured: {kwargs}", component="AutoTrader")

    # -- Event handlers --
    def _on_trade_closed(
        self,
        symbol: str,
        side: str,
        entry: float,
        exit: float,
        pnl: float,
        ts: float,
        meta: Dict[str, Any] | None = None,
        **_,
    ) -> None:
        self._closed_pnls.append(pnl)
        try:
            prometheus_metrics.record_trade_close(symbol, float(pnl))
        except Exception:
            logger.debug("Prometheus record_trade_close skipped", exc_info=True)
        pf, exp, win_rate = self._compute_metrics()
        self.emitter.emit(
            "metrics_updated",
            pf=pf,
            expectancy=exp,
            win_rate=win_rate,
            window=len(self._closed_pnls),
            ts=time.time(),
        )
        self._persist_performance_metrics(symbol, pf, exp, win_rate)

        # Check thresholds
        trigger_reason = None
        details: Dict[str, Any] = {}
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
    def _compute_metrics(self) -> tuple[Optional[float], Optional[float], Optional[float]]:
        if not self._closed_pnls:
            return (None, None, None)
        pnls = list(self._closed_pnls)
        wins = [p for p in pnls if p > 0]
        losses = [-p for p in pnls if p < 0]
        gross_profit = sum(wins)
        gross_loss = sum(losses)
        pf = (gross_profit / gross_loss) if gross_loss > 0 else (float('inf') if gross_profit > 0 else None)
        # Expectancy per trade (avg pnl)
        expectancy = statistics.mean(pnls) if pnls else None
        win_rate = (len(wins) / len(pnls)) if pnls else None
        return (pf, expectancy, win_rate)

    def _resolve_db(self):
        db = self._db_manager
        if db is None:
            candidate = getattr(self.gui, "db", None)
            if candidate is not None:
                db = candidate
                self._db_manager = candidate
        if db is None or not hasattr(db, "sync"):
            return None
        return db

    def _resolve_mode(self) -> str:
        network_var = getattr(self.gui, "network_var", None)
        if network_var is not None and hasattr(network_var, "get"):
            try:
                network = str(network_var.get()).strip().lower()
            except Exception:
                network = ""
            if network in {"testnet", "paper", "demo"}:
                return "paper"
        return "live"

    def _log_metric(
        self,
        metric: str,
        value: Optional[float],
        *,
        symbol: str,
        window: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if value is None:
            return
        db = self._resolve_db()
        if db is None:
            return
        context: Dict[str, Any] = {"symbol": symbol, "window": window, "source": "AutoTrader"}
        if extra:
            context.update(extra)
        payload = {
            "metric": metric,
            "value": float(value),
            "window": window,
            "symbol": symbol,
            "mode": self._resolve_mode(),
            "context": context,
        }
        try:
            db.sync.log_performance_metric(payload)
        except Exception:
            logger.exception("Nie udało się zapisać metryki %s", metric)

    def _persist_performance_metrics(
        self,
        symbol: str,
        pf: Optional[float],
        expectancy: Optional[float],
        win_rate: Optional[float],
    ) -> None:
        window = len(self._closed_pnls)
        extra = {
            "profit_factor": pf,
            "expectancy": expectancy,
            "win_rate": win_rate,
        }
        self._log_metric("auto_trader_expectancy", expectancy, symbol=symbol, window=window, extra=extra)
        self._log_metric("auto_trader_profit_factor", pf, symbol=symbol, window=window, extra=extra)
        self._log_metric("auto_trader_win_rate", win_rate, symbol=symbol, window=window, extra=extra)

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
                logger.exception("AI retrain thread failed to start")
        else:
            self.emitter.log("AI manager not available; reopt event emitted only.", level="WARNING", component="AutoTrader")

    def _call_train_safe(self, ai) -> None:
        try:
            self.emitter.log("AI retrain started...", component="AutoTrader")
            ai.train()
            self.emitter.log("AI retrain finished.", component="AutoTrader")
        except Exception as e:
            self.emitter.log(f"AI retrain error: {e!r}", level="ERROR", component="AutoTrader")
            logger.exception("AI retrain raised an exception")

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
                logger.exception("Walk-forward loop error")
            self._stop.wait(1.0)

    def _run_service_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._service_loop = loop
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._service_loop_main())
        except Exception:  # pragma: no cover - defensywny log
            logger.exception("Service-based auto trade loop crashed")
        finally:
            try:
                pending = [task for task in asyncio.all_tasks(loop=loop) if not task.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                logger.exception("Failed to shutdown auto trade service loop")
            finally:
                asyncio.set_event_loop(None)
                loop.close()
                self._service_loop = None

    async def _service_loop_main(self) -> None:
        try:
            while not self._stop.is_set():
                schedule = self._resolve_schedule_entries()
                await self._ensure_service_schedule(schedule)
                await asyncio.sleep(0.5)
        finally:
            await self._cancel_service_tasks()

    async def _ensure_service_schedule(self, schedule: List[Tuple[str, str]]) -> None:
        desired = {(sym, tf) for sym, tf in schedule if sym}
        for key in list(self._service_tasks):
            if key not in desired:
                task = self._service_tasks.pop(key)
                task.cancel()
                with exception_guard("AutoTrader.scheduler"):
                    await asyncio.gather(task, return_exceptions=True)
        for entry in schedule:
            if entry not in self._service_tasks:
                symbol, timeframe = entry
                if not symbol:
                    continue
                self._service_tasks[entry] = asyncio.create_task(
                    self._symbol_service_loop(symbol, timeframe)
                )

    async def _cancel_service_tasks(self) -> None:
        if not self._service_tasks:
            return
        tasks = list(self._service_tasks.values())
        for task in tasks:
            task.cancel()
        with exception_guard("AutoTrader.scheduler"):
            await asyncio.gather(*tasks, return_exceptions=True)
        self._service_tasks.clear()

    async def _symbol_service_loop(self, symbol: str, timeframe: str) -> None:
        interval = max(0.1, float(self.auto_trade_interval_s))
        while not self._stop.is_set():
            if not self.enable_auto_trade:
                await asyncio.sleep(interval)
                continue
            if self._is_symbol_on_cooldown(symbol):
                await asyncio.sleep(interval)
                continue
            await self._trade_once(symbol, timeframe)
            await asyncio.sleep(interval)

    def _resolve_schedule_entries(self) -> List[Tuple[str, str]]:
        timeframe = self._resolve_timeframe()
        entries: List[Tuple[str, str]] = []
        try:
            raw = self.symbol_getter()
        except Exception:
            logger.exception("Symbol getter failed")
            return entries

        if isinstance(raw, str):
            symbol = raw.strip()
            if symbol:
                entries.append((symbol, timeframe))
            return entries
        if isinstance(raw, Mapping):
            for sym, tf in raw.items():
                symbol = str(sym).strip()
                tf_value = str(tf).strip() or timeframe
                if symbol:
                    entries.append((symbol, tf_value))
            return entries
        if isinstance(raw, Iterable):
            for item in raw:
                if isinstance(item, tuple) and len(item) >= 2:
                    symbol = str(item[0]).strip()
                    tf_value = str(item[1]).strip() or timeframe
                else:
                    symbol = str(item).strip()
                    tf_value = timeframe
                if symbol:
                    entries.append((symbol, tf_value))
            return entries
        if raw:
            symbol = str(raw).strip()
            if symbol:
                entries.append((symbol, timeframe))
        return entries

    def _resolve_timeframe(self) -> str:
        timeframe = "1m"
        tf_var = getattr(self.gui, "timeframe_var", None)
        if tf_var is not None and hasattr(tf_var, "get"):
            try:
                value = tf_var.get()
            except Exception:
                value = None
            if value:
                timeframe = str(value)
        return timeframe

    def _is_symbol_on_cooldown(self, symbol: str) -> bool:
        with self._lock:
            until = self._cooldowns.get(symbol, 0.0)
            if not until:
                return False
            now = time.time()
            if now >= until:
                self._cooldowns.pop(symbol, None)
                return False
            return True

    def _register_cooldown(self, symbol: str, reason: str, duration: Optional[float] = None) -> None:
        cfg = self._get_strategy_config()
        cooldown = float(duration) if duration is not None else max(float(cfg.violation_cooldown_s), float(self.auto_trade_interval_s))
        until = time.time() + cooldown
        with self._lock:
            self._cooldowns[symbol] = until
        self.emitter.log(
            f"Cooldown applied for {symbol}: {reason} (until {until:.0f})",
            level="WARNING",
            component="AutoTrader",
        )

    async def _trade_once(self, symbol: str, timeframe: str) -> None:
        if self._data_provider is None:
            return
        with exception_guard("AutoTrader.trade"):
            cfg = self._get_strategy_config()
            strategy_name = cfg.preset or "SAFE"
            metadata = self._resolve_strategy_metadata(strategy_name)
            market_payload = await self._build_market_payload(symbol, timeframe)
            if not market_payload:
                return
            self._execution_service.update_market_data(symbol, timeframe, market_payload)
            price = float(market_payload.get("price") or 0.0)
            portfolio_snapshot = self._resolve_portfolio_snapshot(symbol, price)
            portfolio_value = float(
                portfolio_snapshot.get("value")
                or portfolio_snapshot.get("portfolio_value")
                or portfolio_snapshot.get("equity")
                or 0.0
            )
            position = float(portfolio_snapshot.get("position") or portfolio_snapshot.get("qty") or 0.0)
            context = self._signal_service.build_context(
                symbol=symbol,
                timeframe=timeframe,
                portfolio_value=portfolio_value,
                position=position,
                metadata=metadata,
                mode=cfg.mode,
            )
            signal = await self._signal_service.run_strategy(
                strategy_name,
                context,
                market_payload,
                self._data_provider,
            )
            if signal is None:
                return
            market_state = self._build_market_state(portfolio_snapshot, market_payload)
            assessment = self._risk_service.assess(signal, context, market_state)
            base_value = portfolio_value if portfolio_value > 0 else 1.0
            fraction = float(assessment.size or 0.0) / base_value
            decision = RiskDecision(
                should_trade=bool(assessment.allow),
                fraction=fraction,
                state="ok" if assessment.allow else "reject",
                reason=assessment.reason,
                details={"market_state": market_state},
                stop_loss_pct=assessment.stop_loss,
                take_profit_pct=assessment.take_profit,
                mode=cfg.mode,
            )
            self._emit_risk_audit(
                symbol,
                signal.action or "HOLD",
                decision,
                float(market_state.get("price") or 0.0),
            )
            if not assessment.allow:
                self._register_cooldown(symbol, assessment.reason or "risk_rejected")
                return

            if assessment.size is not None:
                signal.size = assessment.size
            if assessment.stop_loss is not None and signal.stop_loss is None:
                signal.stop_loss = assessment.stop_loss
            if assessment.take_profit is not None and signal.take_profit is None:
                signal.take_profit = assessment.take_profit
            signal.payload.setdefault("market_state", market_state)
            signal.payload.setdefault("price", market_state.get("price"))

            context.require_demo_mode()
            result = await self._execution_service.execute(signal, context)
            if result is not None:
                try:
                    prometheus_metrics.record_order(symbol, signal.action, float(signal.size or 0.0))
                except Exception:
                    logger.debug("Prometheus record_order skipped", exc_info=True)
                self.emitter.emit("auto_trade_tick", symbol=symbol, ts=time.time())
                self.emitter.log(
                    f"Auto-trade executed: {symbol} {signal.action}",
                    component="AutoTrader",
                )
            with self._lock:
                self._cooldowns.pop(symbol, None)

    async def _build_market_payload(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        if self._data_provider is None:
            return None
        with exception_guard("AutoTrader.market_data"):
            ohlcv = await self._data_provider.get_ohlcv(symbol, timeframe, limit=256)
            ticker = await self._data_provider.get_ticker(symbol)
        payload: Dict[str, Any] = {
            "ohlcv": ohlcv or {},
            "ticker": ticker or {},
        }
        payload["price"] = self._extract_price_from_payload(payload)
        return payload

    def _build_market_state(
        self,
        portfolio_snapshot: Mapping[str, Any],
        market_payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        price = float(market_payload.get("price") or 0.0)
        daily_loss = float(portfolio_snapshot.get("daily_loss_pct") or portfolio_snapshot.get("daily_loss") or 0.0)
        return {
            "price": price,
            "daily_loss_pct": daily_loss,
        }

    def _extract_price_from_payload(self, market_payload: Mapping[str, Any]) -> float:
        ticker = market_payload.get("ticker") if isinstance(market_payload, Mapping) else None
        if isinstance(ticker, Mapping):
            for key in ("last", "close", "bid", "ask", "price"):
                value = ticker.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
        ohlcv = market_payload.get("ohlcv") if isinstance(market_payload, Mapping) else None
        if isinstance(ohlcv, Mapping):
            close = ohlcv.get("close")
            if isinstance(close, (int, float)):
                return float(close)
        if isinstance(ohlcv, pd.DataFrame) and not ohlcv.empty:
            try:
                return float(ohlcv["close"].iloc[-1])
            except Exception:
                pass
        if isinstance(ohlcv, Iterable):
            try:
                last = list(ohlcv)[-1]
                if isinstance(last, Mapping):
                    value = last.get("close")
                    if isinstance(value, (int, float)):
                        return float(value)
                elif isinstance(last, (list, tuple)) and last:
                    candidate = last[4] if len(last) > 4 else last[-2]
                    if isinstance(candidate, (int, float)):
                        return float(candidate)
            except Exception:
                pass
        return 0.0

    def _resolve_strategy_metadata(self, strategy_name: str) -> StrategyMetadata:
        try:
            strategy_cls = self._signal_service._registry.get(strategy_name)
        except KeyError:
            return StrategyMetadata(name=strategy_name or "Unknown", description="AutoTrader context")
        metadata = getattr(strategy_cls, "metadata", None)
        if isinstance(metadata, StrategyMetadata):
            return metadata
        return StrategyMetadata(name=strategy_cls.__name__, description="AutoTrader context")

    def _resolve_portfolio_snapshot(self, symbol: str, price: float) -> Dict[str, Any]:
        adapter_snapshot = self._execution_service.portfolio_snapshot(symbol)
        if adapter_snapshot:
            return {
                "value": float(adapter_snapshot.get("value", 0.0)),
                "position": float(adapter_snapshot.get("position", 0.0)),
                "daily_loss_pct": 0.0,
                "price": float(adapter_snapshot.get("price", price)),
            }
        snapshot_fn = getattr(self.gui, "get_portfolio_snapshot", None)
        if callable(snapshot_fn):
            try:
                snapshot = snapshot_fn(symbol=symbol)
                if isinstance(snapshot, Mapping):
                    return dict(snapshot)
            except Exception:
                logger.exception("Portfolio snapshot retrieval failed")
        balance = float(getattr(self.gui, "paper_balance", 0.0) or 0.0)
        positions = getattr(self.gui, "_open_positions", {})
        qty = 0.0
        if isinstance(positions, Mapping):
            entry = positions.get(symbol)
            if isinstance(entry, Mapping):
                qty = float(entry.get("qty") or entry.get("quantity") or 0.0)
                entry_price = float(entry.get("entry") or price or 0.0)
            else:
                entry_price = price
        else:
            entry_price = price
        notional = qty * float(price or entry_price or 0.0)
        return {
            "value": balance + notional,
            "position": qty,
            "daily_loss_pct": 0.0,
        }

    def _refresh_execution_mode(self) -> None:
        cfg = self._get_strategy_config()
        exchange_cfg = self._exchange_config or {}
        testnet = bool(exchange_cfg.get("testnet", True))
        if cfg.mode == "demo" and not testnet:
            self._enable_paper_trading()
        else:
            self._disable_paper_trading()

    def _enable_paper_trading(self) -> None:
        if self._paper_enabled:
            return
        self._paper_adapter = PaperTradingAdapter(initial_balance=10_000.0)
        self._execution_service.set_adapter(self._paper_adapter)
        self._paper_enabled = True
        self.emitter.log("Paper trading engine enabled", component="AutoTrader")

    def _disable_paper_trading(self) -> None:
        if not self._paper_enabled:
            return
        if self._live_execution_adapter is not None:
            self._execution_service.set_adapter(self._live_execution_adapter)
        self._paper_adapter = None
        self._paper_enabled = False
        self.emitter.log("Paper trading engine disabled", component="AutoTrader")

    def _auto_trade_loop(self) -> None:
        if self._service_mode_enabled:
            self._run_service_loop()
            return
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

                if hasattr(self.gui, "is_demo_mode_active"):
                    try:
                        if not self.gui.is_demo_mode_active():
                            guard_fn = getattr(self.gui, "is_live_trading_allowed", None)
                            if callable(guard_fn) and not guard_fn():
                                msg = "Auto-trade blocked: live trading requires explicit confirmation."
                                self.emitter.log(msg, level="WARNING", component="AutoTrader")
                                logger.warning("%s Skipping auto trade for %s", msg, symbol)
                                self._stop.wait(self.auto_trade_interval_s)
                                continue
                    except Exception:
                        logger.exception("Failed to evaluate live trading guard")

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
                        logger.exception("predict_series failed during auto trade loop")

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

                if side is None:
                    self.emitter.log(
                        f"Auto-trade skipped for {symbol}: no valid model signal.",
                        level="WARNING",
                        component="AutoTrader",
                    )
                    logger.info("Skipping auto trade for %s due to missing model signal", symbol)
                    self._stop.wait(self.auto_trade_interval_s)
                    continue

                if last_price is None:
                    self.emitter.log(
                        f"Auto-trade skipped for {symbol}: missing market price.",
                        level="WARNING",
                        component="AutoTrader",
                    )
                    logger.warning("Skipping auto trade for %s due to missing market price", symbol)
                    self._stop.wait(self.auto_trade_interval_s)
                    continue

                signal_payload = self._build_signal_payload(symbol, side, last_pred)
                decision = self._evaluate_risk(symbol, side, float(last_price), signal_payload, df)
                self._emit_risk_audit(symbol, side, decision, float(last_price))
                if not decision.should_trade:
                    self._stop.wait(self.auto_trade_interval_s)
                    continue

                if hasattr(self.gui, "_bridge_execute_trade"):
                    try:
                        setattr(self.gui, "_autotrade_risk_context", decision.to_dict())
                    except Exception:
                        pass
                    self.gui._bridge_execute_trade(symbol, side.lower(), float(last_price))
                    try:
                        prometheus_metrics.record_order(symbol, side, decision.fraction)
                    except Exception:
                        logger.debug("Prometheus record_order skipped", exc_info=True)
                    self.emitter.emit("auto_trade_tick", symbol=symbol, ts=time.time())
                    self.emitter.log(f"Auto-trade executed: {symbol} {side}", component="AutoTrader")
                    logger.info("Auto trade executed for %s (%s)", symbol, side)
                else:
                    self.emitter.log(
                        "_bridge_execute_trade missing on GUI",
                        level="ERROR",
                        component="AutoTrader",
                    )
                    logger.error("GUI bridge missing for auto trade execution")
            except Exception as e:
                self.emitter.log(f"Auto trade tick error: {e!r}", level="ERROR", component="AutoTrader")
                logger.exception("Unhandled exception inside auto trade loop")
            self._stop.wait(self.auto_trade_interval_s)

    # --- Prediction helpers ---
    def _resolve_prediction_result(self, result: Any, *, context: str) -> Any:
        if not inspect.isawaitable(result):
            return result

        async def _await_result(awaitable: Any) -> Any:
            return await awaitable

        try:
            if self._service_loop is not None:
                try:
                    if self._service_loop.is_running():
                        future = asyncio.run_coroutine_threadsafe(
                            _await_result(result), self._service_loop
                        )
                        return future.result()
                except Exception:
                    # Jeśli pętla nie działa lub zgłosi błąd – fallback do lokalnego wykonania
                    pass

            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None

            if running_loop is not None:
                container: Dict[str, Any] = {}
                errors: List[BaseException] = []

                def _execute_in_thread() -> None:
                    try:
                        container["value"] = asyncio.run(_await_result(result))
                    except BaseException as thread_exc:  # pragma: no cover - propagacja błędu
                        errors.append(thread_exc)

                thread = threading.Thread(target=_execute_in_thread, daemon=True)
                thread.start()
                thread.join(timeout=30.0)
                if thread.is_alive():
                    raise TimeoutError("Timed out waiting for coroutine result")
                if errors:
                    raise errors[0]
                return container.get("value")

            return asyncio.run(_await_result(result))
        except Exception as exc:
            message = f"{context} coroutine failed: {exc!r}"
            try:
                self.emitter.log(message, level="ERROR", component="AutoTrader")
            except Exception:
                logger.exception("Emitter failed while logging coroutine error")
            logger.error("Coroutine execution failed during %s", context, exc_info=exc)
            return None

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
                preds = self._resolve_prediction_result(
                    preds, context=f"predict_series[{symbol}]"
                )
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
                        preds = self._resolve_prediction_result(
                            preds, context=f"predict_series[{symbol}]"
                        )
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
        if self._market_data_provider is not None:
            try:
                from KryptoLowca.data.market_data import MarketDataRequest

                request = MarketDataRequest(symbol=symbol, timeframe=timeframe, limit=256)
                df = self._market_data_provider.get_historical(request)
                return df
            except Exception as exc:
                self.emitter.log(
                    f"MarketDataProvider failed: {exc!r}",
                    level="ERROR",
                    component="AutoTrader",
                )
                logger.exception("MarketDataProvider.get_historical failed")
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
        if self._market_data_provider is not None:
            price = self._market_data_provider.get_latest_price(symbol)
            if price is not None:
                return float(price)
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

    # --- Strategy & risk helpers -------------------------------------------------
    def _update_strategy_config(self, value: Any) -> None:
        try:
            if isinstance(value, StrategyConfig):
                cfg = value.validate()
            elif isinstance(value, str):
                cfg = StrategyConfig.from_preset(value)
            elif isinstance(value, dict):
                cfg = StrategyConfig(**value).validate()
            else:
                raise TypeError("Nieobsługiwany format konfiguracji strategii")
            if cfg.mode == "live":
                backtest_ts = getattr(cfg, "backtest_passed_at", None)
                freshness_window = getattr(cfg, "BACKTEST_VALIDITY_WINDOW_S", 0.0)
                reason: Optional[str] = None
                if not backtest_ts:
                    reason = "brak potwierdzonego backtestu"
                else:
                    age = time.time() - float(backtest_ts)
                    if freshness_window and age > float(freshness_window):
                        hours = max(1, int(freshness_window // 3600))
                        reason = (
                            "wynik backtestu jest przeterminowany (starszy niż "
                            f"{hours}h)"
                        )
                if reason:
                    message = (
                        "Odrzucono przełączenie strategii w tryb LIVE – "
                        f"{reason}. Uruchom backtest i ponów próbę."
                    )
                    self.emitter.log(message, level="WARNING", component="AutoTrader")
                    logger.warning("%s", message)
                    return
        except Exception as exc:  # pragma: no cover - logujemy i utrzymujemy stare ustawienia
            self.emitter.log(
                f"Nieprawidłowa konfiguracja strategii: {exc!r}",
                level="ERROR",
                component="AutoTrader",
            )
            logger.exception("Strategy config update failed")
            return
        if cfg.mode == "live":
            passed_at = cfg.backtest_passed_at or 0.0
            now_ts = time.time()
            if passed_at <= 0:
                message = (
                    "Odrzucono przełączenie strategii w tryb LIVE: brak potwierdzonego backtestu."
                )
                self.emitter.log(message, level="WARNING", component="AutoTrader")
                logger.warning(message)
                return
            if now_ts - passed_at > self.BACKTEST_GUARD_MAX_AGE_S:
                message = (
                    "Odrzucono przełączenie strategii w tryb LIVE: wynik backtestu jest przestarzały."
                )
                self.emitter.log(message, level="WARNING", component="AutoTrader")
                logger.warning(message)
                return
        with self._lock:
            self._strategy_config = cfg
            self._strategy_override = True
        self.emitter.log(
            f"Strategia zaktualizowana: {cfg.preset} mode={cfg.mode} max_notional={cfg.max_position_notional_pct}",
            level="INFO",
            component="AutoTrader",
        )

    def _get_strategy_config(self) -> StrategyConfig:
        cfg = self._strategy_config
        if self._strategy_override:
            return cfg
        cfg_manager = getattr(self.gui, "cfg", None)
        loader = getattr(cfg_manager, "load_strategy_config", None) if cfg_manager else None
        if callable(loader):
            try:
                loaded = loader()
                if isinstance(loaded, StrategyConfig):
                    cfg = loaded.validate()
                elif isinstance(loaded, dict):
                    cfg = StrategyConfig(**loaded).validate()
                self._strategy_config_error_notified = False
            except Exception as exc:  # pragma: no cover - unikamy zalewania logów
                if not self._strategy_config_error_notified:
                    self.emitter.log(
                        f"Nie udało się wczytać konfiguracji strategii: {exc!r}",
                        level="WARNING",
                        component="AutoTrader",
                    )
                    logger.warning("Failed to refresh strategy config", exc_info=True)
                    self._strategy_config_error_notified = True
        self._strategy_config = cfg
        return cfg

    @staticmethod
    def _build_signal_payload(symbol: str, side: str, prediction: Optional[float]) -> Dict[str, Any]:
        try:
            pred_value = float(prediction) if prediction is not None else 0.0
        except Exception:
            pred_value = 0.0
        direction = "LONG" if str(side).upper() == "BUY" else "SHORT"
        strength = abs(pred_value)
        confidence = min(1.0, max(0.0, strength * 10.0))
        return {
            "symbol": symbol,
            "direction": direction,
            "prediction": pred_value,
            "strength": strength,
            "confidence": confidence,
        }

    def _evaluate_risk(
        self,
        symbol: str,
        side: str,
        price: float,
        signal_payload: Dict[str, Any],
        market_df: Optional[pd.DataFrame],
    ) -> RiskDecision:
        with self._lock:
            strategy_cfg = self._get_strategy_config()
        now = time.time()
        side_u = side.upper()

        ro_until = self._reduce_only_until.get(symbol, 0.0)
        if ro_until and now < ro_until and side_u == "BUY":
            details = {"until": ro_until, "now": now, "policy": "reduce_only"}
            return RiskDecision(
                should_trade=False,
                fraction=0.0,
                state="lock",
                reason="reduce_only_active",
                details=details,
                stop_loss_pct=strategy_cfg.default_sl,
                take_profit_pct=strategy_cfg.default_tp,
                mode=strategy_cfg.mode,
            )

        if self._risk_lock_until and now < self._risk_lock_until and side_u == "BUY":
            details = {"until": self._risk_lock_until, "now": now, "policy": "cooldown"}
            return RiskDecision(
                should_trade=False,
                fraction=0.0,
                state="lock",
                reason="cooldown_active",
                details=details,
                stop_loss_pct=strategy_cfg.default_sl,
                take_profit_pct=strategy_cfg.default_tp,
                mode=strategy_cfg.mode,
            )

        env_mode = self._resolve_mode()
        if strategy_cfg.mode == "demo" and env_mode != "paper":
            details = {"configured_mode": strategy_cfg.mode, "env_mode": env_mode}
            return RiskDecision(
                should_trade=False,
                fraction=0.0,
                state="lock",
                reason="demo_mode_enforced",
                details=details,
                stop_loss_pct=strategy_cfg.default_sl,
                take_profit_pct=strategy_cfg.default_tp,
                mode=strategy_cfg.mode,
            )

        compliance_state = {
            "compliance_confirmed": bool(strategy_cfg.compliance_confirmed),
            "api_keys_configured": bool(strategy_cfg.api_keys_configured),
            "acknowledged_risk_disclaimer": bool(
                strategy_cfg.acknowledged_risk_disclaimer
            ),
        }
        if strategy_cfg.mode == "live":
            missing_checks = [name for name, ok in compliance_state.items() if not ok]
            if missing_checks:
                summary = ", ".join(missing_checks)
                log_message = (
                    "Live trading blocked: missing compliance confirmations -> "
                    f"{summary}"
                )
                self.emitter.log(log_message, level="WARNING", component="AutoTrader")
                return RiskDecision(
                    should_trade=False,
                    fraction=0.0,
                    state="lock",
                    reason="live_compliance_missing",
                    details={
                        "missing_checks": missing_checks,
                        "compliance_state": compliance_state,
                    },
                    stop_loss_pct=strategy_cfg.default_sl,
                    take_profit_pct=strategy_cfg.default_tp,
                    mode=strategy_cfg.mode,
                )

        portfolio_ctx = self._build_portfolio_context(symbol, price)
        risk_mgr = getattr(self.gui, "risk_mgr", None)
        fraction = strategy_cfg.trade_risk_pct
        details: Dict[str, Any] = {}
        risk_engine_details: Optional[Dict[str, Any]] = None

        try:
            positions_ctx = portfolio_ctx.get("positions") or {}
            open_positions = sum(
                1 for entry in positions_ctx.values() if (entry or {}).get("size")
            )
            prometheus_metrics.set_open_positions(count=open_positions, mode=strategy_cfg.mode)
        except Exception:
            logger.debug("Nie udało się ustawić metryki open_positions", exc_info=True)

        stop_loss_pct = strategy_cfg.default_sl
        take_profit_pct = strategy_cfg.default_tp

        market_payload: Any
        if isinstance(market_df, pd.DataFrame):
            market_payload = market_df
        else:
            market_payload = {"price": price}

        if risk_mgr is not None and hasattr(risk_mgr, "calculate_position_size"):
            try:
                prepared_kwargs, request_details = self._prepare_risk_kwargs(
                    risk_mgr,
                    symbol=symbol,
                    signal_payload=signal_payload,
                    market_payload=market_payload,
                    portfolio_ctx=portfolio_ctx,
                    price=price,
                )
                result: Any = None
                if prepared_kwargs is not None:
                    try:
                        result = risk_mgr.calculate_position_size(**prepared_kwargs)
                    except TypeError as exc:
                        logger.warning(
                            "Risk manager %s signature call failed (%s); falling back to legacy invocation",
                            type(risk_mgr).__name__,
                            exc,
                        )
                        result = None
                if result is None:
                    legacy_args = [symbol, signal_payload, market_payload, portfolio_ctx]
                    if request_details:
                        legacy_args.append(True)
                    try:
                        result = risk_mgr.calculate_position_size(*legacy_args)
                    except TypeError as exc:
                        if request_details:
                            logger.warning(
                                "Risk manager %s rejected return_details flag (%s); retrying without details",
                                type(risk_mgr).__name__,
                                exc,
                            )
                            result = risk_mgr.calculate_position_size(
                                symbol,
                                signal_payload,
                                market_payload,
                                portfolio_ctx,
                            )
                            request_details = False
                        else:
                            raise
                fraction_val, details_val, sl_override, tp_override = self._normalize_risk_result(result)
                if fraction_val is not None:
                    fraction = fraction_val
                if details_val:
                    details = details_val
                if sl_override is not None:
                    stop_loss_pct = sl_override
                if tp_override is not None:
                    take_profit_pct = tp_override
            except Exception as exc:
                self.emitter.log(
                    f"Risk sizing error: {exc!r}", level="ERROR", component="AutoTrader"
                )
                logger.exception("Risk manager calculate_position_size failed")
                fraction = 0.0
                details = {"error": str(exc)}
        else:
            details["risk_mgr"] = "missing"

        try:
            fraction = float(fraction)
        except Exception:
            fraction = 0.0
        fraction = max(0.0, min(1.0, fraction))

        if fraction is not None and "recommended_size" not in details:
            details["recommended_size"] = fraction

        try:
            if stop_loss_pct is not None:
                stop_loss_pct = float(stop_loss_pct)
        except Exception:
            stop_loss_pct = strategy_cfg.default_sl

        try:
            if take_profit_pct is not None:
                take_profit_pct = float(take_profit_pct)
        except Exception:
            take_profit_pct = strategy_cfg.default_tp

        state = "ok"
        limit_events: List[Dict[str, Any]] = []

        # (opcjonalnie) przepisanie szczegółów od silnika ryzyka z innego wariantu
        if risk_engine_details:
            risk_engine_details = {
                key: (float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value)
                for key, value in risk_engine_details.items()
            }
            risk_engine_details.setdefault(
                "source", getattr(getattr(risk_mgr, "__class__", None), "__name__", type(risk_mgr).__name__)
            )
            details["risk_engine"] = risk_engine_details

        # limit pozycyjny względem notional
        max_pct = float(strategy_cfg.max_position_notional_pct)
        if max_pct > 0.0 and fraction > max_pct:
            limit_events.append({
                "type": "max_position_notional_pct",
                "value": fraction,
                "threshold": max_pct,
            })
            fraction = max_pct
            state = "warn"

        account_value = self._resolve_account_value(portfolio_ctx)
        positions = portfolio_ctx.get("positions") or {}
        position_ctx = positions.get(symbol, {})
        symbol_notional = float(position_ctx.get("notional", 0.0) or 0.0)
        total_notional = float(portfolio_ctx.get("total_notional", 0.0) or 0.0)

        if account_value <= 0.0:
            if strategy_cfg.reduce_only_after_violation:
                self._trigger_reduce_only(symbol, "no_account_value", strategy_cfg)
            details.update({"account_value": account_value, "portfolio_ctx": portfolio_ctx})
            return RiskDecision(
                should_trade=False,
                fraction=0.0,
                state="lock",
                reason="account_value_non_positive",
                details=details,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                mode=strategy_cfg.mode,
            )

        projected_notional = total_notional
        if side_u == "BUY":
            projected_notional += fraction * account_value
        else:
            projected_notional = max(total_notional - symbol_notional, 0.0)

        leverage_after = projected_notional / max(account_value, 1e-9)
        if side_u == "BUY" and leverage_after > strategy_cfg.max_leverage + 1e-6:
            limit_events.append({
                "type": "max_leverage",
                "value": leverage_after,
                "threshold": strategy_cfg.max_leverage,
            })
            if strategy_cfg.reduce_only_after_violation:
                self._trigger_reduce_only(symbol, "max_leverage", strategy_cfg)
            return RiskDecision(
                should_trade=False,
                fraction=0.0,
                state="lock",
                reason="max_leverage_exceeded",
                details={
                    "leverage_after": leverage_after,
                    "max_leverage": strategy_cfg.max_leverage,
                    "limit_events": limit_events,
                },
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                mode=strategy_cfg.mode,
            )

        if fraction <= 0.0:
            if strategy_cfg.reduce_only_after_violation:
                self._trigger_reduce_only(symbol, "fraction_non_positive", strategy_cfg)
            details["limit_events"] = limit_events
            return RiskDecision(
                should_trade=False,
                fraction=0.0,
                state="lock",
                reason="risk_fraction_zero",
                details=details,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                mode=strategy_cfg.mode,
            )

        if side_u == "SELL" and ro_until and now < ro_until:
            # pozwól zamknąć pozycję i wyczyść reduce-only
            self._reduce_only_until.pop(symbol, None)
            details["reduce_only_cleared"] = True

        if limit_events:
            details.setdefault("limit_events", []).extend(limit_events)
            if state != "lock":
                state = "warn"

        decision_details = {
            **details,
            "account_value": account_value,
            "projected_notional": projected_notional,
            "current_notional": total_notional,
            "symbol_notional": symbol_notional,
        }

        decision = RiskDecision(
            should_trade=True,
            fraction=fraction,
            state=state,
            reason="risk_ok" if state == "ok" else "risk_clamped",
            details=decision_details,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            mode=strategy_cfg.mode,
        )
        self._apply_violation_cooldown(symbol, side_u, strategy_cfg, decision)
        return decision

    def _emit_risk_audit(self, symbol: str, side: str, decision: RiskDecision, price: float) -> None:
        payload = {
            "symbol": symbol,
            "side": side,
            "state": decision.state,
            "reason": decision.reason,
            "fraction": float(decision.fraction),
            "price": float(price),
            "mode": decision.mode,
            "details": decision.details,
            "stop_loss_pct": decision.stop_loss_pct,
            "take_profit_pct": decision.take_profit_pct,
            "ts": time.time(),
            "schema_version": 1,
        }
        self._last_risk_audit = payload
        try:
            prometheus_metrics.observe_risk(symbol, decision.state, decision.fraction, decision.mode)
        except Exception:
            logger.debug("Prometheus observe_risk skipped", exc_info=True)
        try:
            self.emitter.emit("risk_guard_event", **payload)
        except Exception:  # pragma: no cover - audyt nie może zatrzymać bota
            logger.exception("Failed to emit risk_guard_event")

        db_manager = self._resolve_db()
        if db_manager is not None:
            limit_events: Optional[List[str]] = None
            if isinstance(decision.details, dict):
                candidate = decision.details.get("limit_events")
                if isinstance(candidate, (list, tuple)):
                    limit_events = [str(item) for item in candidate]
            db_payload = {
                "symbol": symbol,
                "state": decision.state,
                "fraction": float(decision.fraction),
                "side": side,
                "reason": decision.reason,
                "price": float(price),
                "mode": decision.mode,
                "limit_events": limit_events,
                "details": decision.details,
                "stop_loss_pct": decision.stop_loss_pct,
                "take_profit_pct": decision.take_profit_pct,
                "should_trade": decision.should_trade,
            }
            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop is not None:
                    async_method = getattr(db_manager, "log_risk_audit", None)
                    if callable(async_method):
                        result = async_method(db_payload)
                        if inspect.isawaitable(result):
                            task = loop.create_task(result)

                            def _handle_task(t: asyncio.Task[Any]) -> None:
                                try:
                                    t.result()
                                except Exception:  # pragma: no cover - logowanie awarii w tle
                                    logger.exception("Async risk audit log failed")

                            task.add_done_callback(_handle_task)
                        else:
                            logger.debug("Async log_risk_audit returned non-awaitable result")
                    else:
                        logger.debug("No async log_risk_audit available on db manager")
                else:
                    sync = getattr(db_manager, "sync", None)
                    log_method = None
                    if sync is not None:
                        log_method = getattr(sync, "log_risk_audit", None)
                    if log_method is None:
                        log_method = getattr(db_manager, "log_risk_audit", None)
                    if callable(log_method):
                        result = log_method(db_payload)
                        if inspect.isawaitable(result):
                            asyncio.run(result)
                    else:
                        logger.debug("No log_risk_audit method available on db manager")
            except Exception:  # pragma: no cover - logowanie awarii
                logger.exception("Failed to persist risk audit log")

        msg = (
            f"Risk state={decision.state} reason={decision.reason} symbol={symbol} side={side} fraction={decision.fraction:.4f}"
        )
        level = "INFO"
        if decision.state == "warn":
            level = "WARNING"
        elif decision.state == "lock":
            level = "WARNING" if decision.should_trade else "ERROR"
        self.emitter.log(msg, level=level, component="AutoTrader")
        if decision.state != "ok":
            severity = AlertSeverity.WARNING if decision.state == "warn" else AlertSeverity.ERROR
            emit_alert(
                f"Risk guard {decision.state} ({decision.reason}) dla {symbol}",
                severity=severity,
                source="risk_guard",
                context={
                    "symbol": symbol,
                    "side": side,
                    "fraction": float(decision.fraction),
                    "state": decision.state,
                    "reason": decision.reason,
                    "limit_events": decision.details.get("limit_events"),
                    "cooldown_until": decision.details.get("cooldown_until"),
                },
            )

    @staticmethod
    def _supports_return_details(
        risk_mgr: Any, signature: inspect.Signature | None = None
    ) -> bool:
        if signature is None:
            try:
                signature = inspect.signature(risk_mgr.calculate_position_size)  # type: ignore[attr-defined]
            except (TypeError, ValueError, AttributeError):
                return False
        return "return_details" in signature.parameters

    # --- wariant używany w tej wersji ---
    @staticmethod
    def _prepare_risk_kwargs(
        risk_mgr: Any,
        *,
        symbol: str,
        signal_payload: Dict[str, Any],
        market_payload: Any,
        portfolio_ctx: Dict[str, Any],
        price: float,
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        try:
            method = risk_mgr.calculate_position_size  # type: ignore[attr-defined]
        except AttributeError:
            return None, False
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Unable to inspect %s.calculate_position_size signature: %s",
                type(risk_mgr).__name__,
                exc,
            )
            return None, False

        kwargs: Dict[str, Any] = {}
        missing_required: List[str] = []
        has_signal_param = False
        has_portfolio_param = False

        for name, param in signature.parameters.items():
            if name == "self":
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if name == "symbol":
                kwargs[name] = symbol
            elif name in {"signal", "signal_data"}:
                kwargs[name] = signal_payload
                has_signal_param = True
            elif name in {"portfolio", "current_portfolio"}:
                kwargs[name] = portfolio_ctx
                has_portfolio_param = True
            elif name in {"market_data", "market", "market_payload", "market_ctx", "market_context"}:
                kwargs[name] = market_payload
            elif name == "price":
                kwargs[name] = price
            elif name == "return_details":
                kwargs[name] = True
            elif param.default is inspect._empty:
                missing_required.append(name)

        request_details = AutoTrader._supports_return_details(risk_mgr, signature)

        if missing_required:
            logger.warning(
                "Risk manager %s.calculate_position_size has unsupported required parameters: %s",
                type(risk_mgr).__name__,
                ", ".join(missing_required),
            )
            return None, request_details

        if not has_signal_param and any(
            alias in signature.parameters for alias in ("signal", "signal_data")
        ):
            logger.warning(
                "Risk manager %s.calculate_position_size signature declares signal data but mapping failed",
                type(risk_mgr).__name__,
            )
            return None, request_details

        if not has_signal_param and not any(
            alias in signature.parameters for alias in ("signal", "signal_data")
        ):
            logger.warning(
                "Risk manager %s.calculate_position_size is missing signal parameter (expected 'signal' or 'signal_data')",
                type(risk_mgr).__name__,
            )
            return None, request_details

        if not has_portfolio_param and any(
            alias in signature.parameters for alias in ("portfolio", "current_portfolio")
        ):
            logger.warning(
                "Risk manager %s.calculate_position_size signature declares portfolio data but mapping failed",
                type(risk_mgr).__name__,
            )
            return None, request_details

        if not has_portfolio_param and not any(
            alias in signature.parameters for alias in ("portfolio", "current_portfolio")
        ):
            logger.warning(
                "Risk manager %s.calculate_position_size is missing portfolio parameter (expected 'portfolio' or 'current_portfolio')",
                type(risk_mgr).__name__,
            )
            return None, request_details

        return kwargs, request_details

    @staticmethod
    def _normalize_risk_result(
        result: Any,
    ) -> Tuple[Optional[float], Dict[str, Any], Optional[float], Optional[float]]:
        fraction: Optional[float] = None
        details: Dict[str, Any] = {}
        stop_loss_override: Optional[float] = None
        take_profit_override: Optional[float] = None

        if hasattr(result, "recommended_size"):
            try:
                recommended = float(getattr(result, "recommended_size", 0.0))
            except Exception:
                recommended = 0.0
            fraction = recommended
            details = {
                "recommended_size": recommended,
                "max_allowed_size": float(
                    getattr(result, "max_allowed_size", recommended) or recommended
                ),
                "kelly_size": float(getattr(result, "kelly_size", recommended) or recommended),
                "risk_adjusted_size": float(
                    getattr(result, "risk_adjusted_size", recommended) or recommended
                ),
            }
            confidence = getattr(result, "confidence_level", None)
            if confidence is not None:
                try:
                    details["confidence_level"] = float(confidence)
                except Exception:
                    details["confidence_level"] = confidence
            reasoning = getattr(result, "reasoning", None)
            if reasoning is not None:
                details["reasoning"] = reasoning
            for attr_name in ("stop_loss_pct", "stop_loss"):
                sl_value = getattr(result, attr_name, None)
                if sl_value is not None:
                    try:
                        stop_loss_override = float(sl_value)
                        break
                    except Exception:
                        continue
            for attr_name in ("take_profit_pct", "take_profit"):
                tp_value = getattr(result, attr_name, None)
                if tp_value is not None:
                    try:
                        take_profit_override = float(tp_value)
                        break
                    except Exception:
                        continue
        elif isinstance(result, tuple) and len(result) == 2:
            fraction, details_val = result
            if isinstance(details_val, dict):
                details = dict(details_val)
            else:
                details = {"details": details_val}
        elif isinstance(result, dict):
            details = dict(result)
            raw_fraction = details.get(
                "recommended_size",
                details.get("fraction", details.get("size")),
            )
            if raw_fraction is not None:
                try:
                    fraction = float(raw_fraction)
                except Exception:
                    fraction = None
        else:
            fraction = result if result is not None else None

        try:
            if fraction is not None:
                fraction = float(fraction)
        except Exception:
            fraction = None

        if fraction is not None:
            details.setdefault("recommended_size", fraction)

        return fraction, details, stop_loss_override, take_profit_override

    # --- alternatywne (zachowane) API wołania risk engine z innej gałęzi ---
    @staticmethod
    def _supports_return_details_legacy(risk_mgr: Any) -> bool:
        try:
            sig = inspect.signature(risk_mgr.calculate_position_size)  # type: ignore[attr-defined]
        except (TypeError, ValueError, AttributeError):
            return False
        return "return_details" in sig.parameters

    def _call_risk_manager(
        self,
        risk_mgr: Any,
        symbol: str,
        signal_payload: Dict[str, Any],
        market_payload: Any,
        portfolio_ctx: Dict[str, Any],
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        supports_details = self._supports_return_details_legacy(risk_mgr)
        args = [symbol, signal_payload, market_payload, portfolio_ctx]
        kwargs: Dict[str, Any] = {"return_details": True} if supports_details else {}

        try:
            result = risk_mgr.calculate_position_size(*args, **kwargs)  # type: ignore[misc]
        except TypeError:
            mapped_kwargs = self._build_risk_kwargs(
                risk_mgr,
                symbol,
                signal_payload,
                market_payload,
                portfolio_ctx,
                request_details=supports_details,
            )
            result = risk_mgr.calculate_position_size(**mapped_kwargs)  # type: ignore[misc]

        fraction: float
        risk_details: Optional[Dict[str, Any]] = None

        if isinstance(result, tuple):
            primary = result[0]
            fraction = float(primary)
            if len(result) > 1:
                extra = result[1]
                if isinstance(extra, dict):
                    risk_details = dict(extra)
                else:
                    risk_details = {"context": extra}
        elif hasattr(result, "recommended_size"):
            fraction = float(getattr(result, "recommended_size", 0.0) or 0.0)
            risk_details = {
                "recommended_size": float(getattr(result, "recommended_size", 0.0) or 0.0),
                "max_allowed_size": float(getattr(result, "max_allowed_size", 0.0) or 0.0),
                "kelly_size": float(getattr(result, "kelly_size", 0.0) or 0.0),
                "risk_adjusted_size": float(getattr(result, "risk_adjusted_size", 0.0) or 0.0),
                "confidence_level": float(getattr(result, "confidence_level", 0.0) or 0.0),
                "reasoning": getattr(result, "reasoning", "") or "",
            }
        else:
            fraction = float(result)

        if risk_details is not None:
            risk_details.setdefault("requested_details", supports_details)
        return fraction, risk_details

    @staticmethod
    def _build_risk_kwargs(
        risk_mgr: Any,
        symbol: str,
        signal_payload: Dict[str, Any],
        market_payload: Any,
        portfolio_ctx: Dict[str, Any],
        *,
        request_details: bool,
    ) -> Dict[str, Any]:
        try:
            sig = inspect.signature(risk_mgr.calculate_position_size)  # type: ignore[attr-defined]
        except (TypeError, ValueError, AttributeError):
            kwargs = {
                "symbol": symbol,
                "signal": signal_payload,
                "market_data": market_payload,
                "portfolio": portfolio_ctx,
            }
            if request_details:
                kwargs["return_details"] = True
            return kwargs

        kwargs: Dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            lname = name.lower()
            if lname in {"symbol", "pair", "instrument"}:
                kwargs[name] = symbol
            elif lname in {"signal", "signal_data", "signal_payload", "signal_ctx"}:
                kwargs[name] = signal_payload
            elif lname in {"market_data", "market", "market_df", "data", "market_ctx"}:
                kwargs[name] = market_payload
            elif lname in {"portfolio", "portfolio_ctx", "current_portfolio"}:
                kwargs[name] = portfolio_ctx
            elif lname == "return_details" and request_details:
                kwargs[name] = True
            elif param.default is inspect._empty:
                kwargs[name] = portfolio_ctx
        return kwargs

    def _resolve_account_value(self, portfolio_ctx: Dict[str, Any]) -> float:
        account_value = portfolio_ctx.get("equity")
        try:
            if account_value is not None:
                return float(account_value)
        except Exception:
            pass
        cash = portfolio_ctx.get("cash")
        try:
            if cash is not None:
                return float(cash)
        except Exception:
            pass
        candidates = ["paper_balance", "account_balance", "equity", "cash"]
        for attr in candidates:
            if hasattr(self.gui, attr):
                try:
                    return float(getattr(self.gui, attr))
                except Exception:
                    continue
        return 0.0

    def _build_portfolio_context(self, symbol: str, ref_price: float) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "positions": {},
            "total_notional": 0.0,
        }
        raw_positions = getattr(self.gui, "_open_positions", None)
        if not isinstance(raw_positions, dict):
            raw_positions = getattr(self.gui, "open_positions", None)
        if isinstance(raw_positions, dict):
            for sym, pos in raw_positions.items():
                try:
                    qty = float(pos.get("qty", 0.0) or 0.0)
                    entry = float(pos.get("entry") or pos.get("price") or ref_price or 0.0)
                except Exception:
                    qty = 0.0
                    entry = ref_price or 0.0
                notional = abs(qty * entry)
                context["positions"][sym] = {
                    "qty": qty,
                    "entry": entry,
                    "side": str(pos.get("side", "")).upper(),
                    "notional": notional,
                }
                context["total_notional"] += notional

        cash_candidates = [
            ("paper_balance", getattr(self.gui, "paper_balance", None)),
            ("account_balance", getattr(self.gui, "account_balance", None)),
        ]
        cash_value = 0.0
        for name, val in cash_candidates:
            try:
                if val is not None:
                    cash_value = float(val)
                    break
            except Exception:
                continue
        context["cash"] = cash_value
        context["equity"] = max(cash_value, cash_value + context["total_notional"])
        return context

    def _trigger_reduce_only(self, symbol: str, reason: str, cfg: StrategyConfig) -> None:
        cooldown = max(float(cfg.violation_cooldown_s), 1.0)
        until = time.time() + cooldown
        self._reduce_only_until[symbol] = until
        self._risk_lock_until = max(self._risk_lock_until, until)
        self.emitter.log(
            f"Reduce-only aktywne dla {symbol} przez {cooldown:.0f}s (powód: {reason})",
            level="WARNING",
            component="AutoTrader",
        )
        emit_alert(
            f"Reduce-only dla {symbol} przez {cooldown:.0f}s",
            severity=AlertSeverity.ERROR,
            source="risk_guard",
            context={
                "symbol": symbol,
                "reason": reason,
                "cooldown_seconds": cooldown,
                "cooldown_until": until,
            },
        )

    def _apply_violation_cooldown(
        self,
        symbol: str,
        side: str,
        cfg: StrategyConfig,
        decision: RiskDecision,
    ) -> None:
        if side != "BUY":
            return
        if decision.state not in {"warn", "lock"}:
            return

        cooldown = max(float(cfg.violation_cooldown_s), 1.0)
        now = time.time()
        previous_lock = self._risk_lock_until
        proposed_until = now + cooldown
        existing_until = float(decision.details.get("cooldown_until", 0.0) or 0.0)
        if existing_until and existing_until >= proposed_until:
            decision.details.setdefault("cooldown_seconds", cooldown)
            decision.details["cooldown_until"] = existing_until
            self._risk_lock_until = max(self._risk_lock_until, existing_until)
            return

        until = proposed_until
        decision.details.setdefault("cooldown_seconds", cooldown)
        decision.details["cooldown_until"] = until
        self._risk_lock_until = max(self._risk_lock_until, until)

        if previous_lock >= self._risk_lock_until:
            return

        self.emitter.log(
            f"Aktywowano cooldown po naruszeniu limitów ({decision.state}) do {until:.0f}",
            level="WARNING",
            component="AutoTrader",
        )
        emit_alert(
            f"Cooldown ryzyka dla {symbol}",
            severity=AlertSeverity.WARNING if decision.state == "warn" else AlertSeverity.ERROR,
            source="risk_guard",
            context={
                "symbol": symbol,
                "state": decision.state,
                "reason": decision.reason,
                "cooldown_seconds": cooldown,
                "cooldown_until": until,
            },
        )
