"""Lightweight auto-trading controller used by tests and runtime scaffolding.

This module re-implements the bare minimum of the legacy ``AutoTrader``
behaviour in a dependency-free manner so that it can operate without the
monolithic application package.  The original implementation pulled a large
amount of infrastructure (event emitters, Prometheus exporters, runtime
services).  For unit tests we only need predictable threading semantics and
state transitions.

The implementation below focuses on deterministic start/stop logic, manual
activation flow and logging hooks.  It still exposes a small ``RiskDecision``
structure for compatibility with code that serialises decisions.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol


LOGGER = logging.getLogger(__name__)


class EmitterLike(Protocol):
    """Minimal protocol expected from GUI/event emitter integrations."""

    def on(self, event: str, callback: Callable[..., Any], *, tag: str | None = None) -> None:
        ...  # pragma: no cover - optional interface used by runtime only

    def off(self, event: str, *, tag: str | None = None) -> None:
        ...  # pragma: no cover - optional interface used by runtime only

    def emit(self, event: str, **payload: Any) -> None:
        ...  # pragma: no cover - optional interface used by runtime only

    def log(self, message: str, *args: Any, **kwargs: Any) -> None:
        ...


@dataclass(slots=True)
class RiskDecision:
    """Serializable snapshot describing the outcome of a risk engine check."""

    should_trade: bool
    fraction: float
    state: str
    reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    mode: str = "demo"

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
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


class AutoTrader:
    """Small cooperative wrapper around an auto-trading loop.

    The class is intentionally tiny – it exists so that unit tests can exercise
    manual confirmation logic without pulling in the whole legacy runtime.  It
    exposes the same public attributes that the tests rely on (``enable_auto_trade``
    and ``_auto_trade_user_confirmed``) and uses an overridable ``_auto_trade_loop``
    method executed inside a worker thread when the user confirms auto-trading.
    """

    def __init__(
        self,
        emitter: EmitterLike,
        gui: Any,
        symbol_getter: Callable[[], str],
        pf_min: float = 1.3,
        expectancy_min: float = 0.0,
        metrics_window: int = 30,
        atr_ratio_threshold: float = 0.5,
        atr_baseline_len: int = 100,
        reopt_cooldown_s: int = 1800,
        walkforward_interval_s: Optional[int] = 3600,
        walkforward_min_closed_trades: int = 10,
        enable_auto_trade: bool = True,
        auto_trade_interval_s: float = 30.0,
        market_data_provider: Optional[Any] = None,
        *,
        signal_service: Optional[Any] = None,
        risk_service: Optional[Any] = None,
        execution_service: Optional[Any] = None,
        data_provider: Optional[Any] = None,
        bootstrap_context: Any | None = None,
        core_risk_engine: Any | None = None,
        core_execution_service: Any | None = None,
        ai_connector: Any | None = None,
    ) -> None:
        self.emitter = emitter
        self.gui = gui
        self.symbol_getter = symbol_getter
        self.market_data_provider = market_data_provider

        self.enable_auto_trade = bool(enable_auto_trade)
        self.auto_trade_interval_s = float(auto_trade_interval_s)

        self.signal_service = signal_service
        self.risk_service = risk_service
        self.execution_service = execution_service
        self.data_provider = data_provider
        self.bootstrap_context = bootstrap_context
        self.core_risk_engine = core_risk_engine
        self.core_execution_service = core_execution_service
        self.ai_connector = ai_connector

        self._stop = threading.Event()
        self._auto_trade_stop = threading.Event()
        self._auto_trade_thread: threading.Thread | None = None
        self._auto_trade_thread_active = False
        self._auto_trade_user_confirmed = False
        self._started = False
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def _log(self, message: str, *, level: int = logging.INFO, **kwargs: Any) -> None:
        if hasattr(self.emitter, "log"):
            try:
                self.emitter.log(message, level=logging.getLevelName(level), component="AutoTrader", **kwargs)
                return
            except Exception:  # pragma: no cover - defensive logging
                LOGGER.log(level, "Emitter logging failed", exc_info=True)
        LOGGER.log(level, message)

    def _run_auto_trade_thread(self) -> None:
        try:
            while not self._auto_trade_stop.is_set() and not self._stop.is_set():
                self._auto_trade_thread_active = True
                try:
                    self._auto_trade_loop()
                finally:
                    self._auto_trade_thread_active = False
                if self._auto_trade_stop.wait(self.auto_trade_interval_s):
                    break
        except Exception:  # pragma: no cover - keep thread resilient
            LOGGER.exception("Auto-trade loop crashed")
        finally:
            self._auto_trade_thread_active = False
            self._auto_trade_stop.set()

    def _start_auto_trade_thread_locked(self) -> None:
        if self._auto_trade_thread is not None and self._auto_trade_thread.is_alive():
            return
        self._auto_trade_stop.clear()
        self._auto_trade_thread = threading.Thread(
            target=self._run_auto_trade_thread,
            name="AutoTraderThread",
            daemon=True,
        )
        self._auto_trade_thread.start()

    def _cancel_auto_trade_thread_locked(self) -> None:
        self._auto_trade_stop.set()
        thread = self._auto_trade_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._auto_trade_thread = None
        self._auto_trade_thread_active = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._stop.clear()
            self._started = True
            if self.enable_auto_trade and not self._auto_trade_user_confirmed:
                self._log("Auto-trade awaiting explicit activation")
            if self.enable_auto_trade and self._auto_trade_user_confirmed:
                self._start_auto_trade_thread_locked()

    def confirm_auto_trade(self, flag: bool) -> None:
        with self._lock:
            self._auto_trade_user_confirmed = bool(flag)
            if not self._started or not self.enable_auto_trade:
                return
            if self._auto_trade_user_confirmed:
                self._start_auto_trade_thread_locked()
            else:
                self._cancel_auto_trade_thread_locked()

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            self._started = False
            self._stop.set()
            self._cancel_auto_trade_thread_locked()
            self._log("AutoTrader stopped.")

    # ------------------------------------------------------------------
    # Extension hook ----------------------------------------------------
    # ------------------------------------------------------------------
    def _auto_trade_loop(self) -> None:
        """Default loop that simply waits until it is stopped.

        Tests override this method with a custom callable, therefore the
        default implementation just waits on the stop event to minimise CPU
        usage.
        """

        self._auto_trade_stop.wait(self.auto_trade_interval_s)

    # Compatibility helpers -------------------------------------------
    def set_enable_auto_trade(self, flag: bool) -> None:
        self.enable_auto_trade = bool(flag)
        if not flag:
            self.confirm_auto_trade(False)

    def is_running(self) -> bool:
        return self._started and not self._stop.is_set()


__all__ = ["AutoTrader", "RiskDecision", "EmitterLike"]
