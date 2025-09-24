# services/walkforward_service.py
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from KryptoLowca.event_emitter_adapter import (
    EventBus,
    EmitterAdapter,
    Event,
    EventType,
)
from backtest.walkforward import (
    Bar,
    WalkForwardConfig,
    ObjectiveWeights,
    WalkForwardResult,
    run_walkforward,
)


# ======================================
# Konfiguracja i provider'y
# ======================================

StrategyFn = Callable[[List[Bar], Dict[str, Any]], List[Dict[str, Any]]]
BarsProvider = Callable[[str], List[Bar]]
GridProvider = Callable[[str], List[Dict[str, Any]]]
ApplyCallback = Callable[[str, Dict[str, Any]], None]  # (symbol, params) -> zastosuj w silniku


@dataclass
class WalkForwardServiceConfig:
    auto_on_trigger: bool = True           # reaguj na EventType.WFO_TRIGGER i RISK_ALERT
    min_seconds_between_runs: int = 300    # anty-zalewanie (per symbol)
    announce_progress: bool = True         # emituj statusy 'progress'
    wf_cfg: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    obj_weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)


# ======================================
# Serwis
# ======================================

class WalkForwardService:
    """
    Serwis WFO:
     - subskrybuje trigger'y z EventBus (opcjonalnie)
     - uruchamia WFO w tle
     - publikuje EventType.WFO_STATUS (started/progress/completed/error)
     - opcjonalnie wywołuje apply_callback(...) ze zwycięskimi parametrami
    """
    def __init__(
        self,
        adapter: EmitterAdapter,
        *,
        data_provider: BarsProvider,
        strategy_fn: StrategyFn,
        grid_provider: GridProvider,
        apply_callback: Optional[ApplyCallback] = None,
        cfg: Optional[WalkForwardServiceConfig] = None,
    ) -> None:
        self.adapter = adapter
        self.bus: EventBus = adapter.bus
        self.data_provider = data_provider
        self.strategy_fn = strategy_fn
        self.grid_provider = grid_provider
        self.apply_callback = apply_callback
        self.cfg = cfg or WalkForwardServiceConfig()

        self._lock = threading.Lock()
        self._active_jobs: Dict[str, str] = {}   # symbol -> job_id
        self._last_run_ts: Dict[str, float] = {} # symbol -> epoch seconds

        if self.cfg.auto_on_trigger:
            self._attach_bus()

    # ---------- Subskrypcje ----------

    def _attach_bus(self) -> None:
        def on_events(batch: List[Event]) -> None:
            for ev in batch:
                try:
                    if ev.etype == EventType.WFO_TRIGGER:
                        symbol = ev.payload.get("symbol") or ev.payload.get("context", {}).get("symbol") or "BTCUSDT"
                        self.reoptimize_async(symbol=symbol, reason=ev.payload.get("reason", "external_trigger"))
                    elif ev.etype == EventType.RISK_ALERT:
                        symbol = ev.payload.get("symbol", "BTCUSDT")
                        self.reoptimize_async(symbol=symbol, reason=ev.payload.get("kind", "risk_alert"))
                except Exception:
                    # nie blokuj busa
                    pass

        # krótki debounce; EventBus dostarczy listy
        from KryptoLowca.event_emitter_adapter import DebounceRule
        self.bus.subscribe(EventType.WFO_TRIGGER, on_events, rule=DebounceRule(window_sec=0.2, max_batch=50))
        self.bus.subscribe(EventType.RISK_ALERT, on_events, rule=DebounceRule(window_sec=0.2, max_batch=50))

    # ---------- Public API ----------

    def reoptimize_async(self, *, symbol: str, reason: str) -> Optional[str]:
        now = time.time()
        with self._lock:
            if symbol in self._active_jobs:
                # już trwa
                self.adapter.push_wfo_status("busy", detail={"symbol": symbol, "job_id": self._active_jobs[symbol]})
                return None
            last = self._last_run_ts.get(symbol, 0.0)
            if (now - last) < self.cfg.min_seconds_between_runs:
                self.adapter.push_wfo_status(
                    "throttled",
                    detail={"symbol": symbol, "wait_sec": int(self.cfg.min_seconds_between_runs - (now - last))},
                )
                return None
            job_id = uuid.uuid4().hex[:8]
            self._active_jobs[symbol] = job_id
            self._last_run_ts[symbol] = now

        t = threading.Thread(target=self._worker, name=f"WFO-{symbol}-{job_id}", args=(symbol, job_id, reason), daemon=True)
        t.start()
        return job_id

    # ---------- Pracownik ----------

    def _worker(self, symbol: str, job_id: str, reason: str) -> None:
        self.adapter.push_wfo_status("started", detail={"symbol": symbol, "job_id": job_id, "reason": reason})
        try:
            bars = self.data_provider(symbol)
            grid = self.grid_provider(symbol)
            if not isinstance(bars, list) or not isinstance(grid, list) or not grid:
                raise ValueError("Brak danych (bars) lub param_grid.")

            # (opcjonalny) progres — przy dużych gridach/utilach można emitować co X kroków.
            result: WalkForwardResult = run_walkforward(
                bars=bars,
                param_grid=grid,
                strategy_fn=self.strategy_fn,
                cfg=self.cfg.wf_cfg,
                weights=self.cfg.obj_weights,
            )

            best = result.best_params_global
            summary = result.oos_summary()

            detail = {
                "symbol": symbol,
                "job_id": job_id,
                "result": {
                    "windows": [
                        {
                            "is": {"i0": w.i0, "i1": w.i1, "metrics": w.is_metrics, "score": w.score_is},
                            "oos": {"j0": w.j0, "j1": w.j1, "metrics": w.oos_metrics, "score": w.score_oos},
                            "params": w.params_best,
                        }
                        for w in result.windows
                    ],
                    "oos_summary": summary,
                    "best_params_global": best,
                },
            }

            self.adapter.push_wfo_status("completed", detail=detail)

            if best and self.apply_callback:
                try:
                    self.apply_callback(symbol, best)
                    # Poinformuj GUI, że zastosowano:
                    self.adapter.push_wfo_status("applied", detail={"symbol": symbol, "job_id": job_id, "params": best})
                    # Możesz też poinformować logiem:
                    self.adapter.push_log(f"WFO: zastosowano nowe parametry dla {symbol}.", level="INFO")
                except Exception as e:
                    self.adapter.push_wfo_status("apply_error", detail={"symbol": symbol, "job_id": job_id, "error": str(e)})

        except Exception as e:
            self.adapter.push_wfo_status("error", detail={"symbol": symbol, "job_id": job_id, "error": str(e)})
        finally:
            with self._lock:
                self._active_jobs.pop(symbol, None)
