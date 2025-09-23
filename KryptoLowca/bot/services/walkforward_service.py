# services/walkforward_service.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Iterable, Union, Dict, Any, Deque, List, Tuple

try:
    from event_emitter_adapter import Event, EventType, EventBus
except Exception as e:
    raise ImportError(f"walkforward_service: brak event_emitter_adapter ({e})")

log = logging.getLogger("services.walkforward_service")


@dataclass
class ObjectiveWeights:
    w_pf: float = 1.0
    w_expectancy: float = 1.0
    w_sharpe: float = 0.2
    w_maxdd: float = 0.5   # kara za DD (odejmowana)
    pf_cap: float = 3.0


@dataclass
class WFOServiceConfig:
    symbol: str = "BTCUSDT"
    cooldown_sec: float = 60.0
    auto_apply: bool = True
    obj_weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    # okna WFO liczone po liczbie barów (ticków)
    min_is_bars: int = 1000
    min_oos_bars: int = 300
    step_bars: int = 100
    price_buffer: int = 5000
    # siatka parametrów strategii (MA-cross)
    fast_grid: Tuple[int, ...] = (10, 15, 20, 30, 40)
    slow_grid: Tuple[int, ...] = (40, 60, 80, 100, 120)
    qty_grid: Tuple[float, ...] = (0.01, 0.02)


WalkForwardServiceConfig = WFOServiceConfig  # alias kompatybilności


class WalkForwardService:
    """
    WFO:
      - buforuje ceny z MARKET_TICK,
      - na WFO_TRIGGER/ATR_SPIKE wykonuje grid-search (MA-cross),
      - liczy PF, Expectancy, Sharpe (z PnL), MaxDD na prostym modelu transakcji,
      - publikuje WFO_STATUS i przy auto_apply emituje AUTOTRADE_STATUS(action=strategy_update).
    """
    def __init__(self, bus: EventBus, cfg: WFOServiceConfig) -> None:
        self.bus = bus
        self.cfg = cfg
        self._last_trigger_ts: float = 0.0
        self._price: Deque[float] = deque(maxlen=cfg.price_buffer)

        self.bus.subscribe(EventType.MARKET_TICK, self._on_tick)
        self.bus.subscribe(EventType.WFO_TRIGGER, self._on_trigger)
        self.bus.subscribe(EventType.ATR_SPIKE, self._on_trigger)

        log.info("WalkForwardService ready (symbol=%s, cooldown=%ss)", cfg.symbol, cfg.cooldown_sec or "n/a")

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

    def _cooldown_active(self) -> bool:
        if not self.cfg.cooldown_sec:
            return False
        return (time.time() - self._last_trigger_ts) < float(self.cfg.cooldown_sec)

    # --- data intake -----------------------------------------------------------------------------

    def _on_tick(self, evs):
        for ev in self._iter(evs):
            p = ev.payload or {}
            if p.get("symbol") and p["symbol"] != self.cfg.symbol:
                continue
            px = p.get("price")
            if px is None:
                continue
            self._price.append(float(px))

    # --- triggers --------------------------------------------------------------------------------

    def _on_trigger(self, evs):
        for ev in self._iter(evs):
            if self._cooldown_active():
                self._status("cooldown_active", extra={"ts": time.time()})
                continue
            self._last_trigger_ts = time.time()
            self._status("trigger_received", extra={"ts": self._last_trigger_ts})
            preset = self._run_wfo()
            if preset and self.cfg.auto_apply:
                self._apply_preset(preset)

    # --- core ------------------------------------------------------------------------------------

    @staticmethod
    def _pf(pl: List[float]) -> float:
        gp = sum(x for x in pl if x > 0)
        gl = sum(-x for x in pl if x < 0)
        if gl <= 1e-12:
            return float("inf") if gp > 0 else 1.0
        return gp / gl

    @staticmethod
    def _exp(pl: List[float]) -> float:
        return (sum(pl) / len(pl)) if pl else 0.0

    @staticmethod
    def _sharpe(pl: List[float]) -> float:
        # pseudo-sharpe z PnL transakcyjnego (nie dzienny), bez RF
        if not pl:
            return 0.0
        m = sum(pl) / len(pl)
        var = sum((x - m) ** 2 for x in pl) / max(1, (len(pl) - 1))
        sd = math.sqrt(var)
        return m / sd if sd > 1e-12 else 0.0

    @staticmethod
    def _maxdd(equity: List[float]) -> float:
        mdd = 0.0
        peak = -1e18
        for v in equity:
            if v > peak:
                peak = v
            dd = (peak - v) / max(1e-12, peak) if peak > 0 else 0.0
            mdd = max(mdd, dd)
        return mdd  # w ułamku (0.2 = 20%)

    def _score(self, pf: float, exp: float, sharpe: float, maxdd: float) -> float:
        ow = self.cfg.obj_weights
        pf_c = min(pf, ow.pf_cap)
        return ow.w_pf * pf_c + ow.w_expectancy * exp + ow.w_sharpe * sharpe - ow.w_maxdd * maxdd

    def _backtest_macross(self, prices: List[float], fast: int, slow: int, qty: float) -> Dict[str, float]:
        if slow <= 1 or fast <= 1 or fast >= slow or len(prices) < (slow + 5):
            return {"pf": 1.0, "exp": 0.0, "sharpe": 0.0, "maxdd": 1.0, "trades": 0}
        # EWMAs dla stabilności
        def ewm(prev, v, n):
            alpha = 2.0 / (n + 1.0)
            return v if prev is None else (alpha * v + (1.0 - alpha) * prev)

        f = s = None
        state = 0
        entry_px = 0.0
        pl: List[float] = []
        equity: List[float] = []
        cash = 0.0
        pos = 0.0

        for px in prices:
            f = ewm(f, px, fast)
            s = ewm(s, px, slow)
            if f is None or s is None:
                equity.append(cash + pos * px)
                continue
            bias = 1 if f > s else -1
            # zmiana biasu -> zamknij poprzednią pozycję, otwórz nową
            if state == 0:
                state = bias
                entry_px = px
                pos = qty * state
            elif bias != state:
                # zamykamy starą
                pl.append((px - entry_px) * pos)
                cash += (px - entry_px) * pos
                # otwieramy nową
                state = bias
                entry_px = px
                pos = qty * state
            equity.append(cash + pos * px)

        # zamknij na końcu (symulacyjnie)
        if state != 0 and entry_px != 0.0 and len(prices) > 0:
            px = prices[-1]
            pl.append((px - entry_px) * pos)
            cash += (px - entry_px) * pos
            equity.append(cash)

        pf = self._pf(pl)
        exp = self._exp(pl)
        sharpe = self._sharpe(pl)
        mdd = self._maxdd(equity)
        return {"pf": pf, "exp": exp, "sharpe": sharpe, "maxdd": mdd, "trades": len(pl)}

    def _run_wfo(self) -> Optional[Dict[str, Any]]:
        prices = list(self._price)
        n = len(prices)
        if n < (self.cfg.min_is_bars + self.cfg.min_oos_bars + 5):
            self._status("wfo_skipped_not_enough_data", extra={"bars": n})
            return None

        # IS/OOS split (prosty jednorazowy podział)
        is_len = self.cfg.min_is_bars
        oos_len = self.cfg.min_oos_bars
        is_prices = prices[-(is_len + oos_len):-oos_len]
        oos_prices = prices[-oos_len:]

        best_score = -1e18
        best = None
        tested = 0

        for fast in self.cfg.fast_grid:
            for slow in self.cfg.slow_grid:
                if fast >= slow:
                    continue
                for qty in self.cfg.qty_grid:
                    # sanity: IS musi mieć sensowną liczbę transakcji
                    res_is = self._backtest_macross(is_prices, fast, slow, qty)
                    if res_is["trades"] < max(3, is_len // 200):
                        continue
                    res_oos = self._backtest_macross(oos_prices, fast, slow, qty)
                    score = self._score(res_oos["pf"], res_oos["exp"], res_oos["sharpe"], res_oos["maxdd"])
                    tested += 1
                    if score > best_score:
                        best_score = score
                        best = {
                            "fast_len": fast,
                            "slow_len": slow,
                            "qty": qty,
                            "score": score,
                            "is": res_is,
                            "oos": res_oos
                        }

        self._status("wfo_completed", extra={"tested": tested, "best": best})
        return best

    def _apply_preset(self, preset: Dict[str, Any]) -> None:
        # publikujemy dla StrategyEngine, by zaktualizował parametry
        self.bus.publish(EventType.AUTOTRADE_STATUS, {
            "component": "WFO",
            "action": "strategy_update",
            "symbol": self.cfg.symbol,
            "params": {
                "fast_len": preset.get("fast_len"),
                "slow_len": preset.get("slow_len"),
                "qty": preset.get("qty"),
                "order_cooldown_sec": 10.0
            },
            "ts": time.time()
        })
        self._status("preset_applied", extra={"preset": preset})

    def _status(self, phase: str, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {"symbol": self.cfg.symbol, "phase": phase, "ts": time.time()}
        if extra:
            payload.update(extra)
        self.bus.publish(EventType.WFO_STATUS, payload)
