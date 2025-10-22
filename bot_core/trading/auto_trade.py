"""Rozszerzony silnik autotradingu wspierający wiele strategii i reżimy."""
from __future__ import annotations

import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from bot_core.ai.regime import (
    MarketRegime,
    MarketRegimeAssessment,
    MarketRegimeClassifier,
    RegimeHistory,
    RegimeStrategyWeights,
    RegimeSummary,
)
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
    strategy_weights: Mapping[str, Mapping[str, float]] | None = None
    regime_window: int = 60
    activation_threshold: float = 0.2
    breakout_window: int = 24
    mean_reversion_window: int = 20
    mean_reversion_z: float = 1.25

    def __post_init__(self) -> None:
        if self.default_params is None:
            self.default_params = {"fast": 10, "slow": 50}
        if self.strategy_weights is None:
            defaults = RegimeStrategyWeights.default()
            self.strategy_weights = {
                regime.value: dict(weights)
                for regime, weights in defaults.weights.items()
            }


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
        self._bars: Deque[Mapping[str, float]] = deque(maxlen=max(self.cfg.regime_window * 3, 200))
        self._params = dict(self.cfg.default_params)
        self._last_signal: Optional[int] = None
        self._enabled: bool = True
        self._risk_frozen_until: float = 0.0
        self._submit_market = broker_submit_market
        self._regime_classifier = MarketRegimeClassifier()
        self._regime_history = RegimeHistory(
            thresholds_loader=self._regime_classifier.thresholds_loader
        )
        self._regime_history.reload_thresholds(
            thresholds=self._regime_classifier.thresholds_snapshot()
        )
        self._strategy_weights = RegimeStrategyWeights(
            weights={
                MarketRegime(regime): dict(weights)
                for regime, weights in self._normalize_strategy_config(self.cfg.strategy_weights).items()
            }
        )
        self._last_regime: MarketRegimeAssessment | None = None
        self._last_summary: RegimeSummary | None = None

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

    @staticmethod
    def _normalize_strategy_config(
        raw: Mapping[str, Mapping[str, float]] | None
    ) -> Dict[MarketRegime, Dict[str, float]]:
        if raw is None:
            defaults = RegimeStrategyWeights.default()
            return {regime: dict(weights) for regime, weights in defaults.weights.items()}
        normalized: Dict[MarketRegime, Dict[str, float]] = {}
        for regime_name, weights in raw.items():
            try:
                regime = MarketRegime(regime_name)
            except ValueError:
                if isinstance(regime_name, str):
                    try:
                        regime = MarketRegime(regime_name.lower())
                    except ValueError:
                        regime = MarketRegime.TREND
                else:
                    regime = MarketRegime.TREND
            normalized[regime] = {str(name): float(value) for name, value in weights.items()}
        return normalized

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
            high = float(bar.get("high", px))
            low = float(bar.get("low", px))
            volume = float(bar.get("volume", bar.get("quoteVolume", 0.0) or 0.0))
            timestamp = float(bar.get("open_time") or payload.get("timestamp") or time.time())
            self._closes.append(px)
            self._bars.append(
                {
                    "timestamp": timestamp,
                    "close": px,
                    "high": high,
                    "low": low,
                    "volume": volume,
                }
            )
            self._maybe_trade()

    def _maybe_trade(self) -> None:
        closes = self._closes
        if len(closes) < max(self._params.get("fast", 10), self._params.get("slow", 50)) + 2:
            return
        if not self._enabled:
            return
        if time.time() < self._risk_frozen_until:
            return
        if len(self._bars) < self.cfg.regime_window:
            return
        frame = pd.DataFrame(list(self._bars)[-self.cfg.regime_window :])
        regime = self._classify_regime(frame)
        weights = self._strategy_weights.weights_for(regime.regime)
        signals = {
            "trend_following": float(self._trend_following_signal(closes)),
            "daily_breakout": float(self._daily_breakout_signal(frame)),
            "mean_reversion": float(self._mean_reversion_signal(closes)),
        }
        numerator = sum(weights.get(name, 0.0) * signals[name] for name in signals)
        denominator = sum(abs(weights.get(name, 0.0)) for name in signals)
        combined = numerator / denominator if denominator else 0.0
        if self.cfg.emit_signals:
            self.adapter.publish(
                EventType.SIGNAL,
                {
                    "symbol": self.cfg.symbol,
                    "direction": combined,
                    "params": dict(self._params),
                    "regime": regime.regime.value,
                    "weights": weights,
                    "signals": signals,
                },
            )
        direction = 0
        if combined > self.cfg.activation_threshold:
            direction = +1
        elif combined < -self.cfg.activation_threshold:
            direction = -1
        if self._last_signal is None:
            self._last_signal = 0
        if direction > 0 and self._last_signal <= 0:
            self._submit_market("buy", self.cfg.qty)
            self._last_signal = +1
            self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                "entry_long",
                detail={"symbol": self.cfg.symbol, "qty": self.cfg.qty, "regime": regime.to_dict()},
            )
        elif direction < 0 and self._last_signal >= 0:
            self._submit_market("sell", self.cfg.qty)
            self._last_signal = -1
            self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                "entry_short",
                detail={"symbol": self.cfg.symbol, "qty": self.cfg.qty, "regime": regime.to_dict()},
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

    def _trend_following_signal(self, closes: List[float]) -> int:
        fast = int(self._params.get("fast", 10))
        slow = int(self._params.get("slow", 50))
        signal = self._last_cross_signal(closes, fast, slow)
        return 0 if signal is None else signal

    def _daily_breakout_signal(self, frame: pd.DataFrame) -> int:
        window = max(2, int(self.cfg.breakout_window))
        if frame.empty or len(frame) < window:
            return 0
        recent = frame.tail(window)
        high = float(recent["high"].max())
        low = float(recent["low"].min())
        last_close = float(frame["close"].iloc[-1])
        if last_close >= high * 0.999:
            return +1
        if last_close <= low * 1.001:
            return -1
        return 0

    def _mean_reversion_signal(self, closes: List[float]) -> int:
        window = max(3, int(self.cfg.mean_reversion_window))
        if len(closes) < window:
            return 0
        subset = np.asarray(closes[-window:], dtype=float)
        mean = float(subset.mean())
        std = float(subset.std())
        if std == 0.0:
            return 0
        zscore = (subset[-1] - mean) / std
        if zscore > self.cfg.mean_reversion_z:
            return -1
        if zscore < -self.cfg.mean_reversion_z:
            return +1
        return 0

    def _classify_regime(self, frame: pd.DataFrame) -> MarketRegimeAssessment:
        if frame.empty:
            if self._last_regime is None:
                metrics: Dict[str, float] = {}
                self._last_regime = MarketRegimeAssessment(
                    regime=MarketRegime.TREND,
                    confidence=0.0,
                    risk_score=0.0,
                    metrics=metrics,
                    symbol=self.cfg.symbol,
                )
            return self._last_regime
        try:
            assessment = self._regime_classifier.assess(frame, symbol=self.cfg.symbol)
        except ValueError:
            if self._last_regime is not None:
                return self._last_regime
            assessment = MarketRegimeAssessment(
                regime=MarketRegime.TREND,
                confidence=0.0,
                risk_score=0.0,
                metrics={},
                symbol=self.cfg.symbol,
            )
        self._regime_history.reload_thresholds(
            thresholds=self._regime_classifier.thresholds_snapshot()
        )
        self._regime_history.update(assessment)
        summary = self._regime_history.summarise()
        should_emit = self._last_regime is None or (
            self._last_regime.regime != assessment.regime
        )
        if not should_emit and summary is not None and self._last_summary is not None:
            if summary.risk_level != self._last_summary.risk_level:
                should_emit = True
            else:
                risk_delta = abs(summary.risk_score - self._last_summary.risk_score)
                if risk_delta >= 0.1:
                    should_emit = True
        if should_emit:
            detail = assessment.to_dict()
            if summary is not None:
                detail["summary"] = summary.to_dict()
                detail["thresholds"] = self._regime_history.thresholds_snapshot()
            self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                "regime_update",
                detail=detail,
            )
        self._last_regime = assessment
        if summary is not None:
            self._last_summary = summary
        return assessment

    def get_last_regime_assessment(self) -> MarketRegimeAssessment | None:
        """Zwróć ostatnią ocenę reżimu (bez możliwości modyfikacji stanu)."""

        if self._last_regime is None:
            return None
        return deepcopy(self._last_regime)

    def get_regime_summary(self) -> RegimeSummary | None:
        """Zwróć wygładzoną historię reżimu jako kopię defensywną."""

        summary = self._regime_history.summarise()
        if summary is None:
            return None
        return deepcopy(summary)

    def get_regime_thresholds(self) -> Mapping[str, Any]:
        """Udostępnij aktualnie aktywne progi klasyfikatora."""

        return self._regime_history.thresholds_snapshot()


__all__ = ["AutoTradeConfig", "AutoTradeEngine"]
