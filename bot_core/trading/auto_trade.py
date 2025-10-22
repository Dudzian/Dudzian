"""Rozszerzony silnik autotradingu wspierający wiele strategii i reżimy."""
from __future__ import annotations

import datetime as dt
import math
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
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
    RiskLevel,
)
from bot_core.backtest.ma import simulate_trades_ma  # noqa: F401 - zachowaj kompatybilność API
from bot_core.events import DebounceRule, Event, EventBus, EventType, EmitterAdapter


@dataclass
class _AutoRiskFreezeState:
    risk_level: RiskLevel | None = None
    risk_score: float | None = None
    triggered_at: float = 0.0
    last_extension_at: float = 0.0


@dataclass
class _ManualRiskFreezeState:
    reason: str | None = None
    triggered_at: float = 0.0
    last_extension_at: float = 0.0


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
    regime_history_maxlen: int = 5
    regime_history_decay: float = 0.65
    auto_risk_freeze: bool = True
    auto_risk_freeze_level: RiskLevel | str = RiskLevel.CRITICAL
    auto_risk_freeze_score: float = 0.8

    def __post_init__(self) -> None:
        if self.default_params is None:
            self.default_params = {"fast": 10, "slow": 50}
        if self.strategy_weights is None:
            defaults = RegimeStrategyWeights.default()
            self.strategy_weights = {
                regime.value: dict(weights)
                for regime, weights in defaults.weights.items()
            }
        self.regime_history_maxlen = int(self.regime_history_maxlen)
        self.regime_history_decay = float(self.regime_history_decay)
        if self.regime_history_maxlen < 1:
            raise ValueError("regime_history_maxlen must be at least 1")
        if not (0.0 < self.regime_history_decay <= 1.0):
            raise ValueError("regime_history_decay must be in the (0, 1] range")
        self.auto_risk_freeze = bool(self.auto_risk_freeze)
        level = self.auto_risk_freeze_level
        if isinstance(level, str):
            try:
                level = RiskLevel(level.lower())
            except ValueError as exc:  # pragma: no cover - walidacja wejścia
                raise ValueError("auto_risk_freeze_level must be a valid RiskLevel") from exc
        elif not isinstance(level, RiskLevel):
            raise TypeError("auto_risk_freeze_level must be RiskLevel or string")
        self.auto_risk_freeze_level = level
        self.auto_risk_freeze_score = float(self.auto_risk_freeze_score)
        if not (0.0 <= self.auto_risk_freeze_score <= 1.0):
            raise ValueError("auto_risk_freeze_score must be in the [0, 1] range")


class AutoTradeEngine:
    """Prosty kontroler autotradingu reagujący na ticki z EventBusa."""

    _RISK_LEVEL_ORDER = {
        RiskLevel.CALM: 0,
        RiskLevel.BALANCED: 1,
        RiskLevel.WATCH: 2,
        RiskLevel.ELEVATED: 3,
        RiskLevel.CRITICAL: 4,
    }

    def __init__(
        self,
        adapter: EmitterAdapter,
        broker_submit_market,
        cfg: Optional[AutoTradeConfig] = None,
        *,
        regime_classifier: MarketRegimeClassifier | None = None,
        regime_history: RegimeHistory | None = None,
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
        self._manual_risk_frozen_until: float = 0.0
        self._auto_risk_frozen_until: float = 0.0
        self._auto_risk_frozen: bool = False
        self._manual_risk_state = _ManualRiskFreezeState()
        self._auto_risk_state = _AutoRiskFreezeState()
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
        self._manual_risk_frozen_until = 0.0
        self._auto_risk_frozen_until = 0.0
        self._auto_risk_frozen = False
        self._manual_risk_state = _ManualRiskFreezeState()
        self._auto_risk_state = _AutoRiskFreezeState()
        self._recompute_risk_freeze_until()
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

    def _install_regime_components(
        self,
        classifier: MarketRegimeClassifier,
        history: RegimeHistory | None = None,
    ) -> None:
        """Powiąż klasyfikator oraz historię, zapewniając spójną konfigurację progów."""

        loader = getattr(classifier, "thresholds_loader", None)
        if loader is None or not callable(loader):  # pragma: no cover - defensywne strażniki
            raise TypeError("classifier must expose a callable thresholds_loader")
        snapshot_getter = getattr(classifier, "thresholds_snapshot", None)
        if snapshot_getter is None or not callable(snapshot_getter):  # pragma: no cover - strażnik
            raise TypeError("classifier must provide thresholds_snapshot()")

        self._regime_classifier = classifier
        supplied_history = history
        if supplied_history is None:
            existing_history = getattr(self, "_regime_history", None)
            if isinstance(existing_history, RegimeHistory):
                history = existing_history
                history.reconfigure(
                    maxlen=self.cfg.regime_history_maxlen,
                    decay=self.cfg.regime_history_decay,
                    keep_history=True,
                )
                history.reload_thresholds(loader=loader)
            else:
                history = RegimeHistory(
                    thresholds_loader=loader,
                    maxlen=self.cfg.regime_history_maxlen,
                    decay=self.cfg.regime_history_decay,
                )
        else:
            history = supplied_history
            history.reload_thresholds(loader=loader)
        thresholds = snapshot_getter()
        history.reload_thresholds(thresholds=thresholds)
        self._regime_history = history

    def set_regime_components(
        self,
        *,
        classifier: MarketRegimeClassifier,
        history: RegimeHistory | None = None,
        reset_state: bool = True,
    ) -> None:
        """Zastąp aktywny klasyfikator i historię autotradera."""

        target_history = history or getattr(self, "_regime_history", None)
        self._install_regime_components(classifier, target_history)
        if reset_state:
            self._last_regime = None
            self._last_summary = None
            self._regime_history.clear()
            self._auto_risk_frozen = False
            self._auto_risk_frozen_until = 0.0
            self._auto_risk_state = _AutoRiskFreezeState()
            self._recompute_risk_freeze_until()

    def configure_regime_history(
        self,
        *,
        maxlen: int | None = None,
        decay: float | None = None,
        reset: bool = False,
    ) -> None:
        """Zmień parametry wygładzania historii reżimu."""

        update_maxlen = maxlen is not None
        update_decay = decay is not None
        if maxlen is None and decay is None:
            if not reset:
                return
            maxlen = self._regime_history.maxlen
            decay = self._regime_history.decay
        keep_history = not reset
        self._regime_history.reconfigure(
            maxlen=maxlen,
            decay=decay,
            keep_history=keep_history,
        )
        if update_maxlen and maxlen is not None:
            self.cfg.regime_history_maxlen = int(maxlen)
        if update_decay and decay is not None:
            self.cfg.regime_history_decay = float(decay)
        if reset:
            self._last_regime = None
            self._last_summary = None

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
        for ev in events:
            now = time.time()
            payload = ev.payload or {}
            if payload.get("symbol") != self.cfg.symbol:
                continue
            expiry = now + self.cfg.risk_freeze_seconds
            reason_code = str(payload.get("kind") or "risk_alert")
            manual_active = now < self._manual_risk_frozen_until
            previous_until = self._manual_risk_frozen_until if manual_active else 0.0
            if not manual_active:
                self._manual_risk_state = _ManualRiskFreezeState(
                    reason=reason_code,
                    triggered_at=now,
                    last_extension_at=now,
                )
                self._manual_risk_frozen_until = float(expiry)
                detail = {
                    "symbol": self.cfg.symbol,
                    "until": self._manual_risk_frozen_until,
                    "reason": reason_code,
                    "triggered_at": now,
                    "last_extension_at": now,
                    "released_at": None,
                    "frozen_for": None,
                }
                self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                    "risk_freeze",
                    detail=detail,
                    level="WARN",
                )
            else:
                should_extend = expiry > self._manual_risk_frozen_until + 1e-6
                state = self._manual_risk_state
                if should_extend and state is not None:
                    previous_reason = state.reason
                    state.reason = reason_code
                    state.last_extension_at = now
                    self._manual_risk_frozen_until = float(expiry)
                    extend_detail = {
                        "symbol": self.cfg.symbol,
                        "extended_from": previous_until,
                        "until": self._manual_risk_frozen_until,
                        "reason": reason_code,
                        "triggered_at": state.triggered_at or now,
                        "last_extension_at": state.last_extension_at,
                        "released_at": None,
                        "frozen_for": None,
                    }
                    if previous_reason and previous_reason != reason_code:
                        extend_detail["previous_reason"] = previous_reason
                    self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                        "risk_freeze_extend",
                        detail=extend_detail,
                        level="WARN",
                    )
                elif state is not None:
                    state.reason = reason_code
                    state.last_extension_at = now
            self._recompute_risk_freeze_until()

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
        self._sync_freeze_state()
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

    def _manual_risk_unfreeze(
        self,
        *,
        reason: str,
        now: float,
    ) -> None:
        state = self._manual_risk_state
        triggered_at = state.triggered_at if state and state.triggered_at else None
        last_extension_at = state.last_extension_at if state and state.last_extension_at else triggered_at
        detail: Dict[str, Any] = {
            "symbol": self.cfg.symbol,
            "reason": reason,
            "triggered_at": triggered_at,
            "released_at": now,
            "frozen_for": (now - triggered_at) if triggered_at is not None else None,
            "last_extension_at": last_extension_at,
        }
        if state and state.reason:
            detail["source_reason"] = state.reason

        self._manual_risk_frozen_until = 0.0
        self._manual_risk_state = _ManualRiskFreezeState()
        self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
            "risk_unfreeze",
            detail=detail,
        )

    def _auto_risk_unfreeze(
        self,
        *,
        reason: str,
        now: float,
        summary: RegimeSummary | None = None,
        score_value: float | None = None,
    ) -> None:
        state = self._auto_risk_state
        triggered_at = state.triggered_at or now
        last_extension_at = state.last_extension_at or triggered_at
        detail: Dict[str, Any] = {
            "symbol": self.cfg.symbol,
            "reason": reason,
            "triggered_at": triggered_at,
            "released_at": now,
            "frozen_for": max(0.0, now - triggered_at),
            "risk_level": None,
            "risk_score": None,
            "last_extension_at": last_extension_at,
        }
        level_source = summary.risk_level if summary else state.risk_level
        score_source: float | None
        if summary is not None:
            score_source = score_value
        else:
            score_source = state.risk_score
        if level_source is not None:
            detail["risk_level"] = level_source.value
        if score_source is not None:
            detail["risk_score"] = float(score_source)

        self._auto_risk_frozen = False
        self._auto_risk_frozen_until = 0.0
        self._auto_risk_state = _AutoRiskFreezeState()
        self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
            "auto_risk_unfreeze",
            detail=detail,
        )

    def _sync_freeze_state(self) -> None:
        now = time.time()

        if self._manual_risk_frozen_until and now >= self._manual_risk_frozen_until:
            self._manual_risk_unfreeze(reason="expired", now=now)

        auto_until = self._auto_risk_frozen_until if self._auto_risk_frozen else 0.0
        if auto_until and now >= auto_until:
            self._auto_risk_unfreeze(reason="expired", now=now)

        if self.cfg.auto_risk_freeze:
            summary = self._regime_history.summarise()
            triggered = False
            level_rank = -1
            score_value: float | None = None
            trigger_reason: str | None = None
            if summary is not None:
                level_rank = self._RISK_LEVEL_ORDER.get(summary.risk_level, -1)
                target_rank = self._RISK_LEVEL_ORDER.get(self.cfg.auto_risk_freeze_level, 99)
                level_triggered = level_rank >= target_rank >= 0
                score_value = float(summary.risk_score)
                score_triggered = score_value >= self.cfg.auto_risk_freeze_score
                triggered = level_triggered or score_triggered
                if level_triggered and score_triggered:
                    trigger_reason = "risk_level_and_score_threshold"
                elif level_triggered:
                    trigger_reason = "risk_level_threshold"
                elif score_triggered:
                    trigger_reason = "risk_score_threshold"
            if triggered:
                previous_until = self._auto_risk_frozen_until if self._auto_risk_frozen else 0.0
                new_expiry = now + float(self.cfg.risk_freeze_seconds)
                effective_expiry = max(previous_until, new_expiry)
                risk_level_value = summary.risk_level.value if summary else None
                if not self._auto_risk_frozen:
                    new_state = _AutoRiskFreezeState(
                        risk_level=summary.risk_level if summary else None,
                        risk_score=score_value,
                        triggered_at=now,
                        last_extension_at=now,
                    )
                    self._auto_risk_state = new_state
                    self._auto_risk_frozen = True
                    self._auto_risk_frozen_until = effective_expiry
                    detail = {
                        "symbol": self.cfg.symbol,
                        "risk_level": risk_level_value,
                        "risk_score": score_value,
                        "until": effective_expiry,
                        "triggered_at": new_state.triggered_at,
                        "last_extension_at": new_state.last_extension_at,
                        "released_at": None,
                        "frozen_for": None,
                    }
                    if trigger_reason:
                        detail["reason"] = trigger_reason
                    self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                        "auto_risk_freeze",
                        detail=detail,
                        level="WARN",
                    )
                else:
                    state = self._auto_risk_state
                    detail = {
                        "symbol": self.cfg.symbol,
                        "risk_level": risk_level_value,
                        "risk_score": score_value,
                        "until": effective_expiry,
                        "triggered_at": state.triggered_at or now,
                        "last_extension_at": state.last_extension_at
                        or (state.triggered_at or now),
                        "released_at": None,
                        "frozen_for": None,
                    }
                    if trigger_reason:
                        detail["reason"] = trigger_reason
                    extend_reason = None
                    previous_level_rank = self._RISK_LEVEL_ORDER.get(state.risk_level, -1)
                    if summary is not None:
                        if level_rank > previous_level_rank:
                            extend_reason = "risk_level_escalated"
                        elif level_rank == previous_level_rank and score_value is not None:
                            prev_score = state.risk_score if state.risk_score is not None else -math.inf
                            if score_value >= prev_score + 0.05:
                                extend_reason = "risk_score_increase"
                    time_remaining = max(previous_until - now, 0.0)
                    if extend_reason is None and time_remaining <= float(self.cfg.risk_freeze_seconds) * 0.25:
                        extend_reason = "expiry_near"
                    should_extend = effective_expiry > previous_until + 1e-6
                    if should_extend:
                        new_state = _AutoRiskFreezeState(
                            risk_level=summary.risk_level if summary else None,
                            risk_score=score_value,
                            triggered_at=state.triggered_at or now,
                            last_extension_at=now,
                        )
                        self._auto_risk_state = new_state
                        self._auto_risk_frozen_until = effective_expiry
                        if extend_reason:
                            extend_detail = dict(detail)
                            extend_detail["extended_from"] = previous_until
                            extend_detail["until"] = effective_expiry
                            extend_detail["reason"] = extend_reason
                            extend_detail["triggered_at"] = new_state.triggered_at
                            extend_detail["last_extension_at"] = new_state.last_extension_at
                            self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                                "auto_risk_freeze_extend",
                                detail=extend_detail,
                                level="WARN",
                            )
                    else:
                        self._auto_risk_state = _AutoRiskFreezeState(
                            risk_level=summary.risk_level if summary else None,
                            risk_score=score_value,
                            triggered_at=state.triggered_at or now,
                            last_extension_at=state.last_extension_at
                            or (state.triggered_at or now),
                        )
            elif self._auto_risk_frozen:
                recovery_reason = None
                target_rank = self._RISK_LEVEL_ORDER.get(self.cfg.auto_risk_freeze_level, 99)
                below_level = level_rank >= 0 and target_rank >= 0 and level_rank < target_rank
                score_margin = max(0.02, min(0.1, float(self.cfg.auto_risk_freeze_score) * 0.2))
                score_cutoff = max(float(self.cfg.auto_risk_freeze_score) - score_margin, 0.0)
                below_score = score_value is not None and score_value <= score_cutoff
                if below_level and below_score:
                    recovery_reason = "risk_recovered"
                elif below_level:
                    recovery_reason = "risk_level_recovered"
                elif below_score:
                    recovery_reason = "risk_score_recovered"
                if recovery_reason:
                    self._auto_risk_unfreeze(
                        reason=recovery_reason,
                        now=now,
                        summary=summary,
                        score_value=score_value,
                    )

        self._recompute_risk_freeze_until()

    def _recompute_risk_freeze_until(self) -> None:
        now = time.time()
        manual_until = (
            self._manual_risk_frozen_until
            if self._manual_risk_frozen_until and now < self._manual_risk_frozen_until
            else 0.0
        )
        auto_until = (
            self._auto_risk_frozen_until
            if self._auto_risk_frozen and now < self._auto_risk_frozen_until
            else 0.0
        )
        self._risk_frozen_until = float(max(manual_until, auto_until, 0.0))

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
