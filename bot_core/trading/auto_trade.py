"""Rozszerzony silnik autotradingu wspierający wiele strategii i reżimy."""
from __future__ import annotations

import datetime as dt
import math
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
    RiskLevel,
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
        self._manual_freeze_reason: Optional[str] = None
        self._submit_market = broker_submit_market
        self._strategy_weights = RegimeStrategyWeights(
            weights={
                MarketRegime(regime): dict(weights)
                for regime, weights in self._normalize_strategy_config(self.cfg.strategy_weights).items()
            }
        )
        self._last_regime: MarketRegimeAssessment | None = None
        self._last_summary: RegimeSummary | None = None
        self._install_regime_components(
            regime_classifier or MarketRegimeClassifier(),
            regime_history,
        )

        batch_rule = DebounceRule(window=0.1, max_batch=1)
        self.bus.subscribe(EventType.MARKET_TICK, self._on_ticks_batch, rule=batch_rule)
        self.bus.subscribe(EventType.WFO_STATUS, self._on_wfo_status_batch, rule=batch_rule)
        self.bus.subscribe(EventType.RISK_ALERT, self._on_risk_alert_batch, rule=batch_rule)

    def enable(self) -> None:
        self._enabled = True
        self._manual_risk_frozen_until = 0.0
        self._auto_risk_frozen_until = 0.0
        self._auto_risk_frozen = False
        self._manual_freeze_reason = None
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
        now = time.time()
        for ev in events:
            payload = ev.payload or {}
            if payload.get("symbol") != self.cfg.symbol:
                continue
            self.freeze_trading(
                self.cfg.risk_freeze_seconds,
                reason=payload.get("kind", "risk_alert"),
                source="risk_alert",
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
        if summary is not None:
            self._handle_auto_risk_freeze(summary)
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

    def _recompute_risk_freeze_until(self) -> None:
        self._risk_frozen_until = max(self._manual_risk_frozen_until, self._auto_risk_frozen_until)

    def _sync_freeze_state(self, *, now: float | None = None) -> None:
        """Zaktualizuj stan zamrożenia po naturalnym wygaśnięciu blokad."""

        current_time = time.time() if now is None else float(now)
        manual_expired = self._manual_risk_frozen_until > 0.0 and current_time >= self._manual_risk_frozen_until
        auto_expired = self._auto_risk_frozen and current_time >= self._auto_risk_frozen_until > 0.0

        if not manual_expired and not auto_expired:
            return

        if manual_expired:
            self._manual_risk_frozen_until = 0.0
            self._manual_freeze_reason = None

        if auto_expired:
            self._auto_risk_frozen = False
            self._auto_risk_frozen_until = 0.0

        self._recompute_risk_freeze_until()

        if manual_expired:
            self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                "risk_freeze_release",
                detail={
                    "symbol": self.cfg.symbol,
                    "reason": "manual_expired",
                    "source": "manual",
                    "until": self._risk_frozen_until,
                },
                level="INFO",
            )

        if auto_expired:
            detail = {
                "symbol": self.cfg.symbol,
                "reason": "auto_risk_expired",
                "source": "auto",
                "until": self._risk_frozen_until,
            }
            if self._last_summary is not None:
                detail["risk_level"] = self._last_summary.risk_level.value
                detail["risk_score"] = self._last_summary.risk_score
            self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                "risk_freeze_release",
                detail=detail,
                level="INFO",
            )

    def is_risk_frozen(self, *, now: float | None = None) -> bool:
        """Sprawdź czy handel jest obecnie zamrożony z powodu ryzyka."""

        current_time = time.time() if now is None else float(now)
        self._sync_freeze_state(now=current_time)
        return current_time < self._risk_frozen_until

    def get_risk_freeze_state(self) -> Mapping[str, Any]:
        """Zwróć defensywną migawkę stanu zamrożenia ryzyka."""

        current_time = time.time()
        self._sync_freeze_state(now=current_time)
        return {
            "frozen": current_time < self._risk_frozen_until,
            "until": self._risk_frozen_until,
            "manual_until": self._manual_risk_frozen_until,
            "auto_until": self._auto_risk_frozen_until,
            "auto_active": self._auto_risk_frozen,
            "manual_reason": self._manual_freeze_reason,
        }

    @staticmethod
    def _normalize_freeze_duration(duration: object) -> float:
        if hasattr(duration, "total_seconds"):
            seconds = float(duration.total_seconds())  # type: ignore[attr-defined]
        else:
            try:
                seconds = float(duration)  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:  # pragma: no cover - sanity
                raise TypeError("duration must be a number, timedelta or None") from exc
        if not math.isfinite(seconds):
            raise ValueError("duration must be a finite positive number")
        return seconds

    @staticmethod
    def _normalize_freeze_until(value: object) -> float:
        if isinstance(value, (int, float)):
            timestamp = float(value)
        elif isinstance(value, np.datetime64):
            timestamp = float(pd.Timestamp(value).timestamp())
        elif isinstance(value, pd.Timestamp):
            timestamp = AutoTradeEngine._normalize_freeze_until(value.to_pydatetime())
        elif isinstance(value, dt.datetime):
            if value.tzinfo is None:
                timestamp = value.replace(tzinfo=dt.timezone.utc).timestamp()
            else:
                timestamp = value.timestamp()
        elif hasattr(value, "timestamp"):
            try:
                timestamp = float(value.timestamp())  # type: ignore[attr-defined]
            except (TypeError, ValueError) as exc:  # pragma: no cover - sanity
                raise TypeError("until must be a datetime, timestamp or datetime-like object") from exc
        elif hasattr(value, "to_pydatetime"):
            timestamp = AutoTradeEngine._normalize_freeze_until(value.to_pydatetime())
        else:
            raise TypeError("until must be a datetime, timestamp or datetime-like object")

        if not math.isfinite(timestamp):
            raise ValueError("until must be a finite timestamp")
        return timestamp

    def freeze_trading(
        self,
        duration: float | None = None,
        *,
        until: object | None = None,
        reason: str = "manual_freeze",
        source: str = "manual",
    ) -> None:
        """Aktywnie zamroź handel na zadany czas."""

        if duration is not None and until is not None:
            raise ValueError("specify either duration or until, not both")

        now = time.time()
        if until is not None:
            expiry = self._normalize_freeze_until(until)
            if expiry <= now:
                raise ValueError("until must be in the future")
        else:
            if duration is None:
                duration_value = float(self.cfg.risk_freeze_seconds)
            else:
                duration_value = self._normalize_freeze_duration(duration)
            if duration_value <= 0:
                raise ValueError("duration must be positive")
            expiry = now + duration_value
        previous_until = self._manual_risk_frozen_until
        previous_reason = self._manual_freeze_reason
        state_changed = False

        if expiry >= previous_until - 1e-9:
            state_changed = (
                previous_until <= 0.0
                or expiry > previous_until + 1e-9
                or previous_reason != reason
            )
            self._manual_risk_frozen_until = expiry
            self._manual_freeze_reason = reason

        self._recompute_risk_freeze_until()

        if not state_changed:
            return

        self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
            "risk_freeze",
            detail={
                "symbol": self.cfg.symbol,
                "until": self._risk_frozen_until,
                "reason": reason,
                "source": source,
            },
            level="WARN",
        )

    def release_manual_freeze(self) -> None:
        """Zwolnij manualne zamrożenie handlu."""

        previous_until = self._manual_risk_frozen_until
        if previous_until <= 0.0:
            return
        self._manual_risk_frozen_until = 0.0
        self._manual_freeze_reason = None
        self._recompute_risk_freeze_until()
        if time.time() <= previous_until:
            self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                "risk_freeze_release",
                detail={
                    "symbol": self.cfg.symbol,
                    "reason": "manual_release",
                    "source": "manual",
                    "until": self._risk_frozen_until,
                },
                level="INFO",
            )

    def _should_auto_freeze(self, summary: RegimeSummary) -> bool:
        if not self.cfg.auto_risk_freeze:
            return False
        level_threshold = self.cfg.auto_risk_freeze_level
        summary_level = summary.risk_level
        if not isinstance(level_threshold, RiskLevel):  # pragma: no cover - sanity
            level_threshold = RiskLevel(level_threshold)
        level_check = (
            self._RISK_LEVEL_ORDER[summary_level]
            >= self._RISK_LEVEL_ORDER[level_threshold]
        )
        score_check = summary.risk_score >= self.cfg.auto_risk_freeze_score
        return level_check or score_check

    def _handle_auto_risk_freeze(self, summary: RegimeSummary) -> None:
        if not self.cfg.auto_risk_freeze:
            if self._auto_risk_frozen:
                self._auto_risk_frozen = False
                self._auto_risk_frozen_until = 0.0
                self._recompute_risk_freeze_until()
            return

        now = time.time()
        should_freeze = self._should_auto_freeze(summary)
        if should_freeze:
            expiry = now + self.cfg.risk_freeze_seconds
            previous_auto_until = self._auto_risk_frozen_until
            if expiry > previous_auto_until:
                self._auto_risk_frozen_until = expiry
                self._recompute_risk_freeze_until()
            if not self._auto_risk_frozen or expiry > previous_auto_until + 1e-9:
                self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                    "risk_freeze",
                    detail={
                        "symbol": self.cfg.symbol,
                        "until": self._risk_frozen_until,
                        "reason": "auto_risk_freeze",
                        "source": "auto",
                        "risk_level": summary.risk_level.value,
                        "risk_score": summary.risk_score,
                    },
                    level="WARN",
                )
            self._auto_risk_frozen = True
            return

        if self._auto_risk_frozen:
            self._auto_risk_frozen = False
            self._auto_risk_frozen_until = 0.0
            self._recompute_risk_freeze_until()
            self.adapter.push_autotrade_status(  # type: ignore[attr-defined]
                "risk_freeze_release",
                detail={
                    "symbol": self.cfg.symbol,
                    "reason": "auto_risk_release",
                    "source": "auto",
                    "risk_level": summary.risk_level.value,
                    "risk_score": summary.risk_score,
                    "until": self._risk_frozen_until,
                },
                level="INFO",
            )


__all__ = ["AutoTradeConfig", "AutoTradeEngine"]
