"""Strategia trend-following i momentum na interwale dziennym."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Mapping, Sequence

from bot_core.strategies.base import MarketSnapshot, StrategyEngine, StrategySignal


@dataclass(slots=True)
class DailyTrendMomentumSettings:
    """Parametry strategii trend/momentum."""

    fast_ma: int = 20
    slow_ma: int = 100
    breakout_lookback: int = 55
    momentum_window: int = 20
    atr_window: int = 14
    atr_multiplier: float = 2.0
    min_trend_strength: float = 0.005
    min_momentum: float = 0.0

    def max_history(self) -> int:
        """Najdłuższe wymagane okno danych."""

        return (
            max(self.slow_ma, self.breakout_lookback, self.momentum_window, self.atr_window)
            + 2
        )


@dataclass(slots=True)
class _SymbolState:
    """Stan strategii dla pojedynczego instrumentu."""

    closes: Deque[float]
    highs: Deque[float]
    lows: Deque[float]
    true_ranges: Deque[float]
    last_close: float | None = None
    in_position: bool = False
    peak_close: float | None = None


class DailyTrendMomentumStrategy(StrategyEngine):
    """Łączy trend-following i momentum na danych dziennych."""

    def __init__(self, settings: DailyTrendMomentumSettings | None = None) -> None:
        self._settings = settings or DailyTrendMomentumSettings()
        self._states: Dict[str, _SymbolState] = {}

    # ------------------------------------------------------------------
    # API strategii
    # ------------------------------------------------------------------
    def warm_up(self, history: Sequence[MarketSnapshot]) -> None:
        for snapshot in history:
            state = self._get_state(snapshot.symbol)
            self._ingest_snapshot(state, snapshot)

    def on_data(self, snapshot: MarketSnapshot) -> Sequence[StrategySignal]:
        state = self._get_state(snapshot.symbol)
        self._ingest_snapshot(state, snapshot)

        if not self._has_sufficient_history(state):
            return []

        metrics = self._calculate_metrics(state, snapshot)
        signals: List[StrategySignal] = []

        if not state.in_position and self._should_enter(metrics, snapshot):
            state.in_position = True
            state.peak_close = snapshot.close
            signals.append(
                StrategySignal(
                    symbol=snapshot.symbol,
                    side="buy",
                    confidence=self._entry_confidence(metrics),
                    metadata=self._build_metadata(metrics, 1.0),
                )
            )
        elif state.in_position and self._should_exit(metrics, snapshot):
            state.in_position = False
            state.peak_close = None
            signals.append(
                StrategySignal(
                    symbol=snapshot.symbol,
                    side="sell",
                    confidence=self._exit_confidence(metrics, snapshot),
                    metadata=self._build_metadata(metrics, 0.0),
                )
            )
        else:
            if state.in_position:
                state.peak_close = max(state.peak_close or snapshot.close, snapshot.close)
        return signals

    # ------------------------------------------------------------------
    # Metody pomocnicze
    # ------------------------------------------------------------------
    def _get_state(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            maxlen = self._settings.max_history()
            self._states[symbol] = _SymbolState(
                closes=deque(maxlen=maxlen),
                highs=deque(maxlen=maxlen),
                lows=deque(maxlen=maxlen),
                true_ranges=deque(maxlen=maxlen),
            )
        return self._states[symbol]

    def _ingest_snapshot(self, state: _SymbolState, snapshot: MarketSnapshot) -> None:
        state.closes.append(snapshot.close)
        state.highs.append(snapshot.high)
        state.lows.append(snapshot.low)
        if state.last_close is None:
            true_range = snapshot.high - snapshot.low
        else:
            true_range = max(
                snapshot.high - snapshot.low,
                abs(snapshot.high - state.last_close),
                abs(snapshot.low - state.last_close),
            )
        state.true_ranges.append(true_range)
        state.last_close = snapshot.close
        if state.in_position:
            state.peak_close = max(state.peak_close or snapshot.close, snapshot.close)

    def _has_sufficient_history(self, state: _SymbolState) -> bool:
        settings = self._settings
        min_required = max(
            settings.slow_ma,
            settings.breakout_lookback + 1,
            settings.momentum_window + 1,
            settings.atr_window,
        )
        return len(state.closes) >= min_required and len(state.true_ranges) >= settings.atr_window

    def _calculate_metrics(
        self, state: _SymbolState, snapshot: MarketSnapshot
    ) -> Mapping[str, float]:
        settings = self._settings

        closes = list(state.closes)
        highs = list(state.highs)
        lows = list(state.lows)
        trs = list(state.true_ranges)

        sma_fast = sum(closes[-settings.fast_ma :]) / settings.fast_ma
        sma_slow = sum(closes[-settings.slow_ma :]) / settings.slow_ma
        trend_strength = (sma_fast / sma_slow) - 1.0 if sma_slow else 0.0

        base_index = -(settings.momentum_window + 1)
        momentum_base = closes[base_index]
        momentum = (snapshot.close / momentum_base) - 1.0 if momentum_base else 0.0

        atr = sum(trs[-settings.atr_window :]) / settings.atr_window
        recent_highs = highs[-(settings.breakout_lookback + 1) : -1]
        recent_lows = lows[-(settings.breakout_lookback + 1) : -1]
        breakout_high = max(recent_highs) if recent_highs else snapshot.high
        breakout_low = min(recent_lows) if recent_lows else snapshot.low

        peak_close = state.peak_close if state.peak_close is not None else snapshot.close
        stop_price = peak_close - atr * settings.atr_multiplier

        return {
            "sma_fast": sma_fast,
            "sma_slow": sma_slow,
            "trend_strength": trend_strength,
            "momentum": momentum,
            "atr": atr,
            "breakout_high": breakout_high,
            "breakout_low": breakout_low,
            "stop_price": stop_price,
            "close": snapshot.close,
        }

    def _should_enter(self, metrics: Mapping[str, float], snapshot: MarketSnapshot) -> bool:
        settings = self._settings
        price_breakout = snapshot.close >= metrics["breakout_high"]
        trend_ok = metrics["trend_strength"] >= settings.min_trend_strength
        momentum_ok = metrics["momentum"] >= settings.min_momentum
        return price_breakout and trend_ok and momentum_ok

    def _should_exit(self, metrics: Mapping[str, float], snapshot: MarketSnapshot) -> bool:
        trend_reversal = metrics["trend_strength"] < 0.0
        stop_hit = snapshot.close <= metrics["stop_price"]
        breakout_failure = snapshot.close <= metrics["breakout_low"]
        return trend_reversal or stop_hit or breakout_failure

    def _entry_confidence(self, metrics: Mapping[str, float]) -> float:
        trend_component = max(0.0, metrics["trend_strength"] - self._settings.min_trend_strength)
        momentum_component = max(0.0, metrics["momentum"] - self._settings.min_momentum)
        confidence = 0.35 + 2.0 * trend_component + 1.5 * momentum_component
        return max(0.1, min(1.0, confidence))

    def _exit_confidence(self, metrics: Mapping[str, float], snapshot: MarketSnapshot) -> float:
        stop_distance = max(0.0, metrics["stop_price"] - snapshot.close) / max(snapshot.close, 1e-8)
        trend_component = max(0.0, -metrics["trend_strength"])
        confidence = 0.35 + 3.0 * stop_distance + 2.0 * trend_component
        return max(0.1, min(1.0, confidence))

    def _build_metadata(self, metrics: Mapping[str, float], position_flag: float) -> Mapping[str, float]:
        return {
            "sma_fast": metrics["sma_fast"],
            "sma_slow": metrics["sma_slow"],
            "trend_strength": metrics["trend_strength"],
            "momentum": metrics["momentum"],
            "atr": metrics["atr"],
            "breakout_high": metrics["breakout_high"],
            "breakout_low": metrics["breakout_low"],
            "stop_price": metrics["stop_price"],
            "position": position_flag,
        }


__all__ = ["DailyTrendMomentumSettings", "DailyTrendMomentumStrategy"]
