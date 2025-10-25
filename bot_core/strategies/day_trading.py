"""Intraday momentum/day-trading strategy engine."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import mean
from typing import Any, Deque, Dict, List, Mapping, Sequence

from bot_core.strategies.base import MarketSnapshot, StrategyEngine, StrategySignal
from bot_core.trading.exit_reasons import ExitReason


@dataclass(slots=True)
class DayTradingSettings:
    """Configuration values for the intraday momentum engine."""

    momentum_window: int = 6
    volatility_window: int = 10
    entry_threshold: float = 0.75
    exit_threshold: float = 0.25
    take_profit_atr: float = 2.0
    stop_loss_atr: float = 2.5
    max_holding_bars: int = 12
    atr_floor: float = 0.0005
    bias_strength: float = 0.2

    def __post_init__(self) -> None:
        self.momentum_window = self._ensure_positive_int(self.momentum_window, "momentum_window")
        self.volatility_window = self._ensure_positive_int(self.volatility_window, "volatility_window")
        self.max_holding_bars = self._ensure_positive_int(self.max_holding_bars, "max_holding_bars")
        self.entry_threshold = self._ensure_positive_float(self.entry_threshold, "entry_threshold")
        self.exit_threshold = self._ensure_positive_float(self.exit_threshold, "exit_threshold")
        if self.exit_threshold >= self.entry_threshold:
            raise ValueError("exit_threshold must be lower than entry_threshold")
        self.take_profit_atr = self._ensure_positive_float(self.take_profit_atr, "take_profit_atr")
        self.stop_loss_atr = self._ensure_positive_float(self.stop_loss_atr, "stop_loss_atr")
        self.atr_floor = self._ensure_positive_float(self.atr_floor, "atr_floor")
        if not 0.0 <= self.bias_strength <= 1.0:
            raise ValueError("bias_strength must be in the range [0, 1]")

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any] | None = None) -> "DayTradingSettings":
        params = dict(parameters or {})
        defaults = cls()
        return cls(
            momentum_window=int(params.get("momentum_window", defaults.momentum_window)),
            volatility_window=int(params.get("volatility_window", defaults.volatility_window)),
            entry_threshold=float(params.get("entry_threshold", defaults.entry_threshold)),
            exit_threshold=float(params.get("exit_threshold", defaults.exit_threshold)),
            take_profit_atr=float(params.get("take_profit_atr", defaults.take_profit_atr)),
            stop_loss_atr=float(params.get("stop_loss_atr", defaults.stop_loss_atr)),
            max_holding_bars=int(params.get("max_holding_bars", defaults.max_holding_bars)),
            atr_floor=float(params.get("atr_floor", defaults.atr_floor)),
            bias_strength=float(params.get("bias_strength", defaults.bias_strength)),
        )

    @staticmethod
    def _ensure_positive_int(value: int, field: str) -> int:
        value = int(value)
        if value < 1:
            raise ValueError(f"{field} must be at least 1")
        return value

    @staticmethod
    def _ensure_positive_float(value: float, field: str) -> float:
        value = float(value)
        if value <= 0:
            raise ValueError(f"{field} must be positive")
        return value


@dataclass(slots=True)
class _DayTradingState:
    closes: Deque[float]
    atrs: Deque[float]
    position: str | None = None
    entry_price: float | None = None
    entry_atr: float = 0.0
    bars_in_position: int = 0


class DayTradingStrategy(StrategyEngine):
    """Intraday engine reacting to short-lived momentum bursts."""

    def __init__(self, settings: DayTradingSettings | None = None) -> None:
        self._settings = settings or DayTradingSettings()
        window = max(self._settings.momentum_window + 1, self._settings.volatility_window)
        self._states: Dict[str, _DayTradingState] = {}
        self._history_size = max(window, 2)

    def warm_up(self, history: Sequence[MarketSnapshot]) -> None:
        for snapshot in history:
            state = self._ensure_state(snapshot.symbol)
            self._update_state(state, snapshot)

    def on_data(self, snapshot: MarketSnapshot) -> Sequence[StrategySignal]:
        state = self._ensure_state(snapshot.symbol)
        self._update_state(state, snapshot)

        if len(state.closes) < 2:
            return []

        signals: List[StrategySignal] = []
        momentum_score = self._momentum_score(state)
        volatility = self._volatility(state)
        signal_strength = momentum_score / max(volatility, self._settings.atr_floor)
        bias = self._intraday_bias(snapshot)
        adjusted_strength = signal_strength + bias * self._settings.bias_strength

        if state.position is None:
            if adjusted_strength >= self._settings.entry_threshold:
                self._open_position(state, price=snapshot.close, atr=volatility, direction="long")
                signals.append(
                    self._build_signal(
                        snapshot,
                        side="buy",
                        confidence=min(1.0, adjusted_strength / self._settings.entry_threshold),
                        strength=adjusted_strength,
                        volatility=volatility,
                        direction="long",
                    )
                )
            elif adjusted_strength <= -self._settings.entry_threshold:
                self._open_position(state, price=snapshot.close, atr=volatility, direction="short")
                signals.append(
                    self._build_signal(
                        snapshot,
                        side="sell",
                        confidence=min(1.0, abs(adjusted_strength) / self._settings.entry_threshold),
                        strength=adjusted_strength,
                        volatility=volatility,
                        direction="short",
                    )
                )
            return signals

        assert state.entry_price is not None
        pnl_ratio = (snapshot.close - state.entry_price) / state.entry_price
        if state.position == "short":
            pnl_ratio = -pnl_ratio

        state.bars_in_position += 1
        exit_reason = None
        if pnl_ratio >= self._settings.take_profit_atr * state.entry_atr:
            exit_reason = ExitReason.TAKE_PROFIT
        elif pnl_ratio <= -self._settings.stop_loss_atr * state.entry_atr:
            exit_reason = ExitReason.STOP_LOSS
        elif abs(adjusted_strength) <= self._settings.exit_threshold:
            exit_reason = ExitReason.MOMENTUM_FADE
        elif state.bars_in_position >= self._settings.max_holding_bars:
            exit_reason = ExitReason.TIME_EXIT

        if exit_reason:
            side = "buy" if state.position == "short" else "sell"
            signals.append(
                self._build_signal(
                    snapshot,
                    side=side,
                    confidence=1.0,
                    strength=adjusted_strength,
                    volatility=volatility,
                    direction="flat",
                    exit_reason=exit_reason,
                )
            )
            self._reset_state(state)
        return signals

    # ------------------------------------------------------------------
    def _ensure_state(self, symbol: str) -> _DayTradingState:
        if symbol not in self._states:
            self._states[symbol] = _DayTradingState(
                closes=deque(maxlen=self._history_size),
                atrs=deque(maxlen=self._history_size),
            )
        return self._states[symbol]

    def _update_state(self, state: _DayTradingState, snapshot: MarketSnapshot) -> None:
        state.closes.append(float(snapshot.close))
        atr = float(snapshot.indicators.get("atr", 0.0))
        state.atrs.append(max(atr, self._settings.atr_floor))

    def _open_position(self, state: _DayTradingState, *, price: float, atr: float, direction: str) -> None:
        state.position = direction
        state.entry_price = price
        state.entry_atr = max(atr, self._settings.atr_floor)
        state.bars_in_position = 0

    def _reset_state(self, state: _DayTradingState) -> None:
        state.position = None
        state.entry_price = None
        state.entry_atr = max(self._settings.atr_floor, 0.0)
        state.bars_in_position = 0

    def _momentum_score(self, state: _DayTradingState) -> float:
        closes = list(state.closes)
        changes: List[float] = []
        for idx in range(1, len(closes)):
            prev = closes[idx - 1]
            if prev == 0:
                continue
            changes.append((closes[idx] - prev) / prev)
        if not changes:
            return 0.0
        window = min(self._settings.momentum_window, len(changes))
        return mean(changes[-window:])

    def _volatility(self, state: _DayTradingState) -> float:
        atrs = list(state.atrs)
        if not atrs:
            return self._settings.atr_floor
        window = min(self._settings.volatility_window, len(atrs))
        return max(mean(atrs[-window:]), self._settings.atr_floor)

    def _intraday_bias(self, snapshot: MarketSnapshot) -> float:
        high = snapshot.high if snapshot.high is not None else snapshot.close
        low = snapshot.low if snapshot.low is not None else snapshot.close
        midpoint = (high + low) / 2.0 if high is not None and low is not None else snapshot.close
        if midpoint == 0:
            return 0.0
        return max(min((snapshot.close - midpoint) / abs(midpoint), 1.0), -1.0)

    def _build_signal(
        self,
        snapshot: MarketSnapshot,
        *,
        side: str,
        confidence: float,
        strength: float,
        volatility: float,
        direction: str,
        exit_reason: str | None = None,
    ) -> StrategySignal:
        metadata: Dict[str, Any] = {
            "strategy": {
                "type": "day_trading",
                "profile": "intraday_momentum",
                "risk_label": "intraday",
            },
            "signal_strength": strength,
            "volatility": volatility,
            "position": direction,
        }
        if exit_reason:
            metadata["exit_reason"] = exit_reason
        return StrategySignal(
            symbol=snapshot.symbol,
            side=side,
            confidence=max(0.0, min(confidence, 1.0)),
            metadata=metadata,
        )


__all__ = ["DayTradingSettings", "DayTradingStrategy"]

