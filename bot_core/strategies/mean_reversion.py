"""Strategia statystycznego powrotu do średniej."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import log, sqrt
from typing import Deque, Dict, List, Sequence

from bot_core.strategies.base import MarketSnapshot, StrategyEngine, StrategySignal


@dataclass(slots=True)
class MeanReversionSettings:
    """Parametry kalibracyjne strategii mean reversion."""

    lookback: int = 96
    entry_zscore: float = 1.8
    exit_zscore: float = 0.4
    max_holding_period: int = 12
    volatility_cap: float = 0.04
    min_volume_usd: float = 100_000.0

    def history_size(self) -> int:
        return max(self.lookback, 2) + 4


@dataclass(slots=True)
class _SymbolState:
    closes: Deque[float]
    volumes: Deque[float]
    returns: Deque[float]
    position: int = 0  # -1 short, 0 flat, 1 long
    bars_in_position: int = 0


class MeanReversionStrategy(StrategyEngine):
    """Bazowy silnik powrotu do średniej oparty o z-score."""

    def __init__(self, settings: MeanReversionSettings | None = None) -> None:
        self._settings = settings or MeanReversionSettings()
        self._states: Dict[str, _SymbolState] = {}

    def warm_up(self, history: Sequence[MarketSnapshot]) -> None:
        for snapshot in history:
            state = self._state_for(snapshot.symbol)
            self._ingest(state, snapshot)

    def on_data(self, snapshot: MarketSnapshot) -> Sequence[StrategySignal]:
        state = self._state_for(snapshot.symbol)
        self._ingest(state, snapshot)

        if len(state.closes) < self._settings.lookback:
            return []

        if snapshot.volume < self._settings.min_volume_usd:
            return []

        zscore = self._compute_zscore(state)
        realized_vol = self._realized_volatility(state)
        signals: List[StrategySignal] = []

        # warunek awaryjnego zamknięcia przy przekroczeniu zmienności
        volatility_stop = realized_vol > self._settings.volatility_cap * 1.5

        if state.position == 0:
            if zscore <= -self._settings.entry_zscore and realized_vol <= self._settings.volatility_cap:
                state.position = 1
                state.bars_in_position = 0
                signals.append(
                    StrategySignal(
                        symbol=snapshot.symbol,
                        side="buy",
                        confidence=self._signal_confidence(zscore),
                        metadata={
                            "zscore": zscore,
                            "volatility": realized_vol,
                            "regime": "long_entry",
                        },
                    )
                )
            elif zscore >= self._settings.entry_zscore and realized_vol <= self._settings.volatility_cap:
                state.position = -1
                state.bars_in_position = 0
                signals.append(
                    StrategySignal(
                        symbol=snapshot.symbol,
                        side="sell",
                        confidence=self._signal_confidence(zscore),
                        metadata={
                            "zscore": zscore,
                            "volatility": realized_vol,
                            "regime": "short_entry",
                        },
                    )
                )
        elif state.position == 1:
            state.bars_in_position += 1
            if (
                zscore >= -self._settings.exit_zscore
                or state.bars_in_position >= self._settings.max_holding_period
                or volatility_stop
            ):
                state.position = 0
                signals.append(
                    StrategySignal(
                        symbol=snapshot.symbol,
                        side="sell",
                        confidence=self._exit_confidence(zscore),
                        metadata={
                            "zscore": zscore,
                            "volatility": realized_vol,
                            "regime": "long_exit",
                        },
                    )
                )
        elif state.position == -1:
            state.bars_in_position += 1
            if (
                zscore <= self._settings.exit_zscore
                or state.bars_in_position >= self._settings.max_holding_period
                or volatility_stop
            ):
                state.position = 0
                signals.append(
                    StrategySignal(
                        symbol=snapshot.symbol,
                        side="buy",
                        confidence=self._exit_confidence(zscore),
                        metadata={
                            "zscore": zscore,
                            "volatility": realized_vol,
                            "regime": "short_exit",
                        },
                    )
                )
        return signals

    def _state_for(self, symbol: str) -> _SymbolState:
        if symbol not in self._states:
            window = self._settings.history_size()
            self._states[symbol] = _SymbolState(
                closes=deque(maxlen=window),
                volumes=deque(maxlen=window),
                returns=deque(maxlen=window),
            )
        return self._states[symbol]

    def _ingest(self, state: _SymbolState, snapshot: MarketSnapshot) -> None:
        if state.closes:
            last_price = state.closes[-1]
            if last_price > 0 and snapshot.close > 0:
                state.returns.append(log(snapshot.close / last_price))
            else:
                state.returns.append(0.0)
        state.closes.append(snapshot.close)
        state.volumes.append(snapshot.volume)

    def _compute_zscore(self, state: _SymbolState) -> float:
        closes = list(state.closes)[-self._settings.lookback :]
        mean_price = sum(closes) / len(closes)
        variance = sum((price - mean_price) ** 2 for price in closes) / max(len(closes) - 1, 1)
        std_dev = sqrt(max(variance, 1e-12))
        return (closes[-1] - mean_price) / std_dev if std_dev else 0.0

    def _realized_volatility(self, state: _SymbolState) -> float:
        returns = list(state.returns)[-self._settings.lookback :]
        if not returns:
            return 0.0
        mean_ret = sum(returns) / len(returns)
        variance = sum((value - mean_ret) ** 2 for value in returns) / max(len(returns) - 1, 1)
        return sqrt(max(variance, 0.0))

    def _signal_confidence(self, zscore: float) -> float:
        magnitude = abs(zscore)
        return min(1.0, magnitude / max(self._settings.entry_zscore, 1e-6))

    def _exit_confidence(self, zscore: float) -> float:
        return min(1.0, max(0.0, abs(zscore) / max(self._settings.exit_zscore, 1e-6)))


__all__ = ["MeanReversionSettings", "MeanReversionStrategy"]
