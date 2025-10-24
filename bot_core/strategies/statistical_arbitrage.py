"""Strategia arbitrażu statystycznego (pairs trading)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Mapping, Sequence

from bot_core.strategies.base import MarketSnapshot, StrategyEngine, StrategySignal


@dataclass(slots=True)
class StatisticalArbitrageSettings:
    """Parametry strategii pairs trading."""

    lookback: int = 30
    spread_entry_z: float = 2.0
    spread_exit_z: float = 0.5
    max_notional: float = 25_000.0

    def __post_init__(self) -> None:
        if int(self.lookback) < 10:
            raise ValueError("lookback must be at least 10 periods")
        self.lookback = int(self.lookback)
        self.spread_entry_z = float(self.spread_entry_z)
        self.spread_exit_z = float(self.spread_exit_z)
        if self.spread_entry_z <= 0:
            raise ValueError("spread_entry_z must be positive")
        if not 0 < self.spread_exit_z < self.spread_entry_z:
            raise ValueError("spread_exit_z must be positive and lower than entry threshold")
        self.max_notional = float(self.max_notional)
        if self.max_notional <= 0:
            raise ValueError("max_notional must be positive")

    @classmethod
    def from_parameters(
        cls, parameters: Mapping[str, Any] | None = None
    ) -> "StatisticalArbitrageSettings":
        params = dict(parameters or {})
        defaults = cls()
        return cls(
            lookback=int(params.get("lookback", defaults.lookback)),
            spread_entry_z=float(params.get("spread_entry_z", defaults.spread_entry_z)),
            spread_exit_z=float(params.get("spread_exit_z", defaults.spread_exit_z)),
            max_notional=float(params.get("max_notional", defaults.max_notional)),
        )


@dataclass(slots=True)
class _PairState:
    spreads: Deque[float]
    position: str | None = None
    entry_z: float | None = None


class StatisticalArbitrageStrategy(StrategyEngine):
    """Wykorzystuje odchylenia spreadu pomiędzy dwoma powiązanymi aktywami."""

    def __init__(self, settings: StatisticalArbitrageSettings | None = None) -> None:
        self._settings = settings or StatisticalArbitrageSettings()
        self._states: Dict[str, _PairState] = {}

    def warm_up(self, history: Sequence[MarketSnapshot]) -> None:
        for snapshot in history:
            self._process_snapshot(snapshot)

    def on_data(self, snapshot: MarketSnapshot) -> Sequence[StrategySignal]:
        return self._process_snapshot(snapshot)

    # ------------------------------------------------------------------
    def _ensure_state(self, symbol: str) -> _PairState:
        if symbol not in self._states:
            self._states[symbol] = _PairState(
                spreads=deque(maxlen=self._settings.lookback),
            )
        return self._states[symbol]

    def _process_snapshot(self, snapshot: MarketSnapshot) -> Sequence[StrategySignal]:
        paired_price = float(snapshot.indicators.get("paired_price", 0.0))
        paired_symbol = str(snapshot.indicators.get("paired_symbol", ""))
        if paired_price <= 0:
            return []

        state = self._ensure_state(snapshot.symbol)
        spread = snapshot.close - paired_price
        state.spreads.append(spread)

        if len(state.spreads) < self._settings.lookback:
            return []

        spreads = list(state.spreads)
        mean_spread = sum(spreads) / len(spreads)
        variance = sum((value - mean_spread) ** 2 for value in spreads) / max(1, len(spreads) - 1)
        std_spread = variance ** 0.5
        if std_spread == 0:
            return []

        z_score = (spread - mean_spread) / std_spread
        signals: List[StrategySignal] = []

        if state.position is None:
            if z_score >= self._settings.spread_entry_z:
                state.position = "short_primary_long_secondary"
                state.entry_z = z_score
                signals.append(
                    StrategySignal(
                        symbol=snapshot.symbol,
                        side="short_primary_long_secondary",
                        confidence=min(1.0, z_score / self._settings.spread_entry_z),
                        metadata=self._build_metadata(
                            snapshot,
                            paired_symbol,
                            z_score=z_score,
                            spread=spread,
                            status="enter",
                        ),
                    )
                )
            elif z_score <= -self._settings.spread_entry_z:
                state.position = "long_primary_short_secondary"
                state.entry_z = z_score
                signals.append(
                    StrategySignal(
                        symbol=snapshot.symbol,
                        side="long_primary_short_secondary",
                        confidence=min(1.0, abs(z_score) / self._settings.spread_entry_z),
                        metadata=self._build_metadata(
                            snapshot,
                            paired_symbol,
                            z_score=z_score,
                            spread=spread,
                            status="enter",
                        ),
                    )
                )
            return signals

        exit_condition = abs(z_score) <= self._settings.spread_exit_z
        if state.entry_z is not None and z_score * state.entry_z <= 0:
            exit_condition = True
        if exit_condition:
            side = "close_" + state.position
            signals.append(
                StrategySignal(
                    symbol=snapshot.symbol,
                    side=side,
                    confidence=1.0,
                    metadata=self._build_metadata(
                        snapshot,
                        paired_symbol,
                        z_score=z_score,
                        spread=spread,
                        status="exit",
                    ),
                )
            )
            state.position = None
            state.entry_z = None
        return signals

    def _build_metadata(
        self,
        snapshot: MarketSnapshot,
        paired_symbol: str,
        *,
        z_score: float,
        spread: float,
        status: str,
    ) -> Dict[str, object]:
        return {
            "strategy": {
                "type": "statistical_arbitrage",
                "profile": "stat_arb_balanced",
                "risk_label": "balanced",
            },
            "paired_symbol": paired_symbol,
            "spread": spread,
            "z_score": z_score,
            "status": status,
            "max_notional": self._settings.max_notional,
            "timestamp": snapshot.timestamp,
        }


__all__ = [
    "StatisticalArbitrageSettings",
    "StatisticalArbitrageStrategy",
]
