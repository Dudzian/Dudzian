"""Futures spread hedging engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from .base import MarketSnapshot, StrategyEngine, StrategySignal


@dataclass(slots=True)
class FuturesSpreadSettings:
    """Parametry kontrolujące strategię spreadową futures."""

    entry_z: float = 1.25
    exit_z: float = 0.4
    max_bars: int = 48
    funding_exit: float = 0.002
    basis_exit: float = 0.02


@dataclass(slots=True)
class _SpreadState:
    direction: int = 0
    bars_open: int = 0
    entry_z: float = 0.0


class FuturesSpreadStrategy(StrategyEngine):
    """Otwiera i zamyka pozycje hedge na podstawie rozjazdu kontraktów futures."""

    def __init__(self, settings: FuturesSpreadSettings | None = None) -> None:
        self._settings = settings or FuturesSpreadSettings()
        self._state: Dict[str, _SpreadState] = {}

    def warm_up(self, history: Sequence[MarketSnapshot]) -> None:
        for snapshot in history:
            self._ensure_state(snapshot.symbol)

    def on_data(self, snapshot: MarketSnapshot) -> Sequence[StrategySignal]:
        state = self._ensure_state(snapshot.symbol)
        zscore = float(snapshot.indicators.get("spread_zscore", snapshot.indicators.get("spread_z", 0.0)))
        basis = float(snapshot.indicators.get("basis", 0.0))
        funding = float(snapshot.indicators.get("funding_rate", 0.0))

        signals: List[StrategySignal] = []

        if state.direction == 0:
            if abs(zscore) >= self._settings.entry_z:
                direction = -1 if zscore > 0 else 1
                state.direction = direction
                state.bars_open = 0
                state.entry_z = zscore
                side = "short_front_long_back" if direction < 0 else "long_front_short_back"
                confidence = min(1.0, abs(zscore) / self._settings.entry_z)
                signals.append(
                    StrategySignal(
                        symbol=snapshot.symbol,
                        side=side,
                        confidence=confidence,
                        metadata={
                            "basis": basis,
                            "funding_rate": funding,
                            "spread_z": zscore,
                            "entry_z": self._settings.entry_z,
                        },
                    )
                )
            return signals

        state.bars_open += 1
        exit_due_to_z = abs(zscore) <= self._settings.exit_z
        exit_due_to_funding = abs(funding) >= self._settings.funding_exit
        exit_due_to_basis = abs(basis) >= self._settings.basis_exit and (basis * state.direction) > 0
        exit_due_to_time = state.bars_open >= self._settings.max_bars

        if exit_due_to_z or exit_due_to_funding or exit_due_to_basis or exit_due_to_time:
            side = "close_spread_position"
            signals.append(
                StrategySignal(
                    symbol=snapshot.symbol,
                    side=side,
                    confidence=1.0,
                    metadata={
                        "basis": basis,
                        "funding_rate": funding,
                        "spread_z": zscore,
                        "entry_z": state.entry_z,
                        "bars_open": state.bars_open,
                        "exit_reason": _exit_reason(
                            exit_due_to_z,
                            exit_due_to_funding,
                            exit_due_to_basis,
                            exit_due_to_time,
                        ),
                    },
                )
            )
            self._state[snapshot.symbol] = _SpreadState()

        return signals

    def _ensure_state(self, symbol: str) -> _SpreadState:
        state = self._state.get(symbol)
        if state is None:
            state = _SpreadState()
            self._state[symbol] = state
        return state


def _exit_reason(z: bool, funding: bool, basis: bool, time_stop: bool) -> str:
    if z:
        return "spread_mean_revert"
    if funding:
        return "funding_risk"
    if basis:
        return "basis_breakout"
    if time_stop:
        return "time_stop"
    return "unknown"


__all__ = ["FuturesSpreadSettings", "FuturesSpreadStrategy"]
