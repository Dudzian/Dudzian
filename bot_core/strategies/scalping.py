"""Strategia scalpingowa nastawiona na krótkoterminowe ruchy cenowe."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from bot_core.strategies.base import MarketSnapshot, StrategyEngine, StrategySignal
from bot_core.trading.exit_reasons import ExitReason


def _ensure_ratio(value: float, *, field: str) -> float:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{field} must be in the range (0, 1)")
    return value


def _ensure_positive_int(value: int, *, field: str) -> int:
    if value < 1:
        raise ValueError(f"{field} must be at least 1")
    return value


@dataclass(slots=True)
class ScalpingSettings:
    """Parametry kontrolujące zachowanie strategii scalpingowej."""

    min_price_change: float = 0.0005
    take_profit: float = 0.0010
    stop_loss: float = 0.0007
    max_hold_bars: int = 5

    def __post_init__(self) -> None:
        self.min_price_change = _ensure_ratio(float(self.min_price_change), field="min_price_change")
        self.take_profit = _ensure_ratio(float(self.take_profit), field="take_profit")
        self.stop_loss = _ensure_ratio(float(self.stop_loss), field="stop_loss")
        self.max_hold_bars = _ensure_positive_int(int(self.max_hold_bars), field="max_hold_bars")

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any] | None = None) -> "ScalpingSettings":
        params = dict(parameters or {})
        defaults = cls()
        return cls(
            min_price_change=float(params.get("min_price_change", defaults.min_price_change)),
            take_profit=float(params.get("take_profit", defaults.take_profit)),
            stop_loss=float(params.get("stop_loss", defaults.stop_loss)),
            max_hold_bars=int(params.get("max_hold_bars", defaults.max_hold_bars)),
        )


@dataclass(slots=True)
class _ScalpingState:
    last_price: float | None = None
    position: str | None = None
    entry_price: float | None = None
    bars_in_position: int = 0


class ScalpingStrategy(StrategyEngine):
    """Prosty silnik scalpingowy bazujący na zmianie ceny między tickami."""

    def __init__(self, settings: ScalpingSettings | None = None) -> None:
        self._settings = settings or ScalpingSettings()
        self._states: Dict[str, _ScalpingState] = {}

    def warm_up(self, history: Sequence[MarketSnapshot]) -> None:
        for snapshot in history:
            self._ensure_state(snapshot.symbol).last_price = snapshot.close

    def on_data(self, snapshot: MarketSnapshot) -> Sequence[StrategySignal]:
        state = self._ensure_state(snapshot.symbol)
        previous_price = state.last_price
        state.last_price = snapshot.close

        signals: List[StrategySignal] = []
        if previous_price is None or previous_price <= 0:
            return signals

        change_ratio = (snapshot.close - previous_price) / previous_price

        if state.position is None:
            if change_ratio >= self._settings.min_price_change:
                self._open_position(state, direction="long", price=snapshot.close)
                signals.append(
                    StrategySignal(
                        symbol=snapshot.symbol,
                        side="buy",
                        confidence=min(1.0, change_ratio / self._settings.min_price_change),
                        metadata=self._build_metadata(
                            snapshot=snapshot,
                            change_ratio=change_ratio,
                            direction="long",
                        ),
                    )
                )
            elif change_ratio <= -self._settings.min_price_change:
                self._open_position(state, direction="short", price=snapshot.close)
                signals.append(
                    StrategySignal(
                        symbol=snapshot.symbol,
                        side="sell",
                        confidence=min(1.0, abs(change_ratio) / self._settings.min_price_change),
                        metadata=self._build_metadata(
                            snapshot=snapshot,
                            change_ratio=change_ratio,
                            direction="short",
                        ),
                    )
                )
            return signals

        assert state.entry_price is not None
        pnl_ratio = (snapshot.close - state.entry_price) / state.entry_price
        if state.position == "short":
            pnl_ratio = -pnl_ratio

        state.bars_in_position += 1
        exit_reason = None
        if pnl_ratio >= self._settings.take_profit:
            exit_reason = ExitReason.TAKE_PROFIT
        elif pnl_ratio <= -self._settings.stop_loss:
            exit_reason = ExitReason.STOP_LOSS
        elif state.bars_in_position >= self._settings.max_hold_bars:
            exit_reason = ExitReason.TIME_EXIT

        if exit_reason:
            side = "buy" if state.position == "short" else "sell"
            signals.append(
                StrategySignal(
                    symbol=snapshot.symbol,
                    side=side,
                    confidence=1.0,
                    metadata=self._build_metadata(
                        snapshot=snapshot,
                        change_ratio=change_ratio,
                        direction="flat",
                        exit_reason=exit_reason,
                    ),
                )
            )
            self._reset_state(state)
        return signals

    # ------------------------------------------------------------------
    def _ensure_state(self, symbol: str) -> _ScalpingState:
        if symbol not in self._states:
            self._states[symbol] = _ScalpingState()
        return self._states[symbol]

    def _open_position(self, state: _ScalpingState, *, direction: str, price: float) -> None:
        state.position = direction
        state.entry_price = price
        state.bars_in_position = 0

    def _reset_state(self, state: _ScalpingState) -> None:
        state.position = None
        state.entry_price = None
        state.bars_in_position = 0

    def _build_metadata(
        self,
        *,
        snapshot: MarketSnapshot,
        change_ratio: float,
        direction: str,
        exit_reason: str | None = None,
    ) -> Dict[str, object]:
        metadata: Dict[str, object] = {
            "strategy": {
                "type": "scalping",
                "profile": "scalping_aggressive",
                "risk_label": "high",
            },
            "change_ratio": change_ratio,
            "last_close": snapshot.close,
        }
        if exit_reason:
            metadata["exit_reason"] = exit_reason
        metadata["position"] = direction
        return metadata


__all__ = ["ScalpingSettings", "ScalpingStrategy"]
