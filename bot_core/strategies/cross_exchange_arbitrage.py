"""Strategia arbitrażu między giełdami."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Sequence

from bot_core.strategies.base import MarketSnapshot, SignalLeg, StrategyEngine, StrategySignal


@dataclass(slots=True)
class CrossExchangeArbitrageSettings:
    """Parametry kontrolujące zachowanie strategii arbitrażowej."""

    primary_exchange: str
    secondary_exchange: str
    spread_entry: float = 0.0015
    spread_exit: float = 0.0005
    max_notional: float = 50_000.0
    max_open_seconds: int = 120


@dataclass(slots=True)
class _PositionState:
    direction: str
    opened_at: datetime
    entry_spread: float


class CrossExchangeArbitrageStrategy(StrategyEngine):
    """Monitoruje różnice cen między dwoma giełdami i generuje sygnały arbitrażowe."""

    def __init__(self, settings: CrossExchangeArbitrageSettings) -> None:
        self._settings = settings
        self._positions: Dict[str, _PositionState] = {}

    def warm_up(self, history: Sequence[MarketSnapshot]) -> None:
        # brak stanu wymagającego rozgrzania – strategia operuje na bieżących spreadach
        return None

    def on_data(self, snapshot: MarketSnapshot) -> Sequence[StrategySignal]:
        primary_bid = float(snapshot.indicators.get("primary_bid", snapshot.close))
        primary_ask = float(snapshot.indicators.get("primary_ask", snapshot.close))
        secondary_bid = float(snapshot.indicators.get("secondary_bid", snapshot.close))
        secondary_ask = float(snapshot.indicators.get("secondary_ask", snapshot.close))
        secondary_ts = int(snapshot.indicators.get("secondary_timestamp", snapshot.timestamp))
        timestamp = _to_datetime(snapshot.timestamp)
        secondary_time = _to_datetime(secondary_ts)

        signals: List[StrategySignal] = []
        position = self._positions.get(snapshot.symbol)

        # Spread dodatni oznacza możliwość kupna na giełdzie primary i sprzedaży na secondary.
        positive_spread = secondary_bid - primary_ask
        negative_spread = primary_bid - secondary_ask
        mid_price = max(primary_bid, primary_ask, secondary_bid, secondary_ask, snapshot.close)
        spread_ratio_positive = positive_spread / mid_price if mid_price else 0.0
        spread_ratio_negative = negative_spread / mid_price if mid_price else 0.0

        if position is None:
            if spread_ratio_positive >= self._settings.spread_entry:
                self._positions[snapshot.symbol] = _PositionState(
                    direction="long_primary_short_secondary",
                    opened_at=timestamp,
                    entry_spread=spread_ratio_positive,
                )
                primary_qty = _notional_to_quantity(self._settings.max_notional, primary_ask)
                secondary_qty = _notional_to_quantity(self._settings.max_notional, secondary_bid)
                signals.append(
                    StrategySignal(
                        symbol=snapshot.symbol,
                        side="long_primary_short_secondary",
                        confidence=min(1.0, spread_ratio_positive / self._settings.spread_entry),
                        intent="multi_leg",
                        legs=(
                            SignalLeg(
                                symbol=snapshot.symbol,
                                side="BUY",
                                quantity=primary_qty,
                                exchange=self._settings.primary_exchange,
                                metadata={
                                    "leg": "primary_entry",
                                    "price": primary_ask,
                                },
                            ),
                            SignalLeg(
                                symbol=snapshot.symbol,
                                side="SELL",
                                quantity=secondary_qty,
                                exchange=self._settings.secondary_exchange,
                                metadata={
                                    "leg": "secondary_entry",
                                    "price": secondary_bid,
                                },
                            ),
                        ),
                        metadata={
                            "primary_exchange": self._settings.primary_exchange,
                            "secondary_exchange": self._settings.secondary_exchange,
                            "spread": positive_spread,
                            "spread_ratio": spread_ratio_positive,
                            "max_notional": self._settings.max_notional,
                        },
                    )
                )
            elif spread_ratio_negative >= self._settings.spread_entry:
                self._positions[snapshot.symbol] = _PositionState(
                    direction="short_primary_long_secondary",
                    opened_at=timestamp,
                    entry_spread=spread_ratio_negative,
                )
                primary_qty = _notional_to_quantity(self._settings.max_notional, primary_bid)
                secondary_qty = _notional_to_quantity(self._settings.max_notional, secondary_ask)
                signals.append(
                    StrategySignal(
                        symbol=snapshot.symbol,
                        side="short_primary_long_secondary",
                        confidence=min(1.0, spread_ratio_negative / self._settings.spread_entry),
                        intent="multi_leg",
                        legs=(
                            SignalLeg(
                                symbol=snapshot.symbol,
                                side="SELL",
                                quantity=primary_qty,
                                exchange=self._settings.primary_exchange,
                                metadata={
                                    "leg": "primary_entry",
                                    "price": primary_bid,
                                },
                            ),
                            SignalLeg(
                                symbol=snapshot.symbol,
                                side="BUY",
                                quantity=secondary_qty,
                                exchange=self._settings.secondary_exchange,
                                metadata={
                                    "leg": "secondary_entry",
                                    "price": secondary_ask,
                                },
                            ),
                        ),
                        metadata={
                            "primary_exchange": self._settings.primary_exchange,
                            "secondary_exchange": self._settings.secondary_exchange,
                            "spread": negative_spread,
                            "spread_ratio": spread_ratio_negative,
                            "max_notional": self._settings.max_notional,
                        },
                    )
                )
            return signals

        # Zarządzanie otwartą pozycją
        elapsed = (timestamp - position.opened_at).total_seconds()
        if position.direction == "long_primary_short_secondary":
            if (
                spread_ratio_positive <= self._settings.spread_exit
                or elapsed >= self._settings.max_open_seconds
            ):
                del self._positions[snapshot.symbol]
                primary_qty = _notional_to_quantity(self._settings.max_notional, primary_bid)
                secondary_qty = _notional_to_quantity(self._settings.max_notional, secondary_ask)
                signals.append(
                    StrategySignal(
                        symbol=snapshot.symbol,
                        side="close_long_primary_short_secondary",
                        confidence=1.0,
                        intent="multi_leg",
                        legs=(
                            SignalLeg(
                                symbol=snapshot.symbol,
                                side="SELL",
                                quantity=primary_qty,
                                exchange=self._settings.primary_exchange,
                                metadata={
                                    "leg": "primary_exit",
                                    "price": primary_bid,
                                },
                            ),
                            SignalLeg(
                                symbol=snapshot.symbol,
                                side="BUY",
                                quantity=secondary_qty,
                                exchange=self._settings.secondary_exchange,
                                metadata={
                                    "leg": "secondary_exit",
                                    "price": secondary_ask,
                                },
                            ),
                        ),
                        metadata={
                            "exit_spread": positive_spread,
                            "entry_spread": position.entry_spread,
                            "elapsed_seconds": elapsed,
                            "secondary_delay_ms": max(
                                0, (timestamp - secondary_time).total_seconds() * 1000
                            ),
                        },
                    )
                )
        elif position.direction == "short_primary_long_secondary":
            if (
                spread_ratio_negative <= self._settings.spread_exit
                or elapsed >= self._settings.max_open_seconds
            ):
                del self._positions[snapshot.symbol]
                primary_qty = _notional_to_quantity(self._settings.max_notional, primary_ask)
                secondary_qty = _notional_to_quantity(self._settings.max_notional, secondary_bid)
                signals.append(
                    StrategySignal(
                        symbol=snapshot.symbol,
                        side="close_short_primary_long_secondary",
                        confidence=1.0,
                        intent="multi_leg",
                        legs=(
                            SignalLeg(
                                symbol=snapshot.symbol,
                                side="BUY",
                                quantity=primary_qty,
                                exchange=self._settings.primary_exchange,
                                metadata={
                                    "leg": "primary_exit",
                                    "price": primary_ask,
                                },
                            ),
                            SignalLeg(
                                symbol=snapshot.symbol,
                                side="SELL",
                                quantity=secondary_qty,
                                exchange=self._settings.secondary_exchange,
                                metadata={
                                    "leg": "secondary_exit",
                                    "price": secondary_bid,
                                },
                            ),
                        ),
                        metadata={
                            "exit_spread": negative_spread,
                            "entry_spread": position.entry_spread,
                            "elapsed_seconds": elapsed,
                            "secondary_delay_ms": max(
                                0, (timestamp - secondary_time).total_seconds() * 1000
                            ),
                        },
                    )
                )

        return signals


def _to_datetime(value: int | float) -> datetime:
    """Zamienia timestamp w sekundach lub milisekundach na datetime UTC."""

    seconds = float(value)
    if seconds > 10_000_000_000:  # heurystyka na wartości w ms
        seconds = seconds / 1000.0
    return datetime.fromtimestamp(seconds, tz=timezone.utc)


__all__ = ["CrossExchangeArbitrageSettings", "CrossExchangeArbitrageStrategy"]


def _notional_to_quantity(max_notional: float, price: float) -> float:
    reference_price = max(price, 1e-9)
    if max_notional <= 0:
        return 1.0
    quantity = max_notional / reference_price
    return max(quantity, 1e-9)
