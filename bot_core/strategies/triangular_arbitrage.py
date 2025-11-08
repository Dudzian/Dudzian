"""Silnik arbitrażu trójkątnego wykorzystujący sygnały multi-leg."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence

from .base import MarketSnapshot, SignalLeg, StrategyEngine, StrategySignal


@dataclass(slots=True)
class TriangularArbitrageSettings:
    """Parametry kontrolujące strategię arbitrażu trójkątnego."""

    min_edge_bps: float = 4.0
    max_leg_latency_ms: float = 250.0
    cooldown_ms: int = 3_000
    notional_cap: float = 25_000.0


@dataclass(slots=True)
class _ArbitrageState:
    last_timestamp: int = 0
    last_edge_bps: float = 0.0


class TriangularArbitrageStrategy(StrategyEngine):
    """Strategia trójkątnego arbitrażu reagująca na najlepsze ścieżki."""

    def __init__(self, settings: TriangularArbitrageSettings | None = None) -> None:
        self._settings = settings or TriangularArbitrageSettings()
        self._states: MutableMapping[str, _ArbitrageState] = {}

    def warm_up(self, history: Sequence[MarketSnapshot]) -> None:
        for snapshot in history:
            state = self._ensure_state(snapshot.symbol)
            state.last_timestamp = snapshot.timestamp
            state.last_edge_bps = float(snapshot.indicators.get("forward_edge_bps", 0.0))

    def on_data(self, snapshot: MarketSnapshot) -> Sequence[StrategySignal]:
        state = self._ensure_state(snapshot.symbol)

        forward_edge = float(snapshot.indicators.get("forward_edge_bps", 0.0))
        reverse_edge = float(snapshot.indicators.get("reverse_edge_bps", 0.0))
        direction = "forward"
        edge_bps = forward_edge

        if reverse_edge > edge_bps:
            direction = "reverse"
            edge_bps = reverse_edge

        if edge_bps < self._settings.min_edge_bps:
            return []

        if snapshot.timestamp - state.last_timestamp < self._settings.cooldown_ms:
            return []

        path_key = f"{direction}_path"
        path_definition = snapshot.indicators.get(path_key) or snapshot.indicators.get("triangular_path")
        if not path_definition:
            return []

        try:
            legs = tuple(_parse_path_definition(path_definition))
        except ValueError:
            return []

        metadata = _build_metadata(snapshot.indicators, direction, edge_bps, self._settings)

        state.last_timestamp = snapshot.timestamp
        state.last_edge_bps = edge_bps

        confidence = min(1.0, edge_bps / max(self._settings.min_edge_bps, 1e-9))

        return (
            StrategySignal(
                symbol=snapshot.symbol,
                side="buy" if direction == "forward" else "sell",
                confidence=confidence,
                intent="multi_leg",
                legs=legs,
                metadata=metadata,
            ),
        )

    def _ensure_state(self, symbol: str) -> _ArbitrageState:
        if symbol not in self._states:
            self._states[symbol] = _ArbitrageState()
        return self._states[symbol]


def _parse_path_definition(path: object) -> Sequence[SignalLeg]:
    if isinstance(path, str):
        entries = [chunk.strip() for chunk in path.split("->") if chunk.strip()]
        path_payload: list[Mapping[str, object]] = []
        for entry in entries:
            if ":" not in entry:
                raise ValueError("Expected leg definition in form SYMBOL:SIDE")
            symbol, side = entry.split(":", 1)
            path_payload.append({"symbol": symbol.strip(), "side": side.strip().lower()})
        return _parse_path_definition(tuple(path_payload))

    if isinstance(path, Mapping):
        path = (path,)

    if not isinstance(path, Sequence):
        raise ValueError("Unsupported path structure")

    legs: list[SignalLeg] = []
    for item in path:
        if not isinstance(item, Mapping):
            raise ValueError("Each leg must be a mapping")
        symbol = str(item.get("symbol") or "").strip()
        side = str(item.get("side") or "").strip().lower() or "buy"
        exchange = item.get("exchange")
        quantity = item.get("quantity")
        confidence = item.get("confidence")
        metadata = item.get("metadata") or {}
        legs.append(
            SignalLeg(
                symbol=symbol,
                side=side,
                exchange=str(exchange).strip() if exchange else None,
                quantity=float(quantity) if quantity is not None else None,
                confidence=float(confidence) if confidence is not None else None,
                metadata=dict(metadata),
            )
        )
    return tuple(legs)


def _build_metadata(
    indicators: Mapping[str, object],
    direction: str,
    edge_bps: float,
    settings: TriangularArbitrageSettings,
) -> Mapping[str, object]:
    metadata: dict[str, object] = {
        "direction": direction,
        "edge_bps": edge_bps,
        "max_leg_latency_ms": settings.max_leg_latency_ms,
        "notional_cap": settings.notional_cap,
    }
    for key in ("quote_latency_ms", "path_hash", "primary_exchange", "secondary_exchange"):
        if key in indicators:
            metadata[key] = indicators[key]
    return metadata


__all__ = ["TriangularArbitrageSettings", "TriangularArbitrageStrategy"]

