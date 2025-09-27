"""Paper trading adapter using the unified matching engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Mapping, MutableMapping

import pandas as pd

from KryptoLowca.logging_utils import get_logger

from KryptoLowca.backtest.simulation import BacktestFill, MatchingConfig, MatchingEngine

logger = get_logger(__name__)


@dataclass
class _PortfolioState:
    cash: float
    position: float = 0.0
    avg_price: float = 0.0
    last_price: float = 0.0
    fills: list[BacktestFill] = field(default_factory=list)
    matching: MatchingEngine | None = None


class PaperTradingAdapter:
    """Adapter zgodny z ExecutionService wykorzystujący symulator matching engine."""

    def __init__(self, *, initial_balance: float = 10_000.0, matching: MatchingConfig | None = None) -> None:
        self._initial_balance = float(initial_balance)
        self._matching_cfg = matching or MatchingConfig()
        self._portfolios: MutableMapping[str, _PortfolioState] = {}

    def _ensure_state(self, symbol: str) -> _PortfolioState:
        state = self._portfolios.get(symbol)
        if state is None:
            state = _PortfolioState(cash=self._initial_balance, matching=MatchingEngine(self._matching_cfg))
            self._portfolios[symbol] = state
        return state

    def update_market_data(self, symbol: str, timeframe: str, market_payload: Mapping[str, object]) -> None:
        state = self._ensure_state(symbol)
        bar = self._extract_bar(market_payload)
        if bar is None:
            return
        fills = state.matching.process_bar(
            index=int(bar.get("index", 0)),
            timestamp=bar.get("timestamp", datetime.now(timezone.utc)),
            bar=bar,
        )
        for fill in fills:
            self._apply_fill(state, fill)
        state.last_price = float(bar.get("close", state.last_price))

    def submit_order(self, *, symbol: str, side: str, size: float, **kwargs) -> Mapping[str, object]:
        state = self._ensure_state(symbol)
        timestamp = datetime.now(timezone.utc)
        order_id = state.matching.submit_market_order(
            side=side,
            size=float(size),
            index=int(kwargs.get("bar_index", 0)),
            timestamp=timestamp,
            stop_loss=kwargs.get("stop_loss"),
            take_profit=kwargs.get("take_profit"),
        )
        logger.debug("Paper order submitted: %s %s size=%s", symbol, side, size)
        return {"status": "accepted", "order_id": order_id, "timestamp": timestamp.isoformat()}

    def portfolio_snapshot(self, symbol: str) -> Mapping[str, float]:
        state = self._ensure_state(symbol)
        value = state.cash + state.position * state.last_price
        return {"value": value, "position": state.position, "price": state.last_price}

    def _apply_fill(self, state: _PortfolioState, fill: BacktestFill) -> None:
        direction = 1 if fill.side == "buy" else -1
        state.cash -= direction * fill.price * fill.size
        state.cash -= fill.fee
        state.position += direction * fill.size
        if state.position:
            state.avg_price = (
                (state.avg_price * (state.position - direction * fill.size)) + fill.price * fill.size
            ) / state.position
        else:
            state.avg_price = 0.0
        state.fills.append(fill)
        logger.info(
            "Paper fill: side=%s price=%.4f size=%.4f cash=%.2f position=%.4f",
            fill.side,
            fill.price,
            fill.size,
            state.cash,
            state.position,
        )

    @staticmethod
    def _extract_bar(market_payload: Mapping[str, object]) -> Dict[str, float] | None:
        if not market_payload:
            return None
        ohlcv = market_payload.get("ohlcv")
        if isinstance(ohlcv, dict) and {"open", "high", "low", "close"}.issubset(ohlcv):
            bar = dict(ohlcv)
        elif isinstance(ohlcv, pd.DataFrame) and not ohlcv.empty:  # type: ignore[name-defined]
            last = ohlcv.iloc[-1]
            bar = {
                "open": float(last.get("open", 0.0)),
                "high": float(last.get("high", 0.0)),
                "low": float(last.get("low", 0.0)),
                "close": float(last.get("close", 0.0)),
                "volume": float(last.get("volume", 0.0)),
            }
        else:
            close = market_payload.get("price")
            if close is None:
                return None
            bar = {"open": float(close), "high": float(close), "low": float(close), "close": float(close), "volume": 0.0}
        bar.setdefault("timestamp", datetime.now(timezone.utc))
        return bar


__all__ = ["PaperTradingAdapter"]
