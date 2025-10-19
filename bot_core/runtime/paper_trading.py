"""Paper trading adapter backed by the unified matching engine."""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Mapping, MutableMapping

import pandas as pd

try:  # pragma: no cover - optional dependency outside desktop builds
    from bot_core.backtest.simulation import BacktestFill, MatchingConfig, MatchingEngine  # type: ignore
except Exception:  # pragma: no cover - backtest module not packaged
    BacktestFill = MatchingConfig = MatchingEngine = None  # type: ignore
    try:
        _simulation = importlib.import_module("KryptoLowca.backtest.simulation")
    except Exception:  # pragma: no cover - brak legacy modułu
        pass
    else:
        BacktestFill = getattr(_simulation, "BacktestFill", None)
        MatchingConfig = getattr(_simulation, "MatchingConfig", None)
        MatchingEngine = getattr(_simulation, "MatchingEngine", None)

LOGGER = logging.getLogger(__name__)


@dataclass
class _PortfolioState:
    cash: float
    position: float = 0.0
    avg_price: float = 0.0
    last_price: float = 0.0
    fills: list[BacktestFill] = field(default_factory=list)
    matching: MatchingEngine | None = None


class PaperTradingAdapter:
    """ExecutionService-compatible adapter that simulates fills locally."""

    def __init__(self, *, initial_balance: float = 10_000.0, matching: MatchingConfig | None = None) -> None:
        if MatchingEngine is None or MatchingConfig is None:
            try:
                from KryptoLowca.backtest.simulation import (  # type: ignore
                    MatchingConfig as _MatchingConfig,
                    MatchingEngine as _MatchingEngine,
                )
            except Exception as exc:  # pragma: no cover - diagnostyka środowiska
                raise RuntimeError(
                    "PaperTradingAdapter wymaga modułu KryptoLowca.backtest.simulation"
                ) from exc
            else:
                globals()["MatchingEngine"] = _MatchingEngine
                globals()["MatchingConfig"] = _MatchingConfig
        if MatchingEngine is None or MatchingConfig is None:
            raise RuntimeError("PaperTradingAdapter wymaga modułu KryptoLowca.backtest.simulation")
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
        if bar is None or state.matching is None:
            return
        fills = state.matching.process_bar(
            index=self._coerce_int(bar.get("index"), default=0),
            timestamp=bar.get("timestamp", datetime.now(timezone.utc)),  # type: ignore[arg-type]
            bar=bar,  # type: ignore[arg-type]
        )
        for fill in fills:
            self._apply_fill(state, fill)
        state.last_price = self._coerce_float(
            bar.get("close", state.last_price),
            default=state.last_price,
        )

    def submit_order(self, *, symbol: str, side: str, size: float, **kwargs) -> Mapping[str, object]:
        state = self._ensure_state(symbol)
        if state.matching is None:
            state.matching = MatchingEngine(self._matching_cfg)
        timestamp = datetime.now(timezone.utc)
        order_id = state.matching.submit_market_order(
            side=side,
            size=float(size),
            index=self._coerce_int(kwargs.get("bar_index"), default=0),
            timestamp=timestamp,
            stop_loss=kwargs.get("stop_loss"),
            take_profit=kwargs.get("take_profit"),
        )
        LOGGER.debug("Paper order submitted: %s %s size=%s", symbol, side, size)
        return {"status": "accepted", "order_id": order_id, "timestamp": timestamp.isoformat()}

    def portfolio_snapshot(self, symbol: str) -> Mapping[str, float]:
        state = self._ensure_state(symbol)
        value = state.cash + state.position * state.last_price
        return {"value": value, "position": state.position, "price": state.last_price}

    def _apply_fill(self, state: _PortfolioState, fill: BacktestFill) -> None:
        direction = 1 if fill.side == "buy" else -1
        fee_paid = float(fill.fee)
        trade_notional = fill.price * fill.size
        previous_position = state.position

        state.cash -= direction * trade_notional
        state.cash -= fee_paid

        state.position = previous_position + direction * fill.size
        if state.position:
            state.avg_price = ((state.avg_price * previous_position) + fill.price * fill.size) / state.position
        else:
            state.avg_price = 0.0

        state.fills.append(fill)
        LOGGER.info(
            "Paper fill: side=%s price=%.4f size=%.4f cash=%.2f position=%.4f",
            fill.side,
            fill.price,
            fill.size,
            state.cash,
            state.position,
        )

    @staticmethod
    def _extract_bar(market_payload: Mapping[str, object]) -> Dict[str, object] | None:
        if not market_payload:
            return None
        ohlcv = market_payload.get("ohlcv")
        if isinstance(ohlcv, dict) and {"open", "high", "low", "close"}.issubset(ohlcv):
            bar: Dict[str, object] = dict(ohlcv)
        elif isinstance(ohlcv, pd.DataFrame) and not ohlcv.empty:
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

    @staticmethod
    def _coerce_int(value: object, *, default: int) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return default
            try:
                return int(stripped)
            except ValueError:
                try:
                    return int(float(stripped))
                except ValueError:
                    return default
        return default

    @staticmethod
    def _coerce_float(value: object, *, default: float) -> float:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return default
            try:
                return float(stripped)
            except ValueError:
                return default
        return default


__all__ = ["PaperTradingAdapter"]
