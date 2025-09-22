# core/trading_engine.py
# -*- coding: utf-8 -*-
"""
Trading Engine for executing trading strategies.

Features:
- Executes trading plans based on AI predictions and TradingStrategies.
- Supports long/short positions with risk management.
- Integrates with ExchangeManager, AIManager, RiskManager, and TradingStrategies.
- Async operations for real-time trading.
- Event-driven architecture with callbacks for GUI updates.
- Inspired by Cryptohopper: strategy execution, risk controls, real-time feedback.
"""
from __future__ import annotations

import asyncio
import logging
import pandas as pd
from typing import Dict, Optional, Callable, Any, Union
from dataclasses import dataclass
import numpy as np

from managers.exchange_manager import ExchangeManager
from managers.ai_manager import AIManager
from managers.risk_manager_adapter import RiskManager
from managers.database_manager import DatabaseManager
from trading_strategies import TradingStrategies, TradingParameters, EngineConfig

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# Domyślny próg sygnału AI w punktach bazowych – wykorzystywany, gdy model nie dostarczy wartości
DEFAULT_AI_THRESHOLD_BPS = 10.0

# --- Custom exceptions ---
class TradingError(Exception):
    """Raised when trading operations fail."""
    pass

class TradingEngine:
    """
    Core trading engine for executing strategies.

    Args:
        db_manager: DatabaseManager for logging and position tracking.
    """
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.ex_mgr: Optional[ExchangeManager] = None
        self.ai_mgr: Optional[AIManager] = None
        self.risk_mgr: Optional[RiskManager] = None
        self.db_manager = db_manager
        self.strategies = TradingStrategies()
        self.tp = TradingParameters()
        self.ec = EngineConfig()
        self._event_callback: Optional[Callable] = None
        self._lock = asyncio.Lock()
        self._user_id: Optional[int] = None
        # Fallback kapitału wykorzystywany, gdy giełda nie zwróci salda
        self._fallback_capital: float = 1000.0

    async def configure(self, ex_mgr: ExchangeManager, ai_mgr: AIManager, risk_mgr: RiskManager):
        """Configure the engine with dependencies."""
        async with self._lock:
            self.ex_mgr = ex_mgr
            self.ai_mgr = ai_mgr
            self.risk_mgr = risk_mgr
            if self.db_manager:
                self._user_id = await self.db_manager.ensure_user("engine_user@example.com")
                await self.db_manager.log(self._user_id, "INFO", "Trading engine configured", category="engine")

    def set_parameters(self, tp: TradingParameters, ec: EngineConfig):
        """Set trading parameters and configuration."""
        self.tp = tp
        self.ec = ec
        if self.db_manager:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.debug("set_parameters: event loop not running; skipping async log")
            else:
                loop.create_task(
                    self.db_manager.log(
                        self._user_id,
                        "INFO",
                        f"Parameters set: {tp.__dict__}, {ec.__dict__}",
                        category="engine",
                    )
                )

    def on_event(self, callback: Callable):
        """Register event callback for GUI updates."""
        self._event_callback = callback

    async def _emit_event(self, event: Dict[str, Any]):
        """Emit an event to the callback."""
        if self._event_callback:
            self._event_callback(event)
        if self.db_manager:
            await self.db_manager.log(self._user_id, "INFO", f"Event emitted: {event}", category="engine")

    async def execute_live_tick(self, symbol: str, df: pd.DataFrame, preds: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Execute trading logic for a single tick.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT").
            df: DataFrame with OHLCV data.
            preds: Series with AI predictions (in basis points).

        Returns:
            Trading plan or None if no action is taken.
        """
        async with self._lock:
            try:
                self._validate_dependencies()

                if not symbol or not isinstance(symbol, str):
                    raise ValueError("Invalid symbol")
                if df is None or df.empty:
                    raise ValueError("Market data frame is empty")
                if len(df) < self.ec.min_data_points:
                    raise ValueError(
                        f"Insufficient data: {len(df)} bars, required {self.ec.min_data_points}"
                    )
                if preds is None or len(preds) != len(df):
                    raise ValueError("Predictions length does not match data length")

                df = df.copy()
                preds = preds.astype(float)

                positions = await self._fetch_positions()
                max_positions = int(max(0, round(self.ec.max_position_size)))
                if max_positions and len(positions) >= max_positions:
                    await self._emit_event(
                        {"type": "max_positions_reached", "symbol": symbol, "positions": len(positions)}
                    )
                    return None

                threshold = self._resolve_ai_threshold()
                latest_pred = float(preds.iloc[-1])
                latest_price = float(df["close"].iloc[-1])

                if not np.isfinite(latest_pred) or not np.isfinite(latest_price):
                    raise ValueError("Latest prediction or price is not finite")

                strength_ratio = abs(latest_pred) / threshold if threshold else 0.0
                if strength_ratio < 1.0:
                    await self._emit_event({"type": "no_signal", "symbol": symbol, "strength": strength_ratio})
                    return None

                side = "buy" if latest_pred > 0 else "sell"
                if side == "sell" and not getattr(self.ec, "enable_shorting", False):
                    await self._emit_event({"type": "shorting_disabled", "symbol": symbol})
                    return None

                capital = await self._estimate_capital()
                if capital <= 0:
                    raise TradingError("Insufficient balance")

                signal_payload = self._build_signal_payload(latest_pred, strength_ratio, side)
                portfolio = {"capital": capital, "positions": positions}
                sizing = self.risk_mgr.calculate_position_size(symbol, signal_payload, df, portfolio)
                allocation = self._extract_allocation(sizing)
                qty = max(0.0, allocation * capital / latest_price)

                atr = self._compute_atr(df, self.tp.atr_period)
                stop_loss, take_profit = self._compute_protective_levels(side, latest_price, atr)

                plan = {
                    "symbol": symbol,
                    "side": side,
                    "qty_hint": qty,
                    "price_ref": latest_price,
                    "strength": strength_ratio,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                }

                await self._emit_event({"type": "plan_created", "symbol": symbol, "plan": plan})
                if self.db_manager:
                    await self.db_manager.log(
                        self._user_id,
                        "INFO",
                        f"Trading plan created for {symbol}: {plan}",
                        category="trade",
                    )
                return plan

            except (ValueError, TradingError) as e:
                logger.error(f"Trading tick failed for {symbol}: {e}")
                if self.db_manager:
                    await self.db_manager.log(
                        self._user_id,
                        "ERROR",
                        f"Trading tick failed for {symbol}: {e}",
                        category="trade",
                    )
                raise
            except Exception as e:
                logger.error(f"Trading tick failed for {symbol}: {e}")
                if self.db_manager:
                    await self.db_manager.log(
                        self._user_id,
                        "ERROR",
                        f"Trading tick failed for {symbol}: {e}",
                        category="trade",
                    )
                await self._emit_event({"type": "error", "symbol": symbol, "error": str(e)})
                return None

    def _validate_dependencies(self) -> None:
        if not self.ex_mgr or not self.ai_mgr or not self.risk_mgr:
            raise TradingError("Trading engine is not fully configured")

    async def _fetch_positions(self) -> list:
        if not self.db_manager or self._user_id is None:
            return []
        try:
            positions = await self.db_manager.get_positions(self._user_id)
            return positions or []
        except Exception as exc:  # pragma: no cover - defensywnie
            logger.error(f"Failed to fetch positions: {exc}")
            return []

    def _resolve_ai_threshold(self) -> float:
        candidate = None
        if self.ai_mgr is not None:
            for attr in ("ai_threshold_bps", "signal_threshold_bps", "threshold_bps"):
                candidate = getattr(self.ai_mgr, attr, None)
                if candidate is not None:
                    break
        try:
            threshold = float(candidate) if candidate is not None else DEFAULT_AI_THRESHOLD_BPS
        except (TypeError, ValueError):
            threshold = DEFAULT_AI_THRESHOLD_BPS
        return max(1e-3, threshold)

    async def _estimate_capital(self) -> float:
        if not self.ex_mgr:
            return self._fallback_capital
        balance_fetcher = getattr(self.ex_mgr, "fetch_balance", None)
        if not callable(balance_fetcher):
            return self._fallback_capital
        try:
            balance = await balance_fetcher()
        except Exception as exc:
            logger.warning(f"fetch_balance failed: {exc}")
            return self._fallback_capital

        if isinstance(balance, dict):
            # CCXT-style strukturę traktujemy priorytetowo
            for key in ("free", "total", "info"):
                sub = balance.get(key)
                if isinstance(sub, dict):
                    amount = self._extract_stablecoin_balance(sub)
                    if amount is not None:
                        return amount
            amount = self._extract_stablecoin_balance(balance)
            if amount is not None:
                return amount

        try:
            return float(balance)
        except (TypeError, ValueError):
            return self._fallback_capital

    @staticmethod
    def _extract_stablecoin_balance(values: Dict[str, Any]) -> Optional[float]:
        for key in ("USDT", "USD", "usdt", "usd", "cash"):
            if key in values:
                try:
                    return float(values[key])
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _build_signal_payload(prediction: float, strength_ratio: float, side: str) -> Dict[str, Any]:
        confidence = min(1.0, strength_ratio)
        return {
            "strength": strength_ratio,
            "confidence": confidence,
            "direction": side,
            "raw_prediction": prediction,
        }

    @staticmethod
    def _extract_allocation(sizing: Union[float, Dict[str, Any]]) -> float:
        if sizing is None:
            return 0.0
        if isinstance(sizing, (int, float)):
            return max(0.0, float(sizing))
        recommended = None
        for key in ("recommended_size", "risk_adjusted_size", "max_allowed_size"):
            if isinstance(sizing, dict) and key in sizing:
                recommended = sizing[key]
                break
            if hasattr(sizing, key):
                recommended = getattr(sizing, key)
                break
        try:
            return max(0.0, float(recommended)) if recommended is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _compute_atr(self, df: pd.DataFrame, period: int) -> float:
        if not {"high", "low", "close"}.issubset(df.columns):
            return 0.0
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_series = tr.rolling(max(1, int(period))).mean()
        atr_value = atr_series.iloc[-1]
        if pd.isna(atr_value):
            atr_value = tr.mean()
        return float(atr_value) if pd.notna(atr_value) else 0.0

    def _compute_protective_levels(self, side: str, price: float, atr: float) -> tuple:
        if atr <= 0:
            buffer = price * 0.01
        else:
            buffer = atr
        stop_mult = max(0.1, float(self.tp.stop_loss_atr_mult))
        take_mult = max(0.1, float(self.tp.take_profit_atr_mult))
        if side == "buy":
            stop_loss = max(0.0, price - buffer * stop_mult)
            take_profit = price + buffer * take_mult
        else:
            stop_loss = price + buffer * stop_mult
            take_profit = max(0.0, price - buffer * take_mult)
        return stop_loss, take_profit
