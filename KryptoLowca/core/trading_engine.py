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
from typing import Dict, Optional, Callable, Any, Awaitable
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

    def _schedule_background(self, coro: Awaitable[Any]) -> None:
        """Uruchamia coroutine w tle niezależnie od tego, czy działa pętla."""

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Brak aktywnej pętli (np. wywołanie z wątku GUI) – tworzymy
            # tymczasową pętlę, aby dokończyć zapis i nie generować
            # "RuntimeError: no running event loop".
            asyncio.run(coro)
        else:
            loop.create_task(coro)

    async def configure(self, ex_mgr: ExchangeManager, ai_mgr: AIManager, risk_mgr: RiskManager):
        """Configure the engine with dependencies."""
        async with self._lock:
            self.ex_mgr = ex_mgr
            self.ai_mgr = ai_mgr
            self.risk_mgr = risk_mgr
            if self.db_manager:
                self._user_id = await self.db_manager.ensure_user("engine_user@example.com")
                await self.db_manager.log(self._user_id, "INFO", "Trading engine configured", category="engine")
            if self.ai_mgr and not hasattr(self.ai_mgr, "ai_threshold_bps"):
                try:
                    setattr(self.ai_mgr, "ai_threshold_bps", float(self.tp.signal_threshold * 10_000))
                except Exception:
                    setattr(self.ai_mgr, "ai_threshold_bps", 5.0)

    @staticmethod
    def _extract_balance_amount(balance: Any, currency: str) -> float:
        try:
            if isinstance(balance, dict):
                if currency in balance and isinstance(balance[currency], (int, float)):
                    return float(balance[currency])
                for key in ("free", "total", "balance"):
                    section = balance.get(key)
                    if isinstance(section, dict) and currency in section:
                        amount = section[currency]
                        if isinstance(amount, (int, float)):
                            return float(amount)
            return float(balance or 0.0)
        except Exception:
            return 0.0

    def set_parameters(self, tp: TradingParameters, ec: EngineConfig):
        """Set trading parameters and configuration."""
        try:
            if hasattr(tp, "validate"):
                tp.validate()
        except Exception as exc:
            raise ValueError(f"Invalid trading parameters: {exc}") from exc
        try:
            if hasattr(ec, "validate"):
                ec.validate()
        except Exception as exc:
            raise ValueError(f"Invalid engine config: {exc}") from exc
        self.tp = tp
        self.ec = ec
        if self.db_manager:
            self._schedule_background(
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
                if not symbol or not isinstance(symbol, str):
                    raise ValueError("Invalid symbol")
                if df.empty or len(df) < self.ec.min_data_points:
                    raise ValueError(f"Insufficient data: {len(df)} bars, required {self.ec.min_data_points}")
                if len(preds) != len(df):
                    raise ValueError("Predictions length does not match data length")

                # Check current positions
                positions = await self.db_manager.get_positions(self._user_id) if self.db_manager else []
                max_positions = getattr(self.tp, "max_position_size", getattr(self.tp, "max_positions", 5))
                try:
                    max_positions = int(max_positions)
                except Exception:
                    max_positions = 5
                if len(positions) >= max_positions:
                    await self._emit_event({"type": "max_positions_reached", "symbol": symbol, "positions": len(positions)})
                    return None

                if hasattr(self.strategies, "run_strategy"):
                    try:
                        self.strategies.run_strategy(df, self.tp)
                    except Exception as exc:
                        logger.warning("Strategy execution skipped: %s", exc)

                # Get latest prediction and price
                latest_pred = preds.iloc[-1]
                latest_price = df["close"].iloc[-1]

                # Determine trade side
                side = "buy" if latest_pred > 0 else "sell" if self.ec.enable_shorting else None
                if side is None:
                    await self._emit_event({"type": "shorting_disabled", "symbol": symbol})
                    return None

                # Calculate position size
                balance_data = self.ex_mgr.fetch_balance() if self.ex_mgr else {}
                if asyncio.iscoroutine(balance_data):
                    balance_data = await balance_data
                capital = self._extract_balance_amount(balance_data, currency="USDT")
                if capital <= 0:
                    raise TradingError("Insufficient balance")
                portfolio = {"capital": capital, "positions": positions}
                qty = self.risk_mgr.calculate_position_size(symbol, latest_pred, df, portfolio)

                # Create trading plan
                plan = {
                    "symbol": symbol,
                    "side": side,
                    "qty_hint": qty,
                    "price_ref": latest_price,
                    "strength": abs(latest_pred) / self.ai_mgr.ai_threshold_bps,
                    "stop_loss": latest_price * (1 - self.tp.stop_loss_atr_mult if side == "buy" else 1 + self.tp.stop_loss_atr_mult),
                    "take_profit": latest_price * (1 + self.tp.take_profit_atr_mult if side == "buy" else 1 - self.tp.take_profit_atr_mult)
                }

                await self._emit_event({"type": "plan_created", "symbol": symbol, "plan": plan})
                if self.db_manager:
                    await self.db_manager.log(self._user_id, "INFO", f"Trading plan created for {symbol}: {plan}", category="trade")
                return plan

            except ValueError as e:
                logger.error(f"Trading tick failed for {symbol}: {e}")
                if self.db_manager:
                    await self.db_manager.log(self._user_id, "ERROR", f"Trading tick failed for {symbol}: {e}", category="trade")
                await self._emit_event({"type": "error", "symbol": symbol, "error": str(e)})
                raise
            except Exception as e:
                logger.error(f"Trading tick failed for {symbol}: {e}")
                if self.db_manager:
                    await self.db_manager.log(self._user_id, "ERROR", f"Trading tick failed for {symbol}: {e}", category="trade")
                await self._emit_event({"type": "error", "symbol": symbol, "error": str(e)})
                return None