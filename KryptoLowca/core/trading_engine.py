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
from typing import Any, Callable, Dict, Optional, Tuple, Union

import pandas as pd

# Odporne importy: najpierw przestrzeń nazw KryptoLowca, potem lokalne
try:  # pragma: no cover
    from KryptoLowca.core.order_executor import ExecutionResult, OrderExecutor  # type: ignore
    from KryptoLowca.managers.ai_manager import AIManager  # type: ignore
    from KryptoLowca.managers.database_manager import DatabaseManager  # type: ignore
    from KryptoLowca.managers.exchange_manager import ExchangeManager  # type: ignore
    from KryptoLowca.managers.risk_manager_adapter import RiskManager  # type: ignore
    from KryptoLowca.trading_strategies import (  # type: ignore
        EngineConfig,
        TradingParameters,
        TradingStrategies,
    )
except Exception:  # pragma: no cover
    from core.order_executor import ExecutionResult, OrderExecutor
    from managers.ai_manager import AIManager
    from managers.database_manager import DatabaseManager
    from managers.exchange_manager import ExchangeManager
    from managers.risk_manager_adapter import RiskManager
    from trading_strategies import EngineConfig, TradingParameters, TradingStrategies

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# --- Custom exceptions ---
class TradingError(Exception):
    """Raised when trading operations fail."""


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
        self._event_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self._lock = asyncio.Lock()
        self._symbol_locks: Dict[str, asyncio.Lock] = {}
        self._user_id: Optional[int] = None
        self._order_executor: Optional[OrderExecutor] = None

    async def configure(self, ex_mgr: ExchangeManager, ai_mgr: AIManager, risk_mgr: RiskManager) -> None:
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

            # jeśli ExchangeManager wspiera alerty – podłącz handler
            if hasattr(self.ex_mgr, "register_alert_handler"):
                try:
                    self.ex_mgr.register_alert_handler(self._handle_exchange_alert)  # type: ignore[arg-type]
                except Exception:  # pragma: no cover
                    logger.warning("Failed to register exchange alert handler", exc_info=True)

            # wykonawca z limitem frakcji zsynchronizowanym z konfiguracją
            self._order_executor = OrderExecutor(
                ex_mgr,
                self.db_manager,
                max_fraction=self._fraction_cap(),
            )
            self._order_executor.set_user(self._user_id)

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

    @staticmethod
    def _derive_quote_currency(symbol: str) -> str:
        if "/" in symbol:
            return symbol.split("/")[-1].upper()
        if "-" in symbol:
            return symbol.split("-")[-1].upper()
        return "USDT"

    def set_parameters(self, tp: TradingParameters, ec: EngineConfig) -> None:
        """Set trading parameters and configuration."""
        try:
            if hasattr(tp, "validate"):
                tp.validate()
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"Invalid trading parameters: {exc}") from exc
        try:
            if hasattr(ec, "validate"):
                ec.validate()
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"Invalid engine config: {exc}") from exc
        self.tp = tp
        self.ec = ec
        if self.db_manager:
            async def _log_params() -> None:
                await self.db_manager.log(
                    self._user_id,
                    "INFO",
                    f"Parameters set: {tp.__dict__}, {ec.__dict__}",
                    category="engine",
                )

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(_log_params())
            else:
                loop.create_task(_log_params())
        if self._order_executor:
            self._order_executor.set_user(self._user_id)
            # utrzymuj cap frakcji w OrderExecutor w sync z tp/ec
            self._order_executor.max_fraction = self._fraction_cap(tp, ec)

    def on_event(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register event callback for GUI updates."""
        self._event_callback = callback

    def _handle_exchange_alert(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Propagate krytyczne alerty z warstwy giełdowej do logów i GUI."""
        logger.critical("[ALERT] %s | context=%s", message, context or {})

        if self.db_manager:
            async def _log_alert() -> None:
                await self.db_manager.log(
                    self._user_id,
                    "CRITICAL",
                    message,
                    category="exchange",
                    context=context,
                )

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(_log_alert())
            else:
                loop.create_task(_log_alert())

        if self._event_callback:
            payload = {"type": "alert", "message": message, "context": context or {}}
            try:
                self._event_callback(payload)
            except Exception:  # pragma: no cover
                logger.warning("Event callback for alert failed", exc_info=True)

    async def _emit_event(self, event: Dict[str, Any]) -> None:
        """Emit an event to the callback."""
        if self._event_callback:
            try:
                self._event_callback(event)
            except Exception as exc:  # pragma: no cover
                logger.warning("Event callback raised error: %s", exc, exc_info=True)
        if self.db_manager:
            await self.db_manager.log(
                self._user_id,
                "INFO",
                f"Event emitted: {event}",
                category="engine",
            )

    async def execute_live_tick(
        self,
        symbol: str,
        df: pd.DataFrame,
        preds: pd.Series,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute trading logic for a single tick.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT").
            df: DataFrame with OHLCV data.
            preds: Series with AI predictions (in basis points).

        Returns:
            Trading plan (possibly with execution details) or ``None`` if no action is taken.
        """
        symbol_key = str(symbol or "").strip()
        if not symbol_key:
            raise ValueError("Invalid symbol")

        # Lock per-symbol to serialize concurrent ticks for the same market
        lock = self._symbol_locks.setdefault(symbol_key.upper(), asyncio.Lock())
        async with lock:
            db_manager: Optional[DatabaseManager] = None
            user_id: Optional[int] = None
            try:
                # Snapshot dependencies under engine lock
                async with self._lock:
                    ex_mgr = self.ex_mgr
                    ai_mgr = self.ai_mgr
                    risk_mgr = self.risk_mgr
                    order_executor = self._order_executor
                    db_manager = self.db_manager
                    user_id = self._user_id
                    tp = self.tp
                    ec = self.ec
                    fraction_cap = self._fraction_cap(tp, ec)

                if df.empty or len(df) < ec.min_data_points:
                    raise ValueError(
                        f"Insufficient data: {len(df)} bars, required {ec.min_data_points}"
                    )
                if len(preds) != len(df):
                    raise ValueError("Predictions length does not match data length")
                # For planning we only require core managers; executor is optional unless auto_execute=True
                if not ex_mgr or not ai_mgr or not risk_mgr:
                    raise TradingError("Trading engine is not fully configured")

                positions = await db_manager.get_positions(user_id) if db_manager else []
                max_positions = getattr(tp, "max_position_size", getattr(tp, "max_positions", 5))
                try:
                    max_positions = int(max_positions)
                except Exception:
                    max_positions = 5
                if len(positions) >= max_positions:
                    await self._emit_event(
                        {"type": "max_positions_reached", "symbol": symbol_key, "positions": len(positions)}
                    )
                    return None

                if hasattr(self.strategies, "run_strategy"):
                    try:
                        self.strategies.run_strategy(df, tp)
                    except Exception as exc:  # pragma: no cover
                        logger.warning("Strategy execution skipped: %s", exc)

                latest_pred = float(preds.iloc[-1])
                latest_price = float(df["close"].iloc[-1])
                if latest_price <= 0:
                    raise ValueError("Latest price must be positive")

                shorting_enabled = bool(getattr(ec, "enable_shorting", False))
                if latest_pred > 0:
                    side = "buy"
                elif latest_pred < 0 and shorting_enabled:
                    side = "sell"
                else:
                    await self._emit_event({"type": "no_trade_signal", "symbol": symbol_key})
                    return None

                balance_data = ex_mgr.fetch_balance()
                if asyncio.iscoroutine(balance_data):
                    balance_data = await balance_data
                quote_currency = self._derive_quote_currency(symbol_key)
                capital = self._extract_balance_amount(balance_data, currency=quote_currency)
                if capital <= 0:
                    raise TradingError("Insufficient balance")

                portfolio_ctx = {"positions": positions, "capital": capital}

                # Risk sizing may return either qty or (qty, details)
                if "return_details" in risk_mgr.calculate_position_size.__code__.co_varnames:  # type: ignore[attr-defined]
                    sizing_outcome: Union[float, Tuple[float, Dict[str, Any]]] = risk_mgr.calculate_position_size(
                        symbol_key, latest_pred, df, portfolio_ctx, return_details=True
                    )
                else:
                    sizing_outcome = risk_mgr.calculate_position_size(symbol_key, latest_pred, df, portfolio_ctx)

                risk_details: Optional[Dict[str, Any]]
                if isinstance(sizing_outcome, tuple):
                    qty_hint, risk_details = sizing_outcome
                else:
                    qty_hint = sizing_outcome  # type: ignore[assignment]
                    getter = getattr(risk_mgr, "last_position_details", None)
                    risk_details = getter() if callable(getter) else None

                try:
                    qty_hint = float(qty_hint)  # type: ignore[arg-type]
                except Exception:
                    qty_hint = 0.0

                if risk_details is None:
                    risk_details = {"recommended_size": qty_hint}
                else:
                    risk_details.setdefault("recommended_size", qty_hint)

                if qty_hint <= 0:
                    await self._emit_event({"type": "no_position_size", "symbol": symbol_key})
                    return None

                # Apply global fraction cap
                risk_details["max_fraction_cap"] = fraction_cap
                if fraction_cap <= 0:
                    await self._emit_event({"type": "fraction_cap_zero", "symbol": symbol_key})
                    return None

                atr_window = max(1, int(getattr(tp, "atr_period", 14)))
                tr = (df["high"] - df["low"]).abs()
                atr = float(tr.tail(atr_window).mean()) if not tr.empty else 0.0
                if atr <= 0:
                    atr = latest_price * 0.01

                threshold_bps = float(getattr(ai_mgr, "ai_threshold_bps", 1.0) or 1.0)
                plan: Dict[str, Any] = {
                    "symbol": symbol_key,
                    "side": side,
                    "qty_hint": qty_hint,
                    "price_ref": latest_price,
                    "strength": abs(latest_pred) / threshold_bps,
                    "stop_loss": latest_price - atr * tp.stop_loss_atr_mult
                    if side == "buy"
                    else latest_price + atr * tp.stop_loss_atr_mult,
                    "take_profit": latest_price + atr * tp.take_profit_atr_mult
                    if side == "buy"
                    else latest_price - atr * tp.take_profit_atr_mult,
                    "order_type": "market",
                    "capital": capital,
                    "portfolio": {"positions": positions},
                    "allow_short": shorting_enabled,
                    "quote_currency": quote_currency,
                    "user_id": user_id,
                    "max_fraction": fraction_cap,
                    "risk": risk_details,
                }

                if qty_hint <= 1.0:
                    plan["applied_fraction"] = min(max(qty_hint, 0.0), fraction_cap)

                await self._emit_event({"type": "plan_created", "symbol": symbol_key, "plan": plan})
                if db_manager:
                    await db_manager.log(
                        user_id,
                        "INFO",
                        f"Trading plan created for {symbol_key}: {plan}",
                        category="trade",
                    )

                auto_execute = bool(getattr(ec, "auto_execute", True))
                if auto_execute and order_executor:
                    await self._emit_event(
                        {"type": "order_submitting", "symbol": symbol_key, "plan": plan}
                    )
                    execution: ExecutionResult = await order_executor.execute_plan(plan)
                    plan["execution"] = execution.to_dict()
                    if plan.get("risk") and isinstance(plan["risk"], dict):
                        plan["risk"]["executed_fraction"] = plan.get("applied_fraction")
                    event_payload = {
                        "symbol": symbol_key,
                        "execution": plan["execution"],
                        "type": "order_filled"
                        if execution.status.upper() == "FILLED"
                        else "order_submitted",
                    }
                    if execution.error:
                        event_payload["type"] = "order_failed"
                        event_payload["error"] = execution.error
                    await self._emit_event(event_payload)
                    if db_manager:
                        await db_manager.log(
                            user_id,
                            "ERROR" if execution.error else "INFO",
                            f"Order execution result for {symbol_key}: {plan['execution']}",
                            category="trade",
                        )

                return plan

            except ValueError as exc:
                logger.error("Trading tick failed for %s: %s", symbol_key, exc)
                if db_manager:
                    await db_manager.log(
                        user_id,
                        "ERROR",
                        f"Trading tick failed for {symbol_key}: {exc}",
                        category="trade",
                    )
                await self._emit_event({"type": "error", "symbol": symbol_key, "error": str(exc)})
                raise
            except Exception as exc:
                logger.error("Trading tick failed for %s: %s", symbol_key, exc, exc_info=True)
                if db_manager:
                    await db_manager.log(
                        user_id,
                        "ERROR",
                        f"Trading tick failed for {symbol_key}: {exc}",
                        category="trade",
                    )
                await self._emit_event({"type": "error", "symbol": symbol_key, "error": str(exc)})
                raise TradingError(str(exc)) from exc

    def _fraction_cap(
        self,
        tp: Optional[TradingParameters] = None,
        ec: Optional[EngineConfig] = None,
    ) -> float:
        """Określ maksymalną frakcję kapitału na trade z konfiguracji."""
        tp = tp or self.tp
        ec = ec or self.ec
        raw_values = []
        for candidate in (
            getattr(ec, "capital_fraction", None),
            getattr(tp, "position_size", None),
        ):
            if candidate is None:
                continue
            try:
                raw_values.append(float(candidate))
            except Exception:
                continue
        if not raw_values:
            return 1.0
        normalised = [max(0.0, min(1.0, value)) for value in raw_values if value >= 0.0]
        if not normalised:
            return 1.0
        return min(normalised)
