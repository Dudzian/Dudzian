# core/order_executor.py
# -*- coding: utf-8 -*-
"""Order execution helper used by :mod:`core.trading_engine`."""
from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from KryptoLowca.managers.exchange_core import Mode, OrderStatus  # type: ignore

try:  # pragma: no cover - opcjonalne
    import ccxt  # type: ignore
except Exception:  # pragma: no cover
    ccxt = None  # type: ignore


logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


@dataclass
class ExecutionResult:
    """Represents the outcome of a single order submission."""

    status: str
    order_id: Any
    client_order_id: str
    quantity: float
    price: float
    notional: float
    mode: str
    raw: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable dictionary."""
        payload = {
            "status": self.status,
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "quantity": self.quantity,
            "price": self.price,
            "notional": self.notional,
            "mode": self.mode,
            "raw": self.raw,
        }
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass
class PreparedOrder:
    """Internal structure holding validated order parameters."""

    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    client_order_id: str
    mode: str
    notional: float
    plan: Dict[str, Any]
    position_size: float
    allow_short: bool
    applied_fraction: Optional[float] = None
    db_order_id: Optional[int] = None


class OrderExecutor:
    """Responsible for translating trading plans into exchange orders."""

    def __init__(
        self,
        exchange_manager: Any,
        db_manager: Optional[Any] = None,
        *,
        max_fraction: float = 0.2,
        fee_buffer: float = 0.0015,
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ) -> None:
        self.exchange_manager = exchange_manager
        self.db_manager = db_manager
        self.max_fraction = max(0.0, float(max_fraction))
        self.fee_buffer = max(0.0, float(fee_buffer))
        self.max_retries = max(0, int(max_retries))
        self.retry_delay = max(0.0, float(retry_delay))
        self._lock = asyncio.Lock()
        self._user_id: Optional[int] = None

    def set_user(self, user_id: Optional[int]) -> None:
        """Store user identifier for structured logging."""
        self._user_id = user_id

    async def execute_plan(self, plan: Dict[str, Any]) -> ExecutionResult:
        """Validate, submit and record an order based on *plan*."""
        async with self._lock:
            prepared: Optional[PreparedOrder] = None
            try:
                prepared = await self._prepare_order(plan)
                result = await self._place_order(prepared)
                return result
            except Exception as exc:  # pragma: no cover - awaryjna ścieżka
                error_msg = str(exc)
                logger.exception("Order execution failed for %s: %s", plan.get("symbol"), error_msg)
                if prepared:
                    await self._mark_failure(prepared, error_msg)
                    return ExecutionResult(
                        status="FAILED",
                        order_id=None,
                        client_order_id=prepared.client_order_id,
                        quantity=prepared.quantity,
                        price=float(prepared.price or plan.get("price_ref") or 0.0),
                        notional=0.0,
                        mode=prepared.mode,
                        raw={"error": error_msg},
                        error=error_msg,
                    )
                return ExecutionResult(
                    status="FAILED",
                    order_id=None,
                    client_order_id=str(plan.get("client_order_id", "")),
                    quantity=0.0,
                    price=float(plan.get("price_ref") or 0.0),
                    notional=0.0,
                    mode=self._mode_string(),
                    raw={"error": error_msg},
                    error=error_msg,
                )

    async def _prepare_order(self, plan: Dict[str, Any]) -> PreparedOrder:
        symbol = str(plan.get("symbol") or "").strip()
        if not symbol:
            raise ValueError("Trading plan missing symbol")

        side = str(plan.get("side") or "").lower()
        if side not in {"buy", "sell"}:
            raise ValueError("Trading plan side must be 'buy' or 'sell'")

        order_type = str(plan.get("order_type", "market")).lower()
        if order_type not in {"market", "limit"}:
            raise ValueError("Unsupported order type")

        price_ref = float(plan.get("price_ref") or 0.0)
        if price_ref <= 0:
            raise ValueError("Trading plan missing reference price")

        qty_hint = float(plan.get("qty_hint") or 0.0)
        if qty_hint <= 0:
            raise ValueError("Trading plan recommends zero size")

        allow_short = bool(plan.get("allow_short"))
        capital = plan.get("capital")
        capital_val = float(capital) if capital is not None else None
        if capital_val is None or capital_val <= 0:
            capital_val = await self._fetch_capital(symbol)

        position_qty, _ = self._extract_position(symbol, plan.get("portfolio"))

        quantity, notional, used_fraction = self._determine_quantity(
            symbol=symbol,
            side=side,
            qty_hint=qty_hint,
            price_ref=price_ref,
            capital=capital_val,
            position_qty=position_qty,
            allow_short=allow_short,
            fraction_limit=plan.get("max_fraction"),
        )

        limit_price = None
        if order_type == "limit":
            limit_price = float(plan.get("limit_price") or price_ref)
            limit_price = self._quantize_price(symbol, limit_price)

        quantity = self._quantize_amount(symbol, quantity)
        if quantity <= 0:
            raise ValueError("Order quantity collapsed to zero after quantization")
        notional = quantity * price_ref

        if used_fraction is not None:
            plan["applied_fraction"] = used_fraction

        min_notional = self._min_notional(symbol)
        if min_notional and notional + 1e-12 < min_notional:
            raise ValueError("Order notional below exchange minimum")

        client_order_id = plan.get("client_order_id") or self._generate_client_order_id(symbol)
        plan["client_order_id"] = client_order_id

        prepared = PreparedOrder(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=limit_price,
            client_order_id=client_order_id,
            mode=self._mode_string(),
            notional=notional,
            plan=plan,
            position_size=position_qty,
            allow_short=allow_short,
            applied_fraction=used_fraction,
        )

        prepared.db_order_id = await self._record_initial_order(prepared)
        await self._log(
            "INFO",
            f"Prepared order {symbol} {side} qty={quantity:.8f} notional={notional:.2f}",
            context={
                "client_order_id": client_order_id,
                "order_type": order_type,
                "stop_loss": plan.get("stop_loss"),
                "take_profit": plan.get("take_profit"),
            },
        )
        return prepared

    async def _place_order(self, prepared: PreparedOrder) -> ExecutionResult:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.exchange_manager.create_order(
                    prepared.symbol,
                    prepared.side.upper(),
                    prepared.order_type.upper(),
                    prepared.quantity,
                    prepared.price,
                    prepared.client_order_id,
                )
                if inspect.isawaitable(response):
                    response = await response
                if response is None:
                    raise RuntimeError("Exchange returned empty response")
                result = self._build_execution_result(prepared, response)
                await self._finalise_success(prepared, result)
                return result
            except Exception as exc:  # pragma: no cover - retry branch ciężki do pokrycia
                last_error = exc
                if not self._should_retry(exc) or attempt >= self.max_retries:
                    break
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(
                    "Order attempt %s/%s failed (%s). Retrying in %.2fs",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        error_msg = str(last_error) if last_error else "Unknown order execution error"
        await self._mark_failure(prepared, error_msg)
        return ExecutionResult(
            status="FAILED",
            order_id=None,
            client_order_id=prepared.client_order_id,
            quantity=prepared.quantity,
            price=float(prepared.price or prepared.plan.get("price_ref") or 0.0),
            notional=0.0,
            mode=prepared.mode,
            raw={"error": error_msg},
            error=error_msg,
        )

    def _build_execution_result(self, prepared: PreparedOrder, response: Any) -> ExecutionResult:
        raw = self._normalise_response(response)
        status = str(raw.get("status") or OrderStatus.FILLED.value).upper()
        price = float(raw.get("price") or prepared.plan.get("price_ref") or 0.0)
        quantity = float(raw.get("quantity") or prepared.quantity)
        order_id = raw.get("id") or raw.get("order_id") or raw.get("orderId")
        notional = quantity * price if price else prepared.notional
        return ExecutionResult(
            status=status,
            order_id=order_id,
            client_order_id=prepared.client_order_id,
            quantity=quantity,
            price=price,
            notional=notional,
            mode=prepared.mode,
            raw=raw,
        )

    async def _finalise_success(self, prepared: PreparedOrder, result: ExecutionResult) -> None:
        if self.db_manager and (prepared.db_order_id or prepared.client_order_id):
            try:
                await self.db_manager.update_order_status(
                    order_id=prepared.db_order_id,
                    client_order_id=prepared.client_order_id,
                    status=result.status,
                    price=result.price,
                    exchange_order_id=str(result.order_id) if result.order_id is not None else None,
                    extra={
                        "execution": result.raw,
                        "notional": result.notional,
                        "mode": result.mode,
                    },
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to update order status: %s", exc)
        await self._log(
            "INFO",
            f"Order executed {prepared.symbol} {prepared.side} qty={result.quantity:.8f} status={result.status}",
            context={
                "client_order_id": prepared.client_order_id,
                "mode": result.mode,
                "price": result.price,
                "notional": result.notional,
            },
        )

    async def _mark_failure(self, prepared: PreparedOrder, error_msg: str) -> None:
        if self.db_manager and (prepared.db_order_id or prepared.client_order_id):
            try:
                await self.db_manager.update_order_status(
                    order_id=prepared.db_order_id,
                    client_order_id=prepared.client_order_id,
                    status=OrderStatus.REJECTED.value,
                    extra={
                        "error": error_msg,
                        "plan": self._safe_json(prepared.plan),
                    },
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to mark order failure: %s", exc)
        await self._log(
            "ERROR",
            f"Order execution failed {prepared.symbol} {prepared.side}: {error_msg}",
            context={
                "client_order_id": prepared.client_order_id,
                "mode": prepared.mode,
            },
        )

    async def _record_initial_order(self, prepared: PreparedOrder) -> Optional[int]:
        if not self.db_manager:
            return None
        try:
            payload = {
                "symbol": prepared.symbol,
                "side": prepared.side.upper(),
                "type": prepared.order_type.upper(),
                "quantity": prepared.quantity,
                "price": prepared.price,
                "status": "NEW",
                "client_order_id": prepared.client_order_id,
                "mode": prepared.mode,
                "extra": {
                    "strength": prepared.plan.get("strength"),
                    "stop_loss": prepared.plan.get("stop_loss"),
                    "take_profit": prepared.plan.get("take_profit"),
                    "applied_fraction": prepared.applied_fraction,
                    "max_fraction": prepared.plan.get("max_fraction"),
                    "risk": prepared.plan.get("risk"),
                },
            }
            order_id = await self.db_manager.record_order(payload)
            return order_id
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to record initial order: %s", exc)
            return None

    async def _fetch_capital(self, symbol: str) -> float:
        balance = self.exchange_manager.fetch_balance()
        if inspect.isawaitable(balance):
            balance = await balance
        quote = self._quote_asset(symbol)
        return self._extract_balance_amount(balance, quote)

    def _determine_quantity(
        self,
        *,
        symbol: str,
        side: str,
        qty_hint: float,
        price_ref: float,
        capital: float,
        position_qty: float,
        allow_short: bool,
        fraction_limit: Optional[float],
    ) -> Tuple[float, float, Optional[float]]:
        min_notional = self._min_notional(symbol)
        fraction: Optional[float] = None
        if qty_hint <= 1.0:
            fraction = max(0.0, qty_hint)
            if self.max_fraction > 0:
                fraction = min(fraction, self.max_fraction)
            if fraction_limit is not None:
                try:
                    cap = float(fraction_limit)
                    fraction = min(fraction, max(0.0, min(1.0, cap)))
                except Exception:
                    pass
        quantity: float
        notional: float
        used_fraction: Optional[float] = None

        if fraction is not None:
            if side == "buy":
                spendable = capital / (1.0 + self.fee_buffer) if capital > 0 else 0.0
                notional = spendable * fraction
                if min_notional and notional < min_notional:
                    notional = min_notional
                if notional * (1.0 + self.fee_buffer) > capital + 1e-9:
                    if capital < min_notional:
                        raise ValueError("Insufficient capital for minimum order size")
                    notional = capital / (1.0 + self.fee_buffer)
                quantity = notional / price_ref
                used_fraction = min(1.0, notional / capital) if capital > 0 else fraction
            else:
                if position_qty > 0 and not allow_short:
                    quantity = position_qty * min(fraction, 1.0)
                    notional = quantity * price_ref
                    used_fraction = min(1.0, notional / capital) if capital > 0 else fraction
                else:
                    if not allow_short and position_qty <= 0:
                        raise ValueError("No position available to sell and shorting disabled")
                    notional = capital * fraction
                    if min_notional and notional < min_notional:
                        notional = min_notional
                    quantity = notional / price_ref
                    used_fraction = min(1.0, notional / capital) if capital > 0 else fraction
        else:
            quantity = qty_hint
            notional = quantity * price_ref

        if side == "sell" and not allow_short and quantity > position_qty + 1e-9:
            quantity = position_qty
            notional = quantity * price_ref

        if quantity <= 0 or notional <= 0:
            raise ValueError("Calculated order size is zero")
        return quantity, notional, used_fraction

    def _quantize_amount(self, symbol: str, amount: float) -> float:
        quantizer = getattr(self.exchange_manager, "quantize_amount", None)
        if callable(quantizer):
            return float(quantizer(symbol, amount))
        return float(f"{amount:.8f}")

    def _quantize_price(self, symbol: str, price: float) -> float:
        quantizer = getattr(self.exchange_manager, "quantize_price", None)
        if callable(quantizer):
            return float(quantizer(symbol, price))
        return float(f"{price:.8f}")

    def _min_notional(self, symbol: str) -> float:
        getter = getattr(self.exchange_manager, "min_notional", None)
        if callable(getter):
            try:
                return float(getter(symbol))
            except Exception:  # pragma: no cover
                return 0.0
        return 0.0

    def _extract_position(self, symbol: str, portfolio: Any) -> Tuple[float, str]:
        positions: List[Dict[str, Any]] = []
        if isinstance(portfolio, dict):
            maybe_positions = portfolio.get("positions")
            if isinstance(maybe_positions, list):
                positions = [p for p in maybe_positions if isinstance(p, dict)]
        elif isinstance(portfolio, list):
            positions = [p for p in portfolio if isinstance(p, dict)]

        for pos in positions:
            if str(pos.get("symbol")) == symbol:
                qty = float(pos.get("qty") or pos.get("quantity") or 0.0)
                side = str(pos.get("side") or "").upper()
                return qty, side
        return 0.0, ""

    def _mode_string(self) -> str:
        mode = getattr(self.exchange_manager, "mode", None)
        if hasattr(mode, "value"):
            return str(getattr(mode, "value"))
        if isinstance(mode, Mode):
            return mode.value
        if isinstance(mode, str):
            return mode
        return Mode.PAPER.value

    def _quote_asset(self, symbol: str) -> str:
        if "/" in symbol:
            return symbol.split("/")[-1].upper()
        if "-" in symbol:
            return symbol.split("-")[-1].upper()
        return "USDT"

    @staticmethod
    def _extract_balance_amount(balance: Any, currency: str) -> float:
        if isinstance(balance, dict):
            if currency in balance and isinstance(balance[currency], (int, float)):
                return float(balance[currency])
            for key in ("free", "total", "balance"):
                section = balance.get(key)
                if isinstance(section, dict) and currency in section:
                    amount = section[currency]
                    if isinstance(amount, (int, float)):
                        return float(amount)
        try:
            return float(balance or 0.0)
        except Exception:  # pragma: no cover
            return 0.0

    def _normalise_response(self, response: Any) -> Dict[str, Any]:
        if hasattr(response, "model_dump"):
            raw = response.model_dump()
        elif isinstance(response, dict):
            raw = dict(response)
        else:
            raw = {"response": str(response)}
        serialised: Dict[str, Any] = {}
        for key, value in raw.items():
            if hasattr(value, "value"):
                serialised[key] = getattr(value, "value")
            elif isinstance(value, (str, int, float, bool)) or value is None:
                serialised[key] = value
            else:
                serialised[key] = str(value)
        if "status" in serialised:
            serialised["status"] = str(serialised["status"]).upper()
        return serialised

    def _should_retry(self, exc: Exception) -> bool:
        retryable = (ConnectionError, asyncio.TimeoutError, TimeoutError)
        if isinstance(exc, retryable):
            return True
        if ccxt:
            network_errors = (
                getattr(ccxt, "NetworkError", Exception),
                getattr(ccxt, "DDoSProtection", Exception),
                getattr(ccxt, "RateLimitExceeded", Exception),
            )
            if isinstance(exc, network_errors):
                return True
        return False

    async def _log(self, level: str, message: str, *, context: Optional[Dict[str, Any]] = None) -> None:
        if not self.db_manager:
            return
        try:
            await self.db_manager.log(
                self._user_id,
                level,
                message,
                category="trade",
                context=self._safe_json(context or {}),
            )
        except Exception:  # pragma: no cover
            logger.debug("Structured log failed", exc_info=True)

    def _generate_client_order_id(self, symbol: str) -> str:
        prefix = symbol.replace("/", "").replace("-", "")[:6].upper()
        return f"BOT-{prefix}-{uuid.uuid4().hex[:12]}"

    def _safe_json(self, data: Any) -> Any:
        if isinstance(data, (str, int, float, bool)) or data is None:
            return data
        if isinstance(data, dict):
            return {str(k): self._safe_json(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._safe_json(v) for v in data]
        return str(data)
