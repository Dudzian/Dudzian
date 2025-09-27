"""Silnik paper trading odzwierciedlający koszty i poślizg."""
from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional

from bot_core.execution.base import ExecutionContext, ExecutionService
from bot_core.exchanges.base import OrderRequest, OrderResult

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class MarketMetadata:
    """Parametry rynku wykorzystywane przez symulator paper tradingu."""

    base_asset: str
    quote_asset: str
    min_quantity: float = 0.0
    min_notional: float = 0.0
    step_size: Optional[float] = None
    tick_size: Optional[float] = None


@dataclass(slots=True)
class LedgerEntry:
    """Pojedynczy wpis audytowy."""

    timestamp: float
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    fee: float
    fee_asset: str
    status: str

    def to_mapping(self) -> Mapping[str, object]:
        return {
            "timestamp": self.timestamp,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "fee": self.fee,
            "fee_asset": self.fee_asset,
            "status": self.status,
        }


class InsufficientBalanceError(RuntimeError):
    """Rzucany, gdy na rachunku brakuje środków na realizację zlecenia."""


class PaperTradingExecutionService(ExecutionService):
    """Symulator giełdy realizujący zlecenia natychmiast z kosztem prowizji i poślizgu."""

    def __init__(
        self,
        markets: Mapping[str, MarketMetadata],
        *,
        initial_balances: Optional[Mapping[str, float]] = None,
        maker_fee: float = 0.0004,
        taker_fee: float = 0.0006,
        slippage_bps: float = 5.0,
        time_source: Optional[Callable[[], float]] = None,
    ) -> None:
        if not markets:
            raise ValueError("Wymagana jest co najmniej jedna definicja rynku.")
        self._markets: Dict[str, MarketMetadata] = dict(markets)
        self._balances: MutableMapping[str, float] = {asset: float(value) for asset, value in (initial_balances or {}).items()}
        self._maker_fee = max(0.0, maker_fee)
        self._taker_fee = max(0.0, taker_fee)
        self._slippage_bps = max(0.0, slippage_bps)
        self._time = time_source or time.time
        self._order_counter = itertools.count(1)
        self._ledger: List[LedgerEntry] = []

    # --- API ExecutionService -------------------------------------------------
    def execute(self, request: OrderRequest, context: ExecutionContext) -> OrderResult:
        symbol = request.symbol
        market = self._markets.get(symbol)
        if market is None:
            raise KeyError(f"Brak konfiguracji rynku dla symbolu {symbol}")

        side = request.side.lower()
        if side not in {"buy", "sell"}:
            raise ValueError("Obsługiwane są wyłącznie zlecenia kupna lub sprzedaży.")

        if request.quantity <= 0:
            raise ValueError("Wielkość zlecenia musi być dodatnia.")

        self._validate_lot_size(request.quantity, market)

        reference_price = self._determine_reference_price(request)
        notional = reference_price * request.quantity
        if notional < market.min_notional:
            raise ValueError(
                f"Notional {notional:.8f} jest mniejszy niż minimalna wartość {market.min_notional:.8f} dla {symbol}."
            )

        fill_price = self._apply_slippage(reference_price, side)
        fee_rate = self._taker_fee if request.order_type.lower() == "market" else self._maker_fee
        fee = request.quantity * fill_price * fee_rate

        if side == "buy":
            self._process_buy(symbol, market, request.quantity, fill_price, fee)
        else:
            self._process_sell(symbol, market, request.quantity, fill_price, fee)

        order_id = f"paper-{next(self._order_counter)}"
        result = OrderResult(
            order_id=order_id,
            status="filled",
            filled_quantity=request.quantity,
            avg_price=fill_price,
            raw_response={
                "environment": context.environment,
                "risk_profile": context.risk_profile,
                "portfolio_id": context.portfolio_id,
                "fee": fee,
                "fee_asset": market.quote_asset if side == "buy" else market.quote_asset,
            },
        )
        self._ledger.append(
            LedgerEntry(
                timestamp=self._time(),
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=request.quantity,
                price=fill_price,
                fee=fee,
                fee_asset=market.quote_asset,
                status=result.status,
            )
        )
        _LOGGER.debug(
            "Paper trade %s %s qty=%s price=%s fee=%s (env=%s)",
            side,
            symbol,
            request.quantity,
            fill_price,
            fee,
            context.environment,
        )
        return result

    def cancel(self, order_id: str, context: ExecutionContext) -> None:  # noqa: ARG002 - wymagane przez interfejs
        # Zlecenia w trybie paper trading są realizowane natychmiast – rejestrujemy jedynie próbę anulacji.
        self._ledger.append(
            LedgerEntry(
                timestamp=self._time(),
                order_id=order_id,
                symbol="*",
                side="cancel",
                quantity=0.0,
                price=0.0,
                fee=0.0,
                fee_asset="",
                status="cancelled",
            )
        )
        _LOGGER.info("Zarejestrowano anulację %s w symulatorze paper trading (brak otwartych zleceń).", order_id)

    def flush(self) -> None:
        # Symulator realizuje zlecenia natychmiast – flush nie musi nic robić.
        _LOGGER.debug("PaperTradingExecutionService.flush() – brak zaległych operacji.")

    # --- Funkcje pomocnicze ---------------------------------------------------
    def _determine_reference_price(self, request: OrderRequest) -> float:
        if request.price is None or request.price <= 0:
            raise ValueError("Do symulacji wymagane jest podanie ceny referencyjnej w polu price.")
        return request.price

    def _apply_slippage(self, price: float, side: str) -> float:
        if self._slippage_bps <= 0:
            return price
        adjustment = price * (self._slippage_bps / 10_000.0)
        if side == "buy":
            return price + adjustment
        return max(0.0, price - adjustment)

    def _validate_lot_size(self, quantity: float, market: MarketMetadata) -> None:
        if quantity < market.min_quantity:
            raise ValueError(
                f"Wielkość zlecenia {quantity:.8f} jest mniejsza niż minimalna {market.min_quantity:.8f}."
            )
        if market.step_size:
            remainder = (quantity / market.step_size) - round(quantity / market.step_size)
            if abs(remainder) > 1e-8:
                raise ValueError(
                    f"Wielkość {quantity:.8f} nie spełnia kroku {market.step_size:.8f} dla rynku {market.base_asset}/{market.quote_asset}."
                )

    def _process_buy(
        self,
        symbol: str,
        market: MarketMetadata,
        quantity: float,
        price: float,
        fee: float,
    ) -> None:
        cost = quantity * price + fee
        quote_balance = self._balances.get(market.quote_asset, 0.0)
        if quote_balance + 1e-12 < cost:
            raise InsufficientBalanceError(
                f"Brak środków {market.quote_asset}. Dostępne {quote_balance:.8f}, wymagane {cost:.8f}."
            )
        self._balances[market.quote_asset] = quote_balance - cost
        self._balances[market.base_asset] = self._balances.get(market.base_asset, 0.0) + quantity
        _LOGGER.debug(
            "BUY %s: -%s %s, +%s %s (fee=%s)",
            symbol,
            cost,
            market.quote_asset,
            quantity,
            market.base_asset,
            fee,
        )

    def _process_sell(
        self,
        symbol: str,
        market: MarketMetadata,
        quantity: float,
        price: float,
        fee: float,
    ) -> None:
        base_balance = self._balances.get(market.base_asset, 0.0)
        if base_balance + 1e-12 < quantity:
            raise InsufficientBalanceError(
                f"Brak pozycji {market.base_asset}. Dostępne {base_balance:.8f}, wymagane {quantity:.8f}."
            )
        proceed = quantity * price - fee
        if proceed < 0:
            proceed = 0.0
        self._balances[market.base_asset] = base_balance - quantity
        self._balances[market.quote_asset] = self._balances.get(market.quote_asset, 0.0) + proceed
        _LOGGER.debug(
            "SELL %s: -%s %s, +%s %s (fee=%s)",
            symbol,
            quantity,
            market.base_asset,
            proceed,
            market.quote_asset,
            fee,
        )

    # --- Funkcje obserwowalne -------------------------------------------------
    def balances(self) -> Mapping[str, float]:
        """Zwraca bieżące saldo konta paper trading."""

        return dict(self._balances)

    def ledger(self) -> Iterable[Mapping[str, object]]:
        """Zwraca kopię wpisów audytowych (do raportów compliance)."""

        return [entry.to_mapping() for entry in self._ledger]


__all__ = [
    "PaperTradingExecutionService",
    "MarketMetadata",
    "LedgerEntry",
    "InsufficientBalanceError",
]
