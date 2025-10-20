"""Adapter margin Binance oparty na implementacji spot."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)
from bot_core.exchanges.binance.spot import (
    BinanceSpotAdapter,
    _convert_to_target,
    _to_float,
)
from bot_core.exchanges.binance.symbols import to_exchange_symbol
from bot_core.exchanges.error_mapping import raise_for_binance_error
from bot_core.exchanges.health import Watchdog


class BinanceMarginAdapter(BinanceSpotAdapter):
    """Adapter REST obsługujący operacje margin na Binance."""

    name = "binance_margin"

    def __init__(
        self,
        credentials: ExchangeCredentials,
        *,
        environment: Environment | None = None,
        settings: Mapping[str, object] | None = None,
        metrics_registry=None,
        watchdog: Watchdog | None = None,
    ) -> None:
        super().__init__(
            credentials,
            environment=environment or credentials.environment,
            settings=settings,
            metrics_registry=metrics_registry,
        )
        config = dict(settings or {})
        margin_type = str(config.get("margin_type", "cross") or "cross").lower()
        if margin_type not in {"cross", "isolated"}:
            margin_type = "cross"
        self._margin_type = margin_type
        self._watchdog = watchdog or Watchdog()

    # ------------------------------------------------------------------
    # ExchangeAdapter API
    # ------------------------------------------------------------------
    def fetch_account_snapshot(self) -> AccountSnapshot:
        """Zwraca znormalizowany stan rachunku margin Binance."""

        def _call() -> AccountSnapshot:
            payload = self._signed_request("/sapi/v1/margin/account")
            if not isinstance(payload, Mapping):
                raise RuntimeError("Niepoprawna odpowiedź margin/account Binance")
            if "code" in payload:
                raise_for_binance_error(
                    status_code=int(payload.get("status", 400) or 400),
                    payload=payload,
                    default_message="Binance margin account error",
                )
            assets = payload.get("userAssets")
            balances: dict[str, float] = {}
            free_balances: dict[str, float] = {}
            if isinstance(assets, Sequence):
                for entry in assets:
                    if not isinstance(entry, Mapping):
                        continue
                    asset = entry.get("asset")
                    if not isinstance(asset, str):
                        continue
                    free_amount = _to_float(entry.get("free", 0.0))
                    locked_amount = _to_float(entry.get("locked", 0.0))
                    borrowed = _to_float(entry.get("borrowed", 0.0))
                    net_asset = _to_float(entry.get("netAsset", free_amount + locked_amount - borrowed))
                    balances[asset] = max(net_asset + borrowed, 0.0)
                    free_balances[asset] = max(free_amount, 0.0)
            ticker_payload = self._public_request("/api/v3/ticker/price")
            prices: dict[str, float] = {}
            if isinstance(ticker_payload, Sequence):
                for entry in ticker_payload:
                    if not isinstance(entry, Mapping):
                        continue
                    symbol = entry.get("symbol")
                    price = _to_float(entry.get("price", 0.0))
                    if isinstance(symbol, str):
                        prices[symbol] = price
            total_equity = 0.0
            available_margin = 0.0
            valuation_currency = self._valuation_asset
            secondaries = self._secondary_valuation_assets or ("USDT", "BUSD")
            for asset, balance in balances.items():
                conversion = _convert_to_target(asset, valuation_currency, prices)
                if conversion is None:
                    for secondary in secondaries:
                        first = _convert_to_target(asset, secondary, prices)
                        if first is None:
                            continue
                        second_rate = _convert_to_target(secondary, valuation_currency, prices)
                        if second_rate is None:
                            continue
                        conversion = first * second_rate
                        break
                if conversion is None:
                    continue
                total_equity += balance * conversion
                available_margin += free_balances.get(asset, 0.0) * conversion
            maintenance_margin = _to_float(payload.get("marginLevel", 0.0))
            if maintenance_margin <= 1.0:
                maintenance_margin = _to_float(payload.get("totalMaintMargin", maintenance_margin))
            return AccountSnapshot(
                balances=balances,
                total_equity=total_equity,
                available_margin=available_margin,
                maintenance_margin=maintenance_margin,
            )

        return self._watchdog.execute("binance_margin_fetch_account", _call)

    def place_order(self, request: OrderRequest) -> OrderResult:
        """Składa zlecenie margin na Binance."""

        def _call() -> OrderResult:
            exchange_symbol = to_exchange_symbol(request.symbol)
            if exchange_symbol is None:
                raise ValueError(f"Symbol {request.symbol!r} nie jest obsługiwany przez Binance margin")
            params = {
                "symbol": exchange_symbol,
                "side": request.side.upper(),
                "type": request.order_type.upper(),
                "quantity": request.quantity,
                "isIsolated": "TRUE" if self._margin_type == "isolated" else "FALSE",
            }
            if request.price is not None:
                params["price"] = request.price
            if request.client_order_id:
                params["newClientOrderId"] = request.client_order_id
            if request.time_in_force:
                params["timeInForce"] = request.time_in_force
            payload = self._signed_request("/sapi/v1/margin/order", method="POST", params=params)
            if not isinstance(payload, Mapping):
                raise RuntimeError("Niepoprawna odpowiedź margin/order Binance")
            if "code" in payload:
                raise_for_binance_error(
                    status_code=int(payload.get("status", 400) or 400),
                    payload=payload,
                    default_message="Binance margin order rejected",
                )
            order_id = str(payload.get("orderId", payload.get("clientOrderId", "")))
            status = str(payload.get("status", "NEW")).upper()
            filled_qty = _to_float(payload.get("executedQty", 0.0))
            cumulative_quote = _to_float(payload.get("cummulativeQuoteQty", 0.0))
            avg_price: Optional[float] = None
            if filled_qty > 0 and cumulative_quote > 0:
                avg_price = cumulative_quote / filled_qty
            return OrderResult(
                order_id=order_id,
                status=status,
                filled_quantity=filled_qty,
                avg_price=avg_price,
                raw_response=payload,
            )

        return self._watchdog.execute("binance_margin_place_order", _call)

    def cancel_order(self, order_id: str, *, symbol: Optional[str] = None) -> None:
        """Anuluje zlecenie margin."""

        def _call() -> None:
            params: dict[str, object] = {"orderId": order_id}
            if symbol:
                exchange_symbol = to_exchange_symbol(symbol)
                if exchange_symbol is None:
                    raise ValueError(f"Symbol {symbol!r} nie jest obsługiwany przez Binance margin")
                params["symbol"] = exchange_symbol
            payload = self._signed_request("/sapi/v1/margin/order", method="DELETE", params=params)
            if isinstance(payload, Mapping) and "code" in payload:
                raise_for_binance_error(
                    status_code=int(payload.get("status", 400) or 400),
                    payload=payload,
                    default_message="Binance margin cancel rejected",
                )

        self._watchdog.execute("binance_margin_cancel_order", _call)


__all__ = ["BinanceMarginAdapter"]
