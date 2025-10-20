"""Adapter margin dla Zonda."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

from bot_core.exchanges.base import AccountSnapshot, Environment, ExchangeCredentials, OrderRequest, OrderResult
from bot_core.exchanges.error_mapping import raise_for_zonda_error
from bot_core.exchanges.health import Watchdog
from bot_core.exchanges.zonda.spot import (
    ZondaSpotAdapter,
    _convert_with_intermediaries,
    _direct_rate,
    _extract_pair,
    _to_float,
)


class ZondaMarginAdapter(ZondaSpotAdapter):
    """Rozszerzenie adaptera spot o endpointy margin."""

    name = "zonda_margin"

    def __init__(
        self,
        credentials: ExchangeCredentials,
        *,
        environment: Environment,
        settings: Mapping[str, object] | None = None,
        metrics_registry=None,
        watchdog: Watchdog | None = None,
    ) -> None:
        super().__init__(
            credentials,
            environment=environment,
            settings=settings,
            metrics_registry=metrics_registry,
            watchdog=watchdog,
        )
        config = dict(settings or {})
        currency = str(config.get("valuation_currency", "PLN") or "PLN").upper()
        self._valuation_currency = currency

    def fetch_account_snapshot(self) -> AccountSnapshot:
        def _call() -> AccountSnapshot:
            response = self._signed_request("POST", "/trading/margin/balance")
            if not isinstance(response, Mapping):
                raise RuntimeError("Niepoprawna odpowiedź Zonda margin balance")
            if response.get("status") == "Fail":
                raise_for_zonda_error(
                    status_code=400,
                    payload=response,
                    default_message="Zonda margin balance error",
                )
            balances_section = response.get("balances")
            balances: dict[str, float] = {}
            free_balances: dict[str, float] = {}
            if isinstance(balances_section, Sequence):
                for entry in balances_section:
                    if not isinstance(entry, Mapping):
                        continue
                    currency = entry.get("currency")
                    if not isinstance(currency, str):
                        continue
                    free_amount = _to_float(entry.get("available", 0.0))
                    locked_amount = _to_float(entry.get("locked", 0.0))
                    borrowed = _to_float(entry.get("borrowed", 0.0))
                    balances[currency] = max(free_amount + locked_amount + borrowed, 0.0)
                    free_balances[currency] = max(free_amount, 0.0)
            ticker_payload = self._public_request("/trading/ticker")
            prices: dict[tuple[str, str], float] = {}
            intermediaries: set[str] = set()
            if isinstance(ticker_payload, Mapping):
                items = ticker_payload.get("items")
                if isinstance(items, Mapping):
                    for symbol, entry in items.items():
                        if not isinstance(entry, Mapping):
                            continue
                        rate = _to_float(entry.get("rate"), 0.0)
                        pair = _extract_pair(str(symbol), entry)
                        if pair and rate > 0:
                            prices[pair] = rate
                            intermediaries.update(pair)
            total_equity = 0.0
            available_margin = 0.0
            for asset, balance in balances.items():
                direct = _direct_rate(asset, self._valuation_currency, prices)
                if direct is None:
                    direct = _convert_with_intermediaries(
                        asset,
                        self._valuation_currency,
                        prices,
                        tuple(intermediaries),
                    )
                if direct is None:
                    continue
                total_equity += balance * direct
                available_margin += free_balances.get(asset, 0.0) * direct
            maintenance_margin = _to_float(response.get("requiredMargin", response.get("maintenanceMargin", 0.0)))
            return AccountSnapshot(
                balances=balances,
                total_equity=total_equity,
                available_margin=available_margin,
                maintenance_margin=maintenance_margin,
            )

        return self._watchdog.execute("zonda_margin_fetch_account", _call)

    def place_order(self, request: OrderRequest) -> OrderResult:
        def _call() -> OrderResult:
            payload = {
                "market": request.symbol,
                "side": request.side.lower(),
                "type": request.order_type.lower(),
                "amount": f"{request.quantity:.8f}",
            }
            if request.price is not None:
                payload["price"] = f"{request.price:.8f}"
            if request.time_in_force:
                payload["timeInForce"] = request.time_in_force
            if request.client_order_id:
                payload["clientOrderId"] = request.client_order_id
            response = self._signed_request("POST", "/trading/margin/offer", data=payload)
            if not isinstance(response, Mapping):
                raise RuntimeError("Niepoprawna odpowiedź margin/offer Zonda")
            if response.get("status") == "Fail":
                raise_for_zonda_error(
                    status_code=400,
                    payload=response,
                    default_message="Zonda margin order error",
                )
            order_section = response.get("order")
            if not isinstance(order_section, Mapping):
                raise RuntimeError("Brak danych zamówienia w odpowiedzi margin")
            order_id = str(order_section.get("id", ""))
            status = str(order_section.get("status", "new")).upper()
            filled = _to_float(order_section.get("filledAmount", 0.0))
            avg_price = _to_float(order_section.get("avgPrice", 0.0)) or None
            return OrderResult(
                order_id=order_id,
                status=status,
                filled_quantity=filled,
                avg_price=avg_price,
                raw_response=order_section,
            )

        return self._watchdog.execute("zonda_margin_place_order", _call)

    def cancel_order(self, order_id: str, *, symbol: Optional[str] = None) -> None:
        def _call() -> None:
            response = self._signed_request("DELETE", f"/trading/margin/order/{order_id}")
            if not isinstance(response, Mapping):
                raise RuntimeError("Niepoprawna odpowiedź margin/order/{id}")
            if response.get("status") == "Fail":
                raise_for_zonda_error(
                    status_code=400,
                    payload=response,
                    default_message="Zonda margin cancel error",
                )

        self._watchdog.execute("zonda_margin_cancel_order", _call)


__all__ = ["ZondaMarginAdapter"]
