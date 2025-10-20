"""Adapter margin dla giełdy Kraken."""
from __future__ import annotations

from typing import Mapping, MutableMapping, Optional, Sequence

from bot_core.exchanges.base import AccountSnapshot, Environment, ExchangeCredentials, OrderRequest, OrderResult
from bot_core.exchanges.error_mapping import raise_for_kraken_error
from bot_core.exchanges.health import Watchdog
from bot_core.exchanges.kraken.spot import (
    KrakenSpotAdapter,
    _RequestContext,
    _to_float,
)


class KrakenMarginAdapter(KrakenSpotAdapter):
    """Rozszerzenie adaptera spot o funkcje margin."""

    name = "kraken_margin"

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
        leverage = str(config.get("leverage", "2") or "2")
        self._leverage = leverage

    def fetch_account_snapshot(self) -> AccountSnapshot:
        def _call() -> AccountSnapshot:
            balance_payload = self._private_request(_RequestContext(path="/0/private/Balance", params={}))
            if not isinstance(balance_payload, Mapping):
                raise RuntimeError("Niepoprawna odpowiedź Kraken Balance")
            raise_for_kraken_error(payload=balance_payload, default_message="Kraken balance error")
            balances: dict[str, float] = {}
            balance_result = balance_payload.get("result")
            if isinstance(balance_result, Mapping):
                for asset, raw_value in balance_result.items():
                    balances[str(asset)] = _to_float(raw_value)

            trade_payload = self._private_request(
                _RequestContext(
                    path="/0/private/TradeBalance",
                    params={"asset": self._valuation_asset},
                )
            )
            if not isinstance(trade_payload, Mapping):
                raise RuntimeError("Niepoprawna odpowiedź Kraken TradeBalance")
            raise_for_kraken_error(payload=trade_payload, default_message="Kraken trade balance error")
            trade_result = trade_payload.get("result") if isinstance(trade_payload, Mapping) else None
            total_equity = _to_float(trade_result.get("eb")) if isinstance(trade_result, Mapping) else 0.0
            available_margin = _to_float(trade_result.get("tb")) if isinstance(trade_result, Mapping) else 0.0
            maintenance_margin = _to_float(trade_result.get("m")) if isinstance(trade_result, Mapping) else 0.0
            return AccountSnapshot(
                balances=balances,
                total_equity=total_equity,
                available_margin=available_margin,
                maintenance_margin=maintenance_margin,
            )

        return self._watchdog.execute("kraken_margin_fetch_account", _call)

    def place_order(self, request: OrderRequest) -> OrderResult:
        def _call() -> OrderResult:
            if "trade" not in self._permission_set:
                raise PermissionError("Poświadczenia Kraken nie mają uprawnień tradingowych.")
            params: MutableMapping[str, object] = {
                "pair": request.symbol,
                "type": request.side.lower(),
                "volume": f"{request.quantity:.10f}",
                "leverage": self._leverage,
            }
            order_type = request.order_type.lower()
            if order_type == "market":
                params["ordertype"] = "market"
            elif order_type == "limit":
                if request.price is None:
                    raise ValueError("Zlecenie limit na Kraken margin wymaga ceny.")
                params["ordertype"] = "limit"
                params["price"] = f"{request.price:.10f}"
            else:
                raise ValueError(f"Nieobsługiwany typ zlecenia Kraken margin: {request.order_type}")
            if request.time_in_force:
                params["timeinforce"] = request.time_in_force.upper()
            if request.client_order_id:
                params["userref"] = request.client_order_id
            payload = self._private_request(_RequestContext(path="/0/private/AddOrder", params=params))
            if not isinstance(payload, Mapping):
                raise RuntimeError("Niepoprawna odpowiedź Kraken AddOrder")
            raise_for_kraken_error(payload=payload, default_message="Kraken margin order error")
            result = payload.get("result") if isinstance(payload, Mapping) else None
            txid_seq = result.get("txid") if isinstance(result, Mapping) else None
            order_id: Optional[str] = None
            if isinstance(txid_seq, Sequence) and txid_seq:
                first = txid_seq[0]
                if isinstance(first, str):
                    order_id = first
            return OrderResult(
                order_id=order_id or "",
                status="NEW",
                filled_quantity=0.0,
                avg_price=None,
                raw_response=result if isinstance(result, Mapping) else {},
            )

        return self._watchdog.execute("kraken_margin_place_order", _call)

    def cancel_order(self, order_id: str, *, symbol: Optional[str] = None) -> None:
        def _call() -> None:
            if "trade" not in self._permission_set:
                raise PermissionError("Poświadczenia Kraken nie mają uprawnień tradingowych.")
            params: Mapping[str, object] = {"txid": order_id}
            payload = self._private_request(_RequestContext(path="/0/private/CancelOrder", params=params))
            if not isinstance(payload, Mapping):
                raise RuntimeError("Niepoprawna odpowiedź Kraken CancelOrder")
            raise_for_kraken_error(payload=payload, default_message="Kraken margin cancel error")
            result = payload.get("result") if isinstance(payload, Mapping) else None
            if not isinstance(result, Mapping) or int(result.get("count", 0)) < 1:
                raise_for_kraken_error(
                    payload={"error": ["EOrder:Cancel rejected"]},
                    default_message="Kraken margin cancel rejected",
                )

        self._watchdog.execute("kraken_margin_cancel_order", _call)


__all__ = ["KrakenMarginAdapter"]
