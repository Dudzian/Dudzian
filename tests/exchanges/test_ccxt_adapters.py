from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from bot_core.exchanges.base import Environment, ExchangeCredentials, OrderRequest
from bot_core.exchanges.bitfinex.spot import BitfinexSpotAdapter
from bot_core.exchanges.bitmex.futures import BitmexFuturesAdapter
from bot_core.exchanges.bitget.spot import BitgetSpotAdapter
from bot_core.exchanges.bitstamp.spot import BitstampSpotAdapter
from bot_core.exchanges.coinbase.spot import CoinbaseSpotAdapter
from bot_core.exchanges.coinbase.futures import CoinbaseFuturesAdapter
from bot_core.exchanges.coinbase.margin import CoinbaseMarginAdapter
from bot_core.exchanges.bybit.spot import BybitSpotAdapter
from bot_core.exchanges.bybit.futures import BybitFuturesAdapter
from bot_core.exchanges.bybit.margin import BybitMarginAdapter
from bot_core.exchanges.deribit.futures import DeribitFuturesAdapter
from bot_core.exchanges.error_mapping import raise_for_http_status
from bot_core.exchanges.errors import (
    ExchangeAPIError,
    ExchangeNetworkError,
    ExchangeThrottlingError,
)
from bot_core.exchanges.okx.spot import OKXSpotAdapter
from bot_core.exchanges.okx.futures import OKXFuturesAdapter
from bot_core.exchanges.okx.margin import OKXMarginAdapter
from bot_core.exchanges.kucoin.spot import KuCoinSpotAdapter
from bot_core.exchanges.gateio.spot import GateIOSpotAdapter
from bot_core.exchanges.gemini.spot import GeminiSpotAdapter
from bot_core.exchanges.health import RetryPolicy, Watchdog
from bot_core.exchanges.huobi.spot import HuobiSpotAdapter
from bot_core.exchanges.mexc.spot import MexcSpotAdapter


@dataclass
class _FakeOrder:
    id: str = "order-1"
    status: str = "open"
    filled: float = 0.0
    remaining: float = 0.5
    price: float | None = 10.5


class _FakeClient:
    def __init__(self) -> None:
        self.symbols = ["BTC/USDT", "ETH/USDT"]
        self._orders: dict[str, _FakeOrder] = {}
        self._cancelled: list[tuple[str, str | None]] = []
        self._calls: dict[str, int] = {}

    def _hit(self, name: str) -> None:
        self._calls[name] = self._calls.get(name, 0) + 1

    def load_markets(self):
        self._hit("load_markets")
        return {symbol: {} for symbol in self.symbols}

    def fetch_balance(self):
        self._hit("fetch_balance")
        return {
            "free": {"BTC": 0.25, "USDT": 5000.0},
            "total": {"BTC": 0.3, "USDT": 5500.0},
            "used": {"BTC": 0.05, "USDT": 500.0},
        }

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None, params=None):
        self._hit("fetch_ohlcv")
        base_timestamp = 1_600_000_000_000
        return [
            [base_timestamp, 10.0, 12.0, 9.0, 11.0, 42.0],
            [base_timestamp + 60_000, 11.0, 13.0, 10.5, 12.5, 43.0],
        ]

    def create_order(self, symbol, order_type, side, amount, price=None, params=None):
        self._hit("create_order")
        order = _FakeOrder(
            id=f"{symbol}-001", status="open", filled=amount / 2, remaining=amount / 2, price=price
        )
        self._orders[order.id] = order
        return {
            "id": order.id,
            "status": order.status,
            "filled": order.filled,
            "remaining": order.remaining,
            "price": order.price,
        }

    def cancel_order(self, order_id, symbol=None, params=None):
        self._hit("cancel_order")
        self._cancelled.append((order_id, symbol))
        return {"status": "cancelled"}


class _OfflineClient(_FakeClient):
    def __init__(self, error: type[Exception]) -> None:
        super().__init__()
        self._error = error

    def fetch_ohlcv(self, *args, **kwargs):  # noqa: D401 - dziedziczenie
        raise self._error("network down")


class _CustomNetworkError(Exception):
    pass


class _CustomBaseError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status: object | None = None,
        status_code: object | None = None,
        http_status: object | None = None,
        response: object | None = None,
        body: object | None = None,
        payload: object | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.status_code = status_code
        self.http_status = http_status
        self.response = response
        self.body = body
        self.payload = payload


class _BaseErrorClient(_FakeClient):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self._error = error

    def fetch_balance(self):
        raise self._error


class _AmbiguousCreateClient(_FakeClient):
    def __init__(self, *, open_orders: list[dict[str, object]]) -> None:
        super().__init__()
        self.open_orders = open_orders
        self.last_create_params: dict[str, object] | None = None

    def create_order(self, symbol, order_type, side, amount, price=None, params=None):
        del symbol, order_type, side, amount, price
        self._hit("create_order")
        self.last_create_params = dict(params or {})
        raise _CustomNetworkError("timeout on create")

    def fetch_open_orders(self, symbol=None, params=None):  # noqa: ARG002
        self._hit("fetch_open_orders")
        return list(self.open_orders)


class _CancelApiErrorClient(_FakeClient):
    def __init__(self, *, status_code: int, payload: object) -> None:
        super().__init__()
        self.status_code = status_code
        self.payload = payload

    def cancel_order(self, order_id, symbol=None, params=None):
        self._hit("cancel_order")
        self._cancelled.append((order_id, symbol))
        raise ExchangeAPIError("cancel failed", status_code=self.status_code, payload=self.payload)


class _AmbiguousCancelClient(_FakeClient):
    def __init__(
        self,
        *,
        cancel_error: Exception,
        open_orders: list[dict[str, object]] | None = None,
        open_orders_error: Exception | None = None,
    ) -> None:
        super().__init__()
        self.cancel_error = cancel_error
        self.open_orders = list(open_orders or [])
        self.open_orders_error = open_orders_error

    def cancel_order(self, order_id, symbol=None, params=None):
        self._hit("cancel_order")
        self._cancelled.append((order_id, symbol))
        raise self.cancel_error

    def fetch_open_orders(self, symbol=None, params=None):  # noqa: ARG002
        self._hit("fetch_open_orders")
        if self.open_orders_error is not None:
            raise self.open_orders_error
        return list(self.open_orders)


class _TransientExchangeError(Exception):
    pass


class _WatchdogMutationClient(_FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.create_attempts = 0
        self.cancel_attempts = 0

    def create_order(self, symbol, order_type, side, amount, price=None, params=None):  # noqa: ARG002
        self.create_attempts += 1
        raise _TransientExchangeError("create timeout")

    def cancel_order(self, order_id, symbol=None, params=None):  # noqa: ARG002
        self.cancel_attempts += 1
        raise _TransientExchangeError("cancel timeout")


class _WatchdogReadClient(_FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.read_attempts = 0

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None, params=None):  # noqa: ARG002
        self.read_attempts += 1
        if self.read_attempts < 3:
            raise _TransientExchangeError("temporary read timeout")
        return [[1_700_000_000_000, 100.0, 101.0, 99.0, 100.5, 10.0]]


class _AlwaysFailWatchdogReadClient(_FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.read_attempts = 0

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None, params=None):  # noqa: ARG002
        self.read_attempts += 1
        raise _TransientExchangeError("permanent read timeout")


class _ProbeWatchdog(Watchdog):
    def __init__(self, *, retry_policy: RetryPolicy) -> None:
        super().__init__(retry_policy=retry_policy, sleep=lambda _: None)
        self.operations: list[str] = []

    def execute(self, operation: str, func):  # type: ignore[override]
        self.operations.append(operation)
        return super().execute(operation, func)


def _build_request(symbol: str = "BTC/USDT") -> OrderRequest:
    return OrderRequest(symbol=symbol, side="buy", quantity=1.0, order_type="limit", price=10.5)


def test_coinbase_adapter_basic_flow():
    credentials = ExchangeCredentials(key_id="k", secret="s", passphrase="p")
    client = _FakeClient()
    adapter = CoinbaseSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        client=client,
    )

    adapter.configure_network(ip_allowlist=())

    symbols = adapter.fetch_symbols()
    assert set(symbols) == {"BTC/USDT", "ETH/USDT"}

    snapshot = adapter.fetch_account_snapshot()
    assert math.isclose(snapshot.total_equity, 5500.3, rel_tol=1e-6)
    assert math.isclose(snapshot.available_margin, 5000.25, rel_tol=1e-6)
    assert math.isclose(snapshot.maintenance_margin, 500.05, rel_tol=1e-6)

    candles = adapter.fetch_ohlcv("BTC/USDT", "1m", start=0, end=2_000_000_000_000, limit=2)
    assert len(candles) == 2

    order = adapter.place_order(_build_request())
    assert order.order_id.endswith("-001")
    assert order.status == "open"

    adapter.cancel_order(order.order_id, symbol="BTC/USDT")
    assert client._cancelled == [(order.order_id, "BTC/USDT")]


def test_adapter_translates_network_errors():
    credentials = ExchangeCredentials(key_id="k")
    client = _OfflineClient(_CustomNetworkError)
    adapter = BitfinexSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        client=client,
        settings={"network_error_types": (_CustomNetworkError,)},
    )

    adapter.configure_network(ip_allowlist=())

    with pytest.raises(ExchangeNetworkError):
        adapter.fetch_ohlcv("BTC/USDT", "1m", start=0, end=1, limit=1)


def test_adapter_base_error_maps_429_to_throttling_with_json_payload():
    credentials = ExchangeCredentials(key_id="k")
    error = _CustomBaseError(
        "rate limited",
        status_code=429,
        payload={"error": {"message": "Too many requests"}},
    )
    adapter = BitfinexSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        client=_BaseErrorClient(error),
        settings={"base_error_types": (_CustomBaseError,)},
    )
    adapter.configure_network(ip_allowlist=())

    with pytest.raises(ExchangeThrottlingError) as exc_info:
        adapter.fetch_account_snapshot()

    assert exc_info.value.status_code == 429
    assert exc_info.value.payload == {"error": {"message": "Too many requests"}}


def test_adapter_base_error_preserves_503_plain_text_payload():
    credentials = ExchangeCredentials(key_id="k")
    error = _CustomBaseError("upstream error", http_status=503, body="gateway timeout")
    adapter = BitfinexSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        client=_BaseErrorClient(error),
        settings={"base_error_types": (_CustomBaseError,)},
    )
    adapter.configure_network(ip_allowlist=())

    with pytest.raises(ExchangeAPIError) as exc_info:
        adapter.fetch_account_snapshot()

    assert exc_info.value.status_code == 503
    assert exc_info.value.payload == "gateway timeout"


def test_adapter_base_error_defaults_when_status_and_payload_missing():
    credentials = ExchangeCredentials(key_id="k")
    error = _CustomBaseError("opaque base error")
    adapter = BitfinexSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        client=_BaseErrorClient(error),
        settings={"base_error_types": (_CustomBaseError,)},
    )
    adapter.configure_network(ip_allowlist=())

    with pytest.raises(ExchangeAPIError) as exc_info:
        adapter.fetch_account_snapshot()

    assert exc_info.value.status_code == 500
    assert exc_info.value.payload == "opaque base error"


def test_raise_for_http_status_distinguishes_429_from_5xx():
    with pytest.raises(ExchangeThrottlingError) as throttle_exc:
        raise_for_http_status(status_code=429, payload="too many requests", default_message="err")
    assert throttle_exc.value.status_code == 429

    with pytest.raises(ExchangeAPIError) as transient_exc:
        raise_for_http_status(
            status_code=503, payload="upstream unavailable", default_message="err"
        )
    assert transient_exc.value.status_code == 503
    assert transient_exc.value.payload == "upstream unavailable"


def test_okx_adapter_respects_offline_cancellation():
    credentials = ExchangeCredentials(key_id="k", secret="s", passphrase="p")
    client = _FakeClient()
    adapter = OKXSpotAdapter(
        credentials,
        environment=Environment.TESTNET,
        client=client,
        settings={"cancel_order_params": {"simulate": True}},
    )

    adapter.configure_network(ip_allowlist=())

    order = adapter.place_order(_build_request("ETH/USDT"))
    adapter.cancel_order(order.order_id, symbol="ETH/USDT")
    assert client._cancelled[-1] == (order.order_id, "ETH/USDT")

    with pytest.raises(NotImplementedError):
        adapter.stream_public_data(channels=["ticker"])


def test_shared_ccxt_cancel_normalizes_obvious_idempotent_api_error():
    credentials = ExchangeCredentials(key_id="k", secret="s", passphrase="p")
    client = _CancelApiErrorClient(
        status_code=404,
        payload='{"error":{"message":"Order not found"}}',
    )
    adapter = CoinbaseSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        client=client,
    )

    adapter.configure_network(ip_allowlist=())
    adapter.cancel_order("order-404", symbol="BTC/USDT")

    assert client._calls["cancel_order"] == 1
    assert client._cancelled == [("order-404", "BTC/USDT")]


def test_shared_ccxt_cancel_raises_for_non_normalized_api_error():
    credentials = ExchangeCredentials(key_id="k", secret="s", passphrase="p")
    client = _CancelApiErrorClient(
        status_code=409,
        payload='{"error":{"message":"Invalid cancel transition"}}',
    )
    adapter = CoinbaseSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        client=client,
    )

    adapter.configure_network(ip_allowlist=())

    with pytest.raises(ExchangeAPIError):
        adapter.cancel_order("order-409", symbol="BTC/USDT")

    assert client._calls["cancel_order"] == 1


def test_shared_ccxt_cancel_ambiguous_network_error_resolves_when_order_absent():
    credentials = ExchangeCredentials(key_id="k", secret="s")
    cancel_error = ExchangeNetworkError("cancel timeout")
    client = _AmbiguousCancelClient(cancel_error=cancel_error, open_orders=[])
    adapter = CoinbaseSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        client=client,
    )
    adapter.configure_network(ip_allowlist=())

    adapter.cancel_order("order-amb", symbol="BTC/USDT")

    assert client._calls["cancel_order"] == 1
    assert client._calls["fetch_open_orders"] == 1


def test_shared_ccxt_cancel_ambiguous_5xx_raises_original_when_order_still_open():
    credentials = ExchangeCredentials(key_id="k", secret="s")
    cancel_error = ExchangeAPIError("cancel ambiguous", status_code=503, payload="upstream timeout")
    client = _AmbiguousCancelClient(
        cancel_error=cancel_error,
        open_orders=[{"id": "order-open"}],
    )
    adapter = CoinbaseSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        client=client,
    )
    adapter.configure_network(ip_allowlist=())

    with pytest.raises(ExchangeAPIError) as exc_info:
        adapter.cancel_order("order-open", symbol="BTC/USDT")

    assert exc_info.value is cancel_error
    assert client._calls["cancel_order"] == 1
    assert client._calls["fetch_open_orders"] == 1


def test_shared_ccxt_cancel_ambiguous_read_failure_raises_original():
    credentials = ExchangeCredentials(key_id="k", secret="s")
    cancel_error = ExchangeThrottlingError("cancel throttled", status_code=429)
    client = _AmbiguousCancelClient(
        cancel_error=cancel_error,
        open_orders_error=ExchangeNetworkError("read failed"),
    )
    adapter = CoinbaseSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        client=client,
    )
    adapter.configure_network(ip_allowlist=())

    with pytest.raises(ExchangeThrottlingError) as exc_info:
        adapter.cancel_order("order-unknown", symbol="BTC/USDT")

    assert exc_info.value is cancel_error
    assert client._calls["cancel_order"] == 1
    assert client._calls["fetch_open_orders"] == 1


def test_place_order_injects_deterministic_anchor_and_reconciles_success():
    credentials = ExchangeCredentials(key_id="k", secret="s")
    client = _AmbiguousCreateClient(
        open_orders=[
            {
                "id": "resolved-1",
                "status": "open",
                "amount": 1.0,
                "remaining": 0.4,
                "filled": 0.6,
                "price": 10.5,
                "clientOrderId": "cli-anchor-1",
            }
        ]
    )
    adapter = CoinbaseSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        client=client,
        settings={
            "network_error_types": (_CustomNetworkError,),
            "client_order_id_param": "newClientOrderId",
        },
    )
    adapter.configure_network(ip_allowlist=())

    result = adapter.place_order(
        OrderRequest(
            symbol="BTC/USDT",
            side="buy",
            quantity=1.0,
            order_type="limit",
            price=10.5,
            client_order_id="cli-anchor-1",
        )
    )

    assert result.order_id == "resolved-1"
    assert client.last_create_params == {"newClientOrderId": "cli-anchor-1"}
    assert client._calls["create_order"] == 1
    assert client._calls["fetch_open_orders"] == 1


def test_place_order_injects_anchor_and_raises_on_reconcile_miss_without_resubmit():
    credentials = ExchangeCredentials(key_id="k", secret="s")
    client = _AmbiguousCreateClient(open_orders=[])
    adapter = CoinbaseSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        client=client,
        settings={
            "network_error_types": (_CustomNetworkError,),
            "client_order_id_param": "newClientOrderId",
        },
    )
    adapter.configure_network(ip_allowlist=())

    with pytest.raises(ExchangeNetworkError):
        adapter.place_order(
            OrderRequest(
                symbol="BTC/USDT",
                side="buy",
                quantity=1.0,
                order_type="limit",
                price=10.5,
                client_order_id="cli-anchor-miss",
            )
        )

    assert client.last_create_params == {"newClientOrderId": "cli-anchor-miss"}
    assert client._calls["create_order"] == 1
    assert client._calls["fetch_open_orders"] == 1


def test_kucoin_adapter_merges_nested_settings():
    credentials = ExchangeCredentials(key_id="k", secret="s")
    client = _FakeClient()
    adapter = KuCoinSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        client=client,
        settings={
            "ccxt_config": {"timeout": 5000, "options": {"adjust": True}},
            "sandbox_mode": False,
        },
    )

    adapter.configure_network(ip_allowlist=())

    assert adapter._settings["ccxt_config"]["timeout"] == 5000
    assert adapter._settings["ccxt_config"]["options"]["defaultType"] == "spot"
    assert adapter._settings["ccxt_config"]["options"]["adjust"] is True
    assert adapter._settings.get("sandbox_mode") is False
    assert adapter._settings["cancel_order_params"]["type"] == "spot"
    rate_rules = adapter._settings["rate_limit_rules"]
    assert rate_rules[0].rate == 30 and rate_rules[0].per == pytest.approx(3.0)
    assert rate_rules[1].rate == 1_800 and rate_rules[1].per == pytest.approx(60.0)


def test_bybit_adapter_provides_spot_defaults():
    credentials = ExchangeCredentials(key_id="k")
    client = _FakeClient()
    adapter = BybitSpotAdapter(
        credentials,
        environment=Environment.TESTNET,
        client=client,
        settings={"fetch_ohlcv_params": {"price": "mark"}},
    )

    adapter.configure_network(ip_allowlist=())

    assert adapter._settings["fetch_ohlcv_params"]["category"] == "spot"


def test_deribit_and_bitmex_futures_apply_environment_stream_defaults() -> None:
    credentials = ExchangeCredentials(key_id="k", secret="s")
    client = _FakeClient()

    deribit = DeribitFuturesAdapter(
        credentials,
        environment=Environment.TESTNET,
        client=client,
    )
    assert deribit._settings["stream"]["base_url"] == "https://stream.sandbox.dudzian.ai/exchanges"

    bitmex = BitmexFuturesAdapter(
        credentials,
        environment=Environment.LIVE,
        client=client,
    )
    assert bitmex._settings["stream"]["base_url"] == "https://stream.hyperion.dudzian.ai/exchanges"
    assert bitmex._settings["retry_policy"]["max_attempts"] == 5
    assert bitmex._settings["retry_policy"]["max_delay"] == pytest.approx(2.5)


def test_huobi_adapter_sets_retry_and_rate_limits():
    credentials = ExchangeCredentials(key_id="k")
    client = _FakeClient()
    adapter = HuobiSpotAdapter(
        credentials,
        environment=Environment.TESTNET,
        client=client,
    )

    adapter.configure_network(ip_allowlist=())

    assert adapter._settings["fetch_ohlcv_params"]["type"] == "spot"
    assert adapter._settings["cancel_order_params"]["type"] == "spot"
    assert adapter._settings["sandbox_mode"] is True
    retry = adapter._retry_policy
    assert retry.max_attempts == 4
    assert retry.base_delay == pytest.approx(0.15)
    rate_rules = adapter._settings["rate_limit_rules"]
    assert rate_rules[0].rate == 90 and rate_rules[0].per == pytest.approx(3.0)
    assert rate_rules[1].rate == 900 and rate_rules[1].per == pytest.approx(60.0)


def test_gemini_adapter_configures_account_and_retry_policy():
    credentials = ExchangeCredentials(key_id="k", secret="s")
    client = _FakeClient()
    adapter = GeminiSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        client=client,
    )

    adapter.configure_network(ip_allowlist=())

    assert adapter._settings["ccxt_config"]["options"]["account"] == "primary"
    assert adapter._settings["sandbox_mode"] is True
    retry = adapter._retry_policy
    assert retry.max_attempts == 5
    assert retry.base_delay == pytest.approx(0.25)
    rate_rules = adapter._settings["rate_limit_rules"]
    assert rate_rules[0].rate == 15 and rate_rules[0].per == pytest.approx(1.0)
    assert rate_rules[1].rate == 1_200 and rate_rules[1].per == pytest.approx(60.0)


def test_bitstamp_adapter_uses_spot_defaults():
    credentials = ExchangeCredentials(key_id="k")
    client = _FakeClient()
    adapter = BitstampSpotAdapter(
        credentials,
        environment=Environment.TESTNET,
        client=client,
    )

    adapter.configure_network(ip_allowlist=())

    assert adapter._settings["ccxt_config"]["options"]["defaultType"] == "spot"
    assert adapter._settings["ccxt_config"]["timeout"] == 20_000


def test_gateio_adapter_configures_spot_behaviour():
    credentials = ExchangeCredentials(key_id="k", secret="s")
    client = _FakeClient()
    adapter = GateIOSpotAdapter(
        credentials,
        environment=Environment.LIVE,
        client=client,
    )

    adapter.configure_network(ip_allowlist=())

    assert adapter._settings["ccxt_config"]["options"]["defaultType"] == "spot"
    assert adapter._settings["fetch_ohlcv_params"]["type"] == "spot"
    assert adapter._settings["cancel_order_params"]["type"] == "spot"


def test_bitget_adapter_configures_spot_category():
    credentials = ExchangeCredentials(key_id="k")
    client = _FakeClient()
    adapter = BitgetSpotAdapter(
        credentials,
        environment=Environment.TESTNET,
        client=client,
    )

    adapter.configure_network(ip_allowlist=())

    ccxt_config = adapter._settings["ccxt_config"]
    assert ccxt_config["options"]["defaultType"] == "spot"
    assert ccxt_config["timeout"] == 15_000
    assert adapter._settings["fetch_ohlcv_params"]["type"] == "spot"
    assert adapter._settings["cancel_order_params"]["type"] == "spot"


def test_mexc_adapter_sets_spot_defaults():
    credentials = ExchangeCredentials(key_id="k", secret="s")
    client = _FakeClient()
    adapter = MexcSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        client=client,
    )

    adapter.configure_network(ip_allowlist=())

    ccxt_config = adapter._settings["ccxt_config"]
    assert ccxt_config["options"]["defaultType"] == "spot"
    assert ccxt_config["timeout"] == 20_000
    assert adapter._settings["fetch_ohlcv_params"]["type"] == "spot"
    assert adapter._settings["cancel_order_params"]["type"] == "spot"


def test_gemini_adapter_merges_settings():
    credentials = ExchangeCredentials(key_id="k")
    client = _FakeClient()
    adapter = GeminiSpotAdapter(
        credentials,
        environment=Environment.LIVE,
        client=client,
        settings={"ccxt_config": {"timeout": 30_000}},
    )

    adapter.configure_network(ip_allowlist=())

    assert adapter._settings["ccxt_config"]["timeout"] == 30_000
    assert adapter._settings["ccxt_config"]["options"]["defaultType"] == "spot"


def test_huobi_adapter_keeps_default_exchange_id():
    credentials = ExchangeCredentials(key_id="k")
    client = _FakeClient()
    adapter = HuobiSpotAdapter(
        credentials,
        environment=Environment.PAPER,
        client=client,
    )

    adapter.configure_network(ip_allowlist=())

    assert adapter._settings["ccxt_config"]["options"]["defaultType"] == "spot"
    assert adapter._exchange_id == "huobi"


_WATCHDOG_FUTURES_MARGIN_ADAPTERS = (
    BybitFuturesAdapter,
    BybitMarginAdapter,
    CoinbaseFuturesAdapter,
    CoinbaseMarginAdapter,
    OKXFuturesAdapter,
    OKXMarginAdapter,
)


@pytest.mark.parametrize("adapter_cls", _WATCHDOG_FUTURES_MARGIN_ADAPTERS)
def test_watchdog_futures_margin_mutations_do_not_retry_explosion(adapter_cls):
    credentials = ExchangeCredentials(key_id="k", secret="s")
    client = _WatchdogMutationClient()
    adapter = adapter_cls(
        credentials,
        environment=Environment.PAPER,
        client=client,
        settings={
            "network_error_types": (_TransientExchangeError,),
            "retry_policy": {
                "max_attempts": 5,
                "base_delay": 0.0,
                "max_delay": 0.0,
                "jitter": (0.0, 0.0),
            },
            "sleep_callable": lambda _: None,
        },
    )
    adapter.configure_network(ip_allowlist=())

    request = OrderRequest(
        symbol="BTC/USDT",
        side="buy",
        quantity=1.0,
        order_type="limit",
        price=10.0,
    )

    with pytest.raises(ExchangeNetworkError):
        adapter.place_order(request)
    with pytest.raises(ExchangeNetworkError):
        adapter.cancel_order("order-1", symbol="BTC/USDT")

    assert client.create_attempts == 1
    assert client.cancel_attempts == 1


@pytest.mark.parametrize("adapter_cls", _WATCHDOG_FUTURES_MARGIN_ADAPTERS)
def test_watchdog_futures_margin_reads_keep_current_retry_semantics(adapter_cls):
    credentials = ExchangeCredentials(key_id="k", secret="s")
    client = _WatchdogReadClient()
    adapter = adapter_cls(
        credentials,
        environment=Environment.PAPER,
        client=client,
        settings={
            "network_error_types": (_TransientExchangeError,),
            "retry_policy": {
                "max_attempts": 3,
                "base_delay": 0.0,
                "max_delay": 0.0,
                "jitter": (0.0, 0.0),
            },
            "sleep_callable": lambda _: None,
        },
    )
    adapter.configure_network(ip_allowlist=())

    candles = adapter.fetch_ohlcv("BTC/USDT", "1m", start=0, end=1_700_000_100_000, limit=1)

    assert len(candles) == 1
    assert client.read_attempts == 3


@pytest.mark.parametrize("adapter_cls", _WATCHDOG_FUTURES_MARGIN_ADAPTERS)
def test_watchdog_retry_true_read_path_uses_watchdog_without_super_runtime_error(adapter_cls):
    credentials = ExchangeCredentials(key_id="k", secret="s")
    client = _WatchdogReadClient()
    watchdog = _ProbeWatchdog(
        retry_policy=RetryPolicy(
            max_attempts=1,
            base_delay=0.0,
            max_delay=0.0,
            jitter=(0.0, 0.0),
        )
    )
    adapter = adapter_cls(
        credentials,
        environment=Environment.PAPER,
        client=client,
        watchdog=watchdog,
        settings={
            "network_error_types": (_TransientExchangeError,),
            "retry_policy": {
                "max_attempts": 3,
                "base_delay": 0.0,
                "max_delay": 0.0,
                "jitter": (0.0, 0.0),
            },
            "sleep_callable": lambda _: None,
        },
    )
    adapter.configure_network(ip_allowlist=())

    candles = adapter.fetch_ohlcv("BTC/USDT", "1m", start=0, end=1_700_000_100_000, limit=1)

    assert len(candles) == 1
    assert client.read_attempts == 3
    assert watchdog.operations == [f"{adapter.name}.fetch_ohlcv"]


@pytest.mark.parametrize("adapter_cls", _WATCHDOG_FUTURES_MARGIN_ADAPTERS)
def test_watchdog_read_path_permanent_failure_attempt_count_is_predictable(adapter_cls):
    credentials = ExchangeCredentials(key_id="k", secret="s")
    client = _AlwaysFailWatchdogReadClient()
    watchdog = _ProbeWatchdog(
        retry_policy=RetryPolicy(
            max_attempts=2,
            base_delay=0.0,
            max_delay=0.0,
            jitter=(0.0, 0.0),
        )
    )
    adapter = adapter_cls(
        credentials,
        environment=Environment.PAPER,
        client=client,
        watchdog=watchdog,
        settings={
            "network_error_types": (_TransientExchangeError,),
            "retry_policy": {
                "max_attempts": 3,
                "base_delay": 0.0,
                "max_delay": 0.0,
                "jitter": (0.0, 0.0),
            },
            "sleep_callable": lambda _: None,
        },
    )
    adapter.configure_network(ip_allowlist=())

    with pytest.raises(ExchangeNetworkError):
        adapter.fetch_ohlcv("BTC/USDT", "1m", start=0, end=1_700_000_100_000, limit=1)

    # Obecna semantyka: inner retry (3 próby) wykonywany dla każdego outer retry watchdog-a (2 próby).
    assert client.read_attempts == 6
    assert watchdog.operations == [f"{adapter.name}.fetch_ohlcv"]
