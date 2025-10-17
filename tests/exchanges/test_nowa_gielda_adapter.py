from __future__ import annotations

import tests._pathbootstrap  # noqa: F401  # pylint: disable=unused-import

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.nowa_gielda import NowaGieldaSpotAdapter
from bot_core.exchanges.nowa_gielda import symbols


def _build_adapter() -> NowaGieldaSpotAdapter:
    credentials = ExchangeCredentials(
        key_id="test-key",
        secret="secret",
        environment=Environment.PAPER,
    )
    return NowaGieldaSpotAdapter(credentials)


def test_symbol_mapping_roundtrip() -> None:
    assert symbols.to_exchange_symbol("BTC_USDT") == "BTC-USDT"
    assert symbols.to_internal_symbol("BTC-USDT") == "BTC_USDT"
    supported = tuple(symbols.supported_internal_symbols())
    assert "ETH_USDT" in supported


def test_sign_request_is_deterministic() -> None:
    adapter = _build_adapter()
    payload = {
        "symbol": "BTC-USDT",
        "type": "limit",
        "quantity": 1,
        "price": 25_000,
    }
    signature = adapter.sign_request(
        1_700_000_000_000,
        "POST",
        "/private/orders",
        body=payload,
    )
    assert signature == "4b55db7d55de11856114ec7f289dca6aa58f813bac576e37e6da609db9bc39a9"


def test_rate_limit_rules() -> None:
    adapter = _build_adapter()

    trading_rule = adapter.rate_limit_rule("POST", "/private/orders")
    assert trading_rule is not None
    assert trading_rule.weight == 5
    assert trading_rule.max_requests == 5

    ticker_rule = adapter.rate_limit_rule("GET", "/public/ticker")
    assert ticker_rule is not None
    assert ticker_rule.weight == 1

    assert adapter.request_weight("GET", "/non-existent") == 1
