import pytest

from KryptoLowca.ui.trading.app import (
    extract_market_price,
    get_default_market_symbol,
    normalize_market_symbol,
)


def test_normalize_market_symbol_replaces_slash_and_uppercases():
    assert normalize_market_symbol("eth/usdt") == "ETH-USDT"


def test_normalize_market_symbol_uses_default_when_empty():
    assert normalize_market_symbol("") == "BTC-PLN"


def test_normalize_market_symbol_uses_env_default(monkeypatch):
    monkeypatch.setenv("TRADING_GUI_DEFAULT_SYMBOL", "eth/pln")

    assert normalize_market_symbol("") == "ETH-PLN"
    assert get_default_market_symbol() == "ETH-PLN"


def test_extract_market_price_from_flat_payload():
    payload = {"last": "12345.67"}
    assert extract_market_price(payload) == pytest.approx(12345.67)


def test_extract_market_price_from_nested_payload():
    payload = {"ticker": {"rate": "9876.54"}}
    assert extract_market_price(payload) == pytest.approx(9876.54)


def test_extract_market_price_from_items_list():
    payload = {"items": [{"stats": {"close": "4321.0"}}]}
    assert extract_market_price(payload) == pytest.approx(4321.0)


def test_extract_market_price_returns_none_when_missing():
    assert extract_market_price({"status": "Ok"}) is None
