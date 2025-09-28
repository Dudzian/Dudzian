"""Tests for the backfill CLI helpers."""
from __future__ import annotations

from bot_core.config.loader import load_core_config
from bot_core.exchanges.base import Environment
from bot_core.exchanges.binance.futures import BinanceFuturesAdapter
from bot_core.exchanges.binance.spot import BinanceSpotAdapter
from bot_core.exchanges.kraken.futures import KrakenFuturesAdapter
from bot_core.exchanges.kraken.spot import KrakenSpotAdapter
from bot_core.exchanges.zonda.spot import ZondaSpotAdapter

from scripts.backfill import _build_public_source


def test_build_public_source_supports_core_multi_exchange() -> None:
    config = load_core_config("config/core.yaml")
    universe = config.instrument_universes["core_multi_exchange"]

    exchanges = {
        exchange_name
        for instrument in universe.instruments
        for exchange_name in instrument.exchange_symbols
    }

    expected_types = {
        "binance_spot": BinanceSpotAdapter,
        "binance_futures": BinanceFuturesAdapter,
        "kraken_spot": KrakenSpotAdapter,
        "kraken_futures": KrakenFuturesAdapter,
        "zonda_spot": ZondaSpotAdapter,
    }

    for exchange in exchanges:
        public_source = _build_public_source(exchange, Environment.PAPER)
        adapter_type = expected_types[exchange]
        assert isinstance(public_source.exchange_adapter, adapter_type)
