"""Read-only market data contract tests."""

from __future__ import annotations

import dataclasses
import inspect
import os
import socket
from pathlib import Path
from unittest.mock import patch

import pytest

from bot_core.runtime.preview_modes import PreviewMode, PreviewModePolicy, RuntimeCapability
from bot_core.runtime.read_only_market_data import (
    InMemoryReadOnlyMarketDataProvider,
    MarketCandle,
    MarketQuote,
    ReadOnlyMarketDataError,
    ReadOnlyMarketDataPolicy,
    ReadOnlyMarketDataProvider,
)


def _provider() -> InMemoryReadOnlyMarketDataProvider:
    return InMemoryReadOnlyMarketDataProvider(
        quotes={
            "BTC/USDT": MarketQuote(
                "BTC/USDT", bid=100.0, ask=101.0, last=100.5, timestamp="2026-06-18T00:00:00Z"
            ),
            "ETH/USDT": MarketQuote(
                "ETH/USDT", bid=10.0, ask=11.0, last=10.5, timestamp="2026-06-18T00:00:01Z"
            ),
        },
        candles={
            ("BTC/USDT", "1m"): (
                MarketCandle(
                    "BTC/USDT", "1m", "2026-06-18T00:00:00Z", 100.0, 102.0, 99.0, 101.0, 5.0
                ),
                MarketCandle(
                    "BTC/USDT", "1m", "2026-06-18T00:01:00Z", 101.0, 103.0, 100.0, 102.0, 6.0
                ),
            )
        },
    )


def test_read_only_market_policy_allows_fetch_and_blocks_live_capabilities() -> None:
    policy = ReadOnlyMarketDataPolicy.for_preview_mode(PreviewMode.READ_ONLY_MARKET)

    assert policy.preview_policy.mode is PreviewMode.READ_ONLY_MARKET
    assert policy.preview_policy.capabilities == (RuntimeCapability.READ_ONLY_MARKET_FETCH,)

    for capability in (
        RuntimeCapability.LIVE_ORDER_SUBMIT,
        RuntimeCapability.REAL_EXCHANGE_FILL,
        RuntimeCapability.LIVE_BALANCE_MUTATION,
        RuntimeCapability.LIVE_ACCOUNT_BALANCE_FETCH,
        RuntimeCapability.LIVE_ACCOUNT_SNAPSHOT_READ,
        RuntimeCapability.LIVE_CREDENTIALS_READ,
        RuntimeCapability.PRODUCTION_CLOUD_SINK,
        RuntimeCapability.EXTERNAL_EXPORT_SINK,
        RuntimeCapability.LIVE_SCHEDULER_WORKER_SIDE_EFFECT,
    ):
        with pytest.raises(ReadOnlyMarketDataError):
            ReadOnlyMarketDataPolicy(
                preview_policy=PreviewModePolicy(
                    mode=PreviewMode.READ_ONLY_MARKET,
                    capabilities=(capability,),
                )
            )


@pytest.mark.parametrize(
    "capability",
    [
        RuntimeCapability.LIVE_ORDER_SUBMIT,
        RuntimeCapability.LIVE_ACCOUNT_BALANCE_FETCH,
        RuntimeCapability.PRODUCTION_CLOUD_SINK,
        RuntimeCapability.EXTERNAL_EXPORT_SINK,
    ],
)
def test_direct_unsafe_read_only_market_policy_construction_raises(
    capability: RuntimeCapability,
) -> None:
    with pytest.raises(ReadOnlyMarketDataError):
        ReadOnlyMarketDataPolicy(
            preview_policy=PreviewModePolicy(
                mode=PreviewMode.READ_ONLY_MARKET,
                capabilities=(capability,),
            )
        )


def test_direct_read_only_market_policy_requires_read_only_market_mode() -> None:
    with pytest.raises(ReadOnlyMarketDataError):
        ReadOnlyMarketDataPolicy(
            preview_policy=PreviewModePolicy(
                mode=PreviewMode.PAPER,
                capabilities=(RuntimeCapability.READ_ONLY_MARKET_FETCH,),
            )
        )


def test_provider_cannot_be_created_with_unsafe_policy() -> None:
    unsafe_policy = object.__new__(ReadOnlyMarketDataPolicy)
    object.__setattr__(
        unsafe_policy,
        "preview_policy",
        PreviewModePolicy(
            mode=PreviewMode.READ_ONLY_MARKET,
            capabilities=(RuntimeCapability.LIVE_ORDER_SUBMIT,),
        ),
    )

    with pytest.raises(ReadOnlyMarketDataError):
        InMemoryReadOnlyMarketDataProvider(
            quotes={},
            candles={},
            policy=unsafe_policy,
        )


def test_in_memory_provider_happy_path_is_immutable_and_deterministic() -> None:
    provider = _provider()

    quote = provider.get_quote("btc/usdt")
    candles = provider.get_candles("BTC/USDT", "1m", 20)
    snapshot = provider.snapshot(("ETH/USDT", "BTC/USDT"))

    assert isinstance(provider, ReadOnlyMarketDataProvider)
    assert dataclasses.is_dataclass(quote)
    assert quote.last == 100.5
    with pytest.raises(dataclasses.FrozenInstanceError):
        quote.last = 1.0  # type: ignore[misc]
    assert isinstance(candles, tuple)
    assert candles == provider.get_candles("BTC/USDT", "1m", 20)
    assert snapshot.symbols == ("BTC/USDT", "ETH/USDT")
    assert tuple(item.symbol for item in snapshot.quotes) == snapshot.symbols
    mutated_return = candles + (MarketCandle("BTC/USDT", "1m", "later", 1.0, 1.0, 1.0, 1.0, 1.0),)
    assert len(mutated_return) == 3
    assert len(provider.get_candles("BTC/USDT", "1m", 20)) == 2


def test_contract_has_no_account_order_or_credentials_methods() -> None:
    provider = _provider()
    forbidden = {
        "get_balance",
        "get_account",
        "get_account_snapshot",
        "get_positions",
        "get_open_orders",
        "create_order",
        "submit_order",
        "cancel_order",
        "read_credentials",
    }
    public_methods = {
        name
        for name, value in inspect.getmembers(provider, predicate=callable)
        if not name.startswith("_")
    }

    assert forbidden.isdisjoint(public_methods)
    for name in forbidden:
        assert not hasattr(provider, name)


def test_invalid_inputs_fail_closed() -> None:
    provider = _provider()

    with pytest.raises(ReadOnlyMarketDataError):
        provider.get_quote("DOGE/USDT")
    with pytest.raises(ReadOnlyMarketDataError):
        provider.get_quote(" ")
    with pytest.raises(ReadOnlyMarketDataError):
        provider.get_candles("BTC/USDT", " ", 1)
    with pytest.raises(ReadOnlyMarketDataError):
        provider.get_candles("BTC/USDT", "1m", 0)
    assert len(provider.get_candles("BTC/USDT", "1m", 99)) == 2


def test_local_only_safety_no_network_env_file_write_cloud_export_or_runtime_loop(
    tmp_path: Path,
) -> None:
    provider = _provider()

    with patch.object(socket, "create_connection", side_effect=AssertionError("network used")):
        with patch.object(os, "getenv", side_effect=AssertionError("env read")):
            with patch.object(Path, "write_text", side_effect=AssertionError("file write")):
                assert provider.get_quote("BTC/USDT").symbol == "BTC/USDT"
                assert provider.get_candles("BTC/USDT", "1m", 1)[0].close == 101.0
                assert provider.snapshot(("BTC/USDT",)).symbols == ("BTC/USDT",)
    assert not hasattr(provider, "export")
    assert not hasattr(provider, "cloud_sink")
    assert not hasattr(provider, "start")
    assert not hasattr(provider, "run")
    assert not (tmp_path / "anything").exists()


def test_no_metadata_surface_for_secret_like_keys() -> None:
    quote = _provider().get_quote("BTC/USDT")
    candle = _provider().get_candles("BTC/USDT", "1m", 1)[0]

    for item in (quote, candle):
        assert not hasattr(item, "metadata")
        for key in (
            "api_key",
            "secret",
            "password",
            "passphrase",
            "credential",
            "credentials",
            "token",
            "private_key",
        ):
            assert not hasattr(item, key)
