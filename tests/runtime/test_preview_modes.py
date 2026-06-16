"""Preview/live mode contract tests."""

from __future__ import annotations

import pytest

from bot_core.runtime.preview_modes import (
    PreviewMode,
    RuntimeCapability,
    is_capability_allowed_in_preview,
    is_live_production_side_effect,
    is_preview_mode_allowed,
    normalize_capability,
    normalize_preview_mode,
)


@pytest.mark.parametrize(
    "mode",
    [
        "local_mock",
        "recorded_replay",
        "recorded_fixture",
        "paper",
        "testnet",
        "sandbox",
        "read_only_market",
    ],
)
def test_preview_is_not_mock_only(mode: str) -> None:
    assert is_preview_mode_allowed(mode) is True


@pytest.mark.parametrize(
    "capability",
    [
        RuntimeCapability.PAPER_ORDER_SUBMIT,
        RuntimeCapability.PAPER_ORDER_LIFECYCLE,
        RuntimeCapability.TESTNET_ORDER_SUBMIT,
        RuntimeCapability.SANDBOX_ORDER_SUBMIT,
        RuntimeCapability.READ_ONLY_MARKET_FETCH,
        RuntimeCapability.RECORDED_FIXTURE_REPLAY,
        RuntimeCapability.LOCAL_MOCK_RUNTIME,
        RuntimeCapability.LOCAL_TELEMETRY_AUDIT,
    ],
)
def test_preview_allows_non_live_production_capabilities(
    capability: RuntimeCapability,
) -> None:
    assert is_capability_allowed_in_preview(capability) is True
    assert is_live_production_side_effect(capability) is False


@pytest.mark.parametrize(
    "capability",
    [
        RuntimeCapability.LIVE_ORDER_SUBMIT,
        RuntimeCapability.REAL_EXCHANGE_FILL,
        RuntimeCapability.LIVE_BALANCE_MUTATION,
        RuntimeCapability.LIVE_ACCOUNT_BALANCE_FETCH,
        RuntimeCapability.LIVE_ACCOUNT_SNAPSHOT_READ,
        RuntimeCapability.LIVE_CREDENTIALS_READ,
        RuntimeCapability.PRODUCTION_CLOUD_SINK,
        RuntimeCapability.EXTERNAL_EXPORT_SINK,
        RuntimeCapability.LIVE_SCHEDULER_WORKER_SIDE_EFFECT,
    ],
)
def test_preview_blocks_live_production_side_effects(
    capability: RuntimeCapability,
) -> None:
    assert is_live_production_side_effect(capability) is True
    assert is_capability_allowed_in_preview(capability) is False


def test_read_only_market_is_not_live_account_access() -> None:
    assert is_capability_allowed_in_preview(RuntimeCapability.READ_ONLY_MARKET_FETCH) is True
    assert is_live_production_side_effect(RuntimeCapability.READ_ONLY_MARKET_FETCH) is False

    for capability in (
        RuntimeCapability.LIVE_ACCOUNT_BALANCE_FETCH,
        RuntimeCapability.LIVE_ACCOUNT_SNAPSHOT_READ,
    ):
        assert is_capability_allowed_in_preview(capability) is False
        assert is_live_production_side_effect(capability) is True


def test_unknown_and_ambiguous_names_fail_closed() -> None:
    assert normalize_preview_mode("live") is None
    assert normalize_preview_mode("production_live") is None
    assert normalize_preview_mode("real_orders") is None
    assert normalize_preview_mode("paper+live") is None
    assert is_preview_mode_allowed("paper+live") is False
    assert normalize_capability("unknown_order_submit") is None
    assert is_capability_allowed_in_preview("unknown_order_submit") is False
    assert is_live_production_side_effect("unknown_order_submit") is False


def test_string_normalization_is_deterministic() -> None:
    assert normalize_preview_mode(" Read-Only Market ") is PreviewMode.READ_ONLY_MARKET
    assert normalize_preview_mode("paper trading") is PreviewMode.PAPER
    assert normalize_capability(" TESTNET-ORDER-SUBMIT ") is RuntimeCapability.TESTNET_ORDER_SUBMIT
