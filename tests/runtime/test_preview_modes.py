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


@pytest.mark.parametrize(
    "capability",
    [
        RuntimeCapability.PAPER_ORDER_SUBMIT,
        "paper_order_lifecycle",
        RuntimeCapability.TESTNET_ORDER_SUBMIT,
        "sandbox_order_submit",
        RuntimeCapability.READ_ONLY_MARKET_FETCH,
        "recorded_fixture_replay",
        RuntimeCapability.LOCAL_TELEMETRY_AUDIT,
    ],
)
def test_validate_preview_capability_allows_preview_safe_inputs(
    capability: str | RuntimeCapability,
) -> None:
    from bot_core.runtime.preview_modes import validate_preview_capability

    normalized = validate_preview_capability(capability)

    assert isinstance(normalized, RuntimeCapability)
    assert is_capability_allowed_in_preview(normalized) is True


@pytest.mark.parametrize(
    "capability",
    [
        RuntimeCapability.LIVE_ORDER_SUBMIT,
        "real_exchange_fill",
        RuntimeCapability.LIVE_BALANCE_MUTATION,
        "live_account_balance_fetch",
        RuntimeCapability.LIVE_ACCOUNT_SNAPSHOT_READ,
        "live_credentials_read",
        RuntimeCapability.PRODUCTION_CLOUD_SINK,
        "external_export_sink",
        RuntimeCapability.LIVE_SCHEDULER_WORKER_SIDE_EFFECT,
    ],
)
def test_validate_preview_capability_blocks_live_production_side_effects(
    capability: str | RuntimeCapability,
) -> None:
    from bot_core.runtime.preview_modes import (
        PreviewModeContractError,
        validate_preview_capability,
    )

    with pytest.raises(PreviewModeContractError):
        validate_preview_capability(capability)


@pytest.mark.parametrize("capability", ["unknown_capability", "paper+live"])
def test_validate_preview_capability_fails_closed_for_unknown_or_ambiguous(
    capability: str,
) -> None:
    from bot_core.runtime.preview_modes import (
        PreviewModeContractError,
        validate_preview_capability,
    )

    with pytest.raises(PreviewModeContractError):
        validate_preview_capability(capability)


@pytest.mark.parametrize("mode", [PreviewMode.PAPER, "testnet", " Read-Only Market "])
def test_validate_preview_mode_accepts_enum_and_string(mode: str | PreviewMode) -> None:
    from bot_core.runtime.preview_modes import validate_preview_mode

    assert isinstance(validate_preview_mode(mode), PreviewMode)


@pytest.mark.parametrize("mode", ["live", "paper+live", ""])
def test_validate_preview_mode_fails_closed_for_unknown_or_ambiguous(mode: str) -> None:
    from bot_core.runtime.preview_modes import PreviewModeContractError, validate_preview_mode

    with pytest.raises(PreviewModeContractError):
        validate_preview_mode(mode)


def test_batch_validation_fails_if_any_capability_is_live_production() -> None:
    from bot_core.runtime.preview_modes import (
        PreviewModeContractError,
        validate_preview_capabilities,
    )

    with pytest.raises(PreviewModeContractError):
        validate_preview_capabilities(
            [RuntimeCapability.PAPER_ORDER_SUBMIT, RuntimeCapability.LIVE_ORDER_SUBMIT]
        )


def test_assert_preview_capabilities_allowed_accepts_safe_batch() -> None:
    from bot_core.runtime.preview_modes import assert_preview_capabilities_allowed

    assert_preview_capabilities_allowed(
        [
            "paper_order_submit",
            RuntimeCapability.TESTNET_ORDER_SUBMIT,
            "read_only_market_fetch",
            RuntimeCapability.RECORDED_FIXTURE_REPLAY,
        ]
    )


def test_preview_policy_with_paper_testnet_read_only_recorded_is_valid() -> None:
    from bot_core.runtime.preview_modes import build_preview_mode_policy

    policy = build_preview_mode_policy(
        "paper",
        [
            "paper_order_submit",
            RuntimeCapability.TESTNET_ORDER_SUBMIT,
            "read_only_market_fetch",
            RuntimeCapability.RECORDED_FIXTURE_REPLAY,
        ],
    )

    assert policy.mode is PreviewMode.PAPER
    assert policy.capabilities == (
        RuntimeCapability.PAPER_ORDER_SUBMIT,
        RuntimeCapability.TESTNET_ORDER_SUBMIT,
        RuntimeCapability.READ_ONLY_MARKET_FETCH,
        RuntimeCapability.RECORDED_FIXTURE_REPLAY,
    )


def test_preview_policy_with_paper_and_live_order_submit_is_invalid() -> None:
    from bot_core.runtime.preview_modes import (
        PreviewModeContractError,
        build_preview_mode_policy,
    )

    with pytest.raises(PreviewModeContractError):
        build_preview_mode_policy(
            PreviewMode.PAPER,
            [RuntimeCapability.PAPER_ORDER_SUBMIT, RuntimeCapability.LIVE_ORDER_SUBMIT],
        )


def test_preview_policy_with_read_only_market_and_live_account_fetch_is_invalid() -> None:
    from bot_core.runtime.preview_modes import (
        PreviewModeContractError,
        build_preview_mode_policy,
    )

    with pytest.raises(PreviewModeContractError):
        build_preview_mode_policy(
            "read_only_market",
            [
                RuntimeCapability.READ_ONLY_MARKET_FETCH,
                RuntimeCapability.LIVE_ACCOUNT_BALANCE_FETCH,
            ],
        )
