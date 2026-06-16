"""Static preview/live mode contract.

Preview is not mock-only.  This module only classifies mode/capability names and
never performs I/O, starts runtime loops, reads secrets, or talks to exchanges.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Iterable


class PreviewModeContractError(ValueError):
    """Raised when preview mode/capability policy fails closed."""


@dataclass(frozen=True)
class PreviewModePolicy:
    """Normalized preview policy with capabilities verified as preview-safe."""

    mode: "PreviewMode"
    capabilities: tuple["RuntimeCapability", ...]


class PreviewMode(StrEnum):
    """Modes that are explicitly allowed inside preview."""

    LOCAL_MOCK = "local_mock"
    RECORDED_REPLAY = "recorded_replay"
    PAPER = "paper"
    TESTNET = "testnet"
    SANDBOX = "sandbox"
    READ_ONLY_MARKET = "read_only_market"


class RuntimeCapability(StrEnum):
    """Capabilities classified by the preview/live side-effect boundary."""

    LOCAL_MOCK_RUNTIME = "local_mock_runtime"
    RECORDED_FIXTURE_REPLAY = "recorded_fixture_replay"
    PAPER_ORDER_SUBMIT = "paper_order_submit"
    PAPER_ORDER_LIFECYCLE = "paper_order_lifecycle"
    TESTNET_ORDER_SUBMIT = "testnet_order_submit"
    SANDBOX_ORDER_SUBMIT = "sandbox_order_submit"
    READ_ONLY_MARKET_FETCH = "read_only_market_fetch"
    LOCAL_TELEMETRY_AUDIT = "local_telemetry_audit"
    LIVE_ORDER_SUBMIT = "live_order_submit"
    REAL_EXCHANGE_FILL = "real_exchange_fill"
    LIVE_BALANCE_MUTATION = "live_balance_mutation"
    LIVE_ACCOUNT_BALANCE_FETCH = "live_account_balance_fetch"
    LIVE_ACCOUNT_SNAPSHOT_READ = "live_account_snapshot_read"
    LIVE_CREDENTIALS_READ = "live_credentials_read"
    PRODUCTION_CLOUD_SINK = "production_cloud_sink"
    EXTERNAL_EXPORT_SINK = "external_export_sink"
    LIVE_SCHEDULER_WORKER_SIDE_EFFECT = "live_scheduler_worker_side_effect"


_NORMALIZED_MODE_ALIASES: Final[dict[str, PreviewMode]] = {
    "local_mock": PreviewMode.LOCAL_MOCK,
    "mock": PreviewMode.LOCAL_MOCK,
    "demo": PreviewMode.LOCAL_MOCK,
    "recorded_fixture": PreviewMode.RECORDED_REPLAY,
    "recorded_replay": PreviewMode.RECORDED_REPLAY,
    "replay": PreviewMode.RECORDED_REPLAY,
    "paper": PreviewMode.PAPER,
    "paper_trading": PreviewMode.PAPER,
    "testnet": PreviewMode.TESTNET,
    "sandbox": PreviewMode.SANDBOX,
    "read_only_market": PreviewMode.READ_ONLY_MARKET,
    "readonly_market": PreviewMode.READ_ONLY_MARKET,
    "read_only": PreviewMode.READ_ONLY_MARKET,
}

_PREVIEW_ALLOWED_CAPABILITIES: Final[frozenset[RuntimeCapability]] = frozenset(
    {
        RuntimeCapability.LOCAL_MOCK_RUNTIME,
        RuntimeCapability.RECORDED_FIXTURE_REPLAY,
        RuntimeCapability.PAPER_ORDER_SUBMIT,
        RuntimeCapability.PAPER_ORDER_LIFECYCLE,
        RuntimeCapability.TESTNET_ORDER_SUBMIT,
        RuntimeCapability.SANDBOX_ORDER_SUBMIT,
        RuntimeCapability.READ_ONLY_MARKET_FETCH,
        RuntimeCapability.LOCAL_TELEMETRY_AUDIT,
    }
)

_LIVE_PRODUCTION_SIDE_EFFECTS: Final[frozenset[RuntimeCapability]] = frozenset(
    {
        RuntimeCapability.LIVE_ORDER_SUBMIT,
        RuntimeCapability.REAL_EXCHANGE_FILL,
        RuntimeCapability.LIVE_BALANCE_MUTATION,
        RuntimeCapability.LIVE_ACCOUNT_BALANCE_FETCH,
        RuntimeCapability.LIVE_ACCOUNT_SNAPSHOT_READ,
        RuntimeCapability.LIVE_CREDENTIALS_READ,
        RuntimeCapability.PRODUCTION_CLOUD_SINK,
        RuntimeCapability.EXTERNAL_EXPORT_SINK,
        RuntimeCapability.LIVE_SCHEDULER_WORKER_SIDE_EFFECT,
    }
)


def _normalize_token(value: str) -> str | None:
    """Normalize one config token, rejecting combined/ambiguous declarations."""

    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return None
    if any(separator in normalized for separator in ("+", ",", ";", "|", "/")):
        return None
    return normalized


def normalize_preview_mode(value: str | PreviewMode) -> PreviewMode | None:
    """Normalize a mode string; unknown/ambiguous names fail closed as ``None``."""

    if isinstance(value, PreviewMode):
        return value
    normalized = _normalize_token(value)
    if normalized is None:
        return None
    return _NORMALIZED_MODE_ALIASES.get(normalized)


def normalize_capability(value: str | RuntimeCapability) -> RuntimeCapability | None:
    """Normalize a capability string; unknown names fail closed as ``None``."""

    if isinstance(value, RuntimeCapability):
        return value
    normalized = _normalize_token(value)
    if normalized is None:
        return None
    try:
        return RuntimeCapability(normalized)
    except ValueError:
        return None


def is_preview_mode_allowed(mode: str | PreviewMode) -> bool:
    """Return whether ``mode`` is an explicitly allowed preview mode."""

    return normalize_preview_mode(mode) is not None


def is_live_production_side_effect(capability: str | RuntimeCapability) -> bool:
    """Return whether ``capability`` is reserved for live production only."""

    normalized = normalize_capability(capability)
    return normalized in _LIVE_PRODUCTION_SIDE_EFFECTS


def is_capability_allowed_in_preview(capability: str | RuntimeCapability) -> bool:
    """Return whether ``capability`` is allowed in preview.

    Unknown capabilities and live-production side effects fail closed.
    """

    normalized = normalize_capability(capability)
    return normalized in _PREVIEW_ALLOWED_CAPABILITIES


def validate_preview_mode(mode: str | PreviewMode) -> PreviewMode:
    """Return normalized preview mode or raise when config is unknown/ambiguous."""

    normalized = normalize_preview_mode(mode)
    if normalized is None:
        raise PreviewModeContractError(f"Preview mode is not allowed: {mode!r}")
    return normalized


def validate_preview_capability(
    capability: str | RuntimeCapability,
) -> RuntimeCapability:
    """Return normalized capability or raise when it is not preview-safe."""

    normalized = normalize_capability(capability)
    if normalized is None:
        raise PreviewModeContractError(
            f"Preview capability is unknown or ambiguous: {capability!r}"
        )
    if normalized in _LIVE_PRODUCTION_SIDE_EFFECTS:
        raise PreviewModeContractError(
            f"Preview capability is a live-production side effect: {normalized.value}"
        )
    if normalized not in _PREVIEW_ALLOWED_CAPABILITIES:
        raise PreviewModeContractError(f"Preview capability is not allowed: {normalized.value}")
    return normalized


def validate_preview_capabilities(
    capabilities: Iterable[str | RuntimeCapability],
) -> tuple[RuntimeCapability, ...]:
    """Validate all capabilities, failing the batch if any item is unsafe."""

    return tuple(validate_preview_capability(capability) for capability in capabilities)


def assert_preview_capabilities_allowed(
    capabilities: Iterable[str | RuntimeCapability],
) -> None:
    """Raise unless every capability is allowed in preview."""

    validate_preview_capabilities(capabilities)


def build_preview_mode_policy(
    mode: str | PreviewMode,
    capabilities: Iterable[str | RuntimeCapability],
) -> PreviewModePolicy:
    """Build a normalized policy after enforcing preview mode/capability gates."""

    return PreviewModePolicy(
        mode=validate_preview_mode(mode),
        capabilities=validate_preview_capabilities(capabilities),
    )


__all__ = [
    "PreviewMode",
    "PreviewModeContractError",
    "PreviewModePolicy",
    "RuntimeCapability",
    "assert_preview_capabilities_allowed",
    "build_preview_mode_policy",
    "is_capability_allowed_in_preview",
    "is_live_production_side_effect",
    "is_preview_mode_allowed",
    "normalize_capability",
    "normalize_preview_mode",
    "validate_preview_capabilities",
    "validate_preview_capability",
    "validate_preview_mode",
]
