from __future__ import annotations

from datetime import date

import pytest

from bot_core.security.capabilities import build_capabilities_from_payload
from bot_core.security.guards import (
    CapabilityGuard,
    LicenseCapabilityError,
    get_capability_guard,
    install_capability_guard,
    reset_capability_guard,
)


def _sample_capabilities():
    payload = {
        "edition": "pro",
        "environments": ["demo", "paper", "live"],
        "exchanges": {"binance_spot": True, "kraken_futures": False},
        "strategies": {"trend_d1": True},
        "runtime": {"multi_strategy_scheduler": True},
        "modules": {"futures": True, "ai_signals": False},
        "limits": {"max_paper_controllers": 2, "max_concurrent_bots": 2},
    }
    return build_capabilities_from_payload(payload, effective_date=date(2025, 7, 1))


def test_guard_module_and_exchange_checks() -> None:
    guard = CapabilityGuard(_sample_capabilities())
    guard.require_module("futures")
    guard.require_exchange("binance_spot")

    with pytest.raises(LicenseCapabilityError):
        guard.require_module("ai_signals")
    with pytest.raises(LicenseCapabilityError):
        guard.require_exchange("kraken_futures")


def test_guard_limits_enforced() -> None:
    guard = CapabilityGuard(_sample_capabilities())
    guard.reserve_slot("paper_controller")
    guard.reserve_slot("paper_controller")
    with pytest.raises(LicenseCapabilityError):
        guard.reserve_slot("paper_controller")

    guard.reserve_slot("bot")
    guard.reserve_slot("bot")
    with pytest.raises(LicenseCapabilityError):
        guard.reserve_slot("bot")


def test_guard_release_slot() -> None:
    guard = CapabilityGuard(_sample_capabilities())
    guard.reserve_slot("paper_controller")
    guard.release_slot("paper_controller")
    guard.reserve_slot("paper_controller")
    guard.reserve_slot("paper_controller")
    with pytest.raises(LicenseCapabilityError):
        guard.reserve_slot("paper_controller")


def test_global_guard_installation() -> None:
    reset_capability_guard()
    assert get_capability_guard() is None
    capabilities = _sample_capabilities()
    guard = install_capability_guard(capabilities)
    assert get_capability_guard() is guard
    reset_capability_guard()
    assert get_capability_guard() is None
