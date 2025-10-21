from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from bot_core.exchanges.base import Environment, ExchangeAdapter, ExchangeCredentials
from bot_core.runtime.bootstrap import _instantiate_adapter
from bot_core.security.capabilities import build_capabilities_from_payload
from bot_core.security.guards import (
    LicenseCapabilityError,
    install_capability_guard,
    reset_capability_guard,
)

from tests._exchange_adapter_helpers import StubExchangeAdapter


@pytest.fixture(autouse=True)
def _reset_guard() -> None:
    reset_capability_guard()
    yield
    reset_capability_guard()


def _install(payload: dict[str, Any]) -> None:
    capabilities = build_capabilities_from_payload(payload, effective_date=date(2025, 1, 1))
    install_capability_guard(capabilities)


def _factory(credentials: ExchangeCredentials, **kwargs: Any) -> ExchangeAdapter:
    return StubExchangeAdapter(credentials)


def test_instantiate_adapter_denies_missing_exchange() -> None:
    payload = {
        "edition": "pro",
        "environments": ["paper", "demo"],
        "exchanges": {"binance_spot": False},
        "modules": {},
    }
    _install(payload)
    creds = ExchangeCredentials(key_id="spot", environment=Environment.PAPER)
    factories = {"binance_spot": _factory}

    with pytest.raises(LicenseCapabilityError):
        _instantiate_adapter(
            "binance_spot",
            creds,
            factories,
            Environment.PAPER,
        )


def test_instantiate_adapter_requires_futures_module() -> None:
    payload = {
        "edition": "pro",
        "environments": ["paper", "demo"],
        "exchanges": {"binance_futures": True},
        "modules": {"futures": False},
    }
    _install(payload)
    creds = ExchangeCredentials(key_id="futures", environment=Environment.PAPER)
    factories = {"binance_futures": _factory}

    with pytest.raises(LicenseCapabilityError) as exc:
        _instantiate_adapter(
            "binance_futures",
            creds,
            factories,
            Environment.PAPER,
        )
    assert exc.value.capability == "futures"


def test_instantiate_adapter_allows_paper_exchange() -> None:
    payload = {
        "edition": "pro",
        "environments": ["paper", "demo"],
        "exchanges": {"binance_spot": True},
        "modules": {},
    }
    _install(payload)
    creds = ExchangeCredentials(key_id="spot", environment=Environment.PAPER)
    factories = {"binance_spot": _factory}

    adapter = _instantiate_adapter(
        "binance_spot",
        creds,
        factories,
        Environment.PAPER,
    )
    assert isinstance(adapter, StubExchangeAdapter)


def test_instantiate_adapter_requires_pro_for_live() -> None:
    payload = {
        "edition": "standard",
        "environments": ["live"],
        "exchanges": {"binance_spot": True},
        "modules": {},
    }
    _install(payload)
    creds = ExchangeCredentials(key_id="spot", environment=Environment.LIVE)
    factories = {"binance_spot": _factory}

    with pytest.raises(LicenseCapabilityError) as exc:
        _instantiate_adapter(
            "binance_spot",
            creds,
            factories,
            Environment.LIVE,
        )
    assert exc.value.capability == "edition"
