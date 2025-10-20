from __future__ import annotations

from datetime import date
from typing import Any, Iterable, Sequence

import pytest

from bot_core.exchanges.base import (
    Environment,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)
from bot_core.runtime.bootstrap import _instantiate_adapter
from bot_core.security.capabilities import build_capabilities_from_payload
from bot_core.security.guards import (
    LicenseCapabilityError,
    install_capability_guard,
    reset_capability_guard,
)


class DummyAdapter(ExchangeAdapter):
    name = "dummy"

    def configure_network(self, *, ip_allowlist: Sequence[str] | None = None) -> None:  # noqa: D401 - prosty stub
        return None

    def fetch_account_snapshot(self) -> Any:
        return {
            "balances": {},
            "total_equity": 0.0,
            "available_margin": 0.0,
            "maintenance_margin": 0.0,
        }

    def fetch_symbols(self) -> Iterable[str]:
        return ()

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        limit: int | None = None,
    ) -> Sequence[Sequence[float]]:
        return ()

    def place_order(self, request: OrderRequest) -> OrderResult:
        raise RuntimeError("not implemented in dummy adapter")

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:
        return None

    def stream_public_data(self, *, channels: Sequence[str]) -> Any:
        return object()

    def stream_private_data(self, *, channels: Sequence[str]) -> Any:
        return object()


@pytest.fixture(autouse=True)
def _reset_guard() -> None:
    reset_capability_guard()
    yield
    reset_capability_guard()


def _install(payload: dict[str, Any]) -> None:
    capabilities = build_capabilities_from_payload(payload, effective_date=date(2025, 1, 1))
    install_capability_guard(capabilities)


def _factory(credentials: ExchangeCredentials, **kwargs: Any) -> ExchangeAdapter:
    return DummyAdapter(credentials)


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
    assert isinstance(adapter, DummyAdapter)


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
