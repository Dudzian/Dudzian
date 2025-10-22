from __future__ import annotations

import pytest

from KryptoLowca.exchanges import (
    AdapterError,
    BitstampAdapter,
    BybitSpotAdapter,
    OKXDerivativesAdapter,
    OKXMarginAdapter,
    ZondaAdapter,
    create_exchange_adapter,
)


class _DummyHTTPClient:
    async def request(self, *args, **kwargs):  # pragma: no cover - nie powinno być użyte
        raise AssertionError("_DummyHTTPClient nie powinien wykonywać zapytań")


@pytest.mark.parametrize(
    "name, expected_type",
    [
        ("bitstamp", BitstampAdapter),
        ("bybit", BybitSpotAdapter),
        ("bybit-spot", BybitSpotAdapter),
        ("okx-margin", OKXMarginAdapter),
        ("okx-derivatives", OKXDerivativesAdapter),
        ("zonda", ZondaAdapter),
        ("bitbay", ZondaAdapter),
    ],
)
def test_create_exchange_adapter_returns_registered_impl(name: str, expected_type: type) -> None:
    adapter = create_exchange_adapter(name, http_client=_DummyHTTPClient())
    assert isinstance(adapter, expected_type)


def test_factory_forwards_adapter_specific_kwargs() -> None:
    adapter = create_exchange_adapter(
        "okx-derivatives", http_client=_DummyHTTPClient(), inst_type="FUTURES"
    )
    assert isinstance(adapter, OKXDerivativesAdapter)
    assert getattr(adapter, "_inst_type") == "FUTURES"


def test_factory_rejects_non_mapping_adapter_kwargs() -> None:
    with pytest.raises(AdapterError):
        create_exchange_adapter(
            "bitstamp", http_client=_DummyHTTPClient(), adapter_kwargs="invalid"
        )
