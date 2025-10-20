import pytest

from bot_core.security.hwid import HwIdProvider, HwIdProviderError


def test_hwid_provider_strips_whitespace() -> None:
    provider = HwIdProvider(fingerprint_reader=lambda: "  ABC  ")
    assert provider.read() == "ABC"


def test_hwid_provider_raises_on_empty() -> None:
    provider = HwIdProvider(fingerprint_reader=lambda: "   ")
    with pytest.raises(HwIdProviderError):
        provider.read()


def test_hwid_provider_wraps_exceptions() -> None:
    def boom() -> str:
        raise RuntimeError("boom")

    provider = HwIdProvider(fingerprint_reader=boom)
    with pytest.raises(HwIdProviderError):
        provider.read()
