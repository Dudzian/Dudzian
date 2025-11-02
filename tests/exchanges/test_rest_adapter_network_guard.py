import base64
import pytest

from bot_core.exchanges.base import Environment, ExchangeCredentials
from bot_core.exchanges.errors import ExchangeNetworkError
from bot_core.exchanges.kraken.futures import KrakenFuturesAdapter
from bot_core.exchanges.network_guard import NetworkAccessViolation
from bot_core.exchanges.nowa_gielda.spot import NowaGieldaSpotAdapter
from bot_core.exchanges.zonda.spot import ZondaSpotAdapter


def test_zonda_public_request_requires_network_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = ExchangeCredentials(key_id="zonda-public", environment=Environment.LIVE)
    adapter = ZondaSpotAdapter(credentials)

    def _fail(*_args, **_kwargs):  # pragma: no cover - zabezpieczenie przed realnym ruchem
        raise AssertionError("ZondaSpotAdapter nie powinien wysyłać żądania HTTP bez configure_network")

    monkeypatch.setattr("bot_core.exchanges.zonda.spot.urlopen", _fail)

    with pytest.raises(ExchangeNetworkError) as excinfo:
        adapter._public_request("/test")

    violation = excinfo.value.reason
    assert isinstance(violation, NetworkAccessViolation)
    assert violation.reason == "network_not_configured"


def test_zonda_signed_request_propagates_guard_violation(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = ExchangeCredentials(
        key_id="zonda-private",
        secret="secret",
        permissions=("trade",),
        environment=Environment.LIVE,
    )
    adapter = ZondaSpotAdapter(credentials)
    adapter.configure_network(ip_allowlist=("10.0.0.1",))

    violation = NetworkAccessViolation(
        reason="source_ip_not_allowed",
        details={"source_ip": "192.0.2.1"},
    )

    def _raise_violation(self, url: str) -> None:  # pragma: no cover - testowa ścieżka
        raise violation

    monkeypatch.setattr(type(adapter._network_guard), "ensure_allowed", _raise_violation)

    with pytest.raises(ExchangeNetworkError) as excinfo:
        adapter._signed_request("GET", "/private/test")

    assert excinfo.value.reason is violation


def test_kraken_public_request_requires_network_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = ExchangeCredentials(
        key_id="kraken-public",
        secret=base64.b64encode(b"secret").decode("utf-8"),
        permissions=("read",),
        environment=Environment.LIVE,
    )
    adapter = KrakenFuturesAdapter(credentials, environment=Environment.LIVE)

    def _fail(*_args, **_kwargs):  # pragma: no cover - zabezpieczenie
        raise AssertionError("KrakenFuturesAdapter nie powinien wysyłać zapytania bez configure_network")

    monkeypatch.setattr("bot_core.exchanges.kraken.futures.urlopen", _fail)

    with pytest.raises(ExchangeNetworkError) as excinfo:
        adapter._public_request("/time", {})

    violation = excinfo.value.reason
    assert isinstance(violation, NetworkAccessViolation)
    assert violation.reason == "network_not_configured"


def test_nowa_gielda_requests_require_network_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credentials = ExchangeCredentials(key_id="nowa-public", environment=Environment.LIVE)
    adapter = NowaGieldaSpotAdapter(credentials)

    def _fail(*_args, **_kwargs):  # pragma: no cover - zabezpieczenie przed ruchem HTTP
        raise AssertionError("NowaGieldaSpotAdapter nie powinien wykonywać requestu bez configure_network")

    monkeypatch.setattr("bot_core.exchanges.nowa_gielda.spot.run_sync", _fail)

    with pytest.raises(ExchangeNetworkError) as excinfo:
        adapter.fetch_ticker("BTC_USDT")

    violation = excinfo.value.reason
    assert isinstance(violation, NetworkAccessViolation)
    assert violation.reason == "network_not_configured"


def test_nowa_gielda_guard_violation_is_propagated(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = ExchangeCredentials(
        key_id="nowa-private",
        secret="top-secret",
        permissions=("trade",),
        environment=Environment.LIVE,
    )
    adapter = NowaGieldaSpotAdapter(credentials)
    adapter.configure_network(ip_allowlist=("10.0.0.1",))

    violation = NetworkAccessViolation(
        reason="source_ip_not_allowed",
        details={"source_ip": "198.51.100.5"},
    )

    def _raise_violation(self, url: str) -> None:  # pragma: no cover - ścieżka testowa
        raise violation

    monkeypatch.setattr(
        type(adapter._network_guard),
        "ensure_allowed",
        _raise_violation,
    )

    def _fail(*_args, **_kwargs):  # pragma: no cover - zabezpieczenie przed realnym requestem
        raise AssertionError("run_sync nie powinno zostać wywołane przy blokadzie guardem")

    monkeypatch.setattr("bot_core.exchanges.nowa_gielda.spot.run_sync", _fail)

    with pytest.raises(ExchangeNetworkError) as excinfo:
        adapter.fetch_orderbook("BTC_USDT")

    assert excinfo.value.reason is violation
