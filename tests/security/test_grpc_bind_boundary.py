"""Plaintext gRPC listeners stay local; TLS retains its separate transport path."""

from types import SimpleNamespace

import grpc
import pytest

from bot_core.api import server as api_server
from bot_core.runtime import _grpc_support, market_data_service, metrics_service, risk_service


KINDS = ("local", "market-data", "metrics", "risk")
NONLOCAL_OR_AMBIGUOUS = (
    "0.0.0.0",
    "::",
    "[::]",
    "192.0.2.1",
    "2001:db8::1",
    "example.invalid",
    "localhost.example.invalid",
    "localhost:50051",
    "127.0.0.1:50051",
    "dns:///127.0.0.1",
    "ipv4:127.0.0.1,192.0.2.1",
    "[::1]:50051",
    "::1%eth0",
    "[::1%lo]",
    "::ffff:127.0.0.1",
    "127.1",
    "127.000.0.1",
    "2130706433",
    "0x7f000001",
    "localhost.",
    "127.0.0.1,192.0.2.1",
    "",
    " ",
    None,
    False,
    1,
    ["127.0.0.1"],
    {"host": "127.0.0.1"},
)


class RecordingServer:
    def __init__(self) -> None:
        self.binds: list[tuple[str, str, object]] = []

    def add_generic_rpc_handlers(self, handlers: object) -> None:
        pass

    def add_registered_method_handlers(self, *args: object) -> None:
        pass

    def add_insecure_port(self, address: str) -> int:
        self.binds.append(("plaintext", address, None))
        return 50051

    def add_secure_port(self, address: str, credentials: object) -> int:
        self.binds.append(("tls", address, credentials))
        return 50051


def build_server(kind: str, **kwargs: object):
    if kind == "local":
        context = SimpleNamespace(
            risk_store=None, marketplace_repository=None, marketplace_enabled=False
        )
        return api_server.LocalRuntimeServer(context, **kwargs)
    constructors = {
        "market-data": market_data_service.MarketDataServer,
        "metrics": metrics_service.MetricsServer,
        "risk": risk_service.RiskServer,
    }
    return constructors[kind](**kwargs)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("host", NONLOCAL_OR_AMBIGUOUS)
def test_plaintext_exposed_or_ambiguous_host_refused_before_server_allocation(
    monkeypatch: pytest.MonkeyPatch, kind: str, host: object
) -> None:
    allocations = []

    def server_factory(*args: object, **kwargs: object) -> RecordingServer:
        server = RecordingServer()
        allocations.append(server)
        return server

    monkeypatch.setattr(grpc, "server", server_factory)
    with pytest.raises(ValueError, match="loopback"):
        build_server(kind, host=host, port=0)
    assert allocations == []


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "[::1]", "localhost"])
def test_plaintext_loopback_binds_real_ephemeral_listener(kind: str, host: str) -> None:
    server = build_server(kind, host=host, port=0)
    try:
        server.start()
        assert int(server.address.rsplit(":", 1)[1]) > 0
        assert server.address.startswith(("127.0.0.1:", "[::1]:"))
    finally:
        server.stop(0)


@pytest.mark.parametrize("kind", ["market-data", "metrics", "risk"])
@pytest.mark.parametrize("host", ["0.0.0.0", "192.0.2.1", "tls-host.example.invalid"])
def test_tls_remote_host_uses_only_secure_binding(
    monkeypatch: pytest.MonkeyPatch, kind: str, host: str
) -> None:
    recording = RecordingServer()
    credentials = object()
    monkeypatch.setattr(grpc, "server", lambda *args, **kwargs: recording)
    server = build_server(kind, host=host, port=50051, server_credentials=credentials)
    assert recording.binds == [("tls", f"{host}:50051", credentials)]
    assert server.address == f"{host}:50051"


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", "127.0.0.1"),
        ("127.0.0.2", "127.0.0.2"),
        ("localhost", "127.0.0.1"),
        (" LOCALHOST ", "127.0.0.1"),
        ("::1", "[::1]"),
        ("[::1]", "[::1]"),
        ("0:0:0:0:0:0:0:1", "[::1]"),
    ],
)
def test_plaintext_guard_normalizes_only_explicit_loopback(host: str, expected: str) -> None:
    assert _grpc_support.validate_plaintext_grpc_host(host) == expected


@pytest.mark.parametrize("host", NONLOCAL_OR_AMBIGUOUS)
def test_plaintext_guard_refuses_nonlocal_or_ambiguous_input(host: object) -> None:
    with pytest.raises(ValueError, match="loopback"):
        _grpc_support.validate_plaintext_grpc_host(host)
