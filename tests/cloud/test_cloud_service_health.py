from pathlib import Path
from types import SimpleNamespace

import grpc
import pytest

from bot_core.cloud.config import CloudMarketplaceConfig, CloudRuntimeConfig, CloudSecurityConfig, CloudServerConfig
from bot_core.cloud.security import CloudAuthServicer, CloudAuthorizationError
from bot_core.cloud.service import CloudRuntimeService
from bot_core.generated import trading_pb2


class _FakeContext:
    def __init__(self) -> None:
        self.started = False
        self.retrain_scheduler = type("Sched", (), {"maybe_run": lambda self: None})()
        self.marketplace_repository = object()

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def reload_marketplace_presets(self) -> None:
        pass


class _FakeServer:
    def __init__(self, context, host: str, port: int, max_workers: int, interceptors=None) -> None:
        self._context = context
        self.address = f"{host}:{port or 55052}"
        self.grpc_server = object()

    def start(self) -> None:
        return None

    def stop(self, _timeout: float) -> None:
        return None

    def wait(self) -> None:
        return None


@pytest.fixture()
def _fake_config(tmp_path: Path) -> CloudServerConfig:
    runtime_cfg = CloudRuntimeConfig(config_path=tmp_path / "runtime.yaml", entrypoint="demo")
    marketplace_cfg = CloudMarketplaceConfig(refresh_interval_seconds=1, auto_reload=True)
    security_cfg = CloudSecurityConfig(require_handshake=False)
    return CloudServerConfig(
        host="127.0.0.1",
        port=0,
        runtime=runtime_cfg,
        marketplace=marketplace_cfg,
        security=security_cfg,
    )


def test_cloud_service_writes_health_and_ready(tmp_path: Path, _fake_config, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[dict[str, object]] = []

    def _builder(config_path, entrypoint):
        return _FakeContext()

    monkeypatch.setattr("bot_core.cloud.service.LocalRuntimeServer", _FakeServer)

    ready_file = tmp_path / "ready.json"
    service = CloudRuntimeService(
        _fake_config,
        context_builder=_builder,
        ready_hook=lambda payload: events.append(dict(payload)),
        health_probe_path=tmp_path / "health.json",
    )
    service.start()
    assert events and events[0]["event"] == "ready"
    health_path = tmp_path / "health.json"
    assert health_path.exists()
    snapshot = service.health_snapshot
    assert snapshot["status"] == "ready"
    service.stop()
    assert service.health_snapshot["status"] == "stopped"


def test_market_data_servicer_propagates_cloud_metadata() -> None:
    class _Ctx:
        cloud_health_headers = {"x-bot-cloud-health": "0", "x-bot-cloud-last-error": "boom"}
        primary_symbol = "BTCUSDT"
        exchange_name = "PAPER"

        class pipeline:
            controller = type("Ctrl", (), {"interval": "1h"})()
            execution_service = type("Svc", (), {"_markets": {}})()

    class _RpcContext:
        def __init__(self) -> None:
            self.initial_metadata: list[tuple[str, str]] | None = None
            self.trailing_metadata: list[tuple[str, str]] | None = None

        def send_initial_metadata(self, metadata):
            self.initial_metadata = list(metadata)

        def set_trailing_metadata(self, metadata):
            self.trailing_metadata = list(metadata)

    class _Request:
        exchange = ""

    from bot_core.api.server import _MarketDataServicer  # lokalny import, aby zachować zależności testu

    servicer = _MarketDataServicer(_Ctx())
    rpc_context = _RpcContext()

    response = servicer.ListTradableInstruments(_Request(), rpc_context)

    assert len(response.instruments) == 0
    assert rpc_context.initial_metadata == rpc_context.trailing_metadata
    assert rpc_context.initial_metadata == [
        ("x-bot-cloud-health", "0"),
        ("x-bot-cloud-last-error", "boom"),
    ]


def test_order_servicer_propagates_cloud_metadata() -> None:
    class _ExecutionService:
        def __init__(self) -> None:
            self.cancelled: list[str] = []

        def execute(self, _order, _context):
            return SimpleNamespace(order_id="abc", raw_response={})

        def cancel(self, order_id, _context):
            self.cancelled.append(order_id)

    class _Ctx:
        def __init__(self) -> None:
            self.cloud_health_headers = {"x-bot-cloud-health": "0", "x-bot-cloud-last-error": "boom"}
            self.primary_symbol = "BTCUSDT"
            self.pipeline = SimpleNamespace(execution_service=_ExecutionService())
            self.execution_context = object()
            self.authorized = False
            self.portfolio_refreshed = False

        def authorize(self, _context):
            self.authorized = True

        def refresh_portfolio(self) -> None:
            self.portfolio_refreshed = True

        def emit_alert(self, *_, **__):
            return None

    class _RpcContext:
        def __init__(self) -> None:
            self.initial_metadata: list[tuple[str, str]] | None = None
            self.trailing_metadata: list[tuple[str, str]] | None = None

        def send_initial_metadata(self, metadata):
            self.initial_metadata = list(metadata)

        def set_trailing_metadata(self, metadata):
            self.trailing_metadata = list(metadata)

    from bot_core.api.server import _OrderServicer  # lokalny import, aby zachować zależności testu

    ctx = _Ctx()
    servicer = _OrderServicer(ctx)
    rpc_context = _RpcContext()

    request = trading_pb2.SubmitOrderRequest(
        instrument=trading_pb2.Instrument(symbol="BTCUSDT"),
        side=trading_pb2.ORDER_SIDE_BUY,
        quantity=1.0,
        type=trading_pb2.ORDER_TYPE_MARKET,
    )

    response = servicer.SubmitOrder(request, rpc_context)

    assert response.status == trading_pb2.ORDER_STATUS_ACCEPTED
    assert ctx.authorized is True
    assert ctx.portfolio_refreshed is True
    assert rpc_context.initial_metadata == rpc_context.trailing_metadata
    assert rpc_context.initial_metadata == [
        ("x-bot-cloud-health", "0"),
        ("x-bot-cloud-last-error", "boom"),
    ]


def test_marketplace_servicer_propagates_cloud_metadata() -> None:
    class _Doc:
        preset_id = "preset-1"
        version = "v1"
        metadata = {}
        payload = {"name": "demo"}
        verification = SimpleNamespace(verified=True)
        path = None
        tags: list[str] = []
        issues: list[str] = []

    class _Ctx:
        def __init__(self) -> None:
            self.cloud_health_headers = {"x-bot-cloud-health": "1", "x-bot-cloud-last-error": ""}
            self.marketplace_repository = object()
            self.marketplace_enabled = True
            self.authorized = False

        def authorize(self, _context):
            self.authorized = True

        def list_marketplace_presets(self):
            return [_Doc()]

    class _RpcContext:
        def __init__(self) -> None:
            self.initial_metadata: list[tuple[str, str]] | None = None
            self.trailing_metadata: list[tuple[str, str]] | None = None

        def send_initial_metadata(self, metadata):
            self.initial_metadata = list(metadata)

        def set_trailing_metadata(self, metadata):
            self.trailing_metadata = list(metadata)

    from bot_core.api.server import _MarketplaceServicer  # lokalny import, aby zachować zależności testu

    ctx = _Ctx()
    servicer = _MarketplaceServicer(ctx)
    rpc_context = _RpcContext()

    response = servicer.ListPresets(trading_pb2.ListMarketplacePresetsRequest(), rpc_context)

    assert ctx.authorized is True
    assert len(response.presets) == 1
    assert rpc_context.initial_metadata == rpc_context.trailing_metadata
    assert rpc_context.initial_metadata == [
        ("x-bot-cloud-health", "1"),
        ("x-bot-cloud-last-error", ""),
    ]


def test_cloud_auth_servicer_propagates_cloud_metadata() -> None:
    headers = {"x-bot-cloud-health": "0", "x-bot-cloud-last-error": "boom"}

    class _Manager:
        def authorize(self, _request):
            raise CloudAuthorizationError("denied")

    class _RpcContext:
        def __init__(self) -> None:
            self.initial_metadata: list[tuple[str, str]] | None = None
            self.trailing_metadata: list[tuple[str, str]] | None = None
            self.code = None
            self.details = None

        def send_initial_metadata(self, metadata):
            self.initial_metadata = list(metadata)

        def set_trailing_metadata(self, metadata):
            self.trailing_metadata = list(metadata)

        def set_code(self, code):
            self.code = code

        def set_details(self, details):
            self.details = details

    servicer = CloudAuthServicer(_Manager(), health_headers_provider=lambda: headers)
    rpc_context = _RpcContext()

    response = servicer.AuthorizeClient(trading_pb2.CloudAuthRequest(), rpc_context)

    assert response.authorized is False
    assert rpc_context.code == grpc.StatusCode.PERMISSION_DENIED
    assert rpc_context.initial_metadata == rpc_context.trailing_metadata
    assert rpc_context.initial_metadata == [
        ("x-bot-cloud-health", "0"),
        ("x-bot-cloud-last-error", "boom"),
    ]
