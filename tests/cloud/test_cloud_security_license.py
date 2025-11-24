from types import SimpleNamespace

import pytest

from bot_core.cloud.config import (
    CloudAllowedClientConfig,
    CloudMarketplaceConfig,
    CloudRuntimeConfig,
    CloudSecurityConfig,
    CloudServerConfig,
)
from bot_core.cloud.service import CloudRuntimeService
from bot_core.security.license_service import LicenseServiceError


class _DummyContext:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.retrain_scheduler = None
        self.marketplace_repository = None

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class _DummyServer:
    def __init__(self, context, host: str, port: int, max_workers: int, *, interceptors=None):
        del host, port, max_workers, interceptors
        self._context = context
        self.grpc_server = SimpleNamespace(
            add_generic_rpc_handlers=lambda _handlers: None,
            add_registered_method_handlers=lambda _name, _handlers: None,
        )
        self.address = "127.0.0.1:0"

    def start(self) -> None:
        return None

    def stop(self, _timeout: float) -> None:
        return None

    def wait(self) -> None:
        return None


def _build_config(tmp_path, *, require_handshake: bool = True) -> CloudServerConfig:
    allowed = (
        CloudAllowedClientConfig(
            license_id="LIC-TEST",
            fingerprint="HW-TEST",
            shared_secret=b"secret",
            license_bundle_path=tmp_path / "bundle.json",
        ),
    )
    return CloudServerConfig(
        host="127.0.0.1",
        port=0,
        runtime=CloudRuntimeConfig(config_path=tmp_path / "runtime.yaml"),
        security=CloudSecurityConfig(require_handshake=require_handshake, allowed_clients=allowed),
        marketplace=CloudMarketplaceConfig(auto_reload=False),
    )


def test_cloud_runtime_service_initializes_license_service_for_client_bundles(tmp_path, monkeypatch):
    calls: list[str] = []

    class _FakeLicenseService:
        def __init__(self) -> None:
            calls.append("init")

    monkeypatch.setattr("bot_core.cloud.service.LicenseService", _FakeLicenseService)
    monkeypatch.setattr("bot_core.cloud.service.LocalRuntimeServer", _DummyServer)

    context = _DummyContext()
    service = CloudRuntimeService(
        _build_config(tmp_path),
        context_builder=lambda **_: context,
    )
    try:
        service.start()
        assert calls == ["init"], "Powinien zostać zainicjalizowany LicenseService dla bundli klientów"
        assert service.security_manager is not None
    finally:
        service.stop()


def test_cloud_runtime_service_stops_context_on_license_error(tmp_path, monkeypatch):
    class _FailingLicenseService:
        def __init__(self) -> None:
            raise LicenseServiceError("brak klucza publicznego")

    monkeypatch.setattr("bot_core.cloud.service.LicenseService", _FailingLicenseService)
    monkeypatch.setattr("bot_core.cloud.service.LocalRuntimeServer", _DummyServer)

    context = _DummyContext()
    service = CloudRuntimeService(
        _build_config(tmp_path),
        context_builder=lambda **_: context,
    )

    with pytest.raises(LicenseServiceError):
        service.start()
    assert context.started is True
    assert context.stopped is True, "W razie błędu licencji kontekst powinien zostać zatrzymany"
