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


def test_cloud_runtime_service_initializes_license_service_for_client_bundles(
    tmp_path, monkeypatch
):
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
        assert calls == ["init"], (
            "Powinien zostać zainicjalizowany LicenseService dla bundli klientów"
        )
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


def test_required_cloud_handshake_never_starts_without_allowed_clients(tmp_path, monkeypatch):
    from bot_core.cloud.security import CloudSecurityError

    monkeypatch.setattr("bot_core.cloud.service.LocalRuntimeServer", _DummyServer)
    config = _build_config(tmp_path)
    config.security = CloudSecurityConfig(require_handshake=True, allowed_clients=())
    context = _DummyContext()
    service = CloudRuntimeService(config, context_builder=lambda **_: context)
    try:
        with pytest.raises(CloudSecurityError):
            service.start()
        assert context.stopped
        assert service.address is None
    finally:
        service.stop()


def test_cloud_service_restart_discards_previous_security_manager(tmp_path, monkeypatch):
    from bot_core.generated import trading_pb2
    from bot_core.security.fingerprint import sign_license_payload
    from bot_core.cloud.security import CloudAuthorizationError

    monkeypatch.setattr("bot_core.cloud.service.LocalRuntimeServer", _DummyServer)
    config = _build_config(tmp_path)
    config.security.allowed_clients = (
        CloudAllowedClientConfig("LIC-TEST", "HW-TEST", b"synthetic"),
    )
    context = _DummyContext()
    service = CloudRuntimeService(config, context_builder=lambda **_: context)
    try:
        service.start()
        previous = service.security_manager
        nonce = previous.issue_challenge(
            trading_pb2.CloudChallengeRequest(license_id="LIC-TEST", fingerprint="HW-TEST")
        )
        signature = sign_license_payload(
            {"license_id": "LIC-TEST", "fingerprint": "HW-TEST", "nonce": nonce},
            fingerprint="HW-TEST",
            secret=b"synthetic",
        )
        request = trading_pb2.CloudAuthRequest(
            license_id="LIC-TEST",
            fingerprint="HW-TEST",
            nonce=nonce,
            signature=trading_pb2.CloudAuthSignature(**signature),
        )
        service.stop()
        service.start()
        assert service.security_manager is not previous
        with pytest.raises(CloudAuthorizationError):
            service.security_manager.authorize(request)
    finally:
        service.stop()


@pytest.mark.parametrize("failure", ["exception", "before_start", "after_start"])
def test_cloud_service_abandoned_start_discards_only_its_security_manager(
    tmp_path, monkeypatch, failure
):
    from bot_core.cloud.security import CloudAuthorizationError
    from bot_core.generated import trading_pb2
    from bot_core.security.fingerprint import sign_license_payload

    captured = {}

    def abandon():
        service.stop()
        captured["replacement"] = service._ensure_security_manager()

    class _CapturingServer(_DummyServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if captured:
                return
            previous = service.security_manager
            captured["previous"] = previous
            nonce = previous.issue_challenge(
                trading_pb2.CloudChallengeRequest(license_id="LIC-TEST", fingerprint="HW-TEST")
            )
            signature = sign_license_payload(
                {"license_id": "LIC-TEST", "fingerprint": "HW-TEST", "nonce": nonce},
                fingerprint="HW-TEST",
                secret=b"synthetic",
            )
            captured["request"] = trading_pb2.CloudAuthRequest(
                license_id="LIC-TEST",
                fingerprint="HW-TEST",
                nonce=nonce,
                signature=trading_pb2.CloudAuthSignature(**signature),
            )
            if failure == "before_start":
                abandon()

    class _Orchestrator:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            if captured.get("triggered"):
                return
            captured["triggered"] = True
            if failure == "exception":
                raise RuntimeError("synthetic failed startup")
            abandon()

        def stop(self):
            pass

        def health_snapshot(self):
            return {}

    monkeypatch.setattr("bot_core.cloud.service.LocalRuntimeServer", _CapturingServer)
    monkeypatch.setattr("bot_core.cloud.service.CloudOrchestrator", _Orchestrator)
    config = _build_config(tmp_path)
    config.security.allowed_clients = (
        CloudAllowedClientConfig("LIC-TEST", "HW-TEST", b"synthetic"),
    )
    service = CloudRuntimeService(config, context_builder=lambda **_: _DummyContext())
    try:
        if failure == "exception":
            with pytest.raises(RuntimeError, match="synthetic failed startup"):
                service.start()
            assert service.security_manager is None
        else:
            service.start()
            assert service.security_manager is captured["replacement"]
        previous = captured["previous"]
        assert not previous._challenges
        captured["triggered"] = True
        service.start()
        assert service.security_manager is not previous
        with pytest.raises(CloudAuthorizationError):
            service.security_manager.authorize(captured["request"])
    finally:
        service.stop()


def test_cloud_service_replaced_start_does_not_adopt_new_security_manager(tmp_path, monkeypatch):
    from bot_core.generated import trading_pb2

    monkeypatch.setattr("bot_core.cloud.service.LocalRuntimeServer", _DummyServer)
    config = _build_config(tmp_path)
    config.security.allowed_clients = (
        CloudAllowedClientConfig("LIC-TEST", "HW-TEST", b"synthetic"),
    )
    service = CloudRuntimeService(config, context_builder=lambda **_: _DummyContext())
    captured = {}

    def replace_start():
        if captured:
            return
        captured["replaced"] = True
        service.stop()
        service.start()
        captured["replacement"] = service.security_manager

    monkeypatch.setattr(service, "_maybe_attach_license", replace_start)
    try:
        service.start()
        assert service.security_manager is captured["replacement"]
        assert service.security_manager.issue_challenge(
            trading_pb2.CloudChallengeRequest(license_id="LIC-TEST", fingerprint="HW-TEST")
        )
    finally:
        service.stop()
