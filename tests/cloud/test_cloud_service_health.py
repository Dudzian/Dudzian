from pathlib import Path

import pytest

from bot_core.cloud.config import CloudMarketplaceConfig, CloudRuntimeConfig, CloudSecurityConfig, CloudServerConfig
from bot_core.cloud.service import CloudRuntimeService


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
