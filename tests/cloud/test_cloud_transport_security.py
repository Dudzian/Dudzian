"""Negative-path tests for the cloud gRPC transport boundary."""

from pathlib import Path

import pytest

from bot_core.cloud.config import (
    CloudConfigError,
    CloudServerConfig,
    load_cloud_server_config,
    validate_cloud_bind_host,
)
from bot_core.cloud.service import CloudRuntimeService


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "::", "[::]", "192.168.0.10", "8.8.8.8", "example.org", ""],
)
def test_cloud_bind_rejects_exposed_or_unresolved_hosts(host: str) -> None:
    with pytest.raises(CloudConfigError):
        validate_cloud_bind_host(host)


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "::1", "[::1]", "localhost"])
def test_cloud_bind_accepts_explicit_loopback_hosts(host: str) -> None:
    validate_cloud_bind_host(host)


def test_cloud_service_checks_bind_before_context_bootstrap() -> None:
    calls: list[object] = []

    def build_context(**kwargs: object) -> None:
        calls.append(kwargs)
        raise AssertionError("Runtime must not start on an exposed gRPC transport")

    service = CloudRuntimeService(
        CloudServerConfig(host="0.0.0.0"),
        context_builder=build_context,
    )
    with pytest.raises(CloudConfigError):
        service.start()
    assert calls == []


def test_cloud_config_defaults_to_loopback(tmp_path: Path) -> None:
    config_file = tmp_path / "cloud.yaml"
    config_file.write_text("port: 50052\n", encoding="utf-8")
    config = load_cloud_server_config(config_file)
    assert config.host == "127.0.0.1"


def test_cloud_config_rejects_unimplemented_tls_flag(tmp_path: Path) -> None:
    config_file = tmp_path / "cloud.yaml"
    config_file.write_text(
        "host: 127.0.0.1\ntls:\n  enabled: true\n", encoding="utf-8"
    )
    with pytest.raises(CloudConfigError, match="tls.enabled"):
        load_cloud_server_config(config_file)


def test_cloud_config_rejects_nonloopback_host(tmp_path: Path) -> None:
    config_file = tmp_path / "cloud.yaml"
    config_file.write_text("host: 0.0.0.0\n", encoding="utf-8")
    with pytest.raises(CloudConfigError):
        load_cloud_server_config(config_file)
