"""Testy pomocnicze dla konfiguracji TLS serwera MetricsService."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from bot_core.runtime import metrics_service


@pytest.fixture(autouse=True)
def stub_grpc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zapewnia atrapę grpc z metodą ``ssl_server_credentials``."""

    stub = SimpleNamespace(
        ssl_server_credentials=lambda *args, **kwargs: (args, kwargs),
    )
    monkeypatch.setattr(metrics_service, "grpc", stub)


def test_build_server_credentials_ignores_disabled_mapping() -> None:
    result = metrics_service._build_server_credentials({"enabled": False})
    assert result is None


def test_build_server_credentials_ignores_disabled_dataclass() -> None:
    @dataclass(slots=True)
    class Dummy:
        enabled: bool = False

    result = metrics_service._build_server_credentials(Dummy(enabled=False))
    assert result is None


def test_build_server_credentials_requires_cert_and_key() -> None:
    with pytest.raises(ValueError):
        metrics_service._build_server_credentials({"enabled": True})


def test_build_server_credentials_passes_material(tmp_path) -> None:
    cert = tmp_path / "server.crt"
    key = tmp_path / "server.key"
    ca = tmp_path / "clients.pem"
    cert.write_text("cert", encoding="utf-8")
    key.write_text("key", encoding="utf-8")
    ca.write_text("ca", encoding="utf-8")

    args, kwargs = metrics_service._build_server_credentials(
        {
            "enabled": True,
            "certificate_path": cert,
            "private_key_path": key,
            "client_ca_path": ca,
            "require_client_auth": True,
        }
    )

    # pierwszy argument to lista par (klucz, certyfikat)
    assert args[0][0][0] == key.read_bytes()
    assert args[0][0][1] == cert.read_bytes()
    assert kwargs["root_certificates"] == ca.read_bytes()
    assert kwargs["require_client_auth"] is True
