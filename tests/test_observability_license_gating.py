"""Testy wymuszania licencji dla modułów observability."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from bot_core.observability.hypercare import ObservabilityCycleConfig, ObservabilityHypercareCycle, SLOOutputConfig
from bot_core.observability.server import start_http_server
from bot_core.observability.ui_metrics import UiTelemetryPrometheusExporter
from bot_core.runtime.metrics_alerts import UiTelemetryAlertSink
from bot_core.security.capabilities import build_capabilities_from_payload
from bot_core.security.guards import (
    LicenseCapabilityError,
    install_capability_guard,
    reset_capability_guard,
)


@pytest.fixture(autouse=True)
def _reset_guard() -> None:
    reset_capability_guard()
    yield
    reset_capability_guard()


def _install(payload: dict[str, object]) -> None:
    capabilities = build_capabilities_from_payload(payload, effective_date=date(2025, 1, 1))
    install_capability_guard(capabilities)


def test_start_http_server_requires_observability_module() -> None:
    payload = {
        "edition": "pro",
        "modules": {"observability_ui": False},
        "environments": ["paper"],
        "exchanges": {"binance_spot": True},
    }
    _install(payload)

    with pytest.raises(LicenseCapabilityError):
        start_http_server(0)


def test_start_http_server_succeeds_with_module_enabled() -> None:
    payload = {
        "edition": "pro",
        "modules": {"observability_ui": True},
        "environments": ["paper"],
        "exchanges": {"binance_spot": True},
    }
    _install(payload)

    server, thread = start_http_server(0)
    try:
        server.shutdown()
        server.server_close()
    finally:
        thread.join(timeout=1.0)


def test_ui_telemetry_exporter_requires_module() -> None:
    payload = {
        "edition": "pro",
        "modules": {"observability_ui": False},
        "environments": ["paper"],
        "exchanges": {"binance_spot": True},
    }
    _install(payload)

    with pytest.raises(LicenseCapabilityError):
        UiTelemetryPrometheusExporter()


def test_ui_telemetry_alert_sink_requires_module() -> None:
    payload = {
        "edition": "pro",
        "modules": {"observability_ui": False},
        "environments": ["paper"],
        "exchanges": {"binance_spot": True},
    }
    _install(payload)

    router = SimpleNamespace(dispatch=lambda *_args, **_kwargs: None)

    with pytest.raises(LicenseCapabilityError):
        UiTelemetryAlertSink(router)  # type: ignore[arg-type]


def test_hypercare_cycle_requires_runtime(tmp_path: Path) -> None:
    payload = {
        "edition": "commercial",
        "modules": {"observability_ui": True},
        "runtime": {"hypercare": False},
        "environments": ["live"],
        "exchanges": {"binance_spot": True},
    }
    _install(payload)

    config = ObservabilityCycleConfig(
        definitions_path=tmp_path / "definitions.json",
        metrics_path=tmp_path / "metrics.json",
        slo=SLOOutputConfig(json_path=tmp_path / "slo.json"),
    )

    cycle = ObservabilityHypercareCycle(config)

    with pytest.raises(LicenseCapabilityError):
        cycle.run()
