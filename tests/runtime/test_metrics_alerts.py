import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from bot_core.alerts import AlertChannel, AlertMessage, DefaultAlertRouter, InMemoryAlertAuditLog
from bot_core.runtime import metrics_alerts
from bot_core.runtime.metrics_alerts import UiTelemetryAlertSink
from google.protobuf import json_format


class _DummyChannel(AlertChannel):
    def __init__(self, name: str = "hypercare-test") -> None:
        self.name = name
        self.messages: list[AlertMessage] = []

    def send(self, message: AlertMessage) -> None:  # pragma: no cover - simple append
        self.messages.append(message)

    def health_check(self) -> dict[str, str]:  # pragma: no cover - not relevant for tests
        return {"status": "ok"}


@pytest.fixture(autouse=True)
def _reset_feed_health_sink() -> None:
    metrics_alerts.reset_feed_health_alert_sink()
    yield
    metrics_alerts.reset_feed_health_alert_sink()


def test_emit_feed_health_event_dispatches_and_logs(tmp_path: Path) -> None:
    audit_log = InMemoryAlertAuditLog()
    router = DefaultAlertRouter(audit_log=audit_log)
    channel = _DummyChannel()
    router.register(channel)

    sink = UiTelemetryAlertSink(
        router,
        jsonl_path=tmp_path / "alerts.jsonl",
        log_reduce_motion_events=False,
        log_reduce_motion_incident_events=False,
        log_overlay_events=False,
        log_jank_events=False,
        log_retry_backlog_events=False,
        log_tag_inactivity_events=False,
        log_performance_events=False,
    )

    sink.emit_feed_health_event(
        severity="critical",
        title="Latency threshold exceeded",
        body="Test notification",
        context={"adapter": "grpc", "metric": "latency"},
        payload={"metric": "latency", "metric_value": 123.0},
    )

    assert channel.messages, "Kanał HyperCare powinien otrzymać komunikat o degradacji feedu"
    message = channel.messages[-1]
    assert message.title == "Latency threshold exceeded"
    assert message.severity == "critical"
    assert message.context["adapter"] == "grpc"

    audit_entries = list(audit_log.export())
    assert audit_entries, "Alert powinien zostać zapisany w audycie routera"
    assert audit_entries[0]["category"] == message.category

    jsonl_path = tmp_path / "alerts.jsonl"
    assert jsonl_path.exists(), "Plik JSONL z alertami feedu powinien zostać utworzony"
    record = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["category"] == message.category
    assert record["payload"]["event"] == "feed_health"
    assert record["context"]["metric"] == "latency"


def test_get_feed_health_alert_sink_uses_provided_router(tmp_path: Path) -> None:
    audit_log = InMemoryAlertAuditLog()
    custom_router = DefaultAlertRouter(audit_log=audit_log)
    sink = metrics_alerts.get_feed_health_alert_sink(
        router=custom_router, jsonl_path=tmp_path / "custom.jsonl"
    )
    assert sink is not None
    assert sink._router is custom_router  # type: ignore[attr-defined]
    assert sink.jsonl_path == tmp_path / "custom.jsonl"

    cached = metrics_alerts.get_feed_health_alert_sink()
    assert cached is sink, "Sink powinien być singletonem w module"


def test_get_feed_health_alert_sink_falls_back_to_memory_router(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called: list[None] = []

    def _fake_builder() -> None:
        called.append(None)
        return None

    monkeypatch.setattr(metrics_alerts, "_build_feed_alert_router", _fake_builder)

    sink = metrics_alerts.get_feed_health_alert_sink(jsonl_path=tmp_path / "fallback.jsonl")
    assert sink is not None
    assert isinstance(sink._router, DefaultAlertRouter)  # type: ignore[attr-defined]
    assert isinstance(sink._router.audit_log, InMemoryAlertAuditLog)  # type: ignore[attr-defined]
    assert called, "Powinien zostać wykonany bootstrap kanałów HyperCare"


def test_env_hypercare_webhook_channel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BOT_CORE_UI_FEED_HYPERCARE_WEBHOOK", "https://hypercare.example/webhook")

    sent: list[dict[str, object]] = []

    def _fake_urlopen(request, timeout=None):  # pragma: no cover - prosty stub
        sent.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "payload": json.loads(request.data.decode("utf-8")),
            }
        )

        class _Response:
            def read(self) -> bytes:
                return b"ok"

        return _Response()

    monkeypatch.setattr(metrics_alerts.urllib_request, "urlopen", _fake_urlopen)
    sink = metrics_alerts.get_feed_health_alert_sink(jsonl_path=tmp_path / "alerts.jsonl")
    assert sink is not None
    registered = [channel.name for channel in sink._router.channels]  # type: ignore[attr-defined]
    assert "hypercare-webhook" in registered

    sink.emit_feed_health_event(
        severity="warning",
        title="Latency warning",
        body="latency 2500ms",
        context={"adapter": "grpc"},
        payload={"metric": "latency", "metric_value": 2500.0},
    )

    assert sent, "Kanał webhook HyperCare powinien otrzymać alert"
    last = sent[-1]
    assert last["url"].startswith("https://hypercare.example")
    assert last["payload"]["severity"] == "warning"
    assert last["payload"]["context"]["adapter"] == "grpc"


def test_cloud_alert_channel_sends_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    if metrics_alerts.struct_pb2 is None or metrics_alerts.timestamp_pb2 is None:
        pytest.skip("protobuf structs not available")

    selection = SimpleNamespace(
        profile_name="remote",
        client=SimpleNamespace(
            address="127.0.0.1:5000",
            use_tls=False,
            metadata={},
            metadata_env={},
            metadata_files={},
        ),
    )

    published: list[dict[str, object]] = []

    class _Stub:
        def PublishAlert(
            self, request, metadata=None, timeout=None
        ) -> None:  # pragma: no cover - simple capture
            payload = json_format.MessageToDict(request)
            payload["metadata"] = metadata or []
            payload["timeout"] = timeout
            published.append(payload)

    channel = metrics_alerts._CloudAlertChannel(
        selection,
        environment="prod",
        channel_factory=lambda _addr: object(),
        stub_factory=lambda _channel: _Stub(),
    )

    message = AlertMessage(
        category="ui.feed.health",
        title="Latency threshold exceeded",
        body="p95 4.2s",
        severity="critical",
        context={"metric": "latency", "adapter": "grpc"},
    )
    channel.send(message)

    assert published, "Kanał cloudowy powinien wysłać payload do CloudAlertService"
    payload = published[-1]
    assert payload["category"] == message.category
    assert payload["severity"] == "critical"
    assert payload["context"]["metric"] == "latency"
    assert payload["environment"] == "prod"


def test_build_cloud_alert_channel_registers_remote_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime_path = tmp_path / "runtime.yaml"
    runtime_path.write_text("cloud: {}\n", encoding="utf-8")

    runtime_config = SimpleNamespace(
        cloud=SimpleNamespace(enabled=True, default_profile="remote"),
    )

    selection = SimpleNamespace(
        profile_name="remote",
        profile=SimpleNamespace(mode="remote"),
        client=SimpleNamespace(
            address="127.0.0.1:5000",
            use_tls=False,
            metadata={},
            metadata_env={},
            metadata_files={},
        ),
    )

    monkeypatch.setattr(
        metrics_alerts,
        "resolve_runtime_cloud_client",
        lambda *_args, **_kwargs: selection,
    )
    monkeypatch.setattr(
        metrics_alerts,
        "grpc",
        SimpleNamespace(insecure_channel=lambda addr: object()),
        raising=False,
    )
    monkeypatch.setattr(
        metrics_alerts, "struct_pb2", metrics_alerts.struct_pb2 or SimpleNamespace(), raising=False
    )
    monkeypatch.setattr(
        metrics_alerts,
        "timestamp_pb2",
        metrics_alerts.timestamp_pb2 or SimpleNamespace(),
        raising=False,
    )

    sentinel = object()

    def _dummy_channel(*_args, **_kwargs):
        return sentinel

    monkeypatch.setattr(metrics_alerts, "_CloudAlertChannel", _dummy_channel)

    channel = metrics_alerts._build_cloud_alert_channel(runtime_path, runtime_config, "prod")
    assert channel is sentinel


def test_build_secret_manager_degrades_without_warning_by_default(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("BOT_CORE_UI_SECRET_PASSPHRASE", raising=False)
    monkeypatch.delenv("BOT_CORE_UI_SECRET_PATH", raising=False)
    monkeypatch.delenv("BOT_CORE_UI_REQUIRE_SECRET_STORE", raising=False)
    monkeypatch.setattr(
        metrics_alerts,
        "create_default_secret_storage",
        lambda **_kwargs: (_ for _ in ()).throw(metrics_alerts.SecretStorageError("missing")),
    )

    with caplog.at_level(logging.WARNING, logger=metrics_alerts.__name__):
        manager = metrics_alerts._build_secret_manager()

    assert manager is None
    assert not any("magazynu sekretów" in message for message in caplog.messages)


def test_build_secret_manager_warns_when_secret_store_required(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("BOT_CORE_UI_REQUIRE_SECRET_STORE", "1")
    monkeypatch.setattr(
        metrics_alerts,
        "create_default_secret_storage",
        lambda **_kwargs: (_ for _ in ()).throw(metrics_alerts.SecretStorageError("missing")),
    )

    with caplog.at_level(logging.WARNING, logger=metrics_alerts.__name__):
        manager = metrics_alerts._build_secret_manager()

    assert manager is None
    assert any("magazynu sekretów" in message for message in caplog.messages)


@pytest.mark.parametrize(
    ("env_name", "env_value"),
    [
        pytest.param("BOT_CORE_UI_SECRET_PASSPHRASE", "explicit-passphrase", id="explicit-passphrase"),
        pytest.param("BOT_CORE_UI_SECRET_PATH", "/tmp/non-existent-secrets.json", id="explicit-path"),
    ],
)
def test_build_secret_manager_warns_for_explicit_secret_config(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    env_name: str,
    env_value: str,
) -> None:
    monkeypatch.setenv(env_name, env_value)
    monkeypatch.delenv("BOT_CORE_UI_REQUIRE_SECRET_STORE", raising=False)
    monkeypatch.setattr(
        metrics_alerts,
        "create_default_secret_storage",
        lambda **_kwargs: (_ for _ in ()).throw(metrics_alerts.SecretStorageError("missing")),
    )

    with caplog.at_level(logging.WARNING, logger=metrics_alerts.__name__):
        manager = metrics_alerts._build_secret_manager()

    assert manager is None
    assert any("magazynu sekretów" in message for message in caplog.messages)
