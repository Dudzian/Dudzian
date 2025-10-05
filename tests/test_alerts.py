"""Testy modułu alertów."""
from __future__ import annotations

import base64
import json
import logging
from email.message import EmailMessage
from pathlib import Path
from typing import List, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from urllib import request
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from bot_core.alerts import (
    AlertThrottle,
    AlertDeliveryError,
    AlertMessage,
    DefaultAlertRouter,
    EmailChannel,
    FileAlertAuditLog,
    InMemoryAlertAuditLog,
    MessengerChannel,
    SMSChannel,
    SignalChannel,
    TelegramChannel,
    WhatsAppChannel,
    build_coverage_alert_context,
    build_coverage_alert_message,
    dispatch_coverage_alert,
    run_coverage_check_and_alert,
    DEFAULT_SMS_PROVIDERS,
    get_sms_provider,
)
from bot_core.alerts.base import AlertChannel
from bot_core.data.ohlcv import SummaryThresholdResult, coerce_summary_mapping
from bot_core.observability.metrics import MetricsRegistry
from tests.test_check_data_coverage_script import (  # noqa: E402
    _generate_rows,
    _last_row_iso,
    _write_cache,
    _write_config,
)


class DummyChannel(AlertChannel):
    name = "dummy"

    def __init__(self) -> None:
        self.messages: List[AlertMessage] = []

    def send(self, message: AlertMessage) -> None:
        self.messages.append(message)

    def health_check(self) -> dict[str, str]:
        return {"status": "ok", "delivered": str(len(self.messages))}


class FailingChannel(AlertChannel):
    name = "failing"

    def send(self, message: AlertMessage) -> None:  # noqa: ARG002 - interfejs wymaga parametru
        raise AlertDeliveryError("celowy błąd")

    def health_check(self) -> dict[str, str]:
        return {"status": "error"}


class HealthFailingChannel(AlertChannel):
    name = "health-failing"

    def __init__(self) -> None:
        self.messages: List[AlertMessage] = []

    def send(self, message: AlertMessage) -> None:
        self.messages.append(message)

    def health_check(self) -> dict[str, str]:  # noqa: D401
        raise RuntimeError("boom")


@pytest.fixture()
def sample_message() -> AlertMessage:
    return AlertMessage(
        category="risk",
        title="Limit ryzyka osiągnięty",
        body="Dzienne straty przekroczyły próg.",
        severity="critical",
        context={"profile": "conservative", "instrument": "BTC/USDT"},
    )


def test_router_dispatch_records_audit(sample_message: AlertMessage) -> None:
    audit = InMemoryAlertAuditLog()
    router = DefaultAlertRouter(audit_log=audit)
    channel = DummyChannel()
    router.register(channel)

    router.dispatch(sample_message)

    assert channel.messages == [sample_message]
    exported = tuple(audit.export())
    assert len(exported) == 1
    assert exported[0]["channel"] == "dummy"
    assert channel.messages[0].severity == "critical"
    assert exported[0]["severity"] == "critical"


def test_router_dispatch_updates_metrics(sample_message: AlertMessage) -> None:
    audit = InMemoryAlertAuditLog()
    registry = MetricsRegistry()
    router = DefaultAlertRouter(
        audit_log=audit,
        metrics_registry=registry,
        metric_labels={"environment": "paper"},
    )
    channel = DummyChannel()
    router.register(channel)

    router.dispatch(sample_message)

    sent_metric = registry.get("alerts_sent_total")
    assert sent_metric.value(
        labels={"environment": "paper", "channel": "dummy", "severity": "critical"}
    ) == pytest.approx(1.0)


def test_router_continues_on_error(sample_message: AlertMessage, caplog: pytest.LogCaptureFixture) -> None:
    audit = InMemoryAlertAuditLog()
    router = DefaultAlertRouter(audit_log=audit)
    router.register(FailingChannel())
    router.register(DummyChannel())

    router.dispatch(sample_message)

    exported = tuple(audit.export())
    assert len(exported) == 1
    assert exported[0]["channel"] == "dummy"
    assert any("Część kanałów zgłosiła błędy" in record.message for record in caplog.records)


def test_router_records_failure_metric(sample_message: AlertMessage) -> None:
    audit = InMemoryAlertAuditLog()
    registry = MetricsRegistry()
    router = DefaultAlertRouter(
        audit_log=audit,
        metrics_registry=registry,
        metric_labels={"environment": "paper"},
    )
    router.register(FailingChannel())
    router.register(DummyChannel())

    router.dispatch(sample_message)

    failures = registry.get("alerts_failed_total")
    assert failures.value(
        labels={"environment": "paper", "channel": "failing", "severity": "critical"}
    ) == pytest.approx(1.0)

    sent_metric = registry.get("alerts_sent_total")
    assert sent_metric.value(
        labels={"environment": "paper", "channel": "dummy", "severity": "critical"}
    ) == pytest.approx(1.0)


def test_file_alert_audit_log_writes_entries(tmp_path: Path, sample_message: AlertMessage) -> None:
    log_dir = tmp_path / "audit"
    audit = FileAlertAuditLog(directory=log_dir, filename_pattern="alerts-%Y%m%d.jsonl", retention_days=30)

    audit.append(sample_message, channel="telegram")

    expected_file = log_dir / sample_message.timestamp.astimezone(timezone.utc).strftime("alerts-%Y%m%d.jsonl")
    assert expected_file.exists()
    contents = expected_file.read_text("utf-8").strip().splitlines()
    assert contents
    record = json.loads(contents[0])
    assert record["channel"] == "telegram"
    assert record["title"] == sample_message.title
    exported = tuple(audit.export())
    assert len(exported) == 1
    assert exported[0]["severity"] == sample_message.severity


def test_file_alert_audit_log_respects_retention(tmp_path: Path) -> None:
    audit = FileAlertAuditLog(directory=tmp_path / "audit", retention_days=3)

    old_ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
    new_ts = datetime(2023, 1, 5, tzinfo=timezone.utc)
    old_message = AlertMessage(
        category="risk",
        title="Old",
        body="Old alert",
        severity="info",
        context={},
        timestamp=old_ts,
    )
    new_message = AlertMessage(
        category="risk",
        title="New",
        body="New alert",
        severity="info",
        context={},
        timestamp=new_ts,
    )

    audit.append(old_message, channel="email")
    old_file = (tmp_path / "audit") / "alerts-20230101.jsonl"
    assert old_file.exists()


    audit.append(new_message, channel="telegram")
    assert not old_file.exists()
    files = list((tmp_path / "audit").glob("alerts-*.jsonl"))
    assert len(files) == 1


def test_build_coverage_alert_context_serializes_thresholds() -> None:
    summary = coerce_summary_mapping(
        {
            "status": "warning",
            "total": 5,
            "ok": 4,
            "warning": 1,
            "error": 0,
            "ok_ratio": 0.8,
            "stale_entries": 0,
            "worst_gap": {
                "symbol": "BTCUSDT",
                "interval": "1h",
                "gap_minutes": 180.0,
                "threshold_minutes": 240,
            },
        }
    )
    threshold = SummaryThresholdResult(
        issues=("max_gap_exceeded:180.0>120.0",),
        thresholds={"max_gap_minutes": 120.0},
        observed={"worst_gap_minutes": 180.0, "ok_ratio": 0.8},
    )

    context = build_coverage_alert_context(summary=summary, threshold_result=threshold)

    summary_payload = json.loads(context["summary"])
    thresholds_payload = json.loads(context["thresholds"])

    assert summary_payload["status"] == "warning"
    assert context["summary_ok_ratio"] == "0.8000"
    assert thresholds_payload["max_gap_minutes"] == pytest.approx(120.0)
    assert "max_gap_exceeded" in context.get("threshold_issues", "")
    assert context["observed_worst_gap_minutes"] == "180.00"


def test_build_coverage_alert_context_handles_missing_data() -> None:
    context = build_coverage_alert_context(summary=None, threshold_result=None)

    assert context["summary_status"] == "unknown"
    assert context["summary_worst_gap_minutes"] == "n/a"
    assert "thresholds" not in context


def _coverage_cli_payload(
    *,
    issues: Sequence[str] | None = None,
    threshold_issues: Sequence[str] | None = None,
    thresholds: Mapping[str, float] | None = None,
    observed: Mapping[str, float | None] | None = None,
    summary_overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    summary = {
        "status": "ok",
        "total": 4,
        "ok": 4,
        "warning": 0,
        "error": 0,
        "ok_ratio": 1.0,
        "stale_entries": 0,
        "worst_gap": {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "gap_minutes": 45.0,
            "threshold_minutes": 30.0,
        },
    }
    if summary_overrides:
        summary.update(summary_overrides)

    threshold_payload = {
        "issues": list(threshold_issues or ()),
        "thresholds": dict(thresholds or {}),
        "observed": dict(observed or {}),
    }

    return {
        "environment": "binance_paper",
        "exchange": "binance_spot",
        "manifest_path": "/tmp/ohlcv_manifest.sqlite",
        "as_of": "2024-01-01T00:00:00+00:00",
        "entries": [],
        "issues": list(issues or ()),
        "summary": summary,
        "threshold_evaluation": threshold_payload,
        "threshold_issues": list(threshold_issues or ()),
        "status": summary.get("status", "ok"),
    }


def test_build_coverage_alert_message_threshold_only() -> None:
    payload = _coverage_cli_payload(
        threshold_issues=("max_gap_exceeded:45.0>30.0",),
        thresholds={"max_gap_minutes": 30.0},
        observed={"worst_gap_minutes": 45.0},
    )

    message = build_coverage_alert_message(payload=payload)

    assert message.severity == "warning"
    assert message.category == "data.ohlcv"
    assert "Naruszenia progów jakości danych" in message.body
    assert message.context["environment"] == "binance_paper"
    assert message.context["threshold_issue_count"] == "1"
    issues = json.loads(message.context["threshold_issues"])
    assert issues == ["max_gap_exceeded:45.0>30.0"]


def test_dispatch_coverage_alert_skips_without_issues() -> None:
    router = DefaultAlertRouter(audit_log=InMemoryAlertAuditLog())
    channel = DummyChannel()
    router.register(channel)
    payload = _coverage_cli_payload()

    dispatched = dispatch_coverage_alert(router, payload=payload)

    assert dispatched is False
    assert channel.messages == []


def test_dispatch_coverage_alert_escalates_ok_ratio() -> None:
    router = DefaultAlertRouter(audit_log=InMemoryAlertAuditLog())
    channel = DummyChannel()
    router.register(channel)
    payload = _coverage_cli_payload(
        threshold_issues=("ok_ratio_below_threshold:0.7000<0.9000",),
        thresholds={"min_ok_ratio": 0.9},
        observed={"ok_ratio": 0.7, "total_entries": 12},
        summary_overrides={"ok_ratio": 0.7, "ok": 7, "total": 10},
    )

    dispatched = dispatch_coverage_alert(router, payload=payload)

    assert dispatched is True
    assert len(channel.messages) == 1
    message = channel.messages[0]
    assert message.severity == "critical"
    assert "ok_ratio_below_threshold" in message.body
    assert message.context["environment"] == "binance_paper"


def test_run_coverage_check_and_alert_dispatches(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache_threshold_alert"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 40)
    _write_cache(cache_dir, rows)
    config_path = _write_config(
        tmp_path,
        cache_dir,
        data_quality={"max_gap_minutes": 60.0},
    )

    router = DefaultAlertRouter(audit_log=InMemoryAlertAuditLog())
    channel = DummyChannel()
    router.register(channel)

    as_of = datetime.fromisoformat(_last_row_iso(rows)) + timedelta(hours=4)

    report, dispatched = run_coverage_check_and_alert(
        config_path=config_path,
        environment_name="binance_smoke",
        router=router,
        as_of=as_of,
    )

    assert dispatched is True
    assert channel.messages
    message = channel.messages[0]
    assert message.category == "data.ohlcv"
    assert "max_gap_exceeded" in message.body
    assert report.threshold_issues
    assert report.payload["status"] == "error"


def test_run_coverage_check_and_alert_uses_risk_profile_threshold(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache_profile_threshold_alert"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 40)
    _write_cache(cache_dir, rows)
    config_path = _write_config(
        tmp_path,
        cache_dir,
        risk_profile_data_quality={"max_gap_minutes": 45.0},
    )

    router = DefaultAlertRouter(audit_log=InMemoryAlertAuditLog())
    channel = DummyChannel()
    router.register(channel)

    as_of = datetime.fromisoformat(_last_row_iso(rows)) + timedelta(hours=2)

    report, dispatched = run_coverage_check_and_alert(
        config_path=config_path,
        environment_name="binance_smoke",
        router=router,
        as_of=as_of,
    )

    assert dispatched is True
    assert channel.messages
    assert any("max_gap_exceeded" in message.body for message in channel.messages)
    assert report.threshold_issues
    assert report.threshold_result is not None
    assert report.threshold_result.thresholds.get("max_gap_minutes") == pytest.approx(45.0)


def test_run_coverage_check_and_alert_skips_when_ok(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache_ok"
    rows = _generate_rows(datetime(2024, 1, 1, tzinfo=timezone.utc), 30)
    _write_cache(cache_dir, rows)
    config_path = _write_config(tmp_path, cache_dir)

    router = DefaultAlertRouter(audit_log=InMemoryAlertAuditLog())
    channel = DummyChannel()
    router.register(channel)

    as_of = datetime.fromisoformat(_last_row_iso(rows))

    report, dispatched = run_coverage_check_and_alert(
        config_path=config_path,
        environment_name="binance_smoke",
        router=router,
        as_of=as_of,
    )

    assert dispatched is False
    assert channel.messages == []
    assert report.threshold_issues == ()
    assert report.summary["status"] == "ok"


class _FakeHTTPResponse:
    def __init__(self, status: int, payload: bytes) -> None:
        self.status = status
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401, ANN001
        return None


class _MutableClock:
    def __init__(self, *, start: datetime | None = None) -> None:
        self._now = start or datetime(2024, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)


def test_telegram_channel_sends_payload(sample_message: AlertMessage) -> None:
    captured: dict[str, request.Request] = {}

    def opener(req: request.Request, *, timeout: float):  # noqa: ANN001
        captured["request"] = req
        return _FakeHTTPResponse(200, b'{"ok": true}')

    channel = TelegramChannel(bot_token="token", chat_id="chat", _opener=opener)
    channel.send(sample_message)

    sent_request = captured["request"]
    assert sent_request.full_url.endswith("/sendMessage")
    body = sent_request.data.decode("utf-8")
    assert "chat" in body
    assert "Limit ryzyka" in body
    assert channel.health_check()["status"] == "ok"


class _FakeSMTP:
    def __init__(self, host: str, port: int, timeout: float) -> None:  # noqa: D401
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sent: list[EmailMessage] = []

    def __enter__(self) -> "_FakeSMTP":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401, ANN001
        return None

    def ehlo(self) -> None:
        return None

    def starttls(self) -> None:
        return None

    def login(self, username: str, password: str) -> None:  # noqa: D401
        self.username = username
        self.password = password

    def send_message(self, message: EmailMessage) -> None:
        self.sent.append(message)


def test_email_channel_builds_message(sample_message: AlertMessage) -> None:
    fake_smtp = _FakeSMTP("localhost", 25, timeout=5)

    def factory(host: str, port: int, timeout: float) -> _FakeSMTP:  # noqa: ANN001
        assert host == "localhost"
        assert port == 25
        assert timeout == 5
        return fake_smtp

    channel = EmailChannel(
        host="localhost",
        port=25,
        from_address="bot@example.com",
        recipients=["user@example.com"],
        username="user",
        password="secret",
        use_tls=True,
        timeout=5,
        _smtp_factory=factory,  # type: ignore[arg-type]
    )

    channel.send(sample_message)

    assert len(fake_smtp.sent) == 1
    email = fake_smtp.sent[0]
    assert email["Subject"] == "[CRITICAL] Limit ryzyka osiągnięty"
    assert "Dzienne straty" in email.get_content()
    assert channel.health_check()["status"] == "ok"


def test_router_throttles_repeated_alerts(sample_message: AlertMessage, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    clock = _MutableClock()
    throttle = AlertThrottle(
        window=timedelta(seconds=120),
        clock=clock,
        exclude_severities=frozenset(),
        exclude_categories=frozenset(),
        max_entries=16,
    )
    audit = InMemoryAlertAuditLog()
    router = DefaultAlertRouter(audit_log=audit, throttle=throttle)
    channel = DummyChannel()
    router.register(channel)

    router.dispatch(sample_message)
    assert len(channel.messages) == 1
    exported = tuple(audit.export())
    assert exported[-1]["channel"] == "dummy"

    router.dispatch(sample_message)
    assert len(channel.messages) == 1  # throttled
    exported = tuple(audit.export())
    assert exported[-1]["channel"] == "__suppressed__"
    assert throttle.remaining_seconds(sample_message) > 0
    assert any("Tłumię powtarzający się alert" in record.message for record in caplog.records)

    clock.advance(180)
    router.dispatch(sample_message)
    assert len(channel.messages) == 2
    exported = tuple(audit.export())
    assert exported[-1]["channel"] == "dummy"


def test_router_records_suppressed_metric() -> None:
    audit = InMemoryAlertAuditLog()
    registry = MetricsRegistry()
    clock = _MutableClock(start=datetime(2024, 1, 1, tzinfo=timezone.utc))
    throttle = AlertThrottle(window=timedelta(seconds=60), clock=clock, exclude_severities=frozenset())
    router = DefaultAlertRouter(
        audit_log=audit,
        throttle=throttle,
        metrics_registry=registry,
        metric_labels={"environment": "paper"},
    )
    channel = DummyChannel()
    router.register(channel)

    message = AlertMessage(
        category="system",
        title="Info",
        body="Powtarzający się alert",
        severity="info",
        context={},
        timestamp=clock(),
    )

    router.dispatch(message)
    clock.advance(10)
    router.dispatch(message)

    suppressed = registry.get("alerts_suppressed_total")
    assert suppressed.value(
        labels={"environment": "paper", "channel": "__suppressed__", "severity": "info"}
    ) == pytest.approx(1.0)


def test_router_health_snapshot_records_metric(sample_message: AlertMessage) -> None:
    audit = InMemoryAlertAuditLog()
    registry = MetricsRegistry()
    router = DefaultAlertRouter(
        audit_log=audit,
        metrics_registry=registry,
        metric_labels={"environment": "paper"},
    )
    channel = HealthFailingChannel()
    router.register(channel)

    router.dispatch(sample_message)
    snapshot = router.health_snapshot()

    assert snapshot[channel.name]["status"] == "error"
    health_errors = registry.get("alert_healthcheck_errors_total")
    assert health_errors.value(labels={"environment": "paper", "channel": channel.name}) == pytest.approx(1.0)


def test_sms_channel_sends_to_all_recipients(sample_message: AlertMessage) -> None:
    captured_requests: list[request.Request] = []

    def opener(req: request.Request, *, timeout: float):  # noqa: ANN001
        captured_requests.append(req)
        return _FakeHTTPResponse(201, b"{}")

    channel = SMSChannel(
        account_sid="AC123",
        auth_token="token",
        from_number="123",
        recipients=("+48111222333", "+3546600000"),
        _opener=opener,
    )

    channel.send(sample_message)

    assert len(captured_requests) == 2
    first_request = captured_requests[0]
    assert first_request.data is not None
    decoded_body = first_request.data.decode("utf-8")
    assert "To=%2B48111222333" in decoded_body
    auth_header = first_request.get_header("Authorization")
    assert auth_header is not None
    decoded_auth = base64.b64decode(auth_header.split()[1]).decode("utf-8")
    assert decoded_auth == "AC123:token"
    assert channel.health_check()["status"] == "ok"


def test_default_sms_providers_cover_regions() -> None:
    required = {"orange_pl", "tmobile_pl", "plus_pl", "play_pl", "nova_is"}
    assert required.issubset(DEFAULT_SMS_PROVIDERS.keys())


def test_sms_channel_uses_provider_metadata(sample_message: AlertMessage) -> None:
    captured = []

    def opener(req: request.Request, *, timeout: float):  # noqa: ANN001
        captured.append(req)
        return _FakeHTTPResponse(200, b"{}")

    provider = get_sms_provider("orange_pl")
    channel = SMSChannel(
        account_sid="AC456",
        auth_token="token",
        from_number="123",
        recipients=("+48555111222",),
        provider=provider,
        _opener=opener,
    )

    channel.send(sample_message)

    assert captured
    assert captured[0].full_url.startswith(provider.api_base_url)
    health = channel.health_check()
    assert health["provider"] == "orange_pl"
    assert health["country"] == "PL"


def test_signal_channel_builds_payload(sample_message: AlertMessage) -> None:
    captured: dict[str, request.Request] = {}

    def opener(req: request.Request, *, timeout: float, context):  # noqa: ANN001
        captured["request"] = req
        assert context is not None
        return _FakeHTTPResponse(200, b"{}")

    channel = SignalChannel(
        service_url="https://signal-gateway.local",
        sender_number="+48500100999",
        recipients=("+48555111222",),
        auth_token="secret",
        _opener=opener,
    )

    channel.send(sample_message)

    req = captured["request"]
    assert req.full_url.endswith("/v2/send")
    body = json.loads(req.data.decode("utf-8"))
    assert body["number"] == "+48500100999"
    assert body["recipients"] == ["+48555111222"]
    assert "Authorization" in req.headers
    assert channel.health_check()["status"] == "ok"


def test_whatsapp_channel_formats_messages(sample_message: AlertMessage) -> None:
    captured: list[request.Request] = []

    def opener(req: request.Request, *, timeout: float):  # noqa: ANN001
        captured.append(req)
        return _FakeHTTPResponse(200, b"{}")

    channel = WhatsAppChannel(
        phone_number_id="10987654321",
        access_token="token",
        recipients=("48555111222", "48555111333"),
        _opener=opener,
    )

    channel.send(sample_message)

    assert len(captured) == 2
    payload = json.loads(captured[0].data.decode("utf-8"))
    assert payload["messaging_product"] == "whatsapp"
    assert payload["to"] == "48555111222"
    assert "Authorization" in captured[0].headers
    assert channel.health_check()["status"] == "ok"


def test_messenger_channel_sends_text(sample_message: AlertMessage) -> None:
    captured: list[request.Request] = []

    def opener(req: request.Request, *, timeout: float):  # noqa: ANN001
        captured.append(req)
        return _FakeHTTPResponse(200, b"{}")

    channel = MessengerChannel(
        page_id="1357924680",
        access_token="token",
        recipients=("2468013579",),
        _opener=opener,
    )

    channel.send(sample_message)

    assert captured
    payload = json.loads(captured[0].data.decode("utf-8"))
    assert payload["recipient"]["id"] == "2468013579"
    assert payload["message"]["text"]
    assert channel.health_check()["status"] == "ok"
