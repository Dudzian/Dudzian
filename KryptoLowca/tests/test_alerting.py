from __future__ import annotations

from typing import Any, Dict, List

import pytest

from bot_core.alerts import AlertEvent, AlertSeverity, emit_alert, get_alert_dispatcher
from KryptoLowca.services.alerting import AlertManager, EmailAlertSink, SlackWebhookSink, WebhookAlertSink


class InMemorySink:
    def __init__(self) -> None:
        self.events: List[AlertEvent] = []

    def send(self, event: AlertEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        self.events.clear()


def test_alert_manager_dispatches_events(monkeypatch: pytest.MonkeyPatch) -> None:
    dispatcher = get_alert_dispatcher()
    dispatcher.clear()
    sink = InMemorySink()
    manager = AlertManager([sink])
    try:
        emit_alert("Test alert", severity=AlertSeverity.WARNING, source="unit")
        assert sink.events, "Alert powinien trafić do sinka"
    finally:
        manager.close()
        dispatcher.clear()


def test_email_sink_uses_smtp(monkeypatch: pytest.MonkeyPatch) -> None:
    sent_messages: List[Dict[str, Any]] = []

    class FakeSMTP:
        def __init__(self, host: str, port: int, timeout: float = 10.0) -> None:
            self.host = host
            self.port = port
            self.timeout = timeout

        def __enter__(self) -> "FakeSMTP":
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def starttls(self, context: Any | None = None) -> None:  # pragma: no cover - brak logiki
            return None

        def login(self, username: str, password: str) -> None:
            sent_messages.append({"action": "login", "username": username, "password": password})

        def send_message(self, msg: Any) -> None:
            sent_messages.append({"action": "send", "subject": msg["Subject"], "to": msg["To"]})

    monkeypatch.setattr("smtplib.SMTP", FakeSMTP)

    sink = EmailAlertSink(
        host="smtp.test",
        port=587,
        username="user",
        password="secret",
        sender="bot@test",
        recipients=["owner@test"],
    )
    event = AlertEvent(message="Boom", severity=AlertSeverity.ERROR, source="risk")
    sink.send(event)

    actions = [entry["action"] for entry in sent_messages]
    assert "login" in actions and "send" in actions


def test_slack_and_webhook_sinks_post(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[Dict[str, Any]] = []

    def fake_post(url: str, json: Dict[str, Any], timeout: float, headers: Dict[str, str] | None = None):
        calls.append({"url": url, "json": json, "timeout": timeout, "headers": headers or {}})

        class Response:
            def raise_for_status(self) -> None:
                return None

        return Response()

    monkeypatch.setattr("requests.post", fake_post)

    slack_sink = SlackWebhookSink(webhook_url="https://hooks.slack.com/test")
    webhook_sink = WebhookAlertSink(url="https://example.com/alert", headers={"X-Test": "1"})
    event = AlertEvent(message="Limit exceeded", severity=AlertSeverity.CRITICAL, source="exchange")
    slack_sink.send(event)
    webhook_sink.send(event)

    assert any(call["url"].startswith("https://hooks.slack.com") for call in calls)
    assert any(call["headers"].get("X-Test") == "1" for call in calls)
