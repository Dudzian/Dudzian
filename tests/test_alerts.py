"""Testy modułu alertów."""
from __future__ import annotations

import base64
import json
from email.message import EmailMessage
from pathlib import Path
from typing import List
from urllib import request

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from bot_core.alerts import (
    AlertDeliveryError,
    AlertMessage,
    DefaultAlertRouter,
    EmailChannel,
    InMemoryAlertAuditLog,
    MessengerChannel,
    SMSChannel,
    SignalChannel,
    TelegramChannel,
    WhatsAppChannel,
    DEFAULT_SMS_PROVIDERS,
    get_sms_provider,
)
from bot_core.alerts.base import AlertChannel


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


def test_telegram_channel_sends_payload(sample_message: AlertMessage) -> None:
    captured: dict[str, request.Request] = {}

    def opener(req: request.Request, *, timeout: float):  # noqa: ANN001
        captured["request"] = req
        return _FakeHTTPResponse(200, b"{\"ok\": true}")

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
