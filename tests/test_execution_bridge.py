from __future__ import annotations

from types import SimpleNamespace

from bot_core.execution.bridge import decision_to_order_request


def test_decision_to_order_request_prefers_payload_client_order_id_when_source_blank() -> None:
    decision = {
        "symbol": "BTCUSDT",
        "action": "buy",
        "quantity": 1.0,
        "client_order_id": "abc",
        "candidate": {
            "symbol": "BTCUSDT",
            "action": "buy",
            "quantity": 1.0,
            "client_order_id": "",
        },
    }

    request = decision_to_order_request(decision)

    assert request.client_order_id == "abc"




def test_decision_to_order_request_prefers_payload_client_order_id_when_source_whitespace() -> None:
    decision = {
        "symbol": "BTCUSDT",
        "action": "buy",
        "quantity": 1.0,
        "client_order_id": "abc",
        "candidate": {
            "symbol": "BTCUSDT",
            "action": "buy",
            "quantity": 1.0,
            "client_order_id": "   ",
        },
    }

    request = decision_to_order_request(decision)

    assert request.client_order_id == "abc"

def test_decision_to_order_request_prefers_source_client_order_id_when_present() -> None:
    decision = {
        "symbol": "BTCUSDT",
        "action": "buy",
        "quantity": 1.0,
        "client_order_id": "abc",
        "candidate": {
            "symbol": "BTCUSDT",
            "action": "buy",
            "quantity": 1.0,
            "client_order_id": "xyz",
        },
    }

    request = decision_to_order_request(decision)

    assert request.client_order_id == "xyz"


def test_decision_to_order_request_generates_client_order_id_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "bot_core.execution.bridge.uuid.uuid4",
        lambda: SimpleNamespace(hex="fixedid"),
    )
    decision = {
        "symbol": "BTCUSDT",
        "action": "buy",
        "quantity": 1.0,
        "candidate": {
            "symbol": "BTCUSDT",
            "action": "buy",
            "quantity": 1.0,
        },
    }

    request = decision_to_order_request(decision)

    assert request.client_order_id == "svc-fixedid"
    assert request.metadata is not None
    assert request.metadata.get("client_order_id") == "svc-fixedid"
