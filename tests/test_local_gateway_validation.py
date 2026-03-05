from __future__ import annotations

import json
from typing import Any

import pytest

from bot_core.runtime.local_gateway import _GatewayMetrics, _process_request, _TokenBucket


class _DummyGateway:
    def __init__(self, result: Any = "ok", raise_exc: Exception | None = None) -> None:
        self.result = result
        self.raise_exc = raise_exc
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def dispatch(self, method: str, params: dict[str, Any]) -> Any:
        self.calls.append((method, params))
        if self.raise_exc:
            raise self.raise_exc
        return self.result


def _make_basic_components(result: Any = "ok", raise_exc: Exception | None = None):
    gateway = _DummyGateway(result=result, raise_exc=raise_exc)
    metrics = _GatewayMetrics()
    limiter = _TokenBucket(capacity=1, refill_rate_per_sec=100)
    return gateway, metrics, limiter


def test_invalid_json_returns_error():
    gateway, metrics, limiter = _make_basic_components()
    response = _process_request("{not-json}", gateway, metrics, limiter)

    assert response["error"]["code"] == "invalid-json"
    assert "Expecting" in response["error"]["message"]
    assert metrics.invalid_json == 1


def test_non_dict_payload_rejected():
    gateway, metrics, limiter = _make_basic_components()
    response = _process_request(json.dumps([1, 2, 3]), gateway, metrics, limiter)

    assert response == {
        "id": None,
        "error": {"code": "invalid-request", "message": "Payload musi być obiektem JSON"},
    }
    assert metrics.invalid_request == 1


def test_missing_method_validation_error():
    gateway, metrics, limiter = _make_basic_components()
    response = _process_request(json.dumps({"id": 1, "params": {}}), gateway, metrics, limiter)

    assert response["error"]["code"] == "invalid-request"
    assert response["error"]["message"] == "Field required"
    assert metrics.invalid_request == 1


def test_rate_limit_blocks_requests():
    gateway, metrics, limiter = _make_basic_components()
    first = _process_request(json.dumps({"id": 1, "method": "ping"}), gateway, metrics, limiter)
    second = _process_request(json.dumps({"id": 2, "method": "ping"}), gateway, metrics, limiter)

    assert first == {"id": 1, "result": "ok"}
    assert second == {
        "id": 2,
        "error": {"code": "rate-limit", "message": "Przekroczono limit zapytań"},
    }
    assert metrics.rate_limited == 1


def test_dispatch_error_propagates_as_error_payload():
    gateway, metrics, limiter = _make_basic_components(raise_exc=RuntimeError("boom"))
    response = _process_request(
        json.dumps({"id": "x", "method": "fail"}), gateway, metrics, limiter
    )

    assert response == {"id": "x", "error": {"code": "dispatch-error", "message": "boom"}}
    assert metrics.dispatch_errors == 1
