import io
import json
import sys
import types

import pytest

from bot_core.runtime import local_gateway
from bot_core.security.cloud_flag import CloudFlagValidationError


def test_main_emits_error_when_config_missing(monkeypatch, capfd):
    def fake_context_builder(*_args, **_kwargs):
        raise FileNotFoundError("missing config/runtime.yaml")

    monkeypatch.setattr(local_gateway, "build_local_runtime_context", fake_context_builder)

    exit_code = local_gateway.main(["--config", "config/runtime.yaml"])
    captured = capfd.readouterr().out.strip().splitlines()

    assert exit_code == 2
    payload = json.loads(captured[-1])
    assert payload == {
        "event": "error",
        "code": 2,
        "message": "Nie udało się zbudować kontekstu runtime",
        "details": {"reason": "missing config/runtime.yaml"},
    }


def test_main_emits_error_when_cloud_flag_invalid(monkeypatch, capfd):
    def fake_validate(_path):
        raise CloudFlagValidationError("invalid signature")

    monkeypatch.setattr(local_gateway, "validate_runtime_cloud_flag", fake_validate)

    exit_code = local_gateway.main(["--enable-cloud-runtime"])
    captured = capfd.readouterr().out.strip().splitlines()

    assert exit_code == 4
    payload = json.loads(captured[-1])
    assert payload["event"] == "error"
    assert payload["code"] == 4
    assert "Walidacja flagi" in payload["message"]
    assert payload["details"]["reason"] == "invalid signature"


def test_main_emits_ready_and_processes_requests(monkeypatch, capfd):
    class DummyContext:
        version = "1.2.3"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyGateway:
        def __init__(self, _context):
            self.calls: list[tuple[str, dict[str, object]]] = []

        def dispatch(self, method: str, params: dict[str, object]):
            self.calls.append((method, params))
            return {"echo": params.get("value")}

    monkeypatch.setattr(local_gateway, "build_local_runtime_context", lambda **_kwargs: DummyContext())
    monkeypatch.setattr(local_gateway, "LocalRuntimeGateway", DummyGateway)

    stdin = io.StringIO('{"id":1,"method":"echo","params":{"value":42}}\n')
    monkeypatch.setattr(
        local_gateway,
        "sys",
        types.SimpleNamespace(stdin=stdin, stdout=sys.stdout, stderr=sys.stderr),
    )

    exit_code = local_gateway.main(["--config", "config/runtime.yaml"])
    captured_lines = capfd.readouterr().out.strip().splitlines()

    assert exit_code == 0
    ready_payload = json.loads(captured_lines[0])
    assert ready_payload == {"event": "ready", "version": "1.2.3"}

    response_payload = json.loads(captured_lines[1])
    assert response_payload == {"id": 1, "result": {"echo": 42}}
