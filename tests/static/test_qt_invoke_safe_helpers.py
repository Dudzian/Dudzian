from __future__ import annotations

from tests.ui import _qt_invoke_safe as safe


def test_invoke_safe_qvariantmap_returns_none_on_win32(monkeypatch):
    monkeypatch.setattr(safe.sys, "platform", "win32")
    assert safe.invoke_safe_qvariantmap({"a": 1}) is None


def test_invoke_safe_qvariantmap_passthrough_on_non_win32(monkeypatch):
    monkeypatch.setattr(safe.sys, "platform", "linux")
    payload = {"a": 1}
    assert safe.invoke_safe_qvariantmap(payload) is payload
