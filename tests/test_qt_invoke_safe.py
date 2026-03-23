from __future__ import annotations

import pytest

from tests.ui._qt_invoke_safe import assert_has_any_overload, assert_has_overload, has_overload


class _FakeMethod:
    def __init__(self, signature):
        self._signature = signature

    def methodSignature(self):
        return self._signature


class _FakeMeta:
    def __init__(self, signatures):
        self._methods = [_FakeMethod(sig) for sig in signatures]

    def indexOfMethod(self, signature):
        # Symulacja API, które przy bytes rzuca TypeError.
        if not isinstance(signature, str):
            raise TypeError("signature must be str")
        for idx, method in enumerate(self._methods):
            raw = method.methodSignature()
            text = raw.decode(errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
            if text == signature:
                return idx
        return -1

    def methodCount(self):
        return len(self._methods)

    def method(self, idx):
        return self._methods[idx]


class _FakeQObject:
    def __init__(self, signatures):
        self._meta = _FakeMeta(signatures)

    def metaObject(self):
        return self._meta


def test_assert_has_overload_accepts_bytes_signatures() -> None:
    qobj = _FakeQObject([b"foo(QString)", b"bar()"])
    assert_has_overload(qobj, "foo(QString)")


def test_assert_has_overload_reports_diagnostics() -> None:
    qobj = _FakeQObject([b"foo(QString)", "foo(int)"])
    with pytest.raises(AssertionError) as excinfo:
        assert_has_overload(qobj, "foo(QVariant)")
    message = str(excinfo.value)
    assert "target='foo(QVariant)'" in message
    assert "candidates=['foo(QString)', 'foo(int)']" in message
    assert "platform=" in message
    assert "binding=PySide6/" in message


def test_has_overload_reports_presence_without_raising() -> None:
    qobj = _FakeQObject([b"foo(QString)", "bar()"])
    assert has_overload(qobj, "foo(QString)") is True
    assert has_overload(qobj, "foo(QVariant)") is False


def test_assert_has_any_overload_accepts_matching_variant() -> None:
    qobj = _FakeQObject([b"foo(QString)", "bar()"])
    assert_has_any_overload(qobj, "foo(QVariant)", "foo(QString)")


def test_assert_has_any_overload_reports_candidates_when_missing() -> None:
    qobj = _FakeQObject([b"foo(QString)", "foo(int)"])
    with pytest.raises(AssertionError) as excinfo:
        assert_has_any_overload(qobj, "foo(QVariant)", "foo(bool)")
    message = str(excinfo.value)
    assert "signatures=('foo(QVariant)', 'foo(bool)')" in message
    assert "candidates=['foo(QString)', 'foo(int)']" in message
    assert "binding=PySide6/" in message
