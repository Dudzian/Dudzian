from __future__ import annotations

from tests.ui import conftest as ui_conftest


def test_settle_qt_application_skips_when_pyside_unavailable(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(ui_conftest, "_PYSIDE6_AVAILABLE", False)
    monkeypatch.setattr(ui_conftest, "settle_qt_application", lambda: calls.append("settle"))
    monkeypatch.setenv("DUDZIAN_QML_SETTLE_APP", "1")

    ui_conftest._settle_qt_application_best_effort()

    assert calls == []


def test_settle_qt_application_skips_when_ci_default(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(ui_conftest, "_PYSIDE6_AVAILABLE", True)
    monkeypatch.setattr(ui_conftest, "settle_qt_application", lambda: calls.append("settle"))
    monkeypatch.setenv("CI", "1")
    monkeypatch.delenv("DUDZIAN_QML_SETTLE_APP", raising=False)

    ui_conftest._settle_qt_application_best_effort()

    assert calls == []


def test_settle_qt_application_runs_when_enabled_in_ci(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(ui_conftest, "_PYSIDE6_AVAILABLE", True)
    monkeypatch.setattr(ui_conftest, "settle_qt_application", lambda: calls.append("settle"))
    monkeypatch.setenv("CI", "1")
    monkeypatch.setenv("DUDZIAN_QML_SETTLE_APP", "1")

    ui_conftest._settle_qt_application_best_effort()

    assert calls == ["settle"]


def test_settle_qt_application_runs_by_default_outside_ci(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(ui_conftest, "_PYSIDE6_AVAILABLE", True)
    monkeypatch.setattr(ui_conftest, "settle_qt_application", lambda: calls.append("settle"))
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("DUDZIAN_QML_SETTLE_APP", raising=False)

    ui_conftest._settle_qt_application_best_effort()

    assert calls == ["settle"]
