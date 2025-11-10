"""Testy sprawdzające fallback LocalSecurityStore w trybie pamięciowym."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

pytestmark = pytest.mark.qml

PySide6 = pytest.importorskip("PySide6", reason="Wymagany PySide6 do testów LocalSecurityStore")

from PySide6.QtCore import QCoreApplication
from PySide6.QtQml import QJSEngine, QJSValue


@pytest.fixture(scope="module")
def qt_app() -> Iterator[QCoreApplication]:
    """Udostępnia pojedynczą instancję QCoreApplication dla całego modułu testowego."""
    app = QCoreApplication.instance() or QCoreApplication([])
    yield app


@pytest.fixture
def store_engine(qt_app: QCoreApplication) -> Iterator[QJSEngine]:
    engine = QJSEngine()
    source_path = Path("ui/qml/components/security/LocalSecurityStore.js").resolve()
    code = source_path.read_text(encoding="utf-8")
    result = engine.evaluate(code, str(source_path))
    if result.isError():
        pytest.fail(result.toString())
    # wymuszamy tryb pamięciowy, aby uniknąć zależności od SQLite w testach
    _eval(engine, "forceMemoryMode()")
    _eval(engine, "clearAudit()")
    yield engine
    _eval(engine, "clearAudit()")
    _eval(engine, "useDiskStorage()")


def _eval(engine: QJSEngine, expression: str) -> QJSValue:
    value = engine.evaluate(expression)
    if value.isError():
        pytest.fail(value.toString())
    return value


def test_memory_fallback_adds_and_fetches_entries(store_engine: QJSEngine) -> None:
    _eval(store_engine, "addLicenseSnapshot({fingerprint: 'mem-1', licenseId: 'L-1'})")
    assert _eval(store_engine, "isMemoryMode()").toBool() is True
    assert _eval(store_engine, "totalAuditCount()").toInt() == 1
    assert _eval(store_engine, "fetchAudit().length").toInt() == 1
    assert _eval(store_engine, "fetchAudit()[0].fingerprint").toString() == "mem-1"


def test_memory_fallback_trims_to_max_rows(store_engine: QJSEngine) -> None:
    _eval(store_engine, "clearAudit()")
    _eval(
        store_engine,
        "(function() { for (var i = 0; i < 250; ++i) addLicenseSnapshot({fingerprint: 'row-' + i}); })()",
    )
    assert _eval(store_engine, "isMemoryMode()").toBool() is True
    assert _eval(store_engine, "totalAuditCount()").toInt() == 200
    assert _eval(store_engine, "fetchAudit()[0].fingerprint").toString() == "row-249"
    assert _eval(store_engine, "fetchAudit()[fetchAudit().length - 1].fingerprint").toString() == "row-50"


def test_clear_audit_resets_memory_state(store_engine: QJSEngine) -> None:
    _eval(store_engine, "addLicenseSnapshot({fingerprint: 'to-clear'})")
    assert _eval(store_engine, "totalAuditCount()").toInt() >= 1
    _eval(store_engine, "clearAudit()")
    assert _eval(store_engine, "totalAuditCount()").toInt() == 0
    assert _eval(store_engine, "fetchAudit().length").toInt() == 0
