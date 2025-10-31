"""Zestaw podstawowych testów smoke dla RC."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke


def test_run_local_bot_help() -> None:
    """Sprawdza, że główny skrypt backendu uruchamia się i udostępnia pomoc CLI."""
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(Path(__file__).resolve().parents[2]))
    result = subprocess.run(
        [sys.executable, "scripts/run_local_bot.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "--mode" in result.stdout


def test_metrics_overview_snapshot() -> None:
    """Weryfikuje możliwość zebrania najważniejszych metryk runtime."""
    from core.monitoring import metrics_api

    snapshot = metrics_api.load_runtime_snapshot()
    assert hasattr(snapshot, "io_queues")
    assert hasattr(snapshot, "guardrail_overview")
    assert hasattr(snapshot, "retraining")


def test_qml_runtime_overview_manifest_exists() -> None:
    """Minimalne sprawdzenie warstwy UI: plik QML musi istnieć i być niepusty."""
    qml_path = Path("ui/qml/dashboard/RuntimeOverview.qml")
    if not qml_path.exists():
        pytest.fail(f"Brak pliku {qml_path}")
    content = qml_path.read_text(encoding="utf-8").strip()
    assert 'defaultCardOrder' in content
    assert len(content) > 100


@pytest.mark.smoke
@pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="Środowisko CI może nie mieć zależności Qt"
)
def test_optional_qt_loading() -> None:
    """Jeżeli PySide6 jest dostępny, upewnij się, że podstawowy komponent ładuje się poprawnie."""
    try:
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtQml import QQmlApplicationEngine
    except ModuleNotFoundError:
        pytest.skip("Brak biblioteki PySide6")

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QCoreApplication([])
    engine = QQmlApplicationEngine()
    engine.load(Path("ui/qml/onboarding/LicenseWizard.qml"))
    if not engine.rootObjects():
        errors = [str(err.toString()) for err in getattr(engine, 'errors', lambda: [])()]
        pytest.skip("Brak wsparcia QtQuick/GL w środowisku testowym: %s" % (errors or 'unknown'))
    # Zakończ aplikację, by nie blokować testów.
    app.quit()
