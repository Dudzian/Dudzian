# Runbook: Testy QML i diagnostyka UI

## Cel
Runbook opisuje sposób uruchamiania i analizowania testów QML dla aplikacji desktopowej bota handlowego. Etap CI `ui-tests` uruchamia pełną suite testów oznaczonych markerem `qml`, generując artefakty (logi Qt, logi runtime oraz zrzuty ekranu) w katalogu `test-results/qml`.

## Wymagania środowiskowe
- Python 3.11
- Pakiety `PySide6`, `PySide6_Addons`, `PySide6_Essentials`, `shiboken6` w wersji zgodnej z `env.PYSIDE6_VERSION`
- Zależności projektu (`pip install .[dev]`)
- Zmienna `QT_QPA_PLATFORM=offscreen` oraz `QT_QUICK_BACKEND=software` (ustawiane automatycznie przez fixture `configure_qt_environment`)
- Opcjonalnie `PYTEST_REQUIRE_QML=1`, aby wymusić błędy przy braku PySide6

## Lokalne uruchomienie
```bash
export QT_QPA_PLATFORM=offscreen
export QT_QUICK_BACKEND=software
pytest -m qml --maxfail=1 --disable-warnings \
  --junitxml=test-results/qml/pytest.xml
```

Aby wygenerować pełen zestaw diagnostyczny:
```bash
export PYTEST_REQUIRE_QML=1
export QML_DIAGNOSTICS_DIR=test-results/qml
pytest -m qml
```

## Artefakty diagnostyczne
Fixture `capture_qml_artifacts` zapisuje:
- Logi Qt (`test-results/qml/logs/<test>.log`)
- Logi Pythonowego modułu `ui.backend.runtime_service` (`*.runtime.log`)
- Zrzuty ekranu (`test-results/qml/screenshots/<test>_*.png` lub placeholder `.txt` przy braku okien`)
- Manifest JSON (`test-results/qml/manifest.json`) z listą testów i statusem

Artefakty są publikowane przez job CI `ui-tests` jako paczka `qml-test-diagnostics`.

## Debugowanie awarii
1. Pobierz artefakt z GitHub Actions (`qml-test-diagnostics`).
2. Sprawdź log odpowiadający testowi – sekcje `[PYTHON]` zawierają logi runtime, `[PYTEST]` status raportu.
3. Przejrzyj zrzut ekranu (`.png`); plik `.txt` oznacza brak okien lub problemy z renderowaniem.
4. W razie błędów feedu sprawdź `decision-feed-metrics.json` z joba `grpc-feed-integration`.

## Integracja z innymi narzędziami
- Runbook `docs/runtime/status_review.md` opisuje metryki feedu – testy QML zapisują zrzuty i logi w tym samym katalogu.
- Benchmark `docs/benchmark/cryptohopper_comparison.md` wymaga załączenia artefaktu `decision-feed-metrics`; QML runbook wskazuje ścieżki diagnostyczne UI.

