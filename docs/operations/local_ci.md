# Lokalny pipeline CI

Dokument opisuje sposób uruchomienia scenariusza demo → paper na stacji roboczej.

## Wymagania wstępne
- Python 3.11 wraz z wirtualnym środowiskiem.
- Zainstalowane zależności projektu (`pip install -e .[dev,compression]`, co doinstaluje
  `brotli` lub `brotlicffi` oraz `zstandard` w zależności od platformy).
- Dostęp do repozytorium z aktualnym kodem bota.

## Uruchomienie
1. Aktywuj środowisko wirtualne i przejdź do katalogu projektu.
2. Uruchom pipeline:
   ```bash
   python -m pip install --upgrade pip
   pip install -e .[dev,compression]
   pytest -m e2e_demo_paper --maxfail=1 --disable-warnings
   pytest -m retraining tests/reporting/test_retraining_reporter.py --maxfail=1 --disable-warnings
   ```
3. Raporty JSON oraz Markdown znajdziesz w `reports/e2e/`. Artefakty logów testu umieszczane są w `logs/e2e/`.
4. Raporty z retrainingu (Markdown i JSON) są zapisywane w `reports/retraining/`.

## Artefakty
- `reports/e2e/` – raporty generowane przez `DemoPaperReport` (JSON + Markdown).
- `reports/retraining/` – raporty generowane przez `RetrainingReport` (JSON + Markdown).
- `logs/e2e/` – logi pomocnicze oraz checkpointy tworzone podczas przebiegu scenariusza.

## Runbook
1. Jeśli test zakończy się niepowodzeniem, sprawdź najnowszy raport w `reports/e2e/` i komunikaty błędów w logach.
2. Usuń zawartość katalogów `logs/e2e/` oraz `reports/e2e/`, jeżeli chcesz wykonać scenariusz ponownie od zera.
3. Po wprowadzeniu poprawek uruchom ponownie polecenia `pytest -m e2e_demo_paper` oraz `pytest -m retraining tests/reporting/test_retraining_reporter.py`.
