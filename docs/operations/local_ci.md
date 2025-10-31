# Lokalny pipeline CI

Dokument opisuje sposób uruchomienia scenariusza demo → paper na stacji roboczej.

## Wymagania wstępne
- Python 3.11 wraz z wirtualnym środowiskiem.
- Zainstalowane zależności projektu (`pip install -e .[dev]`).
- Dostęp do repozytorium z aktualnym kodem bota.

## Uruchomienie
1. Aktywuj środowisko wirtualne i przejdź do katalogu projektu.
2. Uruchom pipeline:
   ```bash
   python -m pip install --upgrade pip
   pip install -e .[dev]
   pytest -m e2e_demo_paper --maxfail=1 --disable-warnings
   ```
3. Raporty JSON oraz Markdown znajdziesz w `reports/e2e/`. Artefakty logów testu umieszczane są w `logs/e2e/`.

## Artefakty
- `reports/e2e/` – raporty generowane przez `DemoPaperReport` (JSON + Markdown).
- `logs/e2e/` – logi pomocnicze oraz checkpointy tworzone podczas przebiegu scenariusza.

## Runbook
1. Jeśli test zakończy się niepowodzeniem, sprawdź najnowszy raport w `reports/e2e/` i komunikaty błędów w logach.
2. Usuń zawartość katalogów `logs/e2e/` oraz `reports/e2e/`, jeżeli chcesz wykonać scenariusz ponownie od zera.
3. Po wprowadzeniu poprawek uruchom ponownie polecenie `pytest -m e2e_demo_paper`.
