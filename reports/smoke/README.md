# Raporty testów smoke

Folder przechowuje artefakty generowane przez `scripts/run_smoke_tests.py`.  Każde uruchomienie
zapisuje parę plików JSON/Markdown o nazwie `smoke_report_<timestamp>.(json|md)`.

## Struktura raportu JSON

```json
{
  "tag": "rc-0.1.0-202501011200",
  "exit_status": 0,
  "tests_collected": 4,
  "passed": 3,
  "skipped": 1,
  "failed": 0,
  "duration_seconds": 12.5
}
```

- `tag` – nazwa tagu RC, dla którego wykonano smoke testy.
- `exit_status` – kod zakończenia pytest (`0` oznacza sukces).
- `tests_collected`, `passed`, `failed`, `skipped` – podsumowanie wyników.
- `duration_seconds` – czas wykonania zestawu smoke.

Raport Markdown zawiera te same informacje w formie tabeli, aby można było go
wprost dołączyć do notatek release.
