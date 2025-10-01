# Log audytu – Paper trading (Etap 1)

Dokument stanowi append-only rejestr zdarzeń operacyjnych związanych z trybem paper tradingu. Każdy wpis posiada unikalny identyfikator, znacznik czasu w UTC (ISO 8601, `Z`), osobę odpowiedzialną oraz opis kontekstu. Plik przechowujemy pod kontrolą wersji oraz w zaszyfrowanych archiwach (retencja min. 24 miesiące).

## Sekcja A – Uruchomienia sesji
| ID | Data (UTC) | Operator | Środowisko | Profil ryzyka | Commit hash | Uwagi |
|----|------------|----------|------------|---------------|-------------|-------|
| A-0001 | YYYY-MM-DDThh:mm:ssZ | imię nazwisko | paper_binance | balanced | `<git-hash>` | „Start pierwszej sesji testowej” |

## Sekcja B – Raporty dzienne
| ID | Data (UTC) | Operator | Artefakt | Hash SHA-256 | Retencja do | Uwagi |
|----|------------|----------|----------|--------------|-------------|-------|
| R-0001 | YYYY-MM-DDThh:mm:ssZ | imię nazwisko | `data/reports/daily/2023-03-31/paper_binance.zip.age` | `<hash>` | 2025-03-31 | „Raport testowy” |

## Sekcja B1 – Smoke testy paper tradingu
| ID | Data (UTC) | Operator | Środowisko | Zakres dat | Raport (`summary.json`) | Hash SHA-256 | Status alertów | Uwagi |
|----|------------|----------|------------|------------|-------------------------|--------------|----------------|-------|
| S-TEST-0001 | YYYY-MM-DDThh:mm:ssZ | imię nazwisko | binance_paper | 2024-01-01 → 2024-02-15 | `/tmp/daily_trend_smoke_xxx/summary.json` | `<hash>` | OK | „Smoke test sanity” |
 | S-0001 | 2025-09-30T16:53:03Z | CI Agent | binance_paper | 2024-01-01 → 2024-02-15 | `n/a` | `n/a` | ERROR (403 Binance API) | Smoke test przerwany – brak dostępu do API Binance (403 Forbidden) |
| S-0002 | 2025-09-30T17:19:30Z | CI Agent | binance_paper | 2024-01-01 → 2024-02-15 | `/tmp/daily_trend_smoke_jk_rha7g/summary.json` | `c694ac951e24fb214fe8b454b4abb9582d94f59e25ed05f697035d2bff713f87` | WARN (alert channels) | Smoke test ukończony na cache offline; wysyłka alertów nieudana (403 Telegram, DNS e-mail). |
## Sekcja C – Incydenty i alerty krytyczne
| ID | Data (UTC) | Operator | Kod alertu | Opis | Działanie naprawcze | Status |
|----|------------|----------|------------|------|---------------------|--------|
| I-0001 | YYYY-MM-DDThh:mm:ssZ | imię nazwisko | `RISK-DAILY-LIMIT` | „Przekroczony limit dzienny w paper – sanity test” | „Weryfikacja logów, potwierdzona likwidacja pozycji” | Zamknięty |

## Sekcja D – Rotacja kluczy i bezpieczeństwo
| ID | Data (UTC) | Operator | Środowisko | Zakres | Hash potwierdzenia | Uwagi |
|----|------------|----------|------------|--------|--------------------|-------|
| S-0001 | YYYY-MM-DDThh:mm:ssZ | imię nazwisko | paper_binance | Rotacja kluczy trade/read | `<hash raportu>` | „Procedura bez-downtime zakończona” |

## Sekcja E – Zmiany konfiguracji
| ID | Data (UTC) | Operator | Element | Poprzednia wartość | Nowa wartość | Uzasadnienie |
|----|------------|----------|---------|-------------------|--------------|--------------|
| C-0001 | YYYY-MM-DDThh:mm:ssZ | imię nazwisko | `strategies.daily_trend.momentum.window_fast` | 20 | 30 | „Optymalizacja walk-forward” |

> **Instrukcja aktualizacji:** dodawaj nowe wiersze na końcu każdej sekcji, zachowując rosnącą numerację ID. Nie modyfikuj historycznych wpisów – w razie pomyłki dodaj nowy wiersz korygujący z referencją do oryginalnego ID.
> Dla smoke testów kopiuj zarówno hash `summary.json`, jak i treść pliku `summary.txt` (skrót można dodać w kolumnie „Uwagi”). Jeśli skrypt utworzył archiwum ZIP (`--archive-smoke`), dopisz ścieżkę pliku w kolumnie „Uwagi” i zabezpiecz archiwum w sejfie audytu. W razie wykonywania weryfikacji `scripts/check_data_coverage.py` dopisz status (`ok` / `error`) do kolumny „Uwagi”.
> Gdy konfiguracja `reporting.smoke_archive_upload` wykona dodatkowy upload (np. do `audit/smoke_archives/` lub koszyka S3), dopisz docelową lokalizację z pola alertu `archive_upload_location` – ułatwi to odtworzenie pakietu podczas audytu.
