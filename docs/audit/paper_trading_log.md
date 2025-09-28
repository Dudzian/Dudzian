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
