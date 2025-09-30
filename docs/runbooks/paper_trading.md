# Runbook: Paper trading – Etap 1

Ten runbook opisuje, jak uruchomić, monitorować i bezpiecznie zatrzymać tryb paper tradingu dla strategii trend-following na interwale dziennym (D1). Dokument przeznaczony jest dla operatorów i analityków, którzy realizują proces backtest → paper → ograniczony live. Wszystkie kroki wykonujemy na kontach demo/testnet z kluczami o minimalnych uprawnieniach (brak wypłat).

## 1. Prerekwizyty operacyjne

### Środowisko i konfiguracja
- System operacyjny: Windows 10/11 (primary) lub macOS 13+ (fallback). W obu przypadkach wymagany jest Python 3.11 oraz dostęp do Windows Credential Manager / macOS Keychain.
- Repozytorium `Dudzian` w wersji zgodnej z `phase1_foundation` (aktualne testy `pytest` przechodzą bez błędów).
- Pliki konfiguracyjne `config/core.yaml` oraz `config/credentials/` dopasowane do środowiska `paper`.
- Katalog roboczy danych: `data/ohlcv` (Parquet + manifest SQLite) oraz `data/reports` na raporty dzienne.

### Klucze API i bezpieczeństwo
- Klucze `read-only` oraz `trade` dla Binance Testnet (spot i futures) zapisane w natywnym keychainie z etykietami odpowiadającymi polom `credential_purpose`.
- Jeśli posiadamy klucze Zondy paper lub Kraken demo, również zapisujemy je w keychainie – jednak w Etapie 1 aktywujemy tylko Binance.
- Lista dozwolonych IP: statyczny adres wyjściowy VPN + awaryjny adres stacji roboczej (paper). Potwierdź w panelu giełdy, że adresy są aktywne.
- Rejestr rotacji kluczy (`security/rotation_log.json`) uzupełniony o datę ostatniej wymiany; kolejne przypomnienie ustawiamy na 90 dni od tej daty.

### Dane historyczne
- Wykonany backfill OHLCV (D1 + 1h) dla koszyka: BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, XRP/USDT, ADA/USDT, LTC/USDT, MATIC/USDT.
- Dane zapisane w strukturze partycjonowanej Parquet: `exchange/symbol/granularity/year=YYYY/month=MM/`.
- Manifest SQLite (`ohlcv_manifest.sqlite`) zawiera zaktualizowane liczniki świec i ostatnie znaczniki czasu.
- **Tryb offline (brak dostępu do API):** uruchom `PYTHONPATH=. python scripts/seed_paper_cache.py --environment binance_paper --days 60 --start-date 2024-01-01`, aby wygenerować deterministyczny cache D1 w katalogu `var/data/binance_paper`.

## 2. Checklista przed startem sesji
1. **Weryfikacja kodu i konfiguracji**
   - `git status` – brak lokalnych, niezatwierdzonych zmian.
   - `python scripts/validate_config.py --config config/core.yaml` – potwierdzenie spójności sekcji środowisk, profili ryzyka i kanałów alertowych.
   - `pytest --override-ini=addopts= tests/test_runtime_pipeline.py` – potwierdzenie, że pipeline przechodzi testy integracyjne.
   - `scripts/check_key_rotation.py --dry-run` – upewnij się, że rotacja kluczy nie jest przeterminowana.
2. **Aktualizacja danych**
   - Uruchom `scripts/backfill.py --environment paper --granularity 1d --since 2016-01-01`.
   - Sprawdź logi (`logs/backfill.log`) pod kątem błędów; w razie limitów API powtórz z większym interwałem throttlingu.
3. **Konfiguracja środowiska**
   - Plik `config/core.yaml` ma aktywne środowisko `paper_binance` i profil ryzyka `balanced` (domyślny).
   - `config/alerts.yaml` (jeśli używany) zawiera aktywne kanały Telegram + e-mail + SMS (Orange jako operator referencyjny).
4. **Alerty i health-checki**
   - Wyślij wiadomość testową: `python scripts/send_alert.py --channel telegram --message "Paper trading start test"`.
   - Zweryfikuj, że alert pojawił się w Telegramie oraz w logu audytu (`logs/alerts_audit.jsonl`).
5. **Raporty i przestrzeń dyskowa**
   - Upewnij się, że katalog `data/reports/daily` posiada co najmniej 2 GB wolnego miejsca.
   - Potwierdź istnienie katalogu `logs/decision_journal` z retencją < 30 dni (starsze pliki powinny być archiwizowane automatycznie).

## 3. Uruchomienie pipeline’u paper trading
1. Aktywuj wirtualne środowisko Pythona: `py -3.11 -m venv .venv && .venv\Scripts\activate` (Windows) lub `python3 -m venv .venv && source .venv/bin/activate` (macOS).
2. Zainstaluj zależności: `pip install -e .[dev]` (pierwsze uruchomienie) lub `pip install -e .` dla aktualizacji.
3. Jeśli łączność z API Binance jest ograniczona, w pierwszej kolejności odtwórz cache poleceniem `PYTHONPATH=. python scripts/seed_paper_cache.py --environment binance_paper --days 60 --start-date 2024-01-01`. Następnie wykonaj smoke test środowiska paper (sprawdzenie backfillu + egzekucji na krótkim oknie):
   ```bash
   PYTHONPATH=. python scripts/run_daily_trend.py \
       --config config/core.yaml \
       --environment binance_paper \
       --paper-smoke \
       --archive-smoke \
       --date-window 2024-01-01:2024-02-15 \
       --run-once
   ```
   - Narzędzie wykona backfill ograniczony do podanego zakresu, uruchomi pojedynczą iterację i zapisze raport tymczasowy (`ledger.jsonl`, `summary.json`, `summary.txt`, `README.txt`).
    - `summary.txt` zawiera gotowe podsumowanie dla zespołu ryzyka (środowisko, okno dat, liczba zleceń, status kanałów alertowych, hash `summary.json`).
    - `README.txt` zawiera skróconą instrukcję audytu (co przepisać do logu, gdzie przechowywać ledger, jak długo archiwizować paczkę).
    - W trybie `--paper-smoke` skrypt podmienia adapter giełdowy na tryb offline i korzysta wyłącznie z lokalnego cache Parquet/SQLite, dlatego przed uruchomieniem wymagany jest kompletny seed danych z kroku wcześniejszego.
    - Jeżeli w keychainie znajdują się placeholderowe tokeny kanałów alertowych, komunikaty o błędach (403/DNS) zostaną zarejestrowane w logu, ale nie przerywają smoke testu; status kanałów należy odnotować w audycie.
   - Użycie flagi `--archive-smoke` tworzy dodatkowo archiwum ZIP z kompletem plików i instrukcją audytu. Ścieżka katalogu raportu i hash `summary.json` pojawią się w logu. Skopiuj hash, treść `summary.txt` oraz status kanałów alertowych do `docs/audit/paper_trading_log.md`. Archiwum ZIP przechowuj w sejfie audytu (retencja ≥ 24 miesiące).
   - Jeśli w `config/core.yaml` skonfigurowano sekcję `reporting.smoke_archive_upload`, CLI automatycznie wykona kopię archiwum (domyślnie do `audit/smoke_archives/`) oraz – w przypadku backendu S3/MinIO – prześle plik do zdefiniowanego koszyka. Ścieżka docelowa trafia do logu oraz kontekstu alertu w polach `archive_upload_backend` i `archive_upload_location`. Zweryfikuj obecność pliku w magazynie i odnotuj lokalizację w audycie.
4. Uruchom tryb jednorazowy (dry-run) w celu sanity check konfiguracji:
   ```bash
   PYTHONPATH=. python scripts/run_daily_trend.py --config config/core.yaml --environment binance_paper --dry-run
   ```
   - Oczekiwany rezultat: brak wyjątków, pipeline zbudowany i natychmiast zakończony.
5. Uruchom tryb ciągły:
   ```bash
   PYTHONPATH=. python scripts/run_daily_trend.py --config config/core.yaml --environment binance_paper --poll-seconds 300
   ```
   - Proces monitoruje świeże świece D1, wykonuje risk checks, loguje decyzje oraz wysyła alerty.
   - Zaleca się uruchomienie w menedżerze procesów (np. `pm2`, `systemd --user`, Windows Task Scheduler) z automatycznym restartem.

## 4. Monitorowanie podczas sesji
- **Metryki** – endpoint HTTP (`http://localhost:9108/metrics`) wystawia liczniki zleceń, sygnałów, błędów adapterów, latencję egzekucji i health-check alertów.
- **Logi runtime** – `logs/runtime/paper_binance.log` (INFO + WARN + ERROR). Krytyczne błędy (np. brak świecy D1, odrzucenie zlecenia) natychmiast generują alert.
- **Dziennik decyzji** – `data/journal/paper_binance/*.jsonl`. Każdy wpis zawiera identyfikator sygnału, snapshot rynku, wynik kontroli ryzyka oraz status egzekucji.
- **Risk dashboard** – CLI: `PYTHONPATH=. python scripts/show_risk_state.py --environment paper_binance` pokazuje bieżące limity, zrealizowaną stratę dzienną i dostępny margines.

### Alarmy krytyczne
| Kod alertu | Opis | Działanie operatora |
|------------|------|---------------------|
| `RISK-DAILY-LIMIT` | Przekroczono dzienny limit straty – system przechodzi w tryb likwidacji i zamyka pozycje | Zweryfikuj logi, potwierdź zamknięcie pozycji, przygotuj raport incydentu |
| `ADAPTER-OFFLINE` | Adapter giełdy zwrócił błędy 5xx/timeout > 5 minut | Sprawdź status API giełdy, rozważ ręczne zatrzymanie pipeline’u |
| `DATA-LAG` | Brak świecy D1 > 12 h lub manifest Parquet zgłasza braki | Uruchom ponownie backfill, sprawdź limity API |
| `ALERT-CHANNEL-FAIL` | Kanał powiadomień nie odpowiada (np. SMS) | Przełącz na fallback (Telegram/e-mail), odnotuj w logu audytu |

## 5. Procedury reagowania na incydenty

### Błąd egzekucji / brak fill
1. Skontroluj log `execution` w `logs/runtime/paper_binance.log` – odczytaj kod błędu i parametry zlecenia.
2. Upewnij się, że risk engine nie blokuje dalszych transakcji (komenda `show_risk_state`).
3. Jeśli problem wynika z chwilowej niedostępności API, pipeline sam ponowi próbę (retry/backoff). Jeśli błąd trwa > 10 minut, zatrzymaj proces.
4. Zanotuj incydent w audycie (`docs/audit/paper_trading_log.md` – sekcja „Incydenty”).

### Brak nowych danych OHLCV
1. Uruchom `scripts/backfill.py --environment paper_binance --granularity 1d --latest-only`.
2. Zweryfikuj, czy manifest SQLite zaktualizował ostatni timestamp (`scripts/inspect_manifest.py`).
3. Jeśli brak reakcji, sprawdź status API giełdy (Binance status page). W razie globalnej awarii odnotuj incydent i zawieś strategię.

### Rotacja kluczy API
1. Uruchom `scripts/check_key_rotation.py --environment paper_binance --update` po wprowadzeniu nowych kluczy w keychainie.
2. Potwierdź działanie przez `scripts/run_daily_trend.py --mode dry-run`.
3. Zaktualizuj wpis w audycie z datą i osobą odpowiedzialną.

## 6. Zatrzymanie sesji
1. Wysyłamy sygnał `SIGINT`/`CTRL+C` do procesu `run_daily_trend.py`.
2. Poczekaj na komunikat „Shutdown complete” w logu – system zamknie otwarte pozycje (paper) i zapisze raport końcowy.
3. Uruchom generowanie raportu:
   ```bash
   PYTHONPATH=. python -c "from bot_core.reporting.paper import generate_daily_paper_report; generate_daily_paper_report('paper_binance')"
   ```
4. Zweryfikuj, że w `data/reports/daily/YYYY-MM-DD/paper_binance.zip` znajdują się: `ledger.csv`, `decisions.jsonl`, `summary.json`.
5. Zaszyfruj paczkę raportową (np. `age -r recipients.txt paper_binance.zip > paper_binance.zip.age`) i przenieś do archiwum zgodnie z polityką retencji.

## 7. Raportowanie i audyt
- Uzupełnij `docs/audit/paper_trading_log.md` (sekcja „Raport dzienny” oraz ewentualne incydenty).
- Zachowaj hash SHA-256 raportu (`shasum -a 256 paper_binance.zip.age`) i dopisz do logu audytu.
- Prześlij raport dzienny (PDF + CSV) na e-mail operatorski; w razie krytycznych alertów dołącz opis incydentu.

## 8. Checklista po sesji
- [ ] Pipeline zatrzymany kontrolowanie, brak procesów zombie (`tasklist` / `ps aux`).
- [ ] Alerty krytyczne zamknięte lub eskalowane.
- [ ] Raport dzienny zarchiwizowany i zabezpieczony.
- [ ] Dziennik decyzji oraz logi runtime skompresowane (starsze niż 30 dni przeniesione do archiwum).
- [ ] Rejestr rotacji kluczy zaktualizowany w razie zmian.
- [ ] Plan na kolejną sesję (ew. zmiany w konfiguracji) potwierdzony w zespole.

## 9. Rozszerzenia na kolejne etapy
- Dodanie równoległych środowisk `paper_kraken`, `paper_zonda` – procedura pozostaje identyczna, różnią się tylko adaptery i parametry prowizji.
- Integracja dodatkowych kanałów alertów (Signal, WhatsApp, Messenger) – aktywacja w konfiguracji i test health-checku.
- Automatyczne generowanie tygodniowego PDF z metrykami portfela i logami zmian konfiguracji.

> **Przypomnienie:** Wszystkie testy i pierwsze wdrożenia zawsze realizujemy w trybie paper/testnet. Przejście na ograniczony live wymaga kompletnego raportu z backtestu, zgodności P&L oraz review bezpieczeństwa (uprawnienia kluczy, IP allowlist, logi audytu).
