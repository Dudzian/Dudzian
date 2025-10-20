# Runbook: Paper trading – Etap 4 (multi-strategy)

Ten runbook opisuje, jak uruchomić, monitorować i bezpiecznie zatrzymać tryb paper tradingu w ramach Etapu 4 – z biblioteką strategii obejmującą trend-following D1, mean reversion, volatility targeting oraz arbitraż międzygiełdowy zarządzany przez harmonogram multi-strategy. Dokument przeznaczony jest dla operatorów i analityków, którzy realizują proces backtest → paper → ograniczony live przy zachowaniu rygorystycznej kontroli ryzyka (ThresholdRiskEngine + decision log podpisany HMAC). Wszystkie kroki wykonujemy na kontach demo/testnet z kluczami o minimalnych uprawnieniach (brak wypłat).

> **AutoTrader – tryb demo/paper**: `bootstrap_environment` dostarcza teraz gotowy `PaperTradingExecutionService`, a AutoTrader automatycznie przełącza ExecutionService na silnik paper zawsze, gdy strategia pracuje w trybie `demo`/`paper`. Paper trading zostaje również wymuszony, gdy brak jest skonfigurowanego adaptera live i aktywny pozostaje profil testnet. Dzięki temu każde środowisko bez produkcyjnych kluczy wykonuje zlecenia w symulatorze paper bez manualnego przełączania.

> **AutoTrader – mostek z kontrolerem realtime**: po potwierdzeniu trybu auto AutoTrader może teraz wykorzystać `DailyTrendRealtimeRunner`, aby pobierać sygnały z aktywnego `DailyTrendController` i delegować je do `TradingController`. Cały przepływ (sygnał → risk check → egzekucja) przebiega identycznie jak przy manualnym uruchomieniu pipeline'u, co pozwala testować paper trading bez kliknięć w GUI. Telemetrię ostatniego cyklu runnera (sygnały, wyniki egzekucji, timestamp startu/końca, liczba zleceń, sekwencja) można odczytać poprzez `AutoTrader.get_last_controller_cycle()` – przydatne do dashboardów oraz smoke testów. AutoTrader utrzymuje też bufor historii (`AutoTrader.get_controller_cycle_history(limit=...)`) ułatwiający budowę widoków typu timeline, a jego rozmiar można modyfikować w locie (`AutoTrader.set_controller_cycle_history_limit(...)`) i czyścić (`AutoTrader.clear_controller_cycle_history()`). Bufor można ograniczać również czasowo (`AutoTrader.set_controller_cycle_history_ttl(...)`), dzięki czemu timeline przechowuje wyłącznie najświeższe cykle bez ręcznego przeglądania i kasowania starych wpisów. Dodatkowo AutoTrader publikuje event `auto_trader.controller_cycle` (wraz z sekwencją, liczbą zleceń i czasem trwania cyklu) przez emitter, aby UI i zewnętrzne integracje mogły reagować na świeże cykle realtime bez aktywnego odpytywania.

  Jeśli potrzebujesz szybkich metryk operacyjnych (ile cykli w danym przedziale, średnia liczba zleceń, rozkład stron sygnałów, statusy egzekucji czy sumaryczny czas pracy runnera), skorzystaj z `AutoTrader.summarize_controller_cycle_history(...)`. Metoda respektuje ograniczenia czasowe (`since`/`until`) oraz limit liczby rekordów i zwraca zagregowane dane gotowe do prezentacji w dashboardzie lub raportach post-mortem.

  Jeżeli zamiast agregatów potrzebujesz szczegółowego widoku do dalszej analizy (np. w notebooku lub narzędziu BI), użyj `AutoTrader.controller_cycle_history_to_dataframe(...)`. Eksport buduje gotowy `DataFrame` Pandas z licznikami sygnałów/zleceń, opcjonalnymi sekwencjami surowych obiektów oraz automatyczną konwersją timestampów do `Timestamp` UTC.
  Dla integracji bez Pandas dostępna jest także metoda `AutoTrader.controller_cycle_history_to_records(...)`, która zwraca listę słowników (z limitami, filtrami `since`/`until`, opcjonalnymi sekwencjami i licznikami), a w razie potrzeby potrafi przekształcić znaczniki czasu na obiekty `datetime` (UTC lub w zadanym `tz`).
  Historia ocen ryzyka może być teraz ograniczana czasowo przy pomocy `AutoTrader.set_risk_evaluations_ttl(...)`, a bieżącą wartość TTL odczytasz przez `AutoTrader.get_risk_evaluations_ttl()`. Dzięki temu logi decyzji ryzyka pozostają zwarte w pamięci, a dashboardy bazujące na `risk_evaluations_to_dataframe(...)` odczytują wyłącznie świeże wpisy.

> **Regresje i testy**: utrzymujemy `pytest KryptoLowca/tests/test_paper_trading_integration.py`, który weryfikuje scenariusze paper na `bot_core.execution.paper.PaperTradingExecutionService`. Wszelkie zmiany w symulatorze muszą przechodzić te testy – legacy `PaperExchange` pozostaje jedynie w katalogu `archive/` dla historycznych porównań i nie powinien być używany w nowych wdrożeniach.
> **AutoTrader – tryb demo/paper**: Launchery (GUI oraz headless) korzystają teraz ze wspólnego `KryptoLowca.runtime.bootstrap.bootstrap_frontend_services`, który konfiguruje `ExchangeManager`, `LiveExecutionRouter`, `MultiExchangeAccountManager` i współdzielony `MarketIntelAggregator`. `bot_core.runtime.bootstrap_environment` nadal odpowiada za ciężki bootstrap środowiska (licencje, alerty, profile ryzyka), lecz fronty od razu otrzymują ten sam kontekst egzekucji i metryki rynku niezależnie od trybu (`demo`/`paper`). Paper trading zostaje również wymuszony, gdy brak jest skonfigurowanego adaptera live i aktywny pozostaje profil testnet – środowiska bez produkcyjnych kluczy zawsze trafiają do symulatora.
>
> **Tryb awaryjny**: `_NullExchangeAdapter` pozostaje jedynie mechanizmem awaryjnym – jeśli AutoTrader nie otrzyma ExecutionService z bootstrapu, loguje ostrzeżenie „paper-only mode”, blokuje flagę live i nie pozwala na aktywację trybu produkcyjnego do czasu uzupełnienia konfiguracji.
>
> **Początkowy balance**: przy włączaniu silnika paper AutoTrader pobiera saldo w kolejności: (1) z bieżącego GUI (`paper_balance`), (2) z konfiguracji strategii lub aktywnego profilu ryzyka (np. limitów `max_position_usd`/`max_position_pct`), a dopiero na końcu używa wartości domyślnej `10_000 USDT`. Aby zmienić startowy balans, ustaw `paper_balance` w GUI (zakładka „Portfolio”) lub skoryguj odpowiednie limity w strategii/profilu ryzyka przed uruchomieniem sesji.

> **Market intel w GUI**: Trading GUI prezentuje bieżące podsumowanie market-intel, historię wpisów (z auto-zapisem, czyszczeniem, eksportem i otwieraniem pliku) oraz aktualną ścieżkę historii. Te same dane wykorzystują dashboard i launcher papierowy – wszystkie fronty korzystają ze wspólnego `MarketIntelAggregator` wstrzykniętego przez `bootstrap_frontend_services`, dzięki czemu logi i raporty są zsynchronizowane.

## 1. Prerekwizyty operacyjne

### Środowisko i konfiguracja
- System operacyjny: Windows 10/11 (primary) lub macOS 13+ (fallback). W obu przypadkach wymagany jest Python 3.11 oraz dostęp do Windows Credential Manager / macOS Keychain.
- Repozytorium `Dudzian` w wersji zgodnej z `phase1_foundation` (aktualne testy `pytest` przechodzą bez błędów).
- Pliki konfiguracyjne `config/core.yaml` oraz `config/credentials/` dopasowane do środowiska `paper`.
- Każde środowisko w `config/core.yaml` ma ustawione pola `default_strategy` i `default_controller`, aby CLI automatycznie wybierało właściwą strategię i kontroler runtime.
- Katalog roboczy danych: `data/ohlcv` (Parquet + manifest SQLite) oraz `data/reports` na raporty dzienne.
- Sekcja `runtime.multi_strategy_schedulers.core_multi_pipeline` w `config/core.yaml` wskazuje kompletny zestaw strategii (trend-following, mean reversion, volatility targeting, cross-exchange arbitrage) wraz z mapowaniem na profile ryzyka i koszyki instrumentów.
- Parametry `risk.decision_log` definiują ścieżkę JSONL, identyfikator klucza HMAC i algorytm podpisu – wartości muszą być spójne z `scripts/verify_decision_log.py`.

### Klucze API i bezpieczeństwo
- Klucze `read-only` oraz `trade` dla Binance Testnet (spot i futures) zapisane w natywnym keychainie z etykietami odpowiadającymi polom `credential_purpose`.
- Jeśli posiadamy klucze Zondy paper lub Kraken demo, również zapisujemy je w keychainie – jednak w Etapie 1 aktywujemy tylko Binance.
- Lista dozwolonych IP: statyczny adres wyjściowy VPN + awaryjny adres stacji roboczej (paper). Potwierdź w panelu giełdy, że adresy są aktywne.
- Rejestr rotacji kluczy (`security/rotation_log.json`) uzupełniony o datę ostatniej wymiany; kolejne przypomnienie ustawiamy na 90 dni od tej daty.
- Klucz HMAC do podpisywania decision logu (`DECISION_LOG_HMAC_KEY`) przechowywany w keychainie z etykietą zgodną z `risk.decision_log.signing_key_id`; dostęp mają jedynie operatorzy paper tradingu (RBAC).

### Dane historyczne
- Wykonany backfill OHLCV (D1 + 1h) dla koszyka: BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, XRP/USDT, ADA/USDT, LTC/USDT, MATIC/USDT.
- Dane zapisane w strukturze partycjonowanej Parquet: `exchange/symbol/granularity/year=YYYY/month=MM/`.
- Manifest SQLite (`ohlcv_manifest.sqlite`) zawiera zaktualizowane liczniki świec i ostatnie znaczniki czasu.
- Zapełnienie koszyków instrumentów (`config/core.yaml -> risk.instrument_buckets`) potwierdzone raportem `scripts/check_data_coverage.py --instrument-bucket <bucket>` dla presetów multi-strategy.
- **Tryb offline (brak dostępu do API):** uruchom `PYTHONPATH=. python scripts/seed_paper_cache.py --environment binance_paper --days 60 --start-date 2024-01-01`, aby wygenerować deterministyczny cache D1 w katalogu `var/data/binance_paper`.

## 2. Checklista przed startem sesji
1. **Weryfikacja kodu i konfiguracji**
   - `git status` – brak lokalnych, niezatwierdzonych zmian.
   - `python scripts/validate_config.py --config config/core.yaml` – potwierdzenie spójności sekcji środowisk, profili ryzyka i kanałów alertowych.
   - `pytest --override-ini=addopts= tests/test_runtime_pipeline.py` – potwierdzenie, że pipeline przechodzi testy integracyjne.
   - `pytest tests/test_mean_reversion_strategy.py tests/test_volatility_target_strategy.py tests/test_cross_exchange_arbitrage_strategy.py tests/test_multi_strategy_scheduler.py` – sanity check nowych strategii i scheduler-a.
   - `PYTHONPATH=. pytest tests/test_risk_engine.py::test_force_liquidation_due_to_drawdown_allows_only_reducing_orders tests/test_risk_engine.py::test_daily_loss_limit_resets_after_new_trading_day` – regresje harnessu silnika ryzyka (scenariusz force liquidation + reset kolejnego dnia).
   - `PYTHONPATH=. python scripts/run_multi_strategy_scheduler.py --config config/core.yaml --environment binance_paper --dry-run --emit-telemetry` – walidacja harmonogramu multi-strategy i emisji metryk przed startem sesji.
   - `scripts/check_key_rotation.py --dry-run` – upewnij się, że rotacja kluczy nie jest przeterminowana.
   - `PYTHONPATH=. python scripts/verify_decision_log.py logs/decision_journal/paper_binance.jsonl --hmac-key-file <ścieżka_do_klucza> --expected-key-id risk-ci --allow-unsigned` – kontrola dostępności klucza HMAC oraz konfiguracji decision logu (plik można wygenerować z keychaina do katalogu tymczasowego).
2. **Aktualizacja danych i pre-check**
   - Uruchom `scripts/backfill.py --environment paper --granularity 1d --since 2016-01-01`.
   - Sprawdź logi (`logs/backfill.log`) pod kątem błędów; w razie limitów API powtórz z większym interwałem throttlingu.
   - Zweryfikuj pokrycie cache: `PYTHONPATH=. python scripts/check_data_coverage.py --config config/core.yaml --environment binance_paper --json`. Status `ok` oznacza komplet danych wymaganych przez backfill. W razie potrzeby możesz ograniczyć raport do wybranych symboli (`--symbol BTC_USDT`) lub interwałów (`--interval 1d`, `--interval D1`) oraz zapisać wynik do pliku (`--output data/reports/coverage/binance_paper.json`).
   - Uruchom kompleksowy pre-check: `PYTHONPATH=. python scripts/paper_precheck.py --config config/core.yaml --environment binance_paper --json`. Skrypt sprawdza poprawność konfiguracji, pokrycie manifestu oraz sanity-check silnika ryzyka. W kolumnie `risk_status` oczekuj wartości `ok`; ostrzeżenia (`warning`) wymagają przeglądu (np. brak docelowej zmienności), a `--skip-risk-check` stosuj wyłącznie w trybie debugowania. Raport JSON jest automatycznie zapisywany w `audit/paper_precheck_reports/` (plik z sygnaturą czasową i sumą SHA-256); `scripts/run_daily_trend.py --paper-smoke` potrafi tę ścieżkę dopisać do `docs/audit/paper_trading_log.md` wraz z hashem (`--paper-smoke-audit-log`, `--paper-smoke-operator`) oraz do ustrukturyzowanego dziennika JSONL (`docs/audit/paper_trading_log.jsonl`, flaga `--paper-smoke-json-log`). Po udanym zapisie wysyłany jest alert kategorii `paper_smoke_compliance` potwierdzający aktualizację logu. Jeżeli w `config/core.yaml` skonfigurowano sekcję `reporting.paper_smoke_json_sync`, CLI automatycznie tworzy kopię dziennika JSONL (np. w katalogu sieciowym lub w koszyku S3), weryfikuje zgodność sum SHA-256 repliki, zapisuje identyfikator wersji oraz numer potwierdzenia odbioru i dołącza lokalizację kopii do alertu compliance. Dodatkowo flaga `--paper-smoke-summary-json` pozwala zapisać znormalizowane podsumowanie smoke testu (hash `summary.json`, ścieżki, metadane potwierdzeń) w pliku JSON wykorzystywanym później przez pipeline CI; plik można natychmiast przekształcić w raport Markdown poleceniem `PYTHONPATH=. python scripts/render_paper_smoke_summary.py --summary-json <plik> --output <ścieżka.md>`, aby zasilić podsumowanie kroku CI. Z kolei `--paper-smoke-auto-publish` po pozytywnym zakończeniu smoke testu automatycznie wywołuje `publish_paper_smoke_artifacts.py`, aby zsynchronizować dziennik JSONL i archiwum ZIP zgodnie z konfiguracją reportingową. Jeśli publikacja artefaktów ma charakter obowiązkowy (np. wymagania CI/compliance), dodaj flagę `--paper-smoke-auto-publish-required` – pominięcie lub niepowodzenie uploadu zakończy smoke test kodem `6`, a alert `paper_smoke` otrzyma poziom `error`. **Uwaga:** `scripts/run_daily_trend.py --paper-smoke` uruchamia ten pre-check automatycznie i zakończy działanie, jeśli raport zwróci status `error`. Flaga `--paper-precheck-fail-on-warnings` wymusza traktowanie ostrzeżeń jako błędów, `--paper-precheck-audit-dir` pozwala wskazać własny katalog, a `--skip-paper-precheck` pozostaje wyłącznie narzędziem diagnostycznym.

   Jeżeli po smoke teście trzeba powtórzyć synchronizację dziennika JSONL (np. po ręcznej korekcie wpisu), uruchom `PYTHONPATH=. python scripts/sync_paper_smoke_json.py --environment <env> --json-log docs/audit/paper_trading_log.jsonl`. Skrypt korzysta z konfiguracji `reporting.paper_smoke_json_sync`, domyślnie wybiera ostatni rekord z logu i zwraca wynik w formacie tekstowym lub JSON (`--json`). Flaga `--dry-run` pozwala zweryfikować konfigurację bez transferu danych, a dla backendu S3 wymagane są parametry magazynu sekretów (`--secret-namespace`, `--headless-passphrase`, `--headless-storage`) identyczne jak w pozostałych narzędziach operacyjnych.

4. **Publikacja artefaktów smoke testu do magazynu audytowego**
  Pipeline CI lub operator może skorzystać z `PYTHONPATH=. python scripts/publish_paper_smoke_artifacts.py --environment <env> --report-dir <katalog_raportu> --json-log docs/audit/paper_trading_log.jsonl --json`, aby w jednej komendzie zsynchronizować dziennik JSONL oraz przesłać archiwum ZIP zgodnie z sekcją `reporting` w CoreConfig (to samo wywołanie wykonuje automatycznie `run_daily_trend.py --paper-smoke --paper-smoke-auto-publish`). Narzędzie automatycznie wyszuka rekord w dzienniku (po `record_id` lub `summary_sha256`), zweryfikuje sumy SHA-256 oraz wypisze metadane potwierdzeń. Jeżeli przekażesz dodatkowo `--summary-json <plik>` (np. wynik `--paper-smoke-summary-json`), CLI odczyta znormalizowane podsumowanie smoke testu, uzupełni brakujące parametry (`json-log`, `record_id`, ścieżkę archiwum) i zweryfikuje zgodność hashów z raportem. Flagi `--skip-json-sync`, `--skip-archive-upload` oraz `--dry-run` pozwalają na selektywną lub bezpieczną walidację, a parametry sekretów (`--secret-namespace`, `--headless-passphrase`, `--headless-storage`) są wymagane tylko dla backendu S3. W pipeline CI rekomendujemy skrypt `PYTHONPATH=. python scripts/run_paper_smoke_ci.py`, który generuje pełną komendę `run_daily_trend.py`, potrafi przyjąć dodatkowe argumenty (`--run-daily-trend-arg "--date-window 2024-01-01:2024-02-01" --run-daily-trend-arg "--run-once"`), może zapisać kluczowe ścieżki/statusy do pliku środowiskowego (`--env-file <ścieżka>`) oraz – przy pomocy `--render-summary-markdown` – wygeneruje raport Markdown. Domyślnie helper CI uruchamia również walidację podsumowania (`validate_paper_smoke_summary.py`) i będzie oczekiwał sukcesu auto-publikacji (sprawdza `publish.status`, `publish.required`, `publish.exit_code == 0`). W razie potrzeby możesz przekazać dodatkowe flagi walidatora, np. `--summary-validator-arg "--require-json-sync-ok"` lub `--summary-validator-arg "--require-archive-upload-ok"`, aby wymusić sukces poszczególnych kroków. Jeśli potrzebujesz raportu Markdown w kolejnych krokach (np. upload do artefaktów CI), wykorzystaj `PYTHONPATH=. python scripts/render_paper_smoke_summary.py --summary <summary.json> --output <plik.md>` lub analogiczną flagę helpera CI. Dla zespołów korzystających z GitHub Actions przygotowaliśmy przykładowy workflow w `deploy/ci/github_actions_paper_smoke.yml`, który demonstruje kolejne kroki (orchestrator, walidacja, renderowanie Markdown, upload artefaktów, **wysłanie alertu compliance przez `notify_paper_smoke_summary.py`**) i użycie pliku środowiskowego (`--env-file`).

3. **Konfiguracja środowiska**
   - Plik `config/core.yaml` ma aktywne środowisko `paper_binance` i profil ryzyka `balanced` (domyślny).
   - `config/alerts.yaml` (jeśli używany) zawiera aktywne kanały Telegram + e-mail + SMS (Orange jako operator referencyjny).
   - `runtime.multi_strategy_schedulers.core_multi_pipeline` wskazuje właściwe strategie (`core_mean_reversion`, `core_volatility_target`, `core_cross_exchange`) oraz posiada skonfigurowany token `CORE_SCHEDULER_TOKEN` (sprawdź obecność w keychainie).
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
  - W trybie `--paper-smoke` skrypt podmienia adapter giełdowy na tryb offline i korzysta wyłącznie z lokalnego cache Parquet/SQLite, dlatego przed uruchomieniem wymagany jest kompletny seed danych z kroku wcześniejszego oraz pozytywna weryfikacja `scripts/check_data_coverage.py`. Pipeline automatycznie wykonuje `paper_precheck`; jeśli raport zgłosi ostrzeżenia, informacja pojawi się w alertach i raporcie smoke. Ustaw `--paper-precheck-fail-on-warnings`, gdy chcesz, aby ostrzeżenia również przerywały test. Opcja `--skip-paper-precheck` służy tylko do debugowania i wymaga ręcznego odnotowania w audycie. Po udanym smoke teście wpis do `docs/audit/paper_trading_log.md` oraz JSONL `docs/audit/paper_trading_log.jsonl` tworzony jest automatycznie (możesz nadpisać lokalizację plików i operatora flagami `--paper-smoke-audit-log`, `--paper-smoke-json-log`, `--paper-smoke-operator`). Alert `paper_smoke_compliance` potwierdza zapis rekordu audytowego i zawiera identyfikator wpisu.
  - Dodatkowe potwierdzenie dla compliance możesz wysłać poleceniem `PYTHONPATH=. python scripts/notify_paper_smoke_summary.py --config config/core.yaml --environment binance_paper --summary-json <ścieżka>`. Skrypt korzysta z tej samej konfiguracji kanałów alertowych, respektuje wymagania `publish.required` i automatycznie uzupełnia kontekst (hash `summary.json`, lokalizacje kopii JSONL/ZIP, status auto-publikacji). W trybie CI przydaje się `--dry-run` dla podglądu payloadu lub `--severity-override`, aby wymusić poziom ważności.
   - Jeżeli w keychainie znajdują się placeholderowe tokeny kanałów alertowych, komunikaty o błędach (403/DNS) zostaną zarejestrowane w logu, ale nie przerywają smoke testu; status kanałów należy odnotować w audycie.
   - Flaga `--risk-profile <nazwa>` pozwala doraźnie przełączyć pipeline na inny profil ryzyka (np. `aggressive`) bez modyfikacji `config/core.yaml`. Używaj tylko profili zdefiniowanych w sekcji `risk_profiles`; wynik smoke testu wpisz do audytu wraz z informacją o użytym profilu.
   - Jeżeli modyfikujesz sygnały ręcznie (profil `manual`) lub stosujesz korekty pozycji po stronie risk engine, pamiętaj, że stop-loss może być ustawiony szerzej niż `stop_loss_atr_multiple` * ATR. Pipeline automatycznie przeliczy maksymalną wielkość pozycji na podstawie faktycznego dystansu stopu. Zbyt ciasny stop (poniżej minimum) spowoduje natychmiastowe odrzucenie zlecenia i komunikat o naruszeniu polityki ryzyka.
   - Użycie flagi `--archive-smoke` tworzy dodatkowo archiwum ZIP z kompletem plików i instrukcją audytu. Ścieżka katalogu raportu i hash `summary.json` pojawią się w logu. Skopiuj hash, treść `summary.txt` oraz status kanałów alertowych do `docs/audit/paper_trading_log.md`. Archiwum ZIP przechowuj w sejfie audytu (retencja ≥ 24 miesiące).
   - Przy pracy nad wariantami strategii wykorzystaj `--pipeline-module <moduł>` i `--realtime-module <moduł>`, aby tymczasowo nadpisać implementację pipeline'u lub runnera realtime. Moduły z CLI są preferowane przed domyślnymi (`bot_core.runtime.pipeline`, `bot_core.runtime.realtime`), ale fallback do `bot_core.runtime` pozostaje aktywny. Każda zmiana modułów musi zostać odnotowana w dzienniku decyzji wraz z hashami repozytoriów.
- Użyj `--print-runtime-modules`, aby audytować kolejność modułów pipeline/realtime po zastosowaniu override'ów. Flaga wypisuje JSON z listą kandydatów, polami `origin`, `resolved_from` oraz `fallback_used` (na poziomie modułów i sekcji pipeline/realtime), dzięki czemu od razu widać, czy skrypt musiał sięgnąć po awaryjny moduł `bot_core.runtime`. Polecenie kończy działanie CLI bez uruchamiania pipeline'u.
- `--print-risk-profiles` wypisuje profile ryzyka z sekcji `risk_profiles` (limity, data quality, powiązane środowiska, obowiązkowy pipeline demo→paper→live). Użyj tej flagi przed zmianą profilu w audycie, aby potwierdzić aktywne limity oraz czy środowisko korzysta z oczekiwanego profilu.
- `--runtime-plan-jsonl <ścieżka>` dopisuje do wskazanego pliku JSONL snapshot planu runtime: aktywne moduły pipeline/realtime (z `origin`, `resolved_from` oraz flagą `fallback_used`), zastosowane override'y CLI/ENV, wybrany profil ryzyka wraz z limitami, metadane repozytorium Git (commit/gałąź/tag, status roboczy), sanitowane metadane `paper_precheck` oraz sumę SHA-256 wraz z rozmiarem i czasem modyfikacji pliku konfiguracyjnego. Wpis obejmuje także sekcje `controller_details` (interwał, tick_seconds, symbole, kontekst egzekucji), `strategy_details` (klasa, wymagane interwały i parametry), `environment_details` (exchange, risk_profile, adapter, status telemetrii) oraz `pipeline_details` (klasa/pochodzenie obiektu pipeline'u). Dodatkowo eksportowana jest sekcja `metrics_service_details` (status `enabled`, ścieżki JSONL wraz z hashami `sha256`, źródło ścieżki alertów UI: `config` lub domyślne `logs/ui_telemetry_alerts.jsonl`, informacja o dostępności sinka `UiTelemetryAlertSink`, konfiguracja TLS/mTLS z metadanymi plików certyfikatów oraz informacja o wymogu uwierzytelniania klientów). Każdy plik w snapshotcie posiada sekcje `permissions`/`security_flags` oraz listę `security_warnings` (np. światowa czytelność/zapisywalność klucza TLS, brak katalogu docelowego dla JSONL), co ułatwia audytowi wychwycenie luk RBAC. Sekcja zawiera również `runtime_state` (rzeczywiste ścieżki użyte przez bootstrap, metadane katalogów i flagę `ui_alert_sink_active`/`service_enabled`), dzięki czemu audyt widzi rozbieżności między konfiguracją a stanem uruchomionego procesu. Wszystkie ścieżki pochodzące z konfiguracji są rozwijane względem katalogu pliku `core.yaml`, dzięki czemu `path` i `absolute_path` odzwierciedlają faktyczną lokalizację artefaktów (JSONL, certyfikaty, logi alertów UI). Artefakt powstaje przed bootstrapem i powinien zostać dołączony do decision logu dla audytu demo→paper→live.
- `--print-runtime-plan` wypisuje na stdout powyższy snapshot (łącznie z sekcjami `controller_details`/`strategy_details`/`environment_details`/`metrics_service_details`, metadanymi Git, sekcją `config_file` oraz flagami `fallback_used`) i kończy działanie CLI przed bootstrapem pipeline'u, co pozwala szybko zweryfikować konfigurację w CI lub podczas przeglądu zmian.
- `--fail-on-security-warnings` (dostępne w `run_daily_trend.py`, `run_metrics_service.py` oraz `run_trading_stub_server.py`) analizuje sekcje `security_warnings` wygenerowane przez moduł `bot_core.runtime.file_metadata` i kończy działanie kodem 3, jeśli w planie runtime/konfiguracji znajdują się ostrzeżenia (np. brak katalogu docelowego JSONL, zbyt luźne prawa chmod dla kluczy TLS, światowa czytelność logów telemetrii). Pozwala to wymusić zgodność z politykami RBAC i standardami CI przed uruchomieniem strategii/stubów.
- Jeśli środowisko paper korzysta z dedykowanego serwera telemetrii (`scripts/run_metrics_service.py`), przed startem możesz wykonać audyt konfiguracji poleceniem `python scripts/run_metrics_service.py --jsonl logs/metrics.jsonl --config-plan-jsonl audit/metrics_service_plan.jsonl --shutdown-after 0`. Zapisany wpis zawiera metadane TLS (hash certyfikatów, katalogi nadrzędne i uprawnienia zapisu) oraz szczegółową sekcję `security_flags`/`security_warnings` dla każdego pliku (np. ostrzeżenie o światowej czytelności/zapisywalności klucza prywatnego). Snapshot obejmuje też ścieżki JSONL, pochodzenie parametrów (`cli`/`env`/`core_config`), status sinków logujących i informację o włączonym sinku alertów UI. Sekcja `runtime_state` raportuje faktyczną liczbę aktywnych sinków, wykryte sinki dodatkowe, status TLS/mTLS po starcie oraz pełną ścieżkę logu alertów UI. Flaga `--print-config-plan` wypisze identyczny snapshot na stdout i zakończy działanie skryptu, umożliwiając szybki przegląd w CI przed uruchomieniem demona telemetrii. Opcjonalne `--core-config config/core.yaml` zastosuje sekcję `runtime.metrics_service` (host, port, log_sink, JSONL, fsync, log alertów UI, TLS/mTLS) jako domyślne ustawienia CLI i doda do artefaktu sekcję `core_config` z hashami, katalogami nadrzędnymi oraz źródłem każdej wartości (`cli`, `core_config`, `core_config_disabled`, `env`). Audyt demo→paper→live dysponuje wówczas jednym snapshotem potwierdzającym zarówno stan uruchomionego procesu, jak i spójność konfiguracji w repozytorium.
- Serwer telemetrii honoruje teraz zmienne środowiskowe `RUN_METRICS_SERVICE_*` (m.in. `HOST`, `PORT`, `HISTORY_SIZE`, `NO_LOG_SINK`, `JSONL`, `JSONL_FSYNC`, `TLS_CERT`, `TLS_KEY`, `TLS_CLIENT_CA`, `TLS_REQUIRE_CLIENT_CERT`, `SHUTDOWN_AFTER`, `LOG_LEVEL`). Każda aktywna zmiana pojawia się w sekcji `environment_overrides` snapshotu wraz z oceną (`applied`/`cli_override`) i zdekodowaną wartością. Kolumna `parameter_sources` ujawnia ostateczne źródło każdej konfiguracji (`default`, `cli`, `env`, `env_disabled`, `core_config`, `core_config_none`, `core_config_disabled`), co pozwala audytorowi łatwo odtworzyć sekwencję priorytetów w pipeline demo→paper→live.
- `scripts/run_trading_stub_server.py` udostępnia flagi `--print-runtime-plan` oraz `--runtime-plan-jsonl`, które zapisują lub wypisują snapshot konfiguracji stubu (lista datasetów, konfiguracja telemetrii, ścieżki JSONL i materiałów TLS wraz z metadanymi bezpieczeństwa). Artefakt wykorzystuje wspólny moduł `bot_core.runtime.file_metadata`, więc w audycie zobaczysz katalogi nadrzędne, uprawnienia, właścicieli oraz ostrzeżenia RBAC dotyczące logów JSONL i kluczy TLS. Snapshot powstaje przed uruchomieniem serwera, co pozwala potwierdzić konfigurację CI jeszcze przed startem telemetrii i strumieni gRPC.
  - CLI raportuje sekcję `environment` z informacją o źródłach parametrów (`parameter_sources`) oraz listą zastosowanych/odrzuconych override'ów środowiskowych (`overrides`). Możesz wymusić konfigurację bez edycji skryptów używając zmiennych `RUN_TRADING_STUB_*` (np. `RUN_TRADING_STUB_ENABLE_METRICS=1`, `RUN_TRADING_STUB_METRICS_JSONL=/var/log/metrics.jsonl`, `RUN_TRADING_STUB_DATASETS=/path/a.yaml:/path/b.yaml` na systemach Unix lub `/path/a.yaml;/path/b.yaml` na Windows, `RUN_TRADING_STUB_NO_DEFAULT_DATASET=true`). Każdy wpis zawiera surową wartość, wynik parsowania i informację, czy został pominięty z powodu flagi CLI (`reason: "cli_override"`).
  - W sekcji `parameter_sources` znajdziesz, czy dana wartość pochodzi z CLI (`cli`), zmiennej środowiskowej (`env`) czy domyślnej konfiguracji (`default`). Dzięki temu decision log papier tradingu obejmuje pełną genezę aktywnej konfiguracji stubu (host/port, dataset-y, telemetria, TLS/mTLS, logi alertów UI).
- Przed bootstrapem pipeline'u CLI waliduje, czy wskazane środowisko, strategia, kontroler i profil ryzyka istnieją w `CoreConfig`. Dodatkowo sprawdzamy, czy środowisko posiada domyślną strategię i kontroler; jeśli pola `default_strategy` / `default_controller` są puste, a operator nie poda flag `--strategy` / `--controller`, wykonanie kończy się kodem `2`. Zarówno błędne nazwy przekazane w CLI, jak i niepoprawne wpisy w konfiguracji środowiska, są raportowane wraz z listą dostępnych wartości, a pipeline nie jest bootstrapowany – pozwala to wychwycić regresje jeszcze w fazie audytu.
- W środowiskach CI/CD można wykorzystać zmienne `RUN_DAILY_TREND_PIPELINE_MODULES` oraz `RUN_DAILY_TREND_REALTIME_MODULES` (lista modułów rozdzielona spacjami, przecinkami lub średnikami), aby ustawić domyślne override'y bez dopisywania flag CLI. Parametry CLI mają pierwszeństwo przed zmiennymi środowiskowymi i powinny być używane w logach audytu. Puste lub niepoprawne wartości są ignorowane z ostrzeżeniem w logach runtime, aby zapobiec cichym regresjom konfiguracji.
   - Opcjonalnie możesz wskazać katalog docelowy na artefakty smoke testu przy pomocy `--smoke-output <ścieżka>`. Wewnątrz katalogu zostanie utworzony podkatalog `daily_trend_smoke_*`; zachowaj go bez zmian do czasu archiwizacji i wpisu w logu audytu.
   - Parametr `--smoke-min-free-mb <wartość>` pozwala narzucić minimalną ilość wolnego miejsca w katalogu raportu. Gdy próg nie jest spełniony, CLI zapisze ostrzeżenie w logu, oznaczy raport w `summary.json` oraz dopisze ostrzeżenie w alercie `paper_smoke`.
   - Dodając `--smoke-fail-on-low-space` wymusisz traktowanie niskiego wolnego miejsca jako błędu operacyjnego – skrypt zakończy się kodem 4 po zapisaniu raportu i wyśle alert o poziomie `warning`.
   - Jeśli w `config/core.yaml` skonfigurowano sekcję `reporting.smoke_archive_upload`, CLI automatycznie wykona kopię archiwum (domyślnie do `audit/smoke_archives/`) oraz – w przypadku backendu S3/MinIO – prześle plik do zdefiniowanego koszyka. Po wysyłce narzędzie oblicza hash SHA-256 archiwum, weryfikuje zgodność kopii (lokalnej lub zdalnej) i zapisuje metadane potwierdzenia (`verified`, `ack_request_id`, `version_id`). Ścieżka docelowa oraz hash pojawiają się w logu i kontekście alertu w polach `archive_upload_backend`, `archive_upload_location`, `archive_sha256`. Zweryfikuj obecność pliku w magazynie i odnotuj lokalizację wraz z numerem potwierdzenia w audycie.

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

### Force liquidation / przekroczony limit strat
1. Zweryfikuj stan profilu: `PYTHONPATH=. python scripts/show_risk_state.py --environment paper_binance --profile balanced` – oczekuj flagi `force_liquidation: true` oraz bieżącej ekspozycji.
2. Uruchom `PYTHONPATH=. pytest tests/test_risk_engine.py::test_force_liquidation_due_to_drawdown_allows_only_reducing_orders` – potwierdza, że jedyne dozwolone transakcje w trybie awaryjnym to redukcja pozycji.
3. Po zamknięciu wszystkich pozycji i rozpoczęciu kolejnego dnia kalendarzowego uruchom `PYTHONPATH=. pytest tests/test_risk_engine.py::test_daily_loss_limit_resets_after_new_trading_day`, aby potwierdzić reset limitów.
4. Wygeneruj wpis w dzienniku decyzji z podpisem HMAC (`logs/decision_journal/paper_binance.jsonl`) i zweryfikuj `scripts/verify_decision_log.py --expected-key-id risk-ci --hmac-key-file <ścieżka_do_klucza>` – zapis audytowy jest wymagany przed wznowieniem handlu.

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
- [ ] Telemetria scheduler-a (`audit/metrics/telemetry.jsonl`) zawiera wpis z `multi_strategy` oraz `latency_ms < 250`.
- [ ] `PYTHONPATH=. python scripts/verify_decision_log.py --log audit/decisions --hmac-key $DECISION_KEY` potwierdził komplet podpisów dla sygnałów multi-strategy.
- [ ] `PYTHONPATH=. python bot_core/runtime/telemetry_risk_profiles.py --summary core` potwierdził, że `strategy_allocations` mieściły się w limitach profilu ryzyka.
- [ ] Rejestr rotacji kluczy zaktualizowany w razie zmian.
- [ ] Plan na kolejną sesję (ew. zmiany w konfiguracji) potwierdzony w zespole.

## 9. Rozszerzenia na kolejne etapy
- Dodanie równoległych środowisk `paper_kraken`, `paper_zonda` – procedura pozostaje identyczna, różnią się tylko adaptery i parametry prowizji.
- Integracja dodatkowych kanałów alertów (Signal, WhatsApp, Messenger) – aktywacja w konfiguracji i test health-checku.
- Automatyczne generowanie tygodniowego PDF z metrykami portfela i logami zmian konfiguracji.

> **Przypomnienie:** Wszystkie testy i pierwsze wdrożenia zawsze realizujemy w trybie paper/testnet. Przejście na ograniczony live wymaga kompletnego raportu z backtestu, zgodności P&L oraz review bezpieczeństwa (uprawnienia kluczy, IP allowlist, logi audytu).

## Monitoring budżetów zasobów
- Konfiguracja limitów znajduje się w `config/core.yaml` → `runtime.resource_limits`.
- `python scripts/load_test_scheduler.py --iterations 60 --schedules 3 --output logs/load_tests/paper_mode.json` – weryfikuje latencję i status budżetów przed startem paper.
- `pytest tests/test_resource_monitor.py` – szybkie potwierdzenie logiki monitoringu (`bot_core.runtime.resource_monitor`).
- `python scripts/audit_security_baseline.py --print --scheduler-required-scope runtime.schedule.write` – łączy audyt RBAC/mTLS z walidacją budżetów CPU/RAM/I/O.
